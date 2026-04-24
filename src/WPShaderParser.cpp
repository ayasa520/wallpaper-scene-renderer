#include "WPShaderParser.hpp"

#include "Fs/IBinaryStream.h"
#include "Utils/Logging.h"
#include "WPJson.hpp"

#include "wpscene/WPUniform.h"
#include "Fs/VFS.h"
#include "Utils/Sha.hpp"
#include "Utils/String.h"
#include "WPCommon.hpp"

#include "Vulkan/ShaderComp.hpp"

#include <regex>
#include <array>
#include <charconv>
#include <cctype>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

static constexpr std::string_view SHADER_PLACEHOLD { "__SHADER_PLACEHOLD__" };

#define SHADER_DIR    "spvs01"
#define SHADER_SUFFIX "spvs"
#define SHADER_META_DIR "pre-shaders01"
#define SHADER_META_SUFFIX "wpmeta"
#define SHADER_SRC_DIR "prepared-shaders01"
#define SHADER_SRC_SUFFIX "wpsrc"

using namespace wallpaper;

namespace
{
struct GlslangRuntimeState {
    std::recursive_mutex mutex;
    std::size_t          depth { 0 };
};

GlslangRuntimeState& GetGlslangRuntimeState() {
    static GlslangRuntimeState state;
    return state;
}

std::size_t GetThreadLogId() {
    return std::hash<std::thread::id> {}(std::this_thread::get_id());
}

static constexpr const char* pre_shader_code = R"(#version 150
#define GLSL 1
#define HLSL 0
#define highp

#define CAST2(x) (vec2(x))
#define CAST3(x) (vec3(x))
#define CAST4(x) (vec4(x))
#define CAST3X3(x) (mat3(x))

vec4 texSample2D(sampler2D tex, vec2 uv) { return texture(tex, uv); }
vec4 texSample2D(sampler2D tex, vec3 uv) { return texture(tex, uv.xy); }
vec4 texSample2D(sampler2D tex, vec4 uv) { return texture(tex, uv.xy); }

vec4 texSample2DLod(sampler2D tex, vec2 uv, float lod) { return textureLod(tex, uv, lod); }
vec4 texSample2DLod(sampler2D tex, vec3 uv, float lod) { return textureLod(tex, uv.xy, lod); }
vec4 texSample2DLod(sampler2D tex, vec4 uv, float lod) { return textureLod(tex, uv.xy, lod); }
#define mul(x, y) ((y) * (x))
#define frac fract
#define atan2 atan
#define fmod(x, y) (x-y*trunc(x/y))
#define ddx dFdx
#define ddy(x) dFdy(-(x))
#define saturate(x) (clamp(x, 0.0, 1.0))

#define max(x, y) max(y, x)

#define float1 float
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define lerp mix

__SHADER_PLACEHOLD__

)";

static constexpr const char* pre_shader_code_vert = R"(
#define attribute in
#define varying out

)";
static constexpr const char* pre_shader_code_frag = R"(
#define varying in
#define gl_FragColor glOutColor
out vec4 glOutColor;

)";

inline std::string LoadGlslInclude(fs::VFS& vfs, const std::string& input) {
    std::string::size_type pos = 0;
    std::string            output;
    std::string::size_type linePos = std::string::npos;

    while (linePos = input.find("#include", pos), linePos != std::string::npos) {
        auto lineEnd  = input.find_first_of('\n', linePos);
        auto lineSize = lineEnd - linePos;
        auto lineStr  = input.substr(linePos, lineSize);
        output.append(input.substr(pos, linePos - pos));

        auto inP         = lineStr.find_first_of('\"') + 1;
        auto inE         = lineStr.find_last_of('\"');
        auto includeName = lineStr.substr(inP, inE - inP);
        auto includeSrc  = fs::GetFileContent(vfs, "/assets/shaders/" + includeName);
        output.append("\n//-----include " + includeName + "\n");
        output.append(LoadGlslInclude(vfs, includeSrc));
        output.append("\n//-----include end\n");

        pos = lineEnd;
    }
    output.append(input.substr(pos));
    return output;
}

inline size_t SkipWhitespace(std::string_view source, size_t pos) {
    while (pos < source.size() && std::isspace(static_cast<unsigned char>(source[pos]))) {
        pos++;
    }
    return pos;
}

inline char PrevNonWhitespace(std::string_view source, size_t pos) {
    while (pos > 0) {
        pos--;
        if (! std::isspace(static_cast<unsigned char>(source[pos]))) return source[pos];
    }
    return '\0';
}

inline bool LooksLikeObjectKeyAfterComma(std::string_view source, size_t pos) {
    pos = SkipWhitespace(source, pos);
    if (pos >= source.size() || source[pos] != '"') return false;

    bool saw_char { false };
    bool escaped { false };
    pos++;
    while (pos < source.size()) {
        const char ch = source[pos];
        if (escaped) {
            escaped  = false;
            saw_char = true;
            pos++;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            pos++;
            continue;
        }
        if (ch == '"') break;
        saw_char = true;
        pos++;
    }

    if (pos >= source.size() || ! saw_char) return false;
    pos = SkipWhitespace(source, pos + 1);
    return pos < source.size() && source[pos] == ':';
}

inline bool LooksLikeContainerClose(std::string_view source, size_t pos) {
    if (pos >= source.size() || (source[pos] != '}' && source[pos] != ']')) return false;

    pos = SkipWhitespace(source, pos + 1);
    return pos >= source.size() || source[pos] == ',' || source[pos] == '}' || source[pos] == ']';
}

inline std::string TruncateMetadataSnippet(std::string_view source) {
    constexpr size_t max_len { 120 };
    if (source.size() <= max_len) return std::string(source);
    return std::string(source.substr(0, max_len - 3)) + "...";
}

inline bool IsIdentifierStart(char ch) {
    const auto uch = static_cast<unsigned char>(ch);
    return std::isalpha(uch) || ch == '_';
}

inline bool IsIdentifierContinue(char ch) {
    const auto uch = static_cast<unsigned char>(ch);
    return std::isalnum(uch) || ch == '_';
}

inline size_t SkipIdentifier(std::string_view source, size_t pos) {
    while (pos < source.size() && IsIdentifierContinue(source[pos])) {
        pos++;
    }
    return pos;
}

inline size_t SkipBalanced(std::string_view source, size_t pos, char open_ch, char close_ch) {
    if (pos >= source.size() || source[pos] != open_ch) return std::string::npos;

    int depth { 0 };
    while (pos < source.size()) {
        if (source[pos] == open_ch) {
            depth++;
        } else if (source[pos] == close_ch) {
            depth--;
            if (depth == 0) return pos + 1;
        }
        pos++;
    }
    return std::string::npos;
}

inline bool IsIdentifierBoundary(std::string_view source, size_t pos) {
    return pos >= source.size() || ! IsIdentifierContinue(source[pos]);
}

inline bool MatchIdentifierAt(std::string_view source, size_t pos, std::string_view ident) {
    if (pos + ident.size() > source.size()) return false;
    if (source.substr(pos, ident.size()) != ident) return false;
    if (pos > 0 && IsIdentifierContinue(source[pos - 1])) return false;
    return IsIdentifierBoundary(source, pos + ident.size());
}

inline bool ReplaceFirstStandaloneWord(std::string& text, std::string_view from,
                                       std::string_view to) {
    size_t pos { 0 };
    while (pos < text.size()) {
        if (! IsIdentifierStart(text[pos])) {
            pos++;
            continue;
        }

        const auto end   = SkipIdentifier(text, pos);
        const auto token = std::string_view(text).substr(pos, end - pos);
        if (token == from) {
            text.replace(pos, from.size(), to);
            return true;
        }
        pos = end;
    }
    return false;
}

inline std::string ReplaceStandaloneIdentifier(std::string_view text, std::string_view from,
                                               std::string_view to) {
    std::string result;
    result.reserve(text.size());

    size_t pos { 0 };
    while (pos < text.size()) {
        if (MatchIdentifierAt(text, pos, from)) {
            result.append(to);
            pos += from.size();
            continue;
        }

        result.push_back(text[pos]);
        pos++;
    }
    return result;
}

inline bool ReplaceAll(std::string& text, std::string_view from, std::string_view to) {
    if (from.empty()) return false;

    bool   changed { false };
    size_t pos { 0 };
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
        changed = true;
    }
    return changed;
}

inline std::string_view TrimLeadingHorizontalWhitespace(std::string_view source) {
    const auto pos = source.find_first_not_of(" \t\r");
    if (pos == std::string_view::npos) return {};
    return source.substr(pos);
}

inline bool StartsWithToken(std::string_view source, std::string_view token) {
    if (source.size() < token.size()) return false;
    if (source.substr(0, token.size()) != token) return false;
    return source.size() == token.size() ||
           ! IsIdentifierContinue(source[token.size()]);
}

inline void UpdateBraceDepth(std::string_view line, int& brace_depth, bool& in_block_comment) {
    bool in_string { false };
    bool escaped { false };
    char quote { '\0' };

    for (size_t i = 0; i < line.size(); i++) {
        const char ch = line[i];
        const char next = i + 1 < line.size() ? line[i + 1] : '\0';

        if (in_block_comment) {
            if (ch == '*' && next == '/') {
                in_block_comment = false;
                i++;
            }
            continue;
        }

        if (in_string) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                continue;
            }
            if (ch == quote) in_string = false;
            continue;
        }

        if (ch == '/' && next == '/') break;
        if (ch == '/' && next == '*') {
            in_block_comment = true;
            i++;
            continue;
        }
        if (ch == '"' || ch == '\'') {
            in_string = true;
            quote     = ch;
            continue;
        }

        if (ch == '{') {
            brace_depth++;
        } else if (ch == '}' && brace_depth > 0) {
            brace_depth--;
        }
    }
}

inline bool TryParseInterfaceDeclaration(std::string_view line, std::string& qualifier,
                                         std::string& name) {
    const auto end = line.find_last_not_of(" \t\r");
    if (end == std::string_view::npos || line[end] != ';') return false;

    size_t qualifier_pos { std::string::npos };
    size_t qualifier_end { std::string::npos };
    size_t pos { 0 };
    while (pos < line.size()) {
        if (! IsIdentifierStart(line[pos])) {
            pos++;
            continue;
        }

        const auto end_pos = SkipIdentifier(line, pos);
        const auto token   = line.substr(pos, end_pos - pos);
        if (token == "in" || token == "out") {
            qualifier_pos = pos;
            qualifier_end = end_pos;
            qualifier     = std::string(token);
            break;
        }
        pos = end_pos;
    }
    if (qualifier_pos == std::string::npos) return false;

    const auto type_pos = SkipWhitespace(line, qualifier_end);
    if (type_pos >= line.size() || ! IsIdentifierStart(line[type_pos])) return false;

    const auto type_end = SkipIdentifier(line, type_pos);
    const auto name_pos = SkipWhitespace(line, type_end);
    if (name_pos >= line.size() || ! IsIdentifierStart(line[name_pos])) return false;

    const auto name_end = SkipIdentifier(line, name_pos);
    auto       tail_pos = SkipWhitespace(line, name_end);
    if (tail_pos < line.size() && line[tail_pos] == '[') {
        tail_pos = SkipBalanced(line, tail_pos, '[', ']');
        if (tail_pos == std::string::npos) return false;
        tail_pos = SkipWhitespace(line, tail_pos);
    }

    if (tail_pos >= line.size() || line[tail_pos] != ';') return false;

    name = std::string(line.substr(name_pos, name_end - name_pos));
    return true;
}

inline bool TryParseTextureSlot(std::string_view line, uint& slot) {
    const auto end = line.find_last_not_of(" \t\r");
    if (end == std::string_view::npos || line[end] != ';') return false;

    bool        saw_uniform { false };
    bool        saw_sampler2d { false };
    std::string name;

    size_t pos { 0 };
    while (pos < line.size()) {
        if (! IsIdentifierStart(line[pos])) {
            pos++;
            continue;
        }

        const auto end_pos = SkipIdentifier(line, pos);
        const auto token   = line.substr(pos, end_pos - pos);
        if (! saw_uniform && token == "uniform") {
            saw_uniform = true;
        } else if (saw_uniform && ! saw_sampler2d && token == "sampler2D") {
            saw_sampler2d = true;
        } else if (saw_sampler2d) {
            name = std::string(token);
            break;
        }
        pos = end_pos;
    }

    if (name.size() <= 9 || name.compare(0, 9, "g_Texture") != 0) return false;

    const auto slot_text = std::string_view(name).substr(9);
    if (slot_text.empty()) return false;

    auto [ptr, ec] = std::from_chars(slot_text.data(), slot_text.data() + slot_text.size(), slot);
    return ec == std::errc() && ptr == slot_text.data() + slot_text.size();
}

inline std::string_view TrimWhitespace(std::string_view source) {
    const auto begin = source.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) return {};
    const auto end = source.find_last_not_of(" \t\r\n");
    return source.substr(begin, end - begin + 1);
}

inline std::string RemoveAsciiWhitespace(std::string_view source) {
    std::string result;
    result.reserve(source.size());
    for (const char ch : source) {
        if (! std::isspace(static_cast<unsigned char>(ch))) result.push_back(ch);
    }
    return result;
}

inline bool TryParseStandaloneIdentifier(std::string_view source, std::string_view& ident) {
    source = TrimWhitespace(source);
    if (source.empty() || ! IsIdentifierStart(source.front())) return false;

    const auto end = SkipIdentifier(source, 0);
    if (end != source.size()) return false;

    ident = source.substr(0, end);
    return true;
}

inline bool TryParseSimpleTypedAssignment(std::string_view statement, std::string_view& type,
                                          std::string_view& lhs, std::string_view& rhs) {
    statement = TrimWhitespace(statement);
    if (statement.empty() || statement.back() != ';') return false;
    statement.remove_suffix(1);
    statement = TrimWhitespace(statement);
    if (statement.empty() || ! IsIdentifierStart(statement.front())) return false;

    const auto type_end = SkipIdentifier(statement, 0);
    type                = statement.substr(0, type_end);

    auto pos = SkipWhitespace(statement, type_end);
    if (pos >= statement.size() || ! IsIdentifierStart(statement[pos])) return false;

    const auto lhs_end = SkipIdentifier(statement, pos);
    lhs                = statement.substr(pos, lhs_end - pos);

    pos = SkipWhitespace(statement, lhs_end);
    if (pos >= statement.size() || statement[pos] != '=') return false;
    if (pos + 1 < statement.size() && statement[pos + 1] == '=') return false;

    rhs = TrimWhitespace(statement.substr(pos + 1));
    return ! rhs.empty();
}

inline bool TryParseConstructorCall(std::string_view source, std::string_view& ctor,
                                    std::string_view& args) {
    source = TrimWhitespace(source);
    if (source.empty() || ! IsIdentifierStart(source.front())) return false;

    const auto ctor_end = SkipIdentifier(source, 0);
    ctor                = source.substr(0, ctor_end);

    auto pos = SkipWhitespace(source, ctor_end);
    if (pos >= source.size() || source[pos] != '(') return false;

    const auto close_pos = SkipBalanced(source, pos, '(', ')');
    if (close_pos == std::string::npos || close_pos != source.size()) return false;

    args = source.substr(pos + 1, close_pos - pos - 2);
    return true;
}

struct ShaderVectorVariable {
    std::string name;
    int         components { 0 };
};

inline int ShaderVectorComponents(std::string_view type) {
    if (type == "float" || type == "float1") return 1;
    if (type == "vec2" || type == "float2") return 2;
    if (type == "vec3" || type == "float3") return 3;
    if (type == "vec4" || type == "float4") return 4;
    return 0;
}

inline bool IsShaderDeclarationQualifier(std::string_view token) {
    return token == "const" || token == "in" || token == "out" || token == "attribute" ||
        token == "varying" || token == "uniform" || token == "flat" || token == "smooth" ||
        token == "noperspective" || token == "centroid" || token == "sample" ||
        token == "patch" || token == "readonly" || token == "writeonly" ||
        token == "coherent" || token == "volatile" || token == "restrict" ||
        token == "highp" || token == "mediump" || token == "lowp";
}

inline bool TryParseVectorDeclarationLine(std::string_view line, std::string_view& type,
                                          std::string_view& name) {
    const auto comment_pos = line.find("//");
    if (comment_pos != std::string_view::npos) line = line.substr(0, comment_pos);

    const auto semicolon_pos = line.find(';');
    if (semicolon_pos == std::string_view::npos) return false;

    line = TrimWhitespace(line.substr(0, semicolon_pos + 1));
    if (line.empty() || line.back() != ';') return false;
    line.remove_suffix(1);
    line = TrimWhitespace(line);
    if (line.empty()) return false;

    size_t pos { 0 };
    while (pos < line.size()) {
        if (! IsIdentifierStart(line[pos])) return false;

        const auto token_end = SkipIdentifier(line, pos);
        const auto token     = line.substr(pos, token_end - pos);
        if (token == "layout") {
            pos = SkipWhitespace(line, token_end);
            if (pos >= line.size() || line[pos] != '(') return false;
            pos = SkipBalanced(line, pos, '(', ')');
            if (pos == std::string::npos) return false;
            pos = SkipWhitespace(line, pos);
            continue;
        }

        if (IsShaderDeclarationQualifier(token)) {
            pos = SkipWhitespace(line, token_end);
            continue;
        }

        if (ShaderVectorComponents(token) == 0) return false;
        type = token;
        pos  = SkipWhitespace(line, token_end);
        break;
    }

    if (pos >= line.size() || ! IsIdentifierStart(line[pos])) return false;

    const auto name_end = SkipIdentifier(line, pos);
    name                = line.substr(pos, name_end - pos);

    pos = SkipWhitespace(line, name_end);
    if (pos < line.size() && line[pos] == '[') {
        pos = SkipBalanced(line, pos, '[', ']');
        if (pos == std::string::npos) return false;
        pos = SkipWhitespace(line, pos);
    }

    return pos >= line.size() || line[pos] == '=' || line[pos] == ',';
}

inline void UpsertShaderVectorVariable(std::vector<ShaderVectorVariable>& variables,
                                       std::string_view name, int components) {
    for (auto it = variables.rbegin(); it != variables.rend(); ++it) {
        if (it->name == name) {
            it->components = components;
            return;
        }
    }
    variables.push_back(ShaderVectorVariable { .name = std::string(name), .components = components });
}

inline int FindShaderVectorComponents(const std::vector<ShaderVectorVariable>& variables,
                                      std::string_view name) {
    for (auto it = variables.rbegin(); it != variables.rend(); ++it) {
        if (it->name == name) return it->components;
    }
    return 0;
}

inline std::string_view NarrowingSwizzle(int components) {
    if (components == 2) return ".xy";
    if (components == 3) return ".xyz";
    return {};
}

inline bool TryParseSimpleUntypedAssignmentLine(std::string_view line, std::string_view& lhs,
                                                std::string_view& rhs, size_t& rhs_begin,
                                                size_t& rhs_len) {
    const auto comment_pos = line.find("//");
    const auto search_end =
        comment_pos == std::string_view::npos ? line.size() : comment_pos;
    line = line.substr(0, search_end);

    const auto semicolon_pos = line.find(';');
    if (semicolon_pos == std::string_view::npos) return false;

    const auto eq_pos = line.find('=');
    if (eq_pos == std::string_view::npos || eq_pos > semicolon_pos) return false;
    if (line.find('=', eq_pos + 1) != std::string_view::npos) return false;
    if (eq_pos > 0) {
        const char prev = line[eq_pos - 1];
        if (prev == '=' || prev == '<' || prev == '>' || prev == '!' || prev == '+' ||
            prev == '-' || prev == '*' || prev == '/' || prev == '%') {
            return false;
        }
    }
    if (eq_pos + 1 < line.size() && line[eq_pos + 1] == '=') return false;

    std::string_view parsed_lhs;
    if (! TryParseStandaloneIdentifier(line.substr(0, eq_pos), parsed_lhs)) return false;

    const auto rhs_segment = line.substr(eq_pos + 1, semicolon_pos - eq_pos - 1);
    const auto rhs_first   = rhs_segment.find_first_not_of(" \t\r\n");
    if (rhs_first == std::string_view::npos) return false;

    std::string_view parsed_rhs;
    if (! TryParseStandaloneIdentifier(rhs_segment, parsed_rhs)) return false;

    lhs       = parsed_lhs;
    rhs       = parsed_rhs;
    rhs_begin = eq_pos + 1 + rhs_first;
    rhs_len   = parsed_rhs.size();
    return true;
}

inline size_t FindStatementTerminator(std::string_view text, size_t pos) {
    bool in_block_comment { false };
    bool in_string { false };
    bool escaped { false };
    char quote { '\0' };
    int  paren_depth { 0 };
    int  bracket_depth { 0 };

    while (pos < text.size()) {
        const char ch   = text[pos];
        const char next = pos + 1 < text.size() ? text[pos + 1] : '\0';

        if (in_block_comment) {
            if (ch == '*' && next == '/') {
                in_block_comment = false;
                pos += 2;
                continue;
            }
            pos++;
            continue;
        }

        if (in_string) {
            if (escaped) {
                escaped = false;
                pos++;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                pos++;
                continue;
            }
            if (ch == quote) in_string = false;
            pos++;
            continue;
        }

        if (ch == '/' && next == '/') {
            const auto line_end = text.find('\n', pos);
            return line_end == std::string_view::npos ? std::string::npos : line_end;
        }
        if (ch == '/' && next == '*') {
            in_block_comment = true;
            pos += 2;
            continue;
        }
        if (ch == '"' || ch == '\'') {
            in_string = true;
            quote     = ch;
            pos++;
            continue;
        }

        if (ch == '(') paren_depth++;
        else if (ch == ')' && paren_depth > 0) paren_depth--;
        else if (ch == '[') bracket_depth++;
        else if (ch == ']' && bracket_depth > 0) bracket_depth--;
        else if (ch == ';' && paren_depth == 0 && bracket_depth == 0) return pos;

        pos++;
    }

    return std::string::npos;
}

inline std::vector<std::string_view> SplitTopLevelArgs(std::string_view args);

inline std::string RewriteSimpleShaderStatements(std::string_view text) {
    std::string result;
    result.reserve(text.size());

    size_t cursor { 0 };
    size_t pos { 0 };
    bool   in_block_comment { false };
    bool   in_string { false };
    bool   escaped { false };
    char   quote { '\0' };
    int    paren_depth { 0 };
    int    bracket_depth { 0 };

    while (pos < text.size()) {
        const char ch   = text[pos];
        const char next = pos + 1 < text.size() ? text[pos + 1] : '\0';

        if (in_block_comment) {
            if (ch == '*' && next == '/') {
                in_block_comment = false;
                pos += 2;
                continue;
            }
            pos++;
            continue;
        }

        if (in_string) {
            if (escaped) {
                escaped = false;
                pos++;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                pos++;
                continue;
            }
            if (ch == quote) in_string = false;
            pos++;
            continue;
        }

        if (ch == '/' && next == '/') {
            const auto line_end = text.find('\n', pos);
            pos                 = line_end == std::string_view::npos ? text.size() : line_end + 1;
            continue;
        }
        if (ch == '/' && next == '*') {
            in_block_comment = true;
            pos += 2;
            continue;
        }
        if (ch == '"' || ch == '\'') {
            in_string = true;
            quote     = ch;
            pos++;
            continue;
        }

        if (ch == '(') {
            paren_depth++;
            pos++;
            continue;
        }
        if (ch == ')') {
            if (paren_depth > 0) paren_depth--;
            pos++;
            continue;
        }
        if (ch == '[') {
            bracket_depth++;
            pos++;
            continue;
        }
        if (ch == ']') {
            if (bracket_depth > 0) bracket_depth--;
            pos++;
            continue;
        }

        if ((paren_depth != 0 || bracket_depth != 0 || ! IsIdentifierStart(ch))) {
            pos++;
            continue;
        }

        const auto ident_end = SkipIdentifier(text, pos);
        const auto type      = text.substr(pos, ident_end - pos);
        if (type != "float" && type != "vec2" && type != "vec3") {
            pos = ident_end;
            continue;
        }

        const auto stmt_end = FindStatementTerminator(text, pos);
        if (stmt_end == std::string::npos) {
            pos = ident_end;
            continue;
        }

        std::string_view lhs;
        std::string_view rhs;
        std::string_view parsed_type;
        const auto       statement = text.substr(pos, stmt_end - pos + 1);
        if (! TryParseSimpleTypedAssignment(statement, parsed_type, lhs, rhs)) {
            pos = ident_end;
            continue;
        }

        std::string replacement;
        if (parsed_type == "float") {
            const auto rhs_trim = TrimWhitespace(rhs);
            const auto star_pos = rhs_trim.find('*');
            if (star_pos != std::string_view::npos &&
                rhs_trim.find('*', star_pos + 1) == std::string_view::npos) {
                const auto left  = TrimWhitespace(rhs_trim.substr(0, star_pos));
                const auto right = TrimWhitespace(rhs_trim.substr(star_pos + 1));

                std::string_view right_ident;
                if (TryParseStandaloneIdentifier(right, right_ident) &&
                    (left == "g_PointerPosition" || left == "g_PointerPosition.xy")) {
                    replacement = "vec2 " + std::string(lhs) + " = " + std::string(left) +
                                  " * " + std::string(right_ident) + ";";
                }
            }
        } else if (parsed_type == "vec2") {
            std::string_view rhs_ident;
            if (TryParseStandaloneIdentifier(rhs, rhs_ident)) {
                replacement = "vec2 " + std::string(lhs) + " = " + std::string(rhs_ident) + ".xy;";
            } else {
                std::string_view ctor;
                std::string_view args;
                if (TryParseConstructorCall(rhs, ctor, args) && (ctor == "vec3" || ctor == "vec4")) {
                    replacement = "vec2 " + std::string(lhs) + " = vec2(" + std::string(args) + ");";
                }
            }
        } else if (parsed_type == "vec3") {
            std::string_view rhs_ident;
            if (TryParseStandaloneIdentifier(rhs, rhs_ident)) {
                replacement = "vec3 " + std::string(lhs) + " = " + std::string(rhs_ident) + ".xyz;";
            } else {
                std::string_view ctor;
                std::string_view args;
                if (TryParseConstructorCall(rhs, ctor, args) && ctor == "vec4") {
                    replacement = "vec3 " + std::string(lhs) + " = vec3(" + std::string(args) + ");";
                }
            }
        }

        if (replacement.empty()) {
            pos = ident_end;
            continue;
        }

        result.append(text.substr(cursor, pos - cursor));
        result.append(replacement);
        cursor = stmt_end + 1;
        pos    = cursor;
    }

    result.append(text.substr(cursor));
    return result;
}

inline bool IsSwizzleChar(char ch) {
    switch (ch) {
    case 'x':
    case 'y':
    case 'z':
    case 'w':
    case 'r':
    case 'g':
    case 'b':
    case 'a': return true;
    default: return false;
    }
}

inline bool TryParseSwizzledLValue(std::string_view source, std::string_view& base,
                                   std::string_view& swizzle) {
    source = TrimWhitespace(source);
    if (source.empty() || ! IsIdentifierStart(source.front())) return false;

    const auto base_end = SkipIdentifier(source, 0);
    if (base_end >= source.size() || source[base_end] != '.') return false;

    const auto swizzle_begin = base_end + 1;
    if (swizzle_begin >= source.size() || ! IsSwizzleChar(source[swizzle_begin])) return false;

    size_t swizzle_end = swizzle_begin;
    while (swizzle_end < source.size() && IsSwizzleChar(source[swizzle_end])) {
        swizzle_end++;
    }

    swizzle = source.substr(swizzle_begin, swizzle_end - swizzle_begin);
    if (swizzle.size() < 2 || swizzle.size() > 4 || swizzle_end != source.size()) return false;

    base = source.substr(0, base_end);
    return true;
}

inline std::optional<size_t> FindTopLevelAssignmentOperator(std::string_view text) {
    bool in_string { false };
    bool escaped { false };
    char quote { '\0' };
    int  paren_depth { 0 };
    int  bracket_depth { 0 };
    int  brace_depth { 0 };

    for (size_t pos = 0; pos < text.size(); pos++) {
        const char ch = text[pos];

        if (in_string) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                continue;
            }
            if (ch == quote) in_string = false;
            continue;
        }

        if (ch == '"' || ch == '\'') {
            in_string = true;
            quote     = ch;
            continue;
        }

        if (ch == '(') paren_depth++;
        else if (ch == ')' && paren_depth > 0) paren_depth--;
        else if (ch == '[') bracket_depth++;
        else if (ch == ']' && bracket_depth > 0) bracket_depth--;
        else if (ch == '{') brace_depth++;
        else if (ch == '}' && brace_depth > 0) brace_depth--;
        else if (ch == '=' && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
            const char prev = pos > 0 ? text[pos - 1] : '\0';
            const char next = pos + 1 < text.size() ? text[pos + 1] : '\0';
            if (prev == '=' || prev == '!' || prev == '<' || prev == '>') continue;
            if (next == '=') continue;
            return pos;
        }
    }

    return std::nullopt;
}

inline bool TryRewriteSwizzledSelfMixAssignment(std::string_view statement,
                                                std::string&     replacement) {
    statement = TrimWhitespace(statement);
    if (statement.empty() || statement.back() != ';') return false;
    statement.remove_suffix(1);

    const auto assignment_pos = FindTopLevelAssignmentOperator(statement);
    if (! assignment_pos.has_value()) return false;

    std::string_view base;
    std::string_view swizzle;
    if (! TryParseSwizzledLValue(statement.substr(0, *assignment_pos), base, swizzle)) return false;

    std::string_view function_name;
    std::string_view call_args_source;
    if (! TryParseConstructorCall(statement.substr(*assignment_pos + 1), function_name, call_args_source)) {
        return false;
    }
    if (function_name != "mix" && function_name != "lerp") return false;

    const auto args = SplitTopLevelArgs(call_args_source);
    if (args.size() != 3) return false;

    std::array<std::string, 3> rewritten_args;
    bool                       changed { false };
    for (size_t i = 0; i < args.size(); i++) {
        const auto trimmed_arg = TrimWhitespace(args[i]);
        std::string_view ident;
        if (i < 2 && TryParseStandaloneIdentifier(trimmed_arg, ident) && ident == base) {
            rewritten_args[i] = std::string(base) + "." + std::string(swizzle);
            changed           = true;
        } else {
            rewritten_args[i] = std::string(trimmed_arg);
        }
    }

    if (! changed) return false;

    replacement = std::string(TrimWhitespace(statement.substr(0, *assignment_pos))) + " = " +
                  std::string(function_name) + "(" + rewritten_args[0] + ", " +
                  rewritten_args[1] + ", " + rewritten_args[2] + ");";
    return true;
}

inline std::string RewriteSwizzledSelfMixAssignments(std::string_view text) {
    std::string result;
    result.reserve(text.size());

    size_t cursor { 0 };
    size_t pos { 0 };
    while (pos < text.size()) {
        const auto stmt_end = FindStatementTerminator(text, pos);
        if (stmt_end == std::string::npos) break;

        std::string replacement;
        if (TryRewriteSwizzledSelfMixAssignment(text.substr(pos, stmt_end - pos + 1), replacement)) {
            result.append(text.substr(cursor, pos - cursor));
            result.append(replacement);
            cursor = stmt_end + 1;
        }

        pos = stmt_end + 1;
    }

    result.append(text.substr(cursor));
    return result;
}

inline size_t FindTopLevelChar(std::string_view text, char target, size_t start = 0) {
    bool in_string { false };
    bool escaped { false };
    char quote { '\0' };
    int  paren_depth { 0 };
    int  bracket_depth { 0 };
    int  brace_depth { 0 };

    for (size_t pos = start; pos < text.size(); pos++) {
        const char ch = text[pos];

        if (in_string) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                continue;
            }
            if (ch == quote) in_string = false;
            continue;
        }

        if (ch == '"' || ch == '\'') {
            in_string = true;
            quote     = ch;
            continue;
        }

        if (ch == '(') paren_depth++;
        else if (ch == ')' && paren_depth > 0) paren_depth--;
        else if (ch == '[') bracket_depth++;
        else if (ch == ']' && bracket_depth > 0) bracket_depth--;
        else if (ch == '{') brace_depth++;
        else if (ch == '}' && brace_depth > 0) brace_depth--;
        else if (ch == target && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
            return pos;
        }
    }

    return std::string::npos;
}

inline bool TryRewriteIntegerForLoopHeader(std::string_view header, std::string& replacement) {
    if (! StartsWithToken(header, "for")) return false;

    auto pos = SkipWhitespace(header, 3);
    if (pos >= header.size() || header[pos] != '(') return false;

    const auto close_pos = SkipBalanced(header, pos, '(', ')');
    if (close_pos == std::string::npos || close_pos != header.size()) return false;

    const auto inside = header.substr(pos + 1, close_pos - pos - 2);
    const auto semi1  = FindTopLevelChar(inside, ';');
    if (semi1 == std::string::npos) return false;
    const auto semi2 = FindTopLevelChar(inside, ';', semi1 + 1);
    if (semi2 == std::string::npos) return false;
    if (FindTopLevelChar(inside, ';', semi2 + 1) != std::string::npos) return false;

    const auto init = TrimWhitespace(inside.substr(0, semi1));
    const auto cond = TrimWhitespace(inside.substr(semi1 + 1, semi2 - semi1 - 1));
    const auto inc  = TrimWhitespace(inside.substr(semi2 + 1));
    if (init.empty() || cond.empty() || inc.empty()) return false;

    if (! StartsWithToken(init, "int")) return false;
    pos = SkipWhitespace(init, 3);
    if (pos >= init.size() || ! IsIdentifierStart(init[pos])) return false;
    const auto name_end = SkipIdentifier(init, pos);
    const auto name     = init.substr(pos, name_end - pos);

    pos = SkipWhitespace(init, name_end);
    if (pos >= init.size() || init[pos] != '=') return false;
    const auto init_expr = TrimWhitespace(init.substr(pos + 1));
    if (init_expr.empty()) return false;

    pos = 0;
    if (! MatchIdentifierAt(cond, 0, name)) return false;
    pos = SkipWhitespace(cond, name.size());

    std::string_view op;
    if (pos + 1 < cond.size() && (cond.substr(pos, 2) == "<=" || cond.substr(pos, 2) == ">=")) {
        op = cond.substr(pos, 2);
        pos += 2;
    } else if (pos < cond.size() && (cond[pos] == '<' || cond[pos] == '>')) {
        op = cond.substr(pos, 1);
        pos++;
    } else {
        return false;
    }

    const auto cond_expr = TrimWhitespace(cond.substr(pos));
    if (cond_expr.empty()) return false;

    const auto normalized_inc = RemoveAsciiWhitespace(inc);
    const auto name_str       = std::string(name);
    if (normalized_inc != "++" + name_str && normalized_inc != name_str + "++") return false;

    replacement = "for (int " + name_str + " = int(" + std::string(init_expr) + "); " +
                  name_str + " " + std::string(op) + " int(" + std::string(cond_expr) + "); " +
                  name_str + "++)";
    return true;
}

inline std::string RewriteIntegerForLoops(std::string_view text) {
    std::string result;
    result.reserve(text.size());

    size_t cursor { 0 };
    size_t pos { 0 };
    bool   in_block_comment { false };
    bool   in_string { false };
    bool   escaped { false };
    char   quote { '\0' };

    while (pos < text.size()) {
        const char ch   = text[pos];
        const char next = pos + 1 < text.size() ? text[pos + 1] : '\0';

        if (in_block_comment) {
            if (ch == '*' && next == '/') {
                in_block_comment = false;
                pos += 2;
                continue;
            }
            pos++;
            continue;
        }

        if (in_string) {
            if (escaped) {
                escaped = false;
                pos++;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                pos++;
                continue;
            }
            if (ch == quote) in_string = false;
            pos++;
            continue;
        }

        if (ch == '/' && next == '/') {
            const auto line_end = text.find('\n', pos);
            pos                 = line_end == std::string_view::npos ? text.size() : line_end + 1;
            continue;
        }
        if (ch == '/' && next == '*') {
            in_block_comment = true;
            pos += 2;
            continue;
        }
        if (ch == '"' || ch == '\'') {
            in_string = true;
            quote     = ch;
            pos++;
            continue;
        }

        if (! MatchIdentifierAt(text, pos, "for")) {
            pos++;
            continue;
        }

        auto header_pos = SkipWhitespace(text, pos + 3);
        if (header_pos >= text.size() || text[header_pos] != '(') {
            pos += 3;
            continue;
        }

        const auto header_end = SkipBalanced(text, header_pos, '(', ')');
        if (header_end == std::string::npos) {
            pos += 3;
            continue;
        }

        std::string replacement;
        const auto  header = text.substr(pos, header_end - pos);
        if (! TryRewriteIntegerForLoopHeader(header, replacement)) {
            pos += 3;
            continue;
        }

        result.append(text.substr(cursor, pos - cursor));
        result.append(replacement);
        cursor = header_end;
        pos    = header_end;
    }

    result.append(text.substr(cursor));
    return result;
}

inline bool ContainsString(const std::vector<std::string>& values, std::string_view needle) {
    for (const auto& value : values) {
        if (value == needle) return true;
    }
    return false;
}

inline bool IsUnsignedIntegerLiteral(std::string_view source) {
    source = TrimWhitespace(source);
    if (source.size() < 2) return false;
    const char suffix = source.back();
    if (suffix != 'u' && suffix != 'U') return false;

    source.remove_suffix(1);
    if (source.empty()) return false;

    if (source.size() > 2 && source[0] == '0' && (source[1] == 'x' || source[1] == 'X')) {
        for (size_t i = 2; i < source.size(); i++) {
            if (! std::isxdigit(static_cast<unsigned char>(source[i]))) return false;
        }
        return source.size() > 2;
    }

    for (const char ch : source) {
        if (! std::isdigit(static_cast<unsigned char>(ch))) return false;
    }
    return true;
}

inline std::string RewriteIntegerLiteralsAsUnsigned(std::string_view source) {
    std::string result;
    result.reserve(source.size() + 4);

    size_t pos { 0 };
    while (pos < source.size()) {
        const char ch = source[pos];
        if (! std::isdigit(static_cast<unsigned char>(ch)) ||
            (pos > 0 && (IsIdentifierContinue(source[pos - 1]) || source[pos - 1] == '.'))) {
            result.push_back(ch);
            pos++;
            continue;
        }

        const size_t literal_begin = pos;
        bool         hexadecimal { false };
        bool         floating_point { false };
        if (pos + 1 < source.size() && source[pos] == '0' &&
            (source[pos + 1] == 'x' || source[pos + 1] == 'X')) {
            hexadecimal = true;
            pos += 2;
            while (pos < source.size() &&
                   std::isxdigit(static_cast<unsigned char>(source[pos]))) {
                pos++;
            }
        } else {
            while (pos < source.size() &&
                   std::isdigit(static_cast<unsigned char>(source[pos]))) {
                pos++;
            }
            if (pos < source.size() && source[pos] == '.') {
                floating_point = true;
                pos++;
                while (pos < source.size() &&
                       std::isdigit(static_cast<unsigned char>(source[pos]))) {
                    pos++;
                }
            }
            if (pos < source.size() && (source[pos] == 'e' || source[pos] == 'E')) {
                floating_point = true;
                pos++;
                if (pos < source.size() && (source[pos] == '+' || source[pos] == '-')) pos++;
                while (pos < source.size() &&
                       std::isdigit(static_cast<unsigned char>(source[pos]))) {
                    pos++;
                }
            }
        }

        const size_t suffix_begin = pos;
        while (pos < source.size() && IsIdentifierContinue(source[pos])) {
            pos++;
        }

        const auto literal = source.substr(literal_begin, pos - literal_begin);
        const auto suffix  = source.substr(suffix_begin, pos - suffix_begin);
        result.append(literal);
        // Wallpaper Engine compatibility shaders freely mix uint variables with unsuffixed
        // integer constants. glslang keeps `%` and neighboring integer arithmetic strictly typed,
        // so constants inside a uint modulo operand need an unsigned suffix before the operand is
        // wrapped or the inner expression can still fail as `uint + int`.
        if (! floating_point && (hexadecimal || suffix.empty()) &&
            suffix.find_first_of("uU") == std::string_view::npos) {
            result.push_back('u');
        }
    }

    return result;
}

inline size_t FindMatchingOpenBackward(std::string_view source, size_t close_pos, char open_ch,
                                       char close_ch) {
    if (close_pos >= source.size() || source[close_pos] != close_ch) return std::string::npos;

    int depth { 0 };
    for (size_t pos = close_pos + 1; pos > 0; pos--) {
        const size_t i = pos - 1;
        if (source[i] == close_ch) {
            depth++;
        } else if (source[i] == open_ch) {
            depth--;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

inline size_t IncludeIdentifierBeforeOperand(std::string_view source, size_t operand_begin) {
    size_t cursor = operand_begin;
    while (cursor > 0 && std::isspace(static_cast<unsigned char>(source[cursor - 1]))) {
        cursor--;
    }
    if (cursor == 0 || ! IsIdentifierContinue(source[cursor - 1])) return operand_begin;

    size_t ident_begin = cursor;
    while (ident_begin > 0 && IsIdentifierContinue(source[ident_begin - 1])) {
        ident_begin--;
    }
    return ident_begin;
}

inline size_t FindModuloLeftOperandBegin(std::string_view source, size_t modulo_pos) {
    size_t cursor = modulo_pos;
    while (cursor > 0 && std::isspace(static_cast<unsigned char>(source[cursor - 1]))) {
        cursor--;
    }
    if (cursor == 0) return std::string::npos;

    const char ch = source[cursor - 1];
    if (ch == ')') {
        const auto open = FindMatchingOpenBackward(source, cursor - 1, '(', ')');
        if (open == std::string::npos) return std::string::npos;
        return IncludeIdentifierBeforeOperand(source, open);
    }
    if (ch == ']') {
        const auto open = FindMatchingOpenBackward(source, cursor - 1, '[', ']');
        if (open == std::string::npos) return std::string::npos;
        return IncludeIdentifierBeforeOperand(source, open);
    }

    size_t begin = cursor;
    while (begin > 0) {
        const char prev = source[begin - 1];
        if (! IsIdentifierContinue(prev) && prev != '.') break;
        begin--;
    }
    return begin == cursor ? std::string::npos : begin;
}

inline size_t FindModuloRightOperandEnd(std::string_view source, size_t operand_pos) {
    size_t cursor = SkipWhitespace(source, operand_pos);
    if (cursor >= source.size()) return std::string::npos;

    if (IsIdentifierStart(source[cursor])) {
        cursor = SkipIdentifier(source, cursor);
        auto after_identifier = SkipWhitespace(source, cursor);
        if (after_identifier < source.size() && source[after_identifier] == '(') {
            const auto close = SkipBalanced(source, after_identifier, '(', ')');
            if (close == std::string::npos) return std::string::npos;
            cursor = close;
        }

        for (;;) {
            auto next = SkipWhitespace(source, cursor);
            if (next < source.size() && source[next] == '.') {
                next++;
                if (next >= source.size() || ! IsIdentifierStart(source[next])) break;
                cursor = SkipIdentifier(source, next);
                continue;
            }
            if (next < source.size() && source[next] == '[') {
                const auto close = SkipBalanced(source, next, '[', ']');
                if (close == std::string::npos) return std::string::npos;
                cursor = close;
                continue;
            }
            break;
        }
        return cursor;
    }

    if (source[cursor] == '(') {
        return SkipBalanced(source, cursor, '(', ')');
    }

    if (std::isdigit(static_cast<unsigned char>(source[cursor]))) {
        bool hexadecimal { false };
        if (cursor + 1 < source.size() && source[cursor] == '0' &&
            (source[cursor + 1] == 'x' || source[cursor + 1] == 'X')) {
            hexadecimal = true;
            cursor += 2;
            while (cursor < source.size() &&
                   std::isxdigit(static_cast<unsigned char>(source[cursor]))) {
                cursor++;
            }
        } else {
            while (cursor < source.size() &&
                   std::isdigit(static_cast<unsigned char>(source[cursor]))) {
                cursor++;
            }
            if (cursor < source.size() && source[cursor] == '.') {
                cursor++;
                while (cursor < source.size() &&
                       std::isdigit(static_cast<unsigned char>(source[cursor]))) {
                    cursor++;
                }
            }
            if (cursor < source.size() && (source[cursor] == 'e' || source[cursor] == 'E')) {
                cursor++;
                if (cursor < source.size() && (source[cursor] == '+' || source[cursor] == '-')) {
                    cursor++;
                }
                while (cursor < source.size() &&
                       std::isdigit(static_cast<unsigned char>(source[cursor]))) {
                    cursor++;
                }
            }
        }
        while (cursor < source.size() && IsIdentifierContinue(source[cursor])) {
            cursor++;
        }
        (void)hexadecimal;
        return cursor;
    }

    return std::string::npos;
}

inline std::string NormalizeUintModuloOperand(std::string_view operand) {
    std::string normalized = RewriteIntegerLiteralsAsUnsigned(TrimWhitespace(operand));
    const auto  trimmed    = TrimWhitespace(normalized);
    if (trimmed.empty()) return normalized;
    if (IsUnsignedIntegerLiteral(trimmed)) return std::string(trimmed);

    std::string_view ctor;
    std::string_view args;
    if (TryParseConstructorCall(trimmed, ctor, args) && ctor == "uint") {
        return std::string(trimmed);
    }

    return "uint(" + std::string(trimmed) + ")";
}

inline bool RewriteModuloOperandsForUint(std::string& expression) {
    bool   changed { false };
    size_t pos { 0 };

    while ((pos = expression.find('%', pos)) != std::string::npos) {
        if (pos + 1 < expression.size() && expression[pos + 1] == '=') {
            pos++;
            continue;
        }

        const auto lhs_begin = FindModuloLeftOperandBegin(expression, pos);
        const auto rhs_end   = FindModuloRightOperandEnd(expression, pos + 1);
        if (lhs_begin == std::string::npos || rhs_end == std::string::npos ||
            lhs_begin >= pos || rhs_end <= pos + 1) {
            pos++;
            continue;
        }

        const auto lhs = std::string_view(expression).substr(lhs_begin, pos - lhs_begin);
        const auto rhs = std::string_view(expression).substr(pos + 1, rhs_end - pos - 1);
        const auto replacement =
            NormalizeUintModuloOperand(lhs) + " % " + NormalizeUintModuloOperand(rhs);
        expression.replace(lhs_begin, rhs_end - lhs_begin, replacement);
        pos = lhs_begin + replacement.size();
        changed = true;
    }

    return changed;
}

inline bool TryParseUintDeclarationLhs(std::string_view lhs, std::string_view& name) {
    lhs = TrimWhitespace(lhs);
    if (lhs.empty()) return false;

    size_t pos { 0 };
    while (pos < lhs.size()) {
        if (! IsIdentifierStart(lhs[pos])) return false;

        const auto token_end = SkipIdentifier(lhs, pos);
        const auto token     = lhs.substr(pos, token_end - pos);
        if (token == "layout") {
            pos = SkipWhitespace(lhs, token_end);
            if (pos >= lhs.size() || lhs[pos] != '(') return false;
            pos = SkipBalanced(lhs, pos, '(', ')');
            if (pos == std::string::npos) return false;
            pos = SkipWhitespace(lhs, pos);
            continue;
        }

        if (IsShaderDeclarationQualifier(token)) {
            pos = SkipWhitespace(lhs, token_end);
            continue;
        }

        if (token != "uint") return false;
        pos = SkipWhitespace(lhs, token_end);
        if (pos >= lhs.size() || ! IsIdentifierStart(lhs[pos])) return false;

        const auto name_end = SkipIdentifier(lhs, pos);
        name                = lhs.substr(pos, name_end - pos);
        auto tail           = SkipWhitespace(lhs, name_end);
        if (tail < lhs.size() && lhs[tail] == '[') {
            tail = SkipBalanced(lhs, tail, '[', ']');
            if (tail == std::string::npos) return false;
            tail = SkipWhitespace(lhs, tail);
        }
        return tail == lhs.size();
    }

    return false;
}

inline bool TryParseUintDeclarationStatement(std::string_view statement, std::string& name) {
    statement = TrimWhitespace(statement);
    if (statement.empty() || statement.back() != ';') return false;
    statement.remove_suffix(1);

    const auto assignment_pos = FindTopLevelAssignmentOperator(statement);
    const auto lhs = assignment_pos.has_value()
        ? statement.substr(0, *assignment_pos)
        : statement;

    std::string_view parsed_name;
    if (! TryParseUintDeclarationLhs(lhs, parsed_name)) return false;
    name = std::string(parsed_name);
    return true;
}

inline bool TryRewriteUintModuloStatement(std::string_view statement,
                                          const std::vector<std::string>& uint_variables,
                                          std::string& replacement) {
    const auto leading = statement.find_first_not_of(" \t\r\n");
    const auto prefix = leading == std::string_view::npos ? std::string_view {} :
        statement.substr(0, leading);

    statement = TrimWhitespace(statement);
    if (statement.empty() || statement.back() != ';') return false;
    statement.remove_suffix(1);

    const auto assignment_pos = FindTopLevelAssignmentOperator(statement);
    if (! assignment_pos.has_value()) return false;

    const auto lhs = TrimWhitespace(statement.substr(0, *assignment_pos));
    const auto rhs = TrimWhitespace(statement.substr(*assignment_pos + 1));
    if (lhs.empty() || rhs.find('%') == std::string_view::npos) return false;

    std::string_view declared_uint_name;
    std::string_view assigned_name;
    const bool target_is_uint =
        TryParseUintDeclarationLhs(lhs, declared_uint_name) ||
        (TryParseStandaloneIdentifier(lhs, assigned_name) &&
         ContainsString(uint_variables, assigned_name));
    if (! target_is_uint) return false;

    std::string rewritten_rhs(rhs);
    if (! RewriteModuloOperandsForUint(rewritten_rhs)) return false;

    // Wallpaper Engine's shader compatibility layer allows values such as
    // `uint wrapped = floatFrequency % 64;`. GLSL requires `%` operands to be integral, and
    // unsigned arithmetic must use unsigned constants. Rewriting every uint-target modulo
    // assignment here keeps the behavior generic for workshop shaders without adding
    // per-shader source exceptions.
    replacement = std::string(prefix) + std::string(lhs) + " = " + rewritten_rhs + ";";
    return true;
}

inline std::string RewriteUintModuloAssignments(std::string_view text) {
    std::string result;
    result.reserve(text.size() + 64);

    std::vector<std::string> uint_variables;
    size_t                   cursor { 0 };
    size_t                   pos { 0 };
    while (pos < text.size()) {
        const auto stmt_end = FindStatementTerminator(text, pos);
        if (stmt_end == std::string::npos) break;

        const auto statement = text.substr(pos, stmt_end - pos + 1);
        std::string replacement;
        if (TryRewriteUintModuloStatement(statement, uint_variables, replacement)) {
            result.append(text.substr(cursor, pos - cursor));
            result.append(replacement);
            cursor = stmt_end + 1;
        }

        std::string declared_name;
        if (TryParseUintDeclarationStatement(statement, declared_name) &&
            ! ContainsString(uint_variables, declared_name)) {
            uint_variables.push_back(declared_name);
        }

        pos = stmt_end + 1;
    }

    result.append(text.substr(cursor));
    return result;
}

inline bool TryParseScalarDeclarationStatement(std::string_view statement, std::string_view type,
                                               std::string& name) {
    statement = TrimWhitespace(statement);
    if (statement.empty() || statement.back() != ';') return false;
    statement.remove_suffix(1);

    const auto assignment_pos = FindTopLevelAssignmentOperator(statement);
    const auto lhs = assignment_pos.has_value()
        ? TrimWhitespace(statement.substr(0, *assignment_pos))
        : TrimWhitespace(statement);
    if (! StartsWithToken(lhs, type)) return false;

    auto pos = SkipWhitespace(lhs, type.size());
    if (pos >= lhs.size() || ! IsIdentifierStart(lhs[pos])) return false;

    const auto name_end = SkipIdentifier(lhs, pos);
    name                = std::string(lhs.substr(pos, name_end - pos));
    pos                 = SkipWhitespace(lhs, name_end);
    return pos == lhs.size();
}

inline std::optional<size_t> FindTopLevelCompoundAssignment(std::string_view text,
                                                            std::string_view op) {
    bool in_string { false };
    bool escaped { false };
    char quote { '\0' };
    int  paren_depth { 0 };
    int  bracket_depth { 0 };
    int  brace_depth { 0 };

    for (size_t pos = 0; pos + op.size() <= text.size(); pos++) {
        const char ch = text[pos];

        if (in_string) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                continue;
            }
            if (ch == quote) in_string = false;
            continue;
        }

        if (ch == '"' || ch == '\'') {
            in_string = true;
            quote     = ch;
            continue;
        }

        if (ch == '(') paren_depth++;
        else if (ch == ')' && paren_depth > 0) paren_depth--;
        else if (ch == '[') bracket_depth++;
        else if (ch == ']' && bracket_depth > 0) bracket_depth--;
        else if (ch == '{') brace_depth++;
        else if (ch == '}' && brace_depth > 0) brace_depth--;
        else if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 &&
                 text.substr(pos, op.size()) == op) {
            return pos;
        }
    }

    return std::nullopt;
}

inline bool TryRewriteFloatBoolMultiplyStatement(std::string_view statement,
                                                 const std::vector<std::string>& float_variables,
                                                 const std::vector<std::string>& bool_variables,
                                                 std::string& replacement) {
    const auto leading = statement.find_first_not_of(" \t\r\n");
    const auto prefix = leading == std::string_view::npos ? std::string_view {} :
        statement.substr(0, leading);

    statement = TrimWhitespace(statement);
    if (statement.empty() || statement.back() != ';') return false;
    statement.remove_suffix(1);

    const auto op_pos = FindTopLevelCompoundAssignment(statement, "*=");
    if (! op_pos.has_value()) return false;

    const auto lhs = TrimWhitespace(statement.substr(0, *op_pos));
    const auto rhs = TrimWhitespace(statement.substr(*op_pos + 2));

    std::string_view lhs_name;
    std::string_view rhs_name;
    if (! TryParseStandaloneIdentifier(lhs, lhs_name) ||
        ! TryParseStandaloneIdentifier(rhs, rhs_name)) {
        return false;
    }
    if (! ContainsString(float_variables, lhs_name) || ! ContainsString(bool_variables, rhs_name)) {
        return false;
    }

    // Wallpaper Engine compatibility shaders use bool values as 0/1 masks in scalar math, for
    // example `floatMask *= boolCondition`. Desktop GLSL does not implicitly convert bool to
    // float, so this pass makes only the proven float-times-bool compound assignment explicit
    // instead of changing unrelated boolean logic.
    replacement = std::string(prefix) + std::string(lhs_name) + " *= float(" +
                  std::string(rhs_name) + ");";
    return true;
}

inline std::string RewriteFloatBoolMultiplyAssignments(std::string_view text) {
    std::string result;
    result.reserve(text.size() + 64);

    std::vector<std::string> float_variables;
    std::vector<std::string> bool_variables;
    size_t                   cursor { 0 };
    size_t                   pos { 0 };
    while (pos < text.size()) {
        const auto stmt_end = FindStatementTerminator(text, pos);
        if (stmt_end == std::string::npos) break;

        const auto statement = text.substr(pos, stmt_end - pos + 1);
        std::string replacement;
        if (TryRewriteFloatBoolMultiplyStatement(
                statement, float_variables, bool_variables, replacement)) {
            result.append(text.substr(cursor, pos - cursor));
            result.append(replacement);
            cursor = stmt_end + 1;
        }

        std::string declared_name;
        if (TryParseScalarDeclarationStatement(statement, "float", declared_name) &&
            ! ContainsString(float_variables, declared_name)) {
            float_variables.push_back(declared_name);
        } else if (TryParseScalarDeclarationStatement(statement, "bool", declared_name) &&
                   ! ContainsString(bool_variables, declared_name)) {
            bool_variables.push_back(declared_name);
        }

        pos = stmt_end + 1;
    }

    result.append(text.substr(cursor));
    return result;
}

inline void CollectPreprocessorInfo(const std::string& src, WPPreprocessorInfo& process_info) {
    process_info.input.clear();
    process_info.output.clear();
    process_info.active_tex_slots.clear();

    size_t pos { 0 };
    while (pos < src.size()) {
        const auto line_end =
            src.find('\n', pos);
        const auto line_len =
            line_end == std::string::npos ? src.size() - pos : line_end - pos;
        const auto line = std::string_view(src).substr(pos, line_len);

        std::string qualifier;
        std::string name;
        if (TryParseInterfaceDeclaration(line, qualifier, name)) {
            if (qualifier == "in") {
                process_info.input[name] = std::string(line);
            } else {
                process_info.output[name] = std::string(line);
            }
        }

        uint slot { 0 };
        if (TryParseTextureSlot(line, slot)) {
            process_info.active_tex_slots.insert(slot);
        }

        if (line_end == std::string::npos) break;
        pos = line_end + 1;
    }
}

inline std::string CommentOutRequireDirectives(const std::string& src) {
    std::string out;
    out.reserve(src.size());

    size_t pos { 0 };
    while (pos < src.size()) {
        const auto line_end =
            src.find('\n', pos);
        const auto line_len =
            line_end == std::string::npos ? src.size() - pos : line_end - pos;
        const auto line = std::string_view(src).substr(pos, line_len);

        const auto first_non_ws = line.find_first_not_of(" \t\r");
        if (first_non_ws != std::string_view::npos &&
            line.substr(first_non_ws).compare(0, 8, "#require") == 0) {
            out.append(line.substr(0, first_non_ws));
            out.append("//");
            out.append(line.substr(first_non_ws));
        } else {
            out.append(line);
        }

        if (line_end == std::string::npos) break;
        out.push_back('\n');
        pos = line_end + 1;
    }
    return out;
}

inline bool IsMutationOperator(std::string_view text, size_t pos) {
    if (pos >= text.size()) return false;

    switch (text[pos]) {
    case '+':
    case '-':
        return pos + 1 < text.size() && (text[pos + 1] == '=' || text[pos + 1] == text[pos]);
    case '*':
    case '/':
    case '%': return pos + 1 < text.size() && text[pos + 1] == '=';
    case '=': return pos + 1 >= text.size() || text[pos + 1] != '=';
    default: return false;
    }
}

inline bool HasMutableUse(std::string_view text, std::string_view name) {
    size_t pos { 0 };
    while (pos < text.size()) {
        pos = text.find(name, pos);
        if (pos == std::string::npos) return false;
        if (! MatchIdentifierAt(text, pos, name)) {
            pos++;
            continue;
        }

        auto cursor = SkipWhitespace(text, pos + name.size());
        while (cursor < text.size()) {
            if (text[cursor] == '.') {
                cursor++;
                if (cursor >= text.size() || ! IsIdentifierStart(text[cursor])) break;
                cursor = SkipIdentifier(text, cursor);
                cursor = SkipWhitespace(text, cursor);
                continue;
            }
            if (text[cursor] == '[') {
                cursor = SkipBalanced(text, cursor, '[', ']');
                if (cursor == std::string::npos) return false;
                cursor = SkipWhitespace(text, cursor);
                continue;
            }
            break;
        }

        if (IsMutationOperator(text, cursor)) return true;
        pos += name.size();
    }
    return false;
}

inline bool MakeMutableInputShim(std::string_view decl, std::string_view name,
                                 std::string& mutable_name, std::string& init_line) {
    const auto first_non_ws = decl.find_first_not_of(" \t");
    const auto indent =
        first_non_ws == std::string_view::npos ? std::string_view {} : decl.substr(0, first_non_ws);

    size_t in_pos { std::string::npos };
    size_t in_end { std::string::npos };
    size_t pos { 0 };
    while (pos < decl.size()) {
        if (! IsIdentifierStart(decl[pos])) {
            pos++;
            continue;
        }

        const auto end_pos = SkipIdentifier(decl, pos);
        if (decl.substr(pos, end_pos - pos) == "in") {
            in_pos = pos;
            in_end = end_pos;
            break;
        }
        pos = end_pos;
    }
    if (in_pos == std::string::npos) return false;

    const auto type_pos = SkipWhitespace(decl, in_end);
    if (type_pos >= decl.size() || ! IsIdentifierStart(decl[type_pos])) return false;

    const auto type_end = SkipIdentifier(decl, type_pos);
    const auto name_pos = SkipWhitespace(decl, type_end);
    if (name_pos >= decl.size() || ! IsIdentifierStart(decl[name_pos])) return false;

    const auto decl_name_end = SkipIdentifier(decl, name_pos);
    const auto decl_name     = decl.substr(name_pos, decl_name_end - name_pos);
    if (decl_name != name) return false;

    auto        tail_pos = SkipWhitespace(decl, decl_name_end);
    std::string suffix;
    if (tail_pos < decl.size() && decl[tail_pos] == '[') {
        const auto suffix_end = SkipBalanced(decl, tail_pos, '[', ']');
        if (suffix_end == std::string::npos) return false;
        suffix = std::string(decl.substr(tail_pos, suffix_end - tail_pos));
        tail_pos = SkipWhitespace(decl, suffix_end);
    }

    if (tail_pos >= decl.size() || decl[tail_pos] != ';') return false;

    mutable_name = "_wp_mutable_" + std::string(name);
    init_line    = std::string(indent) + std::string(decl.substr(type_pos, type_end - type_pos)) +
                " " + mutable_name + suffix + " = " + std::string(name) + ";\n";
    return true;
}

inline std::string RepairMissingStringQuotes(std::string_view source) {
    std::string repaired(source);
    bool        changed { false };
    bool        in_string { false };
    bool        escaped { false };
    char        string_context { '\0' };

    for (size_t i = 0; i < repaired.size(); i++) {
        const char ch = repaired[i];

        if (! in_string) {
            if (ch == '"') {
                in_string      = true;
                escaped        = false;
                string_context = PrevNonWhitespace(repaired, i);
            }
            continue;
        }

        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            in_string = false;
            continue;
        }

        // Wallpaper Engine workshop shaders occasionally omit the closing quote
        // in inline metadata comments. Only attempt a conservative repair after
        // strict parsing fails, and only for string values that appear to run
        // into the next object key or the end of the object.
        if (string_context != ':') continue;

        if (ch == ',' && LooksLikeObjectKeyAfterComma(repaired, i + 1)) {
            repaired.insert(i, 1, '"');
            changed   = true;
            in_string = false;
            i++;
            continue;
        }

        if ((ch == '}' || ch == ']') && LooksLikeContainerClose(repaired, i)) {
            repaired.insert(i, 1, '"');
            changed   = true;
            in_string = false;
            i++;
        }
    }

    if (! changed) return {};
    return repaired;
}

inline bool IsIdentChar(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

inline std::optional<size_t> FindMatchingParen(std::string_view source, size_t open_pos) {
    if (open_pos >= source.size() || source[open_pos] != '(') return std::nullopt;

    int depth { 1 };
    for (size_t i = open_pos + 1; i < source.size(); i++) {
        if (source[i] == '(') depth++;
        else if (source[i] == ')') {
            depth--;
            if (depth == 0) return i;
        }
    }
    return std::nullopt;
}

inline std::string TrimCopy(std::string_view value) {
    size_t begin = 0;
    size_t end   = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) begin++;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) end--;
    return std::string(value.substr(begin, end - begin));
}

inline std::vector<std::string_view> SplitTopLevelArgs(std::string_view args) {
    std::vector<std::string_view> result;
    size_t                        begin { 0 };
    int                           paren_depth { 0 };
    int                           bracket_depth { 0 };
    int                           brace_depth { 0 };

    for (size_t i = 0; i < args.size(); i++) {
        switch (args[i]) {
        case '(': paren_depth++; break;
        case ')': paren_depth--; break;
        case '[': bracket_depth++; break;
        case ']': bracket_depth--; break;
        case '{': brace_depth++; break;
        case '}': brace_depth--; break;
        case ',':
            if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
                result.push_back(args.substr(begin, i - begin));
                begin = i + 1;
            }
            break;
        default: break;
        }
    }

    result.push_back(args.substr(begin));
    return result;
}

inline std::string TrimOverlongVectorConstructors(std::string_view source) {
    std::string current(source);

    for (;;) {
        bool        changed { false };
        std::string next;
        next.reserve(current.size());

        size_t pos = 0;
        while (pos < current.size()) {
            size_t vec2_pos = current.find("vec2(", pos);
            size_t vec3_pos = current.find("vec3(", pos);

            size_t match_pos = std::string::npos;
            bool   is_vec2 { false };
            if (vec2_pos == std::string::npos) {
                match_pos = vec3_pos;
            } else if (vec3_pos == std::string::npos || vec2_pos < vec3_pos) {
                match_pos = vec2_pos;
                is_vec2   = true;
            } else {
                match_pos = vec3_pos;
            }

            if (match_pos == std::string::npos) {
                next.append(current.substr(pos));
                break;
            }

            if (match_pos > 0 && IsIdentChar(current[match_pos - 1])) {
                next.append(current.substr(pos, match_pos - pos + 1));
                pos = match_pos + 1;
                continue;
            }

            const size_t open_pos = match_pos + 4;
            const auto   close_pos = FindMatchingParen(current, open_pos);
            if (! close_pos.has_value()) {
                next.append(current.substr(pos));
                break;
            }

            next.append(current.substr(pos, match_pos - pos));

            const auto args = SplitTopLevelArgs(
                std::string_view(current).substr(open_pos + 1, *close_pos - open_pos - 1));
            const size_t arg_limit = is_vec2 ? 2 : 3;
            if (args.size() > arg_limit) {
                next.append(is_vec2 ? "vec2(" : "vec3(");
                for (size_t i = 0; i < arg_limit; i++) {
                    if (i > 0) next.append(", ");
                    next.append(TrimCopy(args[i]));
                }
                next.push_back(')');
                changed = true;
            } else {
                next.append(current.substr(match_pos, *close_pos - match_pos + 1));
            }

            pos = *close_pos + 1;
        }

        if (! changed) return current;
        current.swap(next);
    }
}

inline std::string RewriteVec2AssignmentsForVec4TexCoord(std::string_view source) {
    if (source.find("vec4 v_TexCoord") == std::string_view::npos ||
        source.find("vec2 ") == std::string_view::npos ||
        source.find("v_TexCoord") == std::string_view::npos) {
        return std::string(source);
    }

    std::string result;
    result.reserve(source.size() + 64);

    size_t pos { 0 };
    while (pos < source.size()) {
        const auto line_end =
            source.find('\n', pos);
        const auto line_len =
            line_end == std::string::npos ? source.size() - pos : line_end - pos;
        const auto line = source.substr(pos, line_len);

        const auto first_non_ws = line.find_first_not_of(" \t\r");
        if (first_non_ws == std::string_view::npos) {
            result.append(line);
        } else {
            const auto trimmed = line.substr(first_non_ws);
            const auto eq_pos = line.find('=');
            const auto semicolon_pos = line.rfind(';');
            const bool is_vec2_assignment =
                trimmed.compare(0, 5, "vec2 ") == 0 &&
                eq_pos != std::string_view::npos &&
                semicolon_pos != std::string_view::npos &&
                eq_pos < semicolon_pos;

            if (! is_vec2_assignment || line.find("v_TexCoord") == std::string_view::npos) {
                result.append(line);
            } else {
                std::string rewritten(line);
                const auto rhs_begin = rewritten.find('=', 0);
                const auto rhs_end   = rewritten.rfind(';');
                if (rhs_begin != std::string::npos && rhs_end != std::string::npos &&
                    rhs_begin < rhs_end) {
                    const auto rhs_view = std::string_view(rewritten).substr(
                        rhs_begin + 1, rhs_end - rhs_begin - 1);
                    std::string rhs;
                    rhs.reserve(rhs_view.size() + 16);

                    size_t cursor { 0 };
                    while (cursor < rhs_view.size()) {
                        const auto match_pos = rhs_view.find("v_TexCoord", cursor);
                        if (match_pos == std::string_view::npos) {
                            rhs.append(rhs_view.substr(cursor));
                            break;
                        }

                        rhs.append(rhs_view.substr(cursor, match_pos - cursor));
                        const bool has_ident_prefix =
                            match_pos > 0 && IsIdentifierContinue(rhs_view[match_pos - 1]);
                        const size_t after_name = match_pos + std::char_traits<char>::length("v_TexCoord");
                        const bool has_ident_suffix =
                            after_name < rhs_view.size() && IsIdentifierContinue(rhs_view[after_name]);
                        const bool has_component_or_index =
                            after_name < rhs_view.size() &&
                            (rhs_view[after_name] == '.' || rhs_view[after_name] == '[');

                        if (!has_ident_prefix && !has_ident_suffix && !has_component_or_index) {
                            rhs.append("v_TexCoord.xy");
                        } else {
                            rhs.append("v_TexCoord");
                        }
                        cursor = after_name;
                    }

                    rewritten.replace(rhs_begin + 1, rhs_end - rhs_begin - 1, rhs);
                }
                result.append(rewritten);
            }
        }

        if (line_end == std::string_view::npos) break;
        result.push_back('\n');
        pos = line_end + 1;
    }

    return result;
}

inline std::string RewriteVectorNarrowingAssignments(std::string_view source) {
    std::string result;
    result.reserve(source.size() + 64);

    std::vector<ShaderVectorVariable> variables;
    size_t                            pos { 0 };
    while (pos < source.size()) {
        const auto line_end = source.find('\n', pos);
        const auto line_len =
            line_end == std::string_view::npos ? source.size() - pos : line_end - pos;
        const auto line = source.substr(pos, line_len);

        std::string_view declared_type;
        std::string_view declared_name;
        if (TryParseVectorDeclarationLine(line, declared_type, declared_name)) {
            UpsertShaderVectorVariable(
                variables,
                declared_name,
                ShaderVectorComponents(declared_type));
        }

        std::string_view lhs;
        std::string_view rhs;
        size_t           rhs_begin { 0 };
        size_t           rhs_len { 0 };
        if (TryParseSimpleUntypedAssignmentLine(line, lhs, rhs, rhs_begin, rhs_len)) {
            const int lhs_components = FindShaderVectorComponents(variables, lhs);
            const int rhs_components = FindShaderVectorComponents(variables, rhs);
            const auto swizzle       = NarrowingSwizzle(lhs_components);
            if (lhs_components > 0 && rhs_components > lhs_components && !swizzle.empty()) {
                // Wallpaper Engine's shader sources are authored for its legacy GLSL/HLSL-like
                // compatibility layer, where assigning a wider vector into a narrower vector is
                // treated as an implicit leading-component truncation. glslang's Vulkan frontend
                // rejects that form, so the compatibility pass makes the truncation explicit for
                // any already-declared vector assignment instead of adding shader-name exceptions.
                std::string rewritten(line);
                rewritten.replace(rhs_begin, rhs_len, std::string(rhs) + std::string(swizzle));
                result.append(rewritten);
                if (line_end == std::string_view::npos) break;
                result.push_back('\n');
                pos = line_end + 1;
                continue;
            }
        }

        result.append(line);
        if (line_end == std::string_view::npos) break;
        result.push_back('\n');
        pos = line_end + 1;
    }

    return result;
}

inline bool TryParseShaderMetadataJson(std::string_view line, const char* kind,
                                       nlohmann::json& result) {
    const size_t json_start = line.find_first_of('{');
    if (json_start == std::string::npos) return false;

    const auto json_source = line.substr(json_start);
    result                 = nlohmann::json::parse(json_source, nullptr, false);
    if (! result.is_discarded()) return true;

    const auto repaired = RepairMissingStringQuotes(json_source);
    if (! repaired.empty()) {
        result = nlohmann::json::parse(repaired, nullptr, false);
        if (! result.is_discarded()) {
            LOG_INFO("ParseWPShader: applied quote-repair to malformed %s metadata: %s",
                     kind,
                     TruncateMetadataSnippet(json_source).c_str());
            return true;
        }
    }

    LOG_INFO("ParseWPShader: skipped malformed %s metadata after quote-repair failed: %s",
             kind,
             TruncateMetadataSnippet(json_source).c_str());
    return false;
}

inline bool TextureSlotCanEnableCombo(i32 index, idx texcount,
                                      const std::vector<WPShaderTexInfo>& texinfos) {
    if (index < 0) return false;

    const auto texture_index = static_cast<usize>(index);
    // Texture-driven combos guard shader branches that sample g_TextureN. Only an authored material
    // texture should enable those branches: shader metadata defaults are editor fallbacks, not proof
    // that the project opted into an optional mask/input. Treating a default such as `util/white` as
    // bound enables Pulse's MASK branch for 2120087071 even though the material has no mask slot,
    // which changes the alpha-pulse path that makes the wallpaper rotate between images.
    return index < texcount && texture_index < texinfos.size() && texinfos[texture_index].enabled;
}

inline void ParseWPShader(const std::string& src, WPShaderInfo* pWPShaderInfo,
                          const std::vector<WPShaderTexInfo>& texinfos) {
    auto& combos       = pWPShaderInfo->combos;
    auto& wpAliasDict  = pWPShaderInfo->alias;
    auto& shadervalues = pWPShaderInfo->svs;
    auto& defTexs      = pWPShaderInfo->defTexs;
    idx   texcount     = std::ssize(texinfos);

    // pos start of line
    std::string::size_type pos = 0, lineEnd = std::string::npos;
    while ((lineEnd = src.find_first_of(('\n'), pos)), true) {
        const auto clineEnd = lineEnd;
        const auto line     = src.substr(pos, lineEnd - pos);

        /*
        if(line.find("attribute ") != std::string::npos || line.find("in ") != std::string::npos) {
            update_pos = true;
        }
        */
        if (line.find("// [COMBO]") != std::string::npos) {
            nlohmann::json combo_json;
            if (TryParseShaderMetadataJson(line, "combo", combo_json)) {
                if (combo_json.contains("combo")) {
                    std::string name;
                    int32_t     value = 0;
                    GET_JSON_NAME_VALUE(combo_json, "combo", name);
                    GET_JSON_NAME_VALUE(combo_json, "default", value);
                    combos[name] = std::to_string(value);
                }
            }
        } else if (line.find("uniform ") != std::string::npos) {
            if (line.find("// {") != std::string::npos) {
                nlohmann::json sv_json;
                if (TryParseShaderMetadataJson(line, "uniform", sv_json)) {
                    std::vector<std::string> defines =
                        utils::SpliteString(line.substr(0, line.find_first_of(';')), ' ');

                    std::string material;
                    GET_JSON_NAME_VALUE_NOWARN(sv_json, "material", material);
                    if (! material.empty()) wpAliasDict[material] = defines.back();

                    ShaderValue sv;
                    std::string name  = defines.back();
                    bool        istex = name.compare(0, 9, "g_Texture") == 0;
                    if (istex) {
                        wpscene::WPUniformTex wput;
                        wput.FromJson(sv_json);
                        i32 index { 0 };
                        STRTONUM(name.substr(9), index);
                        if (! wput.default_.empty()) defTexs.push_back({ index, wput.default_ });
                        if (! wput.combo.empty()) {
                            const bool combo_enabled =
                                TextureSlotCanEnableCombo(index, texcount, texinfos);
                            combos[wput.combo] = combo_enabled ? "1" : "0";
                            if (! combo_enabled) {
                                LOG_INFO("ParseWPShader: texture combo '%s' disabled for "
                                         "g_Texture%d because the slot has no bound texture",
                                         wput.combo.c_str(),
                                         index);
                            }
                        }
                        if (index >= 0 && index < texcount && texinfos[(usize)index].enabled) {
                            auto& compos = texinfos[(usize)index].composEnabled;

                            usize num = std::min(std::size(compos), std::size(wput.components));
                            for (usize i = 0; i < num; i++) {
                                if (compos[i]) combos[wput.components[i].combo] = "1";
                            }
                        }

                    } else {
                        if (sv_json.contains("default")) {
                            auto        value = sv_json.at("default");
                            ShaderValue sv;
                            name = defines.back();
                            if (value.is_string()) {
                                std::vector<float> v;
                                GET_JSON_VALUE(value, v);
                                sv = std::span<const float>(v);
                            } else if (value.is_number()) {
                                sv.setSize(1);
                                GET_JSON_VALUE(value, sv[0]);
                            }
                            shadervalues[name] = sv;
                        }
                        if (sv_json.contains("combo")) {
                            std::string name;
                            GET_JSON_NAME_VALUE(sv_json, "combo", name);
                            combos[name] = "1";
                        }
                    }
                    if (defines.back()[0] != 'g') {
                        LOG_INFO("PreShaderSrc User shadervalue not supported: %s %s",
                                 defines.back().c_str(),
                                 sv_json.dump().c_str());
                    }
                }
            }
        }

        // end
        if (line.find("void main()") != std::string::npos || clineEnd == std::string::npos) {
            break;
        }
        pos = lineEnd + 1;
    }
}

inline usize FindIncludeInsertPos(const std::string& src, usize startPos) {
    /* rule:
    after attribute/varying/uniform/struct
    befor any func
    not in {}
    not in #if #endif
    */
    const auto mainPos = src.find("void main(", startPos);
    if (mainPos == std::string::npos) return 0;
    if (src.find("void main(", mainPos + 1) != std::string::npos) return 0;

    usize candidate_pos { 0 };
    usize pos { startPos };
    int   brace_depth { 0 };
    int   if_depth { 0 };
    bool  in_block_comment { false };
    bool  pending_struct_close { false };

    while (pos < mainPos) {
        const auto raw_line_end = src.find('\n', pos);
        const auto line_end =
            raw_line_end == std::string::npos || raw_line_end > mainPos ? mainPos : raw_line_end;
        const auto next_pos = raw_line_end == std::string::npos || raw_line_end > mainPos
                                ? line_end
                                : raw_line_end + 1;
        const auto line = std::string_view(src).substr(pos, line_end - pos);
        const auto trimmed = TrimLeadingHorizontalWhitespace(line);
        const auto brace_depth_before = brace_depth;

        const bool starts_if    = trimmed.rfind("#if", 0) == 0;
        const bool starts_endif = trimmed.rfind("#endif", 0) == 0;
        const bool top_level    = brace_depth_before == 0 && if_depth == 0;
        const bool starts_decl =
            top_level &&
            (StartsWithToken(trimmed, "attribute") || StartsWithToken(trimmed, "varying") ||
             StartsWithToken(trimmed, "uniform"));
        const bool starts_struct = top_level && StartsWithToken(trimmed, "struct");

        UpdateBraceDepth(line, brace_depth, in_block_comment);

        if (starts_decl) {
            candidate_pos = next_pos;
        }

        if (starts_struct) {
            if (line.find('{') != std::string_view::npos && brace_depth > brace_depth_before) {
                pending_struct_close = true;
            } else {
                candidate_pos = next_pos;
            }
        } else if (pending_struct_close && brace_depth_before > 0 && brace_depth == 0) {
            candidate_pos         = next_pos;
            pending_struct_close = false;
        }

        if (starts_if) {
            if_depth++;
        } else if (starts_endif && if_depth > 0) {
            if_depth--;
        }

        pos = next_pos;
    }

    return candidate_pos;
}

inline EShLanguage ToGLSL(ShaderType type) {
    switch (type) {
    case ShaderType::VERTEX: return EShLangVertex;
    case ShaderType::FRAGMENT: return EShLangFragment;
    default: return EShLangVertex;
    }
}

inline std::string SanitizeBrokenPreprocessorDirectives(const std::string& src,
                                                        ShaderType         type) {
    std::string out;
    out.reserve(src.size());

    std::vector<std::string::size_type> if_stack;
    std::string::size_type              pos { 0 };
    usize                               removed_endifs { 0 };

    while (pos < src.size()) {
        auto line_end = src.find('\n', pos);
        auto line_len =
            line_end == std::string::npos ? src.size() - pos : line_end - pos;
        auto line = src.substr(pos, line_len);

        auto first_non_ws = line.find_first_not_of(" \t\r");
        bool handled      = false;
        if (first_non_ws != std::string::npos && line[first_non_ws] == '#') {
            auto directive = line.substr(first_non_ws);
            if (directive.rfind("#if", 0) == 0) {
                if_stack.push_back(pos);
            } else if (directive.rfind("#endif", 0) == 0) {
                if (if_stack.empty()) {
                    out.append(line.substr(0, first_non_ws));
                    out.append("// stripped unmatched #endif");
                    handled = true;
                    removed_endifs++;
                } else {
                    if_stack.pop_back();
                }
            }
        }

        if (!handled) out.append(line);
        if (line_end == std::string::npos) break;
        out.push_back('\n');
        pos = line_end + 1;
    }

    if (removed_endifs > 0) {
        LOG_INFO("SanitizeBrokenPreprocessorDirectives stripped %zu unmatched #endif line(s) "
                 "from %s shader",
                 removed_endifs,
                 type == ShaderType::VERTEX ? "vertex" : "fragment");
    }
    return out;
}

inline std::string Preprocessor(const std::string& in_src, ShaderType type, const Combos& combos,
                                WPPreprocessorInfo& process_info) {
    std::string res;

    std::string src =
        wallpaper::WPShaderParser::PreShaderHeader(
            SanitizeBrokenPreprocessorDirectives(in_src, type), combos, type);

    // workaround #require directive
    src = CommentOutRequireDirectives(src);

    glslang::TShader::ForbidIncluder includer;
    glslang::TShader                 shader(ToGLSL(type));
    const EShMessages emsg { (EShMessages)(EShMsgDefault | EShMsgSpvRules | EShMsgRelaxedErrors |
                                           EShMsgSuppressWarnings | EShMsgVulkanRules) };

    auto* data = src.c_str();
    shader.setStrings(&data, 1);
    if (! shader.preprocess(&vulkan::DefaultTBuiltInResource,
                            110,
                            EProfile::ECoreProfile,
                            false,
                            false,
                            emsg,
                            &res,
                            includer)) {
        LOG_ERROR("glslang(preprocess): %s", shader.getInfoLog());
        return src;
    }

    CollectPreprocessorInfo(res, process_info);
    return res;
}

inline std::string Finalprocessor(const WPShaderUnit& unit, const WPPreprocessorInfo* pre,
                                  const WPPreprocessorInfo* next) {
    auto ReplaceDeclaration = [](std::string& text, const std::string& from, const std::string& to) {
        const auto pos = text.find(from);
        if (pos == std::string::npos) return false;

        text.replace(pos, from.size(), to);
        return true;
    };
    auto AsInputDecl = [](const std::string& decl) {
        auto result = decl;
        ReplaceFirstStandaloneWord(result, "out", "in");
        return result;
    };
    auto AsOutputDecl = [](const std::string& decl) {
        auto result = decl;
        ReplaceFirstStandaloneWord(result, "in", "out");
        return result;
    };
    auto FindMainBody = [](const std::string& text) -> std::optional<std::pair<size_t, size_t>> {
        const auto main_pos = text.find("void main");
        if (main_pos == std::string::npos) return std::nullopt;

        const auto brace_open = text.find('{', main_pos);
        if (brace_open == std::string::npos) return std::nullopt;

        int    depth { 1 };
        size_t cursor { brace_open + 1 };
        while (cursor < text.size() && depth > 0) {
            if (text[cursor] == '{') depth++;
            else if (text[cursor] == '}') depth--;
            cursor++;
        }
        if (depth != 0 || cursor <= brace_open + 1) return std::nullopt;

        return std::pair<size_t, size_t> { brace_open + 1, cursor - 1 };
    };
    auto HasLocalDeclarationInMainBody = [&](const std::string& text, const std::string& name) {
        const auto main_body = FindMainBody(text);
        if (! main_body.has_value()) return false;

        auto [body_begin, body_end] = *main_body;
        const auto  body = std::string_view(text).substr(body_begin, body_end - body_begin);
        size_t      pos { 0 };
        bool        in_block_comment { false };
        bool        in_string { false };
        bool        escaped { false };
        char        quote { '\0' };

        while (pos < body.size()) {
            const char ch   = body[pos];
            const char next = pos + 1 < body.size() ? body[pos + 1] : '\0';

            if (in_block_comment) {
                if (ch == '*' && next == '/') {
                    in_block_comment = false;
                    pos += 2;
                    continue;
                }
                pos++;
                continue;
            }

            if (in_string) {
                if (escaped) {
                    escaped = false;
                    pos++;
                    continue;
                }
                if (ch == '\\') {
                    escaped = true;
                    pos++;
                    continue;
                }
                if (ch == quote) in_string = false;
                pos++;
                continue;
            }

            if (ch == '/' && next == '/') {
                const auto line_end = body.find('\n', pos);
                pos                 = line_end == std::string_view::npos ? body.size() : line_end + 1;
                continue;
            }
            if (ch == '/' && next == '*') {
                in_block_comment = true;
                pos += 2;
                continue;
            }
            if (ch == '"' || ch == '\'') {
                in_string = true;
                quote     = ch;
                pos++;
                continue;
            }
            if (! IsIdentifierStart(ch)) {
                pos++;
                continue;
            }

            const auto first_end   = SkipIdentifier(body, pos);
            auto       cursor      = SkipWhitespace(body, first_end);
            auto       type_begin  = pos;
            auto       type_end    = first_end;

            if (body.substr(pos, first_end - pos) == "const") {
                if (cursor >= body.size() || ! IsIdentifierStart(body[cursor])) {
                    pos = first_end;
                    continue;
                }
                type_begin = cursor;
                type_end   = SkipIdentifier(body, cursor);
                cursor     = SkipWhitespace(body, type_end);
            }

            if (type_begin == type_end || cursor >= body.size() || ! IsIdentifierStart(body[cursor])) {
                pos = first_end;
                continue;
            }

            const auto decl_name_end = SkipIdentifier(body, cursor);
            const auto decl_name     = body.substr(cursor, decl_name_end - cursor);
            if (decl_name != name) {
                pos = decl_name_end;
                continue;
            }

            cursor = SkipWhitespace(body, decl_name_end);
            if (cursor < body.size() && body[cursor] == '[') {
                cursor = SkipBalanced(body, cursor, '[', ']');
                if (cursor == std::string::npos) return false;
                cursor = SkipWhitespace(body, cursor);
            }

            if (cursor < body.size() &&
                (body[cursor] == '=' || body[cursor] == ';' || body[cursor] == ',')) {
                return true;
            }

            pos = decl_name_end;
        }

        return false;
    };
    auto ReplaceNameInMainBody = [&](std::string& text, const std::string& from,
                                     const std::string& to) {
        const auto main_body = FindMainBody(text);
        if (! main_body.has_value()) return false;

        auto [body_begin, body_end] = *main_body;
        std::string body            = text.substr(body_begin, body_end - body_begin);
        body = ReplaceStandaloneIdentifier(body, from, to);
        text.replace(body_begin, body_end - body_begin, body);
        return true;
    };

    std::string insert_str {};
    auto&       cur = unit.preprocess_info;
    if (pre != nullptr) {
        for (auto& [k, v] : pre->output) {
            if (! exists(cur.input, k)) {
                auto n = AsInputDecl(v);
                insert_str += n + '\n';
            }
        }
    }
    if (next != nullptr) {
        for (auto& [k, v] : next->input) {
            if (! exists(cur.output, k)) {
                auto n = AsOutputDecl(v);
                insert_str += n + '\n';
            }
        }
    }
    // LOG_INFO("insert: %s", insert_str.c_str());
    // return std::regex_replace(
    //    std::regex_replace(cur.result, re_hold, insert_str), std::regex(R"(\s+\n)"), "\n");
    auto result = unit.src;
    ReplaceAll(result, SHADER_PLACEHOLD, insert_str);

    // Wallpaper Engine shaders sometimes declare the same stage interface with different types
    // across stages. Only reconcile the consumer side with the previous stage's output.
    // Rewriting both sides from stale preprocess metadata can flip the mismatch instead of
    // fixing it.
    if (pre != nullptr) {
        for (const auto& [name, prev_decl] : pre->output) {
            if (! exists(cur.input, name)) continue;

            const auto expected_input = AsInputDecl(prev_decl);
            if (cur.input.at(name) == expected_input) continue;

            ReplaceDeclaration(result, cur.input.at(name), expected_input);
        }
    }

    // Some Wallpaper Engine effect shaders survive preprocessing with invalid simple
    // declarations. Normalize those with a token-aware statement rewrite.
    result = RewriteSimpleShaderStatements(result);
    result = RewriteVectorNarrowingAssignments(result);
    result = RewriteSwizzledSelfMixAssignments(result);

    // Wallpaper Engine sometimes stores integer loop bounds in float uniforms or expressions.
    result = RewriteIntegerForLoops(result);
    result = RewriteUintModuloAssignments(result);
    result = RewriteFloatBoolMultiplyAssignments(result);
    result = TrimOverlongVectorConstructors(result);
    result = RewriteVec2AssignmentsForVec4TexCoord(result);

    // Some effect fragment shaders write back into stage inputs such as v_TexCoord. GLSL
    // forbids that, so rewrite those inputs to immutable varyings plus mutable locals.
    if (unit.stage == ShaderType::FRAGMENT) {
        std::string init_lines;
        for (const auto& [name, decl] : cur.input) {
            if (! HasMutableUse(result, name)) continue;
            if (HasLocalDeclarationInMainBody(result, name)) continue;

            std::string mutable_name;
            std::string init_line;
            if (! MakeMutableInputShim(decl, name, mutable_name, init_line)) {
                continue;
            }
            if (! ReplaceNameInMainBody(result, name, mutable_name)) continue;
            init_lines += init_line;
        }

        if (! init_lines.empty()) {
            if (const auto main_body = FindMainBody(result); main_body.has_value()) {
                result.insert(main_body->first, "\n" + init_lines);
            }
        }
    }

    // HLSL-style shaders often write scalar-first max/min calls such as max(0.0, vec2Expr).
    // GLSL only accepts the scalar as the second argument for vector overloads.
    static const std::regex re_scalar_first_minmax(
        R"(\b(max|min)\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*((?:[^(),\n]+|\([^()]*\))+)\s*\))");
    result = std::regex_replace(result, re_scalar_first_minmax, "$1($3, $2)");

    return result;
}

inline std::string GenSha1(std::span<const WPShaderUnit> units) {
    std::string shas;
    for (auto& unit : units) {
        shas += utils::genSha1(unit.src);
    }
    return utils::genSha1(shas);
}
inline std::string GetCachePath(std::string_view scene_id, std::string_view filename) {
    return std::string("/cache/") + std::string(scene_id) + "/" SHADER_DIR "/" +
           std::string(filename) + "." SHADER_SUFFIX;
}

inline std::string GetPreShaderCachePath(std::string_view filename) {
    return std::string("/cache/") + SHADER_META_DIR + "/" + std::string(filename) + "." +
           SHADER_META_SUFFIX;
}

inline std::string GetPreparedShaderCachePath(std::string_view scene_id, std::string_view filename) {
    return std::string("/cache/") + std::string(scene_id) + "/" SHADER_SRC_DIR "/" +
           std::string(filename) + "." SHADER_SRC_SUFFIX;
}

inline bool LoadShaderFromFile(std::vector<ShaderCode>& codes, fs::IBinaryStream& file) {
    codes.clear();
    i32 ver = ReadSPVVesion(file);

    usize count = file.ReadUint32();
    assert(count <= 16 && count >= 0);
    if (count > 16) return false;

    codes.resize(count);
    for (usize i = 0; i < count; i++) {
        auto& c = codes[i];

        u32 size = file.ReadUint32();
        assert(size % 4 == 0);
        if (size % 4 != 0) return false;

        c.resize(size / 4);
        file.Read((char*)c.data(), size);
    }
    return true;
}

inline void SaveShaderToFile(std::span<const ShaderCode> codes, fs::IBinaryStreamW& file) {
    char nop[256] { '\0' };

    WriteSPVVesion(file, 1);
    file.WriteUint32((u32)codes.size());
    for (const auto& c : codes) {
        u32 size = (u32)c.size() * 4;
        file.WriteUint32(size);
        file.Write((const char*)c.data(), size);
    }
    file.Write(nop, sizeof(nop));
}

inline std::string GenPreparedShaderSha1(std::span<const WPShaderUnit> units, const Combos& combos) {
    std::ostringstream out;
    out << "prepared-shader-v4\n";
    for (const auto& unit : units) {
        out << static_cast<int>(unit.stage) << '\n';
        out << utils::genSha1(unit.src) << '\n';
    }
    for (const auto& [name, value] : combos) {
        out << name << '=' << value << '\n';
    }
    const auto data = out.str();
    return utils::genSha1(std::span<const char>(data.data(), data.size()));
}

inline std::string GenPreShaderSha1(std::string_view expanded_src,
                                    std::span<const WPShaderTexInfo> texinfos) {
    std::ostringstream out;
    out << "pre-shader-v3-authored-texture-combo-boundness\n";
    out << utils::genSha1(expanded_src) << '\n';
    for (const auto& texinfo : texinfos) {
        out << static_cast<int>(texinfo.enabled);
        for (const auto component_enabled : texinfo.composEnabled) {
            out << static_cast<int>(component_enabled);
        }
        out << '\n';
    }

    const auto data = out.str();
    return utils::genSha1(std::span<const char>(data.data(), data.size()));
}

inline void WriteString(fs::IBinaryStreamW& file, std::string_view value) {
    file.WriteUint32(static_cast<u32>(value.size()));
    if (! value.empty()) file.Write(value.data(), value.size());
}

inline bool ReadString(fs::IBinaryStream& file, std::string& value) {
    const auto size = file.ReadUint32();
    value.resize(size);
    if (size > 0) file.Read(value.data(), size);
    return true;
}

inline bool LoadStringMap(Map<std::string, std::string>& map, fs::IBinaryStream& file) {
    map.clear();

    const auto count = file.ReadUint32();
    for (u32 i = 0; i < count; i++) {
        std::string key;
        std::string value;
        if (! ReadString(file, key)) return false;
        if (! ReadString(file, value)) return false;

        map.emplace(std::move(key), std::move(value));
    }
    return true;
}

inline void SaveStringMap(const Map<std::string, std::string>& map, fs::IBinaryStreamW& file) {
    file.WriteUint32(static_cast<u32>(map.size()));
    for (const auto& [key, value] : map) {
        WriteString(file, key);
        WriteString(file, value);
    }
}

inline bool LoadShaderValue(ShaderValue& value, fs::IBinaryStream& file) {
    const auto size = file.ReadUint32();
    std::vector<float> values(size);
    for (u32 i = 0; i < size; i++) {
        values[i] = file.ReadFloat();
    }
    value = ShaderValue(values);
    return true;
}

inline void SaveShaderValue(const ShaderValue& value, fs::IBinaryStreamW& file) {
    file.WriteUint32(static_cast<u32>(value.size()));
    for (size_t i = 0; i < value.size(); i++) {
        const float component = value[i];
        file.Write(&component, sizeof(component));
    }
}

inline bool LoadShaderValueMap(ShaderValueMap& map, fs::IBinaryStream& file) {
    map.clear();

    const auto count = file.ReadUint32();
    for (u32 i = 0; i < count; i++) {
        std::string key;
        ShaderValue value;
        if (! ReadString(file, key)) return false;
        if (! LoadShaderValue(value, file)) return false;
        map.emplace(std::move(key), std::move(value));
    }
    return true;
}

inline void SaveShaderValueMap(const ShaderValueMap& map, fs::IBinaryStreamW& file) {
    file.WriteUint32(static_cast<u32>(map.size()));
    for (const auto& [key, value] : map) {
        WriteString(file, key);
        SaveShaderValue(value, file);
    }
}

inline bool LoadDefaultTexs(WPDefaultTexs& def_texs, fs::IBinaryStream& file) {
    def_texs.clear();

    const auto count = file.ReadUint32();
    def_texs.reserve(count);
    for (u32 i = 0; i < count; i++) {
        const auto slot = file.ReadInt32();
        std::string default_tex;
        if (! ReadString(file, default_tex)) return false;
        def_texs.emplace_back(slot, std::move(default_tex));
    }
    return true;
}

inline void SaveDefaultTexs(const WPDefaultTexs& def_texs, fs::IBinaryStreamW& file) {
    file.WriteUint32(static_cast<u32>(def_texs.size()));
    for (const auto& [slot, default_tex] : def_texs) {
        file.WriteInt32(slot);
        WriteString(file, default_tex);
    }
}

inline bool LoadPreShaderInfo(WPShaderInfo& shader_info, fs::IBinaryStream& file) {
    const auto version = ReadVersion("WSHM", file);
    if (version != 1) return false;

    shader_info = {};
    if (! LoadStringMap(shader_info.combos, file)) return false;
    if (! LoadStringMap(shader_info.alias, file)) return false;
    if (! LoadShaderValueMap(shader_info.svs, file)) return false;
    if (! LoadDefaultTexs(shader_info.defTexs, file)) return false;
    return true;
}

inline void SavePreShaderInfo(const WPShaderInfo& shader_info, fs::IBinaryStreamW& file) {
    WriteVersion("WSHM", file, 1);
    SaveStringMap(shader_info.combos, file);
    SaveStringMap(shader_info.alias, file);
    SaveShaderValueMap(shader_info.svs, file);
    SaveDefaultTexs(shader_info.defTexs, file);
}

inline void MergeShaderInfo(WPShaderInfo& into, const WPShaderInfo& from) {
    for (const auto& [key, value] : from.combos) {
        into.combos[key] = value;
    }
    for (const auto& [key, value] : from.alias) {
        into.alias[key] = value;
    }
    for (const auto& [key, value] : from.svs) {
        into.svs[key] = value;
    }
    into.defTexs.insert(into.defTexs.end(), from.defTexs.begin(), from.defTexs.end());
}

struct ExpandedShaderSource {
    std::string expanded_src;
    std::string src_without_includes;
    std::string include_src;
};

inline ExpandedShaderSource ExpandShaderSource(fs::VFS& vfs, const std::string& src) {
    ExpandedShaderSource   result { .expanded_src = src, .src_without_includes = src, .include_src = {} };
    std::string::size_type pos = 0;
    while (pos = src.find("#include", pos), pos != std::string::npos) {
        auto begin = pos;
        pos        = src.find_first_of('\n', pos);
        result.src_without_includes.replace(begin, pos - begin, pos - begin, ' ');
        result.include_src.append(src.substr(begin, pos - begin) + "\n");
    }

    result.include_src = LoadGlslInclude(vfs, result.include_src);
    result.expanded_src = result.src_without_includes;
    result.expanded_src.insert(FindIncludeInsertPos(result.expanded_src, 0), result.include_src);
    return result;
}

inline bool LoadPreparedShaderUnits(std::span<WPShaderUnit> units, fs::IBinaryStream& file) {
    const auto version = ReadVersion("WSRC", file);
    if (version != 2) return false;

    const auto count = file.ReadUint32();
    if (count != units.size()) return false;

    for (usize i = 0; i < units.size(); i++) {
        const auto stage = file.ReadInt32();
        if (stage != static_cast<int32_t>(units[i].stage)) return false;

        const auto size = file.ReadUint32();
        std::string src(size, '\0');
        if (size > 0) file.Read(src.data(), size);
        units[i].src = std::move(src);

        auto& info = units[i].preprocess_info;
        info       = {};

        if (! LoadStringMap(info.input, file)) return false;
        if (! LoadStringMap(info.output, file)) return false;

        const auto active_count = file.ReadUint32();
        for (u32 j = 0; j < active_count; j++) {
            info.active_tex_slots.insert(file.ReadUint32());
        }
    }
    return true;
}

inline void SavePreparedShaderUnits(std::span<const WPShaderUnit> units, fs::IBinaryStreamW& file) {
    WriteVersion("WSRC", file, 2);
    file.WriteUint32(static_cast<u32>(units.size()));
    for (const auto& unit : units) {
        file.WriteInt32(static_cast<int32_t>(unit.stage));
        file.WriteUint32(static_cast<u32>(unit.src.size()));
        if (! unit.src.empty()) file.Write(unit.src.data(), unit.src.size());

        SaveStringMap(unit.preprocess_info.input, file);
        SaveStringMap(unit.preprocess_info.output, file);

        file.WriteUint32(static_cast<u32>(unit.preprocess_info.active_tex_slots.size()));
        for (const auto slot : unit.preprocess_info.active_tex_slots) {
            file.WriteUint32(slot);
        }
    }
}

inline void PrepareShaderUnits(std::span<WPShaderUnit> units, WPShaderInfo* shader_info) {
    std::for_each(units.begin(), units.end(), [shader_info](auto& unit) {
        unit.src = Preprocessor(unit.src, unit.stage, shader_info->combos, unit.preprocess_info);
    });

    for (usize i = 0; i < units.size(); i++) {
        auto&               unit      = units[i];
        WPPreprocessorInfo* pre_info  = i >= 1 ? &units[i - 1].preprocess_info : nullptr;
        WPPreprocessorInfo* post_info = i + 1 < units.size() ? &units[i + 1].preprocess_info
                                                             : nullptr;
        unit.src = Finalprocessor(unit, pre_info, post_info);
    }
}

} // namespace

std::string WPShaderParser::PreShaderSrc(fs::VFS& vfs, const std::string& src,
                                         WPShaderInfo*                       pWPShaderInfo,
                                         const std::vector<WPShaderTexInfo>& texinfos) {
    auto expanded = ExpandShaderSource(vfs, src);

    if (! vfs.IsMounted("cache")) {
        ParseWPShader(expanded.include_src, pWPShaderInfo, texinfos);
        ParseWPShader(expanded.src_without_includes, pWPShaderInfo, texinfos);
        return expanded.expanded_src;
    }

    const auto cache_key  = GenPreShaderSha1(expanded.expanded_src, texinfos);
    const auto cache_path = GetPreShaderCachePath(cache_key);

    WPShaderInfo cached_info;
    if (vfs.Contains(cache_path)) {
        auto cache_file = vfs.Open(cache_path);
        if (! cache_file || ! LoadPreShaderInfo(cached_info, *cache_file)) {
            LOG_ERROR("load pre-shader metadata from '%s' failed", cache_path.c_str());
            return {};
        }
    } else {
        ParseWPShader(expanded.include_src, &cached_info, texinfos);
        ParseWPShader(expanded.src_without_includes, &cached_info, texinfos);

        if (auto cache_file = vfs.OpenW(cache_path); cache_file) {
            SavePreShaderInfo(cached_info, *cache_file);
        }
    }

    MergeShaderInfo(*pWPShaderInfo, cached_info);
    return expanded.expanded_src;
}

std::string WPShaderParser::PreShaderHeader(const std::string& src, const Combos& combos,
                                            ShaderType type) {
    std::string pre(pre_shader_code);
    if (type == ShaderType::VERTEX) pre += pre_shader_code_vert;
    if (type == ShaderType::FRAGMENT) pre += pre_shader_code_frag;
    std::string header(pre);
    for (const auto& c : combos) {
        std::string cup(c.first);
        std::transform(c.first.begin(), c.first.end(), cup.begin(), ::toupper);
        if (c.second.empty()) {
            LOG_ERROR("combo '%s' can't be empty", cup.c_str());
            continue;
        }
        header.append("#define " + cup + " " + c.second + "\n");
    }
    return header + src;
}

void WPShaderParser::InitGlslang(std::string_view reason) {
    auto& state = GetGlslangRuntimeState();
    state.mutex.lock();
    if (state.depth == 0) {
        glslang::InitializeProcess();
    }
    state.depth++;
    LOG_INFO("GlslangScope: action=enter reason='%.*s' thread=%zu depth=%zu",
             static_cast<int>(reason.size()),
             reason.data(),
             GetThreadLogId(),
             state.depth);
}

void WPShaderParser::FinalGlslang(std::string_view reason) {
    auto& state = GetGlslangRuntimeState();
    if (state.depth == 0) {
        LOG_ERROR("GlslangScope: action=leave reason='%.*s' thread=%zu depth-underflow=true",
                  static_cast<int>(reason.size()),
                  reason.data(),
                  GetThreadLogId());
        return;
    }

    state.depth--;
    LOG_INFO("GlslangScope: action=leave reason='%.*s' thread=%zu depth=%zu",
             static_cast<int>(reason.size()),
             reason.data(),
             GetThreadLogId(),
             state.depth);
    if (state.depth == 0) {
        glslang::FinalizeProcess();
    }
    state.mutex.unlock();
}

bool WPShaderParser::CompileToSpv(std::string_view scene_id, std::span<WPShaderUnit> units,
                                  std::vector<ShaderCode>& codes, fs::VFS& vfs,
                                  WPShaderInfo* shader_info, std::span<const WPShaderTexInfo> texs) {
    (void)texs;

    auto compile = [](std::span<WPShaderUnit> units, std::vector<ShaderCode>& codes) {
        std::vector<vulkan::ShaderCompUnit> vunits(units.size());
        for (usize i = 0; i < units.size(); i++) {
            auto&               unit     = units[i];
            auto&               vunit    = vunits[i];

            vunit.src   = unit.src;
            vunit.stage = ToGLSL(unit.stage);
        }

        vulkan::ShaderCompOpt opt;
        opt.client_ver             = glslang::EShTargetVulkan_1_1;
        opt.auto_map_bindings      = true;
        opt.auto_map_locations     = true;
        opt.relaxed_errors_glsl    = true;
        opt.relaxed_rules_vulkan   = true;
        opt.suppress_warnings_glsl = true;

        std::vector<vulkan::Uni_ShaderSpv> spvs(units.size());

        if (! vulkan::CompileAndLinkShaderUnits(vunits, opt, spvs)) {
            return false;
        }

        codes.clear();
        for (auto& spv : spvs) {
            codes.emplace_back(std::move(spv->spirv));
        }
        return true;
    };

    bool has_cache_dir = vfs.IsMounted("cache");

    if (has_cache_dir) {
        const std::string prepared_sha1            = GenPreparedShaderSha1(units, shader_info->combos);
        const std::string prepared_cache_file_path = GetPreparedShaderCachePath(scene_id, prepared_sha1);

        if (vfs.Contains(prepared_cache_file_path)) {
            auto cache_file = vfs.Open(prepared_cache_file_path);
            if (! cache_file || ! ::LoadPreparedShaderUnits(units, *cache_file)) {
                LOG_ERROR("load prepared shader from '%s' failed", prepared_cache_file_path.c_str());
                return false;
            }
        } else {
            PrepareShaderUnits(units, shader_info);
            if (auto cache_file = vfs.OpenW(prepared_cache_file_path); cache_file) {
                ::SavePreparedShaderUnits(units, *cache_file);
            }
        }

        std::string sha1            = GenSha1(units);
        std::string cache_file_path = GetCachePath(scene_id, sha1);

        if (vfs.Contains(cache_file_path)) {
            auto cache_file = vfs.Open(cache_file_path);
            if (! cache_file || ! ::LoadShaderFromFile(codes, *cache_file)) {
                LOG_ERROR("load shader from \'%s\' failed", cache_file_path.c_str());
                return false;
            }
        } else {
            if (! compile(units, codes)) return false;
            if (auto cache_file = vfs.OpenW(cache_file_path); cache_file) {
                ::SaveShaderToFile(codes, *cache_file);
            }
        }
        return true;

    } else {
        PrepareShaderUnits(units, shader_info);
        return compile(units, codes);
    }
}
