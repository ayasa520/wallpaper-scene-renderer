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
#include <stack>
#include <charconv>
#include <cctype>
#include <sstream>
#include <string>

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

static constexpr const char* pre_shader_code = R"(#version 150
#define GLSL 1
#define HLSL 0
#define highp

#define CAST2(x) (vec2(x))
#define CAST3(x) (vec3(x))
#define CAST4(x) (vec4(x))
#define CAST3X3(x) (mat3(x))

#define texSample2D texture
#define texSample2DLod textureLod
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
                            if (index >= texcount)
                                combos[wput.combo] = "0";
                            else
                                combos[wput.combo] = "1";
                        }
                        if (index < texcount && texinfos[(usize)index].enabled) {
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
    (void)startPos;

    auto NposToZero = [](usize p) {
        return p == std::string::npos ? 0 : p;
    };
    auto search = [](const std::string& p, usize pos, const auto& re) {
        auto        startpos = p.begin() + (isize)pos;
        std::smatch match;
        if (startpos < p.end() && std::regex_search(startpos, p.end(), match, re)) {
            return pos + (usize)match.position();
        }
        return std::string::npos;
    };
    auto searchLast = [](const std::string& p, const auto& re) {
        auto        startpos = p.begin();
        std::smatch match;
        while (startpos < p.end() && std::regex_search(startpos, p.end(), match, re)) {
            startpos++;
            startpos += match.position();
        }
        return startpos >= p.end() ? std::string::npos : usize(startpos - p.begin());
    };
    auto nextLinePos = [](const std::string& p, usize pos) {
        return p.find_first_of('\n', pos) + 1;
    };

    usize mainPos  = src.find("void main(");
    bool  two_main = src.find("void main(", mainPos + 2) != std::string::npos;
    if (two_main) return 0;

    usize pos;
    {
        const std::regex reAfters(R"(\n(attribute|varying|uniform|struct) )");
        usize            afterPos = searchLast(src, reAfters);
        if (afterPos != std::string::npos) {
            afterPos = nextLinePos(src, afterPos + 1);
        }
        pos = std::min({ NposToZero(afterPos), mainPos });
    }
    {
        std::stack<usize> ifStack;
        usize             nowPos { 0 };
        const std::regex  reIfs(R"((#if|#endif))");
        while (true) {
            auto p = search(src, nowPos + 1, reIfs);
            if (p > mainPos || p == std::string::npos) break;
            if (src.substr(p, 3) == "#if") {
                ifStack.push(p);
            } else {
                if (ifStack.empty()) break;
                usize ifp = ifStack.top();
                ifStack.pop();
                usize endp = p;
                if (pos > ifp && pos <= endp) {
                    pos = nextLinePos(src, endp + 1);
                }
            }
            nowPos = p;
        }
        pos = std::min({ pos, mainPos });
    }

    return NposToZero(pos);
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
    {
        std::regex re_require("(^|\r?\n)#require (.+)(\r?\n)");
        src = std::regex_replace(src, re_require, "$1//#require $2$3");
    }

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

    std::regex re_io(R"(.+\s(in|out)\s[\s\w]+\s(\w+)\s*;)", std::regex::ECMAScript);
    for (auto it = std::sregex_iterator(res.begin(), res.end(), re_io);
         it != std::sregex_iterator();
         it++) {
        std::smatch mc = *it;
        if (mc[1] == "in") {
            process_info.input[mc[2]] = mc[0].str();
        } else {
            process_info.output[mc[2]] = mc[0].str();
        }
    }

    std::regex re_tex(R"(uniform\s+sampler2D\s+g_Texture(\d+))", std::regex::ECMAScript);
    for (auto it = std::sregex_iterator(res.begin(), res.end(), re_tex);
         it != std::sregex_iterator();
         it++) {
        std::smatch mc  = *it;
        auto        str = mc[1].str();
        uint        slot;
        auto [ptr, ec] { std::from_chars(str.c_str(), str.c_str() + str.size(), slot) };
        if (ec != std::errc()) continue;
        process_info.active_tex_slots.insert(slot);
    }
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
        return std::regex_replace(decl, std::regex(R"(\bout\b)"), "in");
    };
    auto AsOutputDecl = [](const std::string& decl) {
        return std::regex_replace(decl, std::regex(R"(\bin\b)"), "out");
    };

    auto EscapeRegex = [](const std::string& text) {
        static const std::regex special(R"([-[\]{}()*+?.,\^$|#\s])");
        return std::regex_replace(text, special, R"(\$&)");
    };
    auto HasMutableUse = [&](const std::string& text, const std::string& name) {
        const auto name_re = EscapeRegex(name);
        const auto pattern = std::string(R"(\b)") + name_re +
                             R"(\b(?:\s*(?:\.[A-Za-z_]\w*|\[[^\]]+\]))*\s*(?:\+=|-=|\*=|/=|%=|=(?!=)|\+\+|--))";
        return std::regex_search(text, std::regex(pattern));
    };
    auto MakeMutableInputShim = [](const std::string& decl, const std::string& name,
                                   std::string& mutable_name, std::string& init_line) {
        std::smatch match;
        if (! std::regex_match(
                decl,
                match,
                std::regex(
                    R"((\s*)in\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)(\s*\[[^\]]+\])?\s*;)"))) {
            return false;
        }

        const auto indent        = match[1].str();
        const auto type          = match[2].str();
        const auto declared_name = match[3].str();
        const auto suffix        = match[4].str();
        if (declared_name != name) return false;

        mutable_name = "_wp_mutable_" + name;
        init_line    = indent + type + " " + mutable_name + suffix + " = " + name + ";\n";
        return true;
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
    auto ReplaceNameInMainBody = [&](std::string& text, const std::string& from,
                                     const std::string& to) {
        const auto main_body = FindMainBody(text);
        if (! main_body.has_value()) return false;

        auto [body_begin, body_end] = *main_body;
        std::string body            = text.substr(body_begin, body_end - body_begin);
        body = std::regex_replace(
            body,
            std::regex(std::string(R"(\b)") + EscapeRegex(from) + R"(\b)"),
            to);
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
    std::regex re_hold(SHADER_PLACEHOLD.data());

    // LOG_INFO("insert: %s", insert_str.c_str());
    // return std::regex_replace(
    //    std::regex_replace(cur.result, re_hold, insert_str), std::regex(R"(\s+\n)"), "\n");
    auto result = std::regex_replace(unit.src, re_hold, insert_str);

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

    // Some Wallpaper Engine effect shaders survive preprocessing with an invalid scalar
    // declaration even though the RHS is a vec2. Fix the generated GLSL before glslang sees it.
    result = std::regex_replace(
        result,
        std::regex(R"(\bfloat\s+pointer\s*=\s*g_PointerPosition\.xy\s*\*\s*u_pointerSpeed\s*;)"),
        "vec2 pointer = g_PointerPosition.xy * u_pointerSpeed;");

    // Wallpaper Engine sometimes stores integer loop bounds in float uniforms or expressions.
    result = std::regex_replace(
        result,
        std::regex(
            R"(for\s*\(\s*int\s+([A-Za-z_]\w*)\s*=\s*([^;]+?)\s*;\s*\1\s*([<>]=?)\s*([^;]+?)\s*;\s*(?:\+\+\s*\1|\1\s*\+\+)\s*\))"),
        "for (int $1 = int($2); $1 $3 int($4); $1++)");

    // Wallpaper Engine shaders also rely on implicit vector narrowing in assignments like
    // `vec2 uv = someVec4;`. GLSL does not allow that, so apply the obvious swizzle.
    result = std::regex_replace(
        result,
        std::regex(R"(\bvec2\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*;)"),
        "vec2 $1 = $2.xy;");
    result = std::regex_replace(
        result,
        std::regex(R"(\bvec3\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*;)"),
        "vec3 $1 = $2.xyz;");

    // Some effect fragment shaders write back into stage inputs such as v_TexCoord. GLSL
    // forbids that, so rewrite those inputs to immutable varyings plus mutable locals.
    if (unit.stage == ShaderType::FRAGMENT) {
        std::string init_lines;
        for (const auto& [name, decl] : cur.input) {
            if (! HasMutableUse(result, name)) continue;

            std::string mutable_name;
            std::string init_line;
            if (! MakeMutableInputShim(decl, name, mutable_name, init_line)) {
                continue;
            }
            if (! ReplaceNameInMainBody(result, name, mutable_name)) continue;
            init_lines += init_line;
        }

        if (! init_lines.empty()) {
            result = std::regex_replace(result,
                                        std::regex(R"(void\s+main\s*\(\s*(?:void)?\s*\)\s*\{)"),
                                        "void main() {\n" + init_lines,
                                        std::regex_constants::format_first_only);
        }
    }

    // HLSL-style shaders often write scalar-first max/min calls such as max(0.0, vec2Expr).
    // GLSL only accepts the scalar as the second argument for vector overloads.
    result = std::regex_replace(
        result,
        std::regex(
            R"(\b(max|min)\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*((?:[^(),\n]+|\([^()]*\))+)\s*\))"),
        "$1($3, $2)");

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
    out << "prepared-shader-v2\n";
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
    out << "pre-shader-v1\n";
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

void WPShaderParser::InitGlslang() { glslang::InitializeProcess(); }
void WPShaderParser::FinalGlslang() { glslang::FinalizeProcess(); }

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
