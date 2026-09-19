#include "WPScriptSource.hpp"

#include <optional>
#include <vector>

namespace wallpaper {
namespace {

size_t SpaceLength(std::string_view source, size_t offset) {
    const unsigned char c = static_cast<unsigned char>(source[offset]);
    if (c == ' ' || (c >= '\t' && c <= '\r')) return 1;
    if (c < 0x80) return 0;
    // ECMAScript whitespace is syntax even when encoded with multiple UTF-8 bytes.
    // Recognize it at token boundaries instead of normalizing the entire script:
    // the same bytes inside a string or template are authored data.
    static constexpr std::string_view spaces[] = {
        "\u00a0", "\u1680", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005",
        "\u2006", "\u2007", "\u2008", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f",
        "\u205f", "\u3000", "\ufeff",
    };
    for (const auto space : spaces) {
        if (source.substr(offset).starts_with(space)) return space.size();
    }
    return 0;
}

bool IsWordByte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '$' || c >= 0x80;
}

struct Token {
    size_t begin {};
    size_t end {};
    bool literal {};
};

class SourceTokens {
public:
    explicit SourceTokens(std::string_view source, size_t offset = 0)
        : m_source(source), m_offset(offset) {}

    Token next() {
        skipTrivia();
        Token token { m_offset, m_offset, false };
        if (m_offset == m_source.size()) return token;
        const char c = m_source[m_offset++];
        if (c == '\'' || c == '"') {
            skipQuoted(c);
            token.literal = true;
        } else if (c == '`') {
            skipTemplate();
            token.literal = true;
        } else if (c == '/' && m_expression_start) {
            bool character_class = false;
            while (m_offset < m_source.size()) {
                const char current = m_source[m_offset++];
                if (current == '\\') skipEscape();
                else if (current == '[') character_class = true;
                else if (current == ']') character_class = false;
                else if (current == '/' && !character_class) break;
                else if (current == '\r' || current == '\n') break;
            }
            while (m_offset < m_source.size() &&
                   IsWordByte(static_cast<unsigned char>(m_source[m_offset])) &&
                   SpaceLength(m_source, m_offset) == 0) ++m_offset;
            token.literal = true;
        } else if (IsWordByte(static_cast<unsigned char>(c))) {
            while (m_offset < m_source.size() &&
                   IsWordByte(static_cast<unsigned char>(m_source[m_offset])) &&
                   SpaceLength(m_source, m_offset) == 0) ++m_offset;
        } else if (m_offset < m_source.size()) {
            const auto pair = m_source.substr(token.begin, 2);
            if (pair == "=>" || pair == "++" || pair == "--" || pair == "?.") ++m_offset;
        }
        token.end = m_offset;
        updateContext(token);
        return token;
    }

    std::string_view text(Token token) const {
        return m_source.substr(token.begin, token.end - token.begin);
    }

private:
    struct Delimiter {
        bool expression_after {};
        std::optional<bool> function_expression;
    };

    void skipEscape() {
        if (m_offset < m_source.size()) ++m_offset;
    }

    void skipQuoted(char quote) {
        while (m_offset < m_source.size()) {
            const char c = m_source[m_offset++];
            if (c == '\\') skipEscape();
            else if (c == quote) break;
        }
    }

    void skipTrivia() {
        while (m_offset < m_source.size()) {
            if (const auto count = SpaceLength(m_source, m_offset)) {
                m_offset += count;
            } else if (m_source.substr(m_offset).starts_with("//")) {
                m_offset += 2;
                while (m_offset < m_source.size() && m_source[m_offset] != '\r' &&
                       m_source[m_offset] != '\n' &&
                       !m_source.substr(m_offset).starts_with("\u2028") &&
                       !m_source.substr(m_offset).starts_with("\u2029")) ++m_offset;
            } else if (m_source.substr(m_offset).starts_with("/*")) {
                const auto end = m_source.find("*/", m_offset + 2);
                m_offset = end == std::string_view::npos ? m_source.size() : end + 2;
            } else {
                break;
            }
        }
    }

    void skipTemplate() {
        // Treat the complete template as one literal for source adaptation. Expressions
        // can contain nested templates, quoted braces and regular expressions; tokenize
        // those expressions recursively so none of their text becomes a module declaration.
        while (m_offset < m_source.size()) {
            const char c = m_source[m_offset++];
            if (c == '\\') skipEscape();
            else if (c == '`') break;
            else if (c == '$' && m_offset < m_source.size() && m_source[m_offset] == '{') {
                SourceTokens expression(m_source, ++m_offset);
                size_t depth = 1;
                while (depth != 0) {
                    const auto token = expression.next();
                    if (token.begin == token.end) break;
                    if (!token.literal) {
                        if (expression.text(token) == "{") ++depth;
                        else if (expression.text(token) == "}") --depth;
                    }
                }
                m_offset = expression.m_offset;
            }
        }
    }

    void updateContext(Token token) {
        const auto word = text(token);
        if (token.literal) {
            m_expression_start = false;
            m_statement_start = false;
        } else if (word == "(") {
            const bool control = m_previous == "if" || m_previous == "while" ||
                                 m_previous == "for" || m_previous == "with" ||
                                 m_previous == "switch" || m_previous == "catch";
            m_delimiters.push_back({ control, m_function_expression });
            m_function_expression.reset();
            m_expression_start = true;
            m_statement_start = false;
        } else if (word == "{" || word == "[") {
            bool after = false;
            if (word == "{") {
                after = m_statement_start || !m_expression_start;
                if (m_body_expression.has_value()) after = !*m_body_expression;
                if (m_class_expression && m_class_expression->first == m_delimiters.size()) {
                    after = !m_class_expression->second;
                    m_class_expression.reset();
                }
                if (m_previous == "=>") after = false;
                m_body_expression.reset();
            }
            m_delimiters.push_back({ after, std::nullopt });
            m_expression_start = true;
            m_statement_start = word == "{" && after;
        } else if (word == ")" || word == "}" || word == "]") {
            Delimiter delimiter;
            if (!m_delimiters.empty()) {
                delimiter = m_delimiters.back();
                m_delimiters.pop_back();
            }
            m_body_expression = delimiter.function_expression;
            m_expression_start = delimiter.expression_after;
            m_statement_start = delimiter.expression_after;
        } else if (IsWordByte(static_cast<unsigned char>(word.front()))) {
            const bool property = m_previous == "." || m_previous == "?.";
            if (word == "function" && !property) m_function_expression = !m_statement_start;
            if (word == "class" && !property)
                m_class_expression = std::pair { m_delimiters.size(), !m_statement_start };
            m_expression_start = !property &&
                (word == "return" || word == "throw" || word == "case" || word == "delete" ||
                 word == "void" || word == "typeof" || word == "new" || word == "yield" ||
                 word == "await" || word == "in" || word == "instanceof" || word == "of");
            m_statement_start = !property && (word == "export" || word == "else" ||
                                             word == "do" || word == "try" || word == "finally" ||
                                             (word == "async" && m_statement_start));
            if (m_statement_start) m_expression_start = true;
        } else if (word != "++" && word != "--") {
            // A slash after an expression is division. At an expression boundary it
            // starts a regexp, whose braces, quotes and keywords must stay opaque.
            // Control-condition parentheses and function bodies restore their own
            // boundary state instead of treating every closing parenthesis alike.
            m_expression_start = word != "." && word != "?.";
            m_statement_start = word == ";";
        }
        m_previous = word;
    }

    std::string_view m_source;
    size_t m_offset {};
    bool m_expression_start { true };
    bool m_statement_start { true };
    std::string_view m_previous;
    std::vector<Delimiter> m_delimiters;
    std::optional<bool> m_function_expression;
    std::optional<bool> m_body_expression;
    std::optional<std::pair<size_t, bool>> m_class_expression;
};

void BlankSyntax(std::string& result, size_t begin, size_t end) {
    // Keep byte offsets and line breaks stable for diagnostics. Only declaration
    // syntax is blanked: the scanner never returns literal contents as keywords.
    for (size_t index = begin; index < end; ++index) {
        if (std::string_view(result).substr(index).starts_with("\u2028") ||
            std::string_view(result).substr(index).starts_with("\u2029")) index += 2;
        else if (result[index] != '\r' && result[index] != '\n') result[index] = ' ';
    }
}

}

std::string AdaptSceneScriptSource(std::string_view source) {
    std::string result(source);
    SourceTokens tokens(source);
    size_t depth = 0;
    std::string_view previous;
    while (true) {
        const auto token = tokens.next();
        if (token.begin == token.end) break;
        const auto word = tokens.text(token);
        if (token.literal) {
            previous = word;
            continue;
        }
        if (depth == 0 && previous != "." && previous != "?." &&
            (word == "export" || word == "import")) {
            auto lookahead = tokens;
            const auto following = lookahead.next();
            const auto declaration = lookahead.text(following);
            if (word == "export" && (declaration == "function" || declaration == "var" ||
                                     declaration == "let" || declaration == "const" ||
                                     declaration == "class" || declaration == "async")) {
                BlankSyntax(result, token.begin, token.end);
            } else if (word == "import" && declaration != "(" && declaration != ".") {
                // The callback environment already provides the named built-in modules.
                // Remove a static declaration through its module specifier, not through
                // the end of its line: another callback may start on that same line.
                // Dynamic import and import.meta remain expressions for the JS compiler.
                auto specifier = following;
                bool from = false;
                while (specifier.begin != specifier.end && !specifier.literal &&
                       lookahead.text(specifier) != ";") {
                    from = lookahead.text(specifier) == "from";
                    specifier = lookahead.next();
                    if (specifier.literal) break;
                }
                if (specifier.literal && (following.literal || from) &&
                    (source[specifier.begin] == '\'' || source[specifier.begin] == '"')) {
                    auto end = specifier.end;
                    auto after = lookahead;
                    const auto terminal = after.next();
                    if (after.text(terminal) == ";") {
                        lookahead = after;
                        end = terminal.end;
                    }
                    BlankSyntax(result, token.begin, end);
                    tokens = lookahead;
                }
            }
        }
        if (word == "{" || word == "(" || word == "[") ++depth;
        else if ((word == "}" || word == ")" || word == "]") && depth != 0) --depth;
        previous = word;
    }
    return result;
}

}
