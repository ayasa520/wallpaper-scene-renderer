#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <string_view>

namespace wallpaper::wpscene
{

inline int32_t ReadEffectInteger(const nlohmann::json& value) {
    // Condition operands and texture slots share raw signed-32 conversion, independently
    // of dimension admission. Integers retain their low word, and doubles truncate toward
    // zero. An unrepresentable double produces the signed minimum; nonnumeric JSON supplies
    // zero without unwrapping property objects or interpreting booleans/strings as numbers.
    if (value.is_number_unsigned()) {
        return std::bit_cast<int32_t>(static_cast<uint32_t>(value.get<uint64_t>()));
    }
    if (value.is_number_integer()) {
        return std::bit_cast<int32_t>(static_cast<uint32_t>(value.get<int64_t>()));
    }
    if (value.is_number_float()) {
        const double integer = std::trunc(value.get<double>());
        constexpr auto minimum = std::numeric_limits<int32_t>::min();
        constexpr auto maximum = std::numeric_limits<int32_t>::max();
        return integer >= minimum && integer <= maximum ? static_cast<int32_t>(integer) : minimum;
    }
    return 0;
}

inline bool MatchesEffectConditions(const nlohmann::json& record, const nlohmann::json& combos) {
    const auto conditions = record.find("conditions");
    if (conditions == record.end() || !conditions->is_array()) return true;

    // A single construction predicate controls framebuffer, pass and texture-binding
    // admission. Array entries and object keys are conjunctive; unsupported entries impose
    // no constraint. Read the raw instance combo table so a wrapped property cannot silently
    // change topology. Apply this before resource loading or binding resolution, including
    // when a pass remains admitted while one of its optional inputs is excluded.
    for (const auto& group : *conditions) {
        if (!group.is_object()) continue;
        for (const auto& [key, rule] : group.items()) {
            const auto combo = combos.find(key);
            const int32_t actual = combo == combos.end() ? 0 : ReadEffectInteger(*combo);
            if (rule.is_number()) {
                if (actual != ReadEffectInteger(rule)) return false;
            } else if (rule.is_object()) {
                const auto value = rule.find("value");
                const int32_t expected = value == rule.end() ? 0 : ReadEffectInteger(*value);
                const auto operation = rule.find("op");
                const std::string_view op = operation != rule.end() && operation->is_string()
                    ? operation->get_ref<const std::string&>() : std::string_view();
                bool matches;
                if (op == "ge") matches = actual >= expected;
                else if (op == "gt") matches = actual > expected;
                else if (op == "le") matches = actual <= expected;
                else if (op == "lt") matches = actual < expected;
                else matches = actual == expected;
                if (!matches) return false;
            }
        }
    }
    return true;
}

} // namespace wallpaper::wpscene
