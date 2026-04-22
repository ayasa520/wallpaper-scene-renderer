#pragma once
#include <cstdint>
#include <unordered_map>
#include <cstdint>
#include "WPJson.hpp"
#include "WPUserProperties.hpp"
#include <nlohmann/json.hpp>

namespace wallpaper
{
namespace fs
{
class VFS;
}

namespace wpscene
{

struct WPSoundObject {
    int32_t                  id { 0 };
    std::string              playbackmode { "loop" };
    float                    maxtime { 10.0f };
    float                    mintime { 0.0f };
    float                    volume { 1.0f };
    bool                     visible { true };
    VisibleBinding           visible_binding;
    std::string              name;
    std::vector<std::string> sound;

    bool FromJson(const nlohmann::json& json, fs::VFS&) {
        GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
        GET_JSON_NAME_VALUE(json, "volume", volume);
        GET_JSON_NAME_VALUE(json, "playbackmode", playbackmode);
        GET_JSON_NAME_VALUE_NOWARN(json, "mintime", mintime);
        GET_JSON_NAME_VALUE_NOWARN(json, "maxtime", maxtime);
        GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
        if (json.contains("visible") && json.at("visible").is_object()) {
            const auto& visible_json = json.at("visible");
            GET_JSON_NAME_VALUE_NOWARN(visible_json, "value", visible_binding.value);
            if (visible_json.contains("user") && ! visible_json.at("user").is_null()) {
                const auto& user = visible_json.at("user");
                if (user.is_string()) {
                    GET_JSON_VALUE(user, visible_binding.user.name);
                } else if (user.is_object()) {
                    GET_JSON_NAME_VALUE_NOWARN(user, "name", visible_binding.user.name);
                    GET_JSON_NAME_VALUE_NOWARN(user, "condition", visible_binding.user.condition);
                }
            }
        }
        GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
        if (! json.contains("sound") || ! json.at("sound").is_array()) {
            return false;
        }
        for (const auto& el : json.at("sound")) {
            std::string name;
            GET_JSON_VALUE(el, name);
            if (! name.empty()) sound.push_back(name);
        }
        return true;
    }
};
} // namespace wpscene
} // namespace wallpaper
