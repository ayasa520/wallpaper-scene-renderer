#include "WPLightObject.hpp"

#include "Utils/Logging.h"
#include "Fs/VFS.h"

using namespace wallpaper::wpscene;

namespace
{

void ReadVisibleBinding(const nlohmann::json& json, wallpaper::VisibleBinding* binding) {
    if (! json.is_object()) return;

    GET_JSON_NAME_VALUE_NOWARN(json, "value", binding->value);
    if (! json.contains("user") || json.at("user").is_null()) return;

    const auto& user = json.at("user");
    if (user.is_string()) {
        GET_JSON_VALUE(user, binding->user.name);
        return;
    }

    if (! user.is_object()) return;

    GET_JSON_NAME_VALUE_NOWARN(user, "name", binding->user.name);
    GET_JSON_NAME_VALUE_NOWARN(user, "condition", binding->user.condition);
}

} // namespace

bool WPLightObject::FromJson(const nlohmann::json& json,  fs::VFS&) {
    GET_JSON_NAME_VALUE(json, "origin", origin);
    GET_JSON_NAME_VALUE(json, "angles", angles);
    GET_JSON_NAME_VALUE(json, "scale", scale);
    GET_JSON_NAME_VALUE(json, "color", color);
    GET_JSON_NAME_VALUE(json, "light", light);
    GET_JSON_NAME_VALUE(json, "radius", radius);
    GET_JSON_NAME_VALUE(json, "intensity", intensity);
    GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
    if (json.contains("visible")) ReadVisibleBinding(json.at("visible"), &visible_binding);
    GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
    GET_JSON_NAME_VALUE_NOWARN(json, "parent", parent);
    GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
    GET_JSON_NAME_VALUE_NOWARN(json, "parallaxDepth", parallaxDepth);
    GET_JSON_NAME_VALUE_NOWARN(json, "castvolumetrics", castvolumetrics);
    GET_JSON_NAME_VALUE_NOWARN(json, "density", density);
    GET_JSON_NAME_VALUE_NOWARN(json, "volumetricsexponent", volumetricsexponent);
    GET_JSON_NAME_VALUE_NOWARN(json, "innercone", innercone);
    GET_JSON_NAME_VALUE_NOWARN(json, "outercone", outercone);
    GET_JSON_NAME_VALUE_NOWARN(json, "usecookie", usecookie);
    GET_JSON_NAME_VALUE_NOWARN(json, "cookie", cookie);
    // Workshop and editor assets write `castshadow`. Accept the plural alias
    // so older hand-authored scene.json still enables the same light flag.
    if (json.contains("castshadow")) {
        GET_JSON_NAME_VALUE_NOWARN(json, "castshadow", castshadows);
    } else {
        GET_JSON_NAME_VALUE_NOWARN(json, "castshadows", castshadows);
    }
    GET_JSON_NAME_VALUE_NOWARN(json, "exponent", exponent);
    GET_JSON_NAME_VALUE_NOWARN(json, "cascadedistance0", cascadedistance0);
    GET_JSON_NAME_VALUE_NOWARN(json, "cascadedistance1", cascadedistance1);
    GET_JSON_NAME_VALUE_NOWARN(json, "cascadedistance2", cascadedistance2);
    return true;
}
