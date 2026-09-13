#include "WPScene.h"

using namespace wallpaper::wpscene;

bool Orthogonalprojection::FromJson(const nlohmann::json& json) {
    if (!json.is_object()) return false;

    // Projection configuration has a literal schema: only boolean true requests an
    // automatic canvas. False and differently typed auto entries still allow an
    // explicit numeric pair; they must not hide authored width and height.
    auto_ = wallpaper::ReadJsonLiteralBoolean(json, "auto", false);
    if (auto_) return true;

    const auto width_value = json.find("width");
    const auto height_value = json.find("height");
    if (width_value == json.end() || !width_value->is_number() ||
        height_value == json.end() || !height_value->is_number()) {
        return false;
    }

    // Admit both numeric values before storing either dimension. Generic property
    // expansion would also accept wrappers or mix one authored dimension with a
    // default. The integer canvas conversion precedes the scene's zero-size gate.
    width = width_value->get<int32_t>();
    height = height_value->get<int32_t>();
    return true;
}

bool WPSceneCamera::FromJson(const nlohmann::json& json) {
    GET_JSON_NAME_VALUE(json, "center", center);
    GET_JSON_NAME_VALUE(json, "eye", eye);
    GET_JSON_NAME_VALUE(json, "up", up);
    if (json.contains("paths") && json.at("paths").is_array()) {
        // Camera path assets are consumed only by the 3D model camera parser. Recording the list
        // here is inert for 2D scenes until WPSceneParser explicitly enables model camera playback.
        paths.clear();
        for (const auto& path : json.at("paths")) {
            if (path.is_string()) paths.push_back(path.get<std::string>());
        }
    }
    return true;
}

bool WPSceneGeneral::FromJson(const nlohmann::json& json) {
    // Scene clearing starts enabled and only a JSON boolean changes it. A dynamic property's
    // value obeys the same rule; strings and numbers do not coerce to bool and must not turn an
    // omitted/invalid value into a disabled clear.
    if (auto it = json.find("clearenabled"); it != json.end()) {
        const auto& value = it->is_object() && it->contains("value") ? it->at("value") : *it;
        if (value.is_boolean()) clearenabled = value.get<bool>();
    }
    GET_JSON_NAME_VALUE(json, "ambientcolor", ambientcolor);
    GET_JSON_NAME_VALUE(json, "skylightcolor", skylightcolor);
	GET_JSON_NAME_VALUE(json, "clearcolor", clearcolor);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloom", bloom);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomstrength", bloomstrength);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomthreshold", bloomthreshold);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomtint", bloomtint);
	GET_JSON_NAME_VALUE_NOWARN(json, "hdr", hdr);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomhdrstrength", bloomhdrstrength);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomhdrthreshold", bloomhdrthreshold);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomhdrscatter", bloomhdrscatter);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomhdrfeather", bloomhdrfeather);
	GET_JSON_NAME_VALUE_NOWARN(json, "bloomhdriterations", bloomhdriterations);
	GET_JSON_NAME_VALUE(json, "cameraparallax", cameraparallax);
	GET_JSON_NAME_VALUE(json, "cameraparallaxamount", cameraparallaxamount);
	GET_JSON_NAME_VALUE(json, "cameraparallaxdelay", cameraparallaxdelay);
	GET_JSON_NAME_VALUE(json, "cameraparallaxmouseinfluence", cameraparallaxmouseinfluence);
	GET_JSON_NAME_VALUE_NOWARN(json, "camerashake", camerashake);
	GET_JSON_NAME_VALUE_NOWARN(json, "camerashakeamplitude", camerashakeamplitude);
	GET_JSON_NAME_VALUE_NOWARN(json, "camerashakeroughness", camerashakeroughness);
	GET_JSON_NAME_VALUE_NOWARN(json, "camerashakespeed", camerashakespeed);
	GET_JSON_NAME_VALUE_NOWARN(json, "zoom", zoom);
	GET_JSON_NAME_VALUE_NOWARN(json, "fov", fov);
	GET_JSON_NAME_VALUE_NOWARN(json, "perspectiveoverridefov", perspectiveoverridefov);
	GET_JSON_NAME_VALUE_NOWARN(json, "nearz", nearz);
	GET_JSON_NAME_VALUE_NOWARN(json, "farz", farz);
    // Perspective remains selected unless a complete typed configuration requests
    // orthographic mode. Test the converted dimensions, so a fractional value that
    // truncates to zero cannot select a canvas projection with a zero dimension.
    isOrtho = false;
    if (const auto ortho = json.find("orthogonalprojection");
        ortho != json.end() && orthogonalprojection.FromJson(*ortho)) {
        isOrtho = orthogonalprojection.auto_ ||
                  (orthogonalprojection.width != 0 && orthogonalprojection.height != 0);
    }
    return true;
}

bool WPScene::FromJson(const nlohmann::json& json) {
    GET_JSON_NAME_VALUE_NOWARN(json, "version", version);
    if(json.contains("camera")) {
        camera.FromJson(json.at("camera"));
    } else {
        LOG_ERROR("scene no camera");
        return false;
    }
    if(json.contains("general")) {
        general.FromJson(json.at("general"));
    } else {
        LOG_ERROR("scene no genera data");
        return false;
    }
    return true; 
}
