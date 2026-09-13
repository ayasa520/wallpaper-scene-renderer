#include "WPMaterial.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <string_view>

using namespace wallpaper::wpscene;

namespace
{
void ReadMaterialEnum(const nlohmann::json& json, std::string_view name,
                      std::initializer_list<std::string_view> choices, std::string& value) {
    const auto entry = json.find(name);
    if (entry == json.end()) return;

    // Static raster entries consume their own literal JSON value, not a dynamic property's
    // value/user wrapper. Omission preserves the initialized state; a present non-string or
    // unrecognized spelling selects the first enum entry. In particular, those two cases
    // differ for depth state. Compare the entire stored string so embedded NUL bytes are not
    // mistaken for the end of a valid name, and do not apply live script string conversion.
    const std::string_view text =
        entry->is_string() ? entry->get_ref<const std::string&>() : std::string_view {};
    const auto selected = std::find(choices.begin(), choices.end(), text);
    value = selected == choices.end() ? *choices.begin() : *selected;
}

void TraceMaterialRenderState(const nlohmann::json& json, const WPMaterial& material) {
    if (std::getenv("WESCENE_TRACE_MATERIAL_STATE") == nullptr) return;

    // Keep raw entry types and omitted keys visible next to the selected material state.
    // The draw trace records effective owner state separately, including depth-write changes
    // required by transparent model draws; those must not overwrite the parsed material.
    auto authored = nlohmann::json::object();
    for (const auto* name : { "blending", "cullmode", "depthtest", "depthwrite", "alphawriting" }) {
        if (const auto entry = json.find(name); entry != json.end()) authored[name] = *entry;
    }
    LOG_INFO("SceneMaterialColdState: shader='%s' authored=%s blending='%s' cullmode='%s' "
             "depthtest='%s' depthwrite='%s' alphawriting='%s'",
             material.shader.c_str(), authored.dump().c_str(), material.blending.c_str(),
             material.cullmode.c_str(), material.depthtest.c_str(), material.depthwrite.c_str(),
             material.alphawriting.c_str());
}
} // namespace

bool WPMaterialPassBindItem::FromJson(const nlohmann::json& json) {
    GET_JSON_NAME_VALUE(json, "name", name);
    GET_JSON_NAME_VALUE(json, "index", index);
    return true;
}

bool WPUserTextureBinding::FromJson(const nlohmann::json& json) {
    if (json.is_string()) {
        GET_JSON_VALUE(json, name);
        return ! name.empty();
    }

    if (! json.is_object()) return false;

    GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
    GET_JSON_NAME_VALUE_NOWARN(json, "type", type);
    if (const auto entry = json.find("keepaspect"); entry != json.end() && entry->is_boolean()) {
        keepaspect = entry->get<bool>();
    }
    return ! name.empty();
}


void WPMaterialPass::Update(const WPMaterialPass& p) {
    if (!p.usertexturereference.is_null()) usertexturereference = p.usertexturereference;
    int32_t i = -1;
    for(const auto& el:p.textures) {
        i++;
        if(p.textures.size() > textures.size())
            textures.resize(p.textures.size());
        if(!el.empty()) {
            textures[i] = el;
        }
    }
    i = -1;
    for (const auto& el : p.usertextures) {
        i++;
        if (p.usertextures.size() > usertextures.size())
            usertextures.resize(p.usertextures.size());
        if (! el.empty()) {
            usertextures[i] = el;
        }
    }
    for(const auto& el:p.constantshadervalues) {
        constantshadervalues[el.first] = el.second;
    }
    for(const auto& el:p.usershadervalues) {
        usershadervalues[el.first] = el.second;
    }
    for(const auto& el:p.combos) {
        combos[el.first] = el.second;
    }
}

void WPMaterial::MergePass(const WPMaterialPass& p) {
    if (!p.usertexturereference.is_null()) usertexturereference = p.usertexturereference;
    int32_t i = -1;
    for(const auto& el:p.textures) {
        i++;
        if(p.textures.size() > textures.size())
            textures.resize(p.textures.size());
        if(!el.empty()) {
            textures[i] = el;
        }
    }
    i = -1;
    for (const auto& el : p.usertextures) {
        i++;
        if (p.usertextures.size() > usertextures.size())
            usertextures.resize(p.usertextures.size());
        if (! el.empty()) {
            usertextures[i] = el;
        }
    }
    for(const auto& el:p.constantshadervalues) {
        constantshadervalues[el.first] = el.second;
    }
    for(const auto& el:p.usershadervalues) {
        usershadervalues[el.first] = el.second;
    }
    for(const auto& el:p.combos) {
        combos[el.first] = el.second;
    }
}

bool WPMaterialPass::FromJson(const nlohmann::json& json) {
    if (const auto entry = json.find("usertexturereference"); entry != json.end()) {
        usertexturereference = *entry;
    }
    if(json.contains("textures")) {
        for(const auto& jT:json.at("textures")) {
            std::string tex;
            if(!jT.is_null())
                GET_JSON_VALUE(jT, tex);
            textures.push_back(tex);
        }
    }
    if (json.contains("usertextures")) {
        for (const auto& jT : json.at("usertextures")) {
            WPUserTextureBinding binding;
            if (! jT.is_null())
                binding.FromJson(jT);
            usertextures.push_back(binding);
        }
    }
    if(json.contains("constantshadervalues")) {
        for(const auto& jC:json.at("constantshadervalues").items()) {
            constantshadervalues[jC.key()] = jC.value();
        }
    }
    if(json.contains("usershadervalues")) {
        for(const auto& jC:json.at("usershadervalues").items()) {
            std::string name;
            std::string value;
            GET_JSON_VALUE(jC.key(), name);
            GET_JSON_VALUE(jC.value(), value);
            usershadervalues[name] = value;
        }
    }
    if(json.contains("combos")) {
        for(const auto& jC:json.at("combos").items()) {
            std::string name;
            int32_t value;
            GET_JSON_VALUE(jC.key(), name);
            GET_JSON_VALUE(jC.value(), value);
            combos[name] = value;
        }
    }
    GET_JSON_NAME_VALUE_NOWARN(json, "target", target);
    GET_JSON_NAME_VALUE_NOWARN(json, "compose", compose);
    if(json.contains("bind")) {
        for(const auto& jB:json.at("bind")) {
            WPMaterialPassBindItem bindItem;
            bindItem.FromJson(jB);
            bind.push_back(bindItem);
        }
    }
    return true;
}

bool WPMaterial::FromJson(const nlohmann::json& json) {
    if(!json.contains("passes") || json.at("passes").size() == 0) {
        LOG_ERROR("material no data");
        return false;
    }
    const auto jContent = json.at("passes").at(0);
    if (const auto entry = jContent.find("usertexturereference"); entry != jContent.end()) {
        usertexturereference = *entry;
    }
    if(!jContent.contains("shader")) {
        LOG_ERROR("material no shader");
        return false;
    }
    ReadMaterialEnum(jContent, "blending",
                     { "normal", "translucent", "additive", "alphatocoverage" }, blending);
    ReadMaterialEnum(jContent, "cullmode", { "normal", "nocull" }, cullmode);
    ReadMaterialEnum(jContent, "alphawriting", { "default", "disabled", "enabled" }, alphawriting);
    ReadMaterialEnum(jContent, "depthtest", { "disabled", "enabled" }, depthtest);
    ReadMaterialEnum(jContent, "depthwrite", { "disabled", "enabled" }, depthwrite);
	GET_JSON_NAME_VALUE(jContent, "shader", shader);
    TraceMaterialRenderState(jContent, *this);
    if(jContent.contains("textures")) {
        for(const auto& jT:jContent.at("textures")) {
            std::string tex;
            if(!jT.is_null())
                GET_JSON_VALUE(jT, tex);
            textures.push_back(tex);
        }
    }
    if (jContent.contains("usertextures")) {
        for (const auto& jT : jContent.at("usertextures")) {
            WPUserTextureBinding binding;
            if (! jT.is_null())
                binding.FromJson(jT);
            usertextures.push_back(binding);
        }
    }
    if(jContent.contains("constantshadervalues")) {
        for(const auto& jC:jContent.at("constantshadervalues").items()) {
            constantshadervalues[jC.key()] = jC.value();
        }
    }
    if(jContent.contains("usershadervalues")) {
        for(const auto& jC:jContent.at("usershadervalues").items()) {
            std::string name;
            std::string value;
            GET_JSON_VALUE(jC.key(), name);
            GET_JSON_VALUE(jC.value(), value);
            usershadervalues[name] = value;
        }
    }
    if(jContent.contains("combos")) {
        for(const auto& jC:jContent.at("combos").items()) {
            std::string name;
            int32_t value;
            GET_JSON_VALUE(jC.key(), name);
            GET_JSON_VALUE(jC.value(), value);
            combos[name] = value;
        }
    }
    return true;
}
