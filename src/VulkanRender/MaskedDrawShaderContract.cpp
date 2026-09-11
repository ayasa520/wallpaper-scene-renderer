#include "MaskedDrawShaderContract.hpp"

#include "Utils/Logging.h"

#include <algorithm>

namespace wallpaper::vulkan
{

bool MaskedDrawShaderContract::hasUniform(std::string_view name) const {
    return !reflection.blocks.empty() && reflection.blocks.front().member_map.contains(name);
}

bool ReflectMaskedDrawShaderContract(const SceneMesh& mesh, const SceneMaterial& material,
                                     std::vector<Uni_ShaderSpv>& stages,
                                     MaskedDrawShaderContract& contract) {
    contract = {};
    contract.texture_bindings.fill(-1);
    if (!GenReflect(material.customShader.shader->codes, stages, contract.reflection)) {
        LOG_ERROR("MaskedDrawReflect: shader='%s' reflection failed", material.name.c_str());
        return false;
    }
    const auto& reflection = contract.reflection;
    if (reflection.blocks.size() > 1) {
        LOG_ERROR("MaskedDrawReflect: shader='%s' uniform blocks=%zu expected-at-most=1",
                  material.name.c_str(), reflection.blocks.size());
        return false;
    }
    contract.descriptors.push_descriptor = true;
    for (const auto& [name, binding] : reflection.binding_map) {
        contract.descriptors.bindings.push_back(binding);
        if (binding.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            contract.uniform_binding = binding.binding;
        }
    }
    for (size_t slot = 0; slot < WE_GLTEX_NAMES.size(); ++slot) {
        const auto it = reflection.binding_map.find(WE_GLTEX_NAMES[slot]);
        if (it != reflection.binding_map.end()) {
            contract.texture_bindings[slot] = static_cast<int32_t>(it->second.binding);
        }
    }

    // Reflected input locations are program-specific, while mesh stream names and offsets
    // remain shared with the visible draw. Locate each attribute in exactly one stream;
    // another executable's numeric locations do not describe this program's vertex ABI.
    for (uint32_t stream = 0; stream < mesh.VertexCount(); ++stream) {
        const auto& vertices = mesh.GetVertexArray(stream);
        contract.bindings.push_back({
            .binding = stream,
            .stride = static_cast<uint32_t>(vertices.OneSizeOf()),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        });
        const auto attributes = vertices.GetAttrOffsetMap();
        for (const auto& [name, input] : reflection.input_location_map) {
            const auto attribute = attributes.find(name);
            if (attribute == attributes.end()) continue;
            contract.attributes.push_back({
                .location = input.location,
                .binding = stream,
                .format = input.format,
                .offset = static_cast<uint32_t>(attribute->second.offset),
            });
        }
    }
    for (const auto& [name, input] : reflection.input_location_map) {
        const auto count = std::count_if(contract.attributes.begin(), contract.attributes.end(),
            [&](const auto& attribute) { return attribute.location == input.location; });
        if (count != 1) {
            LOG_ERROR("MaskedDrawReflect: shader='%s' input='%s' streams=%zu expected=1",
                      material.name.c_str(), name.c_str(), static_cast<size_t>(count));
            return false;
        }
    }
    return true;
}

} // namespace wallpaper::vulkan
