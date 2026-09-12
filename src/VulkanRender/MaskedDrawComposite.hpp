#pragma once

#include "Resource.hpp"
#include "Vulkan/GraphicsPipeline.hpp"

namespace wallpaper::vulkan
{

// Multiplies completed single-sample mask coverage. Source shading and the enclosing
// destination stay with MaskedDrawRenderer; this pass owns only the R8 composition program.
class MaskedDrawComposite {
public:
    bool prepare(const Device&, RenderingResources&);
    void record(RenderingResources&, VkFramebuffer, VkExtent3D, const ImageParameters&) const;
    void destroy();

private:
    PipelineParameters m_pipeline;
};

} // namespace wallpaper::vulkan
