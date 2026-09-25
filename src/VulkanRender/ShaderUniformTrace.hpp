#pragma once

#include "FrameTraceDump.hpp"
#include "Scene/SceneShader.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace wallpaper::vulkan
{

struct TracedUniformWrite {
    std::string name;
    ShaderValue value;
    std::size_t offset;
    std::size_t reflected_size;
};

// Uniform uploads happen before command recording, including during graph preparation. Keep
// diagnostic copies here until the corresponding draw consumes them. The selected empty type
// owns no vector and returns a constant false at the update boundary, so production performs
// neither extra reflection lookups nor payload copies in its upload callback.
class ShaderUniformRecorder {
public:
    bool beginUpdate() {
        const bool active =
            (wallpaper::diagnostics::Options().trace_render_commands &&
             wallpaper::diagnostics::Options().trace_render_uniforms) ||
            FrameTraceDump::Configured();
        m_writes.clear();
        return active;
    }

    void record(std::string_view name, const ShaderValue& value,
                std::size_t offset, std::size_t reflected_size) {
        m_writes.push_back({std::string(name), value, offset, reflected_size});
    }

    auto begin() const { return m_writes.begin(); }
    auto end() const { return m_writes.end(); }

private:
    std::vector<TracedUniformWrite> m_writes;
};

class DisabledShaderUniformTrace {
public:
    // The upload callback captures this result. A type-level false stays constant even when
    // that callback is stored in std::function, instead of becoming a captured runtime flag.
    static constexpr std::false_type beginUpdate() { return {}; }
    static constexpr void record(std::string_view, const ShaderValue&,
                                  std::size_t, std::size_t) {}
    static constexpr const TracedUniformWrite* begin() { return nullptr; }
    static constexpr const TracedUniformWrite* end() { return nullptr; }
};

using ShaderUniformTrace = std::conditional_t<wallpaper::diagnostics::Enabled,
                                             ShaderUniformRecorder, DisabledShaderUniformTrace>;

} // namespace wallpaper::vulkan
