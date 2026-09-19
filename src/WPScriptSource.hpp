#pragma once

#include <string>
#include <string_view>

namespace wallpaper {

// Adapt the declaration syntax used by the scene callback wrapper without changing
// authored literal data or explicit strict directives. Import bindings are supplied
// by that wrapper; resolving additional modules is a separate host responsibility.
std::string AdaptSceneScriptSource(std::string_view source);

}
