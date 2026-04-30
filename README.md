# Wallpaper Engine Scene Renderer

Open source scene renderer, mostly for linux.  
Made this for fun.

<!-- Documentation note: this README tracks the compatibility layer used by Hanabi's native
scene backend. Items marked as supported can still be compatibility subsets rather than a complete
Wallpaper Engine editor/runtime clone. -->

- vulkan 1.1
- render graph for automatic pass dependencies
- QuickJS-backed compatibility layer for common SceneScript property bindings
- Pango/Cairo text rasterization for first-class text layers

## Supported

- [x] Layer
  - [x] Image
  - [x] Composition / Fullscreen
  - [x] Text
  - [x] Sound
  - [x] Particle
  - [x] Light
  - [x] Shape direct draw
- [x] Effect
  - [x] Basic
  - [x] Mouse position with delay
  - [x] Parallax
    - [x] Depth Parallax
  - [x] ColorBlendMode
  - [x] PBR light
  - [x] Global bloom
- [x] Camera
  - [x] Zoom
  - [x] Path
  - [ ] Shake
  - [ ] Fade
- [x] Audio
  - [x] Loop
  - [x] Random
  - [x] SceneScript audio buffers from host-provided spectrum data
- [x] Particle System
  - [x] Renderers
  - [x] Emitters
    - [ ] Duration
  - [x] Initializers
  - [x] Operators
  - [x] Control Points
    - [x] Control point force / attract / repel
    - [ ] Full editor mouse-follow feature set
  - [x] Children
    - [ ] Audio Response
- [x] Puppet warp
- [x] 3D model
- [x] Property / timeline animations
- [x] SceneScript compatibility subset
  - [x] `init`, `update`, `destroy`, timers, and property scripts
  - [x] `applyUserProperties` and `applyGeneralSettings`
  - [x] Cursor enter/leave/move/down/up/click events
  - [x] Media thumbnail/properties/playback events
  - [x] Texture animation and video texture control helpers
  - [ ] Complete Wallpaper Engine API surface
- [x] User Properties
