## Image models
- material (STRING): Set "material": "materials/MYMATERIAL.json" to specify the material for the image model.
- height (INT): Set "height": 1024 to specify the height in pixels.
- width (INT): Set "width": 1024 to specify the width in pixels.
- fullscreen (BOOL): Set "fullscreen": true to create a signed unit quad that you can render as a fullscreen pass. Skip model/view transforms in the shader with e.g. gl_Position = vec4(a_Position, 1.0);

## Textures
A compiler config JSON file to control how your texture is compiled  
- format (STRING): Set "format" to one of these values: "rgba8888", "rgba8888n", "dxt5", "dxt5+", "dxt5n", "dxt5n+" (or dxt3, dxt1 respectively). The 'n' suffix will enable normal map swizzling for compressed normal maps (use the shader utility DecompressNormal to decompress). The '+' suffix will enable a higher quality compression method.
- nointerpolation (BOOL): Set "nointerpolation": true to disable bilinear interpolation.
- clampuvs (BOOL): Set "clampuvs": true to enable UV clamping.
- nomip (BOOL): Set "nomip": true to disable mip map generation and mip mapping for this texture.
- nonpoweroftwo (BOOL): Set "nonpoweroftwo": true to allow non power of two dimensions. This will pad the texture with black to fill missing pixels and round up to the next power of two.
- croptoaspectratio (FLOAT): Set "croptoaspectratio": 1.0 To crop the texture to a specific aspect ratio, like 1.0 in this example.

## Materials
- blending (STRING): Set "blending" to "translucent" or "additive" to enable translucent or additive blending respectively.
- depthtest (STRING): Set "depthtest" to "disabled" to disable depth testing.
- depthwrite (STRING): Set "depthwrite" to "disabled" to disable depth writing.
- cullmode (STRING): Set "cullmode" to "nocull" to disable culling.
- shader (STRING): Set "shader" to any name of a shader that exists in your shader/ directory. A .frag and .vert will be required with this name.
- textures (ARRAY of STRINGS): Set "textures" to an array of texture names. The textures will be bound to g_Texture0, g_Texture1, g_TextureX respectively. Special texture names will allow you to access rendertargets: "_rt_FullFrameBuffer", "_rt_Reflection".
- constantshadervalues (OBJECT): Defines constant values used by uniforms from the shader.
- combos (OBJECT): Defines combo constants (preprocessor) set when compiling the shader. Use this to enable/disables features and build an 'uber shader'.
- usershadervalues (OBJECT): Links uniform values to user values, which show up in the wallpaper settings dialog for a user to change.

<!-- Documentation note: the renderer accepts many Wallpaper Engine value wrappers directly from
scene.json. Documenting the shared wrapper shape here avoids repeating the same rule for each layer,
material, text, effect, and script-controlled property. -->
## User-bound, scripted, and animated values
Many numeric, vector, color, text, visibility, material, and transform fields may be authored either as a literal value or as an object that wraps the value.

- value (VARYING): Set `"value"` to the fallback or initial value that should be used before user bindings, scripts, or animations update the property.
- user (STRING or OBJECT): Bind the value to a user property. A string uses that property name directly; an object can provide `"name"` and `"condition"` so the binding only applies when the user property condition matches.
- script (STRING): Evaluate a SceneScript property script. The renderer supports the common parser-time and runtime `init` / `update` flow for scene, layer, text, material, sound, and effect properties.
- scriptproperties (OBJECT): Supplies nested properties for a property script. Each nested property may use the same literal/user/script/animation wrapper form.
- animation (OBJECT): Defines keyframed property animation data. Loop and mirror animation modes are supported for numeric and vector properties.

## Particle systems
Particle JSON is parsed from the Wallpaper Engine particle object data.

- Initializers: `colorrandom`, `lifetimerandom`, `sizerandom`, `alpharandom`, `velocityrandom`, `rotationrandom`, `angularvelocityrandom`, and `turbulentvelocityrandom`.
- Emitters: `boxrandom` and `sphererandom`.
- Operators: `movement`, `angularmovement`, `sizechange`, `alphafade`, `alphachange`, `colorchange`, `oscillatealpha`, `oscillatesize`, `oscillateposition`, `turbulence`, `vortex`, and `controlpointattract`.

### controlpointattract operator
The `controlpointattract` operator applies acceleration between particles and a selected control point.

- controlpoint (INT): Selects the control point index. Wallpaper Engine scenes commonly use indices from 0 to 7.
- scale (FLOAT): Controls force strength. Positive values attract particles toward the control point; negative values repel particles away from it.
- threshold (FLOAT): Maximum distance at which the force is applied.
- threadhold (FLOAT): Historical misspelling accepted by many authored Wallpaper Engine assets. The renderer reads both `threshold` and `threadhold`, with the later parsed field taking the final value.
- offset (ARRAY of FLOAT): Positional offset from the control point center.

Particles exactly at the force center are ignored by a tiny dead-zone so the normalized force direction remains finite.

## Models
3D models will be compiled from .obj or .fbx files into the custom .mdl format. Only the .mdl file is required to use a model in Wallpaper Engine.  
An optional JSON file can be used to configure how your model is compiled  
- normals (BOOL): Add "normals": true to your config file to compile normals into your model.
- tangentspace (BOOL): Add "tangentspace": true to your config file to add tangent space vectors to your model file (bitangent and tangent vectors).
- seconduvchannel (BOOL): Add "seconduvchannel": true to your config file to enable a second UV channel (typically used for lightmaps). This option is only valid for .fbx files. The second set of UVs must be named "lightmapSet". This allows you to use attribute vec4 a_TexCoordVec4 in the shader instead of attribute vec2 a_TexCoord.
- materialdirectory (STRING): Add "materialdirectory": "MYPATH" to your config to use a specific material base directory. This allows you to share materials between models more easily.
- skins (ARRAY): An array of objects with keys/strings that define multiple materials for a model. Use "skin" in scene.json to select a skin by index.
