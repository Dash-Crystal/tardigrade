parser.add_argument("--water-wave-iterations", type=int, default=0,
                    help="override g_nWaveIterations (the `%%16582` "
                         "UNIFORM-bounded wave loop). 0 = the material's "
                         "own value, which is what the shader reads")
parser.add_argument("--water-ssr-steps", type=int, default=0,
                    help="override g_nSSRMaxForwardSteps (the `%%23225` "
                         "loop). 0 = the material's own value")
parser.add_argument("--water-rain", type=float, default=1.0,
                    help="the per-view rain multiplier the shader reads "
                         "from set 1/binding 3 and multiplies by "
                         "g_flRainStrength (:499) to gate the 2-iteration "
                         "rain-ripple loop. 1.0 = the material's own "
                         "g_flRainStrength decides, which is the shipped "
                         "behaviour; 0 forces the loop off")
parser.add_argument("--water-effects-map", default=None,
                    help="image for g_tWaterEffectsMap (set 1/binding 64), "
                         "the per-view target S_INTERACTION_EFFECTS reads. "
                         "It is an ENGINE target -- no .vmat binds it and "
                         "no .vcs carries its content -- so without this "
                         "the renderer builds a procedural stand-in with "
                         "the layout the shader's own decode fixes. It is "
                         "never absent, because an absent target would "
                         "silently make the whole axis a no-op.")
parser.add_argument("--water-selftest", action="store_true",
                    help="run water_fancy_shade once per value of every "
                         "combo axis the family declares and report, per "
                         "axis value, the pixels it reaches and the "
                         "max|delta| against the material's own value")
parser.add_argument("--unlit", action="store_true",
                    help="csgo_black_unlit (2) + csgo_lightmappedgeneric's "
                         "`black` (1): emit the albedo unlit rather than "
                         "lighting a white sentinel texture")
parser.add_argument("--complex-tint", action="store_true",
                    help="csgo_complex/csgo_static_overlay g_vColorTint "
                         "applied WITHOUT the F_TINT_MASK gate (2 "
                         "static_overlay materials carry a non-white tint "
                         "and no mask, so it is dropped entirely today)")
# ==============================================================# THE VERTEX STAGE.  Every flag below drives a transcription in
# cs2_vertex_stage.py of a named region of a committed csgo_*_vs listing;
# the reference lines live in that module's docstrings, not here.
#
# UNITS: these are per-VERTEX axes.  de_inferno is ~8.6e6 vertices against
# 2.07e6 raster pixels (vertex:pixel = 4.15:1,
# SHADER_CALLFLOW_vertex_stages.md:14-17).  Nothing below converts between
# the two and no per-pixel figure is quoted against a per-vertex one.
#
# REACHABILITY: every parameter default here is non-inert.  Turning an axis
# on with no further arguments moves geometry -- `--vertex-stage-selftest`
# prints max|delta| per axis and exits non-zero on a zero, because a
# default that quietly disables the path it advertises is worse than the
# path being absent.
parser.add_argument("--vertex-stage-selftest", action="store_true",
                    help="run every vertex axis at its DEFAULT parameters "
                         "on the loaded geometry, print max|delta| for "
                         "each, and exit. Non-zero exit if any axis fails "
                         "to move anything -- the reachability check")
parser.add_argument("--compressed-normals", choices=("off", "on"),
                    default="off",
                    help="D_COMPRESSED_NORMALS_AND_TANGENTS. `on` routes "
                         "the pack's float normal+tangent through the u32 "
                         "CompressedTangentFrame encode/decode the shader "
                         "uses (10-bit octahedral normal + 11-bit Rodrigues "
                         "tangent angle + sign), so the quantisation the "
                         "engine actually ships is present rather than "
                         "assumed away. `off` is the shader's own "
                         "D_COMPRESSED=0 branch (separate vNormalOs + "
                         "vTangentUOs_flTangentVSign), which is what our "
                         "glTF carries -- both are real shader variants, "
                         "neither is a fallback")
parser.add_argument("--secondary-uv", choices=("off", "material", "0",
                                               "1", "2"),
                    default="off",
                    help="S_SECONDARY_UV, VERTEX-ATTRIBUTE SIDE. The "
                         "per-layer g_nUVSet int: 0 = pass through (the "
                         "PS takes its biplanar branch, owned elsewhere), "
                         "1 = UV set 0, 2 = UV set 1. `material` reads the "
                         "pack's face_secondary flag; a digit forces it "
                         "globally. This is the second vec2 that "
                         "csgo_environment_blend_vs takes and "
                         "csgo_complex_vs does not. "
                         "DEFAULT IS `off` FOR A MEASURED REASON, not "
                         "timidity: our `uvs2` comes from glTF TEXCOORD_2, "
                         "whose density is 0.0028 repeats/m against the "
                         "tiling stream's 0.485 (UV_CONSTRUCTION.md:90-97). "
                         "THAT CITATION IS glTF-ERA AND SUPERSEDED: it "
                         "measured a pipeline we replaced (it says uvs2 comes "
                         "from glTF TEXCOORD_2, and the ASH source declares "
                         "no TEXCOORD_2 at all -- 0, 1, 3, 4 only; ASH maps "
                         "TEXCOORD_1 -> uvs2). Re-measured 2026-08-11 on the "
                         "current pack, SCOPED to the materials that actually "
                         "select a UV set, because the whole-map 0.0163 hides "
                         "p10 0.0000 / p90 0.9251: uvs2 is EXACTLY ZERO on "
                         "53.97%% of the selecting surface and a 100%% "
                         "duplicate of uvs where nothing selects it. So it is "
                         "not a unique unwrap -- it is ABSENT on half the "
                         "surface that uses it, which is a strictly worse "
                         "reason to keep this off than the recorded one. "
                         "Cause not yet typed. `material` prints the density "
                         "so the mismatch is visible, not assumed")
parser.add_argument("--foliage-animation", action="store_true",
                    help="S_FOLIAGE_ANIMATION: the THREE-tap pivot "
                         "hierarchy (trunk rotation about a wind-derived "
                         "base, branch rotation about vPivotPaint, detail "
                         "displacement) plus the 4-band flutter "
                         "interpolant. Needs a vertex-rate 3D fetch")
parser.add_argument("--foliage-uv-animation", action="store_true",
                    help="S_FOLIAGE_UV_ANIMATION: uv += scroll * time")
parser.add_argument("--foliage-uv-scroll", type=float, nargs=2,
                    default=(0.02, 0.0), metavar=("DU", "DV"),
                    help="UV units per second. Non-zero by default so the "
                         "axis is not silently inert when enabled")
parser.add_argument("--vertex-animation", type=int, default=0,
                    choices=(0, 1, 2),
                    help="S_VERTEX_ANIMATION (declared range 0..2): the "
                         "NINE-tap world-noise displacement with the "
                         "tangent frame rebuilt by finite differences at "
                         "P, P+T, P+B. A DIFFERENT animation from "
                         "--foliage-animation, not a setting of it. What "
                         "value 2 changes over 1 was not isolated in the "
                         "corpus and is not guessed: 1 and 2 both run the "
                         "full nine-tap form")
parser.add_argument("--cs-vertex-animation", action="store_true",
                    help="D_CS_VERTEX_ANIMATION. The per-instance transform "
                         "re-index branch IS transcribed; the animated "
                         "transform buffer's CONTENTS are ours, because no "
                         "committed listing selects this axis. See the "
                         "docstring in cs2_vertex_stage.py")
parser.add_argument("--cs-vertex-animation-rate", type=float, default=0.6,
                    help="rad/s of the per-instance transform animation")
parser.add_argument("--cs-vertex-animation-amp", type=float, default=0.05,
                    help="metres of per-instance translation amplitude")
parser.add_argument("--prebaked-vertex-animation", action="store_true",
                    help="S_PRE_BAKED_VERTEX_ANIMATION (csgo_complex). "
                         "Requires --prebaked-anim-npz; it REFUSES to run "
                         "without a table rather than silently no-opping")
parser.add_argument("--prebaked-anim-npz", default=None,
                    help="npz with `dpos` (F,N,3) and optionally `dnrm` "
                         "(F,N,3), or glTF morph targets carried by the "
                         "pack as `morph_dpos`")
parser.add_argument("--prebaked-anim-rate", type=float, default=8.0,
                    help="keyframes per second")
parser.add_argument("--vertex-break-animation", action="store_true",
                    help="S_VERTEX_BREAK_ANIMATION (csgo_glass, live on "
                         "de_inferno at combo s4). The glass VERTEX stage "
                         "was never decompiled; this is reconstructed from "
                         "the four uniform NAMES the glass document quotes "
                         "and every added term is named as an addition")
parser.add_argument("--break-time", type=float, default=0.35,
                    help="g_flBreakTime, seconds since the break. Non-zero "
                         "by default: at 0 the pane is unbroken and the "
                         "axis is correctly inert, which would read as a "
                         "disabled path")
parser.add_argument("--break-shot-pos", type=float, nargs=3,
                    default=None, metavar=("X", "Y", "Z"),
                    help="g_vBreakShotPosition in world metres; default is "
                         "the centroid of the glass-family geometry, so "
                         "the axis affects something wherever the map is")
parser.add_argument("--break-shot-size", type=float, default=1.5,
                    help="g_flBreakShotSize, metres")
parser.add_argument("--break-bounce-floor", type=float, default=None,
                    help="g_flBreakBounceFloor, world Y in metres; default "
                         "is the glass geometry's own minimum Y")
parser.add_argument("--vertex-noise-volume", default=None,
                    help="npy (D,H,W,3) for the vertex-rate sampler3D. The "
                         "CS2 asset is chosen by a RUNTIME descriptor index "
                         "and is not nameable from the bytecode, so the "
                         "default is a deterministic tiling value-noise "
                         "volume; the fetch path, coordinates, LOD-0 "
                         "semantics, wrap and both decodes are the "
                         "shader's. THIS IS A DISPLACEMENT FIELD, not a "
                         "lighting volume -- it must never be wired into "
                         "the probe atlas")
parser.add_argument("--vertex-noise-size", type=int, default=32,
                    help="edge of the generated volume")
parser.add_argument("--vertex-noise-seed", type=int, default=1)
parser.add_argument("--wind-dir", type=float, nargs=3,
                    default=(0.80, 0.0, 0.60), metavar=("X", "Y", "Z"),
                    help="world-space wind vector for every animation axis")
parser.add_argument("--wind-strength", type=float, default=1.0)
parser.add_argument("--wind-gust", type=float, default=0.35,
                    help="the per-view gust scalar the shader reads from "
                         "PerViewConstantBuffer offset 288")
parser.add_argument("--vertex-anim-fps", type=float, default=None,
                    help="seconds per rendered frame for every time-driven "
                         "vertex axis; default is 1/--fps, i.e. the "
                         "animation runs at the video's own rate")
# --- render / tools modes ---------------------------------------------
parser.add_argument("--tools-vis", default="off",
                    choices=("off", "normal", "tangent", "uv0", "uv1",
                             "uvset", "vertexcolor", "pervertexlighting",
                             "pivot", "foliageparams", "bend",
                             "displacement", "packedframe-error"),
                    help="S_MODE_TOOLS_VIS. This is a declared static combo "
                         "axis of all four vertex families "
                         "(SHADER_CALLFLOW_vertex_stages.md:204-207) but no "
                         "module selecting it was decompiled in this "
                         "corpus, so the SELECTOR is reproduced and the "
                         "channels are built from the quantities the vertex "
                         "stage genuinely produces. `displacement` and "
                         "`bend` show what the animation axes did; "
                         "`packedframe-error` shows the "
                         "D_COMPRESSED_NORMALS_AND_TANGENTS quantisation")
parser.add_argument("--tools-shading-complexity", action="store_true",
                    help="S_MODE_TOOLS_SHADING_COMPLEXITY: heat map of "
                         "per-pixel shader cost, from the measured FLOP "
                         "tables. Per-PIXEL and per-VERTEX are rendered on "
                         "SEPARATE scales and the mode prints which; the "
                         "extrapolated rows carry their own caveat")
parser.add_argument("--tools-complexity-unit",
                    choices=("pixel", "vertex"), default="pixel",
                    help="which cost unit the complexity heat map shows. "
                         "They are different units and are never mixed")
parser.add_argument("--tools-complexity-max", type=float, default=None,
                    help="FLOPs at the top of the ramp; default is the "
                         "maximum of the table actually in use")
parser.add_argument("--quad-overdraw", action="store_true",
                    help="D_QUAD_OVERDRAW: per-2x2-QUAD fragment count as a "
                         "heat map. In CS2 this is 3 OpImageTexelPointer "
                         "feeding 3 image atomics -- the only 6 unclassified "
                         "instructions in the whole 3.0e6-instruction corpus "
                         "(SHADER_CALLFLOW_METHOD.md:78-88). Counted here "
                         "across all three alpha classes")
parser.add_argument("--quad-overdraw-max", type=float, default=8.0,
                    help="fragments per quad at the top of the ramp")
parser.add_argument("--quad-overdraw-layers", type=int, default=8,
                    help="depth-peeled layers the counter walks. A pixel "
                         "with more fragments than this is UNDERCOUNTED, so "
                         "the image is a floor; the saturating fraction is "
                         "printed so the floor is quantified rather than "
                         "assumed small")
# --- GROUND-TRUTH MATERIAL-SURFACE COMBO AXES --------------------------
# One flag per Source 2 static-combo axis, transcribed from the shaders
# under docs/projects/counter-strike-sft/. Each axis is resolved through
# its feature-array position (SHADER_CALLFLOW_csgo_complex.md §1,
# SHADER_CALLFLOW_csgo_static_overlay.md, FOG_AND_COMBO_MAPPING.md §3),
# never by string-matching an F_ name -- Valve's own files carry
# `F_DIFFSUE_WRAP`, a typo matching nothing, and several F_ names present
# in one shader's materials are absent from another's feature array.
#
# All of these need the fast_pack_fam.py side table (--fam-side); the
# renderer refuses to start if the table predates the column it needs
# rather than silently reading a default.
parser.add_argument("--detail-texture", action="store_true",
                    help="S_DETAIL_TEXTURE (csgo_complex, m_iFeatureIndex "
                         "11). RANGE 0..4, NOT a boolean: the combo VALUE "
                         "selects the blend mode. g_tDetail on its own "
                         "g_vDetailTexCoordScale/Offset, composited by "
                         "g_flDetailBlendFactor. See --detail-mode.")
parser.add_argument("--detail-mode", type=int, default=-1,
                    help="override the per-material S_DETAIL_TEXTURE value "
                         "(0=off 1=mod2x 2=multiply 3=overlay 4=additive). "
                         "-1 = read F_DETAIL_TEXTURE off each material, "
                         "which is the ground-truth path and the default.")
parser.add_argument("--detail-normal", action="store_true",
                    help="S_DETAIL_NORMAL (csgo_environment bit 1, "
                         "csgo_environment_blend bit 1). Per-layer "
                         "g_tNormalDetail1/2 composited onto the base "
                         "normal by g_flDetailNormalStrength.")
parser.add_argument("--tint-mask-src",
                    choices=("auto", "tintmask-tex", "height-g"),
                    default="auto",
                    help="S_TINT_MASK mask source. csgo_environment_blend_"
                         "ps.glsl:593 reads the mask from the HEIGHT "
                         "texture's GREEN channel; csgo_complex has a real "
                         "g_tTintMask. auto = g_tTintMask where the "
                         "material binds one, height.g otherwise.")
parser.add_argument("--metalness-channel", type=int, default=1,
                    help="which g_tMetalness channel S_METALNESS_TEXTURE "
                         "reads (VRF packs de_inferno's in G).")
parser.add_argument("--self-illum-channel", type=int, default=0,
                    help="which g_tSelfIllumMask channel S_SELF_ILLUM "
                         "reads.")
# NOTE: S_SECONDARY_UV is ONE axis and therefore ONE flag. The pixel
# side and the vertex-attribute side were implemented independently
# and both declared "--secondary-uv"; argparse raises on the second,
# so merged main could not build its parser at all. The richer
# definition above (choices off/material/0/1/2) is kept and BOTH
# sides read it. Consumers must test != "off" -- the string "off" is
# truthy, so a bare boolean test is permanently ON.
parser.add_argument("--decal-texture", action="store_true",
                    help="S_DECAL_TEXTURE (csgo_complex m_iFeatureIndex "
                         "21). g_tDecal over the albedo on its own UV, by "
                         "decal alpha * g_flDecalBlendFactor.")
parser.add_argument("--aniso-gloss", action="store_true",
                    help="S_ANISOTROPIC_GLOSS (csgo_complex "
                         "m_iFeatureIndex 9). Splits roughness into "
                         "tangent/bitangent halves so the specular lobe "
                         "stretches along the brushed direction.")
parser.add_argument("--layer3", action="store_true",
                    help="S_ENABLE_LAYER_3 (csgo_environment_blend, bit "
                         "weight 4). The THIRD blend layer: g_tColor3 / "
                         "g_tNormal3 / g_tHeight3 over the 1->2 result, on "
                         "the weight csgo_environment_blend_ps.glsl:260-262 "
                         "builds from COLOR_0.y.")
parser.add_argument("--texture-animation", action="store_true",
                    help="S_TEXTURE_ANIMATION (csgo_complex vs). Exact "
                         "port of csgo_complex_vs_max.glsl:256-282: "
                         "sprite-sheet cell selection by one of three "
                         "g_nAnimationMethod modes, plus the unconditional "
                         "g_vTexCoordScrollSpeed * time scroll.")
parser.add_argument("--overlay-mode", type=int, default=-1,
                    help="override g_nColorOverlayMode for "
                         "S_SHARED_COLOR_OVERLAY (0=multiply 1=overlay "
                         "2=mod2x 3=replace). -1 = per material, the "
                         "ground-truth path and the default.")
parser.add_argument("--blend-effects", action="store_true",
                    help="S_BLEND_EFFECTS -- csgo_static_overlay ONLY "
                         "(m_iFeatureIndex 11). NOT the environment_blend "
                         "seams: RENDER_SETTINGS_AXES.md line 144 declares "
                         "it in so:ps alone. Constrained: the shader "
                         "REQUIRES S_MATERIAL_REFERENCE != 0, so it is "
                         "gated on that per material, not applied blindly.")
parser.add_argument("--blend-effects-2", action="store_true",
                    help="S_BLEND_EFFECTS_2 (csgo_environment_blend "
                         "m_iFeatureIndex 8) -- the 1->2 seam, set on 34 of "
                         "43 de_inferno env-blend materials, the LIVE one. "
                         "Turns on the whole block: border tint, "
                         "F_BORDER_ROUGHNESS_2 and the bevel. "
                         "--blend-border/--bevel remain as the individual "
                         "terms so they can still be scored apart.")
parser.add_argument("--blend-effects-3", action="store_true",
                    help="S_BLEND_EFFECTS_3 (csgo_environment_blend "
                         "m_iFeatureIndex 11) -- the 2->3 seam. The shader "
                         "REQUIRES S_ENABLE_LAYER_3 (RENDER_SETTINGS_AXES"
                         ".md line 327), so this implies --layer3 and says "
                         "so rather than silently doing nothing.")
parser.add_argument("--use-new-blending", action="store_true",
                    help="S_USE_NEW_BLENDING (csgo_environment_blend "
                         "m_iFeatureIndex 16). Selects Source 2's newer "
                         "height-blend transition; 0 on all 43 de_inferno "
                         "materials, which is a usage fact, not a "
                         "capability fact.")
parser.add_argument("--enable-visualizations", action="store_true",
                    help="S_ENABLE_VISUALIZATIONS (csgo_environment_blend "
                         "m_iFeatureIndex 21) + F_VISUALIZATION_MODE: the "
                         "authoring visualisation outputs (blend weight / "
                         "height / border band / layer id) in place of the "
                         "shaded colour. 0 on all 43 de_inferno materials.")
parser.add_argument("--alpha-test-gt", action="store_true",
                    help="S_ALPHA_TEST: use the material's own "
                         "F_ALPHA_TEST + g_flAlphaTestReference instead of "
                         "VRF's blanket alphaCutoff=0.5, on EVERY family "
                         "that declares the axis (--alpha-ref only "
                         "rewrites csgo_environment's).")
parser.add_argument("--alpha-test-layer", action="store_true",
                    help="S_ALPHA_TEST_LAYER (csgo_environment bit 4): the "
                         "alpha test runs on the BLENDED layer alpha with "
                         "a per-layer reference, not on layer 1's.")
parser.add_argument("--material-reference", action="store_true",
                    help="S_MATERIAL_REFERENCE (csgo_environment "
                         "m_iFeatureIndex 7, bit weight 2; 51/151 "
                         "materials). A second colour transform of the "
                         "albedo, selected per pixel.")
parser.add_argument("--gt-axes-reach", action="store_true",
                    help="DIAGNOSTIC: fraction of shaded pixels each of "
                         "the material-surface combo axes actually "
                         "reaches, so 'no visible change' can be told "
                         "apart from 'no pixels' or 'disabled by a "
                         "default'.")
# --- baked environment cubemap probes (cube_extract_fam.py) ------------
# maps/de_inferno/cubemaps/env_cubemap_array.vtex_c is a 256^2 BC6H cube
# ARRAY of 116 probes with 7 prefiltered mips -- that mip chain is the
# output of convolve_environment_map_vulkan_50_ps. The same vtex carries
# VTEX_EXTRA_DATA_CUBEMAP_RADIANCE_SH, 116 x 3 x 9 SH-L2 coefficients,
# which is the DIFFUSE half already convolved and needs no BC6H decode.
parser.add_argument("--cube-probes", default=None,
                    help="cube_extract_fam.py probe table (cubeprobes.pt)")
parser.add_argument("--ibl-sh", action="store_true",
                    help="ambient from the per-probe SH-L2 radiance instead "
                         "of the fitted 2-colour hemisphere. This is the "
                         "L1/L2 lobe the lightmap DC term does not carry.")
parser.add_argument("--ibl-sh-mode",
                    choices=("dir", "dirlum", "modulate", "replace"),
                    default="dir",
                    help="dir: the DIRECTIONAL lobe only -- the probe's own "
                         "irradiance divided by its own DC term, so its mean "
                         "over directions is 1 by construction and it "
                         "composes with the lightmap indirect instead of "
                         "overwriting it. This is the only part the lightmap "
                         "does not already carry. modulate: spatial ratio too "
                         "(REPLACES the lightmap term -- an A/B of which "
                         "spatial source is better, not an addition). "
                         "replace: probe irradiance outright.")
parser.add_argument("--ibl-sh-gain", type=float, default=1.0)
parser.add_argument("--ibl-normal", choices=("shaded", "geometric"),
                    default="shaded",
                    help="which normal the SH ambient is evaluated on. A "
                         "probe is a LOW-frequency lighting term, so "
                         "evaluating it on the normal-mapped normal makes "
                         "the ambient carry the bump map's spatial "
                         "frequency -- measurable in hf_energy.py.")
parser.add_argument("--ibl-nearest", action="store_true",
                    help="DIAGNOSTIC: pick one probe per voxel instead of "
                         "trilinearly interpolating the SH coefficients. "
                         "Nearest-probe puts a hard seam on every probe "
                         "boundary, which shows up as +2.6 points of "
                         "high-frequency energy in hf_energy.py.")
parser.add_argument("--ibl-spec", action="store_true",
                    help="SPECULAR IBL from the same environment probes. "
                         "The renderer runs GGX on the SUN ONLY ('the "
                         "ambient term has no direction to reflect') -- with "
                         "probes it does. Uses the SH-L2 radiance evaluated "
                         "along the reflection vector, which is a genuinely "
                         "prefiltered radiance for ROUGH surfaces and "
                         "over-blurred for smooth ones; the sharp end needs "
                         "the BC6H mip chain (see the report).")
parser.add_argument("--ibl-spec-gain", type=float, default=1.0)
parser.add_argument("--ibl-brdf", choices=("lazarov", "simple"),
                    default="lazarov",
                    help="lazarov: the split-sum BRDF the real "
                         "csgo_environment_blend PS uses, constants verbatim "
                         "from the decompiled shader (there is no DFG LUT "
                         "texture). simple: my first pass -- Schlick times "
                         "(1-roughness), kept so the BRDF change can be "
                         "scored on its own.")
parser.add_argument("--ibl-dominant-dir", type=int, default=1,
                    help="1 = Frostbite dominant specular direction, which "
                         "the real shader uses: it lerps the sample "
                         "direction from the mirror vector toward N as "
                         "roughness rises. 0 = pure mirror reflection.")
parser.add_argument("--ibl-cube", default=None,
                    help="directory of the HDR BC6H decode "
                         "(env_cubemap_mip{0..6}_f16.npy): the REAL "
                         "prefiltered radiance chain, the output of "
                         "convolve_environment_map, instead of SH-L2 "
                         "standing in for it. lod = scale*sqrt(roughness) "
                         "only means something once this chain exists.")
parser.add_argument("--ibl-cube-mips", type=int, default=7,
                    help="how many mips to keep resident. mip0 alone is "
                         "274 MB; rough surfaces only need the small ones.")
parser.add_argument("--ibl-lod-scale", type=float, default=None,
                    help="g_flEnvMapLodScale in lod = scale*sqrt(roughness). "
                         "DEFAULT IS DERIVED FROM THE ASSET at load: "
                         "nmips-1, so sqrt(roughness) in [0,1] spans the "
                         "whole chain. A fixed default cannot do this -- a "
                         "previous 1.0 made mips 2..6 UNREACHABLE BY "
                         "CONSTRUCTION, so the prefilter was never sampled "
                         "and every measurement through it described a "
                         "chain with its filtering switched off.")
parser.add_argument("--ibl-cube-norm", type=int, default=1,
                    help="g_bCubemapNormalization: chroma-preserving "
                         "magnitude clamp at 1.75, from the shader.")
parser.add_argument("--ibl-spec-flat", action="store_true",
                    help="CONTROL: keep the ambient-specular term but throw "
                         "the probe DIRECTIONALITY away (ratio forced to 1). "
                         "If this scores the same as --ibl-spec then the "
                         "gain is just a missing isotropic specular term "
                         "acting as a global brightener and the cubemaps "
                         "are doing no work.")
parser.add_argument("--ibl-grid", type=float, default=1.0,
                    help="metres; edge of the voxel grid that maps a world "
                         "position to its probe")
_f3 = lambda s: [float(x) for x in s.split(",")]   # noqa: E731
# --- CS2 post-process chain -------------------------------------------
# Every constant below is READ OUT of the shipped map, not assumed:
#   de_inferno.vpk : maps/de_inferno/entities/default_ents.vents_c holds
#   exactly one post_processing_volume ("[PR#]postprocess",
#   hammerUniqueId 13885, model .../postprocess_13885.vmdl, master=true,
#   StartDisabled=false) with
#       enableexposure    true
#       minexposure       0.8
#       maxexposure       1.1
#       exposurespeedup   1.0    exposurespeeddown 1.0   fadetime 1.5
#       postprocessing    lighting/postprocessing/de_inferno_prefab/
#                         de_inferno_prefab.vpost
#   and that .vpost (pak01_dir.vpk) carries
#       m_bHasTonemapParams true, Hable/Uncharted2 filmic:
#         ShoulderStrength .15  LinearStrength .5  LinearAngle .1
#         ToeStrength .2  ToeNum .02  ToeDenom .3  WhitePoint 4.0
#         ExposureBias 0.0 (and 0.0 for both the shadow/highlight biases)
#       m_bHasBloomParams true: BLOOM_BLEND_ADD, strength .12,
#         threshold .599, thresholdWidth .906, skyboxStrength .1557,
#         5 blur levels weighted .2 each, all tints white
#       m_bHasColorCorrection true: 32^3 RGBA8 volume LUT
#       vignette / local-contrast / fog-scattering: all disabled
parser.add_argument("--tonemap", choices=("off", "filmic"), default="off",
                    help="'filmic' = the map's own Hable curve; the lighting "
                         "coefficients must be refit UNDER it (fit_tone.py)")
parser.add_argument("--white-point", type=float, default=4.0,
                    help="m_flWhitePoint from the vpost")
parser.add_argument("--pre-exposure", type=str, default="1.0",
                    help="linear gain applied BEFORE the tonemap. Our "
                         "lighting terms are in arbitrary units; this is "
                         "what puts them on the curve's scale. 1 or 3 "
                         "comma-separated values (fit_tone.py emits it)")
parser.add_argument("--post-cc", default=None,
                    help="path to the vpost's 32^3 colour-correction volume "
                         "(cc_lut.npy). Applied in sRGB space, trilinear, "
                         "as the engine does")
parser.add_argument("--bloom", action="store_true",
                    help="the vpost's additive 5-level bloom pyramid")
parser.add_argument("--bloom-strength", type=float, default=0.12)
parser.add_argument("--bloom-threshold", type=float, default=0.599)
parser.add_argument("--bloom-threshold-width", type=float, default=0.906)
parser.add_argument("--auto-exposure", action="store_true",
                    help="per-frame exposure = clamp(K / meanlum, "
                         "minexposure, maxexposure) with the entity's own "
                         "0.8/1.1 bounds; K is one global constant")
parser.add_argument("--ae-key", type=float, default=0.5,
                    help="the K above")
parser.add_argument("--min-exposure", type=float, default=0.8)
parser.add_argument("--max-exposure", type=float, default=1.1)
parser.add_argument("--post-order", choices=("legacy", "engine"),
                    default="legacy",
                    help="'engine' = the order decompiled from "
                         "post_process_vulkan_50_ps")
parser.add_argument("--srgb-encode", choices=("pow22", "true"),
                    default="pow22",
                    help="'true' = the shader's sRGB curve with its linear "
                         "toe (exponent 1/2.4), not pow(1/2.2)")
parser.add_argument("--bloom-curve", choices=("hable", "reinhard"),
                    default="hable",
                    help="D_BLOOM_MODE. 'hable' (mode 1) runs the bloom "
                         "through the same Hable curve + sRGB encode as the "
                         "scene. 'reinhard' (mode 0) is what the combo "
                         "matching OUR feature set ships (combo 36); its "
                         "mode-1 twin is PRUNED and does not exist for this "
                         "configuration, so mode 0 is the engine's answer")
parser.add_argument("--bloom-mode0-strength", type=float, default=1.0,
                    help="g_vNormalizedBloomStrengths.y; engine-managed, "
                         "not in the .vpost")
parser.add_argument("--bloom-blend", choices=("screen", "add"),
                    default="screen",
                    help="the shader screens; the .vpost says "
                         "BLOOM_BLEND_ADD, so this is likely combo-selected")
# --lut-texel DELETED 2026-08-08. It offered "edge" (c*(N-1), the plain
# spatial-sampler convention) against "halftexel", and DEFAULTED to "edge"
# -- the branch its own help text said was not the shader's.
# cs2_post_process_ps.glsl:124 fetches at
#     clamp(c,0,1) * 0.96875 + 0.015625
# and 0.96875 = 31/32, 0.015625 = 0.5/32, i.e. (c*(N-1) + 0.5)/N at N=32.
# The half-texel inset is the only convention the reference can take, so
# "edge" was not a config value: a config axis selects between values the
# REFERENCE can take, and an arm the reference cannot take is a fuzzer
# input wearing a flag. Per the standing rule, the validated replacement
# means the old arm is deleted rather than kept as a gate. _apply_cc now
# does the reference fetch unconditionally.
parser.add_argument("--cc-amount", type=float, default=1.0,
                    help="the shader's mix(lut, c, m2) blend")
parser.add_argument("--post-decode", action="store_true",
                    help="shader steps 8+11: sRGB-decode after the LUT then "
                         "the adjustable final gamma")
parser.add_argument("--final-gamma", type=float, default=1.0,
                    help="the shader's m5 in pow(c, m5/2.2)")
parser.add_argument("--tm-prescale", type=float, default=1.0,
                    help="the shader's hardcoded x2.8 before the Hable curve")
parser.add_argument("--tm-clamp", type=float, default=1e9,
                    help="the shader's min(rgb*2.8, m7) clamp")
parser.add_argument("--lod-per-frame", action="store_true",
                    help="select LOD from EACH FRAME's own eye instead "
                         "of the chunk-mean eye. Without this the LOD a "
                         "cluster receives depends on how frames happen "
                         "to be grouped, so --batch silently changes the "
                         "render: batch 16 vs 48 differ on 21.181%% of "
                         "pixels (identical only under --no-lod). "
                         "Implemented by reducing the rasterization "
                         "chunk to one frame, so per-frame selection is "
                         "true by construction rather than by new "
                         "distance code; it costs the cross-frame "
                         "instancing, so it is off by default.")
parser.add_argument("--ext2-zero", default=None,
                    help="DIAGNOSTIC: comma-separated ext2 column names to "
                         "force to 0 for every material, so a single "
                         "column's contribution can be A/B'd without "
                         "unbinding the whole block. Refuses on an unknown "
                         "name and on a column that is already all-zero, "
                         "because both would make the arms identical and "
                         "report 'no effect'.")
parser.add_argument("--matid-dump", default=None,
                    help="write the FINAL-OWNER material id per pixel, "
                         "one uint16 .npy per frame, tracked across the "
                         "opaque/mask/blend classes so it follows the "
                         "same compositing the frames do. NOTE: matid "
                         "depends on geometry and LOD, and LOD is chosen "
                         "from the CHUNK-MEAN eye position -- so a dump "
                         "is only valid for the frames of the SAME run "
                         "at the SAME --batch (batch 16 vs 48 differ on "
                         "21.181%% of pixels; identical only under "
                         "--no-lod).")
parser.add_argument("--hdr-dump", default=None,
                    help="write pre-tonemap linear frames as .npy for the "
                         "under-the-curve refit")
# The lighting constants below were fitted (fit_lighting2.py) against a
# G-buffer of FLAT FACE normals. Changing the normal field changes the
# ndl/up distribution the fit solved for, so any normal-mapping A/B that
# keeps them frozen is measuring a stale fit, not the feature. These make
# the refit injectable instead of a source edit.
_f3 = lambda s: [float(x) for x in s.split(",")]
# --- the PRESET post axes: hdr_detail, fsr_detail, cmaa ---------------
# Three cvars every GT capture carries and this renderer had no axis for.
# Each selects a VALUE inside a pass that always runs in the reference;
# none of them is a feature switch.
parser.add_argument("--hdr-scene-format", default="off",
                    choices=("off",
                             "VK_FORMAT_R16G16B16A16_SFLOAT",
                             "VK_FORMAT_B10G11R11_UFLOAT_PACK32"),
                    help="videocfg_hdr_detail: the format of the HDR SCENE "
                         "COLOUR attachment the scene pass writes and the "
                         "post chain reads. The tier -> format map is READ "
                         "from the engine's own RT enumeration across all "
                         "30 GT arms with no exceptions: -1 -> RGBA16F, "
                         "3 -> R11G11B10 (post.video_config."
                         "HDR_FORMAT_BY_TIER). WHAT THE AXIS DOES IS "
                         "MANTISSA WIDTH: 10 bits against 6/6/5, a 16x "
                         "coarser quantisation on every scene pixel. The "
                         "range clamp the two formats differ by is inert "
                         "on this map -- nothing reaches 64512 -- so a "
                         "renderer implementing only the clamp emits "
                         "BIT-IDENTICAL images at both tiers. 'off' "
                         "emulates no attachment store at all and is this "
                         "renderer's historical behaviour; --preset always "
                         "sets one of the two real formats.")
parser.add_argument("--fsr-detail", type=int, default=0,
                    choices=(0, 1, 2, 3, 4),
                    help="videocfg_fsr_detail. 0 is native. A non-zero "
                         "tier SHRINKS THE SCENE RASTER by that tier's "
                         "render scale and presents through EASU (+RCAS); "
                         "it is not a post-process, because every LOD and "
                         "every derivative downstream is then taken at the "
                         "smaller size. preset0 and preset1 carry tier 3, "
                         "preset2 and preset3 carry 0, so this is one of "
                         "the largest single deltas in the preset ladder. "
                         "Tiers whose render scale has not been READ out "
                         "of CS2 REFUSE rather than assume AMD's published "
                         "quality-mode ratios -- see "
                         "post.video_config.FSR_TIER_SCALE.")
parser.add_argument("--fsr-sharpness", type=float, default=0.25,
                    help="RCAS sharpness; con.x = exp2(-sharpness) "
                         "(cs2_fsr_rcas.glsl:39). BOTH HALVES ARE NOW "
                         "READ. WHETHER RCAS runs is per-tier "
                         "(post.video_config.FSR_TIER_RCAS: tiers 2-4 yes, "
                         "tier 1 no -- Ultra Quality loses energy in every "
                         "band and allocates no second full-res LDR "
                         "target). WHAT IT RUNS WITH is 0.25f, READ out of "
                         "ws-1's libclient.so: at the single PC-relative "
                         "site that resolves to the r_csgo_fsr_rcas_"
                         "sharpness string, the movss loading xmm0 "
                         "immediately before the ConVar name lea targets "
                         "va 0xafbf04, whose bytes are 00 00 80 3e. "
                         "con.x = exp2(-0.25) = 0.8408964152537145. The "
                         "reader is calibrated two-sided: on the BOOLEAN "
                         "r_csgo_fsr_enable_mip_bias the same walk finds "
                         "no movss and an integer immediate instead, so it "
                         "declines rather than inventing a float. This is "
                         "no longer a fitted value and no longer defaults "
                         "to skipping RCAS.")
# THE LITERAL AND THE READ CONSTANT MUST AGREE, AND A MISMATCH REFUSES.
# argparse carries the LITERAL 0.25 rather than a reference to
# post.video_config.FSR_RCAS_SHARPNESS on purpose: renderer_defaults()
# reads defaults STATICALLY out of this file's source, and a computed
# expression resolves to the marker '<computed>', which would silently
# drop this flag out of the #55 identity pin -- two runs at different
# sharpnesses would then compare as EQUAL. The literal keeps the pin; this
# check keeps the literal honest.
#
# IT SITS HERE, BEFORE parse_args, AND USES THE DEST NOT THE FLAG. The
# first version of this guard did neither and was SILENT under
# perturbation: `get_default("--fsr-sharpness")` returns None because
# argparse keys defaults by DEST, and a check placed after parse_args is
# never reached by `--help`, which exits inside it. Calibrated two-sided
# now -- it fires when the constant is moved to 0.30 and is clean at 0.25.
if parser.get_default("fsr_sharpness") != _po_vcfg.FSR_RCAS_SHARPNESS:
    raise SystemExit(
        f"REFUSED: --fsr-sharpness defaults to "
        f"{parser.get_default('fsr_sharpness')} but the value READ from "
        f"libclient.so is {_po_vcfg.FSR_RCAS_SHARPNESS} "
        f"(post.video_config.FSR_RCAS_SHARPNESS). The literal exists so "
        f"the flag stays in the #55 identity pin; if the read is revised, "
        f"the literal has to move with it.")

parser.add_argument("--texture-filtering-quality", type=int, default=None,
                    choices=(0, 1, 2, 3, 4, 5),
                    help="r_texturefilteringquality. READ from the "
                         "engine's own UI resource (game/csgo/pak01_dir.vpk "
                         "-> panorama/layout/settings/settings_video.vxml_c, "
                         "element ids matforceaniso0..5 against Bilinear / "
                         "Trilinear / Anisotropic 2X/4X/8X/16X): aniso "
                         "1,1,2,4,8,16 with the MIP FILTER changing "
                         "point->linear at 0->1, and no per-tier LOD bias. "
                         "Sets --max-aniso / --max-aniso-aux and the mip "
                         "filter together, because the tier is ONE state "
                         "and setting the aniso count alone would put "
                         "tier 1 (a mip-filter change) on the wrong axis. "
                         "None leaves this renderer's own --max-aniso "
                         "values alone, which is NO CS2 tier; --preset "
                         "always sets it.")
parser.add_argument("--texture-detail", type=int, default=None,
                    choices=(0, 1, 2),
                    help="videocfg_texture_detail. The tier -> (mip bias, "
                         "max resident mip) table is NOT READ, so setting "
                         "this RECORDS the tier and REFUSES to invent its "
                         "values -- post.detail_tiers.TEXTURE_DETAIL. The "
                         "AXIS exists (--mip-bias, --mip-max-levels); this "
                         "row read GAP_NO_AXIS for most of the project on "
                         "the false basis that it did not.")
parser.add_argument("--particle-detail", type=int, default=None,
                    choices=(0, 1, 2, 3),
                    help="videocfg_particle_detail. Table NOT READ; the "
                         "axis exists (smoke_mboit's own D_SMOKE_QUALITY "
                         "via --smoke-quality, plus D_SMOKE_FULLRES and "
                         "the light-step / shadow-tap / DDA budgets). "
                         "Setting it also prints whether THIS FIXTURE CAN "
                         "EXHIBIT the axis at all: on a render with no "
                         "particles a null across tiers is vacuous, not a "
                         "finding, and the 48-pose de_inferno teleport set "
                         "the arms use may be exactly that case.")
parser.add_argument("--ao-detail", type=int, default=None,
                    choices=(0, 1, 2, 3),
                    help="videocfg_ao_detail. Table NOT READ; the axis "
                         "exists (--ssao and the --ssao-* family, driving "
                         "a transcription of all three shipped SSAO "
                         "modules with its own conformance family). The "
                         "superseded gap row named three BAKED-AO flags "
                         "and concluded no axis existed. NOTE THE DOMAIN: "
                         "the 53 GT configs carry only 0, 2 and 3 -- tier "
                         "1 is never exercised, so it cannot be read from "
                         "this corpus at all.")
parser.add_argument("--cmaa", choices=("on", "off"), default="off",
                    help="r_csgo_cmaa_enable: Conservative Morphological "
                         "AA 2, all three shipped compute shaders "
                         "(cs2_cmaa2_cs_edges / _process_candidates / "
                         "_deferred_apply), transcribed in post/cmaa2.py. "
                         "It is 0 in ALL FOUR presets, so it contributes "
                         "nothing to the preset ladder; the arm that "
                         "exercises it is gtq/caps/r_csgo_cmaa_enable-1.")
# --- env_gradient_fog -------------------------------------------------
# de_inferno.vpk : maps/de_inferno/entities/default_ents.vents_c holds one
# env_gradient_fog (hammerUniqueId 13276:11, StartDisabled false):
#     fogstart 500      fogend 15000      fogfalloffexponent 1.0
#     fogstartheight 0  fogendheight 10000  fogverticalexponent 1.0
#     heightfog true    fogcolor [128,172,212]
#     fogstrength 1.0   fogmaxopacity 1.0   farz -1
# This is a DIFFERENT entity from the env_cubemap_fog the sky work
# measured as inert (that one starts at 800 u and ends at 280000 u).
# Distances below are Source units; the world is metres * 0.0254.
parser.add_argument("--fog", action="store_true",
                    help="the map's own env_gradient_fog")
parser.add_argument("--fog-start", type=float, default=500.0)
parser.add_argument("--fog-end", type=float, default=15000.0)
parser.add_argument("--fog-exponent", type=float, default=1.0)
parser.add_argument("--fog-start-height", type=float, default=0.0)
parser.add_argument("--fog-end-height", type=float, default=10000.0)
parser.add_argument("--fog-vertical-exponent", type=float, default=1.0)
parser.add_argument("--fog-height", type=int, default=1,
                    help="heightfog key (1 = as shipped)")
parser.add_argument("--fog-color", type=_f3, default=[128.0, 172.0, 212.0],
                    help="entity fogcolor, 0-255 sRGB")
parser.add_argument("--fog-strength", type=float, default=1.0)
parser.add_argument("--fog-max-opacity", type=float, default=1.0)
parser.add_argument("--fog-color-space", choices=("srgb", "linear"),
                    default="srgb",
                    help="'srgb' decodes fogcolor with ^2.2 before mixing in "
                         "linear space (entity colours are sRGB bytes); "
                         "'linear' is the control")
parser.add_argument("--fog-stats", action="store_true",
                    help="print the fog distance/opacity distribution once")
parser.add_argument("--fog-sky", action="store_true",
                    help="also fog the no-geometry pixels. OFF by default: "
                         "Source 2 fogs the skybox through env_cubemap_fog, "
                         "not through env_gradient_fog")
# --- light_environment ------------------------------------------------
# Same lump, hammerUniqueId 13276:10, enabled true:
#     angles [63, 218, 0]   brightness 3.0   color [255,247,235]
#     skycolor [136,199,255]  skyintensity 0.875  skytexturescale 1.0
#     bouncescale 1.75  skybouncescale 1.5  angulardiameter 0.3
#     brightnessscale 1.0   skyambientbounce [0,0,0]
# Source angles are [pitch, yaw, roll] of the direction the light POINTS,
# so the direction TO the sun is elevation = pitch = 63, azimuth =
# yaw - 180 = 38. Our fitted sun is el 48 / az 45.
# Radiometric units are NOT transferable (Source's brightness 3.0 is in
# engine light units, ours are arbitrary), so --light-entity takes the
# entity's CHROMATICITY verbatim and leaves only a scalar magnitude per
# term to be fitted -- the specified quantity is implemented, the
# unspecified one is fitted.
parser.add_argument("--sun-entity", action="store_true",
                    help="sun direction from light_environment angles "
                         "(el 63, az 38) instead of the fitted el 48/az 45")
parser.add_argument("--light-entity", action="store_true",
                    help="SUN_COLOR and AMBIENT_SKY chromaticity from "
                         "light_environment's color / skycolor")
parser.add_argument("--light-env", type=str, default="auto",
                    help="per-map light_environment sidecar JSON. DEFAULT "
                         "'auto' = <world pack>.light_environment.json "
                         "beside the pack, the same convention --irr-npy "
                         "and --sky-cube use, because the sun belongs to "
                         "the map and must travel with it. Keys: angles "
                         "[pitch, yaw, roll], color [r,g,b] 0-255, "
                         "brightness, angulardiameter, and OPTIONALLY "
                         "sun_angular_w when a capture has been read for "
                         "that map. 'none' keeps the built-in constants "
                         "and says so.")
parser.add_argument("--sky-ambient-bounce", choices=("read", "zero"),
                    default="read",
                    help="the candidate constant ambient in the SUN path, "
                         "i.e. g_vSunAmbient at byte offset +288. 'read' "
                         "takes the map's own `skyambientbounce` from the "
                         "light_environment sidecar; 'zero' holds it at 0. "
                         "THIS IS AN OPEN QUESTION, NOT A SETTING. The "
                         "buffer member was READ as (0,0,0) on de_inferno "
                         "-- whose skyambientbounce is ALSO [0,0,0], so "
                         "that read cannot tell the two apart. de_dust2 "
                         "ships [151,151,151]: if the member IS "
                         "skyambientbounce, 'read' is correct everywhere "
                         "and 'zero' is wrong on dust2; if they are "
                         "different quantities, 'zero' is correct "
                         "everywhere and 'read' is wrong on dust2. The two "
                         "arms are identical on the other six maps BY "
                         "CONSTRUCTION, since their skyambientbounce is "
                         "[0,0,0] -- so dust2 is the only frame that can "
                         "decide it, and a dust2 RenderDoc capture would "
                         "convert the verdict into a read.")
parser.add_argument("--sky-cube", type=str, default="auto",
                    help="npz holding the reference's own 1024^2 x6 BC6H "
                         "sky cube, decoded from the capture's Initial "
                         "Contents (Valve's mip chain, no resample of "
                         "ours). #48: CS2 draws the sky as ONE degenerate-"
                         "depth [1,1] pass sampling this cube BY VIEW "
                         "DIRECTION with a world Z-rotation and NO "
                         "translation. Our SKY_TOP/SKY_HORIZON gradient is "
                         "a screen-locked image with no orientation state "
                         "-- a different operation, not this one missing a "
                         "constant, which is why it cannot respond to yaw. "
                         "W1 resolved the pages to image 77649 / view "
                         "77656, the object the sky draw binds directly at "
                         "set=1 binding=52 (NOT via the bindless table), so "
                         "this is the sky's texture by identity.")
parser.add_argument("--sky-cube-zrot", type=float, default=115.0,
                    help="degrees. READ from the sky draw's entire uniform "
                         "state -- chunkIndex 40411 binds one 64-byte "
                         "buffer holding ONE matrix: a pure Z-rotation, "
                         "unit scale, no translation. cos(115) = "
                         "-0.4226183 against a read -0.4226194 and "
                         "sin(115) = 0.9063078 against +0.9063073, both "
                         "~1e-6. No translation is the tell: the sky does "
                         "not move with the camera.")
parser.add_argument("--sun-scale", type=float, default=1.0)
parser.add_argument("--sky-level", type=float, default=None,
                    help="override the sky pass's cube-to-framebuffer "
                         "level. DEFAULT is per-map: de_inferno gets 1.6, "
                         "READ from the executed module (78427, set 1 "
                         "binding 0, +72 * +76 * +80 = 2.0 * 1.0 * "
                         "(0.8,0.8,0.8)); the other six carry that value as "
                         "a STATED stand-in with a GAP row, because the "
                         "product's achromatic SHAPE is established "
                         "corpus-wide but its MAGNITUDE is one capture's. "
                         "MEASURED on de_dust2 f2512, where 1.6 is the "
                         "stand-in and not a read: nrmse 0.9707 -> 0.9416, "
                         "kl_rgb 3.4746 -> 3.2071, ncc 0.3377 -> 0.3665. "
                         "All three improve, which CORROBORATES the shape "
                         "and says nothing about whether 1.6 is dust2's "
                         "own number.")
parser.add_argument("--sky-scale", type=float, default=1.0)
parser.add_argument("--ground-scale", type=float, default=1.0)
# The lighting constants below were fitted (fit_lighting2.py) against a
# G-buffer of FLAT FACE normals. Changing the normal field changes the
# ndl/up distribution the fit solved for, so any normal-mapping A/B that
# keeps them frozen is measuring a stale fit, not the feature. These make
# the refit injectable instead of a source edit.
_f3 = lambda s: [float(x) for x in s.split(",")]
parser.add_argument("--amb-ground", type=_f3, default=None)
parser.add_argument("--amb-sky", type=_f3, default=None)
parser.add_argument("--sun-color", type=_f3, default=None)
parser.add_argument("--sky-top", type=_f3, default=None)
parser.add_argument("--sky-horizon", type=_f3, default=None)
# READ, not fitted. The direction TO the sun at byte offset +31120 of the
# set=3 binding=0 block is (0.357750, 0.279504, 0.891006), |v| = 1.00000000,
# identical in the wallA2 daylight and church-interior captures. Elevation
# 63 / azimuth 38 derived from the map's own light_environment angles
# [63, 218, 0] reproduces it to max|delta| = 1.0e-6, so the entity lump and
# the GPU buffer -- independent sources -- agree. It replaces a fitted
# el 48 / az 45 that GROUND_TRUTH_METRICS.md:328 already flagged as chosen
# "IN PREFERENCE to the map's shipped light_environment".
# READ from the capture, two captures agreeing.
# DEFAULT None, NOT 63/38, so "the caller asked for this" is distinguishable
# from "nobody said". With numeric defaults the sidecar overwrote them
# unconditionally a few lines into the resolve, which made both flags DEAD
# LEVERS on every corpus render -- every map ships a sidecar, so the value
# a caller passed was replaced before it could reach anything. 63/38 is
# still the fallback when there is no sidecar and no flag; it just is not
# spelled as an argparse default any more, because that spelling is what
# made an override indistinguishable from a default.
parser.add_argument("--sun-el", type=float, default=None)
parser.add_argument("--sun-az", type=float, default=None)
# Camera latent variables, fitted against ground truth rather than assumed.
# Source defines FOV horizontally at 4:3 and applies Hor+ scaling for
# widescreen, so fov_desired 90 is NOT 90 degrees horizontal at 16:9.
#
# THE DEFAULT WAS 90, DIRECTLY BELOW THE COMMENT SAYING 90 IS WRONG.
# Hor+ is exact, not fitted: a 90-degree horizontal FOV at 4:3 gives
# vfov = 2*atan(tan(45)*3/4) = 73.7398, and holding that vertical at 16:9
# gives hfov = 2*atan(tan(vfov/2)*16/9) = 106.2602. fit_camera_params.py
# calls the 90 "a systematic error" and its sweep picks 106.26.
#
# The cost of the wrong default was not marginal. Structure against the
# holdout reads ncc 0.0013 at hfov 90 -- statistically indistinguishable
# from the mispaired null of -0.0003 -- and 0.0836 at 106.26. A 64x
# difference from one number, and it was published as a renderer baseline.
#
# It defaults here rather than living in render_gt.sh, because a canonical
# value that exists only in a wrapper is the fragility the wrapper was
# written to remove: every arm-B run today needed --hfov added by hand,
# which means the canonical file as committed did not produce the
# canonical render.
parser.add_argument("--hfov", type=float, default=106.2602)
parser.add_argument("--eye-dz", type=float, default=0.0, help="units")
parser.add_argument("--yaw-offset", type=float, default=0.0, help="degrees")
parser.add_argument("--pitch-offset", type=float, default=0.0, help="degrees")
# =======================================================================
# VIEW PUNCH -- the recoil kick on the recording player's own camera
# =======================================================================
# An ego camera that does not kick when the player fires is wrong in every
# frame of a firefight, and the .dem carries the quantity: cs2_demo_camera
# writes aim_punch_{pitch,yaw} and view_punch_{pitch,yaw} per tick.
#
# WHICH ONE IS A SEMANTICS QUESTION THIS FILE DOES NOT GET TO SETTLE BY
# TASTE. harness/cs2_demo_camera.py:20-24 refuses to compose them and says
# why: "how punch composes into the RENDERED view angle ... Anyone who needs
# the composed angle has to settle it against a reference frame first --
# adding them because it looks right is a fit." That refusal stands; this
# does not settle it either. It applies what the COLUMN NAME says -- the
# `view_punch` column is the punch on the VIEW -- and prints STATED, with
# the other readings available as arms so a reference frame can decide
# later without a code change.
#
# MEASURED on the test demo, 57,970 ticks, so the choice is not between one
# live term and three dead ones:
#     aim_punch_pitch    47.61% nonzero, max 7.8494 deg
#     aim_punch_yaw      39.40% nonzero, max 1.1901 deg
#     view_punch_pitch   74.47% nonzero, max 4.7423 deg
#     view_punch_yaw     27.39% nonzero, max 1.5274 deg
#
# NO BASELINE MOVES. The cs1k corpus paths are schema
# iji/cs1k-ego-camera-path/v1 and carry NO punch columns at all -- checked,
# all four absent across 3,284 ticks -- so every existing scored arm renders
# byte-identically and prints a GAP line. Only demo-derived paths
# (iji/cs2-demo-ego-camera-path/v1) can exercise this.
# =======================================================================
# FLASHBANG WHITEOUT -- the screen the blinded player actually sees
# =======================================================================
# Makes the corpus's `flash` class renderable at all: a blinded frame is
# currently indistinguishable from an unblinded one, so every flashbang in
# the corpus scores as though nothing happened.
#
# THE INPUT IS DERIVED, NOT GUESSED, and the derivation is in
# cs2_demo_camera because only there are all the ticks present. READ from
# the wire: `flash_duration` is CONSTANT through an episode -- it is the
# TOTAL blindness set at detonation, not a countdown (tick 4213 onward
# holds 0.2637767493724823 unchanged). So the value alone cannot drive a
# decay; `flash_elapsed_s` and `flash_start_tick` come from the 0 -> v
# transition, computed over the UNDECIMATED tick list.
#
# THE CURVE IS NOT READ AND SAYS SO. What the engine does between onset and
# recovery is not in anything opened here. The default is exponential, per
# the ordering, and its rate is DERIVED rather than fitted: it decays to one
# 8-bit level (1/255) exactly at the blind duration, so K = ln(255) and
# there is no free parameter to tune. `linear` is the parameter-free
# alternative. Both print STATED, because a curve that looks right is the
# thing this file keeps refusing to call read.
parser.add_argument("--flash-whiteout", default="on",
                    choices=("on", "off"),
                    help="full-frame white overlay while the recording "
                         "player is flashbang-blind, decaying over the "
                         "blind duration READ from the wire. A path with no "
                         "flash_elapsed_s prints a GAP and renders "
                         "unchanged.")
parser.add_argument("--flash-curve", default="exp",
                    choices=("exp", "linear"),
                    help="how the whiteout decays. STATED, not read: the "
                         "engine's actual curve is not in anything opened "
                         "here. exp decays to one 8-bit level at the blind "
                         "duration (K = ln 255, DERIVED from output "
                         "precision, no free parameter); linear is "
                         "1 - elapsed/duration.")
parser.add_argument("--view-punch", default="view",
                    choices=("view", "aim", "sum", "none"),
                    help="which punch rotates the camera. view = the "
                         "view_punch columns (DEFAULT, what the column name "
                         "says, STATED not settled). aim = aim_punch. sum = "
                         "both added, which is a COMPOSITION CLAIM nobody "
                         "has verified against a reference frame. none = "
                         "no punch, the pre-existing behaviour. A path "
                         "without punch columns prints a GAP and renders "
                         "unchanged whatever this says.")
parser.add_argument("--frustum-ss", default=None,
                    help="emit frustum state space (eye, rotation basis, 8 world "
                         "corners, htan/vtan/near/far) as JSONL, one record per "
                         "pose. The GT side emits the SAME schema from netcon; "
                         "diffing the two is what qualifies every other measure.")
# --- texture mipmapping / anisotropic filtering -------------------------
# The renderer sampled every map with ONE wrapped-bilinear tap at level 0.
# World UVs tile to +-24x, so on a floor at a grazing angle a single screen
# pixel spans dozens of texels and that one tap is a point sample of a
# high-frequency signal: it aliases. A mip chain plus screen-space-
# derivative LOD selection replaces the point sample with the (band-
# limited) average over the pixel footprint, which is what CS2's hardware
# sampler does.
parser.add_argument("--mip", choices=("off", "on"), default="off",
                    help="trilinear (and, with --max-aniso>1, anisotropic) "
                         "filtering of the colour and aux texture arrays")
parser.add_argument("--mip-bias", type=float, default=0.0,
                    help="added to the computed LOD. >0 blurs, <0 sharpens "
                         "and re-aliases. Swept, not guessed")
parser.add_argument("--max-aniso", type=int, default=1,
                    help="taps along the major footprint axis for the "
                         "COLOUR array. 1 = plain trilinear")
parser.add_argument("--max-aniso-aux", type=int, default=0,
                    help="same for the aux array (normal/height/AO/overlay/"
                         "selfillum); 0 = follow --max-aniso")
parser.add_argument("--mip-color", choices=("on", "off"), default="on")
parser.add_argument("--mip-aux", choices=("on", "off"), default="on")
parser.add_argument("--mip-gamma", choices=("linear", "srgb"),
                    default="linear",
                    help="space the COLOUR mip chain is box-averaged in. "
                         "'linear' decodes ^2.2, averages, re-encodes (what "
                         "a hardware sRGB sampler does); 'srgb' averages the "
                         "stored bytes. The aux array is always raw")
parser.add_argument("--mip-uvconv", choices=("legacy", "half"),
                    default="legacy",
                    help="texel addressing. 'legacy' keeps the existing "
                         "u = frac(uv)*(S-1); 'half' uses the correct "
                         "u = frac(uv)*S - 0.5 texel-centre convention")
parser.add_argument("--mip-max-levels", type=int, default=0,
                    help="cap the mip chain depth (0 = full chain to 1x1)")
parser.add_argument("--irr-bilinear", action="store_true",
                    help="bilinear instead of nearest for the lightmap / "
                         "baked-irradiance atlas lookups")
parser.add_argument("--mip-diag", action="store_true",
                    help="print LOD/derivative statistics for frame 0 and "
                         "exit; validates the nvdiffrast derivative units "
                         "against a finite difference of the interpolated UV")

# =====================================================================
# GROUND-TRUTH LIGHTING / SHADOW COMBO AXES
# =====================================================================
# Ported from the decompiled CS2 pixel shaders committed under
# docs/projects/counter-strike-sft/.  Reference for each term is named on
# the function that implements it, by file and line.  These are the shader
# combos, not a re-derivation:
#
#   D_BAKED_LIGHTING_FROM_LIGHTMAP        csgo_environment_ps.glsl:756-758,
#                                         888, 1067, 1227-1240
#   D_BAKED_LIGHTING_FROM_PROBE           csgo_complex_ps.glsl:302-489
#   D_BAKED_LIGHTING_FROM_VERTEX_STREAM   COLOR1 `PerVertexLighting`,
#                                         SHADER_CALLFLOW_vertex_stages.md:55-71,165
#   (none)                                csgo_complex_ps.glsl:480-489 tail
#   S_LIT                                 SHADER_CALLFLOW_csgo_static_overlay.md:35,61
#   S_ANIMATED_SHADOWS                    SHADER_CALLFLOW_vertex_stages.md:207
#   D_SPECULAR_CUBE_MAP_STATIC            csgo_complex_ps.glsl:922-945 (q0)
#                                         csgo_complex_ps_shaderquality1.glsl:1056-1187 (q1)
#   S_SHADER_QUALITY 0 / 1                csgo_complex_ps.glsl vs
#                                         csgo_complex_ps_shaderquality1.glsl
#
# `--gt-lighting 0` is the composition that existed before these axes
# landed.  It is NOT a fallback for a broken path: every axis below runs
# its full call flow at its own defaults, which --gt-lighting-selftest
# proves pixel-by-pixel.
parser.add_argument("--gt-surface", choices=("auto", "on", "off"),
                    default="auto",
                    help="the GT-SURFACE path as its own axis instead of a "
                         "side effect. It was reachable ONLY as "
                         "any(GT_AXES), so asking for one surface feature "
                         "turned the whole path on for the frame and any "
                         "arm pricing that feature priced both -- the same "
                         "aliasing as --vmat-ao riding --normal-map. "
                         "'auto' is the historical coupling and changes no "
                         "existing command line; 'on'/'off' address the "
                         "axis directly. It is NOT a shading model: it "
                         "decides whether MAT_EXT2 is bound, so with it "
                         "off the gt_* transcriptions have no parameters "
                         "and are absent rather than substituted.")
parser.add_argument("--gt-lighting", type=int, default=1,
                    help="1 = shade through the ported CS2 combo axes "
                         "below instead of the pre-existing fitted "
                         "composition. 0 = the pre-existing composition, "
                         "unchanged. FLIPPED TO 1 AT INTEGRATION, which "
                         "is what the merge blocker that stood here "
                         "demanded: the 0 default existed only to keep "
                         "concurrently-running agents' baselines stable, "
                         "those agents have finished and merged, and a 0 "
                         "default made every path below unreachable at "
                         "its default -- the --ibl-lod-scale 1.0 failure. "
                         "Every --gt-* flag "
                         "is inert at 0 and the startup banner announces "
                         "that; an announced disabled state and a silent "
                         "one are different objects, and this one is "
                         "announced.")
parser.add_argument("--baked-lighting",
                    choices=("auto", "none", "lightmap", "probe",
                             "vertex-stream"),
                    default="auto",
                    help="D_BAKED_LIGHTING_FROM_*. VERIFIED MUTUALLY "
                         "EXCLUSIVE, not assumed: the 16 shipped dynamic "
                         "combo ids of csgo_environment_ps "
                         "(SHADER_CALLFLOW_csgo_environment.md:551-566) are "
                         "{0,1,2,4,8,9,10,12,16,17,18,20,24,25,26,28} and no "
                         "id sets two of the baked bits 1/2/4 -- they "
                         "factor as 4 baked states x D_OPAQUE_FADE x "
                         "D_SPECULAR_CUBE_MAP_STATIC. 'none' is the real "
                         "4th state (the constant-buffer SH tail), not an "
                         "off switch. 'auto' = the engine's own split: "
                         "lightmapped world geometry takes the lightmap, "
                         "everything else takes the probe, which is why "
                         "csgo_environment's primary is d4 and "
                         "csgo_complex's is d2.")
parser.add_argument("--preset", type=int, choices=(0, 1, 2, 3), default=None,
                    help="Render at a CS2 QUALITY BRACKET (#91). The "
                         "preset's whole 11-cvar vector is READ from the "
                         "committed gtq manifest -- not a fidelity label, "
                         "and not invented here. Every cvar that resolves "
                         "to one of our axes is applied and PRINTED; every "
                         "cvar that does not is printed on a GAP LIST with "
                         "its kind (NO_AXIS / UNREAD_TABLE / UNRESOLVED) "
                         "and its basis, and the run then REFUSES. The gap "
                         "list is the deliverable: it is the inventory of "
                         "reference options this renderer does not "
                         "implement. A flag typed on the command line wins "
                         "over the preset and the collision is named.")
parser.add_argument("--preset-accept-gaps", action="store_true",
                    help="Render anyway when --preset leaves cvars "
                         "unresolved. The gap list still prints and the "
                         "artifact must quote it. NOT the default: a "
                         "bracket claim nobody had to acknowledge is a "
                         "bracket claim nobody checked.")
parser.add_argument("--shader-quality", type=int, choices=(0, 1), default=0,
                    help="S_SHADER_QUALITY. NOT a quality scale -- a "
                         "different renderer. Both are shipped for all 11 "
                         "de_inferno csgo_complex static combos and nothing "
                         "in the VPK says which the runtime picks "
                         "(SHADER_CALLFLOW_csgo_complex.md:430-436), so "
                         "both are implemented and this selects. 0 is the "
                         "capture host's cs2_video.txt setting.shaderquality "
                         "(SHADER_CALLFLOW_csgo_environment.md:125-133), "
                         "which is an inference from a name, not a proven "
                         "binding -- hence 1 is equally complete here.")
parser.add_argument("--s-lit", type=int, choices=(0, 1), default=1,
                    help="S_LIT (csgo_static_overlay, combo weight 14). 1 = "
                         "decals/overlays receive the full lit chain; 0 = "
                         "they emit albedo through the blend mode with no "
                         "lighting. 17/17 de_inferno csgo_static_overlay "
                         "materials set F_LIT=1 "
                         "(SHADER_CALLFLOW_csgo_static_overlay.md:44-51), "
                         "which is why 1 is the default.")
parser.add_argument("--animated-shadows", type=int, choices=(0, 1),
                    default=0,
                    help="S_ANIMATED_SHADOWS (csgo_foliage vs+ps). 1 = the "
                         "sun depth pass is rebuilt per frame from the "
                         "current caster pose AND honours the caster's "
                         "alpha cutoff, so an alpha-tested leaf casts its "
                         "silhouette; 0 = one rest-pose depth pass with no "
                         "alpha test. See gt_build_csm() for exactly which "
                         "half of this axis is observable on static world "
                         "geometry.")
parser.add_argument("--spec-cube-static", type=int, choices=(0, 1),
                    default=0,
                    help="D_SPECULAR_CUBE_MAP_STATIC. CONSTRAINED: it "
                         "REQUIRES S_SHADER_QUALITY == 1 in cx:ps, env:ps, "
                         "eb:ps and fol:ps (RENDER_SETTINGS_AXES.md sec 4), "
                         "so at --shader-quality 0 the axis does not exist "
                         "and 1 is refused rather than silently ignored. "
                         "At q=1 the axis REMOVES the binned cubemap-volume "
                         "loop: verified on the shipped enumeration, not "
                         "assumed -- SHADER_CALLFLOW_csgo_environment.md "
                         "line 668 vs 676 is loops 6 -> 4 and FLOPs "
                         "2975 -> 2896, and line 667 vs 675 is 10 -> 8 and "
                         "2965 -> 2895, i.e. exactly the two loops of the "
                         "volume walk. 0 = the binned, box-projected "
                         "(parallax-corrected) volume loop "
                         "(shaderquality1.glsl:1068-1187); 1 = one static "
                         "cubemap with the same downstream "
                         "renormalisation. Three cubemap call flows total, "
                         "counting the unconditional q=0 one at "
                         "csgo_complex_ps.glsl:945; all three implemented.")
# --- psrs: the render-state stage -------------------------------------
# SUPERSEDED, and the earlier note here was wrong in a way worth stating
# rather than quietly deleting: it said "psrs has NO BYTECODE ... these
# two are NOT transcribed from a call flow -- there is no call flow", on
# RENDER_SETTINGS_AXES.md §9's "their render-state payload was not
# decoded". That was true when written and is now false.
#
# SHADER_CALLFLOW_psrs_stage.md decodes it. `psrs` is PIXEL SHADER RENDER
# STATE: it carries no shader program, but it DOES carry executable
# content -- VFX expression programs, whose eight-opcode bytecode ISA is
# recovered in §3 and closes 35/35 of the programs in these six families.
# Both axes below are now transcribed from that, with the same
# file-and-line standard as every other axis in this file.
#
# The one thing still not determined is which of CullMode 1 and 2 is the
# front face; §4.1 says so explicitly and --cullmode-12 keeps it a fork
# rather than resolving it. That is a named gap, not an undecoded stage.
parser.add_argument("--front-face-cull", type=int, choices=(0, 1),
                    default=0,
                    help="D_FRONT_FACE_CULL (psrs, all six families). "
                         "SHADER_CALLFLOW_psrs_stage.md §4.1: "
                         "CullMode = F_RENDER_BACKFACES ? 0 : 1 at axis 0, "
                         "? 0 : 2 at axis 1, exact by set equality in all "
                         "six families, and the ONLY change the axis makes "
                         "over all 656 0->1 pairs. 0 is 'no culling'. "
                         "NOT --backfaces: F_RENDER_BACKFACES is a "
                         "per-MATERIAL feature (cull nothing) and this is "
                         "a per-DRAW axis (flip which side is culled, "
                         "without changing the material) -- two "
                         "mechanisms, implemented as two things.")
parser.add_argument("--cullmode-12", choices=("2-front", "1-front"),
                    default="2-front",
                    help="WHICH OF CullMode 1 AND 2 IS THE FRONT FACE IS "
                         "NOT DETERMINED by any shipped data. psrs §4.1 "
                         "fixes the partition {0} vs {1,2} and the pairing "
                         "(axis=0 <-> 1, axis=1 <-> 2) by set equality, "
                         "but says plainly that assigning 2->FRONT is the "
                         "one step in that document leaning on the axis's "
                         "declared name, and that nothing else in the six "
                         "packages disambiguates it. So it is a FLAG, not "
                         "a constant: '2-front' is the doc's tentative "
                         "reading and the default only because it is the "
                         "doc's; '1-front' is the equally-supported "
                         "alternative. Both are reachable and "
                         "--gt-lighting-selftest checks they produce "
                         "different depth atlases, so the ambiguity stays "
                         "a live fork instead of a buried choice.")
parser.add_argument("--disable-depth-bias", type=int, choices=(0, 1),
                    default=0,
                    help="D_DISABLE_DEPTH_BIAS (psrs, all six families). "
                         "psrs §4.2: its effect DIFFERS BY FAMILY. On "
                         "env/fol/gl it removes SlopeScaleDepthBias, "
                         "live iff (D_DISABLE_DEPTH_BIAS==0 and "
                         "S_MODE_DEPTH==1), exact by set equality over "
                         "24/24/96 records. On cx/eb/so it changes "
                         "NOTHING (0 of 240, 0 of 96, 0 of 32) -- their "
                         "bias comes from F_DEPTH_BIAS and is always "
                         "live. Distinct from --gt-csm-bias, the SHADER's "
                         "g_flShadowBias added to the reference at sample "
                         "time (csgo_complex_ps.glsl:570); CS2 has both.")
parser.add_argument("--gt-depth-bias-pass", type=float, default=5.0,
                    help="SlopeScaleDepthBias for the DEPTH-PASS "
                         "mechanism, env/fol/gl (psrs §4.3b). MEASURED: "
                         "+5.0, a constant with no expression, and "
                         "DepthBias/DepthBiasClamp are not declared at all "
                         "in those three families. de_inferno reaches it "
                         "on all 15 csgo_environment, both csgo_foliage "
                         "and both csgo_glass materials. NOTE THE SIGN "
                         "against --gt-depth-bias-material: opposite "
                         "polarity, different pass, not normalised.")
parser.add_argument("--gt-depth-bias-material", type=float, nargs=3,
                    default=[-64.0, -2.0, -5.0000002e-04],
                    help="(DepthBias, SlopeScaleDepthBias, "
                         "DepthBiasClamp) for the MATERIAL mechanism, "
                         "cx/eb/so, gated on F_DEPTH_BIAS (psrs §4.3a). "
                         "MEASURED from the expression bytecode: "
                         "-64 / -2.0 / -5.0000002e-04. csgo_static_overlay "
                         "ships the same triple as literal negatives with "
                         "no expression, and that cross-family agreement "
                         "is what fixes opcode 0x18 as NEG -- max|delta| "
                         "between the two routes is 0 on all three. "
                         "DepthBias is D3D-style integer units of 1/2^24.")
parser.add_argument("--gt-csm", type=int, default=2048,
                    help="per-cascade sun depth map resolution for the GT "
                         "cascade path. NON-ZERO BY DEFAULT ON PURPOSE: "
                         "S_SHADER_QUALITY=1's headline difference is three "
                         "9-tap PCF kernels, and a 0 default would make "
                         "them unreachable by construction -- the "
                         "--ibl-lod-scale 1.0 failure. Independent of "
                         "--shadow, which drives the pre-existing "
                         "single-map path.")
parser.add_argument("--gt-csm-cascades", type=int, default=3,
                    help="cascade count. READ: 3. The engine's own value, "
                         "at byte offset +31248 of the set=3 binding=0 "
                         "block, identical in both the wallA2 daylight and "
                         "the church-interior captures and across all 718 "
                         "block copies found (READ, two captures, 718 block copies). "
                         "The bound 1..4 still holds -- the loop body "
                         "indexes a mat4[4] and a vec4 by the loop counter "
                         "(SHADER_CALLFLOW_csgo_complex.md:301) -- and the "
                         "capture carries FOUR matrices of which the loop "
                         "reaches three; the fourth has a different depth "
                         "range and is never selected at count 3.")
parser.add_argument("--gt-csm-split", type=float, default=0.7,
                    help="practical-split-scheme lambda blending the "
                         "logarithmic and uniform cascade splits. The "
                         "shader reads its cascade extents from a uniform "
                         "(g_vCascadeExtent, _5538._m16) that no committed "
                         "artefact carries, so the split has to be "
                         "generated; the CONSUMPTION of it -- "
                         "max(|x|,|y|) < extent[i] -- is the shader's, "
                         "verbatim.")
parser.add_argument("--gt-csm-bias", type=float, default=-1.52587890625e-05,
                    help="g_flShadowBias. READ: -2**-16 exactly, at byte "
                         "offset +31256, identical in both captures and "
                         "all 718 block copies (READ, two captures, 718 block copies). "
                         "It replaces a swept -0.0015, which was 98x "
                         "larger. Applied exactly where the shader applies "
                         "it -- added to the REFERENCE depth, then clamped "
                         "to [0,1], before the depth compare "
                         "(csgo_complex_ps.glsl:570, "
                         "csgo_environment_ps.glsl:840). The sign is the "
                         "reference's, not an argument about acne; and a "
                         "negative bias into a [0,1] clamp only makes "
                         "sense with non-negative stored depths, so the "
                         "read value CONFIRMS the cascade depth-sign fix "
                         "(the cascade depth-sign fix) from a second direction.")
parser.add_argument("--gt-probe-fade", type=float, default=0.1,
                    help="SPECULAR-CUBE border fade only, as a fraction of "
                         "the box extent, for the binned cube walk at "
                         "csgo_complex_ps.glsl:1074/:1145. NO LONGER FEEDS "
                         "THE PROBE VOLUMES: this help used to say "
                         "'probe_field.npz does not carry it', which was "
                         "true of the table and never true of the map -- "
                         "edge_fade_dists is an authored per-axis field on "
                         "every light_probe_volume entity, the builder now "
                         "reads it, and the probe path uses it (:426-428) "
                         "instead of this fraction. TWO CHAINS, DECLARED: "
                         "the cube walk reuses the probe BOXES as cube "
                         "bounds and has no per-box fade of its own to "
                         "read, so its fraction stands until someone reads "
                         "ITS source; sweeping it out with the probe one "
                         "would have silently retuned a term nobody "
                         "measured.")
parser.add_argument("--gt-csm-fade", type=float, nargs=2,
                    default=[-9.0, 10.0],
                    help="g_vCascadeBlend = (_5538._m17, _5538._m18): "
                         "fade = 1 - clamp(|xy|*m18 + m17, 0, 1) per axis, "
                         "the two multiplied "
                         "(csgo_complex_ps.glsl:553-558). READ at byte "
                         "offsets +31280 and +31284: (-9.0, +10.0), "
                         "identical in both captures "
                         "(READ, two captures, 718 block copies). This pair was "
                         "GUESSED from 'the outer tenth' before it was "
                         "read and the guess was exactly right, so the "
                         "default is unchanged -- recorded because a "
                         "confirmed guess and a read value are different "
                         "epistemic objects and only the second is now "
                         "load-bearing. Cascade coords "
                         "are clip space, so |xy| runs 0..1 and the "
                         "values put the blend band at |xy| in "
                         "[0.9, 1.0] -- the outer tenth. Values that put "
                         "the band outside [0,1] would make the "
                         "next-cascade branch at :573 unreachable, which "
                         "--gt-lighting-selftest reports as a live count "
                         "of 0.")
parser.add_argument("--gt-csm-distance-fade", type=float, nargs=2,
                    default=None,
                    help="(_5538._m19, _5538._m20): shadow fades to fully "
                         "lit with clamp(distance(wpos,eye)*m20 + m19, 0, 1) "
                         "(csgo_complex_ps.glsl:602), metres. DEFAULT IS "
                         "DERIVED FROM THE SCENE at CSM build: the ramp "
                         "runs from 0.8x to 1.0x the world radius. A fixed "
                         "pair cannot do this -- on a 50 m map any "
                         "hundreds-of-metres default leaves the term "
                         "identically 0 and the fade never runs, which is "
                         "the --ibl-lod-scale failure with a different "
                         "constant.")
parser.add_argument("--gt-pcf-weights", type=float, nargs=3,
                    default=[0.0625, 0.125, 0.25],
                    help="S_SHADER_QUALITY=1 9-tap kernel weights "
                         "(corner=_5538._m0.w, edge=_5538._m1.x, "
                         "centre=_5538._m1.y; "
                         "csgo_complex_ps_shaderquality1.glsl:634,649). The "
                         "uniforms are not in any committed artefact; the "
                         "defaults are the separable binomial 1-2-1 x 1-2-1 "
                         "that sums to exactly 1 for these three tap "
                         "classes (4*0.0625 + 4*0.125 + 0.25 = 1).")
parser.add_argument("--gt-pcf-early-out", type=int, default=1,
                    help="the shader's 4-tap early exit: if the corner "
                         "average is exactly 0 or exactly 1, the 5 "
                         "remaining taps are skipped and the average is "
                         "returned (shaderquality1.glsl:635-648). Changes "
                         "cost, and -- because a 4-tap mean and a 9-tap "
                         "binomial mean are not the same estimator -- the "
                         "result on fully-lit/fully-shadowed texels too. "
                         "DIAGNOSTIC: 0 forces all 9 taps.")
parser.add_argument("--term-values-json", default=None,
                    help="write the TERM VALUES table as json here, for "
                         "gate.py's inert-term precondition. Contains "
                         "mean/min/max/frac_at_identity per instrumented "
                         "term. It does NOT contain frac_reaching_output "
                         "and says why in the file: that is a property of "
                         "a term's CONSUMER, not of the term.")
parser.add_argument("--gt-lm-direction", default=None,
                    help="the THIRD lightmap sampler2DArray, bound only at "
                         "S_SHADER_QUALITY=1 (_4565._m2, DECLARED BYTE "
                         "OFFSET 20 -- this help said _m3 until the offsets "
                         "were re-read, and _m3 is offset 24, which is the "
                         "OCCLUSION page and is the same slot q0 calls _m2; "
                         "`_mN` ordinals are not stable across combos and "
                         "naming by ordinal is how the two got swapped. "
                         "csgo_environment_ps_shaderquality1.glsl:817). "
                         ".npy of (H,W,>=3) uint8 or float. THE ENCODING IS "
                         "READ FROM THE MODULE, not from the page's "
                         "statistics: Q1:818-828 decodes .xy as a "
                         "hemispherical disc direction and Q1:829 consumes "
                         ".z as the confidence in "
                         "clamp(dirPage.z + bias, 0, 1). Without it the q1 "
                         "lightmap cannot respond to the normal map, which "
                         "is the largest behavioural difference between the "
                         "two quality levels on the 95/153 materials that "
                         "take D_BAKED_LIGHTING_FROM_LIGHTMAP.")
parser.add_argument("--gt-lm-direction-bias", type=float, default=0.0,
                    help="g_flLmBias at Q1:829, added to the confidence "
                         "channel before the clamp. 0.0 until it is READ "
                         "from a bound constant buffer -- it is not in any "
                         "committed artefact, and the term prints its own "
                         "note saying so rather than implying 0 was read.")
parser.add_argument("--gt-soft-terminator", choices=("on", "off"),
                    default="off",
                    help="the q1-ONLY soft terminator, "
                         "csgo_environment_ps_shaderquality1.glsl:1113-1119 "
                         "and :1440-1447: both the direct diffuse and the "
                         "direct specular are multiplied by "
                         "smoothstep(0, 1, clamp(dot(n,l) / "
                         "sqrt(max(1 - matAO, 1e-5)), 0, 1)), which widens "
                         "the sun's terminator in proportion to the "
                         "material's own occlusion. q0 has NO counterpart. "
                         "DEFAULTS OFF because the shader's own selector "
                         "for it -- notEqual(_5037._m2, ivec4(0)).w -- is "
                         "not in any committed artefact, so neither state "
                         "is READ. Off does not mean unimplemented: the "
                         "expression is here and the registry scores it.")
parser.add_argument("--gt-lm-bicubic", choices=("on", "off"), default="on",
                    help="the q1 B-spline bicubic on the OCCLUSION page "
                         "(_4565._m3, offset 24; "
                         "csgo_environment_ps_shaderquality1.glsl:839-864). "
                         "IT IS ONE PAGE, NOT ALL THREE: the irradiance "
                         "page (:809) and the direction page (:817) are "
                         "plain texture() taps at BOTH qualities. This "
                         "renderer bicubic'd the irradiance page too until "
                         "#57, which low-passed the dominant light term at "
                         "q1 and not at q0 -- a q1-only loss on the 95/153 "
                         "materials that take the lightmap. "
                         "The reference's :277-286 gate "
                         "(lightmap density > _5037._m18 AND "
                         "min(fwidth(lmUV.zw)) > 0.1) is NOT evaluated "
                         "here: this pack carries no second lightmap UV "
                         "pair to take fwidth of. So 'on' runs the filter "
                         "the reference runs when the gate passes and 'off' "
                         "runs the :866 fallback it runs when the gate "
                         "fails, and the banner says the precondition was "
                         "not checked instead of implying it held.")
parser.add_argument("--gt-lm-occlusion", default="auto",
                    help="the SECOND lightmap sampler2DArray "
                         "(csgo_environment_ps.glsl:758): a per-light baked "
                         "occlusion vec4 consumed as 1 - dot(occ, mask), "
                         "NOT an irradiance. .npy of (H,W,4). DEFAULT "
                         "'auto' = <world pack>.direct_light_shadows.npy "
                         "beside the pack, the same convention --irr-npy "
                         "and --sky-cube use, because the page was baked "
                         "for that map and must travel with it. 'none' "
                         "disables it explicitly. Without it the lightmap "
                         "path runs with occ = 0, which is the shader's own "
                         "value when the map is unbound -- a DEFINED state, "
                         "not a substitution, which is why a missing page "
                         "says so loudly instead of exiting the way "
                         "--irr-npy does.")
parser.add_argument("--gt-lm-tint", type=int, default=0,
                    help="g_bLightmapTint, the int at BYTE OFFSET 60 of the "
                         "set-1 binding-0 block, spelled _5618._m6 in "
                         "csgo_environment_ps.glsl and gated at :525. The "
                         "INDIRECT TERM -- irradiance times AO, and at q=1 "
                         "times the energy factor as well -- gets "
                         "pow(x, vec3(1.05, 0.95, 0.80)) mixed in at full "
                         "weight (:1230-1240 at q=0, :1588-1598 at q=1). "
                         "Applied after AO, not to the bare lightmap: pow "
                         "is not linear, so the two differ. "
                         "DEFAULT 0 IS A READ, not a preference: the member "
                         "reads int 0 (bytes 00000000) on all 86 env_out "
                         "draws of the wallA2 de_inferno capture, and every "
                         "value seen is in {0,1} so it is the bool the "
                         "shader treats it as. ONE capture, ONE map -- the "
                         "block is per-material, so another map may tint; "
                         "this is the only read that exists and it beats "
                         "the 1 that was here on no read at all.")
parser.add_argument("--gt-vertex-lighting", default=None,
                    help="COLOR1 `PerVertexLighting` for "
                         "D_BAKED_LIGHTING_FROM_VERTEX_STREAM, .npy of "
                         "(nverts,3). This is NOT COLOR0 "
                         "`VertexPaintTintColor` and NOT --vertex-color-mode "
                         "(SHADER_CALLFLOW_vertex_stages.md:165-168: "
                         "'supplying one where the other is expected is a "
                         "silent error'). If absent the stream is BAKED at "
                         "load from the same baked irradiance CS2's own "
                         "baker reads -- see gt_bake_vertex_stream().")
parser.add_argument("--gt-ambient-sh", type=float, nargs=12, default=None,
                    help="g_vAmbientCube, the 3 vec4 of the "
                         "D_BAKED_LIGHTING=none tail "
                         "(csgo_complex_ps.glsl:483-484): irradiance = "
                         "vec3(dot(sh[c], vec4(n,1))). DEFAULT IS DERIVED "
                         "FROM AMBIENT_SKY/AMBIENT_GROUND, which the affine "
                         "form represents exactly (see gt_ambient_sh()); a "
                         "fixed default would silently replace the fitted "
                         "hemisphere.")
parser.add_argument("--gt-sun-occ-mask", type=float, nargs=4,
                    default=[1.0, 0.0, 0.0, 0.0],
                    help="g_vSunLightOcclusionMask (_5538._m13): the sun's "
                         "attenuation is shadow * (1 - dot(bakedOcc, mask)) "
                         "(csgo_complex_ps.glsl:618). Slot 0 = the sun.")
parser.add_argument("--gt-spec-cube-lod-scale", type=float, default=None,
                    help="g_flEnvMapLodScale in lod = scale*sqrt(roughness) "
                         "(csgo_complex_ps.glsl:945, _5538._m9.y). DEFAULT "
                         "IS DERIVED FROM THE ASSET (nmips-1) so that "
                         "sqrt(roughness) in [0,1] spans the whole chain -- "
                         "the same defaulting --ibl-lod-scale had to adopt "
                         "after a fixed 1.0 made mips 2..6 unreachable.")
parser.add_argument("--gt-spec-cube-clamp", type=float, default=1.75,
                    help="q0 cubemap renormalisation magnitude clamp "
                         "(csgo_complex_ps.glsl:945).")
parser.add_argument("--gt-spec-cube-boost", type=float, nargs=2,
                    default=[0.0, 1.0],
                    help="q1 luminance-ratio ceiling max(rough*x + y, 1.0) "
                         "(_5538._m4.xy; shaderquality1.glsl:1187).")
parser.add_argument("--gt-parallax-box", type=float, default=1.0,
                    help="q1 box-projection extent multiplier on the probe "
                         "volume used for the parallax correction "
                         "(shaderquality1.glsl:1139-1143). 1.0 = the probe "
                         "volume's own box, which is what the shader "
                         "intersects.")
parser.add_argument("--gt-probe-blend", type=int, default=1,
                    help="the probe path's soft volume blend: "
                         "w += t*t*(3-2*t) * (1-w) over the binned volume "
                         "list, early-out at w > 0.99 "
                         "(csgo_complex_ps.glsl:354, 440-441). 0 = "
                         "DIAGNOSTIC hard single-volume select, which is "
                         "what sample_probes() did before this axis landed.")
parser.add_argument("--gt-lighting-selftest", action="store_true",
                    help="run every path this file adds at ITS OWN "
                         "DEFAULTS on a synthetic pixel batch, print how "
                         "many pixels took each path and max|delta| against "
                         "the path suppressed, then EXIT NON-ZERO if any "
                         "selected path is dead. A code-path reachability "
                         "proof, not an image metric: no ground-truth "
                         "frame is read. EXECUTED 2026-08-08 on ws-1 "
                         "(RTX 5090, torch 2.11) against de_inferno: exit "
                         "0, all required paths live. The fault-injection "
                         "loop was run for all ten --gt-selftest-inject "
                         "names and FIVE fire (csm-dead, pcf-collapse, "
                         "dfg-zero, distfade-full, ambientsh-zero); five "
                         "return 3 -- INJECTION DID NOT FIRE (probe-zero, "
                         "speccube-zero, lightmap-zero, cullmode-tie, "
                         "biasfam-collapse), because each needs an input "
                         "this invocation has no artefact for. So the "
                         "reachability claim covers five paths, not ten: "
                         "see the docstring on gt_lighting_selftest().")
parser.add_argument("--gt-selftest-inject", default=None,
                    help="deliberately break one named path, so the "
                         "self-test can be made to FAIL ON PURPOSE. "
                         "CHECKS_THAT_CANNOT_FAIL.md: 'Can this check "
                         "fail? Make it fail on purpose. If you cannot, "
                         "it is not a check' -- and its instance 5 is a "
                         "checker that was wrong three times and was only "
                         "trusted after known defects were injected and "
                         "all three fired. Until every name below has "
                         "been injected and seen to fire, a clean "
                         "self-test run is indistinguishable from a "
                         "self-test that never ran. Names: "
                         "csm-dead, pcf-collapse, dfg-zero, probe-zero, "
                         "distfade-full, speccube-zero, ambientsh-zero, "
                         "lightmap-zero, cullmode-tie, biasfam-collapse. "
                         "'all' lists them and exits.")
# =====================================================================
# csgo_lightmappedgeneric.vfx / generic.vfx (csgo_core) / the
# csgo_imported search path's dev placeholder.
#
# EVERY FLAG BELOW IS NAMESPACED `--lmg-`. Three agents landed features
# into this file in the same session and a duplicate `add_argument` name
# raises at import, which takes the parser down for EVERY invocation, not
# just the one that uses the flag. `--lmg-` is a prefix nothing else in
# this file uses; the replay in lmg_parser_replay() proves it.
#
# NO STRING-VALUED FLAG HERE MAY BE TESTED WITH BARE TRUTHINESS: "off"
# is a true string. Every one is compared with `== "on"` explicitly.
# =====================================================================
parser.add_argument("--lmg-family", choices=("on", "off"), default="on",
                    help="shade csgo_lightmappedgeneric.vfx, generic.vfx "
                         "(the csgo_core copy) and the csgo_imported dev "
                         "placeholder through their OWN transcribed pixel "
                         "shaders instead of the generic composite. ON by "
                         "default: it costs nothing where those families "
                         "own no pixels, and a default of off would make "
                         "three ported shaders unreachable while reading "
                         "as tested. Needs --fam-side, because without the "
                         "family side table no pixel knows which family "
                         "owns it -- that is the same precondition every "
                         "other family axis in this file has, not a quiet "
                         "disable.")
parser.add_argument("--lmg-baked", default="auto",
                    choices=("auto", "lightmap", "vertex-stream", "probe",
                             "none"),
                    help="D_BAKED_LIGHTING_FROM_* for these three "
                         "families. All four values are real shipped "
                         "modules of static combo 0: r0_m3 (LIGHTMAP, "
                         "dyn 4), r0_m1 (VERTEX_STREAM, dyn 1), r0_m2 "
                         "(PROBE, dyn 2) and r0_m0 (dyn 0, the ambient-SH "
                         "path). 'auto' takes the lightmap where the pixel "
                         "has one and the shipped dyn-0 module where it "
                         "does not, which is the engine's own split.")
parser.add_argument("--lmg-opaque-fade", choices=("on", "off"),
                    default="off",
                    help="D_OPAQUE_FADE (dynamic stride 16) for these "
                         "families. Default off because the module "
                         "de_inferno selects is dyn 4, NOT dyn 20 -- this "
                         "is the reference's own dynamic selection, not a "
                         "feature switched off. Both values are exercised "
                         "unconditionally by --lmg-selftest.")
parser.add_argument("--lmg-lm-direction", default=None,
                    help="the THIRD lightmap sampler2DArray generic.vfx "
                         "fetches (generic csgo_core r0_m3:270 -- xy = the "
                         "dominant baked direction, z = the baked AO "
                         "level). csgo_lightmappedgeneric has no such "
                         "array; only generic.vfx does. With no file the "
                         "directional reconstruction degenerates to the "
                         "shader's own dir.z==1 case, which is the "
                         "isotropic lightmap, and the substitution is "
                         "PRINTED rather than silently taken.")
parser.add_argument("--lmg-selftest", choices=("on", "off"), default="on",
                    help="direct-call reachability proof for all three "
                         "families at every value of every combo axis "
                         "they declare. Prints, per family and per axis "
                         "value, how many synthetic pixels executed and "
                         "the max|delta| against the neighbouring value, "
                         "so an axis that changes nothing is visible as a "
                         "zero. This is how csgo_lightmappedgeneric -- "
                         "which owns 0 pixels in all 48 GT poses -- is "
                         "shown to be called at all.")
parser.add_argument("--lmg-selftest-inject", default=None,
                    help="break one named path on purpose so the "
                         "self-test can be made to FAIL. Names: "
                         "lmg-albedo-passthrough, lmg-metal-ignored, "
                         "gen-direction-ignored, imported-unrouted, "
                         "combo-bitmask. The last one decodes the static "
                         "combo with & instead of mixed-radix division, "
                         "which is the documented container trap "
                         "(vcs/README.md section 1) -- it must move the "
                         "reported combo ids of csgo_lightmappedgeneric, "
                         "whose S_TEXTURETRANSFORMS sits at stride 96.")
# =======================================================================
# APPEND-ONLY BLOCK -- CS2 SSAO SUBSYSTEM + DEPTH-KEYED FILTER FAMILIES
# (impl-ssao-filters). Everything below is ONE contiguous block appended
# at the very end of the parser list. Do not interleave: four merges on
# 2026-08-07 needed hand-resolution because two agents appended to the
# same anchor and git spliced one string literal's tail onto another's
# head. Add your own block AFTER this banner's block, not inside it.
#
# EVERY FLAG HERE IS NAMESPACED --cs2ao-* / --aoproxy-* / --atrous-* /
# --dnb-* / --bwd-* / --cs2filt-*.
#
# SLICE (stated per the brief; sibling impl-filters-binners takes the
# other 16 filters + 2 binners, agreed by message):
#   ssao, ssao_scalable_ambient_obscurance, ssao_bilateral_blur,
#   aoproxy_splat, atrous_filter, denoise_blur, blur_with_depth
# plus the two SSAO-chain producers ssao_convert_depth and
# ssao_downsample_depth, which are not in the 41 (zero loops) but are
# the inputs the estimator cannot run without.
#
# PROVENANCE: transcribed from SPIR-V decompiled off the render host
# (nkcut2:/home/bigboi/cs2/game/{core,csgo}/shaders_vulkan_dir.vpk ->
# spirv-cross --vulkan-semantics). 69 modules, container-declared size
# == instruction-walk on 69/69, spirv-cross exit 0 on 69/69. Line
# citations below are into .scratch-ssao/<family>/r<rec>m<mod>.glsl.
#
# THESE ARE FILTER KERNELS, NOT BINNERS. Every loop here is a fixed-tap
# kernel with a literal bound (11 / 9 / 16 / 25 / 5 / 7 / 13 / 3x3 /
# 5x5) except aoproxy_splat's two, whose bounds are a per-instance
# uvec3 range. NO findLSB appears in any module of any of the seven, so
# none of them is a clustered-light loop.
# =======================================================================
parser.add_argument("--cs2ao-estimator",
                    choices=("fitted", "sao", "ssao", "both"),
                    default="sao",
                    help="which ambient-obscurance estimator --ssao runs. "
                         "CS2 ships TWO, side by side, and they are "
                         "different algorithms rather than quality levels "
                         "of one. 'sao' = ssao_scalable_ambient_obscurance "
                         "(11 spiral taps on a camera-space-Z mip chain). "
                         "'ssao' = ssao.vfx (9/16/25 world-space sphere "
                         "taps against a RAY-DISTANCE buffer, kernel "
                         "reflected about a noise vector). 'both' "
                         "multiplies them. 'fitted' is the pre-existing "
                         "McGuire-2012-paper stand-in this file carried "
                         "before the bytecode was read -- KEPT because "
                         "the prior measured numbers used it, not because "
                         "it is the reference. STRING-VALUED: never "
                         "tested for truthiness, every value here is "
                         "truthy including 'fitted'.")
parser.add_argument("--cs2ao-quality", type=int, default=2,
                    choices=(0, 1, 2),
                    help="S_SSAO_QUALITY, ssao.vfx static place value 1, "
                         "m_nMin 0 m_nMax 2. Selects the sample-kernel "
                         "TABLE, and the three tables are different "
                         "hard-coded constant arrays in three different "
                         "bytecode records: 9 taps (r0m0:4), 16 (r1m0:4), "
                         "25 (r2m0:4), with the final divisor 1/9, 1/16, "
                         "1/25 at r0m0:85 / r1m0:85 / r2m0:85. All three "
                         "tables are transcribed verbatim; per "
                         "CHECKS_THAT_CANNOT_FAIL.md a quality level "
                         "never gates implementation. Default 2 = the "
                         "widest, so the default exercises the most code.")
parser.add_argument("--cs2ao-alpha-blend", type=int, default=0,
                    choices=(0, 1),
                    help="S_ALPHA_BLEND, ssao.vfx static place value 3. "
                         "DEDUPLICATES COMPLETELY: 6 static combos map to "
                         "3 bytecode records, and the axis changes no "
                         "instruction -- it selects host blend state, not "
                         "shader code. Recorded and exercised so the "
                         "dedup is visible as a measured zero rather than "
                         "an untested assumption.")
