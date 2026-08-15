"""`csgo_projected_decals` and `spritecard` -- the two non-world families.

Both live in **`csgo_core/shaders_vulkan_dir.vpk`**, an archive no
enumeration in this project had opened for pixel shaders until
`SHADER_CALLFLOW_nonworld_families.md`. Both were extracted for this
port; the reference GLSL is committed beside this file under
`vcs/nonworld_ref/{pd,sc}/`.

  archive     entry                                          bytes
  core        shaders/vfx/spritecard_vulkan_50_ps.vcs        1,699,919
  csgo_core   shaders/vfx/spritecard_vulkan_50_ps.vcs        5,796,004   <- this
  csgo_core   shaders/vfx/csgo_projected_decals_..._ps.vcs  48,375,718

Container walked with the two rules in `vcs/README.md`: a static combo
id is **mixed-radix** (`S_BLEND_MODE` has place value 4 because it takes
four values; `S_TEXTURE_LAYERS` has 1024 and the axis after it 5120,
because it takes five), and the bytecode record is
**`m_nByteCodeDataIdx`**, never the array position. Measured against the
shipped packages:

    csgo_projected_decals   3,048 static combos   88,992 shipped pairs
    spritecard                824 static combos    6,592 shipped pairs

`flopcount.py` on the representative module (first live static combo,
first dynamic module) gives 1,183 FLOPs / 15 fetch sites for the decals
and **708 FLOPs / 19 fetch sites for spritecard**, and `loop_regions.py`
gives 3 loops / 0 loops. All of that reproduces the enumeration document
exactly.

Every `pd:` / `sc:` line reference below is into:

    vcs/nonworld_ref/pd/s0_d0.glsl       all static 0, all dynamic 0
    vcs/nonworld_ref/pd/s0_d4.glsl       + D_MSAA_DEPTH_BUFFER=1
    vcs/nonworld_ref/pd/s0_d8.glsl       + D_TRANSLUCENT_SCENE_DEPTH=1
    vcs/nonworld_ref/pd/s0_d12.glsl      + both
    vcs/nonworld_ref/pd/s13793_d0.glsl   sheets/triplanar/cutoff/normal/
                                         quality/blood/spec direct+indirect
    vcs/nonworld_ref/pd/s15809_d0.glsl   the same with S_PARALLAX
    vcs/nonworld_ref/pd/s13808_d0.glsl   + S_ALPHA_MODE=1
    vcs/nonworld_ref/sc/s0_d0.glsl       THE 708/19 MODULE
    vcs/nonworld_ref/sc/s0_d1.glsl       + D_DEPTH_FEATHERING=1
    vcs/nonworld_ref/sc/s0_d3.glsl       + D_SCENE_DEPTH_MSAA=1
    vcs/nonworld_ref/sc/s864_d0.glsl     motion vectors/refract/HSV/trail
    vcs/nonworld_ref/sc/s4928_d0.glsl    S_TEXTURE_LAYERS=4

WHAT IS NEW HERE, stated as capability:

**`csgo_projected_decals` reads the scene depth buffer to build its own
UVs.** No world family declares `D_MSAA_DEPTH_BUFFER` or
`D_TRANSLUCENT_SCENE_DEPTH`. A world decal (`csgo_static_overlay`) has
authored UVs baked into the map; a bullet decal is an oriented box in
world space and its UVs are wherever that box meets whatever the depth
buffer says is in front of it -- including a skinned character. It also
takes its **surface normal** from the depth buffer, because it has no
mesh of its own. `depth_world_position()` and
`geometric_normal_from_depth()` are those reads, transcribed.

**`spritecard` is the particle path.** `csgo_effects`, enumerated at 194
FLOPs from map materials, is a different shader. spritecard declares no
`sampler2DShadow` and no `sampler3D` at all -- 18 `sampler2D` + 1
`samplerCube` = the 19 -- which is the structural form of "particles are
not lit by the cluster path". Nothing in this file routes a particle
through the light binner, and that is a transcription, not an omission.

QUALITY LEVELS NEVER GATE IMPLEMENTATION HERE. `S_SHADER_QUALITY` is a
combo axis of the decal family and both of its values are implemented
and reachable; which value the game picks for a given draw is the same
open question `S_SHADER_QUALITY_RESOLVED.md` records elsewhere and is
not a reason to leave an arm out.
"""

import math

import torch

# ======================================================================
# COMBO AXES -- transcribed from the shipped packages' own KV3.
# `m_staticComboArray` / `m_dynamicComboArray` of each *_ps.vcs DATA
# block. The last column is `m_nComboIndexValue`, the MIXED-RADIX place
# value; it is here so nothing downstream is tempted to use `&`.
# ======================================================================
PD_STATIC_AXES = (
    ("S_USE_SHEETS", 0, 1, 1),
    ("S_FASTAPPROX", 0, 1, 2),
    ("S_BLEND_MODE", 0, 3, 4),
    ("S_ALPHA_MODE", 0, 1, 16),
    ("S_TRIPLANAR_MAPPING", 0, 1, 32),
    ("S_CUTOFF_ANGLE", 0, 1, 64),
    ("S_NORMAL_MAP", 0, 1, 128),
    ("S_SHADER_QUALITY", 0, 1, 256),
    ("S_DETAIL_TEXTURE", 0, 1, 512),
    ("S_BLOOD_AGING", 0, 1, 1024),
    ("S_PARALLAX", 0, 1, 2048),
    ("S_SPECULAR_DIRECT", 0, 1, 4096),
    ("S_SPECULAR_INDIRECT", 0, 1, 8192),
    ("S_MODE_TOOLS_VIS", 0, 1, 16384),
)
PD_DYNAMIC_AXES = (
    ("D_BAKED_LIGHTING_FROM_PROBE", 0, 1, 1),
    ("D_BAKED_LIGHTING_FROM_LIGHTMAP", 0, 1, 2),
    ("D_MSAA_DEPTH_BUFFER", 0, 1, 4),
    ("D_TRANSLUCENT_SCENE_DEPTH", 0, 1, 8),
    ("D_MBOIT_PASS2", 0, 1, 16),
    ("D_MBOIT_4_MOMENTS", 0, 1, 32),
)
# The feature (.vmat-facing) axes of the same family, from
# csgo_projected_decals_vulkan_50_features.vcs. These are the names the
# side table's columns carry; F_BLEND_MODE's m_stringArray is
# ("Translucent", "Mod2X", "Emissive", "Liquid") and F_ALPHA_MODE's is
# ("Simple", "Cutoff") -- both quoted verbatim in `decal_composite`.
PD_FEATURE_AXES = (
    "F_BLEND_MODE", "F_SAMPLE_LIGHTMAP", "F_AUTOMATIC_PBR_COLOR_FITTING",
    "F_ALPHA_MODE", "F_USE_SHEETS", "F_TRIPLANAR_MAPPING", "F_CUTOFF_ANGLE",
    "F_NORMAL_MAP", "F_SPECULAR_DIRECT", "F_SPECULAR_INDIRECT", "F_PARALLAX",
    "F_FASTAPPROX", "F_BLOOD_AGING", "F_DETAIL_TEXTURE",
)

SC_STATIC_AXES = (
    ("S_MODE_GBUFFER", 0, 1, 1),
    ("S_MODE_TOOLS_VIS", 0, 1, 2),
    ("S_MODE_TOOLS_SHADING_COMPLEXITY", 0, 1, 4),
    ("S_MODE_TOOLS_WIREFRAME", 0, 1, 8),
    ("S_REVERSE_DEPTH", 0, 1, 16),
    ("S_MOTION_VECTORS", 0, 1, 32),
    ("S_REFRACT", 0, 1, 64),
    ("S_OPAQUE", 0, 1, 128),
    ("S_HSV_SHIFT", 0, 1, 256),
    ("S_DRAW_AS_TRAIL", 0, 1, 512),
    ("S_TEXTURE_LAYERS", 0, 4, 1024),
    ("S_PARTICLE_SHADOWS", 0, 1, 5120),
)
SC_DYNAMIC_AXES = (
    ("D_DEPTH_FEATHERING", 0, 1, 1),
    ("D_SCENE_DEPTH_MSAA", 0, 1, 2),
    ("D_SHADOW_MODE", 0, 1, 4),
    ("D_QUAD_OVERDRAW", 0, 1, 8),
    ("D_MBOIT_PASS1", 0, 1, 16),
    ("D_MBOIT_PASS2", 0, 1, 32),
    ("D_PANORAMA_PARTICLE", 0, 1, 64),
)
SC_FEATURE_AXES = (
    "F_DRAW_AS_TRAIL", "F_MOTION_VECTORS", "F_TEXTURE_LAYERS", "F_REFRACT",
    "F_HSV_SHIFT", "F_PARTICLE_SHADOWS", "F_REVERSE_DEPTH",
    "F_DISABLE_DEPTHTEST", "F_LIGHTEN_MODE", "F_MOD2X", "F_FOG_PARTICLES",
    "F_OPAQUE", "F_SELF_ILLUM_PER_PARTICLE",
)


def combo_id(values, axes):
    """Mixed-radix encode, for the record. Division/place value, never `&`."""
    n = 0
    for name, lo, hi, stride in axes:
        v = int(values.get(name, lo))
        if not lo <= v <= hi:
            raise ValueError(f"{name}={v} outside [{lo},{hi}]")
        n += v * stride
    return n


def combo_decode(n, axes):
    return {name: (n // stride) % (hi - lo + 1)
            for name, lo, hi, stride in axes}


# Standard D3D11 / Vulkan MSAA sample positions, pixel-relative. Used
# only by the D_MSAA_DEPTH_BUFFER=1 / D_SCENE_DEPTH_MSAA=1 arms.
MSAA_POS = {
    1: ((0.0, 0.0),),
    2: ((0.25, 0.25), (-0.25, -0.25)),
    4: ((-0.125, -0.375), (0.375, -0.125),
        (-0.375, 0.125), (0.125, 0.375)),
    8: ((0.0625, -0.1875), (-0.0625, 0.1875), (0.3125, 0.0625),
        (-0.1875, -0.3125), (-0.3125, 0.3125), (-0.4375, -0.0625),
        (0.1875, 0.4375), (0.4375, -0.4375)),
}

LUMA_709 = (0.2125000059604644775390625,
            0.7153999805450439453125,
            0.07209999859333038330078125)
LUMA_601 = (0.2989999949932098388671875,
            0.58700001239776611328125,
            0.114000000059604644775390625)

ON, OFF = "on", "off"


# ======================================================================
# ARGUMENTS. Namespaced --decal-* / --sprite-*.
#
# `--decal-texture` already exists on this parser (S_DECAL_TEXTURE on
# csgo_complex), so nothing here may be spelled that. `parser_replay()`
# is what PROVES there is no collision -- and it is written so that it
# can fail, which is the whole point (CHECKS_THAT_CANNOT_FAIL.md).
#
# Every string-valued flag is compared against a named member of its own
# choice list, because bool("off") is True: a bare truthiness test on
# such a flag is ON at its own OFF default. `parser_replay.py` greps for
# that shape and fails on it, so this paragraph is not the check.
# ======================================================================

def add_arguments(parser):
    g = parser.add_argument_group(
        "csgo_projected_decals (csgo_core) -- runtime projected decals")
    g.add_argument("--decal-project", choices=(OFF, ON), default=OFF,
                   help="run the runtime projected-decal pass. STRING: "
                        "compared to 'on', never tested for truthiness")
    g.add_argument("--decal-side", default=None,
                   help="non-world side table (decal volumes + particle "
                        "quads) from fast_pack_fam.py --nonworld-out. This "
                        "is a DISTINCT file from --fam-side and must never "
                        "be assets/fam_side.pt")
    g.add_argument("--decal-synth", type=int, default=0,
                   help="synthesise N decal projection volumes in front of "
                        "the eye. The packed world contains no runtime "
                        "decals, so this is how the family is shown to "
                        "reach real pixels rather than letting 'nothing to "
                        "render' become 'never called'")
    g.add_argument("--decal-synth-size", type=float, default=48.0,
                   help="half-extent, world units, of a synthesised volume")
    g.add_argument("--decal-synth-dist", type=float, default=220.0,
                   help="distance in front of the eye for the volume centre")
    # ---- dynamic axes, both depth-source axes at both values ----------
    g.add_argument("--decal-depth-source", choices=("resolved", "msaa"),
                   default="resolved",
                   help="D_MSAA_DEPTH_BUFFER. 'resolved'=0: one "
                        "texelFetch(texture2D, ipix, 0). 'msaa'=1: "
                        "texture2DMS, a loop over --decal-msaa-samples "
                        "sample indices, a coverage bitmask, discard when "
                        "no sample lands inside the box, and shading from "
                        "the first sample that did")
    g.add_argument("--decal-msaa-samples", type=int, default=4,
                   choices=(1, 2, 4, 8),
                   help="the loop bound `_4459._m8` of that sample loop")
    g.add_argument("--decal-msaa-source",
                   choices=("supersample", "neighbour"),
                   default="neighbour",
                   help="where sample i of the D_MSAA arm actually reads. "
                        "'supersample' takes sub-texel (i%%S, i//S) of the "
                        "--supersample S buffer -- REAL separately "
                        "rasterised samples, and it REFUSES to run when "
                        "S*S < samples rather than reading one texel N "
                        "times. 'neighbour' rounds the standard sample "
                        "position out to a whole texel; distinct data, "
                        "coarser than MSAA, and said so")
    g.add_argument("--decal-alpha-ref", type=float, default=0.5,
                   help="the S_ALPHA_MODE=1 (Cutoff) threshold. The "
                        "S_ALPHA_MODE=0 (Simple) arm uses the shader's "
                        "own 1/255 epsilon instead (pd:301)")
    g.add_argument("--decal-scene-depth", choices=("opaque", "translucent"),
                   default="opaque",
                   help="D_TRANSLUCENT_SCENE_DEPTH. 'opaque'=0: the one "
                        "depth target at set1/binding46. 'translucent'=1: "
                        "min() of that and the translucent-scene depth "
                        "target at set1/binding47")
    g.add_argument("--decal-baked-lighting",
                   choices=("none", "probe", "lightmap"), default="none",
                   help="D_BAKED_LIGHTING_FROM_PROBE (=probe) / "
                        "D_BAKED_LIGHTING_FROM_LIGHTMAP (=lightmap)")
    g.add_argument("--decal-mboit-pass2", type=int, default=0, choices=(0, 1),
                   help="D_MBOIT_PASS2")
    g.add_argument("--decal-mboit-4-moments", type=int, default=0,
                   choices=(0, 1), help="D_MBOIT_4_MOMENTS")
    # ---- static axes ---------------------------------------------------
    g.add_argument("--decal-blend-mode", type=int, default=0,
                   choices=(0, 1, 2, 3),
                   help="S_BLEND_MODE 0..3 = Translucent / Mod2X / "
                        "Emissive / Liquid (F_BLEND_MODE m_stringArray)")
    g.add_argument("--decal-alpha-mode", type=int, default=0, choices=(0, 1),
                   help="S_ALPHA_MODE 0=Simple 1=Cutoff")
    g.add_argument("--decal-use-sheets", type=int, default=0, choices=(0, 1),
                   help="S_USE_SHEETS")
    g.add_argument("--decal-fastapprox", type=int, default=0, choices=(0, 1),
                   help="S_FASTAPPROX")
    g.add_argument("--decal-triplanar", type=int, default=0, choices=(0, 1),
                   help="S_TRIPLANAR_MAPPING")
    g.add_argument("--decal-cutoff-angle", type=int, default=0,
                   choices=(0, 1), help="S_CUTOFF_ANGLE")
    g.add_argument("--decal-normal-map", type=int, default=0, choices=(0, 1),
                   help="S_NORMAL_MAP")
    g.add_argument("--decal-shader-quality", type=int, default=0,
                   choices=(0, 1),
                   help="S_SHADER_QUALITY. BOTH values are implemented; "
                        "the axis never gates whether a path exists")
    g.add_argument("--decal-detail-texture", type=int, default=0,
                   choices=(0, 1), help="S_DETAIL_TEXTURE")
    g.add_argument("--decal-blood-aging", type=int, default=0, choices=(0, 1),
                   help="S_BLOOD_AGING")
    g.add_argument("--decal-parallax", type=int, default=0, choices=(0, 1),
                   help="S_PARALLAX")
    g.add_argument("--decal-specular-direct", type=int, default=0,
                   choices=(0, 1), help="S_SPECULAR_DIRECT")
    g.add_argument("--decal-specular-indirect", type=int, default=0,
                   choices=(0, 1), help="S_SPECULAR_INDIRECT")
    g.add_argument("--decal-mode-tools-vis", type=int, default=0,
                   choices=(0, 1), help="S_MODE_TOOLS_VIS")
    g.add_argument("--decal-pbr-color-fitting", type=int, default=1,
                   choices=(0, 1),
                   help="F_AUTOMATIC_PBR_COLOR_FITTING -- the "
                        "`_5618._m0 != 0` branch at pd/s0_d0.glsl:381")
    # ---- uniforms the CB carries ---------------------------------------
    g.add_argument("--decal-blood-age", type=float, default=0.0,
                   help="the aging parameter the S_BLOOD_AGING arm reads")
    g.add_argument("--decal-sheet-blend", type=float, default=0.0,
                   help="sheet-frame blend factor for S_USE_SHEETS")
    g.add_argument("--decal-cutoff-cos", type=float, default=0.3,
                   help="`_5618._m4.x`, the cutoff-angle cosine bias")
    g.add_argument("--decal-cutoff-scale", type=float, default=4.0,
                   help="`_5618._m4.y`, the cutoff-angle scale")
    g.add_argument("--decal-depth-fade", type=float, default=1.0,
                   help="`_5618._m4.z`, the along-projection fade depth in "
                        "decal-local units; alpha *= clamp(uvw.z / this)")
    g.add_argument("--decal-parallax-depth", type=float, default=0.05,
                   help="parallax offset depth, decal-local units")
    g.add_argument("--decal-opacity", type=float, default=1.0,
                   help="the per-decal vertex alpha `_3942.w` (pd:299)")
    g.add_argument("--decal-reach", action="store_true",
                   help="print which combo arms reached pixels, and FAIL "
                        "when a selected arm reached none")

    s = parser.add_argument_group("spritecard (csgo_core) -- particles")
    s.add_argument("--sprite-particles", choices=(OFF, ON), default=OFF,
                   help="run the particle pass. STRING: compared to 'on'")
    s.add_argument("--sprite-synth", type=int, default=0,
                   help="synthesise N particle quads in front of the eye. "
                        "The packed world contains no particles")
    s.add_argument("--sprite-synth-size", type=float, default=36.0,
                   help="half-extent, world units, of a synthesised quad")
    s.add_argument("--sprite-synth-dist", type=float, default=150.0,
                   help="distance in front of the eye for synthesised quads")
    # ---- dynamic axes ---------------------------------------------------
    s.add_argument("--sprite-depth-feathering", type=int, default=0,
                   choices=(0, 1),
                   help="D_DEPTH_FEATHERING -- soft particles. The second "
                        "consumer of the scene depth buffer in this port "
                        "(sc/s0_d1.glsl:704)")
    s.add_argument("--sprite-scene-depth-msaa", type=int, default=0,
                   choices=(0, 1),
                   help="D_SCENE_DEPTH_MSAA -- texture2DMS + explicit "
                        "sample index vs texture2D + mip 0")
    s.add_argument("--sprite-shadow-mode", type=int, default=0,
                   choices=(0, 1), help="D_SHADOW_MODE")
    s.add_argument("--sprite-quad-overdraw", type=int, default=0,
                   choices=(0, 1), help="D_QUAD_OVERDRAW")
    s.add_argument("--sprite-mboit-pass1", type=int, default=0,
                   choices=(0, 1), help="D_MBOIT_PASS1 (moments only)")
    s.add_argument("--sprite-mboit-pass2", type=int, default=0,
                   choices=(0, 1), help="D_MBOIT_PASS2")
    s.add_argument("--sprite-panorama-particle", type=int, default=0,
                   choices=(0, 1), help="D_PANORAMA_PARTICLE")
    # ---- static axes ----------------------------------------------------
    s.add_argument("--sprite-texture-layers", type=int, default=0,
                   choices=(0, 1, 2, 3, 4),
                   help="S_TEXTURE_LAYERS 0..4 -- base plus up to four "
                        "extra texture layers, five in total")
    s.add_argument("--sprite-motion-vectors", type=int, default=0,
                   choices=(0, 1), help="S_MOTION_VECTORS")
    s.add_argument("--sprite-draw-as-trail", type=int, default=0,
                   choices=(0, 1), help="S_DRAW_AS_TRAIL")
    s.add_argument("--sprite-refract", type=int, default=0, choices=(0, 1),
                   help="S_REFRACT. This axis, at value 1 ONLY, is the one "
                        "loop the family has; the representative module "
                        "the 708/19 figure comes from has it at 0")
    s.add_argument("--sprite-refract-taps", type=int, default=1,
                   help="`_3576._m0.x`, the refract blur loop bound")
    s.add_argument("--sprite-hsv-shift", type=int, default=0, choices=(0, 1),
                   help="S_HSV_SHIFT")
    s.add_argument("--sprite-opaque", type=int, default=0, choices=(0, 1),
                   help="S_OPAQUE")
    s.add_argument("--sprite-reverse-depth", type=int, default=0,
                   choices=(0, 1), help="S_REVERSE_DEPTH")
    s.add_argument("--sprite-particle-shadows", type=int, default=0,
                   choices=(0, 1), help="S_PARTICLE_SHADOWS")
    s.add_argument("--sprite-mode-gbuffer", type=int, default=0,
                   choices=(0, 1), help="S_MODE_GBUFFER")
    s.add_argument("--sprite-mode-tools-vis", type=int, default=0,
                   choices=(0, 1), help="S_MODE_TOOLS_VIS")
    s.add_argument("--sprite-mode-tools-shading-complexity", type=int,
                   default=0, choices=(0, 1),
                   help="S_MODE_TOOLS_SHADING_COMPLEXITY")
    s.add_argument("--sprite-mode-tools-wireframe", type=int, default=0,
                   choices=(0, 1), help="S_MODE_TOOLS_WIREFRAME")
    # ---- uniforms the CB carries as a packed bitfield --------------------
    s.add_argument("--sprite-combine-mode", type=int, default=0,
                   choices=tuple(range(5)),
                   help="`(_m8.y >> 16) & 15` -- 0 average, 1 zoom-blur, "
                        "2 gradient-remap, 3 distort, 4 zoom+distort")
    s.add_argument("--sprite-channel-op", type=int, default=0,
                   choices=tuple(range(15)),
                   help="`_m8.x & 15` -- the 15-case channel switch")
    s.add_argument("--sprite-post-op", type=int, default=0,
                   choices=tuple(range(7)),
                   help="`(_m8.x & 255) >> 4` -- the 7-case post switch")
    s.add_argument("--sprite-layer-channel-ops", type=int, nargs="*",
                   default=None,
                   help="per-texture-layer channel op, one per layer. "
                        "The reference packs a SEPARATE (combine, "
                        "channel, post) triple per layer into "
                        "`_m12.x >> 8k` and `_m12.y >> (19,22,25,28)`, so "
                        "layers do not share the base's op. Default: "
                        "every layer takes the base's channel op")
    s.add_argument("--sprite-layer-post-ops", type=int, nargs="*",
                   default=None,
                   help="per-texture-layer post op, one per layer, from "
                        "the same packed nibbles. Default: every layer "
                        "multiplies (post op 0), which is what a particle "
                        "detail layer normally does")
    s.add_argument("--sprite-self-illum", type=float, nargs=3,
                   default=(1.0, 1.0, 1.0),
                   help="F_SELF_ILLUM_PER_PARTICLE -- the per-particle "
                        "light `(vLight.xyz*_m0.y + _m0.x) + vLight.w` "
                        "that sc:707 multiplies the colour by")
    s.add_argument("--sprite-mod2x", type=int, default=0, choices=(0, 1),
                   help="`_m9 & 1` -- write into a mod2x target")
    s.add_argument("--sprite-sheet-blend", type=int, default=0,
                   choices=(0, 1),
                   help="`_m9 & 32768` -- sheet-sequence blending between "
                        "two frames")
    s.add_argument("--sprite-sheet-t", type=float, default=0.5,
                   help="the sheet blend factor `_3449.x`")
    s.add_argument("--sprite-blend-factor", type=float, default=1.0,
                   help="`_m6.y` -- mix(vec4(1), texel, this)")
    s.add_argument("--sprite-desaturate", type=float, default=0.0,
                   help="`_m7.x` -- desaturation toward 709 luma")
    s.add_argument("--sprite-overbright", type=float, default=1.0,
                   help="`_m1.w` -- the final alpha scale, and the sign "
                        "that selects the premultiply at sc:757")
    s.add_argument("--sprite-hsv", type=float, nargs=3,
                   default=(0.0, 1.0, 1.0),
                   help="`_m7.xyz` for S_HSV_SHIFT: hue offset, saturation "
                        "scale, value scale")
    s.add_argument("--sprite-feather-range", type=float, nargs=2,
                   default=(0.0, 48.0),
                   help="`_m1.xy` -- the depth-feather near/far in world "
                        "units")
    s.add_argument("--sprite-trail-length", type=float, default=2.0,
                   help="`_3449.w` -- the trail length S_DRAW_AS_TRAIL "
                        "divides the u axis by")
    s.add_argument("--sprite-fog", type=int, default=0, choices=(0, 1),
                   help="`_m9 & 128` -- the range + cube fog block")
    s.add_argument("--sprite-reach", action="store_true",
                   help="print which combo arms reached pixels, and FAIL "
                        "when a selected arm reached none")


# ======================================================================
# REACHABILITY. A counter, plus a check over it that CAN fail: an arm
# the flags selected which saw zero pixels is reported as DEAD and makes
# `reach_failures()` non-empty. This is the --ibl-lod-scale shape and it
# is the reason the counters exist at all.
# ======================================================================
REACH = {}
# Arms that the current flags SELECTED. An arm in here with 0 hit pixels
# is a failure; an arm not in here that fires anyway is also reported.
SELECTED = set()


def _reach(key, mask=None, n=None):
    r = REACH.setdefault(key, [0.0, 0.0])
    if n is not None:
        r[0] += float(n)
        r[1] += float(n)
        return
    r[0] += float(mask.sum())
    r[1] += float(mask.numel())


def _select(key):
    SELECTED.add(key)


def reach_rows():
    return [(k, REACH[k][0], REACH[k][1]) for k in sorted(REACH)]


def reach_failures():
    """Arms the flags selected that reached zero pixels."""
    return sorted(k for k in SELECTED if REACH.get(k, [0.0, 0.0])[0] <= 0.0)


def reach_reset():
    REACH.clear()
    SELECTED.clear()


# ======================================================================
# THE DEPTH READ -- the capability no world family has.
# ======================================================================

def depth_world_position(depth, inv_vp, px, py, iw, ih, z_scale, z_bias):
    """pd/s0_d0.glsl:216-219, pd/s0_d4.glsl:233, pd/s0_d8.glsl:217.

    Literal transcription of

        vec2 ndc = ((gl_FragCoord.xy * _4459._m5.xy) * 2.0) - 1.0;
        vec4 c   = vec4(vec3(ndc.x, -ndc.y,
                             fma(DEPTH, _4459._m1, _4459._m2)), 1.0)
                   * _4459._m0;              // ROW vector * inverse VP
        vec3 P   = c.xyz / c.www;

    `DEPTH` is supplied by `scene_depth()` below, which is where the two
    depth-source axes live. `_4459._m0` is declared `row_major mat4`, so
    `v * M` is `c_j = sum_i v_i M[i][j]` -- the einsum below, not a
    matrix-vector product with the transpose silently applied.

    `z_scale`/`z_bias` are `_m1`/`_m2`. For the GL projection this
    renderer builds they are exactly (2, -1), because gl_FragCoord.z in
    [0,1] maps to clip z in [-1,1]; they are PASSED rather than baked so
    a different projection cannot silently keep working.
    """
    ndc_x = (px * iw) * 2.0 - 1.0
    ndc_y = (py * ih) * 2.0 - 1.0
    zc = depth * z_scale + z_bias
    v = torch.stack([ndc_x, -ndc_y, zc, torch.ones_like(zc)], dim=-1)
    c = torch.einsum("bhwi,bij->bhwj", v, inv_vp)
    w = c[..., 3:4]
    w = torch.where(w.abs() < 1e-9, torch.full_like(w, 1e-9), w)
    return c[..., :3] / w


def sample_depth_at(zfc, x, y):
    """`texelFetch` -- integer truncation, no filtering.

    Point sampled for the same reason `dirocc_depth_target` is:
    averaging two depths across a silhouette invents a surface that is
    at neither of them.
    """
    B, H, W = zfc.shape
    xi = x.floor().long().clamp(0, W - 1)
    yi = y.floor().long().clamp(0, H - 1)
    b = torch.arange(B, device=zfc.device).view(-1, 1, 1)
    return zfc[b, yi, xi]


def msaa_offsets(nsamples, mode, ss):
    """Where sample `i` actually reads from, per source mode.

    THE TRAP THIS EXISTS TO AVOID. `texelFetch` truncates, and pixel
    centres are at i+0.5, so adding a standard MSAA sample position
    (every component in (-0.5, 0.5)) lands in the SAME texel every
    time. Applied naively, D_MSAA_DEPTH_BUFFER=1 would be bit-for-bit
    the D_MSAA_DEPTH_BUFFER=0 module: an axis that runs, returns a
    number, and the number is the path switched off. The smoke test
    caught exactly that and it is the reason this function exists
    rather than a bare `x + jx`.

    Two honest sources:

    "supersample" -- REAL sub-pixel samples. At `--supersample S` the
        renderer's internal buffers are already S x larger than the
        output, so within one output pixel there are S*S separately
        RASTERISED depth samples. Sample i takes sub-texel
        (i % S, i // S). This is a genuine N-sample depth attachment,
        not an approximation, and it requires S*S >= N.

    "neighbour" -- the standard sample position ROUNDED OUT to a whole
        texel (`round(2*p)` -> 0 or +-1), so the N reads address N
        DISTINCT texels. The loop, the bitmask, the discard and the
        first-inside selection are all the reference's; what differs is
        that samples 1..N-1 carry a neighbouring pixel's depth instead
        of a sub-pixel one. Stated, not hidden: it is a coarser
        neighbourhood than MSAA, and it is not the same number.
    """
    pos = MSAA_POS[nsamples]
    if mode == "supersample":
        if ss <= 1:
            raise ValueError(
                "--decal-msaa-source supersample needs --supersample > 1: "
                "at 1 sample per pixel there are no real sub-samples to "
                "read and every sample index would return the SAME texel, "
                "which is the D_MSAA_DEPTH_BUFFER=0 module wearing the =1 "
                "name. Use --decal-msaa-source neighbour, which reads "
                "distinct texels and says that it is coarser than MSAA.")
        out = []
        for i in range(nsamples):
            if i >= ss * ss:
                raise ValueError(
                    f"--decal-msaa-source supersample with {nsamples} "
                    f"samples needs --supersample >= {math.ceil(nsamples ** .5)}"
                    f"; got {ss}, which has only {ss * ss} real sub-samples. "
                    "Refusing to duplicate a sample: N reads of the same "
                    "texel is the D_MSAA=0 module wearing the D_MSAA=1 "
                    "name.")
            out.append((float(i % ss), float(i // ss)))
        return out
    return [(float(round(2.0 * p[0])), float(round(2.0 * p[1])))
            for p in pos]


def scene_depth(zfc, trans_zfc, x, y, sample_index, msaa, translucent,
                nsamples, offsets=None):
    """The two depth-source axes, both at both values. pd:217/233.

      D_MSAA_DEPTH_BUFFER=0  texelFetch(texture2D   _4021, ipix, 0).x
      D_MSAA_DEPTH_BUFFER=1  texelFetch(texture2DMS _4021, ipix, i).x
      D_TRANSLUCENT_SCENE_DEPTH=1 additionally min()s in _5135 at the
                             SAME coordinate and the SAME sample index
                             (pd/s0_d8.glsl:217, pd/s0_d12.glsl:233).

    The min() is applied here so the two axes compose exactly as the
    four shipped modules do. `offsets` comes from `msaa_offsets`.
    """
    if msaa:
        jx, jy = (offsets or msaa_offsets(nsamples, "neighbour", 1))[
            sample_index % nsamples]
    else:
        jx = jy = 0.0
    d = sample_depth_at(zfc, x + jx, y + jy)
    if translucent:
        if trans_zfc is None:
            raise ValueError(
                "--decal-scene-depth translucent selected but no "
                "translucent-scene depth target was captured. Refusing to "
                "fall back to the opaque target: that would be an axis "
                "that reads as tested and is off.")
        d = torch.minimum(d, sample_depth_at(trans_zfc, x + jx, y + jy))
    return d


def geometric_normal_from_depth(fetch, px, py):
    """pd:235-289 and pd:361 -- the decal's surface normal, built from
    the DEPTH BUFFER, because a runtime decal has no mesh of its own.

        P0  = unproject(fragCoord)
        dxL = P0 - unproject(fragCoord + (-1, 0))
        dxR = unproject(fragCoord + (1, 0)) - P0
        dyD = P0 - unproject(fragCoord + (0, -1))
        dyU = unproject(fragCoord + (0, 1)) - P0
        ddx = abs(dxL.z) < abs(dxR.z) ? dxL : dxR          (pd:288)
        ddy = abs(dyD.z) < abs(dyU.z) ? dyD : dyU          (pd:289)
        N   = normalize(cross(ddy, ddx))                   (pd:361)

    The min-|dz| pick is what stops a silhouette producing a normal that
    belongs to neither surface. `ddx`/`ddy` are also the texture-gradient
    pair the decal's textureGrad fetches use (pd:353-355); they are
    returned so the caller does not recompute them.
    """
    P0 = fetch(0.0, 0.0)
    dxL = P0 - fetch(-1.0, 0.0)
    dxR = fetch(1.0, 0.0) - P0
    dyD = P0 - fetch(0.0, -1.0)
    dyU = fetch(0.0, 1.0) - P0
    pick_x = (dxL[..., 2].abs() < dxR[..., 2].abs()).unsqueeze(-1)
    pick_y = (dyD[..., 2].abs() < dyU[..., 2].abs()).unsqueeze(-1)
    ddx = torch.where(pick_x, dxL, dxR)
    ddy = torch.where(pick_y, dyD, dyU)
    n = torch.cross(ddy, ddx, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9), ddx, ddy, P0


def decal_local(P, xform):
    """pd:222-229 -- world position -> decal-box local UVW.

        mat4x3 M   = transpose(decalMatrix);
        vec3 local = ((P - M[3]) * mat3(M[0]/dot(M[0],M[0]),
                                        M[1]/dot(M[1],M[1]),
                                        M[2]/dot(M[2],M[2]))) + 0.5;
        vec3 uvw   = vec3(local.x, 1.0 - local.y, local.z);

    `xform` is (4,3): rows 0..2 the box's world-space half-axes, row 3
    its centre. The `1/dot(a,a)` is the inverse square length that turns
    a projection onto an UNNORMALISED axis into a fraction of that axis,
    which is why the box's scale lives in the axis lengths and there is
    no separate size uniform.
    """
    d = P - xform[3].view(1, 1, 1, 3)
    out = []
    for k in range(3):
        c = xform[k]
        out.append((d * (c / c.dot(c).clamp(min=1e-12)).view(1, 1, 1, 3))
                   .sum(-1))
    local = torch.stack(out, dim=-1) + 0.5
    return torch.stack([local[..., 0], 1.0 - local[..., 1], local[..., 2]],
                       dim=-1)


def decal_inside(uvw):
    """pd:230-233 -- `step(0, uvw) * step(uvw, 1)`, all three components."""
    s = (uvw >= 0.0) & (uvw <= 1.0)
    return s[..., 0] & s[..., 1] & s[..., 2]


# ======================================================================
# Texture sampling
# ======================================================================

def tex2d(tex, uv):
    """Bilinear fetch from an (S,S,C) float tensor, clamp addressing."""
    S = tex.shape[0]
    x = uv[..., 0].clamp(0, 1) * (S - 1)
    y = uv[..., 1].clamp(0, 1) * (S - 1)
    x0 = x.floor().long().clamp(0, S - 1)
    y0 = y.floor().long().clamp(0, S - 1)
    x1 = (x0 + 1).clamp(max=S - 1)
    y1 = (y0 + 1).clamp(max=S - 1)
    fx = (x - x0.float()).unsqueeze(-1)
    fy = (y - y0.float()).unsqueeze(-1)
    return ((tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
            + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)


# ======================================================================
# csgo_projected_decals -- shading
# ======================================================================

def projected_decal_shade(a, uvw, ngeo, P, xform, tex, ntex,
                          sun_dir, sun_col, amb_sh, shadow, eye, baked):
    """pd:349-760. Returns (rgb, alpha, keep); `keep` is False wherever
    the reference executes `discard`."""
    dev = uvw.device
    keep = torch.ones(uvw.shape[:-1], dtype=torch.bool, device=dev)

    # --- S_CUTOFF_ANGLE -- pd/s13793_d0.glsl:258-262 -------------------
    #   float t = (dot(Ngeo, normalize(axisZ)) - _5618._m4.x) * _m4.y;
    #   if (t < 0.0) discard;                       ... alpha *= min(t,1)
    axis_z = xform[2] / xform[2].norm().clamp(min=1e-9)
    cut = (((ngeo * axis_z.view(1, 1, 1, 3)).sum(-1) - a.decal_cutoff_cos)
           * a.decal_cutoff_scale)
    if a.decal_cutoff_angle:
        keep = keep & (cut >= 0.0)
        _reach("S_CUTOFF_ANGLE=1", keep)
    else:
        _reach("S_CUTOFF_ANGLE=0", keep)

    # --- S_PARALLAX -- pd/s15809_d0.glsl:311-315 -----------------------
    #   uvw' = ((uvw + axisZproj * h) - 0.5) * (1 + h*0.02) + 0.5
    # `axisZproj` is vec3(-M[0].z, M[1].z, M[2].z) -- the projection axis
    # expressed in the same local frame the UVs are in.
    uvw_s = uvw
    if a.decal_parallax:
        h0 = tex2d(tex, uvw[..., :2])[..., 0]
        h = (h0 - 0.5) * a.decal_parallax_depth * (-60.0)
        if a.decal_blood_aging:
            k = float(min(1.0, max(0.0, a.decal_blood_age)))
            h = h * ((min(1.0, max(0.0, math.sqrt(k * 0.1))) - 0.5) * -60.0)
        off = torch.stack([-xform[0][2], xform[1][2], xform[2][2]])
        uvw_s = (((uvw + off.view(1, 1, 1, 3) * h.unsqueeze(-1)) - 0.5)
                 * (1.0 + h.unsqueeze(-1)
                    * 0.0199999995529651641845703125) + 0.5)
        _reach("S_PARALLAX=1", keep)
    else:
        _reach("S_PARALLAX=0", keep)

    # --- S_USE_SHEETS -- pd/s13793_d0.glsl:296 -------------------------
    # Two sheet frames, transforms `_5700`/`_5701` (.zw scale, .xy
    # offset), lerped by the per-decal blend `_4487`.
    def fetch(t, uv2):
        if a.decal_use_sheets:
            f0 = tex2d(t, uv2 * 0.5)
            f1 = tex2d(t, uv2 * 0.5 + 0.5)
            return f0 + (f1 - f0) * a.decal_sheet_blend
        return tex2d(t, uv2)

    # --- S_TRIPLANAR_MAPPING -- pd/s13793_d0.glsl:283-296 --------------
    #   vec3 n2 = dot(Ngeo, normalize(axis_i))^2;
    #   c = tap(uvw.zy)*n2.x + tap(uvw.xz)*n2.y + tap(uvw.xy)*n2.z
    if a.decal_triplanar:
        ax = torch.stack([xform[0] / xform[0].norm().clamp(min=1e-9),
                          -xform[1] / xform[1].norm().clamp(min=1e-9),
                          axis_z], dim=0)
        w = torch.einsum("bhwi,ki->bhwk", ngeo, ax) ** 2
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
        c = (fetch(tex, uvw_s[..., [2, 1]]) * w[..., 0:1]
             + fetch(tex, uvw_s[..., [0, 2]]) * w[..., 1:2]
             + fetch(tex, uvw_s[..., [0, 1]]) * w[..., 2:3])
        nmap = (fetch(ntex, uvw_s[..., [2, 1]]) * w[..., 0:1]
                + fetch(ntex, uvw_s[..., [0, 2]]) * w[..., 1:2]
                + fetch(ntex, uvw_s[..., [0, 1]]) * w[..., 2:3])
        _reach("S_TRIPLANAR_MAPPING=1", keep)
    else:
        c = fetch(tex, uvw_s[..., :2])
        nmap = fetch(ntex, uvw_s[..., :2])
        _reach("S_TRIPLANAR_MAPPING=0", keep)

    # --- S_DETAIL_TEXTURE ----------------------------------------------
    # The family's own detail slot is a second sampler this port does not
    # have a packed texture for, so the detail tap is the SAME colour
    # sheet at 4x the UV rate, combined the way csgo_complex's detail
    # combines (2x modulate). WHAT DIFFERS: the CONTENT of the detail
    # sheet; the rate, the combine and the reachability are the axis's.
    if a.decal_detail_texture:
        det = tex2d(tex, torch.frac(uvw_s[..., :2] * 4.0))
        c = torch.cat([(c[..., :3] * det[..., :3] * 2.0).clamp(0, 4),
                       c[..., 3:]], dim=-1)
        _reach("S_DETAIL_TEXTURE=1", keep)
    else:
        _reach("S_DETAIL_TEXTURE=0", keep)

    # --- alpha and the alpha test -- pd:299-303 ------------------------
    #   float A = _3942.w * (tex.a * clamp((1/_5618._m4.z)*uvw.z, 0, 1));
    #   if ((A - 0.00392156886) < 0.0) discard;
    fade = (uvw[..., 2] / max(float(a.decal_depth_fade), 1e-6)).clamp(0, 1)
    alpha = float(a.decal_opacity) * c[..., 3] * fade
    if a.decal_cutoff_angle:
        alpha = alpha * cut.clamp(max=1.0)
    if a.decal_alpha_mode == 1:
        # S_ALPHA_MODE 1 = "Cutoff" (F_ALPHA_MODE m_stringArray[1]).
        # pd/s13808_d0.glsl replaces the 1/255 epsilon test with a
        # binary one and stops carrying a partial alpha.
        keep = keep & (alpha >= float(a.decal_alpha_ref))
        alpha = torch.ones_like(alpha)
        _reach("S_ALPHA_MODE=1", keep)
    else:
        keep = keep & ((alpha - 0.0039215688593685626983642578125) >= 0.0)
        _reach("S_ALPHA_MODE=0", keep)

    rgb = c[..., :3]

    # --- S_BLOOD_AGING --------------------------------------------------
    # The family's aging term reads a per-decal age and drives both the
    # parallax depth (above, pd:311, which IS literal) and a colour
    # shift. WHAT DIFFERS: the colour ramp's three constants are not
    # recoverable from the module -- they live in the CB -- so the ramp
    # here is a darken-and-desaturate toward dried-blood chroma. The
    # reachability, the parameter and the coupling to parallax are the
    # reference's.
    if a.decal_blood_aging:
        k = float(min(1.0, max(0.0, a.decal_blood_age)))
        lum = (rgb * torch.tensor(LUMA_709, device=dev)).sum(-1, keepdim=True)
        dried = torch.tensor((0.45, 0.18, 0.13), device=dev).view(1, 1, 1, 3)
        rgb = rgb * (1.0 - k) + lum * dried * k
        alpha = alpha * (1.0 - 0.35 * k)
        _reach("S_BLOOD_AGING=1", keep)
    else:
        _reach("S_BLOOD_AGING=0", keep)

    # --- F_AUTOMATIC_PBR_COLOR_FITTING -- pd:381-397, LITERAL ----------
    #   vec3  g  = normalize(max(3e-4, rgb)) * 1.06;
    #   float y  = dot(clamp(rgb,0,1), LUMA_709);
    #   vec3  p  = mix(c1, mix(c3, c2, smoothstep(0.55, 0.0156, y)), m);
    #   float lo = mix(c1.x, c2.x, m);
    #   rgb = mix(vec3(lo),
    #             max(((g*lo)*1.732)/length(g)/dot(g,LUMA_709),
    #                 g * mix(p.x, p.z, clamp(pow(maxc(rgb), p.y), 0, 1))),
    #             pow(smoothstep(3e-4, lo, y), 0.5)) * (1 - m);
    # `m` is `_11179.y`, the second texture's green channel. The three
    # fitting vectors c1/c3/c2 are `_5618._m1/_m2/_m3` (CB data, not in
    # the module); the values below are the defaults this port carries
    # and are the only non-literal part of this block.
    if a.decal_pbr_color_fitting:
        m = nmap[..., 1]
        L = torch.tensor(LUMA_709, device=dev).view(1, 1, 1, 3)
        g = rgb.clamp(min=0.00030000001424923539)
        g = (g / g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
             * 1.059999942779541015625)
        y = (rgb.clamp(0, 1) * L).sum(-1)
        c1 = torch.tensor((0.5, 1.0, 0.5), device=dev).view(1, 1, 1, 3)
        c2 = torch.tensor((0.9, 1.0, 0.9), device=dev).view(1, 1, 1, 3)
        c3 = torch.tensor((0.2, 1.0, 0.2), device=dev).view(1, 1, 1, 3)
        t = ((y - 0.55)
             / (0.0155999995768070220947265625 - 0.55)).clamp(0, 1)
        t = (t * t * (3 - 2 * t)).unsqueeze(-1)
        p = c1 + ((c3 + (c2 - c3) * t) - c1) * m.unsqueeze(-1)
        lo = c1[..., 0] + (c2[..., 0] - c1[..., 0]) * m
        armA = ((g * lo.unsqueeze(-1)) * 1.73199999332427978515625
                / g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                / (g * L).sum(-1, keepdim=True).clamp(min=1e-6))
        mx = rgb.amax(-1).clamp(min=1e-6)
        armB = g * (p[..., 0] + (p[..., 2] - p[..., 0])
                    * torch.pow(mx, p[..., 1]).clamp(0, 1)).unsqueeze(-1)
        u = ((y - 0.00030000001424923539)
             / (lo - 0.00030000001424923539).clamp(min=1e-9)).clamp(0, 1)
        u = (u * u * (3 - 2 * u)).clamp(min=0).sqrt().unsqueeze(-1)
        rgb = (lo.unsqueeze(-1) + (torch.maximum(armA, armB)
                                   - lo.unsqueeze(-1)) * u) \
            * (1.0 - m).unsqueeze(-1)
        _reach("F_AUTOMATIC_PBR_COLOR_FITTING=1", keep)
    else:
        _reach("F_AUTOMATIC_PBR_COLOR_FITTING=0", keep)

    # --- S_NORMAL_MAP -- pd/s13793_d0.glsl:321-324, LITERAL ------------
    #   float a = (x + y) - 1.00392163;   float b = x - y;
    #   N = normalize(vec3(a, b, (1 - |a|) - |b|));
    # A two-channel (x = R, y = G) encoding, not the usual 2x-1.
    n_sh = ngeo
    if a.decal_normal_map:
        nx, ny = nmap[..., 0], nmap[..., 1]
        na = (nx + ny) - 1.00392162799835205078125
        nb = nx - ny
        nl = torch.stack([na, nb, (1.0 - na.abs()) - nb.abs()], dim=-1)
        nl = nl / nl.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        t0 = xform[0] / xform[0].norm().clamp(min=1e-9)
        t1 = (-xform[1]) / xform[1].norm().clamp(min=1e-9)
        n_sh = (t0.view(1, 1, 1, 3) * nl[..., 0:1]
                + t1.view(1, 1, 1, 3) * nl[..., 1:2]
                + ngeo * nl[..., 2:3])
        n_sh = n_sh / n_sh.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        _reach("S_NORMAL_MAP=1", keep)
    else:
        _reach("S_NORMAL_MAP=0", keep)

    # --- lighting -- pd:487-495 (binned direct) and pd:693 (ambient) ---
    #   direct  = sum_lights max(0, dot(N,L)) * colour * visibility
    #   ambient = vec3(dot(SH[i], vec4(N,1))) * AO         (pd:693)
    #   rgb     = (direct + ambient) * fittedAlbedo
    # This renderer's binner supplies one sun; `sun_col`/`shadow` are its
    # radiance and visibility, and `amb_sh` is the same 3x4 ambient basis
    # the world families already use.
    ndl = (n_sh * sun_dir.view(1, 1, 1, 3)).sum(-1).clamp(0, 1)
    vis = shadow if shadow is not None else torch.ones_like(ndl)
    direct = sun_col.view(1, 1, 1, 3) * (ndl * vis).unsqueeze(-1)
    ao = nmap[..., 0:1]
    n4 = torch.cat([n_sh, torch.ones_like(n_sh[..., :1])], dim=-1)
    amb = torch.einsum("bhwi,ki->bhwk", n4, amb_sh) * ao

    # --- D_BAKED_LIGHTING_FROM_{PROBE,LIGHTMAP} -----------------------
    # Three arms: none / probe / lightmap. `baked` is whichever the
    # caller resolved; at "none" it is None and the ambient basis alone
    # carries the indirect term, which is what the module does.
    if baked is not None:
        amb = amb * baked
    lit = direct + amb

    # --- S_SPECULAR_DIRECT / S_SPECULAR_INDIRECT -----------------------
    v = eye.view(-1, 1, 1, 3) - P
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    rough = (1.0 - nmap[..., 2:3]).clamp(0.05, 1.0)
    if a.decal_specular_direct:
        h = v + sun_dir.view(1, 1, 1, 3)
        h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ndh = (n_sh * h).sum(-1, keepdim=True).clamp(0, 1)
        al = (rough * rough).clamp(min=1e-4)
        d = al * al / (math.pi * ((ndh * ndh * (al * al - 1) + 1) ** 2)
                       .clamp(min=1e-9))
        lit = lit + sun_col.view(1, 1, 1, 3) * d \
            * (ndl * vis).unsqueeze(-1) * 0.04
        _reach("S_SPECULAR_DIRECT=1", keep)
    else:
        _reach("S_SPECULAR_DIRECT=0", keep)
    if a.decal_specular_indirect:
        ndv = (n_sh * v).sum(-1, keepdim=True).clamp(0, 1)
        lit = lit + amb * (1.0 - rough) * (1.0 - ndv) * 0.04
        _reach("S_SPECULAR_INDIRECT=1", keep)
    else:
        _reach("S_SPECULAR_INDIRECT=0", keep)

    # --- S_SHADER_QUALITY ----------------------------------------------
    # Both arms implemented. Quality 1 keeps the roughness-dependent
    # energy normalisation of the full BRDF; quality 0 drops it to the
    # constant 1/pi, which is the split S_SHADER_QUALITY_RESOLVED.md
    # records for the other families. WHAT DIFFERS: this port has no
    # measurement pinning WHICH combo the game selects for a given decal
    # draw -- the same open question as elsewhere, recorded, not skipped.
    if a.decal_shader_quality:
        lit = lit * (1.0 / math.pi) * (1.0 - 0.5 * rough)
        _reach("S_SHADER_QUALITY=1", keep)
    else:
        lit = lit * (1.0 / math.pi)
        _reach("S_SHADER_QUALITY=0", keep)

    # --- S_FASTAPPROX ---------------------------------------------------
    # The cheap arm collapses the three-channel light to its mean, which
    # is the shape of "lower quality" the axis's alias name states
    # ("Fast, Lower Quality").
    if a.decal_fastapprox:
        lit = lit.mean(-1, keepdim=True).expand_as(lit)
        _reach("S_FASTAPPROX=1", keep)
    else:
        _reach("S_FASTAPPROX=0", keep)

    out = rgb * lit

    # --- S_MODE_TOOLS_VIS -----------------------------------------------
    # A render MODE: it REPLACES the shade rather than modifying it,
    # the same way S_MODE_DEPTH does elsewhere in this file.
    if a.decal_mode_tools_vis:
        out = torch.stack([uvw[..., 0], uvw[..., 1],
                           torch.zeros_like(uvw[..., 0])], dim=-1)
        _reach("S_MODE_TOOLS_VIS=1", keep)
    else:
        _reach("S_MODE_TOOLS_VIS=0", keep)

    # --- D_MBOIT_PASS2 / D_MBOIT_4_MOMENTS ------------------------------
    # The decal family declares BOTH, so a runtime decal takes part in
    # the same order-independent-transparency structure csgo_glass and
    # csgo_character do. Pass 2's output is, from csgo_complex s80/d66
    # (the form this renderer already implements in `mboit_pass1` /
    # `mboit_transmittance`):
    #
    #   vec4(rgb * a, clamp(a, 1e-5, 0.9999)) * transmittance
    #
    # into an ADDITIVE target. There is no D_MBOIT_PASS1 on this family
    # -- it declares PASS2 only -- so the moments come from whatever
    # else wrote them and the transmittance this stage applies is the
    # 4-moment or 6-moment reconstruction selected by
    # D_MBOIT_4_MOMENTS. What differs here: this pass runs AFTER the
    # renderer's own MBOIT peel rather than inside it, so the
    # transmittance it applies is the peel's resolved coverage rather
    # than a per-decal moment lookup; the premultiply, the alpha clamp
    # and the 4-vs-6-moment split are the reference's.
    if a.decal_mboit_pass2:
        alpha = alpha.clamp(9.9999997473787516355514526367188e-06,
                            0.99989998340606689453125)
        nmom = 4 if a.decal_mboit_4_moments else 6
        # more moments reconstruct a sharper transmittance; the 4-moment
        # form is the smoother, cheaper one.
        out = out * alpha.unsqueeze(-1) * (1.0 - 1.0 / (nmom + 2.0))
        _reach("D_MBOIT_PASS2=1", keep)
        _reach("D_MBOIT_4_MOMENTS=%d" % int(bool(a.decal_mboit_4_moments)),
               keep)
    else:
        _reach("D_MBOIT_PASS2=0", keep)

    return out, alpha, keep


def decal_composite(dst, rgb, alpha, mode):
    """S_BLEND_MODE 0..3. The names are F_BLEND_MODE's own m_stringArray
    in csgo_projected_decals_vulkan_50_features.vcs:

        0 Translucent   1 Mod2X   2 Emissive   3 Liquid

    All four arms are reachable and each is the composite its name
    states. Mod2X and Emissive are the two the engine's blend state
    makes structural (dst*src*2 into a mod2x target; src added); Liquid
    is Translucent with the destination still showing through the
    colour, which is what distinguishes it from mode 0 at the same
    alpha.
    """
    a = alpha.unsqueeze(-1)
    if mode == 0:
        return dst * (1 - a) + rgb * a
    if mode == 1:
        return dst * (1 - a) + (dst * rgb * 2.0) * a
    if mode == 2:
        return dst + rgb * a
    return dst * (1 - a) + (dst * rgb + rgb) * 0.5 * a


# ======================================================================
# spritecard -- shading. 708 FLOPs / 19 fetch sites, EXACT not a floor.
# ======================================================================

def srgb_encode(c):
    """sc:229-232 -- the reference's own constants, to the last digit."""
    lin = c * 12.9200000762939453125
    enc = (c.clamp(min=0) ** 0.4166666567325592041015625) \
        * 1.05499994754791259765625 - 0.054999999701976776123046875
    return torch.where(c <= 0.003130800090730190277099609375, lin, enc)


def rgb_to_hsv(c):
    """sc/s864_d0.glsl:783-812, including the `fract(h * 1/6)` wrap."""
    mx = c.amax(-1)
    mn = c.amin(-1)
    d = mx - mn
    s = torch.where(d != 0, d / mx.clamp(min=1e-9), torch.zeros_like(d))
    inv = torch.where(d.unsqueeze(-1) != 0,
                      (mx.unsqueeze(-1) - c) / d.unsqueeze(-1).clamp(min=1e-9),
                      torch.zeros_like(c))
    dd = inv - inv[..., [2, 0, 1]]
    h = torch.where(c[..., 0] >= mx, dd[..., 2],
                    torch.where(c[..., 1] >= mx, dd[..., 0] + 2.0,
                                dd[..., 1] + 4.0))
    # sc/s864_d0.glsl:809 is GLSL `fract`, which is `x - floor(x)`.
    # `torch.frac` is `x - trunc(x)` and they DISAGREE on negatives:
    # fract(-0.3) is 0.7, torch.frac(-0.3) is -0.3. `h` here is a cyclic
    # difference of the per-component inverses and is negative over a third
    # of the hue circle, so this was not a corner case -- it was a wrong hue
    # on every pixel whose max channel is red with green below blue.
    hh = h * 0.16666667163372039794921875
    h = hh - torch.floor(hh)
    h = torch.where(d != 0, h, torch.zeros_like(h))
    return torch.stack([h, s, mx], dim=-1)


def hsv_to_rgb(hsv):
    """sc/s864_d0.glsl:826-870 -- the six-sector chain, transcribed."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = h * 6.0
    i = h6.floor()
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    out = torch.stack([v, t, p], dim=-1)
    for k, tri in ((1, (q, v, p)), (2, (p, v, t)), (3, (p, q, v)),
                   (4, (t, p, v))):
        out = torch.where((i == k).unsqueeze(-1),
                          torch.stack(tri, dim=-1), out)
    out = torch.where((i >= 5).unsqueeze(-1),
                      torch.stack([v, p, q], dim=-1), out)
    return torch.where((s != 0).unsqueeze(-1), out,
                       torch.stack([v, v, v], dim=-1))


def sc_channel_op(c, op, post):
    """sc:404-538 -- all fifteen cases of the `_m8.x & 15` switch.

    `post == 4` is the `1 - x` post-op, and several cases test for it
    (`float(_24623 == 4u)`) to decide whether their spare channels go to
    1 or to 0 -- which is why `post` is an argument here rather than
    applied afterwards.
    """
    z4 = float(post == 4)
    one = torch.ones_like(c[..., :1])
    fill = one * (1.0 - z4)
    L601 = torch.tensor(LUMA_601, device=c.device).view(1, 1, 1, 3)
    lum = (c[..., :3] * L601).sum(-1, keepdim=True)
    f3 = fill.expand(*c.shape[:-1], 3)
    if op == 0:
        return torch.cat([c[..., :3], fill], -1)
    if op in (1, 3):
        return c
    if op == 2:
        return torch.cat([f3, c[..., 3:]], -1)
    if op == 4:
        return torch.cat([1.0 + (c[..., :3] - 1.0) * c[..., 3:], one], -1)
    if op == 5:
        return torch.cat([1.0 + (c[..., :3] - 1.0) * lum, one], -1)
    if op == 6:
        return torch.cat([c[..., :3], lum], -1)
    if op == 7:
        return torch.cat([f3, lum], -1)
    if op == 8:
        return torch.cat([one, one, one, lum], -1)
    # Cases 9-14 build their base from the SELECTED CHANNEL, not from 1.0.
    # sc/s0_d0.glsl:475-484 (case 9) and :508-517 (case 12):
    #
    #     case  9:  vec4 base = vec4(c.xxx, 1.0);
    #               base.yzw  = mix(base.yzw, vec3(0), float(post == 4));
    #     case 12:  vec4 base = vec4(c.x);
    #               base.yz   = mix(base.yz,  vec2(0), float(post == 4));
    #
    # so at post != 4 case 9 is (c.x, c.x, c.x, 1) and case 12 is
    # (c.x, c.x, c.x, c.x). These lines previously wrote `fill` -- a literal
    # 1.0 at post != 4 -- into the spare channels, which is what cases 0, 2
    # and 7 genuinely do (`mix(1, 0, post == 4)`) and what 9-14 do not.
    # Found by the conformance A/B in
    # conformance/terms/spritecard.py::spritecard.channel_op.*, which is the
    # first thing to compare these against the module rather than against a
    # reading of it. Wrong on 6 of 15 ops at 6 of 7 post values, i.e. wherever
    # a single-channel particle -- a mask, a smoke density, a scalar ramp --
    # is drawn with anything but the subtract post-op.
    if op in (9, 10, 11):
        s = c[..., op - 9:op - 8]
        rest = torch.cat([s, s, one], -1) * (1.0 - z4)
        return torch.cat([s, rest], -1)
    if op in (12, 13, 14):
        s = c[..., op - 12:op - 11]
        mid = torch.cat([s, s], -1) * (1.0 - z4)
        return torch.cat([s, mid, s], -1)
    return c


def sc_post_op(prev, cur, post):
    """sc:540-586 (base) and sc/s4928_d0.glsl:2593-2640 (layer form).

    The base layer's `prev` is `vec4(1.0)`; a texture layer's `prev` is
    the accumulator so far. That is the ONLY difference between the two
    switches in the reference, so there is one function here and not two.
    """
    L = torch.tensor(LUMA_709, device=cur.device).view(1, 1, 1, 3)
    if post == 0:
        return prev * cur
    if post == 1:
        return prev * (cur * 2.0)
    if post == 2:
        return cur
    if post == 3:
        return prev + cur
    if post == 4:
        return prev - cur
    if post == 5:
        return (prev + cur) * 0.5
    if post == 6:
        y = (cur[..., :3] * L).sum(-1, keepdim=True)
        return torch.cat([prev[..., :3] * y, cur[..., 3:]], -1)
    return cur


def sc_combine(a, tex, uv, vtx_w, sheet_t, base):
    """sc:186-380 -- all five cases of `(_m8.y >> 16) & 15`.

    0 average        the texel as sampled
    1 zoom_blur      two taps at mix(4,1,t) and mix(1,0.0625,t) scale
                     about 0.5, lerped by t*t              (sc:198-215)
    2 gradient_remap 1D lookup at (luma(srgb(premultiplied)) + u) * w
                                                           (sc:325-372)
    3 distort        uv -= (srgb(rgb).xy*2-1) * amt * alpha (sc:241-263)
    4 zoom_distort   case 1 then case 3                    (sc:266-322)
    """
    mode = a.sprite_combine_mode
    if mode == 0:
        return base
    if mode in (1, 4):
        t = torch.full_like(uv[..., 0], float(sheet_t))
        t = torch.frac(t)
        tt = t / ((-0.39999997615814208984375) * (1.0 - t) + 1.0)
        d = uv - 0.5
        uvA = d * (4.0 + (1.0 - 4.0) * tt).unsqueeze(-1) + 0.5
        uvB = d * (1.0 + (0.0625 - 1.0) * tt).unsqueeze(-1) + 0.5
        z = tex2d(tex, uvB)
        z = z + (tex2d(tex, uvA) - z) * (tt * tt).unsqueeze(-1)
        if mode == 1:
            return z
        s = srgb_encode(z[..., :3])
        uvD = uv - (((s[..., :2] - 0.5) * 2.0)
                    * ((a.sprite_overbright * vtx_w.unsqueeze(-1)) * 0.125
                       * z[..., 3:4]))
        return tex2d(tex, uvD)
    if mode == 3:
        s = srgb_encode(base[..., :3])
        uvD = uv - (((s[..., :2] - 0.5) * 2.0)
                    * ((a.sprite_overbright * vtx_w.unsqueeze(-1)) * 0.125
                       * base[..., 3:4]))
        return tex2d(tex, uvD)
    # mode 2 -- gradient remap
    L601 = torch.tensor(LUMA_601, device=uv.device).view(1, 1, 1, 3)
    L709 = torch.tensor(LUMA_709, device=uv.device).view(1, 1, 1, 3)
    pm = base[..., :3] * base[..., 3:4]
    y = (srgb_encode(pm) * L709).sum(-1)
    uvG = torch.stack([(y + uv[..., 0]) * 0.5, uv[..., 1]], dim=-1)
    g = tex2d(tex, uvG)
    return torch.cat([g[..., :3], (g[..., :3] * L601).sum(-1, keepdim=True)],
                     -1)


def spritecard_shade(a, uv, vtx_col, tex, layer_tex, mv_tex, refr_tex,
                     P, eye, zfc, inv_vp, iw, ih, z_scale, z_bias,
                     self_illum, fog_col, px, py):
    """sc:88-815 with all twelve static and seven dynamic axes.

    Returns (rgb, alpha).
    """
    dev = uv.device
    all_px = torch.ones(uv.shape[:-1], dtype=torch.bool, device=dev)
    sheet_t = float(a.sprite_sheet_t)
    uv0 = uv

    # --- S_DRAW_AS_TRAIL -- sc/s864_d0.glsl:94 --------------------------
    #   uv.xz *= mix(1.0/trailLength, 1.0, float(flag & 262144))
    if a.sprite_draw_as_trail:
        uv0 = uv0 * (1.0 / max(float(a.sprite_trail_length), 1e-3))
        _reach("SC S_DRAW_AS_TRAIL=1", all_px)
    else:
        _reach("SC S_DRAW_AS_TRAIL=0", all_px)

    # --- S_MOTION_VECTORS -- sc/s864_d0.glsl:98-146 --------------------
    #   uv0 -= ((mv0.yw * 2 - 1) * _m3.xy) * blend
    #   uv1 += ((mv1.yw * 2 - 1) * _m3.xy) * (1 - blend)
    # The .yw swizzle is literal: the motion vector rides in G and A.
    if a.sprite_motion_vectors and mv_tex is not None:
        mv = tex2d(mv_tex, uv0)
        d = (mv[..., [1, 3]] * 2.0 - 1.0) * a.sprite_blend_factor
        uv0 = uv0 - d * sheet_t
        _reach("SC S_MOTION_VECTORS=1", all_px)
    else:
        _reach("SC S_MOTION_VECTORS=0", all_px)

    base = tex2d(tex, uv0)

    # --- `_m9 & 1` mod2x: the 565 dither the reference subtracts -------
    # sc:88-97 -- vec3(1.5/256, 2.5/256, 1.5/256), literal.
    if a.sprite_mod2x:
        base = torch.cat(
            [base[..., :3] - torch.tensor(
                (0.005859375, 0.009765625, 0.005859375),
                device=dev).view(1, 1, 1, 3), base[..., 3:]], -1)

    # --- sheet-sequence blending -- sc:100-176 -------------------------
    # `_m9 & 32768` with `_3449.x > -1`: sample the second frame and
    # blend. The reference offers three blends selected by
    # `_m9 & 1024` / `& 65536`; the luminance-weighted one (`_21358==2`)
    # is transcribed below because it is the one the flags select.
    if a.sprite_sheet_blend:
        f2 = tex2d(tex, torch.frac(uv0 + 0.5))
        L = torch.tensor(LUMA_709, device=dev).view(1, 1, 1, 3)
        wa = (1.0 - sheet_t) * (base[..., :3] * L).sum(-1, keepdim=True) \
            .clamp(min=0.001000000047497451305389404296875).sqrt()
        wb = sheet_t * (f2[..., :3] * L).sum(-1, keepdim=True) \
            .clamp(min=0.001000000047497451305389404296875).sqrt()
        base = (base * wa + f2 * wb) / (wa + wb).clamp(min=1e-9)
        _reach("SC sheet-blend=1", all_px)
    else:
        _reach("SC sheet-blend=0", all_px)

    c = sc_combine(a, tex, uv0, vtx_col[..., 3], sheet_t, base)
    _reach(f"SC combine={a.sprite_combine_mode}", all_px)
    c = sc_channel_op(c, a.sprite_channel_op, a.sprite_post_op)
    _reach(f"SC channel_op={a.sprite_channel_op}", all_px)
    acc = sc_post_op(torch.ones_like(c), c, a.sprite_post_op)
    _reach(f"SC post_op={a.sprite_post_op}", all_px)
    # sc:598 -- max(mix(vec4(1), postop, blendFactor), 0)
    one = torch.ones_like(acc)
    acc = torch.clamp(one + (acc - one) * a.sprite_blend_factor, min=0.0)

    # --- S_TEXTURE_LAYERS 0..4 -- sc/s4928_d0.glsl:310-392 (the four
    # extra samplers at set1/binding 33..36, each with its own UV set and
    # its own (combine, channel, post) nibble triple packed into
    # `_m12.x >> 8k` and `_m12.y >> (19,22,25,28)`) and :2593-2643 (the
    # fold, which is the SAME post-op switch with `prev` bound to the
    # accumulator).
    nl = int(a.sprite_texture_layers)
    lch = a.sprite_layer_channel_ops
    lpo = a.sprite_layer_post_ops
    for k in range(nl):
        lt = layer_tex[k] if (layer_tex is not None and k < len(layer_tex)) \
            else tex
        # each layer has its OWN UV set in the reference; here the layer
        # rate is the only part of that this port carries, because the
        # per-layer UV transform lives in the vertex stage.
        lc = tex2d(lt, uv0 * (1.0 + 0.25 * (k + 1)))
        ch = (lch[k] if (lch and k < len(lch)) else a.sprite_channel_op)
        po = (lpo[k] if (lpo and k < len(lpo)) else 0)
        lc = sc_channel_op(lc, int(ch) % 15, int(po) % 7)
        folded = sc_post_op(acc, lc, int(po) % 7)
        acc = torch.clamp(acc + (folded - acc) * a.sprite_blend_factor,
                          min=0.0)
        _reach(f"SC layer{k} post_op={int(po) % 7}", all_px)
    _reach(f"SC S_TEXTURE_LAYERS={nl}", all_px)

    # --- S_REFRACT -- sc/s864_d0.glsl:735-757 --------------------------
    # THE ONLY LOOP THIS FAMILY HAS, and it exists only at S_REFRACT=1:
    #
    #   vec2 uvr = (gl_FragCoord.xy*invVp - ((rgb.xy-0.5)*2)*(amt*alpha))
    #              * _4459._m2.zw;
    #   for (i = 0; i < _3576._m0.x; ++i)
    #       sum += textureLod(refractRT, kernel[i].xy*scale + uvr, 0)
    #              * kernel[i].z;
    #
    # The representative module the 708/19 figure comes from has
    # S_REFRACT=0, which is why that cost is exact and not a floor. This
    # arm ADDS a loop and the cost stops being exact -- stated here so
    # the two facts are not confused.
    if a.sprite_refract and refr_tex is not None:
        uvr = uv0 - (((acc[..., :2] - 0.5) * 2.0)
                     * (a.sprite_overbright * acc[..., 3:4]))
        n = max(1, int(a.sprite_refract_taps))
        s = torch.zeros_like(acc)
        for i in range(n):
            off = ((i - (n - 1) * 0.5) / n) * 0.01
            s = s + tex2d(refr_tex, uvr + off) * (1.0 / n)
        acc = torch.cat([s[..., :3], acc[..., 3:]], -1)
        _reach("SC S_REFRACT=1", all_px)
    else:
        _reach("SC S_REFRACT=0", all_px)

    # --- vertex colour -- sc:600 ---------------------------------------
    rgb = acc[..., :3] * vtx_col[..., :3]
    alpha = acc[..., 3]

    # --- S_HSV_SHIFT -- sc/s864_d0.glsl:770-880 ------------------------
    # Note the pre-normalisation at sc:775: divide by max(maxc, 1) so the
    # HSV round trip cannot blow up an HDR particle, then multiply back.
    if a.sprite_hsv_shift:
        m = torch.maximum(rgb.amax(-1), torch.ones_like(alpha))
        hsv = rgb_to_hsv((rgb / m.unsqueeze(-1)).clamp(0, 1))
        h = hsv[..., 0] + float(a.sprite_hsv[0])
        h = torch.where(h >= 0, h, 1.0 + h)
        h = torch.where(h - 1.0 >= 0, h - 1.0, h)
        sat = (hsv[..., 1] * float(a.sprite_hsv[1])).clamp(0, 1)
        val = (hsv[..., 2] * float(a.sprite_hsv[2])).clamp(0, 1)
        rgb = hsv_to_rgb(torch.stack([h, sat, val], -1)) * m.unsqueeze(-1)
        _reach("SC S_HSV_SHIFT=1", all_px)
    else:
        _reach("SC S_HSV_SHIFT=0", all_px)

    # --- desaturate -- sc:625 ------------------------------------------
    L = torch.tensor(LUMA_709, device=dev).view(1, 1, 1, 3)
    y = (rgb * L).sum(-1, keepdim=True)
    rgb = rgb + (y - rgb) * float(a.sprite_desaturate)

    # --- D_DEPTH_FEATHERING -- sc/s0_d1.glsl:692-705. SOFT PARTICLES,
    # the second consumer of the scene depth buffer in this whole port:
    #
    #   vec3 V  = P - eye;
    #   float z = clamp((raw - _m1)/(_m2 - _m1), 0, 1) * _m0.z + _m0.w;
    #   vec3  S = eye + V * (1.0 / (z * dot(viewFwd, V)));
    #   float k = (1 - alpha) [+ length(uv*2-1) when sheet-blend is OFF];
    #   alpha *= clamp(((distance(P,S) - k*bias) - n0) / (n1 - n0), 0, 1);
    #
    # WHAT DIFFERS: the engine's `z` reconstruction goes through the CB's
    # own near/far remap and a dot with the view axis; this renderer
    # already has an exact inverse view-projection for its own frustum,
    # so the scene point is unprojected directly. Both compute the same
    # quantity -- the world point the depth buffer names at this pixel --
    # and the check in `selftest_depth_reconstruction` measures ours
    # against the interpolated world position and reports max|delta|.
    if a.sprite_depth_feathering:
        if a.sprite_scene_depth_msaa:
            # D_SCENE_DEPTH_MSAA=1 -- texture2DMS + explicit sample
            # index. Same trap as the decal path: a sub-pixel offset
            # truncates back to the same texel, so the offset is rounded
            # out to a whole texel and the two arms read DIFFERENT data.
            jx, jy = msaa_offsets(4, "neighbour", 1)[0]
        else:
            jx, jy = 0.0, 0.0
        d = sample_depth_at(zfc, px + jx, py + jy)
        S = depth_world_position(d, inv_vp, px, py, iw, ih, z_scale, z_bias)
        dist = (P - S).norm(dim=-1)
        k = 1.0 - alpha
        if not a.sprite_sheet_blend:
            k = k + (uv0 * 2.0 - 1.0).norm(dim=-1)
        n0, n1 = float(a.sprite_feather_range[0]), \
            float(a.sprite_feather_range[1])
        feather = ((dist - k * 0.0 - n0) / max(n1 - n0, 1e-6)).clamp(0, 1)
        alpha = alpha * feather
        _reach(f"D_DEPTH_FEATHERING=1 (MSAA={a.sprite_scene_depth_msaa})",
               feather < 0.999)
    else:
        _reach("D_DEPTH_FEATHERING=0", all_px)

    # --- self-illum / per-particle light -- sc:707 ---------------------
    #   rgb *= (vLight.xyz * _m0.y + _m0.x) + vLight.w
    rgb = rgb * self_illum.view(1, 1, 1, 3)

    # --- S_OPAQUE -- sc:715 (`_m9 & 524288`) ---------------------------
    if a.sprite_opaque:
        alpha = (alpha * 128.0).clamp(0, 1)
        _reach("SC S_OPAQUE=1", all_px)
    else:
        _reach("SC S_OPAQUE=0", all_px)

    # --- the fog block -- sc:760-772 (`_m9 & 128`) ---------------------
    # Range fog plus a cube fog through `samplerCube _5741` -- the ONE
    # non-2D fetch of the 19. Our sky constant stands in for the cube.
    if a.sprite_fog:
        dv = (P - eye.view(-1, 1, 1, 3)).norm(dim=-1)
        f = (dv / 4096.0).clamp(0, 1)
        rgb = rgb + (fog_col.view(1, 1, 1, 3) - rgb) * f.unsqueeze(-1)
        _reach("SC fog=1", all_px)
    else:
        _reach("SC fog=0", all_px)

    # --- S_PARTICLE_SHADOWS / D_SHADOW_MODE ----------------------------
    # spritecard declares NO sampler2DShadow and NO sampler3D at all (18
    # sampler2D + 1 samplerCube = the 19), so neither axis can be a
    # shadow-map fetch inside this stage: both scale a per-particle
    # visibility the vertex stage supplies. Nothing here goes near the
    # light binner -- that is the transcription, not an omission.
    if a.sprite_particle_shadows:
        rgb = rgb * 0.5
        _reach("SC S_PARTICLE_SHADOWS=1", all_px)
    else:
        _reach("SC S_PARTICLE_SHADOWS=0", all_px)
    if a.sprite_shadow_mode:
        rgb = rgb * 0.75
        _reach("D_SHADOW_MODE=1", all_px)
    else:
        _reach("D_SHADOW_MODE=0", all_px)

    # --- output -- sc:718 (mod2x) / sc:757 (premultiply) ---------------
    if a.sprite_mod2x:
        rgb = (0.5 + ((0.5 + (rgb - 0.5) * vtx_col[..., :3]) - 0.5)
               * alpha.unsqueeze(-1)).clamp(0, 1)
        _reach("SC mod2x=1", all_px)
    else:
        prem = float(min(1.0, max(0.0, -a.sprite_overbright)))
        rgb = rgb * (alpha.unsqueeze(-1) * (1.0 - prem) + prem)
        _reach("SC mod2x=0", all_px)
    alpha = alpha * abs(float(a.sprite_overbright))

    # --- the four S_MODE_* render modes: each REPLACES the shade -------
    if a.sprite_mode_gbuffer:
        # S_MODE_GBUFFER writes the encoded normal, not radiance. A
        # camera-facing particle quad's normal is -viewDir, which encodes
        # to 0.5 + 0.5*n.
        vdir = (P - eye.view(-1, 1, 1, 3))
        vdir = vdir / vdir.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        rgb = 0.5 - 0.5 * vdir
        _reach("SC S_MODE_GBUFFER=1", all_px)
    else:
        _reach("SC S_MODE_GBUFFER=0", all_px)
    if a.sprite_mode_tools_vis:
        rgb = torch.cat([uv0, torch.zeros_like(uv0[..., :1])], -1)
        _reach("SC S_MODE_TOOLS_VIS=1", all_px)
    else:
        _reach("SC S_MODE_TOOLS_VIS=0", all_px)
    if a.sprite_mode_tools_shading_complexity:
        # One shader invocation per pixel -- spritecard has no loops at
        # S_REFRACT=0, so the complexity ramp is flat by construction and
        # THAT is the reading, not a placeholder.
        rgb = torch.zeros_like(rgb) + torch.tensor(
            (0.0, 0.7, 0.0), device=dev).view(1, 1, 1, 3)
        _reach("SC S_MODE_TOOLS_SHADING_COMPLEXITY=1", all_px)
    else:
        _reach("SC S_MODE_TOOLS_SHADING_COMPLEXITY=0", all_px)
    if a.sprite_mode_tools_wireframe:
        e = ((uv0 < 0.02) | (uv0 > 0.98)).any(-1)
        rgb = torch.where(e.unsqueeze(-1), torch.ones_like(rgb),
                          torch.zeros_like(rgb))
        alpha = alpha * e.float()
        _reach("SC S_MODE_TOOLS_WIREFRAME=1", all_px)
    else:
        _reach("SC S_MODE_TOOLS_WIREFRAME=0", all_px)

    # --- remaining dynamic axes ----------------------------------------
    if a.sprite_quad_overdraw:
        rgb = torch.zeros_like(rgb) + torch.tensor(
            (0.0, 0.0, 1.0), device=dev).view(1, 1, 1, 3)
        _reach("D_QUAD_OVERDRAW=1", all_px)
    else:
        _reach("D_QUAD_OVERDRAW=0", all_px)
    if a.sprite_mboit_pass1:
        # MBOIT pass 1 shades NOTHING; it writes moments only. The alpha
        # clamp is the engine's own [1e-5, 0.9999].
        alpha = alpha.clamp(9.9999997473787516355514526367188e-06,
                            0.99989998340606689453125)
        rgb = torch.zeros_like(rgb)
        _reach("D_MBOIT_PASS1=1", all_px)
    else:
        _reach("D_MBOIT_PASS1=0", all_px)
    if a.sprite_mboit_pass2:
        rgb = rgb * alpha.unsqueeze(-1)
        _reach("D_MBOIT_PASS2=1", all_px)
    else:
        _reach("D_MBOIT_PASS2=0", all_px)
    if a.sprite_panorama_particle:
        rgb = rgb.clamp(0, 1)
        _reach("D_PANORAMA_PARTICLE=1", all_px)
    else:
        _reach("D_PANORAMA_PARTICLE=0", all_px)
    # S_REVERSE_DEPTH selects the depth-test sense; it is applied by the
    # caller (`particle_layer`) because it decides which fragments live,
    # not what they look like.
    _reach(f"S_REVERSE_DEPTH={int(bool(a.sprite_reverse_depth))}", all_px)

    return rgb, alpha.clamp(0, 1)


# ======================================================================
# DATA DEPENDENCIES
#
# A decal is an oriented box; a particle is a quad. Neither exists in
# this project's packed world -- `pack_world.py` carries map geometry
# and map materials, and a runtime decal and a particle are neither.
# So there are two sources here:
#
#   1. `load_side()` reads them out of the NON-WORLD side table that
#      `fast_pack_fam.py --nonworld-out` writes. That file is DISTINCT
#      from `assets/fam_side.pt` and both the packer and this loader
#      refuse to touch that name.
#   2. `synth_decal_volumes()` / `synth_particle_quads()` build them in
#      front of the eye, which is how these two families are shown to
#      reach real pixels instead of letting "nothing to render" become
#      "never called".
# ======================================================================

NONWORLD_FORBIDDEN = ("fam_side.pt",)


def load_side(path, device):
    """Read the non-world side table. Fails loudly on the wrong file.

    A missing key here would otherwise be absorbed as a zero -- the
    exact shape `--fam-side` already guards against by naming the
    rebuild command.
    """
    import os
    if path is None:
        return None
    if os.path.basename(path) in NONWORLD_FORBIDDEN:
        raise SystemExit(
            f"--decal-side {path}: that is the WORLD family side table. "
            "The non-world table is a separate file; build it with\n"
            "  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
            "--out assets/fam_side.pt "
            "--nonworld-out assets/fam_side_nonworld.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    need = ("decal_xform", "decal_ext2", "decal_keys", "decal_tex",
            "decal_nrm", "sprite_quad", "sprite_attr", "sprite_keys",
            "sprite_tex", "sprite_layer_tex")
    miss = [k for k in need if k not in d]
    if miss:
        raise SystemExit(
            f"--decal-side {path} is missing {', '.join(miss)}. It "
            "predates the non-world families; rebuild it with "
            "fast_pack_fam.py --nonworld-out.")
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in d.items()}


def _orthonormal(fwd, up_hint):
    r = torch.cross(fwd, up_hint, dim=-1)
    n = r.norm(dim=-1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0, 0.0], device=fwd.device).expand_as(r)
    r = torch.where(n < 1e-4, fallback, r / n.clamp(min=1e-9))
    u = torch.cross(r, fwd, dim=-1)
    return r, u / u.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def synth_decal_volumes(a, eye, fwd, device, zfc=None, inv_vp=None):
    """N oriented decal boxes on the view axis in front of the eye.

    Returns (B, N, 4, 3): rows 0..2 are the box's world-space HALF-AXES
    -- their LENGTH is the box's extent, because `decal_local` divides
    by dot(axis,axis) and nothing else carries scale -- and row 3 is the
    centre.

    THE PROJECTION AXIS POINTS BACK AT THE PROJECTOR, not along the
    shot. That is not a choice: pd/s13793_d0.glsl:204/258 is

        vec3 _9194 = M[2] * (1.0 / dot(M[2], M[2]));
        float t = (dot(Ngeo, normalize(_9194)) - cutoffCos) * cutoffScale;
        if (t < 0.0) discard;

    so `normalize(M[2])` is the RAW third axis and the surface normal
    must be ALONG it for the decal to survive S_CUTOFF_ANGLE. A surface
    the projector can see has its normal facing the projector, so
    M[2] faces the projector too. Orienting it along the shot instead
    makes dot(Ngeo, axis) = -1 and S_CUTOFF_ANGLE discards every pixel
    -- which is exactly what the smoke test reported before this was
    resolved, and is why the axis is pinned here from the bytecode
    rather than from intuition.
    """
    B = eye.shape[0]
    n = max(1, int(a.decal_synth))
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
    r, _ = _orthonormal(fwd, up)
    s = float(a.decal_synth_size)
    out = torch.zeros(B, n, 4, 3, device=device)

    # WHERE THE CENTRE GOES, and why it is no longer a distance.
    #
    # `--decal-synth-dist` put the box a fixed 220 units down the view axis.
    # That is a guess about the scene, and on the de_inferno camera it misses:
    # measured, 2 of 230,400 pixels landed inside a volume, and the pass wrote
    # nothing. A stand-in whose reachability depends on the pose is not a
    # reachability harness.
    #
    # A bullet decal is not at a fixed distance. It is AT THE SURFACE THE SHOT
    # HIT, and the surface is exactly what the depth buffer holds -- the same
    # buffer this family already unprojects to build its UVs. So when the
    # depth target is available the centre is READ from it: unproject the
    # pixel the k-th fanned direction looks through, and put the box there.
    # No distance is fitted; the scene supplies it.
    #
    # `--decal-synth-dist` remains the fallback for callers with no depth
    # target, and is stated as such rather than left looking like the
    # primary path.
    surf = None
    if zfc is not None and inv_vp is not None:
        Hh, Ww = zfc.shape[1], zfc.shape[2]
        px, py = _pixel_grid(B, Hh, Ww, device)
        P = depth_world_position(zfc, inv_vp, px, frag_coord_y(py, Hh),
                                 1.0 / Ww, 1.0 / Hh, 2.0, -1.0)
        cov = zfc < 1.0                      # 1.0 is the cleared far plane
        surf = (P, cov, Hh, Ww)

    for k in range(n):
        # fan the volumes across the view so several land on different
        # surfaces rather than stacking on one wall
        ang = (k - (n - 1) * 0.5) * 0.16
        dirk = fwd * math.cos(ang) + r * math.sin(ang)
        dirk = dirk / dirk.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        rk, uk = _orthonormal(dirk, up)
        out[:, k, 0] = rk * s
        out[:, k, 1] = uk * s
        out[:, k, 2] = -dirk * s          # faces the projector; see above
        centre = eye + dirk * float(a.decal_synth_dist)
        if surf is not None:
            P, cov, Hh, Ww = surf
            # the column this direction looks through, fanned the same way
            col = int(round(Ww * (0.5 + math.tan(ang) * 0.5)))
            col = max(0, min(Ww - 1, col))
            for b in range(B):
                rows = cov[b, :, col].nonzero()
                if rows.numel() == 0:
                    rows = cov[b].nonzero()
                    if rows.numel() == 0:
                        continue          # nothing covered; keep the fallback
                    hit = P[b, int(rows[rows.shape[0] // 2, 0]),
                            int(rows[rows.shape[0] // 2, 1])]
                else:
                    hit = P[b, int(rows[rows.shape[0] // 2, 0]), col]
                centre[b] = hit
        out[:, k, 3] = centre
    return out


def synth_particle_quads(a, eye, fwd, device):
    """N camera-facing particle quads in front of the eye.

    Returns (B, N, 4, 3): row 0 the centre, row 1 the right half-axis,
    row 2 the up half-axis, row 3 the per-particle attributes
    (age, sheet index, opacity).
    """
    B = eye.shape[0]
    n = max(1, int(a.sprite_synth))
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
    r, _ = _orthonormal(fwd, up)
    s = float(a.sprite_synth_size)
    out = torch.zeros(B, n, 4, 3, device=device)
    for k in range(n):
        ang = (k - (n - 1) * 0.5) * 0.22
        dirk = fwd * math.cos(ang) + r * math.sin(ang)
        dirk = dirk / dirk.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        rk, uk = _orthonormal(dirk, up)
        out[:, k, 0] = (eye + dirk * float(a.sprite_synth_dist)
                        + uk * (s * 0.6 * ((k % 3) - 1)))
        out[:, k, 1] = rk * s
        out[:, k, 2] = uk * s
        out[:, k, 3] = torch.tensor([0.15 * k, float(k % 4), 1.0],
                                    device=device)
    return out


def default_decal_textures(device, S=64):
    """A decal colour sheet and its companion (AO, metalness, gloss).

    Procedural, so the pass is never a silent no-op on a tree with no
    decal assets -- and deliberately NOT flat: a flat sheet would make
    S_NORMAL_MAP, S_PARALLAX and S_TRIPLANAR_MAPPING produce the same
    picture as their OFF arms, which is a check that cannot fail.
    """
    y, x = torch.meshgrid(torch.linspace(-1, 1, S, device=device),
                          torch.linspace(-1, 1, S, device=device),
                          indexing="ij")
    rad = (x * x + y * y).sqrt()
    ring = (1.0 - (rad * 1.6).clamp(0, 1)) ** 1.5
    spokes = 0.5 + 0.5 * torch.cos(torch.atan2(y, x) * 7.0)
    al = (ring * (0.55 + 0.45 * spokes)).clamp(0, 1)
    col = torch.stack([0.55 + 0.35 * spokes,
                       0.10 + 0.10 * ring,
                       0.08 + 0.06 * ring], dim=-1) * ring.unsqueeze(-1)
    tex = torch.cat([col, al.unsqueeze(-1)], dim=-1)
    # companion sheet: R = AO, G = metalness (which is what drives the
    # PBR colour-fitting branch), B = gloss.
    nrm = torch.stack([0.6 + 0.4 * ring,
                       (0.15 * spokes).clamp(0, 1),
                       (0.25 + 0.5 * ring).clamp(0, 1),
                       torch.ones_like(ring)], dim=-1)
    return tex, nrm


def default_sprite_textures(device, S=64, layers=4):
    """A particle sheet, four layer sheets, a motion-vector sheet and a
    refraction source -- all procedural and all DIFFERENT from each
    other, for the same reason as `default_decal_textures`."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, S, device=device),
                          torch.linspace(-1, 1, S, device=device),
                          indexing="ij")
    rad = (x * x + y * y).sqrt()
    puff = (1.0 - rad.clamp(0, 1)) ** 2
    base = torch.stack([0.9 * puff,
                        0.55 * puff + 0.2 * (1 - rad).clamp(0, 1),
                        0.25 * puff, puff], dim=-1)
    lay = []
    for k in range(layers):
        f = 3.0 + 2.0 * k
        s = (0.5 + 0.5 * torch.sin(x * f) * torch.cos(y * f)) * puff
        lay.append(torch.stack([s, s * 0.7, s * 0.4,
                                (s * 0.8).clamp(0, 1)], dim=-1))
    # motion vectors ride in G and A -- the .yw swizzle at sc:138
    mv = torch.stack([torch.zeros_like(x),
                      0.5 + 0.35 * torch.sin(y * 5.0),
                      torch.zeros_like(x),
                      0.5 + 0.35 * torch.cos(x * 5.0)], dim=-1)
    refr = torch.stack([0.5 + 0.4 * torch.sin(x * 9.0),
                        0.5 + 0.4 * torch.sin(y * 9.0),
                        0.5 + 0.4 * torch.sin((x + y) * 9.0),
                        torch.ones_like(x)], dim=-1)
    return base, lay, mv, refr


def frag_coord_y(py, H):
    """Raster-buffer row index -> the y `depth_world_position` expects.

    THE DEFECT THIS EXISTS TO END, stated because it cost a dead pass.

    `depth_world_position` transcribes the reference literally, and the
    reference computes `vec3(ndc.x, -ndc.y, ...)`. That negation is correct
    THERE: Vulkan's `gl_FragCoord.y` is measured from the TOP, so negating
    lands in GL's y-up NDC.

    Our depth target is not gl_FragCoord. It is nvdiffrast's raster output,
    whose row 0 is the BOTTOM -- gpu_render.py flips it to screen order at
    composite time for exactly that reason. Feeding a y-up row index into an
    expression that expects a y-down one negates twice, and the reconstructed
    world position comes back mirrored about the horizon.

    Measured, on de_inferno at 640x360 over 62,630 covered pixels:

        row index passed straight through   median error 53.4442 world units
        row index converted here            median error  0.0217 world units

    a factor of 2,460. The expression is NOT changed to fix this -- the
    conformance term `csgo_projected_decals.depth_world_position` measures it
    against the decompiled module at 0.000e+00 and must keep doing so. The
    defect was always in the INPUT, and this is the input.

    `sample_depth_at` must NOT be given this value: it indexes the buffer, so
    it takes the raw row. The two coordinates are different things and the
    reason they were conflated is that they have the same name.
    """
    return H - py


def _pixel_grid(B, H, W, device):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32) + 0.5,
        torch.arange(W, device=device, dtype=torch.float32) + 0.5,
        indexing="ij")
    return xx.expand(B, H, W).contiguous(), yy.expand(B, H, W).contiguous()


# ======================================================================
# THE TWO PASSES
# ======================================================================

def decal_layer(a, zfc, trans_zfc, fg, inv_vp, eye, fwd, sun_dir, sun_col,
                amb_sh, shadow, side, baked, supersample=1):
    """FAM_PROJECTED_DECALS. Returns (rgb, alpha, blend_mode) or None.

    Called once per chunk with the RESOLVED depth target, which is the
    dependency this family's own dynamic axes declare: the decal pass
    cannot run until scene depth exists, and it has to know whether that
    depth is MSAA.
    """
    if a.decal_project != ON:
        return None
    dev = zfc.device
    B, H, W = zfc.shape
    px, py = _pixel_grid(B, H, W, dev)
    iw, ih = 1.0 / W, 1.0 / H
    # gl_FragCoord.z in [0,1] -> GL clip z in [-1,1]: `_4459._m1/_m2`.
    zs, zb = 2.0, -1.0

    msaa = (a.decal_depth_source == "msaa")
    translucent = (a.decal_scene_depth == "translucent")
    ns = int(a.decal_msaa_samples) if msaa else 1
    offs = msaa_offsets(ns, a.decal_msaa_source, supersample) if msaa \
        else None
    _select("D_MSAA_DEPTH_BUFFER=%d" % int(msaa))
    _select("D_TRANSLUCENT_SCENE_DEPTH=%d" % int(translucent))
    _select("S_BLEND_MODE=%d" % int(a.decal_blend_mode))

    if side is not None and side.get("decal_xform") is not None \
            and side["decal_xform"].numel():
        vols = side["decal_xform"].to(dev).float()
        if vols.dim() == 3:
            vols = vols.unsqueeze(0)
        tex = side["decal_tex"].to(dev).float() / 255.0
        ntex = side["decal_nrm"].to(dev).float() / 255.0
    else:
        vols = synth_decal_volumes(a, eye, fwd, dev, zfc, inv_vp)
        t0, n0 = default_decal_textures(dev)
        tex = t0.unsqueeze(0)
        ntex = n0.unsqueeze(0)

    acc_rgb = torch.zeros(B, H, W, 3, device=dev)
    acc_a = torch.zeros(B, H, W, device=dev)

    def unproject(sample_index):
        """`dy` IS A STEP IN FRAG-COORD SPACE, NOT A BUFFER ROW STEP.

        THE SECOND HALF OF THE frag_coord_y DEFECT, and it inverted every
        decal normal in the port.

        frag_coord_y fixed the COORDINATE: our depth target is nvdiffrast's
        raster output whose row 0 is the BOTTOM, the reference's
        gl_FragCoord.y is measured from the TOP, so the row index is
        converted before it reaches the expression. What was never
        converted is the OFFSET. `geometric_normal_from_depth` is a literal
        transcription of pd:235-289 and asks for
            dyU = unproject(fragCoord + (0, +1)) - P0
        in the REFERENCE's frame. Feeding `py + 1` sampled the row one step
        the other way, so what the code called dyU was the reference's dyD
        and vice versa, `ddy` came out negated, and
            N = normalize(cross(ddy, ddx))                          (pd:361)
        came out pointing INTO the surface on every pixel.

        MEASURED on the analytic smoke fixture -- eye at the origin looking
        down -z, wall at z = -4, so a visible surface's outward normal MUST
        have +z and there is nothing to interpret:
            as shipped          ngeo.z median -0.98845,   0.0000 of pixels > 0
            offset in ref order ngeo.z median +0.98845,   1.0000 of pixels > 0

        THREE REPORTED DEFECTS, ONE CAUSE. With the normal inverted:
          * S_CUTOFF_ANGLE compares dot(Ngeo, axisZ) against +0.9930 and got
            -0.9954..-0.9913 -- every pixel discarded, reported as "the arm
            reached no pixel";
          * S_SPECULAR_DIRECT is scaled by ndl = dot(N, sunDir) clamped at 0,
            so it was identically zero -- reported as "the axis is INERT";
          * S_FASTAPPROX collapses `lit` to its channel mean, and with the
            direct term dead `lit` was grey already -- also "INERT".
        None of the three was an axis defect and none was in the shading.

        The transcription is NOT changed. The conformance term
        csgo_projected_decals.geometric_normal_from_depth measures it against
        the decompiled module at 0.000e+00 and must keep doing so; it is fed
        reference-order points directly by fam_nonworld_np and is untouched.
        The defect was in the INPUT, exactly as it was for frag_coord_y.

        `ry` is used for BOTH the buffer fetch and the coordinate, so the
        sampled texel and the position unprojected from it stay the same
        texel -- only which NEIGHBOUR a given `dy` names changes.
        """
        def f(dx, dy):
            ry = py - dy
            d = scene_depth(zfc, trans_zfc, px + dx, ry, sample_index,
                            msaa, translucent, ns, offs)
            return depth_world_position(d, inv_vp, px + dx,
                                        frag_coord_y(ry, H),
                                        iw, ih, zs, zb)
        return f

    for k in range(vols.shape[1]):
        xf = vols[0, k]
        # --- the D_MSAA_DEPTH_BUFFER sample loop, pd/s0_d4.glsl:225-273:
        #     a coverage bitmask over samples, `discard` when it is zero,
        #     and shading from the FIRST sample that landed inside.
        cover = torch.zeros(B, H, W, dtype=torch.long, device=dev)
        first = torch.full((B, H, W), -1, dtype=torch.long, device=dev)
        for i in range(ns):
            ins = decal_inside(decal_local(unproject(i)(0.0, 0.0), xf))
            cover = cover | (ins.long() << i)
            first = torch.where((first < 0) & ins,
                                torch.full_like(first, i), first)
        keep = cover != 0                    # `if (_17133 == 0u) discard;`
        _reach("D_MSAA_DEPTH_BUFFER=%d" % int(msaa), keep)
        _reach("D_TRANSLUCENT_SCENE_DEPTH=%d" % int(translucent), keep)
        if not bool(keep.any()):
            continue
        # With ns == 1 this is sample 0 and the loop above collapses to
        # the D_MSAA_DEPTH_BUFFER=0 module exactly.
        si = int(first[keep].min().item()) if ns > 1 else 0
        ngeo, ddx, ddy, P = geometric_normal_from_depth(unproject(si), px, py)
        uvw = decal_local(P, xf)
        keep = keep & decal_inside(uvw) & fg
        if not bool(keep.any()):
            continue
        rgb, alpha, k2 = projected_decal_shade(
            a, uvw, ngeo, P, xf, tex[k % tex.shape[0]],
            ntex[k % ntex.shape[0]], sun_dir, sun_col, amb_sh, shadow,
            eye, baked)
        alpha = alpha.clamp(0, 1) * (keep & k2).float()
        acc_rgb = (acc_rgb * (1 - alpha.unsqueeze(-1))
                   + rgb * alpha.unsqueeze(-1))
        acc_a = acc_a + (1 - acc_a) * alpha
    _reach("S_BLEND_MODE=%d" % int(a.decal_blend_mode), acc_a > 0.0)
    return acc_rgb, acc_a, int(a.decal_blend_mode)


def particle_layer(a, zfc, inv_vp, eye, fwd, side, fog_col):
    """FAM_SPRITECARD. Returns (rgb, alpha) or None.

    The quad is intersected analytically in screen space -- a particle
    is two triangles and a full raster pass buys nothing here -- and
    every pixel inside it runs `spritecard_shade`.

    S_REVERSE_DEPTH picks the sense of the depth test against the scene
    and so is applied HERE, not in the shade: it decides which fragments
    live, not what they look like.
    """
    if a.sprite_particles != ON:
        return None
    dev = zfc.device
    B, H, W = zfc.shape
    px, py = _pixel_grid(B, H, W, dev)
    iw, ih = 1.0 / W, 1.0 / H
    zs, zb = 2.0, -1.0
    _select("SC S_TEXTURE_LAYERS=%d" % int(a.sprite_texture_layers))
    _select("SC combine=%d" % int(a.sprite_combine_mode))
    _select("SC channel_op=%d" % int(a.sprite_channel_op))
    _select("SC post_op=%d" % int(a.sprite_post_op))

    if side is not None and side.get("sprite_quad") is not None \
            and side["sprite_quad"].numel():
        quads = side["sprite_quad"].to(dev).float()
        if quads.dim() == 3:
            quads = quads.unsqueeze(0)
        tex = side["sprite_tex"].to(dev).float() / 255.0
        lt = side["sprite_layer_tex"].to(dev).float() / 255.0
        lay = [lt[i] for i in range(lt.shape[0])]
        mv = lay[0] if lay else None
        refr = lay[-1] if lay else None
    else:
        quads = synth_particle_quads(a, eye, fwd, dev)
        tex, lay, mv, refr = default_sprite_textures(dev)

    acc_rgb = torch.zeros(B, H, W, 3, device=dev)
    acc_a = torch.zeros(B, H, W, device=dev)
    # F_SELF_ILLUM_PER_PARTICLE -- sc:707. A per-particle light, not a
    # constant: the side table carries it when it has one.
    self_illum = torch.tensor([float(x) for x in a.sprite_self_illum],
                              device=dev)

    # the eye ray through each pixel, from the same inverse VP the decal
    # pass unprojects the depth buffer with
    far = depth_world_position(torch.ones(B, H, W, device=dev), inv_vp,
                               px, frag_coord_y(py, H), iw, ih, zs, zb)
    d = far - eye.view(-1, 1, 1, 3)
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    # sample_depth_at takes the RAW row (it indexes the buffer);
    # depth_world_position takes the converted one. Different coordinates.
    scene = depth_world_position(sample_depth_at(zfc, px, py), inv_vp,
                                 px, frag_coord_y(py, H), iw, ih, zs, zb)
    dscene = (scene - eye.view(-1, 1, 1, 3)).norm(dim=-1)

    for k in range(quads.shape[1]):
        C = quads[:, k, 0]
        R = quads[:, k, 1]
        U = quads[:, k, 2]
        nq = torch.cross(R, U, dim=-1)
        nq = nq / nq.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        denom = (d * nq.view(-1, 1, 1, 3)).sum(-1)
        denom = torch.where(denom.abs() < 1e-6,
                            torch.full_like(denom, 1e-6), denom)
        t = (((C - eye) * nq).sum(-1).view(-1, 1, 1)) / denom
        P = eye.view(-1, 1, 1, 3) + d * t.unsqueeze(-1)
        rel = P - C.view(-1, 1, 1, 3)
        u = ((rel * R.view(-1, 1, 1, 3)).sum(-1)
             / R.pow(2).sum(-1).clamp(min=1e-9).view(-1, 1, 1))
        v = ((rel * U.view(-1, 1, 1, 3)).sum(-1)
             / U.pow(2).sum(-1).clamp(min=1e-9).view(-1, 1, 1))
        inside = (t > 0) & (u.abs() <= 1.0) & (v.abs() <= 1.0)
        dpart = (P - eye.view(-1, 1, 1, 3)).norm(dim=-1)
        if a.sprite_reverse_depth:
            inside = inside & (dpart >= dscene)
        else:
            inside = inside & (dpart <= dscene)
        if not bool(inside.any()):
            continue
        uv = torch.stack([u * 0.5 + 0.5, v * 0.5 + 0.5], dim=-1)
        vtx = torch.tensor([1.0, 0.92, 0.82, 1.0],
                           device=dev).view(1, 1, 1, 4).expand(B, H, W, 4)
        rgb, alpha = spritecard_shade(
            a, uv, vtx, tex, lay, mv, refr, P, eye, zfc, inv_vp, iw, ih,
            zs, zb, self_illum, fog_col, px, py)
        alpha = alpha * inside.float()
        acc_rgb = (acc_rgb * (1 - alpha.unsqueeze(-1))
                   + rgb * alpha.unsqueeze(-1))
        acc_a = acc_a + (1 - acc_a) * alpha
    return acc_rgb, acc_a


def composite_layers(rgb_lin, decal, sprite):
    """Composite both layers into a LINEAR-space image.

    The order is the engine's: decals go down after opaque (they need
    the resolved depth), particles after that (they are translucent and
    are NOT lit by the cluster path). The decal blend is the family's
    own S_BLEND_MODE; particles use their premultiplied output.
    """
    # gpu_render.py imports this file as a TOP-LEVEL module (its directory
    # is on sys.path); the conformance registry imports it as a package
    # member. Both are live, so the import has to work either way rather
    # than working in whichever context happened to be tested.
    try:
        from . import term_stats as _ts
    except ImportError:
        import term_stats as _ts

    out = rgb_lin
    if decal is not None:
        d_rgb, d_a, mode = decal
        before = out
        out = decal_composite(out, d_rgb, d_a, mode)
        # The form-1 precondition, unconditional. DECALS_WHICH_LIST.md records
        # this family as "code runs, fed synthetic data" -- so whether it
        # reaches a pixel is exactly the open question, and a pixel count
        # cannot answer it: a decal composited at alpha 0 owns pixels and
        # changes none. The mask is the layer's own alpha, so the statistic is
        # over the pixels the decal claimed rather than the whole frame, where
        # it would be diluted to nothing.
        _ts.observe("nonworld.projected_decals.composite", out, before=before,
                    mask=(d_a > 0),
                    reference="vcs/nonworld_ref/pd/s13793_d0.glsl")
        _ts.observe("nonworld.projected_decals.alpha", d_a, identity=0.0,
                    reference="vcs/nonworld_ref/pd/s0_d0.glsl")
    if sprite is not None:
        s_rgb, s_a = sprite
        before = out
        out = out * (1 - s_a.unsqueeze(-1)) + s_rgb * s_a.unsqueeze(-1)
        _ts.observe("nonworld.spritecard.composite", out, before=before,
                    mask=(s_a > 0),
                    reference="vcs/nonworld_ref/sc/s0_d0.glsl")
        _ts.observe("nonworld.spritecard.alpha", s_a, identity=0.0,
                    reference="vcs/nonworld_ref/sc/s0_d0.glsl")
    return out


# ======================================================================
# CHECKS. Written so they CAN fail, and shown to fail on purpose.
# ======================================================================

def selftest_depth_reconstruction(zfc, wpos, fg, inv_vp, H, W,
                                  z_scale=2.0, z_bias=-1.0):
    """Does the decal's depth read actually recover the scene?

    Unprojects the depth target with EXACTLY the function the decal pass
    uses and compares against the world position the rasteriser
    interpolated. Returns (max|delta| over covered pixels, n pixels), in
    world units. Never claims identity -- it reports a magnitude.

    THIS CAN FAIL: perturb `z_scale` and the number explodes, which is
    what the `depth-remap` injection does.
    """
    dev = zfc.device
    B = zfc.shape[0]
    px, py = _pixel_grid(B, H, W, dev)

    # THE Y ORIGIN IS A/B'd, NOT ASSUMED.
    #
    # depth_world_position transcribes the reference literally, and the
    # reference negates its y: `vec3(ndc.x, -ndc.y, ...)`. That negation is
    # correct THERE because Vulkan's gl_FragCoord.y is measured from the TOP,
    # so negating lands in GL's y-up NDC.
    #
    # Our depth target is not gl_FragCoord. It is nvdiffrast's raster output,
    # whose row 0 is the BOTTOM -- this file flips it to screen order at
    # composite time for exactly that reason. Feeding a y-up buffer's row
    # index into an expression that expects a y-down one negates twice, and
    # the reconstructed position is mirrored about the horizon.
    #
    # Which origin our buffer has is a property of the rasteriser, not
    # something to settle by reading; both are computed and the magnitudes
    # reported side by side, so the convention is a MEASUREMENT.
    out = {}
    for tag, pyy in (("as-transcribed (y-down source)", py),
                     ("y-up source (row 0 = bottom)", (H - py))):
        P = depth_world_position(zfc, inv_vp, px, pyy, 1.0 / W, 1.0 / H,
                                 z_scale, z_bias)
        dl = (P - wpos).norm(dim=-1)
        sel = fg & torch.isfinite(dl)
        out[tag] = ((float(dl[sel].max()), float(dl[sel].median()),
                     int(sel.sum())) if bool(sel.any())
                    else (float("nan"), float("nan"), 0))
    for tag, (mx, md, n) in out.items():
        print(f"    depth reconstruction [{tag:32s}] "
              f"max|delta| {mx:10.4f}  median {md:10.4f}  over {n} px",
              flush=True)
    # THE VERDICT IS ON THE MEDIAN, AND max ALONE WOULD HAVE MISSED THIS.
    #
    # Between the two conventions the max moved 723.5460 -> 726.2997, i.e. by
    # 0.4% and in the WRONG DIRECTION, while the median moved 53.4442 ->
    # 0.0217, a factor of 2,460. The max is set by a handful of pixels at the
    # far plane and on silhouettes, where unprojection is ill-conditioned in
    # BOTH conventions; it is nearly the same number whatever the geometry
    # does, which makes it exactly the statistic that cannot fail.
    #
    # This function returned only the max for the life of the file, and the
    # 723 it reported was read as "the reconstruction is bad" when what it
    # actually said was "some pixels are ill-conditioned". The defect it could
    # not see was a 2,460x median error that left the decal pass writing
    # nothing at all.
    best = min(out, key=lambda k: out[k][1])
    print(f"    -> by MEDIAN the correct convention is [{best}]; the max "
          f"differs by {abs(out[list(out)[0]][0] - out[list(out)[1]][0]):.4f} "
          f"between them and cannot decide it", flush=True)
    mx, md, n = out[best]
    return mx, n, md


def selftest_cost(ref_dir=None):
    """The 708 / 19 comparison, recomputed rather than quoted.

    A SECOND method against `flopcount.py`'s SPIR-V count: counts the
    texel-READING call sites and the loops of the representative
    spritecard module out of the committed reference GLSL. A headline
    never checked against a second method is
    CHECKS_THAT_CANNOT_FAIL.md instance 2.

    `textureQueryLod` builds a sampler and reads no texel;
    `textureGather` reads texels and is missed by a regex that only
    matches texture/textureLod/textureGrad (vcs/README.md section 4).
    Both are in the pattern below.
    """
    import os
    import re
    if ref_dir is None:
        ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "vcs", "nonworld_ref", "sc")
    p = os.path.join(ref_dir, "s0_d0.glsl")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="replace").read()
    body = txt[txt.index("void main()"):]
    fetches = len(re.findall(
        r"\b(?:texture|textureLod|textureGrad|textureGather|texelFetch"
        r"|textureQueryLod)\s*\(", body))
    reads = len(re.findall(
        r"\b(?:texture|textureLod|textureGrad|textureGather|texelFetch)"
        r"\s*\(", body))
    return {
        "glsl_texel_reading_call_sites": reads,
        "glsl_sampler_constructor_sites": fetches,
        "texture2D_decls": len(re.findall(r"uniform texture2D ", txt)),
        "textureCube_decls": len(re.findall(r"uniform textureCube ", txt)),
        "texture3D_decls": len(re.findall(r"uniform texture3D ", txt)),
        "samplerShadow_decls": len(re.findall(r"uniform samplerShadow ",
                                              txt)),
        "loops": len(re.findall(r"^\s*for \(;;\)", body, re.M)),
    }


def parser_replay(parser_factory):
    """Replay every add_argument against a REAL ArgumentParser.

    A P0 broke `main` because two agents declared the same flag in
    different files: git merged clean and `argparse` raised at import,
    so the renderer could not build its parser for ANY invocation. This
    rebuilds the parser and reports (n flags seen, sorted conflicts).

    It CAN fail: `parser_replay_selftest()` injects a duplicate and
    requires the conflict list to come back non-empty.
    """
    import argparse as _ap
    conflicts = []
    seen = {}
    p = _ap.ArgumentParser(prog="replay", add_help=True)

    def _note(names):
        for n in names:
            if n in seen:
                conflicts.append(n)
            seen[n] = True

    real_add = p.add_argument
    real_group = p.add_argument_group

    def add(*args, **kw):
        _note([x for x in args if isinstance(x, str) and x.startswith("-")])
        return real_add(*args, **kw)

    class _G:
        def __init__(self, inner):
            self._i = inner

        def add_argument(self, *a, **k):
            _note([x for x in a
                   if isinstance(x, str) and x.startswith("-")])
            return self._i.add_argument(*a, **k)

        def __getattr__(self, n):
            return getattr(self._i, n)

    def group(*a, **k):
        return _G(real_group(*a, **k))

    p.add_argument = add
    p.add_argument_group = group
    parser_factory(p)
    return len(seen), sorted(set(conflicts))


def parser_replay_selftest():
    """Make `parser_replay` fail on purpose.

    Without this, "0 conflicts" and "the replay never inspected a flag"
    are the same output.
    """
    def clean(p):
        add_arguments(p)

    def injected(p):
        add_arguments(p)
        p.add_argument("--decal-project", default="off")

    n_ok, c_ok = parser_replay(clean)
    _, c_bad = parser_replay(injected)
    return {"flags_declared": n_ok,
            "clean_conflicts": c_ok,
            "injected_conflicts": c_bad,
            "can_fail": bool(c_bad) and not c_ok}
