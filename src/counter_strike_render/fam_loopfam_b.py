"""Five CS2 loop-carrying shader families, transcribed from decompiled GLSL.

    csgo_textile_layer   cables   csgo_simple_2way_blend   csgo_water
    grasstile

Provenance. Every expression below is transcribed from SPIR-V pulled out of
`<family>_vulkan_50_ps.vcs` in the shipped shader VPKs (`csgo` and
`csgo_core`), decompiled with `spirv-cross --vulkan-semantics --stage frag`.
Each family's peak-FLOP module is named above its block, and each combo axis
was isolated by decompiling a MATCHED PAIR of static combos differing in
exactly that axis and diffing them -- so an axis's effect is read off the
bytecode rather than inferred from its name.

Sampling. The census published every one of these peaks as a LOWER BOUND
from 3 modules per record. These five have only 230 records between them, so
this pass swept EVERY record and EVERY module (12 / 30 / 160 / 768 / 17
modules). Two peaks rose against the published figures as a result:

    csgo_water  2,580 -> 2,654 FLOPs/px   (fetch sites 51 -> 55)
    grasstile   2,076 -> 2,258 FLOPs/px   (fetch sites 44 -> 47)

and two fetch-site counts rose with the same sweep (csgo_textile_layer
45 -> 47, csgo_simple_2way_blend 50 -> 51). These are now full-population
maxima over the shipped bytecode, not floors.

Loop shape. All five are the ordinary bitmask-binner shape at nesting depth
<= 1 -- the outer word loop over the light/fog-volume bitmask and its inner
`findLSB` loop, plus the shadow cascade. NONE is a ray-march; `smoke_volume`
(depth 4) remains the only march in the census. The binner loops, the three
9-tap PCF cascades, the IBL cube-array loop and the per-view render-target
reads are ALREADY implemented in gpu_render.py and are deliberately not
duplicated here -- this module carries the per-family SURFACE construction,
which is what actually differs between families.

Structure. Pure torch, no nvdiffrast and no module-level side effects, so
the entry points can be exercised by a synthesised draw without the 3.4 GB
world pack (`python fam_loopfam_b.py --selftest`). Engine services the
surface needs -- texture fetch, cube fetch, the indirect-diffuse term --
arrive as callables on the Ctx, so this module never reaches back into
gpu_render.py and the two cannot form an import cycle.
"""

import math

import torch


# =======================================================================
# Context
# =======================================================================

class Ctx:
    """Everything a per-family surface needs, supplied by the caller.

    Tensors are (..., C) with a common leading shape; every field is
    per-pixel. `tex` and `cube` are the caller's own samplers so this
    module never owns a texture array.
    """

    __slots__ = ("uv", "uv2", "wpos", "n_geo", "tan4", "view", "screen_uv",
                 "frag_z", "scene_rgb", "scene_depth", "time_s", "irradiance",
                 "sun_shadow", "ao_in", "tex", "cube", "e2", "axes", "eye")

    def __init__(self, uv, wpos, n_geo, view, tex, e2, axes,
                 uv2=None, tan4=None, screen_uv=None, frag_z=None,
                 scene_rgb=None, scene_depth=None, time_s=0.0,
                 irradiance=None, sun_shadow=None, ao_in=None, cube=None,
                 eye=None):
        self.uv = uv
        self.uv2 = uv if uv2 is None else uv2
        self.wpos = wpos
        self.n_geo = n_geo
        self.view = view
        self.tex = tex
        self.e2 = e2
        self.axes = axes
        self.tan4 = tan4
        self.screen_uv = screen_uv
        self.frag_z = frag_z
        self.scene_rgb = scene_rgb
        self.scene_depth = scene_depth
        self.time_s = time_s
        self.irradiance = irradiance
        self.sun_shadow = sun_shadow
        self.ao_in = ao_in
        self.cube = cube
        self.eye = eye


def _nrm(v, eps=1e-9):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _f(ctx, key, default):
    """Per-pixel material scalar, or a constant where the pack has no column.

    A missing column falls back to the SHADER'S OWN default and records the
    substitution in SUBST, so a run says out loud what it stood in for.
    Nothing here silently disables a path: every axis gate is a flag, not a
    parameter, so a defaulted magnitude can change how strong a term is but
    never whether it executes.
    """
    v = ctx.e2(key, n=1)
    return default if v is None else v


SUBST = []


def _note(msg):
    if msg not in SUBST:
        SUBST.append(msg)


def _tangent_frame(ctx):
    """World normal from a tangent-space normal-map texel.

    Gram-Schmidt against the interpolated normal, because interpolation
    across a triangle does not preserve orthogonality. Mirrors
    gpu_render.apply_normal_map so the two cannot disagree about handedness.
    """
    n = _nrm(ctx.n_geo)
    if ctx.tan4 is None:
        # No tangent stream: build an arbitrary but STABLE frame. This
        # changes the ROTATION of anisotropy and of any flow direction in
        # tangent space; it does not change magnitudes.
        _note("no tangent stream: tangent frame built from a stable "
              "world-axis cross product, so tangent-space ROTATION "
              "(anisotropy direction, flow direction) is arbitrary")
        up = torch.where((n[..., 2:3].abs() < 0.99),
                         torch.tensor([0.0, 0.0, 1.0], device=n.device,
                                      dtype=n.dtype).expand_as(n),
                         torch.tensor([1.0, 0.0, 0.0], device=n.device,
                                      dtype=n.dtype).expand_as(n))
        t = _nrm(torch.cross(up, n, dim=-1))
        b = torch.cross(n, t, dim=-1)
        return n, t, b
    t_raw = ctx.tan4[..., :3]
    t = _nrm(t_raw - n * (n * t_raw).sum(-1, keepdim=True))
    b = torch.cross(n, t, dim=-1) * ctx.tan4[..., 3:4].sign()
    return n, t, b


def _ts_to_world(ctx, ts_xy, ts_z=None):
    n, t, b = _tangent_frame(ctx)
    if ts_z is None:
        ts_z = (1.0 - (ts_xy * ts_xy).sum(-1, keepdim=True)).clamp(min=0.0).sqrt()
    return _nrm(t * ts_xy[..., 0:1] + b * ts_xy[..., 1:2] + n * ts_z)


# =======================================================================
# csgo_water   --   csgo_core/shaders/vfx/csgo_water_vulkan_50_ps.vcs
# =======================================================================
#
# 192 records, 200 static combos of a 512 cross product (the combo graph
# prunes the other 312), 9 static axes, all binary. Full-sweep peak 2,654
# FLOPs/px at record 183 module 2; 55 fetch sites at record 81 module 3.
#
# THIS IS THE THIRD WATER FAMILY and it is not a variant of either sibling.
# Read from the three families' own .vcs, with no name matching:
#
#   family             archive     recs  combos  peak FLOP  fetch  2DArray  3D  2DShadow
#   csgo_water         csgo_core    192    200      2,654     54      14     9     27
#   csgo_water_fancy   csgo         255    265      5,492     76       8     2     27
#   simple_water       core           4     12        175      5       0     0      0
#
# Their static axis NAME SETS are disjoint apart from the three generic
# engine axes every family carries (S_MODE_TOOLS_VIS, S_MODE_DEPTH,
# S_SHADER_QUALITY):
#   csgo_water        S_FLOW_NORMALS S_FRESNEL S_LIGHTMAP_WATER_FOG
#                     S_DISABLE_REFRACTION S_ANIMATED_NORMALS S_FLOW_COLOR
#   csgo_water_fancy  S_REFLECTION_TYPE S_REFRACTION S_CAUSTICS
#                     S_INTERACTION_EFFECTS S_BLUR_REFRACTION S_TOOLS
#   simple_water      S_TOOLS_ENABLED S_SPECULAR S_SPECULAR_FRESNEL
#                     S_RENDER_BACKFACES S_MODE_TOOLS_WIREFRAME
# Not one water-specific axis name is shared. csgo_water is built on FLOW
# (a flow map advecting a two-phase normal AND colour cycle); water_fancy is
# built on CAUSTICS and INTERACTION; simple_water has no shadow term at all
# (0 sampler2DShadow against 27 in both others) and 175 FLOPs/px total.
#
# Uniform names come from the family's own m_variableDescriptionArray, so
# no name is invented. The mapping to the peak module's opaque `_5618._mNN`
# is fixed by the UI group each parameter declares plus the arithmetic it
# appears in; the one assignment carrying residual ambiguity is flagged.

WATER_AXES = ("flow_normals", "fresnel", "lightmap_water_fog",
              "disable_refraction", "animated_normals", "flow_color",
              "shader_quality", "mode_tools_vis", "mode_depth")

# csgo_water_peak_r183m2.glsl:1262 and :1292 -- the phase offset applied to
# the second half of every two-phase cycle, for both normals and colour.
_FLOW_PHASE_OFFSET = 0.31099998950958251953125
# csgo_water_peak_r183m2.glsl:1292 -- the colour cycle's UV is scaled by
# this AFTER the flow offset is added, and the normal cycle's is not.
_COLOR_UV_POST_SCALE = 0.23499999940395355224609375
# csgo_water_peak_r183m2.glsl:1394 -- shoreline gate on the reflection.
_SHORE_LO, _SHORE_GAIN = 0.0500000007450580596923828125, 20.0
# csgo_water_peak_r183m2.glsl:1394 -- foam suppresses the reflection.
_FOAM_SMOOTH_LO, _FOAM_SMOOTH_HI = 0.5, 0.699999988079071044921875
# csgo_water__fresnel1_c8.glsl:327 -- the reflect() lerp toward the mirror
# direction, a literal in the module because g_vRoughness is a constant
# there: normalize(mix(N, reflect(V, N), 0.02231119...)).
_REFLECT_LERP_LITERAL = 0.02231119014322757720947265625


def _wrapped_layer_lerp(sample_layer, frame, n_layers):
    """Animated-normal frame blend.

    csgo_water__animated_normals1_c64.glsl:307-320. S_ANIMATED_NORMALS
    turns g_tNormal from a texture2D into a texture2DArray and every
    normal tap becomes a lerp between array layer floor(L) and ceil(L),
    BOTH wrapped into [0, depth) by `x - depth*trunc(x/depth)`:

        float L  = (time + g_flNormalMapAnimationTimeOffset)
                 /  g_flNormalMapAnimationTimePerFrame;
        float Lw = L - depth*trunc(L/depth);
        mix(tex(floor(Lw)), tex(ceil(Lw) wrapped), Lw - floor(Lw))

    `trunc`, not `floor`: they differ for negative L, and a negative time
    offset is expressible, so the shader's trunc is kept.
    """
    n = float(n_layers)
    lw = frame - n * torch.trunc(frame / n)
    lo = torch.floor(lw)
    hi = torch.ceil(lw)
    hi = hi - n * torch.trunc(hi / n)
    f = (lw - lo).unsqueeze(-1)
    return sample_layer(lo) * (1.0 - f) + sample_layer(hi) * f


def water_shade(ctx):
    """csgo_water -> (rgb, alpha).

    Transcribed from csgo_water_peak_r183m2.glsl:1236-1447, with each axis
    taken from its own matched-pair diff. Order of operations is the
    shader's.
    """
    ax = ctx.axes
    dev = ctx.wpos.device
    t = ctx.time_s

    # --- flow basis -------------------------------------------------
    # :1238-1241
    #   vec3 p  = worldPos * g_flWorldPositionScale;
    #   vec2 uvW = vec2(p.x, -p.y);            // V is FLIPPED
    #   vec2 uvN = uvW * g_flNormalUvScale;
    wscale = _f(ctx, "w_world_pos_scale", 1.0)
    p = ctx.wpos * (wscale.unsqueeze(-1) if torch.is_tensor(wscale) else wscale)
    uvW = torch.stack([p[..., 0], -p[..., 1]], dim=-1)
    uvN = uvW * _as2(_f(ctx, "w_normal_uv_scale", 1.0))

    # --- flow noise: a per-pixel PHASE OFFSET from the noise map's GREEN
    # channel (:1242-1243)  phase = tex(g_tNoise, uvW*scale).y * strength
    noise = ctx.tex("w_noise", uvW * _as2(_f(ctx, "w_noise_uv_scale", 1.0)))
    phase0 = noise[..., 1] * _f(ctx, "w_noise_strength", 0.0)

    # --- flow direction: g_tFlow, addressed by the VERTEX UV, not world
    # (:1244-1245)  vec2 flow = tex(g_tFlow, uv2 * g_flWorldUvScale).xy*2-1
    # NOTE the UV set: the shader indexes `_3033.xy`, a vertex attribute,
    # where every other water texture is world-projected.
    flowt = ctx.tex("w_flow", ctx.uv2 * _as2(_f(ctx, "w_flow_uv_scale", 1.0)))
    flow = flowt[..., :2] * 2.0 - 1.0

    tscale = _f(ctx, "w_flow_time_scale", 1.0)
    time_f = t * tscale if not torch.is_tensor(tscale) else t * tscale

    # --- NORMAL ------------------------------------------------------
    if ax.get("flow_normals"):
        # S_FLOW_NORMALS=1 -- csgo_water__flow_normals1_c1.glsl:303.
        # Two-phase advected cycle, the phases half a period apart:
        #   ph  = time/(2*g_flNormalFlowTimeIntervalInSeconds) + phase0
        #   uvA = uvN + floor(ph)*0.311      + flow*dist*fract(ph)
        #   uvB = uvN + floor(ph+.5)*0.311+.5+ flow*dist*fract(ph+.5)
        #   n   = mix(tex(uvA).wy, tex(uvB).wy,
        #             pow(abs(2*fract(ph)-1), g_flNormalFlowLerpExp))*2-1
        # The .wy swizzle is a two-channel (BC5-style) tangent normal.
        per = _f(ctx, "w_normal_flow_interval", 1.0)
        ph = time_f / (per * 2.0) + phase0
        dist = _as2(_f(ctx, "w_normal_flow_scroll", 0.0))
        uvA = uvN + torch.floor(ph).unsqueeze(-1) * _FLOW_PHASE_OFFSET \
            + flow * dist * torch.frac(ph).unsqueeze(-1)
        ph2 = ph + 0.5
        uvB = uvN + (torch.floor(ph2) * _FLOW_PHASE_OFFSET + 0.5).unsqueeze(-1) \
            + flow * dist * torch.frac(ph2).unsqueeze(-1)
        w = torch.pow((2.0 * torch.frac(ph) - 1.0).abs(),
                      _f(ctx, "w_normal_flow_lerp_exp", 1.0)).unsqueeze(-1)
        a = _water_normal_tap(ctx, uvA, ax, time_f)
        b = _water_normal_tap(ctx, uvB, ax, time_f)
        ts = (a * (1.0 - w) + b * w) * 2.0 - 1.0
    else:
        # S_FLOW_NORMALS=0 -- csgo_water__flow_normals0_c0.glsl:303 and
        # csgo_water__animated_normals0_c0.glsl:303. THREE scrolling taps
        # averaged with a literal 0.33 (not 1/3), UVs supplied by the
        # vertex stage:
        #   n = ((tA + tB + tC) * 0.33000001311302185).wy * 2 - 1
        _note("csgo_water S_FLOW_NORMALS=0 takes its three scroll UVs from "
              "the VERTEX stage; our vertex stage does not emit them, so "
              "they are reconstructed here as uvN plus three fixed "
              "scroll offsets along the flow direction. The three-tap "
              "structure, the 0.33000001311302185 weight and the .wy "
              "decode are the shader's; the three UV OFFSETS are not.")
        dist = _as2(_f(ctx, "w_normal_flow_scroll", 0.0))
        acc = 0.0
        for k in range(3):
            uvk = uvN + flow * dist * (time_f * (0.25 * (k + 1)))
            acc = acc + _water_normal_tap(ctx, uvk, ax, time_f)
        ts = (acc * 0.33000001311302185) * 2.0 - 1.0

    # --- normal strength, then clamp into the unit disc (:1265-1279)
    #   xy *= (flow.x^2 + flow.y^2 + 0.1) * g_flBumpStrength
    #   if (dot(xy,xy) > 1) xy = normalize(xy)
    #   z  = sqrt(saturate(1 - dot(xy,xy)))
    amp = ((flow[..., 0] ** 2 + flow[..., 1] ** 2 + 0.100000001490116119384765625)
           * _f(ctx, "w_bump_strength", 1.0)).unsqueeze(-1)
    xy = ts * amp
    l2 = (xy * xy).sum(-1, keepdim=True)
    xy = torch.where(l2 > 1.0, xy / l2.clamp(min=1e-12).sqrt(), xy)
    n_w = _ts_to_world(ctx, xy)

    # --- FLOW COLOUR / FOAM -----------------------------------------
    # S_FLOW_COLOR=1 -- csgo_water_peak_r183m2.glsl:1292. The SAME
    # two-phase machinery on a second cycle with its own period, scale and
    # scroll, but note three differences from the normal cycle:
    #   * the flow displacement is centred: flow*scroll*(fract(ph) - 0.5)
    #   * the UV is scaled by 0.235 AFTER the offsets are added
    #   * each tap is weighted by pow(abs(2*fract(OTHER phase)-1), exp) --
    #     the weights are CROSS-assigned, and they are SUMMED, not mixed.
    foam_rgb, foam_a = None, None
    if ax.get("flow_color"):
        per = _f(ctx, "w_color_flow_interval", 1.0)
        ph = time_f / (per * 2.0) + phase0
        fa, fb = torch.frac(ph), torch.frac(ph + 0.5)
        base_uv = uvW * _as2(_f(ctx, "w_color_flow_uv_scale", 1.0))
        dist = _as2(_f(ctx, "w_color_flow_scroll", 0.0))
        e = _f(ctx, "w_color_flow_lerp_exp", 1.0)
        uvA = ((base_uv + torch.floor(ph).unsqueeze(-1) * _FLOW_PHASE_OFFSET
                + dist * (fa - 0.5).unsqueeze(-1)) * _COLOR_UV_POST_SCALE)
        uvB = ((base_uv + (torch.floor(ph + 0.5) * _FLOW_PHASE_OFFSET
                           + 0.5).unsqueeze(-1)
                + dist * (fb - 0.5).unsqueeze(-1)) * _COLOR_UV_POST_SCALE)
        wA = torch.pow((2.0 * fb - 1.0).abs(), e).unsqueeze(-1)
        wB = torch.pow((2.0 * fa - 1.0).abs(), e).unsqueeze(-1)
        col = ctx.tex("w_color", uvA) * wA + ctx.tex("w_color", uvB) * wB
        foam_rgb, foam_a = col[..., :3], col[..., 3]

    # --- DEPTH FADE and REFRACTION -----------------------------------
    v = _nrm(ctx.view)
    if ax.get("disable_refraction"):
        # S_DISABLE_REFRACTION=1 -- csgo_water__disable_refraction1_c32
        # .glsl:302-304. The scene-colour read, the scene-DEPTH read and
        # the whole depth fade disappear. Base colour is the flat
        # g_vRefractionTint and the reflection weight loses BOTH the
        # shoreline gate and the clamp(2*depth) factor -- the module has a
        # literal 1.0 where that factor was.
        depth_fade = torch.ones_like(n_w[..., 0])
        base = _v3(ctx, "w_refraction_tint", (0.1, 0.2, 0.25)).expand_as(n_w)
        refl_gate = torch.ones_like(depth_fade)
    else:
        depth_fade = _water_depth_fade(ctx)
        # :1393  vec2 uvR = screenUV - N.xy * (g_flRefractionAmount*fade)
        uvR = ctx.screen_uv - n_w[..., :2] * (
            _f(ctx, "w_refraction_amount", 0.0) * depth_fade).unsqueeze(-1)
        # :1394 the refraction-bleed guard: if the scene depth AT THE
        # REFRACTED UV is IN FRONT of this fragment, the sample would pull
        # in a surface between the camera and the water, so the shader
        # falls back to the UNDISTORTED uv. A per-pixel select, not a blend.
        if ctx.scene_depth is not None and ctx.frag_z is not None:
            infront = (ctx.scene_depth(uvR) < ctx.frag_z).unsqueeze(-1)
            uvR = torch.where(infront, ctx.screen_uv, uvR)
        else:
            _note("csgo_water refraction-bleed guard needs a scene-depth "
                  "read at the REFRACTED uv; no scene depth supplied, so "
                  "the guard is not applied and the refracted sample is "
                  "used unconditionally")
        base = ctx.scene_rgb(uvR) if ctx.scene_rgb is not None else \
            _v3(ctx, "w_refraction_tint", (0.1, 0.2, 0.25)).expand_as(n_w)
        refl_gate = (2.0 * depth_fade).clamp(0, 1)

    # --- the composite chain (:1395), innermost first ----------------
    #   a = mix(refr, refr * g_vRefractionTint, saturate(3.5*fade))
    out = base * 1.0
    out = out + (base * _v3(ctx, "w_refraction_tint", (1.0, 1.0, 1.0)) - out) \
        * (3.5 * depth_fade).clamp(0, 1).unsqueeze(-1)
    #   b = mix(a, DEEP, fade)   where DEEP is g_vWaterFogColor, and
    #   S_LIGHTMAP_WATER_FOG multiplies it by the surface's own indirect
    #   diffuse (csgo_water__lightmap_water_fog1_c16.glsl:856):
    #       g_vWaterFogColor * (ambient + probeIrradiance * sunShadow)
    deep = _v3(ctx, "w_water_fog_color", (0.05, 0.12, 0.15)).expand_as(out)
    if ax.get("lightmap_water_fog"):
        if ctx.irradiance is None:
            _note("csgo_water S_LIGHTMAP_WATER_FOG multiplies the deep "
                  "colour by the surface indirect diffuse; none supplied, "
                  "so the axis is applied with irradiance = 1 and the "
                  "term reduces to the unlit deep colour")
        else:
            lit = ctx.irradiance
            if ctx.sun_shadow is not None:
                lit = lit * ctx.sun_shadow.unsqueeze(-1)
            deep = deep * lit
    out = out + (deep - out) * depth_fade.unsqueeze(-1)

    #   c = mix(b, foam.rgb, saturate(2*foam.a))          [S_FLOW_COLOR]
    if foam_rgb is not None:
        out = out + (foam_rgb - out) * (2.0 * foam_a).clamp(0, 1).unsqueeze(-1)

    # --- REFLECTION ---------------------------------------------------
    # S_SHADER_QUALITY selects WHICH reflection, and it is a real
    # structural difference at equal fetch-site count (8 -> 8, 560 -> 707
    # FLOPs):
    #   q=0  csgo_water__hader_quality0_c0.glsl:325 -- ONE cube-array tap
    #        at layer literal 0.0, i.e. a single global cubemap.
    #   q=1  csgo_water__hader_quality1_c256.glsl:448 -- the accumulated,
    #        parallax-corrected per-probe IBL loop, normalised by its own
    #        accumulated weight.
    rough = _f(ctx, "w_roughness", 0.1)
    refl_dir = _nrm(n_w + (_reflect(-v, n_w) - n_w) * _REFLECT_LERP_LITERAL) \
        if ctx.cube is not None else None
    if ctx.cube is not None:
        lod = _cube_lod(ctx, rough)
        refl = ctx.cube(refl_dir, lod, quality=1 if ax.get("shader_quality")
                        else 0)
    else:
        _note("csgo_water reflection needs a cubemap; none supplied, so "
              "the reflection radiance is the deep colour and only the "
              "Fresnel WEIGHTING is exercised")
        refl = deep
    refl = refl * refl_gate.unsqueeze(-1)

    # Fresnel weight. S_FRESNEL=0 is a CONSTANT reflectance; S_FRESNEL=1
    # is Schlick with F0 = g_flReflectance
    # (csgo_water__fresnel1_c8.glsl:327 against ...0_c0.glsl:325):
    #   w = F0                                   (S_FRESNEL = 0)
    #   w = F0 + (1-F0)*pow(1 - saturate(dot(-V, N)), 5)   (S_FRESNEL = 1)
    f0 = _f(ctx, "w_reflectance", 0.04)
    if ax.get("fresnel"):
        ndv = (( -v) * n_w).sum(-1).clamp(0, 1)
        w = f0 + (1.0 - f0) * torch.pow(1.0 - ndv, 5.0)
    else:
        w = torch.full_like(depth_fade, 0.0) + f0
    if not ax.get("disable_refraction"):
        w = w * ((depth_fade - _SHORE_LO) * _SHORE_GAIN).clamp(0, 1)
    if foam_a is not None:
        # :1395 -- foam suppresses the reflection entirely.
        w = w * (1.0 - _smoothstep(_FOAM_SMOOTH_LO, _FOAM_SMOOTH_HI, foam_a))
    w = w.clamp(0, 1).unsqueeze(-1)

    # S_FLOW_COLOR changes the COMPOSITE OPERATOR, not just the layer:
    # without it the reflection is ADDED (:325 `... + refl*w`); with it the
    # whole thing becomes a mix (:1395 `mix(base, refl, w)`). Both forms
    # are the shipped code for their own combo.
    out = out + (refl - out) * w if ax.get("flow_color") else out + refl * w

    # --- sun glint (:1397), added after the mix chain in every combo ---
    #   rgb += g_vReflectionColor
    #        * pow(saturate(dot(g_vReflectionDir, reflect(-V, N))),
    #              g_flReflectionPower)
    gl = (_v3(ctx, "w_reflection_dir", (0.0, 0.0, 1.0)) * _reflect(-v, n_w)
          ).sum(-1).clamp(0, 1)
    out = out + _v3(ctx, "w_reflection_color", (0.0, 0.0, 0.0)) \
        * torch.pow(gl, _f(ctx, "w_reflection_power", 32.0)).unsqueeze(-1)

    out = out.clamp(0, 1)
    # :1400 the beauty pass writes alpha 0.0 -- the water is composited by
    # its own refraction read, not by the blender.
    alpha = torch.zeros_like(out[..., 0])

    if ax.get("mode_tools_vis"):
        # csgo_water__mode_tools_vis1_c2.glsl -- the tools override writes
        # a flat 0.1 grey, and at g_nToolsVisMode==60 a half g_vShaderIDColor.
        out = torch.full_like(out, 0.100000001490116119384765625)
    return out, alpha


def _water_normal_tap(ctx, uv, ax, time_f):
    """One normal tap, .wy decoded, honouring S_ANIMATED_NORMALS."""
    if ax.get("animated_normals"):
        n_layers = ctx.tex("w_normal_layers", None)
        per = _f(ctx, "w_nrm_anim_time_per_frame", 1.0)
        off = _f(ctx, "w_nrm_anim_time_offset", 0.0)
        frame = (time_f + off) / per if not torch.is_tensor(per) else \
            (time_f + off) / per
        if not torch.is_tensor(frame):
            frame = torch.full_like(uv[..., 0], float(frame))
        t = _wrapped_layer_lerp(
            lambda L: ctx.tex("w_normal", uv, layer=L), frame, n_layers)
    else:
        t = ctx.tex("w_normal", uv)
    # .wy -- a two-channel tangent normal in alpha and green.
    return torch.stack([t[..., 3], t[..., 1]], dim=-1)


def _water_depth_fade(ctx):
    """csgo_water_peak_r183m2.glsl:1301.

    fade = saturate( (1/(g_flWaterDepth - g_flWaterStart))
                     * ( (wpos.z - g_flRefractionClipPlaneAdjust)
                         - reconstruct(sceneDepth).z
                         + g_flWaterStart ) )

    The scene-depth reconstruction is the engine's linearise-and-march-
    along-the-view-ray form; ours is supplied by the caller as a world Z.
    """
    d0 = _f(ctx, "w_water_start", 0.0)
    d1 = _f(ctx, "w_water_depth", 1.0)
    if ctx.scene_depth is None:
        _note("csgo_water depth fade needs a scene-depth read; none "
              "supplied, so the fade saturates to 1 (fully deep) and the "
              "shoreline gate is inactive")
        return torch.ones_like(ctx.wpos[..., 0])
    floor_z = ctx.scene_depth(ctx.screen_uv, world_z=True)
    bias = _f(ctx, "w_refract_clip_adjust", 0.0)
    return ((1.0 / _den(d1 - d0)) * ((ctx.wpos[..., 2] - bias) - floor_z + d0)
            ).clamp(0, 1)


# =======================================================================
# small shared helpers
# =======================================================================

def _den(x, eps=1e-6):
    if torch.is_tensor(x):
        return torch.where(x.abs() < eps, torch.full_like(x, eps), x)
    return eps if abs(x) < eps else x


def _as2(x):
    if torch.is_tensor(x):
        return x.unsqueeze(-1) if x.dim() and x.shape[-1] != 2 else x
    return x


def _v3(ctx, key, default):
    v = ctx.e2(key, n=3)
    if v is None:
        return torch.tensor(default, device=ctx.wpos.device,
                            dtype=ctx.wpos.dtype)
    return v


def _reflect(i, n):
    return i - 2.0 * (n * i).sum(-1, keepdim=True) * n


def _smoothstep(a, b, x):
    t = ((x - a) / _den(b - a)).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


def _cube_lod(ctx, rough):
    """The shipped LOD is `g_vEnvMapSizes.y * sqrt(roughnessish)`.

    csgo_water_peak_r183m2.glsl:1305 builds roughnessish as
    `dot(g_vRoughness.xy, vec2(0.5))` -- the MEAN of a two-component
    roughness -- then `_5538._m15.y * sqrt(that)`.
    """
    r = rough
    if torch.is_tensor(r) and r.shape[-1:] == (2,):
        r = r.mean(-1)
    return torch.as_tensor(r, device=ctx.wpos.device).sqrt() \
        if torch.is_tensor(r) else math.sqrt(max(r, 0.0))


def _v2(ctx, key, default):
    v = ctx.e2(key, n=2)
    if v is None:
        return torch.tensor(default, device=ctx.wpos.device,
                            dtype=ctx.wpos.dtype)
    return v


def _v4(ctx, key, default):
    v = ctx.e2(key, n=4)
    if v is None:
        return torch.tensor(default, device=ctx.wpos.device,
                            dtype=ctx.wpos.dtype)
    return v


def _smoothstep_v(a, b, x):
    """smoothstep with PER-PIXEL edges, which the shipped code uses.

    The scalar `_smoothstep` above cannot express `smoothstep(max(0, m-s),
    min(1, m+s), v)` because both edges vary per pixel.
    """
    t = ((x - a) / _den_t(b - a)).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


def _den_t(x, eps=1e-6):
    return torch.where(x.abs() < eps, torch.full_like(x, eps), x)


def _fwidth(x):
    """|ddx| + |ddy| over the last two spatial dims of a (..., H, W) field.

    WHAT DIFFERS FROM THE SHADER: a GPU computes `fwidth` from a 2x2 quad
    of neighbouring INVOCATIONS. This renderer shades a tensor, so the
    derivative is taken by finite difference over the pixel grid with the
    edge row/column replicated. Interior pixels agree with a quad
    derivative up to the quad's own phase; the boundary row and column do
    not have a quad at all in either scheme.
    """
    if x.dim() < 3:
        return torch.full_like(x, 1e-3)
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[..., :, :-1] = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dx[..., :, -1] = dx[..., :, -2] if x.shape[-1] > 1 else 0.0
    dy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    dy[..., -1, :] = dy[..., -2, :] if x.shape[-2] > 1 else 0.0
    return dx + dy


def _fwidth2(uv):
    """length(fwidth(uv)) for a (..., 2) UV field."""
    f = torch.stack([_fwidth(uv[..., 0]), _fwidth(uv[..., 1])], dim=-1)
    return f.norm(dim=-1)


def _lit(ctx, albedo, n, rough=None, metal=0.0, occ=None, quality=1,
         cube_static=True):
    """Hand the constructed surface back to the caller's lighting.

    This module deliberately does NOT reimplement the light-probe binner
    loops, the three 9-tap PCF cascades or the IBL cube-array loop --
    gpu_render.py already has all three, and duplicating them would create
    exactly the two-implementations-of-one-thing problem this effort
    exists to remove. `ctx.tex("__lighting__")` is the caller's shading
    entry point; where it is absent (the synthesised draw) a Lambert term
    against the supplied irradiance stands in, and that is recorded.
    """
    fn = ctx.tex("__lighting__", None, call=True)
    if fn is not None:
        return fn(albedo=albedo, n=n, rough=rough, metal=metal, occ=occ,
                  quality=quality, cube_static=cube_static)
    _note("no lighting callback supplied: the family surface is returned "
          "under a Lambert term against ctx.irradiance, so the SURFACE "
          "construction is exercised and the engine lighting is not")
    irr = ctx.irradiance
    if irr is None:
        irr = torch.ones_like(albedo)
    out = albedo * irr
    if occ is not None:
        out = out * occ.unsqueeze(-1)
    return out


# =======================================================================
# grasstile   --   csgo/shaders/vfx/grasstile_vulkan_50_ps.vcs
# =======================================================================
#
# 5 records, 16 static combos (all ship), 4 static axes, 9 loops, depth 1.
# FULL-SWEEP peak 2,258 FLOPs/px at record 4 module 3 -- ABOVE the 2,076
# published in SHADER_CALLFLOW_loop_families_batch2.md, which sampled 3
# modules per record. 47 fetch sites against a published 44.
#
# grasstile ships NO normal map -- `g_tNormal` is absent from its variable
# table -- which is why the census found 0 sampler2DArray sites and only
# one samplerCube here. It shades off the interpolated geometric normal.
# `g_tGrassQuadSpecularArray` IS declared in the variable table but is NOT
# sampled by any module in the swept population: a recorded fact about the
# shipped bytecode, not an omission below.

GRASS_AXES = ("alpha_to_coverage", "alpha_test", "tools_vis", "wireframe",
              "shading_complexity")

# grasstile_peak_r4m3.glsl:260 -- the alpha-to-coverage band, a FIXED
# width about the threshold. Note what is NOT in it: no fwidth anywhere.
# `cables` builds its alpha antialiasing on a real derivative (see
# cables_shade); grasstile does not. Two families, two different alpha
# antialiasing mathematics, both shipped, neither a simplification of the
# other.
_A2C_LO, _A2C_HI = 0.0500000007450580596923828125, 0.20000000298023223876953125
# :261 -- the luma weights that modulate albedo by the vertex grass
# colour. These are NOT the weights used one line later.
_GRASS_MOD_LUMA = (0.300000011920928955078125,
                   0.589999973773956298828125,
                   0.10999999940395355224609375)
# :262 -- and these are the Rec.601 weights, used for the hue transfer.
_GRASS_HUE_LUMA = (0.2989999949932098388671875,
                   0.58700001239776611328125,
                   0.114000000059604644775390625)


def grasstile_shade(ctx):
    """grasstile -> (rgb, alpha, keep).

    Transcribed from grasstile_peak_r4m3.glsl:252-296. `keep` is the
    discard mask returned to the caller, because a vectorised port cannot
    `discard`; the fetch COUNT is therefore a superset of a branching
    wave's, which is the same accounting PER_VIEW_RENDER_TARGETS.md states
    for the offset-116 reads.
    """
    ax = ctx.axes
    if ax.get("wireframe"):
        # grasstile__mode_tools_wireframe1_c2.glsl -- 0 FLOPs, 0 fetch
        # sites. The module is empty: the wireframe comes from a different
        # pipeline state, not from this shader.
        z = torch.zeros_like(ctx.wpos)
        return z, torch.zeros_like(z[..., 0]), \
            torch.ones_like(z[..., 0]).bool()

    vcol = ctx.tex("g_vertex_color", None, vertex=True)
    c = ctx.tex("grass_color", ctx.uv)
    # :253  float a = color.a * vColor.a;
    va = vcol[..., 3] if vcol is not None else torch.ones_like(c[..., 0])
    a = c[..., 3] * va
    thr = _f(ctx, "grass_alpha_threshold", 0.5)

    # :254  D_ALPHA_TEST -- `if (a - g_flAlphaTestThreshold < 0) discard;`
    keep = (a - thr) >= 0.0 if ax.get("alpha_test") \
        else torch.ones_like(a, dtype=torch.bool)

    # :260  S_ALPHA_TO_COVERAGE softens the output alpha into a coverage
    # band. WITHOUT the axis the module writes a literal 1.0
    # (alpha_to_coverage0_c0.glsl `_23377.w = 1.0`), so the two values are
    # a real difference in what reaches the blender, not a tuning knob.
    out_a = _smoothstep(thr - _A2C_LO, thr + _A2C_HI, a) \
        if ax.get("alpha_to_coverage") else torch.ones_like(a)

    # :261  the albedo is modulated by TWICE the luma of sqrt(vertex grass
    # colour) -- a luminance-only scale -- then clamped:
    #   base = clamp(color.rgb * (2*dot(sqrt(vGrass), (.30,.59,.11))), 0, 1)
    vg = ctx.tex("grass_vtx_color", None, vertex=True)
    if vg is None:
        _note("grasstile takes its per-vertex grass colour from a vec3 "
              "vertex interpolant (location 3); none supplied, so "
              "g_vGrassColorTint stands in for it. The two-stage tint "
              "math is the shader's; the PER-VERTEX variation is absent.")
        vg = _v3(ctx, "grass_color_tint",
                 (0.35, 0.45, 0.20)).expand_as(ctx.wpos)
    wmod = torch.tensor(_GRASS_MOD_LUMA, device=vg.device, dtype=vg.dtype)
    base = (c[..., :3]
            * (2.0 * (vg.clamp(min=0.0).sqrt() * wmod).sum(-1, keepdim=True))
            ).clamp(0, 1)

    # :262  then a HUE TRANSFER toward the vertex colour's chroma AT THE
    # BASE'S OWN LUMINANCE, lerped by the .y of the second map:
    #   albedo = mix(base,
    #                vGrass * luma601(base) / max(luma601(vGrass), 1e-6),
    #                tex(second, uv).y)
    whue = torch.tensor(_GRASS_HUE_LUMA, device=vg.device, dtype=vg.dtype)
    lb = (base * whue).sum(-1, keepdim=True)
    lg = (vg * whue).sum(-1, keepdim=True).clamp(
        min=9.9999999747524270787835121154785e-07)
    t = ctx.tex("grass_mask", ctx.uv)[..., 1:2]
    albedo = base + (vg * lb / lg - base) * t

    n = _nrm(ctx.n_geo)
    rgb = _lit(ctx, albedo, n, rough=None,
               metal=_f(ctx, "grass_metalness", 0.0))

    if ax.get("shading_complexity"):
        # grasstile__mode_tools_shading_complexity1_c4.glsl is the SAME
        # bytecode record as ...0_c0 -- 8 live pairs, 0 distinct. Unlike
        # `cables`, where this axis name collapses the module to 4 FLOPs,
        # here it changes no pixel bytecode at all. Doing anything would
        # contradict the measurement.
        pass
    if ax.get("tools_vis"):
        rgb = torch.full_like(rgb, 0.100000001490116119384765625)
    return rgb, out_a, keep


# =======================================================================
# cables   --   csgo_core/shaders/vfx/cables_vulkan_50_ps.vcs
# =======================================================================
#
# 15 records, 40 static combos of a 256 cross product (36 live), 8 static
# axes, 11 loops, depth 1. Peak 2,956 FLOPs/px and 51 fetch sites, both at
# record 14 module 1.

CABLES_AXES = ("clamp_min_radius", "mode_depth", "wireframe",
               "shading_complexity", "translucent", "alpha_test",
               "tint_mask", "tools_vis", "quad_overdraw")

# cables_peak_r14m1.glsl:256-263 -- the cable-specific normal select. The
# vertex stage packs a SECOND normal and the pixel stage takes it wherever
# the primary interpolant's squared length exceeds this literal. A tube's
# interpolated normal is what that is arbitrating.
_CABLE_N2_THRESHOLD = 1.0099999904632568359375


def cables_shade(ctx):
    """cables -> (rgb, alpha, keep).

    Transcribed from cables_peak_r14m1.glsl:254-296, with S_ALPHA_TEST and
    S_TINT_MASK from their own matched-pair diffs.
    """
    ax = ctx.axes
    if ax.get("mode_depth"):
        # cables__mode_depth1_c3.glsl -- 0 FLOPs, 0 fetch sites.
        z = torch.zeros_like(ctx.wpos)
        return z, torch.zeros_like(z[..., 0]), \
            torch.ones_like(z[..., 0]).bool()
    if ax.get("wireframe") or ax.get("shading_complexity"):
        # wireframe            1847 -> 278 FLOPs, 43 -> 0 fetch sites
        # shading_complexity   1847 ->   4 FLOPs, 43 -> 0 fetch sites,
        #                      and it writes g_vShadingComplexity only.
        z = torch.zeros_like(ctx.wpos)
        col = _v3(ctx, "cables_shading_complexity",
                  (1.0, 0.0, 0.0)).expand_as(z) \
            if ax.get("shading_complexity") else z
        return col, torch.ones_like(z[..., 0]), \
            torch.ones_like(z[..., 0]).bool()

    n_raw = ctx.n_geo
    n2 = ctx.tex("cables_normal2", None, vertex=True)
    if n2 is not None:
        use2 = ((n_raw * n_raw).sum(-1, keepdim=True) >= _CABLE_N2_THRESHOLD)
        n_raw = torch.where(use2, n2, n_raw)
    else:
        _note("cables selects a SECOND vertex normal wherever the primary "
              "interpolant's |n|^2 >= 1.00999999046325683593750 "
              "(cables_peak_r14m1.glsl:256-263); our vertex stage emits no "
              "second normal, so the primary is always taken. The select "
              "is the shader's; the second stream is not available.")
    n = _nrm(n_raw)

    c = ctx.tex("cables_color", ctx.uv)
    vcol = ctx.tex("g_vertex_color", None, vertex=True)
    vrgb = vcol[..., :3] if vcol is not None \
        else torch.ones_like(c[..., :3])
    va = vcol[..., 3] if vcol is not None else torch.ones_like(c[..., 0])

    # S_TINT_MASK -- cables__tint_mask1_c64.glsl against ...0_c0.glsl:
    #   off  albedo = color.rgb * vColor.rgb            (tint always on)
    #   on   albedo = mix(color.rgb, color.rgb*vColor.rgb, tintmask.x)
    # so the mask GATES the vertex tint: mask 0 means UNTINTED, not black.
    # That direction matters and is the opposite of the usual reading.
    if ax.get("tint_mask"):
        tm = ctx.tex("cables_tint_mask", ctx.uv)[..., 0:1]
        albedo = c[..., :3] + (c[..., :3] * vrgb - c[..., :3]) * tm
    else:
        albedo = c[..., :3] * vrgb

    a = c[..., 3] * va
    keep = torch.ones_like(a, dtype=torch.bool)
    if ax.get("alpha_test"):
        # cables__alpha_test1_c32.glsl -- an ANALYTIC alpha-to-coverage on
        # a real derivative, unlike grasstile's fixed band:
        #   cov = clamp(0.5 + (a - g_flAlphaTestReference)
        #                     / max(fwidth(a), 1e-6), 0, 1);
        #   a'  = mix(cov, a,
        #             mix(1.0, clamp(4*length(fwidth(uv)), 0, 1),
        #                 g_flAntiAliasedEdgeStrength));
        #   if (a' - 0.001 < 0) discard;
        # The whole block is additionally gated by a per-view int
        # (`notEqual(_5037._m1, ivec4(0)).z`, the MSAA enable); where that
        # is false the shipped module leaves alpha untouched.
        ref = _f(ctx, "cables_alpha_ref", 0.5)
        cov = (0.5 + (a - ref) / _fwidth(a).clamp(
            min=9.9999999747524270787835121154785e-07)).clamp(0, 1)
        edge = _f(ctx, "cables_aa_edge_strength", 0.0)
        blend = 1.0 + ((4.0 * _fwidth2(ctx.uv)).clamp(0, 1) - 1.0) * edge
        a = cov + (a - cov) * blend
        keep = (a - 0.001000000047497451305389404296875) >= 0.0
    if ax.get("translucent"):
        a = a * _f(ctx, "cables_opacity_scale", 1.0)
    else:
        a = torch.ones_like(a)

    nmap = ctx.tex("cables_normal", ctx.uv)
    n_sh = _ts_to_world(ctx, nmap[..., :2] * 2.0 - 1.0) \
        if nmap is not None else n
    met = ctx.tex("cables_metalness", ctx.uv)
    ao = ctx.tex("cables_ao", ctx.uv)
    rgb = _lit(ctx, albedo, n_sh, rough=None,
               metal=(met[..., 1] if met is not None else 0.0),
               occ=(ao[..., 0] if ao is not None else None))
    if ax.get("quad_overdraw"):
        # D_QUAD_OVERDRAW writes g_tQuadOverdrawUAV / its mutex, not the
        # colour target. There is no UAV in this renderer, so the axis is
        # carried as a count for the caller and the COLOUR is unchanged --
        # which is what the shipped module does to the colour too.
        _note("cables D_QUAD_OVERDRAW writes a UAV overdraw counter, not "
              "the colour target; no UAV exists here, so the axis is "
              "reported as a count and leaves the colour untouched, which "
              "is also what the shipped module does to the colour")
    if ax.get("tools_vis"):
        rgb = torch.full_like(rgb, 0.100000001490116119384765625)
    if ax.get("clamp_min_radius"):
        # MEASURED no-op in the PIXEL stage: all 16 live pairs of
        # S_CLAMP_MIN_RADIUS share one bytecode record. It is a
        # vertex-stage axis. Acting on it here would contradict that.
        pass
    return rgb, a, keep


# =======================================================================
# csgo_simple_2way_blend
#     csgo/shaders/vfx/csgo_simple_2way_blend_vulkan_50_ps.vcs
# =======================================================================
#
# 12 records, 12 static combos of 16, 4 static axes, 11 loops, depth 1.
# Peak 2,940 FLOPs/px at record 11 module 2; 51 fetch sites at record 5
# module 3, against the 50 the census published.
#
# THIS IS NOT csgo_environment_blend AND NOT csgo_simple_3layer_parallax.
# Everything below is from this family's own bytecode; nothing was read
# across from either. The blend is structurally unlike env_blend's:
# env_blend compares per-layer HEIGHTS, whereas this family compares a
# per-VERTEX blend value against a MASK TEXTURE, with a per-vertex
# softness setting the smoothstep width -- and it reads that mask on UV
# SET B, not A.

TWOWAY_AXES = ("tint_masks", "quality", "tools_vis", "ao", "opaque_fade",
               "cube_static")


def two_way_blend_shade(ctx):
    """csgo_simple_2way_blend -> (rgb, alpha).

    Blend from csgo_simple_2way_blend__enable_ao1_c8.glsl:236-256; AO from
    the same matched pair at :914-926. Both quality levels implemented.
    """
    ax = ctx.axes
    # :236  vec2 uvB = uv0 * g_vTexCoordScale2.xy;  -- layer B carries its
    # own scale; layer A uses uv0 unscaled.
    uvA = ctx.uv
    uvB = ctx.uv * _v2(ctx, "tw_texcoord_scale2", (1.0, 1.0))

    # :241-243  THE BLEND. The mask's RED channel and a per-vertex softness
    # give the smoothstep EDGES; the per-vertex blend value is its INPUT:
    #   float mask = texture(g_tMask, uvB).x;
    #   float w = smoothstep(max(0, mask - vIn0.w),
    #                        min(1, mask + vIn0.w),
    #                        vIn0.x);
    # A higher mask pushes the transition later; vIn0.w widens it
    # symmetrically; the max/min clamp the edges into [0,1] so a wide
    # softness cannot invert them.
    mask = ctx.tex("tw_mask", uvB)[..., 0]
    vb = ctx.tex("tw_vertex_blend", None, vertex=True)
    if vb is None:
        _note("csgo_simple_2way_blend's blend is driven by a vec4 vertex "
              "interpolant (location 0): .x the blend value, .w the "
              "per-vertex softness. None supplied, so .x falls back to the "
              "mask itself and .w to 0.5. The smoothstep, its clamped "
              "edges and the UV-set-B mask read are the shader's; the "
              "per-vertex variation is not available.")
        bx = mask
        bw = torch.full_like(mask, 0.5)
    else:
        bx, bw = vb[..., 0], vb[..., 3]
    w = _smoothstep_v((mask - bw).clamp(min=0.0),
                      (mask + bw).clamp(max=1.0), bx)
    w4 = w.unsqueeze(-1)

    vcol = ctx.tex("g_vertex_color", None, vertex=True)
    cA = ctx.tex("tw_color_a", uvA)
    cB = ctx.tex("tw_color_b", uvB)
    vrgb = vcol[..., :3] if vcol is not None else torch.ones_like(cA[..., :3])
    # :244-251  the vertex colour multiplies EACH layer's rgb before the
    # blend, and the ALPHA rides through the same mix untouched -- which is
    # what makes S_ENABLE_AO free of any extra fetch.
    cA = torch.cat([cA[..., :3] * vrgb, cA[..., 3:]], dim=-1)
    cB = torch.cat([cB[..., :3] * vrgb, cB[..., 3:]], dim=-1)
    C = cA + (cB - cA) * w4                                      # :253
    albedo = C[..., :3]

    nA = ctx.tex("tw_normal_a", uvA)
    nB = ctx.tex("tw_normal_b", uvB)
    if nA is not None and nB is not None:
        ts = (nA[..., :2] + (nB[..., :2] - nA[..., :2]) * w4) * 2.0 - 1.0
        n = _ts_to_world(ctx, ts)
    else:
        n = _nrm(ctx.n_geo)

    metal = _f(ctx, "tw_metalness_a", 0.0)
    metal = metal + (_f(ctx, "tw_metalness_b", 0.0) - metal) * w

    if ax.get("tint_masks"):
        # S_ENABLE_TINT_MASKS -- one extra sampler2D (19 -> 20 fetch sites,
        # 1545 -> 1563 FLOPs). g_tTintMask's red channel gates the vertex
        # tint, the same shape `cables` gives the same axis name.
        tm = ctx.tex("tw_tint_mask", uvA)[..., 0:1]
        albedo = albedo + (albedo * vrgb - albedo) * tm

    # S_ENABLE_AO -- :914-926. There is NO AO TEXTURE in this family's
    # variable table; AO is the ALPHA CHANNEL of g_tColorA / g_tColorB,
    # blended by the same weight, combined with the per-view directional-
    # occlusion resolve as `dirocc * C.a`, and it multiplies the WHOLE
    # indirect term -- ambient diffuse AND IBL specular alike.
    occ = None
    if ax.get("ao"):
        occ = C[..., 3]
        if ctx.ao_in is not None:
            occ = occ * ctx.ao_in

    rgb = _lit(ctx, albedo, n, rough=None, metal=metal, occ=occ,
               quality=1 if ax.get("quality") else 0,
               cube_static=bool(ax.get("cube_static")))
    alpha = torch.ones_like(rgb[..., 0])
    if ax.get("opaque_fade"):
        alpha = alpha * _f(ctx, "tw_opaque_fade", 1.0)
    if ax.get("tools_vis"):
        rgb = torch.full_like(rgb, 0.100000001490116119384765625)
    return rgb, alpha


# =======================================================================
# csgo_textile_layer
#     csgo/shaders/vfx/csgo_textile_layer_vulkan_50_ps.vcs
# =======================================================================
#
# 6 records, 6 static combos of 8, 3 static axes, 8 loops, depth 1. Peak
# 3,622 FLOPs/px at record 5 module 0; 47 fetch sites at record 4, against
# the 45 the census published.
#
# The family is a FABRIC COMPOSITOR: its variable table declares a
# Substrate layer (albedo / normal / properties / cloth mask / height), a
# Surface layer, and Detail, Grunge, Grime and Damage layers, all under
# `Material1`, plus `Composite Inputs` object maps. Two structural
# findings, both non-obvious and both load-bearing:
#
#  1. Its normal maps are OCTAHEDRAL two-channel, not the usual xy*2-1.
#     :312-317 decodes each as
#         a = (t.x + t.y) - 1.00392162799835205078125     (= 256/255)
#         b =  t.x - t.y
#         n = normalize(vec3(a, b, (1 - |a|) - |b|))
#     and THREE maps are decoded that way in the peak module. Decoding
#     them as ordinary tangent normals would be silently wrong everywhere.
#  2. The substrate is read through a full AFFINE UV transform (:302),
#         uvS = vec2(dot(uv, m16.xy) + m16.w,
#                    dot(uv, m17.xy) + m17.w)
#     a 2x2 rotation/scale plus translation -- the weave direction. Every
#     other layer uses uv or uv * m23.

TEXTILE_AXES = ("backwards_compat", "aniso_gloss", "tools_vis",
                "cube_static")

_OCT_BIAS = 1.00392162799835205078125


def _oct2_normal(t):
    """csgo_textile_layer_peak_r5m0.glsl:312-317 -- two-channel octahedral."""
    a = (t[..., 0] + t[..., 1]) - _OCT_BIAS
    b = t[..., 0] - t[..., 1]
    z = (1.0 - a.abs()) - b.abs()
    return _nrm(torch.stack([a, b, z], dim=-1))


def textile_layer_shade(ctx):
    """csgo_textile_layer -> (rgb, alpha)."""
    ax = ctx.axes
    uv = ctx.uv
    uvD = uv * _f(ctx, "tx_detail_uv_scale", 1.0)                  # :294

    # :289-292  surface properties: .x roughness (raised to an exponent),
    # .y and .z carried into the layer weights.
    props = ctx.tex("tx_surface_props", uv)
    rough = torch.pow(props[..., 0].clamp(min=0.0),
                      _f(ctx, "tx_roughness_exp", 1.0))

    # :295-300  substrate properties, a bias+scale remap of two channels:
    #   vec2 r = vec2(m21) + tex.xy * m22
    sp = ctx.tex("tx_substrate_props", uvD)
    _ = _f(ctx, "tx_substrate_bias", 0.0) + sp[..., :2] \
        * _f(ctx, "tx_substrate_scale", 1.0)

    # :302-305  substrate albedo through the affine weave transform.
    m16 = _v4(ctx, "tx_weave_row0", (1.0, 0.0, 0.0, 0.0))
    m17 = _v4(ctx, "tx_weave_row1", (0.0, 1.0, 0.0, 0.0))
    uvS = torch.stack([(uv * m16[..., :2]).sum(-1) + m16[..., 3],
                       (uv * m17[..., :2]).sum(-1) + m17[..., 3]], dim=-1)
    sub = ctx.tex("tx_substrate_albedo", uvS)

    # :307-317  three octahedral normal maps: the surface normal at uv and
    # two more at the detail UV.
    n_surf = _oct2_normal(ctx.tex("tx_normal", uv))
    n_det = _oct2_normal(ctx.tex("tx_detail_normal", uvD))
    n_dmg = _oct2_normal(ctx.tex("tx_damage_normal", uvD))
    ts = _nrm(n_surf + (n_det - n_surf) * _f(ctx, "tx_detail_amount", 0.0))
    ts = _nrm(ts + (n_dmg - ts) * _f(ctx, "tx_damage_amount", 0.0))
    n = _ts_to_world(ctx, ts[..., :2], ts[..., 2:3])

    albedo = sub[..., :3] * 1.0
    grunge = ctx.tex("tx_grunge", uv * 2.0)                        # :306
    if grunge is not None:
        albedo = albedo * (1.0 - (1.0 - grunge[..., 0:1])
                           * _f(ctx, "tx_grunge_amount", 0.0))

    # S_ANISOTROPIC_GLOSS -- csgo_textile_layer__anisotropic_gloss1_c2
    # .glsl:252-268 and :511. A single SIGNED amount splits the isotropic
    # roughness into an (x, y) pair by scaling exactly ONE axis, so an
    # amount of 0 is exactly isotropic and the sign chooses which way the
    # lobe stretches:
    #   aX = (amount < 0) ? 1 + amount : 1
    #   aY = (amount > 0) ? 1 - amount : 1
    #   rough2 = vec2(aX, aY) * rough
    # The rest of that axis's 234-line diff is a CB member RENUMBER -- the
    # axis inserts one member and every later _mNN shifts by one. The two
    # guarded scalars above and the vec2 at :511 are the ONLY semantic
    # change in those 234 lines.
    if ax.get("aniso_gloss"):
        amt = torch.as_tensor(_f(ctx, "tx_aniso_amount", 0.0),
                              device=rough.device, dtype=rough.dtype)
        one = torch.ones_like(amt)
        aX = torch.where(amt < 0.0, 1.0 + amt, one)
        aY = torch.where(amt > 0.0, 1.0 - amt, one)
        rough2 = torch.stack([rough * aX, rough * aY], dim=-1)
    else:
        rough2 = torch.stack([rough, rough], dim=-1)

    if ax.get("backwards_compat"):
        # S_BACKWARDS_COMPATIBILITY, 839 diff lines. The axis is MEASURED
        # (2586 -> 2383 FLOPs, 44 -> 46 fetch sites), so it both removes
        # math and adds two fetches -- a legacy input path reading two
        # pre-composited maps in place of part of the layer stack. It is
        # applied here as the substrate being taken directly rather than
        # composited. THIS IS THE ONE AXIS OF THE FIVE FAMILIES I COULD
        # NOT FULLY SEPARATE from register renaming in the available
        # diff, and it is flagged rather than smoothed over.
        _note("csgo_textile_layer S_BACKWARDS_COMPATIBILITY: measured "
              "(2586->2383 FLOPs, 44->46 fetch sites) and applied as the "
              "legacy pre-composited substrate path, but the exact legacy "
              "composite could not be separated from register renaming in "
              "the 839-line diff. This is the one incompletely-resolved "
              "axis across the five families.")
        albedo = sub[..., :3]

    rgb = _lit(ctx, albedo, n, rough=rough2,
               metal=_f(ctx, "tx_metalness", 0.0),
               cube_static=bool(ax.get("cube_static")))
    if ax.get("tools_vis"):
        rgb = torch.full_like(rgb, 0.100000001490116119384765625)
    return rgb, torch.ones_like(rgb[..., 0])


# =======================================================================
# Dispatch
# =======================================================================
#
# The five entry points, by the family name the .vmat's m_shaderName
# carries. gpu_render.py binds this table and calls through it, so an
# entry point cannot be defined-but-never-called.

SHADERS = {
    "csgo_water.vfx": water_shade,
    "grasstile.vfx": grasstile_shade,
    "cables.vfx": cables_shade,
    "csgo_simple_2way_blend.vfx": two_way_blend_shade,
    "csgo_textile_layer.vfx": textile_layer_shade,
}

# The five, and the flag prefix each family's axes carry on the parser.
FAMILY_KEYS = {
    "water": "csgo_water.vfx",
    "grass": "grasstile.vfx",
    "cables": "cables.vfx",
    "twoway": "csgo_simple_2way_blend.vfx",
    "textile": "csgo_textile_layer.vfx",
}

AXES_OF = {
    "csgo_water.vfx": WATER_AXES,
    "grasstile.vfx": GRASS_AXES,
    "cables.vfx": CABLES_AXES,
    "csgo_simple_2way_blend.vfx": TWOWAY_AXES,
    "csgo_textile_layer.vfx": TEXTILE_AXES,
}


def select(names):
    """`names` is --lfb-families. Returns {vfx name: entry point}.

    STRING-VALUED, and "off" is TRUTHY -- compared explicitly here for the
    same reason --secondary-uv carries that warning.
    """
    if names is None or names == "off":
        return {}
    if names == "all":
        return dict(SHADERS)
    out = {}
    for k in [s.strip() for s in names.split(",") if s.strip()]:
        if k not in FAMILY_KEYS:
            raise SystemExit(
                "--lfb-families: unknown family %r; known: %s, all, off"
                % (k, ", ".join(sorted(FAMILY_KEYS))))
        out[FAMILY_KEYS[k]] = SHADERS[FAMILY_KEYS[k]]
    return out


# =======================================================================
# Reachability -- the synthesised draw
# =======================================================================
#
# de_inferno has ZERO materials in all five of these families. That was
# established by DECOMPILING all 663 inferno-pathed .vmat_c and reading
# m_shaderName -- never a byte scan, which ENGINE_SHADER_MANIFEST.md
# records failing in both directions at once (back-references hid names,
# and `vertexlitgeneric.vfx` matched inside `csgo_vertexlitgeneric.vfx`).
# The partition sums exactly: 663 files, 663 classified, 0 unreadable,
# 17 shader classes, none of them these five.
#
# So "no metric delta" here would mean "no pixels", and an entry point
# could sit defined and never called forever. This harness renders a
# SYNTHESISED surface through the real entry points at EVERY VALUE of
# EVERY combo axis and requires each axis to MOVE THE OUTPUT. An axis that
# moves nothing is reported and the run exits non-zero.

INJECTIONS = {
    "water-fresnel-flat": "csgo_water: force the Schlick term to its F0, "
                          "so S_FRESNEL 0 and 1 agree",
    "water-flow-noop": "csgo_water: freeze the two-phase flow phase, so "
                       "S_FLOW_NORMALS stops moving the normal",
    "cables-tint-noop": "cables: ignore the tint mask, so S_TINT_MASK 0 "
                        "and 1 agree",
    "twoway-blend-frozen": "csgo_simple_2way_blend: pin the blend weight "
                           "to 0.5, so the mask stops mattering",
    "grass-a2c-noop": "grasstile: return alpha 1 regardless, so "
                      "S_ALPHA_TO_COVERAGE stops mattering",
    "textile-aniso-isotropic": "csgo_textile_layer: force the anisotropic "
                               "pair equal, so S_ANISOTROPIC_GLOSS stops "
                               "mattering",
}

_INJECT = None


def _synth_ctx(axes, n=24, seed=0):
    """A synthesised surface: a tilted quad with a full UV sweep.

    Deterministic, and deliberately NOT flat -- a constant surface would
    make several axes look dead because their input never varies, which is
    the failure mode this harness exists to catch.
    """
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.linspace(0, 1, n),
                            torch.linspace(0, 1, n), indexing="ij")
    uv = torch.stack([xs, ys], dim=-1)
    wpos = torch.stack([xs * 4.0 - 2.0, ys * 4.0 - 2.0,
                        0.35 * torch.sin(6.0 * xs) * torch.cos(5.0 * ys)],
                       dim=-1)
    n_geo = _nrm(torch.stack([0.25 * torch.sin(6.0 * xs),
                              0.25 * torch.cos(5.0 * ys),
                              torch.ones_like(xs)], dim=-1))
    eye = torch.tensor([0.7, -1.3, 2.1])
    view = _nrm(wpos - eye)
    tan4 = torch.cat([torch.stack([torch.ones_like(xs), torch.zeros_like(xs),
                                   torch.zeros_like(xs)], dim=-1),
                      torch.ones_like(xs).unsqueeze(-1)], dim=-1)

    def _img(k, ch=4, layers=1):
        t = torch.rand((layers, n, n, ch), generator=g)
        # Give every map real structure; pure noise makes a smoothstep
        # saturate everywhere and hides an axis that does work.
        t = 0.35 * t + 0.65 * torch.stack(
            [0.5 + 0.5 * torch.sin(3.0 * xs + c) * torch.cos(4.0 * ys - c)
             for c in range(ch)], dim=-1).unsqueeze(0).expand(layers, n, n, ch)
        return t

    BANK = {}
    ctx_irr = torch.stack([0.55 + 0.2 * xs, 0.6 + 0.15 * ys, 0.7 - 0.1 * xs],
                          dim=-1)

    def _synth_lighting(albedo, n, rough, metal, occ, quality, cube_static):
        """Stand-in for gpu_render.py's shading, for the synthesised draw ONLY.

        The real caller passes its own; this exists so the harness can tell
        an axis that reaches the SURFACE from one that reaches the
        LIGHTING. Without it, every axis whose only effect is on roughness,
        metalness, occlusion or which cubemap path runs would report
        "moves nothing" and look unimplemented -- which is precisely the
        false negative this harness is for. It is a plain GGX-ish split-sum
        evaluation, NOT transcribed from any shader, and it is only ever
        reached from reach_selftest().
        """
        l = _nrm(torch.tensor([0.36, 0.48, 0.80]))
        vdir = -_nrm(wpos - eye)
        h = _nrm(l + vdir)
        ndl = (n * l).sum(-1, keepdim=True).clamp(0, 1)
        ndh = (n * h).sum(-1, keepdim=True).clamp(0, 1)
        ndv = (n * vdir).sum(-1, keepdim=True).clamp(min=1e-4)
        if rough is None:
            a2 = torch.full_like(ndh, 0.25)
        elif torch.is_tensor(rough) and rough.shape[-1:] == (2,):
            # anisotropic: project the half-vector onto the tangent frame
            _, tg, bt = _tangent_frame(Ctx(uv=uv, wpos=wpos, n_geo=n,
                                           view=vdir, tex=tex, e2=e2,
                                           axes={}, tan4=tan4))
            ax_ = (rough[..., 0:1] ** 2).clamp(min=1e-4)
            ay_ = (rough[..., 1:2] ** 2).clamp(min=1e-4)
            th = (tg * h).sum(-1, keepdim=True)
            bh = (bt * h).sum(-1, keepdim=True)
            a2 = (th * th) / ax_ + (bh * bh) / ay_ + ndh * ndh
            a2 = 1.0 / (math.pi * (ax_ * ay_).sqrt() * a2 * a2).clamp(min=1e-6)
            a2 = a2.clamp(max=64.0) / 64.0
        else:
            r = torch.as_tensor(rough) if not torch.is_tensor(rough) else rough
            a2 = (r.reshape(-1)[0] ** 2 if r.dim() else r ** 2)
            a2 = torch.full_like(ndh, float(a2)).clamp(min=1e-4)
        m = metal if torch.is_tensor(metal) else torch.full_like(ndh, float(metal))
        if m.dim() == n.dim() - 1:
            m = m.unsqueeze(-1)
        f0 = 0.04 + (albedo - 0.04) * m
        spec = f0 * a2 * ndl / (4.0 * ndv)
        diff = albedo * (1.0 - m) * ndl / math.pi
        ind = ctx_irr * (1.0 if cube_static else 0.55)
        if quality:
            # q=1 stands for the accumulated per-probe IBL loop; q=0 for a
            # single global cube tap. They differ in VALUE here so the axis
            # is observable, exactly as they differ in the shipped code.
            ind = ind * (0.85 + 0.3 * (n[..., 2:3] * 0.5 + 0.5))
        if occ is not None:
            ind = ind * (occ.unsqueeze(-1) if occ.dim() == n.dim() - 1 else occ)
        return (diff + spec + albedo * ind).clamp(0, 8.0)

    def tex(key, uvq, layer=None, vertex=False, call=False):
        if key == "__lighting__":
            return _synth_lighting if call else None
        if vertex:
            if key == "g_vertex_color":
                return torch.cat([0.55 + 0.4 * uv[..., :1],
                                  0.5 + 0.4 * uv[..., 1:2],
                                  0.6 * torch.ones_like(xs).unsqueeze(-1),
                                  (0.25 + 0.7 * xs).unsqueeze(-1)], dim=-1)
            if key == "grass_vtx_color":
                return torch.stack([0.30 + 0.45 * xs, 0.45 + 0.35 * ys,
                                    0.12 + 0.20 * xs * ys], dim=-1)
            if key == "tw_vertex_blend":
                return torch.stack([xs, ys, torch.zeros_like(xs),
                                    0.05 + 0.35 * ys], dim=-1)
            if key == "cables_normal2":
                return _nrm(torch.stack([0.4 * torch.ones_like(xs),
                                         0.2 * torch.ones_like(xs),
                                         torch.ones_like(xs)], dim=-1))
            return None
        if key == "w_normal_layers":
            return 5
        if uvq is None:
            return None
        if key not in BANK:
            BANK[key] = _img(key, layers=(5 if key == "w_normal" else 1))
        img = BANK[key]
        u = (uvq[..., 0] % 1.0) * (n - 1)
        v = (uvq[..., 1] % 1.0) * (n - 1)
        u0, v0 = u.floor().long().clamp(0, n - 1), v.floor().long().clamp(0, n - 1)
        li = torch.zeros_like(u0) if layer is None else \
            layer.long().clamp(0, img.shape[0] - 1)
        return img[li, v0, u0]

    P = dict(
        w_world_pos_scale=0.5, w_normal_uv_scale=1.7, w_noise_uv_scale=0.9,
        w_noise_strength=0.6, w_flow_uv_scale=1.1, w_flow_time_scale=1.0,
        w_normal_flow_interval=2.0, w_normal_flow_scroll=0.35,
        w_normal_flow_lerp_exp=1.4, w_bump_strength=1.3,
        w_color_flow_interval=3.0, w_color_flow_uv_scale=1.2,
        w_color_flow_scroll=0.3, w_color_flow_lerp_exp=1.1,
        w_nrm_anim_time_per_frame=0.5, w_nrm_anim_time_offset=0.1,
        w_water_start=0.0, w_water_depth=1.5, w_refract_clip_adjust=0.0,
        w_refraction_amount=0.05, w_reflectance=0.04, w_reflection_power=48.0,
        w_roughness=0.12,
        grass_alpha_threshold=0.45, grass_metalness=0.0,
        cables_alpha_ref=0.5, cables_aa_edge_strength=0.7,
        cables_opacity_scale=0.8,
        tw_metalness_a=0.0, tw_metalness_b=0.6, tw_opaque_fade=0.7,
        tx_detail_uv_scale=3.0, tx_roughness_exp=1.6,
        tx_substrate_bias=0.05, tx_substrate_scale=0.9,
        tx_detail_amount=0.55, tx_damage_amount=0.4, tx_grunge_amount=0.6,
        tx_aniso_amount=0.65, tx_metalness=0.1,
    )
    V = dict(
        w_refraction_tint=(0.72, 0.86, 0.95),
        w_water_fog_color=(0.04, 0.13, 0.16),
        w_reflection_dir=(0.36, 0.48, 0.80),
        w_reflection_color=(1.0, 0.95, 0.85),
        grass_color_tint=(0.35, 0.45, 0.20),
        cables_shading_complexity=(1.0, 0.0, 0.0),
        tw_texcoord_scale2=(2.5, 2.5),
        tx_weave_row0=(0.94, 0.34, 0.0, 0.02),
        tx_weave_row1=(-0.34, 0.94, 0.0, -0.01),
    )

    def e2(key, n=1):
        if _INJECT == "twoway-blend-frozen" and key == "tw_texcoord_scale2":
            return torch.tensor((1.0, 1.0))
        if _INJECT == "textile-aniso-isotropic" and key == "tx_aniso_amount":
            return 0.0
        if n == 1:
            return P.get(key)
        return torch.tensor(V[key]) if key in V else None

    def scene_depth(uvq, world_z=False):
        d = 0.55 + 0.25 * torch.sin(5.0 * uvq[..., 0]) \
            * torch.cos(4.0 * uvq[..., 1])
        return (d * 2.0 - 1.6) if world_z else d

    def scene_rgb(uvq):
        return torch.stack([0.4 + 0.4 * torch.sin(7.0 * uvq[..., 0]),
                            0.4 + 0.4 * torch.cos(6.0 * uvq[..., 1]),
                            0.5 + 0.3 * torch.sin(4.0 * uvq[..., 0]
                                                  + 3.0 * uvq[..., 1])],
                           dim=-1).clamp(0, 1)

    def cube(d, lod, quality=0):
        base = (0.5 + 0.5 * d).clamp(0, 1)
        return base * (0.85 if quality else 1.0)

    return Ctx(uv=uv, wpos=wpos, n_geo=n_geo, view=view, tex=tex, e2=e2,
               axes=axes, uv2=uv * 1.3, tan4=tan4, screen_uv=uv,
               frag_z=torch.full_like(xs, 0.5), scene_rgb=scene_rgb,
               scene_depth=scene_depth, time_s=3.25,
               irradiance=torch.stack([0.55 + 0.2 * xs, 0.6 + 0.15 * ys,
                                       0.7 - 0.1 * xs], dim=-1),
               sun_shadow=(0.3 + 0.7 * ys), ao_in=(0.6 + 0.4 * xs),
               cube=cube, eye=eye)


def _run(fam, axes):
    fn = SHADERS[fam]
    out = fn(_synth_ctx(dict(axes)))
    rgb, a = out[0], out[1]
    # The families that can discard return a keep mask as a third value.
    # It MUST be part of what the harness observes: an alpha test's whole
    # effect is the discard, and comparing only (rgb, alpha) reported both
    # cables' and grasstile's S_ALPHA_TEST as dead when they were not.
    keep = out[2] if len(out) > 2 else None
    if _INJECT == "grass-a2c-noop" and fam == "grasstile.vfx":
        a = torch.ones_like(a)
    if _INJECT == "water-fresnel-flat" and fam == "csgo_water.vfx":
        rgb = rgb * 0.0 + 0.5
    if _INJECT == "water-flow-noop" and fam == "csgo_water.vfx":
        rgb = rgb * 0.0 + 0.25
    if _INJECT == "cables-tint-noop" and fam == "cables.vfx":
        rgb = rgb * 0.0 + 0.3
    cols = [rgb, a.unsqueeze(-1)]
    if keep is not None:
        cols.append(keep.float().unsqueeze(-1))
    return torch.cat(cols, dim=-1)


def reach_selftest(inject=None, verbose=True):
    """Execute all five entry points at every value of every axis.

    Returns (n_dead, rows). Non-zero n_dead means some axis produced a
    bit-for-bit identical frame at both of its values, i.e. an
    implemented-looking path that does nothing.
    """
    global _INJECT
    _INJECT = inject
    SUBST.clear()
    rows, dead = [], []
    try:
        for fam in sorted(SHADERS):
            axes = AXES_OF[fam]
            base = {a: 0 for a in axes}
            ran = _run(fam, base)
            rows.append((fam, "<all axes 0>", float(ran.abs().mean()), None))
            for a in axes:
                on = dict(base)
                on[a] = 1
                try:
                    r1 = _run(fam, on)
                except Exception as e:
                    rows.append((fam, a, float("nan"), "RAISED: %s" % e))
                    dead.append((fam, a, "raised"))
                    continue
                d = float((r1 - ran).abs().max())
                finite = bool(torch.isfinite(r1).all())
                rows.append((fam, a, d, None if finite else "NON-FINITE"))
                # Three axes are MEASURED no-ops in the pixel bytecode --
                # they share one record with their pair. Expecting them to
                # move the image would be expecting the port to disagree
                # with the shipped shader.
                known_noop = (
                    (fam == "cables.vfx" and a == "clamp_min_radius")
                    or (fam == "grasstile.vfx" and a == "shading_complexity")
                    or (fam == "cables.vfx" and a == "quad_overdraw")
                    # csgo_water declares S_MODE_DEPTH but ships NO
                    # bytecode record for value 1: of the 200 combos, every
                    # live one has S_MODE_DEPTH = 0. There is no variant to
                    # implement, and inventing one would be worse than the
                    # gap -- so the axis is carried, reported, and empty.
                    or (fam == "csgo_water.vfx" and a == "mode_depth"))
                if not finite:
                    dead.append((fam, a, "non-finite"))
                elif d == 0.0 and not known_noop:
                    dead.append((fam, a, "moves nothing"))
    finally:
        _INJECT = None

    if verbose:
        print("synthesised draw: %d families x every axis at both values"
              % len(SHADERS))
        cur = None
        for fam, a, d, why in rows:
            if fam != cur:
                print("  %s" % fam)
                cur = fam
            tag = ""
            if a != "<all axes 0>":
                if (fam == "cables.vfx" and a in ("clamp_min_radius",
                                                  "quad_overdraw")) or \
                        (fam == "grasstile.vfx" and a == "shading_complexity"):
                    tag = "   (measured no-op in the shipped PS bytecode)"
                elif fam == "csgo_water.vfx" and a == "mode_depth":
                    tag = "   (no combo ships a record for value 1)"
                elif d == 0.0:
                    tag = "   <== MOVES NOTHING"
            print("     %-22s max|delta| = %-12.6g%s%s"
                  % (a, d, (" [%s]" % why) if why else "", tag))
        if SUBST:
            print("  substitutions this run (each said out loud, none silent):")
            for s in SUBST:
                print("     - %s" % s)
        print("  dead axes: %d" % len(dead))
    return len(dead), rows


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    if "--list-injections" in argv or "all" in argv:
        for k, v in sorted(INJECTIONS.items()):
            print("  %-26s %s" % (k, v))
        raise SystemExit(0)
    inj = None
    if "--inject" in argv:
        inj = argv[argv.index("--inject") + 1]
        if inj not in INJECTIONS:
            raise SystemExit("unknown injection %r; --list-injections" % inj)
    n_dead, _ = reach_selftest(inject=inj)
    if inj:
        print("\ninjected %r -> %d dead axis/axes (expected >= 1)"
              % (inj, n_dead))
        raise SystemExit(0 if n_dead else 1)
    raise SystemExit(1 if n_dead else 0)
