#!/usr/bin/env python3
"""The post chain, in the order `cs2_post_process_ps.glsl` runs it.

THE ORDER, READ NOT INFERRED (cs2_post_process_ps.glsl:46-136, combo 36 =
`D_BLOOM=1, D_COLOR_CORRECTION_LUT=1`, every other axis 0):

    :48  uv          = gl_FragCoord.xy * invViewport
    :49  scene       = textureLod(sceneColor, uv * _5618._m0)
    :50  exposure    = _4459._m0 * _4349._m0
    :51  scene      *= exposure
    :56  bloom       = textureLod(bloomTex, min(uv, _m1.zw) * _m1.xy)
    :57  scene       = scene*(1 - _m4.z) + bloom*(_m3.z * exposure)
    :62  scene      += bloom * _m3.x
    :67  x           = min(scene * 2.8, vec3(_4349._m7))
    :68  hable       = <the curve, coefficients from _4349>
    :71  srgb        = TRUE sRGB, per channel, cutoff 0.0031308
    :104 bloomE      = bloom * exposure
    :109 bloomC      = clamp((bloomE/(bloomE + 0.187)) * 1.035, 0, 1)
    :114 bloomC     *= _m3.y
    :119 out         = (srgb + bloomC) - srgb*bloomC        <- SCREEN
    :124 out         = mix(LUT(out*0.96875 + 0.015625), out, _m2)
    :129 out         = mix(out, mix(_m5.rgb, _m5.rgb*out, _m6), _m5.w)
    :130 out         = pow(out, _m7 * 0.454545438289642333984375)
    :135 alpha       = the SCENE's alpha, untouched from :49

FIVE THINGS THIS SETTLES THAT WERE PREVIOUSLY ARGUED
-----------------------------------------------------
1. **The bloom is composited TWICE and on both sides of the curve.** Once
   in linear before the tonemap (:57, :62 — the lens/veil term) and once
   in display space after the sRGB encode (:119 — the screen blend). Every
   previous "is bloom before or after the curve" verdict picked one; the
   shader does both, with different weights (`_m3.x`/`_m3.z` linear,
   `_m3.y` display).
2. **The blend is a SCREEN, `(a+b) - a*b`,** despite the .vpost naming
   `BLOOM_BLEND_ADD`. The name is not the operation.
3. **The colour-correction mix runs the OTHER WAY.** `mix(lut, c, _m2)`
   weights toward the UNCORRECTED value, so `_m2` is the amount of
   original, not the amount of LUT. Our `--cc-amount` had the complement,
   which silently inverts the axis.
4. **The LUT fetch coefficients are 0.96875 and 0.015625** — that is
   31/32 and 0.5/32, i.e. `(c*(N-1) + 0.5)/N` at N=32, the half-texel
   convention, and the 32 is confirmed by the literals themselves rather
   than by the asset's dimensions.
5. **There IS a tint term** (:129), which POSTPROCESS_FINDINGS.md:142-147
   lists under "Not implemented, flagged not guessed". It is a nested mix:
   an inner one between a flat tint colour and a modulated tint, weighted
   by `_m6`, and an outer one back toward the untinted value weighted by
   `_m5.w`.

WHERE THIS SITS: RESOLVE FIRST, THEN TONEMAP — READ FROM THE CAPTURE
---------------------------------------------------------------------
This chain runs on a SINGLE-SAMPLED image. The order was an open question
and the capture answers it, by barrier and by shader declaration:

    scene renders into MSAA HDR (image 82012, R16G16B16A16_SFLOAT 4x)
    -> barrier COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL
    -> vkCmdResolveImage 82012 (4x) -> 82055 (1x), 1280x720
    -> barrier TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    -> the bloom/downsample chain, every target 1x
    -> THIS PASS (execution position 77 of 90)
    -> writes into a 4x B8G8R8A8_SRGB target
    -> 2 HUD draws
    -> vkCmdResolveImage that 4x sRGB -> 1x -> blit to the swapchain

The decisive evidence is not the barrier order, which could be argued
about; it is that the tonemap's own fragment module declares `texture2D`
and `texture3D` and NO `texture2DMS`, no `subpassInputMS`, no
`gl_SampleID`. A shader that declares no multisampled sampler cannot read
multisample data, so every input to this pass is already resolved. That
holds without tracing a single bindless descriptor.

The SECOND multisampled target is not a second HDR resolve. The tonemap
writes a fullscreen triangle with `sampleShadingEnable = VK_FALSE`, so it
writes the same value to all four samples of every pixel; that target's
samples diverge only where the two HUD draws rasterise. It exists to
antialias the UI. A pipeline compositing an unaliased HUD does not need
it, and saying so is a scope statement, not a deferral.

NOTHING HERE IS GATED
---------------------
Every term above runs on every frame. `_5618._m2`, `_m5.w`, `_m3.*` and
`_m6` are the VALUES that decide how much each contributes, and they come
from the config. A zero weight makes a term contribute nothing while still
executing — which is what the reference does, and is why the runtime
report below prints each stage's mean and fraction-at-identity rather than
a flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._backend import clip, maximum, minimum
from .tonemap import (TM_PRESCALE, hable, hable_white, srgb_encode,
                      TM_A, TM_B, TM_C, TM_D, TM_E, TM_F, TM_WHITE_POINT)

# :109 -- the D_BLOOM_MODE 0 curve's two literals.
BLOOM_REINHARD_K = 0.1870000064373016357421875
BLOOM_REINHARD_SCALE = 1.03499996662139892578125

# :124 -- the LUT fetch, N=32. 0.96875 = 31/32, 0.015625 = 0.5/32.
LUT_SCALE = 0.96875
LUT_BIAS = 0.015625
LUT_N = 32

# :130 -- 1/2.2, to the bit.
INV_2_2 = 0.454545438289642333984375


@dataclass
class PostParams:
    """The uniforms the chain reads, by accessor path.

    Named after the decompiled member, not after a guessed engine variable
    name: `csgo_glass`'s callflow was caught asserting names and offsets
    that the artifact contradicts, and the same discipline applies here
    even though THIS shader does carry `layout(offset = N)` decorations
    (:8-22) — the offsets are read, the NAMES are not."""
    # _4349 block -- the tonemap parameters. Defaults are the de_inferno
    # .vpost's decoded values; they are READ constants, not fallbacks.
    view_exposure: float = 1.0        # _4459._m0
    post_exposure: float = 1.0        # _4349._m0
    tm_a: float = TM_A                # _4349._m1  ShoulderStrength
    tm_b: float = TM_B                # _4349._m2  LinearStrength
    tm_c: float = TM_C                # _4349._m3  LinearAngle
    tm_d: float = TM_D                # _4349._m4  ToeStrength
    tm_e: float = TM_E                # _4349._m5  ToeNum
    tm_f: float = TM_F                # _4349._m6  ToeDenom
    tm_clamp: float = 65504.0         # _4349._m7  the min() ceiling
    tm_white_recip: float = field(default=0.0)   # _4349._m8 = 1/F(W)

    # _5618 block
    bloom_linear: float = 0.0         # _m3.x  linear-side bloom weight
    bloom_display: float = 0.0        # _m3.y  display-side (screen) weight
    bloom_veil: float = 0.0           # _m3.z  pre-curve veil weight
    scene_attenuation: float = 0.0    # _m4.z  scene weight complement
    cc_keep_original: float = 0.0     # _m2    mix() weight toward UNcorrected
    tint_rgb: tuple = (0.0, 0.0, 0.0)  # _m5.xyz
    tint_amount: float = 0.0          # _m5.w  outer mix weight
    tint_modulate: float = 0.0        # _m6    inner mix weight
    final_gamma: float = 1.0          # _m7    exponent numerator

    def __post_init__(self):
        if not self.tm_white_recip:
            self.tm_white_recip = 1.0 / hable_white(TM_WHITE_POINT)


def hable_uniform(x, tm_a=TM_A, tm_b=TM_B, tm_c=TM_C, tm_d=TM_D,
                  tm_e=TM_E, tm_f=TM_F, tm_white_recip=None):
    """:68-69 -- the curve with its coefficients taken from the uniform
    block rather than baked.

    The coefficients are SCALAR ARGUMENTS, not a params object. A kernel
    whose signature takes a dataclass cannot be driven by the conformance
    generator, and a term the A/B cannot reach is a term whose conformance
    is an assertion — so the params object unpacks into this call rather
    than being passed through it."""
    if tm_white_recip is None:
        w = ((TM_WHITE_POINT * (tm_a * TM_WHITE_POINT + tm_c * tm_b)
              + tm_d * tm_e)
             / (TM_WHITE_POINT * (tm_a * TM_WHITE_POINT + tm_b) + tm_d * tm_f)
             ) - tm_e / tm_f
        tm_white_recip = 1.0 / w
    ax = x * tm_a
    num = x * (ax + tm_c * tm_b) + tm_d * tm_e
    den = x * (ax + tm_b) + tm_d * tm_f
    return (num / den - tm_e / tm_f) * tm_white_recip


def bloom_curve_mode0(bloom_exposed):
    """:109 -- `clamp((b / (b + 0.187)) * 1.035, 0, 1)`.

    A Reinhard-shaped curve on the BLOOM ALONE, on the exposed value, and
    NOT the scene's Hable curve and NOT sRGB-encoded. Both D_BLOOM_MODE
    variants screen, so the mode selects this curve, not the blend."""
    b = maximum(bloom_exposed, 0.0)
    return clip((b / (b + BLOOM_REINHARD_K)) * BLOOM_REINHARD_SCALE, 0.0, 1.0)


def screen_blend(a, b):
    """:119 -- `(a + b) - a*b`. The .vpost says BLOOM_BLEND_ADD; the
    shader screens. The name is not the operation."""
    return (a + b) - a * b


def apply_cc_lut(c, lut: Optional[np.ndarray], keep_original: float):
    """:124 -- `mix(LUT(clamp(c,0,1)*0.96875 + 0.015625), c, _m2)`.

    NOTE THE DIRECTION: `_m2` weights toward the UNCORRECTED value. A
    `cc_amount` that weights toward the LUT is the complement and inverts
    the axis. `lut` is [B][G][R][3]; None means the pass runs with the
    identity, which is what a LUT of the identity ramp would produce —
    stated so a missing asset is visible rather than silently skipping the
    stage."""
    src = clip(c, 0.0, 1.0)
    if lut is None:
        return src
    coord = src * LUT_SCALE + LUT_BIAS          # in [0,1], half-texel inset
    g = coord * (LUT_N - 1)
    i0 = np.floor(g).astype(np.int64)
    i1 = np.minimum(i0 + 1, LUT_N - 1)
    f = g - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    def L(bi, gi, ri):
        return lut[bi, gi, ri]

    c00 = L(b0, g0, r0) * (1 - fr) + L(b0, g0, r1) * fr
    c01 = L(b0, g1, r0) * (1 - fr) + L(b0, g1, r1) * fr
    c10 = L(b1, g0, r0) * (1 - fr) + L(b1, g0, r1) * fr
    c11 = L(b1, g1, r0) * (1 - fr) + L(b1, g1, r1) * fr
    out = ((c00 * (1 - fg) + c01 * fg) * (1 - fb)
           + (c10 * (1 - fg) + c11 * fg) * fb)
    return out + (src - out) * keep_original


def apply_tint(c, p: PostParams):
    """:129 -- the nested tint mix.

        mix(c, mix(tint, tint * c, vec3(_m6)), vec3(_m5.w))

    The inner mix chooses between a FLAT tint colour and the tint
    MODULATING the image; the outer one weights that against leaving the
    image alone. Listed as 'not implemented, flagged not guessed' in
    POSTPROCESS_FINDINGS.md; it is in the shader and it is implemented."""
    t = np.asarray(p.tint_rgb, dtype=np.float64) if isinstance(c, np.ndarray) \
        else p.tint_rgb
    inner = t + (t * c - t) * p.tint_modulate
    return c + (inner - c) * p.tint_amount


def post_chain(scene_rgba, bloom_rgb, p: PostParams,
               lut: Optional[np.ndarray] = None):
    """The whole chain, :48-135. `scene_rgba` is (..., 4); the alpha is the
    SCENE's and passes through untouched (:135)."""
    e = p.view_exposure * p.post_exposure

    scene = scene_rgba[..., :3] * e                              # :51
    x = scene * (1.0 - p.scene_attenuation) \
        + bloom_rgb * (p.bloom_veil * e)                         # :57
    x = x + bloom_rgb * p.bloom_linear                           # :62
    x = minimum(x * TM_PRESCALE, p.tm_clamp)                     # :67
    x = hable_uniform(x, p.tm_a, p.tm_b, p.tm_c, p.tm_d, p.tm_e,
                      p.tm_f, p.tm_white_recip)                  # :68
    srgb = srgb_encode(x)                                        # :71-99

    b = bloom_curve_mode0(bloom_rgb * e) * p.bloom_display       # :104-114
    out = screen_blend(srgb, b)                                  # :119
    out = apply_cc_lut(out, lut, p.cc_keep_original)             # :124
    out = apply_tint(out, p)                                     # :129
    out = maximum(out, 0.0) ** (p.final_gamma * INV_2_2)         # :130

    a = scene_rgba[..., 3:4]
    return np.concatenate([out, a], -1) if isinstance(out, np.ndarray) \
        else __import__("torch").cat([out, a], -1)


def post_chain_report(stage_name: str, before, after) -> str:
    """Per-stage runtime print. `fraction-at-identity` is the share of
    values the stage left EXACTLY unchanged: a stage that reports 100%
    ran and contributed nothing, which is data about the weights, not
    evidence the stage is absent."""
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    d = np.abs(a - b)
    n = max(d.size, 1)
    return (f"post/{stage_name}: mean {a.mean():.6g} "
            f"fraction-at-identity {float((d == 0.0).sum()) / n:.4%} "
            f"|delta| mean {d.mean():.6g} max {d.max():.6g}")
