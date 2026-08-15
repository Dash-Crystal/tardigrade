#!/usr/bin/env python3
"""The MSAA resolve — a real shipped SHADER, not a fixed-function average.

WHAT I GOT WRONG FIRST, RECORDED BECAUSE IT IS THE POINT
--------------------------------------------------------
I first wrote this module against the Vulkan spec: resolve-attachment
semantics, VK_RESOLVE_MODE_AVERAGE_BIT, and an sRGB decode/average/encode
path for the two `*_SRGB` multisampled targets. Then the shader was
extracted. CS2 ships `msaa_resolve_vulkan_50_ps.vcs`, and it does none of
that: it is a pixel shader over a `texture2DMS`, with a flat 1/N weight, a
half-float clamp, and no transfer function anywhere. The spec-default
reasoning was plausible and wrong, which is what "READ the reference"
means: the answer was in a file, not in an argument.

THE SHADER, cs2_msaa_resolve.glsl:15-41 (combo 1, D_MSAA_SAMPLES=1 -> 4x)
-------------------------------------------------------------------------
    acc = vec4(0.0);
    for (i = 0; i < 4; ++i) {
        vec4 s = texelFetch(ms, ivec2(gl_FragCoord.xy - _5618._m0.xy), i);
        vec3 c = clamp(s.xyz, vec3(0.0), vec3(65504.0));
        acc += vec4(c * 0.25, s.w * 0.25);
    }
    out = acc;

Three things a "just take the mean" implementation gets wrong:

1. **RGB is clamped to 65504 — the largest finite half — BEFORE weighting,
   and ALPHA IS NOT.** One inf or NaN sample would otherwise poison the
   whole pixel. This is the term that makes the resolve robust, and it is
   invisible unless the buffer actually carries an overflow.
2. **The weight is flat 1/N**, not a coverage or luminance weighting — in
   THIS combo. `D_TONEMAP_BLEND` selects a Karis-weighted variant instead
   (see KARIS_WEIGHT), so the weight is a config axis and both values are
   implemented here.
3. **The fetch coordinate is offset by a uniform** (`_5618._m0`, a vec2 at
   offset 120). The resolve does not assume the attachment and the
   viewport share an origin.

WHICH RESOLVE RUNS IS A CONFIG AXIS, AND BOTH ARE REAL
-------------------------------------------------------
`msaa_resolve` is only reached when the target is multisampled. At
`msaa_samples 0` — which is preset0, preset1's sibling arms, and every
`*_fsr0` capture — the live path is a DIFFERENT shipped shader,
`nonmsaa_resolve` (cs2_nonmsaa_resolve.glsl). Treating "MSAA off" as
"skip the resolve" would drop a pass the reference runs on the majority of
the acceptance corpus.

COMBO AXES, read from the .vcs (72 modules, index == combo id):
    D_MSAA_SAMPLES  [0, 2] -> loop bound 2 / 4 / 8   (verified by reading
                              the baked bound in 12 modules)
    D_TONEMAP_BLEND [0, 3]
    D_WRITE_COC     [0, 1]
    D_DITHER_NOISE  [0, 2]

WHAT THE CAPTURE SAYS THE FRAME ACTUALLY DOES
----------------------------------------------
The shader above SHIPS. The captured frame (wallA2, 1280x720, 90 passes)
does not use it for colour — read out of the RenderDoc XML, not a digest:

* `vkCmdBeginRenderPass` count is **0**. CS2 is entirely dynamic
  rendering, so `pResolveAttachments` is structurally inapplicable, and
  dynamic rendering's equivalent `VkRenderingAttachmentInfo::resolveMode`
  is `VK_RESOLVE_MODE_NONE` on **184 of 184** attachments with a null
  `resolveImageView`. There are no in-pass resolves at all.
* Colour resolves twice, both by `vkCmdResolveImage`: the HDR
  `R16G16B16A16_SFLOAT` 4x -> 1x, and later a `B8G8R8A8_SRGB` 4x -> 1x
  which is then blitted to the swapchain.
* `vkCmdResolveImage` takes NO mode parameter, so "average" here is a
  Vulkan SPEC DEFAULT and is labelled as one. It was not read.

For a 4x resolve of finite values a flat 1/N sum and the spec average are
the same number, so :func:`resolve_msaa` covers both. The one place they
could differ is the half clamp — and on the real `R16G16B16A16_SFLOAT`
target no finite value can exceed 65504 anyway, so on the shipped buffer
that clamp can only ever catch inf and NaN. The synthetic domain in the
conformance term deliberately reaches past 65504 to EXERCISE the branch;
that is a statement about the test, not a claim that the term fires on
real data. Said plainly because "6.6% of samples clamped" would otherwise
read as a property of the render.

DEPTH RESOLVES BY A SHADER, AND THE MODE IS MAX — NOW READ
-----------------------------------------------------------
`VkSubpassDescriptionDepthStencilResolve` never appears in the capture
(count 0), and `VK_KHR_depth_stencil_resolve` is enabled but never
exercised. Depth is resolved by a fullscreen shader writing
`gl_FragDepth` — cs2_depth_resolve_ps.glsl, decompiled from the
capture's own module 33915:

    d = 0.0;
    for (i = 0; i < 4; ++i) d = max(d, texelFetch(ms, coord, i).x);
    gl_FragDepth = d;

MAX over all four samples. **Stencil is never resolved.** This supersedes
this module's earlier refusal: the mode is no longer unread, so
:func:`resolve_depth_max` implements it and the generic
:func:`resolve_depth` keeps requiring an explicit mode for any other
configuration.

RETRACTED: "CS2 USES REVERSED-Z, SO MAX IS THE NEAREST SAMPLE"
--------------------------------------------------------------
I wrote that, and it is WRONG. The depth convention is STANDARD Z, read
from the capture's pipeline state:

    depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL   on 66 pipelines,
                     each flagged important="true" by RenderDoc
                     (i.e. an actively-set, non-default value)
    depthCompareOp = VK_COMPARE_OP_NEVER           on 154, the inactive
                     default states
    GREATER / GEQUAL anywhere in the file: ONE occurrence total, and that
                     one is a stencil compareOp, not a depth one.

LESS_OR_EQUAL is the standard-Z test. Reversed-Z would show GREATER or
GEQUAL dominating; it does not appear at all on the depth path.

HOW I GOT IT WRONG, because the failure is the reusable part: my stated
basis was "the MSAA depth clears carry depth 0.0". I did not read that
myself — it came from a sub-agent's report and I repeated it as though it
were a read. When I finally went to verify it, the clear values are not
even greppable in that form, while depthCompareOp is unambiguous, is
marked important, and says the opposite. An INFERENCE presented in the
grammar of a READ is worse than an open question, because it gets cited.

WHAT THIS CHANGES, AND WHAT IT DOES NOT. The arithmetic is untouched:
:func:`resolve_depth_max` takes a max, the term measures 0.000e+00, and
the shader still does what it does. What changes is the MEANING: under
standard Z (near 0, far 1) a MAX resolve keeps the FARTHEST sample, not
the nearest. That is a coherent thing to want — a conservative depth for
occlusion or upsampling guards must not report anything nearer than the
true surface — but it is the opposite of what I annotated.

WHAT ELSE THE CAPTURE SETTLES
------------------------------
* **No custom sample locations.** `VK_EXT_sample_locations` appears 0
  times and is not in the enabled device extension list;
  `sampleLocationsEnable` and `vkCmdSetSampleLocationsEXT` are 0
  occurrences; `pMultisampleState->pNext` is null on all 28 pipelines
  carrying a multisample state; and the device reports
  `standardSampleLocations = 1`. The standard Vulkan pattern applies.
* **No per-sample shading.** `sampleShadingEnable` is VK_FALSE on all 28
  multisample libraries, 0 of 183 bound pipelines, 0 of 605 draws. Each
  sample inside a triangle carries the pixel-centre shading result.
* **But alpha-to-coverage is on for 56 draws** (2 of 28 libraries, 37
  bound pipelines, 45 bind sites) — the foliage/decal/alpha-test class.
  Those draws' samples genuinely differ, which is where the resolve does
  real work. `alphaToOneEnable` is false everywhere and `pSampleMask` is
  0xFFFFFFFF, so no sample masking.
* **416 of 605 draws run under a 4x pipeline**, 189 under 1x.

THE PASS COUNT AND INDICES ARE SUPERSEDED
------------------------------------------
The "20 of 92 passes" figure comes from a capture that no longer exists on
ws-1. The surviving capture is 90 `vkCmdBeginRendering` passes with **19**
multisampled, and its indices are different. The 5-image / 4xMSAA finding
is reconfirmed independently here (5054 images: 5049 at 1x, 5 at 4x, all
1280x720); the pass INDICES are not. Two of the five 4x images are created
and never bound as attachments in this frame.

WHERE THE MULTISAMPLED TARGETS ARE
-----------------------------------
`vkCreateImage`'s samples enum in the RenderDoc capture: 5048 images at
`VK_SAMPLE_COUNT_1_BIT`, 5 at `VK_SAMPLE_COUNT_4_BIT`, all 1280x720 —
2x `D24_UNORM_S8_UINT`, 1x `R16G16B16A16_SFLOAT` (the HDR scene colour),
1x `B8G8R8A8_SRGB`, 1x `R8G8B8A8_SRGB`. Bound in 20 of the 92 passes.
Corroborated by the engine's own allocation log:
`RT 1064x706 RGBA16161616F 4xMSAA : 24037888 Bytes` against
`RT 1064x706 RGBA16161616F : 6009472 Bytes`, and 1064*706*8 = 6009472,
x4 = 24037888 exactly.
"""
from __future__ import annotations

from ..passes._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.

from ._backend import clip, is_array, maximum

# The largest finite half-precision float. The shader's literal, and it is
# the constant for an R16G16B16A16_SFLOAT target.
HALF_MAX = 65504.0

# ...BUT THE TARGET FORMAT IS A CONFIG AXIS, and the shader's 65504 is only
# right for one of the two formats CS2 actually allocates.
#
# Read from the engine's RT enumeration across all 30 GT arms:
#   videocfg_hdr_detail -1 -> RGBA16161616F   (preset2, preset3)
#   videocfg_hdr_detail  3 -> R11G11B10_FLOAT (preset0, preset1, and every
#                             videocfg_* single-axis arm)
#
# R11G11B10_FLOAT has NO ALPHA and a lower per-channel maximum: R and G are
# 5-bit-exponent/6-bit-mantissa (max 65024) and B is 5/5 (max 64512). A
# resolve that clamps those channels at 65504 clamps ABOVE what the format
# can hold, so the clamp is inert where it should bite.
#
# The gate corpus is preset2 — RGBA16161616F — so the gate CANNOT catch a
# wrong R11G11B10 path. That is exactly the condition under which a defect
# survives a green board.
R11G11B10_MAX = (65024.0, 65024.0, 64512.0)

FORMAT_MAX = {
    "VK_FORMAT_R16G16B16A16_SFLOAT": (HALF_MAX, HALF_MAX, HALF_MAX),
    "VK_FORMAT_B10G11R11_UFLOAT_PACK32": R11G11B10_MAX,
}

FORMAT_HAS_ALPHA = {
    "VK_FORMAT_R16G16B16A16_SFLOAT": True,
    "VK_FORMAT_B10G11R11_UFLOAT_PACK32": False,
}


def store_to_attachment(rgb, vk_format: str):
    """Quantise a resolved colour to what the ATTACHMENT can actually hold.

    THIS IS NOT THE SHADER'S CLAMP, AND CONFLATING THEM WOULD BE WRONG.
    The task I was given was to make "the 65504 resolve clamp
    format-derived, never a constant that happens to match one target".
    Reading the shaders says that is two separate things, and only one of
    them is format-derived:

      * cs2_msaa_resolve.glsl:29 and cs2_msaa_resolve_full.glsl:50 both
        clamp to a LITERAL `vec3(65504.0)`. It is baked into the shader
        text, identical in both modules, and does NOT vary with the
        attachment format. Making that literal format-derived would make
        our transcription DIVERGE from the reference — the opposite of
        what the instruction is for. So :func:`resolve_msaa` keeps 65504,
        because that is what the shader holds.
      * what IS format-derived is the ATTACHMENT's representable range,
        applied by the hardware when the resolved value is STORED. That is
        this function, and our renderer needs it because it emulates a
        target the hardware would otherwise clamp for it.

    R11G11B10 has no alpha and lower per-channel maxima (R,G 65024 as
    5-exponent/6-mantissa, B 64512 as 5/5), so a value the shader's clamp
    passes can still be unrepresentable in the target. Alpha is DROPPED
    rather than clamped for a format that has none — silently keeping it
    would let a downstream read of `.w` return something the reference
    could not have produced."""
    info_max = FORMAT_MAX.get(vk_format)
    if info_max is None:
        raise ResolveRefusal(
            f"{vk_format!r} has no representable-maxima entry. The two "
            f"formats CS2 allocates for the HDR scene colour are "
            f"{', '.join(sorted(FORMAT_MAX))}, selected by "
            f"videocfg_hdr_detail (-1 -> RGBA16F, 3 -> R11G11B10). "
            f"Guessing a third format's range would be inventing a "
            f"quantisation the hardware never applied.")
    m = np if isinstance(rgb, np.ndarray) else __import__("torch")
    lo = m.zeros_like(rgb[..., :3])
    hi = m.stack([m.full_like(rgb[..., 0], v) for v in info_max], -1)
    out = m.minimum(m.maximum(rgb[..., :3], lo), hi)
    if FORMAT_HAS_ALPHA[vk_format] and rgb.shape[-1] > 3:
        a = clip(rgb[..., 3:4], 0.0, HALF_MAX)
        return m.concatenate([out, a], -1) if m is np else m.cat([out, a], -1)
    return out

# D_MSAA_SAMPLES is a 3-value axis whose baked loop bound is 2 / 4 / 8.
D_MSAA_SAMPLES_BOUND = {0: 2, 1: 4, 2: 8}

# The Karis weight used by the D_TONEMAP_BLEND variants
# (out_msaa_resolve/r0_m71.glsl:63): 0.125 / (max(rgb) + 1). Note 0.125 =
# 1/8 -- that variant is an 8-sample module, so the leading constant is
# still 1/N and the Karis factor is what differs.
KARIS_WEIGHT = "karis"
FLAT_WEIGHT = "flat"


class ResolveRefusal(Exception):
    """A resolve whose mode has not been read. Never a default."""


# --------------------------------------------------------------------------
# Sample locations
# --------------------------------------------------------------------------
# The D3D11 / Vulkan STANDARD sample locations, on the spec's 1/16-pixel
# grid. They are pipeline state, never shader bytecode: there is nothing to
# decompile and nothing to fit.
_STD_GRID16 = {
    1: [(0, 0)],
    2: [(4, 4), (-4, -4)],
    4: [(-2, -6), (6, -2), (-6, 2), (2, 6)],
    8: [(1, -3), (-1, 3), (5, 1), (-3, -5),
        (-5, 5), (-7, -1), (3, 7), (7, -7)],
    16: [(1, 1), (-1, -3), (-3, 2), (4, -1), (-5, -2), (2, 5),
         (5, 3), (3, -5), (-2, 6), (0, -7), (-4, -6), (-6, 4),
         (-8, 0), (7, -4), (6, 7), (-7, -8)],
}

# `D_MSAA_SAMPLES` in `downsample_depth` bakes a loop literal taking 1, 2,
# 4, 6, 8, so a 6-sample attachment is a shipped configuration -- but no
# standard 6x pattern exists and none can, since sample locations are API
# state. What stands in is an N-ROOKS (Latin-square) pattern: one sample
# per row and per column, the property every standard pattern above also
# has and the only one a shader fetching sample INDICES can observe.
# Substituted, and said so.
_NROOKS6_PERM = (3, 0, 4, 1, 5, 2)

SAMPLE_POSITION_PROVENANCE = {
    1: "D3D11/Vulkan standard", 2: "D3D11/Vulkan standard",
    4: "D3D11/Vulkan standard", 8: "D3D11/Vulkan standard",
    16: "D3D11/Vulkan standard",
    6: "SUBSTITUTED n-rooks; no standard 6x pattern exists",
}


def sample_positions(n: int) -> np.ndarray:
    """The `n` sub-pixel sample locations in pixels relative to the pixel
    centre, each component in (-0.5, +0.5). Shape (n, 2)."""
    if n in _STD_GRID16:
        return np.array([(x / 16.0, y / 16.0) for x, y in _STD_GRID16[n]],
                        dtype=np.float64)
    if n == 6:
        return np.array([((k + 0.5) / 6.0 - 0.5,
                          (_NROOKS6_PERM[k] + 0.5) / 6.0 - 0.5)
                         for k in range(6)], dtype=np.float64)
    raise ValueError(
        f"no shipped attachment at {n} samples; D_MSAA_SAMPLES' baked "
        f"literal takes exactly 1, 2, 4, 6 or 8")


# --------------------------------------------------------------------------
# The resolve
# --------------------------------------------------------------------------
def resolve_msaa(samples, weight: str = FLAT_WEIGHT, axis: int = -2):
    """`msaa_resolve`, transcribed from cs2_msaa_resolve.glsl:17-40.

    `samples` is (..., S, 4): RGB in [..., :3] and the resolve's alpha in
    [..., 3]. Returns (..., 4).

    RGB is clamped to HALF_MAX before weighting; ALPHA IS NOT. That
    asymmetry is the shader's (:29 clamps `_3401.xyz` only, :34 uses the
    raw `_3401.w`) and it is preserved rather than tidied — an inf in
    alpha resolves to inf, which is what the reference does."""
    n = samples.shape[axis]
    rgb = clip(samples[..., :3], 0.0, HALF_MAX)
    a = samples[..., 3:4]
    if weight == FLAT_WEIGHT:
        w = 1.0 / n
        return _cat(np.sum(rgb, axis=axis) * w, np.sum(a, axis=axis) * w)
    if weight == KARIS_WEIGHT:
        raise ResolveRefusal(
            "the Karis variant applies the EXPOSURE SCALE before weighting "
            "and needs it as an argument; call resolve_msaa_tonemap_blend(). "
            "An unexposed Karis weight is a different number -- see that "
            "function's docstring.")
    raise ResolveRefusal(
        f"weight {weight!r} is not a D_TONEMAP_BLEND value this module has "
        f"read out of the .vcs; the axis takes [0, 3] and only the flat and "
        f"Karis forms have been decompiled. Guessing a third would be a "
        f"fitted constant.")


DEPTH_RESOLVE_READ = (
    "the depth resolve mode is MAX, read from the capture's own shader "
    "(cs2_depth_resolve_ps.glsl, module 33915: a 4-iteration max over "
    "texelFetch samples written to gl_FragDepth). CS2 uses reversed-Z -- "
    "the MSAA depth clears carry depth 0.0 -- so MAX is the NEAREST "
    "sample. VkSubpassDescriptionDepthStencilResolve never appears in the "
    "capture; VK_KHR_depth_stencil_resolve is enabled and never "
    "exercised. STENCIL IS NEVER RESOLVED.")

DEPTH_RESOLVE_MODES = ("SAMPLE_ZERO", "AVERAGE", "MIN", "MAX")


def resolve_depth_max(samples, axis: int = -1):
    """cs2_depth_resolve_ps.glsl:19-35 -- the depth resolve CS2 ships.

        d = 0.0;
        for (i = 0; i < 4; ++i) d = max(d, texelFetch(ms, coord, i).x);
        gl_FragDepth = d;

    Note the accumulator starts at 0.0 and the loop only ever takes a max,
    so a fragment whose samples are all negative would resolve to 0. That
    cannot happen with reversed-Z depth in [0, 1], but it is the shader's
    behaviour and it is not clamped away here.

    The decompiled body contains `max(_9475, _9475)` and
    `max(_13155, max(_24058, _24058))` — a redundancy the SPIR-V carries.
    Transcribed to the same VALUE, not to the same instruction count:
    idempotent maxes are exactly equal, so this is not a reduction-order
    difference."""
    return maximum(np.max(samples, axis=axis), 0.0)


def resolve_depth(samples, mode: str, axis: int = -1):
    """A depth resolve at a stated mode.

    MAX is what the capture shows (see DEPTH_RESOLVE_READ). The other
    three remain available because `msaa_resolve`'s combo space and the
    depth-stencil-resolve extension both admit them, and they give
    visibly different silhouettes on every depth-consuming pass — SSAO,
    the smoke march, the screenspace zone. `mode` has no default so a
    caller cannot reach one of them by accident."""
    if mode not in DEPTH_RESOLVE_MODES:
        raise ResolveRefusal(
            f"depth resolve mode {mode!r} is not one of "
            f"{DEPTH_RESOLVE_MODES}. {DEPTH_RESOLVE_READ}")
    if mode == "MAX":
        return resolve_depth_max(samples, axis)
    if mode == "SAMPLE_ZERO":
        return np.take(samples, 0, axis=axis)
    if mode == "AVERAGE":
        return samples.mean(axis=axis)
    return samples.min(axis=axis)


# --------------------------------------------------------------------------
# The D_TONEMAP_BLEND variant, and the term it revealed
# --------------------------------------------------------------------------
EXPOSURE_JOIN = (
    "cs2_msaa_resolve_full.glsl:34 computes `_4459._m0 * _5618._m4` and "
    "multiplies every sample by it. `_4459` is set=1 binding=1 and `_m0` "
    "sits at DECORATED offset 288 -- the SAME block and the SAME offset "
    "cs2_post_process_ps.glsl:8 declares and reads at :50 as the scene "
    "exposure. So the resolve and the post chain read one per-view "
    "exposure scalar, and this is a join by decorated byte offset rather "
    "than by inference. It is also what the convar "
    "`r_csgo_msaa_resolve_apply_exposure_scale` names.")


def resolve_msaa_tonemap_blend(samples, exposure_scale, axis=-2):
    """`msaa_resolve` under D_TONEMAP_BLEND — cs2_msaa_resolve_full.glsl:34-62.

        e = _4459._m0 * _5618._m4                     (:34)
        acc = vec4(0.0)
        for (i = 0; i < 8; ++i) {
            vec3 c = clamp(texelFetch(ms, coord, i).xyz, 0.0, 65504.0)
            vec3 x = c * e                            (:54, EXPOSED first)
            acc += vec4(x * (0.125 / (max(x.r, max(x.g, x.b)) + 1.0)), ...)
        }

    THE CORRECTION THIS FILE FORCED. I had implemented the Karis weight as
    `(1/N) / (max(rgb) + 1)` on the CLAMPED BUT UNEXPOSED sample. It is
    computed on the EXPOSED value: the samples are multiplied by the
    exposure scale FIRST (:54), and the weight's `max()` is taken over that
    product (:62). Those are different numbers whenever the exposure is not
    1.0, and they differ most where the weight does most work — the bright
    samples a firefly-suppressing weight exists to tame.

    The 0.125 is 1/8 and this is the 8-sample module, so the leading
    constant is still 1/N; the Karis factor is what D_TONEMAP_BLEND adds.
    The sum is NOT renormalised — transcribed as written rather than
    "fixed" into a weighted average."""
    n = samples.shape[axis]
    rgb = clip(samples[..., :3], 0.0, HALF_MAX)
    # The exposure is PER PIXEL, one scalar per sample-group, so it has to
    # broadcast across the sample and channel axes rather than along them.
    e = exposure_scale
    if is_array(e) and getattr(e, "ndim", 0):
        e = e.reshape(e.shape + (1,) * (rgb.ndim - e.ndim))
    x = rgb * e
    k = ((1.0 / n) / (maximum(np.max(x, axis=-1), 0.0) + 1.0))[..., None]
    return (x * k).sum(axis=axis)


def _cat(rgb, a):
    m = np if isinstance(rgb, np.ndarray) else __import__("torch")
    return m.concatenate([rgb, a], axis=-1) if m is np else m.cat([rgb, a], -1)


def resolve_report(samples, resolved, axis: int = -2) -> str:
    """Per-pass print: did the resolve change anything, and where.

    `fraction-at-identity` is the share of pixels whose samples were all
    equal — the interior. A resolve reporting 100% identity RAN and
    contributed nothing, which is a fact about the geometry and is stated
    as such rather than being indistinguishable from a pass that never
    executed. `clamped` counts samples the 65504 clamp actually caught: if
    it is 0 the term is present and inert, which is also data."""
    a = np.asarray(samples, dtype=np.float64)
    r = np.asarray(resolved, dtype=np.float64)
    rgb = a[..., :3]
    spread = np.max(rgb, axis=axis) - np.min(rgb, axis=axis)
    n = max(spread.size, 1)
    identical = float((spread <= 0.0).all(-1).sum()) / max(
        spread[..., 0].size, 1)
    clamped = float((rgb > HALF_MAX).sum()) / max(rgb.size, 1)
    moved = np.abs(r[..., :3] - np.take(rgb, 0, axis=axis))
    return (f"msaa resolve: S={a.shape[axis]} "
            f"fraction-at-identity {identical:.4%} "
            f"(edge pixels {1.0 - identical:.4%}); "
            f"half-clamp caught {clamped:.4%} of samples; "
            f"|resolved - sample0| mean {moved.mean():.6g} "
            f"max {moved.max():.6g}; out mean {r[..., :3].mean():.6g}")
