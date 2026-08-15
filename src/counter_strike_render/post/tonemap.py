#!/usr/bin/env python3
"""Exposure, the Hable curve, and the sRGB transfer function.

WHAT REPLACED `--pre-exposure` — AND THE CORRECTION THE SHADER FORCED
----------------------------------------------------------------------
`--pre-exposure` was a FITTED scalar (2.05, and 2.2 in older scripts) whose
job was to put our arbitrary lighting units on the tonemap curve's scale.
POSTPROCESS_FINDINGS.md:135-141 flags it as such and asserts "The scene
exposure multiply is not in this shader — only a hardcoded x2.8 on the
tonemap input". **That assertion is WRONG, and the extracted shader says
so.** cs2_post_process_ps.glsl:50-51 is:

    float _5099 = _4459._m0 * _4349._m0;     // THE scene exposure
    vec3  _9407 = _10875.xyz * _5099;        // applied to the scene fetch

A per-view scalar times a post-parameter scalar, applied to the scene
colour at :51 and to the bloom at :104 — a real exposure multiply, in this
shader, exactly where the fitted flag was standing in. The 2.8 at :67 is a
SEPARATE, later term (`min(rgb * 2.8, vec3(_4349._m7))`), and treating it
as the exposure conflates two multiplies that sit on opposite sides of the
bloom composite. Both are implemented, separately.

THE THREE EXPOSURES ARE DIFFERENT THINGS AND ALL THREE ARE HERE
----------------------------------------------------------------
1. The SCENE EXPOSURE, `_4459._m0 * _4349._m0` (:50), applied before the
   bloom composite. :func:`scene_exposure`.
2. The SHADER's fixed pre-curve gain `TM_PRESCALE = 2.8` with its
   `min(rgb * 2.8, vec3(_4349._m7))` clamp (:67), applied after it.
3. The `post_processing_volume` entity's AUTO-EXPOSURE band —
   enableexposure true, minexposure 0.8, maxexposure 1.1, speed up/down
   1.0, fadetime 1.5 (gpu_render.py:1855-1862, decoded from the map). This
   is a temporal adaptation computed on the CPU and is one of the two
   factors of (1); :func:`adapt_exposure` is the recurrence and
   :func:`exposure_gain` its clamped output.

The fitted flag is not deprecated-with-a-warning; it is deleted, per the
standing rule that a validated replacement means the old config goes rather
than becoming an env gate.

THE HABLE COEFFICIENTS ARE UNIFORMS, NOT LITERALS, AND THEY MATCH
------------------------------------------------------------------
The curve at :68-69 reads its coefficients from the `_4349` block rather
than baking them, and the mapping onto Hable's A..F is exact:

    _21494 = x * _m1
    F(x) = ((x*(_21494 + _m3*_m2) + _m4*_m5)
          / (x*(_21494 + _m2)     + _m4*_m6)) - _m5/_m6) * _m8

  => A=_m1, B=_m2, C=_m3, D=_m4, E=_m5, F=_m6, _m7 = the clamp ceiling,
     _m8 = the white-point normaliser 1/F(W).

Substituting the .vpost's decoded values (.15/.5/.1/.2/.02/.3, W=4.0)
gives the constants below. Two independent sources — the shader's
structure and the map's parameter block — agree on the same seven numbers.

The auto-exposure was previously measured as a no-op — the clamp saturates
on all 48 frames (POSTPROCESS_FINDINGS.md:46-63). That is a statement about
our linear units against a corpus, not about the computation, so it is not
a reason to omit the term. It IS a reason to print the saturation fraction
at runtime, which :func:`exposure_report` does: a term pinned at a clamp
bound on every frame is data, not a silent pass-through.

CONSTANT PROVENANCE
-------------------
Every constant below is READ. The Hable coefficients and white point come
from `de_inferno_prefab.vpost`'s tonemap block, transcribed at
gpu_render.py:1863-1872 and POSTPROCESS_FINDINGS.md:10-12; the sRGB
constants are the shader's own (exponent 0.41666666 = 1/2.4, NOT 1/2.2);
the 2.8 and the white-point normalisation by a CPU-side `m8 = 1/F(W)` are
read from `post_process_vulkan_50_ps`.
"""
from __future__ import annotations

from ._backend import (clip, is_array as _is_array, maximum, minimum,
                       scalar_like as _scalar_like, where)

# -- de_inferno_prefab.vpost, m_bHasTonemapParams block --------------------
# The vpost's field names map 1:1 onto Hable/Uncharted-2's A..F,W.
TM_A = 0.15   # m_flShoulderStrength
TM_B = 0.50   # m_flLinearStrength
TM_C = 0.10   # m_flLinearAngle
TM_D = 0.20   # m_flToeStrength
TM_E = 0.02   # m_flToeNum
TM_F = 0.30   # m_flToeDenom
TM_WHITE_POINT = 4.0   # m_flWhitePoint -- NOT the usual 11.2
TM_EXPOSURE_BIAS = 0.0  # all three ExposureBias fields are 0.0

# -- post_process_vulkan_50_ps ---------------------------------------------
# The shader's own hardcoded gain into the curve, and the clamp that
# follows it. This is the term `--pre-exposure` was standing in for.
TM_PRESCALE = 2.8

# -- post_processing_volume (hammerUniqueId 13885), decoded from the map ----
AE_MIN = 0.8
AE_MAX = 1.1
AE_SPEED_UP = 1.0
AE_SPEED_DOWN = 1.0
AE_FADE_TIME = 1.5

# -- sRGB transfer function, THE SHADER'S LITERALS, NOT THE IDEAL VALUES ---
# These are transcribed from cs2_post_process_ps.glsl:71-99 exactly as they
# appear, because they are float32 constants and the textbook values are
# not the same numbers:
#
#     1.0/2.4 = 0.4166666666666667   the shader: 0.4166666567325592041015625
#     1.055                          the shader: 1.05499994754791259765625
#     0.055                          the shader: 0.054999999701976776123046875
#
# Using the ideal values measured max|delta| 5.215e-08 against the
# transcription. That is INSIDE the 1e-6 epsilon and it would have passed
# unnoticed -- and it is a difference we introduced by rounding the
# reference to what the constant "means" rather than to what it IS. Same
# rule as the 1e-8 floor removed earlier: the epsilon band is for float and
# reduction order, not for our own edits to the reference.
SRGB_LINEAR_CUTOFF = 0.003130800090730190277099609375
SRGB_LINEAR_SLOPE = 12.9200000762939453125
SRGB_ALPHA = 1.05499994754791259765625
SRGB_OFFSET = 0.054999999701976776123046875
SRGB_EXPONENT = 0.4166666567325592041015625
SRGB_DECODE_CUTOFF = 0.04045

# Rec.709 luminance weights -- the coefficients the exposure meter uses.
LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722


# --------------------------------------------------------------------------
# The curve
# --------------------------------------------------------------------------
def hable(x):
    """Uncharted-2 filmic, with the vpost's coefficients.

        F(x) = ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F

    Unnormalised. `x` is linear radiance; the caller divides by F(W)."""
    return ((x * (TM_A * x + TM_C * TM_B) + TM_D * TM_E)
            / (x * (TM_A * x + TM_B) + TM_D * TM_F)) - TM_E / TM_F


def hable_white(white_point: float = TM_WHITE_POINT) -> float:
    """F(W). The shader does not evaluate this inline: it multiplies by a
    CPU-side uniform `m8` which holds 1/F(W). Same number, computed once."""
    return float(hable(white_point))


def tonemap(x, prescale: float = TM_PRESCALE, clamp_max: float = None,
            white_point: float = TM_WHITE_POINT):
    """The shader's full tonemap: prescale, optional clamp, curve, divide
    by F(W), saturate.

    `clamp_max` is the shader's `min(rgb * 2.8, m7)`. m7 is a per-frame
    uniform this project has never read; None leaves the clamp out rather
    than substituting a value, so a run cannot silently carry a made-up
    ceiling. Passing a number applies the clamp exactly."""
    x = maximum(x, 0.0)
    x = x * prescale
    if clamp_max is not None:
        x = minimum(x, clamp_max)
    return clip(hable(x) / hable_white(white_point), 0.0, 1.0)


# --------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------
def scene_exposure(view_exposure: float, post_exposure: float) -> float:
    """cs2_post_process_ps.glsl:50 -- `_4459._m0 * _4349._m0`.

    The scene exposure is a PRODUCT of two uniforms: a per-view scalar
    (the auto-exposure controller's output) and a post-parameter scalar.
    Kept as two arguments rather than one because they come from different
    places and only one of them adapts over time; collapsing them is how a
    fitted single scalar got here in the first place."""
    return view_exposure * post_exposure


def luminance(rgb):
    """Rec.709 relative luminance of a (..., 3) linear array."""
    return (rgb[..., 0] * LUMA_R + rgb[..., 1] * LUMA_G
            + rgb[..., 2] * LUMA_B)


def exposure_gain(avg_luminance, key: float,
                  lo: float = AE_MIN, hi: float = AE_MAX):
    """The entity's exposure, clamped to its own band.

    `key` is the controller's target luminance. It is NOT read out of the
    engine — see EXPOSURE_KEY_UNREAD — so it is a required argument with no
    default: a caller must state the number it is using, and it lands in
    the render-identity stamp instead of hiding in a signature."""
    return clip(key / maximum(avg_luminance, 1e-5), lo, hi)


EXPOSURE_KEY_UNREAD = (
    "post_processing_volume's exposure TARGET luminance has not been read "
    "out of CS2; only the band (min 0.8, max 1.1), the speeds (1.0/1.0) and "
    "the fade time (1.5) are decoded from the map. `key` is therefore an "
    "explicit caller argument with no default. Route to read it: the "
    "engine's exposure controller state, or r_ exposure convars on the CS2 "
    "install. Measured consequence of the band alone: the clamp saturates "
    "on all 48 frames of the scored corpus, so the term is pinned rather "
    "than free -- report the saturation fraction, do not assume it is 1.0.")


def adapt_exposure(prev, target, dt: float,
                   speed_up: float = AE_SPEED_UP,
                   speed_down: float = AE_SPEED_DOWN,
                   fade_time: float = AE_FADE_TIME):
    """The entity's temporal adaptation toward `target`.

    Exponential approach with separate up/down speeds, as the entity's
    exposurespeedup / exposurespeeddown pair specifies, over `fade_time`
    seconds. Frame-rate independent: the per-step factor is derived from
    dt, so a 64 Hz and a 128 Hz replay of the same demo converge the same
    way rather than at different rates.

    The up and down speeds are separate multipliers on the approach rate,
    which is what the entity's two fields mean: a scene getting brighter and
    one getting darker adapt at independently-set rates. Here both are 1.0,
    so the branches coincide numerically — a property of de_inferno's
    values, not of the code. The branch is written rather than folded away
    so a map with different speeds is expressible.
    """
    import math
    base = 1.0 - math.exp(-dt / max(fade_time, 1e-6))
    up, down = base * speed_up, base * speed_down
    if _is_array(prev) or _is_array(target):
        arr = prev if _is_array(prev) else target
        rate = where(target > prev, _scalar_like(arr, up), _scalar_like(arr, down))
    else:
        rate = up if target > prev else down
    return prev + (target - prev) * clip(rate, 0.0, 1.0)


def exposure_report(gain, lo: float = AE_MIN, hi: float = AE_MAX) -> str:
    """Runtime print: mean and the fraction pinned at each bound.

    A term that sits on a clamp bound on every frame contributes a
    constant, and a constant is indistinguishable from an absent term
    unless the pinning is measured and printed."""
    import numpy as _np
    g = _np.asarray(gain, dtype=_np.float64).ravel()
    n = max(g.size, 1)
    at_lo = float((g <= lo + 1e-9).sum()) / n
    at_hi = float((g >= hi - 1e-9).sum()) / n
    return (f"exposure gain mean {g.mean():.4f} min {g.min():.4f} max "
            f"{g.max():.4f}; pinned at lo({lo}) {at_lo:.1%} hi({hi}) "
            f"{at_hi:.1%}; free {1.0 - at_lo - at_hi:.1%}")


# --------------------------------------------------------------------------
# The transfer function
# --------------------------------------------------------------------------
def srgb_encode(x):
    """TRUE sRGB with the linear toe, as post_process_vulkan_50_ps writes
    it. The shader's exponent literal is 0.41666666 = 1/2.4; pow(1/2.2) is
    a different curve and was never what the reference did."""
    # The shader has NO epsilon floor here (:72 is a bare pow). An added
    # 1e-8 guard measured 5.215e-08 against the transcription -- inside
    # the 1e-6 epsilon, and still a divergence introduced by us rather
    # than read from the reference, so it is removed. The preceding
    # max(x, 0) already makes the base non-negative, which is what the
    # guard was for.
    x = maximum(x, 0.0)
    return where(x <= SRGB_LINEAR_CUTOFF,
                 x * SRGB_LINEAR_SLOPE,
                 x ** SRGB_EXPONENT * SRGB_ALPHA - SRGB_OFFSET)


def srgb_decode(x):
    """The shader decodes back to linear after the colour-correction LUT."""
    x = clip(x, 0.0, 1.0)
    return where(x <= SRGB_DECODE_CUTOFF,
                 x / SRGB_LINEAR_SLOPE,
                 maximum((x + SRGB_OFFSET) / SRGB_ALPHA, 1e-8) ** 2.4)


def final_gamma(x, m5: float = 1.0):
    """The shader's adjustable final gamma, `pow(c, m5 / 2.2)`."""
    return maximum(x, 1e-8) ** (m5 / 2.2)
