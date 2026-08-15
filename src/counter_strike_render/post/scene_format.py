#!/usr/bin/env python3
"""`videocfg_hdr_detail` — the HDR scene-colour format, as a REAL axis.

WHAT WAS ALREADY THERE, AND WHY IT WAS NOT ENOUGH
-------------------------------------------------
`msaa_resolve.store_to_attachment()` already applied the per-format
representable MAXIMA (65504 for RGBA16F; 65024/65024/64512 for
R11G11B10), and `video_config.HDR_FORMAT_BY_TIER` already carried the
READ tier->format map. The preset table nevertheless recorded this axis
as GAP_NO_AXIS with the note "both tiers render the same buffer", and
that note was RIGHT, because a range clamp is not the difference between
these two formats.

On a de_inferno frame nothing reaches 64512. Clamping at 65024 instead of
65504 changes nothing anywhere, so a renderer that implements only the
clamp produces BIT-IDENTICAL images at hdr_detail 3 and -1 — the exact
"deleting a FIT silently kills its ratio consumers" shape, where an A/B
comes back identical and gets read as "doesn't matter" when in fact
neither arm was doing anything.

WHAT ACTUALLY DIFFERS: MANTISSA BITS
-------------------------------------
    VK_FORMAT_R16G16B16A16_SFLOAT   sign + 5 exp + 10 mantissa, x4
    VK_FORMAT_B10G11R11_UFLOAT_PACK32
        R  no sign + 5 exp + 6 mantissa
        G  no sign + 5 exp + 6 mantissa
        B  no sign + 5 exp + 5 mantissa
        (no alpha channel at all)

Both are exponent-bias-15 floats with denormals, so the two agree on
DYNAMIC RANGE and disagree on PRECISION: 6 mantissa bits give 64 steps
per octave against half's 1024, a 16x coarser quantisation, and 5 bits in
blue give 32. That is what the reference's 2.53x measured response to
this axis is made of, and it shows up as banding in smooth gradients —
sky, fog, large flat lit surfaces — not as clipping in highlights.

It is also why the axis is worth implementing rather than declared
cosmetic: every value the scene pass writes goes through it, on preset0
and preset1 and on every single-axis arm in the corpus. The gate corpus
is preset2, i.e. RGBA16F, so the gate cannot see this path at all.

ROUNDING MODE
--------------
Round-to-nearest-EVEN, which is what Vulkan requires of a colour
attachment store (VK spec, "Conversion to Unsigned Floating-Point"
defers to the IEEE-754 roundTiesToEven default; no CS2 pipeline in the
capture sets a rounding-mode execution mode). Round-to-nearest-even is
therefore READ from the specification of the operation the hardware
performs, not chosen because it looked reasonable.
"""
from __future__ import annotations

import numpy as np

from ._backend import is_torch, xp
from .msaa_resolve import (FORMAT_HAS_ALPHA, FORMAT_MAX, HALF_MAX,
                           ResolveRefusal, store_to_attachment)

# (mantissa bits, has sign) per channel, per format. Exponent width is 5
# and the bias is 15 in every one of these, which is why only the mantissa
# width and the sign bit are tabulated.
FORMAT_MANTISSA_BITS = {
    "VK_FORMAT_R16G16B16A16_SFLOAT": (10, 10, 10),
    "VK_FORMAT_B10G11R11_UFLOAT_PACK32": (6, 6, 5),
}
FORMAT_SIGNED = {
    "VK_FORMAT_R16G16B16A16_SFLOAT": True,
    "VK_FORMAT_B10G11R11_UFLOAT_PACK32": False,
}

# Both formats: 5 exponent bits, bias 15. The smallest NORMAL value is
# 2^(1-15) = 2^-14; below that the encoding is denormal with a fixed
# 2^-14 scale, which is a DIFFERENT quantisation step and is handled
# separately below.
EXP_BIAS = 15
MIN_NORMAL_EXP = 1 - EXP_BIAS          # -14


def _round_ties_even(x):
    """Round-to-nearest-even on an array of non-negative floats."""
    m = xp(x)
    if m is np:
        return np.rint(x)
    # torch.round is round-half-to-even, matching np.rint.
    return m.round(x)


def quantize_channel(x, mantissa_bits: int, signed: bool, vmax: float):
    """One channel through a (1, 5, `mantissa_bits`) float encode+decode.

    Written as an explicit frexp/ldexp round-trip rather than a bit-cast
    so it holds for float32 and float64 inputs alike and so the denormal
    branch is visible instead of implied."""
    m = xp(x)
    sign = None
    if signed:
        if m is np:
            sign = np.where(x < 0, -1.0, 1.0)
        else:
            sign = m.where(x < 0, -m.ones_like(x), m.ones_like(x))
        a = abs(x)
    else:
        # UFLOAT: negatives are not representable. The hardware clamps
        # them to zero on store; so does this.
        a = m.clamp(x, min=0.0) if m is not np else np.maximum(x, 0.0)
    a = m.minimum(a, m.full_like(a, vmax)) if m is not np \
        else np.minimum(a, vmax)

    step = float(2 ** mantissa_bits)
    if m is np:
        frac, exp = np.frexp(a)                  # a = frac * 2**exp
    else:
        exp = m.floor(m.log2(m.clamp(a, min=1e-45))) + 1
        frac = a * m.pow(2.0, -exp)
    # NORMAL path: a = 1.f * 2**(exp-1), so the stored mantissa is
    # (frac*2 - 1) in [0, 1) with `mantissa_bits` of resolution.
    e = exp - 1
    mant = _round_ties_even((frac * 2.0 - 1.0) * step) / step
    # A mantissa that rounds up to exactly 1.0 carries into the exponent.
    carry = mant >= 1.0
    mant = m.where(carry, mant * 0.0, mant) if m is not np \
        else np.where(carry, 0.0, mant)
    e = m.where(carry, e + 1, e) if m is not np else np.where(carry, e + 1, e)
    normal = (1.0 + mant) * (2.0 ** e if m is np else m.pow(2.0, e))

    # DENORMAL path: below 2**-14 the exponent is pinned and the step is
    # 2**-14 / 2**mantissa_bits, an ABSOLUTE step rather than a relative
    # one. Dropping this branch would quantise near-black to zero.
    dstep = (2.0 ** MIN_NORMAL_EXP) / step
    denorm = _round_ties_even(a / dstep) * dstep

    is_denorm = a < (2.0 ** MIN_NORMAL_EXP)
    out = m.where(is_denorm, denorm, normal) if m is not np \
        else np.where(is_denorm, denorm, normal)
    if m is np:
        out = np.minimum(out, vmax)
    else:
        out = m.minimum(out, m.full_like(out, vmax))
    return out * sign if signed else out


def quantize_to_format(rgb, vk_format: str):
    """The scene colour as the ATTACHMENT actually holds it.

    Range first (`store_to_attachment`, which is the existing clamp and
    the alpha drop), then PRECISION, which is the part that was missing
    and is the part the axis is made of."""
    bits = FORMAT_MANTISSA_BITS.get(vk_format)
    if bits is None:
        raise ResolveRefusal(
            f"{vk_format!r} has no mantissa-width entry. The two formats "
            f"videocfg_hdr_detail selects are "
            f"{', '.join(sorted(FORMAT_MANTISSA_BITS))}. Quantising to a "
            f"guessed width would invent a precision the hardware never "
            f"had -- the same class of error as inventing its range.")
    clamped = store_to_attachment(rgb, vk_format)
    signed = FORMAT_SIGNED[vk_format]
    vmax = FORMAT_MAX[vk_format]
    m = xp(clamped)
    chans = [quantize_channel(clamped[..., c], bits[c], signed, vmax[c])
             for c in range(3)]
    if FORMAT_HAS_ALPHA[vk_format] and clamped.shape[-1] > 3:
        # Alpha of RGBA16F is a half like the others.
        chans.append(quantize_channel(clamped[..., 3], 10, True,
                                      FORMAT_MAX[vk_format][0]))
    stacked = [c[..., None] for c in chans]
    return (np.concatenate(stacked, -1) if m is np
            else __import__("torch").cat(stacked, -1))


def _to_host(x):
    """torch tensor (possibly on CUDA) -> numpy, anything else unchanged.

    np.asarray() on a CUDA tensor raises; the tensor has to come across the
    device boundary explicitly. Everything else in this module already
    branches on `m is np`, so the torch path was known -- this one call site
    just did not take it.
    """
    try:
        import torch
    except Exception:                                            # noqa: BLE001
        return x
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def quantize_report(rgb, out, vk_format: str) -> str:
    """Runtime line. Prints the fraction the quantisation MOVED and the
    largest step it took, so "this tier ran and changed nothing" and
    "this tier was skipped" cannot print the same thing (#34).

    A REPORT MUST NOT BE ABLE TO KILL THE THING IT REPORTS ON. This function
    raised `TypeError: can't convert cuda:0 device type tensor to numpy` and
    took the whole render down with it -- and only on the --preset path, since
    that is what selects an explicit hdr_scene_format and routes into
    _hdr_store's reporting. Four preset arms died in a diagnostic line while
    the render itself was fine. So there are two fixes here and both matter:
    the device transfer, and the guarantee that any future failure in this
    function degrades to a STATED unavailable line instead of an exception.
    A diagnostic that can abort its subject is worse than no diagnostic,
    because it converts a working render into a failed one.
    """
    bits = FORMAT_MANTISSA_BITS[vk_format]
    try:
        a = np.asarray(_to_host(rgb), dtype=np.float64)[..., :3]
        b = np.asarray(_to_host(out), dtype=np.float64)[..., :3]
        d = np.abs(a - b)
        n = max(d.size, 1)
        return (f"post/scene_format: {vk_format} mantissa {bits} "
                f"moved {float((d > 0).sum()) / n:.4%} of samples "
                f"|delta| mean {d.mean():.3e} max {d.max():.3e}")
    except Exception as exc:                                     # noqa: BLE001
        # Stated, not silent: the reader must be able to tell "the report
        # could not be computed" from "the tier changed nothing".
        return (f"post/scene_format: {vk_format} mantissa {bits} "
                f"REPORT UNAVAILABLE ({type(exc).__name__}: {exc}) -- the "
                "quantisation still ran; only this line failed")


# --------------------------------------------------------------------------
# The hardware oracle
# --------------------------------------------------------------------------
def selftest(n: int = 20001) -> int:
    """Validate the encoder where an INDEPENDENT ORACLE exists.

    This term is not in the conformance registry, and that is the
    harness being right rather than a gap: its anchor requires a
    citation into a registered DECOMPILED SOURCE, and this reference is
    the Vulkan specification of a colour-attachment store. Faking a
    `.glsl:NN` citation to earn a green row would defeat the one thing
    the anchor checks.

    What replaces it is stronger than a synthetic A/B against a second
    transcription I would also have written: at M=10 with a sign bit,
    :func:`quantize_channel` IS IEEE binary16, and numpy's float16 is
    the hardware's. So the same code path, same branches, same rounding,
    is compared against an oracle nobody in this project wrote. If it
    reproduces binary16 bit-exactly including denormals, the 6- and
    5-bit widths are that expression with a different constant.

    Run: `python -m counter_strike_render.post.scene_format`
    """
    rng = np.linspace(0.0, 4.0, n)
    dom = np.concatenate([
        rng,
        np.array([0.0, 2.0 ** -24, 2.0 ** -15, 2.0 ** -14, 1.0,
                  65504.0, 1e-8, 3e-6, 6e-5]),
    ])
    ours = quantize_channel(dom, 10, True, HALF_MAX)
    oracle = dom.astype(np.float16).astype(np.float64)
    d = float(np.abs(ours - oracle).max())
    ok = d == 0.0
    print(f"scene_format: M=10 signed vs numpy float16 over {dom.size} "
          f"samples (incl. denormals to 2^-24 and the format maximum): "
          f"maxdiff {d:g} -- {'BIT-EXACT' if ok else 'DIVERGES'}")

    # THE ORACLE IS NOT VALID ABOVE THE FORMAT MAXIMUM, and the reason is
    # a real semantic difference rather than a domain nicety: numpy's
    # float16 CAST overflows to +inf, while a colour-attachment STORE
    # clamps to the largest representable value (VK spec: values outside
    # the representable range are converted to the nearest representable,
    # and inf is only produced by an inf input). So 70000 is checked
    # against the store's rule, not against numpy's.
    over = quantize_channel(np.array([70000.0, 1e30]), 10, True, HALF_MAX)
    ok_over = bool(np.all(over == HALF_MAX))
    print(f"scene_format: past the maximum -> {over.tolist()} "
          f"(numpy's cast gives inf here; an attachment store CLAMPS) "
          f"-- {'CLAMPS' if ok_over else 'DOES NOT CLAMP'}")
    ok = ok and ok_over

    # And the property the AXIS is made of, as a number rather than a
    # claim: how much coarser R11G11B10 actually is.
    unit = np.linspace(0.0, 1.0, 4001)
    lv_h = len(np.unique(quantize_channel(unit, 10, True, HALF_MAX)))
    lv_r = len(np.unique(quantize_channel(unit, 6, False, 65024.0)))
    lv_b = len(np.unique(quantize_channel(unit, 5, False, 64512.0)))
    print(f"scene_format: distinct levels over [0,1] -- "
          f"RGBA16F {lv_h}, R11G11B10 R/G {lv_r}, B {lv_b} "
          f"({lv_h / max(lv_r, 1):.1f}x and {lv_h / max(lv_b, 1):.1f}x "
          f"coarser). THIS is videocfg_hdr_detail; the range clamp the "
          f"two formats differ by is inert on a de_inferno frame.")
    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(selftest())
