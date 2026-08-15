#!/usr/bin/env python3
"""FidelityFX Super Resolution — EASU + RCAS, transcribed from the two
shipped CS2 modules.

SOURCES, and they are the whole basis of this file
---------------------------------------------------
    docs/projects/counter-strike-sft/cs2_fsr_easu.glsl   (222 lines)
    docs/projects/counter-strike-sft/cs2_fsr_rcas.glsl   ( 44 lines)

Both are SPIR-V -> GLSL of the modules CS2 ships. Every expression below
carries the `:line` it came from. Nothing here is taken from AMD's
published ffx_fsr1.h: the two agree, but the artifact is what runs, and
the project law is READ the field, never derive it from the name.

THE THREE MAGIC INTEGERS, DECODED
---------------------------------
The decompiler leaves AMD's bit-trick approximations as raw integer
arithmetic on float bit patterns. They are named functions in `ffx_a.h`
and they are NOT interchangeable with the exact operation:

    easu:210 etc  uintBitsToFloat(2129690299u - floatBitsToUint(x))
                  2129690299 = 0x7EF07EBB  ->  APrxLoRcpF1(x)   ~= 1/x
    easu:88       uintBitsToFloat(1597275508u - (floatBitsToUint(x) >> 1))
                  1597275508 = 0x5F347D74  ->  APrxLoRsqF1(x)   ~= 1/sqrt(x)
    rcas:41       uintBitsToFloat(2129764351u - floatBitsToUint(x))
                  2129764351 = 0x7EF19FFF  ->  APrxMedRcpF1 stage 1, then
                  :42 b*(-b*a + 2.0) is the Newton step that finishes it.

APrxLoRcpF1(1.0) is 0.93943, not 1.0 — a 6% error that lands directly in
the edge weights. Substituting a true reciprocal would be a DIFFERENT
filter, so the bit tricks are reproduced bit-exactly here (see
:func:`aprx_lo_rcp`). This is the same class as `no-fitted-floats`: the
constant is read, and the read constant is an approximation.

THE 12-TAP NEIGHBOURHOOD, DERIVED FROM THE SHADER'S OWN UV MATH
----------------------------------------------------------------
easu:21-24 builds four gather UVs; :25-36 issues twelve `textureGather`s
(four positions x three channels). `textureGather` returns, in order,
the texels at (i0,j1), (i1,j1), (i1,j0), (i0,j0) where
i0 = floor(u*W - 0.5) and j0 = floor(v*H - 0.5). Resolving that against
the uniform values FsrEasuCon writes gives the following texel offsets
relative to `fp` (:19, the floor of the input-space position):

    p0 = fp + (1,-1)   -> .x b(0,-1)  .y c(1,-1)  .zw unused
    p1 = fp + (0, 1)   -> .x i(-1,1)  .y j(0,1)   .z f(0,0)  .w e(-1,0)
    p2 = fp + (2, 1)   -> .x k(1,1)   .y l(2,1)   .z h(2,0)  .w g(1,0)
    p3 = fp + (1, 3)   -> .zw          o(1,2)                 n(0,2)

                    b c
                  e f g h
                  i j k l
                    n o

which is FSR's named pattern. This module therefore indexes texels
directly with those integer offsets rather than re-deriving them from
floating-point UVs each call: the derivation above is the provenance, and
doing the UV arithmetic again would only re-introduce rounding the
integer form does not have. ADDRESSING IS CLAMP-TO-EDGE; the sampler at
easu:12 (set 1, binding 14) was not dumped, and clamp is what every FSR
integration binds. That is the one assumption in this file and it is
stated rather than buried.

WHAT IS NOT IN THESE MODULES
-----------------------------
* No RCAS denoise path (`FSR_RCAS_DENOISE`): rcas:33-38 takes plain
  min/max over the five-tap cross with no `nz` term.
* No alpha passthrough: rcas:43 writes a literal `1.0` into .w, and
  easu:221 does the same. Both outputs are opaque by construction.
* No gamma/linear conversion anywhere in either module. FSR runs on
  whatever the previous pass wrote, which in this chain is the
  DISPLAY-SPACE output of the tonemap (chain.py:130), not linear light.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._backend import clip, is_torch, maximum, minimum, xp

# ffx_a.h bit-trick magic numbers, read off the decompiled modules.
APRX_LO_RCP_MAGIC = 0x7EF07EBB      # 2129690299 -- easu, many lines
APRX_LO_RSQ_MAGIC = 0x5F347D74      # 1597275508 -- easu:88
APRX_MED_RCP_MAGIC = 0x7EF19FFF     # 2129764351 -- rcas:41

# easu:86 -- the direction-length epsilon, exactly 2^-15.
EASU_DIR_EPSILON = 3.0517578125e-05
# easu:92-93 -- the length->window shaping constants.
EASU_LEN_W = 0.25
EASU_LEN_STRETCH = -0.125
EASU_CLIP_BASE = 0.5
EASU_CLIP_SLOPE = -0.072499997913837432861328125
# easu:107-109 -- the two-lobe window.
EASU_LOBE_A = 0.4000000059604644775390625
EASU_LOBE_B = 1.5625
EASU_LOBE_C = -0.5625
# rcas:39 -- FSR_RCAS_LIMIT = 0.25 - 1/16, negated as the lobe floor.
RCAS_LIMIT = -0.1875


# --------------------------------------------------------------------------
# The bit tricks. Written once, against the resolved array module.
# --------------------------------------------------------------------------
def _bits(x):
    """float32 -> int32 with the SAME bits. Both libraries, one expression.

    int32 rather than uint32 is safe HERE and only here: every value these
    approximations are applied to in both modules is non-negative (each is
    an `abs()`, a `max()` of `abs()`s, or a sum of squares), so the sign
    bit is clear and the pattern is below 2^31."""
    if is_torch(x):
        import torch
        return x.to(torch.float32).contiguous().view(torch.int32)
    return np.ascontiguousarray(x, dtype=np.float32).view(np.int32)


def _unbits(i, like):
    if is_torch(like):
        import torch
        return i.contiguous().view(torch.float32).to(like.dtype)
    return i.view(np.float32).astype(like.dtype)


def aprx_lo_rcp(x):
    """`uintBitsToFloat(0x7EF07EBB - floatBitsToUint(x))` -- APrxLoRcpF1.

    ~1/x with about 6% error. NOT 1/x; see the module docstring."""
    return _unbits(APRX_LO_RCP_MAGIC - _bits(x), x)


def aprx_lo_rsq(x):
    """easu:88 -- `uintBitsToFloat(0x5F347D74 - (floatBitsToUint(x) >> 1))`,
    APrxLoRsqF1, the Quake-style rsqrt WITHOUT the Newton step."""
    b = _bits(x)
    if is_torch(x):
        shifted = b.bitwise_right_shift(1)
    else:
        shifted = np.right_shift(b, 1)
    return _unbits(APRX_LO_RSQ_MAGIC - shifted, x)


def aprx_med_rcp(x):
    """rcas:41-42 -- `b = uintBitsToFloat(0x7EF19FFF - bits(a));
    b * (-b*a + 2.0)`. APrxMedRcpF1: the low approximation plus one
    Newton-Raphson step, which is why the magic differs from the lo one."""
    b = _unbits(APRX_MED_RCP_MAGIC - _bits(x), x)
    return b * ((-b) * x + 2.0)


# --------------------------------------------------------------------------
# EASU
# --------------------------------------------------------------------------
# easu:19-24. `con0.xy` is the ONLY place the render scale enters the
# filter: it is inputViewport / outputSize. Everything else is a function
# of the input size. So "what does fsr_detail tier N do" reduces to "what
# does it set con0.xy to", which is a table this project READS rather than
# assumes (see video_config.FSR_TIER_SCALE).
def easu_con(in_w: int, in_h: int, out_w: int, out_h: int):
    """FsrEasuCon's four uniform vectors, as the decompiled code consumes
    them (`_5540._m0.._m3`).

    Only the members the module actually reads are returned: :18 reads
    `_m0.xy` and `_m0.zw`, :21 `_m1.xy`/`_m1.zw`, :22-24 `_m2.xy`,
    `_m2.zw`, `_m3.xy`. `_m3.zw` is never read."""
    sx, sy = in_w / out_w, in_h / out_h
    return dict(
        m0=(sx, sy, 0.5 * sx - 0.5, 0.5 * sy - 0.5),      # :18
        m1=(1.0 / in_w, 1.0 / in_h, 1.0 / in_w, -1.0 / in_h),   # :21
        m2=(-1.0 / in_w, 2.0 / in_h, 1.0 / in_w, 2.0 / in_h),   # :22-23
        m3=(0.0, 4.0 / in_h, 0.0, 0.0),                          # :24
    )


def _tap(src, ix, iy, in_w, in_h):
    """One texel with CLAMP-TO-EDGE addressing. `src` is (H, W, C)."""
    m = xp(src)
    if m is np:
        ix = np.clip(ix, 0, in_w - 1)
        iy = np.clip(iy, 0, in_h - 1)
    else:
        ix = ix.clamp(0, in_w - 1)
        iy = iy.clamp(0, in_h - 1)
    return src[iy, ix]


def _luma(t):
    """easu:37-40 -- `B*0.5 + (R*0.5 + G)`.

    Written in the shader's own association order. It is NOT normalised
    (the weights sum to 2), and it is NOT a perceptual luma; it is FSR's
    cheap green-weighted proxy and the edge logic is calibrated to its
    scale."""
    return t[..., 2] * 0.5 + (t[..., 0] * 0.5 + t[..., 1])


def _edge(a, b, hi0, hi1):
    """The `clamp(abs(a - b) * APrxLoRcp(max(hi0, hi1)), 0, 1)` shape that
    easu:60/63/66/69/73/75/79/82 repeats eight times."""
    return clip(abs(a - b) * aprx_lo_rcp(maximum(hi0, hi1)), 0.0, 1.0)


def easu(src, out_w: int, out_h: int):
    """cs2_fsr_easu.glsl:16-222, whole. `src` is (H, W, >=3), display-space.

    Returns (out_h, out_w, 4); .w is the literal 1.0 of :221."""
    m = xp(src)
    in_h, in_w = int(src.shape[0]), int(src.shape[1])
    con = easu_con(in_w, in_h, out_w, out_h)

    # :18-20 -- output pixel -> input space, split into cell and fraction.
    if m is np:
        py, px = np.meshgrid(np.arange(out_h, dtype=np.float64),
                             np.arange(out_w, dtype=np.float64), indexing="ij")
    else:
        import torch
        py, px = torch.meshgrid(
            torch.arange(out_h, dtype=src.dtype, device=src.device),
            torch.arange(out_w, dtype=src.dtype, device=src.device),
            indexing="ij")
    ppx = px * con["m0"][0] + con["m0"][2]
    ppy = py * con["m0"][1] + con["m0"][3]
    fpx, fpy = m.floor(ppx), m.floor(ppy)
    rx, ry = ppx - fpx, ppy - fpy                                   # :20
    ix = fpx.to(__import__("torch").int64) if m is not np else fpx.astype(np.int64)
    iy = fpy.to(__import__("torch").int64) if m is not np else fpy.astype(np.int64)

    # :25-36 resolved onto the neighbourhood derived in the docstring.
    OFF = dict(b=(0, -1), c=(1, -1),
               e=(-1, 0), f=(0, 0), g=(1, 0), h=(2, 0),
               i=(-1, 1), j=(0, 1), k=(1, 1), l=(2, 1),
               n=(0, 2), o=(1, 2))
    T = {name: _tap(src, ix + dx, iy + dy, in_w, in_h)
         for name, (dx, dy) in OFF.items()}
    L = {name: _luma(t) for name, t in T.items()}                   # :37-40

    # :53-57, 64, 70, 76 -- the four bilinear corner weights.
    w00 = (1.0 - rx) * (1.0 - ry)
    w10 = rx * (1.0 - ry)
    w01 = (1.0 - rx) * ry
    w11 = rx * ry

    # :58-82 -- eight edge measures and the two direction accumulators.
    # Named for the tap pair each one straddles; the shader's ids are in
    # the trailing comment so the transcription is checkable line by line.
    d_fg = L["g"] - L["f"]                                          # :59 _14242
    a_gj = abs(L["g"] - L["j"])                                     # :58 _8529
    e_fg = _edge(L["g"], L["f"], a_gj, abs(L["j"] - L["f"]))        # :60 _16881
    d_jb = L["j"] - L["b"]                                          # :62 _14243
    a_jf = abs(L["j"] - L["f"])                                     # :61 _8530
    e_jb = _edge(L["j"], L["b"], a_jf, abs(L["f"] - L["b"]))        # :63 _16843
    d_hf = L["h"] - L["f"]                                          # :65 _16461
    e_hf = _edge(L["h"], L["f"], abs(L["h"] - L["g"]), a_gj)        # :66 _16919
    d_kc = L["k"] - L["c"]                                          # :68 _16462
    a_kg = abs(L["k"] - L["g"])                                     # :67 _8531
    e_kc = _edge(L["k"], L["c"], a_kg, abs(L["g"] - L["c"]))        # :69 _16920
    d_ki = L["k"] - L["i"]                                          # :72 _16463
    a_kj = abs(L["k"] - L["j"])                                     # :71 _8532
    e_ki = _edge(L["k"], L["i"], a_kj, abs(L["j"] - L["i"]))        # :73 _16922
    d_nf = L["n"] - L["f"]                                          # :74 _16464
    e_nf = _edge(L["n"], L["f"], abs(L["n"] - L["j"]), a_jf)        # :75 _16923
    d_lj = L["l"] - L["j"]                                          # :77 _16465
    e_lj = _edge(L["l"], L["j"], abs(L["l"] - L["k"]), a_kj)        # :79 _16925
    d_og = L["o"] - L["g"]                                          # :80 _16466
    e_og = _edge(L["o"], L["g"], abs(L["o"] - L["k"]), a_kg)        # :82 _16926

    dir_x = d_fg * w00 + d_hf * w10 + d_ki * w01 + d_lj * w11       # :78
    dir_y = d_jb * w00 + d_kc * w10 + d_nf * w01 + d_og * w11       # :81
    # :83 -- the LENGTH accumulator: squared edge measures, corner-weighted.
    ln = ((w00 * (e_fg * e_fg + e_jb * e_jb)
           + (e_hf * e_hf) * w10) + (e_kc * e_kc) * w10
          + (e_ki * e_ki) * w01 + (e_nf * e_nf) * w01
          + (e_lj * e_lj) * w11 + (e_og * e_og) * w11)

    # :84-88 -- normalise the direction. The degenerate case substitutes
    # (1, dir_y) and a unit scale rather than dividing by zero.
    d2 = dir_x * dir_x + dir_y * dir_y                              # :85
    zero = d2 < EASU_DIR_EPSILON                                    # :86
    dx0 = m.where(zero, m.ones_like(dir_x), dir_x)                  # :87
    inv = m.where(zero, m.ones_like(d2), aprx_lo_rsq(d2))           # :88
    dx, dy = dx0 * inv, dir_y * inv                                 # :88

    ln2 = ln * ln                                                   # :89
    # :92 -- the anisotropic stretch, then the perpendicular squash.
    stretch = ((dx * dx + dy * dy)
               * aprx_lo_rcp(maximum(abs(dx), abs(dy)))) - 1.0
    wx = 1.0 + stretch * (EASU_LEN_W * ln2)
    wy = 1.0 + ln2 * EASU_LEN_STRETCH
    clp = EASU_CLIP_BASE + ln2 * EASU_CLIP_SLOPE                    # :93
    clp_max = aprx_lo_rcp(clp)                                      # :94

    def _w(dx_off, dy_off):
        """:99-109 -- one tap's window weight, the twelve-times-repeated
        block. `(ox, oy)` is the tap's offset from the cell corner."""
        ox = dx_off - rx
        oy = dy_off - ry
        # :103 -- rotate into the edge frame, then scale by (wx, wy).
        vx = (ox * dx + oy * dy) * wx
        vy = (ox * (-dy) + oy * dx) * wy
        d = minimum(vx * vx + vy * vy, clp_max)                     # :106
        a = EASU_LOBE_A * d - 1.0                                   # :107
        b = clp * d - 1.0                                           # :108
        return ((EASU_LOBE_B * (a * a)) + EASU_LOBE_C) * (b * b)    # :109

    W = {name: _w(float(dx_off), float(dy_off))
         for name, (dx_off, dy_off) in OFF.items()}

    # :210, :221 -- the weighted sum, in the shader's accumulation order.
    ORDER = ("b", "c", "i", "j", "f", "e", "k", "l", "h", "g", "o", "n")
    acc = None
    wsum = None
    for name in ORDER:
        w = W[name][..., None]
        acc = T[name][..., :3] * w if acc is None else acc + T[name][..., :3] * w
        wsum = W[name] if wsum is None else wsum + W[name]
    out = acc * (1.0 / wsum)[..., None]

    # :221 -- clamp into the 2x2 CENTRE's range (f, g, j, k), which is the
    # ringing guard. min(max(...)) with the arguments in the shader's own
    # nesting: the outer min is against the max of the four, the inner max
    # against their min.
    ctr = [T[nm][..., :3] for nm in ("f", "g", "j", "k")]
    hi = maximum(maximum(ctr[0], maximum(ctr[1], ctr[2])), ctr[3])
    lo = minimum(minimum(ctr[0], minimum(ctr[1], ctr[2])), ctr[3])
    out = minimum(hi, maximum(lo, out))
    one = (np.ones_like(out[..., :1]) if m is np
           else __import__("torch").ones_like(out[..., :1]))
    return (np.concatenate([out, one], -1) if m is np
            else __import__("torch").cat([out, one], -1))


# --------------------------------------------------------------------------
# RCAS
# --------------------------------------------------------------------------
def rcas_con(sharpness: float) -> float:
    """FsrRcasCon: `con.x = exp2(-sharpness)`.

    rcas:39 reads it as `uintBitsToFloat(_5540._m0.x)` and multiplies the
    lobe by it, so a LARGER `sharpness` gives a SMALLER multiplier and a
    weaker lobe... which is backwards from the name, and is why the
    parameter this module exposes is the raw multiplier `con` as well as
    the sharpness that produces it. Neither is guessed at from a tier: see
    video_config.FSR_TIER_SHARPNESS."""
    return float(2.0 ** (-sharpness))


def rcas(src, con: float):
    """cs2_fsr_rcas.glsl:13-44, whole. `src` is (H, W, >=3); `con` is the
    already-exponentiated multiplier of :39, not the sharpness."""
    m = xp(src)
    h, w = int(src.shape[0]), int(src.shape[1])

    def T(dx, dy):
        """:16-20 -- `texelFetch` with an integer offset. texelFetch is
        UNCLAMPED in GLSL and out-of-range is undefined; the reference
        binds a full-viewport pass, so the edge behaviour only shows on
        the one-pixel border. Clamped here, and said out loud rather than
        left to whatever indexing does."""
        if m is np:
            ys = np.clip(np.arange(h) + dy, 0, h - 1)
            xs = np.clip(np.arange(w) + dx, 0, w - 1)
            return src[ys][:, xs]
        import torch
        ys = torch.arange(h, device=src.device).add(dy).clamp(0, h - 1)
        xs = torch.arange(w, device=src.device).add(dx).clamp(0, w - 1)
        return src.index_select(0, ys).index_select(1, xs)

    up, lf, ct, rt, dn = T(0, -1), T(-1, 0), T(0, 0), T(1, 0), T(0, 1)

    lobe = None
    for ch in range(3):
        # :33-38 -- min/max of the five-tap cross, per channel.
        a, b, c, d = up[..., ch], lf[..., ch], rt[..., ch], dn[..., ch]
        mn = minimum(minimum(a, minimum(b, c)), d)
        mx = maximum(maximum(a, maximum(b, c)), d)
        # :39 -- the per-channel lobe: the sharper of the two dark/bright
        # limits. `0.25/mx` and `1/(4*mn - 4)` are FSR's hardness terms.
        v = maximum(-(mn * (0.25 / mx)), (1.0 - mx) * (1.0 / (4.0 * mn - 4.0)))
        lobe = v if lobe is None else maximum(lobe, v)
    lobe = maximum(RCAS_LIMIT, minimum(lobe, 0.0)) * con             # :39

    rcp = aprx_med_rcp(4.0 * lobe + 1.0)                             # :40-42
    out = None
    for ch in range(3):
        s = (up[..., ch] + lf[..., ch]) + dn[..., ch] + rt[..., ch]
        v = ((lobe * s) + ct[..., ch]) * rcp                         # :43
        out = v[..., None] if out is None else (
            np.concatenate([out, v[..., None]], -1) if m is np
            else __import__("torch").cat([out, v[..., None]], -1))
    one = (np.ones_like(out[..., :1]) if m is np
           else __import__("torch").ones_like(out[..., :1]))
    return (np.concatenate([out, one], -1) if m is np
            else __import__("torch").cat([out, one], -1))


# --------------------------------------------------------------------------
# The pass pair, in the order the reference runs them
# --------------------------------------------------------------------------
def fsr_pass(src, out_w: int, out_h: int, rcas_con_value: Optional[float]):
    """EASU then RCAS, which is FSR 1.0's fixed order.

    `rcas_con_value` None runs EASU alone. That is a legal FSR
    configuration (RCAS is optional in FSR 1.0) and it is the honest
    representation of "we have not read this tier's sharpness" — running
    RCAS at a made-up sharpness would put a fitted float in the image."""
    up = easu(src, out_w, out_h)
    if rcas_con_value is None:
        return up
    return rcas(up, rcas_con_value)


def fsr_report(src, out) -> str:
    """Runtime line. Prints the SCALE it actually ran at, because a tier
    that resolved to 1.0 and a tier that was skipped produce the same
    pixels and must not produce the same log.

    The mean is taken with `.mean()` on the object itself, NOT through
    np.asarray(). numpy's converter raises on a CUDA tensor --

        TypeError: can't convert cuda:0 device type tensor to numpy.
                   Use Tensor.cpu() to copy the tensor to host memory first.

    -- and in the live renderer `out` is exactly that. `.mean()` exists on
    both numpy arrays and torch tensors, returns a 0-d value in each case,
    and float() takes it from either without dragging the whole image
    across the device boundary to print one number.

    This crashed EVERY preset0/preset1 render (the FSR-upscaled tiers) while
    preset2/3 rendered clean, because those resolve to fsr tier 0 and never
    reach this line. The crash was in the REPORTING call, not in EASU or
    RCAS: the pixels were correct and the line describing them killed the
    job. sibling scene_format._to_host carries the warning about this exact
    numpy behaviour, written before this line was -- the rule was known and
    this call site simply did not take it.
    """
    ih, iw = int(src.shape[0]), int(src.shape[1])
    oh, ow = int(out.shape[0]), int(out.shape[1])
    return (f"post/fsr: {iw}x{ih} -> {ow}x{oh} "
            f"scale {iw / ow:.6f}x{ih / oh:.6f} "
            f"mean {float(out[..., :3].mean()):.6g}")
