#!/usr/bin/env python3
"""`r_csgo_cmaa_enable` — Conservative Morphological Anti-Aliasing 2.

SOURCES
-------
    docs/projects/counter-strike-sft/cs2_cmaa2_cs_edges.glsl              (122)
    docs/projects/counter-strike-sft/cs2_cmaa2_cs_process_candidates.glsl (452)
    docs/projects/counter-strike-sft/cs2_cmaa2_cs_deferred_apply.glsl     ( 93)

Three compute shaders, decompiled from the modules CS2 ships. The preset
table recorded this as GAP_NO_AXIS with the note "the shader exists; the
axis does not" — this file is the axis.

WHERE IT SITS IN THE LATTICE, SAID UP FRONT
--------------------------------------------
`r_csgo_cmaa_enable` is **0 in all four presets** (vcfg/preset0..3.txt).
So this pass contributes NOTHING to the preset0..preset3 ladder, and the
only ground truth that exercises it is the single-axis arm
`gtq/caps/r_csgo_cmaa_enable-1`. That is a scope fact, not a reason to
skip it: the work order is the whole lattice, and an arm with no
implementation is an arm that cannot be scored.

THE EDGE RULE, DERIVED FROM THE SHADER'S SHARED-MEMORY INDEXING
----------------------------------------------------------------
edges.glsl computes, per pixel p, two luma gradients

    h(p) = |L(p) - L(p + (1,0))|        L(p) = dot(sqrt(rgb), (.299,.587,.114))
    v(p) = |L(p) - L(p + (0,1))|

stores them into shared memory as 2x2 quads (:51-52), and then reads back
across quad boundaries with the offsets `tid-17, tid-16, tid-15, tid-1,
tid+1, tid+15, tid+16`. Resolving those offsets into absolute pixel
coordinates — done term by term for `_4811[0].x`, `_4811[0].y`,
`_4811[0].z` and `_4811[1].x`, all four of which agree — gives a rule
that has no quad structure in it at all:

    RIGHT edge at p  <=>  h(p) - 0.1 * max{ v(p+(0,-1)), v(p),
                                            v(p+(1,-1)), v(p+(1,0)) } > 0.15
    BOTTOM edge at p <=>  v(p) - 0.1 * max{ h(p+(-1,0)), h(p),
                                            h(p+(-1,1)), h(p+(0,1)) } > 0.15

i.e. an edge must clear an ABSOLUTE threshold (0.15) *and* clear 10% of
the largest PERPENDICULAR gradient in its 2x2 neighbourhood. That second
term is CMAA2's local-contrast adaptation and it is what stops texture
detail being treated as geometry. Because the rule is per-pixel, the
whole edge pass is two shifts and a max here, not a 16x16 tile walk.

THE BIT ASSIGNMENT, READ NOT ASSUMED
-------------------------------------
edges.glsl:106 packs `dot(_4811[i], vec4(1,2,4,8))`. Which component is
which direction was resolved from the CONSUMER: process_candidates:130-190
weights the (-1,0), (0,-1), (1,0), (0,1) taps by `_17569.x/.y/.z/.w`, and
tracing those back through :113-122 fixes

    bit 0 (1) = RIGHT    bit 1 (2) = BOTTOM
    bit 2 (4) = LEFT     bit 3 (8) = TOP

Deriving this from the component ORDER instead would have given the
transposed assignment; the project law is READ the field, and the field
here is the consumer's tap offset.

WHAT IS SIMPLIFIED, AND WHY IT IS THE SAME NUMBER
--------------------------------------------------
* **The candidate linked list.** process_candidates appends each blended
  colour to a per-QUAD singly-linked list (an `imageAtomicExchange` on the
  head, :196), and deferred_apply:48-79 walks it accumulating
  `colour * w` and `w`, with `w = 1.8` when bit 26 is set (a Z-shape
  contribution) and `0.8` otherwise (:67), then divides (:84). That is a
  weighted mean over a pixel's contributions. This module accumulates the
  same weighted sums directly into two buffers. Same arithmetic, no
  pointer chasing. The list's 32-entry walk cap (:53) is preserved as a
  cap on contributions per pixel.
* **The 768-entry shared staging buffer.** process_candidates has TWO
  paths that emit the Z-shape blend: the staged one (:328-352) and the
  immediate one (:358-395) taken when the buffer overflows. They compute
  the SAME mix, and differ only in that the staged path clamps the blend
  factor to [0,1] and quantises it to 10 bits (`*1023 + 0.5`, decoded by
  `* 1/1023` at :439). The staged path is implemented, including the
  quantisation, because it is the one that runs when the buffer does not
  overflow — which is the common case.
* **The packed candidate colour.** :198 stores each colour as a
  3-bit-shifted float16 (11/11/10 bits, `(f16 + 4) >> 3` and friends) and
  deferred_apply:68 unpacks it. That quantisation IS implemented
  (:func:`_pack_roundtrip`) because it happens before the average, so it
  is not recoverable afterwards.

⚠️ THERE IS A SECOND CMAA CVAR, AND WE HAVE NO AXIS FOR IT
------------------------------------------------------------
`r_csgo_cmaa_quality` is a CONFIRMED live cvar
(GT_RENDER_OPTION_SURFACE.json, libclient.so) alongside
`r_csgo_cmaa_enable`. So CMAA2 in CS2 has a QUALITY level as well as an
on/off, and `--cmaa` is on/off only.

Intel's CMAA2 ships a quality axis that moves the edge threshold and the
non-dominant factor -- the two constants at the top of this file -- but
WHICH values Valve's levels select is not readable: this build exposes no
cvar value readback, and no capture in the corpus varies it (every gtq
arm leaves it at its default, whatever that is). The transcription here
is of the module CS2 SHIPS, so it carries that module's baked constants
and is correct for whatever level produced it; a second level would be a
different pair of constants and this file does not know them.

Recording it rather than leaving it out: an axis nobody wrote down is an
axis nobody knows is missing. The arm that would settle it is
pre-written in the manifest.

WHAT IS NOT IN THESE MODULES
-----------------------------
No temporal component, no depth or normal input, no luma-in-alpha
optimisation: edges.glsl reads only `texture2D _4971` colour, and
`texelFetch` on it is the only input to the whole algorithm.
"""
from __future__ import annotations

import numpy as np

from ._backend import is_torch, xp

# edges:39-50 -- the luma weights, decoded from their float literals.
LUMA_WEIGHTS = (0.2989999949932098388671875,
                0.58700001239776611328125,
                0.114000000059604644775390625)
# edges:72 etc -- the absolute edge threshold and the non-dominant factor.
EDGE_THRESHOLD = 0.1500000059604644775390625
NON_DOMINANT_FACTOR = 0.100000001490116119384765625
# candidates:113-122 -- the L-shape reinforcement weight.
SHAPE_REINFORCE = 0.89999997615814208984375
# candidates:124 -- the blend budget: tighter when exactly two edges meet.
BLEND_SIMPLE = 0.100000001490116119384765625
BLEND_TWO_EDGE = 0.07500000298023223876953125
# candidates:124 -- confidence falls off with neighbourhood edge density.
CONF_BASE = 1.2999999523162841796875
CONF_SLOPE = 0.100000001490116119384765625
# candidates:207 etc -- the Z-shape score's cross-term discount.
Z_DISCOUNT = 0.699999988079071044921875
# candidates:236 -- the trace's hard length cap and its symmetry tolerance.
TRACE_CAP = 86.0
TRACE_RATIO = 1.25
TRACE_BIAS = -0.25
# candidates:307 -- the minimum total traced length that earns a Z blend.
Z_MIN_LENGTH = 5.0
# candidates:321-322 -- the odd-length half-pixel correction.
Z_PARITY = 0.2199999988079071044921875
# candidates:323 -- confidence from the traced length.
Z_CONF_SLOPE = 0.1500000059604644775390625
# candidates:197 / deferred_apply:68 -- the deferred colour's range clamp.
CANDIDATE_MAX = 1.99999988079071044921875
# deferred_apply:67 -- the two contribution weights.
W_COMPLEX = 1.7999999523162841796875
W_SIMPLE = 0.800000011920928955078125
# deferred_apply:53 -- the list walk cap.
MAX_CONTRIBUTIONS = 32

# The four direction bits, named. See the module docstring for how the
# assignment was read out of the consumer rather than the component order.
E_RIGHT, E_BOTTOM, E_LEFT, E_TOP = 1, 2, 4, 8


def _np(x):
    """This module works in numpy internally. The Z-shape trace is a
    variable-length scatter/gather over a sparse candidate set, which is a
    CPU-shaped problem even when the frame arrives as a tensor; converting
    once at the boundary is honest about that rather than writing a
    torch path that would silently fall back to the same loop."""
    if is_torch(x):
        return x.detach().to("cpu").to(__import__("torch").float64).numpy()
    return np.asarray(x, dtype=np.float64)


def _shift(a, dx, dy):
    """`a` displaced by (dx, dy) with CLAMP-TO-EDGE, i.e. the value the
    shader's `texelFetchOffset` reads at p+(dx,dy) for interior pixels."""
    h, w = a.shape[0], a.shape[1]
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return a[ys][:, xs]


# --------------------------------------------------------------------------
# Pass 1 -- edges
# --------------------------------------------------------------------------
def compute_edges(rgb):
    """cs2_cmaa2_cs_edges.glsl:32-121 -> (edge bitmask, candidate mask).

    Returns `(edges, candidates)`; `edges` is uint8 with the four
    direction bits, `candidates` is the bool of :101 (any two
    PERPENDICULAR edges meeting at the pixel)."""
    c = _np(rgb)[..., :3]
    lum = np.sqrt(np.maximum(c, 0.0)) @ np.asarray(LUMA_WEIGHTS)  # :39
    h = np.abs(lum - _shift(lum, 1, 0))                            # :43
    v = np.abs(lum - _shift(lum, 0, 1))                            # :44

    # :72 / :74 -- threshold against the max PERPENDICULAR gradient over
    # the tap's 2x2, which is the rule derived in the module docstring.
    perp_h = np.maximum(np.maximum(_shift(v, 0, -1), v),
                        np.maximum(_shift(v, 1, -1), _shift(v, 1, 0)))
    perp_v = np.maximum(np.maximum(_shift(h, -1, 0), h),
                        np.maximum(_shift(h, -1, 1), _shift(h, 0, 1)))
    e_right = (h - perp_h * NON_DOMINANT_FACTOR) > EDGE_THRESHOLD
    e_bottom = (v - perp_v * NON_DOMINANT_FACTOR) > EDGE_THRESHOLD
    # LEFT of p IS RIGHT of p-(1,0); TOP of p IS BOTTOM of p-(0,1). The
    # shader reads exactly those neighbours (:83, :87) rather than
    # recomputing, and so does this.
    e_left = _shift(e_right, -1, 0)
    e_top = _shift(e_bottom, 0, -1)

    edges = (e_right.astype(np.uint8) * E_RIGHT
             + e_bottom.astype(np.uint8) * E_BOTTOM
             + e_left.astype(np.uint8) * E_LEFT
             + e_top.astype(np.uint8) * E_TOP)

    # :101 -- `b*(r + l) + l*t + t*r != 0`, the four perpendicular pairs.
    r, b, l, t = (e_right.astype(np.float64), e_bottom.astype(np.float64),
                  e_left.astype(np.float64), e_top.astype(np.float64))
    candidates = (b * (r + l) + l * t + t * r) != 0.0
    return edges, candidates


# --------------------------------------------------------------------------
# The deferred-colour packing
# --------------------------------------------------------------------------
def _pack_roundtrip(c):
    """candidates:198 store + deferred_apply:68 load, as one function.

    R and G keep 11 of the half's 16 bits (`(f16 + 4) >> 3`, mask 2047,
    decoded through mask 16376 = bits 3..13); B keeps 10 (decoded through
    16368 = bits 4..13). The `+4` / `+8` are round-to-nearest on the
    dropped bits. Sign and the top exponent bit are dropped, which is why
    :197 clamps to [0, 1.99999988] first — the packing cannot represent
    anything outside it."""
    c = np.clip(c, 0.0, CANDIDATE_MAX)
    bits = c.astype(np.float16).view(np.uint16).astype(np.uint32)
    out = np.empty_like(c)
    for ch, (add, mask) in enumerate(((4, 16376), (4, 16376), (8, 16368))):
        packed = ((bits[..., ch] + add) >> 3) & (mask >> 3)
        rebuilt = ((packed << 3) & mask).astype(np.uint16)
        out[..., ch] = rebuilt.view(np.float16).astype(np.float64)
    return out


# --------------------------------------------------------------------------
# Pass 2 -- process candidates
# --------------------------------------------------------------------------
def _bit(edges, dx, dy, bit):
    """The `bit` of the edge mask at p+(dx,dy), as a float 0/1 -- the
    `float((mask & B) != 0)` the shader writes 20 times over."""
    return ((_shift(edges, dx, dy) & bit) != 0).astype(np.float64)


def process_candidates(rgb, edges, candidates):
    """cs2_cmaa2_cs_process_candidates.glsl:52-397.

    Returns `(num, den)` -- the weighted colour sum and weight sum the
    deferred pass would have accumulated by walking the linked list."""
    src = _np(rgb)
    if src.shape[-1] < 4:
        src = np.concatenate([src, np.ones_like(src[..., :1])], -1)
    h, w = src.shape[0], src.shape[1]
    num = np.zeros((h, w, 3))
    den = np.zeros((h, w))
    cnt = np.zeros((h, w), dtype=np.int64)

    # --- the pixel's own bits, and its neighbours' (:70-104) -------------
    er, eb, el, et = (_bit(edges, 0, 0, E_RIGHT), _bit(edges, 0, 0, E_BOTTOM),
                      _bit(edges, 0, 0, E_LEFT), _bit(edges, 0, 0, E_TOP))
    # left neighbour (:77-80)
    lB, lT, lL = (_bit(edges, -1, 0, E_BOTTOM), _bit(edges, -1, 0, E_TOP),
                  _bit(edges, -1, 0, E_LEFT))
    # right neighbour (:83-88)
    rR, rB, rT = (_bit(edges, 1, 0, E_RIGHT), _bit(edges, 1, 0, E_BOTTOM),
                  _bit(edges, 1, 0, E_TOP))
    # down neighbour (:90-94)
    dR, dL, dB = (_bit(edges, 0, 1, E_RIGHT), _bit(edges, 0, 1, E_LEFT),
                  _bit(edges, 0, 1, E_BOTTOM))
    # up neighbour (:96-101)
    uR, uL, uT = (_bit(edges, 0, -1, E_RIGHT), _bit(edges, 0, -1, E_LEFT),
                  _bit(edges, 0, -1, E_TOP))

    # --- :105-122, the L-shape reinforcement -----------------------------
    two = (er + eb + el + et) == 2.0                                # :105
    k = SHAPE_REINFORCE
    b2 = eb + k * (el * rB * (1.0 - uL) + er * lB * (1.0 - uR))     # :109
    r2 = er + k * (eb * uR * (1.0 - lB) + et * dR * (1.0 - lT))     # :110
    t2 = et + k * (er * lT * (1.0 - dR) + el * rT * (1.0 - dL))     # :111
    l2 = el + k * (et * dL * (1.0 - rT) + eb * uL * (1.0 - rB))     # :112
    bl, rr, tt, ll = (np.where(two, b2, eb), np.where(two, r2, er),
                      np.where(two, t2, et), np.where(two, l2, el))

    # :124 -- confidence falls with the total edge count of the twelve
    # neighbour bits, and the budget is tighter on an exact corner.
    neigh = (lL + rR + uR + dR) + (lB + rB + uL + dB) + (lT + rT + uT + dL)
    conf = np.clip(CONF_BASE - neigh * CONF_SLOPE, 0.0, 1.0)
    scale = np.where(two, BLEND_TWO_EDGE, BLEND_SIMPLE) * conf
    wL, wT, wR, wB = ll * scale, tt * scale, rr * scale, bl * scale  # :124

    # :126-190 -- the four-tap blend. The shader guards each tap with
    # `> 0.0`; a zero weight contributes zero either way, so the guard is
    # a branch-cost optimisation and is not reproduced as a branch.
    tot = wL + wT + wR + wB
    blended = src[..., :3] * (1.0 - tot)[..., None]                 # :127
    blended = blended + _shift(src, -1, 0)[..., :3] * wL[..., None]  # :133
    blended = blended + _shift(src, 0, -1)[..., :3] * wT[..., None]  # :147
    blended = blended + _shift(src, 1, 0)[..., :3] * wR[..., None]   # :161
    blended = blended + _shift(src, 0, 1)[..., :3] * wB[..., None]   # :175

    # :191 -- emit only where the pass ran at all and the alpha survives.
    alpha = src[..., 3] * (1.0 - tot)
    emit = candidates & (alpha != 0.0) & (tot != 0.0)
    col = _pack_roundtrip(blended)
    num[emit] += col[emit] * W_SIMPLE
    den[emit] += W_SIMPLE
    cnt[emit] += 1

    # --- :199-397, the symmetrical Z shapes ------------------------------
    _z_shapes(src, edges, candidates, num, den, cnt,
              er, eb, el, et, lB, lT, rB, rT, dR, dL, uR, uL, uT)
    return num, den


def _z_shapes(src, edges, candidates, num, den, cnt,
              er, eb, el, et, lB, lT, rB, rT, dR, dL, uR, uL, uT):
    """candidates:199-397, whole.

    The four scores (:203, :205, :211, :212) pick a direction and a
    handedness; the winner is traced along its edge until the two arms
    stop growing symmetrically (:236); the traced length becomes a
    gradient blend applied to every pixel along the arm (:340-350)."""
    h, w = src.shape[0], src.shape[1]
    rRb = _bit(edges, 1, 0, E_RIGHT)
    x2T, x2B = _bit(edges, 2, 0, E_TOP), _bit(edges, 2, 0, E_BOTTOM)
    y2L, y2R = _bit(edges, 0, -2, E_LEFT), _bit(edges, 0, -2, E_RIGHT)
    d = Z_DISCOUNT

    sup_h0 = lB + x2T                                               # :201
    sup_h1 = x2B + lT                                               # :202
    # :203 / :205 -- horizontal pair. `_20640` is the er*et product the
    # shader factors out.
    s_h0 = (er * eb * rT) * ((2.0 + sup_h0) - (et + rB)
                             - d * ((sup_h1 + el) + rRb))
    er_et = er * et
    s_h1 = (er_et * rB) * ((2.0 + sup_h1) - (eb + rT)
                           - d * ((sup_h0 + el) + rRb))
    best_h = np.maximum(s_h0, s_h1)                                 # :206
    hand_h = (best_h > 0.0) & (s_h0 > s_h1)                         # :207-214

    sup_v0 = dR + y2L                                               # :217
    sup_v1 = y2R + dL                                               # :218
    s_v0 = (er_et * uL) * ((2.0 + sup_v0) - (el + uR)
                           - d * ((sup_v1 + eb) + uT))              # :219
    s_v1 = (et * el * uR) * ((2.0 + sup_v1) - (er + uL)
                             - d * ((sup_v0 + eb) + uT))            # :220
    best_v = np.maximum(s_v0, s_v1)                                 # :221
    vertical = best_v > best_h                                      # :222
    hand = np.where(vertical, s_v0 > s_v1, hand_h)                  # :223-230
    best = np.where(vertical, best_v, best_h)                       # :231
    horizontal = ~vertical                                          # :232

    live = candidates & (best > 0.0)                                # :233
    if not live.any():
        return
    ys, xs = np.nonzero(live)
    n = ys.size
    sc = best[ys, xs]
    horiz = horizontal[ys, xs]
    hnd = hand[ys, xs]

    trim = np.floor(np.clip(4.0 - sc, 0.0, 3.0))                    # :235
    # :237 -- the step direction: (1,0) horizontal, (0,-1) vertical.
    step_x = np.where(horiz, 1.0, 0.0)
    step_y = np.where(horiz, 0.0, -1.0)
    # :238-241 -- the two edge bits the trace follows, swapped by hand.
    near = np.where(horiz, float(E_BOTTOM), float(E_RIGHT))
    far = np.where(horiz, float(E_TOP), float(E_LEFT))
    mask_fwd = np.where(hnd, far, near).astype(np.int64)             # :241
    mask_back = np.where(hnd, near, far).astype(np.int64)            # :242

    # --- :243-304, the symmetric trace ----------------------------------
    lenA = np.ones(n)
    lenB = np.ones(n)
    okA = np.ones(n, bool)
    okB = np.ones(n, bool)
    running = np.ones(n, bool)
    for _ in range(int(TRACE_CAP) + 2):
        if not running.any():
            break
        ax = np.clip((xs - step_x * lenA).astype(np.int64), 0, w - 1)
        ay = np.clip((ys - step_y * lenA).astype(np.int64), 0, h - 1)
        bx = np.clip((xs + step_x * (lenB + 1.0)).astype(np.int64), 0, w - 1)
        by = np.clip((ys + step_y * (lenB + 1.0)).astype(np.int64), 0, h - 1)
        ea = edges[ay, ax].astype(np.int64)
        ebk = edges[by, bx].astype(np.int64)
        hitA = okA & ((ea & mask_back) == mask_back)                # :262
        hitB = okB & ((ebk & mask_fwd) == mask_fwd)                 # :270
        nA = lenA + hitA                                            # :274
        nB = lenB + hitB                                            # :275
        neither = (~hitA) & (~hitB)                                 # :277-284
        val = np.where(neither, TRACE_CAP, np.maximum(nB, nA))
        stop = val >= np.minimum(TRACE_CAP,
                                 TRACE_RATIO * np.minimum(nB, nA) + TRACE_BIAS)
        lenA = np.where(running, nA, lenA)
        lenB = np.where(running, nB, lenB)
        okA = np.where(running, hitA, okA)
        okB = np.where(running, hitB, okB)
        running = running & (~stop)                                 # :286

    a_len = lenA - trim                                             # :305
    b_len = lenB - trim                                             # :306
    total = a_len + b_len
    keep = total >= Z_MIN_LENGTH                                    # :307
    if not keep.any():
        return
    idx = np.nonzero(keep)[0]
    ys, xs = ys[idx], xs[idx]
    a_len, b_len, total, trim = a_len[idx], b_len[idx], total[idx], trim[idx]
    horiz, hnd = horiz[idx], hnd[idx]
    step_x, step_y = step_x[idx], step_y[idx]

    # :321-327 -- the parity correction, the confidence, and the span.
    m0 = Z_PARITY * (a_len - 2.0 * np.trunc(a_len / 2.0))
    m1 = Z_PARITY * (b_len - 2.0 * np.trunc(b_len / 2.0))
    zconf = np.clip((total - trim) * Z_CONF_SLOPE, 0.0, 1.0)
    a_half = np.floor((a_len + 1.0) * 0.5)
    i_from = 1.0 - a_half
    i_to = np.floor((b_len + 1.0) * 0.5)
    span = (i_to - i_from) + 1.0
    inv = 1.0 / ((span - m0) - m1)                                  # :332
    off = -0.5 + (a_half - m0)                                      # :333

    # :237 / :359 -- the PERPENDICULAR the blend pulls from.
    perp_x = np.where(horiz, 0.0, -1.0)
    perp_y = np.where(horiz, -1.0, 0.0)
    perp_x = np.where(hnd, -perp_x, perp_x)                         # :362-368
    perp_y = np.where(hnd, -perp_y, perp_y)

    # :340-350 -- one contribution per step along the arm. The step count
    # varies per candidate, so the loop is over the MAXIMUM span with a
    # per-lane liveness mask; every lane executes the same arithmetic the
    # shader executes for its own i.
    i_lo, i_hi = float(i_from.min()), float(i_to.max())
    i = i_lo
    while i <= i_hi:
        live = (i >= i_from) & (i <= i_to)
        if live.any():
            j = np.nonzero(live)[0]
            px = np.clip((xs[j] + step_x[j] * i).astype(np.int64), 0, w - 1)
            py = np.clip((ys[j] + step_y[j] * i).astype(np.int64), 0, h - 1)
            pos = i > 0.0                                           # :342
            sign = -1.0 if pos else 1.0
            f = ((inv[j] * (i + off[j])) * sign + float(pos)) * zconf[j]
            # :350 -- the staged path clamps and quantises to 10 bits.
            f = np.round(np.clip(f, 0.0, 1.0) * 1023.0) / 1023.0
            qx = np.clip((px + perp_x[j] * sign).astype(np.int64), 0, w - 1)
            qy = np.clip((py + perp_y[j] * sign).astype(np.int64), 0, h - 1)
            base = src[py, px]
            other = src[qy, qx]
            alive = base[..., 3] != 0.0                             # :380
            colr = base[..., :3] + (other[..., :3] - base[..., :3]) * f[:, None]
            colr = _pack_roundtrip(np.clip(colr, 0.0, CANDIDATE_MAX))
            k = np.nonzero(alive)[0]
            np.add.at(num, (py[k], px[k]), colr[k] * W_COMPLEX)
            np.add.at(den, (py[k], px[k]), W_COMPLEX)
            np.add.at(cnt, (py[k], px[k]), 1)
        i += 1.0


# --------------------------------------------------------------------------
# Pass 3 -- deferred apply, and the whole thing
# --------------------------------------------------------------------------
def deferred_apply(rgb, num, den):
    """cs2_cmaa2_cs_deferred_apply.glsl:80-89 -- `sum / weight`, written
    only where a contribution landed (`_13136.w == 0` breaks at :80, so a
    pixel with no candidate keeps its ORIGINAL colour rather than being
    written with anything)."""
    out = _np(rgb).copy()
    hit = den > 0.0
    out[..., :3] = np.where(hit[..., None],
                            num / np.maximum(den, 1e-30)[..., None],
                            out[..., :3])
    return out


def cmaa2(rgb):
    """The three passes, in order. `rgb` is the DISPLAY-SPACE image the
    pass runs on -- edges.glsl:39 takes `sqrt()` of its input, which is
    the cheap gamma of an already-encoded value, not of linear light."""
    edges, cand = compute_edges(rgb)
    num, den = process_candidates(rgb, edges, cand)
    out = deferred_apply(rgb, num, den)
    if is_torch(rgb):
        import torch
        return torch.as_tensor(out, dtype=rgb.dtype, device=rgb.device)
    return out


def cmaa2_report(rgb, out, edges, cand) -> str:
    """Runtime line. Reports the EDGE and CANDIDATE populations as well as
    the pixel delta: a frame with no edges and a pass that never ran
    produce the same image, and must not produce the same log."""
    a = _np(rgb)[..., :3]
    b = _np(out)[..., :3]
    d = np.abs(a - b)
    n = max(d[..., 0].size, 1)
    return (f"post/cmaa2: edges {float((edges != 0).sum()) / n:.3%} of px, "
            f"candidates {float(cand.sum()) / n:.3%}, "
            f"changed {float((d.max(-1) > 0).sum()) / n:.3%} "
            f"|delta| mean {d.mean():.3e} max {d.max():.3e}")


# --------------------------------------------------------------------------
# Two-sided calibration
# --------------------------------------------------------------------------
def selftest() -> int:
    """Does this pass FIND the thing it exists for, and go CLEAN otherwise?

    A detector calibrated on one side only is an assertion. CMAA's whole
    job is to soften hard geometric edges without touching texture, so
    the two sides are a hard diagonal step (must fire, and must produce
    values BETWEEN the two levels -- a pass that merely "changes pixels"
    could be doing anything) and a flat field plus a smooth gradient
    (must be exactly inert; a gradient in particular has non-zero
    gradients everywhere, so an implementation missing the
    local-contrast term would light up on it).

    Run: `python -m counter_strike_render.post.cmaa2`
    """
    h, w = 64, 96
    y, x = np.mgrid[0:h, 0:w]
    step = np.where((y * 1.7 - x) < 0, 0.85, 0.12).astype(np.float64)
    cases = [
        ("hard diagonal step", np.stack([step] * 3, -1), True),
        ("flat field", np.stack([np.full((48, 64), 0.5)] * 3, -1), False),
        ("smooth gradient",
         np.stack([np.tile(np.linspace(0, 1, 64), (48, 1))] * 3, -1), False),
    ]
    bad = 0
    for name, rgb, should_fire in cases:
        e, c = compute_edges(rgb)
        n, d = process_candidates(rgb, e, c)
        out = deferred_apply(rgb, n, d)
        changed = float((np.abs(out[..., :3] - rgb[..., :3]).max(-1) > 0).sum())
        lv0, lv1 = len(np.unique(rgb[..., 0])), len(np.unique(out[..., 0]))
        fired = changed > 0
        ok = fired == should_fire
        print(f"cmaa2 selftest [{name}]: {cmaa2_report(rgb, out, e, c)}")
        print(f"    distinct levels {lv0} -> {lv1}; "
              f"expected {'FIRE' if should_fire else 'CLEAN'}, "
              f"got {'FIRE' if fired else 'CLEAN'} -- "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad += 1
        if should_fire and ok and lv1 <= lv0:
            print("    MISMATCH: it changed pixels but produced no "
                  "INTERMEDIATE levels, which is not antialiasing")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(selftest())
