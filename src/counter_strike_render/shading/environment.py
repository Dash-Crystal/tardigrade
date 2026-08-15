#!/usr/bin/env python3
"""`csgo_environment` — the lightmap subsystem, from the reference modules.

Sources of truth, both committed and cited by line:

  q0  docs/projects/counter-strike-sft/csgo_environment_ps.glsl
      static combo 0 x dynamic 4 (D_BAKED_LIGHTING_FROM_LIGHTMAP),
      record 0 module 3, sha256 36b067d51788a6c5..., 2,304 FLOPs/px,
      24 fetch sites.
  q1  docs/projects/counter-strike-sft/csgo_environment_ps_shaderquality1.glsl
      static combo 16 x dynamic 4, record 8 module 3, sha256
      c05ce145c6b94f07..., 2,975 FLOPs/px, 55 fetch sites.

WHY THIS FILE EXISTS AND WHY IT IS SEPARATE FROM gpu_render.py

Defect #57 -- our `--shader-quality 1` losing to `--shader-quality 0`
against the q1 ground truth, which is the gate corpus -- was localised to
ONE term: `_gt_bicubic4`, carrying **83.8% of the MSE of our q0->q1 delta
and 100% of the ncc regression** (Q1_LOCALISATION.md). Reverting it alone
recovered ncc 0.4013 -> 0.4185 against GT preset0, where q0 sits at
0.4176.

And the reason it was the defect is recorded in its own docstring: a
fetch COUNT (2 -> 8 sites) was read out of a callflow document, and the
KERNEL was **chosen** to produce that count. Its author wrote the
uncertainty down -- "no q=1 csgo_environment GLSL is committed, so the
exact tap pattern could not be read" -- and the READ/inferred boundary
landed exactly on the term carrying 83.8% of the divergence.

That GLSL is committed now. What it says is more interesting than "the
guess was wrong":

  * the kernel FAMILY was right. The reference is a 4-tap cubic B-spline
    (Sigg-Hadwiger) and so was the guess; the weight polynomials are
    algebraically identical.
  * the ADDRESSING was wrong, in a way this codebase has already been
    bitten by once. Ours mapped `u * (N - 1)` -- corner to corner. The
    reference maps texel CENTRES. `cube_sample()`'s docstring in
    gpu_render.py records finding and fixing exactly this: "an earlier
    version used u*(N-1), which maps corner-to-corner and compresses the
    whole face onto the texel CENTRES -- at N=4 that is a 12.5% angular
    warp toward face centres."
  * the FILTERING was wrong: the reference issues four `textureGrad`
    calls carrying the derivatives of the ORIGINAL uv, so all four taps
    share the unfiltered footprint's mip level. Ours took four plain
    bilinear taps with no LOD at all.
  * and **four of the six extra fetches are not a filter at all.** The
    q1 module binds a THIRD lightmap page and runs a direction decode
    and a directional reconstruction off it. At q1 the reference
    lightmap RESPONDS TO THE NORMAL MAP; at q0 it cannot; ours cannot at
    either quality. For the 95 of 153 materials on
    D_BAKED_LIGHTING_FROM_LIGHTMAP that is the largest behavioural
    difference between the two quality levels, and we implemented the
    widening without the thing being widened.

DUAL BACKEND, AND WHY IT IS NOT A CONVENIENCE

Every function here runs on numpy arrays or torch tensors. The renderer
calls them on the GPU; `conformance/terms/csgo_environment.py` registers
them as `impl=` and the runner calls the SAME code objects on numpy. If
the runner had to call a numpy re-implementation, the thing under test
would be the re-implementation -- which is the SELF_PAIRED case the
registry refuses, arrived at by a longer route.

WHAT IS STILL MISSING, NAMED RATHER THAN FAKED

`lightmap_q1()` takes `dir_page` and returns unchanged irradiance when it
is None, and says so through the `note` it returns. **We have no
direction page.** The pack ships irradiance and occlusion; the third
array is not extracted. It is an ASSET hole of the same class as the
cube array that turned out to be why the environment specular had been
returning exactly zero on every render this project ever made -- and it
is recorded here, in the ledger, and at runtime, rather than being
approximated by something plausible.
"""

from ..passes._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.

try:
    import torch
except ImportError:                                        # pragma: no cover
    torch = None


def _xp(a):
    """numpy or torch, chosen by what the caller handed us."""
    if torch is not None and isinstance(a, torch.Tensor):
        return torch
    return np


def _clip(a, lo, hi):
    xp = _xp(a)
    if xp is np:
        return np.clip(a, lo, hi)
    # torch refuses a positional (Tensor, int, Tensor) mix; keywords take
    # tensor or scalar bounds interchangeably.
    lo = lo if hasattr(lo, "dtype") else xp.as_tensor(lo, dtype=a.dtype,
                                                      device=a.device)
    hi = hi if hasattr(hi, "dtype") else xp.as_tensor(hi, dtype=a.dtype,
                                                      device=a.device)
    return xp.clamp(a, min=lo, max=hi)


def _stack(parts, axis=-1):
    return _xp(parts[0]).stack(parts, axis) if _xp(parts[0]) is torch \
        else np.stack(parts, axis)


# =====================================================================
# TEXEL ADDRESSING -- one definition, because two readers of one atlas
# is how the bug recurred
# =====================================================================
#
# `TEXEL_ADDRESSING_SWEEP.md`: corner-to-corner addressing was found and
# fixed in `cube_sample()`, stood for the project's whole life in
# `_gt_tex2d`, and then turned up A THIRD TIME in `_atlas()` -- all
# three reading the same class of input. The transferable finding was
# that the defect recurred **wherever the atlas is read**, because each
# reader was written against the array's SHAPE rather than against the
# sampler convention the reference uses.
#
# A per-reader fix that leaves three readers is a fourth instance
# waiting. These two functions are the single definition; every reader
# calls one of them.

def texel_coord(u, v, wt, ht):
    """Normalised UV -> continuous texel coordinate. TEXEL CENTRES.

        x = u*W - 0.5 ; y = v*H - 0.5

    This is what `texture(sampler2D, uv)` does: texel i covers
    [i/W, (i+1)/W) and its centre is at (i+0.5)/W, so the inverse is
    `u*W - 0.5`.

    NOT `u*(W-1)`, which maps u=0 and u=1 to the CENTRES of the first
    and last texels and compresses the page by (W-1)/W. On the 64-wide
    lightmap pages this pack ships that is a half-texel shift plus a
    64/63 scale.

    NOT `u*(W-1) + 0.5` either -- that is the colour-LUT convention
    (`_apply_cc`'s `--lut-texel halftexel`), correct for a table whose
    endpoints are defined AT its domain's extremes and wrong for a
    spatial sampler. The two are genuinely different and this is why the
    sweep audited each site instead of substituting globally.
    """
    return u * wt - 0.5, v * ht - 0.5


def nearest_index(u, v, wt, ht):
    """Nearest-neighbour tap under the texel-centre convention.

        i = clamp(floor(u*W), 0, W-1)

    `floor(u*W)` and `round(u*W - 0.5)` are the same integer, and both
    differ from `(u*(W-1)).long()` by up to one whole texel near the
    page edges. On a NEAREST tap the half-texel shift is not blurred by
    interpolation -- it quantises into the wrong luxel outright, which
    at an atlas chart edge means reading a neighbouring chart's
    irradiance.
    """
    xp = _xp(u)
    x = xp.floor(u * wt)
    y = xp.floor(v * ht)
    return (_clip(x, 0, wt - 1), _clip(y, 0, ht - 1))


def texel_coord_flat(u, v, wt, ht):
    """`texel_coord` stacked. ADAPTER for the array-shaped runner."""
    x, y = texel_coord(u, v, wt, ht)
    return _stack([x, y], -1)


def nearest_index_flat(u, v, wt, ht):
    """`nearest_index` stacked. ADAPTER for the array-shaped runner."""
    x, y = nearest_index(u, v, wt, ht)
    return _stack([x, y], -1)


# =====================================================================
# The 4-tap cubic B-spline, addressed the way the reference addresses it
# csgo_environment_ps_shaderquality1.glsl:841-862
# =====================================================================

# Q1:849-852. The shipped constant, not exact 1/6 -- and the difference
# is measurable: the four weights sum to 1 only to 2.980e-08, which is
# 4x this value's f32 error against 1/6. Using exact 1/6 here would make
# our partition tighter than Valve's, which is a different renderer.
BSPLINE_SIXTH = 0.16666667163372039794921875


def bspline4_weights(uv, texel):
    """Q1:844-853 -- offsets and weights for four bilinear taps.

        p  = uv / texel - 0.5                                  Q1:844
        i  = floor(p) ; f = fract(p)                           Q1:845-846
        w1 = ( 3*f3 - 6*f2 + 4) / 6                            Q1:849
        w3 = f3 / 6                                            Q1:850
        g0 = w0 + w1  where w0 = (-f3 + 3*(f2 - f) + 1)/6      Q1:851
        g1 = w2 + w3  where w2 = (3*(f2 - f3 + f) + 1)/6       Q1:852
        h  = vec4(w1/g0 - 1, w3/g1 + 1)                        Q1:853

    `texel` is `1/textureSize` (Q1:841-842), so `uv/texel` is the TEXEL
    CENTRE coordinate. The tap positions are then `((i + h) - 0.5) *
    texel` (Q1:862) -- note the second `- 0.5`, which is in the module
    and is not the one already inside `p`.

    THE DEFECT THIS REPLACES. `gpu_render.py:_gt_bicubic4` computed
    `x = u * (wt - 1)` and stepped by `sx = 1/(wt - 1)`: corner-to-corner
    addressing, which compresses the whole page onto the texel centres.
    On a 64-wide page that is a 1/64 systematic shift plus a scale error
    of 64/63, applied to the term the localisation measured as 83.8% of
    the q1 divergence.

    Returns (i, h, g0, g1) with `h` shaped (..., 4) as `vec4(h0, h1)`
    laid out `[hx0, hy0, hx1, hy1]` -- the module's `.xy` and `.zw`.
    """
    p = uv / texel - 0.5
    xp = _xp(p)
    i = xp.floor(p)
    f = p - i
    f2 = f * f
    f3 = f2 * f
    w1 = ((f3 * 3.0) - (f2 * 6.0) + 4.0) * BSPLINE_SIXTH
    w3 = f3 * BSPLINE_SIXTH
    g0 = ((-f3) + (f2 - f) * 3.0 + 1.0) * BSPLINE_SIXTH + w1
    g1 = (((f2 - f3 + f) * 3.0 + 1.0) * BSPLINE_SIXTH) + w3
    h0 = w1 / g0 - 1.0
    h1 = w3 / g1 + 1.0
    h = _stack([h0[..., 0], h0[..., 1], h1[..., 0], h1[..., 1]], -1)
    return i, h, g0, g1


def bspline4_tap_uvs(i, h, texel):
    """Q1:862 -- the four tap coordinates, in the module's own order.

    `xy`, `zy`, `xw`, `zw` of `vec4(h0, h1)`, i.e.
    (h0.x, h0.y), (h1.x, h0.y), (h0.x, h1.y), (h1.x, h1.y).
    """
    hx0, hy0, hx1, hy1 = h[..., 0], h[..., 1], h[..., 2], h[..., 3]
    ix, iy = i[..., 0], i[..., 1]
    tx, ty = texel[..., 0], texel[..., 1]

    def _uv(hx, hy):
        return _stack([((ix + hx) - 0.5) * tx, ((iy + hy) - 0.5) * ty], -1)
    return (_uv(hx0, hy0), _uv(hx1, hy0), _uv(hx0, hy1), _uv(hx1, hy1))


def bspline4_weights_flat(uv, texel):
    """`bspline4_weights` with its four outputs concatenated.

    An ADAPTER, not a second implementation: it calls the function above
    and joins the results. The conformance runner compares arrays, and
    the natural return here is a ragged tuple ((N,2), (N,4), (N,2),
    (N,2)) that `np.asarray` cannot form. Concatenating in the adapter
    keeps the shape contract at the boundary instead of distorting the
    function the renderer calls.
    """
    i, h, g0, g1 = bspline4_weights(uv, texel)
    xp = _xp(i)
    parts = [i[..., 0], i[..., 1], h[..., 0], h[..., 1], h[..., 2],
             h[..., 3], g0[..., 0], g0[..., 1], g1[..., 0], g1[..., 1]]
    return xp.stack(parts, -1) if xp is torch else np.stack(parts, -1)


def bspline4_tap_uvs_flat(i, h, texel):
    """`bspline4_tap_uvs` with its four (u, v) pairs concatenated."""
    uvs = bspline4_tap_uvs(i, h, texel)
    xp = _xp(uvs[0])
    parts = []
    for uv in uvs:
        parts.extend([uv[..., 0], uv[..., 1]])
    return xp.stack(parts, -1) if xp is torch else np.stack(parts, -1)


def bspline4_combine(t_xy, t_zy, t_xw, t_zw, g0, g1):
    """Q1:862 -- the separable weighting of the four taps."""
    gx0, gy0 = g0[..., 0:1], g0[..., 1:2]
    gx1, gy1 = g1[..., 0:1], g1[..., 1:2]
    return (t_xy * (gx0 * gy0) + t_zy * (gx1 * gy0)
            + t_xw * (gx0 * gy1) + t_zw * (gx1 * gy1))


def bicubic4(sample, uv, texel):
    """The whole filter. `sample(uv)` issues one bilinear tap.

    The reference passes `textureGrad(..., ddx(uvOriginal), ddy(
    uvOriginal))` at all four sites (Q1:855-862) -- the derivatives of
    the ORIGINAL coordinate, not of the offset ones. So a caller whose
    `sample` selects a mip must select it from the unfiltered footprint;
    choosing per-tap would put the four taps on different levels and
    they would stop describing one surface. Stated here because the
    signature cannot enforce it.
    """
    i, h, g0, g1 = bspline4_weights(uv, texel)
    a, b, c, d = bspline4_tap_uvs(i, h, texel)
    return bspline4_combine(sample(a), sample(b), sample(c), sample(d),
                            g0, g1)


# =====================================================================
# The direction page -- the four fetches that are not a filter
# csgo_environment_ps_shaderquality1.glsl:818-831
# =====================================================================

LM_DISC_R = 0.996190845966339111328125          # Q1:822
LM_SHARP_LO = 0.100000001490116119384765625     # Q1:828
LM_SHARP_HI = 2.0                               # Q1:828
LM_EDGE0 = 0.100000001490116119384765625        # Q1:828
LM_EDGE1 = 0.00999999977648258209228515625      # Q1:828
LM_K = 0.800000011920928955078125               # Q1:828


def _smoothstep(e0, e1, x):
    t = _clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lm_direction_decode(tex_xy, curv_n, curv_p):
    """Q1:818-828 -- decode the direction page to a unit vector.

        t   = tex.xy * 2 - 1
        p   = t * (R / max(R, length(t)))                       Q1:822
        z   = sqrt(1 - dot(p, p))                               Q1:827
        k   = smoothstep(0.1, 0.01, |fwidth(n)| / |fwidth(P)|) * 0.8
        s   = mix(1, mix(0.1, 2.0, clamp((1 - z) * 1.5, 0, 1)), k)
        dir = normalize(vec3(p, z * s))                         Q1:828

    A hemispherical DISC encode. Three properties that a reasonable
    reimplementation loses:

      * the clamp is `R / max(R, |t|)` with R = 0.99619..., so it
        rescales only vectors LONGER than R and leaves shorter ones
        exactly alone. A plain normalize() would rescale every vector
        and destroy the encoded elevation. It is also what keeps `z`
        away from 0, which is why the divide at Q1:831 is unguarded --
        the clamp is load-bearing, not cosmetic.
      * `s` runs from 0.1 to 2.0, so it can DOUBLE the z component. This
        is a sharpening of the directional lobe, not a normalisation.
      * the smoothstep edges run DOWNWARD (0.1 -> 0.01), so flat
        surfaces get sharpened and curved ones are left alone. Reversing
        them inverts which surfaces are affected.

    `curv_n` and `curv_p` are `length(fwidth(...))` of the geometric
    normal and the world position; the caller owns the derivative,
    because a vectorised port has no `fwidth`.
    """
    xp = _xp(tex_xy)
    t = tex_xy * 2.0 - 1.0
    ln = xp.sqrt((t * t).sum(-1))[..., None]
    p = t * (LM_DISC_R / xp.maximum(ln, xp.full_like(ln, LM_DISC_R)))
    z = xp.sqrt(_clip(1.0 - (p * p).sum(-1), 0.0, 1.0))
    k = _smoothstep(LM_EDGE0, LM_EDGE1,
                    curv_n / xp.maximum(curv_p,
                                        xp.full_like(curv_p, 1e-30))) * LM_K
    inner = LM_SHARP_LO + (LM_SHARP_HI - LM_SHARP_LO) \
        * _clip((1.0 - z) * 1.5, 0.0, 1.0)
    s = 1.0 + (inner - 1.0) * k
    v = _stack([p[..., 0], p[..., 1], z * s], -1)
    return v / xp.sqrt((v * v).sum(-1))[..., None]


def lm_directional_reconstruct(lm_rgb, conf, bias, lm_dir, n_tangent):
    """Q1:829-831 -- redistribute the baked irradiance by the normal.

        amb = lm.rgb * clamp(dirPage.z + g_flLmBias, 0, 1)       Q1:829
        out = amb + ((lm.rgb - amb) / dir.z)
                    * max(0, dot(dir, normalize(nTangent)))      Q1:831

    The total is split into an ambient floor scaled by the direction
    page's confidence channel, and a directional remainder redistributed
    by the cosine between the baked direction and the shading normal,
    divided by `dir.z` so the split is energy-preserving when the two
    coincide.

    q0 has no counterpart: it uses the lightmap flat (:756-758, then
    :1227-1228 multiply it by nothing). So **at q1 the lightmap responds
    to the normal map and at q0 it cannot** -- and that, not the filter,
    is what the quality axis buys on the 95-of-153 materials that take
    D_BAKED_LIGHTING_FROM_LIGHTMAP.
    """
    xp = _xp(lm_rgb)
    amb = lm_rgb * _clip(conf + bias, 0.0, 1.0)[..., None]
    nt = n_tangent / xp.sqrt((n_tangent * n_tangent).sum(-1))[..., None]
    c = xp.maximum((lm_dir * nt).sum(-1),
                   xp.zeros_like(lm_dir[..., 0]))[..., None]
    return amb + ((lm_rgb - amb) / lm_dir[..., 2:3]) * c


# =====================================================================
# The family path
# =====================================================================
def lightmap(quality, sample_irr, sample_occ, uv, texel,
             sample_dir=None, curv_n=None, curv_p=None, n_tangent=None,
             lm_bias=0.0):
    """The `D_BAKED_LIGHTING_FROM_LIGHTMAP` path at either quality.

    Returns `(irradiance, occlusion4, notes)`. `notes` is a list of
    strings naming every input that was absent and what the module would
    have done with it -- the caller prints them, so a missing asset is a
    runtime line and not a silent default.

    q0 (csgo_environment_ps.glsl:756-758): two bilinear taps at one
    coordinate, array slice literal 0.0, irradiance used flat.

    q1: the same two arrays through the 4-tap B-spline above, PLUS a
    third page carrying direction and confidence, decoded and used to
    redistribute the irradiance. 2 -> 8 sampler2DArray sites, of which
    the callflow doc's count is the total including the DFG LUT.
    """
    notes = []
    if quality == 0:
        return sample_irr(uv), sample_occ(uv), notes
    irr = bicubic4(sample_irr, uv, texel)
    occ = bicubic4(sample_occ, uv, texel)
    if sample_dir is None:
        notes.append(
            "lightmap q1: NO DIRECTION PAGE. The reference binds a third "
            "sampler2DArray (_4565._m3, Q1:817) and redistributes the "
            "irradiance by dot(bakedDirection, normal) at Q1:829-831, so "
            "at q1 its lightmap responds to the normal map. Ours cannot: "
            "the pack ships irradiance and occlusion only. The filter "
            "widening is implemented and the thing being widened is not. "
            "ASSET HOLE, same class as the cube array.")
        return irr, occ, notes
    dpage = bicubic4(sample_dir, uv, texel)
    d = lm_direction_decode(dpage[..., 0:2], curv_n, curv_p)
    irr = lm_directional_reconstruct(irr, dpage[..., 2], lm_bias, d,
                                     n_tangent)
    return irr, occ, notes
