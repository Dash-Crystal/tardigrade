"""Cascade selection, PCF and screen-space visibility:
`deferred_particle_shadows`, `player_visibility`, `tools_grid`.

`deferred_particle_shadows` is the one that matters beyond its own pass:
it contains the ENGINE'S CASCADE SELECTOR and its PCF kernel, written out
in full, in a module small enough to read end to end. The world families
carry the same structure buried in thousands of lines; this is the same
arithmetic in 181.
"""
from .._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.


# ---------------------------------------------------------------------------
# deferred_particle_shadows -- the cascade path
# ---------------------------------------------------------------------------
def uv_inside_unit_square(uv):
    """dot(clamp(uv, 0, 1) - uv, vec2(1.0)) == 0.0

    A branchless inside test. It is NOT equivalent to
    `all(uv >= 0 && uv <= 1)` in general: the two clamped deviations are
    SUMMED before the comparison, so an overshoot on one axis could in
    principle cancel an undershoot on the other. It cannot here, because
    clamp(x,0,1) - x is <= 0 for x > 1 and >= 0 for x < 0 and both signs
    can appear -- so a point at (-a, 1+a) sums to exactly zero and reads
    as INSIDE. That is a real property of the selector, not a rewrite of
    a bounds check, and it is transcribed rather than repaired."""
    return np.einsum("...i,i->...", np.clip(uv, 0.0, 1.0) - uv,
                     np.like([1.0, 1.0], uv)) == 0.0


def pcf_gather_16(g0, g1, g2, g3):
    """Four textureGatherOffset results, each dotted with vec4(0.0625),
    summed.

    Sixteen depth comparisons at 1/16 each -- a 4x4 PCF built from four
    2x2 gathers at offsets (-2,-2), (0,-2), (-2,0), (0,0). The weight is a
    LITERAL 0.0625 in all four dots, so the kernel is a box, not a
    gaussian, and no tap is privileged."""
    w = 0.0625
    return (np.einsum("...i->...", g0) * w + np.einsum("...i->...", g1) * w
            + np.einsum("...i->...", g2) * w + np.einsum("...i->...", g3) * w)


def cascade_border_factor(uv, inset, slope):
    """1 - (1 - sat((|u-0.5| - inset) * slope)) * (1 - sat((|v-0.5| - inset) * slope))

    Zero in the cascade's interior and rising toward its edge on EITHER
    axis, because the two per-axis terms are combined as a product of
    complements. It is the blend weight toward the NEXT cascade, so a
    sign error here does not brighten or darken -- it swaps which cascade
    a pixel reads, which is far harder to see and exactly the class of
    defect this project has already shipped once."""
    t = 1.0 - np.clip((np.abs(uv - 0.5) - inset[..., None])
                      * slope[..., None], 0.0, 1.0)
    return 1.0 - t[..., 0] * t[..., 1]


def cascade_blend(shadow_this, shadow_next, factor):
    """mix(this, next, clamp(f, 0, 1)) -- and `next` is 1.0 (unshadowed)
    when this is the LAST cascade, not the first. Wrapping around would
    blend the far cascade into the near one at the horizon."""
    f = np.clip(factor, 0.0, 1.0)
    return shadow_this + (shadow_next - shadow_this) * f


def shadow_distance_fade(shadow, dist, start, inv_range):
    """mix(shadow, 1.0, clamp((d - start) * invRange, 0, 1))

    Fades toward UNSHADOWED with distance. The fade is applied only when
    at least one cascade exists (`_5464._m8 > 0`); with none the raw value
    passes through, so a scene with shadows disabled is not silently
    whitened by this term."""
    return shadow + (1.0 - shadow) * np.clip((dist - start) * inv_range, 0.0, 1.0)


def shadow_uv_from_cascade(uv, scale_bias, jitter):
    """(uv * sb.zw + sb.xy) + jitter

    The per-frame jitter is added AFTER the atlas scale/bias, i.e. in
    atlas texel space, so the same jitter shifts every cascade by the same
    number of texels rather than the same world distance."""
    return uv * scale_bias[..., 2:] + scale_bias[..., :2] + jitter


# ---------------------------------------------------------------------------
# player_visibility
# ---------------------------------------------------------------------------
def visibility_tap_mask(occlusion_x):
    """step(9.9999997473787516355514526367188e-06, x)

    A 1e-5 threshold, not zero: the target is written by an additive pass
    and its idle value is small-but-nonzero. Testing `> 0` here lights up
    every pixel."""
    return np.where(occlusion_x < 9.9999997473787516355514526367188e-06, 0.0, 1.0)


def visibility_split_means(acc_lit, acc_unlit, n_lit, n_total=16.0):
    """lit / max(1, n) and unlit / max(1, 16 - n)

    TWO means over the same 4x4 footprint, partitioned by the mask, each
    guarded by max(1, .) so an empty partition returns its accumulator
    rather than a nan. 16 is the tap count as a literal."""
    return (acc_lit / np.maximum(1.0, n_lit)[..., None],
            acc_unlit / np.maximum(1.0, n_total - n_lit)[..., None])


def visibility_contrast_boost(luma_unlit, luma_lit):
    """d = L_unlit - L_lit, then the boost is
       (L_unlit + max(0, 4d)) + min(0, 2d)

    ASYMMETRIC by design: a positive difference is amplified 4x and a
    negative one only 2x, so the silhouette brightens harder than it
    darkens. Using one gain for both makes the outline vanish against a
    brighter background."""
    d = luma_unlit - luma_lit
    return (luma_unlit + np.maximum(0.0, d * 4.0)) + np.minimum(0.0, d * 2.0)


def visibility_edge_weight(count_weight, coverage, contrast, luma_unlit, luma_lit):
    """mix(c*c, 1, coverage) * pow(1 - sat(|d| * smoothstep(0.7, 0.25, Lu*Ll)), 8)

    The smoothstep is DESCENDING and its argument is the PRODUCT of the
    two lumas, so the contrast term is suppressed where both sides are
    bright. The eighth power makes the whole factor near-binary."""
    d = luma_unlit - luma_lit
    s = _ss(0.699999988079071044921875, 0.25, luma_unlit * luma_lit)
    base = count_weight * count_weight
    return (base + (1.0 - base) * coverage) * np.power(
        1.0 - np.clip(np.abs(d) * s, 0.0, 1.0), 8.0)


# ---------------------------------------------------------------------------
# tools_grid
# ---------------------------------------------------------------------------
def grid_plane_intersect(eye, dir_, plane_origin, u_axis, v_axis):
    """eye + dir * (dot(o - eye, n) / dot(dir, n)) - o, with n = cross(u, v)

    The normal is rebuilt from the two grid axes by a cross product INSIDE
    the loop (`vec3 _7727 = cross(_5618._m0, _5618._m1);` at :43, loop-
    invariant but recomputed 25 times). Returns the hit point RELATIVE to
    the grid origin, which is what the axis dots then consume."""
    n = np.cross(u_axis, v_axis)
    t = (np.einsum("...i,...i->...", plane_origin - eye, n)
         / np.einsum("...i,...i->...", dir_, n))
    return eye + dir_ * t[..., None] - plane_origin


def grid_line_coverage(cell_coord, fw, scale):
    """1 - sat(|c - round(c)| / (fwidth * 0.7 * scale))

    Distance to the nearest grid line in cells, normalized by the SCREEN-
    SPACE derivative so the line stays a constant pixel width as the
    camera moves. 0.7 is a literal half-width factor."""
    d = np.abs(cell_coord - np.round(cell_coord))
    return 1.0 - np.clip(
        d / (fw * (0.699999988079071044921875 * scale[..., None])), 0.0, 1.0)


def grid_fade_by_derivative(fw):
    """1 - sat((fwidth - 12) * 0.083333335816860198974609375)

    0.0833333358... is float32 1/12. The grid fades out once one cell
    spans more than 12 pixels of derivative -- i.e. when it would alias --
    and is fully gone at 24. A distance-based fade would behave
    differently under a wide field of view."""
    return 1.0 - np.clip((fw - 12.0) * 0.083333335816860198974609375, 0.0, 1.0)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
