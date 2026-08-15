"""Screen-space filters: `atrous_filter`, `general_filter`, `blur`,
`blur_with_depth`, `denoise_blur`.

Five families that all look like "a weighted tap loop" and are five
DIFFERENT filters. What separates them is the weight, and every one of the
five weights is built from a different quantity:

    atrous_filter   luminance distance / sigma, depth slope, normal power
    general_filter  the tap's own alpha, times a per-tap table weight
    blur            a fixed 5-tap gaussian, in sRGB-decoded space
    blur_with_depth a fixed 5-tap gaussian gated by a STEP on a projected
                    scalar field -- taps are admitted or rejected, not faded
    denoise_blur    a linear depth-difference ramp, plus a separate
                    saturation mask on one channel only

The constant tables are READ out of the modules, not chosen: `blur` and
`blur_with_depth` ship the IDENTICAL 5-tap offsets
(-3, -1.182425022125244140625, 0, ...) and weights
(0.0044329999946057796478271484375, 0.2960419952869415283203125, ...), to
the last bit. Two families, one kernel, no fitting anywhere.
"""
from . import tables as _t  # single source for every read table
from .._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.

# Read from blur_r6m2.glsl:16-17 and blur_with_depth_r1m4.glsl:17-18.
# Identical in both modules.
# atrous_filter_r0m1.glsl:5-6
ATROUS_VAR_KERNEL = np.array([[0.25, 0.125], [0.125, 0.0625]])
ATROUS_TAP_KERNEL = np.array([1.0, 0.666666686534881591796875,
                              0.16666667163372039794921875])
# ssao_bilateral_blur_r0m0.glsl:16
# denoise_blur_r1m0.glsl:16



# ---------------------------------------------------------------------------
# atrous_filter
# ---------------------------------------------------------------------------
# Single source: every read table lives in tables.py, machine-emitted from
# the pinned .glsl. These are re-exported so callers keep one import.
GAUSS5_OFFSETS = _t.GAUSS5_OFFSETS
GAUSS5_WEIGHTS = _t.GAUSS5_WEIGHTS
ATROUS_VARIANCE_2X2 = _t.ATROUS_VARIANCE_2X2
ATROUS_TAP_3 = _t.ATROUS_TAP_3
BILATERAL_KERNEL = _t.BILATERAL_5
DENOISE_OFFSETS_5 = _t.DENOISE_OFFSETS_5
LUMA_709 = np.array([0.2125000059604644775390625,
                     0.7153999805450439453125,
                     0.07209999859333038330078125])


def luma(rgb):
    """dot(rgb, (0.2125.., 0.7154.., 0.0721..)) -- Rec.709 luma, at the
    float32 literals the module stores, not the textbook 4-digit ones."""
    return np.einsum("...i,i->...", rgb, LUMA_709)


def octahedral_normal_decode(zw):
    """The THIRD distinct normal encoding in this census, and it is not the
    other two.

        e = zw * 2 - 1
        z = (1 - |e.x|) - |e.y|
        t = clamp(-z, 0, 1)
        e += (e >= 0) ? -t : +t          [per component]
        n  = normalize(e.x, e.y, z)

    `csgo_glass` decodes a DIAGONAL pair with a 256/255 bias and an L1
    complement; `ssao` unpacks a plain 2*t-1 triple. This one is an
    octahedral map with the standard sign-folded lower hemisphere. Three
    families, three encodings; nothing here generalises from one to
    another, which is why each is transcribed separately."""
    e = zw * 2.0 - 1.0
    z = (1.0 - np.abs(e[..., 0])) - np.abs(e[..., 1])
    t = np.clip(-z, 0.0, 1.0)
    fold = np.where(e >= 0.0, -t[..., None], t[..., None])
    n = np.concatenate([e + fold, z[..., None]], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def variance_sigma(acc, scale):
    """scale * sqrt(max(0, acc + 9.9999999747524270787835121154785e-07))

    The epsilon sits INSIDE the max, so a slightly negative accumulator is
    clamped to the epsilon rather than to zero, and sigma never reaches 0
    -- which matters because it is a DENOMINATOR two lines later."""
    return scale * np.sqrt(np.maximum(0.0, acc + 9.9999999747524270787835121154785e-07))


def depth_slope_reject(depth_center, depth_tap, slope, offset_xy):
    """|dz| / (slope * |offset|), with the zero-denominator case returning 0.

    The shader tests `== 0.0` exactly and yields 0 -- NOT infinity and not
    a clamped large number. A tap at zero offset therefore gets the BEST
    possible depth score, not the worst."""
    d = slope * np.linalg.norm(offset_xy, axis=-1)
    return np.where(d == 0.0, 0.0, np.abs(depth_center - depth_tap) / np.where(d == 0.0, 1.0, d))


def atrous_weight(dz_term, luma_center, luma_tap, sigma,
                  n_center, n_tap, n_power, kernel_prod):
    """exp(-max(0,dz) - max(0,|dL|)/sigma) * pow(clamp(dot(n0,n),0,1), p) * k

    One exp over the SUM of the two rejections, not a product of two exps
    -- algebraically the same, and transcribed as written so the reduction
    order matches."""
    dl = np.maximum(0.0, np.abs(luma_center - luma_tap)) / sigma
    edge = np.exp(-np.maximum(0.0, dz_term) - dl)
    nd = np.power(np.clip(np.einsum("...i,...i->...", n_center, n_tap), 0.0, 1.0),
                  n_power)
    return edge * nd * kernel_prod


def atrous_resolve_rgb(rgb_acc, w):
    """rgb / w."""
    return rgb_acc / w[..., None]


def atrous_resolve_variance(var_acc, w):
    """variance / w SQUARED, because :141 accumulated w-squared
    contributions (`_20940 * (_6029 * _6029)`). Dividing by w -- the
    treatment the rgb channel gets one expression earlier -- leaves the
    variance biased by a factor of w, and the next a-trous iteration
    builds its sigma out of this channel."""
    return var_acc / (w * w)


# ---------------------------------------------------------------------------
# general_filter / blur / blur_with_depth: the shared sampling rect
# ---------------------------------------------------------------------------
def sample_rect(origin, size, texel):
    """The rect taps are clamped into, INSET BY 1.5 TEXELS on the far edge.

        r = vec4(origin, size + origin) * texel.xyxy
        r.zw = r.zw - texel * 1.5

    1.5 not 0.5: the far edge is pulled in by a texel and a half, which is
    what keeps a bilinear tap at the boundary from reaching outside the
    viewport's live region. Only the far edge moves; the near edge is the
    raw origin."""
    lo = origin * texel
    hi = (size + origin) * texel - texel * 1.5
    return np.concatenate([lo, hi], -1)


def clamp_to_rect(uv, rect):
    return np.clip(uv, rect[..., :2], rect[..., 2:])


# ---------------------------------------------------------------------------
# general_filter
# ---------------------------------------------------------------------------
def alpha_tap_weight(texel, tap_weight):
    """w = tap_weight * texel.a.

    The per-tap table weight is modulated by the TAP'S OWN ALPHA, so a
    transparent tap withdraws from the average instead of dragging it
    toward zero. The weight accumulator starts at
    1.0000000133514319600180897396058e-10, which is the family's divide
    guard and is read, not added by us."""
    return tap_weight * texel[..., 3]


def alpha_tap_contribution(texel, tap_weight):
    """texel * w, with w as above -- the FULL rgba, so the alpha channel
    is weighted by itself and the accumulator's .w is a sum of squares."""
    return texel * (tap_weight * texel[..., 3])[..., None]


def general_filter_resolve(acc, wsum, tint, vertex_color):
    """(acc / wsum) * tint * vertexColor."""
    return (acc * (1.0 / wsum)[..., None]) * tint * vertex_color


# ---------------------------------------------------------------------------
# blur
# ---------------------------------------------------------------------------
def srgb_to_linear(c):
    """The EXACT piecewise sRGB EOTF this module ships.

        c <= 0.040449999272823333740234375 : c * 0.077399380505084991455078125
        else : pow(c * 0.947867333889007568359375
                     + 0.052132703363895416259765625, 2.400000095367431640625)

    0.0773993805... is float32 1/12.92 and 0.9478673338... is float32
    1/1.055. They are the standard constants at the precision the bytecode
    stores; none of them is fitted, and rounding any of them to four
    digits changes the low decade visibly."""
    lin_lo = c * 0.077399380505084991455078125
    lin_hi = np.power(c * 0.947867333889007568359375
                      + 0.052132703363895416259765625, 2.400000095367431640625)
    return np.where(c <= 0.040449999272823333740234375, lin_lo, lin_hi)


def msaa_pair_average(s0, s1):
    """(s0 + s1) * 0.5 -- the module resolves TWO msaa samples inside the
    blur, before the sRGB decode, so the decode is applied to the resolved
    value and not per sample. Order matters: the EOTF is not linear."""
    return (s0 + s1) * 0.5


# ---------------------------------------------------------------------------
# blur_with_depth
# ---------------------------------------------------------------------------
def ndc_from_uv_depth(uv, depth01, m):
    """clip = M * vec4(uv.x*2-1, (1-uv.y)*2-1, depth01, 1), then /w.

    The y flip is INSIDE the ndc construction (`(1 - uv.y) * 2 - 1`), and
    the third component is the LINEARLY REMAPPED depth in [0,1], not a
    device-z. Feeding a raw device-z here produces a plausible image with
    the wrong field."""
    v = np.stack([uv[..., 0] * 2.0 - 1.0,
                  (1.0 - uv[..., 1]) * 2.0 - 1.0,
                  depth01,
                  np.ones_like(depth01)], -1)
    clip = np.einsum("...ij,...j->...i", m, v)
    return clip[..., :3] / clip[..., 3:4]


def projected_field(ndc, plane, ramp):
    """clamp(ramp.x * dot(plane, vec4(ndc,1)) + ramp.y, 0, 1) * ramp.z

    A signed-distance-to-plane run through a linear ramp and a saturate,
    then scaled. This is the quantity the tap gate compares, and it is a
    scalar FIELD -- not a depth, not an alpha."""
    d = np.einsum("...i,...i->...", plane,
                  np.concatenate([ndc, np.ones_like(ndc[..., :1])], -1))
    return np.clip(ramp[..., 0] * d + ramp[..., 1], 0.0, 1.0) * ramp[..., 2]


def depth_tap_gate(kernel_w, field_tap, field_center, tap_alpha, center_alpha,
                   lo, hi, floor_w):
    """k * step(|f_tap - f_c|, max(floor, s^2) + step(lo, a_tap + a_c))

    where s = smoothstep(lo, hi, min(f_tap, f_c)).

    A STEP, so a tap is admitted whole or rejected whole -- this filter
    does not fade taps. The inner `step` ADDS 1.0 to the tolerance when the
    two alphas together clear `lo`, which effectively disables the gate on
    already-covered pixels; reading it as a multiply inverts that."""
    s = _smoothstep(lo, hi, np.minimum(field_tap, field_center))
    tol = np.maximum(floor_w, s * s) + _step(lo, tap_alpha + center_alpha)
    return kernel_w * _step(np.abs(field_tap - field_center), tol)


def _step(edge, x):
    return np.where(x < edge, 0.0, 1.0)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def alpha_promote(alpha, field, threshold):
    """if (a > threshold) a = max(a, field); else a unchanged.

    Only ABOVE the threshold does the field get to raise alpha. Applying
    the max unconditionally would lift every transparent pixel."""
    return np.where(alpha > threshold, np.maximum(alpha, field), alpha)


# ---------------------------------------------------------------------------
# denoise_blur
# ---------------------------------------------------------------------------
def reciprocal_depth(scale, texel_x):
    """scale / texel.x -- the depth target here stores a RECIPROCAL, so the
    linear depth is a divide, not a multiply-add remap. Getting this
    backwards leaves the range weight monotone but with the wrong
    curvature."""
    return scale / texel_x


def depth_range_weight(d_center, d_tap, width):
    """1 - clamp(|d_tap - d_c| / width, 0, 1) -- a LINEAR ramp to zero, not
    an exponential and not a gaussian."""
    return 1.0 - np.clip(np.abs(d_tap - d_center) / width, 0.0, 1.0)


def unsaturated_mask(x):
    """float(x < 1.0). Taps that already reached 1.0 are excluded from the
    first channel's average entirely -- a count-based mask with its own
    accumulator, separate from the depth weight."""
    return np.astype((x < 1.0), np.float64)


def guarded_mean(acc, w, fallback=1.0):
    """acc / w when w > 0, else `fallback`. The fallback is 1.0, not 0.0:
    an unfiltered pixel reads fully unoccluded rather than fully dark."""
    return np.where(w > 0.0, acc / np.where(w > 0.0, w, 1.0), fallback)
