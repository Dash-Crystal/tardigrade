"""Ambient occlusion: `ssao`, `ssao_scalable_ambient_obscurance`,
`ssao_bilateral_blur`, `aoproxy_splat`.

Four families, three DIFFERENT occlusion estimators, and they are not
interchangeable:

* `ssao` r0/m0 -- a 9-tap kernel reflected into the hemisphere, occlusion
  decided by two hard step functions (slope 1e6), output cubed.
* `ssao_scalable_ambient_obscurance` r0/m3 -- an 11-tap spiral with a
  per-tap mip selected from the tap radius, a cosine-weighted falloff, and
  a 2x2 checkerboard de-noise built out of dFdx/dFdy.
* `aoproxy_splat` r0/m0 -- not a screen-space estimator at all: a
  per-proxy-volume PRODUCT of four 3D-texture lookups, one per SH-ish
  basis row, accumulated multiplicatively over two loops.
* `ssao_bilateral_blur` r0/m0 -- the separable cross-bilateral filter that
  cleans up whichever of the above ran.
"""
from .._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.


# THE BACKEND PROBLEM IS SOLVED CENTRALLY, NOT HERE. I first added local
# _i64/_clip/_maximum adapters to this module after it raised
# AttributeError on `.astype` the moment the renderer called it with
# tensors. `passes/_arraylib.py` landed the same fix for the whole repo --
# 132 of 240 impls raised on a CUDA tensor before it -- so the local
# adapters were a second mechanism for one job, which is the duplication
# this campaign exists to remove. Deleted; the expressions below are
# numpy-shaped and the facade dispatches them.

# ---------------------------------------------------------------------------
# ssao  (docs/.../glsl_loopfam/ssao_r0m0.glsl)
# ---------------------------------------------------------------------------
def normal_decode_unpack(texel_rgb):
    """normalize(texel * 2 - 1) -- the plain unpack, not glass's diagonal."""
    n = texel_rgb * 2.0 - 1.0
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def face_normal_min_delta(d_right, d_left, d_up, d_down):
    """Face normal from the SMALLER of each opposed screen-space delta.

    Picking the shorter delta on each axis keeps the estimate on the near
    side of a depth discontinuity instead of straddling it. The reference
    then NEGATES the cross product; the sign is part of the term, because
    an AO estimator with an inverted normal still produces a plausible
    image and would never be caught by an eyeball comparison.
    """
    dx = np.where((_len(d_left) < _len(d_right))[..., None], d_left, d_right)
    dy = np.where((_len(d_up) < _len(d_down))[..., None], d_up, d_down)
    c = np.cross(dx, dy)
    c = -c
    return c / np.linalg.norm(c, axis=-1, keepdims=True)


def _len(v):
    return np.linalg.norm(v, axis=-1)


def sample_origin(pos, face_n, depth, bias):
    """Push the sampling origin off the surface by depth * bias."""
    return pos + face_n * (depth * bias)[..., None]


def hemisphere_reflect(kernel_vec, radius, map_n, face_n):
    """Kernel vector into the hemisphere: reflect about the MAP normal
    always, then about the FACE normal only when it still points inward.

    Two different normals, in that order. Collapsing them to one -- the
    obvious simplification -- changes the sample distribution, and the
    reference keeps them distinct."""
    v = _reflect(kernel_vec * radius[..., None], map_n)
    inward = (np.einsum("...i,...i->...", face_n, v) < 0.0)[..., None]
    return np.where(inward, _reflect(v, face_n), v)


def _reflect(i, n):
    return i - 2.0 * np.einsum("...i,...i->...", n, i)[..., None] * n


def project_clamped_uv(clip, rect):
    """Clip -> uv with the family's own half-scale/bias, clamped to `rect`.

    The scale and bias are built from rect.zw ALONE:
        scale = (0.5, -0.5) * rect.zw
        bias  = 0.5 * rect.zw + rect.xy
    and then the result is clamped to [rect.xy, rect.zw]. The same member
    supplies both the mapping and the bound, which only agrees with a
    normalized uv space when rect.xy is the origin. Transcribed as written,
    not repaired."""
    scale = np.stack([0.5 * rect[..., 2], -0.5 * rect[..., 3]], -1)
    bias = 0.5 * rect[..., 2:] + rect[..., :2]
    uv = clip[..., :2] / clip[..., 3:4] * scale + bias
    return np.clip(uv, rect[..., :2], rect[..., 2:])


def occlusion_step(sample_pos, eye, sampled_depth, origin, radius2):
    """One tap's occlusion: two saturated 1e6-slope steps, multiplied.

    1e6 makes each `clamp(x * 1e6, 0, 1)` a step function that is smooth
    only across a 1e-6-wide band -- so this is a hard binary test written
    without a branch, and NOT a soft range falloff. Treating it as a
    falloff (the natural-looking reading) changes the estimator."""
    dv = sample_pos - eye
    dist = _len(dv)
    hit = eye + (dv / dist[..., None]) * sampled_depth[..., None]
    d = hit - origin
    in_range = np.clip((radius2 - np.einsum("...i,...i->...", d, d)) * 1e6, 0.0, 1.0)
    occluded = np.clip((dist - sampled_depth) * 1e6, 0.0, 1.0)
    return in_range * occluded


def ao_resolve_cubed(acc, n_taps=9):
    """v = 1 - acc/9, output v^3. The cube is in the shader, not a curve
    we chose; it is written `_19065 * (_19065 * _19065)`."""
    v = 1.0 - acc * (1.0 / n_taps)
    return v * (v * v)


# ---------------------------------------------------------------------------
# ssao_scalable_ambient_obscurance
# ---------------------------------------------------------------------------
def sao_spiral_tap(i, jitter, n_taps=11):
    """Tap i -> (angle, radius_fraction) of the spiral.

    angle = ((i + 0.5) * 3.996364116668701171875 + jitter) mod 2pi
    r     = (i + 0.5) / 11

    3.9963641... is not 2pi/phi or any constant we get to choose; it is the
    literal in the module and is stored to full float32 precision."""
    h = i + 0.5
    a = h * 3.996364116668701171875 + jitter
    a = a - 6.283185482025146484375 * np.trunc(a / 6.283185482025146484375)
    return np.stack([a, h * (1.0 / n_taps)], -1)


def sao_hash_jitter(px, py, phase):
    """((3*x) ^ (y + x*y)) * 8 + phase -- an integer XOR hash, per pixel."""
    ix, iy = np.int64(px), np.int64(py)
    h = (3 * ix) ^ (iy + ix * iy)
    return np.float64(h) * 8.0 + phase


def sao_mip_from_radius(r_px):
    """clamp(floor(log2(r)) - 3, 0, 5). Selecting the mip from the tap
    radius is what keeps the spiral's outer taps from thrashing the cache;
    it is also why the estimator is not scale-free."""
    lg = np.floor(np.log2(np.maximum(r_px, 1e-300)))
    return np.int64(np.clip(lg - 3.0, 0.0, 5.0))


def sao_tap_weight(dv, n, inv_r2, bias):
    """4 * max(1 - |dv|^2 * inv_r2, 0) * max(dot(dv, n) - bias, 0)

    Note the SECOND max is not clamped above: a tap far in front of the
    plane contributes proportionally to its distance, unbounded. That is
    the estimator's shape and the normalizer downstream is what bounds it."""
    d2 = np.einsum("...i,...i->...", dv, dv)
    falloff = 4.0 * np.maximum(1.0 - d2 * inv_r2, 0.0)
    cosine = np.maximum(np.einsum("...i,...i->...", dv, n) - bias, 0.0)
    return falloff * cosine


def sao_resolve(acc, intensity, norm, n_taps=11):
    """pow(clamp(1 - intensity*acc / (11 * norm), 0, 1), 1.4)."""
    return np.power(np.clip(1.0 - (intensity * acc) / (n_taps * norm), 0.0, 1.0),
                    1.39999997615814208984375)


def sao_checker_denoise(v, dv_dx, parity):
    """v - dFdx(v) * (parity - 0.5): the 2x2 quad de-checker.

    Applied on x then on y, each guarded by |dFdx(depth)| < 1 so it does
    not smear across a silhouette. The guard is on the DEPTH derivative
    while the correction uses the AO derivative -- two different
    quantities, and pairing them is the point."""
    # `parity` is a 0/1 selector and torch refuses `-` on a bool tensor.
    # Promoting with *1.0 keeps the arithmetic identical for a float input
    # and makes a bool one legal; inverting with ~ would be a different
    # expression.
    return v - dv_dx * (parity * 1.0 - 0.5)


# ---------------------------------------------------------------------------
# ssao_bilateral_blur
# ---------------------------------------------------------------------------
def bilateral_weight(tap_kernel, depth_center, depth_tap):
    """(0.3 + k[|i|]) * max(0, 1 - 2000 * |dz|)

    The range term dies over a depth difference of 1/2000, so on any
    ordinary scene scale this filter is essentially edge-stopping rather
    than edge-aware; the 0.3 floor is what keeps the spatial kernel alive
    when the range term saturates."""
    return (0.3 + tap_kernel) * np.maximum(0.0, 1.0 - 2000.0 * np.abs(depth_tap - depth_center))


def bilateral_resolve(num, den):
    """num / (den + 9.9999997473787516355514526367188e-05)

    The epsilon is IN THE SHADER. We added an epsilon of our own to a
    different family once and it cost a retraction; this one is read."""
    return num / (den + 9.9999997473787516355514526367188e-05)


# ---------------------------------------------------------------------------
# aoproxy_splat
# ---------------------------------------------------------------------------
def proxy_box_fade(local_pos):
    """clamp(3 * min(1-|p|), 0, 1) * clamp(2.8 - 2*|p|, 0, 1)

    A box fade and a sphere fade MULTIPLIED, not chosen between. The
    shader discards below 0.01 of the product."""
    m = 1.0 - np.abs(local_pos)
    box = np.clip(3.0 * np.min(m, axis=-1), 0.0, 1.0)
    sphere = np.clip(2.7999999523162841796875
                     - 2.0 * np.linalg.norm(local_pos, axis=-1), 0.0, 1.0)
    return box * sphere


def proxy_sphere_lookup_uvw(basis_row_dir, dir_to_proxy, inv_dist, slice_w):
    """(dot(basis, d) * 0.5 + 0.5, inv_dist, slice) -- the 3D LUT coord.

    The second coordinate is the INVERSE distance (an `inversesqrt`
    result), not the distance. A LUT indexed by 1/r has its resolution
    where the proxy is close, which is the opposite of what indexing by r
    would give."""
    u = np.einsum("...i,...i->...", basis_row_dir, dir_to_proxy) * 0.5 + 0.5
    return np.stack([u, inv_dist, slice_w], -1)


def proxy_box_inv_dist(local_pos, half_extent):
    """1 / (1 + dot(|q| / L1(|q|), pi / (e.yzx * e.zxy)) * |q|)

    where q = clamp(p, -e, e) * 0.9 - p. This is the box proxy's stand-in
    for a distance: an L1-normalized direction weighted by the reciprocal
    of the two PERPENDICULAR extents per axis (yzx * zxy is the face area),
    so a long thin box and a cube at the same distance do not read the
    same LUT row."""
    q = np.clip(local_pos, -half_extent, half_extent) * 0.89999997615814208984375 - local_pos
    a = np.abs(q)
    l1 = a[..., 0] + a[..., 1] + a[..., 2]
    face = half_extent[..., [1, 2, 0]] * half_extent[..., [2, 0, 1]]
    w = np.einsum("...i,...i->...", a / l1[..., None],
                  3.1415927410125732421875 / face)
    return 1.0 / (1.0 + w * np.linalg.norm(q, axis=-1))


def proxy_composite(acc, fade, vertex_alpha, instance_scale):
    """mix(1, acc, fade * vcol.a * scale) -- fades toward UNOCCLUDED (1),
    so a proxy that misses contributes nothing rather than darkening."""
    a = (fade * vertex_alpha * instance_scale)[..., None]
    return 1.0 + (acc - 1.0) * a
