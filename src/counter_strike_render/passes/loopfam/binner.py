"""`generic` and `lpv_debug_grid` -- the two WAVE-COHERENT binners.

THESE TWO ARE NOT THE BINNER MY OWN DOCUMENTS DESCRIBED. Both prior
loop-family documents characterise the idiom as "an outer word loop over a
cull bitmask, one inner loop peeling set bits". That is the shape here too,
with one addition that changes what the loop costs:

    uint mask = subgroupOr(cullA[base + w] & cullB[base2 + w]);

The per-lane intersection is OR-ed ACROSS THE SUBGROUP before the peel, so
every lane iterates the UNION of what any lane in the wave needs. The trip
count is a property of the WAVE, not of the pixel. Verified in the
bytecode: OpGroupNonUniformBitwiseOr x3 in generic_r69m3 and x4 in
lpv_debug_grid_r2m2, both declaring OpCapability GroupNonUniformArithmetic.
A sweep of all 27 extracted modules found subgroup ops in these two ONLY.

The consequence for this project is specific and not cosmetic: a per-pixel
FLOP figure for either family is a LOWER BOUND that can be exceeded by up
to the wave's divergence factor, and no amount of deeper record sampling
narrows it -- the quantity is not a property of the module at all.
"""
from .._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.


# ---------------------------------------------------------------------------
# The wave-coherent binner
# ---------------------------------------------------------------------------
def wave_coherent_mask(cull_a, cull_b, subgroup_ids):
    """subgroupOr(cull_a & cull_b), evaluated per subgroup.

    Every lane in a wave receives the same result, so modelling this as a
    plain `a & b` -- which is what a per-pixel reading gives -- produces a
    SUBSET of the real mask on every divergent wave and silently skips a
    light another lane needs.
    """
    a = np.astype(cull_a, np.int64)
    b = np.astype(cull_b, np.int64)
    g = np.astype(subgroup_ids, np.int64)
    inter = a & b
    n = inter.shape[0]
    # FAST PATH: lanes numbered in contiguous equal-size blocks, which is
    # what a wave is. A reshape-and-reduce is exact and stays on the
    # device; the general path below calls np.unique per group and drags
    # every id back to host memory, which on CUDA is not a slowdown but a
    # TypeError.
    if n:
        first = g[0]
        width = int((g == first).sum())
        if width and n % width == 0:
            blocks = g.reshape(n // width, width)
            if bool((blocks == blocks[:, :1]).all()):
                red = np.bitwise_or.reduce(inter.reshape(n // width, width),
                                           axis=1)
                return np.astype(np.repeat(red, width), np.float64)
    out = np.zeros_like(inter)
    for w in np.unique(g):
        m = g == w
        out[m] = np.bitwise_or.reduce(inter[m])
    return np.astype(out, np.float64)


def peel_lowest_bit(mask):
    """(findLSB(mask), mask & (mask - 1)) -- the peel, unchanged from the
    per-pixel idiom. Only the mask's PROVENANCE changed."""
    m = np.astype(np.asarray(mask), np.int64)
    lsb = np.zeros_like(m)
    nz = m != 0
    lsb[nz] = np.astype(np.log2(m[nz] & (-m[nz])), np.int64)
    return lsb, m & (m - 1)


# ---------------------------------------------------------------------------
# generic -- the bicubic B-spline lightmap filter
# ---------------------------------------------------------------------------
def bspline_weights(t):
    """The cubic B-spline basis, in the exact grouping the module writes.

        t2 = t*t ; t3 = t2*t
        w1 = (t3*3 - t2*6 + 4) / 6
        w3 =  t3 / 6
        w0 = (-t3 + (t2 - t)*3 + 1) / 6
        w2 = ((t2 - t3 + t)*3 + 1) / 6

    and the module then forms w01 = w0 + w1 and w23 = w2 + w3 in place --
    it never materialises w0 and w2 separately, so the four weights below
    are returned as the two PAIR SUMS the taps actually use, plus the two
    numerators the offsets need."""
    t2 = t * t
    t3 = t2 * t
    w1 = (t3 * 3.0 - t2 * 6.0 + 4.0) * 0.16666667163372039794921875
    w3 = t3 * 0.16666667163372039794921875
    w01 = ((t3 * -1.0 + (t2 - t) * 3.0) + 1.0) * 0.16666667163372039794921875 + w1
    w23 = (((t2 - t3 + t) * 3.0) + 1.0) * 0.16666667163372039794921875 + w3
    return np.stack([w1, w3, w01, w23], -1)


def bspline_offsets(w1, w3, w01, w23):
    """(w1/w01 - 1, w3/w23 + 1) -- the two bilinear tap offsets.

    This is what turns 16 texture reads into 4: each offset places a
    BILINEAR tap so the hardware's own interpolation performs two of the
    four cubic taps. Reading it as a 4x4 gather is arithmetically
    equivalent and four times the bandwidth."""
    return np.stack([w1 / w01 - 1.0, w3 / w23 + 1.0], -1)


# ---------------------------------------------------------------------------
# generic -- directional lightmap
# ---------------------------------------------------------------------------
_DIR_CLAMP = 0.996190845966339111328125


def directional_lightmap_decode(texel_xy):
    """xy * (0.996190845966339111328125 / max(0.99619..., |xy|)), then
    z = sqrt(1 - |xy|^2).

    The constant is a LENGTH CEILING, not a scale: vectors shorter than it
    pass through untouched and only longer ones are shrunk. cos(5 degrees)
    is 0.9961947, so this caps the direction at 5 degrees off the surface
    -- it keeps z from reaching zero, which is exactly the divide two
    expressions later."""
    n = np.linalg.norm(texel_xy, axis=-1, keepdims=True)
    xy = texel_xy * (_DIR_CLAMP / np.maximum(_DIR_CLAMP, n))
    z = np.sqrt(np.maximum(0.0, 1.0 - np.einsum("...i,...i->...", xy, xy)))
    return np.concatenate([xy, z[..., None]], -1)


def directional_sharpen(z, deriv_ratio):
    """z * mix(1, mix(0.1, 2, sat((1-z)*1.5)),
               smoothstep(0.1, 0.01, ratio) * 0.8)

    A TEXEL-DENSITY-DRIVEN sharpen: `ratio` is
    length(fwidth(N)) / length(fwidth(P)), so it is large where geometry
    curves fast relative to world size. The inner mix spans 0.1 to 2.0 --
    it can BLUNT the direction as well as sharpen it, depending on z. The
    outer smoothstep is DESCENDING and its output is scaled by 0.8, so the
    effect never fully engages."""
    inner = 0.100000001490116119384765625 + (2.0 - 0.100000001490116119384765625) \
        * np.clip((1.0 - z) * 1.5, 0.0, 1.0)
    amt = _ss(0.100000001490116119384765625, 0.00999999977648258209228515625,
              deriv_ratio) * 0.800000011920928955078125
    return z * (1.0 + (inner - 1.0) * amt)


def directional_irradiance(color, ambient_scale, ambient_bias, z, ndotl):
    """amb = color * sat(ambient_scale + bias)
       out = amb + ((color - amb) / z) * max(0, dot(dir, n))

    The DIVIDE BY z is what makes this a directional lightmap rather than
    a lerp: the stored colour is the integral over the hemisphere, so
    recovering the directional part means dividing out the direction's own
    cosine. z is bounded away from zero by the 0.99619 length ceiling
    upstream, which is the only reason this divide is safe."""
    amb = color * np.clip(ambient_scale + ambient_bias, 0.0, 1.0)[..., None]
    return amb + ((color - amb) / z[..., None]) * np.maximum(0.0, ndotl)[..., None]


# ---------------------------------------------------------------------------
# generic -- surface
# ---------------------------------------------------------------------------
def normal_decode_wy(texel, scale):
    """(texel.wy * 2 - 1), z = sqrt(sat(1 - |xy|^2)), THEN y negated, THEN
    xy scaled.

    A FOURTH normal encoding, and the channel choice is the tell: .w and
    .y, i.e. a BC5-style two-channel map stored in alpha and green. The y
    flip happens BEFORE the xy scale and AFTER the z reconstruct, so
    reordering any two of the three changes the result."""
    xy = texel[..., [3, 1]] * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - np.einsum("...i,...i->...", xy, xy), 0.0, 1.0))
    xy = np.stack([xy[..., 0], -xy[..., 1]], -1) * scale[..., None]
    return np.concatenate([xy, z[..., None]], -1)


def specular_aa_roughness(base_rough, ddx_n, ddy_n):
    """max(base, pow(sat(max(dot(ddx,ddx), dot(ddy,ddy))),
                     0.333000004291534423828125))

    Normal-variance roughening: where the shading normal changes fast
    across a pixel, the surface is treated as rougher so the highlight
    does not alias. The exponent is 0.333000004291534423828125 -- close to
    a cube root but NOT 1/3, and it is a literal in the module. The max
    with `base` means this can only ever roughen."""
    v = np.maximum(np.einsum("...i,...i->...", ddx_n, ddx_n),
                   np.einsum("...i,...i->...", ddy_n, ddy_n))
    return np.maximum(base_rough,
                      np.power(np.clip(v, 0.0, 1.0), 0.333000004291534423828125))


def bitangent_with_flips(normal, tangent4, instance_flip, back_facing):
    """cross(n * (backFacing ? -1 : 1), t.xyz) * (t.w > 0 ? 1 : -1),
    then negated again when the instance flag is set.

    THREE INDEPENDENT SIGN SOURCES on one vector: the facing flip on the
    normal, the tangent's own w handedness, and a per-instance flag. Two
    of them can cancel, so a renderer that implements any two and drops
    the third is correct on exactly half its geometry -- which looks like
    a content bug rather than a shader one."""
    n = normal * np.where(back_facing, -1.0, 1.0)[..., None]
    b = np.cross(n, tangent4[..., :3]) \
        * np.where(tangent4[..., 3] > 0.0, 1.0, -1.0)[..., None]
    return np.where(instance_flip[..., None], -b, b)


def quaternion_axis_y(q, scale):
    """The middle column of a quaternion's rotation matrix, times a scale.

        (2xy - 2wz,  1 - 2x^2 - 2z^2,  2yz + 2wx) * s

    Written out from _m5 = (x, y, z, w) with every product formed
    separately. It is the light's own +Y axis in world space, and it is
    the half-extent vector of the capsule two expressions later -- so the
    quaternion is not orientation-only, its LENGTH is the light's size."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2.0 * x * y - 2.0 * w * z,
                     (1.0 - 2.0 * x * x) - 2.0 * z * z,
                     2.0 * y * z + 2.0 * w * x], -1) * scale[..., None]


def capsule_closest_point(to_light, half_axis):
    """Closest point on the segment [c - a, c + a] to the origin.

        p0 = to_light - a ; d = (to_light + a) - p0
        t  = dot(-p0, d)
        return t <= 0 ? p0 : p0 + d * min(1, t / dot(d, d))

    The `t <= 0` early-out returns the SEGMENT ENDPOINT, not a clamped
    projection, and the far end is clamped by min(1, .) rather than by a
    second branch -- so the two ends are not handled symmetrically even
    though the geometry is."""
    p0 = to_light - half_axis
    d = (to_light + half_axis) - p0
    t = np.einsum("...i,...i->...", -p0, d)
    dd = np.einsum("...i,...i->...", d, d)
    far = p0 + d * np.minimum(1.0, t / np.where(dd == 0.0, 1.0, dd))[..., None]
    return np.where((t <= 0.0)[..., None], p0, far)


def spot_endcap_fade(ndc_z, near_slope, far_slope):
    """smoothstep(0, 1, z * near) * smoothstep(0, 1, (1 - z) * far),
    each factor applied only when its slope is > 0.

    Two independent end caps on the projected z, each disabled by a
    NON-POSITIVE slope rather than by a flag. A slope of exactly zero
    leaves that end unfaded, which a `>= 0` test would invert."""
    a = np.where(near_slope > 0.0, _ss(0.0, 1.0, ndc_z * near_slope), 1.0)
    b = np.where(far_slope > 0.0, a * _ss(0.0, 1.0, (1.0 - ndc_z) * far_slope), a)
    return b


# ---------------------------------------------------------------------------
# lpv_debug_grid / generic -- probe volume selection
# ---------------------------------------------------------------------------
def volume_inset_weight(local_pos, lo, hi, inv_fade):
    """min over 6 faces of clamp((p - lo) * k, 0, 1) and clamp((hi - p) * k, 0, 1)

    A MIN, not a product: the weight is set by the NEAREST face alone, so
    a probe in a corner is not doubly attenuated. Zero means fully outside
    and the module `continue`s on exactly `== 0.0`."""
    a = np.clip((local_pos - lo) * inv_fade, 0.0, 1.0)
    b = np.clip((hi - local_pos) * inv_fade, 0.0, 1.0)
    return np.minimum(np.min(a, axis=-1), np.min(b, axis=-1))


def volume_smooth_and_accumulate(w, covered):
    """s = w*w*(-2w + 3) ; contribution = s * (1 - covered)

    A smoothstep polynomial applied to the inset weight, then FRONT-TO-BACK
    compositing against the coverage accumulated so far. The loop breaks
    once coverage exceeds 0.9900000095367431640625, so probe order is
    load-bearing -- a renderer that sums all probes and normalises gets a
    different answer wherever two volumes overlap."""
    s = (w * w) * (-2.0 * w + 3.0)
    return s * (1.0 - covered)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# csgo_tools_shading_complexity -- the arithmetic of the atomic path
# ---------------------------------------------------------------------------
# WHAT IS AND IS NOT IMPLEMENTED HERE. Each atomic is a PURE FUNCTION of
# (the storage texel's current value, this invocation's primitive id): given
# what it read, what does it write and what does it return. That much is
# ordinary arithmetic and is implemented and A/B'd below.
#
# What is NOT here, and is recorded as its own ABSENT inventory row rather
# than faked: WHICH value an invocation reads depends on the order other
# invocations touched the same texel. That is a property of the execution
# model, not of an expression, no single-term A/B can express it, and this
# renderer has no storage-image model to run it against. Same category as
# dispatch wiring -- named, not invented.
TOOLS_SENTINEL = 4294967295


def tools_tile_coord(frag_xy):
    """uvec2(gl_FragCoord.xy * 0.5) -- a HALF-RESOLUTION tile index.

    The truncation is the uint cast, not a floor, so this is only the same
    thing for non-negative coordinates. gl_FragCoord is non-negative, which
    is why the shader can get away with it and why a signed reimplementation
    would diverge on nothing until it was used somewhere else."""
    return np.astype(np.floor(np.asarray(frag_xy) * 0.5), np.int64)


def tools_compswap(current, primitive_id, sentinel=TOOLS_SENTINEL):
    """imageAtomicCompSwap(img, p, 4294967295u, primitiveID)

    Returns the value that was THERE. The write happens only when the texel
    still holds the sentinel, so the FIRST primitive to reach a tile claims
    it and every later one reads that claim back. Returned as (old, new)."""
    cur = np.astype(np.asarray(current), np.int64)
    pid = np.astype(np.asarray(primitive_id), np.int64)
    new = np.where(cur == sentinel, pid, cur)
    return cur, new


def tools_exchange(current, sentinel=TOOLS_SENTINEL):
    """imageAtomicExchange(img, p, 4294967295u) -- unconditionally RESTORES
    the sentinel and returns the old value.

    It fires on the SECOND claim only (`_5526 == 2`), which is what releases
    the tile so a third primitive can claim it in turn. Reading this as a
    plain store loses the return value the loop's exit test consumes."""
    cur = np.astype(np.asarray(current), np.int64)
    return cur, np.full_like(cur, sentinel)


def tools_overdraw_add(counter, fired):
    """imageAtomicAdd(img2, p, 1u), guarded by `_17133 != 0`.

    A SECOND storage image, counting tiles that claimed at least once --
    not fragments. A per-fragment increment would produce overdraw; this
    produces distinct-primitive coverage, which is what the family is
    named for."""
    c = np.astype(np.asarray(counter), np.int64)
    return c + np.astype((np.asarray(fired) != 0), np.int64)


def tools_claim_state(old, primitive_id, claimed):
    """claimed = (old == primitiveID) ? true : claimedSoFar

    A LATCH: once this invocation sees its own id in the texel it stops
    issuing atomics for the remaining iterations (the compare-swap at the
    top is guarded by `!claimed`). The 16-iteration bound is therefore a
    RETRY BUDGET, not a tap count -- an invocation that claims on the first
    try still runs 16 iterations but issues one atomic."""
    return np.where(np.astype(np.asarray(old), np.int64)
                    == np.astype(np.asarray(primitive_id), np.int64),
                    1.0, np.astype((np.asarray(claimed) != 0), np.float64))
