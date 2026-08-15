"""Screen-space effect passes: `inferno`, `bomb_blast`, `csgo_grenade_camera`,
`outlines`, `panorama_alpha`, `convolve_environment_map`,
`tools_terrain_composite_slope`, `csgo_volume_viewer`, `visualize_depth`,
`msaa_resolve`, `dof2`, `sfm_volumetrics_frustum`,
`deferred_particle_shadows`, `player_visibility`, `tools_grid`.

`inferno` is the interesting one: a TRUE clustered binner, two nested
findLSB loops over a per-tile bitmask, with a 3D noise fetch inside the
outer loop. It is the molotov fire, and the structure is the same
light-binner shape the world families use -- applied to fire volumes
instead of lights.
"""
from .._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.


# ---------------------------------------------------------------------------
# The near/far circle-of-confusion field. THREE families, one expression.
# ---------------------------------------------------------------------------
def coc_two_sided(field, near, far):
    """max(clamp(f*near.x + near.y, 0, 1), clamp(f*far.x + far.y, 0, 1))

    Two independent linear ramps with a MAX, not one ramp: a near field
    and a far field, each saturating, and a pixel is blurred if EITHER
    claims it. Written identically in dof2_r0m1.glsl:29,
    msaa_resolve_r0m39.glsl:47 and (as the alpha output)
    blur_with_depth's neighbourhood -- so it lives here once.

    `near` and `far` are (scale, bias) pairs; in the modules they are the
    xy and zw halves of a single vec4."""
    a = np.clip(field * near[..., 0] + near[..., 1], 0.0, 1.0)
    b = np.clip(field * far[..., 0] + far[..., 1], 0.0, 1.0)
    return np.maximum(a, b)


# ---------------------------------------------------------------------------
# inferno
# ---------------------------------------------------------------------------
def inferno_tile_index(uv):
    """dot(floor(uv * 4), (1, 4)) -- a 4x4 tile id from the interpolated uv,
    then split into a word index and a bit lane by /4 and %4. The cluster
    grid here is in UV, not in screen pixels."""
    return np.einsum("...i,i->...", np.floor(uv * 4.0),
                     np.like([1.0, 4.0], uv))


def inferno_blob_field(world_pos, blob_center, blob_radius):
    """length(p - c with z DOUBLED) - r

    The z component of the difference is multiplied by 2 BEFORE the length,
    which makes the blob an ellipsoid squashed vertically -- fire that
    spreads wider than it rises. Scaling after the length, or not at all,
    gives a sphere and a different silhouette."""
    d = world_pos - blob_center
    d = np.stack([d[..., 0], d[..., 1], d[..., 2] * 2.0], -1)
    return np.linalg.norm(d, axis=-1) - blob_radius


def inferno_blob_weight(sdf, radius, age_delta):
    """smoothstep(100, 50, sdf) * smoothstep(0, 1, r) * smoothstep(6.5, 5.5, age)

    Three smoothsteps multiplied, two of them DESCENDING (edge0 > edge1),
    which inverts the ramp. The blob is culled outright at sdf >= 100 by a
    `continue` one line earlier, so the first factor never actually
    evaluates below its own edge1."""
    return (_ss(100.0, 50.0, sdf) * _ss(0.0, 1.0, radius)
            * _ss(6.5, 5.5, age_delta))


def inferno_noise_lookup_pos(world_pos, lift):
    """The march position is LIFTED before the 3D fetch:

        p.z += 100.0 * smoothstep(7.0, 0.0, lifetime)
        uvw = p * 0.007000000216066837310791015625

    A young fire samples the noise volume 100 units higher than an old
    one, which is what makes the texture appear to fall as the fire dies.
    The scale is a literal, not a uniform."""
    p = np.stack([world_pos[..., 0], world_pos[..., 1],
                  world_pos[..., 2] + 100.0 * _ss(7.0, 0.0, lift)], -1)
    return p * 0.007000000216066837310791015625


def inferno_shade(noise_xy, w):
    """clamp((n.y + w) * w, 0, 1) * (0.85 + n.x * 0.1)

    Two DIFFERENT noise channels doing two different jobs: .y perturbs the
    coverage before the square, .x tints the result over a narrow 0.85-0.95
    band. Swapping them changes both the silhouette and the brightness."""
    cov = np.clip((noise_xy[..., 1] + w) * w, 0.0, 1.0)
    return cov * (0.85000002384185791015625 + noise_xy[..., 0] * 0.100000001490116119384765625)


# ---------------------------------------------------------------------------
# bomb_blast
# ---------------------------------------------------------------------------
def blast_shock_profile(dist, radius):
    """smoothstep(r + 40, r, d) * pow(smoothstep(r - 3000, r, d), 2.0)

    An outer edge 40 units wide and an inner ramp 3000 units wide, squared.
    The two smoothsteps run in OPPOSITE directions about the same radius,
    so the product is a ring, not a disc; the 3000-unit inner ramp is what
    gives it a long trailing tail toward the centre."""
    return _ss(radius + 40.0, radius, dist) * np.power(
        _ss(radius - 3000.0, radius, dist), 2.0)


def blast_refract_offset(uv, centre_uv, strength):
    """dir = normalize(c - uv); mag = w * 0.03 * (saturate(|c-uv|*2)*0.9999 + 1e-4)

    The magnitude never reaches zero: the 9.9999997473787516355514526367188e-05
    floor keeps a minimum displacement even at the exact centre, where the
    direction is undefined."""
    d = centre_uv - uv
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    direction = d / n
    mag = (strength * 0.02999999932944774627685546875) * (
        np.clip(np.clip(n[..., 0], 0.0, 1.0) * 2.0, 0.0, 1.0)
        * 0.99989998340606689453125 + 9.9999997473787516355514526367188e-05)
    return direction * mag[..., None]


def blast_bloom(rgb):
    """c + pow(c, 8) -- an eighth power added to the base, so anything under
    1.0 contributes essentially nothing and anything over it explodes. Not
    a threshold-and-scale bloom; a pure polynomial."""
    return rgb + np.power(rgb, 8.0)


# ---------------------------------------------------------------------------
# csgo_grenade_camera
# ---------------------------------------------------------------------------
def grenade_vignette(uv, aspect):
    """Four smoothsteps multiplied -- two per axis, and the X pair uses an
    ASPECT-SCALED width (0.1 * aspect) while the Y pair uses a bare 0.1 and
    0.9. The frame border is therefore not symmetric in pixels unless the
    target is square, which is the intent for a 4:3 grenade view."""
    w = 0.100000001490116119384765625 * aspect
    return (_ss(0.0, w, uv[..., 0]) * _ss(1.0, 1.0 - w, uv[..., 0])
            * _ss(0.0, 0.100000001490116119384765625, uv[..., 1])
            * _ss(1.0, 0.89999997615814208984375, uv[..., 1]))


def grenade_scanline_alpha(uv, vignette, hard, phase):
    """mix(v, 1, hard) * smoothstep(s - 0.1, s, phase * 8)
    with s = uv.x - uv.y * 0.1

    A DIAGONAL wipe: the threshold is a linear function of both uv axes, so
    the sweep line is tilted by 0.1 rather than vertical."""
    s = uv[..., 0] - uv[..., 1] * 0.100000001490116119384765625
    return (vignette + (1.0 - vignette) * hard) * _ss(
        s - 0.100000001490116119384765625, s, phase * 8.0)


def grenade_gamma_out(rgb):
    """pow(rgb, 2.2000000476837158203125) -- applied to the OUTPUT, i.e.
    this pass writes a de-gamma'd value, not an encoded one."""
    return np.power(rgb, 2.2000000476837158203125)


# ---------------------------------------------------------------------------
# outlines
# ---------------------------------------------------------------------------
def outline_edge_metric(inside_count, n_taps=5):
    """|count/5 - 0.5| * 2, then DISCARD where 0.999 - m < 0.

    A pixel survives only when the 5 taps are split -- all-inside and
    all-outside both give m = 1 and are killed. So this is an edge
    detector built out of a majority count, not a gradient."""
    return np.abs(inside_count * (1.0 / n_taps) - 0.5) * 2.0


def outline_shade(rgb_mean, metric, radial):
    """rgb * ((1 - m) * 4) * smoothstep(-0.2, 0.4, 0.6 - r)

    The (1-m)*4 factor is a gain of up to 4x at a perfectly split pixel,
    and the radial term fades the whole outline toward the frame edge. The
    smoothstep's edge0 is NEGATIVE, so it is already partly on at r = 0.6."""
    return rgb_mean * ((1.0 - metric) * 4.0)[..., None] * _ss(
        -0.20000000298023223876953125, 0.4000000059604644775390625,
        0.60000002384185791015625 - radial)[..., None]


def outline_tap_offset_scale(radial, width):
    """mix(width, 0, r) = width * (1 - r)

    The sampling radius SHRINKS to zero at the frame edge -- taps converge
    on the centre pixel there, so the outline thins out radially rather
    than being clipped."""
    return width * (1.0 - radial)


# ---------------------------------------------------------------------------
# panorama_alpha
# ---------------------------------------------------------------------------
def panorama_is_background(rgb, depth, ref):
    """(r < 0.005 && g < 0.005 && b < 0.005) && depth >= ref - 0.01

    ALL THREE colour channels must be near-black AND the depth must be at
    the far reference. Either alone is not enough, and the short-circuit
    chain in the module makes the order explicit."""
    dark = np.all(rgb < 0.004999999888241291046142578125, axis=-1)
    far = depth >= (ref - 0.00999999977648258209228515625)
    return dark & far


def panorama_coverage(count):
    """smoothstep(4, 8, n) over 8 ring taps -- fewer than 4 non-background
    neighbours gives zero, 8 gives one. The premultiplied output is
    `rgb * c` with alpha `c`."""
    return _ss(4.0, 8.0, count)


# ---------------------------------------------------------------------------
# convolve_environment_map
# ---------------------------------------------------------------------------
def convolve_tangent_basis(n):
    """Tangent frame with the up-vector chosen by |n.z| < 0.999:

        up = (|n.z| < 0.999) ? (0,0,1) : (1,0,0)
        t  = normalize(cross(up, n))
        b  = cross(n, t)
        M  = mat3(t, b, n)

    `mix(a, b, bvec)` picks b where TRUE, so the (0,0,1) branch is the
    COMMON case and (1,0,0) is the pole fallback -- the reverse of how the
    expression reads left to right."""
    up = np.where((np.abs(n[..., 2]) < 0.999000012874603271484375)[..., None],
                  np.like([0.0, 0.0, 1.0], n), np.like([1.0, 0.0, 0.0], n))
    t = np.cross(up, n)
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    b = np.cross(n, t)
    return np.stack([t, b, n], axis=-1)


def convolve_accumulate(sample_rgb, weight):
    """max(sample * w, 0) accumulated -- the MAX is per-sample and inside
    the loop, so a negative-lobe kernel cannot subtract; it can only fail
    to add."""
    return np.maximum(sample_rgb * weight[..., None], 0.0)


# ---------------------------------------------------------------------------
# tools_terrain_composite_slope
# ---------------------------------------------------------------------------
def slope_metric(d):
    """d.z^2 / dot(d, d) -- the squared cosine between the height difference
    and the full 3-vector difference, i.e. how much of the change is
    vertical. Taken as a MAX over 8 ring taps, so it reports the steepest
    direction, not the average."""
    return (d[..., 2] * d[..., 2]) / np.einsum("...i,...i->...", d, d)


# ---------------------------------------------------------------------------
# csgo_volume_viewer
# ---------------------------------------------------------------------------
def volume_march_start(world_pos):
    """(p + 16) * 0.03125 -- a fixed [-16, 16] box mapped to [0,1]. 0.03125
    is 1/32 exactly and the offset is 16, so the volume this viewer shows
    is 32 units on a side regardless of what is in it."""
    return (world_pos + 16.0) * 0.03125


def volume_march_step(uvw, direction):
    """uvw + dir * 0.0199999995529651641845703125 -- a fixed step in
    NORMALIZED volume space, so the world-space step size depends on the
    box, not on the ray."""
    return uvw + direction * 0.0199999995529651641845703125


def volume_resolve(acc, steps):
    """acc / float(steps) where `steps` is the loop counter AT EXIT.

    The loop breaks early when the ray leaves the box, so the divisor is
    the number of steps actually taken -- a ray that exits after 3 steps
    divides by 3, not by 64. Dividing by the loop bound instead darkens
    every ray that leaves early, which is most of them."""
    return acc / steps[..., None]


# ---------------------------------------------------------------------------
# msaa_resolve (r0/m39 -- the TONEMAPPED weighted resolve)
# ---------------------------------------------------------------------------
def resolve_tonemap_weight(rgb):
    """rgb * (0.5 / (max3(rgb) + 1))

    A Karis-style luminance-weighted average, keyed on the MAX CHANNEL, not
    on luma -- so a saturated red and a white of the same luma get
    different weights. The 0.5 is 1/nSamples for this 2x module and is a
    LITERAL, so it does not follow the sample count."""
    m = np.max(rgb, axis=-1)
    return rgb * (0.5 / (m + 1.0))[..., None]


def resolve_tonemap_invert(rgb):
    """rgb / (1 - max3(rgb)), after a clamp to 0.99 -- the inverse of the
    weight above. The clamp is what keeps the divide finite, and it is the
    reason the resolve cannot represent anything at or above 1.0 after
    weighting."""
    m = np.max(rgb, axis=-1)
    return rgb * (1.0 / (1.0 - m))[..., None]


def resolve_dither(frag_xy, phase, amount):
    """(fract(dot((131, 312), frag + phase) * k) - 0.5) * 0.0667 * amount

    with k = (0.00970873795449733734130859375,
              0.0140845067799091339111328125,
              0.010309278033673763275146484375)
      = 1/103, 1/71, 1/97 in float32 -- three coprime-ish reciprocals, so
    the three channels decorrelate. A single k would dither all three
    identically and produce coloured banding rather than noise."""
    k = np.like([0.00970873795449733734130859375,
                 0.0140845067799091339111328125,
                 0.010309278033673763275146484375], frag_xy)
    d = np.einsum("...i,i->...", frag_xy + phase[..., None],
                  np.like([131.0, 312.0], frag_xy))
    x = d[..., None] * k
    # GLSL fract is x - floor(x) and is NON-NEGATIVE for negative x.
    # numpy's modf keeps the sign, which would make the dither one-sided on
    # half the screen.
    return (x - np.floor(x) - 0.5) * 0.066666670143604278564453125 * amount[..., None]


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
