"""`sfm_volumetrics_frustum` -- a volumetric ray-march through a projected
light frustum, with a ROUNDED-RECTANGLE mask.

The second march found in the whole census, after `smoke_volume`. Unlike
smoke's depth-4 nesting this is one loop, but the loop body is where the
family's real content sits: a projected-space rounded-rect coverage
function with four algebraic branches, none of which is a smoothstep.
"""
from .._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.


def slab_clip(t_max, plane_n, plane_d, origin, direction):
    """t = min(t, (-d - dot(n, o)) / dot(n, dir)) when dot(n, dir) < 1e-6.

    Six of these fold the ray against the frustum's six planes. The
    condition is `< 1e-6`, NOT `!= 0` and not `> 0`: a ray heading away
    from a plane is skipped, and so is a ray very nearly parallel to it,
    which is what keeps the divide from producing a huge t rather than a
    guard on the result."""
    dn = np.einsum("...i,...i->...", plane_n, direction)
    safe = np.where(dn < 9.9999999747524270787835121154785e-07, dn, 1.0)
    hit = (-plane_d - np.einsum("...i,...i->...", plane_n, origin)) / safe
    return np.where(dn < 9.9999999747524270787835121154785e-07,
                    np.minimum(t_max, hit), t_max)


def march_step_size(t_max, ndc_near, ndc_far, density, lo=1.0, hi=100.0):
    """t_max / clamp(density * |(0.5, 0.5, 1) * (ndc_far - ndc_near)|, 1, 100)

    The step count comes from the ray's extent IN PROJECTED SPACE, weighted
    (0.5, 0.5, 1) so the depth axis counts double against the two lateral
    ones. A ray that crosses the frustum edge-on therefore gets few steps
    and one going down its axis gets many -- the opposite of a fixed step
    count, and the reason this pass does not shimmer at grazing angles."""
    extent = np.linalg.norm(np.like([0.5, 0.5, 1.0], ndc_far) * (ndc_far - ndc_near),
                            axis=-1)
    return t_max / np.minimum(np.maximum(density * extent, lo), hi)


def hash_dither(frag_xy, scale, phase, step):
    """The four-term fract cascade this module uses to offset the first step.

        s = sin(dot(frag * scale, (12.98980045318603515625,
                                   78.233001708984375)))
        h = fract(fract(s*43758.546875) + fract(s*21879.2734375)
                  + fract(s*10939.63671875) + fract(s*5469.818359375))
        t0 = fract(h + phase) * step

    The four multipliers are 43758.546875 and then successive HALVES of it
    (21879.2734375, 10939.63671875, 5469.818359375). Summing a hash with
    its own halved-frequency copies is what flattens the sin's banding;
    one term alone leaves visible stripes at grazing angles."""
    s = np.sin(np.einsum("...i,i->...", frag_xy * scale,
                         np.like([12.98980045318603515625,
                                  78.233001708984375], frag_xy)))
    h = _fract(_fract(s * 43758.546875) + _fract(s * 21879.2734375)
               + _fract(s * 10939.63671875) + _fract(s * 5469.818359375))
    return _fract(h + phase) * step


def rounded_rect_mask(ndc_xy, corner, softness):
    """The projected-space coverage function, all four branches.

    `corner` and `softness` are each `clamp(u, 0, 1) * 0.9998 + 1e-4`, so
    both live strictly inside (0, 1) -- the shader normalizes them the same
    way and neither can reach an exact endpoint.

    Branch 1, outside the unit square: 0.
    Branch 2, softness >= 1: a pure RADIAL falloff (the rect degenerates to
      a disc).
    Branch 3, softness <= 0 or the point is nearer an EDGE than a corner:
      a Chebyshev (max-norm) falloff -- straight edges.
    Branch 4, otherwise: the corner arc, solved as a quadratic in the
      projected radius, and the coverage is the position between the inner
      and outer roots.

    Three different norms in one function -- L2, L-infinity and the corner
    quadratic -- chosen per pixel. Approximating it with any single one is
    correct in its own region and wrong in the other two."""
    ax = np.abs(ndc_xy[..., 0])
    ay = np.abs(ndc_xy[..., 1])
    mx = np.maximum(ax, ay)
    c = np.clip(corner, 0.0, 1.0) * 0.999800026416778564453125
    cw = c + 9.9999997473787516355514526367188e-05
    s = np.clip(softness, 0.0, 1.0) * 0.999800026416778564453125
    sw = s + 9.9999997473787516355514526367188e-05
    k = 0.99989998340606689453125

    radial = np.clip(1.0 - ((np.sqrt(ax * ax + ay * ay) - (k - c)) / cw), 0.0, 1.0)
    cheby = np.clip(1.0 - ((mx - (k - c)) / cw), 0.0, 1.0)

    inner = k - c
    su = k - s
    r2 = ax * ax + ay * ay
    b = (ax + ay) * su
    disc = (2.0 - 4.0 * sw) + sw * sw
    with np.errstate(invalid="ignore", divide="ignore"):
        t_out = (b + np.sqrt(b * b - r2 * disc)) / r2
        nx, ny = ax / inner, ay / inner
        n2 = nx * nx + ny * ny
        bn = (nx + ny) * su
        t_in = (bn + np.sqrt(bn * bn - n2 * disc)) / n2
        corner_cov = np.clip(1.0 - ((1.0 - t_in) / (t_out - t_in)), 0.0, 1.0)
    corner_cov = np.where((ax + ay) <= inner * (1.0 + k - s), 1.0, corner_cov)
    corner_cov = np.where(1.0 > t_out, 0.0, corner_cov)

    edge_case = (sw <= 0.0) | ((np.minimum(ax, ay) / mx) < (k - s))
    out = np.where(edge_case, cheby, corner_cov)
    out = np.where(sw >= 1.0, radial, out)
    return np.where(mx > 1.0, 0.0, out)


def _fract(x):
    return x - np.floor(x)
