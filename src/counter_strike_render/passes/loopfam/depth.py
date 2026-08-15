"""Depth-buffer reads shared across the loop-carrying families.

The remap-and-divide position reconstruction below is not one family's
idiom: `aoproxy_splat` r0/m0 and `bomb_blast` r0/m0 both open with the
IDENTICAL expression, character for character, before either does anything
family-specific. It is the engine's world-position-from-depth, so it lives
here and both families call it rather than each carrying a copy.
"""
from .._arraylib import np  # the numpy-shaped facade: dispatches to torch when handed
                              # tensors, so every expression below is
                              # unchanged. See passes/_arraylib.py --
                              # 132 of 240 impls raised on a CUDA
                              # tensor before this line.


# ---------------------------------------------------------------------------
# Backend adapters -- FOURTH instance of the numpy-only trap
# ---------------------------------------------------------------------------
# glass.py raised TypeError on module-level numpy constants; ao.py raised
# AttributeError on `.astype`; this module raised on `np.clip`. All three
# were green their whole lives because the conformance harness drives every
# term with numpy.
#
# THIS ONE IS DEVICE-DEPENDENT AND THAT IS THE WORSE PROPERTY. `np.clip` on
# a CPU tensor succeeds -- numpy converts through `__array__` -- and on a
# CUDA tensor raises "can't convert cuda:0 device type tensor to numpy". So
# a CPU smoke passes and the GPU run dies. Any torch-callability probe that
# does not run ON THE DEVICE will certify this module as fine.
def _clip(x, lo, hi):
    return np.clip(x, lo, hi) if isinstance(x, np.ndarray) else x.clamp(lo, hi)


def _dot(a, b):
    if isinstance(a, np.ndarray):
        return np.einsum("...i,...i->...", a, b)
    return (a * b).sum(-1)


def linear_depth_remap(raw, near, far, scale, bias):
    """Depth-buffer value -> the reciprocal-space coordinate the divide uses.

    clamp((raw - near) / (far - near), 0, 1) * scale + bias

    `near`/`far` are the CB's two depth-range members, not the projection's
    near and far planes; the name follows their use, which is a linear
    remap of the stored value onto [0,1] before the scale/bias.
    """
    t = _clip((raw - near) / (far - near), 0.0, 1.0)
    return t * scale + bias


def world_pos_from_ray(eye, ray, fwd, raw, near, far, scale, bias):
    """World position where the camera ray through this pixel meets the depth.

    eye + ray * (1 / (remap(raw) * dot(fwd, ray)))

    `ray` is NOT normalized here. Both call sites hand this function a
    vector they normalized themselves (`bomb_blast` negates and normalizes
    the interpolated view vector; `aoproxy_splat` passes an interpolated
    attribute unnormalized), and the shader does not normalize inside the
    expression. Doing it here would silently change one of the two.
    """
    k = linear_depth_remap(raw, near, far, scale, bias)
    denom = k * _dot(fwd, ray)
    return eye + ray * (1.0 / denom)[..., None]


def view_pos_from_uv_scalebias(frag_xy, depth, sb):
    """SAO's position rebuild: ((frag * sb.xy) + sb.zw) * depth, then depth.

    z is the SAMPLED VALUE ITSELF, not a transformed one -- the family
    stores view-space distance in the depth target, so the third component
    needs no reconstruction. Writing `-depth` or a projection inverse here
    would be a plausible-looking change that the reference does not make.
    """
    xy = ((frag_xy * sb[..., :2]) + sb[..., 2:]) * depth[..., None]
    cat = np.concatenate if isinstance(xy, np.ndarray) \
        else __import__("torch").cat
    return cat([xy, depth[..., None]], -1)
