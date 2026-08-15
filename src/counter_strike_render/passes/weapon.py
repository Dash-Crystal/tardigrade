#!/usr/bin/env python3
"""csgo_weapon / csgo_legs_prepass term expressions, lifted out of the renderer.

WHY THIS MODULE EXISTS
----------------------
The conformance registry resolves ``impl=`` as a real import. Every one of
these expressions used to live inline in ``gpu_render.py``, a 27k-line script
that runs argparse at import time and cannot be imported from a test. The
registry's own instruction for that case is to lift the expression into a
module both the renderer and the registry call, which is what this is. The
renderer imports these; it does not keep a second copy.

THE REFERENCE
-------------
Everything here is transcribed from files committed under
``docs/projects/counter-strike-sft/reference/csgo_weapon/``. The default
citation module is

    csgo_weapon__allon_c3041_d2.glsl          6,946 lines

-- static combo 3041, the maximal *beauty* combo (every optional surface axis
on, neither mode axis) at ``D_BAKED_LIGHTING_FROM_PROBE``. It is the one file
in which every optional term appears under one set of line numbers.

Two terms are cited against their own modules instead, because they replace
the shader rather than extend it:

    csgo_weapon__mode_depth1_c1024_d2.glsl       26 lines
    csgo_legs_prepass__only_d0.glsl              34 lines

A CITATION WITHOUT A MODULE IS NOT A CITATION. db11cafd cited ``glsl:313``,
``glsl:1321``, ``:6894-6899`` against a combo it did not name and did not
commit; :6894-6899 lands ~44 lines off the term it claimed in this module.
Every line range below names its file.

BACKEND
-------
These run on torch tensors inside the renderer and on numpy arrays inside the
conformance runner, so they use only arithmetic that both spell the same way,
plus the four dispatched helpers below. Nothing here allocates a device
tensor or reads ``args``: an expression that needs engine data takes it as a
parameter, so the same call is checkable on synthetic inputs.
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Backend dispatch. Four operations spell differently in torch and numpy; the
# rest (+ - * / ** abs()) do not.
# --------------------------------------------------------------------------


def _clip(x, lo, hi):
    if hasattr(x, "clamp"):
        return x.clamp(lo, hi)
    import numpy as np
    return np.clip(x, lo, hi)


def _sin(x):
    if hasattr(x, "sin"):
        return x.sin()
    import numpy as np
    return np.sin(x)


def _sqrt(x):
    if hasattr(x, "sqrt"):
        return x.sqrt()
    import numpy as np
    return np.sqrt(x)


def _fract(x):
    return x - _floor(x)


def _floor(x):
    if hasattr(x, "floor"):
        return x.floor()
    import numpy as np
    return np.floor(x)


def _norm_last(v):
    """Euclidean norm over the last axis, keeping the axis."""
    if hasattr(v, "norm"):
        return v.norm(dim=-1, keepdim=True)
    import numpy as np
    return np.linalg.norm(v, axis=-1, keepdims=True)


def _last1(x):
    """Give a per-sample scalar a trailing axis so it broadcasts against a
    vec2/vec3. A shader uniform is a scalar; a batched transcription of it is
    an (N,) array, and (N,) * (N,3) is an error in both backends."""
    if hasattr(x, "unsqueeze"):
        return x.unsqueeze(-1)
    if getattr(x, "shape", ()):
        import numpy as np
        return np.expand_dims(x, -1)
    return x


def _stack_last(parts):
    first = parts[0]
    if hasattr(first, "unsqueeze"):
        import torch
        return torch.stack(parts, dim=-1)
    import numpy as np
    return np.stack(parts, axis=-1)


# GLSL's smoothstep, including the e0 > e1 direction csgo_weapon uses at the
# proximity dissolve. The spec calls e0 >= e1 undefined; every implementation
# including the one this shader was compiled for evaluates the same formula,
# which descends, and the shader relies on that.
def glsl_smoothstep(e0, e1, x):
    t = _clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def glsl_mix(a, b, t):
    """GLSL mix(x, y, a) = x*(1-a) + y*a.

    Spelled out because getting the operand order backwards is silent: it is
    still a lerp, still in range, still smooth. The shimmer term shipped
    inverted for exactly this reason.
    """
    return a + (b - a) * t


# --------------------------------------------------------------------------
# Surface set
# --------------------------------------------------------------------------
# The diagonal two-channel normal decode. 1.00392162799835205078125 is
# 256/255 exactly, so a mid-grey pair decodes to zero, and z is an L1
# complement rather than a sqrt.
OCTA_BIAS = 1.00392162799835205078125

# Rec.709-ish luma, as the module spells it. Every luma in csgo_weapon --
# shimmer, tools-vis, the glitter desaturations -- uses this same triple.
LUMA = (0.2125000059604644775390625,
        0.7153999805450439453125,
        0.07209999859333038330078125)


def octa_normal_decode(r, g):
    """allon_c3041_d2:540-542.

        _16783 = (r + g) - 1.00392162799835205078125
        _11176 = r - g
        normalize(vec3(vec2(_16783, _11176),
                       (1.0 - abs(_16783)) - abs(_11176)))
    """
    a = (r + g) - OCTA_BIAS
    b = r - g
    z = (1.0 - abs(a)) - abs(b)
    n = _stack_last([a, b, z])
    return n / _norm_last(n)


def metal_remap(lo, hi, mask_y):
    """allon_c3041_d2:533 -- ``mix(_5618._m24.x, _5618._m24.y, _21761.y)``.

    This is the term db11cafd first read as a ROUGHNESS remap. It is not:
    roughness is ``mask.x`` at :535, and what :533 produces is spent as the
    metalness lerp into F0. Reading it as roughness threw the term away and
    left F0 dielectric everywhere.
    """
    return glsl_mix(lo, hi, mask_y)


def vcolor_tint(colour_rgb, vcolor_rgb, tint_mask):
    """allon_c3041_d2:528-529.

        _6652  = colour.xyz
        _10910 = mix(_6652, _6652 * _3942.xyz, vec3(_19334.x))

    The vertex colour multiplies the albedo only where the tint mask says so;
    the mask is a texture channel, not a flag.
    """
    return glsl_mix(colour_rgb, colour_rgb * vcolor_rgb,
                    _last1(tint_mask))


def tangent_frame_normal(tangent, bitangent, normal, n_ts):
    """allon_c3041_d2:586.

        normalize(((T * n.x) + (B * n.y)) + (N * n.z))
    """
    v = (tangent * n_ts[..., 0:1]
         + bitangent * n_ts[..., 1:2]
         + normal * n_ts[..., 2:3])
    return v / _norm_last(v)


def self_illum(tex_rgb, tint_rgb, albedo_rgb, albedo_blend):
    """allon_c3041_d2:588-589.

        _11580 = texture(...).xyz * _5618._m19.xyz
        _17133 = mix(_11580, _11580 * _10910.xyz, vec3(_5618._m21))

    The emissive is tinted, then optionally re-modulated by the albedo. The
    scroll that produced its UV is :587 and is `self_illum_uv` below.
    """
    lit = tex_rgb * tint_rgb
    return glsl_mix(lit, lit * albedo_rgb, _last1(albedo_blend))


def self_illum_uv(uv, scroll_xy, t):
    """allon_c3041_d2:587 -- ``uv + fract(_5618._m20.xy * _4459._m1)``.

    The fract is INSIDE, on the scroll offset, not on the summed coordinate.
    """
    return uv + _fract(scroll_xy * _last1(t))


# --------------------------------------------------------------------------
# The SFX mask flow
# --------------------------------------------------------------------------
def sfx_gate(sfx_amount, per_view_scale):
    """allon_c3041_d2:451-452 -- ``(g_flSfxAmount * perViewScale) > 0``.

    The whole SFX block is behind this one product, and it is a UNIFORM gate.
    S_ENABLE_SFX_MASK, the static axis that names it, emits no instruction of
    its own: at d2, static combos 0 and 128 resolve to SPIR-V with the same
    sha256 838cd370e8c6095e... and share record 0. So the axis may not be
    implemented as a branch -- this product is the only gate there is.
    """
    return sfx_amount * per_view_scale


def sfx_uv_scale(uv, texel_ratio):
    """allon_c3041_d2:461-470.

        _18197 = uv * 2.5
        ratio  = length(cross(vec3(dFdx(uv), 0), vec3(dFdy(uv), 0)))
               / max(1e-4, length(cross(dFdx(P), dFdy(P))))
        _20826 = (ratio < 0.002) ? _18197 * 3.0 : _18197

    A texel-density branch: where the UV derivative is small against the
    world derivative -- a densely mapped patch -- the flow map is read three
    times finer. `texel_ratio` is that quotient, computed by the caller
    because it needs screen-space derivatives.
    """
    base = uv * 2.5
    if getattr(texel_ratio, "shape", ()):
        sel = _last1(texel_ratio < 0.00200000009499490261077880859375)
        return base + ((base * 3.0) - base) * sel
    return base * 3.0 if texel_ratio < 0.002 else base


def sfx_wipe(gate, phase, wipe_width, local_z):
    """allon_c3041_d2:476.

        clamp((((gate * 0.25) - fract(phase)) * 5.0) / (width + 0.001),
              0, 1)
        * clamp((local.z + 0.75) * 4.0, 0, 1)

    Two clamps multiplied: an animated band that sweeps as `phase` advances,
    times a hard cutoff below local z = -0.75. The 0.001 in the denominator
    is the shader's, not a guard added here.
    """
    band = _clip((((gate * 0.25) - _fract(phase)) * 5.0)
                 / (wipe_width + 0.001000000047497451305389404296875),
                 0.0, 1.0)
    cut = _clip((local_z + 0.75) * 4.0, 0.0, 1.0)
    return band * cut


def sfx_uv_offset(uv, flow_xy, wipe, flow_a):
    """allon_c3041_d2:477.

        uv + ((flow.xy * -0.02) * wipe) * flow.w

    `flow_xy` is the decoded ``(tex.xy * 2 - 1)`` with y negated (:472-473).
    """
    k = _last1(wipe * flow_a)
    return uv + (flow_xy * -0.0199999995529651641845703125) * k


# --------------------------------------------------------------------------
# The tail: shimmer, dissolve, screen door
# --------------------------------------------------------------------------
def shimmer(rgb, ndv, dither_y, tint_rgb, amount, t):
    """allon_c3041_d2:6884-6888, the SFX shimmer.

        if (_5618._m197 > 0.0) {
          _11479 = 1.0 - clamp(dot(N, V), 0.0, 1.0);
          rgb = mix(rgb,
                    _5618._m198 * (mix(dot(rgb, LUMA), 0.5, 0.5)
                                   + 4.0 * pow(mix(_11479 * dither.y,
                                                   1.0, _11479),
                                               mix(3.0, 6.0,
                                                   1.0 + sin(t*20)*0.5))),
                    vec3(_5618._m197));
        }

    TWO THINGS THE SHAPE INVITES YOU TO GET WRONG, and db11cafd got both:

    * ``mix(k * dither.y, 1.0, k)`` interpolates FROM the dithered term TO
      1.0 BY k. Written as ``k*dither + (1-k)`` it is still a lerp, still in
      [0,1], still smooth -- and it is the other one. Facing the viewer
      (k -> 0) the reference gives the dithered term times ~0; the inversion
      gives 1.0. That is the whole term, backwards, silently.
    * ``_5618._m198`` is a **vec3** tint. It was implemented as a scalar read
      off an unrelated column (`wpn_refract_tint`), which cannot carry a
      colour.

    The exponent's mix parameter is ``1 + sin(t*20)*0.5``, which leaves [0,1]
    for half its period, so the pow exponent EXTRAPOLATES past 6 rather than
    oscillating between 3 and 6. Transcribed, not tidied.
    """
    k = 1.0 - _clip(ndv, 0.0, 1.0)
    lum = (rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1]
           + rgb[..., 2] * LUMA[2])
    inner = glsl_mix(k * dither_y, 1.0, k)
    expo = glsl_mix(3.0, 6.0, 1.0 + _sin(t * 20.0) * 0.5)
    val = glsl_mix(lum, 0.5, 0.5) + 4.0 * (inner ** expo)
    return glsl_mix(rgb, tint_rgb * _last1(val), _last1(amount))


def dissolve_threshold(dist, strength):
    """allon_c3041_d2:6930-6933.

        f = clamp((smoothstep(6.0, 2.0, distance(origin, P)) - 1.0) * -1.0,
                  0.0, 1.0)
        t = mix(f * 0.51 + 0.5, f * 0.91 + 0.1,
                smoothstep(0.1, 1.0, strength))

    smoothstep(6, 2, d) DESCENDS, so f is 0 within 2 units of the origin and
    1 beyond 6: the threshold RISES with distance and the fragment dissolves
    away from the origin, not toward it.
    """
    f = _clip((glsl_smoothstep(6.0, 2.0, dist) - 1.0) * (-1.0), 0.0, 1.0)
    return glsl_mix(f * 0.5099999904632568359375 + 0.5,
                    f * 0.90999996662139892578125
                    + 0.100000001490116119384765625,
                    glsl_smoothstep(0.100000001490116119384765625, 1.0,
                                    strength))


def dissolve_keep(dither_y, threshold):
    """allon_c3041_d2:6933-6936 -- ``if (dither.y - t < 0.0) discard;``.

    Returns the KEEP mask, so the sense is inverted exactly once, here.
    """
    return (dither_y - threshold) >= 0.0


def screendoor_keep(alpha, dither_y):
    """The stochastic-alpha discard, spelled identically in three modules:

        allon_c3041_d2:6938-6944
        mode_depth1_c1024_d2:19-25
        csgo_legs_prepass__only_d0:28

        if (alpha < 1.0)
            if (fma(alpha, 2.0, -1.5) + dither.y < 0.0) discard;

    The outer ``alpha < 1.0`` matters: a fully opaque fragment is never
    dithered, which is not the same as one that happens to pass. Over a
    uniform dither the keep probability is clamp(2*alpha - 0.5, 0, 1), so
    alpha <= 0.25 never survives and alpha >= 0.75 always does.

    Returns the KEEP mask.
    """
    keep = (alpha * 2.0 - 1.5 + dither_y) >= 0.0
    return keep | (alpha >= 1.0)


def legs_fade(fade_distance, cam_dist, local_pos,
              cone_cos=0.800000011920928955078125, eye_z=55.0):
    """csgo_legs_prepass__only_d0:28, the whole varying part of that shader.

        smoothstep(F - 35.0, F, length(camPos - (worldPos + viewOrigin)))
        * step(0.8, dot(normalize(local - vec3(0,0,55)), vec3(0,0,-1)))

    F is a DISTANCE, not an opacity: the legs fade IN as the camera gets
    further than F-35 from them. The cone is anchored at (0, 0, 55) -- hip
    height in this unit system -- and opens downward, cos 0.8.

    The result is fed to the screen door as ``fade * 3.0`` (:28), not as an
    alpha, which is why nearly the whole cone survives the dither.

    ``cone_cos`` and ``eye_z`` are the shader's own literals -- 0.8 and 55.0
    -- named as parameters so the renderer can carry them as material
    columns without editing the transcription. Their defaults ARE the
    literals, so a call that omits them is the reference.
    """
    band = glsl_smoothstep(fade_distance - 35.0, fade_distance, cam_dist)
    d = local_pos - _stack_last(
        [local_pos[..., 0] * 0.0, local_pos[..., 1] * 0.0,
         local_pos[..., 2] * 0.0 + eye_z])
    d = d / _norm_last(d)
    step_ = (-d[..., 2] >= cone_cos)
    if hasattr(step_, "to"):
        step_ = step_.to(band.dtype)
    else:
        step_ = step_.astype(band.dtype)
    return band * step_

# --------------------------------------------------------------------------
# S_OPAQUE_REFRACT + D_USE_BGREFRACT
# --------------------------------------------------------------------------
# WHERE THIS CODE ACTUALLY LIVES. S_OPAQUE_REFRACT emits no instruction of
# its own: at d2, static combos 0 and 2048 both resolve to SPIR-V sha256
# 838cd370e8c6095ec76a973f9202437a279c125f636e3805d7c83821257c9b49. The
# refraction is behind the DYNAMIC axis D_USE_BGREFRACT, which ships only
# under S_OPAQUE_REFRACT -- 128 statics x 16 dynamic ids = the 2,048 pairs
# the container reports. db11cafd transcribed this path from c2048/d2, a
# module that provably does not contain it.
#
# Two committed modules carry it and they AGREE structurally:
#     allon_c3041_bgr1_d34:5534-5550
#     opaque_refract1_c2048_bgr1_d34:512-528
# with DIFFERENT member numbers for the same uniforms (_m201/_m202/_m204..
# vs _m29/_m30/_m32..) and different per-view members for the same view
# vectors (_4459._m9/_m8 vs _m8/_m7). Same family, two combos, incompatible
# _mN identity -- so the view vectors below are named `view_a`/`view_b` by
# their ROLE in the expression, not by a guess at what they are.


def refract_screen_uv(frag_xy, inv_screen, n_world, view_a, view_b, scale):
    """allon_c3041_bgr1_d34:5534 / c2048_bgr1_d34:512.

        (gl_FragCoord.xy * invScreen.xy)
        + vec2(dot(cross(view_a, view_b), N), dot(view_b, N)) * scale

    The offset is the shading normal projected onto a 2-frame built from
    the per-view vectors: cross(a, b) for x and b itself for y. Note the
    SAME `view_a` reappears in the edge term below -- one vector, two uses,
    which is how the two are known to be the same member.
    """
    off = _stack_last([_dot_last(_cross3(view_a, view_b), n_world),
                       _dot_last(view_b, n_world)])
    return frag_xy * inv_screen + off * _last1(scale)


def refract_blur5(centre, tap_pp, tap_mp, tap_pm, tap_mm):
    """allon_c3041_bgr1_d34:5538-5543 / c2048_bgr1_d34:516-521.

        if (blur > 0.0) {
            s = blur * 4.0;
            bg = (bg + bg(uv + ( ix,  iy)*s) + bg(uv + (-ix,  iy)*s)
                     + bg(uv + ( ix, -iy)*s) + bg(uv + (-ix, -iy)*s)) * 0.2;
        }

    FIVE taps at a flat 1/5, not four -- the centre tap is inside the sum.
    The offsets are the SCREEN-SIZE reciprocal `invScreen.xy` times
    `blur * 4`, i.e. the blur radius is in texels of the render target, not
    in UV. The four taps are passed in already sampled because the fetch is
    the caller's; what is transcribed here is the weighting.
    """
    return ((((centre + tap_pp) + tap_mp) + tap_pm) + tap_mm) * \
        0.20000000298023223876953125


def refract_blend(albedo, bg, mask_x, view_a, n_world, edge_k, amount,
                  contrast, tint, sticker_a=0.0):
    """allon_c3041_bgr1_d34:5550 / c2048_bgr1_d34:528.

        mix(albedo,
            mix(vec3(0.5), bg, vec3(1.0 + contrast)) * tint,
            vec3((((clamp((dot(view_a, N) - (-1.0)) / (1.0 - edgeK),
                          0.0, 1.0) * (-1.0)) + 1.0)
                  * mask.x) * amount))

    `tint` is a FLOAT here (`float _m209` / `float _m37`), not a colour --
    which is why taking it for the shimmer's vec3 g_vShimmerTint was wrong
    in both directions.

    NO GUARD ON THE DENOMINATOR. At edgeK == 1 the reference divides by
    zero, the clamp takes the resulting inf to 1, and the edge term becomes
    0. Adding a `max(1e-3)` there changes the answer at exactly the value
    the shader handles by construction, so it is not added.

    `sticker_a` is an INTERACTION, not part of this term: the all-on module
    multiplies the blend weight by `(1.0 - stickerAccum.x)` at :5550 and
    the sticker-free module (c2048) has no such factor. It defaults to 0 --
    the value it takes when S_STICKERS is off -- so a call that omits it is
    the c2048 expression exactly.
    """
    edge = (_clip((_dot_last(view_a, n_world) - (-1.0)) / (1.0 - edge_k),
                  0.0, 1.0) * (-1.0)) + 1.0
    w = ((edge * mask_x) * amount) * (1.0 - sticker_a)
    mixed = glsl_mix(0.5, bg, _last1(1.0 + contrast)) * _last1(tint)
    return glsl_mix(albedo, mixed, _last1(w))


def _dot_last(a, b):
    p = a * b
    return p.sum(-1) if hasattr(p, "dim") else p.sum(axis=-1)


def _cross3(a, b):
    return _stack_last([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]])

# --------------------------------------------------------------------------
# S_STICKERS -- five slots at a 27-uniform stride
# --------------------------------------------------------------------------
# The five slot bodies in allon_c3041_d2 begin at :828, :1760, :2692, :3624
# and :4556 -- a uniform 932-line stride -- and their per-slot uniform blocks
# are _m36.., _m63.., _m90.., _m117.., _m144.., a stride of 27 members,
# confirmed by the five texture handles _m42, _m69, _m96, _m123, _m150.
# The slots are IDENTICAL code, so one transcription serves all five and the
# renderer loops it.
#
# The db11cafd citations for this block (:671-760, :752, :1455, :2158, :2861,
# :3564) came from the dead module and are marked UNSOURCED in gpu_render.py.
# What follows was re-read at the line ranges named here.


def sticker_uv(uv_zw, offset, scale, rot):
    """allon_c3041_d2:829-835, slot 0.

        vec2 _17352 = ((_23261.zw - vec2(0.5)) - _5618._m43)
                      * abs(_5618._m44).x;
        float _10689 = _5618._m45 * 6.28318023681640625;
        vec2 _15799 = vec2((x * cos(a)) - (y * sin(a)),
                           (x * sin(a)) + (y * cos(a))) + vec2(0.5);

    Three things a paraphrase loses:
      * the coordinate is `.zw` of the interpolated texcoord -- the SECOND
        uv pair, not the albedo's `.xy`;
      * the scale is `abs(...)` and then `.x` of a vector, so a negative
        authored scale does not mirror the sticker;
      * the turns-to-radians constant is 6.28318023681640625, which is NOT
        float32 2*pi (6.2831854820251465). It is transcribed, not replaced
        by `2 * math.pi`.
    """
    q = ((uv_zw - 0.5) - offset) * _last1(abs(scale))
    a = rot * 6.28318023681640625
    ca, sa = _cos(a), _sin(a)
    x, y = q[..., 0], q[..., 1]
    return _stack_last([(x * ca) - (y * sa), (x * sa) + (y * ca)]) + 0.5


def sticker_outside(q):
    """allon_c3041_d2:836-846.

        if (clamp(q.x, 0.0, 1.0) != q.x) outside = true;
        else                             outside = clamp(q.y,0,1) != q.y;

    A CLAMP-INEQUALITY test, not `q < 0 || q > 1`. The two agree on every
    ordinary value and disagree on NaN: `clamp(nan) != nan` is true, so a
    NaN coordinate is REJECTED here, where the comparison form would keep
    it. Returns the OUTSIDE mask.
    """
    return (_clip(q[..., 0], 0.0, 1.0) != q[..., 0]) | \
           (_clip(q[..., 1], 0.0, 1.0) != q[..., 1])


def sticker_alpha(tex_a, wear):
    """allon_c3041_d2:869-870.

        float _8480 = clamp(_20322.w * 12.75, 0.0, 1.0) * _19148;

    12.75 is 255/20: the authored alpha saturates over the bottom 20/255 of
    the channel, so all but the faintest edge is fully opaque before wear
    scales it.
    """
    return _clip(tex_a * 12.75, 0.0, 1.0) * wear


def sticker_lod_fade(lod):
    """allon_c3041_d2:876-877.

        float _8795 = 1.0 - clamp(_6345.x - 3.0, 0.0, 1.0);

    The sticker fades out over ONE mip level starting at LOD 3, and the
    whole holo branch below is gated on this being > 0 (:878). The LOD is
    queried at :875 through `textureQueryLod` on the coordinate clamped to
    half a texel inside the page (:873-875) -- the query, not the fetch, is
    what the clamp protects.
    """
    return 1.0 - _clip(lod - 3.0, 0.0, 1.0)


def sticker_holo_alpha(alpha, holo_a, fade):
    """allon_c3041_d2:885.

        mix(_8480, max(_8480, _holoTap * 0.699999988079071044921875), _8795)

    `max` then `mix`, in that order: the holo tap can only ADD opacity, and
    the LOD fade then interpolates back toward the plain alpha. Reversing
    the two lets a holo tap REMOVE opacity, which the module never does.
    """
    return glsl_mix(alpha, _maximum(alpha, holo_a * 0.699999988079071044921875),
                    fade)


def sticker_holo_uv(q, view_a, view_b, n_world, rot):
    """allon_c3041_d2:882-885, the offset the second tap is taken at.

        float _15250 = dot(cross(_4459._m9, _4459._m8), -_15878);
        float _24057 = dot(_4459._m8, _15878);
        clamp(((_9983 - vec2(0.5))
               - (vec2((_15250 * cos(a)) - (_24057 * sin(a)),
                       (_15250 * sin(a)) + (_24057 * cos(a))) * 0.01))
              + vec2(0.5), vec2(0.0), vec2(1.0))

    The parallax vector is built from the SAME two per-view vectors the
    refraction uses (cross(a,b) and b) and rotated by the SAME slot angle --
    so the holo shift follows the sticker's own rotation, not the screen.
    Note the asymmetry that is easy to normalise away: the first dot takes
    the NEGATED normal, the second does not.
    """
    px = _dot_last(_cross3(view_a, view_b), -n_world)
    py = _dot_last(view_b, n_world)
    a = rot * 6.28318023681640625
    ca, sa = _cos(a), _sin(a)
    d = _stack_last([(px * ca) - (py * sa), (px * sa) + (py * ca)]) * \
        0.00999999977648258209228515625
    return _clip(((q - 0.5) - d) + 0.5, 0.0, 1.0)


def _cos(x):
    if hasattr(x, "cos"):
        return x.cos()
    import numpy as np
    return np.cos(x)


def _maximum(a, b):
    if hasattr(a, "maximum"):
        return a.maximum(b) if hasattr(a, "dim") else a.maximum(b)
    import numpy as np
    return np.maximum(a, b)

# --------------------------------------------------------------------------
# S_ENABLE_ADJUSTMENTS -- the hue rotation and the 6-segment rainbow ramp
# --------------------------------------------------------------------------
# db11cafd cited these at :396-412 and :464-519 of the dead module and they
# are marked UNSOURCED in gpu_render.py. Re-read at allon_c3041_d2:709-723
# and :5759-5800.


def hue_angle(strength, view_dir, n_world, mask_z):
    """allon_c3041_d2:709.

        float _21433 = (_5618._m200
                        * (1.0 - dot(normalize(_4459._m7.xyz - _7715.xyz),
                                     _15878))) * _24590;

    A Fresnel-shaped ANGLE, in radians, gated by the mask's .z channel --
    the same channel S_ENABLE_ADJUSTMENTS reads. The shift is strongest at
    grazing angles and zero head-on.

    `view_dir` IS the `normalize(camPos - worldPos)` of the line, taken as
    an input rather than computed here, and that is a statement about the
    DOMAIN rather than a convenience. Registered with the two positions
    instead, the conformance runner refused the term REF_NAN: the generator
    emits the same corner for both position domains, the difference is the
    zero vector, and `normalize(0)` is non-finite -- in the reference too.
    That input is a fragment sitting exactly at the eye, which is behind
    the near plane and never rasterised, so it is outside the term's domain
    and the right fix is to say so, not to nudge the generator until the
    degenerate sample stops appearing.
    """
    return (strength * (1.0 - _dot_last(view_dir, n_world))) * mask_z


def hsv_saturation(rgb):
    """allon_c3041_d2:714-721.

        float _18475 = max(r, max(g, b));
        if (_18475 == 0.0) { s = 0.0; break; }
        s = (_18475 - min(r, min(g, b))) / _18475;

    HSV saturation with an EXPLICIT zero-guard on black, written as an
    early break rather than a max() in the denominator. Transcribed in that
    shape: `s = where(mx == 0, 0, (mx - mn) / mx)` divides by zero on black
    in a vectorised form, so the guard is the branch, not an epsilon.
    """
    mx = _max_last(rgb)
    mn = _min_last(rgb)
    safe = _where(mx == 0.0, _ones_like(mx), mx)
    return _where(mx == 0.0, _zeros_like(mx), (mx - mn) / safe)


# 1/sqrt(3) as the module spells it -- the grey axis of the RGB cube, and
# the rotation axis of the hue shift.
GREY_AXIS = 0.57735002040863037109375


def hue_rotate(rgb, angle, saturation):
    """allon_c3041_d2:723 -- Rodrigues about the grey axis.

        mix(vec3(dot(c, LUMA)),
            ((c * cos(a)) + (cross(vec3(0.57735002040863037109375), c)
                             * sin(a)))
            + ((vec3(0.57735002040863037109375)
                * dot(vec3(0.57735002040863037109375), c)) * (1.0 - cos(a))),
            vec3(pow(saturation, 0.125)));

    Rodrigues' formula with k = (1,1,1)/sqrt(3): rotating a colour about
    the grey axis is a hue shift that preserves luma. Two details a
    paraphrase drops:

      * the axis constant is 0.57735002040863037109375 in all three
        components -- the vector is NOT re-normalised, and 3 * k^2 is
        0.99999995 rather than 1, so the rotation is very slightly
        non-orthonormal exactly as shipped;
      * the blend weight is `pow(saturation, 0.125)`, an EIGHTH root, which
        is ~0.84 already at saturation 0.25. Near-grey pixels still take
        most of the rotation; only a true grey is exempt.
    """
    k = GREY_AXIS
    ca, sa = _cos(angle), _sin(angle)
    kv = _stack_last([rgb[..., 0] * 0.0 + k, rgb[..., 1] * 0.0 + k,
                      rgb[..., 2] * 0.0 + k])
    rot = ((rgb * _last1(ca)) + (_cross3(kv, rgb) * _last1(sa))) \
        + (kv * _last1(_dot_last(kv, rgb)) * _last1(1.0 - ca))
    lum = (rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1]
           + rgb[..., 2] * LUMA[2])
    grey = _stack_last([lum, lum, lum])
    return glsl_mix(grey, rot, _last1(saturation ** 0.125))


def rainbow_ramp(t6):
    """allon_c3041_d2:5760-5800, the 6-segment hue ramp.

        float seg = floor(t6);
        float f   = t6 - seg;
        float g   = 1.0 - f;
        seg == 0 -> vec3(1, f, 0)
        seg == 1 -> vec3(g, 1, 0)
        seg == 2 -> vec3(0, 1, f)
        seg == 3 -> vec3(0, g, 1)
        seg == 4 -> vec3(f, 0, 1)
        else     -> vec3(1, 0, g)

    `t6` is `fract(phase) * 6.0` from :5759, so it lies in [0, 6) and the
    final `else` is segment 5. The chain is written as nested if/else in the
    decompilation, not a table lookup, and the LAST branch is unguarded --
    any t6 >= 5 lands there, which is what makes the ramp total rather than
    leaving a hole at exactly 6.0.
    """
    seg = _floor(t6)
    f = t6 - seg
    g = 1.0 - f
    z = f * 0.0
    o = z + 1.0
    out = _stack_last([o, z, g])                      # segment 5 / else
    for k, tri in ((4.0, (f, z, o)), (3.0, (z, g, o)), (2.0, (z, o, f)),
                   (1.0, (g, o, z)), (0.0, (o, f, z))):
        out = _where(_last1(seg == k), _stack_last(list(tri)), out)
    return out


def _min_last(v):
    return v.min(dim=-1).values if hasattr(v, "dim") else v.min(axis=-1)


def _max_last(v):
    return v.max(dim=-1).values if hasattr(v, "dim") else v.max(axis=-1)


def _where(c, a, b):
    if hasattr(c, "dim"):
        import torch
        return torch.where(c, a, b)
    import numpy as np
    return np.where(c, a, b)


def _zeros_like(x):
    return x * 0.0


def _ones_like(x):
    return x * 0.0 + 1.0
