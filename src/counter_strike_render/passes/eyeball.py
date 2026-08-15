#!/usr/bin/env python3
"""`csgo_eyeball` pixel-stage kernels.

Reference: `docs/projects/counter-strike-sft/decompiled/csgo_eyeball_ps_r1_m2.metal`
(committed a3fefe94; MSL 2.3, for the reason in the manifest).

`csgo_eyeball` is A SEPARATE FAMILY, not csgo_character's `S_EYEBALLS`
combo. Three things read out of the bytecode that a textbook eye shader
gets wrong, and that are kept here because they are what shipped:

  * there is NO `refract()` anywhere in the family -- the
    `normalize(Ne - (0,0,0.5))` cup at :389 is the whole of the corneal bend;
  * there is NO cornea specular lobe, only a roughness swap to 0.1 (:390);
  * the aim/walleye terms (`g_flEyeBallWalleyeL1/R1`, `g_nEyeTargetBindIdx`)
    are in the VERTEX package only. The pixel shader never reads an aim
    direction, so all directional eye behaviour comes from the vertex stage
    moving the geometry, and adding a pixel-side aim would be inventing a
    term the reference does not have.

See `character.py`'s header for why these live in an importable module.
"""
from __future__ import annotations

from ._chargl import (clip, const_vec, cos, cross, dot, length, maximum,
                      normalize, sin, stack, where)


def eyeball_ray_sphere(wpos, eye, centre, radius):
    """The iris ray-sphere trace. r1_m2.metal:344-350.

        r    = g_flEyeBallRadius1 * vRadiusScale                     :344
        dir  = normalize(P - cameraPos)                              :345
        oc   = cameraPos - centre                                    :346
        b    = dot(oc, dir)                                          :347
        disc = b*b - (dot(oc,oc) - 576*r*r)         # 576 = 24^2      :348
        t    = disc > 0 ? -b - sqrt(disc) : 0                        :350

    Returns `t`. `dir` is `normalize(P - cameraPos)` and the caller has it
    already; the discard where `t - 0.001 < 0` (:358) is the caller's too.

    THE 576. It is a literal in the shader, not a scale the caller passes,
    and 576 = 24^2 means the radius uniform is in units of 1/24 of whatever
    `centre` is in -- so `radius` here is NOT a world-space radius and must
    not be "fixed" to one.

    WHAT DIFFERS: `centre` is `float3(16, 16, 0)`, a literal, in every
    static and dynamic combo of the family, in the space of
    `vPositionWs + perViewOffset`; no per-eye centre uniform or varying
    reaches the pixel shader at all. That literal cannot be right for two
    eyes on a moving player, so the caller supplies the centre and the
    shipped literal is the default (`--eyeball-centre`).
    """
    d = normalize(wpos - eye)
    oc = eye - centre
    b = dot(oc, d)
    disc = b * b - (dot(oc, oc) - 576.0 * radius * radius)
    return where(disc > 0.0, -b - maximum(disc, 0.0) ** 0.5, disc * 0.0)


def eyeball_iris_uv(ne, iris_size):
    """Iris-plane projection and the radial pinch. r1_m2.metal:368-370.

        p    = (Ne.x, dot(Ne, (0,-1,0)))                             :368
        rad  = length(p)                                             :369
        uvW  = mix(p * (2 - g_flEyeIrisSize1), p, vec2(rad))         :370

    The blend weight is the RADIUS itself, not a smoothstep of it: the pinch
    is full-strength at the pupil and vanishes at the limbus where rad -> 1.
    Returns `uvW`; the sampler UV is `uvW * 0.5 + 0.5` at :371, with an
    explicit LOD bias of -0.5. `eyeball_iris_radius` is the same `rad` when
    the caller needs it separately.
    """
    p2 = stack([ne[..., 0], dot(ne, const_vec(ne, [0.0, -1.0, 0.0]))])
    rad = length(p2)[..., None]
    pinched = p2 * (2.0 - iris_size[..., None])
    return pinched + (p2 - pinched) * rad


def eyeball_iris_radius(ne):
    """`length(vec2(Ne.x, dot(Ne, (0,-1,0))))`. r1_m2.metal:368-369.

    The iris-plane radius: 0 at the pupil centre, 1 at the limbus. It gates
    the pupil smoothstep (:377) as well as the pinch, so it is a term in its
    own right and not an intermediate.
    """
    return length(stack([ne[..., 0],
                         dot(ne, const_vec(ne, [0.0, -1.0, 0.0]))]))


def eyeball_hue_rotate(rgb, alpha, angle):
    """Hue rotation about the grey axis. r1_m2.metal:373-375.

    Rodrigues about `(0.57735002040863037109375,)*3` = 1/sqrt(3), then
    blended back toward the untouched colour by the iris texture's ALPHA:

        rot  = c*v + sin*cross(axis, v) + axis*dot(axis, v)*(1 - c)
        hued = mix(v, rot, iris.w)

    `iris.w` gating the rotation is what confines the tint to the iris disc
    without a second mask, and it is why the sclera stays white.
    """
    ax = 0.57735002040863037109375
    c = cos(angle)[..., None]
    s = sin(angle)[..., None]
    axis = const_vec(rgb, [ax, ax, ax]) + rgb * 0.0
    rot = (rgb * c + cross(axis, rgb) * s
           + axis * dot(axis, rgb)[..., None] * (1.0 - c))
    return rgb + (rot - rgb) * alpha[..., None]


def eyeball_normal_blend(n_geo, ne, iris_a, blend):
    """The corneal normal. r1_m2.metal:389.

        N = normalize(mix(N_geom,
                          normalize(mix(Ne, normalize(Ne - (0,0,0.5)),
                                        iris.w)),
                          pow(blend, 0.5)))

    `normalize(Ne - (0,0,0.5))` is the entire corneal refraction model in
    this family -- a sphere normal pulled toward -Z and renormalised. There
    is no `refract()` and no IOR. The outer weight is `sqrt(blend)`, not
    `blend`, which widens the geometric-to-corneal transition.
    """
    cup = normalize(ne - const_vec(ne, [0.0, 0.0, 0.5]))
    inner = normalize(ne + (cup - ne) * iris_a[..., None])
    w = maximum(blend, 0.0)[..., None] ** 0.5
    return normalize(n_geo + (inner - n_geo) * w)


def eyeball_roughness(blend):
    """`mix(vec2(0.5), vec2(0.1), vec2(blend))`. r1_m2.metal:390.

    The whole of the family's specular response to the cornea: a roughness
    SWAP, no extra lobe. Both components move together -- the reference
    writes a float2 and the two halves are the same expression, so the eye
    is isotropic even though the surrounding pipeline is not.
    """
    return 0.5 + (0.10000000149011611938476562 - 0.5) * blend


def eyeball_blend_mask(mask_x, face_on):
    """`eyeMask.x * smoothstep(0.1, 0.3, saturate(Ne.z))`. r1_m2.metal:387,
    with `faceOn` from :376.

    The smoothstep band is 0.1..0.3 on the face-on term, so the iris fades
    out well before the silhouette; a wider band is the difference between
    an eye and a decal, and the band is read, not chosen.
    """
    t = clip((face_on - 0.100000001490116119384765625)
             / (0.300000011920928955078125 - 0.100000001490116119384765625),
             0.0, 1.0)
    return mask_x * (t * t * (3.0 - 2.0 * t))
