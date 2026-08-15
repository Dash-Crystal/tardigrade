#!/usr/bin/env python3
"""`csgo_character` pixel-stage kernels, lifted out of the renderer script.

WHY THIS FILE EXISTS AT ALL
---------------------------
These expressions were written into `gpu_render.py`, which builds an
`argparse.ArgumentParser` at module scope (line 45) and imports nvdiffrast.
It cannot be imported, so the conformance registry could not resolve
`impl=` against it and the family's terms could not be A/B'd against the
reference at all. The registry's own instruction for that case is to lift
the expression into a module BOTH the renderer and the registry call, which
is what this is. `gpu_render.py` imports these names; there is one copy of
each expression, and the number the registry reports is about the code the
renderer runs.

REFERENCE
---------
`docs/projects/counter-strike-sft/decompiled/csgo_character_ps_*.metal`,
committed in a3fefe94 with their provenance chain. They are MSL, not GLSL,
because spirv-cross cannot express this family's uniform block in std140,
std430 or scalar (its own `.err`, quoted in the manifest) -- so `powr` is
`pow`, `fast::clamp` is `clamp`, `float3` is `vec3`, and `_5538._m18` is a
stripped uniform member whose NAME is not recoverable from the shipped data.

Every function below cites the line range it came from. The citations are
to the committed copies, so they resolve.

WHAT IS NOT HERE
----------------
The paths that read the material side-table (`_e2`), sample a texture
(`sample_fam`) or branch on a CLI flag stay in `gpu_render.py`, which calls
these kernels with the values it read. That split is the registry's stated
preference -- a term driven by scalars can be spanned by a generator; one
that takes a params object cannot be driven at all.
"""
from __future__ import annotations

import math

from ._chargl import (clip, const_vec, cos, cross, dot, length, log, maximum,
                      minimum, normalize, sign, sin, stack, step, where)

_R5400 = "csgo_character_ps_r5400_m0.metal"
_R1050 = "csgo_character_ps_r1050_m0.metal"
_R3000 = "csgo_character_ps_r3000_m0.metal"
_R5145 = "csgo_character_ps_r5145_m{4,12}.metal"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def char_normal_decode(tex):
    """2-channel hemi-octahedral ("diamond") normal decode. r5400_m0:513-515.

        x = (r + g) - 256/255      y = r - g      z = (1 - |x|) - |y|

    NOT the `2*t - 1` decode `apply_normal_map` uses for the world families:
    csgo_character's normal maps carry only two channels and rebuild the
    third, and z comes from an L1 complement rather than a sqrt.
    1.00392162799835205078125 is 256/255 and is the shipped literal.
    """
    r, g = tex[..., 0], tex[..., 1]
    x = (r + g) - 1.00392162799835205078125
    y = r - g
    z = (1.0 - abs(x)) - abs(y)
    return normalize(stack([x, y, z]))


def char_tbn(n_geo, tan_xyz, tan_w, n_ts):
    """Bitangent + tangent-to-world. r5400_m0:537 and :1062.

        B = cross(N_geo, T.xyz) * sign(T.w)
        N = normalize(T.xyz*n.x + B*n.y + N_geo*n.z)

    The backface flip and the green flip that precede it in the shipped
    order are driven by the material side table, so they stay with the
    caller; `n_geo` and `n_ts` arrive already flipped.
    """
    b = cross(n_geo, tan_xyz) * sign(tan_w)[..., None]
    n = normalize(tan_xyz * n_ts[..., 0:1] + b * n_ts[..., 1:2]
                  + n_geo * n_ts[..., 2:3])
    return n, b


def char_aniso_axes(n, tan_xyz, b):
    """The anisotropic axis pair. r5400_m0:1100-1101.

        T' = normalize(cross(B_ws, N))      B' = normalize(cross(N, T_ws))
    """
    return normalize(cross(b, n)), normalize(cross(n, tan_xyz))


def char_spherical_tangent(wpos, origin, n):
    """S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS.

    **NOT TRANSCRIBED — the axis is 0 in all five csgo_character modules
    decompiled, so its body was never seen.** What is known is the axis's
    name and that both the vs and the ps package declare it. This is the
    closest complete form, with the difference stated rather than a stub:
    build the frame from the surface point's position on a sphere about the
    model origin, which is the only construction the name can describe and
    the only one that needs no UV.

        d  = normalize(P - origin)
        up = |d.y| < 0.999 ? (0,1,0) : (1,0,0)
        T' = normalize(cross(up, d)), reprojected into the tangent plane of N
        B' = cross(N, T')

    WHAT DIFFERS FROM THE REFERENCE: the sphere ORIGIN. The engine has the
    model's object-space origin per draw; a packed world has no per-draw
    model transform, so `origin` is the centroid of the geometry the family
    owns, and gpu_render.py prints that substitution at runtime. Everything
    downstream of the axes is the transcribed path and is shared with the
    UV form, so the two modes differ ONLY in the frame.
    """
    d = normalize(wpos - origin)
    up_y = const_vec(d, [0.0, 1.0, 0.0])
    up_x = const_vec(d, [1.0, 0.0, 0.0])
    up = where((abs(d[..., 1:2]) < 0.999), up_y + d * 0.0, up_x + d * 0.0)
    t = normalize(cross(up, d))
    t = normalize(t - n * dot(n, t)[..., None])
    return t, normalize(cross(n, t))


# --------------------------------------------------------------------------
# Roughness
# --------------------------------------------------------------------------
def char_aniso_roughness(gloss, lobe_radius, on):
    """S_ANISOTROPIC_GLOSS channel map + the specular-AA floor.

    r5400_m0:516 samples `g_tAnisoGloss` and takes `.xy`; the ISOTROPIC
    sibling module reads `.zz` from the SAME slot
    (`csgo_character_ps_r0_m0.metal:473`, `_19372.zz`), and that one line is
    what pins the channel map:

        .x = perceptual roughness along T
        .y = perceptual roughness along B
        .z = the isotropic roughness, used when S_ANISOTROPIC_GLOSS = 0

    THERE IS NO ROTATION CHANNEL. `gt_aniso_gloss()` (the csgo_complex form)
    rotates the tangent by the map's R and scales anisotropy by its G; the
    two are kept separate rather than one being "fixed" to the other,
    because they are different shaders.

        rough2 = max(rough2, lightSourceRadius)          r5400_m0:1347
    """
    aniso = stack([gloss[..., 0], gloss[..., 1]])
    iso = stack([gloss[..., 2], gloss[..., 2]])
    r2 = iso + (aniso - iso) * on[..., None]
    return clip(maximum(r2, lobe_radius[..., None]), 0.0, 1.0)


# --------------------------------------------------------------------------
# The four specular lobes
# --------------------------------------------------------------------------
def char_ggx_aniso(r2, tx, bx, n, l, v, f0):
    """Anisotropic GGX + Smith-Schlick, as shipped. r5400_m0:1358-1367, 1380-1381, 1411.

        a2 = rough2 * rough2                                       :1360
        S  = (T.H/a2.x)^2 + (B.H/a2.y)^2 + (N.H)^2                 :1361-1362
        k  = (max(rx,ry) + 1)^2 * 0.125                            :1364-1365
        DV = 1 / (S^2 * a2.x * a2.y * 4*(NdotL(1-k)+k)*(NdotV(1-k)+k))  :1367
        F  = F0 + (1-F0)*pow(max(1e-6, 1 - max(0, L.H)), 5)        :1380-1381
        lobe = F * DV * NdotL                                      :1411

    The pi is folded out on BOTH the specular and the diffuse, which is
    Valve's convention; keeping it here would double-count against the rest
    of this renderer, so it is kept out.
    """
    h = normalize(l + v)
    a2 = r2 * r2
    th = dot(tx, h) / maximum(a2[..., 0], 1e-8)
    bh = dot(bx, h) / maximum(a2[..., 1], 1e-8)
    nh = dot(n, h)
    s = th * th + bh * bh + nh * nh
    k = (maximum(r2[..., 0], r2[..., 1]) + 1.0) ** 2 * 0.125
    ndl = maximum(dot(n, l), 0.0)
    ndv = maximum(dot(n, v), 0.0)
    dv = 1.0 / maximum(s * s * a2[..., 0] * a2[..., 1] * 4.0
                       * (ndl * (1 - k) + k) * (ndv * (1 - k) + k), 1e-9)
    lh = maximum(dot(l, h), 0.0)
    f = f0 + (1.0 - f0) * maximum(1.0 - lh, 1e-6)[..., None] ** 5
    return f * (dv * ndl)[..., None]


def char_iso_ggx(rough, n, l, v, f0):
    """The ISOTROPIC baseline the renderer's own chain already computes,
    written in csgo_character's OWN convention so the delta cancels exactly
    when every csgo_character axis is off. Same expression as
    `char_ggx_aniso` with alpha_x == alpha_y == rough.
    """
    r2 = stack([rough, rough])
    ax = normalize(cross(n, v) + 1e-6)
    bx = normalize(cross(n, ax))
    return char_ggx_aniso(r2, ax, bx, n, l, v, f0)


def char_hair_lobe(r2, tx, bx, n, l, v, f0_hair, shift, rough_scale):
    """S_ANISOTROPIC_HAIR: a SECOND aniso GGX with a Kajiya-Kay tangent
    shift. r5400_m0:1382-1402.

        if (rx >= ry): x = dot(H,T');  y = dot(H, normalize(B' + N*shift))
        else:          x = dot(H, normalize(T' + N*shift));  y = dot(H,B')
        z  = dot(H,N)
        rh = clamp(rough2 * g_flHairGlossRoughnessScale, 0, 1)      :1395
        S2 = x^2/rh.x^4 + y^2/rh.y^4 + z^2                          :1398-1399
        k2 = (max(rh) + 1)^2 * 0.125                                :1400-1401

    The shift lands on whichever axis carries the SMALLER roughness. The
    branch at :1385 is `step(_15596.y, _15596.x)` over
    `_15596 = r.xy / r.yx` (:1382), i.e. `rx/ry >= ry/rx`, i.e. `rx >= ry`;
    its TRUE arm shifts `_14964`, and `_14964` is the axis paired with
    `a2.y` at :1361. Transcribed from the branch, not inferred from what a
    hair shader ought to do.
    """
    h = normalize(l + v)
    swap = (r2[..., 0] >= r2[..., 1])
    tsh = normalize(tx + n * shift[..., None] + 1e-6)
    bsh = normalize(bx + n * shift[..., None] + 1e-6)
    xp_ = where(swap, dot(h, tx), dot(h, tsh))
    yp = where(swap, dot(h, bsh), dot(h, bx))
    zp = dot(h, n)
    rh = clip(r2 * rough_scale[..., None], 1e-3, 1.0)
    s2 = (xp_ * xp_ / maximum(rh[..., 0] ** 4, 1e-9)
          + yp * yp / maximum(rh[..., 1] ** 4, 1e-9) + zp * zp)
    k2 = (maximum(rh[..., 0], rh[..., 1]) + 1.0) ** 2 * 0.125
    ndl = maximum(dot(n, l), 0.0)
    ndv = maximum(dot(n, v), 0.0)
    dv2 = 1.0 / maximum(s2 * s2 * rh[..., 0] ** 2 * rh[..., 1] ** 2 * 4.0
                        * (ndl * (1 - k2) + k2) * (ndv * (1 - k2) + k2), 1e-9)
    lh = maximum(dot(l, h), 0.0)
    f = f0_hair + (1.0 - f0_hair) * maximum(1.0 - lh, 1e-6)[..., None] ** 5
    return f * (dv2 * ndl)[..., None]


def char_cloth_lobe(r2, n, l, v):
    """F_CLOTH_SHADING: the Estevez-Kulla "Charlie" sheen NDF with Ashikhmin
    visibility. r5400_m0:1369-1374 (again at 1795, and at 2313).

        a  = max(dot(rough2, (0.5,0.5))^2, 1e-05)                   :1371-1372
        charlie = ((2 + 1/a) * pow(1 - (N.H)^2, 0.5/a))
                  / (NdotL + NdotV' - NdotL*NdotV')
                  * 0.124999918043613433837890625 * NdotL           :1374

    where NdotV' = clamp(NdotV + 0.001, 0, 1) (:1373). The trailing constant
    is the shipped literal `0.124999918043613433837890625`, which is NOT
    1/8 -- it is 1/8 minus 8.2e-8, and it is transcribed rather than
    rounded because rounding it is a fitted float by another name.
    """
    h = normalize(l + v)
    a = maximum(dot(r2, const_vec(r2, [0.5, 0.5])) ** 2, 1e-5)
    nh = dot(n, h)
    ndl = maximum(dot(n, l), 0.0)
    ndv = clip(dot(n, v) + 0.001, 0.0, 1.0)
    d = (2.0 + 1.0 / a) * maximum(1.0 - nh * nh, 0.0) ** (0.5 / a)
    return (d / maximum(ndl + ndv - ndl * ndv, 1e-6)
            * 0.124999918043613433837890625 * ndl)


def char_retro_lobe(r2, n, l, v, f0):
    """S_RETRO_REFLECTIVE. r5400_m0:1404-1411.

        Hr = normalize(L - reflect(V, N))       # Hr == N when L == V   :1404
        a  = rough2.x^2                         # ONLY the x axis        :1406
        D  = a / ((Hr.N)^2 * (a^2 - 1) + 1)     # then SQUARED           :1407
        k  = (rough2.x + 1)^2 * 0.125                                   :1408-1409
        retro = (F0 + (1-F0)*pow(max(0, dot(L,Hr)), 5))
                * (D^2 / (4*(NdotL(1-k)+k)*(NdotV(1-k)+k))) * NdotL     :1411

    MSL's `reflect(I, N)` is `I - 2*dot(N,I)*N`, so the expanded form below
    is the same vector, not a re-derivation.

    A GGX lobe peaked at L == V rather than at the mirror direction, which
    is what makes it retro-reflective. It reads only `rough2.x`; feeding it
    the anisotropic pair is a defect this transcription does not introduce.
    """
    refl = v - 2.0 * dot(n, v)[..., None] * n
    hr = normalize(l - refl)
    a = r2[..., 0] ** 2
    d = a / maximum(dot(hr, n) ** 2 * (a * a - 1.0) + 1.0, 1e-9)
    k = (r2[..., 0] + 1.0) ** 2 * 0.125
    ndl = maximum(dot(n, l), 0.0)
    ndv = maximum(dot(n, v), 0.0)
    lhr = maximum(dot(l, hr), 0.0)
    f = f0 + (1.0 - f0) * (lhr ** 5)[..., None]
    return f * ((d * d / maximum(4.0 * (ndl * (1 - k) + k)
                                 * (ndv * (1 - k) + k), 1e-9)) * ndl)[..., None]


# --------------------------------------------------------------------------
# Subsurface scattering
# --------------------------------------------------------------------------
def char_sss_lut_uv(n_tex, l, curvature, curvature_scale):
    """The pre-integrated-skin LUT coordinate. r1050_m0:929, :941.

        u = fma(dot(N_tex, L), 0.5, 0.5)                              :941
        v = curvature * g_flCurvatureScale * 0.3976777493953704833984375
            + (-0.0101010091602802276611328125)                       :929

    Two things a reader should not assume. The u axis is driven by the RAW
    normal-mapped normal, not the shading normal the diffuse uses -- that is
    what makes the lookup a function of the high-frequency surface rather
    than of the smoothed one. And THE UV IS NOT CLAMPED IN THE SHADER: it
    relies on the sampler's clamp mode, so a curvature outside the baked
    range reads the edge texel rather than wrapping, and reproducing this
    with a repeat-mode sampler is a different image.

    The 0.39767.../-0.01010... pair is the shipped literal, not 0.4/-0.01.
    """
    v = (curvature * curvature_scale * 0.3976777493953704833984375
         + (-0.0101010091602802276611328125))
    u = dot(n_tex, l) * 0.5 + 0.5
    return stack([u, v])


def char_sss_bent(n_shade, n_tex, l, bleed):
    """S_SUBSURFACE_SCATTERING: the three-channel bent-normal N.L.
    r1050_m0:930-940 (sun) and :1342-1346 (per punctual light).

        ndl = dot(N_tex, L)                                           :931
        w   = clamp(1 - ndl, 0, 1);  w2 = w*w                         :932, :934
        for c in {r,g,b}:
            n_c = normalize(mix(N_shading, N_tex, bleed[c] + w2*(1-bleed[c])))
            d_c = clamp(dot(n_c, L), 0, 1)                            :940

    THREE SEPARATE NORMALS, one per channel, each a different blend between
    the shading normal and the texture normal -- that is the scattering
    model, and collapsing it to one normal times a colour is a different
    (and much cheaper) shader. The blend weight per channel rises toward the
    terminator, because `w2` is largest where `ndl` is smallest.

    `bleed` is the float3 RGB scattering width at a material slot the
    stripped corpus does not name (`_5618._m17`). The caller supplies it;
    gpu_render.py supplies (1.0, 0.3, 0.2), the standard red-deepest
    ordering, and prints that substitution at runtime. It is an INPUT, not a
    constant of this expression, which is why it is a parameter here and can
    be spanned by a generator.
    """
    ndl = dot(n_tex, l)
    w = clip(1.0 - ndl, 0.0, 1.0)
    w2 = w * w
    d = []
    for c in range(3):
        bc = bleed[..., c]
        k = (bc + w2 * (1.0 - bc))[..., None]
        n_c = normalize(n_shade + (n_tex - n_shade) * k)
        d.append(clip(dot(n_c, l), 0.0, 1.0))
    return stack(d)


def char_sss_compose(d3, lut, ndl_shade, mask):
    """The second half of r1050_m0:941.

        sss     = clamp(d3 + (lut.xyz * 0.5 - 0.25), 0, 1)
        diffuse = mix(max(0, dot(N_shading, L)), sss, sssMaskTex.y)

    Two things this REFUTES about the obvious guess, both read from the
    bytecode: the pre-integrated lookup does NOT replace the diffuse -- it
    is remapped to +/-0.25 and ADDED to a three-channel bent-normal N.L --
    and it is per-light, not a single ambient wrap.
    """
    sss = clip(d3 + (lut[..., :3] * 0.5 - 0.25), 0.0, 1.0)
    base = maximum(ndl_shade, 0.0)[..., None] + d3 * 0.0
    m = mask[..., None]
    return base + (sss - base) * m


# --------------------------------------------------------------------------
# The two extra probe-volume binner pairs
# --------------------------------------------------------------------------
def char_probe_slice_offsets(dirv):
    """`mix((0, 1/6, 1/3), (0.5, 2/3, 5/6), step(dir, 0))`. r5400_m0:3636.

    Sign-selected slice offsets into a packed-Z volume: six slices, one per
    signed axis, addressed by which side of zero the direction is on.
    """
    neg = step(dirv, 0.0)
    lo = const_vec(dirv, [0.0, 0.16666667163372039794921875,
                          0.3333333432674407958984375])
    hi = const_vec(dirv, [0.5, 0.666666686534881591796875,
                          0.833333313465118408203125])
    return lo + (hi - lo) * neg


def char_probe_ambient_cube(dirv, cx, cy, cz):
    """The csgo_character probe reconstruction: a Valve AMBIENT CUBE packed
    in Z, NOT a trilinear interpolation. r5400_m0:3636-3653.

        d2  = dir*dir
        irr = cX.rgb*d2.x + cY.rgb*d2.y + cZ.rgb*d2.z

    Callers supply the three slice fetches, addressed by
    `char_probe_slice_offsets`. This is the body of the TWO EXTRA binner
    pairs csgo_character has beyond the light and cubemap pairs every world
    family has -- 146 FLOPs and 3 sampler3D each, evaluated once along +V
    and once along -V inside the distance-contrast block.
    """
    d2 = dirv * dirv
    return cx * d2[..., 0:1] + cy * d2[..., 1:2] + cz * d2[..., 2:3]


# --------------------------------------------------------------------------
# Output-side terms
# --------------------------------------------------------------------------
def char_alpha_test(a, alpha_fw, uv_fw, alpha_ref, aa_strength):
    """S_ALPHA_TEST with g_flAntiAliasedEdgeStrength. r3000_m0:503-512.

        aa = clamp(0.5 + (a - ref) / max(fwidth(a), 1e-06), 0, 1)     :506
        t  = mix(aa, a, mix(1, clamp(4*length(fwidth(uv)), 0, 1),
                            g_flAntiAliasedEdgeStrength))             :506
        discard where (t - 0.001) < 0                                 :512

    TWO DIFFERENT DERIVATIVES, and conflating them was a defect this file
    fixes. The DIVISOR is `fwidth(alpha)` -- how fast the alpha channel
    itself changes across the pixel, which is what turns a hard test into a
    one-pixel ramp. `4*length(fwidth(uv))` appears in the SECOND `mix`, as
    the BLEND WEIGHT between the antialiased and the raw alpha, and is a
    different quantity with a different clamp. The lifted-from version
    divided by the uv derivative; on a surface whose alpha is flat but whose
    UV is compressed (any distant or grazing draw) that is the wrong ramp
    width, and at `aa_strength = 0` the two forms disagree outright.

    Both derivatives are now the caller's to supply, so a renderer that has
    only one of them substitutes visibly at the call site instead of
    silently inside the term. The `else` arm (the AA feature off) is
    `t = a - ref` (:510), and the discard is the caller's -- this returns
    the tested alpha.
    """
    aa = clip(0.5 + (a - alpha_ref) / maximum(alpha_fw, 1e-6), 0.0, 1.0)
    k = 1.0 + (clip(4.0 * uv_fw, 0.0, 1.0) - 1.0) * aa_strength
    return aa + (a - aa) * k


def char_invuln_exponent(t_now):
    """`mix(3, 6, 1 + sin(time*20)*0.5)`. r5400_m0:3958.

    A python float in, a python float out: `t_now` is per-frame, not
    per-pixel, so this is the one part of the term that is not an array.
    Note the mix weight EXCEEDS 1 for half the cycle (it spans 0.5..1.5),
    so the exponent ranges 4.5..7.5 and never actually reaches 3 -- reading
    `mix(3, 6, ...)` as "between 3 and 6" is wrong, and the extrapolation is
    what gives the effect its hard pulse.
    """
    return 3.0 + 3.0 * (1.0 + math.sin(t_now * 20.0) * 0.5)


def char_invulnerability(rgb, n, v, noise, amount, color, exponent):
    """g_flSpawnInvulnerability. r5400_m0:3955-3958, the last operation
    before the write.

        f   = 1 - clamp(dot(V, N), 0, 1)                             :3957
        rgb = mix(rgb, g_cInvulnerabilityColor
                       * (mix(luma(rgb), 0.5, 0.5)
                          + 4*pow(mix(f*noise, 1, f),
                                  mix(3, 6, 1 + sin(time*20)*0.5))),
                  g_flSpawnInvulnerability)                          :3958

    THE LUMA WEIGHTS ARE VALVE'S, NOT THE ITU ONES. :3958 reads
    `(0.2125000059604644775390625, 0.7153999805450439453125,
    0.07209999859333038330078125)` -- the same triple csgo_eyeball uses --
    and the lifted-from version had `(0.2126, 0.7152, 0.0722)`, which is the
    Rec.709 rounding the REST of gpu_render.py uses. It is a 1e-4 error on a
    coefficient that is READ, so it is corrected here rather than kept for
    consistency with the renderer's other families.

    `color` is a parameter, not a constant, because this module broadcasts a
    SCALAR uniform (`float3(_5618._m48)`) into the colour slot while the
    material table carries three channels; which is right is a question
    about the material, and the caller answers it.

    `exponent` is `char_invuln_exponent(t_now)`, kept out of this term
    because it is per-frame rather than per-pixel.
    """
    f = 1.0 - clip(dot(v, n), 0.0, 1.0)
    lum = dot(rgb, const_vec(rgb, [0.2125000059604644775390625,
                                   0.7153999805450439453125,
                                   0.07209999859333038330078125]))[..., None]
    base = maximum(f * noise + (1.0 - f * noise) * f, 1e-6)
    val = (lum * 0.5 + 0.25) + 4.0 * base[..., None] ** exponent[..., None]
    return rgb + (color * val - rgb) * amount[..., None]


def char_mboit_z(zlin, near, far):
    """MBOIT's warped depth. r5145_m4:1979 (identical at r5145_m12:1978).

        z = ((log(depth) - logNear) / (logFar - logNear)) * 2 - 1

    `logNear`/`logFar` arrive at the shader ALREADY LOGGED, as uniform
    members `_5037._m22` and `_m23`; this takes linear near/far and logs
    them, which is the same number by a different route and is stated
    because the two are easy to conflate. `depth` in the reference is
    `dot(cameraForward, P)` -- a view-space depth along the forward axis,
    not a distance.
    """
    ln = log(maximum(near, 1e-6))
    lf = log(maximum(far, maximum(near, 1e-6) * 1.0000001 + 1e-6))
    return ((log(maximum(zlin, 1e-6)) - ln) / (lf - ln)) * 2.0 - 1.0


def char_mboit_absorbance(alpha):
    """MBOIT's optical depth. r5145_m4:1980, r5145_m12:1979.

        a = -log(1 - clamp(alpha, 1e-05, 0.99989998340606689453125))

    BOTH bounds are shipped float32 literals and neither is its decimal
    idealisation. The lower one is 9.9999997473787516355514526367188e-06,
    which is float32(1e-05) and is NOT the double 1e-05 -- writing the tidy
    form put a 2.5e-13 error into the absorbance at alpha = 0, and the
    conformance A/B is what found it. The upper one is
    0.99989998340606689453125: at alpha = 1 the log diverges, so that bound
    sets exactly how much optical depth one fully-opaque blended fragment
    may contribute. csgo_character's literals were compared against
    csgo_glass's rather than assumed to match; they do.
    """
    return -log(1.0 - clip(alpha, 9.9999997473787516355514526367188e-06,
                0.99989998340606689453125))


def char_mboit_moments4(alpha, zlin, near, far):
    """D_MBOIT_PASS1, four moments. r5145_m12:1978-1982.

        out1 = (z, z^2, z^3, z^4) * a

    One colour target of four moments, plus the absorbance in a separate
    single-channel target (:1981).
    """
    z = char_mboit_z(zlin, near, far)
    a = char_mboit_absorbance(alpha)
    z2 = z * z
    return stack([z, z2, z2 * z, z2 * z2]) * a[..., None]


def char_mboit_moments6(alpha, zlin, near, far):
    """D_MBOIT_PASS1, six moments. r5145_m4:1979-1985.

        out1 = (z, z^2) * a                                            :1984
        out2 = (z^3, z^4, z^5, z^6) * a                                :1985

    Returned CONCATENATED as six, because the split into a 2-channel and a
    4-channel target is a render-target packing decision and not part of the
    arithmetic. The six-moment record is r5145_m4 and the four-moment one is
    r5145_m12 -- checked by reading both files, since the moment count is
    exactly the kind of thing that gets swapped between two near-identical
    modules.
    """
    z = char_mboit_z(zlin, near, far)
    a = char_mboit_absorbance(alpha)
    z2 = z * z
    z4 = z2 * z2
    return stack([z, z2, z2 * z, z4, z4 * z, z4 * z2]) * a[..., None]
