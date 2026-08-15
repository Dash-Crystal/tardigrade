#!/usr/bin/env python3
"""`csgo_customglove` kernels — the paint-kit MATERIAL COMPOSITOR.

Reference:
`docs/projects/counter-strike-sft/decompiled/csgo_customglove_ps_r{0,1,2,3}_m0.glsl`
(committed a3fefe94; these four ARE GLSL -- spirv-cross accepts this
family's block layout, unlike csgo_character's and csgo_eyeball's).

THIS IS NOT A LIT SURFACE SHADER, and every term below has to be read in
that light. It renders to a texture atlas, not to the framebuffer;
`g_nOutputMode` selects which map is being baked (Albedo / Normal /
Metalness / AO / Roughness). Across all four records there are ZERO cube,
3D or array samplers, ZERO SSBOs, no `reflect`, no `refract`, no
`gl_FragCoord`, no `gl_FrontFacing`, and the only uniform block in the
entire shader is the material CB -- so there is no camera, no view matrix,
no light list, no SH and no time. The lighting happens later, in the
surface shaders that sample the baked atlas. Full enumeration:
`SHADER_CALLFLOW_csgo_customglove.md`.

`r3_m0` at 3,769 FLOPs/px and 67 static fetch sites is the SHIPPING
MAXIMUM: the PS declares exactly three static axes and no tools axis, so
nothing here is discounted for being a tools path.
"""
from __future__ import annotations

from ._chargl import (clip, const_vec, dot, maximum, minimum, normalize,
                      stack, where)


def glove_aniso_split(a):
    """S_ANISOTROPIC_GLOSS. r2_m0.glsl:213-231.

        aniso = mix(mix(mix(l1, l2, wR), l3, wG), l4, wB)             :213
        scaleT = (aniso < 0) ? 1 + aniso : 1                          :215-222
        scaleB = (aniso > 0) ? 1 - aniso : 1                          :224-231

    ONE SIGNED SCALAR, and one of the two multipliers is always exactly 1.
    That is a THIRD anisotropy encoding, distinct from both of the others in
    this renderer: `gt_aniso_gloss()` (csgo_complex) is a symmetric split
    with a rotation channel, and `char_aniso_roughness()` (csgo_character)
    is two independent roughness CHANNELS in a texture with no rotation at
    all. Three families, three encodings, each read from its own bytecode;
    none is "the" anisotropic gloss and none has been made to agree with the
    others.

    It is applied to the BASE roughness only (:750 in r2), before the damage
    and pattern overrides -- both of which write an isotropic `vec2(scalar)`
    and therefore erase the anisotropy wherever they dominate.
    """
    one = a * 0.0 + 1.0
    return where(a < 0.0, one + a, one), where(a > 0.0, one - a, one)


def glove_layer_weights(mask_rgb):
    """The four RENORMALISED layer weights. r3_m0.glsl:235-237.

        sum = (b + g) + r                                             :235
        w4  = max(0, 1 - sum)                                         :236
        w   = vec4(r, g, b, w4) / (sum + w4)                          :237

    The fourth layer is the REMAINDER of the first three, and then all four
    are divided by their own total -- so the weights sum to 1 even where the
    mask does not, and an over-bright mask desaturates rather than
    over-composites. This is the BC=1 ("Source2 Height Blending") form; the
    BC=0 records blend three layers with a plain nested `mix` and no
    renormalisation at all (r0_m0.glsl), which is a different shader rather
    than a flag.
    """
    r = mask_rgb[..., 0]
    g = mask_rgb[..., 1]
    b = mask_rgb[..., 2]
    s = (b + g) + r
    w4 = maximum(1.0 - s, 0.0)
    return stack([r, g, b, w4]) / (s + w4)[..., None]


def glove_id_coverage(onehot):
    """Per-id 3x3 coverage. r3_m0.glsl:281-322.

        if   (c[4] == c[1] && c[4] == c[7]) w = c[4]                  :281-292
        elif (c[4] == c[3] && c[4] == c[5]) w = c[4]                  :297-308
        else w = 0.03125*(c[0]+c[2]+c[6]+c[8])
               + 0.09375*(c[2]+c[3]+c[5]+c[7])
               + 0.5    * c[4]                                        :312

    Tap order is `_1555` (r3_m0.glsl:17):
    (-1,-1)(0,-1)(1,-1)(-1,0)(0,0)(1,0)(-1,1)(0,1)(1,1); tap 4 is the centre.
    Two interior early-outs -- a vertical and a horizontal run through the
    centre skip the filter entirely -- then a normalised tent
    (4*0.03125 + 4*0.09375 + 0.5 = 1.0).

    AS SHIPPED, THE EDGE TERM USES INDEX [2] -- A CORNER -- AND INDEX [1] IS
    NEVER REFERENCED. That is what is in the SPIR-V. It is transcribed, not
    corrected: "fixing" it would make this renderer disagree with the engine
    on every id boundary, which is the opposite of the goal.

    `onehot` is (..., 8, 9): 8 ids x 9 taps, each 0 or 1.
    """
    c = onehot
    vert = (c[..., 4] == c[..., 1]) & (c[..., 4] == c[..., 7])
    horiz = (c[..., 4] == c[..., 3]) & (c[..., 4] == c[..., 5])
    tent = (0.03125 * (c[..., 0] + c[..., 2] + c[..., 6] + c[..., 8])
            + 0.09375 * (c[..., 2] + c[..., 3] + c[..., 5] + c[..., 7])
            + 0.5 * c[..., 4])
    return where(vert | horiz, c[..., 4] * 1.0, tent)


def glove_tint_composite(tint_rgba, coverage):
    """The 8-way tint composite. r3_m0.glsl:483-503.

        acc  = sum_i vec4(C_i.xyz * C_i.w, C_i.w) * coverage[i]       :483-484
        tint = (acc.w > 0) ? vec4(acc.xyz / acc.w, acc.w)             :487-495
                           : vec4(1, 1, 1, acc.w)                     :497-503

    PREMULTIPLIED sum, then UN-premultiplied. Summing the colours directly
    would let a low-alpha id drag the hue; this weights each id's colour by
    its own alpha and divides the alpha back out, so a fully-covered pixel
    reproduces its id's colour exactly.

    The fallback at zero total alpha is WHITE, not black and not the first
    id -- which matters because that is the value every uncovered texel of
    the atlas bakes to.

    `tint_rgba` is (..., 8, 4), `coverage` is (..., 8). Returns the RGB;
    the alpha is `glove_tint_alpha` and is carried through unchanged from
    the accumulator (:486), so the two are separate reads of one sum.
    """
    w = tint_rgba[..., 3]
    acc_rgb = (tint_rgba[..., :3] * w[..., None] * coverage[..., None]).sum(-2)
    acc_a = (w * coverage).sum(-1)
    ones = acc_rgb * 0.0 + 1.0
    return where((acc_a > 0.0)[..., None],
                 acc_rgb / maximum(acc_a, 1e-30)[..., None], ones)


def glove_tint_alpha(tint_rgba, coverage):
    """The tint accumulator's alpha. r3_m0.glsl:483-486.

        acc.w = sum_i C_i.w * coverage[i]

    It survives BOTH arms of the :487 branch untouched -- the zero-alpha
    fallback replaces only the RGB with white -- so the composite's alpha is
    this sum and nothing else.
    """
    return (tint_rgba[..., 3] * coverage).sum(-1)


def glove_octahedral_encode(n_tbn):
    """The baked normal's DIAGONAL octahedral encode. r3_m0.glsl:1943-1946.

        o   = vec3(dot(N,T), dot(N,B), N.z) / L1(that vec3)            :1943
        enc = (vec2(o.x + o.y, o.x - o.y) * 0.5) + 0.5                 :1946

    The inverse of `char_normal_decode`'s `(r+g)-256/255, r-g` -- the same
    diagonal basis, which is why the character family can decode two
    channels and rebuild the third. The divisor is the L1 norm
    (`|x| + |y| + |z|`), not the L2 norm, and using L2 here is the classic
    way to get an encode that round-trips at the poles and nowhere else.

    `n_tbn` is `vec3(dot(N,T), dot(N,B), N.z)` AS ONE VECTOR, which is how
    the reference writes it and which is not cosmetic: the argument is a
    unit normal projected onto a basis, so `|x| + |y| + |z| >= 1` always and
    the divide cannot be zero. Taking the three components as independent
    scalars admits (0,0,0) -- and the conformance sample, which spans its
    corners deliberately, produced exactly that and refused the term with
    REF_NAN until the signature said what the input is.
    """
    l1 = (abs(n_tbn[..., 0]) + abs(n_tbn[..., 1])) + abs(n_tbn[..., 2])
    o = n_tbn / l1[..., None]
    return stack([o[..., 0] + o[..., 1], o[..., 0] - o[..., 1]]) * 0.5 + 0.5


def glove_srgb_to_linear(x):
    """The output-side transfer. r3_m0.glsl:1948-1976 and the four repeats.

        lo = x * 0.077399380505084991455078125          # 1/12.92
        hi = pow(x * 0.947867333889007568359375         # 1/1.055
                 + 0.052132703363895416259765625, 2.4)  # 0.055/1.055
        out = (x <= 0.040449999272823333740234375) ? lo : hi

    APPLIED ON THE WAY OUT, so that an sRGB render target re-encodes the
    data unchanged and the atlas stores exactly what was computed. Output
    mode 0 (Albedo) correctly SKIPS it -- the RT does the decode there --
    and the four data modes (Normal, Metalness/AO, AO, Roughness) apply it.
    Reading this as "the shader gamma-corrects its output" inverts what it
    is for.
    """
    lo = x * 0.077399380505084991455078125
    hi = (x * 0.947867333889007568359375
          + 0.052132703363895416259765625) ** 2.4
    return where(x <= 0.040449999272823333740234375, lo, hi)


def glove_roughness_backcompat(surface_rough_xz, aniso_scale, aniso_amount):
    """The `S_BACKWARDS_COMPATIBILITY = 1` roughness pair.
    r3_m0.glsl:1155-1156 (again at :1765-1767).

        rough   = surfaceNormalTex.xz * anisoScale                    :1155
        rough.y = mix(rough.x, rough.y, anisoAmount)                  :1156

    The BC=1 branch does NOT fork the bytecode on `S_ANISOTROPIC_GLOSS` --
    it interpolates. `anisoAmount = 0` collapses the pair to isotropic;
    `= 1` passes the second channel through untouched. Both scalars are
    4-layer accumulations of adjacent per-layer CB slots (:974-975 for
    layer 4, :877-878 for layer 3, :693-694 for the layer-1 seed).

    `.x` and `.z` of the surface normal texture are THE ROUGHNESS PAIR --
    `.w` and `.y` are the two normal channels (:416-422). Reading .xy here
    would take one normal channel as a roughness.
    """
    r = surface_rough_xz * aniso_scale[..., None]
    return stack([r[..., 0], r[..., 0] + (r[..., 1] - r[..., 0])
                  * aniso_amount])
