#!/usr/bin/env python3
"""`csgo_glass` pixel stage.

SCOPE, STATED BEFORE ANY EXPRESSION
-----------------------------------
`docs/projects/counter-strike-sft/csgo_glass_ps.glsl` is ONE decompiled
module — combo `r0_m3`, static 0 / dynamic 8, i.e. plain transmissive with
`D_BAKED_LIGHTING_FROM_PROBE`. It is not the union of the family. Verified
against the artifact: `refract` occurs 0 times (so
`S_OPAQUE_CUBEMAP_REFRACTION` is 0 here) and `MBOIT` occurs 0 times, the
shader declares exactly one colour output at :206 and writes it once at
:1047, and there is no `imageStore` and no `early_fragment_tests`.

So the MBOIT WRITE SIDE IS NOT IN THIS FILE. w1-storage-tables has since
identified both modules BY CONTENT — by their colour-attachment count
across the 64 that decompile — rather than by combo arithmetic:

    r*_m4..m7    3 attachments  ->  MBOIT PASS1 (moment accumulation)
    r*_m12..m15  2 attachments  ->  MBOIT PASS2 (resolve)
    r*_m0..m3, m8..m11   1 attachment  ->  plain transmissive (this file)

MY OWN GUESS WAS HALF WRONG AND IS CORRECTED HERE: I had said PASS2 was
r0_m8. It is r0_m12; r0_m8 has ONE attachment and is a plain transmissive
module. PASS1 = r0_m4 was right. Both bodies are committed at w1's
e47e79ae with their offset tables. Neither uses `imageStore` — MBOIT here
is plain MRT, not UAV writes — so the blendStateDesc is the one remaining
write-side gap.

WHAT MAY NOT BE CITED FROM THE CALLFLOW DOC
-------------------------------------------
`csgo_glass_ps.glsl` contains ZERO `offset` decorations (the substring does
not occur). Every byte-offset and material-variable-name claim in
SHADER_CALLFLOW_csgo_glass.md is therefore unverifiable from any artifact
in this repo, and three of them are CONTRADICTED by the file's own
`layout(set=, binding=)` decorations:

  * doc :355 says the 7 probe `sampler3D` taps resolve to set 1 / binding 2;
    the file declares its images at set 4 / binding 46 (:193, :195, :197,
    :198) and has no set-1/binding-2 image at all.
  * doc :356/:376 says the 2 `sampler3D` at :858/:867 are set 3 / binding 0;
    no such image declaration exists.
  * doc :372-374 says the probe volume list is the set-3/binding-31 SSBO;
    that SSBO (`_5570`, :188) is read ONLY in the light loop (:702, :726,
    :735-919). The probe loop indexes `_5538._m7._m0[...]` (UBO `_939`,
    set 3, :174) and `_3705._m0._m0[...]` (UBO `_1611`, set 1, :186).

Consequently every uniform below is addressed by its DECOMPILED ACCESSOR
PATH as an opaque handle, never by a byte offset and never by a name. The
re-decompile with offset decorations is owned by w1-storage-tables.

BRDF DIVERGENCE FROM `cables` — DO NOT SHARE AN IMPLEMENTATION
--------------------------------------------------------------
Glass's direct specular (:670, :918) is GGX-D over
`denom^2 * LdotH^2 * (4r+2)` with NO Fresnel term; cables (:692, :990) is
Schlick-F x D^2 x Smith-Schlick-GGX with `k = (r+1)^2/8`. Numerically
distinct. Glass's diffuse accumulates `max(0, -N.L)` — BACK-lit, which is
the whole point of the transmissive family; cables accumulates
`max(0, +N.L)`. Sign-flipped. Two families, two implementations.
"""
from __future__ import annotations

import numpy as np

from ..post._backend import clip, maximum, where

MBOIT_MODULES = (
    "csgo_glass's MBOIT write side is PASS1 = r0_m4 (3 colour attachments, "
    "locations 0/1/2) and PASS2 = r0_m12 (2 attachments, locations 0/1), "
    "identified by attachment count rather than combo arithmetic and "
    "committed by w1-storage-tables at e47e79ae with their offset tables. "
    "My earlier guess of r0_m8 for PASS2 was WRONG -- r0_m8 has one "
    "attachment and is plain transmissive. Neither pass uses imageStore, "
    "so MBOIT here is plain MRT rather than UAV writes, and the "
    "blendStateDesc is the one remaining gap on the write side. This "
    "module implements the transmissive path (r0_m3) only.")

# The Rec.709 weights the shader writes into its alpha channel. Note the
# literals: 0.2125, 0.7154, 0.0721 -- the shader's own values, which are
# NOT the 0.2126/0.7152/0.0722 used by the post chain's exposure meter.
# Transcribed as written; the difference is the shader's, not ours.
GLASS_LUMA = (0.2125000059604644775390625,
              0.7153999805450439453125,
              0.07209999859333038330078125)

# 256/255, exactly. The normal decode's bias is a byte-range constant, not
# the 1.0 a `2t-1` decode would use.
NORMAL_DECODE_BIAS = 1.00392162799835205078125

# Approximately 1/3 but NOT 1/3. Transcribed to the bit.
SPEC_AA_EXPONENT = 0.333000004291534423828125

# Schlick with F0 = 0.04 and exponent 5, then COMPLEMENTED: this is the
# transmission gate, not a reflectance.
FRESNEL_F0 = 0.039999999105930328369140625

# 4/9. The absorption path length's obliquity factor.
ABSORPTION_OBLIQUITY = 0.4444444477558135986328125
ABSORPTION_MIN_COS = 0.00999999977648258209228515625

# Second-order tint exponent.
TINT_BOUNCE_EXPONENT = 1.60000002384185791015625

# The horizon clamp's epsilon and its length ceiling.
HORIZON_EPS = 0.001000000047497451305389404296875
HORIZON_MAX_LEN = 1.75


# --------------------------------------------------------------------------
# Stage B -- material decode
# --------------------------------------------------------------------------
def normal_decode_diagonal(rg):
    """csgo_glass_ps.glsl:247-251 -- the DIAGONAL two-channel decode.

    Not `2*t - 1`. The two stored channels are the rotated pair
    (a+b, a-b), so the decode is a subtraction of 256/255 on the sum and a
    plain difference on the other, with z reconstructed from the L1
    complement rather than by a sqrt.

    `rg` is (..., 2) in [0, 1]. Returns a unit (..., 3)."""
    a = (rg[..., 0] + rg[..., 1]) - NORMAL_DECODE_BIAS
    b = rg[..., 0] - rg[..., 1]
    z = (1.0 - abs(a)) - abs(b)
    n = _stack3(a, b, z)
    return n / _norm(n)


def transmission_remap(mask_a, vcolor_a, lo, hi):
    """:301 -- masked alpha remapped into the material's [lo, hi] band.

        (clamp(mask.w * vColor.w, 0, 1) * (hi - lo)) + lo

    `lo`/`hi` are `_5618._m3.x` / `_5618._m3.y`, addressed as accessor
    paths; no byte offset is asserted for them."""
    return clip(mask_a * vcolor_a, 0.0, 1.0) * (hi - lo) + lo


def albedo_tint(albedo_rgb, vcolor_rgb, amount):
    """:303 -- vertex-colour tint as a LERP WEIGHT, not a raw multiply.

        albedo.rgb * mix(vec3(1.0), vColor.rgb, vec3(_5618._m4))

    amount = 0 leaves the albedo untinted; amount = 1 is the plain
    multiply. A renderer that multiplied unconditionally would be correct
    only at the endpoint."""
    a = amount[..., None] if getattr(amount, "ndim", 0) else amount
    # GLSL `mix(x, y, a)` is defined as `x*(1-a) + y*a`. Written in that
    # form rather than the algebraically-equal fma form `1 + (y-1)*a`,
    # which measured 1.110e-16 against the transcription -- machine
    # epsilon, and still an ordering WE chose rather than the one the
    # shader specifies.
    return albedo_rgb * ((1.0 - a) + vcolor_rgb * a)


def spec_aa_roughness(rough_z, vcolor_a, ddx_n, ddy_n):
    """:309-314 -- roughness floored by GEOMETRIC-normal curvature.

        max((normal.zz * vColor.w), pow(clamp(max(|ddx|^2, |ddy|^2), 0, 1),
                                        0.333000004291534423828125))

    Two details that are easy to get wrong and are both load-bearing: the
    exponent is 0.333000004291534423828125, which is near 1/3 and not 1/3;
    and the derivatives are of the GEOMETRIC normal `_24347` (:309), not
    the mapped one. Glass also multiplies the stored roughness by
    `vColor.w`, which cables does not."""
    curv = maximum((ddx_n * ddx_n).sum(-1), (ddy_n * ddy_n).sum(-1))
    floor = clip(curv, 0.0, 1.0) ** SPEC_AA_EXPONENT
    return maximum(rough_z * vcolor_a, floor)


# --------------------------------------------------------------------------
# Stage F -- the sun BRDF (no Fresnel; this is the glass form)
# --------------------------------------------------------------------------
def ggx_specular_noF(rough, ndoth, ldoth, ndotl):
    """:641-643 + :670 -- GGX D with a combined visibility and NO Fresnel.

        a  = r*r ;  a2 = a*a
        d  = ((NdotH^2) * (a2 - 1)) + 1
        S  = a2 / ((d^2) * (LdotH^2) * ((r * 4) + 2))
        out = S * NdotL

    The `(4r + 2)` denominator is the shader's own combined D*V normaliser;
    it is not a Smith term and it is not separable into one. F0 enters this
    family as a flat broadcast (`_11894`, :245) multiplied outside, which
    is why there is no Schlick call anywhere in the module."""
    a = rough * rough
    a2 = a * a
    d = (ndoth * ndoth) * (a2 - 1.0) + 1.0
    return (a2 / ((d * d) * (ldoth * ldoth) * (rough * 4.0 + 2.0))) * ndotl


def diffuse_backlit(ambient_neg_n, ndotl_back, sun_color, shadow):
    """:669 -- the transmissive diffuse: ambient cube on -N, plus a sun
    term gated by `max(0, -dot(N, L))`.

        ambient(-N) + (max(0, -N.L) * sunColor * shadow)

    The MINUS is the family. Light arriving from behind the surface is what
    a transmissive material shows; sharing cables' `+N.L` accumulation here
    would silently make glass opaque-shaded."""
    return ambient_neg_n + (ndotl_back[..., None] * sun_color
                            * shadow[..., None])


# --------------------------------------------------------------------------
# Stage H -- IBL, transmission, absorption
# --------------------------------------------------------------------------
_ENV_A = np.array([-1.0, -0.0274999998509883880615234375,
                   -0.572000026702880859375,
                   0.02199999988079071044921875])
_ENV_B = np.array([1.0, 0.0425000004470348358154296875,
                   1.03999996185302734375,
                   -0.039999999105930328369140625])
_ENV_C = np.array([-1.03999996185302734375, 1.03999996185302734375])
_ENV_EXP2_K = -9.27999973297119140625


def _as_backend(const, like):
    """Return `const` in the same array backend/device/dtype as `like`.

    numpy passes straight through. For torch this is the difference between
    running and raising: `np.ndarray.__mul__(Tensor)` is a TypeError, not a
    broadcast, so a module-level numpy literal makes its whole expression
    torch-hostile no matter how backend-agnostic the rest of the file is.
    """
    if isinstance(like, np.ndarray):
        return const
    torch = __import__("torch")
    return torch.as_tensor(const, dtype=like.dtype, device=like.device)


def env_brdf_analytic(avg_rough, ndotv):
    """:937-943 -- the ANALYTIC split-sum approximation.

    csgo_glass has NO BRDF LUT texture. `cables` does (a sampler2DArray at
    layer literal 1.0), so the two families answer the same question with
    different machinery and must not share an implementation.

        c  = A*r + B                       (A, B the two vec4 literals)
        x  = c.x
        t  = min(x*x, exp2(-9.27999973297119140625 * max(0, N.V)))
        fg = (vec2(-1.04, 1.04) * (t*x + c.y)) + c.zw
        return vec2(fg.y, fg.x + fg.y)

    Returns (..., 2) = the (scale, bias) pair the composite lerps between
    by F0."""
    r = avg_rough[..., None]
    # THE CONSTANTS MUST FOLLOW THE BACKEND. `_ENV_A` and friends are numpy
    # arrays; `numpy_array * torch_tensor` raises TypeError outright, so
    # this function could only ever run under numpy. It was green in the
    # conformance table -- which drives it with numpy -- and threw the
    # moment the renderer called it with tensors. Every other expression in
    # this module reaches the backend through the shared `_backend` shim or
    # through duck-typed operators and was unaffected; this one holds
    # module-level literals and is the only one that needed adapting.
    A, B, C = (_as_backend(v, r) for v in (_ENV_A, _ENV_B, _ENV_C))
    c = A * r + B
    x = c[..., 0]
    t = _min(x * x, _exp2(_ENV_EXP2_K * maximum(ndotv, 0.0)))
    fg = C * (t * x + c[..., 1])[..., None] + c[..., 2:4]
    y = fg[..., 1]
    return _stack2(y, fg[..., 0] + y)


def fresnel_transmission(ndotv):
    """:958 -- Schlick, exponent 5, F0 = 0.04, COMPLEMENTED.

        1 - mix(0.04, 1.0, pow(1 - max(0, N.V), 5))

    The complement is what makes this a transmission gate rather than a
    reflectance: at grazing incidence it goes to 0 and the surface stops
    transmitting."""
    f = FRESNEL_F0 + (1.0 - FRESNEL_F0) * (1.0 - maximum(ndotv, 0.0)) ** 5.0
    return 1.0 - f


def absorption(albedo, ndotv):
    """:963 -- Beer-like absorption as an exponent on the albedo.

        albedo ^ (1 / max(0.01, sqrt(1 - (4/9)*(1 - N.V^2))))

    The exponent is one over an obliquity-corrected cosine: a ray at
    grazing incidence traverses more medium, so the albedo is raised to a
    LARGER power and darkens. The 0.01 floor bounds the exponent; without
    it the grazing limit is a divide by zero."""
    c = _sqrt(1.0 - ABSORPTION_OBLIQUITY * (1.0 - ndotv * ndotv))
    e = 1.0 / maximum(c, ABSORPTION_MIN_COS)
    return maximum(albedo, 0.0) ** e[..., None]


def tint_bounce(albedo):
    """:974 -- the second-order tint, albedo raised to 1.6.

    Light that has passed through the medium twice picks up the tint again
    but not squared; 1.6 is the shader's literal."""
    return maximum(albedo, 0.0) ** TINT_BOUNCE_EXPONENT


def horizon_clamp(irradiance):
    """:953 -- the IBL horizon term's direction/length clamp.

        normalize(v + 0.001) * clamp(length(v), 0, 1.75)

    The epsilon is inside the normalize, not added to the result: it fixes
    the direction of a near-zero vector rather than biasing a valid one."""
    v = irradiance
    d = v + HORIZON_EPS
    return d / _norm(d) * clip(_len(v), 0.0, HORIZON_MAX_LEN)[..., None]


def fog_range_height(dist, world_z, m8, m9, m10w):
    """:995-996 -- the two-axis fog factor: a PRODUCT OF POWERS in
    (range, height), not a linear ramp.

        f = clamp(m8.xy + m8.zw * vec2(length(v), worldZ), 0, 1)
        a = pow(f.x, m9.x) * pow(f.y, m9.y) * m10.w

    A start/end linear ramp cannot match a product of powers on any
    constants, which is exactly the defect class a per-term A/B surfaces
    and a whole-frame nrmse absorbs."""
    f = clip(m8[..., :2] + m8[..., 2:4] * _stack2(dist, world_z), 0.0, 1.0)
    return (f[..., 0] ** m9[..., 0]) * (f[..., 1] ** m9[..., 1]) * m10w


def output_alpha(transmission_rgb):
    """:1047 -- the output alpha is the Rec.709 luma of the TRANSMISSION
    vector, with the shader's own weights (0.2125 / 0.7154 / 0.0721).

    This is a fixed-function-blend transparent output. It is the evidence
    that this module is not an MBOIT pass: a moment write would need extra
    attachments or storage images, and the file has one colour output and
    no imageStore."""
    return (transmission_rgb[..., 0] * GLASS_LUMA[0]
            + transmission_rgb[..., 1] * GLASS_LUMA[1]
            + transmission_rgb[..., 2] * GLASS_LUMA[2])


# --------------------------------------------------------------------------
# small array helpers, backend-agnostic
# --------------------------------------------------------------------------
def _stack2(a, b):
    m = np if isinstance(a, np.ndarray) else __import__("torch")
    return m.stack([a, b], -1)


def _stack3(a, b, c):
    m = np if isinstance(a, np.ndarray) else __import__("torch")
    return m.stack([a, b, c], -1)


def _norm(v):
    return _sqrt((v * v).sum(-1))[..., None]


def _len(v):
    return _sqrt((v * v).sum(-1))


def _sqrt(x):
    return maximum(x, 0.0) ** 0.5


def _exp2(x):
    return 2.0 ** x


def _min(a, b):
    from ..post._backend import minimum
    return minimum(a, b)
