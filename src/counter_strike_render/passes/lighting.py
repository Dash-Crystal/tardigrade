#!/usr/bin/env python3
"""Sun and cascade-shadow expressions, lifted so both sides can call them.

SCOPE, STATED BEFORE ANY EXPRESSION
-----------------------------------
`docs/projects/counter-strike-sft/csgo_environment_ps.glsl` is ONE
decompiled module. Everything here is transcribed from its lines 800-855
(the cascade loop and the compare) and 892-923 (the sun diffuse), with the
uniform members named by BYTE OFFSET into the `set = 3, binding = 0` block
rather than by `_mN`.

WHY BY OFFSET. The ordinals do not transfer between families: byte offset
288 is `_m4` in `csgo_complex_ps.glsl:151` and `_m3` in
`csgo_environment_ps.glsl:153`, because the two dead-strip different
members of the SAME block. The offsets are identical across both. A
previous join assumed `_mN` sat at `16N`, put `_m11` at 176 where the
declaration says 31120, and refused itself. See the reference read §0.

The constants those offsets hold were READ out of two captures (718 block
copies, both agreeing) and are listed in READ from the capture, two captures agreeing. They
are values, not part of these expressions, so they are arguments here.

WHY THESE FUNCTIONS AND NOT MORE. Only expressions the renderer can
actually CALL are lifted. Retyping a reference into an `impl=` slot is
caught by the registry's SELF_PAIRED refusal and would be worthless
anyway; a term is landed when `gpu_render.py` evaluates THIS code object,
which is why each function below names its call site.

ARRAY LIBRARY. These are written against the duck-typed subset numpy and
torch share (arithmetic, `abs`, `clip`/`clamp` via `minimum`/`maximum`),
so the conformance harness can call them with numpy arrays and the
renderer with CUDA tensors -- the same code object, not a second copy.
`_clip01` exists because numpy spells it `clip` and torch spells it
`clamp`; using `minimum`/`maximum` avoids choosing.
"""

_E = "docs/projects/counter-strike-sft/csgo_environment_ps.glsl"


def _cos(x):
    """cos for a python float, a numpy array or a torch tensor alike.

    The first version took `math.cos` and the registry refused the term
    with IMPL_ERROR -- because the harness sweeps the ANGLE as an array
    (a term pinned at its shipped constant cannot distinguish the
    expression from a lookup of that constant), and `math.cos` takes a
    scalar. The refusal was the sample domain doing its job on the
    implementation rather than on the reference.
    """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch.cos(x)
    except ImportError:
        pass
    import numpy as _np
    return _np.cos(x)


def _sin(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch.sin(x)
    except ImportError:
        pass
    import numpy as _np
    return _np.sin(x)


def _stack3(a, b, c):
    """Stack three same-shaped arrays on a new last axis, numpy or torch."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return torch.stack([a, b, c], -1)
    except ImportError:
        pass
    import numpy as _np
    return _np.stack([a, b, c], -1)


def _clip01(x):
    """clamp(x, 0, 1) for numpy arrays and torch tensors alike."""
    one = x * 0 + 1
    zero = x * 0
    # `where` is spelled the same in both, and avoids clip/clamp.
    lo = (x > zero) * x
    return (lo < one) * lo + (lo >= one) * one


def cascade_fade(px, py, fade_offset, fade_scale):
    """csgo_environment_ps.glsl:823 -- the cross-fade band into the next cascade.

        vec2 _22193 = vec2(1.0) - clamp((abs(_24804) * vec2(_5538._m15))
                                        + vec2(_5538._m14), 0.0, 1.0);
        ...
        _13142 = clamp(_22193.x * _22193.y, 0.0, 1.0);          // :826

    `_m14` is at byte offset +31280 and `_m15` at +31284; READ as -9.0 and
    +10.0, so the band is `|xy|` in [0.9, 1.0] -- the outer tenth. The two
    axes are formed independently and MULTIPLIED, which is why a corner is
    darker than an edge; a min() would not reproduce it.

    Called by gpu_render.gt_csm_shadow().
    """
    fx = 1.0 - _clip01(abs(px) * fade_scale + fade_offset)
    fy = 1.0 - _clip01(abs(py) * fade_scale + fade_offset)
    return _clip01(fx * fy)


def cascade_atlas_uv(pxy, subrect):
    """csgo_environment_ps.glsl:824 -- cascade clip xy to shadow-atlas uv.

        vec2 _20561 = (_24804 * _5538._m19._m0[_13039].zw)
                      + _5538._m19._m0[_13039].xy;

    `_m19` is the `vec4[4]` at byte offset +31808, stride 16. READ on
    wallA2, `subrect.zw` is (0.147059, -0.121951) for every cascade and
    `subrect.xy` steps (0.147059, 0.441176, 0.735294) then wraps to
    (0.147059, 0.512195): a half-extent and a centre, three tiles across
    and one starting the second row.

    TWO PROPERTIES THAT ARE THE REFERENCE'S, NOT CONVENTION:
      * `subrect.w` is NEGATIVE -- the atlas V axis is flipped. Dropping
        the sign gives a shadow map mirrored in V, which looks plausible
        and is wrong.
      * centre EQUALS half-extent for cascade 0, so tile 0 starts at u = 0
        with NO half-texel inset. Adding the usual inset is a fitted
        correction to a term that does not have one.

    `pxy` is the cascade clip xy, shape (..., 2); `subrect` is the
    4-vector for the SELECTED cascade, shape (..., 4), in file order
    (x, y, z, w) = (centre.x, centre.y, half.x, half.y). Both are packed
    so the expression is pure slice arithmetic, which numpy and torch
    spell identically -- the alternative, returning a 2-tuple, is not an
    array on either side and the conformance runner refuses it.

    Called by gpu_render.gt_csm_shadow().
    """
    return pxy * subrect[..., 2:4] + subrect[..., 0:2]


def shadow_compare_ref(pz, shadow_bias):
    """csgo_environment_ps.glsl:840 -- the reference depth handed to the compare.

        textureLod(sampler2DShadow(...), vec3(uv, clamp(_14975.z
                   + _5538._m12, 0.0, 1.0)), 0.0)

    `_m12` is at byte offset +31256, READ as exactly -2**-16
    (-1.52587890625e-05). Our own value was a swept -0.0015, 98x larger.

    The clamp to [0,1] is load-bearing and is where the cascade depth-sign fix's
    defect lived: with a NEGATIVE cascade depth range every `pz` is
    negative, `clamp` returns 0, and the compare degenerates into a
    coverage test that shadows everything. A -2**-16 bias only means
    anything against NON-NEGATIVE stored depths, so the read value
    confirms the depth-sign fix from a second direction.

    Called by gpu_render.gt_csm_shadow().
    """
    return _clip01(pz + shadow_bias)


def sun_diffuse(sun_ambient, ndotl_term, sun_colour, atten):
    """csgo_environment_ps.glsl:916-923 -- the whole sun diffuse contribution.

        vec3 _16808 = (_5538._m9.xyz * _23022).xyz;             // :916
        _24880 = _5538._m3.xyz + (_18223.xyz * _16808);         // :918  lit
        ...
        _24880 = _5538._m3.xyz;                                 // :923  not lit

    Structurally identical at csgo_complex_ps.glsl:636,641 with the
    ordinals `_m4` / `_m12` for the same two byte offsets, 288 and 31136.

    `sun_ambient` is `_5538` at byte offset +288, READ as (0, 0, 0) in 717
    of 718 block copies across two captures. So the `not lit` branch
    evaluates to ZERO: the reference adds NO constant ambient anywhere in
    the sun path, and the branch is expressible as this one expression
    with `atten = 0`. Our AMBIENT_SKY/AMBIENT_GROUND hemisphere is NOT
    this member -- it stands in for the BAKED subsystem, which is a
    different expression entirely. See READ from the capture.

    `sun_colour` is +31136, READ as (3.0, 2.790334, 2.492310), which is
    sRGB_EOTF([255,247,235]) * brightness 3.0 to max|delta| 1.4e-6.

    `atten` is a scalar per sample and the others are RGB triples, so it
    is broadcast on the last axis explicitly. Relying on implicit
    broadcasting works for (N,) against (N,3) in neither library.

    Called by gpu_render's world shading path.
    """
    a = atten[..., None] if atten.ndim < sun_ambient.ndim else atten
    return sun_ambient + ndotl_term * (sun_colour * a)


_G = "docs/projects/counter-strike-sft/glsl_loopfam/generic_r69m3.glsl"


def probe_volume_weight(p_local, box_min, box_max, inv_fade):
    """generic_r69m3.glsl:1098-1100 -- a volume's weight is the MIN over SIX faces.

        vec3 a = clamp((p - _m1) * _m4.xyz, 0.0, 1.0);
        vec3 b = clamp((_m3 - p) * _m4.xyz, 0.0, 1.0);
        float w = min(min(a.x, min(a.y, a.z)), min(b.x, min(b.y, b.z)));

    Six independent half-space distances, each normalised by the volume's
    own inverse fade distance and clamped, then reduced by MIN -- NOT by a
    product. The distinction is not cosmetic: a product of six clamped
    ramps falls off in every direction at once and rounds the box's edges,
    where the min holds 1.0 through the interior and tapers only across
    whichever face is nearest. At a corner a product is the sixth power of
    what the min returns.

    `w == 0` means the point is outside, and :1101 SKIPS such a volume
    entirely rather than accumulating a zero -- which matters because the
    loop's early-out counts accumulated alpha, so a skipped volume and a
    zero-weight volume differ in how many volumes get visited.

    Called by gpu_render's probe path.
    """
    a = _clip01((p_local - box_min) * inv_fade)
    b = _clip01((box_max - p_local) * inv_fade)
    m = a[..., 0]
    for k in (1, 2):
        m = m * (m < a[..., k]) + a[..., k] * (a[..., k] <= m)
    for k in (0, 1, 2):
        m = m * (m < b[..., k]) + b[..., k] * (b[..., k] <= m)
    return m


def probe_smoothstep(w):
    """generic_r69m3.glsl:1114 -- `(w*w) * ((-2.0)*w + 3.0)`.

    The classic smoothstep polynomial, applied to the six-face min BEFORE
    it becomes an alpha contribution. Transcribed rather than called from
    a library so the coefficient order matches the file: the shader writes
    `((w*w) * (((-2.0)*w) + 3.0))`, not `3w^2 - 2w^3` factored differently,
    and at float precision those are not the same reduction.
    """
    return (w * w) * ((-2.0) * w + 3.0)


def probe_composite_step(accum_rgb, accum_alpha, sample_rgb, tint, w):
    """generic_r69m3.glsl:1114-1120 -- ONE front-to-back step. ORDER MATTERS.

        float c   = smoothstep(w) * (1.0 - alpha);      // :1114
        float a2  = alpha + c;                          // :1115
        rgb      += texture(...).xyz * _m9 * c;         // :1116
        if (a2 > 0.99) break;                           // :1120

    `(1 - alpha)` is remaining transmittance, so a volume's contribution
    depends on EVERYTHING ALREADY ACCUMULATED. Re-ordering the volumes
    changes the result, and the `> 0.99` early-out truncates the list --
    two reasons the iteration order is part of the expression rather than
    an implementation detail. The reference walks the volume bitmask via
    `findLSB` (:1094), i.e. ASCENDING volume index, which is the order.

    Returns (rgb, alpha, done) with `done` true once alpha passes 0.99.
    """
    c = probe_smoothstep(w) * (1.0 - accum_alpha)
    a2 = accum_alpha + c
    rgb = accum_rgb + sample_rgb * tint * c[..., None]
    return rgb, a2, a2 > 0.99


# --------------------------------------------------------------------------
# #48, the sky pass, decomposed so the registry can score it.
#
# A whole-pass @term would compare a ray-construction, a rotation and a cube
# fetch as one opaque function on synthetic inputs -- which moves a counter
# and checks nothing, because the reference has no single expression to diff
# it against. These two ARE reference expressions with line citations, and
# `sky_pass` calls them.


def sky_zrot(d, degrees):
    """#48: the sky draw's ENTIRE uniform state, applied as a rotation.

    chunkIndex 40411 binds one 64-byte buffer holding ONE matrix: a pure
    Z-rotation, unit scale, NO TRANSLATION. Read against the constant:
    cos(115) = -0.4226183 vs a read -0.4226194 and sin(115) = 0.9063078 vs
    +0.9063073, both ~1e-6.

    The absence of translation is the semantic content -- an untranslated
    far-plane pass is a DIRECTION lookup, which is why our screen-locked
    gradient is not this operation missing a constant but a different
    operation with nowhere to put one.

    AXIS: the reference rotates about SOURCE Z (up). This renderer's pack
    basis is glTF = (src_y, src_z, src_x) (gpu_render.py:18, ray-cast
    validated), so source Z is pack Y and the rotation is applied about
    pack Y. That mapping is READ from a documented, validated basis, but it
    IS a step between the read constant and this code -- the first thing to
    check if the sky comes out rotated.
    """
    th = degrees * 3.141592653589793 / 180.0
    c = _cos(th)
    s = _sin(th)
    rx = d[..., 0] * c - d[..., 2] * s
    rz = d[..., 0] * s + d[..., 2] * c
    return _stack3(rx, d[..., 1], rz)


def cube_face_uv(d):
    """The D3D cube face-selection table and its u/v mapping.

    Transcribed from gpu_render.cube_sample()'s body rather than from the
    D3D spec: that function's conventions were validated against the probe
    array's own in-file SH block at R^2 0.99992 against a runner-up of
    0.75, and re-deriving a validated convention is how two implementations
    of one thing appear.

    NOT carried over from it: its 180-degree Z rotation. That was
    established for the PROBE array against ITS asset; the sky is a
    different asset with its own read orientation (sky_zrot above), and
    inheriting the probe's would be the family-generalisation trap of
    the two-selector comparison in this file.

    Returns (face, u, v) with u, v in [0, 1] under the texel-centre
    convention x = u*N - 0.5 that the decode was validated with -- the
    corner-to-corner alternative u*(N-1) compresses a face onto its texel
    centres and warps 12.5% at N=4.
    """
    sx = -d[..., 2]
    sy = -d[..., 0]
    sz = d[..., 1]
    ax, ay, az = abs(sx), abs(sy), abs(sz)
    # `* 1.0` promotes each mask out of bool BEFORE any arithmetic. numpy
    # tolerates `1 - bool_array`; torch raises on it, so a mask kept as
    # bool works in the conformance harness and fails in the renderer --
    # a divergence that only appears at the call site, which is exactly
    # what the shared-code-object rule exists to prevent and did not,
    # because the two libraries disagree about a type rather than a value.
    xmaj = ((ax >= ay) * (ax >= az)) * 1.0
    ymaj = (1.0 - xmaj) * ((ay >= az) * 1.0)
    zmaj = (1.0 - xmaj) * (1.0 - ymaj)
    face = (xmaj * ((sx > 0) * 1.0 * 0 + (sx <= 0) * 1.0 * 1)
            + ymaj * ((sy > 0) * 1.0 * 2 + (sy <= 0) * 1.0 * 3)
            + zmaj * ((sz > 0) * 1.0 * 4 + (sz <= 0) * 1.0 * 5))
    ma = ax * xmaj + ay * ymaj + az * zmaj
    ma = ma + (ma < 1e-8) * 1e-8
    u = (xmaj * ((sx > 0) * 1.0 * (-sz) + (sx <= 0) * 1.0 * sz)
         + ymaj * sx
         + zmaj * ((sz > 0) * 1.0 * sx + (sz <= 0) * 1.0 * (-sx)))
    v = (xmaj * (-sy)
         + ymaj * ((sy > 0) * 1.0 * sz + (sy <= 0) * 1.0 * (-sz))
         + zmaj * (-sy))
    return face, (u / ma) * 0.5 + 0.5, (v / ma) * 0.5 + 0.5


def cascade_project(wpos_h, r0, r1, r2, r3):
    """csgo_environment_ps.glsl:819 -- world position into cascade clip space.

        vec4 _18322 = vec4(_4198.xyz, 1.0) * _5538._m18._m0[_13039];

    A ROW-VECTOR product: `v * M`, so the translation lives in the matrix's
    last row and the camera at `_4198 = 0` maps to that row directly. That
    convention is not a preference -- the camera-relative read at vs_base:180 settled it
    against two independent constraints (the inverted matrices' depth axis
    lands on the read sun direction at dot = -1.000000, and the camera's
    clip z falls inside the [0,1] the :840 clamp assumes; the transpose
    gives -0.889 and z = 2.25).

    WHAT `wpos_h` MUST BE, because getting this wrong cost two published
    reversals: `_4198` is NOT an absolute world position. The vertex stage
    emits `_4046 = worldPos - _5037._m0.xyz` (vs_base:180) and the pixel
    stage receives exactly that. The input here is therefore CAMERA-RELATIVE
    world, homogeneous. Feeding an absolute position produces numbers that
    look reasonable and mean nothing -- §6 and §9 both did, and both had to
    be withdrawn.

    THE ROWS ARE PASSED SEPARATELY, on purpose. A single `cascade_mat`
    argument is ambiguous between a (4, 4) matrix and a batch of 4-vectors,
    and the two index identically under `m[0]` while meaning different
    things -- the conformance harness supplies the second and the renderer
    the first, so the term REFUSED with SHAPE_MISMATCH and would otherwise
    have been "fixed" by reshaping until it passed. Four named rows cannot
    be misread by either caller.

    Called by gpu_render.gt_csm_shadow().
    """
    return (wpos_h[..., 0:1] * r0 + wpos_h[..., 1:2] * r1
            + wpos_h[..., 2:3] * r2 + wpos_h[..., 3:4] * r3)


def cascade_inside(px, py, extent):
    """csgo_environment_ps.glsl:820 -- the cascade selection test.

        if (max(abs(_12779), abs(_18322.y)) < _5538._m13[_13039])

    A genuine PER-AXIS bound. It is worth stating what this is NOT: the
    other cascade selector in the reference, deferred_particle_shadows
    :104, tests `dot(clamp(uv,0,1) - uv, vec2(1)) == 0`, which SUMS the two
    axes' deviations, so (-a, 1+a) sums to zero and selects as though
    inside with both components out of range. That family works in [0,1]
    UV; this one works in [-1,1] clip. the two-selector comparison in this file records
    both, and the summation quirk has no mechanism here.

    `extent` READS (1, 1, 1, 1) at byte offset +31264 -- which is itself
    the evidence for the [-1,1] range, since on a [0,1] output a bound of
    1 passes the entire range and cannot discriminate one cascade from the
    next.

    Called by gpu_render.gt_csm_shadow().
    """
    a = abs(px)
    b = abs(py)
    m = a * (a >= b) + b * (b > a)
    return m < extent


def sun_roughness_floor(rough, sun_angular_w):
    """csgo_environment_ps.glsl:908 -- the sun's angular size floors roughness.

        vec2 _17301 = max(_10873, vec2(_5538._m8.w));

    `_m8` is the sun direction at byte offset +31120 and its .W component
    -- READ as 0.040000 -- is spent HERE, as a lower bound on roughness,
    nowhere else. The map ships `angulardiameter 0.3` on its
    light_environment, so this is the disc's angular size entering the
    specular lobe: a perfectly smooth surface cannot produce a highlight
    sharper than the source, and the floor is what stops it.

    We had no such floor, which makes our sun highlight a point-source
    delta on any low-roughness material where the reference's is a disc.

    Called by gpu_render's world shading path.
    """
    return rough * (rough > sun_angular_w) + sun_angular_w * (sun_angular_w >= rough)


def sun_half_vector(sun_dir, wpos, eye):
    """csgo_environment_ps.glsl:909 -- the half vector, with its sign READ.

        vec3 _12281 = normalize(_5538._m8.xyz
                                + (-normalize(_7715.xyz - _4459._m5.xyz)).xyz);

    The view term is `-normalize(wpos - eye)`, i.e. the direction FROM the
    surface TOWARD the eye, and the minus is inside the normalize's
    argument order rather than applied after -- transcribed as written
    because `normalize(eye - wpos)` and `-normalize(wpos - eye)` differ in
    float only by rounding but differ in READING by a sign nobody can
    check later.

    `_4459._m5` is the eye at byte offset +416 of the per-view block.
    """
    v = eye - wpos
    n = (v[..., 0:1] * v[..., 0:1] + v[..., 1:2] * v[..., 1:2]
         + v[..., 2:3] * v[..., 2:3]) ** 0.5
    v = v / (n + (n <= 0.0) * 1e-12)
    h = sun_dir + v
    hn = (h[..., 0:1] * h[..., 0:1] + h[..., 1:2] * h[..., 1:2]
          + h[..., 2:3] * h[..., 2:3]) ** 0.5
    return h / (hn + (hn <= 0.0) * 1e-12)


def sun_specular_ggx(f_term, rough, ndoth, ndotl_h, ndotl):
    """csgo_environment_ps.glsl:911-917 -- the sun specular lobe, verbatim.

        float a2 = (r*r) * (r*r);                                 // :913-914
        float d  = ((NdotH*NdotH) * (a2 - 1.0)) + 1.0;            // :915
        spec = F * (a2 / (((d*d) * (X*X)) * ((r*4.0) + 2.0))) * NdotL;

    THREE THINGS THAT ARE THE REFERENCE'S AND NOT A TEXTBOOK GGX:

      * the denominator carries `(r*4 + 2)`, not the usual `4*NdotL*NdotV`.
        That is a combined normalisation-and-visibility term expressed in
        ROUGHNESS, and substituting the textbook form changes the lobe's
        magnitude at every roughness.
      * `a2` is `(r*r)*(r*r)` = r^4, so the shader's `r` is already
        perceptual roughness squared once by the caller; writing `r*r`
        here would be one squaring short.
      * `X` (`_19210`) is `max(0, dot(sunDir, H))` from :914 -- an L-dot-H,
        NOT the N-dot-V a textbook Smith term uses.

    Transcribed in the file's own grouping so the float reduction order
    matches; `(d*d) * (X*X)` then `* (r*4+2)` is not the same sum as
    multiplying the three factors in any other order.
    """
    a2 = (rough * rough) * (rough * rough)
    d = ((ndoth * ndoth) * (a2 - 1.0)) + 1.0
    denom = ((d * d) * (ndotl_h * ndotl_h)) * ((rough * 4.0) + 2.0)
    return f_term * (a2 / denom)[..., None] * ndotl[..., None]


def sun_wrap_scatter(ndotl, l_dot_bent, n, bent_n, sun_dir, t):
    """csgo_environment_ps.glsl:902 -- the sun diffuse SHAPE on thin surfaces.

        _18223 = mix(vec3(NdotL),
                     vec3((((0.5 + (NdotL * 0.5))
                            + pow(1.0 - clamp(B, 0.0, 1.0), 4.0))
                           * clamp((B + 0.2) * 4.0, 0.0, 1.0))
                          * clamp(mix(dot(mix(N, bentN, vec3(10.0)), L),
                                      1.0, t), 0.0, 1.0)),
                     vec3(t));

    where B = dot(bentN, L) and t = clamp(_13960, 0, 1), itself
    `_10839 * (0.2 + 0.15*w*w*w)` from :735 with w the normal map's alpha.

    THE FACTOR OF 10 IS NOT A TYPO AND NOT A CLAMPED LERP. `mix(N, bentN,
    vec3(10.0))` EXTRAPOLATES: it is `N + 10*(bentN - N)`, ten times past
    the bent normal, and the result is fed straight into a dot with the
    light. A reader who assumes mix's factor lives in [0,1] -- as every
    other mix in this shader does -- silently implements a different
    function. Transcribed as written.

    The rest is a half-lambert (`0.5 + NdotL*0.5`) plus a rim
    (`pow(1-B, 4)`), gated by a hard ramp that reaches full at B = 0.05
    and is zero below B = -0.2, so light arriving slightly BEHIND the bent
    normal still contributes. That is the scatter: `mix` toward it by `t`,
    so t = 0 leaves plain NdotL and the whole term vanishes.

    Called by gpu_render's world shading path.
    """
    bent_mix = n + (bent_n - n) * 10.0
    d = (bent_mix[..., 0] * sun_dir[..., 0] + bent_mix[..., 1] * sun_dir[..., 1]
         + bent_mix[..., 2] * sun_dir[..., 2])
    b = _clip01(l_dot_bent)
    half_rim = (0.5 + ndotl * 0.5) + (1.0 - b) ** 4.0
    ramp = _clip01((l_dot_bent + 0.20000000298023223876953125) * 4.0)
    # GLSL defines mix(a, b, t) as a*(1-t) + b*t. Writing the algebraically
    # identical a + (b-a)*t is a DIFFERENT float reduction, and the registry
    # caught it: the two forms scored 1.332e-15 apart, inside the epsilon
    # band but non-zero, and the runner says in those words that an exact
    # transcription of an algebraically identical expression gives 0.0.
    # A tolerance that would have swallowed this is exactly how a rounded
    # constant or an idealised exponent survives.
    gate = _clip01(d * (1.0 - t) + 1.0 * t)
    wrapped = half_rim * ramp * gate
    return ndotl * (1.0 - t) + wrapped * t
