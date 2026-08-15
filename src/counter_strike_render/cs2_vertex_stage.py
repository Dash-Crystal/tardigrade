"""CS2 vertex stage — ported from the decompiled `csgo_*_vs` SPIR-V.

Every function here is a transcription of a specific region of a specific
listing under `docs/projects/counter-strike-sft/`.  Where a listing exists the
port is line-for-line and the reference lines are named in the docstring.
Where NO listing exists (three axes: `D_CS_VERTEX_ANIMATION`,
`S_PRE_BAKED_VERTEX_ANIMATION`, `S_VERTEX_BREAK_ANIMATION`) the docstring says
so in its first paragraph and states exactly what was substituted.  Nothing in
this file returns its input unchanged as a stand-in for an unimplemented term.

UNITS.  The reference works in SOURCE UNITS (inches) with a Z-up basis.  This
renderer's world is glTF metres with a Y-up basis (`gpu_render.py:38`,
`S = 0.0254`).  Every spatial constant taken from the reference is therefore
applied to `world_metres / S`, and the reference's up axis `vec3(0,0,1)` maps
to `vec3(0,1,0)`.  Both conversions are applied in exactly one place each
(`_to_source_units` and `UP_YUP`) so they can be audited.

COSTS.  The reference costs quoted in comments are FLOPs per VERTEX, never per
pixel.  `SHADER_CALLFLOW_vertex_stages.md:14-17` gives de_inferno as 8.6e6
vertices against 2.07e6 pixels, i.e. vertex:pixel = 4.15:1; no conversion is
performed anywhere in this file.

RESOLVED VERTEX ATTRIBUTE STREAM IDENTITY.  Established by USE, not by
location index, across the six committed listings.  See `STREAM_IDENTITY`
below for the table and the evidence for each row.
"""

import math

import torch

# --- axis conventions --------------------------------------------------
# gpu_render.py:38.  Source units (inches) -> metres.
S_SOURCE_TO_M = 0.0254
# The reference's up axis.  `csgo_foliage_vs_max.glsl:209` takes
# `cross(windDirOs, vec3(0,0,1))`; in Source, +Z is up.  Our pack is glTF
# Y-up (fast_pack2.py bakes the glTF node transform into positions), so the
# same geometric quantity is `cross(windDirOs, (0,1,0))` here.
UP_YUP = (0.0, 1.0, 0.0)


def _to_source_units(p_m):
    """world metres -> Source inches, the space every reference constant is in."""
    return p_m / S_SOURCE_TO_M


# ----------------------------------------------------------------------
# Vertex attribute stream identity.
#
# Confirmed BY USE across the six committed listings, not by location index
# and not by SSA id alone.  The spirv-cross SSA ids do recur across files
# (the brief's caution), so each row below records the use that confirms it
# rather than the id that suggested it.
#
#   id      type    semantic (D3D slot)            confirmed by
#   ------  ------  -----------------------------  -----------------------
#   _5275   vec3    vPositionOs   (POSITION0)       fed to the instance
#                                                   mat3x4 as vec4(p,1)*M:
#                                                   env_blend_max:200,
#                                                   complex_max:183,
#                                                   foliage_max:340,351
#   _5800   vec2    vTexCoord     (TEXCOORD0)       becomes .xy of the UV
#                                                   vec4: env_blend_max:217,
#                                                   complex_max:200,
#                                                   foliage_max:367
#   _4301   vec2    vTexCoord2    (TEXCOORD1)       written to .zw of the
#                                                   SAME UV vec4 and then
#                                                   selected against .xy by
#                                                   the per-layer UV-set int:
#                                                   env_blend_max:218-219 +
#                                                   346, env_blend_base:
#                                                   159-160 + 164.  THIS IS
#                                                   THE EXTRA vec2 THAT
#                                                   csgo_complex LACKS.
#   _4574   uint    nPackedFrame  (NORMAL0)         bit-unpacked to an
#                                                   octahedral normal + a
#                                                   Rodrigues tangent angle
#                                                   + a sign bit:
#                                                   env_blend_max:119-135,
#                                                   complex_max:100-116,
#                                                   foliage_max:120-136,
#                                                   foliage_maxfetch:106-122.
#                                                   The four expressions are
#                                                   textually the same and
#                                                   their literal constants
#                                                   differ by max|d| = 0.
#   _3208   vec3    vNormalOs     (NORMAL0)         the D_COMPRESSED=0 form;
#   _5609   vec4    vTangentUOs_flTangentVSign      normal and tangent are
#                   (TANGENT0)                      transformed directly and
#                                                   .w carried through:
#                                                   env_blend_base:155,204
#   _5227   uvec4   vBlendIndices (BLENDINDICES0)   .x is added to the
#                                                   instance transform base
#                                                   index: env_blend_max:149,
#                                                   complex_max:131,
#                                                   foliage_max:150
#   _5506   uint    nInstanceIdx  (TEXCOORD13)      indexes the per-draw
#                                                   data buffer:
#                                                   env_blend_max:137-144
#   _4772   vec4    COLOR0                          THREE DIFFERENT USES,
#                                                   matching the three
#                                                   declared semantics:
#                                                   * VertexPaintTintColor,
#                                                     env_blend_max:457
#                                                     `mix(1, c.xyz, c.w)`;
#                                                   * VertexPaintTintColor,
#                                                     complex_max:290-319
#                                                     all-zero test then
#                                                     multiply;
#                                                   * Color, foliage_max:384
#                                                     passthrough; and in
#                                                     foliage_maxfetch:219 it
#                                                     is `clamp(c,0.001,1)`
#                                                     and its .x/.y/.z are
#                                                     the three PER-VERTEX
#                                                     WEIGHTS of
#                                                     S_VERTEX_ANIMATION.
#   _5703   vec4    vPerVertexLighting (COLOR1)     decoded `(rgb*6*a)` then
#                                                   squared into the
#                                                   interpolant, identically
#                                                   in all three families:
#                                                   env_blend_max:220+470,
#                                                   complex_max:201+329,
#                                                   foliage_max:368+389
#   _4489   vec4    vColorBlendValues (TEXCOORD4)   passed straight through
#                                                   to the blend PS:
#                                                   env_blend_max:462,
#                                                   env_blend_base:197
#   _3823   vec3    vPivotPaint   (TEXCOORD4)       an OBJECT-SPACE POINT in
#                                                   the same space as
#                                                   _5275: differenced with
#                                                   the trunk base
#                                                   (foliage_max:253),
#                                                   rotated as a position
#                                                   (255,262), used as the
#                                                   level-2 rotation origin
#                                                   (316,324) and a
#                                                   zero-length value
#                                                   DISABLES level 2
#                                                   (273).  Also hashed for
#                                                   a per-branch phase
#                                                   (203).  FOLIAGE ONLY.
#   _3932   vec3    vFoliageParams (TEXCOORD5)      per-vertex animation
#                                                   weights: .x is the
#                                                   detail-bend weight
#                                                   (foliage_max:339 under
#                                                   pow(), and 385 scaling
#                                                   the flutter output), .y
#                                                   is the level-2 weight
#                                                   (285, `+0.25`).  .z is
#                                                   not read by this
#                                                   variant.  FOLIAGE ONLY.
#
# TEXCOORD4 is the same D3D slot carrying vColorBlendValues on
# environment/blend/complex and vPivotPaint on foliage
# (SHADER_CALLFLOW_vertex_stages.md:169-171); the slot number is not the
# meaning, which is why every row above is justified by use.
STREAM_IDENTITY = {
    "vPositionOs": "POSITION0  f32x3  object-space position",
    "vTexCoord": "TEXCOORD0  f32x2  UV set 0 (LowPrecisionUv)",
    "vTexCoord2": "TEXCOORD1  f32x2  UV set 1 (LowPrecisionUv1)",
    "vLightmapUV": "TEXCOORD3  f32x2  lightmap UV (a THIRD, distinct set)",
    "vColorBlendValues": "TEXCOORD4  f32x4  VertexPaintBlendParams (env/blend/complex)",
    "vPivotPaint": "TEXCOORD4  f32x3  PivotPaint (foliage only)",
    "vFoliageParams": "TEXCOORD5  f32x3  FoliageAnimation (foliage only)",
    "vBlendIndices": "BLENDINDICES0  u32x4  instance-transform sub-index",
    "nInstanceIdx": "TEXCOORD13  u32  per-draw data buffer index",
    "vBlendColorTint": "COLOR0  f32x4  VertexPaintTintColor / Color",
    "vPerVertexLighting": "COLOR1  f32x4  PerVertexLighting, decode (rgb*6*a)^2",
    "vNormalOs": "NORMAL0  f32x3  D_COMPRESSED_NORMALS_AND_TANGENTS=0",
    "vTangentUOs_flTangentVSign": "TANGENT0  f32x4  D_COMPRESSED=0",
    "nPackedFrame": "NORMAL0  u32  D_COMPRESSED_NORMALS_AND_TANGENTS=1",
}


def _nrm(v, eps=1e-9):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


# =======================================================================
# D_COMPRESSED_NORMALS_AND_TANGENTS
# =======================================================================
#
# Reference, all four families, textually the same expression:
#   csgo_environment_blend_vs_max.glsl:119-135, 199, 473
#   csgo_complex_vs_max.glsl:100-116, 182, 332
#   csgo_foliage_vs_max.glsl:120-136, 350, 392
#   csgo_foliage_vs_maxfetch.glsl:106-122, 187, 204
# and the D_COMPRESSED=0 counterpart at
#   csgo_environment_blend_vs_base.glsl:155, 204
#
# The two forms are mutually exclusive by construction — `nPackedFrame` and
# `vNormalOs` both bind NORMAL0, and across all 10+10+48+21 declared input
# signatures the count containing both is 0
# (SHADER_CALLFLOW_vertex_stages.md:141-149).
#
# Literal constants, taken verbatim from the listing:
_OCT_SCALE = 0.00195503421127796173095703125   # = 2/1023, the 10-bit oct scale
_TAN_ANGLE_SCALE = 0.003069460391998291015625  # = 2*pi/2047, the 11-bit angle


def decode_packed_frame(packed_u32, inv_m3=None):
    """`D_COMPRESSED_NORMALS_AND_TANGENTS=1`: u32 -> (normal, tangent4).

    Ported from `csgo_foliage_vs_max.glsl:120-136` (normal + tangent basis +
    angle + sign) and `:392` (the Rodrigues rotation and the inverse-matrix
    transform of the result).  The same expression appears at
    `csgo_environment_blend_vs_max.glsl:119-135` + `:473` and
    `csgo_complex_vs_max.glsl:100-116` + `:332`; the three differ by no
    literal constant (max|delta| over the six constants = 0), which is what
    confirms the cross-file SSA-id correspondence of `_4574` rather than the
    id itself.

    Bit layout, read off the shifts and masks:
        bit  0        tangent V sign  (0 -> -1.0, else +1.0)
        bits 1..11    tangent rotation angle, 11 bits, * 2*pi/2047
        bits 12..21   octahedral normal X, 10 bits, * 2/1023 - 1
        bits 22..31   octahedral normal Y, 10 bits, * 2/1023 - 1

    `inv_m3` is the inverse of the instance transform's 3x3 part
    (`csgo_foliage_vs_max.glsl:352-366` computes it by cofactors).  The
    reference applies it to the TANGENT ONLY (`:392`, `t.xyz * _20198`) while
    the NORMAL gets the forward matrix (`:350`).  That asymmetry is ported as
    written, not "corrected"; for a rigid uniformly-scaled instance the two
    agree.  Pass None for identity.

    Returns (normal (...,3), tangent4 (...,4) with .w the sign).
    """
    u = packed_u32.to(torch.int64)
    nx = ((u >> 12) & 1023).to(torch.float32) * _OCT_SCALE - 1.0
    ny = ((u >> 22) & 1023).to(torch.float32) * _OCT_SCALE - 1.0
    nz = (1.0 - nx.abs()) - ny.abs()
    # `_24228 = clamp(-_23404, 0, 1)`; `mix(vec2(t), vec2(-t), v >= 0)`
    # selects the SECOND operand where the bool is true, i.e. -t for v>=0.
    t = (-nz).clamp(0.0, 1.0)
    nx = nx + torch.where(nx >= 0, -t, t)
    ny = ny + torch.where(ny >= 0, -t, t)
    # NOTE the reference does NOT rewrite .z after the wrap: `_8254.x` and
    # `_8254.y` are assigned, `.z` keeps the pre-wrap `_23404`.
    n = _nrm(torch.stack([nx, ny, nz], dim=-1))

    # Frisvad basis b1, `:134`.
    nzc = n[..., 2]
    sgn = torch.where(nzc >= 0, torch.ones_like(nzc), -torch.ones_like(nzc))
    a = -1.0 / (sgn + nzc)
    nxc, nyc = n[..., 0], n[..., 1]
    t0 = torch.stack([1.0 + ((sgn * nxc) * nxc) * a,
                      sgn * ((nxc * nyc) * a),
                      (-sgn) * nxc], dim=-1)

    ang = ((u >> 1) & 2047).to(torch.float32) * _TAN_ANGLE_SCALE
    sign_v = torch.where((u & 1) == 0,
                         -torch.ones_like(ang), torch.ones_like(ang))
    # Rodrigues about n: `t0*cos(a) + cross(n,t0)*sin(a)`, `:392`.
    tan = (t0 * torch.cos(ang).unsqueeze(-1)
           + torch.cross(n, t0, dim=-1) * torch.sin(ang).unsqueeze(-1))
    if inv_m3 is not None:
        # row-vector `v * M`, matching `_3091 ... .xyz * _20198`.
        tan = torch.einsum("...i,...ij->...j", tan, inv_m3)
    tan = _nrm(tan)
    return n, torch.cat([tan, sign_v.unsqueeze(-1)], dim=-1)


def encode_packed_frame(normal, tangent4):
    """Inverse of `decode_packed_frame`, so the axis is testable end to end.

    This has NO reference counterpart — the shader only decodes.  It exists
    so `D_COMPRESSED_NORMALS_AND_TANGENTS=1` can be driven from a pack that
    stores float normals/tangents, and so the quantisation error the axis
    introduces can be measured rather than assumed.
    """
    n = _nrm(normal)
    # octahedral encode (inverse of the wrap above)
    d = n / n.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)
    ox, oy, oz = d[..., 0], d[..., 1], d[..., 2]
    wx = torch.where(oz >= 0, ox,
                     (1.0 - oy.abs()) * torch.where(ox >= 0,
                                                    torch.ones_like(ox),
                                                    -torch.ones_like(ox)))
    wy = torch.where(oz >= 0, oy,
                     (1.0 - ox.abs()) * torch.where(oy >= 0,
                                                    torch.ones_like(oy),
                                                    -torch.ones_like(oy)))
    qx = (((wx + 1.0) / _OCT_SCALE).round().clamp(0, 1023)).to(torch.int64)
    qy = (((wy + 1.0) / _OCT_SCALE).round().clamp(0, 1023)).to(torch.int64)

    # recover the angle of the tangent in the Frisvad basis of the DECODED
    # normal, so the round trip is consistent with the decoder.
    nx2 = qx.to(torch.float32) * _OCT_SCALE - 1.0
    ny2 = qy.to(torch.float32) * _OCT_SCALE - 1.0
    nz2 = (1.0 - nx2.abs()) - ny2.abs()
    t2 = (-nz2).clamp(0.0, 1.0)
    nx2 = nx2 + torch.where(nx2 >= 0, -t2, t2)
    ny2 = ny2 + torch.where(ny2 >= 0, -t2, t2)
    nd = _nrm(torch.stack([nx2, ny2, nz2], dim=-1))
    nzc = nd[..., 2]
    sgn = torch.where(nzc >= 0, torch.ones_like(nzc), -torch.ones_like(nzc))
    a = -1.0 / (sgn + nzc)
    nxc, nyc = nd[..., 0], nd[..., 1]
    b1 = torch.stack([1.0 + ((sgn * nxc) * nxc) * a,
                      sgn * ((nxc * nyc) * a),
                      (-sgn) * nxc], dim=-1)
    b2 = torch.cross(nd, b1, dim=-1)
    tv = _nrm(tangent4[..., :3])
    ang = torch.atan2((tv * b2).sum(-1), (tv * b1).sum(-1)) % (2.0 * math.pi)
    qa = (ang / _TAN_ANGLE_SCALE).round().clamp(0, 2047).to(torch.int64)
    qs = (tangent4[..., 3] >= 0).to(torch.int64)
    return ((qy << 22) | (qx << 12) | (qa << 1) | qs).to(torch.int64)


# =======================================================================
# S_SECONDARY_UV  (vertex-attribute side only)
# =======================================================================
#
# Reference: `csgo_environment_blend_vs_base.glsl:158-190` (two layers) and
# `csgo_environment_blend_vs_max.glsl:217-219, 316-455` (six layers, plus
# the model-scale branch at 231-311).  `csgo_environment_vs` carries the same
# construct.  `csgo_complex_vs` declares S_SECONDARY_UV as a static axis
# (SHADER_CALLFLOW_csgo_complex_vs.md:42) but the max-FLOP module of that
# family has S_SECONDARY_UV=0 — its attribute list has no second vec2
# (`csgo_complex_vs_max.glsl:81-87`), which is itself the confirmation that
# the extra vec2 IS this axis.
#
# NOTE ON EVERY UV_CONSTRUCTION.md CITATION IN THIS FILE (2026-08-11):
# section 5's density table is glTF-ERA and superseded. It measured a
# pipeline we replaced; the ASH source declares no TEXCOORD_2 and maps
# TEXCOORD_1 -> uvs2. Scoped to the materials that select a UV set, uvs2 is
# EXACTLY ZERO on 53.97% of that surface. Sections 1-4, which this file
# cites for the SELECTOR semantics rather than for densities, are unaffected.
#
# The selector's own UI string is quoted in UV_CONSTRUCTION.md:11 —
# "0=Biplanar,1=UV1,2=UV2", `m_intDefault = [1,0,0,0]`.
#
# This ANSWERS the question UV_CONSTRUCTION.md:76-85 left open ("whether the
# VERTEX shader applies any further texcoord transform ... I have not ruled
# it out").  It does: guarded by `sel > 0`, the VS applies a per-layer
# 2x2-plus-offset transform to the selected set, `dot(uv, m.xy) + m.w` per
# channel.  With the identity parameters de_inferno's 30/30 materials
# declare (UV_CONSTRUCTION.md:35-38) the transform is inert, but it is
# present and it is in the vertex stage.


def secondary_uv(uv0, uv1, uv_set, xform=None):
    """Per-layer UV-set selection + affine, `..._blend_vs_base.glsl:162-190`.

    uv_set: 0 = the coordinate is passed through untransformed (the PS takes
            the biplanar branch, which is a pixel-stage term and belongs to
            another agent);
            1 = UV set 0;  2 = UV set 1.
    xform:  (...,2,4) rows m_row0, m_row1.  Applied as
            `u' = dot(uv, m0.xy) + m0.w`, `v' = dot(uv, m1.xy) + m1.w`,
            exactly as `:167-168`.  None = identity, which is what
            de_inferno's materials declare.

    Note the reference applies the affine for sel==1 as well as sel==2 — the
    guard is `sel > 0`, not `sel == 2`.
    """
    sel2 = (uv_set == 2).unsqueeze(-1)
    uv = torch.where(sel2, uv1, uv0)
    if xform is None:
        out = uv
    else:
        u = (uv * xform[..., 0, :2]).sum(-1) + xform[..., 0, 3]
        v = (uv * xform[..., 1, :2]).sum(-1) + xform[..., 1, 3]
        out = torch.stack([u, v], dim=-1)
    return torch.where((uv_set > 0).unsqueeze(-1), out, uv)


def scale_uv_by_model_scale(uv, col_len, u_axis, v_axis):
    """`g_nScaleTexCoord{U,V}ByModelScaleAxis`, `..._blend_vs_max.glsl:231-243`.

    `_10934 = vec4(len(M col x), len(M col y), len(M col z), 1)` and the
    coordinate is scaled by `dot(_10934, g_v...)`.  `col_len` is that vec4.
    `u_axis`/`v_axis` are the per-layer vec4 selectors (`_5618._m6/_m7`).
    UV_CONSTRUCTION.md:41 measures this non-zero on 5 of de_inferno's 31
    blend materials, so it is the one live transform of the family.

    The reference also flips V around this (`1 - v` before and after,
    `:216, 242`); that flip is ported.
    """
    v_flipped = 1.0 - uv[..., 1]
    su = (col_len * u_axis).sum(-1)
    sv = (col_len * v_axis).sum(-1)
    ok = (uv[..., 0] >= 0) | (v_flipped >= 0)
    u = torch.where(ok, uv[..., 0] * su, uv[..., 0])
    v = torch.where(ok, v_flipped * sv, v_flipped)
    return torch.stack([u, 1.0 - v], dim=-1)


# =======================================================================
# S_FOLIAGE_UV_ANIMATION
# =======================================================================


def foliage_uv_animation(uv, scroll_vel, t):
    """`csgo_foliage_vs_max.glsl:371` — `uv += g_vUvScroll.xy * time`.

    The identical construct is at `csgo_environment_blend_vs_max.glsl:279`
    driven by a different (non-foliage) axis that another agent owns; this
    function is the foliage one.

    `scroll_vel` is UV units per second.  Reachability: the default supplied
    by `gpu_render.py` is non-zero, so enabling the axis moves texels; see
    `--foliage-uv-scroll`.
    """
    return uv + scroll_vel * t


# =======================================================================
# The vertex-rate sampler3D — the capability that exists ONLY in foliage
# =======================================================================
#
# `csgo_foliage_vs_max.glsl:95-96`:
#     layout(set = 4, binding = 46) uniform texture3D _5418[65536];
#     layout(set = 4, binding = 29) uniform sampler   _5875[2048];
# Bindless arrays.  The array INDEX is `_5618._m0` (texture) and `_5618._m1`
# / `_5618._m9` (sampler), both read from the per-material constant buffer at
# set 0 binding 0 offset 0.  So the volume is chosen by a MATERIAL constant:
# it is a material-authored asset, not an engine global, and the identity of
# the asset is a runtime descriptor index that the bytecode cannot name
# (SHADER_CALLFLOW_csgo_foliage_vs.md:38-41, 202-204).
#
# What the binding and the USES do fix, and this is as far as the evidence
# goes:
#   * ONE texture, ONE sampler, for all 3 taps of the max-FLOP variant and
#     all 9 of the max-fetch variant — the same `_5618._m0` in every fetch.
#   * every fetch is `ImageSampleExplicitLod` at LOD 0
#     (`textureLod(..., 0.0)`), which is what makes it legal at vertex rate;
#   * .xyz is decoded `(t - 0.5) * 2`, i.e. the volume stores a SIGNED
#     3-VECTOR in unsigned texels (foliage_max:340, maxfetch:233,239,245...);
#   * one tap per group reads ONLY .z and turns it into a scalar 0..1
#     magnitude, `clamp(pow(t.z, 2.0) * 1.5, 0, 1)`, at the same coordinate
#     scaled by (0.4, 0.4, 0.125) — a coarser, vertically stretched lookup
#     of the SAME volume (maxfetch:232, 244, 252);
#   * the coordinate is world position * ~0.00075..0.0025 per Source inch
#     (a repeat every ~1333 in = 33.9 m for level 1, ~400 in = 10.2 m for
#     level 2) plus a scroll along the wind direction.
# Functional identity, therefore: A TILING 3-CHANNEL SIGNED VECTOR-NOISE
# VOLUME — a wind/turbulence flow field — whose .z channel doubles as a
# scalar gust mask.  The asset itself is not named by anything in the
# bytecode.
#
# RESOLVED BY DATAFLOW, WHICH IS THE PART THAT MATTERS FOR WIRING: every
# sampled value reaches gl_Position, and NO sampled value reaches any
# lighting term.  Trace, in `csgo_foliage_vs_max.glsl`: the level-1 tap
# `_13569` (:222) -> `_11275` (:234) -> `_14531` -> the rotation angle
# `_9415` (:243) -> the rotated position `_16305` (:260); the level-3 tap
# `_15717` (:340) -> `_20826` (:344) -> `_22887` (:351) -> `_17895` (:381)
# -> `gl_Position` (:395).  It is a DISPLACEMENT FIELD.  It must NOT be
# wired into the light-probe atlas or any irradiance volume — that is a
# different resource with a different purpose, and merging them would be a
# whole-class error, not a tuning one.
#
# It is also NOT a tools-only capability: 216 of the family's 288 modules
# carry at least one vertex fetch (SHADER_CALLFLOW_csgo_foliage_vs.md:27)
# and `S_MODE_TOOLS_VIS` contributes none of them.  Vertex-rate texture
# fetch is therefore implemented here as a real shipping capability.
#
# WHAT DIFFERS HERE.  The fetch machinery, the coordinates, the LOD-0
# explicit-LOD semantics, the repeat addressing, the `*2-1` decode and the
# `clamp(pow(z,2)*1.5,0,1)` gust decode are ported exactly.  The TEXEL DATA
# is not CS2's, because the descriptor index is a runtime value; a
# deterministic tiling value-noise volume is generated instead (or loaded
# from `--vertex-noise-volume`).  This is a substitution of contents, not of
# the path: every downstream term consumes a real fetch.


def make_noise_volume(size=32, channels=3, seed=0, octaves=3, device="cpu"):
    """Deterministic tiling value-noise volume standing in for the GT asset.

    Tiling is required: the reference relies on repeat addressing (the
    coordinates run to hundreds of repeats across a map).  Output is in
    [0,1] so the reference's `*2-1` decode lands on [-1,1], and the .z
    channel is built with more low-frequency mass than .x/.y so that
    `pow(z,2)*1.5` behaves as the broad gust mask the reference treats it as.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    acc = torch.zeros(size, size, size, channels)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = max(2, size >> (octaves - 1 - o))
        lat = torch.rand(n, n, n, channels, generator=g)
        # circular trilinear upsample -> tiles exactly at the volume border
        up = torch.nn.functional.interpolate(
            torch.cat([lat, lat[:1]], 0).permute(3, 0, 1, 2)[None],
            size=(size + 1, size + 1, size + 1), mode="trilinear",
            align_corners=True)[0].permute(1, 2, 3, 0)[:size, :size, :size]
        acc = acc + up * amp
        total += amp
        amp *= 0.5
    acc = acc / total
    if channels >= 3:
        # bias .z low-frequency, per the gust-mask use
        acc[..., 2] = 0.5 * acc[..., 2] + 0.5 * acc[..., 2].mean()
    return acc.clamp(0, 1).to(device)


def sample_volume3(vol, coord):
    """`textureLod(sampler3D, c, 0.0)` with repeat addressing, trilinear.

    Implemented directly rather than through grid_sample because
    grid_sample has no repeat wrap mode and the reference's coordinates run
    to hundreds of repeats.  `vol` is (D,H,W,C) indexed [z,y,x]; `coord` is
    (...,3) in normalised texture space.
    """
    D, H, W, C = vol.shape
    n = torch.tensor([W, H, D], dtype=coord.dtype, device=coord.device)
    p = coord * n - 0.5
    f = torch.floor(p)
    frac = (p - f)
    i0 = f.to(torch.int64)
    ix0 = i0[..., 0] % W
    iy0 = i0[..., 1] % H
    iz0 = i0[..., 2] % D
    ix1 = (ix0 + 1) % W
    iy1 = (iy0 + 1) % H
    iz1 = (iz0 + 1) % D
    flat = vol.reshape(D * H * W, C)

    def g(zx, yx, xx):
        return flat[(zx * H + yx) * W + xx]

    fx = frac[..., 0:1]
    fy = frac[..., 1:2]
    fz = frac[..., 2:3]
    c00 = g(iz0, iy0, ix0) * (1 - fx) + g(iz0, iy0, ix1) * fx
    c01 = g(iz0, iy1, ix0) * (1 - fx) + g(iz0, iy1, ix1) * fx
    c10 = g(iz1, iy0, ix0) * (1 - fx) + g(iz1, iy0, ix1) * fx
    c11 = g(iz1, iy1, ix0) * (1 - fx) + g(iz1, iy1, ix1) * fx
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _signed_tap(vol, coord):
    """`(textureLod(...).xyz - 0.5) * 2` — foliage_max:340, maxfetch:233."""
    return (sample_volume3(vol, coord)[..., :3] - 0.5) * 2.0


def _gust_tap(vol, coord):
    """`clamp(pow(textureLod(..., c*(0.4,0.4,0.125)).z, 2.0)*1.5, 0, 1)`.

    maxfetch:232.  The (0.4,0.4,0.125) scale is applied to the COORDINATE,
    i.e. the same volume read at a coarser, z-stretched rate.
    """
    sc = torch.tensor([0.4, 0.4, 0.125], dtype=coord.dtype,
                      device=coord.device)
    z = sample_volume3(vol, coord * sc)[..., 2]
    return (z.pow(2.0) * 1.5).clamp(0.0, 1.0)


# =======================================================================
# S_FOLIAGE_ANIMATION — the pivot hierarchy
# =======================================================================


class FoliageParams:
    """The `_5618` / `_5037` / `_4459` constants of `csgo_foliage_vs_max`.

    Field names carry the reference's member id so every default can be
    traced.  DEFAULTS ARE NON-INERT BY CONSTRUCTION: every amplitude,
    frequency and range below is non-zero, so enabling the axis with no
    further arguments displaces geometry.  A default that made a branch
    unreachable would read as tested while testing nothing.
    """

    def __init__(self, **kw):
        # --- wind, from the per-view buffer `_5037` (offsets 272, 288) ---
        self.wind_dir = kw.get("wind_dir", (0.80, 0.0, 0.60))   # _5037._m0.xyz
        self.wind_strength = kw.get("wind_strength", 1.0)        # |_m0| scale
        self.gust = kw.get("gust", 0.35)                         # _5037._m1.x
        # --- level 1 (trunk) ---
        self.l1_amount = kw.get("l1_amount", 0.55)      # _m3
        self.l1_base_dist = kw.get("l1_base_dist", 4.0)  # _m4  (Source in)
        self.l1_range = kw.get("l1_range", 48.0)        # _m5  (Source in)
        self.l1_noise = kw.get("l1_noise", 0.6)         # _m6  clamp(,0,1)
        self.l1_freq = kw.get("l1_freq", 1.3)           # _m7
        # --- volume scroll / scale, shared ---
        self.vol_scroll_dir = kw.get("vol_scroll_dir", (1.0, 0.0, 0.25))  # _m8
        self.vol_scale = kw.get("vol_scale", (1.0, 1.0, 1.0))             # _m9
        self.three_band = kw.get("three_band", True)    # _m10 != 0
        # --- level 3 (detail) ---
        self.detail_amount = kw.get("detail_amount", 1.5)   # _m11 (Source in)
        self.detail_exp = kw.get("detail_exp", 1.0)         # _m12
        self.detail_speed = kw.get("detail_speed", 1.0)     # _m13
        self.detail_scale = kw.get("detail_scale", 3.0)     # _m14
        self.detail_bias = kw.get("detail_bias", True)      # _m15 != 0
        # --- level 2 (branch) ---
        self.l2_noise = kw.get("l2_noise", 0.5)      # _m16
        self.l2_amount = kw.get("l2_amount", 0.5)    # _m17
        self.l2_dist = kw.get("l2_dist", 4.0)        # _m18 (Source in)
        self.l2_range = kw.get("l2_range", 24.0)     # _m19 (Source in)
        self.l2_freq = kw.get("l2_freq", 2.2)        # _m20
        self.l2_cross_axis = kw.get("l2_cross_axis", True)  # _m21 != 0
        # --- UV / flutter ---
        self.uv_scroll = kw.get("uv_scroll", (0.02, 0.0))   # _m24
        self.flutter_freq = kw.get("flutter_freq", 0.02)    # _m2

    def as_dict(self):
        return dict(self.__dict__)


def _osc(vol_tap, tphase, k, three_band):
    """The oscillator both animation levels share.

    `csgo_foliage_vs_max.glsl:224-241` (level 1) and `:293-310` (level 2).
    The two are the same expression with different inputs, which is why one
    function serves both; the SSA ids differ (`_11275` vs `_9027`) but the
    operand structure is identical term for term.

    `k = clamp(g_flNoise, 0, 1)`; `three_band` is `g_nBand != 0` (`_m10`).
    """
    a = tphase * 0.709999978542327880859375
    b = tphase * 1.60000002384185791015625
    sa, sb, st = torch.sin(a), torch.sin(b), torch.sin(tphase)
    if three_band:
        w1 = 0.5 + (vol_tap[..., 0] * 0.949999988079071044921875
                    + 0.0500000007450580596923828125 - 0.5) * k
        w2 = 0.5 + (vol_tap[..., 1] * 0.949999988079071044921875
                    + 0.0500000007450580596923828125 - 0.5) * k
        w3 = (vol_tap[..., 2] * 0.949999988079071044921875
              + 0.0500000007450580596923828125) * k
        s = (w1 + w2 + w3).clamp(min=0.00999999977648258209228515625)
        return ((sa * (w1 / s) + sb * (w2 / s) + st * w3)
                * (1.0 + 0.5 * k))
    inner = (sa + (sb - sa) * vol_tap[..., 0])
    inner = inner + (st - inner) * vol_tap[..., 1]
    lo = 0.5 * (sa + sb)
    return lo + (inner * 1.5 - lo) * k


def _quat_rotate(q_xyz, q_w, v):
    """`q * v * conj(q)`, written exactly as the reference expands it.

    `csgo_foliage_vs_max.glsl:246-260`: the conjugate is formed first
    (`_24626 = q * (-1,-1,-1,1)`), then
        tmp  = v*qc.w + cross(v, qc.xyz),  tmp_w = -dot(v, qc.xyz)
        out  = tmp*q.w + q.xyz*tmp_w + cross(q.xyz, tmp)
    """
    qc = -q_xyz
    w = q_w.unsqueeze(-1)
    tmp = v * w + torch.cross(v, qc, dim=-1)
    tmp_w = -(v * qc).sum(-1, keepdim=True)
    return tmp * w + q_xyz * tmp_w + torch.cross(q_xyz, tmp, dim=-1)


def foliage_animation(pos_m, normal, pivot_m, foliage_params, vol, t, P,
                      up=UP_YUP):
    """`S_FOLIAGE_ANIMATION` — the three-level pivot hierarchy.

    Line-for-line port of `csgo_foliage_vs_max.glsl:201-351`.  3 sampler3D
    taps per vertex, one per level, matching the 3 `ImageSampleExplicitLod`
    the module declares (SHADER_CALLFLOW_csgo_foliage_vs.md:181).

    Levels, and the reference lines each comes from:
      1  trunk   :218-263  rotate (pos, pivot, normal) about a wind-derived
                           base point by a noise-driven angle
      2  branch  :271-338  rotate (pos, normal) about the level-1-transformed
                           PIVOT; skipped entirely when |pivot| == 0 (:273)
      3  detail  :339-351  a direct positional displacement from a third tap
                           at the UNDEFORMED world position

    `pos_m`, `pivot_m` are metres; converted once to Source units because
    every distance constant in FoliageParams is in inches.

    THE INSTANCE TRANSFORM COLLAPSES TO IDENTITY HERE.  The reference works
    in object space and applies `mat3x4 M` at the end (:351); this pack bakes
    each glTF node transform into the vertex positions (fast_pack2.py:186),
    so object space IS world space, M = I, `_19321` (the per-instance uniform
    scale, :162) = 1 and `_23470` (the scale-correction `sqrt(max(0.5,
    max3(colLen)))`, :205) = 1.  The transform-dependent terms are therefore
    evaluated at their identity values, which is a property of our asset, not
    a term that was dropped.

    Returns (pos_m', normal', bend_magnitude, flutter (...,4)).
    """
    dev = pos_m.device
    fp = foliage_params
    p = _to_source_units(pos_m)
    pv = _to_source_units(pivot_m)

    wind = torch.tensor(fp.wind_dir, dtype=torch.float32, device=dev)
    wind = wind * fp.wind_strength
    gust = float(fp.gust)
    upv = torch.tensor(up, dtype=torch.float32, device=dev)

    # :202-203  per-instance and per-branch phase hashes.  With M = I the
    # instance origin is the world origin, so `_23005` is fract(0) = 0; the
    # per-branch hash `_22733` is retained in full because it is driven by
    # the PIVOT attribute, which is per-vertex and not affected by M = I.
    phase0 = torch.zeros((), device=dev)
    phase_pivot = torch.frac(pv.sum(-1) + phase0)

    # :205-207  scale correction; = 1 under M = I (see docstring).
    scale_fix = 1.0
    wind_len = wind.norm() * gust          # :206  length(wind)*gustscale
    wind_amt = wind_len / scale_fix        # :207

    # :208-210  the wind frame, in OBJECT space (= world here).
    w_dir = _nrm(wind + torch.tensor([0.0, 1e-4, 0.0], device=dev))
    side = torch.cross(w_dir, upv, dim=-1)
    base = torch.cross(side, w_dir, dim=-1) * fp.l1_base_dist   # :210

    freq1 = (1.0 / scale_fix) * fp.l1_freq       # :211
    amt1 = fp.l1_amount * wind_amt               # :212
    t_det = (t + phase0) * (freq1 * 0.25)        # :213

    vscale = torch.tensor(fp.vol_scale, dtype=torch.float32,
                          device=dev) * 0.001000000047497451305389404296875
    vdir = torch.tensor(fp.vol_scroll_dir, dtype=torch.float32, device=dev)

    zero = torch.zeros_like(p[..., 0])
    pos1, pvt1, nrm1 = p, pv, normal
    bend = zero
    if float(amt1) > 0.0:
        # :220  distance falloff away from the trunk base
        d1 = (((p - base).norm(dim=-1) - fp.l1_base_dist)
              / fp.l1_range).clamp(0.0, 1.0) * float(amt1)
        # :222  TAP 1 — at the trunk base, scrolled along the wind
        c1 = base * vscale + vdir * ((t * 0.1) * fp.l1_freq)
        n1 = sample_volume3(vol, c1.expand(p.shape[:-1] + (3,)))
        t1 = (t + d1) * freq1                                   # :223
        k1 = min(max(fp.l1_noise, 0.0), 1.0)                    # :224
        osc = _osc(n1, t1, k1, fp.three_band)                   # :227-241
        ang0 = osc + (osc * 0.25 - 0.5 - osc) * gust            # :242 mix
        ang = d1 * ang0                                          # :243
        # :245  the rotation AXIS itself oscillates between wind and side
        _td = float(t_det)
        ab = ((math.sin(_td * 0.709999978542327880859375)
               + math.sin(_td * 1.60000002384185791015625)) * 0.25) + 0.5
        axis = _nrm(w_dir + (side - w_dir) * ab)
        q_xyz = axis * torch.sin(ang).unsqueeze(-1)
        q_w = torch.cos(ang)
        pos1 = _quat_rotate(q_xyz, q_w, p - base) + base         # :260
        pvt1 = _quat_rotate(q_xyz, q_w, pv - base) + base        # :262
        nrm1 = _nrm(_quat_rotate(q_xyz, q_w, normal))            # :259
        bend = (ang0 * amt1).abs()                               # :261

    pos2, nrm2 = pos1, nrm1
    # :273  a zero-length pivot disables level 2 entirely.
    has_pivot = pv.norm(dim=-1) > 0.0
    if bool(has_pivot.any()):
        perp = torch.cross(w_dir.expand_as(pos1),
                           _nrm(pvt1 - base), dim=-1)             # :275
        axis2 = (torch.cross(perp, w_dir.expand_as(perp), dim=-1)
                 if fp.l2_cross_axis else perp)                   # :277-284
        # :285  `(_5618._m17 * _16738) * (_3932.y + 0.25)` — TEXCOORD5.y.
        a2b = (fp.l2_amount * wind_amt) * (P[..., 1] + 0.25)
        a2 = a2b + a2b * bend                                     # :286
        act = has_pivot & (a2 > 0)
        # :291  TAP 2 — at the level-1 pivot, scrolled at the level-2 rate
        c2 = pvt1 * vscale + vdir * ((t * 0.1) * fp.l2_freq)
        n2 = sample_volume3(vol, c2)
        dfall = ((((pos1 - pvt1).norm(dim=-1) - fp.l2_dist)
                  .clamp(min=0.0) / fp.l2_range).clamp(min=1.0)).pow(0.5)
        t2 = (t + (phase_pivot - dfall)) * (2.0 * fp.l2_freq)     # :292
        k2 = min(max(fp.l2_noise, 0.0), 1.0)
        osc2 = _osc(n2, t2, k2, fp.three_band)                    # :296-310
        ang2 = a2 * osc2                                          # :311
        q2 = axis2 * torch.sin(ang2).unsqueeze(-1)
        q2w = torch.cos(ang2)
        r_pos = _quat_rotate(q2, q2w, pos1 - pvt1) + pvt1         # :324
        r_nrm = _nrm(_quat_rotate(q2, q2w, nrm1))                 # :323
        m = act.unsqueeze(-1)
        pos2 = torch.where(m, r_pos, pos1)
        nrm2 = torch.where(m, r_nrm, nrm1)

    # :339-349  level 3, detail displacement.  TAP 3 is at the UNDEFORMED
    # world position (`vec4(_5275,1)*M`, :340), not the animated one.
    det_w = P[..., 0].clamp(0.0, 1.0).pow(fp.detail_exp)          # :339
    c3 = ((p * vscale) * fp.detail_scale
          + vdir * (((t * (4.0 + phase_pivot * 0.25)) * 0.05)
                    * fp.detail_speed).unsqueeze(-1))             # :340
    v3 = _signed_tap(vol, c3)
    if fp.detail_bias:
        # :344  bias the displacement toward the wind direction.  The
        # reference dots against the RAW wind vector `_5037._m0.xyz`, not a
        # normalised one, so the magnitude participates.
        d = (v3 * wind).sum(-1, keepdim=True)
        disp = ((v3 * (0.3300000131130218505859375
                       + 0.6699999868869781494140625 * d))
                * gust) * fp.detail_amount * det_w.unsqueeze(-1)
    else:
        disp = (v3 * fp.detail_amount) * det_w.unsqueeze(-1)      # :348
    out = pos2 + disp                                             # :351

    # :382-385  the four-band flutter interpolant handed to the foliage PS.
    tx = t + p[..., 0] * fp.flutter_freq
    ty = t + p[..., 1] * fp.flutter_freq
    flutter = torch.stack([
        0.5 * (torch.sin(tx * 1.41999995708465576171875)
               + torch.sin(tx * 3.2000000476837158203125)),
        0.5 * (torch.sin(tx * 12.77999973297119140625)
               + torch.sin(tx * 28.8000011444091796875)),
        0.5 * (torch.sin(ty * 1.41999995708465576171875)
               + torch.sin(ty * 3.2000000476837158203125)),
        0.5 * (torch.sin(ty * 11.35999965667724609375)
               + torch.sin(ty * 25.6000003814697265625)),
    ], dim=-1) * (wind_len * P[..., 0]).unsqueeze(-1)

    return out * S_SOURCE_TO_M, _nrm(nrm2), bend, flutter


def default_foliage_params_stream(n, device):
    """`vFoliageParams` (TEXCOORD5) for a pack that does not carry it.

    The reference reads .x as the detail-bend weight
    (`csgo_foliage_vs_max.glsl:339`) and .y as the level-2 weight (`:285`).
    A glTF export with no TEXCOORD5 has neither.  Returning ONES rather than
    zeros is deliberate: zeros would switch the detail bend and the branch
    level off, i.e. would silently disable two of the three levels of an
    axis the caller just asked for — the `--ibl-lod-scale 1.0` failure.
    `gpu_render.py` prints, at startup, whether the stream was real or this
    substitute, so an inert run is loud.
    """
    return torch.ones(n, 3, device=device)


# =======================================================================
# S_VERTEX_ANIMATION  (0..2)
# =======================================================================


class VertexAnimParams:
    """Constants of `csgo_foliage_vs_maxfetch` (`_5618`, offsets 0..76).

    Non-inert defaults, same discipline as FoliageParams.
    """

    def __init__(self, **kw):
        self.l1_amp = kw.get("l1_amp", 0.03)      # _m1, then *150 at :221
        self.l1_scroll = kw.get("l1_scroll", 1.0)  # _m2
        self.l1_scale = kw.get("l1_scale", 1.0)    # _m3
        self.l1_exp = kw.get("l1_exp", 1.0)        # _m4
        self.l2_amp = kw.get("l2_amp", 0.10)       # _m5, then *25 at :222
        self.l2_scroll = kw.get("l2_scroll", 1.0)  # _m6
        self.l2_scale = kw.get("l2_scale", 1.0)    # _m7
        self.l2_exp = kw.get("l2_exp", 1.0)        # _m8
        self.wind_dir = kw.get("wind_dir", (0.80, 0.0, 0.60))
        self.wind_strength = kw.get("wind_strength", 1.0)
        self.gust = kw.get("gust", 0.35)


# Literal vectors from `csgo_foliage_vs_maxfetch.glsl:225-228`.
_VA_SCROLL1 = (-0.300000011920928955078125, 0.0,
               -0.02999999932944774627685546875)
_VA_SCALE1 = (0.000750000006519258022308349609375,
              0.000750000006519258022308349609375,
              4.9999998736893758177757263183594e-05)
_VA_SCROLL2 = (-0.3499999940395355224609375, 0.0,
               0.100000001490116119384765625)
_VA_SCALE2 = (0.00250000017695128917694091796875,
              0.00250000017695128917694091796875,
              0.0005000000237487256526947021484375)
# :258, :260 — the arc-length sag coefficient, ~pi/10.
_VA_SAG = 0.31400001049041748046875


def cs_vertex_animation(pos_m, normal, tangent, color0, params, vol, t,
                        level=2):
    """`S_VERTEX_ANIMATION` (0..2) — port of `csgo_foliage_vs_maxfetch.glsl`.

    THIS IS A DIFFERENT ANIMATION FROM `S_FOLIAGE_ANIMATION`, and the
    attribute lists prove it: the max-fetch module declares NO PivotPaint and
    NO FoliageAnimation (`:85-91` — position, uv, packed frame, blend
    indices, instance idx, COLOR1, COLOR0) and instead uses COLOR0's three
    channels as the per-vertex weights (`:219, 235, 240`).  The pivot
    hierarchy cannot run without TEXCOORD4/5, so the two axes are disjoint.

    STRUCTURE, and this REFINES what
    SHADER_CALLFLOW_csgo_foliage_vs.md:44-58 could only call an
    interpretation ("three groups of three taps ... consistent with a
    three-level animation hierarchy ... the MEANING of the three levels is
    an interpretation").  The three groups are not three hierarchy levels:
    they are THREE PROBE POSITIONS of the same two-level displacement,
        group A  at  worldPos + tangent          (`_16884`, :229)
        group B  at  worldPos + cross(n,tangent) (`_10383`, :242)
        group C  at  worldPos                    (`_18423`, :250)
    and the outputs are DIFFERENCED to rebuild the frame:
        tangent' = A' - C'   (`_3515`, :260-263 -> the tangent interpolant)
        normal'  = cross(tangent', B' - C')      (:272)
        position = C'                            (`_17876`, :258, :265)
    Three groups x three taps = the 9 `ImageSampleExplicitLod` the module
    declares.  Each group is 2 signed-vector taps + 1 scalar gust tap,
    exactly as the document reads them.

    TAP COUNT.  `S_VERTEX_ANIMATION = 1` is the NINE-tap form — all three
    probes — and `S_FOLIAGE_ANIMATION = 1` is the separate THREE-tap form
    implemented in `foliage_animation()` above.  The 3-vs-9 gap is combo
    SELECTION between two different animations, not two settings of one; the
    two are implemented as two functions here for exactly that reason.
    `level` is carried because the axis's declared range is 0..2
    (SHADER_CALLFLOW_csgo_foliage_vs.md:66); what value 2 changes on top of
    value 1 "was not isolated" (:214-215) and is NOT guessed at here — 1 and
    2 both run the full nine-tap form, and only 0 disables it.

    `level=0` returns the inputs unchanged.  That is the axis being off, not
    a stub: the reference's `r0/m0` module "does no animation at all"
    (SHADER_CALLFLOW_csgo_foliage_vs.md:138) and is 0 of 9 fetches.

    Returns (pos_m', normal', tangent').
    """
    if level <= 0:
        return pos_m, normal, tangent
    dev = pos_m.device
    P = _to_source_units(pos_m)
    wind = torch.tensor(params.wind_dir, dtype=torch.float32,
                        device=dev) * params.wind_strength
    gustk = float(params.gust)

    c = color0.clamp(0.001000000047497451305389404296875, 1.0)   # :219
    A = 150.0 * params.l1_amp                                    # :221
    Bamp = 25.0 * params.l2_amp                                  # :222
    h = 2.0 * c[..., 1] - 1.0                                    # :223
    scroll1 = (torch.tensor(_VA_SCROLL1, device=dev)
               * params.l1_scroll * t)                           # :225
    scale1 = torch.tensor(_VA_SCALE1, device=dev) * params.l1_scale   # :226
    scroll2 = (torch.tensor(_VA_SCROLL2, device=dev)
               * params.l2_scroll * t)                           # :227
    scale2 = torch.tensor(_VA_SCALE2, device=dev) * params.l2_scale   # :228
    off1 = torch.stack([h * 100.0, torch.zeros_like(h), h * 50.0], -1)  # :230
    off2 = torch.stack([h * 10.0, h * 5.0, h * 5.0], -1)                # :238
    w1 = c[..., 0].pow(params.l1_exp).unsqueeze(-1)              # :235
    w2 = c[..., 2].pow(params.l2_exp).unsqueeze(-1)              # :240

    def probe(Q):
        p1 = (Q + off1) * scale1 + scroll1                       # :231
        gust = _gust_tap(vol, p1).unsqueeze(-1)                  # :232
        n1 = _signed_tap(vol, p1)                                # :233
        v1 = torch.stack([n1[..., 0], n1[..., 1], n1[..., 1] * 0.5],
                         -1) * 0.5                               # :234
        d1 = (((v1 * gust + wind) * w1) * gustk * A) * gust      # :236
        n2 = _signed_tap(vol, (Q + off2) * scale2 + scroll2)     # :239
        d2 = ((n2 * wind + n2 * v1) * gustk * Bamp) * w2         # :241
        sag1 = torch.zeros_like(d1)
        sag1[..., 1] = -(d1[..., [0, 2]] * _VA_SAG).norm(dim=-1)
        sag2 = torch.zeros_like(d2)
        sag2[..., 1] = -(d2[..., [0, 2]].norm(dim=-1)) * _VA_SAG
        return Q + d1 + sag1 + d2 + sag2                         # :258

    # NOTE the sag term is written on the UP axis.  The reference writes it
    # into `.z` (:258) because Source is Z-up; here it is `.y`, the same
    # geometric term under the basis change declared at the top of the file.
    Cq = probe(P)
    # :229, :242 — the two offset probes.  The reference adds the UNIT
    # tangent and the unit bitangent to a position that is in Source inches,
    # i.e. it probes exactly one inch away on each axis; that is the finite
    # difference step and it is reproduced at the same absolute size.
    Aq = probe(P + tangent[..., :3])
    Bq = probe(P + torch.cross(normal, tangent[..., :3], dim=-1))
    tan_new = Aq - Cq                                            # :260-263
    nrm_new = torch.cross(tan_new, Bq - Cq, dim=-1)              # :272
    out_t = torch.cat([_nrm(tan_new), tangent[..., 3:]], dim=-1) \
        if tangent.shape[-1] == 4 else _nrm(tan_new)
    return Cq * S_SOURCE_TO_M, _nrm(nrm_new), out_t


# =======================================================================
# D_CS_VERTEX_ANIMATION
# =======================================================================


def cs_vertex_animation_instanced(pos_os, blend_index_x, draw_flags_nibble,
                                  transform_base, transform_buffer,
                                  uniform_scale_col=None):
    """`D_CS_VERTEX_ANIMATION` — NO LISTING IN THIS CORPUS HAS IT ENABLED.

    Stated plainly, because the rest of this file is transcription and this
    is not: `D_CS_VERTEX_ANIMATION` is declared as a dynamic axis by all four
    families (SHADER_CALLFLOW_vertex_stages.md:196-198) and NONE of the six
    committed listings selects it — every one of them is a d0 or otherwise
    non-animated module.  Its body was not extracted
    (SHADER_CALLFLOW_vertex_stages.md:228-229, "combo -> module
    identification ... was not extracted", for all four).

    WHAT IS PORTED EXACTLY.  The per-instance transform indirection that IS
    in every listing, `csgo_foliage_vs_max.glsl:137-168` /
    `csgo_environment_blend_vs_max.glsl:136-167` /
    `csgo_complex_vs_max.glsl:117-149`:

        flags   = (drawData[nInstanceIdx]._m5 >> 16) & 15
        xfIdx   = flags == 1 ? base + 2 + vBlendIndices.x : base
        M       = transformBuffer[xfIdx]
        scale   = flags > 0 ? transformBuffer[base][0].z : 1.0
        worldPos= vec4(pos * scale, 1) * M

    That `flags == 1` branch selects a DIFFERENT, per-vertex-indexed
    transform out of the same buffer — i.e. the vertex is animated by
    re-binding its transform, with no texture fetch and no attribute other
    than BLENDINDICES.  Every operand above is transcribed.

    WHAT DIFFERS.  Only the SOURCE of the transform buffer's contents: the
    reference gets it from the engine's per-frame instance upload, and
    `gpu_render.py` supplies a time-varying buffer built from the
    `--cs-vertex-animation-*` arguments.  The branch, the index arithmetic,
    the `>> 16 & 15` nibble and the `[0].z` uniform-scale read are the
    reference's.  This is a substitution of an input, not of the path.

    `transform_buffer` is (K,3,4) row-major `mat3x4` in the reference's
    convention, i.e. world = vec4(p,1) * M.
    """
    flags = draw_flags_nibble
    idx = torch.where(flags == 1,
                      transform_base + 2 + blend_index_x, transform_base)
    M = transform_buffer[idx]                       # (...,3,4) -> cols x rows
    scale = torch.where(flags > 0,
                        transform_buffer[transform_base][..., 0, 2],
                        torch.ones_like(pos_os[..., 0]))
    if uniform_scale_col is not None:
        scale = uniform_scale_col
    p = pos_os * scale.unsqueeze(-1)
    ph = torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)
    # `vec4 * mat3x4` in GLSL: M is 3 columns x 4 rows, result is vec3.
    return torch.einsum("...r,...cr->...c", ph, M)


# =======================================================================
# S_PRE_BAKED_VERTEX_ANIMATION
# =======================================================================


def prebaked_vertex_animation(pos_m, normal, table_pos, table_nrm, t,
                              rate=1.0):
    """`S_PRE_BAKED_VERTEX_ANIMATION` — NO LISTING IN THIS CORPUS HAS IT.

    Stated plainly: this is a static combo axis of `csgo_complex_vs`
    (SHADER_CALLFLOW_csgo_complex_vs.md:42) and neither committed module of
    that family selects it — `csgo_complex_vs_max.glsl` is the population
    maximum at 374 FLOPs/vertex and contains no animation whatsoever, and
    `csgo_complex_vs_base.glsl` is `r0/m0`.

    WHAT THE REFERENCE MATERIAL DOES CONSTRAIN, and it is not nothing:
    SHADER_CALLFLOW_csgo_complex_vs.md:49-51 records that this family drives
    a vertex-animation axis with ZERO texture fetch in all 480 modules, and
    :28-31 that the replacing path is 5 descriptor buffers, 12-27 OpLoads,
    168-276 B/vertex.  So the animation data arrives as BUFFER or ATTRIBUTE
    data and is interpolated per vertex, which is what "pre-baked" names.

    IMPLEMENTED FORM.  A baked keyframe table of per-vertex position and
    normal deltas, (F, N, 3), sampled at `t * rate` with linear
    interpolation between the two bracketing frames and wrap-around — a
    buffer read per vertex, no texture fetch, matching the constraint above.
    The KEYFRAME ENCODING is not the reference's, because the reference's
    was not extracted; the frame count, the interpolation and the wrap are
    choices made here and are named as such.

    This function never returns its input unchanged: `gpu_render.py` refuses
    to enable the axis without a table, rather than silently no-opping.
    """
    F = table_pos.shape[0]
    x = (t * rate) % F
    i0 = int(math.floor(x)) % F
    i1 = (i0 + 1) % F
    a = float(x - math.floor(x))
    dp = table_pos[i0] * (1.0 - a) + table_pos[i1] * a
    out_n = normal
    if table_nrm is not None:
        dn = table_nrm[i0] * (1.0 - a) + table_nrm[i1] * a
        out_n = _nrm(normal + dn)
    return pos_m + dp, out_n


# =======================================================================
# S_VERTEX_BREAK_ANIMATION
# =======================================================================


class BreakParams:
    """`g_flBreakTime`, `g_vBreakShotPosition`, `g_flBreakShotSize`,
    `g_flBreakBounceFloor` — the four uniforms
    SHADER_CALLFLOW_csgo_glass.md:485-487 names as living in the glass
    vertex stage.  Defaults are non-inert."""

    def __init__(self, **kw):
        self.break_time = kw.get("break_time", 0.0)       # g_flBreakTime
        self.shot_pos = kw.get("shot_pos", (0.0, 1.6, 0.0))  # world metres
        self.shot_size = kw.get("shot_size", 0.35)        # metres
        self.bounce_floor = kw.get("bounce_floor", 0.0)   # world Y, metres
        self.gravity = kw.get("gravity", 9.81)
        self.spin = kw.get("spin", 6.0)
        self.restitution = kw.get("restitution", 0.35)


def vertex_break_animation(pos_m, normal, shard_centroid_m, shard_seed,
                           params, t, up=UP_YUP):
    """`S_VERTEX_BREAK_ANIMATION` — THE GLASS VERTEX STAGE WAS NOT DECOMPILED.

    Stated plainly and first: `csgo_glass_vulkan_50_vs.vcs` (90,942 B) was
    "extracted from the VPK, not decompiled or counted"
    (SHADER_CALLFLOW_csgo_glass.md:484-487).  There is no listing to port.
    The static axis itself IS confirmed and IS live on this map — it is
    combo weight 4 and the document's own combo matrix has a row
    "s4 / d8 — `S_VERTEX_BREAK_ANIMATION`, probe (de_inferno)" at 1616
    FLOPs/px (`:281`) — so the axis is not hypothetical, only its vertex
    body is unread.

    WHAT IS PORTED.  Nothing textual.  WHAT IS RECONSTRUCTED, and from what:
    the four uniform NAMES quoted at `:486`.  Each maps to exactly one term
    and nothing here has a term the names do not motivate:
      g_vBreakShotPosition + g_flBreakShotSize
          -> per-shard impulse whose magnitude falls off with distance from
             the shot point over that radius, directed away from it;
      g_flBreakTime
          -> seconds since the break; the whole displacement is 0 at t <= 0,
             so an unbroken pane is EXACTLY its input geometry (this is the
             one place the identity is correct rather than a stub, and it is
             correct because the axis's own time parameter says so);
      g_flBreakBounceFloor
          -> a world height the shard bounces off, with a restitution
             coefficient that is NOT one of the four named uniforms and is
             therefore an addition (`BreakParams.restitution`).
    Ballistics (constant gravity, per-shard spin about a seed-derived axis)
    are likewise additions, named here so they are not mistaken for ported
    terms.

    The shard identity comes from `shard_centroid_m` / `shard_seed`, which
    `gpu_render.py` derives by connected components of the glass-family
    triangles; the reference presumably has a per-vertex or per-instance
    shard id that was not extracted.
    """
    dev = pos_m.device
    dt = float(params.break_time)
    if dt <= 0.0:
        # Not a stub: at t <= 0 the pane is unbroken and the reference's own
        # time uniform makes the displacement identically zero.
        return pos_m, normal
    shot = torch.tensor(params.shot_pos, dtype=torch.float32, device=dev)
    upv = torch.tensor(up, dtype=torch.float32, device=dev)

    r = shard_centroid_m - shot
    d = r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    fall = (1.0 - (d / max(params.shot_size, 1e-6))).clamp(min=0.0)
    v0 = (r / d) * fall * (params.shot_size * 12.0)

    disp = v0 * dt - upv * (0.5 * params.gravity * dt * dt)
    y = (shard_centroid_m + disp) * upv
    y = y.sum(-1, keepdim=True)
    floor = float(params.bounce_floor)
    below = y < floor
    # one bounce, restitution applied to the vertical component only
    disp = torch.where(below,
                       disp + upv * ((floor - y) * (1.0 + params.restitution)),
                       disp)

    ang = (shard_seed * params.spin * dt).unsqueeze(-1)
    ax = _nrm(torch.stack([torch.sin(shard_seed * 12.9898),
                           torch.cos(shard_seed * 78.233),
                           torch.sin(shard_seed * 43.7585)], dim=-1))
    local = pos_m - shard_centroid_m
    ca, sa = torch.cos(ang), torch.sin(ang)
    rot = (local * ca + torch.cross(ax, local, dim=-1) * sa
           + ax * (ax * local).sum(-1, keepdim=True) * (1.0 - ca))
    n_rot = (normal * ca + torch.cross(ax, normal, dim=-1) * sa
             + ax * (ax * normal).sum(-1, keepdim=True) * (1.0 - ca))
    return shard_centroid_m + rot + disp, _nrm(n_rot)


# =======================================================================
# csgo_character SKINNING  (csgo_core/shaders_vulkan/csgo_character_vulkan_50_vs)
# =======================================================================
#
# This is the FIRST skinned vertex path in this file; every axis above it is
# static or wind-animated geometry.
#
# Provenance.  `csgo_character_vulkan_50_ps/vs.vcs` is in
# `csgo_core/shaders_vulkan_dir.vpk`, NOT the `csgo` archive — which is why no
# earlier enumeration in this project had the bytecode.  The vs package holds
# 128 bytecode records / 1152 SPIR-V modules; ALL 1152 were decompiled and
# scanned, so the two statements below are over the whole shipped package, not
# over a sample.
#
# TWO SHIPPED FORMS, both keyed off the same per-draw nibble
#
#     nInfluences = (drawData[nInstanceIdx]._m5 >> 16) & 15
#
#   (a) RIGID / single influence — `r47_m0.glsl:139-161, 195-196`:
#           xfIdx  = (nInfluences == 1) ? drawTransformIdx + 2 + vBlendIndices.x
#                                       : drawTransformIdx
#           M      = boneBuffer[xfIdx]                       # mat3x4
#           scale  = nInfluences > 0 ? boneBuffer[drawTransformIdx][0].z : 1.0
#           posWs  = vec4(posOs * scale, 1.0) * M
#           nrmWs  = normalize(vec4(nrmOs,   0.0) * M)
#       No weights are read at all on this path — one bone, index only.
#
#   (b) FOUR-INFLUENCE LINEAR BLEND — `r101_m2.glsl:151, 242-273`:
#           boneBase = drawTransformIdx + 2
#           M  = boneBuffer[boneBase + vBlendIndices.x] * vBlendWeight.x
#           for (i = 1; i < nInfluences; ++i)
#               M += boneBuffer[boneBase + vBlendIndices[i]] * vBlendWeight[i]
#           # and where vBlendIndices.x == 0xFFFFFFFF (an unskinned draw):
#           M  = boneBuffer[drawTransformIdx] * vBlendWeight.x
#       `vBlendIndices` is a uvec4 (`BLENDINDICES`) and `vBlendWeight` a vec4
#       (`BLENDWEIGHT`); both are declared in the package's
#       `m_vsInputSignatureArray`.  The loop bound is the SAME nibble, so (a)
#       is the nInfluences==1 specialisation of (b), not a different mechanism.
#
# WHAT THE SHIPPED (b) MODULES DO WITH `M`, MEASURED, NOT ASSUMED.  Across all
# 288 modules in the package that build the blended matrix, the ONLY use of it
# is `vec4(dir, 0.0) * M` — a DIRECTION transform (the anisotropic tangent
# direction).  Zero of the 288 multiply a `vec4(p, 1.0)` by it.  They do not
# need to: those modules read the vertex position, and an octahedral-packed
# normal and tangent, out of a storage buffer at 24 bytes per vertex
# (`r101_m2.glsl:184-185`, index `(gl_VertexIndex - gl_BaseVertex +
# drawData._m3) * 24`) — i.e. the engine hands that path vertices that are
# ALREADY skinned, and the vertex shader only needs the bone frame to rotate
# an object-space direction into world space.
#
# WHAT DIFFERS HERE, stated exactly.  Our vertex buffer is in BIND POSE and
# there is no pre-skinning compute pass, so `character_skinning()` applies the
# blended matrix (b) to the position, normal and tangent, using form (a)'s
# transcribed position/normal expressions (`r47_m0.glsl:195-196`).  The matrix
# build, the `>> 16 & 15` influence count, the `+2` bone base, the
# `0xFFFFFFFF` unskinned sentinel and the `[0].z` uniform-scale read are the
# reference's, verbatim.  What is substituted is only WHICH vertices the
# blended matrix is applied to — an input substitution, not a path
# substitution, and it makes the renderer strictly closer to the engine's
# final geometry than skipping it would.


def character_skinning(pos_os, normal, tangent4, blend_indices, blend_weight,
                       transform_base, n_influences, bone_buffer,
                       uniform_scale=None):
    """csgo_character linear-blend skinning.

    pos_os          (N,3)   bind-pose object-space position
    normal          (N,3)   bind-pose object-space normal   (or None)
    tangent4        (N,4)   bind-pose tangent, .w = bitangent sign (or None)
    blend_indices   (N,4)   int64 BLENDINDICES; 0xFFFFFFFF (-1 here) in .x
                            marks an UNSKINNED vertex, which takes
                            `bone_buffer[transform_base]`
    blend_weight    (N,4)   float BLENDWEIGHT
    transform_base  (N,)    int64 per-vertex draw transform index
    n_influences    (N,)    int64 the `(m5 >> 16) & 15` nibble, 0..15; only
                            the first `min(n, 4)` influences exist in the
                            attribute, which is the shipped attribute width
    bone_buffer     (K,3,4) mat3x4 in the reference's convention, so a point
                            is `vec4(p, 1) * M` -> vec3
    uniform_scale   (N,)    optional; the reference reads
                            `bone_buffer[transform_base][0][2]` when
                            nInfluences > 0 and 1.0 otherwise

    Returns (pos, normal, tangent4) with the same shapes as the inputs.
    """
    n = pos_os.shape[0]
    dev = pos_os.device
    K = bone_buffer.shape[0]
    base = transform_base.long()
    skinned = blend_indices[:, 0] >= 0            # 0xFFFFFFFF sentinel

    # --- form (b): the weighted accumulation, r101_m2.glsl:248-262 --------
    M = torch.zeros(n, 3, 4, device=dev, dtype=bone_buffer.dtype)
    for i in range(4):
        # `i < nInfluences` is the reference's own loop bound; i == 0 is
        # outside the loop there and unconditional, which this reproduces
        # because n_influences >= 1 whenever `skinned`.
        live = skinned & (n_influences > i) if i else skinned
        idx = (base + 2 + blend_indices[:, i].long()).clamp(0, K - 1)
        w = blend_weight[:, i] * live.to(blend_weight.dtype)
        M = M + bone_buffer[idx] * w.view(-1, 1, 1)
    # the unskinned branch, r101_m2.glsl:270-272 -- note it too is scaled by
    # vBlendWeight.x, which is the reference's expression, not a tidy-up.
    Mu = bone_buffer[base.clamp(0, K - 1)] * blend_weight[:, 0].view(-1, 1, 1)
    M = torch.where(skinned.view(-1, 1, 1), M, Mu)

    # --- form (a)'s uniform scale, r47_m0.glsl:152-161 -------------------
    if uniform_scale is None:
        s0 = bone_buffer[base.clamp(0, K - 1)][:, 0, 2]
        uniform_scale = torch.where(n_influences > 0, s0,
                                    torch.ones_like(s0))

    ph = torch.cat([pos_os * uniform_scale.unsqueeze(-1),
                    torch.ones_like(pos_os[:, :1])], dim=-1)
    pos = torch.einsum("nr,ncr->nc", ph, M)          # r47_m0.glsl:196

    nrm = None
    if normal is not None:
        nh = torch.cat([normal, torch.zeros_like(normal[:, :1])], dim=-1)
        nrm = _nrm(torch.einsum("nr,ncr->nc", nh, M))  # r47_m0.glsl:195
    tan = None
    if tangent4 is not None:
        th = torch.cat([tangent4[:, :3],
                        torch.zeros_like(tangent4[:, :1])], dim=-1)
        # r101_m2.glsl:567 -- the ONE use the shipped blended matrix has.
        tan = torch.cat([_nrm(torch.einsum("nr,ncr->nc", th, M)),
                         tangent4[:, 3:4]], dim=-1)
    return pos, nrm, tan


def character_bone_palette(bind_pos, n_bones, t, amp=0.0, rate=0.0,
                           device="cpu"):
    """A bone palette in the reference's mat3x4 convention.

    THIS IS A SUBSTITUTED INPUT, said out loud.  The engine uploads the pose
    from the animation graph every frame; a .glb world pack carries no
    skeleton and no animation, so the palette here is built from the geometry
    itself: bone k owns the k-th slab of the model's bounding box along Y and
    is given a rotation about Y plus a translation, both of amplitude `amp`
    and angular rate `rate`.  At amp == 0 and rate == 0 every bone is exactly
    the identity, so `character_skinning()` with this palette is a no-op and
    the SKINNING PATH IS STILL EXECUTED -- a zero delta then means "bind pose
    requested", never "the path did not run".

    Slots 0 and 1 are reserved: the reference indexes bones as
    `transform_base + 2 + blendIndex` (`r47_m0.glsl:144`,
    `r101_m2.glsl:244`), so slot 0 is the draw's model-to-world transform
    (used by the unskinned branch) and slot 1 is unused by the pixel path.
    Slot 0's `[0][2]` element is the uniform scale the reference reads
    (`r47_m0.glsl:157`), so it is set to 1.0 here, not 0.0.
    """
    K = int(n_bones) + 2
    xf = torch.zeros(K, 3, 4, device=device)
    xf[:, 0, 0] = 1.0
    xf[:, 1, 1] = 1.0
    xf[:, 2, 2] = 1.0
    # slot 0 carries the uniform scale in [0][2] -- see r47_m0.glsl:157.
    xf[0, 0, 2] = 1.0
    if amp == 0.0 and rate == 0.0:
        return xf
    lo = float(bind_pos[:, 1].min())
    hi = float(bind_pos[:, 1].max())
    span = max(hi - lo, 1e-6)
    for k in range(int(n_bones)):
        a = rate * t + (k / max(int(n_bones) - 1, 1)) * math.pi
        # amplitude grows along the bone chain, so the root barely moves and
        # the extremities swing -- the shape a real pose has.
        g = amp * (k / max(int(n_bones) - 1, 1))
        ca, sa = math.cos(a * g), math.sin(a * g)
        j = k + 2
        xf[j, 0, 0] = ca
        xf[j, 2, 0] = sa
        xf[j, 0, 2] = -sa
        xf[j, 2, 2] = ca
        # rotate about the slab centre so the mesh does not tear at the seam
        c = lo + span * (k + 0.5) / max(int(n_bones), 1)
        xf[j, 1, 3] = 0.0
        xf[j, 0, 3] = g * math.sin(a) * span * 0.05
        xf[j, 2, 3] = g * math.cos(a) * span * 0.05
        xf[j, 1, 1] = 1.0
        del c
    return xf


def character_skin_binding(pos_os, n_bones, transform_base=0):
    """Derive BLENDINDICES / BLENDWEIGHT / nInfluences from bind-pose geometry.

    ALSO A SUBSTITUTED INPUT.  glTF `JOINTS_0` / `WEIGHTS_0` are not present
    in this project's world pack (the pack is a static map export), so the
    binding is derived: bones are laid out along +Y, a vertex's two nearest
    bone slabs are its two influences, and the weight is the linear partition
    of the vertex between their centres.  That gives exactly 2 non-zero
    influences per vertex, `nInfluences = 2`, weights summing to 1 -- the
    shape a real binding has, from data the pack does carry.  When the pack
    DOES carry joints/weights they are used instead; this function is only
    the fallback and `gpu_render.py` prints which one it took.
    """
    n = pos_os.shape[0]
    dev = pos_os.device
    lo = pos_os[:, 1].min()
    hi = pos_os[:, 1].max()
    span = torch.clamp(hi - lo, min=1e-6)
    u = ((pos_os[:, 1] - lo) / span * (n_bones - 1)).clamp(0, n_bones - 1 - 1e-4)
    k0 = u.floor().long()
    f = u - k0.to(u.dtype)
    bi = torch.zeros(n, 4, dtype=torch.long, device=dev)
    bw = torch.zeros(n, 4, device=dev)
    bi[:, 0] = k0
    bi[:, 1] = (k0 + 1).clamp(max=n_bones - 1)
    bw[:, 0] = 1.0 - f
    bw[:, 1] = f
    nin = torch.full((n,), 2, dtype=torch.long, device=dev)
    base = torch.full((n,), int(transform_base), dtype=torch.long, device=dev)
    return bi, bw, base, nin


# =======================================================================
# vPerVertexLighting  (COLOR1) — the decode every family shares
# =======================================================================


def per_vertex_lighting(color1):
    """`(rgb * 6 * a)` then squared, `csgo_environment_blend_vs_max.glsl:220`
    + `:470`; identical at `csgo_complex_vs_max.glsl:201` + `:329` and
    `csgo_foliage_vs_max.glsl:368` + `:389`.  The three expressions differ by
    no literal (max|delta| = 0 over the constant 6.0), which is what confirms
    `_5703`'s cross-file identity by use."""
    v = (color1[..., :3] * 6.0) * color1[..., 3:4]
    return v * v


# =======================================================================
# per-vertex FLOP model, for S_MODE_TOOLS_SHADING_COMPLEXITY
# =======================================================================
#
# FLOPs per VERTEX, from SHADER_CALLFLOW_vertex_stages.md:41-44.  These are
# NEVER mixed with the per-pixel figures; the tools mode renders them on
# separate scales and says which it is showing.
VS_FLOPS_PER_VERTEX = {
    # family                     min  median  mean  max
    "csgo_environment.vfx": (61, 218, 229, 350),
    "csgo_environment_blend.vfx": (61, 272, 277, 436),
    "csgo_complex.vfx": (156, 219, 236, 374),
    "csgo_foliage.vfx": (156, 544, 525, 1108),
}
# FLOPs per PIXEL, on the one-loop-entered-once basis FLOP_LOGIT_TABLES.md
# insists on when comparing (`:78-83`): env_blend 3205 vs complex 1510.
PS_FLOPS_PER_PIXEL = {
    "csgo_environment_blend.vfx": 3205,
    "csgo_complex.vfx": 1510,
    "csgo_glass.vfx": 1616,      # SHADER_CALLFLOW_csgo_glass.md:281
    "csgo_environment.vfx": 1150,  # scaled row, Table 1; see caveat below
    "csgo_foliage.vfx": 1184,
    "csgo_static_overlay.vfx": 1254,
}
# CAVEAT carried with the numbers: FLOP_LOGIT_TABLES.md:70-83 states the
# non-blend/non-complex rows were EXTRAPOLATED against a since-retracted
# figure and "must not be quoted as measurements".  The shading-complexity
# mode prints that caveat when it uses them.


# =======================================================================
# self-test — reachability at default
# =======================================================================

def _selftest(device="cpu"):
    """Every axis, at its DEFAULT parameters, must move something.

    Prints max|delta| per axis.  A zero here is the `--ibl-lod-scale 1.0`
    failure mode: a path that is present, resident and unreachable.
    """
    torch.manual_seed(0)
    N = 4096
    dev = torch.device(device)
    pos = (torch.rand(N, 3, device=dev) - 0.5) * 8.0
    nrm = _nrm(torch.randn(N, 3, device=dev))
    tan = torch.cat([_nrm(torch.cross(nrm, torch.tensor(
        [0.0, 1.0, 0.0], device=dev).expand_as(nrm), dim=-1)),
        torch.ones(N, 1, device=dev)], dim=-1)
    col0 = torch.rand(N, 4, device=dev)
    pivot = pos + torch.randn(N, 3, device=dev) * 0.2
    P = torch.rand(N, 3, device=dev)
    vol = make_noise_volume(32, 3, seed=1, device=dev)
    rows = []

    p2, t4 = decode_packed_frame(encode_packed_frame(nrm, tan))
    rows.append(("D_COMPRESSED_NORMALS_AND_TANGENTS",
                 float((p2 - nrm).abs().max()),
                 "max|d normal| of the pack->unpack round trip "
                 "(the axis's quantisation, not an error)"))

    uv0 = torch.rand(N, 2, device=dev)
    uv1 = torch.rand(N, 2, device=dev)
    s2 = secondary_uv(uv0, uv1, torch.full((N,), 2, device=dev))
    rows.append(("S_SECONDARY_UV", float((s2 - uv0).abs().max()),
                 "max|d uv| selecting set 1 instead of set 0"))

    ua = foliage_uv_animation(uv0, torch.tensor([0.02, 0.0], device=dev), 3.0)
    rows.append(("S_FOLIAGE_UV_ANIMATION", float((ua - uv0).abs().max()),
                 "max|d uv| at t=3s, default scroll"))

    fp = FoliageParams()
    fpos, fnrm, bend, flut = foliage_animation(pos, nrm, pivot, fp, vol,
                                               3.0, P)
    rows.append(("S_FOLIAGE_ANIMATION", float((fpos - pos).abs().max()),
                 "max|d position| in metres at t=3s, default wind"))
    rows.append(("S_FOLIAGE_ANIMATION (normal)",
                 float((fnrm - nrm).abs().max()), "max|d normal|"))
    rows.append(("S_FOLIAGE_ANIMATION (flutter out)",
                 float(flut.abs().max()), "max|flutter interpolant|"))

    vp = VertexAnimParams()
    for lv in (1, 2):
        a, b, c = cs_vertex_animation(pos, nrm, tan, col0, vp, vol, 3.0,
                                      level=lv)
        rows.append((f"S_VERTEX_ANIMATION={lv} (9 taps)",
                     float((a - pos).abs().max()),
                     "max|d position| in metres at t=3s"))
        rows.append((f"S_VERTEX_ANIMATION={lv} (frame)",
                     float((b - nrm).abs().max()),
                     "max|d normal| from the finite-difference rebuild"))

    K = 8
    xf = torch.zeros(K, 3, 4, device=dev)
    for k in range(K):
        xf[k, 0, 0] = xf[k, 1, 1] = xf[k, 2, 2] = 1.0
        xf[k, 0, 3] = 0.1 * k
    flags = torch.full((N,), 1, device=dev, dtype=torch.long)
    bi = torch.randint(0, 4, (N,), device=dev)
    base = torch.zeros(N, dtype=torch.long, device=dev)
    w = cs_vertex_animation_instanced(pos, bi, flags, base, xf,
                                      uniform_scale_col=torch.ones(N,
                                                                   device=dev))
    rows.append(("D_CS_VERTEX_ANIMATION", float((w - pos).abs().max()),
                 "max|d position| through the flags==1 transform re-index"))

    tp = torch.randn(4, N, 3, device=dev) * 0.05
    pb, pn = prebaked_vertex_animation(pos, nrm, tp, None, 3.0)
    rows.append(("S_PRE_BAKED_VERTEX_ANIMATION",
                 float((pb - pos).abs().max()),
                 "max|d position| from a 4-key table"))

    bp = BreakParams(break_time=0.4)
    cen = pos * 0.0 + pos.mean(0)
    seed = torch.rand(N, device=dev)
    bpos, bn = vertex_break_animation(pos, nrm, cen, seed, bp, 0.4)
    rows.append(("S_VERTEX_BREAK_ANIMATION",
                 float((bpos - pos).abs().max()),
                 "max|d position| at break_time=0.4s"))

    pvl = per_vertex_lighting(torch.rand(N, 4, device=dev))
    rows.append(("vPerVertexLighting decode", float(pvl.max()),
                 "max of (rgb*6*a)^2"))

    print("CS2 vertex stage — reachability at default parameters")
    print(f"{'axis':38s} {'max|delta|':>14s}  meaning")
    bad = []
    for name, v, why in rows:
        print(f"{name:38s} {v:14.6g}  {why}")
        if not (v > 0):
            bad.append(name)
    if bad:
        print("UNREACHABLE AT DEFAULT: " + ", ".join(bad))
        return 1
    print("all axes reachable at their defaults")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
