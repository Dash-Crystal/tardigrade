"""GLB -> GPU pack with Source 2 material semantics preserved.

Beyond fast_pack.py this carries what CS2 actually shades with, all of
which VRF already puts in the glTF but a naive baseColor read discards:

  * layer 2 of `csgo_environment_blend` materials (365 of 923 here) —
    g_tColor2, resolved by matching the vmat's .vtex name against the
    glTF image names;
  * COLOR_0 vertex colours (the per-vertex blend weight);
  * TEXCOORD_1 (layer-2 UV set, F_SECONDARY_UV);
  * per-layer colour correction (tint / brightness / contrast /
    saturation), which CS2 applies in-shader and which desaturates the
    whole frame if skipped;
  * vertex normals + TANGENT for normal mapping;
  * alphaMode / alphaCutoff.

fast_pack_normals.py additionally emits the PBR surface set that the
baseline pack throws away entirely:

  * TANGENT (VEC4, w = bitangent sign), world-transformed, per vertex —
    present on all 6795 primitives;
  * normalTexture (918/923 materials) plus the blend layer's g_tNormal2
    (365/923, resolved by vtex name exactly like g_tColor2);
  * metallicRoughnessTexture (743), occlusionTexture (213),
    emissiveTexture (51);
  * a per-material table (indexed by a new per-face `face_matid`)
    instead of per-face float replication — face_params alone is already
    ~516 MB at 8.6 M faces, so five more slots go in a table, not a
    per-face array.

These live in a SEPARATE texture array `aux` at --aux-size (default 512)
because they are linear-space data (no sRGB decode) and because at
960x540 the normal/MR/AO maps are minified far past 1024.

fast_pack_shaders.py adds the vmat texture slots that have no glTF
equivalent at all, resolved the same way g_tColor2 already was (vtex
name -> glTF image name), plus the scalar/flag block each one needs:

  * g_tHeight1 (700) / g_tHeight2 (365) — Source 2 blends two layers by
    HEIGHT, not by a linear lerp, with g_flHeightMapScale1/2,
    g_flHeightMapZeroPoint1/2, g_flBlendSoftness2 and the
    F_BLEND_EFFECTS_2 bevel/border block (g_flBevel*, g_flBorder*,
    g_vBorderTint2, g_fBorderRoughness2);
  * g_tSharedColorOverlay (167, only 3 distinct images — a map-wide
    grunge pass) with g_flOverlayBrightness/DarknessContrast,
    g_nColorOverlayMode, g_nColorOverlayTintMask;
  * g_tAmbientOcclusion (212) — the vmat AO slot, 4x denser than the
    glTF occlusionTexture VRF wrote;
  * g_tTintMask (104) + F_TINT_MASK with g_fTintMaskBrightness/Contrast;
  * g_tTransmissiveColor (67) + F_TRANSMISSIVE_BACKFACE_NDOTL /
    F_USE_ALBEDO_FOR_TRANSMISSIVE — foliage translucency;
  * g_tMetalness (43) + F_METALNESS_TEXTURE, g_flMetalness;
  * g_tSelfIllumMask (51) + F_SELF_ILLUM, g_flSelfIllumAlbedoFactor;
  * F_RENDER_BACKFACES (89), so the renderer can stop one-sided-lighting
    a surface CS2 draws from both sides.

All of the above is checked in the export: every one of these vtex names
resolves against a glTF image (0 unresolved across all 923 materials),
and every g_v*TexCoordScale/Offset/Center and g_flTexCoordRotation is
identity, so no per-slot UV transform is needed.

  venv/bin/python fast_pack_shaders.py --glb world.glb --out world_s.pt
"""

import argparse
import concurrent.futures as futures
import json
import os
import struct
import time

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--glb", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tex-size", type=int, default=1024)
parser.add_argument("--aux-size", type=int, default=512)
parser.add_argument("--premultiply-bug", action="store_true",
                    help="reproduce the Pillow-12 alpha-premultiplying "
                         "resize the earlier packs were built with, so a "
                         "pack can be compared against the published "
                         "1.5304 / 2.8296 baseline; see _resize_rgba")
parser.add_argument("--no-alpha-bleed", action="store_true",
                    help="reproduce the pre-alpha_bleed colour pack "
                         "(world3.pt) so PBR deltas are measured against "
                         "the published 1.5477 / 2.9647 baseline")
args = parser.parse_args()
t0 = time.time()
device = torch.device("cuda")

raw = open(args.glb, "rb").read()
offset, gltf, bin_chunk = 12, None, None
while offset < len(raw):
    clen, ctype = struct.unpack_from("<II", raw, offset)
    offset += 8
    chunk = raw[offset:offset + clen]
    offset += clen
    if ctype == 0x4E4F534A:
        gltf = json.loads(chunk)
    elif ctype == 0x004E4942:
        bin_chunk = chunk

buffer_views, accessors = gltf["bufferViews"], gltf["accessors"]
COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
             5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor_array(index):
    acc = accessors[index]
    view = buffer_views[acc["bufferView"]]
    dtype, width = COMPONENT[acc["componentType"]], WIDTH[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride, itemsize = view.get("byteStride"), np.dtype(dtype).itemsize * width
    count = acc["count"]
    if stride and stride != itemsize:
        rows = np.lib.stride_tricks.as_strided(
            np.frombuffer(bin_chunk, np.uint8, offset=start,
                          count=stride * count),
            shape=(count, itemsize), strides=(stride, 1))
        arr = rows.reshape(-1).view(dtype).reshape(count, width)
    else:
        arr = np.frombuffer(bin_chunk, dtype, offset=start,
                            count=count * width).reshape(count, width)
    if acc.get("normalized"):
        arr = arr.astype(np.float32) / np.iinfo(dtype).max
    return arr


def node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = m @ np.diag(list(node["scale"]) + [1.0])
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        rm = np.eye(4); rm[:3, :3] = r
        m = rm @ m
    if "translation" in node:
        tm = np.eye(4); tm[:3, 3] = node["translation"]
        m = tm @ m
    return m


nodes = gltf["nodes"]
world_of = {}


def walk(index, parent):
    m = parent @ node_matrix(nodes[index])
    if "mesh" in nodes[index]:
        world_of.setdefault(index, m)
    for child in nodes[index].get("children", ()):
        walk(child, m)


for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
    walk(root, np.eye(4))

# --- flatten, keeping uv2 / vertex colour / normals / tangents ---------
V, UV, UV2, LM, VC, NRM, TAN, F, FM = [], [], [], [], [], [], [], [], []
# CS2 vertex-stage streams; see the block at the attribute reads below.
C1, BV, PV, FPR = [], [], [], []
MORPH_DP, MORPH_DN, MORPH_OK = [], [], [True]
base_v = 0
for node_index, world in world_of.items():
    mesh = gltf["meshes"][nodes[node_index]["mesh"]]
    wt = torch.from_numpy(np.ascontiguousarray(world, np.float32)).to(device)
    # A mirrored node flips handedness, which flips the bitangent the
    # glTF TANGENT.w sign was authored for.
    det_sign = 1.0 if np.linalg.det(world[:3, :3]) >= 0 else -1.0
    for prim in mesh["primitives"]:
        a = prim["attributes"]
        if "POSITION" not in a or "indices" not in prim:
            continue
        idx = torch.from_numpy(
            accessor_array(prim["indices"]).reshape(-1).astype(np.int64)
        ).to(device)
        used, remap = torch.unique(idx, return_inverse=True)
        pos = torch.from_numpy(np.ascontiguousarray(
            accessor_array(a["POSITION"]), np.float32)).to(device)[used]
        V.append(pos @ wt[:3, :3].T + wt[:3, 3])

        def attr(name, width, default=0.0):
            if name in a:
                arr = np.ascontiguousarray(accessor_array(a[name]), np.float32)
                return torch.from_numpy(arr).to(device)[used][:, :width]
            return torch.full((len(used), width), default, device=device)

        UV.append(attr("TEXCOORD_0", 2))
        # TEXCOORD_1 / TEXCOORD_2 are 0..1 lightmap-atlas coordinates,
        # not the blend layer's UV set. CS2 bakes its lighting into
        # maps/<map>/lightmaps/*.vtex and addresses it with these.
        UV2.append(attr("TEXCOORD_2", 2) if "TEXCOORD_2" in a
                   else UV[-1].clone())
        LM.append(attr("TEXCOORD_1", 2) if "TEXCOORD_1" in a
                  else torch.full((len(used), 2), -1.0, device=device))
        # Absent COLOR_0 must mean "layer 1 only". Defaulting to 1.0
        # made every unpainted primitive render as pure layer 2, which
        # turned grey plaster orange.
        VC.append(attr("COLOR_0", 4, 0.0))
        # --- the streams the CS2 VERTEX STAGE reads ------------------
        # The four decompiled csgo_*_vs families do not take the same
        # attributes, and the difference IS the specification of what a
        # packer must supply (SHADER_CALLFLOW_vertex_stages.md:80-137).
        # Aligned by declared semantic, not by slot:
        #
        #   COLOR1     PerVertexLighting      (env/blend/complex/foliage)
        #   TEXCOORD4  VertexPaintBlendParams (env/blend/complex)
        #              -- ALSO TEXCOORD4 --
        #   TEXCOORD4  PivotPaint             (foliage ONLY)
        #   TEXCOORD5  FoliageAnimation       (foliage ONLY)
        #
        # TEXCOORD4 is one D3D slot carrying two different things; the
        # slot number is not the meaning, so both readings are packed and
        # the renderer picks by shader family, never by slot.
        #
        # Widths are the widths the SHADER sees after vertex-fetch format
        # expansion, read from the SPIR-V type table; the ON-WIRE format
        # is declared by the mesh and was not extracted, so a glTF that
        # stores these narrower still expands to float here.
        C1.append(attr("COLOR_1", 4, 0.0))
        # vColorBlendValues, f32x4 (env_blend_vs_max:102 loc8, passed
        # straight to the blend PS at :462).  The existing --vpaint
        # sidecar carries only .x of this; the full vec4 goes in the pack.
        BV.append(attr("TEXCOORD_4", 4, 0.0))
        # vPivotPaint, f32x3, foliage only (foliage_vs_max:102 loc4).  It
        # is an OBJECT-SPACE POINT, so it takes the same node transform as
        # POSITION -- packing it untransformed would put every branch
        # pivot at the world origin.
        _pp = attr("TEXCOORD_4", 3)
        PV.append(_pp @ wt[:3, :3].T + wt[:3, 3])
        # vFoliageParams, f32x3, foliage only (foliage_vs_max:103 loc5).
        # .x detail-bend weight (:339), .y branch weight (:285).
        FPR.append(attr("TEXCOORD_5", 3, 1.0))
        n = attr("NORMAL", 3)
        n = n @ wt[:3, :3].T
        NRM.append(n / n.norm(dim=1, keepdim=True).clamp(min=1e-9))
        if "TANGENT" in a:
            tg = attr("TANGENT", 4)
            tw = tg[:, 3:4] * det_sign
            tv = tg[:, :3] @ wt[:3, :3].T
            tv = tv / tv.norm(dim=1, keepdim=True).clamp(min=1e-9)
            TAN.append(torch.cat([tv, tw], dim=1))
        else:
            TAN.append(torch.zeros(len(used), 4, device=device))
        # --- S_PRE_BAKED_VERTEX_ANIMATION: glTF morph targets ---------
        # csgo_complex_vs drives a vertex-animation axis with ZERO texture
        # fetch in all 480 of its modules
        # (SHADER_CALLFLOW_csgo_complex_vs.md:28-31, 49-51), i.e. the
        # animation data arrives as buffer/attribute data.  A glTF morph
        # target is exactly that shape, so it is what the pack carries.
        # The reference's own keyframe ENCODING was not extracted, and
        # the renderer says so where it uses this.
        _tg = prim.get("targets") or []
        if _tg:
            _dp, _dn = [], []
            for _tt in _tg:
                _d = (torch.from_numpy(np.ascontiguousarray(
                    accessor_array(_tt["POSITION"]), np.float32)
                ).to(device)[used] @ wt[:3, :3].T if "POSITION" in _tt
                    else torch.zeros(len(used), 3, device=device))
                _dp.append(_d)
                _dn.append(torch.from_numpy(np.ascontiguousarray(
                    accessor_array(_tt["NORMAL"]), np.float32)
                ).to(device)[used] @ wt[:3, :3].T if "NORMAL" in _tt
                    else torch.zeros(len(used), 3, device=device))
            MORPH_DP.append(torch.stack(_dp))
            MORPH_DN.append(torch.stack(_dn))
        else:
            MORPH_DP.append(None)
            MORPH_DN.append(None)
        F.append(remap.reshape(-1, 3).int() + base_v)
        FM.append(torch.full((len(remap) // 3,), prim.get("material", 0),
                             dtype=torch.int32, device=device))
        base_v += len(used)

vertices = torch.cat(V); uvs = torch.cat(UV); uvs2 = torch.cat(UV2)
lmuv = torch.cat(LM)
vcolor = torch.cat(VC); vnormal = torch.cat(NRM); vtangent = torch.cat(TAN)
vcolor1 = torch.cat(C1); vblendvals = torch.cat(BV)
vpivot = torch.cat(PV); vfoliage = torch.cat(FPR)
faces = torch.cat(F); face_prim_mat = torch.cat(FM)
# Morph targets, padded to a common key count so the table is (F, N, 3).
# A primitive with no targets contributes zeros, i.e. it does not animate,
# which is the correct result for a mesh with no baked animation and is
# distinguishable from "the axis is off" because the renderer refuses to
# enable S_PRE_BAKED_VERTEX_ANIMATION on an all-zero table.
_nk = max((0 if m is None else m.shape[0]) for m in MORPH_DP)
morph_dpos = morph_dnrm = None
if _nk:
    morph_dpos = torch.zeros(_nk, len(vertices), 3, device=device)
    morph_dnrm = torch.zeros(_nk, len(vertices), 3, device=device)
    _o = 0
    for _mp, _mn, _vv in zip(MORPH_DP, MORPH_DN, V):
        if _mp is not None:
            morph_dpos[:_mp.shape[0], _o:_o + len(_vv)] = _mp
            morph_dnrm[:_mn.shape[0], _o:_o + len(_vv)] = _mn
        _o += len(_vv)
    print(f"S_PRE_BAKED_VERTEX_ANIMATION: {_nk} morph keys, "
          f"max|dpos| {float(morph_dpos.abs().max()):.4g} m over "
          f"{int((morph_dpos.abs().sum((0, 2)) > 0).sum()):,} verts",
          flush=True)
print("CS2 vertex-stage streams: COLOR_1 (PerVertexLighting) on "
      f"{float((vcolor1.abs().sum(1) > 0).float().mean()):.1%} of verts, "
      f"TEXCOORD_4 (VertexPaintBlendParams / PivotPaint) on "
      f"{float((vblendvals.abs().sum(1) > 0).float().mean()):.1%}, "
      f"TEXCOORD_5 (FoliageAnimation) on "
      f"{float((vfoliage != 1.0).any(1).float().mean()):.1%}", flush=True)
print(f"flattened {time.time()-t0:.1f}s: {len(vertices):,} verts, "
      f"{len(faces):,} tris, "
      f"{float((vtangent[:, :3].norm(dim=1) > 0.5).float().mean()):.1%} "
      f"of verts carry a tangent", flush=True)

e1 = vertices[faces[:, 1].long()] - vertices[faces[:, 0].long()]
e2 = vertices[faces[:, 2].long()] - vertices[faces[:, 0].long()]
fn = torch.cross(e1, e2, dim=1)
face_normals = fn / fn.norm(dim=1, keepdim=True).clamp(min=1e-9)

# --- materials: both layers + colour correction + PBR slots ------------
materials = gltf.get("materials", [])
images = gltf.get("images", [])
tex_to_image = [t.get("source", -1) for t in gltf.get("textures", [])]
name_to_image = {im.get("name"): i for i, im in enumerate(images)}
uri_of = [im.get("uri") for im in images]
base_dir = os.path.dirname(os.path.abspath(args.glb))
T = args.tex_size
TA = args.aux_size
MODE = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}


def cc_params(vmat, suffix):
    fp = vmat.get("FloatParams", {})
    vp = vmat.get("VectorParams", {})
    tint = vp.get(f"g_vTextureColorTint{suffix}", [1, 1, 1, 1])[:3]
    return (list(tint)
            + [fp.get(f"g_fTextureColorBrightness{suffix}", 1.0),
               fp.get(f"g_fTextureColorContrast{suffix}", 1.0),
               fp.get(f"g_fTextureColorSaturation{suffix}", 1.0)])


def tex_img(d):
    """glTF textureInfo -> image index, or -1."""
    if not d:
        return -1
    return tex_to_image[d["index"]]


img1, img2, params, modes, cutoffs, secondary = [], [], [], [], [], []
imgN1, imgN2, imgMR, imgAO, imgEM = [], [], [], [], []
pbr_scalar = []          # (M, 4): metallic, roughness, bump, emissive scale
em_tint = []             # (M, 3)

# --- vmat-only slots (no glTF equivalent) ------------------------------
# Every one of these resolves by vtex name against a glTF image; the
# audit over all 923 materials reports 0 unresolved for each slot.
imgH1, imgH2, imgAOV, imgOVL = [], [], [], []
imgTM, imgTR, imgMET, imgSI = [], [], [], []
ext = []                 # (M, EXT_N) scalar/flag block, see EXT_* below
border_tint = []         # (M, 3)
color_tint = []          # (M, 3): g_vColorTint * g_flModelTintAmount

# Named offsets into the per-material `ext` row. The renderer imports the
# same names so a slot can never be read at the wrong index.
EXT_KEYS = (
    "h_scale1", "h_scale2", "h_zero1", "h_zero2", "blend_soft2",
    "bevel_strength2", "bevel_soft2", "bevel_spread2", "bevel_curve2",
    "border_offset2", "border_soft2", "border_spread2", "border_rough2",
    "f_blend_effects2", "f_border_rough2",
    "ovl_bright_contrast", "ovl_dark_contrast",
    "f_overlay", "overlay_mode", "overlay_tintmask",
    "f_tintmask", "tm_bright1", "tm_contrast1", "tm_bright2", "tm_contrast2",
    "f_transmissive_ndotl", "f_albedo_for_transmissive",
    "f_render_backfaces", "f_metalness_tex", "metalness",
    "f_self_illum", "si_albedo_factor",
    "vcmode1", "vcmode2",
    "has_h1", "has_h2", "has_aov", "has_ovl", "has_tm", "has_tr",
    "has_met", "has_si",
)
EXT_IDX = {k: i for i, k in enumerate(EXT_KEYS)}


def _vmat_slot(tp, name, sink):
    """Resolve a vmat .vtex reference to a glTF image index (or -1)."""
    v = tp.get(name)
    i = name_to_image.get(v, -1) if v else -1
    sink.append(i)
    return i


for m in materials:
    pbr = m.get("pbrMetallicRoughness", {})
    tex = pbr.get("baseColorTexture")
    img1.append(tex_to_image[tex["index"]] if tex else -1)
    vmat = m.get("extras", {}).get("vmat", {}) or {}
    tp = vmat.get("TextureParams", {})
    fp = vmat.get("FloatParams", {})
    vp = vmat.get("VectorParams", {})
    ip = vmat.get("IntParams", {})
    second = tp.get("g_tColor2")
    img2.append(name_to_image.get(second, -1) if second else -1)
    secondary.append(1 if ip.get("F_SECONDARY_UV") else 0)
    factor = pbr.get("baseColorFactor", [1, 1, 1, 1])[:3]
    p1 = cc_params(vmat, 1)
    p2 = cc_params(vmat, 2)
    params.append(factor + p1 + p2)
    modes.append(MODE.get(m.get("alphaMode", "OPAQUE"), 0))
    cutoffs.append(float(m.get("alphaCutoff", 0.5)))

    # --- PBR slots ---------------------------------------------------
    imgN1.append(tex_img(m.get("normalTexture")))
    n2 = tp.get("g_tNormal2")
    imgN2.append(name_to_image.get(n2, -1) if n2 else -1)
    imgMR.append(tex_img(pbr.get("metallicRoughnessTexture")))
    imgAO.append(tex_img(m.get("occlusionTexture")))
    imgEM.append(tex_img(m.get("emissiveTexture")))
    # glTF gives metallicFactor on the 178 materials with no MR texture
    # and omits it (default 1.0) on the 743 that have one, but VRF wrote
    # the raw g_tMetalness vtex into the MR slot rather than a glTF-packed
    # MR, so the renderer treats the MR texture as metalness-only.
    metallic = pbr.get("metallicFactor")
    metallic = 1.0 if metallic is None else float(metallic)
    # roughnessFactor is absent on all 923; Source 2's csgo_* shaders
    # carry roughness in the NORMAL map's alpha channel, which is why the
    # exported normal PNGs have a non-trivial alpha (mean 0.5-0.79) while
    # the MR PNGs are flat zero.
    rough = float(pbr.get("roughnessFactor", 1.0))
    bump = float(fp.get("g_flBumpStrength", 1.0))
    em = float(fp.get("g_flSelfIllumBrightness", 0.0)) * \
        float(fp.get("g_flSelfIllumScale", 1.0))
    pbr_scalar.append([metallic, rough, bump, em])
    et = (vp.get("g_vSelfIllumTint") or [1, 1, 1, 1])[:3]
    ef = m.get("emissiveFactor")
    if ef is not None and any(x > 0 for x in ef[:3]):
        et = [a * b for a, b in zip(et, ef[:3])]
    em_tint.append(list(et))

    # --- vmat-only slots ---------------------------------------------
    h1 = _vmat_slot(tp, "g_tHeight1", imgH1)
    h2 = _vmat_slot(tp, "g_tHeight2", imgH2)
    aov = _vmat_slot(tp, "g_tAmbientOcclusion", imgAOV)
    ovl = _vmat_slot(tp, "g_tSharedColorOverlay", imgOVL)
    tm = _vmat_slot(tp, "g_tTintMask", imgTM)
    tr = _vmat_slot(tp, "g_tTransmissiveColor", imgTR)
    met = _vmat_slot(tp, "g_tMetalness", imgMET)
    si = _vmat_slot(tp, "g_tSelfIllumMask", imgSI)
    row = dict(
        # Source 2 height blend. HeightMapScale defaults to 1 rather
        # than 0: a 0 default would silently disable the height term on
        # the 417/700 materials that omit g_flHeightMapScale1.
        h_scale1=fp.get("g_flHeightMapScale1", 1.0),
        h_scale2=fp.get("g_flHeightMapScale2", 1.0),
        h_zero1=fp.get("g_flHeightMapZeroPoint1", 0.0),
        h_zero2=fp.get("g_flHeightMapZeroPoint2", 0.0),
        blend_soft2=fp.get("g_flBlendSoftness2", 0.1),
        bevel_strength2=fp.get("g_flBevelStrength2", 0.0),
        bevel_soft2=fp.get("g_flBevelSoftness2", 0.1),
        bevel_spread2=fp.get("g_flBevelSpread2", 0.0),
        bevel_curve2=fp.get("g_flBevelCurve2", 1.0),
        border_offset2=fp.get("g_flBorderOffset2", 0.0),
        border_soft2=fp.get("g_flBorderSoftness2", 0.1),
        border_spread2=fp.get("g_flBorderSpread2", 0.0),
        border_rough2=fp.get("g_fBorderRoughness2", 0.0),
        f_blend_effects2=1.0 if ip.get("F_BLEND_EFFECTS_2") else 0.0,
        f_border_rough2=1.0 if ip.get("F_BORDER_ROUGHNESS_2") else 0.0,
        ovl_bright_contrast=fp.get("g_flOverlayBrightnessContrast", 0.0),
        ovl_dark_contrast=fp.get("g_flOverlayDarknessContrast", 0.0),
        f_overlay=1.0 if ip.get("F_SHARED_COLOR_OVERLAY") else 0.0,
        overlay_mode=float(ip.get("g_nColorOverlayMode", 0) or 0),
        overlay_tintmask=float(ip.get("g_nColorOverlayTintMask", 0) or 0),
        f_tintmask=1.0 if ip.get("F_TINT_MASK") else 0.0,
        tm_bright1=fp.get("g_fTintMaskBrightness1", 1.0),
        tm_contrast1=fp.get("g_fTintMaskContrast1", 1.0),
        tm_bright2=fp.get("g_fTintMaskBrightness2", 1.0),
        tm_contrast2=fp.get("g_fTintMaskContrast2", 1.0),
        f_transmissive_ndotl=(
            1.0 if ip.get("F_TRANSMISSIVE_BACKFACE_NDOTL") else 0.0),
        f_albedo_for_transmissive=(
            1.0 if ip.get("F_USE_ALBEDO_FOR_TRANSMISSIVE") else 0.0),
        f_render_backfaces=1.0 if ip.get("F_RENDER_BACKFACES") else 0.0,
        f_metalness_tex=1.0 if ip.get("F_METALNESS_TEXTURE") else 0.0,
        metalness=fp.get("g_flMetalness", 0.0),
        f_self_illum=1.0 if ip.get("F_SELF_ILLUM") else 0.0,
        si_albedo_factor=fp.get("g_flSelfIllumAlbedoFactor", 0.0),
        # g_nVertexColorMode<N> selects what COLOR_0 MEANS for layer N.
        # -1 = the key is absent (779 / 787 materials). Present values are
        # 0 (120/112), 1 (21/17) and 2 (3/7). The three materials with
        # mode1 == 2 are exactly inferno_stonefloor07_{dirt,concrete,
        # gravel}_blend -- the plaza ground whose colour does not match GT.
        vcmode1=ip.get("g_nVertexColorMode1", -1),
        vcmode2=ip.get("g_nVertexColorMode2", -1),
        has_h1=float(h1 >= 0), has_h2=float(h2 >= 0),
        has_aov=float(aov >= 0), has_ovl=float(ovl >= 0),
        has_tm=float(tm >= 0), has_tr=float(tr >= 0),
        has_met=float(met >= 0), has_si=float(si >= 0),
    )
    ext.append([float(row[k]) for k in EXT_KEYS])
    bt = (vp.get("g_vBorderTint2") or [1, 1, 1, 1])[:3]
    border_tint.append(list(bt))
    ct = (vp.get("g_vColorTint") or [1, 1, 1, 1])[:3]
    amt = float(fp.get("g_flModelTintAmount", 1.0))
    color_tint.append([1.0 + (c - 1.0) * amt for c in ct])

need = sorted({i for i in img1 + img2 if i >= 0})
VMAT_LISTS = (imgH1, imgH2, imgAOV, imgOVL, imgTM, imgTR, imgMET, imgSI)
aux_need = sorted({i for i in imgN1 + imgN2 + imgMR + imgAO + imgEM
                   + [j for lst in VMAT_LISTS for j in lst] if i >= 0})
print(f"layer1 refs {sum(1 for i in img1 if i>=0)}, "
      f"layer2 refs {sum(1 for i in img2 if i>=0)}, "
      f"distinct colour images {len(need)}", flush=True)
print(f"aux refs: normal1 {sum(1 for i in imgN1 if i>=0)}, "
      f"normal2 {sum(1 for i in imgN2 if i>=0)}, "
      f"mr {sum(1 for i in imgMR if i>=0)}, ao {sum(1 for i in imgAO if i>=0)}, "
      f"em {sum(1 for i in imgEM if i>=0)}; distinct aux images "
      f"{len(aux_need)} @ {TA}", flush=True)
print("vmat slot refs: " + ", ".join(
    f"{n} {sum(1 for i in lst if i >= 0)}" for n, lst in zip(
        ("height1", "height2", "ao_vmat", "overlay", "tintmask",
         "transmissive", "metalness", "selfillum"), VMAT_LISTS)), flush=True)


def _resize_rgba(img, size):
    """Resize RGBA WITHOUT alpha premultiplication.

    Pillow 12 premultiplies alpha inside resize() for RGBA images, so a
    texel with alpha 0 comes back with rgb 0 no matter what colour it
    held. Measured on this export: plaster_facade_01b_height (a real
    2048^2 height map whose alpha happens to be 0) resizes from channel
    means (0.326, 0.734) to (0.000, 0.0001) -- the data is not attenuated
    but ANNIHILATED. A synthetic rgb=200/alpha=0 image resizes to 0 as
    well, so this is unconditional, not an artefact of one file.

    That silently zeroed every vmat slot whose alpha is 0 (all 700
    g_tHeight1 / 365 g_tHeight2 maps, g_tTransmissiveColor's 0.733 grey)
    and darkened every alpha-tested COLOUR texture in proportion to its
    own alpha. Resizing colour and alpha as separate images is exact.
    """
    from PIL import Image
    rgb = Image.merge("RGB", img.split()[:3]).resize(size, Image.BILINEAR)
    a = img.split()[3].resize(size, Image.BILINEAR)
    return Image.merge("RGBA", rgb.split() + (a,))


def decode(i):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(os.path.join(base_dir, uri_of[i])).convert("RGBA")
    if args.premultiply_bug:
        return i, np.asarray(img.resize((T, T), Image.BILINEAR), np.uint8)
    return i, np.asarray(_resize_rgba(img, (T, T)), np.uint8)


def decode_aux(i):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(os.path.join(base_dir, uri_of[i])).convert("RGBA")
    if args.premultiply_bug:
        return i, np.asarray(img.resize((TA, TA), Image.BILINEAR), np.uint8)
    return i, np.asarray(_resize_rgba(img, (TA, TA)), np.uint8)


def alpha_bleed(arr, iters=6, thresh=128):
    """Dilate RGB into transparent texels (fixes black alpha-test seams).

    Alpha-tested CS2 textures store BLACK rgb where alpha is 0 (measured:
    70% of texels on the worst foliage layer). Bilinear filtering near a
    leaf edge pulls alpha above the cutoff while rgb is still black, so
    the surface gets a black fringe. Bleeding valid colour outward makes
    the interpolated rgb sane regardless of the alpha ramp.
    """
    t = torch.from_numpy(arr).to(device).float()
    rgb, a = t[..., :3], t[..., 3]
    valid = (a >= thresh).float().unsqueeze(-1)
    if float(valid.mean()) in (0.0, 1.0):
        return arr
    rgb = rgb * valid
    for _ in range(iters):
        r = rgb.permute(2, 0, 1)[None]
        v = valid.permute(2, 0, 1)[None]
        k = torch.ones(1, 1, 3, 3, device=device)
        rs = torch.nn.functional.conv2d(
            r, k.expand(3, 1, 3, 3), padding=1, groups=3)
        vs = torch.nn.functional.conv2d(v, k, padding=1)
        fill = (vs > 0) & (v == 0)
        newrgb = torch.where(fill.expand_as(rs), rs / vs.clamp(min=1), r)
        rgb = newrgb[0].permute(1, 2, 0)
        valid = ((vs > 0).float())[0].permute(1, 2, 0)
    out = arr.copy()
    out[..., :3] = rgb.clamp(0, 255).byte().cpu().numpy()
    return out


layer_of, layers = {}, []
with futures.ProcessPoolExecutor(max_workers=16) as pool:
    for i, arr in pool.map(decode, need, chunksize=8):
        layer_of[i] = len(layers)
        layers.append(arr if args.no_alpha_bleed else alpha_bleed(arr))
white = np.full((T, T, 4), 255, np.uint8)
layer_of[-1] = len(layers)
layers.append(white)
print(f"textures decoded {time.time()-t0:.1f}s: {len(layers)} layers",
      flush=True)

# --- aux array: normal / metal-rough / AO / emissive -------------------
# No alpha_bleed here: alpha is roughness on normal maps, not coverage.
aux_of, aux_layers = {}, []
with futures.ProcessPoolExecutor(max_workers=16) as pool:
    for i, arr in pool.map(decode_aux, aux_need, chunksize=8):
        aux_of[i] = len(aux_layers)
        aux_layers.append(arr)
AUX_FLAT = len(aux_layers)      # neutral tangent-space normal (0,0,1), rough .5
aux_layers.append(np.tile(np.array([128, 128, 255, 128], np.uint8),
                          (TA, TA, 1)))
AUX_WHITE = len(aux_layers)     # AO = 1
aux_layers.append(np.full((TA, TA, 4), 255, np.uint8))
AUX_BLACK = len(aux_layers)     # emissive = 0, metalness = 0
aux_layers.append(np.zeros((TA, TA, 4), np.uint8))
AUX_HALF = len(aux_layers)      # height = 0.5 -> no local height preference
aux_layers.append(np.full((TA, TA, 4), 128, np.uint8))
print(f"aux decoded {time.time()-t0:.1f}s: {len(aux_layers)} layers @{TA}",
      flush=True)


def aux_index(lst, fallback):
    return torch.tensor([aux_of[i] if i >= 0 else fallback for i in lst],
                        dtype=torch.int32)


mat_l1 = torch.tensor([layer_of[i] for i in img1], dtype=torch.int32)
mat_l2 = torch.tensor([layer_of[i] for i in img2], dtype=torch.int32)
mat_has2 = torch.tensor([1 if i >= 0 else 0 for i in img2], dtype=torch.int8)
mat_params = torch.tensor(params, dtype=torch.float32)   # (M, 3+6+6)
mat_secondary = torch.tensor(secondary, dtype=torch.int8)
mat_mode = torch.tensor(modes, dtype=torch.int8)
mat_cut = torch.tensor(cutoffs, dtype=torch.float32)
pm = face_prim_mat.long().cpu().clamp(max=len(materials) - 1)

# Material TABLES, indexed per face by face_matid. face_params alone is
# already 15 floats x 8.6 M faces; five more per-face slots would add
# another ~170 MB for data that has only 923 distinct values.
mat_nrm1 = aux_index(imgN1, AUX_FLAT)
mat_nrm2 = aux_index(imgN2, AUX_FLAT)
mat_mr = aux_index(imgMR, AUX_BLACK)
mat_ao = aux_index(imgAO, AUX_WHITE)
mat_em = aux_index(imgEM, AUX_BLACK)
mat_has_nrm = torch.tensor([1 if i >= 0 else 0 for i in imgN1],
                           dtype=torch.int8)
mat_has_em = torch.tensor([1 if i >= 0 else 0 for i in imgEM],
                          dtype=torch.int8)
mat_pbr = torch.tensor(pbr_scalar, dtype=torch.float32)     # (M, 4)
mat_emtint = torch.tensor(em_tint, dtype=torch.float32)     # (M, 3)

# --- vmat-only slot tables --------------------------------------------
# Fallbacks are the value that makes the feature a NO-OP, so a material
# without the slot shades exactly as it did before the feature existed:
# height -> 0.5 (no local preference), AO/tintmask -> 1, overlay ->
# mid-grey (overlay neutral), transmissive/metalness/selfillum -> 0.
mat_h1 = aux_index(imgH1, AUX_HALF)
mat_h2 = aux_index(imgH2, AUX_HALF)
mat_aov = aux_index(imgAOV, AUX_WHITE)
mat_ovl = aux_index(imgOVL, AUX_HALF)
mat_tm = aux_index(imgTM, AUX_WHITE)
mat_tr = aux_index(imgTR, AUX_BLACK)
mat_met = aux_index(imgMET, AUX_BLACK)
mat_si = aux_index(imgSI, AUX_BLACK)
mat_ext = torch.tensor(ext, dtype=torch.float32)            # (M, EXT_N)
mat_btint = torch.tensor(border_tint, dtype=torch.float32)  # (M, 3)
mat_ctint = torch.tensor(color_tint, dtype=torch.float32)   # (M, 3)

torch.save({
    "mat_h1": mat_h1, "mat_h2": mat_h2, "mat_aov": mat_aov,
    "mat_ovl": mat_ovl, "mat_tm": mat_tm, "mat_tr": mat_tr,
    "mat_met": mat_met, "mat_si": mat_si,
    "mat_ext": mat_ext, "mat_btint": mat_btint, "mat_ctint": mat_ctint,
    "ext_keys": list(EXT_KEYS),
    "vertices": vertices.cpu(), "uvs": uvs.cpu(), "uvs2": uvs2.cpu(),
    "vcolor": vcolor.cpu(), "vnormal": vnormal.cpu(),
    "vtangent": vtangent.cpu(),
    "lmuv": lmuv.cpu(),
    # CS2 vertex-stage streams.  Keyed by SEMANTIC, not by D3D slot,
    # because TEXCOORD4 carries VertexPaintBlendParams on
    # environment/blend/complex and PivotPaint on foliage.
    "vcolor1": vcolor1.cpu(),          # COLOR1  PerVertexLighting
    "vblendvals": vblendvals.cpu(),    # TEXCOORD4 VertexPaintBlendParams
    "vpivot": vpivot.cpu(),            # TEXCOORD4 PivotPaint (foliage)
    "vfoliage": vfoliage.cpu(),        # TEXCOORD5 FoliageAnimation
    **({"morph_dpos": morph_dpos.cpu(), "morph_dnrm": morph_dnrm.cpu()}
       if morph_dpos is not None else {}),
    "faces": faces.cpu(), "face_normals": face_normals.cpu(),
    "face_mat": mat_l1[pm], "face_mat2": mat_l2[pm],
    "face_has2": mat_has2[pm], "face_secondary": mat_secondary[pm],
    "face_params": mat_params[pm], "face_alpha_mode": mat_mode[pm],
    "face_alpha_cutoff": mat_cut[pm],
    "face_matid": pm.to(torch.int32),
    "textures": torch.from_numpy(np.stack(layers)),
    "aux": torch.from_numpy(np.stack(aux_layers)),
    "mat_nrm1": mat_nrm1, "mat_nrm2": mat_nrm2, "mat_mr": mat_mr,
    "mat_ao": mat_ao, "mat_em": mat_em,
    "mat_has_nrm": mat_has_nrm, "mat_has_em": mat_has_em,
    "mat_pbr": mat_pbr, "mat_emtint": mat_emtint,
}, args.out)
print(f"wrote {args.out} in {time.time()-t0:.1f}s "
      f"({int(mat_has2[pm].sum()):,} faces have a second layer, "
      f"{int(mat_has_nrm[pm].sum()):,} have a normal map)", flush=True)
_h2 = mat_ext[:, EXT_IDX["has_h2"]][pm]
_ov = mat_ext[:, EXT_IDX["f_overlay"]][pm]
_ao = mat_ext[:, EXT_IDX["has_aov"]][pm]
print(f"faces reached: height-blend {int(_h2.sum()):,}, "
      f"shared overlay {int(_ov.sum()):,}, vmat AO {int(_ao.sum()):,}",
      flush=True)
