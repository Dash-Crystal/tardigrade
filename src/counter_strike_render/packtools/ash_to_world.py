"""Flatten an ASH extraction into the flat world .pt that gpu_render.py loads.

ASH `geometry.pt` is a SCENE GRAPH -- {models, instances, world_nodes} where a
model owns several vertex buffers, several index buffers, and draw calls that
name a slice of one index buffer plus the vertex buffers bound alongside it.
`gpu_render.py --world` wants the opposite shape: ONE indexed mesh, with the
per-draw material already resolved down to a per-face id.

This writes the keys gpu_render.py reads UNCONDITIONALLY off `data`:

    vertices uvs faces face_mat face_normals textures   (+ uvs2 lmuv)

plus `face_matid` and `mat_paths` for the fam side table (see below). Every
other read in that file is either `data.get(...)` or sits under `if FAMILY:` /
`if PBR:` / `if args.normal_map:`, so the base raster path is complete with
the six above. This pack therefore renders the base raster path ONLY; the
PBR/VMAT flags need an `aux` array and the mat_* tables that are not
synthesised here, and gpu_render.py refuses those flags by name rather than
reading a missing key as zero.

MATERIAL NUMBERING IS A CONTRACT WITH THE FAM SIDE TABLE, whose row i is
sorted(mat_paths)[i] and which the renderer indexes by face_matid. Numbering
in first-seen draw order -- the order materials actually arrive in -- puts
every face on another material's parameters, and the frame still renders, so
nothing announces the error. Materials are therefore sorted by path here, and
`mat_paths` ships in the pack so the two sides can be compared rather than
assumed aligned.

THREE STRUCTURAL FACTS the naive flattening gets wrong:

1. A draw call's `vertex_buffers` is a LIST of PARALLEL ATTRIBUTE STREAMS over
   the SAME vertices, not a list of separate meshes. On de_inferno 619 of 784
   models split POSITION_0 into one buffer (stride 12) and TEXCOORD_0/NORMAL_0/
   COLOR_0 into a second (stride 20) with an IDENTICAL count. Treating them as
   two meshes emits the position buffer with no UVs and the attribute buffer
   with no positions. The streams are merged by vertex index here.

2. Draw calls PARTITION their index buffer -- they are not LODs of one another.
   The 446 draw calls of n0_lr0_agg_merge_inferno_stone_wall_09_trim_0 sum to
   exactly the 2624310 indices of its single index buffer. So every draw call
   is emitted; nothing is deduplicated, and no LOD selection is needed.

3. Source units are INCHES with Z up; the renderer works in METRES with Y up
   and reads the camera json in source units, applying S itself as
   world = (y_src*S, eye_z_src*S, x_src*S). So a vertex maps

       world = (src.y, src.z, src.x) * S

   which is a CYCLIC permutation (determinant +1) and therefore preserves
   triangle winding -- the normals below can be computed after the swap.

FLOOR NORMAL SIGN is measured, not assumed. synth_min_world.py shipped floors
with normal.y = -1 and nothing crashed: the base raster path neither culls
backfaces nor uses the normal, so a flipped floor only surfaces later, in a
camera sampler concluding the world has no standable geometry. Here the
area-weighted sign of near-horizontal faces is computed and PRINTED, and
--winding auto flips the emitted winding when floors come out pointing down.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time

import numpy as np
import torch

S = 0.0254                      # source inches -> metres
TEX = 64                        # synthesised texture edge (power of two: --mip
                                # builds a mip pyramid off textures.shape[1])

# Tool materials are non-visual volumes (blocklight, clip, nodraw). They are
# in the extraction because they are in the map; drawing them fills the frame
# with solid boxes. Dropped by prefix, and the count is reported.
TOOL_PREFIXES = ("materials/tools/",)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# .vtex_c container -> one mip's compressed block bytes
#
# The texels are NOT in the DATA block. DATA holds only the header and the
# extra-data tables (~1-2 KB); the texel bytes TRAIL every declared block,
# running from the end of the last block to the end of the file. The
# resource header's own file-size field describes the block region alone,
# so it agrees with `blocks_end`, not with the file -- reading it as the
# file length finds no texels at all and raises nothing.
#
# Per-mip storage is NOT uniformly raw. Checking one file per format said
# it was; checking all 794 said 190 of 713. The stored size of each level
# comes from the COMPRESSED_MIP_SIZE extra, and only the top level or two
# are ever actually compressed -- so the small level this pack wants is
# raw in practice and LZ4 is not a dependency. A compressed pick is NAMED,
# never handed to a block decoder as though it were block data.
#
# The header layout is the one thing ash_extract.py's texture_format()
# gets wrong -- it reads 6 reflectivity floats and puts the extra-data
# offsets before w/h/depth. The wrong layout still unpacks, so it fails
# silently and reports UNKNOWN for all 794. Correct order below.
VTEX_HDR = "<HH4f3HBBIII"      # ver flags refl[4] w h depth fmt mips
                               # picmip0res extraOff extraCnt
# TRANSCRIBED FROM ash_extract.py, which transcribed it from source. This
# file previously carried its own copy that was OFF BY ONE from index 23:
# it had ATI1N at 23, so the 78 single-channel BC4 maps per map -- the
# _ao/_mask/_trans slots -- read as RG11_EAC. Both names mean "compressed"
# and both are plausible, so nothing raised and nothing looked empty; it
# would simply have fed a one-channel BC4 surface to a two-channel decoder.
# Only the LABEL decides the decoder, which is why the format number is
# what gets passed to bc_decode.decode() below -- it owns the number->name
# mapping, and this table exists only to size the mip walk.
VTEX_FORMAT = {
    0: "UNKNOWN", 1: "DXT1", 2: "DXT5", 3: "I8", 4: "RGBA8888",
    5: "R16", 6: "RG1616", 7: "RGBA16161616", 8: "R16F", 9: "RG1616F",
    10: "RGBA16161616F", 11: "R32F", 12: "RG3232F", 13: "RGB323232F",
    14: "RGBA32323232F", 15: "JPEG_RGBA8888", 16: "PNG_RGBA8888",
    17: "JPEG_DXT5", 18: "PNG_DXT5", 19: "BC6H", 20: "BC7",
    21: "ATI2N", 22: "IA88", 23: "ETC2", 24: "ETC2_EAC", 25: "R11_EAC",
    26: "RG11_EAC", 27: "ATI1N", 28: "BGRA8888", 29: "WEBP_RGBA8888",
    30: "WEBP_DXT5",
}
# Bytes per 4x4 block, or per pixel for the uncompressed forms.
BLOCK_BYTES = {"DXT1": 8, "BC4": 8, "ATI1N": 8,
               "DXT5": 16, "BC7": 16, "BC6H": 16, "ATI2N": 16,
               "RG11_EAC": 16, "ETC2_EAC": 16, "ETC2": 8}
RAW_STORED = ("BC7", "DXT1", "DXT5", "ATI1N", "ATI2N",
              "RGBA8888", "BGRA8888")
# The uncompressed forms need no codec at all, so they exercise container
# -> mip -> resize -> pack -> render with NOTHING mocked. That end-to-end
# path being green before bc_decode.py exists is worth more than the one
# material it currently textures.
UNCOMPRESSED = ("RGBA8888", "BGRA8888")


def _level_bytes(w, h, fmt):
    if fmt in ("RGBA8888", "BGRA8888"):
        return w * h * 4
    bpb = BLOCK_BYTES.get(fmt)
    if bpb is None:
        return None
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bpb


def read_vtex(path, want=TEX, allow_lz4=False):
    """(fmt, w, h, block_bytes) for the smallest mip at least `want` wide.

    Reading a small mip rather than level 0 is the point: the pack needs a
    64x64 texel square, and de_inferno's textures run to 2048x2048, so
    decoding level 0 would cost ~500x the work and then be thrown away.
    Mip levels are stored SMALLEST FIRST, so level L sits after every
    level below it.

    `allow_lz4` is OPT-IN and defaults off, so no existing caller changes
    behaviour by this argument existing. With it off a level whose stored
    size differs from its raw size is named "COMPRESSED:<fmt>" and no
    bytes come back, which is what every caller before this saw. With it
    on the level is LZ4-block-decompressed and the result is checked to be
    exactly `raw[pick]` bytes long before it is returned -- a wrong
    decompression is a length mismatch, not a plausible image. It matters
    at 512^2 rather than 64^2: at `want=64` almost every level is stored
    raw, but at `want=512` 8 of de_cache's 19 g_tColor3 textures land on a
    compressed level, so refusing them would have silently dropped 42% of
    that map's layer-3 albedo.
    """
    with open(path, "rb") as fh:
        b = fh.read()
    _fs, _hv, _ver, bo, bc = struct.unpack_from("<IHHII", b, 0)
    p, end, d0 = 8 + bo, 0, None
    for _ in range(bc):
        tag = b[p:p + 4].decode("ascii", "replace")
        off, size = struct.unpack_from("<II", b, p + 4)
        if tag == "DATA":
            d0 = p + 4 + off
        end = max(end, p + 4 + off + size)
        p += 12
    if d0 is None:
        return None
    (ver, _flags, _r, _g, _b, _a, w, h, _dep,
     fmt, nmip, _pic, _xo, _xc) = struct.unpack_from(VTEX_HDR, b, d0)
    if nmip < 1:
        return None
    if ver != 1:
        return None
    name = VTEX_FORMAT.get(fmt, f"fmt_{fmt}")
    levels = [(max(1, w >> i), max(1, h >> i)) for i in range(nmip)]
    raw = [_level_bytes(lw, lh, name) for lw, lh in levels]
    if any(r is None for r in raw):
        return ("UNSUPPORTED:" + name, w, h, None, fmt)

    # Per-mip STORED sizes. Most files carry a COMPRESSED_MIP_SIZE extra
    # (type 4), and without it the walk lands mid-texel on 523 of the 713
    # BC7/DXT1/DXT5 files here -- the earlier "they are all stored raw"
    # reading came from checking ONE file per format, and 190 of 713 is
    # what that actually generalises to.
    #
    # The entry is (unknown, mipsOffset, count) and the size array begins
    # one word AFTER mipsOffset. Reading it at mipsOffset yields an array
    # led by `count` itself and short by one, which sums to trail minus
    # (last_level - nmip) -- close enough to look like padding and not be.
    # Sizes are indexed by MIP LEVEL, level 0 first.
    sizes = None
    xbase = d0 + 32 + _xo
    for i in range(_xc):
        es = xbase + i * 12
        etype, eoff, _esz = struct.unpack_from("<III", b, es)
        if etype == 4:
            dat = es + 4 + eoff
            _u, moff, cnt = struct.unpack_from("<III", b, dat)
            if cnt == nmip:
                sizes = list(struct.unpack_from(f"<{cnt}I", b, dat + moff + 4))
    if sizes is None:
        sizes = raw
    if end + sum(sizes) != len(b):
        return ("SIZEMISMATCH:" + name, w, h, None, fmt)

    # Pick the smallest level at least `want` wide.
    pick = 0
    for i in range(nmip - 1, -1, -1):
        if levels[i][0] >= want:
            pick = i
            break
    if sizes[pick] != raw[pick] and not allow_lz4:
        # A compressed level. Only the top one or two ever are, and the
        # 64x64 level this pack wants is stored raw in practice -- so LZ4
        # is not a dependency at want=64. Named rather than fed to the
        # decoder as if it were block data.
        return ("COMPRESSED:" + name, w, h, None, fmt)
    if name not in RAW_STORED:
        return ("UNSUPPORTED:" + name, w, h, None, fmt)
    # Levels are stored SMALLEST FIRST, so level L follows every level
    # below it.
    off = end + sum(sizes[nmip - 1:pick:-1])
    lw, lh = levels[pick]
    blk = b[off:off + sizes[pick]]
    if sizes[pick] != raw[pick]:
        # LZ4 BLOCK format (no frame header), uncompressed size known from
        # the mip geometry. The length check below is the falsifier: LZ4
        # block decompression given a wrong uncompressed_size either raises
        # or returns short, and a short buffer fed to the BC decoder would
        # produce a partly-garbage image with nothing to see. Checked here
        # so the caller gets bytes or an explanation, never a truncation.
        try:
            import lz4.block as _lz4b
        except ImportError:
            return ("NOLZ4:" + name, w, h, None, fmt)
        try:
            blk = _lz4b.decompress(blk, uncompressed_size=raw[pick])
        except Exception:                                # noqa: BLE001
            return ("LZ4FAIL:" + name, w, h, None, fmt)
        if len(blk) != raw[pick]:
            return ("LZ4SHORT:" + name, w, h, None, fmt)
    return (name, lw, lh, blk, fmt)


def mat_color(path, tint):
    """Deterministic per-material hue x a checker, modulated by g_vColorTint.

    A flat colour would hide a broken UV and make every surface in the frame
    the same, so the geometry could not be read by eye. The hue comes from a
    hash of the material path so the same material is the same colour in every
    map, and the checker makes a wrong or unscaled UV visible immediately.
    """
    h = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16)
    hue = (h & 0xFFFF) / 65535.0
    i = int(hue * 6.0) % 6
    f = hue * 6.0 - int(hue * 6.0)
    v, p, q, t = 0.95, 0.35, 0.95 * (1 - 0.6 * f), 0.35 + 0.6 * f * 0.95
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    rgb = np.array(rgb, dtype=np.float32) * np.asarray(tint[:3], dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(TEX), np.arange(TEX), indexing="ij")
    checker = (((xx // 8) + (yy // 8)) % 2).astype(np.float32) * 0.35 + 0.6
    img = np.clip(checker[..., None] * rgb[None, None, :], 0.0, 1.0)
    a = np.ones((TEX, TEX, 1), dtype=np.float32)
    return (np.concatenate([img, a], axis=-1) * 255).astype(np.uint8)


_BC = "unchecked"


def decode_albedo(path, want=TEX):
    """(HxWx4 uint8, None) or (None, reason). Never raises, never guesses.

    The codec lives in bc_decode.py and is material-pipeline's -- the
    per-(family, slot) encoding semantics belong with it, since the same
    bytes in g_tNormal are DXT5nm in one family and hemi-octahedral in
    another. This is only the seam: container, mip pick, resize, and an
    honest reason string when a texel cannot be produced. Every failure
    falls back to the checker so a partly-decoded map still renders and
    the gap is visible by eye rather than silent.
    """
    global _BC
    if _BC == "unchecked":
        try:
            import bc_decode as _m
            _BC = _m
            # WHICH bc_decode ran, not which one was reviewed. Two deployed
            # copies were 647 and 662 lines against the repo's 823, so the
            # module that was measured and committed was not the module
            # executing -- and the only symptom was a missing attribute.
            # A validated artifact and a deployed artifact are different
            # objects until something checks they are the same.
            try:
                _b = open(_m.__file__, "rb").read()
                _nv = "yes" if hasattr(_m, "normal_validity") else "NO"
                _tn = "yes" if hasattr(_m, "to_normal") else "NO"
                _nl = _b.count(b"\n") + 1
                _sh = hashlib.sha256(_b).hexdigest()[:16]
                log(f"  bc_decode: {_m.__file__} sha256={_sh} "
                    f"lines={_nl} to_normal={_tn} normal_validity={_nv}")
            except Exception as _e:                       # noqa: BLE001
                # Named, not swallowed. "Unreadable" is worth nothing; the
                # reason is the point -- this one was bytes.count(str),
                # and it hid the provenance line written to stop exactly
                # this class of thing.
                log(f"  bc_decode: provenance FAILED "
                    f"({type(_e).__name__}: {_e})")
        except Exception:                                 # noqa: BLE001
            _BC = None
    if not os.path.exists(path):
        return None, "no .vtex_c on disk"
    try:
        got = read_vtex(path, want)
    except Exception as e:                                # noqa: BLE001
        return None, f"container unreadable ({type(e).__name__})"
    if got is None:
        return None, "no DATA block / bad version"
    fmt, w, h, blocks, fmtnum = got
    if blocks is None:
        return None, fmt.split(":")[0].lower() + " " + fmt.split(":")[-1]
    if fmt in UNCOMPRESSED:
        a = np.frombuffer(blocks, dtype=np.uint8).reshape(h, w, 4)
        if fmt == "BGRA8888":
            a = a[..., [2, 1, 0, 3]]
        return _fit(a, w, h), None
    if _BC is None:
        return None, "no bc_decode.py in the tree yet"
    fn = getattr(_BC, "decode", None)
    if fn is None:
        return None, "bc_decode.py exposes no decode()"
    try:
        # bc_decode.decode(data, fmt, width, height) -- and fmt is
        # given as the NUMBER so that file owns the number->name
        # mapping. Passing a name from here would reintroduce the
        # duplicate enum that was already wrong once.
        rgba = fn(blocks, fmtnum, w, h)
    except Exception as e:                                # noqa: BLE001
        return None, f"{fmt} decode raised ({type(e).__name__})"
    return _fit(np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4), w, h), None


def _resolve_side(mat_paths_dir, n):
    """Side-table path for map `n`, or None -- and SAY which.

    Two naming conventions in the wild: fam_side_<map>.pt in the fam_build
    assets dir, and <map>.pt in ~/fam_side. A resolver that knows only one
    reports "no table" for a directory that is full of them -- 43 silent
    fallbacks to local ordering, which renders clean and shades the wrong
    material. Try both, and say which was found.

    ONE implementation, shared by the sweep and the single-file path. It
    lived only in the sweep, so `--ash` accepted --mat-paths-dir and
    ignored it, and the layer-2 pages vanished with no line printed.
    """
    for cand in (f"{n}.pt", f"fam_side_{n}.pt"):
        c = os.path.join(mat_paths_dir, cand)
        if os.path.exists(c):
            return c
    log(f"  NO side table for {n} in {mat_paths_dir} (tried {n}.pt and "
        f"fam_side_{n}.pt): own sorted order, NOT aligned to any table")
    return None


def _fit(a, w, h):
    """Nearest-resample to the pack's square.

    Cheap on purpose: read_vtex already picked a mip close to TEX, so this
    is a small adjustment and not a full-resolution downsample.
    """
    yi = (np.arange(TEX) * h // TEX).clip(0, h - 1)
    xi = (np.arange(TEX) * w // TEX).clip(0, w - 1)
    return np.ascontiguousarray(a[yi][:, xi])


# ----------------------------------------------------------------------
# The mat_* scaffold
#
# gpu_render.py's `if PBR:` block reads 11 keys and its `if VMAT:` block a
# further 11. Only four are geometry (vnormal, vtangent, face_matid,
# mat_paths); the rest are per-material texture-layer INDICES into `aux`
# plus scalar parameter columns. The texture side is blocked behind BC
# decode -- the same blocker as the 324/327 albedo slots -- so what is
# emitted here is a SCAFFOLD: every feature flag off, every scalar at its
# neutral value, every layer index pointing at a sentinel.
#
# THE DANGER IS THAT THIS WORKS. With the scaffold present the PBR path
# stops refusing and renders, and the output is plausible -- flat-lit,
# unmapped, but a picture. Anything measured against it is measuring the
# scaffold, not the map. So the pack carries `mat_scaffold: True` as a
# MACHINE-CHECKABLE marker rather than only a printed line, because a log
# line does not survive into whoever loads the pack three steps later.
# When real material data lands, that key goes to False and any consumer
# gating on it starts trusting the numbers on the same run.
#
# Column names come from the renderer's own _ext() call sites, not from a
# guess at the Source 2 feature set: gpu_render.py builds EXT from
# data["ext_keys"] specifically so the packer owns the layout, and reading
# them off the consumer is what keeps the two in step.
EXT_KEYS = [
    # feature flags -- all OFF in the scaffold
    "f_albedo_for_transmissive", "f_blend_effects2", "f_border_rough2",
    "f_metalness_tex", "f_overlay", "f_render_backfaces", "f_self_illum",
    "f_tintmask", "f_transmissive_ndotl",
    # slot-presence flags -- all ABSENT in the scaffold
    "has_h1", "has_h2", "has_met", "has_aov", "has_tr",
    # scalar parameters
    "h_scale1", "h_scale2", "h_zero1", "h_zero2",
    "blend_soft2", "bevel_soft2", "bevel_strength2",
    "border_offset2", "border_rough2", "border_soft2", "border_spread2",
    # The overlay's per-layer enables and per-layer tint-mask gates.
    # These four REPLACE the single `overlay_tintmask` boolean, which is
    # deleted rather than left beside them: it stood for "gate the
    # overlay by the tint mask", and the shader has no such switch. It
    # has g_nColorOverlayMode and g_nColorOverlayTintMask, both of them
    # LAYER SELECTORS, folded here into the booleans the shader actually
    # reads (glsl:1188, and SHARED_COLOR_OVERLAY.md for the derivation):
    #     ovl_lN    = (g_nColorOverlayMode     == 0) || (== N)
    #     ovl_maskN = (g_nColorOverlayTintMask == 0) || (== N)
    "ovl_l1", "ovl_l2", "ovl_mask1", "ovl_mask2",
    "ovl_bright_contrast", "ovl_dark_contrast",
    "si_albedo_factor",
    "tm_bright1", "tm_bright2", "tm_contrast1", "tm_contrast2",
    "vcmode1", "vcmode2",
]
# Neutral value per scalar column. A multiplier's neutral is 1 and an
# offset's is 0; getting that backwards is invisible while the feature
# flags are off and wrong the moment one is set.
EXT_NEUTRAL = {
    "h_scale1": 1.0, "h_scale2": 1.0,
    "tm_bright1": 1.0, "tm_bright2": 1.0,
    "tm_contrast1": 1.0, "tm_contrast2": 1.0,
    # ovl_bright_contrast / ovl_dark_contrast ARE NOT MULTIPLIERS, and
    # they used to be listed here at 1.0 with the six above. The reason
    # given at the time was that the consumer used them as lerp weights.
    # The consumer has since been replaced by the shader's own expression
    # (csgo_environment_blend_ps_c258_d0.glsl:1212), where they are
    # EXPONENTS as well as scales:
    #
    #     f = (1 - pow(1 - max(0,s), B)) * B + (pow(1 + min(0,s), D) - 1) * D + 1
    #
    # The conclusion survives the rewrite and is now derived from the
    # right function: at B = D = 0 both terms carry a factor of 0, so
    # f = 1 and the composite is the identity. 0.0 is the neutral. Note
    # the shader's own declared default for both is 1.0 -- a default is
    # not a neutral, and all 11 de_inferno materials that set the feature
    # set both values explicitly, so the default is never reached.
    #
    # Left OUT of the table rather than written as 0.0, so the column
    # takes the block's own zero fill and there is one fewer place for a
    # neutral to be asserted instead of derived.
}
# Sentinel aux layers. The renderer's own note says mat_nrm1 is >= 0 for
# EVERY material because the packer substitutes a flat layer for a missing
# normal map, and that mat_has_nrm is the real flag -- so the sentinel is
# expected, and the flag is where honesty lives. It is False everywhere
# here. RGB is a DIRECTION and alpha is Source 2 roughness, so the flat
# normal is (128,128,255) and not white.
AUX_FLAT_NRM, AUX_WHITE, AUX_BLACK, AUX_HALF = 0, 1, 2, 3


def build_mat_scaffold(n_mat, tex_edge):
    aux = np.zeros((4, tex_edge, tex_edge, 4), dtype=np.uint8)
    aux[AUX_FLAT_NRM] = (128, 128, 255, 255)   # +Z normal, roughness 1
    aux[AUX_WHITE] = 255                       # no occlusion / full mask
    aux[AUX_BLACK] = (0, 0, 0, 255)            # no emission / non-metal
    # NEUTRAL FOR A COMPOSITE IS NOT NEUTRAL FOR A MASK. 255 is right for
    # the occlusion and mask pages above and catastrophic for an overlay
    # page. This value was landed on measurement before the composite had
    # been read; the read now says WHY it is right, exactly. The shader
    # forms s = 2*texel - 1 and multiplies by a factor that is 1 at s = 0
    # (glsl:1211-1217), so the identity texel is 0.5 -- and sample_aux()
    # does no sRGB decode, so 128/255 = 0.502 IS that 0.5 to within
    # s = +0.004. Real overlay pages are written into this array in
    # LINEAR for the same reason.
    aux[AUX_HALF] = (128, 128, 128, 255)       # s = 2t-1 = 0, composite identity
    lay = lambda v: torch.full((n_mat,), v, dtype=torch.long)  # noqa: E731
    ext = torch.zeros((n_mat, len(EXT_KEYS)), dtype=torch.float32)
    for k, v in EXT_NEUTRAL.items():
        ext[:, EXT_KEYS.index(k)] = v
    return {
        "aux": torch.from_numpy(aux),
        "mat_nrm1": lay(AUX_FLAT_NRM), "mat_nrm2": lay(AUX_FLAT_NRM),
        "mat_ao": lay(AUX_WHITE), "mat_em": lay(AUX_BLACK),
        "mat_h1": lay(AUX_WHITE), "mat_h2": lay(AUX_WHITE),
        # mat_ovl gets AUX_HALF, not AUX_WHITE. See OVERLAY_PAGE_AND_K.md.
        # The overlay composite at mode 1 is
        #     o > 0.5 ? 1 - 2(1-rgb)(1-o) : 2*rgb*o
        # so o = 0.5 is its exact algebraic identity and o = 1.0 is PURE
        # WHITE at any k > 0. AUX_WHITE was the worst value available and
        # was here because 255 genuinely IS the neutral for the mask and
        # occlusion pages sharing this dict.
        # MEASURED, not merely argued: with this layer the overlay axis is
        # inert to within 0.0003 of not running at all on nrmse/KL_rgb/
        # KL_lum/exposure/ncc over holdout66, which is the correct
        # behaviour for an unloaded texture. AUX_WHITE instead cost nrmse
        # +0.0825 while IMPROVING KL_rgb by 0.0825 -- a placeholder that
        # flatters a headline metric.
        "mat_aov": lay(AUX_WHITE), "mat_ovl": lay(AUX_HALF),
        "mat_tm": lay(AUX_WHITE), "mat_tr": lay(AUX_BLACK),
        "mat_met": lay(AUX_BLACK), "mat_si": lay(AUX_BLACK),
        # metal, roughness, bump scale, emissive scale.
        "mat_pbr": torch.tensor([[0.0, 1.0, 1.0, 0.0]]).repeat(n_mat, 1),
        "mat_emtint": torch.ones((n_mat, 3), dtype=torch.float32),
        "mat_btint": torch.ones((n_mat, 3), dtype=torch.float32),
        "mat_ctint": torch.ones((n_mat, 3), dtype=torch.float32),
        # The real flag, and it is False: this pack has no normal maps.
        "mat_has_nrm": torch.zeros((n_mat,), dtype=torch.bool),
        "mat_ext": ext,
        "ext_keys": list(EXT_KEYS),
    }


def decode_normal(path, encoding):
    """(TEXxTEXx4 uint8 packed normal, None) or (None, reason).

    The ENCODING is not ours to choose and not ours to default. DXT5nm is
    the intuitive guess and it is wrong here: measured over de_inferno's
    323 layer-1 normal maps, reading them as DXT5nm puts x^2+y^2 > 1 on
    87.78% of texels, where the sqrt has no real root and z is clamped to
    a flat normal invented from nothing. Read as hemioct_diag, 0.31% are
    invalid and mean z is 0.9012 -- normals pointing out of the surface,
    as a tangent-space map must. Both produce a perfectly plausible image.
    """
    got = read_vtex(path, TEX)
    if got is None:
        return None, "container unreadable"
    fmt, w, h, blocks, fmtnum = got
    if blocks is None:
        return None, fmt.split(":")[0].lower()
    if _BC is None or not hasattr(_BC, "to_normal"):
        return None, "bc_decode has no to_normal()"
    try:
        rgba = np.asarray(_BC.decode(blocks, fmtnum, w, h))
        n = np.asarray(_BC.to_normal(rgba, encoding), dtype=np.float32)
    except Exception as e:                                # noqa: BLE001
        return None, f"{encoding} refused ({str(e)[:40]})"
    img = np.empty((h, w, 4), dtype=np.uint8)
    img[..., :3] = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    # Alpha is Source 2's roughness on this array, and this pack does not
    # know which source channel carries it for a hemi-oct slot. 255 is a
    # STAND-IN, not a measurement, and it is named in the DEFAULTED lines
    # rather than passed off as roughness data.
    img[..., 3] = 255
    return _fit(img, w, h), None


def convert(ash_dir, out_path, winding="auto", keep_tools=False,
            drop_effects=False,
            camera_out=None, frames=4, mat_paths_from=None,
            tex_dir=None, height_pages=False, overlay_pages=False,
            ao_pages=False):
    t0 = time.time()
    geo = torch.load(os.path.join(ash_dir, "geometry.pt"),
                     map_location="cpu", weights_only=False)
    models, instances = geo["models"], geo["instances"]
    if geo.get("problems"):
        log(f"  ash problems: {geo['problems']}")

    # Material params, if the extraction carried them. Only g_vColorTint is
    # used, and only to modulate the hash hue -- this is NOT a material pack.
    tints, texslot, alpha_of, defaulted = {}, {}, {}, []
    uvset_of = {}
    effects = set()
    rows_tp, rows_fam, rows_ip, rows_fp = {}, {}, {}, {}
    nrm_slot = nrm_enc = nrm_basis = None
    nrm_slot2 = nrm_enc2 = None
    nrm_enc_names = []
    alb_slot2 = []
    alpha_mode_ovl = alpha_cut_ovl = None
    mp = os.path.join(ash_dir, "materials.pt")
    if os.path.exists(mp):
        try:
            rows = torch.load(mp, map_location="cpu",
                              weights_only=False).get("rows", {})
            for k, v in rows.items():
                c = (v.get("vector_params") or {}).get("g_vColorTint")
                if c:
                    tints[k] = list(c)
                # ALPHA HANDLING IS PER MATERIAL and the renderer takes it
                # per face. Without it every alpha-masked surface renders
                # OPAQUE -- which is what the pale cards fanning out of the
                # trees were: foliage cut-out cards drawn as solid quads.
                # Nothing errors, the map just grows sails.
                ip = v.get("int_params") or {}
                fpp = v.get("float_params") or {}
                if ip.get("F_ALPHA_TEST"):
                    alpha_of[k] = (1, float(fpp.get(
                        "g_flAlphaTestReference",
                        fpp.get("g_flAlphaTestReference1", 0.5)) or 0.5))
                elif ip.get("F_TRANSLUCENT") or ip.get("F_ADDITIVE_BLEND"):
                    # F_ADDITIVE_BLEND is how csgo_effects declares smoke and
                    # steam. It is neither alpha-test nor F_TRANSLUCENT, so a
                    # mapping that checks only those two calls it OPAQUE --
                    # which is what drew de_inferno's steam cards as solid
                    # pale wedges hanging over the map. 160 faces, 576 m2,
                    # traced by ray-casting the pixels rather than guessed.
                    alpha_of[k] = (2, 0.5)
                # THE UV-SET SELECTOR, per material, for face_secondary.
                # gpu_render.py turns face_secondary into the per-vertex
                # `uv_set` that vsx.secondary_uv() switches on, and it is
                # the route --secondary-uv material actually drives. It
                # used to be an all-zero constant built below and never
                # read from anything, so that route was inert on every
                # map -- one of four independent zeros that together made
                # real TEXCOORD_1 land in the pack and change nothing.
                #
                # READ, not derived: the field is g_nUVSet1, the same one
                # fast_pack_fam.py's uvsel_c1 now reads. 2 = TEXCOORD_1,
                # 1 = TEXCOORD_0, and -1/0/absent mean "inherit the
                # previous slot", which for slot 1 is TEXCOORD_0. So the
                # test is >= 2 and everything else is 0, matching the
                # reference's own `_m0 == 2` stream test at
                # csgo_environment_blend_vs_max.glsl:346.
                uvset_of[k] = 1 if int(ip.get("g_nUVSet1", 0) or 0) >= 2 \
                    else 0
                if v.get("shader_name") == "csgo_effects.vfx":
                    effects.add(k)
                tp = v.get("texture_params") or {}
                rows_tp[k] = tp
                rows_fam[k] = v.get("shader_name") or ""
                # These two were read for the alpha mode on the line above
                # and then dropped, so every mat_ext column stayed at its
                # scaffold neutral even though the material had just told
                # us the answer. Keeping them costs one dict assignment.
                rows_ip[k] = ip
                rows_fp[k] = fpp
                # g_tColor is the albedo slot on every family here. The
                # other slots are normals/AO/masks and are NOT albedo --
                # binding one as colour would be a measured-looking wrong
                # texel, which is worse than an obvious checker.
                # g_tColor covers only 108 of de_inferno's 327 materials.
                # The two biggest families -- csgo_environment (148) and
                # csgo_environment_blend (69) -- do not declare it at all;
                # their albedo is the LAYER slot g_tColor1, 217 materials.
                # Taking layer 1 alone is a deliberate approximation: the
                # blend families composite layer 1 over layer 2 by a mask
                # this pack does not evaluate. Layer 1 is the base coat, so
                # it is the honest single-texture stand-in, and it is named
                # here rather than passed off as the composited result.
                for slot in ("g_tColor", "g_tColorTexture", "g_tColor1",
                             "g_tColorA"):
                    if tp.get(slot):
                        texslot[k] = tp[slot]
                        break
        except Exception as e:                            # noqa: BLE001
            defaulted.append(f"materials.pt unreadable ({e}) -> white tints")
    else:
        defaulted.append("no materials.pt -> white tints")

    V, UV, F, FM, LM, VC, VP = [], [], [], [], [], [], []
    UV2 = []
    mat_ids, mat_names = {}, []
    base = 0
    n_dc = n_missing_pos = n_tool = n_bad = n_effect = 0
    n_nomodel = n_nontri = n_nolm = n_tinted = 0
    n_vp = n_novp = 0
    missing_uv = 0
    n_uv2 = n_nouv2 = 0

    for inst in instances:
        m = models.get(inst["model"])
        if m is None:
            # An instance naming a model the extraction does not carry is a
            # MISSING PROP, not corrupt geometry. Counting it as "malformed"
            # alongside index-range failures read as data corruption at
            # 0.08% and was in fact 114 whole objects absent from 43 maps.
            n_nomodel += 1
            continue
        tr = np.asarray(inst.get("transform") or
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                        dtype=np.float32)
        R, T = tr[:, :3], tr[:, 3]
        # PER-DRAW-CALL TINT, recovered from the aggregate schema.
        # m_aggregateSceneObjects nests its tint inside m_aggregateMeshes,
        # keyed by m_nDrawCallIndex -- per draw call, not per instance,
        # which is why reading it at instance level found only None. 1691
        # of de_inferno's 6176 aggregate meshes carry a non-identity tint
        # and every one of them was discarded.
        dct = inst.get("draw_call_tint") or {}
        vbs, ibs = m["vertex_buffers"], m["index_buffers"]
        for _dci, dc in enumerate(m["draw_calls"]):
            if dc.get("primitive") not in (None, "RENDER_PRIM_TRIANGLES"):
                n_nontri += 1
                continue
            mat = dc.get("material") or "materials/dev/black_simple.vmat"
            # Register the material BEFORE dropping tool faces. Dropping the
            # tool material from the NUMBERING as well would shift every path
            # sorting after materials/tools/*, silently offsetting this pack
            # against any side table built from the same material set.
            # The faces go; the row stays.
            if mat not in mat_ids:
                mat_ids[mat] = len(mat_names)
                mat_names.append(mat)
            if not keep_tools and mat.startswith(TOOL_PREFIXES):
                n_tool += 1
                continue
            # csgo_effects.vfx is PARTICLE geometry -- steam and smoke cards
            # -- not world surface. It reached the world tri-soup as opaque
            # quads and drew de_inferno's pale wedges. F_ADDITIVE_BLEND
            # cannot be expressed by the base raster path, so carrying the
            # faces with alpha mode 2 changes their shade and still leaves
            # sails. They belong to fam_nonworld's particle path.
            #
            # RETRACTED AS A DEFAULT. F_ADDITIVE_BLEND turned out to be
            # fully implemented -- the renderer composites these dst+src*a
            # and reports 28190 of 32734 BLEND px composited on de_inferno.
            # Dropping them was the packer making a rendering decision it
            # had no standing to make, on the strength of a grep for GPU-API
            # names (depth_write, blend_dst, src_alpha) in a renderer that
            # composites with tensor arithmetic. --drop-effects still exists
            # for a run with --additive-blend off, where they draw opaque.
            if drop_effects and mat in effects:
                n_effect += 1
                continue
            # A draw call binds SEVERAL vertex buffers as PARALLEL streams.
            bound = [vbs[b["m_hBuffer"]] for b in (dc.get("vertex_buffers")
                     or [{"m_hBuffer": 0}]) if b["m_hBuffer"] < len(vbs)]
            pos = uv = lmv = vpt = uv2 = None
            for vb in bound:
                if pos is None and "POSITION_0" in vb["attrs"]:
                    pos = vb["attrs"]["POSITION_0"]
                if uv is None and "TEXCOORD_0" in vb["attrs"]:
                    uv = vb["attrs"]["TEXCOORD_0"]
                # TEXCOORD_1 is the SECONDARY UV -- the one
                # g_nColorOverlayUVSet's default `2 = UV2` selects, and
                # the one F_SECONDARY_UV binds the second layer stack to.
                # It was in the extraction all along, on 286 of 1428
                # vertex buffers (20.0%), and this file shipped
                # `uvs2 = uvs` and logged it as DEFAULTED. Every
                # texture-space term that reads UV2 was therefore sampled
                # at layer 1's coordinate -- a SPATIAL error, which is the
                # signature ncc is sensitive to and nrmse is not, and
                # that mismatch is what put this on the critical path.
                if uv2 is None and "TEXCOORD_1" in vb["attrs"]:
                    uv2 = vb["attrs"]["TEXCOORD_1"]
                # TEXCOORD_3 is the LIGHTMAP UV, identified by the same
                # luxel-density criterion gpu_render.py:4130 uses to pick
                # it and :4190 uses to validate it: median 2^-8.75 uv/m,
                # 100% inside the 2^-11.5..2^-7.5 band, against
                # TEXCOORD_0's 2^-0.57 and 0%. Eight octaves apart, so the
                # selection is not a judgement call.
                #
                # It was in the extraction all along -- 755 of 784 models,
                # 8.4M vertices -- and this file shipped lmuv as zeros
                # because no lightmap TEXTURE had been extracted, which
                # made the addressing look moot. It was not: the texture
                # was six files away in the same archive.
                if lmv is None and "TEXCOORD_3" in vb["attrs"]:
                    lmv = vb["attrs"]["TEXCOORD_3"]
                # TEXCOORD_4.x is the csgo_environment_blend LAYER WEIGHT --
                # the vertex paint. NOT COLOR_0. gpu_render.py's own
                # --blend-channel help records the measurement that settled
                # it (blend_diag5.py, 271 M vertices of the 365
                # csgo_environment_blend primitives): COLOR_0 is grey with
                # alpha 1.0 and modes at exactly 1.0/0.8/0.6/0.4, 66.9% of
                # verts at 1.0 -- a per-vertex TINT. Reading it as the
                # weight puts two thirds of every blended surface on pure
                # layer 2 and the rest on pure layer 1, which is what
                # de_inferno's courtyard was doing: GT pavesit in flagstone
                # and we rendered dirt.
                #
                # The stream is used by exactly the shader that needs it --
                # non-zero on 12.9% of blend verts against 0.1% of every
                # other shader's -- and corr(TEXCOORD_4.x, COLOR_0.r) is
                # 0.229, so the two are not substitutes for one another.
                #
                # It was in the extraction all along: 287 vertex buffers on
                # de_inferno. This file read TEXCOORD_0 and _3 and stopped,
                # so the renderer had no weight to blend on and fell back to
                # a knob whose own help text says it is the wrong stream.
                # Same failure as lmuv, one attribute over.
                if vpt is None and "TEXCOORD_4" in vb["attrs"]:
                    vpt = vb["attrs"]["TEXCOORD_4"]
            if pos is None:
                n_missing_pos += 1
                continue
            ib = ibs[dc.get("index_buffer", {}).get("m_hBuffer", 0)]
            si, ic = int(dc["start_index"]), int(dc["index_count"])
            idx = ib["indices"][si:si + ic].to(torch.int64) \
                + int(dc.get("base_vertex") or 0)
            if idx.numel() < 3:
                continue
            idx = idx[:(idx.numel() // 3) * 3]
            if int(idx.max()) >= pos.shape[0]:
                n_bad += 1
                continue
            # Compact to the vertices this draw call actually touches, so the
            # pack carries ~15M vertices rather than 25M loose corners.
            uniq, inv = torch.unique(idx, return_inverse=True)
            p = pos[uniq].numpy().astype(np.float32)
            p = p @ R.T + T                          # instance transform
            # inches/Z-up -> metres/Y-up, matching view_matrix()'s convention.
            V.append(np.stack([p[:, 1], p[:, 2], p[:, 0]], axis=1) * S)
            if uv is not None and uv.shape[0] >= pos.shape[0]:
                UV.append(uv[uniq].numpy().astype(np.float32))
            else:
                UV.append(np.zeros((uniq.numel(), 2), dtype=np.float32))
                missing_uv += 1
            # A buffer with no TEXCOORD_1 falls back to UV1, which is what
            # the shader does too: a single-layer material has no second
            # coordinate and its UV2 reads are the same surface. That is a
            # fallback with a reason, not padding, and it is counted.
            if uv2 is not None and uv2.shape[0] >= pos.shape[0]:
                UV2.append(uv2[uniq].numpy().astype(np.float32))
                n_uv2 += 1
            else:
                UV2.append(UV[-1])
                n_nouv2 += 1
            if lmv is not None and lmv.shape[0] >= pos.shape[0]:
                LM.append(lmv[uniq].numpy().astype(np.float32))
            else:
                # Zeros are the NO-LIGHTMAP signal lm_valid tests for at
                # :4190; they are not padding. A model with no TEXCOORD_3
                # genuinely has no lightmap chart.
                LM.append(np.zeros((uniq.numel(), 2), dtype=np.float32))
                n_nolm += 1
            # Zero is the shader's own "pure layer 1", not padding: the
            # remap clamp(w*1.1-0.05,0,1) at glsl:260 sends everything
            # below 0.0455 to exactly layer 1. A model with no TEXCOORD_4
            # is not blended, so 0 is the correct value and not a gap.
            if vpt is not None and vpt.shape[0] >= pos.shape[0]:
                _vp = vpt[uniq].numpy().astype(np.float32)
                VP.append(_vp[:, 0] if _vp.ndim > 1 else _vp)
                n_vp += 1
            else:
                VP.append(np.zeros(uniq.numel(), dtype=np.float32))
                n_novp += 1
            F.append(inv.numpy().astype(np.int64).reshape(-1, 3) + base)
            # Tint is 0-255 per channel; identity is (255,255,255). It
            # multiplies albedo, so it rides as a per-vertex colour --
            # the carrier gpu_render.py already reads at :5058 and
            # defaults to ones.
            _t = dct.get(_dci) or dct.get(str(_dci))
            if _t is None:
                _t = dc.get("tint")
                _t = None if _t is None else [c * 255.0 for c in _t[:3]]
            if _t is None:
                VC.append(np.ones((uniq.numel(), 4), dtype=np.float32))
            else:
                _c4 = np.ones((uniq.numel(), 4), dtype=np.float32)
                _c4[:, :3] = np.asarray(_t[:3], dtype=np.float32) / 255.0
                VC.append(_c4)
                if tuple(_t[:3]) != (255.0, 255.0, 255.0):
                    n_tinted += 1
            base += int(uniq.numel())
            FM.append(np.full(idx.numel() // 3, mat_ids[mat], dtype=np.int32))
            n_dc += 1

    if not F:
        raise RuntimeError("no drawable triangles survived the flattening")

    vertices = torch.from_numpy(np.concatenate(V))
    uvs = torch.from_numpy(np.concatenate(UV))
    # NON-FINITE UVs CRASH THE RENDERER, AND NOT WHERE THEY ENTER.
    # sample_textures computes its index as (uv % 1.0) * (T-1), floored to
    # long. On a NaN that is an arbitrary huge integer, so the failure is a
    # CUDA device-side assert inside an indexing kernel, with a traceback
    # pointing at textures[layer, v0, u0] and every shipped index array
    # provably in bounds. de_boulder carries 48 such components -- 1 map of
    # 43 -- and it took a repro plus instrumentation to get from the assert
    # back to here, because the pack that produced it looks clean by every
    # check that examines the arrays it ships.
    #
    # Zeroed, not clamped and not dropped: a non-finite texture coordinate
    # has no defensible value, so the vertex samples texel (0,0) and the
    # count is printed. Dropping the faces would hide which models carry
    # them; leaving them crashes a map.
    _uvbad = ~torch.isfinite(uvs)
    n_uvbad = int(_uvbad.any(dim=1).sum())
    if n_uvbad:
        uvs = torch.where(_uvbad, torch.zeros_like(uvs), uvs)
    faces = torch.from_numpy(np.concatenate(F).astype(np.int32))
    lmuv = torch.from_numpy(np.concatenate(LM))
    vcolor = torch.from_numpy(np.concatenate(VC))
    vpaint = torch.from_numpy(np.concatenate(VP))
    # Report the SHAPE of the stream, not just that it was found. Vertex
    # paint is expected to be mostly-zero (92.9% over the reference pack)
    # because most surfaces are not blended at all -- so "mean is small"
    # is the healthy reading and an all-zero stream is the failure. Say
    # which, here, rather than leaving a later reader to infer it from a
    # flat render.
    _vpnz = float((vpaint > 0).float().mean())
    log(f"  vertex paint (TEXCOORD_4.x): {n_vp} draw calls carried it, "
        f"{n_novp} did not; {_vpnz:.1%} of vertices non-zero, "
        f"mean {float(vpaint.mean()):.4f}, max {float(vpaint.max()):.3f}")
    if n_vp and _vpnz == 0.0:
        log("  VERTEX PAINT ALL ZERO despite being present -- the layer-2 "
            "weight would silently render every blended surface as pure "
            "layer 1. Treat this as a read failure, not an absence.")
    fm = np.concatenate(FM)
    del V, UV, F, FM, VP

    # MATERIAL NUMBERING IS A CROSS-AGENT CONTRACT, NOT A LOCAL CHOICE.
    # fam_side's table is built with row i == sorted(mat_paths)[i], and the
    # renderer indexes those rows by face_matid. Numbering materials in
    # first-seen draw order instead -- the obvious thing, since that is the
    # order they arrive in -- puts every face on some OTHER material's
    # parameters, and the frame still renders, so nothing announces it.
    # Sort, and ship mat_paths alongside so the two tables can be COMPARED
    # rather than assumed aligned: `world["mat_paths"] == fam["mat_paths"]`.
    # Sorting is necessary but NOT sufficient: sorted position only aligns
    # two tables that hold the SAME SET. Measured against the real
    # fam_side.pt, they do not -- 327 material paths here against 340 there,
    # mine a strict subset, the 13 extra being tools/skybox materials that a
    # .vmat_c scan sees and no draw call ever names. Sorting both then
    # disagrees from the first insertion onward, every face lands on a
    # neighbouring material's row, and the frame still renders. So when an
    # authoritative table is given, its ordering is ADOPTED outright rather
    # than independently re-derived and hoped to match.
    auth = None
    if mat_paths_from:
        t = torch.load(mat_paths_from, map_location="cpu", weights_only=False)
        auth = list(t.get("mat_paths") or [])
        extra = sorted(set(mat_names) - set(auth))
        if not auth:
            log(f"  WARNING: {mat_paths_from} carries no mat_paths; "
                f"falling back to this pack's own sorted order")
            auth = None
        elif extra:
            # Reported, never silently remapped: a material of ours absent
            # from theirs has no row to land on, and inventing one would put
            # real faces on invented parameters.
            log(f"  REFUSED to adopt {mat_paths_from}: {len(extra)} materials "
                f"here are absent from it, e.g. {extra[:3]}. This pack is NOT "
                f"a subset of that table, so no alignment exists to adopt. "
                f"Keeping this pack's own sorted order; the two sides must be "
                f"reconciled, not remapped.")
            auth = None
    if auth is not None:
        # The same table carries the normal slot and its ESTABLISHED
        # encoding per row, in this ordering. Taking them here, from the
        # object whose ordering we just adopted, is what keeps row j
        # meaning the same material in both.
        alb_slot2 = list(t.get("albedo_slot2") or [])
        alpha_mode_ovl = t.get("alpha_mode_ovl")
        alpha_cut_ovl = t.get("alpha_cut_ovl")
        nrm_basis = t.get("nrm_enc_basis")
        nrm_slot = t.get("nrm_slot1")
        nrm_enc = t.get("nrm_enc1")
        # LAYER 2 WAS NEVER READ. The side table has carried nrm_slot2 /
        # nrm_enc2 all along and this file took only the `1` columns, then
        # set mat_nrm2 = mat_nrm1.clone() -- so on a blend material the
        # second layer was shaded with the FIRST layer's normal map. That
        # is not a missing feature, it is a wrong one: it looks like a
        # normal map, blends on the right weight, and is the wrong image.
        nrm_slot2 = t.get("nrm_slot2")
        nrm_enc2 = t.get("nrm_enc2")
        nrm_enc_names = list(t.get("nrm_enc_names") or [])
        idx_of = {p: i for i, p in enumerate(auth)}
        remap = np.asarray([idx_of[p] for p in mat_names], dtype=np.int32)
        log(f"  adopted mat_paths from {os.path.basename(mat_paths_from)}: "
            f"{len(mat_names)} of {len(auth)} rows carry faces, "
            f"{len(auth) - len(mat_names)} unused")
        mat_names = auth
    else:
        order = sorted(range(len(mat_names)), key=lambda i: mat_names[i])
        remap = np.empty(len(mat_names), dtype=np.int32)
        remap[np.asarray(order, dtype=np.int64)] = np.arange(len(order),
                                                            dtype=np.int32)
        mat_names = [mat_names[i] for i in order]
    face_mat = torch.from_numpy(remap[fm])
    del fm

    def normals(fa):
        p = vertices[fa.long()]
        n = torch.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], dim=1)
        return n / n.norm(dim=1, keepdim=True).clamp(min=1e-9), \
            n.norm(dim=1) * 0.5

    fn, area = normals(faces)
    # Area-weighted vote over NEAR-HORIZONTAL faces. In a playable map the
    # standable floor area exceeds the ceiling area, so if the up-facing share
    # is the minority the winding is inverted.
    #
    # The vote must be DECISIVE to act on, and the margin is not cosmetic.
    # As-extracted is the physically motivated default -- the axis map is a
    # cyclic permutation, so winding is carried through untouched -- and two
    # real maps confirm it (de_inferno 71.2% up, de_nuke 84.2%). A bare
    # `frac < 0.5` test then flips on noise: de_nuke_vanity voted 49.6% over
    # 4531 m2 and lobby_mapveto 49.7% over ~0 m2, and both got inverted by a
    # coin toss. That is worse than not deciding, because the 43 packs would
    # no longer share one convention and the inconsistency is invisible
    # downstream. Genuine inversions are not close: the four scenes that
    # really are wound the other way voted 0.0-1.2% up. So flip only well
    # below the tie, require some horizontal area to vote at all, and say so
    # out loud when the evidence is too thin to decide either way.
    horiz = fn[:, 1].abs() > 0.9
    h_area = float(area[horiz].sum())
    up = float((area[horiz] * (fn[horiz, 1] > 0)).sum())
    dn = float((area[horiz] * (fn[horiz, 1] < 0)).sum())
    frac = up / max(up + dn, 1e-9)
    decisive = h_area > 1.0 and (frac < 0.35 or frac > 0.65)
    flipped = False
    if winding == "flip" or (winding == "auto" and decisive and frac < 0.35):
        faces = faces[:, [0, 2, 1]].contiguous()
        fn, area = normals(faces)
        flipped = True
        frac = 1.0 - frac
    log(f"  horizontal faces: {h_area:.0f} m2, "
        f"{frac * 100:.1f}% point +Y (up) after winding="
        f"{'FLIPPED' if flipped else 'as-extracted'}")
    if winding == "auto" and not decisive:
        log(f"  UNDECIDED: the floor-normal vote is {frac * 100:.1f}% up over "
            f"{h_area:.0f} m2 -- too close to a tie to act on. Winding kept "
            f"as-extracted, which is what every decided map agrees on. The "
            f"up-axis of this pack is NOT confirmed.")

    layers, n_real, why = [], 0, {}
    for n in mat_names:
        img = None
        rel = texslot.get(n)
        if tex_dir and not rel:
            # Counted, not silent. Without this the summary reports a real
            # percentage and an unexplained remainder: 477 of the fleet's
            # 479 checkers had NO reason recorded, because a material with
            # no albedo slot never reaches the decoder that records one.
            # An uncounted fallback is the same defect as an uncounted
            # failure -- it just reads as success.
            why["no albedo slot declared"] = \
                why.get("no albedo slot declared", 0) + 1
        if tex_dir and rel:
            img, reason = decode_albedo(os.path.join(tex_dir, rel + "_c"))
            if img is None:
                why[reason] = why.get(reason, 0) + 1
            else:
                n_real += 1
        layers.append(img if img is not None
                      else mat_color(n, tints.get(n, (1.0, 1.0, 1.0, 1.0))))
    textures = torch.from_numpy(np.stack(layers))
    if tex_dir:
        log(f"  textures: {n_real}/{len(layers)} REAL texels, "
            f"{len(layers) - n_real} hash-hue checkers")
        for r, c in sorted(why.items(), key=lambda kv: -kv[1]):
            log(f"    checker because {r}: {c}")
    else:
        defaulted.append(f"textures: {len(layers)} synthesised {TEX}x{TEX} "
                         f"hash-hue checkers (no --tex-dir given)")
    # THIS LINE USED TO LIE. It claimed "lmuv = zeros (no lightmap in this
    # pack)" unconditionally -- written when that was true, never revisited
    # once TEXCOORD_3 was bound, and still printing on packs whose lmuv is
    # 99.9% populated. A DEFAULTED line is a claim about what is MISSING,
    # so a stale one is worse than no line: anyone grepping DEFAULTED to
    # find the holes is handed a hole that was filled two commits ago, and
    # anyone checking whether lightmapping landed is told it did not.
    # Report the measured state instead of a remembered one.
    uvs2_t = None
    if n_uv2:
        # Built here where UV2 is in scope, ATTACHED at the pack dict
        # below -- `pack` does not exist yet at this point, and the first
        # version of this assigned into it anyway and died on an
        # UnboundLocalError before writing a byte. Caught by running it.
        uvs2_t = torch.from_numpy(np.concatenate(UV2))
        log(f"  uvs2: TEXCOORD_1 on {n_uv2}/{n_uv2 + n_nouv2} draw calls "
            f"({100.0 * n_uv2 / max(1, n_uv2 + n_nouv2):.1f}%); the rest "
            f"fall back to UV1, which is what a single-layer material's "
            f"UV2 reads are anyway. This slot used to be `uvs2 = uvs` "
            f"unconditionally and was logged as DEFAULTED.")
    else:
        defaulted.append("uvs2 = uvs (no TEXCOORD_1 found in ANY buffer)")
    _lmnz = float((lmuv != 0).any(dim=1).float().mean())
    if _lmnz == 0.0:
        defaulted.append("lmuv = zeros (no TEXCOORD_3 on any draw call in "
                         "this pack): the renderer's lm_valid reads that "
                         "as no-lightmap, which is correct here")
    else:
        log(f"  lmuv: {_lmnz:.1%} of vertices carry a lightmap chart "
            f"(TEXCOORD_3); the remainder are the no-lightmap zeros "
            f"lm_valid tests for, not padding")

    # LAYER 2. `textures` is (n_mat, T, T, 4) -- one slot per material --
    # so a 2-layer material cannot carry its stack in it. The renderer
    # already has the mechanism: face_mat2 indexes a SECOND entry in the
    # same array and face_has2 says whether the face has one. So the
    # layer-2 pages are appended past the per-material block and pointed
    # at per face. Layer 3 has no equivalent key and is NOT expressible
    # here -- 295 rows declare one and this pack cannot carry them, which
    # is a shape limit and is stated rather than counted as coverage.
    lay2, mat2_of, n_l2 = [], {}, 0
    # alb_slot2 arrived ONLY from an adopted fam_side_<map>.pt. Without
    # --mat-paths it stays empty, the block below is skipped, and the pack
    # ships 0 layer-2 pages -- the exact failure this file's own --mat-paths
    # note describes: "327-texture pack where the sweep produces 399, with
    # face_has2 at 0.0% instead of 31.3%, and nothing said so". A run of mine
    # produced 327 and 0.0%, so the note was describing my run before I made
    # it.
    #
    # The side table was never the source of that fact. materials.pt carries
    # it: on de_inferno g_tColor2 is declared by 69 materials and there are
    # exactly 69 csgo_environment_blend.vfx materials -- the match is exact,
    # so every blend material names its own second layer. Reading it here
    # makes the side table an OVERRIDE rather than the only route, and a
    # pack built without one stops silently losing a third of its faces.
    if not alb_slot2 and rows_tp:
        alb_slot2 = []
        for nm in mat_names:
            tp2 = rows_tp.get(nm) or {}
            for slot in ("g_tColor2", "g_tLayer2Color", "g_tColorB"):
                if tp2.get(slot):
                    alb_slot2.append(slot)
                    break
            else:
                alb_slot2.append("")
        _declared = sum(1 for x in alb_slot2 if x)
        log(f"  layer 2 slots from materials.pt: {_declared}/"
            f"{len(mat_names)} materials declare a second albedo "
            f"(no side table adopted)")
    if tex_dir and alb_slot2:
        for j, nm in enumerate(mat_names):
            slot = alb_slot2[j] if j < len(alb_slot2) else ""
            if not slot:
                continue
            rel = (rows_tp.get(nm) or {}).get(slot)
            if not rel:
                continue
            img, _r = decode_albedo(os.path.join(tex_dir, rel + "_c"))
            if img is None:
                continue
            mat2_of[j] = len(layers) + len(lay2)
            lay2.append(img)
            n_l2 += 1
    if lay2:
        textures = torch.from_numpy(np.stack(layers + lay2))
    fm_np = face_mat.numpy()
    m2 = np.zeros(len(mat_names), dtype=np.int32)
    h2 = np.zeros(len(mat_names), dtype=np.int8)
    for j, k in mat2_of.items():
        m2[j], h2[j] = k, 1
    pack_face_mat2 = torch.from_numpy(m2[fm_np])
    pack_face_has2 = torch.from_numpy(h2[fm_np])
    # face_mat2 / face_has2 / face_secondary / face_params are ONE GROUP.
    # gpu_render.py tests only face_mat2 for absence and, finding it,
    # skips the branch that defaults ALL FOUR -- so supplying the pair
    # alone leaves the other two as None and the renderer dies on
    # `face_sec_0.to(device)`. Emitting a subset of a coupled group is
    # worse than emitting none of it: the absence test is on one member.
    # face_params defaults to the renderer's own 15 ones.
    # face_secondary IS NOW READ. It was `torch.zeros(len(faces))` -- a
    # constant built here and derived from nothing -- so vsx.secondary_uv
    # saw uv_set == 1 on every vertex and `--secondary-uv material` could
    # not select TEXCOORD_1 anywhere, on any map. That was one of four
    # independent zeros behind "real TEXCOORD_1 lands in the pack and
    # changes nothing", and the only one on this file.
    _sec_by_mat = np.zeros(len(mat_names), dtype=np.int8)
    for _j, _nm in enumerate(mat_names):
        _sec_by_mat[_j] = uvset_of.get(_nm, 0)
    pack_face_sec = torch.from_numpy(_sec_by_mat[fm_np])
    _n_sec = int((_sec_by_mat != 0).sum())
    _n_sec_f = int((pack_face_sec != 0).sum())
    log(f"  face_secondary: {_n_sec}/{len(mat_names)} materials select "
        f"TEXCOORD_1 via g_nUVSet1 >= 2, covering {_n_sec_f}/{len(faces)} "
        f"faces ({100.0 * _n_sec_f / max(len(faces), 1):.2f}%)"
        + ("  -- a MEASURED zero: no material on this map asks for the "
           "secondary UV set, so --secondary-uv is correctly inert here "
           "and an identical frame with and without it proves nothing "
           "about the term" if _n_sec == 0 else ""))
    pack_face_params = torch.ones(len(faces), 15, dtype=torch.float32)
    log(f"  layer 2: {n_l2}/{len(mat_names)} materials carry a second "
        f"albedo page (textures now {textures.shape[0]} entries)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    # Per-vertex normals, area-weighted by construction: each face's
    # UNNORMALISED cross product is accumulated to its three corners, so a
    # large triangle contributes proportionally more than a sliver. This is
    # synthesised from the geometry, NOT the NORMAL_0 attribute the models
    # carry -- those are packed under m_bUseCompressedNormalTangent and
    # decoding them is a separate question. Smooth-shading a hard edge is
    # the cost; the alternative is the PBR path having no vnormal at all.
    vnormal = torch.zeros_like(vertices)
    fl = faces.long()
    fw = fn * (2.0 * area).unsqueeze(1)
    for k in range(3):
        vnormal.index_add_(0, fl[:, k], fw)
    # A vertex whose every incident face is degenerate accumulates nothing,
    # and dividing by the clamp leaves it length ZERO rather than unit --
    # which reads as a normal pointing nowhere, silently, since nothing
    # downstream re-checks the length. Substitute +Y so the frame is always
    # well defined, and count them rather than let the substitution hide
    # how much geometry is degenerate.
    vlen = vnormal.norm(dim=1, keepdim=True)
    n_nonorm = int((vlen.squeeze(1) < 1e-9).sum())
    up_axis = torch.zeros_like(vnormal)
    up_axis[:, 1] = 1.0
    vnormal = torch.where(vlen > 1e-9, vnormal / vlen.clamp(min=1e-9), up_axis)

    # Per-vertex tangent frame from UV DERIVATIVES -- the surface direction
    # in which u increases -- accumulated per face and orthogonalised
    # against vnormal. This is the basis a tangent-space normal map is
    # expressed in, so it has to come from the same UVs the map is sampled
    # with; any other unit vector perpendicular to the normal is a
    # plausible-looking frame that rotates every normal-mapped detail.
    #
    # vtangent is Nx4: xyz plus a HANDEDNESS SIGN in w, which gpu_render.py
    # reads as vtangent[:, 3].sign(). Mirrored UV shells -- common on map
    # geometry, where an artist flips a chart to reuse texture space --
    # have the opposite bitangent, and dropping the sign flips their normal
    # maps inside out while leaving everything else correct.
    p = vertices[fl]
    q = uvs[fl]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    # A degenerate UV triangle has no tangent to give. Zero it rather than
    # dividing by ~0 and accumulating an enormous vector into its corners,
    # which would corrupt the frame of every vertex it touches.
    ok = det.abs() > 1e-12
    r = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)),
                    torch.zeros_like(det)).unsqueeze(1)
    tf = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    bf = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    tacc = torch.zeros_like(vertices)
    bacc = torch.zeros_like(vertices)
    for k in range(3):
        tacc.index_add_(0, fl[:, k], tf)
        bacc.index_add_(0, fl[:, k], bf)
    # Gram-Schmidt against the normal.
    t = tacc - vnormal * (vnormal * tacc).sum(1, keepdim=True)
    tn = t.norm(dim=1, keepdim=True)
    # Vertices with no usable UV gradient get an arbitrary perpendicular
    # rather than a zero vector: gpu_render.py gates on
    # vtangent[:, :3].norm() > 0.5, so a zero would read as "no frame".
    fallback = torch.zeros_like(vnormal)
    fallback[:, 0] = 1.0
    alt = torch.zeros_like(vnormal)
    alt[:, 1] = 1.0
    fallback = torch.where((vnormal[:, 0:1].abs() > 0.9), alt, fallback)
    fallback = fallback - vnormal * (vnormal * fallback).sum(1, keepdim=True)
    fallback = fallback / fallback.norm(dim=1, keepdim=True).clamp(min=1e-9)
    t = torch.where(tn > 1e-9, t / tn.clamp(min=1e-9), fallback)
    hand = torch.sign((torch.cross(vnormal, t, dim=1) * bacc).sum(1))
    hand = torch.where(hand == 0, torch.ones_like(hand), hand)
    vtangent = torch.cat([t, hand.unsqueeze(1)], dim=1)
    # THE BITANGENT IS RECONSTRUCTED, SO MEASURE THE RECONSTRUCTION.
    # The consumer rebuilds B = cross(N, T) * w, which is exact only where
    # the UV chart is unskewed. Where it is skewed the true bitangent is
    # not perpendicular to T, and the error shows up as ordinary-looking
    # shading -- so it cannot be found by looking at a frame. Compare the
    # reconstruction against the bitangent accumulated straight from the
    # UV derivatives and report the angular deviation, which is the
    # falsifiable form of the assumption.
    # SUBSAMPLE FIRST. Computed over every vertex this needs four 21M x 3
    # float temporaries on de_boulder and the sweep was OOM-killed -- a
    # diagnostic that takes down the run it is reporting on is worse than
    # no diagnostic. A deterministic stride keeps the figure reproducible
    # from the pack, and 2M vertices is far more than these quantiles need.
    _step = max(1, vertices.shape[0] // 2_000_000)
    _sl = slice(None, None, _step)
    b_rec = torch.cross(vnormal[_sl], t[_sl], dim=1) * hand[_sl].unsqueeze(1)
    b_s = bacc[_sl]
    b_len = b_s.norm(dim=1)
    ok_b = b_len > 1e-9
    if int(ok_b.sum()):
        b_meas = b_s[ok_b] / b_len[ok_b].unsqueeze(1)
        cosang = (b_rec[ok_b] * b_meas).sum(1).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cosang))
        q = torch.quantile(ang, torch.tensor([0.5, 0.95, 0.99]))
        defaulted.append(
            f"vtangent B is RECONSTRUCTED as cross(N,T)*w; measured against "
            f"the UV-derivative bitangent on a 1-in-{_step} sample "
            f"({int(ok_b.sum())} verts) the angular deviation is median "
            f"{float(q[0]):.2f} deg, p95 {float(q[1]):.2f}, p99 "
            f"{float(q[2]):.2f}, max {float(ang.max()):.2f} "
            f"({100 * float((ang > 5).float().mean()):.2f}% over 5 deg)")
        del b_meas, cosang, ang
    del b_rec, b_s, b_len, ok_b, tacc, bacc
    n_degen = int((~ok).sum())
    defaulted.append("vnormal/vtangent synthesised from geometry and UVs "
                     "(NORMAL_0 is packed and not decoded); "
                     f"{n_degen} faces had degenerate UVs and contributed "
                     f"no tangent; {n_nonorm} vertices had no non-degenerate "
                     f"face at all and were set to +Y")

    # face_matid == face_mat: one numbering, sorted material path, used both
    # to index `textures` and to index the fam side table. Two numberings
    # would be a second thing to keep aligned for no gain.
    pack = {
        # uvs2 is NOT shipped: gpu_render.py already does
        # data.get("uvs2", data["uvs"]), so a clone was 170 MB per big map
        # to say "same as uvs". lmuv IS shipped -- its zeros are the
        # no-lightmap signal that lm_valid tests for, and the renderer
        # defaults an absent lmuv to uvs, which is a different claim.
        "vertices": vertices, "uvs": uvs,
        # Real TEXCOORD_1 where the buffer had one; absent means every
        # buffer fell back, and gpu_render's data.get("uvs2", data["uvs"])
        # is then the same answer without the 170 MB clone.
        **({"uvs2": uvs2_t} if uvs2_t is not None else {}),
        # vpaint IS shipped, and as a pack key rather than the `--vpaint`
        # SIDECAR the renderer also accepts. A sidecar the canonical
        # invocation does not pass is loaded-or-not by accident, which is
        # exactly how the layer weight came to be silently absent: the
        # renderer implements the blend correctly, down to the glsl:260
        # dead-zone remap, and had nothing to blend on.
        "lmuv": lmuv, "vcolor": vcolor, "vpaint": vpaint,
        "faces": faces, "face_mat": face_mat, "face_matid": face_mat,
        "face_normals": fn, "vnormal": vnormal, "vtangent": vtangent,
        "textures": textures,
        "face_mat2": pack_face_mat2, "face_has2": pack_face_has2,
        # face_params is NOT shipped: it is all-ones, a constant
        # gpu_render.py builds itself, and shipping it cost 1173 MB on
        # de_boulder alone, resident in VRAM because the world loads with
        # map_location=device. Requires the per-key defaults at :5058;
        # a renderer older than that gates all four on face_mat2 and will
        # die on face_sec_0.to(device).
        #
        # face_secondary IS shipped now -- but only when it is not the
        # all-zero constant the renderer would build anyway. It stopped
        # being a constant the moment it started being READ from
        # g_nUVSet1, and the VRAM argument only ever applied to constants.
        # On a map where no material selects TEXCOORD_1 the key is omitted
        # exactly as before and the renderer's default is correct; on one
        # where some do, omitting it would silently discard the read.
        **({"face_secondary": pack_face_sec}
           if bool((pack_face_sec != 0).any()) else {}),
        "mat_paths": mat_names,
    }
    # 0 = OPAQUE, 1 = MASK, 2 = BLEND, per face, from the material.
    am = np.zeros(len(mat_names), dtype=np.int64)
    ac = np.full(len(mat_names), 0.5, dtype=np.float32)
    for i, nm in enumerate(mat_names):
        if nm in alpha_of:
            am[i], ac[i] = alpha_of[nm]
    # THE SIDE TABLE'S ANSWER IS STRICTLY BETTER THAN THIS ONE, so adopt it
    # where it exists. Established by decal-vfx: the derivation above reads
    # the material's own F_ALPHA_TEST/F_TRANSLUCENT/F_ADDITIVE_BLEND, which
    # is right for most families and wrong for the overlay/decal ones --
    # 25 decal materials, 23,943 faces, OPAQUE on 16,631 of them, 5.10% of
    # visible pixels drawn with the wrong blend state. alpha_mode_ovl is
    # already loaded, already correct, and was being ignored one branch
    # away, which is the same shape as vpaint and alb_slot2.
    #
    # REPORT THE DIFF, never remap silently: two sources for one fact must
    # be reconciled out loud or the quieter one is lost the next time
    # somebody wonders which was used. The cutoff comes across with the
    # mode because it is per material (0.200..0.545 here) -- collapsing it
    # to 0.5 is the thing 43c4708 explicitly refused.
    if alpha_mode_ovl is not None:
        _n = min(len(mat_names), len(alpha_mode_ovl))
        _dm = _dc = 0
        for i in range(_n):
            _m = int(alpha_mode_ovl[i])
            if _m != int(am[i]):
                _dm += 1
            am[i] = _m
            if alpha_cut_ovl is not None and i < len(alpha_cut_ovl):
                _c = float(alpha_cut_ovl[i])
                if abs(_c - float(ac[i])) > 1e-6:
                    _dc += 1
                ac[i] = _c
        log(f"  alpha: ADOPTED alpha_mode_ovl from the side table for "
            f"{_n} materials; it disagreed with the derivation on {_dm} "
            f"modes and {_dc} cutoffs, and the table wins on both")
    n_mask = int((am == 1).sum())
    n_blend = int((am == 2).sum())
    fmi = face_mat.numpy()
    pack["face_alpha_mode"] = torch.from_numpy(am[fmi]).to(torch.int64)
    pack["face_alpha_cutoff"] = torch.from_numpy(ac[fmi])
    log(f"  alpha: {n_mask} masked + {n_blend} blended of {len(mat_names)} "
        f"materials -> per-face modes")
    scaf = build_mat_scaffold(len(mat_names), TEX)
    # What stopped being a sentinel on THIS run. Appended to mat_measured
    # so a consumer can gate on the specific key it needs rather than on
    # the global mat_scaffold boolean, which stays True until the whole
    # 35-column block plus its aux pages are real.
    mat_real = []
    # RAW texels by AUX INDEX, for the reference-layout page arrays the
    # renderer contracts for (mat_normal_pages, mat_ao_pages). Filled by
    # the slot blocks below and emitted ONCE at the end, because each
    # block appends to `aux` and a page array frozen mid-way is shorter
    # than the indices a later block hands out. The arrays are the same
    # length as the final aux and share its indices, so mat_nrm1 /
    # mat_nrm2 / mat_ao address both without a second index table.
    RAW_PAGES = {"normal": {}, "ao": {}}
    # Real normal maps, if the side table established the encoding for the
    # slot. The table's nrm_slot/nrm_enc columns are CONSUMED, never
    # re-derived: g_tNormal appears under 15 families and the same bytes
    # decode to different normals, so the slot has to say which. A row
    # whose encoding is unestablished keeps the flat sentinel and
    # mat_has_nrm stays False for it -- a tilted normal from a guessed
    # encoding is the one error no frame reveals.
    # BOTH LAYERS, and both layouts. Three things were wrong here and none
    # of them had ever shown up in a frame, because no shipped pack has
    # run this block -- every de_inferno pack on the cluster has aux of 4
    # layers and mat_measured == [], so the whole block is code that has
    # never executed in anger.
    #
    #   1. `nrm1[j] = 3 + len(extra)` is OFF BY ONE. The scaffold has FOUR
    #      sentinel layers (0..3), so the first appended page is index 4;
    #      at 3 the first material with a real normal would have been
    #      handed AUX_HALF -- flat grey read as a normal -- and every
    #      other material shifted one page off its own texture. Written as
    #      the array's own length so it cannot drift from the sentinels
    #      again.
    #   2. LAYER 2 was `mat_nrm2 = mat_nrm1.clone()`: on a blend material
    #      the second layer got the FIRST layer's normal map. The side
    #      table has carried nrm_slot2 / nrm_enc2 all along.
    #   3. The DECODED page is not what the reference reads. gpu_render
    #      declares a contract for `mat_normal_pages` -- "(L,E,E,C>=3), .x
    #      / .y the octahedral pair, .z the ROUGHNESS, indexed by the
    #      EXISTING mat_nrm1 / mat_nrm2" -- and prints a GAP every frame
    #      because no packer ever wrote the key. The aux array gets the
    #      decoded normals it has always wanted (RGB a direction, alpha
    #      roughness) and mat_normal_pages gets the RAW texels at the same
    #      indices, so the two paths cannot disagree about which page
    #      belongs to which material.
    #
    # Roughness rides in .z of the raw page, which is why the raw texels
    # have to be kept: decode_normal() returns a direction and the .z
    # channel is consumed by it. `HemiOctIsoRoughness_RG_B` is the
    # shipping mip-processing command on g_tNormal1's channel processor --
    # the encoding named by the shader's own tables, and the same answer
    # bc_decode's table already carries for this family.
    n_nrm, n_nrm2, nrm_why = 0, 0, {}
    if tex_dir and nrm_slot is not None:
        base = int(scaf["aux"].shape[0])
        nrm1 = scaf["mat_nrm1"].clone()
        nrm2 = scaf["mat_nrm2"].clone()
        has = scaf["mat_has_nrm"].clone()
        basis = [""] * len(mat_names)
        extra, raw, seen = [], [], {}

        def _nrm_page(nm, slots, encs, lay):
            """(aux index, raw texels) for one material's layer, or None.

            Deduplicated by (path, encoding): de_inferno's 327 materials
            share far fewer distinct normal textures than they have slots,
            and a page decoded twice is two copies of the same 1024x1024x4
            in the pack.
            """
            slot = slots[j] if (slots is not None and j < len(slots)) else ""
            enc = (nrm_enc_names[int(encs[j])]
                   if encs is not None and j < len(encs) else "")
            # Two DIFFERENT conditions, and lumping them said "192
            # unestablished encodings" fleet-wide when most of those rows
            # declare no normal map at all. One is work owed by whoever
            # reads the shaders; the other is a material that simply has
            # no normal, and nothing is owed. Same sin as n_bad.
            if not slot:
                nrm_why[f"L{lay}: material declares no normal slot"] = \
                    nrm_why.get(f"L{lay}: material declares no normal slot",
                                0) + 1
                return None
            if not enc:
                nrm_why[f"L{lay}: slot declared, encoding NOT established"] = \
                    nrm_why.get(
                        f"L{lay}: slot declared, encoding NOT established",
                        0) + 1
                return None
            rel = (rows_tp.get(nm) or {}).get(slot)
            if not rel:
                nrm_why[f"L{lay}: slot declared, no texture bound"] = \
                    nrm_why.get(f"L{lay}: slot declared, no texture bound",
                                0) + 1
                return None
            key = (rel, enc)
            if key in seen:
                return seen[key], slot
            path = os.path.join(tex_dir, rel + "_c")
            img, reason = decode_normal(path, enc)
            if img is None:
                nrm_why[f"L{lay}: {reason}"] = \
                    nrm_why.get(f"L{lay}: {reason}", 0) + 1
                return None
            texels, why2 = decode_albedo(path)
            if texels is None:
                # The direction decoded and the texels did not, which
                # cannot happen through one container -- reported rather
                # than papered over with a flat page.
                nrm_why[f"L{lay}: decoded as normal but NOT as texels "
                        f"({why2})"] = 1 + nrm_why.get(
                            f"L{lay}: decoded as normal but NOT as texels "
                            f"({why2})", 0)
                return None
            idx = base + len(extra)
            extra.append(img)
            raw.append(texels)
            seen[key] = idx
            return idx, slot

        for j, nm in enumerate(mat_names):
            got1 = _nrm_page(nm, nrm_slot, nrm_enc, 1)
            if got1:
                nrm1[j], slot = got1
                has[j] = True
                # "shader" (read off the decompiled pixel shader) and
                # "falsifier" (DXT5nm ruled out from the data) are the SAME
                # answer at DIFFERENT confidence. Collapsing them would let
                # a weaker claim inherit a stronger one's authority, so the
                # basis is carried per material rather than summarised away.
                fam = (rows_fam.get(nm) or "")
                basis[j] = (nrm_basis or {}).get(f"{fam}::{slot}", "unknown")
                n_nrm += 1
            got2 = _nrm_page(nm, nrm_slot2, nrm_enc2, 2)
            if got2:
                nrm2[j] = got2[0]
                n_nrm2 += 1
            elif got1:
                # A single-layer material has no layer-2 normal to be
                # wrong about; pointing it at layer 1 is what the blend
                # weight already does with the albedo. A material that
                # DECLARES a layer-2 normal and failed to decode keeps the
                # flat sentinel instead, so a failure never masquerades as
                # a page.
                nrm2[j] = nrm1[j]
        if extra:
            scaf["aux"] = torch.from_numpy(
                np.concatenate([scaf["aux"].numpy(), np.stack(extra)]))
            # The RAW texels are held by aux index and emitted once, after
            # every block that appends to aux has run. Building the page
            # array here instead would freeze it at this block's length,
            # and the overlay block below appends three more layers -- so
            # a later index would address a row the array does not have.
            RAW_PAGES["normal"].update(
                {base + i: t for i, t in enumerate(raw)})
            scaf["mat_nrm1"] = nrm1
            scaf["mat_nrm2"] = nrm2
            scaf["mat_has_nrm"] = has
            scaf["mat_nrm_basis"] = basis
            mat_real.append("mat_nrm1/mat_nrm2 + mat_has_nrm + "
                            "mat_normal_pages (.xy octahedral, .z roughness)")
        log(f"  normals: L1 {n_nrm}/{len(mat_names)} materials, L2 "
            f"{n_nrm2}/{len(mat_names)}, with a REAL normal map "
            f"({len(extra)} distinct pages, {scaf['aux'].shape[0]} aux "
            f"layers); {len(RAW_PAGES['normal'])} raw pages held for "
            f"mat_normal_pages, which is emitted after every slot block")
        import collections as _c
        for b, n in _c.Counter(x for x in basis if x).most_common():
            log(f"    encoding established by {b}: {n}")

    # ---- HEIGHT PAGES, and the layer-2 blend parameters that ride them --
    #
    # gpu_render.py:19645 gates the height blend on
    # `_ext(e,"has_h2") * face_has2`, and samples MAT_H2 at :19631. So
    # has_h2 and the aux page are ONE FACT: setting the flag while mat_h2
    # still points at the AUX_WHITE sentinel makes the blend run against a
    # flat white height, which is a plausible picture carrying no measured
    # data -- the exact failure the scaffold header warns about. The flag
    # is therefore set ONLY where a page actually decoded, never from the
    # material's declaration alone.
    #
    # The scalars are different in kind: g_flHeightMapScale/ZeroPoint and
    # g_flBlendSoftness2 are READ from the material, not fitted, and they
    # are meaningless without the page, so they are written under the same
    # condition.
    # OFF BY DEFAULT BECAUSE IT WAS MEASURED AND IT LOST. Turning has_h2 on
    # with real decoded pages, pose 0 of de_inferno, lit, canonical
    # render_gt.sh, against GT:
    #
    #   arm                    ratio      MSE      KL       ncc   ncc-BOT
    #   vpaint, no height      1.155  0.04658  0.2587  -0.0449   -0.1614
    #   vpaint + REAL height   1.299  0.10298  3.8353  -0.2067   -0.4088
    #
    # Every metric worse, KL by 14x, and the ground -- the region this was
    # supposed to fix -- worst of all. So the pages decode and the flags
    # and scalars are read correctly from the material, and something about
    # how they are INTERPRETED is wrong. The two leading suspects, neither
    # checked yet: decode_albedo() is an albedo path and a height map is
    # single-channel linear data, so the channel semantics may not survive
    # it; and gpu_render.py:593 says csgo_environment_blend carries its
    # TINT MASK in g_tHeight's green channel, so the page is not a plain
    # height field for the family that matters most here.
    #
    # This is a DIAGNOSTIC gate, not a scope cut: the work is here, the
    # refutation is here with its numbers, and the default is the arm that
    # measured better. It must not be turned on again without a decode
    # that has been checked against what the shader actually reads.
    n_h, h_why = 0, {}
    if height_pages and tex_dir and rows_ip:
        _ext_t = scaf["mat_ext"]
        _base = int(scaf["aux"].shape[0])
        _h1 = scaf["mat_h1"].clone()
        _h2 = scaf["mat_h2"].clone()
        _hx = []
        for j, nm in enumerate(mat_names):
            tp = rows_tp.get(nm) or {}
            fp = rows_fp.get(nm) or {}
            got = {}
            for lay, slot in ((1, "g_tHeight1"), (2, "g_tHeight2")):
                rel = tp.get(slot)
                if not rel:
                    # Not a defect: most materials are single-layer and owe
                    # no height map at all. Counted separately from a slot
                    # that is declared and fails, for the same reason the
                    # normal block splits those two.
                    h_why[f"declares no {slot}"] = \
                        h_why.get(f"declares no {slot}", 0) + 1
                    continue
                img, reason = decode_albedo(os.path.join(tex_dir, rel + "_c"))
                if img is None:
                    h_why[f"{slot} bound, {reason}"] = \
                        h_why.get(f"{slot} bound, {reason}", 0) + 1
                    continue
                got[lay] = _base + len(_hx)
                _hx.append(img)
            if not got:
                continue
            for lay, idx in got.items():
                (_h1 if lay == 1 else _h2)[j] = idx
                _ext_t[j, EXT_KEYS.index(f"has_h{lay}")] = 1.0
                # Scale/zero are the material's own numbers. Their neutral
                # is 1.0/0.0 and EXT_NEUTRAL already holds h_scale at 1.0,
                # so a material that declares a page but no scale keeps the
                # neutral rather than picking up a zero.
                _s = fp.get(f"g_flHeightMapScale{lay}")
                if _s is not None:
                    _ext_t[j, EXT_KEYS.index(f"h_scale{lay}")] = float(_s)
                _z = fp.get(f"g_flHeightMapZeroPoint{lay}")
                if _z is not None:
                    _ext_t[j, EXT_KEYS.index(f"h_zero{lay}")] = float(_z)
            n_h += 1
        if _hx:
            scaf["aux"] = torch.from_numpy(
                np.concatenate([scaf["aux"].numpy(), np.stack(_hx)]))
            scaf["mat_h1"], scaf["mat_h2"] = _h1, _h2
            scaf["mat_ext"] = _ext_t
            mat_real.append("has_h1/has_h2 + h_scale/h_zero + mat_h1/mat_h2")
        log(f"  height: {n_h}/{len(mat_names)} materials with a REAL height "
            f"page ({len(_hx)} pages, {scaf['aux'].shape[0]} aux layers); "
            f"has_h2 set on {int((_ext_t[:, EXT_KEYS.index('has_h2')] > 0).sum())}")
        for r, c in sorted(h_why.items(), key=lambda kv: -kv[1])[:6]:
            log(f"    {r}: {c}")
        for r, c in sorted(nrm_why.items(), key=lambda kv: -kv[1]):
            log(f"    no normal because {r}: {c}")

    # ---- SHARED COLOUR OVERLAY PAGE, and the five numbers that drive it -
    #
    # g_tSharedColorOverlay was declared by 11 de_inferno materials and
    # carried by none: mat_ovl pointed at AUX_HALF on all 327, which the
    # composite treats as the exact identity, so the term was wired end to
    # end and multiplying by 1.0 everywhere.
    #
    # THE PAGE GOES IN AS LINEAR. gpu_render.sample_aux() does no sRGB
    # decode -- its docstring says the aux array must not be decoded,
    # because a normal is a direction and a height is not a colour -- but
    # this page IS a colour: its channel processor carries
    # m_outputColorSpace = 1, the same value g_tColor1 carries and the
    # opposite of g_tNormal1 and g_tHeight1. So the decode happens here,
    # once, at pack time, rather than at a sampler that is right to refuse
    # it. Not an assumption -- the three pages measure it. Decoded as
    # sRGB their linear means are 0.5012, 0.4877 and 0.5858, i.e. two of
    # three sit within 0.025 of the exact identity 0.5; read as raw bytes
    # they would sit at s = +0.43, +0.45, +0.54, a strong brightening
    # everywhere, which is not what a grunge page is for.
    #
    # The cost of doing it here is 8-bit LINEAR quantisation, coarser in
    # the darks than 8-bit sRGB would be: the step is 1/255 in linear, so
    # 2/255 = 0.008 in s. Stated because it is a real loss, not because
    # it is large.
    #
    # The five scalars are the material's own numbers, read from the vmat:
    # two contrasts and the two enum-derived layer selectors. None fitted.
    n_ovl, ovl_why = 0, {}
    if overlay_pages and tex_dir:
        _ext_t = scaf["mat_ext"]
        _ovl = scaf["mat_ovl"].clone()
        _ox, _seen = [], {}
        _K = EXT_KEYS.index
        for j, nm in enumerate(mat_names):
            rel = (rows_tp.get(nm) or {}).get("g_tSharedColorOverlay")
            if not rel:
                # Not a defect: 316 of 327 declare no overlay and owe no
                # page. Counted apart from a slot that is declared and
                # fails, as the normal and height blocks above do.
                ovl_why["declares no g_tSharedColorOverlay"] = \
                    ovl_why.get("declares no g_tSharedColorOverlay", 0) + 1
                continue
            if rel in _seen:
                idx = _seen[rel]
            else:
                img, reason = decode_albedo(os.path.join(tex_dir, rel + "_c"))
                if img is None:
                    ovl_why[f"overlay bound, {reason}"] = \
                        ovl_why.get(f"overlay bound, {reason}", 0) + 1
                    continue
                lin = np.empty_like(img)
                c = img[..., :3].astype(np.float32) / 255.0
                lin[..., :3] = np.rint(
                    np.where(c <= 0.04045, c / 12.92,
                             ((c + 0.055) / 1.055) ** 2.4) * 255.0
                ).astype(np.uint8)
                lin[..., 3] = img[..., 3]
                idx = int(scaf["aux"].shape[0]) + len(_ox)
                _ox.append(lin)
                _seen[rel] = idx
            _ovl[j] = idx
            ip = rows_ip.get(nm) or {}
            fp = rows_fp.get(nm) or {}
            # The declared defaults, from the shader's own variable table:
            # mode and tintMask default to 0 ("All Layers" / "Mask All").
            mode = int(ip.get("g_nColorOverlayMode", 0) or 0)
            tmk = int(ip.get("g_nColorOverlayTintMask", 0) or 0)
            _ext_t[j, _K("f_overlay")] = 1.0
            _ext_t[j, _K("ovl_l1")] = float(mode in (0, 1))
            _ext_t[j, _K("ovl_l2")] = float(mode in (0, 2))
            _ext_t[j, _K("ovl_mask1")] = float(tmk in (0, 1))
            _ext_t[j, _K("ovl_mask2")] = float(tmk in (0, 2))
            _b = fp.get("g_flOverlayBrightnessContrast")
            _d = fp.get("g_flOverlayDarknessContrast")
            if _b is not None:
                _ext_t[j, _K("ovl_bright_contrast")] = float(_b)
            if _d is not None:
                _ext_t[j, _K("ovl_dark_contrast")] = float(_d)
            n_ovl += 1
        if _ox:
            scaf["aux"] = torch.from_numpy(
                np.concatenate([scaf["aux"].numpy(), np.stack(_ox)]))
            scaf["mat_ovl"] = _ovl
            scaf["mat_ext"] = _ext_t
            mat_real.append("f_overlay + ovl_l1/l2 + ovl_mask1/2 + "
                            "ovl_bright/dark_contrast + mat_ovl")
        log(f"  overlay: {n_ovl}/{len(mat_names)} materials with a REAL "
            f"g_tSharedColorOverlay page ({len(_ox)} distinct pages, "
            f"{scaf['aux'].shape[0]} aux layers)")
        if n_ovl:
            _bc = _ext_t[:, _K("ovl_bright_contrast")]
            _dc = _ext_t[:, _K("ovl_dark_contrast")]
            _on = _ext_t[:, _K("f_overlay")] > 0.5
            log(f"    B in [{float(_bc[_on].min()):.3f}, "
                f"{float(_bc[_on].max()):.3f}], D in "
                f"[{float(_dc[_on].min()):.3f}, {float(_dc[_on].max()):.3f}]"
                f"; ovl_l1 on {int((_ext_t[:, _K('ovl_l1')] > 0.5).sum())}, "
                f"ovl_l2 on {int((_ext_t[:, _K('ovl_l2')] > 0.5).sum())}, "
                f"ovl_mask1 on {int((_ext_t[:, _K('ovl_mask1')] > 0.5).sum())}")
        for r, c in sorted(ovl_why.items(), key=lambda kv: -kv[1])[:6]:
            log(f"    {r}: {c}")

    # ---- AMBIENT OCCLUSION PAGES ---------------------------------------
    #
    # 103 de_inferno materials declare g_tAmbientOcclusion and the pack
    # carried none; mat_ao pointed at AUX_WHITE on all 327, i.e. 1.0,
    # i.e. unoccluded everywhere. gpu_render has had the consumer for it
    # all along -- `mat_ao_pages` .x, multiplied into the INDIRECT term
    # inside the _gt_lm_tint argument (glsl:1230 q0, :1588-1592 q1) --
    # and prints a GAP for the missing key every frame.
    #
    # LAYER 2 IS NOT CARRIED, and that is a stated partial rather than a
    # silent one: g_tLayer2AmbientOcclusion is declared 127 times across
    # the 7 maps (once on de_inferno) and there is no second AO index in
    # the renderer for it to reach. Carrying it would need MAT_AO2 and a
    # blend on the albedo's weight; the slot is counted in the log below
    # so it does not read as absent.
    #
    # Occlusion is linear data -- its channel processor carries
    # m_outputColorSpace = 0, the same as g_tNormal1 and g_tHeight1 and
    # the opposite of g_tColor1 -- so the texels go in raw, with no
    # decode of any kind.
    n_ao, ao_why = 0, {}
    if ao_pages and tex_dir:
        _ao = scaf["mat_ao"].clone()
        _base_ao = int(scaf["aux"].shape[0])
        _ax, _seen_ao = [], {}
        for j, nm in enumerate(mat_names):
            tp = rows_tp.get(nm) or {}
            rel = tp.get("g_tAmbientOcclusion") or tp.get(
                "g_tLayer1AmbientOcclusion")
            if not rel:
                ao_why["declares no AO slot"] = \
                    ao_why.get("declares no AO slot", 0) + 1
                continue
            if rel in _seen_ao:
                _ao[j] = _seen_ao[rel]
                n_ao += 1
                continue
            img, reason = decode_albedo(os.path.join(tex_dir, rel + "_c"))
            if img is None:
                ao_why[f"AO bound, {reason}"] = \
                    ao_why.get(f"AO bound, {reason}", 0) + 1
                continue
            idx = _base_ao + len(_ax)
            _ax.append(img)
            _seen_ao[rel] = idx
            _ao[j] = idx
            n_ao += 1
        if _ax:
            scaf["aux"] = torch.from_numpy(
                np.concatenate([scaf["aux"].numpy(), np.stack(_ax)]))
            RAW_PAGES["ao"].update(
                {_base_ao + i: t for i, t in enumerate(_ax)})
            scaf["mat_ao"] = _ao
            mat_real.append("mat_ao + mat_ao_pages (.x occlusion)")
        _n_l2 = sum(1 for nm in mat_names
                    if (rows_tp.get(nm) or {}).get("g_tLayer2AmbientOcclusion"))
        log(f"  AO: {n_ao}/{len(mat_names)} materials with a REAL "
            f"occlusion page ({len(_ax)} distinct pages, "
            f"{scaf['aux'].shape[0]} aux layers); "
            f"g_tLayer2AmbientOcclusion declared by {_n_l2} and NOT "
            f"carried -- the renderer has no second AO index")
        for r, c in sorted(ao_why.items(), key=lambda kv: -kv[1])[:6]:
            log(f"    {r}: {c}")

    # ---- EMIT THE REFERENCE-LAYOUT PAGE ARRAYS --------------------------
    # One pass, after every block that appends to aux. Rows default to the
    # aux row at the same index, so an index that never got a raw page --
    # every sentinel, and every material without this slot -- reads the
    # value it already read. AUX_WHITE is 255 = occlusion 1.0 and
    # AUX_FLAT_NRM is (128,128,255) = a = 0 exactly with roughness 1, so
    # the no-page rows are the right no-op in the reference layout too.
    for _kind, _key in (("normal", "mat_normal_pages"), ("ao", "mat_ao_pages")):
        if RAW_PAGES[_kind]:
            _p = scaf["aux"].numpy().copy()
            for _i, _t in RAW_PAGES[_kind].items():
                _p[_i] = _t
            scaf[_key] = torch.from_numpy(_p)
            log(f"  {_key}: {tuple(scaf[_key].shape)}, "
                f"{len(RAW_PAGES[_kind])} rows are decoded pages and the "
                f"rest are the aux row at that index")
    pack.update(scaf)
    # Machine-checkable, not just a log line: a printed warning does not
    # survive into whoever loads this pack three steps later.
    # Still a scaffold: normals may now be measured, but the other 18
    # mat_* keys are not. Flipping this to False because ONE key became
    # real would be the manufactured-completeness move. `mat_measured`
    # names exactly what is real, so a consumer can gate on the key it
    # actually needs instead of on a single global boolean.
    # THE MAP THIS PACK IS OF, so a sidecar resolver never has to guess it
    # from the file name. Three sidecar families (.irradiance.npy,
    # .light_environment.json, .sky_cube.npz) auto-resolve by splitting the
    # world path's extension, and every variant pack this lane built --
    # de_inferno_ovl, _nrm, _ao -- missed all three, most damagingly the
    # sky, which every arm then rendered without behind a GAP line that was
    # printed and read past. Symlinks fix a day; the name in the pack fixes
    # the class.
    pack["map"] = os.path.basename(ash_dir.rstrip("/")).replace(".ash", "")
    pack["mat_scaffold"] = True
    pack["mat_measured"] = ((["mat_nrm1", "mat_nrm2", "mat_has_nrm"]
                             if n_nrm else []) + mat_real)
    # WHICH ANSWER IS THIS. Two generations have now differed by a
    # deliberate content decision -- effects dropped versus kept -- and
    # their frames are indistinguishable unless you already know to look
    # for steam. mtime cannot tell them apart either: one 5-minute sweep
    # and two sweeps 5 minutes apart have the same spread. So the pack
    # carries a signature derived from the CODE and the OPTIONS that
    # produced it, and a downstream reader can compare two packs without
    # asking anyone which run they came from.
    pack["effects_kept"] = not drop_effects
    try:
        _src = open(os.path.abspath(__file__), "rb").read()
        _opt = repr((winding, keep_tools, drop_effects, bool(tex_dir),
                     bool(mat_paths_from), TEX)).encode()
        pack["pack_generation"] = hashlib.sha256(_src + _opt).hexdigest()[:16]
    except Exception as _e:                                   # noqa: BLE001
        pack["pack_generation"] = f"UNKNOWN ({type(_e).__name__})"
        log(f"  pack_generation FAILED ({type(_e).__name__}: {_e})")
    torch.save(pack, out_path)
    # THIS BANNER USED TO SAY "every feature flag OFF, every scalar
    # neutral, N sentinel aux layers" UNCONDITIONALLY, which stopped being
    # true the moment a slot block above it wrote a real page -- it
    # printed "7 sentinel aux layers" over a pack whose last 3 were
    # decoded overlay pages, one line under the log line saying so. A
    # banner that cannot be wrong is the failure this project keeps
    # finding; it is now derived from mat_real.
    _nsent = 4
    _real = int((scaf["mat_ext"] != 0).any(dim=1).sum())
    if mat_real:
        log(f"  mat_* PARTLY MEASURED: {len(scaf) - 1} keys + "
            f"{len(EXT_KEYS)} ext columns; {scaf['aux'].shape[0]} aux "
            f"layers of which {_nsent} are sentinels; {_real} of "
            f"{len(mat_names)} materials have a non-zero ext row. REAL: "
            f"{'; '.join(mat_real)}. EVERYTHING ELSE IS STILL SCAFFOLD -- "
            f"gate on pack['mat_measured'], which names exactly these, "
            f"not on pack['mat_scaffold'], which stays True.")
    else:
        log(f"  mat_* SCAFFOLD: {len(scaf) - 1} keys + {len(EXT_KEYS)} ext "
            f"columns, every feature flag OFF, every scalar neutral, "
            f"{scaf['aux'].shape[0]} sentinel aux layers, mat_has_nrm False "
            f"on all {len(mat_names)} materials. The PBR path will RUN and "
            f"the picture will be plausible; it carries NO measured "
            f"material data. Gate on pack['mat_scaffold'] before trusting "
            f"any number from it.")

    lo = vertices.min(0).values.tolist()
    hi = vertices.max(0).values.tolist()
    log(f"  wrote {out_path}: {vertices.shape[0]} verts, {faces.shape[0]} tris,"
        f" {len(mat_names)} materials, bounds "
        f"[{lo[0]:.1f},{lo[1]:.1f},{lo[2]:.1f}].."
        f"[{hi[0]:.1f},{hi[1]:.1f},{hi[2]:.1f}] m")
    if n_uvbad:
        log(f"  NON-FINITE UVs: {n_uvbad} vertices zeroed. Left in place "
            f"they index textures[] out of bounds and abort the render "
            f"with a device-side assert, from a pack whose own index "
            f"arrays are all in range.")
    log(f"  tinted draw calls: {n_tinted} carry a non-identity tint "
        f"-> per-vertex vcolor")
    log(f"  draw calls kept {n_dc}, tools dropped {n_tool}, "
        f"no-POSITION {n_missing_pos}, index-out-of-range {n_bad}, "
        f"missing-model {n_nomodel}, non-triangle {n_nontri}, "
        f"effects dropped {n_effect}, "
        f"UV-less {missing_uv}, {time.time() - t0:.1f}s")
    for d in defaulted:
        log(f"  DEFAULTED: {d}")

    if camera_out:
        write_camera(vertices, fn, area, camera_out, frames)
    return {"verts": int(vertices.shape[0]), "tris": int(faces.shape[0]),
            "materials": len(mat_names), "flipped_winding": flipped,
            "up_fraction": frac}


def write_camera(vertices, fn, area, path, frames):
    """A camera json in SOURCE units that stands on the biggest floor.

    Schema is {"ticks":[{tick,x,y,eye_z,yaw_degrees,pitch_degrees}]}; x is
    source X, y is source Y, eye_z is source Z, all inches. ~/poseset48.json
    is a DIFFERENT schema and gpu_render.py will not read it.
    """
    # An ELEVATED ORBIT looking in at the map, not an eye-height pose at the
    # centroid. The eye-height version put cs_italy's camera inside a
    # building -- the map renders, and the frame is a wall. Standing a
    # camera on the floor needs clearance the pack does not carry, whereas
    # a viewpoint above the geometry looking down cannot be occluded by it
    # and so is non-degenerate for every map without a per-map check.
    lo = vertices.min(0).values
    hi = vertices.max(0).values
    c = (lo + hi) * 0.5
    span = float(torch.max(hi[0] - lo[0], hi[2] - lo[2]))
    radius = max(span * 0.55, 4.0)
    height = float(hi[1]) + max(span * 0.18, 3.0)
    ticks = []
    for i in range(frames):
        th = 2.0 * math.pi * i / max(frames, 1)
        ex = float(c[0]) + radius * math.sin(th)
        ez = float(c[2]) + radius * math.cos(th)
        dx, dy, dz = float(c[0]) - ex, float(c[1]) - height, float(c[2]) - ez
        n = max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-9)
        # yaw 0 looks down world +Z; +pitch is DOWN, since f_y = -sin(pitch).
        ticks.append({
            "tick": i,
            "x": ez / S,                      # world Z <- source X
            "y": ex / S,                      # world X <- source Y
            "eye_z": height / S,              # world Y <- source Z
            "yaw_degrees": math.degrees(math.atan2(dx, dz)),
            "pitch_degrees": math.degrees(math.asin(max(-1.0, min(1.0,
                                                                 -dy / n)))),
        })
    json.dump({"schema": "iji/gt-camera/v1", "ticks": ticks},
              open(path, "w"), indent=1)
    log(f"  wrote {path}: {len(ticks)} ticks, eye_z={ticks[0]['eye_z']:.0f} src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ash")
    ap.add_argument("--out")
    ap.add_argument("--ash-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--winding", choices=["auto", "keep", "flip"],
                    default="auto")
    ap.add_argument("--drop-effects", action="store_true",
                    help="drop csgo_effects.vfx particle cards (steam/smoke). "
                    "They are KEPT by default: F_ADDITIVE_BLEND is fully "
                    "implemented in the renderer and composites them "
                    "dst+src*a under --additive-blend. Without that flag they "
                    "draw opaque, which is a renderer axis being off and not "
                    "a defect in this pack -- so the pack carries the map and "
                    "the renderer decides how to shade it.")
    ap.add_argument("--keep-tools", action="store_true",
                    help="keep materials/tools/* (blocklight/clip volumes)")
    ap.add_argument("--camera-out")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--tex-edge", type=int, default=None,
                    help="texture edge in the pack. The default 64 was "
                    "chosen when textures were synthesised hash-hue "
                    "checkers, which had no detail to lose; it now "
                    "resamples real BC7 assets. Sweep it against a metric "
                    "before changing it -- a constant correct for the data "
                    "it was written against is not automatically correct "
                    "for the data that replaced it.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--tex-dir", help="the `textures/` dir of an ASH "
                    "extraction made WITHOUT --no-textures; albedo texels "
                    "are decoded from it, falling back to the checker per "
                    "material with the reason counted")
    ap.add_argument("--mat-paths", help="a fam_side_<map>.pt whose mat_paths "
                    "ordering this pack should ADOPT for face_matid, rather "
                    "than re-deriving a sorted order and hoping the two sets "
                    "match. Refused, loudly, if this pack is not a subset.")
    ap.add_argument("--tex-dir-root", help="dir of <map>.ash extractions "
                    "carrying textures, for --ash-dir sweeps")
    ap.add_argument("--mat-paths-dir", help="dir of fam_side_<map>.pt, for "
                    "--ash-dir sweeps; missing files fall back per map")
    ap.add_argument("--ao-pages", action="store_true",
                    help="carry g_tAmbientOcclusion (or "
                         "g_tLayer1AmbientOcclusion) into the aux array "
                         "and write mat_ao_pages. Needs --tex-dir. Its "
                         "own flag for the same reason --overlay-pages "
                         "is: one slot family per arm or neither is "
                         "priced.")
    ap.add_argument("--overlay-pages", action="store_true",
                    help="carry g_tSharedColorOverlay into the aux array "
                         "and write the five scalars that drive it. Needs "
                         "--tex-dir. Its own flag, not folded into "
                         "--tex-dir, so a pack can differ from the "
                         "previous one in the overlay ALONE -- --tex-dir "
                         "on its own is also the normal block's gate, and "
                         "an arm that moves two slots prices neither.")
    ap.add_argument("--height-pages", action="store_true",
                    help="DIAGNOSTIC, off by default because it MEASURED "
                         "WORSE. Decode g_tHeight1/2 into aux and set "
                         "has_h1/has_h2 + h_scale/h_zero from the material. "
                         "On pose 0 of de_inferno, lit, this took ncc "
                         "-0.0449 -> -0.2067, ground -0.1614 -> -0.4088, "
                         "MSE 0.04658 -> 0.10298 and KL 0.2587 -> 3.8353. "
                         "The pages decode and the scalars are read "
                         "correctly; the interpretation is wrong. See the "
                         "block in convert() for the two open suspects.")
    a = ap.parse_args()

    if a.tex_edge:
        globals()["TEX"] = int(a.tex_edge)
    if a.ash:
        # --mat-paths-dir used to be read ONLY by the --ash-dir sweep, so a
        # single-file run accepted the flag and silently ignored it. The
        # cost is not cosmetic: alb_slot2 comes off the adopted side table,
        # so ignoring it drops every layer-2 albedo page -- this produced a
        # 327-texture pack where the sweep produces 399, with face_has2 at
        # 0.0% instead of 31.3%, and nothing said so. The tell was the one
        # I had already written down: a real read logs "adopted" or
        # "REFUSED", and that run logged NEITHER.
        mpf = a.mat_paths
        if a.mat_paths_dir:
            _n = os.path.basename(a.ash)
            _n = _n[:-4] if _n.endswith(".ash") else _n
            mpf = _resolve_side(a.mat_paths_dir, _n)
        convert(a.ash, a.out, a.winding, a.keep_tools, a.drop_effects,
                a.camera_out,
                a.frames, mpf, a.tex_dir, a.height_pages,
                a.overlay_pages, a.ao_pages)
        return 0

    if not a.ash_dir:
        ap.error("need --ash or --ash-dir")
    # THE SET IS THE ARTIFACT, NOT THE FILE. Per-map atomic writes make a
    # partial FILE impossible and a partial SET routine: for the two
    # minutes a sweep runs, ~/worlds/ holds a mix of generations and every
    # reader gets a different answer with no evidence it happened. Two
    # agents reported 33+10 and 43/43 of the same directory within
    # minutes; both were correct and nothing in either artifact could
    # reconcile them.
    #
    # A transient straddle is WORSE than a permanent one -- a permanently
    # mixed directory is at least stable enough for two readers to agree
    # they disagree.
    #
    # So the sweep builds into a sibling directory and swaps it in at the
    # end. A reader sees the old set or the new set, never a blend.
    final_dir = a.out_dir
    a.out_dir = final_dir.rstrip("/") + ".incoming"
    if os.path.exists(a.out_dir):
        shutil.rmtree(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    names = sorted(d[:-4] for d in os.listdir(a.ash_dir) if d.endswith(".ash"))
    ok, bad = [], []
    for i, n in enumerate(names):
        out = os.path.join(a.out_dir, n + ".pt")
        if a.skip_existing and os.path.exists(out):
            log(f"[{i+1}/{len(names)}] {n}: exists, skipped")
            ok.append(n)
            continue
        log(f"[{i+1}/{len(names)}] {n}")
        # Per-map isolation: one map's extraction defect must not take the
        # other 42 down with it.
        try:
            mpf = a.mat_paths
            if a.mat_paths_dir:
                mpf = _resolve_side(a.mat_paths_dir, n)
            convert(os.path.join(a.ash_dir, n + ".ash"), out, a.winding,
                    a.keep_tools, a.drop_effects,
                    os.path.join(a.out_dir, n + ".camera.json"), a.frames,
                    mpf,
                    (os.path.join(a.tex_dir_root, n + ".ash", "textures")
                     if a.tex_dir_root else a.tex_dir), a.height_pages,
                    a.overlay_pages, a.ao_pages)
            ok.append(n)
        except Exception as e:                            # noqa: BLE001
            import traceback
            traceback.print_exc()
            bad.append((n, f"{type(e).__name__}: {e}"))
    # COMPLETE IS NOT VALID. rename(2) guarantees no reader sees a blend;
    # it guarantees nothing about whether the new set is worth seeing, and
    # a sweep that writes 43 unloadable packs satisfies "complete" exactly.
    # `bad` only catches converters that RAISED -- a pack written truncated,
    # or with a key the renderer indexes out of range, lands in `ok`. So
    # the incoming set is re-opened and checked BEFORE the commit point,
    # because a check after the rename is a check after it is too late.
    if not bad:
        for n in ok:
            p = os.path.join(a.out_dir, n + ".pt")
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:                        # noqa: BLE001
                bad.append((n, f"UNLOADABLE {type(e).__name__}: {e}"))
                continue
            why = None
            need = ("vertices", "faces", "face_matid", "mat_paths",
                    "textures", "uvs")
            miss = [k for k in need if d.get(k) is None]
            nv = 0 if d.get("vertices") is None else len(d["vertices"])
            if miss:
                why = "missing " + ", ".join(miss)
            elif nv == 0 or len(d["faces"]) == 0:
                why = f"empty ({nv} verts, {len(d['faces'])} faces)"
            elif int(d["faces"].max()) >= nv:
                why = (f"face index {int(d['faces'].max())} >= {nv} verts "
                       "-- would fault the rasteriser")
            elif int(d["face_matid"].max()) >= len(d["mat_paths"]):
                why = (f"face_matid {int(d['face_matid'].max())} >= "
                       f"{len(d['mat_paths'])} materials")
            elif not bool(torch.isfinite(d["vertices"]).all()):
                why = "non-finite vertex positions"
            else:
                # Per-vertex streams must be index-aligned or every one of
                # them shades the wrong vertex, silently and plausibly.
                for k in ("uvs", "lmuv", "vcolor", "vnormal", "vtangent",
                          "vpaint"):
                    if d.get(k) is not None and len(d[k]) != nv:
                        why = f"{k} has {len(d[k]):,} rows against {nv:,} verts"
                        break
            del d
            if why:
                bad.append((n, "INVALID " + why))
        if bad:
            log(f"\n=== {len(bad)} pack(s) VERIFIED BAD after writing ===")
    # Swap only on a COMPLETE and VALID set. A partial or broken sweep
    # leaves the previous generation in place rather than publishing a
    # mixture -- a failed sweep should change nothing, not change some of it.
    if bad:
        log(f"\n=== {len(ok)}/{len(names)} converted -- NOT SWAPPED IN ===")
        log(f"    incomplete set left at {a.out_dir}; {final_dir} still "
            f"holds the previous generation, unmixed")
    else:
        _prev = final_dir.rstrip("/") + ".prev"
        if os.path.exists(_prev):
            shutil.rmtree(_prev)
        if os.path.exists(final_dir):
            os.rename(final_dir, _prev)
        os.rename(a.out_dir, final_dir)
        if os.path.exists(_prev):
            shutil.rmtree(_prev)
        log(f"    swapped {final_dir} atomically ({len(ok)} packs)")
    log(f"\n=== {len(ok)}/{len(names)} converted ===")
    for n, why in bad:
        log(f"FAILED {n}: {why}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
