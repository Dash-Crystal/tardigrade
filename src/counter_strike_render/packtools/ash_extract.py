"""Extract a CS2 map VPK straight into an ASH history. No glTF anywhere.

    ash_extract.py --map ~/cs2/game/csgo/maps/de_inferno.vpk --out de_inferno.ash

WHY THIS EXISTS. The chain into the renderer ran map -> Source2Viewer-CLI
-> glTF -> packer. That hop is lossy in a way that is measured rather
than suspected: de_inferno through `Source2Viewer-CLI` emits **6,107
material-channel exceptions** (`GetGltfChannels`), one per channel the
spec cannot express. glTF carries one base colour where
`csgo_environment_blend` shades from two full layer stacks plus a third,
one normal encoding where CS2 has at least three that decode differently
and are indistinguishable by inspection, no combo axes, and it discards
`m_shaderName` outright -- the single field that says which of 179
families shades a surface.

THE RULE THIS FILE IS BUILT ON. Store what the reference reads, keyed by
what the reference calls it. No renaming, no canonicalisation into
another material model, no closest-equivalent. Every material row keeps
its ENTIRE decoded `DATA` tree under `raw`, verbatim, in addition to the
indexed views -- because an unresolved parameter that is stored can be
resolved later and one that is dropped cannot.

WHAT IT READS, AND WITH WHOSE DECODER

    VPK directory      harness/gpu_render/vcs/vpk.py     (in-repo)
    KV3 v1-v5          harness/gpu_render/vcs/kv3v5.py   (in-repo)
    meshopt vertex/index   libmeshopt.so via ctypes      (see --meshopt)

`kv3v5.py` rather than `kv3.py` is not a preference. Every resource in
the shipped build is **KV3 version 5**, and `kv3.py` reads version 4;
v5 is a different container (two LZ4 buffers, 120-byte header, object
lengths moved out of the ints buffer, a new auxiliary-buffer array
type). A v4 reader pointed at a v5 file does not report the wrong
version -- it raises mid-match-copy and reads like a corrupt file.

WHAT IS EXTRACTED

  geometry    per model: every vertex buffer with its FULL input layout
              (semantic name, semantic index, DXGI format, offset, slot,
              shader semantic), decoded per-semantic arrays for every UV
              set, COLOR_0 and COLOR_1, BLENDINDICES/BLENDWEIGHT, plus
              the raw interleaved bytes so a format this table does not
              yet decode is still present. Per-instance transforms from
              the world node scene objects.
  materials   ONE ROW PER MATERIAL: `m_shaderName` verbatim, every F_*,
              every g_* under its own name, texture slot -> path with the
              slot's declared encoding, and the whole raw tree.
  textures    `.vtex_c` bytes copied in their SOURCE encoding, with the
              declared image format recorded per file.

WHAT IS NOT (stated here rather than discovered later): see the notes
written into the manifest by `--out`.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "vcs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kv3v5  # noqa: E402
from ash import AshWriter  # noqa: E402
from vpk import VPK  # noqa: E402

EXTRACTOR_VERSION = "ash_extract/1"

# ----------------------------------------------------------------------
# DXGI vertex formats, by the number the input layout actually carries.
#
# The layout field is `m_Format`, a DXGI_FORMAT. Decoding it by guessing
# from the semantic name is how a 4-byte compressed normal gets read as
# three floats; the number is authoritative and the name is not.
# (id: (numpy dtype, components, bytes, scale-to-unit))
# ----------------------------------------------------------------------
DXGI = {
    2:  ("<f4", 4, 16, None),      # R32G32B32A32_FLOAT
    3:  ("<u4", 4, 16, None),      # R32G32B32A32_UINT
    4:  ("<i4", 4, 16, None),      # R32G32B32A32_SINT
    6:  ("<f4", 3, 12, None),      # R32G32B32_FLOAT
    7:  ("<u4", 3, 12, None),      # R32G32B32_UINT
    8:  ("<i4", 3, 12, None),      # R32G32B32_SINT
    10: ("<f2", 4, 8,  None),      # R16G16B16A16_FLOAT
    11: ("<u2", 4, 8,  65535.0),   # R16G16B16A16_UNORM
    12: ("<u2", 4, 8,  None),      # R16G16B16A16_UINT
    13: ("<i2", 4, 8,  32767.0),   # R16G16B16A16_SNORM
    14: ("<i2", 4, 8,  None),      # R16G16B16A16_SINT
    16: ("<f4", 2, 8,  None),      # R32G32_FLOAT
    17: ("<u4", 2, 8,  None),      # R32G32_UINT
    18: ("<i4", 2, 8,  None),      # R32G32_SINT
    24: ("<u4", 1, 4,  None),      # R10G10B10A2_UNORM (kept packed)
    26: ("<u4", 1, 4,  None),      # R11G11B10_FLOAT   (kept packed)
    28: ("<u1", 4, 4,  255.0),     # R8G8B8A8_UNORM
    29: ("<u1", 4, 4,  255.0),     # R8G8B8A8_UNORM_SRGB
    30: ("<u1", 4, 4,  None),      # R8G8B8A8_UINT
    31: ("<i1", 4, 4,  127.0),     # R8G8B8A8_SNORM
    32: ("<i1", 4, 4,  None),      # R8G8B8A8_SINT
    34: ("<f2", 2, 4,  None),      # R16G16_FLOAT
    35: ("<u2", 2, 4,  65535.0),   # R16G16_UNORM
    36: ("<u2", 2, 4,  None),      # R16G16_UINT
    37: ("<i2", 2, 4,  32767.0),   # R16G16_SNORM
    38: ("<i2", 2, 4,  None),      # R16G16_SINT
    41: ("<f4", 1, 4,  None),      # R32_FLOAT
    42: ("<u4", 1, 4,  None),      # R32_UINT
    43: ("<i4", 1, 4,  None),      # R32_SINT
    49: ("<u1", 2, 2,  255.0),     # R8G8_UNORM
    54: ("<f2", 1, 2,  None),      # R16_FLOAT
    56: ("<u2", 1, 2,  65535.0),   # R16_UNORM
    57: ("<u2", 1, 2,  None),      # R16_UINT
    58: ("<i2", 1, 2,  32767.0),   # R16_SNORM
    61: ("<u1", 1, 1,  255.0),     # R8_UNORM
    62: ("<u1", 1, 1,  None),      # R8_UINT
}


class ExtractError(Exception):
    pass


def decode_normal_r32uint(np, packed):
    """CS2's compressed tangent frame (R32_UINT) -> unit normals.

    NOT inferred. Read off `csgo_environment_vs_max.glsl` in this repo,
    which is SPIR-V-derived, so the field widths come from the mask
    constants the shader actually applies rather than from anybody's
    reading of a name. Guessing widths from text has already cost this
    project a 43% retraction once.

        bit  0       bitangent sign  (0 -> -1, 1 -> +1)
        bits 1..11   tangent rotation, 2047 steps
        bits 12..21  octahedral X, 1023 steps
        bits 22..31  octahedral Y, 1023 steps

    The X/Y pair is an octahedral encoding, so the wrap fold below is
    load-bearing: dropping it leaves every normal in the lower hemisphere
    mirrored, which produces a plausible image and a wrong one.

    Only the NORMAL is returned. The full tangent basis needs the second
    basis vector of the frame, which is further down that shader than the
    region read here; the rotation and the bitangent sign are handed back
    unreduced so it can be finished without re-reading the buffer.
    """
    p = np.asarray(packed, dtype=np.uint32).reshape(-1)
    x = ((p >> np.uint32(12)) & np.uint32(1023)).astype(np.float32)
    y = ((p >> np.uint32(22)) & np.uint32(1023)).astype(np.float32)
    # 0.00195503421127796173 is 2/1023, verbatim from the shader.
    x = x * np.float32(2.0 / 1023.0) - np.float32(1.0)
    y = y * np.float32(2.0 / 1023.0) - np.float32(1.0)
    z = np.float32(1.0) - np.abs(x) - np.abs(y)
    t = np.clip(-z, 0.0, 1.0).astype(np.float32)
    x = x + np.where(x >= 0, -t, t)
    y = y + np.where(y >= 0, -t, t)
    n = np.stack([x, y, z], axis=1)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = (n / ln).astype(np.float32)
    rot = (((p >> np.uint32(1)) & np.uint32(2047)).astype(np.float32)
           * np.float32(0.003069460391998291))
    bsign = np.where((p & np.uint32(1)) == 0, np.float32(-1.0),
                     np.float32(1.0))
    return n, rot, bsign


# ----------------------------------------------------------------------
# resource containers
# ----------------------------------------------------------------------
def ordered_blocks(data):
    """[(tag, bytes)] IN BLOCK-INDEX ORDER.

    Order is load-bearing and a dict is wrong here: a vertex buffer names
    its storage by `m_nBlockIndex`, and a model with two vertex buffers
    has TWO `MVTX` blocks. Keying by tag collapses them and the second
    buffer silently reads the index block instead.
    """
    _fs, _hv, _ver, bo, bc = struct.unpack_from("<IHHII", data, 0)
    p, out = 8 + bo, []
    for _ in range(bc):
        tag = data[p:p + 4].decode("ascii", "replace")
        off, size = struct.unpack_from("<II", data, p + 4)
        out.append((tag, bytes(data[p + 4 + off:p + 4 + off + size])))
        p += 12
    return out


class Content:
    """Every VPK that could answer a path, searched in one place.

    A map VPK holds its world and its baked world-node models; the
    materials and textures those name live in the game archives. A
    resolver that only looks in the map reports a material as missing
    when it is merely elsewhere.
    """

    ARCHIVES = ("csgo/pak01_dir.vpk", "core/pak01_dir.vpk",
                "csgo_core/pak01_dir.vpk", "csgo_imported/pak01_dir.vpk",
                "csgo_lv/pak01_dir.vpk")

    def __init__(self, map_vpk, game_dir, map_name=None):
        self.map = VPK(map_vpk)
        self.vpks = [("<map>", self.map)]
        self.missing = collections.Counter()
        for rel in self.ARCHIVES:
            p = os.path.join(game_dir, rel)
            if os.path.exists(p):
                self.vpks.append((rel, VPK(p)))
        # Community maps ship their OWN assets in a per-map archive under
        # csgo_community_addons/<name>/, not in pak01. Six maps
        # (de_boulder, cs_shelter, de_fachwerk, de_debris, de_eldorado,
        # de_poseidon) reported hundreds of "missing" models purely
        # because this directory was not searched -- and a one-level glob
        # over game/*/*.vpk does not reveal it, which is how I concluded
        # the models were not shipped when they were sitting one
        # directory deeper. Absence of evidence was read as evidence of
        # absence; the archive was there the whole time.
        addons = os.path.join(game_dir, "csgo_community_addons")
        if os.path.isdir(addons):
            names = ([map_name] if map_name else []) + sorted(
                os.listdir(addons))
            seen = set()
            for nm in names:
                if not nm or nm in seen:
                    continue
                seen.add(nm)
                p = os.path.join(addons, nm, f"{nm}_dir.vpk")
                if os.path.exists(p):
                    self.vpks.append((f"csgo_community_addons/{nm}",
                                      VPK(p)))
        self._cache = {}

    def read(self, path):
        """Bytes for a resource path. COMPILED FORM FIRST.

        The `_c` form is tried before the bare name, and the order is not
        cosmetic. Map VPKs ship the UNCOMPILED SOURCE `.vmat` next to the
        compiled `.vmat_c` that lives in pak01 -- `ar_pool_day` carries
        `materials/decals/blacknumber1.vmat` as an 800-byte text file
        while the real resource is 3,392 bytes in `csgo/pak01_dir.vpk`.
        Preferring the bare name handed the text file to the resource
        reader, which then read a block table out of ASCII and raised;
        16 decal materials per affected map were dropped that way, as
        "material problems" rather than as anything anyone would look at.
        """
        key = path.replace("\\", "/").lstrip("/")
        for cand in (key + "_c", key):
            for _label, v in self.vpks:
                if cand in v.entries:
                    return v.read(cand)
        return None

    def find(self, path):
        b = self.read(path)
        if b is None:
            self.missing[path] += 1
        return b


# ----------------------------------------------------------------------
# meshopt
# ----------------------------------------------------------------------
def load_meshopt(hint=None):
    """Import the ctypes meshopt decoder, or say exactly why not.

    Returned as an object rather than imported at module scope so that
    the material half of an extraction still runs on a host without the
    codec -- and so that a run WITHOUT it is visible in the manifest
    instead of quietly producing a history with no triangles.
    """
    # `~/ashwork/meshopt` used to be searched first. That directory held the
    # ONLY copy of the decoder, was never committed, and was deleted on
    # 2026-08-07 as a "regenerable intermediate" -- taking the extraction's
    # geometry with it. The decoder now lives in-repo beside this file, and the
    # dangling home-directory candidate is gone so it can never again shadow it
    # with an untracked copy.
    for cand in ([hint] if hint else []) + [
            os.path.join(_HERE, "meshopt")]:
        if cand and os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import meshopt
                return meshopt
            except Exception as e:            # noqa: BLE001
                raise ExtractError(
                    f"meshopt found at {cand} but did not import: {e}")
    return None


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def decode_vertex_buffer(np, meshopt, desc, blocks):
    """One vertex buffer -> {semantic: array}, plus the raw bytes.

    Both are kept. The named arrays are what a renderer binds; the raw
    interleaved bytes are what makes a format this table does not yet
    decode recoverable instead of lost.
    """
    count = int(desc["m_nElementCount"])
    size = int(desc["m_nElementSizeInBytes"])
    # MBUF carries its payload inline; MVTX/MIDX name a resource block.
    raw = desc.get("_data")
    if raw is None:
        raw = blocks[int(desc["m_nBlockIndex"])][1]
    total = count * size

    if desc.get("m_bCompressedZSTD"):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(
            raw, max_output_size=total)
    elif desc.get("m_bMeshoptCompressed"):
        if meshopt is None:
            return None, None, "meshopt decoder unavailable"
        raw = meshopt.decode_vertex_buffer(raw, count, size)
    if len(raw) < total:
        return None, None, f"buffer {len(raw)}B < declared {total}B"
    raw = raw[:total]

    buf = np.frombuffer(raw, dtype=np.uint8).reshape(count, size)
    attrs, undecoded = {}, []
    for f in desc.get("m_inputLayoutFields", []):
        name = f.get("m_pSemanticName")
        if isinstance(name, (bytes, bytearray)):
            name = name.rstrip(b"\0").decode("utf-8", "replace")
        idx = int(f.get("m_nSemanticIndex", 0))
        fmt = int(f.get("m_Format", 0))
        off = int(f.get("m_nOffset", 0))
        key = f"{str(name).upper()}_{idx}"
        spec = DXGI.get(fmt)
        if spec is None:
            undecoded.append({"semantic": key, "format": fmt, "offset": off})
            continue
        dt, comps, width, scale = spec
        if off + width > size:
            undecoded.append({"semantic": key, "format": fmt, "offset": off,
                              "why": "field runs past the vertex stride"})
            continue
        sl = np.ascontiguousarray(buf[:, off:off + width])
        a = sl.view(np.dtype(dt)).reshape(count, comps)
        if scale is not None:
            a = a.astype(np.float32) / np.float32(scale)
        elif a.dtype == np.float16:
            a = a.astype(np.float32)
        attrs[key] = a
        if key.startswith("NORMAL") and fmt == 42:
            # Stored ALONGSIDE the packed value, never instead of it.
            # The packed uint is what the buffer holds and what the
            # reference reads; the decode is a view, and a view that
            # deletes its source is exactly the canonicalisation this
            # format exists to refuse.
            n, rot, bsign = decode_normal_r32uint(np, a)
            attrs[key + "_decoded"] = n
            attrs[key + "_tangent_rotation"] = rot.reshape(-1, 1)
            attrs[key + "_bitangent_sign"] = bsign.reshape(-1, 1)
    return attrs, (raw, undecoded), None


def decode_index_buffer(np, meshopt, desc, blocks):
    count = int(desc["m_nElementCount"])
    size = int(desc["m_nElementSizeInBytes"])
    raw = desc.get("_data")
    if raw is None:
        raw = blocks[int(desc["m_nBlockIndex"])][1]
    total = count * size
    if desc.get("m_bCompressedZSTD"):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(
            raw, max_output_size=total)
    elif desc.get("m_bMeshoptCompressed"):
        if meshopt is None:
            return None, "meshopt decoder unavailable"
        if desc.get("m_bMeshoptIndexSequence"):
            raw = meshopt.decode_index_sequence(raw, count, size)
        else:
            raw = meshopt.decode_index_buffer(raw, count, size)
    if len(raw) < total:
        return None, f"index buffer {len(raw)}B < declared {total}B"
    dt = np.uint16 if size == 2 else np.uint32
    return np.frombuffer(raw[:total], dtype=dt).copy(), None


def parse_mbuf(mbuf):
    """The pre-MVTX/MIDX VBIB block -> the same descriptors CTRL gives.

    Older maps (cs_italy, ar_pool_day, the workshop previews) put every
    buffer in ONE `MBUF` block with its own offset table, instead of one
    resource block per buffer plus a CTRL describing them. Same data,
    different container -- so this returns dicts shaped like CTRL's
    `m_vertexBuffers` entries and the rest of the path is unchanged.

    Layout, and it self-checks: the header is two (offset, count) pairs,
    each offset relative to where it was read. Each buffer descriptor is
    24 bytes -- element count, element size, input-layout offset/count,
    data offset, total size -- with the two internal offsets also
    relative to their own field. Each input-layout field is 56 bytes: a
    32-byte semantic name then six 4-byte fields.

    The arithmetic is checkable against the block length rather than
    trusted: on cs_italy's first world node the index data ends at byte
    525 of a 525-byte block, which is the kind of agreement that says
    the strides are right rather than merely plausible.

    `total_size != count * stride` is how a buffer declares it is
    meshopt compressed here; there is no `m_bMeshoptCompressed` flag in
    this layout.
    """
    vb_off, vb_cnt = struct.unpack_from("<II", mbuf, 0)
    ib_off, ib_cnt = struct.unpack_from("<II", mbuf, 8)
    vb_off += 0
    ib_off += 8

    def descriptors(base, n):
        out = []
        for i in range(n):
            p = base + i * 24
            # The element size is SIGNED, and its high bit is a flag, not
            # magnitude: an index buffer reports 0x80000002, which is
            # "2 bytes, meshopt compressed" and not a 2.1-billion-byte
            # stride. VRF marks this field `isSizeNegative` with a TODO;
            # what it indicates is the compression, confirmed here by 200
            # cs_italy models decoding cleanly once the bit is split off.
            count, raw_stride = struct.unpack_from("<Ii", mbuf, p)
            stride = raw_stride & 0x7FFFFFFF
            flagged = raw_stride < 0
            lay_off, lay_cnt = struct.unpack_from("<II", mbuf, p + 8)
            data_off, total = struct.unpack_from("<Ii", mbuf, p + 16)
            lay_off += p + 8
            data_off += p + 16
            fields = []
            for j in range(lay_cnt):
                q = lay_off + j * 56
                name = mbuf[q:q + 32].split(b"\0")[0].decode(
                    "utf-8", "replace")
                (sem_idx, fmt, off, slot, slot_type,
                 step) = struct.unpack_from("<iIIiIi", mbuf, q + 32)
                fields.append({
                    "m_pSemanticName": name,
                    "m_nSemanticIndex": sem_idx,
                    "m_Format": fmt,
                    "m_nOffset": off,
                    "m_nSlot": slot,
                    "m_nSlotType": slot_type,
                    "m_nInstanceStepRate": step,
                })
            end = data_off + (total if total > 0 else count * stride)
            if end > len(mbuf):
                raise ExtractError(
                    f"MBUF buffer {i} ends at {end} past the {len(mbuf)}-"
                    f"byte block; the descriptor stride is wrong")
            out.append({
                "m_nElementCount": count,
                "m_nElementSizeInBytes": stride,
                "m_inputLayoutFields": fields,
                "_data": bytes(mbuf[data_off:data_off + (
                    total if total > 0 else count * stride)]),
                # Either signal is sufficient: the high-bit flag, or a
                # payload shorter than count * stride.
                "m_bMeshoptCompressed": flagged or (
                    total > 0 and total != count * stride),
                "m_bCompressedZSTD": False,
                "m_bMeshoptIndexSequence": False,
            })
        return out

    return descriptors(vb_off, vb_cnt), descriptors(ib_off, ib_cnt)


def read_model(np, meshopt, content, path, keep_raw):
    """A `.vmdl_c` -> vertex/index buffers, draw calls, and its materials."""
    data = content.find(path)
    if data is None:
        return None, f"model not found: {path}"
    blocks = ordered_blocks(data)
    by_tag = {}
    for t, b in blocks:
        by_tag.setdefault(t, b)
    if "CTRL" not in by_tag:
        return None, f"{path}: no CTRL block (nothing describes the buffers)"

    ctrl = kv3v5.parse(by_tag["CTRL"])

    mdat = kv3v5.parse(by_tag["MDAT"]) if "MDAT" in by_tag else {}

    out = {"path": path, "vertex_buffers": [], "index_buffers": [],
           "draw_calls": [], "problems": []}

    if "MBUF" in by_tag:
        vbs, ibs = parse_mbuf(by_tag["MBUF"])
        meshes = [{"m_vertexBuffers": vbs, "m_indexBuffers": ibs}]
    else:
        meshes = ctrl.get("embedded_meshes", [])

    for emb in meshes:
        for vb in emb.get("m_vertexBuffers", []):
            attrs, rawpair, err = decode_vertex_buffer(
                np, meshopt, vb, blocks)
            rec = {
                "count": int(vb["m_nElementCount"]),
                "stride": int(vb["m_nElementSizeInBytes"]),
                "layout": vb.get("m_inputLayoutFields", []),
                "meshopt": bool(vb.get("m_bMeshoptCompressed")),
            }
            if err:
                rec["error"] = err
                out["problems"].append(f"vertex buffer: {err}")
            else:
                rec["attrs"] = attrs
                rec["undecoded_fields"] = rawpair[1]
                if keep_raw:
                    rec["raw"] = np.frombuffer(rawpair[0], dtype=np.uint8)
            out["vertex_buffers"].append(rec)
        for ib in emb.get("m_indexBuffers", []):
            arr, err = decode_index_buffer(np, meshopt, ib, blocks)
            rec = {"count": int(ib["m_nElementCount"]),
                   "index_size": int(ib["m_nElementSizeInBytes"])}
            if err:
                rec["error"] = err
                out["problems"].append(f"index buffer: {err}")
            else:
                rec["indices"] = arr
            out["index_buffers"].append(rec)

    if not out["vertex_buffers"]:
        # A model that yields no buffers must SAY so. Older maps
        # (cs_italy, ar_pool_day, the workshop previews) ship an `MBUF`
        # block -- the earlier VBIB layout -- instead of MVTX/MIDX, and
        # this walk finds nothing in it. Reported as a gap rather than
        # returned as a mesh with zero triangles, because a silent zero
        # is indistinguishable from an empty model.
        tags = sorted({t for t, _ in blocks})
        out["problems"].append(
            f"no vertex buffers decoded; blocks present {tags}. "
            f"{'MBUF (pre-MVTX VBIB layout) is NOT yet read' if 'MBUF' in tags else 'CTRL declared no embedded_meshes'}")

    for so in mdat.get("m_sceneObjects", []):
        for dc in so.get("m_drawCalls", []):
            out["draw_calls"].append({
                "material": dc.get("m_material"),
                "primitive": dc.get("m_nPrimitiveType"),
                "base_vertex": dc.get("m_nBaseVertex"),
                "vertex_count": dc.get("m_nVertexCount"),
                "start_index": dc.get("m_nStartIndex"),
                "index_count": dc.get("m_nIndexCount"),
                "index_buffer": dc.get("m_indexBuffer"),
                "vertex_buffers": dc.get("m_vertexBuffers"),
                "tint": dc.get("m_vTintColor"),
                "alpha": dc.get("m_flAlpha"),
                "uv_density": dc.get("m_flUvDensity"),
                # These two decide how the vertex data is INTERPRETED,
                # not merely how it looks, so they travel with the call.
                "compressed_normal_tangent":
                    dc.get("m_bUseCompressedNormalTangent"),
                "baked_lighting_from_lightmap":
                    dc.get("m_bHasBakedLightingFromLightMap"),
            })
    return out, None


def build_flat(np, models, instances):
    """One world-space triangle soup a rasteriser can bind directly.

    AXIS CONVENTION, stated rather than assumed. Positions are CS2 world
    space, **Z-up**, in inches, with NO axis conversion applied -- the
    map's own bounds confirm it (de_inferno spans Z -280..1624 while X
    and Y span thousands). A consumer that wants Y-up must convert, and
    a packer that silently converted is where a floor ends up reporting
    `normal.y = -1`.

    `face_normal` is GEOMETRIC: cross(v1-v0, v2-v0), normalised, from the
    index winding as shipped. It is derived here rather than taken from
    the vertex normals so that it can be checked against them -- and the
    check is the point, because a sign error in either one produces a
    plausible image rather than an obvious failure.
    """
    P, F, FM, VN = [], [], [], []
    mats, mat_id = [], {}
    base = 0
    skipped = 0
    for inst in instances:
        m = models.get(inst["model"])
        if not m or not m["vertex_buffers"] or not m["index_buffers"]:
            skipped += 1
            continue
        vb = None
        for cand in m["vertex_buffers"]:
            if "attrs" in cand and "POSITION_0" in cand["attrs"]:
                vb = cand
                break
        if vb is None or "indices" not in m["index_buffers"][0]:
            skipped += 1
            continue
        pos = np.asarray(vb["attrs"]["POSITION_0"], dtype=np.float32)
        nrm = None
        for cand in m["vertex_buffers"]:
            a = cand.get("attrs", {})
            if "NORMAL_0_decoded" in a and len(a["NORMAL_0_decoded"]) == len(pos):
                nrm = np.asarray(a["NORMAL_0_decoded"], dtype=np.float32)
                break

        T = np.asarray(inst.get("transform") or
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                       dtype=np.float32)
        if T.shape != (3, 4):
            skipped += 1
            continue
        world = pos @ T[:, :3].T + T[:, 3]
        R = T[:, :3]
        if nrm is not None:
            wn = nrm @ R.T
            ln = np.linalg.norm(wn, axis=1, keepdims=True)
            ln[ln == 0] = 1.0
            wn = wn / ln
        else:
            wn = np.zeros_like(world)

        idx = np.asarray(m["index_buffers"][0]["indices"]).astype(np.int64)
        # Collect this instance's draw calls FIRST. Appending the vertex
        # block once per draw call instead of once per instance inflates
        # the vertex count by the draw-call fan-out -- it read as 883M
        # vertices for a 15M-vertex map, which is the kind of number that
        # is only obviously wrong if someone looks at it.
        got = []
        for dc in m["draw_calls"]:
            ic = int(dc.get("index_count") or 0)
            si = int(dc.get("start_index") or 0)
            bv = int(dc.get("base_vertex") or 0)
            if ic < 3:
                continue
            tri = idx[si:si + (ic // 3) * 3]
            if tri.size < 3:
                continue
            tri = tri.reshape(-1, 3) + bv
            if tri.max() >= len(world):
                skipped += 1
                continue
            mp = dc.get("material") or ""
            if mp not in mat_id:
                mat_id[mp] = len(mats)
                mats.append(mp)
            got.append((tri, mat_id[mp]))
        if not got:
            continue
        P.append(world)
        VN.append(wn)
        for tri, mid in got:
            F.append(tri + base)
            FM.append(np.full(len(tri), mid, dtype=np.int32))
        base += len(world)

    if not F:
        return None, skipped
    positions = np.concatenate(P).astype(np.float32)
    vnormals = np.concatenate(VN).astype(np.float32)
    faces = np.concatenate(F).astype(np.int32)
    face_mat = np.concatenate(FM)
    e1 = positions[faces[:, 1]] - positions[faces[:, 0]]
    e2 = positions[faces[:, 2]] - positions[faces[:, 0]]
    fn = np.cross(e1, e2)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    fn = (fn / ln).astype(np.float32)
    return {"positions": positions, "faces": faces, "face_material": face_mat,
            "face_normal": fn, "vertex_normal": vnormals,
            "materials": mats, "up_axis": "z",
            "units": "inches (CS2 world space, unconverted)"}, skipped


# ----------------------------------------------------------------------
# materials
# ----------------------------------------------------------------------
def _named(rows, value_key):
    """[{m_name, <value_key>}] -> {name: value}, under the SOURCE name."""
    out = {}
    for r in rows or []:
        if isinstance(r, dict) and "m_name" in r:
            out[r["m_name"]] = r.get(value_key)
    return out


def read_material(content, path):
    data = content.find(path)
    if data is None:
        return None, f"material not found: {path}"
    blocks = dict(ordered_blocks(data))
    if "DATA" not in blocks:
        return None, f"{path}: no DATA block"
    m = kv3v5.parse(blocks["DATA"])

    shader = m.get("m_shaderName")
    if not shader:
        # The field glTF drops. Without it the row cannot be dispatched
        # to one of the 179 families, so it is refused rather than
        # defaulted -- a guessed family is worse than a named gap.
        return None, f"{path}: no m_shaderName in DATA"

    ints = _named(m.get("m_intParams"), "m_nValue")
    floats = _named(m.get("m_floatParams"), "m_flValue")
    vectors = _named(m.get("m_vectorParams"), "m_value")
    textures = _named(m.get("m_textureParams"), "m_pValue")

    row = {
        "shader_name": shader,
        "material_name": m.get("m_materialName", path),
        # F_* are the feature/combo axes; they are separated for lookup
        # but NOT removed from int_params -- the split is a view, and a
        # view that deletes its source is a canonicalisation.
        "features": {k: v for k, v in ints.items() if k.startswith("F_")},
        "int_params": ints,
        "float_params": floats,
        "vector_params": vectors,
        "texture_params": textures,
        "dynamic_params": _named(m.get("m_dynamicParams"), "m_value"),
        "dynamic_texture_params": _named(
            m.get("m_dynamicTextureParams"), "m_value"),
        "int_attributes": _named(m.get("m_intAttributes"), "m_nValue"),
        "float_attributes": _named(m.get("m_floatAttributes"), "m_flValue"),
        "vector_attributes": _named(m.get("m_vectorAttributes"), "m_value"),
        "string_attributes": _named(m.get("m_stringAttributes"), "m_value"),
        "texture_attributes": _named(m.get("m_textureAttributes"), "m_value"),
        "render_attributes_used": m.get("m_renderAttributesUsed"),
        # THE RULE: everything, verbatim, even what is already indexed
        # above. A parameter whose meaning is unresolved is still stored.
        "raw": m,
        "resource_refs": kv3v5.resource_refs(blocks.get("RERL")),
    }
    return row, None


# vtex_c image formats, by the number the texture header declares. This
# is the FILE's encoding. It is not the same question as the slot's.
# Transcribed from ValveResourceFormat's `VTexFormat.cs`, NOT from
# memory. The first version of this table was written from memory and was
# off by one from index 23 onward, which is not a harmless slip: it
# reported 78 of de_inferno's textures as `RG11_EAC` when they are
# `ATI1N`. Both are "compressed", so nothing raised and nothing looked
# empty -- it would simply have told the BC decoder to read a
# single-channel BC4 surface as a two-channel ETC2 one. Same bytes,
# wrong decoder, which is the exact failure ASH exists to prevent, and
# it was caught only because 78 mobile-format textures in a PC build did
# not make sense.
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


def texture_format(data):
    """The declared image format of a `.vtex_c`, without decoding it.

    Read off the resource header rather than inferred from the bytes:
    BC7 and DXT5 blocks are the same size and telling them apart by
    inspection is the mistake this field exists to prevent.
    """
    try:
        blocks = dict(ordered_blocks(data))
        d = blocks.get("DATA")
        if not d:
            return None
        # vtex_c DATA is a fixed struct, not KV3, and the field order is
        # exact: version u16, flags u16, FOUR reflectivity floats, then
        # width/height/depth u16, then the format as a SINGLE BYTE.
        # An earlier guess here put six floats and three u32s ahead of
        # the dimensions; it did not raise, it just read a byte from the
        # wrong place and reported UNKNOWN for all 20,419 textures --
        # a wrong answer that looked like an unsupported field rather
        # than a bad offset.
        (ver, flags, r0, r1, r2, r3,
         w, h, depth, fmt, mips) = struct.unpack_from("<HH4f3HBB", d, 0)
        if ver != 1:
            return {"format_name": f"unreadable_vtex_version_{ver}"}
        return {"m_nImageFormat": fmt,
                "format_name": VTEX_FORMAT.get(fmt, f"format_{fmt}"),
                "width": w, "height": h, "depth": depth,
                "mip_levels": mips, "flags": flags,
                "reflectivity": [r0, r1, r2, r3]}
    except Exception:                                    # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# world traversal
# ----------------------------------------------------------------------
def world_instances(content, map_name):
    """Every scene object in the map, with its transform and model.

    Both `m_sceneObjects` and `m_aggregateSceneObjects` are walked; the
    aggregates are where the merged instance streams live and skipping
    them drops most of the visible surface of a map.
    """
    wpath = f"maps/{map_name}/world.vwrld"
    wdata = content.find(wpath)
    if wdata is None:
        return None, None, f"no {wpath} in this VPK"
    world = kv3v5.parse(dict(ordered_blocks(wdata))["DATA"])

    instances, nodes, problems = [], [], []
    # CONSERVATION PER RECORD TYPE. A .get() miss is indistinguishable
    # from a field that is legitimately absent, which is how a whole
    # schema went unread while the loop reported success. Counting
    # recognised against seen makes the next one loud on the first run.
    import collections as _c
    n_seen, n_ok = _c.Counter(), _c.Counter()
    for wn in world.get("m_worldNodes", []):
        prefix = str(wn.get("m_worldNodePrefix", "")).replace("\\", "/")
        if not prefix:
            continue
        ndata = content.find(prefix + ".vwnod")
        if ndata is None:
            problems.append(f"world node {prefix}.vwnod_c missing")
            continue
        node = kv3v5.parse(dict(ordered_blocks(ndata))["DATA"])
        nodes.append({"prefix": prefix,
                      "min": wn.get("m_vMinBounds"),
                      "max": wn.get("m_vMaxBounds"),
                      "origin": wn.get("m_vOrigin")})
        # THE TWO COLLECTIONS ARE DIFFERENT RECORD TYPES. They share
        # exactly ONE field name, m_renderableModel. Reading both with
        # m_sceneObjects' schema made every other .get() return None, so
        # 346 of de_inferno's 784 instances -- carrying 8,380,186 of
        # 8,419,800 triangles, 99.5% of the map -- were appended as dicts
        # of Nones and counted as decoded.
        #
        # It survived because the ONE field whose absence is harmless is
        # the one that would have raised the alarm: aggregates are
        # pre-merged in WORLD space, so transform=None yields correct
        # geometry. The other seven hid behind it.
        for so in node.get("m_sceneObjects", []):
            model = so.get("m_renderableModel") or so.get("m_renderable")
            if not model:
                continue
            n_seen["m_sceneObjects"] += 1
            n_ok["m_sceneObjects"] += 1
            instances.append({
                "model": model, "node": prefix, "kind": "m_sceneObjects",
                "transform": so.get("m_vTransform"),
                "tint": so.get("m_vTintColor"),
                "flags": so.get("m_nObjectTypeFlags"),
                "fade_start": so.get("m_flFadeStartDistance"),
                "fade_end": so.get("m_flFadeEndDistance"),
                "lod_override": so.get("m_nLODOverride"),
                "lighting_origin": so.get("m_vLightingOrigin"),
                "skin": so.get("m_skin"),
            })
        for so in node.get("m_aggregateSceneObjects", []):
            model = so.get("m_renderableModel") or so.get("m_renderable")
            if not model:
                continue
            n_seen["m_aggregateSceneObjects"] += 1
            # The tint lives one level DEEPER, per mesh, keyed by the
            # draw call it applies to -- not per instance at all, which
            # is why fixing the instance-level read could never have
            # found it. 1691 of de_inferno's 6176 aggregate meshes carry
            # a NON-identity tint (27.4%); the sample [106,106,106] is a
            # 41% darkening and (91,77,68) a warm brown.
            #
            # m_vertexAlbedoStream and m_instanceStream are both -1 on
            # every aggregate here, so the tint is NOT a vertex stream --
            # checked, not assumed.
            meshes = so.get("m_aggregateMeshes") or []
            dc_tint = {}
            for me in meshes:
                dci = me.get("m_nDrawCallIndex")
                if dci is None:
                    continue
                t = me.get("m_vTintColor")
                if t is not None:
                    dc_tint[int(dci)] = list(t)
            rec = {
                "model": model, "node": prefix,
                "kind": "m_aggregateSceneObjects",
                # Aggregates are pre-merged in world space; a per-instance
                # transform is absent BY DESIGN, not by decode failure.
                "transform": None,
                "transform_absent_by_design": True,
                # m_allFlags / m_anyFlags are this type's flag fields.
                "flags": so.get("m_allFlags"),
                "any_flags": so.get("m_anyFlags"),
                "layer": so.get("m_nLayer"),
                "fragment_transforms": so.get("m_fragmentTransforms"),
                "draw_call_tint": dc_tint,
                "n_meshes": len(meshes),
                "tint": None, "fade_start": None, "fade_end": None,
                "lod_override": None, "lighting_origin": None, "skin": None,
            }
            if any(v is not None for v in
                   (rec["flags"], rec["layer"])) or dc_tint or meshes:
                n_ok["m_aggregateSceneObjects"] += 1
            instances.append(rec)
    for _k in n_seen:
        if n_ok[_k] != n_seen[_k]:
            problems.append(
                f"{_k}: decoded {n_ok[_k]} of {n_seen[_k]} records -- a "
                f"record type yielding no recognised fields is a SCHEMA "
                f"MISMATCH, not an absent field")
    return world, {"instances": instances, "nodes": nodes,
                   "problems": problems,
                   "decoded_by_type": {k: [n_ok[k], n_seen[k]]
                                       for k in n_seen}}, None


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", required=True, help="path to <name>.vpk")
    ap.add_argument("--out", required=True, help="output <name>.ash dir")
    ap.add_argument("--game-dir",
                    default=os.path.expanduser("~/cs2/game"),
                    help="CS2 game dir holding csgo/pak01_dir.vpk")
    ap.add_argument("--meshopt", default=None,
                    help="dir holding meshopt.py + libmeshopt.so")
    ap.add_argument("--max-models", type=int, default=0,
                    help="cap distinct models decoded (0 = all)")
    ap.add_argument("--no-geometry", action="store_true")
    ap.add_argument("--no-textures", action="store_true")
    ap.add_argument("--keep-raw-vertex", action="store_true",
                    help="also store the raw interleaved vertex bytes")
    ap.add_argument("--allow-no-geometry", action="store_true",
                    help="do not refuse when the meshopt decoder is missing; "
                         "the extraction then carries materials and textures "
                         "and NO triangles, on purpose")
    ap.add_argument("--report", default=None,
                    help="write the per-map summary as JSON here too")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    t0 = time.time()
    map_name = os.path.basename(args.map)
    for suf in ("_dir.vpk", ".vpk"):
        if map_name.endswith(suf):
            map_name = map_name[:-len(suf)]
            break

    content = Content(args.map, args.game_dir, map_name=map_name)
    meshopt = None if args.no_geometry else load_meshopt(args.meshopt)

    w = AshWriter(args.out, source=os.path.abspath(args.map),
                  tool=EXTRACTOR_VERSION,
                  command=" ".join([os.path.basename(sys.argv[0])] +
                                   (argv if argv is not None else
                                    sys.argv[1:])))
    w.meta["map"] = map_name
    w.meta["game_dir"] = os.path.abspath(args.game_dir)
    w.meta["archives"] = [lbl for lbl, _ in content.vpks]

    world, scene, err = world_instances(content, map_name)
    if err:
        raise ExtractError(f"{map_name}: {err}")

    summary = {"map": map_name, "instances": len(scene["instances"]),
               "world_nodes": len(scene["nodes"]),
               "problems": list(scene["problems"])}

    # -- geometry -------------------------------------------------------
    models, model_problems = {}, []
    if not args.no_geometry:
        wanted = []
        seen = set()
        for inst in scene["instances"]:
            p = inst["model"]
            if p not in seen:
                seen.add(p)
                wanted.append(p)
        if args.max_models:
            dropped = max(0, len(wanted) - args.max_models)
            if dropped:
                # A cap that is not reported reads as full coverage.
                model_problems.append(
                    f"--max-models {args.max_models} dropped {dropped} of "
                    f"{len(wanted)} distinct models")
            wanted = wanted[:args.max_models]
        for i, p in enumerate(wanted):
            try:
                m, e = read_model(np, meshopt, content, p,
                                  args.keep_raw_vertex)
            except Exception as exc:                     # noqa: BLE001
                m, e = None, f"{p}: {type(exc).__name__}: {exc}"
            if e:
                model_problems.append(e)
                continue
            models[p] = m
            model_problems.extend(f"{p}: {x}" for x in m["problems"])
            if (i + 1) % 200 == 0:
                print(f"  models {i + 1}/{len(wanted)}", flush=True)
        summary["models_decoded"] = len(models)
        summary["models_wanted"] = len(wanted)

    def to_t(o):
        if isinstance(o, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(o))
        if isinstance(o, dict):
            return {k: to_t(v) for k, v in o.items()}
        if isinstance(o, list):
            return [to_t(v) for v in o]
        if isinstance(o, (bytes, bytearray)):
            return torch.frombuffer(bytes(o), dtype=torch.uint8)
        return o

    tris = 0
    verts = 0
    for m in models.values():
        for vb in m["vertex_buffers"]:
            verts += vb.get("count", 0)
        for ib in m["index_buffers"]:
            tris += ib.get("count", 0) // 3
    summary["vertices"] = verts
    summary["triangles"] = tris

    flat, flat_skipped = (None, 0)
    if models:
        flat, flat_skipped = build_flat(np, models, scene["instances"])
    if flat is not None:
        fn = flat["face_normal"]
        # Report the floor sign rather than assert it. A packer elsewhere
        # in this project emitted floors with normal.y = -1 and a
        # sign-trusting sampler concluded the world had no standable
        # geometry; the fix is to state the convention and the measured
        # distribution, not to flip until it looks right.
        # THE DECISIVE CHECK. The geometric normal comes from the index
        # winding; the vertex normal comes from the shipped compressed
        # tangent frame. They are two independent derivations, so their
        # agreement tests BOTH the winding convention and the octahedral
        # decode at once. Half-and-half would mean one of them is wrong;
        # a strong majority agreeing means neither is inverted.
        vn = flat["vertex_normal"]
        if vn.any():
            fv = vn[flat["faces"]].mean(axis=1)
            ln = np.linalg.norm(fv, axis=1, keepdims=True)
            ln[ln == 0] = 1.0
            dot = (fn * (fv / ln)).sum(axis=1)
            ok = np.abs(dot) > 0.1
            summary["faces_with_usable_vertex_normal"] = int(ok.sum())
            summary["geometric_vs_vertex_normal_agree_frac"] = round(
                float((dot[ok] > 0).mean()) if ok.any() else -1.0, 4)
        up = np.abs(fn[:, 2]) > 0.9
        summary["flat_triangles"] = int(len(flat["faces"]))
        summary["flat_vertices"] = int(len(flat["positions"]))
        summary["flat_materials"] = int(len(flat["materials"]))
        summary["near_horizontal_faces"] = int(up.sum())
        summary["of_which_normal_z_positive"] = int(
            (fn[up, 2] > 0).sum()) if up.any() else 0
        summary["flat_skipped_instances"] = int(flat_skipped)
        summary["bbox_min"] = [float(x) for x in flat["positions"].min(0)]
        summary["bbox_max"] = [float(x) for x in flat["positions"].max(0)]

    w.add_geometry(models=to_t(models), instances=scene["instances"],
                   world_nodes=scene["nodes"],
                   problems=model_problems,
                   flat=to_t(flat) if flat is not None else None)

    # -- materials ------------------------------------------------------
    mat_paths = set()
    for m in models.values():
        for dc in m["draw_calls"]:
            if dc.get("material"):
                mat_paths.add(dc["material"])
    if not mat_paths:
        # Geometry off (or capped): take the materials from the world
        # nodes' own resource lists so the material half still runs.
        for inst in scene["instances"]:
            d = content.read(inst["model"])
            if d:
                for r in kv3v5.resource_refs(dict(ordered_blocks(d))
                                             .get("RERL")):
                    if r.endswith(".vmat"):
                        mat_paths.add(r)

    rows, mat_problems = {}, []
    for p in sorted(mat_paths):
        try:
            row, e = read_material(content, p)
        except Exception as exc:                          # noqa: BLE001
            row, e = None, f"{p}: {type(exc).__name__}: {exc}"
        if e:
            mat_problems.append(e)
            continue
        rows[p] = row

    # ENCODING TRAVELS WITH THE SLOT, NOT THE FILE. `g_tNormal` is
    # DXT5nm in one family and hemi-octahedral in another, so the key is
    # (shader family, slot) and never the texture path.
    slot_enc, tex_paths = {}, {}
    for p, row in rows.items():
        fam = row["shader_name"]
        for slot, tpath in row["texture_params"].items():
            if not isinstance(tpath, str) or not tpath:
                continue
            tex_paths.setdefault(tpath, set()).add((fam, slot))
            slot_enc.setdefault(f"{fam}::{slot}", {
                "shader_name": fam, "slot": slot,
                "file_formats": {}, "materials": 0})
            slot_enc[f"{fam}::{slot}"]["materials"] += 1

    shaders = collections.Counter(r["shader_name"] for r in rows.values())
    summary["materials"] = len(rows)
    summary["distinct_shader_names"] = len(shaders)
    summary["shader_histogram"] = shaders.most_common()
    summary["material_problems"] = len(mat_problems)

    # -- textures -------------------------------------------------------
    n_tex, tex_bytes = 0, 0
    if not args.no_textures:
        for tpath in sorted(tex_paths):
            data = content.read(tpath)
            if data is None:
                content.missing[tpath] += 1
                continue
            info = texture_format(data)
            name = tpath.replace("\\", "/").lstrip("/")
            if not name.endswith("_c"):
                name += "_c"
            w.add_texture(name, data,
                          encoding=(info or {}).get("format_name"))
            n_tex += 1
            tex_bytes += len(data)
            for fam, slot in tex_paths[tpath]:
                k = f"{fam}::{slot}"
                ff = slot_enc[k]["file_formats"]
                fn = (info or {}).get("format_name", "unreadable")
                ff[fn] = ff.get(fn, 0) + 1
    summary["textures"] = n_tex
    summary["texture_bytes"] = tex_bytes

    w.add_materials(rows, encodings=slot_enc)

    # -- what this history does NOT claim -------------------------------
    w.note("Extracted directly from the map VPK through vcs/vpk.py and "
           "vcs/kv3v5.py. glTF is not in this path; no material channel "
           "passed through a format that cannot express it.")
    w.note("Every resource in this build is KV3 version 5. vcs/kv3.py "
           "reads version 4 only and raises mid-payload on these files; "
           "vcs/kv3v5.py is the decoder used here.")
    w.note("slot_encodings is keyed (shader_name, slot) and records the "
           "DECLARED IMAGE FORMAT of the .vtex_c files bound to that "
           "slot. That is the file's encoding. The slot's SEMANTIC "
           "encoding -- DXT5nm .wy vs hemi-octahedral vs the cables "
           "diagonal form -- is a property of the shader family's own "
           "declaration in its .vcs and is NOT resolved here. The bytes "
           "and the binding are both stored, so it can be resolved "
           "later; it is not being claimed now.")
    w.note("Textures are stored as the .vtex_c resource verbatim, in "
           "source encoding. They are not decoded to RGBA.")
    w.note("Animation is not extracted. Pass ordering, resolution and "
           "scheduling are not in any shader package and are not here.")
    if meshopt is None and not args.no_geometry:
        w.note("GEOMETRY IS ABSENT OR PARTIAL: the meshopt decoder was "
               "not available, and every CS2 world buffer is meshopt "
               "compressed.")
    for pr in model_problems[:200]:
        w.note("model: " + pr)
    for pr in mat_problems[:200]:
        w.note("material: " + pr)
    if content.missing:
        w.note(f"{len(content.missing)} referenced path(s) not found in "
               f"any archive; first: "
               f"{list(content.missing)[:5]}")

    summary["unresolved_paths"] = len(content.missing)
    summary["model_problems"] = len(model_problems)
    summary["seconds"] = round(time.time() - t0, 1)
    w.meta["summary"] = summary
    w.close()

    print(json.dumps(summary, indent=1)[:4000], flush=True)
    print(f"MESHOPT: {getattr(meshopt, 'library_path', lambda: None)()}",
          flush=True)

    # REFUSE a geometry-free extraction unless it was asked for.
    #
    # Without this the run exits 0 and writes a well-formed .ash. The obvious
    # guard does not catch it either: summary["triangles"] is the DECLARED
    # element count read from the buffer descriptors, and reads in the millions
    # on a run that decoded nothing -- measured on de_train 2026-08-08,
    # 7,578,208 declared against 32 flattened. The number that discriminates is
    # flat_triangles. A silent zero-geometry map is how ~/ashtex_inferno.log
    # came to exist.
    if (not args.no_geometry and meshopt is None
            and not args.allow_no_geometry):
        sys.stderr.write(
            "REFUSING: the meshopt decoder was not available and every CS2 "
            "world buffer is meshopt compressed, so this extraction decoded "
            f"{summary.get('flat_triangles', 0):,} triangles of "
            f"{summary.get('triangles', 0):,} declared. The .ash at "
            f"{args.out} is materials-and-textures only.\n"
            "  * the decoder ships in harness/gpu_render/meshopt/; point "
            "--meshopt at it if this ran from elsewhere\n"
            "  * set MESHOPT_LIB if libmeshoptimizer is not where it looks\n"
            "  * pass --allow-no-geometry to say you meant materials only\n")
        return 2
    return 0
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
