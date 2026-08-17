"""Strict, read-only loaders for the Source 1 asset formats used by CS:GO.

These readers are intentionally small.  They expose the binary facts needed by
the reference backend without pretending to implement the Source 1 runtime.
Layouts follow Valve's public Source SDK 2013 headers.  CS:GO shipped forks of
some formats, so an unfamiliar version is a refusal rather than a best guess.
"""

from __future__ import annotations

import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


class Source1FormatError(ValueError):
    """An asset is truncated, inconsistent, or not a supported Source 1 form."""


def _range(data: bytes, offset: int, size: int, label: str) -> memoryview:
    if offset < 0 or size < 0 or offset > len(data) - size:
        raise Source1FormatError(
            f"{label} range [{offset}, {offset + size}) exceeds {len(data)} bytes"
        )
    return memoryview(data)[offset : offset + size]


def _cstr(data: bytes, offset: int, limit: int, label: str) -> str:
    raw = bytes(_range(data, offset, limit, label))
    end = raw.find(b"\0")
    if end < 0:
        raise Source1FormatError(f"{label} has no NUL terminator in {limit} bytes")
    try:
        return raw[:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Source1FormatError(f"{label} is not UTF-8") from exc


def _fixed_cstr(data: bytes, offset: int, size: int, label: str) -> str:
    """Decode a fixed C character array, which need not end in NUL.

    Studio model names are diagnostic fixed-width buffers.  Real CS:GO 49
    assets contain names that fill all 64 bytes; treating those arrays like
    offset strings incorrectly rejects an otherwise exact structure.
    """
    raw = bytes(_range(data, offset, size, label))
    end = raw.find(b"\0")
    if end >= 0:
        raw = raw[:end]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Source1FormatError(f"{label} is not UTF-8") from exc


@dataclass(frozen=True)
class Source1VPKEntry:
    path: str
    crc32: int
    preload: bytes
    archive_index: int
    offset: int
    length: int


class Source1VPK:
    """Valve VPK v1/v2 directory reader with bounds and CRC validation."""

    SIGNATURE = 0x55AA1234

    def __init__(self, directory_path: str | Path) -> None:
        self.path = Path(directory_path)
        self._directory = self.path.read_bytes()
        if len(self._directory) < 12:
            raise Source1FormatError("VPK header is truncated")
        signature, self.version, tree_size = struct.unpack_from(
            "<III", self._directory, 0
        )
        if signature != self.SIGNATURE:
            raise Source1FormatError(
                f"VPK signature is 0x{signature:08x}, expected 0x{self.SIGNATURE:08x}"
            )
        if self.version == 1:
            header_size = 12
        elif self.version == 2:
            if len(self._directory) < 28:
                raise Source1FormatError("VPK v2 header is truncated")
            header_size = 28
        else:
            raise Source1FormatError(f"unsupported VPK version {self.version}")
        _range(self._directory, header_size, tree_size, "VPK directory tree")
        self._data_offset = header_size + tree_size
        self.entries = self._parse_tree(header_size, self._data_offset)

    def _parse_tree(self, start: int, end: int) -> Mapping[str, Source1VPKEntry]:
        data = self._directory
        cursor = start

        def token(label: str) -> str:
            nonlocal cursor
            if cursor >= end:
                raise Source1FormatError(f"VPK tree ended while reading {label}")
            nul = data.find(b"\0", cursor, end)
            if nul < 0:
                raise Source1FormatError(f"VPK {label} is not NUL terminated")
            try:
                value = data[cursor:nul].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise Source1FormatError(f"VPK {label} is not UTF-8") from exc
            cursor = nul + 1
            return value

        entries: dict[str, Source1VPKEntry] = {}
        while True:
            extension = token("extension")
            if not extension:
                break
            while True:
                directory = token("directory")
                if not directory:
                    break
                while True:
                    stem = token("filename")
                    if not stem:
                        break
                    if cursor > end - 18:
                        raise Source1FormatError("VPK entry descriptor is truncated")
                    crc, preload_size, archive, offset, length, terminator = (
                        struct.unpack_from("<IHHIIH", data, cursor)
                    )
                    cursor += 18
                    if terminator != 0xFFFF:
                        raise Source1FormatError(
                            f"VPK entry {stem!r} has terminator 0x{terminator:04x}"
                        )
                    preload = bytes(_range(data, cursor, preload_size, "VPK preload"))
                    cursor += preload_size
                    prefix = "" if directory == " " else directory + "/"
                    path = (prefix + stem + "." + extension).replace("\\", "/")
                    key = path.casefold()
                    if key in entries:
                        raise Source1FormatError(f"duplicate VPK path {path!r}")
                    entries[key] = Source1VPKEntry(
                        path, crc, preload, archive, offset, length
                    )
        if cursor != end:
            raise Source1FormatError(
                f"VPK tree consumed {cursor - start} of {end - start} bytes"
            )
        return entries

    def _archive_path(self, index: int) -> Path:
        name = self.path.name
        if name.endswith("_dir.vpk"):
            base = name[: -len("_dir.vpk")]
        elif name.endswith(".vpk"):
            base = name[:-4]
        else:
            raise Source1FormatError("VPK directory filename must end in .vpk")
        return self.path.with_name(f"{base}_{index:03d}.vpk")

    def read(self, path: str, *, verify_crc: bool = True) -> bytes:
        key = path.replace("\\", "/").lstrip("/").casefold()
        try:
            entry = self.entries[key]
        except KeyError as exc:
            raise FileNotFoundError(f"{path!r} is not present in {self.path}") from exc
        if entry.archive_index == 0x7FFF:
            payload = bytes(
                _range(
                    self._directory,
                    self._data_offset + entry.offset,
                    entry.length,
                    f"VPK inline entry {entry.path}",
                )
            )
        else:
            archive = self._archive_path(entry.archive_index)
            with archive.open("rb") as handle:
                handle.seek(entry.offset)
                payload = handle.read(entry.length)
            if len(payload) != entry.length:
                raise Source1FormatError(
                    f"VPK archive entry {entry.path!r} is truncated in {archive}"
                )
        result = entry.preload + payload
        if verify_crc and (zlib.crc32(result) & 0xFFFFFFFF) != entry.crc32:
            raise Source1FormatError(f"VPK CRC mismatch for {entry.path!r}")
        return result


@dataclass(frozen=True)
class Source1BSPLump:
    index: int
    offset: int
    length: int
    version: int
    four_cc: bytes


@dataclass(frozen=True)
class Source1BSPMesh:
    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    face_texinfo: tuple[int, ...]


class Source1BSP:
    """Parser for uncompressed VBSP v19-v21 world-face geometry."""

    IDENT = b"VBSP"
    HEADER_LUMPS = 64
    LUMP_VERTEXES = 3
    LUMP_FACES = 7
    LUMP_EDGES = 12
    LUMP_SURFEDGES = 13

    def __init__(self, data: bytes, *, label: str = "<memory>") -> None:
        self.data = data
        self.label = label
        if len(data) < 8 + self.HEADER_LUMPS * 16 + 4:
            raise Source1FormatError(f"{label}: BSP header is truncated")
        if data[:4] != self.IDENT:
            raise Source1FormatError(f"{label}: not a VBSP file")
        self.version = struct.unpack_from("<i", data, 4)[0]
        if self.version not in (19, 20, 21):
            raise Source1FormatError(
                f"{label}: unsupported BSP version {self.version}; expected 19-21"
            )
        lumps: list[Source1BSPLump] = []
        for index in range(self.HEADER_LUMPS):
            offset, length, version, four_cc = struct.unpack_from(
                "<iii4s", data, 8 + index * 16
            )
            if length:
                _range(data, offset, length, f"{label}: BSP lump {index}")
            lumps.append(Source1BSPLump(index, offset, length, version, four_cc))
        self.lumps = tuple(lumps)
        self.map_revision = struct.unpack_from("<i", data, 8 + 64 * 16)[0]

    @classmethod
    def from_path(cls, path: str | Path) -> "Source1BSP":
        source = Path(path)
        return cls(source.read_bytes(), label=str(source))

    def lump_bytes(self, index: int) -> bytes:
        if isinstance(index, bool) or not 0 <= index < self.HEADER_LUMPS:
            raise IndexError("BSP lump index must be in [0, 64)")
        lump = self.lumps[index]
        if lump.four_cc != b"\0\0\0\0":
            raise Source1FormatError(
                f"{self.label}: compressed BSP lump {index} is not implemented"
            )
        return bytes(_range(self.data, lump.offset, lump.length, f"BSP lump {index}"))

    def world_mesh(self) -> Source1BSPMesh:
        vertex_data = self.lump_bytes(self.LUMP_VERTEXES)
        edge_data = self.lump_bytes(self.LUMP_EDGES)
        surfedge_data = self.lump_bytes(self.LUMP_SURFEDGES)
        face_data = self.lump_bytes(self.LUMP_FACES)
        if len(vertex_data) % 12 or len(edge_data) % 4 or len(surfedge_data) % 4:
            raise Source1FormatError(f"{self.label}: malformed BSP geometry lump size")
        if len(face_data) % 56:
            raise Source1FormatError(
                f"{self.label}: face lump size is not a multiple of dface_t (56)"
            )
        vertices = tuple(struct.iter_unpack("<fff", vertex_data))
        edges = tuple(struct.iter_unpack("<HH", edge_data))
        surfedges = tuple(v[0] for v in struct.iter_unpack("<i", surfedge_data))
        triangles: list[tuple[int, int, int]] = []
        triangle_texinfo: list[int] = []
        for face_no in range(len(face_data) // 56):
            base = face_no * 56
            firstedge = struct.unpack_from("<i", face_data, base + 4)[0]
            numedges, texinfo = struct.unpack_from("<hh", face_data, base + 8)
            if numedges < 3:
                continue
            if firstedge < 0 or firstedge > len(surfedges) - numedges:
                raise Source1FormatError(
                    f"{self.label}: face {face_no} surfedge range is invalid"
                )
            polygon: list[int] = []
            for signed_edge in surfedges[firstedge : firstedge + numedges]:
                edge_index = abs(signed_edge)
                if edge_index >= len(edges):
                    raise Source1FormatError(
                        f"{self.label}: face {face_no} names missing edge {edge_index}"
                    )
                edge = edges[edge_index]
                vertex = edge[0] if signed_edge >= 0 else edge[1]
                if vertex >= len(vertices):
                    raise Source1FormatError(
                        f"{self.label}: edge {edge_index} names missing vertex {vertex}"
                    )
                polygon.append(vertex)
            for index in range(1, len(polygon) - 1):
                triangles.append((polygon[0], polygon[index], polygon[index + 1]))
                triangle_texinfo.append(texinfo)
        return Source1BSPMesh(vertices, tuple(triangles), tuple(triangle_texinfo))


@dataclass(frozen=True)
class Source1MDLBone:
    name: str
    parent: int
    flags: int


@dataclass(frozen=True)
class Source1MDLSequence:
    label: str
    activity: str
    flags: int
    activity_id: int
    blend_count: int


@dataclass(frozen=True)
class Source1MDLMesh:
    material: int
    vertex_count: int
    vertex_offset: int
    flex_count: int


@dataclass(frozen=True)
class Source1MDLModel:
    name: str
    vertex_count: int
    vertex_base: int
    meshes: tuple[Source1MDLMesh, ...]


@dataclass(frozen=True)
class Source1MDLBodyPart:
    name: str
    models: tuple[Source1MDLModel, ...]


@dataclass(frozen=True)
class Source1MDL:
    name: str
    version: int
    checksum: int
    flags: int
    bones: tuple[Source1MDLBone, ...]
    sequences: tuple[Source1MDLSequence, ...]
    body_parts: tuple[Source1MDLBodyPart, ...]

    @classmethod
    def parse(cls, data: bytes, *, label: str = "<memory>") -> "Source1MDL":
        if len(data) < 244 or data[:4] != b"IDST":
            raise Source1FormatError(f"{label}: not a complete Source 1 MDL header")
        version, checksum = struct.unpack_from("<ii", data, 4)
        # The public Source SDK layout below is STUDIO_VERSION 48; CS:GO also
        # shipped version 49 with the same prefix used here.  Older versions
        # differ enough that interpreting them through this structure would
        # be guesswork.
        if version not in (48, 49):
            raise Source1FormatError(
                f"{label}: unsupported MDL version {version}; expected 48 or 49"
            )
        declared_length = struct.unpack_from("<i", data, 76)[0]
        if declared_length <= 0 or declared_length > len(data):
            raise Source1FormatError(
                f"{label}: declared MDL length {declared_length} exceeds file"
            )
        name = _cstr(data, 12, 64, f"{label}: model name")
        flags = struct.unpack_from("<i", data, 152)[0]
        bone_count, bone_offset = struct.unpack_from("<ii", data, 156)
        seq_count, seq_offset = struct.unpack_from("<ii", data, 188)
        body_part_count, body_part_offset = struct.unpack_from("<ii", data, 232)
        if not 0 <= bone_count <= 256:
            raise Source1FormatError(f"{label}: invalid bone count {bone_count}")
        if not 0 <= seq_count <= 16384:
            raise Source1FormatError(f"{label}: invalid sequence count {seq_count}")
        if not 0 <= body_part_count <= 256:
            raise Source1FormatError(
                f"{label}: invalid body-part count {body_part_count}"
            )
        bones: list[Source1MDLBone] = []
        bone_size = 216
        _range(data, bone_offset, bone_count * bone_size, f"{label}: bones")
        for index in range(bone_count):
            base = bone_offset + index * bone_size
            relative_name, parent = struct.unpack_from("<ii", data, base)
            bone_name = _relative_cstr(data, base, relative_name, label, "bone")
            bone_flags = struct.unpack_from("<i", data, base + 160)[0]
            if parent < -1 or parent >= bone_count:
                raise Source1FormatError(
                    f"{label}: bone {index} has invalid parent {parent}"
                )
            bones.append(Source1MDLBone(bone_name, parent, bone_flags))
        sequences: list[Source1MDLSequence] = []
        seq_size = 212
        _range(data, seq_offset, seq_count * seq_size, f"{label}: sequences")
        for index in range(seq_count):
            base = seq_offset + index * seq_size
            label_offset, activity_offset, seq_flags, activity_id = struct.unpack_from(
                "<iiii", data, base + 4
            )
            blend_count = struct.unpack_from("<i", data, base + 56)[0]
            sequences.append(
                Source1MDLSequence(
                    _relative_cstr(data, base, label_offset, label, "sequence label"),
                    _relative_cstr(
                        data, base, activity_offset, label, "sequence activity"
                    ),
                    seq_flags,
                    activity_id,
                    blend_count,
                )
            )
        body_parts: list[Source1MDLBodyPart] = []
        body_part_size = 16
        model_size = 148
        mesh_size = 116
        _range(
            data,
            body_part_offset,
            body_part_count * body_part_size,
            f"{label}: body parts",
        )
        for body_part_no in range(body_part_count):
            base = body_part_offset + body_part_no * body_part_size
            name_offset, model_count, _body_base, model_offset = struct.unpack_from(
                "<iiii", data, base
            )
            if not 0 <= model_count <= 256:
                raise Source1FormatError(
                    f"{label}: body part {body_part_no} has invalid model count "
                    f"{model_count}"
                )
            models_base = base + model_offset
            _range(
                data,
                models_base,
                model_count * model_size,
                f"{label}: body part {body_part_no} models",
            )
            models: list[Source1MDLModel] = []
            for model_no in range(model_count):
                model_base = models_base + model_no * model_size
                model_name = _fixed_cstr(
                    data, model_base, 64, f"{label}: model {body_part_no}/{model_no} name"
                )
                mesh_count, mesh_offset, vertex_count, vertex_index = (
                    struct.unpack_from("<iiii", data, model_base + 72)
                )
                if not 0 <= mesh_count <= 4096 or not 0 <= vertex_count <= 65536:
                    raise Source1FormatError(
                        f"{label}: model {body_part_no}/{model_no} has invalid "
                        "mesh or vertex count"
                    )
                if vertex_index < 0 or vertex_index % 48:
                    raise Source1FormatError(
                        f"{label}: model {body_part_no}/{model_no} vertexindex "
                        f"{vertex_index} is not aligned to mstudiovertex_t (48)"
                    )
                meshes_base = model_base + mesh_offset
                _range(
                    data,
                    meshes_base,
                    mesh_count * mesh_size,
                    f"{label}: model {body_part_no}/{model_no} meshes",
                )
                meshes: list[Source1MDLMesh] = []
                for mesh_no in range(mesh_count):
                    mesh_base = meshes_base + mesh_no * mesh_size
                    material = struct.unpack_from("<i", data, mesh_base)[0]
                    mesh_vertices, vertex_offset, flex_count = struct.unpack_from(
                        "<iii", data, mesh_base + 8
                    )
                    if (
                        mesh_vertices < 0
                        or vertex_offset < 0
                        or flex_count < 0
                        or vertex_offset > vertex_count - mesh_vertices
                    ):
                        raise Source1FormatError(
                            f"{label}: mesh {body_part_no}/{model_no}/{mesh_no} "
                            "vertex/flex range is invalid"
                        )
                    meshes.append(
                        Source1MDLMesh(
                            material, mesh_vertices, vertex_offset, flex_count
                        )
                    )
                models.append(
                    Source1MDLModel(
                        model_name,
                        vertex_count,
                        vertex_index // 48,
                        tuple(meshes),
                    )
                )
            body_parts.append(
                Source1MDLBodyPart(
                    _relative_cstr(
                        data, base, name_offset, label, "body-part name"
                    ),
                    tuple(models),
                )
            )
        return cls(
            name,
            version,
            checksum,
            flags,
            tuple(bones),
            tuple(sequences),
            tuple(body_parts),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "Source1MDL":
        source = Path(path)
        return cls.parse(source.read_bytes(), label=str(source))

    def validate_companions(self, mdl_path: str | Path) -> Mapping[str, Path]:
        """Require VVD and DX90.VTX siblings and match their MDL checksums."""

        mdl = Path(mdl_path)
        stem = mdl.with_suffix("")
        vvd = stem.with_suffix(".vvd")
        vtx = stem.with_suffix(".dx90.vtx")
        missing = [str(path) for path in (vvd, vtx) if not path.is_file()]
        if missing:
            raise Source1FormatError(
                "model companion files are required: " + ", ".join(missing)
            )
        vvd_data = vvd.read_bytes()
        if len(vvd_data) < 12 or vvd_data[:4] != b"IDSV":
            raise Source1FormatError(f"{vvd}: invalid VVD header")
        vvd_checksum = struct.unpack_from("<i", vvd_data, 8)[0]
        vtx_data = vtx.read_bytes()
        if len(vtx_data) < 20:
            raise Source1FormatError(f"{vtx}: truncated VTX header")
        vtx_checksum = struct.unpack_from("<i", vtx_data, 16)[0]
        if vvd_checksum != self.checksum or vtx_checksum != self.checksum:
            raise Source1FormatError(
                f"model checksum {self.checksum} does not match VVD/VTX "
                f"({vvd_checksum}/{vtx_checksum})"
            )
        return {"mdl": mdl, "vvd": vvd, "vtx": vtx}


@dataclass(frozen=True)
class Source1VVDVertex:
    weights: tuple[float, ...]
    bones: tuple[int, ...]
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    texcoord: tuple[float, float]


@dataclass(frozen=True)
class Source1VVD:
    checksum: int
    root_lod: int
    lod_vertex_counts: tuple[int, ...]
    vertices: tuple[Source1VVDVertex, ...]

    @classmethod
    def parse(
        cls,
        data: bytes,
        *,
        expected_checksum: int | None = None,
        root_lod: int = 0,
        label: str = "<memory>.vvd",
    ) -> "Source1VVD":
        header_size = 64
        if len(data) < header_size or data[:4] != b"IDSV":
            raise Source1FormatError(f"{label}: invalid VVD header")
        version, checksum, num_lods = struct.unpack_from("<iii", data, 4)
        if version != 4:
            raise Source1FormatError(
                f"{label}: unsupported VVD version {version}; expected 4"
            )
        if expected_checksum is not None and checksum != expected_checksum:
            raise Source1FormatError(
                f"{label}: checksum {checksum} does not match MDL {expected_checksum}"
            )
        if not 1 <= num_lods <= 8:
            raise Source1FormatError(f"{label}: invalid VVD LOD count {num_lods}")
        if isinstance(root_lod, bool) or not 0 <= root_lod < num_lods:
            raise Source1FormatError(
                f"{label}: root LOD {root_lod!r} is outside [0, {num_lods})"
            )
        lod_counts = struct.unpack_from("<8i", data, 16)[:num_lods]
        if any(count < 0 or count > 4_194_304 for count in lod_counts):
            raise Source1FormatError(f"{label}: invalid VVD LOD vertex count")
        num_fixups, fixup_offset, vertex_offset, tangent_offset = struct.unpack_from(
            "<iiii", data, 48
        )
        if num_fixups < 0 or num_fixups > 4_194_304:
            raise Source1FormatError(f"{label}: invalid VVD fixup count {num_fixups}")
        _range(data, fixup_offset, num_fixups * 12, f"{label}: fixup table")
        stored_count = lod_counts[0]
        _range(
            data,
            vertex_offset,
            stored_count * 48,
            f"{label}: mstudiovertex_t array",
        )
        if tangent_offset:
            _range(data, tangent_offset, stored_count * 16, f"{label}: tangents")
        all_vertices = tuple(
            _parse_vvd_vertex(data, vertex_offset + index * 48, label, index)
            for index in range(stored_count)
        )
        if num_fixups:
            selected: list[Source1VVDVertex] = []
            for fixup_no in range(num_fixups):
                lod, source, count = struct.unpack_from(
                    "<iii", data, fixup_offset + fixup_no * 12
                )
                if lod < 0 or source < 0 or count < 0 or source > stored_count - count:
                    raise Source1FormatError(
                        f"{label}: fixup {fixup_no} has an invalid source range"
                    )
                if lod >= root_lod:
                    selected.extend(all_vertices[source : source + count])
            vertices = tuple(selected)
        else:
            vertices = all_vertices[: lod_counts[root_lod]]
        if len(vertices) != lod_counts[root_lod]:
            raise Source1FormatError(
                f"{label}: fixups produced {len(vertices)} vertices; LOD "
                f"{root_lod} declares {lod_counts[root_lod]}"
            )
        return cls(checksum, root_lod, tuple(lod_counts), vertices)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        expected_checksum: int | None = None,
        root_lod: int = 0,
    ) -> "Source1VVD":
        source = Path(path)
        return cls.parse(
            source.read_bytes(),
            expected_checksum=expected_checksum,
            root_lod=root_lod,
            label=str(source),
        )


def _parse_vvd_vertex(
    data: bytes, offset: int, label: str, index: int
) -> Source1VVDVertex:
    raw_weights = struct.unpack_from("<fff", data, offset)
    raw_bones = struct.unpack_from("<bbb", data, offset + 12)
    bone_count = data[offset + 15]
    if not 1 <= bone_count <= 3:
        raise Source1FormatError(
            f"{label}: vertex {index} has invalid bone count {bone_count}"
        )
    weights = tuple(float(value) for value in raw_weights[:bone_count])
    bones = tuple(int(value) for value in raw_bones[:bone_count])
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise Source1FormatError(f"{label}: vertex {index} has invalid bone weights")
    if abs(sum(weights) - 1.0) > 1e-4:
        raise Source1FormatError(
            f"{label}: vertex {index} bone weights sum to {sum(weights)!r}, not 1"
        )
    if any(bone < 0 for bone in bones):
        raise Source1FormatError(f"{label}: vertex {index} has a negative bone id")
    position = struct.unpack_from("<fff", data, offset + 16)
    normal = struct.unpack_from("<fff", data, offset + 28)
    texcoord = struct.unpack_from("<ff", data, offset + 40)
    if not all(math.isfinite(value) for value in (*position, *normal, *texcoord)):
        raise Source1FormatError(f"{label}: vertex {index} contains non-finite data")
    return Source1VVDVertex(weights, bones, position, normal, texcoord)


@dataclass(frozen=True)
class Source1VTXMesh:
    body_part: int
    model: int
    mesh: int
    material: int
    vertex_indices: tuple[int, ...]
    triangles: tuple[tuple[int, int, int], ...]


class Source1VTX:
    """Exact OPTIMIZED_MODEL_FILE_VERSION 7 DX90 topology reader."""

    def __init__(
        self,
        data: bytes,
        *,
        expected_checksum: int | None = None,
        label: str = "<memory>.dx90.vtx",
    ) -> None:
        self.data = data
        self.label = label
        if len(data) < 36:
            raise Source1FormatError(f"{label}: VTX header is truncated")
        (
            self.version,
            _cache_size,
            _max_bones_strip,
            _max_bones_tri,
            max_bones_vertex,
            self.checksum,
            self.num_lods,
            _replacement_offset,
            self.num_body_parts,
            self.body_part_offset,
        ) = struct.unpack_from("<iiHHiiiiii", data, 0)
        if self.version != 7:
            raise Source1FormatError(
                f"{label}: unsupported VTX version {self.version}; expected 7"
            )
        if max_bones_vertex != 3:
            raise Source1FormatError(
                f"{label}: maxBonesPerVert is {max_bones_vertex}, expected 3"
            )
        if expected_checksum is not None and self.checksum != expected_checksum:
            raise Source1FormatError(
                f"{label}: checksum {self.checksum} does not match MDL "
                f"{expected_checksum}"
            )
        if not 1 <= self.num_lods <= 8 or not 0 <= self.num_body_parts <= 256:
            raise Source1FormatError(f"{label}: invalid VTX LOD/body-part counts")
        _range(
            data,
            self.body_part_offset,
            self.num_body_parts * 8,
            f"{label}: body-part headers",
        )

    @classmethod
    def from_path(
        cls, path: str | Path, *, expected_checksum: int | None = None
    ) -> "Source1VTX":
        source = Path(path)
        return cls(
            source.read_bytes(), expected_checksum=expected_checksum, label=str(source)
        )

    def assemble_lod0(
        self,
        mdl: Source1MDL,
        vvd: Source1VVD,
        *,
        bodygroups: Sequence[int] | None = None,
    ) -> tuple[Source1VTXMesh, ...]:
        if vvd.root_lod != 0:
            raise Source1FormatError(
                f"{self.label}: only exact LOD0 assembly is implemented"
            )
        if self.checksum != mdl.checksum or vvd.checksum != mdl.checksum:
            raise Source1FormatError("MDL/VVD/VTX checksums do not agree")
        if self.num_body_parts != len(mdl.body_parts):
            raise Source1FormatError(
                f"{self.label}: VTX has {self.num_body_parts} body parts but MDL "
                f"has {len(mdl.body_parts)}"
            )
        selected = _bodygroup_selection(mdl, bodygroups)
        output: list[Source1VTXMesh] = []
        for body_part_no, model_no in enumerate(selected):
            mdl_body = mdl.body_parts[body_part_no]
            mdl_model = mdl_body.models[model_no]
            body_base = self.body_part_offset + body_part_no * 8
            model_count, model_offset = struct.unpack_from("<ii", self.data, body_base)
            if model_count != len(mdl_body.models):
                raise Source1FormatError(
                    f"{self.label}: body part {body_part_no} model count differs "
                    "from MDL"
                )
            models_base = body_base + model_offset
            _range(
                self.data,
                models_base,
                model_count * 8,
                f"{self.label}: body part {body_part_no} models",
            )
            model_base = models_base + model_no * 8
            lod_count, lod_offset = struct.unpack_from("<ii", self.data, model_base)
            if lod_count < 1 or lod_count > self.num_lods:
                raise Source1FormatError(
                    f"{self.label}: model {body_part_no}/{model_no} has invalid LOD count"
                )
            lod_base = model_base + lod_offset
            _range(self.data, lod_base, lod_count * 12, f"{self.label}: LOD headers")
            mesh_count, mesh_offset = struct.unpack_from("<ii", self.data, lod_base)
            if mesh_count != len(mdl_model.meshes):
                raise Source1FormatError(
                    f"{self.label}: model {body_part_no}/{model_no} LOD0 has "
                    f"{mesh_count} meshes but MDL has {len(mdl_model.meshes)}"
                )
            meshes_base = lod_base + mesh_offset
            _range(self.data, meshes_base, mesh_count * 9, f"{self.label}: meshes")
            for mesh_no, mdl_mesh in enumerate(mdl_model.meshes):
                if mdl_mesh.flex_count:
                    raise Source1FormatError(
                        f"{self.label}: mesh {body_part_no}/{model_no}/{mesh_no} "
                        "has flex vertices; evaluated flex geometry is not implemented"
                    )
                mesh_base = meshes_base + mesh_no * 9
                group_count, group_offset = struct.unpack_from(
                    "<ii", self.data, mesh_base
                )
                groups_base = mesh_base + group_offset
                _range(
                    self.data,
                    groups_base,
                    group_count * 25,
                    f"{self.label}: strip groups",
                )
                all_indices: list[int] = []
                triangles: list[tuple[int, int, int]] = []
                for group_no in range(group_count):
                    group_base = groups_base + group_no * 25
                    group_vertices, group_triangles = self._strip_group(
                        group_base,
                        mdl_model,
                        mdl_mesh,
                        body_part_no,
                        model_no,
                        mesh_no,
                        len(vvd.vertices),
                    )
                    vertex_base = len(all_indices)
                    all_indices.extend(group_vertices)
                    triangles.extend(
                        tuple(vertex_base + index for index in triangle)
                        for triangle in group_triangles
                    )
                output.append(
                    Source1VTXMesh(
                        body_part_no,
                        model_no,
                        mesh_no,
                        mdl_mesh.material,
                        tuple(all_indices),
                        tuple(triangles),
                    )
                )
        return tuple(output)

    def _strip_group(
        self,
        base: int,
        model: Source1MDLModel,
        mesh: Source1MDLMesh,
        body_part_no: int,
        model_no: int,
        mesh_no: int,
        vvd_count: int,
    ) -> tuple[list[int], list[tuple[int, int, int]]]:
        (
            vertex_count,
            vertex_offset,
            index_count,
            index_offset,
            strip_count,
            strip_offset,
            group_flags,
        ) = struct.unpack_from("<iiiiiiB", self.data, base)
        label = (
            f"{self.label}: strip group {body_part_no}/{model_no}/{mesh_no}"
        )
        if min(vertex_count, index_count, strip_count) < 0:
            raise Source1FormatError(f"{label} has negative counts")
        if group_flags & 0x01:
            raise Source1FormatError(
                f"{label} is flexed; evaluated flex geometry is not implemented"
            )
        vertices_base = base + vertex_offset
        indices_base = base + index_offset
        strips_base = base + strip_offset
        _range(self.data, vertices_base, vertex_count * 9, f"{label} vertices")
        _range(self.data, indices_base, index_count * 2, f"{label} indices")
        _range(self.data, strips_base, strip_count * 27, f"{label} strips")
        mapped_vertices: list[int] = []
        for vertex_no in range(vertex_count):
            vertex_base = vertices_base + vertex_no * 9
            bone_count = self.data[vertex_base + 3]
            original = struct.unpack_from("<H", self.data, vertex_base + 4)[0]
            if not 1 <= bone_count <= 3 or original >= mesh.vertex_count:
                raise Source1FormatError(f"{label} vertex {vertex_no} is invalid")
            global_vertex = model.vertex_base + mesh.vertex_offset + original
            if global_vertex >= vvd_count:
                raise Source1FormatError(
                    f"{label} vertex {vertex_no} maps beyond VVD vertex array"
                )
            mapped_vertices.append(global_vertex)
        group_indices = tuple(
            value[0]
            for value in struct.iter_unpack(
                "<H", bytes(_range(self.data, indices_base, index_count * 2, label))
            )
        )
        if any(index >= vertex_count for index in group_indices):
            raise Source1FormatError(f"{label} contains an out-of-range index")
        triangles: list[tuple[int, int, int]] = []
        for strip_no in range(strip_count):
            strip_base = strips_base + strip_no * 27
            (
                strip_index_count,
                strip_index_offset,
                strip_vertex_count,
                strip_vertex_offset,
                _strip_bone_count,
                flags,
                change_count,
                change_offset,
            ) = struct.unpack_from("<iiiihBii", self.data, strip_base)
            if (
                strip_index_count < 0
                or strip_index_offset < 0
                or strip_index_offset > index_count - strip_index_count
                or strip_vertex_count < 0
                or strip_vertex_offset < 0
                or strip_vertex_offset > vertex_count - strip_vertex_count
                or change_count < 0
            ):
                raise Source1FormatError(f"{label} strip {strip_no} has invalid ranges")
            _range(
                self.data,
                strip_base + change_offset,
                change_count * 8,
                f"{label} bone-state changes",
            )
            indices = group_indices[
                strip_index_offset : strip_index_offset + strip_index_count
            ]
            if flags == 0x01:
                if len(indices) % 3:
                    raise Source1FormatError(
                        f"{label} triangle-list strip index count is not divisible by 3"
                    )
                candidates = (
                    tuple(indices[index : index + 3])
                    for index in range(0, len(indices), 3)
                )
            elif flags == 0x02:
                candidates = []
                for index in range(len(indices) - 2):
                    triangle = (indices[index], indices[index + 1], indices[index + 2])
                    if index & 1:
                        triangle = (triangle[1], triangle[0], triangle[2])
                    candidates.append(triangle)
            else:
                raise Source1FormatError(
                    f"{label} strip {strip_no} has unsupported flags 0x{flags:02x}"
                )
            triangles.extend(
                triangle for triangle in candidates if len(set(triangle)) == 3
            )
        return mapped_vertices, triangles


def _bodygroup_selection(
    mdl: Source1MDL, bodygroups: Sequence[int] | None
) -> tuple[int, ...]:
    if bodygroups is None:
        if any(len(body_part.models) != 1 for body_part in mdl.body_parts):
            raise Source1FormatError(
                "bodygroups are required because the MDL contains selectable models"
            )
        return (0,) * len(mdl.body_parts)
    if isinstance(bodygroups, (str, bytes)) or len(bodygroups) != len(mdl.body_parts):
        raise Source1FormatError(
            f"bodygroups must contain {len(mdl.body_parts)} model indices"
        )
    selected: list[int] = []
    for index, (value, body_part) in enumerate(zip(bodygroups, mdl.body_parts)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(body_part.models):
            raise Source1FormatError(
                f"bodygroups[{index}]={value!r} is outside this body part"
            )
        selected.append(value)
    return tuple(selected)


def _relative_cstr(
    data: bytes, base: int, relative: int, label: str, field: str
) -> str:
    if relative == 0:
        return ""
    absolute = base + relative
    if absolute < 0 or absolute >= len(data):
        raise Source1FormatError(f"{label}: {field} offset is out of range")
    end = data.find(b"\0", absolute)
    if end < 0:
        raise Source1FormatError(f"{label}: {field} is not NUL terminated")
    try:
        return data[absolute:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Source1FormatError(f"{label}: {field} is not UTF-8") from exc


_VMT_TOKEN = re.compile(
    r'\s*(?:(//[^\n]*(?:\n|$))|("(?:\\.|[^"\\])*")|([{}])|([^\s{}"]+))',
    re.MULTILINE,
)


def parse_source1_vmt(text: str) -> Mapping[str, Any]:
    """Parse KeyValues1 material text, preserving duplicate keys as lists."""

    tokens: list[str] = []
    cursor = 0
    while cursor < len(text):
        match = _VMT_TOKEN.match(text, cursor)
        if not match:
            if text[cursor:].strip():
                raise Source1FormatError(f"VMT token error at character {cursor}")
            break
        cursor = match.end()
        comment, quoted, brace, bare = match.groups()
        if comment is not None:
            continue
        if quoted is not None:
            try:
                tokens.append(bytes(quoted[1:-1], "utf-8").decode("unicode_escape"))
            except UnicodeDecodeError as exc:
                raise Source1FormatError("invalid VMT quoted escape") from exc
        elif brace is not None:
            tokens.append(brace)
        elif bare is not None:
            tokens.append(bare)
    position = 0

    def object_value() -> dict[str, Any]:
        nonlocal position
        result: dict[str, Any] = {}
        while position < len(tokens) and tokens[position] != "}":
            key = tokens[position]
            position += 1
            if key == "{":
                raise Source1FormatError("VMT object has a value without a key")
            if position >= len(tokens):
                raise Source1FormatError(f"VMT key {key!r} has no value")
            if tokens[position] == "{":
                position += 1
                value: Any = object_value()
                if position >= len(tokens) or tokens[position] != "}":
                    raise Source1FormatError(f"VMT object {key!r} is not closed")
                position += 1
            elif tokens[position] == "}":
                raise Source1FormatError(f"VMT key {key!r} has no value")
            else:
                value = tokens[position]
                position += 1
            folded = key.casefold()
            if folded in result:
                old = result[folded]
                result[folded] = old + [value] if isinstance(old, list) else [old, value]
            else:
                result[folded] = value
        return result

    document = object_value()
    if position != len(tokens):
        raise Source1FormatError("VMT has an unmatched closing brace")
    if len(document) != 1:
        raise Source1FormatError("VMT must contain exactly one root shader")
    return document


__all__ = [
    "Source1BSP",
    "Source1BSPMesh",
    "Source1FormatError",
    "Source1MDL",
    "Source1MDLBone",
    "Source1MDLBodyPart",
    "Source1MDLMesh",
    "Source1MDLModel",
    "Source1MDLSequence",
    "Source1VTX",
    "Source1VTXMesh",
    "Source1VVD",
    "Source1VVDVertex",
    "Source1VPK",
    "Source1VPKEntry",
    "parse_source1_vmt",
]
