from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from counter_strike_render.source1_formats import (
    Source1BSP,
    Source1FormatError,
    Source1MDL,
    Source1VPK,
    Source1VTX,
    Source1VVD,
    parse_source1_vmt,
)


def _fixture_bsp() -> bytes:
    header_size = 8 + 64 * 16 + 4
    vertices = struct.pack(
        "<fffffffff",
        4.0, -1.0, -1.0,
        4.0, 1.0, -1.0,
        4.0, 0.0, 1.0,
    )
    edges = struct.pack("<HHHHHH", 0, 1, 1, 2, 2, 0)
    surfedges = struct.pack("<iii", 0, 1, 2)
    face = bytearray(56)
    struct.pack_into("<i", face, 4, 0)
    struct.pack_into("<hh", face, 8, 3, 7)
    chunks = {3: vertices, 7: bytes(face), 12: edges, 13: surfedges}
    result = bytearray(header_size + sum(len(item) for item in chunks.values()))
    result[:4] = b"VBSP"
    struct.pack_into("<i", result, 4, 20)
    cursor = header_size
    for index, payload in chunks.items():
        struct.pack_into("<iii4s", result, 8 + index * 16,
                         cursor, len(payload), 0, b"\0\0\0\0")
        result[cursor:cursor + len(payload)] = payload
        cursor += len(payload)
    struct.pack_into("<i", result, 8 + 64 * 16, 17)
    return bytes(result)


def _fixture_vpk(path: str, payload: bytes) -> bytes:
    directory, filename = path.rsplit("/", 1)
    stem, extension = filename.rsplit(".", 1)
    tree_prefix = (
        extension.encode() + b"\0" + directory.encode() + b"\0" +
        stem.encode() + b"\0"
    )
    descriptor = struct.pack(
        "<IHHIIH", zlib.crc32(payload) & 0xFFFFFFFF,
        len(payload), 0x7FFF, 0, 0, 0xFFFF,
    )
    tree = tree_prefix + descriptor + payload + b"\0\0\0"
    return struct.pack("<III", 0x55AA1234, 1, len(tree)) + tree


def _fixture_mdl(version=48) -> bytes:
    bone_offset = 244
    sequence_offset = bone_offset + 216
    string_offset = sequence_offset + 212
    bone_name = b"root\0"
    sequence_name = b"idle\0"
    activity_name = b"ACT_IDLE\0"
    size = string_offset + len(bone_name) + len(sequence_name) + len(activity_name)
    data = bytearray(size)
    data[:4] = b"IDST"
    struct.pack_into("<ii", data, 4, version, 0x12345678)
    data[12:12 + len(b"fixture.mdl\0")] = b"fixture.mdl\0"
    struct.pack_into("<i", data, 76, size)
    struct.pack_into("<i", data, 152, 7)
    struct.pack_into("<ii", data, 156, 1, bone_offset)
    struct.pack_into("<ii", data, 188, 1, sequence_offset)
    struct.pack_into("<ii", data, bone_offset,
                     string_offset - bone_offset, -1)
    struct.pack_into("<i", data, bone_offset + 160, 0x100)
    sequence_string = string_offset + len(bone_name)
    activity_string = sequence_string + len(sequence_name)
    struct.pack_into(
        "<iiii", data, sequence_offset + 4,
        sequence_string - sequence_offset,
        activity_string - sequence_offset,
        1,
        5,
    )
    struct.pack_into("<i", data, sequence_offset + 56, 1)
    data[string_offset:string_offset + len(bone_name)] = bone_name
    data[sequence_string:sequence_string + len(sequence_name)] = sequence_name
    data[activity_string:activity_string + len(activity_name)] = activity_name
    return bytes(data)


def _fixture_model_triplet(checksum=0x12345678):
    bone_offset = 244
    body_part_offset = bone_offset + 216
    model_offset = body_part_offset + 16
    mesh_offset = model_offset + 148
    string_offset = mesh_offset + 116
    strings = b"root\0body\0"
    mdl = bytearray(string_offset + len(strings))
    mdl[:4] = b"IDST"
    struct.pack_into("<ii", mdl, 4, 48, checksum)
    mdl[12:12 + len(b"fixture.mdl\0")] = b"fixture.mdl\0"
    struct.pack_into("<i", mdl, 76, len(mdl))
    struct.pack_into("<ii", mdl, 156, 1, bone_offset)
    struct.pack_into("<ii", mdl, 188, 0, 0)
    struct.pack_into("<ii", mdl, 232, 1, body_part_offset)
    struct.pack_into("<ii", mdl, bone_offset,
                     string_offset - bone_offset, -1)
    struct.pack_into("<i", mdl, bone_offset + 160, 0x100)
    body_name_offset = string_offset + len(b"root\0")
    struct.pack_into(
        "<iiii", mdl, body_part_offset,
        body_name_offset - body_part_offset, 1, 1, 16,
    )
    mdl[model_offset:model_offset + len(b"triangle\0")] = b"triangle\0"
    struct.pack_into("<iiii", mdl, model_offset + 72, 1, 148, 3, 0)
    struct.pack_into("<i", mdl, mesh_offset, 7)
    struct.pack_into("<iii", mdl, mesh_offset + 8, 3, 0, 0)
    mdl[string_offset:string_offset + len(strings)] = strings

    vvd = bytearray(64 + 3 * 48)
    vvd[:4] = b"IDSV"
    struct.pack_into("<iii", vvd, 4, 4, checksum, 1)
    struct.pack_into("<8i", vvd, 16, 3, 0, 0, 0, 0, 0, 0, 0)
    struct.pack_into("<iiii", vvd, 48, 0, 0, 64, 0)
    positions = ((0.0, -1.0, -1.0), (0.0, 1.0, -1.0), (0.0, 0.0, 1.0))
    for index, position in enumerate(positions):
        base = 64 + index * 48
        struct.pack_into("<fff", vvd, base, 1.0, 0.0, 0.0)
        struct.pack_into("<bbbB", vvd, base + 12, 0, 0, 0, 1)
        struct.pack_into("<fff", vvd, base + 16, *position)
        struct.pack_into("<fff", vvd, base + 28, 1.0, 0.0, 0.0)
        struct.pack_into("<ff", vvd, base + 40, 0.0, 0.0)

    # optimize.h structures are byte-packed.  The offsets below exercise the
    # exact 36/8/8/12/9/25/9/27 byte layout chain.
    vtx = bytearray(158)
    struct.pack_into(
        "<iiHHiiiiii", vtx, 0,
        7, 32, 3, 9, 3, checksum, 1, 0, 1, 36,
    )
    struct.pack_into("<ii", vtx, 36, 1, 8)
    struct.pack_into("<ii", vtx, 44, 1, 8)
    struct.pack_into("<iif", vtx, 52, 1, 12, 0.0)
    struct.pack_into("<iiB", vtx, 64, 1, 9, 0)
    struct.pack_into("<iiiiiiB", vtx, 73, 3, 25, 3, 52, 1, 58, 0x02)
    for index in range(3):
        struct.pack_into(
            "<BBBBHbbb", vtx, 98 + index * 9,
            0, 1, 2, 1, index, 0, 0, 0,
        )
    struct.pack_into("<HHH", vtx, 125, 0, 1, 2)
    struct.pack_into("<iiiihBii", vtx, 131, 3, 0, 3, 0, 1, 0x01, 0, 0)
    return bytes(mdl), bytes(vvd), bytes(vtx)


class Source1FormatTests(unittest.TestCase):
    def test_bsp_world_faces_are_triangulated_from_real_lump_layout(self):
        bsp = Source1BSP(_fixture_bsp(), label="fixture.bsp")
        mesh = bsp.world_mesh()
        self.assertEqual(bsp.version, 20)
        self.assertEqual(bsp.map_revision, 17)
        self.assertEqual(mesh.triangles, ((0, 1, 2),))
        self.assertEqual(mesh.face_texinfo, (7,))

    def test_bsp_rejects_unknown_version(self):
        data = bytearray(_fixture_bsp())
        struct.pack_into("<i", data, 4, 99)
        with self.assertRaisesRegex(Source1FormatError, "unsupported BSP version"):
            Source1BSP(bytes(data))

    def test_vpk_reads_preload_entry_and_checks_crc(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pak01_dir.vpk"
            path.write_bytes(_fixture_vpk("materials/test.vmt", b"hello"))
            archive = Source1VPK(path)
            self.assertEqual(archive.read("MATERIALS\\TEST.VMT"), b"hello")

            corrupt = bytearray(path.read_bytes())
            corrupt[-4] ^= 0x01
            path.write_bytes(corrupt)
            with self.assertRaisesRegex(Source1FormatError, "CRC mismatch"):
                Source1VPK(path).read("materials/test.vmt")

    def test_vmt_parser_handles_comments_nesting_and_duplicate_keys(self):
        parsed = parse_source1_vmt('''
            // fixture material
            "VertexLitGeneric" {
                "$basetexture" "models/a"
                "$basetexture" "models/b"
                "Proxies" { "Sine" { "resultVar" "$alpha" } }
            }
        ''')
        material = parsed["vertexlitgeneric"]
        self.assertEqual(material["$basetexture"], ["models/a", "models/b"])
        self.assertEqual(
            material["proxies"]["sine"]["resultvar"], "$alpha"
        )

    def test_mdl_reads_sdk48_bone_and_sequence_prefixes(self):
        mdl = Source1MDL.parse(_fixture_mdl())
        self.assertEqual(mdl.name, "fixture.mdl")
        self.assertEqual(mdl.version, 48)
        self.assertEqual(mdl.bones[0].name, "root")
        self.assertEqual(mdl.bones[0].parent, -1)
        self.assertEqual(mdl.sequences[0].label, "idle")
        self.assertEqual(mdl.sequences[0].activity, "ACT_IDLE")

    def test_mdl_model_name_accepts_full_fixed_width_buffer(self):
        mdl, _vvd, _vtx = _fixture_model_triplet()
        data = bytearray(mdl)
        model_offset = 244 + 216 + 16
        data[model_offset:model_offset + 64] = b"x" * 64
        parsed = Source1MDL.parse(bytes(data), label="full-model-name.mdl")
        self.assertEqual(parsed.body_parts[0].models[0].name, "x" * 64)

    def test_mdl_refuses_layout_versions_it_does_not_implement(self):
        with self.assertRaisesRegex(Source1FormatError, "expected 48 or 49"):
            Source1MDL.parse(_fixture_mdl(version=47))

    def test_vvd_and_dx90_vtx_assemble_exact_lod0_triangle(self):
        mdl_data, vvd_data, vtx_data = _fixture_model_triplet()
        mdl = Source1MDL.parse(mdl_data)
        vvd = Source1VVD.parse(vvd_data, expected_checksum=mdl.checksum)
        topology = Source1VTX(
            vtx_data, expected_checksum=mdl.checksum
        ).assemble_lod0(mdl, vvd, bodygroups=[0])

        self.assertEqual(len(vvd.vertices), 3)
        self.assertEqual(vvd.vertices[0].weights, (1.0,))
        self.assertEqual(vvd.vertices[0].bones, (0,))
        self.assertEqual(topology[0].vertex_indices, (0, 1, 2))
        self.assertEqual(topology[0].triangles, ((0, 1, 2),))
        self.assertEqual(topology[0].material, 7)

    def test_vvd_fixups_rebuild_selected_lod_order(self):
        _mdl, base, _vtx = _fixture_model_triplet()
        data = bytearray(base)
        # Add one source vertex then insert two fixups before the vertex block.
        old_vertices = bytes(data[64:]) + bytes(data[64:112])
        rebuilt = bytearray(64 + 24 + len(old_vertices))
        rebuilt[:64] = data[:64]
        struct.pack_into("<i", rebuilt, 12, 2)
        struct.pack_into("<8i", rebuilt, 16, 4, 3, 0, 0, 0, 0, 0, 0)
        struct.pack_into("<iiii", rebuilt, 48, 2, 64, 88, 0)
        struct.pack_into("<iii", rebuilt, 64, 0, 0, 1)
        struct.pack_into("<iii", rebuilt, 76, 1, 1, 3)
        rebuilt[88:] = old_vertices

        lod0 = Source1VVD.parse(bytes(rebuilt), root_lod=0)
        lod1 = Source1VVD.parse(bytes(rebuilt), root_lod=1)
        self.assertEqual(len(lod0.vertices), 4)
        self.assertEqual(len(lod1.vertices), 3)
        self.assertEqual(lod1.vertices[0].position, (0.0, 1.0, -1.0))


if __name__ == "__main__":
    unittest.main()
