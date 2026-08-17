"""Adversarial Source 1 parity and state/pixel-contract regressions.

These fixtures are deliberately tiny, but the expected behavior comes from
Valve's csgo-demoinfo algorithms rather than from the implementation under
test.  In particular, they exercise failure atomicity and facts which must
survive the boundary between the demo decoder, integrator, and renderer.
"""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from counter_strike_render.source1_backend import (  # noqa: E402
    Source1RenderConfig,
    Source1RenderRefusal,
    render_snapshot,
)
from state_replay.integrator import state_hash  # noqa: E402
from state_replay.source1_adapter import (  # noqa: E402
    CanonicalSource1Reducer,
    ReductionContext,
)
from state_replay.source1_demo import Source1DemoError  # noqa: E402
from state_replay.source1_entities import (  # noqa: E402
    DPT_DATATABLE,
    DPT_INT,
    EntityChange,
    EntityState,
    PacketEntityDecoder,
    StringTableDecoder,
    flatten_send_tables,
)
from state_replay.source1_net import (  # noqa: E402
    DataTableClass,
    DataTables,
    decode_game_event,
)


class Bits:
    def __init__(self) -> None:
        self.values: list[int] = []

    def put(self, value: int, count: int) -> None:
        self.values.extend((value >> bit) & 1 for bit in range(count))

    def cstring(self, value: str) -> None:
        for byte in value.encode("ascii") + b"\0":
            self.put(byte, 8)

    def bytes(self) -> bytes:
        result = bytearray((len(self.values) + 7) // 8)
        for index, value in enumerate(self.values):
            result[index >> 3] |= value << (index & 7)
        return bytes(result)


def _vi(value: int) -> bytes:
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _pv(number: int, value: int) -> bytes:
    return _vi(number << 3) + _vi(value)


def _pb(number: int, value: bytes) -> bytes:
    return _vi((number << 3) | 2) + _vi(len(value)) + value


def _int_prop(name: str) -> dict[str, object]:
    return {
        "type": DPT_INT,
        "var_name": name,
        "flags": 1,
        "priority": 0,
        "data_table_name": None,
        "num_elements": 0,
        "low_value": 0.0,
        "high_value": 0.0,
        "num_bits": 8,
    }


def _one_class_tables() -> DataTables:
    return DataTables(
        ({"name": "DT_Test", "properties": [_int_prop("health")]},),
        (DataTableClass(0, "CTest", "DT_Test"),),
    )


def _component(value, provenance="recorded", **metadata):
    return {"provenance": provenance, "value": value, **metadata}


def _render_snapshot() -> dict[str, object]:
    camera = {
        "id": "camera",
        "generation": 0,
        "class": "camera",
        "components": {"camera": _component({
            "origin": [0.0, 0.0, 0.0],
            "yaw_degrees": 0.0,
            "pitch_degrees": 0.0,
            "fov": 90.0,
        })},
    }
    visible = {
        "id": "visible",
        "generation": 0,
        "class": "prop",
        "components": {
            "source1_geometry": _component({
                "space": "world",
                "vertices": [[4.0, -2.0, -0.8], [4.0, -0.5, -0.8],
                             [4.0, -1.25, 0.8]],
                "triangles": [[0, 1, 2]],
                "color": [220, 30, 20],
            }),
            "visibility": _component({"in_pvs": True}),
        },
    }
    hidden = {
        "id": "hidden",
        "generation": 0,
        "class": "prop",
        "components": {
            "source1_geometry": _component({
                "space": "world",
                "vertices": [[4.0, 0.5, -0.8], [4.0, 2.0, -0.8],
                             [4.0, 1.25, 0.8]],
                "triangles": [[0, 1, 2]],
                "color": [20, 220, 30],
            }),
            "visibility": _component({"in_pvs": False}),
        },
    }
    entities = [camera, visible, hidden]
    return {
        "schema": "tardigrade/state-snapshot/v1",
        "stream_id": "qa/source1",
        "tick": 1,
        "subtick": 0,
        "state_hash": state_hash(entities),
        "entities": entities,
    }


class Source1ProtocolParityQA(unittest.TestCase):
    def test_valve_noncollapsible_child_is_flattened_before_parent_locals(self):
        child = {"name": "DT_Child", "properties": [_int_prop("child_value")]}
        nested = {
            "type": DPT_DATATABLE,
            "var_name": "nested",
            "flags": 0,
            "priority": 0,
            "data_table_name": "DT_Child",
            "num_elements": 0,
            "low_value": 0.0,
            "high_value": 0.0,
            "num_bits": 0,
        }
        root = {
            "name": "DT_Root",
            "properties": [_int_prop("before"), nested, _int_prop("after")],
        }
        tables = DataTables(
            (root, child), (DataTableClass(0, "CRoot", "DT_Root"),)
        )

        # Valve GatherProps(child) appends the child table before the current
        # table's temporary scalar list is appended.
        self.assertEqual(
            ["nested.child_value", "before", "after"],
            [item.path for item in flatten_send_tables(tables)[0]],
        )

    def test_negative_protobuf_int32_game_event_value_is_sign_extended(self):
        # int32 -1 uses protobuf's ten-byte, two's-complement varint form.
        key = _pv(1, 3) + _pv(4, (1 << 64) - 1)
        event = _pb(1, b"qa_negative") + _pv(2, 7) + _pb(3, key)
        decoded = decode_game_event(event, {})
        self.assertEqual(-1, decoded["values"][0]["value"])

    def test_game_event_type_must_match_populated_value_field(self):
        # TYPE_STRING is 1, but field 4 is val_long. Accepting this pair makes
        # the logged event depend on decoder guesswork rather than wire state.
        key = _pv(1, 1) + _pv(4, 7)
        event = _pb(1, b"qa_mismatch") + _pv(2, 8) + _pb(3, key)
        with self.assertRaises(Source1DemoError):
            decode_game_event(event, {})

    def test_malformed_baseline_does_not_install_decoded_prefix(self):
        decoder = PacketEntityDecoder(_one_class_tables())
        bits = Bits()
        bits.put(0, 1)  # old field-index coding
        bits.put(0x7F, 7)
        bits.put(0x7F, 7)  # 0xfff end marker
        malformed = bits.bytes() + b"\x01"

        with self.assertRaises(Source1DemoError):
            decoder.install_baseline(0, malformed)
        self.assertEqual({}, decoder.baselines)

    def test_malformed_stringtable_snapshot_is_transactional(self):
        entities = PacketEntityDecoder(_one_class_tables())
        decoder = StringTableDecoder(entities)
        decoder.create({
            "name": "modelprecache",
            "max_entries": 8,
            "num_entries": 0,
            "user_data_fixed_size": False,
            "user_data_size": 0,
            "user_data_size_bits": 0,
            "_string_data_bytes": b"\0",
        })
        before = list(decoder.tables)

        snapshot = Bits()
        snapshot.put(1, 8)
        snapshot.cstring("modelprecache")
        snapshot.put(1, 16)
        snapshot.cstring("models/qa.mdl")
        snapshot.put(0, 1)  # no server user data
        snapshot.put(0, 1)  # no client-side table
        malformed = snapshot.bytes() + b"\x01"
        with self.assertRaises(Source1DemoError):
            decoder.snapshot(malformed)

        self.assertEqual(before, decoder.tables)
        self.assertEqual({}, decoder.tables[0].entries)

    def test_full_stringtable_snapshot_removes_stale_server_entries(self):
        entities = PacketEntityDecoder(_one_class_tables())
        decoder = StringTableDecoder(entities)
        decoder.create({
            "name": "modelprecache",
            "max_entries": 8,
            "num_entries": 0,
            "user_data_fixed_size": False,
            "user_data_size": 0,
            "user_data_size_bits": 0,
            "_string_data_bytes": b"\0",
        })
        decoder.tables[0].entries[1] = ("models/stale.mdl", b"")

        snapshot = Bits()
        snapshot.put(1, 8)
        snapshot.cstring("modelprecache")
        snapshot.put(1, 16)
        snapshot.cstring("models/current.mdl")
        snapshot.put(0, 1)
        snapshot.put(0, 1)
        decoder.snapshot(snapshot.bytes())

        self.assertEqual(
            {0: ("models/current.mdl", b"")}, decoder.tables[0].entries
        )

    def test_duplicate_sendtable_names_are_refused_before_last_wins(self):
        duplicate = DataTables(
            (
                {"name": "DT_Test", "properties": [_int_prop("first")]},
                {"name": "DT_Test", "properties": [_int_prop("second")]},
            ),
            (DataTableClass(0, "CTest", "DT_Test"),),
        )

        # Server classes refer to a send table by name. Silently choosing the
        # last duplicate would make the flattened class layout ambiguous.
        with self.assertRaisesRegex(Source1DemoError, "duplicate send-table"):
            flatten_send_tables(duplicate)

    def test_duplicate_stringtable_names_cannot_alias_canonical_resource(self):
        entities = PacketEntityDecoder(_one_class_tables())
        decoder = StringTableDecoder(entities)
        empty = {
            "name": "modelprecache",
            "max_entries": 8,
            "num_entries": 0,
            "user_data_fixed_size": False,
            "user_data_size": 0,
            "user_data_size_bits": 0,
            "_string_data_bytes": b"\0",
        }
        decoder.create(empty)

        # Wire table IDs are distinct, but canonical resources are keyed by
        # name. A duplicate must not alias one hashed resource entity.
        with self.assertRaisesRegex(Source1DemoError, "duplicate string-table"):
            decoder.create(empty)
        self.assertEqual(1, len(decoder.tables))


class Source1StateRendererQA(unittest.TestCase):
    def test_reducer_failure_does_not_poison_generation_state(self):
        reducer = CanonicalSource1Reducer()
        entity = EntityState(0, 0, "CTest", 3, {}, True, False)
        decoder = SimpleNamespace(entities={0: entity})
        changes = (
            EntityChange("enter", 0, 0, "CTest", 3, {}, False),
            EntityChange("delete", 99, 0, "CTest", 1, {}, False),
        )
        with self.assertRaises(Source1DemoError):
            reducer.on_entities(changes, decoder, ReductionContext(1, 0, 0))
        self.assertEqual({}, reducer.active)
        self.assertEqual({}, reducer.watermarks)

    def test_renderer_rejects_forged_state_hash_before_pixels(self):
        snapshot = _render_snapshot()
        snapshot["state_hash"] = "0" * 64
        with self.assertRaisesRegex(Source1RenderRefusal, "state-hash-mismatch"):
            render_snapshot(
                snapshot,
                Source1RenderConfig(width=32, height=24,
                                    allow_synthetic_geometry=True),
            )

    def test_leave_pvs_geometry_contributes_no_pixels(self):
        rendered = render_snapshot(
            _render_snapshot(),
            Source1RenderConfig(width=96, height=64,
                                allow_synthetic_geometry=True),
        )
        pixels = rendered.ppm.split(b"\n", 3)[3]
        colors = {tuple(pixels[index:index + 3])
                  for index in range(0, len(pixels), 3)}
        self.assertIn((220, 30, 20), colors)
        self.assertNotIn((20, 220, 30), colors)


if __name__ == "__main__":
    unittest.main()
