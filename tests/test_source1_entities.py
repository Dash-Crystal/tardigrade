from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay.source1_adapter import (  # noqa: E402
    CanonicalSource1Reducer,
    ReductionContext,
)
from state_replay import StateIntegrator  # noqa: E402
from state_replay.source1_demo import Source1DemoError  # noqa: E402
from state_replay.source1_entities import (  # noqa: E402
    EntityChange,
    EntityState,
    PacketEntityDecoder,
    StringTableDecoder,
    StringTableState,
    flatten_send_tables,
)
from state_replay.source1_net import DataTableClass, DataTables  # noqa: E402
from state_replay.source1_resource_contract import (  # noqa: E402
    MODEL_BINDING_DERIVATION,
    STRING_TABLE_RESOURCE_DERIVATION,
    Source1ResourceContractError,
    make_string_table_entry,
    string_table_entry_document_sha256,
    string_table_entry_sha256,
    validate_string_table_entry,
)


class Bits:
    def __init__(self): self.bits = []
    def put(self, value, count):
        self.bits.extend((value >> index) & 1 for index in range(count))
    def cstring(self, value):
        for byte in value.encode() + b"\0": self.put(byte, 8)
    def bytes(self):
        data = bytearray((len(self.bits) + 7) // 8)
        for index, value in enumerate(self.bits):
            data[index >> 3] |= value << (index & 7)
        return bytes(data)


def end_fields(bits):
    bits.put(0, 1); bits.put(0, 1); bits.put(0x7f, 7); bits.put(0x7f, 7)


def tables(changes_priority=False):
    props = [
        {"type": 0, "var_name": "health", "flags": 1 | ((1 << 18)
         if changes_priority else 0), "priority": 0, "data_table_name": None,
         "num_elements": 0, "low_value": 0.0, "high_value": 0.0, "num_bits": 8},
        {"type": 0, "var_name": "armor", "flags": 1, "priority": 1,
         "data_table_name": None, "num_elements": 0, "low_value": 0.0,
         "high_value": 0.0, "num_bits": 8},
    ]
    return DataTables(({"name": "DT_Test", "properties": props},),
                      (DataTableClass(0, "CTest", "DT_Test"),))


def fields(values):
    b = Bits(); b.put(1, 1)
    for _ in values: b.put(1, 1)
    end_fields(b)
    for value in values: b.put(value, 8)
    return b.bytes()


def enter(values=(10, 20), serial=1, extra=b""):
    b = Bits(); b.put(0, 6); b.put(0, 1); b.put(1, 1); b.put(0, 1); b.put(serial, 10)
    b.put(1, 1)
    for _ in values: b.put(1, 1)
    end_fields(b)
    for value in values: b.put(value, 8)
    return b.bytes() + extra


def envelope(raw, is_delta=False):
    return {"max_entries": 16, "updated_entries": 1, "is_delta": is_delta,
            "_entity_data_bytes": raw}


def delta_one(value):
    b = Bits(); b.put(0, 6); b.put(0, 1); b.put(0, 1)
    b.put(1, 1); b.put(1, 1); end_fields(b); b.put(value, 8)
    return b.bytes()


class Source1EntitiesTest(unittest.TestCase):
    def test_shared_resource_contract_is_exact_and_reducer_uses_it(self):
        entry = make_string_table_entry(3, "models/a.mdl", b"A")
        self.assertEqual({"index", "string", "user_data_base64",
                          "content_provenance",
                          "canonical_presence_provenance"}, set(entry))
        self.assertEqual(64, len(string_table_entry_sha256(
            3, "models/a.mdl", b"A")))
        self.assertEqual((3, "models/a.mdl", b"A"),
                         validate_string_table_entry(entry))
        self.assertEqual(string_table_entry_sha256(3, "models/a.mdl", b"A"),
                         string_table_entry_document_sha256(entry))
        with self.assertRaises(Source1ResourceContractError):
            validate_string_table_entry({**entry, "extra": True})
        with self.assertRaisesRegex(Source1ResourceContractError, "canonical"):
            validate_string_table_entry({**entry, "user_data_base64": "QR=="})
        reducer = CanonicalSource1Reducer()
        table = StringTableState(0, "modelprecache", 8, False, 0, 0,
                                 {3: ("models/a.mdl", b"A")})
        component = reducer._string_table_component(table, 0, "create")
        self.assertEqual(STRING_TABLE_RESOURCE_DERIVATION,
                         component["derivation"])
        self.assertEqual(entry, component["value"]["server_entries"][0])
        reducer.modelprecache = {3: ("models/a.mdl", b"A")}
        reducer.table_revisions = {"modelprecache": 0}
        model, _ = reducer._model_component(EntityState(
            0, 0, "CTest", 1, {"m_nModelIndex": 3}))
        self.assertEqual(MODEL_BINDING_DERIVATION, model["derivation"])
        self.assertEqual(string_table_entry_sha256(3, "models/a.mdl", b"A"),
                         model["value"]["source_entry_sha256"])

    def test_modelprecache_join_emits_binding_without_inventing_lod(self):
        reducer = CanonicalSource1Reducer()
        table = StringTableState(0, "modelprecache", 8, False, 0, 0,
                                 {3: ("models/player/test.mdl", b"")})
        entity = EntityState(0, 0, "CTest", 1,
                             {"baseclass.m_nModelIndex": 3}, True, True)
        fake = SimpleNamespace(entities={0: entity})
        change = EntityChange("enter", 0, 0, "CTest", 1, {}, True)
        reducer.on_string_table("snapshot", table, (3,),
                                ReductionContext(0, 0, 0), fake)
        components = reducer.on_entities((change,), fake,
            ReductionContext(1, 0, 1))[0]["entity"]["components"]
        self.assertNotIn("animation", components)
        self.assertNotIn("physics", components)
        model = components["source1_model"]
        self.assertEqual("derived", model["provenance"])
        self.assertEqual(
            "lossless join of decoded m_nModelIndex to recorded modelprecache entry",
            model["derivation"])
        self.assertEqual("models/player/test.mdl", model["value"]["source1_mdl"])
        self.assertEqual("unavailable", model["value"]["lod"]["provenance"])

    def test_noncollapsible_child_precedes_parent_local_props(self):
        scalar = lambda name: {"type": 0, "var_name": name, "flags": 1,
            "priority": 0, "data_table_name": None, "num_elements": 0,
            "low_value": 0.0, "high_value": 0.0, "num_bits": 8}
        nested = {"type": 6, "var_name": "child", "flags": 0,
            "priority": 0, "data_table_name": "DT_Child", "num_elements": 0,
            "low_value": 0.0, "high_value": 0.0, "num_bits": 0}
        data = DataTables((
            {"name": "DT_Root", "properties": [scalar("A"), nested, scalar("B")]},
            {"name": "DT_Child", "properties": [scalar("C")]},
        ), (DataTableClass(0, "CTest", "DT_Root"),))
        self.assertEqual(["child.C", "A", "B"],
            [item.path for item in flatten_send_tables(data)[0]])

    def test_bit_exact_enter_and_malformed_tail_rolls_back(self):
        decoder = PacketEntityDecoder(tables())
        with self.assertRaisesRegex(Source1DemoError, "unconsumed"):
            decoder.apply(envelope(enter(extra=b"\x01")))
        self.assertEqual({}, decoder.entities)
        change = decoder.apply(envelope(enter()))[0]
        self.assertEqual("enter", change.kind)
        self.assertEqual({"health": 10, "armor": 20},
                         decoder.entities[0].properties)

    def test_full_update_removes_omitted_ghost_and_old_delta_base_reconciles(self):
        decoder = PacketEntityDecoder(tables())
        initial = envelope(enter((10, 20)))
        decoder.apply(initial, tick=1)
        delta = envelope(delta_one(30), is_delta=True)
        delta["delta_from"] = 1
        decoder.apply(delta, tick=2)
        self.assertEqual(30, decoder.entities[0].properties["health"])
        # C is based on A and contains no headers, so it must revert B's
        # health even though the wire contains no explicit entity update.
        omitted = {"max_entries": 16, "updated_entries": 0, "is_delta": True,
                   "delta_from": 1, "_entity_data_bytes": b""}
        changes = decoder.apply(omitted, tick=3)
        self.assertEqual("delta", changes[0].kind)
        self.assertEqual(10, decoder.entities[0].properties["health"])
        full_empty = {"max_entries": 16, "updated_entries": 0,
                      "is_delta": False, "_entity_data_bytes": b""}
        changes = decoder.apply(full_empty, tick=4)
        self.assertEqual("delete", changes[0].kind)
        self.assertEqual({}, decoder.entities)

    def test_changes_often_is_effective_priority_64(self):
        flat = flatten_send_tables(tables(changes_priority=True))[0]
        self.assertEqual(["armor", "health"], [item.path for item in flat])
        decoder = PacketEntityDecoder(tables(changes_priority=True))
        decoder.apply(envelope(enter((44, 99))))
        self.assertEqual({"armor": 44, "health": 99},
                         decoder.entities[0].properties)

    def test_instancebaseline_is_applied_before_enter_delta(self):
        decoder = PacketEntityDecoder(tables())
        string_tables = StringTableDecoder(decoder)
        baseline = fields((50, 5))
        b = Bits()
        b.put(0, 1)  # no dictionary
        b.put(1, 1)  # sequential index zero
        b.put(1, 1); b.put(0, 1); b.cstring("0")
        b.put(1, 1); b.put(len(baseline), 14)
        for byte in baseline: b.put(byte, 8)
        table, changed = string_tables.create({
            "name": "instancebaseline", "max_entries": 16, "num_entries": 1,
            "user_data_fixed_size": False, "user_data_size": 0,
            "user_data_size_bits": 0, "_string_data_bytes": b.bytes()})
        self.assertEqual((0,), changed)
        # Enter changes only field zero; armor survives from baseline.
        raw = enter((80,))
        change = decoder.apply(envelope(raw))[0]
        self.assertFalse(change.complete_from_baseline)
        self.assertTrue(decoder.entities[0].baseline_present)
        self.assertEqual({"health": 80, "armor": 5}, decoder.entities[0].properties)

    def test_string_resources_and_model_rejoins_are_hashed_seekable_and_persistent(self):
        reducer = CanonicalSource1Reducer()
        decoder = PacketEntityDecoder(tables())
        context = ReductionContext(1, 0, 0)
        table = StringTableState(0, "modelprecache", 8, False, 0, 0,
            {3: ("models/a.mdl", b"A")},
            {0: ("client-only", b"C")})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "journal"
            stream = "source1/resources"
            journal = StateIntegrator.create(root, {
                "schema": "tardigrade/state-manifest/v1", "stream_id": stream,
                "tick_rate_hz": 64, "source": {"kind": "test"}},
                StateIntegrator.make_checkpoint(stream, 0, 0, 0, []))

            def ingest(tick, ops):
                return journal.ingest({
                    "schema": "tardigrade/state-transaction/v1",
                    "stream_id": stream,
                    "sequence": journal.current_snapshot()["sequence"] + 1,
                    "tick": tick, "subtick": 0,
                    "base_hash": journal.current_snapshot()["state_hash"],
                    "ops": list(ops)})

            ingest(1, reducer.on_string_table("create", table, (3,), context,
                                              decoder))
            first = journal.current_snapshot()
            resource = next(e for e in first["entities"]
                            if e["class"] == "Source1StringTable")
            self.assertEqual("derived", resource["components"][
                "source1_string_table"]["provenance"])
            origins = resource["components"]["source1_string_table"][
                "value"]["field_origins"]
            self.assertEqual("derived", origins["server_entries"]["provenance"])
            self.assertEqual("recorded", origins["server_entries"][
                "entry_content_provenance"])
            self.assertEqual("derived", origins["table_id"]["provenance"])
            self.assertEqual("derived", origins["revision"]["provenance"])
            self.assertEqual("QQ==", resource["components"][
                "source1_string_table"]["value"]["server_entries"][0][
                    "user_data_base64"])
            self.assertEqual("client-only", resource["components"][
                "source1_string_table"]["value"]["client_entries"][0]["string"])

            entity = EntityState(0, 0, "CTest", 1,
                {"m_nModelIndex": 3}, True, False, False)
            decoder.entities[0] = entity
            ingest(2, reducer.on_entities((EntityChange(
                "enter", 0, 0, "CTest", 1, {}, False),), decoder, context))

            updated = StringTableState(0, "modelprecache", 8, False, 0, 0,
                {3: ("models/b.mdl", b"B")})
            ingest(3, reducer.on_string_table("update", updated, (3,), context,
                                              decoder))
            current = journal.current_snapshot()
            pawn = next(e for e in current["entities"] if e["id"] == "source1:edict:0")
            self.assertEqual("models/b.mdl", pawn["components"][
                "source1_model"]["value"]["source1_mdl"])
            self.assertNotEqual(first["state_hash"], current["state_hash"])
            self.assertEqual("models/a.mdl", next(e for e in journal.state_at(2)[
                "entities"] if e["id"] == "source1:edict:0")["components"][
                    "source1_model"]["value"]["source1_mdl"])
            journal.checkpoint()
            expected = journal.current_snapshot()
            journal.close()
            journal = StateIntegrator.open(root, stream)
            self.assertEqual(expected["state_hash"], journal.current_snapshot()["state_hash"])
            ingest(4, reducer.on_string_tables_removed(
                (updated,), context, decoder))
            removed = journal.current_snapshot()
            self.assertNotIn("source1:stringtable:modelprecache",
                             {e["id"] for e in removed["entities"]})
            pawn = next(e for e in removed["entities"]
                        if e["id"] == "source1:edict:0")
            self.assertNotIn("source1_model", pawn["components"])
            journal.close()

    def test_sparse_instancebaseline_is_never_called_complete(self):
        reducer = CanonicalSource1Reducer()
        entity = EntityState(0, 0, "CTest", 1, {"health": 50},
                             True, False, True)
        components = reducer._components(entity)
        self.assertEqual("partial-instancebaseline-missing-constructor-defaults",
                         components["source1_netprops"]["completeness"])

    def test_baseline_and_table_failures_are_atomic_and_snapshot_recovers_baseline(self):
        decoder = PacketEntityDecoder(tables())
        valid = fields((50, 5))
        with self.assertRaises(Source1DemoError):
            decoder.install_baseline(0, valid + b"\x01")
        self.assertEqual({}, decoder.baselines)

        string_tables = StringTableDecoder(decoder)
        bad = Bits(); bad.put(0, 1)
        for name, data in (("0", valid), ("99", valid)):
            bad.put(1, 1); bad.put(1, 1); bad.put(0, 1); bad.cstring(name)
            bad.put(1, 1); bad.put(len(data), 14)
            for byte in data: bad.put(byte, 8)
        with self.assertRaisesRegex(Source1DemoError, "unknown class"):
            string_tables.create({"name": "instancebaseline", "max_entries": 16,
                "num_entries": 2, "user_data_fixed_size": False,
                "user_data_size": 0, "user_data_size_bits": 0,
                "_string_data_bytes": bad.bytes()})
        self.assertEqual([], string_tables.tables)
        self.assertEqual({}, decoder.baselines)

        # dem_stringtables command snapshot: table count/name, server entries,
        # then the client-entry presence bit.
        snap = Bits(); snap.put(1, 8); snap.cstring("instancebaseline")
        snap.put(1, 16); snap.cstring("0"); snap.put(1, 1)
        snap.put(len(valid), 16)
        for byte in valid: snap.put(byte, 8)
        snap.put(0, 1)
        result = string_tables.snapshot(snap.bytes())
        self.assertEqual("instancebaseline", result[0][0].name)
        self.assertEqual({"health": 50, "armor": 5}, decoder.baselines[0])

    def test_stringtable_snapshot_replaces_stale_server_entries(self):
        decoder = PacketEntityDecoder(tables())
        strings = StringTableDecoder(decoder)
        strings.tables.append(StringTableState(0, "modelprecache", 8,
            False, 0, 0, {0: ("zero", b""), 1: ("stale", b"")}))
        snap = Bits(); snap.put(1, 8); snap.cstring("modelprecache")
        snap.put(1, 16); snap.cstring("fresh"); snap.put(0, 1)
        snap.put(0, 1)
        strings.snapshot(snap.bytes())
        self.assertEqual({0: ("fresh", b"")}, strings.tables[0].entries)

    def test_leave_delete_and_serial_wrap_use_distinct_canonical_lifetimes(self):
        reducer = CanonicalSource1Reducer()
        context = ReductionContext(1, 0, 100)
        first = EntityState(0, 0, "CTest", 1023, {}, True, True)
        fake = SimpleNamespace(entities={0: first})
        enter1 = EntityChange("enter", 0, 0, "CTest", 1023, {}, True)
        create = reducer.on_entities((enter1,), fake, context)[0]
        self.assertEqual(0, create["entity"]["generation"])
        leave = EntityChange("leave", 0, 0, "CTest", 1023, {}, True)
        self.assertEqual("update", reducer.on_entities((leave,), fake, context)[0]["op"])
        second = EntityState(0, 0, "CTest", 0, {}, True, True)
        fake.entities[0] = second
        enter2 = EntityChange("enter", 0, 0, "CTest", 0, {}, True)
        ops = reducer.on_entities((enter2,), fake, context)
        self.assertEqual(["destroy", "create"], [op["op"] for op in ops])
        self.assertEqual(1, ops[1]["entity"]["generation"])
        self.assertEqual(0, ops[1]["entity"]["components"][
            "source1_identity"]["value"]["serial"])

    def test_reducer_multi_change_failure_rolls_back_lifetime_bookkeeping(self):
        reducer = CanonicalSource1Reducer()
        entity = EntityState(0, 0, "CTest", 4, {}, True, True)
        fake = SimpleNamespace(entities={0: entity})
        changes = (
            EntityChange("enter", 0, 0, "CTest", 4, {}, True),
            EntityChange("delete", 9, 0, "CTest", 1, {}, True),
        )
        with self.assertRaisesRegex(Source1DemoError, "untracked"):
            reducer.on_entities(changes, fake, ReductionContext(1, 0, 0))
        self.assertEqual({}, reducer.active)
        self.assertEqual({}, reducer.watermarks)


if __name__ == "__main__":
    unittest.main()
