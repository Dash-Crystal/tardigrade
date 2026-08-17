from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay import StateIntegrator  # noqa: E402
from state_replay.source1_adapter import Source1ReplayAdapter  # noqa: E402
from state_replay.source1_demo import Source1DemoError, Source1DemoReader  # noqa: E402
from counter_strike_render.replay_bridge import StateToRenderBridge  # noqa: E402


def vi(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def pv(number, value):
    return vi(number << 3) + vi(value)


def pb(number, value):
    return vi((number << 3) | 2) + vi(len(value)) + value


def pf(number, value):
    return vi((number << 3) | 5) + struct.pack("<f", value)


def net(message_type, payload):
    return vi(message_type) + vi(len(payload)) + payload


class Bits:
    def __init__(self):
        self.bits = []

    def put(self, value, count):
        self.bits.extend((value >> n) & 1 for n in range(count))

    def bytes(self):
        out = bytearray((len(self.bits) + 7) // 8)
        for n, value in enumerate(self.bits):
            out[n // 8] |= value << (n % 8)
        return bytes(out)


def end_fields(bits):
    bits.put(0, 1)
    bits.put(0, 1)
    bits.put(0x7f, 7)
    bits.put(0x7f, 7)


def entity_enter(serial=7, health=100, origin=(1.0, 2.0, 3.0)):
    b = Bits()
    b.put(0, 6)  # ReadUBitVar: edict zero
    b.put(0, 1)  # not leave
    b.put(1, 1)  # enter
    b.put(0, 1)  # one server class -> one class bit
    b.put(serial, 10)
    b.put(1, 1)  # new field-index encoding
    b.put(1, 1)  # field 0
    b.put(1, 1)  # field 1
    end_fields(b)
    b.put(health, 8)
    for value in origin:
        b.put(struct.unpack("<I", struct.pack("<f", value))[0], 32)
    return b.bytes()


def packet_entities(entity_data):
    return (pv(1, 16) + pv(2, 1) + pv(3, 0) + pb(7, entity_data))


def send_tables():
    health = pv(1, 0) + pb(2, b"m_iHealth") + pv(3, 1) + pv(4, 0) + pv(9, 8)
    origin = (pv(1, 2) + pb(2, b"m_vecOrigin") + pv(3, 4) + pv(4, 0) +
              pf(7, 0.0) + pf(8, 0.0) + pv(9, 32))
    table = pb(2, b"DT_Test") + pb(4, health) + pb(4, origin)
    end = pv(1, 1)
    return (net(9, table) + net(9, end) + struct.pack("<hh", 1, 0) +
            b"CTestEntity\x00DT_Test\x00")


def header(stamp=b"HL2DEMO\x00"):
    fixed = lambda value: value + bytes(260 - len(value))
    return (struct.pack("<8sii", stamp, 4, 13790) + fixed(b"server") +
            fixed(b"client") + fixed(b"de_test") + fixed(b"csgo") +
            struct.pack("<fiii", 2.0, 128, 3, 0))


def command(kind, tick, payload=b"", cmdinfo=None):
    result = struct.pack("<BiB", kind, tick, 0)
    if kind in (1, 2):
        result += (bytes(152) if cmdinfo is None else cmdinfo) + struct.pack("<ii", 10, 11)
    if kind in (1, 2, 4, 6, 9):
        result += struct.pack("<i", len(payload)) + payload
    return result


class Source1DemoTest(unittest.TestCase):
    def test_container_and_packet_entity_reach_integrator(self):
        demo = (header() + command(6, 0, send_tables()) +
                command(2, 1, net(26, packet_entities(entity_enter()))) +
                command(7, 2))
        reader = Source1DemoReader.from_bytes(demo)
        with tempfile.TemporaryDirectory() as temp:
            summary = Source1ReplayAdapter().ingest(
                reader, Path(temp) / "journal", "test/source1")
            self.assertEqual(1, summary.entity_updates)
            journal = StateIntegrator.open(Path(temp) / "journal", "test/source1")
            try:
                snapshot = journal.current_snapshot()
                StateToRenderBridge().convert(snapshot)
                entity = next(item for item in snapshot["entities"]
                              if item["id"] == "source1:edict:0")
                self.assertEqual("CTestEntity", entity["class"])
                self.assertEqual(100, entity["components"][
                    "player_state"]["value"]["m_iHealth"])
                self.assertEqual([1.0, 2.0, 3.0], entity["components"][
                    "transform"]["value"]["origin"])
                self.assertEqual("partial-no-instancebaseline", entity[
                    "components"]["source1_netprops"]["completeness"])
                self.assertNotIn("animation", entity["components"])
                self.assertNotIn("physics", entity["components"])
                self.assertEqual("unavailable", entity["components"][
                    "source1_reconstruction"]["value"]["animation"]["status"])
                camera = next(item for item in journal.current_snapshot()["entities"]
                              if item["class"] == "camera")
                self.assertEqual("unavailable", camera["components"]["camera"]
                    ["value"]["fov"]["provenance"])
            finally:
                journal.close()

    def test_minus_one_preamble_is_preserved_but_canonicalized(self):
        demo = (header() + command(1, -1, b"") +
                command(6, 0, send_tables()) + command(7, 0))
        reader = Source1DemoReader.from_bytes(demo)
        commands = list(reader.commands())
        self.assertEqual([-1, 0, 0], [item.tick for item in commands])
        reader = Source1DemoReader.from_bytes(demo)
        with tempfile.TemporaryDirectory() as temp:
            Source1ReplayAdapter().ingest(reader, Path(temp) / "j", "preamble")
            journal = StateIntegrator.open(Path(temp) / "j", "preamble")
            try:
                self.assertEqual(2, journal.current_snapshot()["subtick"])
                action = journal.action_context("source1:command:1")["action"]
                self.assertEqual(-1, action["payload"]["tick"])
            finally:
                journal.close()

    def test_resampled_camera_pose_and_all_views_are_preserved(self):
        split0 = struct.pack("<i18f", 3, *(
            1, 2, 3, 10, 20, 30, 4, 5, 6,
            101, 102, 103, 40, 50, 60, 7, 8, 9))
        split1 = struct.pack("<i18f", 0, *([0] * 18))
        demo = (header() + command(6, 0, send_tables()) +
                command(2, 1, b"", split0 + split1) + command(7, 2))
        with tempfile.TemporaryDirectory() as temp:
            Source1ReplayAdapter().ingest(Source1DemoReader.from_bytes(demo),
                Path(temp) / "j", "camera")
            journal = StateIntegrator.open(Path(temp) / "j", "camera")
            try:
                camera = next(item for item in journal.current_snapshot()["entities"]
                              if item["class"] == "camera")
                value = camera["components"]["camera"]["value"]
                self.assertEqual([101.0, 102.0, 103.0], value["origin"])
                self.assertEqual((40.0, 50.0),
                    (value["pitch_degrees"], value["yaw_degrees"]))
                action = journal.action_context("source1:command:2")["action"]
                self.assertEqual(2, len(action["payload"]["views"]))
            finally:
                journal.close()

    def test_rejects_source2_compression_truncation_and_trailing_bytes(self):
        with self.assertRaisesRegex(Source1DemoError, "not a Source 1"):
            Source1DemoReader.from_bytes(header(b"PBDEMS2\x00"))
        compressed = header() + struct.pack("<BiB", 0x82, 0, 0)
        with self.assertRaisesRegex(Source1DemoError, "compressed"):
            list(Source1DemoReader.from_bytes(compressed).commands())
        truncated = header() + command(4, 0, b"abc")[:-1]
        with self.assertRaisesRegex(Source1DemoError, "truncated"):
            list(Source1DemoReader.from_bytes(truncated).commands())
        trailing = header() + command(7, 0) + b"x"
        with self.assertRaisesRegex(Source1DemoError, "trailing"):
            list(Source1DemoReader.from_bytes(trailing).commands())

    def test_late_corruption_never_publishes_partial_journal(self):
        demo = header() + command(6, 0, send_tables()) + command(4, 1, b"bad")[:-1]
        with tempfile.TemporaryDirectory() as temp:
            final = Path(temp) / "journal"
            with self.assertRaisesRegex(Source1DemoError, "truncated"):
                Source1ReplayAdapter().ingest(
                    Source1DemoReader.from_bytes(demo), final, "atomic")
            self.assertFalse(final.exists())
            self.assertEqual([], list(Path(temp).glob(".journal.partial-*")))

    def test_adapter_instance_is_explicitly_one_shot(self):
        demo = header() + command(6, 0, send_tables()) + command(7, 1)
        adapter = Source1ReplayAdapter()
        with tempfile.TemporaryDirectory() as temp:
            adapter.ingest(Source1DemoReader.from_bytes(demo),
                           Path(temp) / "one", "one")
            with self.assertRaisesRegex(Source1DemoError, "one-shot"):
                adapter.ingest(Source1DemoReader.from_bytes(demo),
                               Path(temp) / "two", "two")
            self.assertFalse((Path(temp) / "two").exists())


if __name__ == "__main__":
    unittest.main()
