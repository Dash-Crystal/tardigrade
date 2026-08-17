from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay.source1_demo import Source1DemoError  # noqa: E402
from state_replay.source1_net import (  # noqa: E402
    decode_game_event,
    decode_packet_entities,
    parse_net_messages,
)


def vi(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128); value >>= 7
    out.append(value)
    return bytes(out)


def pv(number, value): return vi(number << 3) + vi(value)
def pb(number, value): return vi((number << 3) | 2) + vi(len(value)) + value


class Source1NetTest(unittest.TestCase):
    def test_signed_int32_narrowing(self):
        negative = (1 << 64) - 1
        packet = decode_packet_entities(pv(1, 16) + pv(2, 0) + pv(6, negative))
        self.assertEqual(-1, packet["delta_from"])
        key = pv(1, 3) + pv(4, negative)
        event = decode_game_event(pv(2, 7) + pb(3, key), {
            7: {"name": "hurt", "keys": [{"name": "delta", "type": 3}]}})
        self.assertEqual(-1, event["values"][0]["value"])

    def test_game_event_type_and_descriptor_must_match_value_field(self):
        good = pv(1, 3) + pv(4, 9)
        event = decode_game_event(pv(2, 7) + pb(3, good), {
            7: {"name": "hurt", "keys": [{"name": "delta", "type": 3}]}})
        self.assertEqual(9, event["values"][0]["value"])
        wrong_wire_type = pv(1, 2) + pv(4, 9)
        with self.assertRaisesRegex(Source1DemoError, "disagrees with long"):
            decode_game_event(pv(2, 7) + pb(3, wrong_wire_type), {})
        with self.assertRaisesRegex(Source1DemoError, "descriptor type"):
            decode_game_event(pv(2, 7) + pb(3, good), {
                7: {"name": "hurt", "keys": [{"name": "delta", "type": 2}]}})

    def test_net_framing_is_bounded_and_fails_truncation(self):
        payload = vi(25) + vi(3) + b"abc"
        self.assertEqual(b"abc", list(parse_net_messages(payload))[0].payload)
        with self.assertRaisesRegex(Source1DemoError, "truncated net message"):
            list(parse_net_messages(payload[:-1]))


if __name__ == "__main__":
    unittest.main()
