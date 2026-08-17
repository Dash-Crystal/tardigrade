from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cs_hid_motion_decode import (  # noqa: E402
    M575_DESCRIPTOR_SHA256,
    MotionDecodeError,
    decode_documents,
    decode_m575_report,
)


DESCRIPTOR = bytes.fromhex(
    "05010902a10185020901a1009510750115002501050919012910810205011601f8"
    "26ff07750c95020930093181061581257f75089501093881069501050c0a380281"
    "06c0c00643ff0a0202a101851175089513150026ff000902810009029100c0"
)


def report(x: int, y: int, buttons: int = 0, wheel: int = 0, pan: int = 0) -> bytes:
    packed = (x & 0xFFF) | ((y & 0xFFF) << 12)
    return (
        bytes((2,))
        + buttons.to_bytes(2, "little")
        + packed.to_bytes(3, "little")
        + bytes((wheel & 0xFF, pan & 0xFF))
    )


def fixture() -> list[dict]:
    device = {
        "vendor_id": 0x046D,
        "product_id": 0xB041,
        "location_id": 7,
        "product": "ERGO M575S",
        "report_descriptor_base64": base64.b64encode(DESCRIPTOR).decode(),
    }
    return [
        {
            "type": "capture_header",
            "schema": "counter-strike-ego-hid-v3",
            "effective_backend": "hid-report",
            "selected_devices": [device],
        },
        {"type": "clock_anchor", "mach_timebase_numer": 125,
         "mach_timebase_denom": 3},
        {"type": "hid_report", "sequence": 2, "source_sequence": 1,
         "hid_report_timestamp_ticks": 1_000_000,
         "received_monotonic_raw_ns": 50,
         "report_base64": base64.b64encode(report(-2, 3, 1)).decode(),
         "device": device},
        {"type": "hid_report", "sequence": 3, "source_sequence": 2,
         "hid_report_timestamp_ticks": 1_192_000,
         "received_monotonic_raw_ns": 8_000_050,
         "report_base64": base64.b64encode(report(4, -5, 0, -1, 1)).decode(),
         "device": device},
        {"type": "capture_end"},
    ]


class HidMotionDecodeTests(unittest.TestCase):
    def test_descriptor_and_signed_report_layout_are_exact(self):
        import hashlib
        self.assertEqual(M575_DESCRIPTOR_SHA256, hashlib.sha256(DESCRIPTOR).hexdigest())
        self.assertEqual(
            {"buttons": 2, "dx_counts": -136, "dy_counts": 50,
             "wheel_counts": -1, "pan_counts": 1},
            decode_m575_report(report(-136, 50, 2, -1, 1)),
        )

    def test_time_series_retains_integrals_and_labels_only_interval_average(self):
        decoded = decode_documents(fixture())
        first, second = decoded[1:3]
        self.assertEqual((-2, 3), (first["dx_counts"], first["dy_counts"]))
        self.assertIsNone(first["interval_average_count_rate_per_second"])
        self.assertEqual((4, -5), (second["dx_counts"], second["dy_counts"]))
        self.assertEqual(192_000, second["interval_hid_ticks"])
        self.assertEqual(8_000_000.0, second["interval_ns"])
        self.assertEqual(
            [500.0, -625.0],
            second["interval_average_count_rate_per_second"],
        )
        self.assertNotIn("velocity_counts_per_second", second)
        self.assertEqual(
            {"nanoseconds_numerator": 125, "nanoseconds_denominator": 3},
            decoded[0]["hid_tick_timebase"],
        )
        self.assertTrue(decoded[-1]["terminal_observed"])

    def test_wrong_descriptor_and_malformed_report_refuse(self):
        documents = fixture()
        documents[0]["selected_devices"][0]["report_descriptor_base64"] = (
            base64.b64encode(DESCRIPTOR[:-1] + b"x").decode()
        )
        with self.assertRaisesRegex(MotionDecodeError, "unrecognized"):
            decode_documents(documents)
        with self.assertRaisesRegex(MotionDecodeError, "8 bytes"):
            decode_m575_report(b"\x02")


if __name__ == "__main__":
    unittest.main()
