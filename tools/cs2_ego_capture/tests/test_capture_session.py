from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/cs2_capture_session.py"
SPEC = importlib.util.spec_from_file_location("cs2_capture_session", SCRIPT)
assert SPEC and SPEC.loader
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)
PROBE_SCRIPT = REPO / "scripts/cs2_hid_probe.py"
PROBE_SPEC = importlib.util.spec_from_file_location("cs2_hid_probe", PROBE_SCRIPT)
assert PROBE_SPEC and PROBE_SPEC.loader
probe = importlib.util.module_from_spec(PROBE_SPEC)
sys.modules[PROBE_SPEC.name] = probe
PROBE_SPEC.loader.exec_module(probe)


class CaptureSessionTests(unittest.TestCase):
    def test_parse_realistic_appmanifest_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "appmanifest_730.acf"
            path.write_text(
                '''"AppState"
{
    "appid" "730"
    "name" "Counter-Strike 2"
    "installdir" "Counter-Strike Global Offensive"
    "buildid" "24701871"
    "InstalledDepots"
    {
        "2347770"
        {
            "manifest" "7645176062026597595"
            "size" "62785669236"
        }
    }
}
''',
                encoding="utf-8",
            )
            result = capture.parse_appmanifest(path)
        self.assertEqual(result["appid"], "730")
        self.assertEqual(result["build_id"], "24701871")
        self.assertEqual(
            result["installed_depots"], {"2347770": "7645176062026597595"}
        )
        self.assertEqual(len(result["sha256"]), 64)

    def test_changed_demo_detects_new_and_modified_only(self) -> None:
        first = Path("/tmp/first.dem")
        second = Path("/tmp/second.dem")
        unchanged = capture.FileObservation(10, 100)
        before = {first: unchanged}
        after = {
            first: capture.FileObservation(11, 101),
            second: capture.FileObservation(5, 99),
        }
        self.assertEqual(capture.changed_demos(before, after), [first, second])

    def test_stable_demo_copy_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source" / "match.dem"
            source.parent.mkdir()
            source.write_bytes(b"PBDEMS2\x00deterministic fixture")
            result = capture.copy_stable_demo(source, root / "session" / "demos")
            stored = Path(result["stored_path"])
            self.assertEqual(stored.read_bytes(), source.read_bytes())
            self.assertEqual(result["sha256"], capture.sha256_file(source))
            self.assertTrue(result["stable_copy"])

    def test_affine_alignment_centers_large_clock_values(self) -> None:
        base = 9_000_000_000_000_000
        result = capture.affine_alignment(
            [(base, 100), (base + 1_000_000, 164), (base + 2_000_000, 228)]
        )
        self.assertEqual(result["status"], "estimated_affine")
        self.assertAlmostEqual(result["slope_ticks_per_ns"], 64 / 1_000_000)
        self.assertLess(result["max_abs_residual_ticks"], 1e-6)
        self.assertIn("not exact", result["interpretation"])

    def test_atomic_json_leaves_only_final_document(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "manifest.json"
            capture.atomic_json(target, {"status": "complete", "n": 3})
            self.assertEqual(json.loads(target.read_text()), {"status": "complete", "n": 3})
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_safe_session_name_rejects_paths(self) -> None:
        with self.assertRaises(ValueError):
            capture.safe_session_name("../escape")
        self.assertEqual(capture.safe_session_name("bot-trackball-01"), "bot-trackball-01")

    def test_supervisor_uses_target_lifecycle_not_duration(self) -> None:
        args = capture.create_parser().parse_args(
            ["--device", "0x046d:0xb041", "--demo-dir", "/tmp"]
        )
        self.assertEqual(args.reminder_minutes, 15)
        self.assertFalse(hasattr(args, "duration_seconds"))

    def test_compile_staging_preserves_executable_basename(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            source = repo / "tools/cs2_ego_capture/Capture.swift"
            source.parent.mkdir(parents=True)
            source.write_text("// fixture\n")
            binary = root / "bin/cs2-ego-capture"

            def fake_run(command, **kwargs):
                if "-o" in command:
                    output = Path(command[command.index("-o") + 1])
                    output.write_bytes(b"stable-binary-fixture")
                return capture.subprocess.CompletedProcess(
                    command, 0, stdout="swift fixture", stderr=""
                )

            with mock.patch.object(capture.subprocess, "run", side_effect=fake_run):
                compiled, record = capture.compile_capture(repo, binary)

        self.assertEqual(compiled.name, "cs2-ego-capture")
        self.assertEqual(Path(record["compile_command"][-1]).name, "cs2-ego-capture")

    def test_probe_interval_summary(self) -> None:
        result = probe.interval_summary([0, 8_000_000, 16_000_000, 24_000_000])
        self.assertEqual(result["p50_ns"], 8_000_000)
        self.assertEqual(result["median_rate_hz"], 125)

    def test_jsonl_terminal_record_streams_to_final_object(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                '{"type":"capture_header"}\nnot-json\n'
                '{"type":"capture_end","observation_status":"events_observed"}\n'
            )
            terminal = capture.jsonl_terminal_record(path)
        self.assertEqual(terminal["type"], "capture_end")
        self.assertEqual(terminal["observation_status"], "events_observed")

    def test_probe_preserves_backend_identity_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            records = [
                {
                    "type": "clock_anchor",
                    "mach_timebase_numer": 1,
                    "mach_timebase_denom": 1,
                },
                {
                    "type": "hid_report",
                    "hid_report_timestamp_ticks": 100,
                    "received_absolute_ticks": 120,
                },
                {
                    "type": "hid_report",
                    "hid_report_timestamp_ticks": 200,
                    "received_absolute_ticks": 225,
                },
                {
                    "type": "cg_event",
                    "cg_event_timestamp_ns_since_startup": 130,
                },
                {"type": "capture_end", "internal_dropped_records": 0},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            result = probe.analyze_jsonl(path)
        self.assertEqual(result["record_counts"]["hid_report"], 2)
        self.assertEqual(result["record_counts"]["cg_event"], 1)
        self.assertEqual(result["callback_observation_lag"]["hid_report"]["p50_ns"], 22.5)
        self.assertIn("Never sum", result["cross_backend_policy"])

    def test_probe_terminal_status_refuses_partial_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            records = [
                {"type": "clock_anchor", "mach_timebase_numer": 1, "mach_timebase_denom": 1},
                {"type": "cg_event", "cg_event_timestamp_ns_since_startup": 130},
                {
                    "type": "capture_end",
                    "observation_status": "inconclusive_zero_events",
                    "zero_event_sources": ["hid_report", "hid_value"],
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            result = probe.analyze_jsonl(path)
        self.assertEqual(result["empirical_status"], "inconclusive_zero_events")


if __name__ == "__main__":
    unittest.main()
