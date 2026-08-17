from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contract_gate import refusal
from inventory import DEFAULT_GAME_ROOT, DEFAULT_MANIFEST, inventory, require_allowlisted
from launch import LaunchRefusal, REPO, ensure_external_root, parser
import monitor as outer_monitor
from wire import CAPABILITIES, SCHEMA, WireError, read_records, terminal_summary, write_fixture


class WireTests(unittest.TestCase):
    def test_fixture_is_explicit_refusal_not_total_capture(self) -> None:
        with tempfile.TemporaryDirectory(dir="/Users/mdot/dox/runs") as temporary:
            path = Path(temporary) / "fixture.jsonl"
            write_fixture(path, "fixture-test")
            summary = terminal_summary(path)
            self.assertEqual("refused", summary["terminal"]["status"])
            self.assertEqual(set(CAPABILITIES), set(summary["capabilities"]))
            self.assertTrue(all(
                item["status"] == "fixture-only"
                for item in summary["capabilities"].values()
            ))
            self.assertNotEqual("tardigrade/total-capture-record/v1", SCHEMA)
            gate = refusal(path)
            self.assertFalse(gate["promotable"])
            self.assertIn("usercmd", gate["incomplete_capabilities"])

    def test_duplicate_capability_and_early_terminal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/Users/mdot/dox/runs") as temporary:
            original = Path(temporary) / "fixture.jsonl"
            write_fixture(original, "fixture-test")
            records = read_records(original)
            duplicate = Path(temporary) / "duplicate.jsonl"
            records.insert(-1, dict(records[1], ordinal=records[-1]["ordinal"]))
            records[-1] = dict(records[-1], ordinal=records[-1]["ordinal"] + 1)
            duplicate.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(WireError, "duplicate capability"):
                terminal_summary(duplicate)


class SafetyTests(unittest.TestCase):
    def test_output_root_resolution_rejects_repository_symlink(self) -> None:
        with tempfile.TemporaryDirectory(dir="/Users/mdot/dox/runs") as temporary:
            link = Path(temporary) / "repo-link"
            link.symlink_to(REPO, target_is_directory=True)
            with self.assertRaisesRegex(LaunchRefusal, "outside"):
                ensure_external_root(link)
            accepted = ensure_external_root(Path(temporary) / "capture")
            self.assertTrue(str(accepted).startswith("/Users/mdot/dox/runs/"))

    def test_launcher_has_no_passthrough_argument_surface(self) -> None:
        with self.assertRaises(SystemExit):
            parser().parse_args([
                "--consent", "--offline-only", "--insecure", "--map", "de_dust2",
                "--device", "0x046d:0xb041",
                "+connect", "example.invalid",
            ])

    def test_launcher_has_no_console_or_command_line_demo_recording(self) -> None:
        source = (Path(__file__).parent / "launch.py").read_text(encoding="utf-8")
        self.assertNotIn('"-console"', source)
        self.assertNotIn('"+record"', source)
        self.assertNotIn("last_growth", source)
        self.assertNotIn("producer_stall", source)

    def test_old_session_path_refuses_csgo_legacy(self) -> None:
        source = (REPO / "scripts" / "cs2_capture_session.py").read_text(encoding="utf-8")
        self.assertIn('if args.game_variant == "csgo_legacy":', source)
        self.assertIn("tools/csgo_total_capture/launch.py", source)

    def test_outer_monitor_activation_is_scoped_and_reversible(self) -> None:
        with tempfile.TemporaryDirectory(dir="/Users/mdot/dox/runs") as temporary:
            root = Path(temporary)
            prior = outer_monitor.ACTIVE_ROOT
            outer_monitor.ACTIVE_ROOT = root / "active"
            try:
                game = root / "game"
                (game / "csgo").mkdir(parents=True)
                session = root / "session"
                plugin = root / "capture.dylib"
                session.mkdir()
                plugin.write_bytes(b"fixture")
                activation = outer_monitor.Activation(
                    game, plugin, session, "outer-test", "tardigrade_outer_test"
                )
                record = activation.install()
                self.assertEqual("temporary Source server-plugin VDF auto-load descriptor",
                                 record["mechanism"])
                self.assertTrue(activation.vdf.is_file())
                self.assertTrue(activation.control.is_file())
                activation.remove()
                self.assertFalse(activation.vdf.exists())
                self.assertFalse(activation.control.exists())
            finally:
                outer_monitor.ACTIVE_ROOT = prior

    def test_coverage_is_focus_simulation_intersection(self) -> None:
        seconds = outer_monitor.interval_intersection_seconds(
            [(0, 10_000_000_000)],
            [(2_000_000_000, 4_000_000_000), (7_000_000_000, 8_000_000_000)],
            1_000_000_000,
            9_000_000_000,
        )
        self.assertEqual(3.0, seconds)

    def test_incremental_coverage_latches_complete_line_corruption(self) -> None:
        with tempfile.TemporaryDirectory(dir="/Users/mdot/dox/runs") as temporary:
            root = Path(temporary)
            hid = root / "hid.jsonl"
            hid.write_bytes(b'{"type":"clock_anchor"}\n{not-json}\n')
            coverage = outer_monitor.IncrementalCoverage(root, root, hid)
            coverage.poll()
            self.assertTrue(coverage.corruption_errors)
            self.assertEqual(0.0, coverage.measure()["active_gameplay_seconds"])


class ExactBuildTests(unittest.TestCase):
    def test_installed_legacy_build_matches_exact_allowlist(self) -> None:
        report = inventory(DEFAULT_GAME_ROOT, DEFAULT_MANIFEST)
        require_allowlisted(report)
        self.assertEqual("12426195", report["manifest"]["build_id"])
        self.assertEqual("csgo_legacy", report["manifest"]["beta_key"])
        self.assertTrue(report["matches_allowlist"])
        self.assertEqual(5, len(report["abi_code_anchors"]))
        self.assertTrue(all(item["matches"] for item in report["abi_code_anchors"].values()))

    def test_plugin_abi_matches_pinned_public_v004_order(self) -> None:
        source = (Path(__file__).parent / "capture_plugin.cpp").read_text(encoding="utf-8")
        ordered = [
            "ClientActive(edict_t *entity)",
            "ClientFullyConnect(edict_t *entity)",
            "ClientDisconnect(edict_t *entity)",
            "OnEdictAllocated(edict_t *edict)",
            "OnEdictFreed(const edict_t *edict)",
            "BNetworkCryptKeyCheckRequired(",
            "BNetworkCryptKeyValidate(",
        ]
        positions = [source.index(value) for value in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('"ISERVERPLUGINCALLBACKS004"', source)
        self.assertNotIn('strcmp(name, "ISERVERPLUGINCALLBACKS003")', source)

    def test_client_vtable_clone_preserves_itanium_abi_prefix(self) -> None:
        source = (Path(__file__).parent / "capture_plugin.cpp").read_text(encoding="utf-8")
        self.assertIn("original_vtable_[-2]", source)
        self.assertIn("original_vtable_[-1]", source)
        self.assertIn("cloned_vtable_storage_.data() + 2", source)

    def test_demo_coordinator_is_reentrant_and_epoch_named(self) -> None:
        source = (Path(__file__).parent / "capture_plugin.cpp").read_text(encoding="utf-8")
        self.assertNotIn("demo_start_requested_", source)
        self.assertNotIn("demo_stop_requested_", source)
        self.assertIn("segment_epoch_", source)
        self.assertIn('"_i" << g_process_instance << "_e"', source)
        self.assertIn('request_demo_stop("periodic_ten_minute_checkpoint")', source)
        self.assertIn("process_instance_id", source)
        self.assertIn('request_demo_stop("level_init")', source)
        self.assertIn('request_demo_stop("local_client_disconnect")', source)

    def test_hid_supports_successive_process_epochs(self) -> None:
        source = (REPO / "tools/cs2_ego_capture/Capture.swift").read_text(encoding="utf-8")
        self.assertIn("--outer-session-owner-pid", source)
        self.assertIn('"process_epoch": targetEpoch', source)
        self.assertIn("successive_matching_processes_until_outer-monitor-termination", source)

    def test_outer_target_uses_only_finalized_focused_coverage(self) -> None:
        source = (Path(__file__).parent / "monitor.py").read_text(encoding="utf-8")
        self.assertIn('gameplay["finalized_focused_simulating_seconds"] >=', source)
        self.assertIn("IncrementalCoverage", source)
        self.assertNotIn("def active_gameplay_measure", source)


if __name__ == "__main__":
    unittest.main()
