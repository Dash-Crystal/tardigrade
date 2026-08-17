from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from state_replay.integrator import StateIntegrator
from state_replay.total_capture_collection import (
    CollectionEpochInput,
    TotalCaptureCollectionIngestor,
)
from state_replay.total_capture_collection_contract import (
    TOTAL_CAPTURE_COLLECTION_SCHEMA,
    TOTAL_CAPTURE_COLLECTION_TERMINAL_SCHEMA,
    TOTAL_CAPTURE_EPOCH_CONTEXT_ENTITY_ID,
    epoch_key,
    epoch_stream_id,
    locate_complete_epoch,
    validate_epoch_context_component,
    validate_collection_manifest,
)
from state_replay.total_capture_contract import (
    TotalCaptureContractError,
    canonical_json,
    incomplete_capture_document,
    seal_document,
)
from state_replay.total_capture_monitor_handoff import (
    diagnostic_monitor_handoff,
    promote_verified_total_epoch,
    validate_monitor_handoff,
)
from tests import test_total_capture as total_capture_fixture


ZERO = "0" * 64
DEMO_NS = 31_250_000
CRASH_END = DEMO_NS + 10_000_000
GAP_END = CRASH_END + 10_000_000
THIRD_END = GAP_END + DEMO_NS
OUTER_END = 100_000_000


def shift_record_clocks(records, offset):
    shifted = []
    for source in records:
        item = copy.deepcopy(source)
        item["clock"]["monotonic_ns"] += offset
        shifted.append(seal_document(item, "record_sha256"))
    return shifted


def collection_fixture(root: Path):
    first_manifest, first_records, first_artifacts = total_capture_fixture.TotalCaptureTests().fixture(
        root / "first")
    third_manifest, third_records, third_artifacts = total_capture_fixture.TotalCaptureTests().fixture(
        root / "third")
    third_records = shift_record_clocks(third_records, GAP_END)
    crash = root / "crash.partial"
    crash.write_bytes(b"truncated demo and monitor evidence")
    crash_hash = hashlib.sha256(crash.read_bytes()).hexdigest()
    salvage = incomplete_capture_document(
        "outer-session/crash", "process exited before dem_stop", {"partial": crash_hash})
    epochs = [
        {
            "ordinal": 0, "process_epoch_id": "process:0",
            "map_epoch_id": "map:de_test:0", "capture_epoch_id": "capture:0",
            "map_name": "de_test",
            "status": "complete",
            "host_horizon": {"start_monotonic_ns": 0, "end_monotonic_ns": DEMO_NS},
            "reset": {"kind": "session-start", "previous_process_epoch_id": None,
                      "previous_map_epoch_id": None,
                      "previous_capture_epoch_id": None,
                      "reason": "monitor started target transcript"},
            "capture_manifest_sha256": first_manifest["manifest_sha256"],
            "salvage_sha256": None,
            "native_demo_sha256": first_manifest["artifacts"]["demo_sha256"],
            "demo_playback_ns": DEMO_NS,
        },
        {
            "ordinal": 1, "process_epoch_id": "process:0",
            "map_epoch_id": "map:de_test:0", "capture_epoch_id": "capture:1",
            "map_name": "de_test",
            "status": "crashed",
            "host_horizon": {"start_monotonic_ns": DEMO_NS,
                             "end_monotonic_ns": CRASH_END},
            "reset": {"kind": "record-rotation",
                      "previous_process_epoch_id": "process:0",
                      "previous_map_epoch_id": "map:de_test:0",
                      "previous_capture_epoch_id": "capture:0",
                      "reason": "monitor rotated native recording"},
            "capture_manifest_sha256": None,
            "salvage_sha256": salvage["salvage_sha256"],
            "native_demo_sha256": None,
            "demo_playback_ns": None,
        },
        {
            "ordinal": 2, "process_epoch_id": "process:1",
            "map_epoch_id": "map:de_test:1", "capture_epoch_id": "capture:2",
            "map_name": "de_test",
            "status": "complete",
            "host_horizon": {"start_monotonic_ns": GAP_END,
                             "end_monotonic_ns": THIRD_END},
            "reset": {"kind": "crash-recovery",
                      "previous_process_epoch_id": "process:0",
                      "previous_map_epoch_id": "map:de_test:0",
                      "previous_capture_epoch_id": "capture:1",
                      "reason": "target relaunched after crash"},
            "capture_manifest_sha256": third_manifest["manifest_sha256"],
            "salvage_sha256": None,
            "native_demo_sha256": third_manifest["artifacts"]["demo_sha256"],
            "demo_playback_ns": DEMO_NS,
        },
    ]
    intervals = [
        {"start_monotonic_ns": 0, "end_monotonic_ns": DEMO_NS,
         "state": "target-active", "process_epoch_id": "process:0",
         "map_epoch_id": "map:de_test:0", "capture_epoch_id": "capture:0",
         "reason": None},
        {"start_monotonic_ns": DEMO_NS, "end_monotonic_ns": CRASH_END,
         "state": "target-active", "process_epoch_id": "process:0",
         "map_epoch_id": "map:de_test:0", "capture_epoch_id": "capture:1",
         "reason": None},
        {"start_monotonic_ns": CRASH_END, "end_monotonic_ns": GAP_END,
         "state": "gap", "process_epoch_id": None, "map_epoch_id": None,
         "capture_epoch_id": None, "reason": "target crash/relaunch gap"},
        {"start_monotonic_ns": GAP_END, "end_monotonic_ns": THIRD_END,
         "state": "target-active", "process_epoch_id": "process:1",
         "map_epoch_id": "map:de_test:1", "capture_epoch_id": "capture:2",
         "reason": None},
        {"start_monotonic_ns": THIRD_END, "end_monotonic_ns": OUTER_END,
         "state": "target-inactive", "process_epoch_id": None,
         "map_epoch_id": None, "capture_epoch_id": None,
         "reason": "post-target monitor shutdown"},
    ]
    monitor_events = root / "outer-monitor-events.jsonl"
    monitor_events.write_text('{"kind":"outer_session_start","monotonic_ns":0}\n',
                              encoding="utf-8")
    ledger = seal_document({
        "schema": "tardigrade/total-capture-coverage-ledger/v1",
        "collection_id": "collection:one-hour-target",
        "clock": {"domain": "host-monotonic-ns", "start_monotonic_ns": 0,
                  "end_monotonic_ns": OUTER_END},
        "intervals": intervals,
    }, "ledger_sha256")
    ledger_path = root / "coverage-ledger.json"
    ledger_path.write_text(canonical_json(ledger) + "\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = seal_document({
        "schema": TOTAL_CAPTURE_COLLECTION_SCHEMA,
        "collection_id": "collection:one-hour-target",
        "request": {"requested_start_monotonic_ns": 0,
                    "required_target_active_ns": 60_000_000,
                    "close_policy": "target-active-threshold",
                    "max_wall_ns": None,
                    "target_executable_sha256": ZERO},
        "clock": {"domain": "host-monotonic-ns", "start_monotonic_ns": 0,
                  "end_monotonic_ns": OUTER_END},
        "coverage": {"intervals": intervals, "target_active_ns": 72_500_000,
                     "complete_target_active_ns": 62_500_000,
                     "target_inactive_ns": 17_500_000, "gap_ns": 10_000_000,
                     "corpus_status": "target-met",
                     "continuity_status": "gapped"},
        "epochs": epochs,
        "outer_artifacts": [
            {"id": "monitor-events", "role": "monitor-events",
             "sha256": digest(monitor_events)},
            {"id": "coverage-ledger", "role": "coverage-ledger",
             "sha256": digest(ledger_path)},
        ],
    }, "collection_manifest_sha256")
    terminal = seal_document({
        "schema": TOTAL_CAPTURE_COLLECTION_TERMINAL_SCHEMA,
        "collection_id": manifest["collection_id"],
        "collection_manifest_sha256": manifest["collection_manifest_sha256"],
        "end_monotonic_ns": OUTER_END, "corpus_status": "target-met",
        "continuity_status": "gapped", "close_reason": "target-met",
    }, "terminal_sha256")
    inputs = {
        epoch_key("process:0", "map:de_test:0", "capture:0"):
            CollectionEpochInput(first_manifest, first_records, first_artifacts),
        epoch_key("process:0", "map:de_test:0", "capture:1"):
            CollectionEpochInput(salvage=salvage,
                                 artifact_paths={"partial": crash}),
        epoch_key("process:1", "map:de_test:1", "capture:2"):
            CollectionEpochInput(third_manifest, third_records, third_artifacts),
    }
    return manifest, terminal, inputs, {
        "monitor-events": monitor_events, "coverage-ledger": ledger_path}


class TotalCaptureCollectionTests(unittest.TestCase):
    def test_current_monitor_diagnostic_is_salvage_not_collection_and_promotes_only_total(self):
        diagnostic = {
            "schema": "tardigrade/csgo-outer-monitor-plan/v1",
            "session_id": "diagnostic-session", "status": "closed-incomplete-total-capture",
            "terminal_reason": "target_active_gameplay",
            "claim": "not total capture; required native faculties unavailable",
            "segment_boundaries": [
                {"kind": "demo_lifecycle", "operation": "record-request",
                 "name": "demo_iabc_e1", "process_instance_id": "abc",
                 "map_epoch": 4, "segment_epoch": 1, "map": "de_test",
                 "monotonic_ns": 10},
                {"kind": "demo_lifecycle", "operation": "stop-request",
                 "name": "demo_iabc_e1", "process_instance_id": "abc",
                 "map_epoch": 4, "segment_epoch": 1, "map": "de_test",
                 "monotonic_ns": 20},
            ],
            "demo_segments": [{"name": "demo_iabc_e1", "status": "closed",
                               "sha256": "a" * 64}],
            "native_captures": [{"sha256": "b" * 64}],
            "hid": {"sha256": "c" * 64},
            "monitor_events": {"sha256": "d" * 64},
        }
        with self.assertRaisesRegex(TotalCaptureContractError, "collection manifest"):
            validate_collection_manifest(diagnostic)
        handoff = diagnostic_monitor_handoff(diagnostic)
        validate_monitor_handoff(handoff)
        self.assertEqual("salvage-only", handoff["status"])
        self.assertEqual("capture:abc:1", handoff["segments"][0]["capture_epoch_id"])
        self.assertEqual("blocked-missing-total-faculties",
                         handoff["segments"][0]["promotion_status"])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, records, artifacts = total_capture_fixture.TotalCaptureTests().fixture(root)
            promoted = promote_verified_total_epoch(
                ordinal=0, process_epoch_id="process:abc",
                map_epoch_id="map:abc:4", capture_epoch_id="capture:abc:2",
                host_horizon={"start_monotonic_ns": 0,
                              "end_monotonic_ns": DEMO_NS},
                reset={"kind": "session-start", "previous_process_epoch_id": None,
                       "previous_map_epoch_id": None,
                       "previous_capture_epoch_id": None, "reason": "verified total"},
                manifest=manifest, records=records, artifact_paths=artifacts)
            self.assertEqual(epoch_key("process:abc", "map:abc:4", "capture:abc:2"),
                             promoted.key)
            self.assertEqual(DEMO_NS, promoted.descriptor["demo_playback_ns"])

    def test_gapped_crash_collection_meets_target_without_bridging_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "first").mkdir()
            (root / "third").mkdir()
            manifest, terminal, inputs, outer_artifacts = collection_fixture(root)
            output = root / "collection"
            summary = TotalCaptureCollectionIngestor().ingest(
                manifest, terminal, inputs, output,
                outer_artifact_paths=outer_artifacts)
            self.assertEqual("target-met", summary.corpus_status)
            self.assertEqual("gapped", summary.continuity_status)
            self.assertEqual((2, 1), (summary.complete_epochs,
                                      summary.salvage_epochs))
            index = json.loads((output / "collection-index.json").read_text())
            self.assertEqual(["complete", "crashed", "complete"],
                             [item["status"] for item in index["epochs"]])
            self.assertIsNone(index["epochs"][1]["stream_id"])
            self.assertIsNone(index["epochs"][1]["journal"])
            for ordinal in (0, 2):
                stream_id = epoch_stream_id(
                    manifest["collection_id"], ordinal,
                    manifest["epochs"][ordinal]["capture_epoch_id"])
                journal = StateIntegrator.open(
                    output / index["epochs"][ordinal]["journal"], stream_id)
                try:
                    context = next(
                        item for item in journal.current_snapshot()["entities"]
                        if item["id"] == TOTAL_CAPTURE_EPOCH_CONTEXT_ENTITY_ID)
                    value = context["components"]["total_capture_epoch_context"]["value"]
                    validate_epoch_context_component(
                        context["components"]["total_capture_epoch_context"], manifest)
                    self.assertEqual(ordinal, value["ordinal"])
                    self.assertEqual(manifest["epochs"][ordinal]["capture_epoch_id"],
                                     value["capture_epoch_id"])
                    self.assertEqual("cmd:1",
                                     journal.action_context("cmd:1")["action"]["action_id"])
                finally:
                    journal.close()
            self.assertEqual(0, locate_complete_epoch(manifest, 50)["ordinal"])
            with self.assertRaisesRegex(TotalCaptureContractError, "salvage"):
                locate_complete_epoch(manifest, DEMO_NS + 5_000_000)
            with self.assertRaisesRegex(TotalCaptureContractError, "no state bridge"):
                locate_complete_epoch(manifest, CRASH_END + 5_000_000)

    def test_target_not_met_needs_explicit_salvage_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "first").mkdir()
            (root / "third").mkdir()
            manifest, _terminal, inputs, outer_artifacts = collection_fixture(root)
            manifest = copy.deepcopy(manifest)
            manifest["request"]["required_target_active_ns"] = 70_000_000
            manifest["coverage"]["corpus_status"] = "target-not-met"
            manifest = seal_document(manifest, "collection_manifest_sha256")
            terminal = seal_document({
                "schema": TOTAL_CAPTURE_COLLECTION_TERMINAL_SCHEMA,
                "collection_id": manifest["collection_id"],
                "collection_manifest_sha256": manifest["collection_manifest_sha256"],
                "end_monotonic_ns": OUTER_END, "corpus_status": "target-not-met",
                "continuity_status": "gapped", "close_reason": "user-stop",
            }, "terminal_sha256")
            with self.assertRaisesRegex(TotalCaptureContractError, "target not met"):
                TotalCaptureCollectionIngestor().ingest(
                    manifest, terminal, inputs, root / "refused",
                    outer_artifact_paths=outer_artifacts)
            self.assertFalse((root / "refused").exists())
            summary = TotalCaptureCollectionIngestor().ingest(
                manifest, terminal, inputs, root / "salvage-index",
                outer_artifact_paths=outer_artifacts,
                allow_target_not_met=True)
            self.assertEqual("target-not-met", summary.corpus_status)

    def test_false_coverage_and_late_epoch_failure_publish_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "first").mkdir()
            (root / "third").mkdir()
            manifest, terminal, inputs, outer_artifacts = collection_fixture(root)
            lie = copy.deepcopy(manifest)
            lie["coverage"]["complete_target_active_ns"] = 72_500_000
            lie = seal_document(lie, "collection_manifest_sha256")
            with self.assertRaisesRegex(TotalCaptureContractError,
                                        "includes incomplete"):
                validate_collection_manifest(lie)
            broken = copy.deepcopy(inputs)
            last_key = epoch_key("process:1", "map:de_test:1", "capture:2")
            broken[last_key].artifact_paths["demo"].write_bytes(b"corrupt late epoch")
            output = root / "atomic"
            with self.assertRaises(TotalCaptureContractError):
                TotalCaptureCollectionIngestor().ingest(
                    manifest, terminal, broken, output,
                    outer_artifact_paths=outer_artifacts)
            self.assertFalse(output.exists())
            self.assertEqual([], list(root.glob(".atomic.partial-*")))


if __name__ == "__main__":
    unittest.main()
