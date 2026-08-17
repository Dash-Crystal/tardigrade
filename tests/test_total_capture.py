from __future__ import annotations

import copy
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from state_replay.integrator import StateIntegrator
from state_replay.total_capture import TotalCaptureIngestor
from state_replay.total_capture_contract import (
    CAPTURE_COMPLETENESS_COMPONENT,
    TOTAL_CAPTURE_MANIFEST_SCHEMA,
    TOTAL_CAPTURE_METADATA_ENTITY_ID,
    TOTAL_CAPTURE_RECORD_SCHEMA,
    TotalCaptureContractError,
    ANGULAR_TARGET_DERIVATION,
    seal_document,
    validate_capture_completeness,
    validate_angular_target_history,
    validate_manifest,
    incomplete_capture_document,
    canonical_json,
)


ZERO = "0" * 64
KINDS = ["usercmd", "entity_lifecycle", "final_pose", "rigid_body", "effects",
         "camera", "ui", "visibility", "render_history"]
FIELDS = {
    "usercmd": ["applied_intent", "applied_outcome", "actor_kind",
                "hid_interval_integrals", "view_angle_anchors", "focus_segments"],
    "entity_lifecycle": ["entity_id", "generation", "class", "operation",
                         "world_transform", "component_state"],
    "final_pose": ["final_bones", "attachments", "viewmodels", "ragdolls",
                   "model_binding"],
    "rigid_body": ["bodies", "constraints"], "effects": ["instances"],
    "camera": ["pose", "fov", "projection", "prediction"],
    "ui": ["diegetic_ui"], "visibility": ["visible_entities"],
    "render_history": ["temporal_state"],
}


def recorded(value):
    return {"provenance": "recorded", "value": value}


def make_manifest(demo_hash, sidecar_hash, monitor_hash, terminal_hash):
    streams = []
    completeness = {}
    for kind in KINDS:
        stream_id = kind
        pov = "ego" if kind in {"camera", "ui", "visibility", "render_history"} else None
        fields = FIELDS[kind]
        streams.append({"id": stream_id, "kind": kind, "pov_id": pov,
                        "producer": "test-capture", "required": True,
                        "horizon": {"start_tick": 0, "end_tick": 2},
                        "fields": fields})
        completeness[stream_id] = {
            "status": "recorded",
            "fields": {field: "recorded" for field in fields}}
    return seal_document({
        "schema": TOTAL_CAPTURE_MANIFEST_SCHEMA, "capture_id": "capture-1",
        "engine": {"name": "source1/csgo", "build_id": "test-build",
                   "build_sha256": ZERO,
                   "content_manifest_sha256": "1" * 64},
        "artifacts": {"demo_sha256": demo_hash,
                      "sidecars": [{"id": "content", "sha256": sidecar_hash,
                                    "role": "content-package"},
                                   {"id": "monitor", "sha256": monitor_hash,
                                    "role": "capture-stream"}]},
        "producer_closure": {
            "ownership": "monitor-owned-native-demo",
            "capture_session_id": "session-1",
            "monitor_sidecar_id": "monitor",
            "native_demo": {"format": "source1-hl2demo-v4",
                            "terminal_command": "dem_stop", "terminal_tick": 2,
                            "playback_ticks": 2, "map_name": "de_test",
                            "server_name": "local"},
            "sidecars": {"monitor": {
                "terminal_schema": "tardigrade/capture-sidecar-terminal/v1",
                "terminal_sha256": terminal_hash, "start_tick": 0,
                "end_tick": 2, "capture_session_id": "session-1"}},
        },
        "tick_rate_hz": 64, "checkpoint_interval_ticks": 2,
        "horizon": {"start_tick": 0, "end_tick": 2},
        "scored_povs": ["ego"], "streams": streams,
        "completeness": completeness,
    }, "manifest_sha256")


def component(kind):
    if kind == "camera":
        return recorded({"pov_id": "ego", "vantage": "first-person",
                         "origin": [0, 0, 64], "yaw_degrees": 0,
                         "pitch_degrees": 0, "roll_degrees": 0, "fov": 90,
                         "projection": "perspective", "near": 1, "far": 4096,
                         "clock": {"monotonic_ns": 6, "engine_tick": 0,
                                   "source_sequence": 0},
                         "view_matrix_4x4": [1, 0, 0, 0, 0, 1, 0, 0,
                                             0, 0, 1, 0, 0, 0, 0, 1],
                         "projection_matrix_4x4": [1, 0, 0, 0, 0, 1, 0, 0,
                                                   0, 0, 1, 0, 0, 0, 0, 1],
                         "viewport": [1920, 1080],
                         "matrix_convention": {
                             "layout": "row-major", "vectors": "column-vectors",
                             "view_handedness": "right-handed",
                             "view_axis_order": ["right", "up", "forward"],
                             "clip_depth_range": "negative-one-to-one",
                             "ndc_y_direction": "up",
                             "view_transform_direction": "world-to-view"},
                         "prediction": {"mode": "recorded-final",
                                        "command_number": 0, "predicted_tick": 0,
                                        "source_sequence": 0,
                                        "prediction_error": [0, 0, 0],
                                        "corrected": False}})
    if kind == "ui":
        return recorded({"pov_id": "ego", "viewport": [1920, 1080],
                         "elements": []})
    if kind == "visibility":
        return recorded({"pov_id": "ego", "visible_entity_ids": [],
                         "method": "recorded-render-list"})
    return recorded({"pov_id": "ego", "samples": []})


def make_records(manifest):
    records = []
    clock = 1
    for kind in KINDS:
        ops = []
        if kind == "entity_lifecycle":
            ops = [{"op": "create", "entity": {
                "id": "total-capture:pov:ego", "generation": 0,
                "class": "CapturedPOV", "components": {"score": recorded(0)}}}]
        elif kind in {"camera", "ui", "visibility", "render_history"}:
            name = "diegetic_ui" if kind == "ui" else kind
            ops = [{"op": "update", "id": "total-capture:pov:ego",
                    "generation": 0, "components": {name: component(kind)}}]
        records.append(record(manifest, kind, "checkpoint", 0, 0, 0, clock,
                              {"full": True, "ops": ops}))
        clock += 1
    intent = {"action_id": "cmd:1", "kind": "usercmd.applied",
              "payload": {"buttons": ["attack"], "mouse_dx": 3}}
    records.append(record(manifest, "usercmd", "delta", 1, 1, 0, clock, {
        "actor_kind": "human", "actor_id": "player:ego", "intent": intent,
        "hid_interval": {"start_monotonic_ns": 100, "end_monotonic_ns": 200,
                         "relative_counts": [3, -1], "report_sequence_start": 9,
                         "report_sequence_end": 12},
        "view_angle_anchor": {"kind": "usercmd_target", "yaw_degrees": 12,
                              "pitch_degrees": -2, "roll_degrees": 0},
        "focus_segment_id": "focus:0",
        "focus_discontinuity_before": True,
        "applied": True, "outcome_ops": [{"op": "update",
            "id": "total-capture:pov:ego", "generation": 0,
            "components": {"score": recorded(1)}}]}))
    clock += 1
    for kind in KINDS:
        seq = 2 if kind == "usercmd" else 1
        records.append(record(manifest, kind, "end", seq, 2, 0, clock,
                              {"status": "complete"}))
        clock += 1
    return records


def record(manifest, stream, kind, sequence, tick, subtick, clock, payload):
    return seal_document({
        "schema": TOTAL_CAPTURE_RECORD_SCHEMA,
        "capture_id": manifest["capture_id"],
        "manifest_sha256": manifest["manifest_sha256"], "stream_id": stream,
        "type": kind, "sequence": sequence, "tick": tick, "subtick": subtick,
        "clock": {"monotonic_ns": clock, "engine_tick": tick,
                  "source_sequence": sequence}, "payload": payload,
    }, "record_sha256")


class TotalCaptureTests(unittest.TestCase):
    def fixture(self, root):
        demo, content = root / "demo.dem", root / "content.pack"
        monitor = root / "monitor.jsonl"
        field = lambda value: value.encode() + b"\0" * (260 - len(value))
        demo.write_bytes(
            struct.pack("<8sii260s260s260s260sfiii", b"HL2DEMO\0", 4, 13790,
                        field("local"), field("ego"), field("de_test"),
                        field("csgo"), 2 / 64, 2, 1, 0) +
            struct.pack("<BiB", 7, 2, 0))
        content.write_bytes(b"content-addressed resources")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        terminal = seal_document({
            "schema": "tardigrade/capture-sidecar-terminal/v1",
            "sidecar_id": "monitor", "capture_session_id": "session-1",
            "start_tick": 0, "end_tick": 2, "status": "complete",
            "native_demo_sha256": digest(demo),
        }, "terminal_sha256")
        monitor.write_text(canonical_json(terminal) + "\n", encoding="utf-8")
        manifest = make_manifest(digest(demo), digest(content), digest(monitor),
                                 terminal["terminal_sha256"])
        return manifest, make_records(manifest), {
            "demo": demo, "content": content, "monitor": monitor}

    def test_ingest_reopen_hash_closure_and_action_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, records, artifacts = self.fixture(root)
            output = root / "journal"
            summary = TotalCaptureIngestor().ingest(
                manifest, records, artifacts, output, "capture")
            reopened = StateIntegrator.open(output, "capture")
            try:
                snapshot = reopened.current_snapshot()
                self.assertEqual(summary.final_state_hash, snapshot["state_hash"])
                metadata = next(entity for entity in snapshot["entities"]
                                if entity["id"] == TOTAL_CAPTURE_METADATA_ENTITY_ID)
                closure = metadata["components"][CAPTURE_COMPLETENESS_COMPONENT]
                self.assertEqual("complete", closure["value"]["status"])
                validate_capture_completeness(manifest, closure["value"])
                action = reopened.action_context("cmd:1")
                self.assertNotEqual(action["pre"]["state_hash"],
                                    action["post"]["state_hash"])
                self.assertEqual("human", action["action"]["actor_kind"])
                at_start = reopened.state_at(0)
                pov = next(entity for entity in at_start["entities"]
                           if entity["id"] == "total-capture:pov:ego")
                self.assertIn("camera", pov["components"])
            finally:
                reopened.close()

    def test_hash_gap_incomplete_and_artifact_fail_without_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, records, artifacts = self.fixture(root)
            cases = []
            corrupt = copy.deepcopy(records)
            corrupt[-1]["payload"] = {"status": "not-complete"}
            cases.append(corrupt)
            incomplete = [r for r in records
                          if not (r["stream_id"] == "camera" and r["type"] == "end")]
            cases.append(incomplete)
            duplicate = copy.deepcopy(records)
            duplicate.append(copy.deepcopy(records[-1]))
            cases.append(duplicate)
            for index, bad in enumerate(cases):
                output = root / f"bad-{index}"
                with self.assertRaises(TotalCaptureContractError):
                    TotalCaptureIngestor().ingest(manifest, bad, artifacts, output)
                self.assertFalse(output.exists())
            artifacts["demo"].write_bytes(b"wrong")
            output = root / "bad-artifact"
            with self.assertRaisesRegex(TotalCaptureContractError, "sha256 mismatch"):
                TotalCaptureIngestor().ingest(manifest, records, artifacts, output)
            self.assertFalse(output.exists())

    def test_manifest_rejects_missing_scored_pov_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _, _ = self.fixture(Path(temp))
            bad = copy.deepcopy(manifest)
            bad["streams"] = [s for s in bad["streams"] if s["kind"] != "camera"]
            bad["completeness"].pop("camera")
            bad = seal_document(bad, "manifest_sha256")
            with self.assertRaisesRegex(TotalCaptureContractError, "scored POV"):
                validate_manifest(bad)

    def test_hashed_crash_partial_demo_is_salvage_only_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, records, artifacts = self.fixture(root)
            demo = artifacts["demo"]
            demo.write_bytes(demo.read_bytes()[:-6])
            partial = copy.deepcopy(manifest)
            partial["artifacts"]["demo_sha256"] = hashlib.sha256(
                demo.read_bytes()).hexdigest()
            partial = seal_document(partial, "manifest_sha256")
            output = root / "partial"
            with self.assertRaisesRegex(TotalCaptureContractError, "dem_stop"):
                TotalCaptureIngestor().ingest(partial, records,
                                              artifacts, output)
            self.assertFalse(output.exists())
            salvage = incomplete_capture_document(
                "session-1", "native process crashed before dem_stop",
                {"demo": partial["artifacts"]["demo_sha256"]})
            self.assertEqual("incomplete", salvage["status"])

    def test_angular_history_is_derived_from_integrals_and_exact_anchors(self):
        envelope = {"provenance": "derived",
                    "derivation": ANGULAR_TARGET_DERIVATION,
                    "value": {
            "focus_segment_id": "focus:0",
            "hid_intervals": [{"start_monotonic_ns": 100,
                               "end_monotonic_ns": 200,
                               "relative_counts": [7, -2],
                               "report_sequence_start": 1,
                               "report_sequence_end": 4}],
            "anchors": [
                {"clock": {"monotonic_ns": 100, "engine_tick": 1,
                           "source_sequence": 1},
                 "angle": {"kind": "usercmd_target", "yaw_degrees": 10,
                           "pitch_degrees": 2, "roll_degrees": 0}},
                {"clock": {"monotonic_ns": 200, "engine_tick": 2,
                           "source_sequence": 2},
                 "angle": {"kind": "render_camera", "yaw_degrees": 11,
                           "pitch_degrees": 1, "roll_degrees": 0}}],
            "samples": [
                {"monotonic_ns": 100, "yaw_degrees": 10,
                 "pitch_degrees": 2, "roll_degrees": 0},
                {"monotonic_ns": 150, "yaw_degrees": 10.5,
                 "pitch_degrees": 1.5, "roll_degrees": 0},
                {"monotonic_ns": 200, "yaw_degrees": 11,
                 "pitch_degrees": 1, "roll_degrees": 0}],
            "source_artifact_sha256s": [ZERO],
        }}
        validate_angular_target_history(envelope)
        bad = copy.deepcopy(envelope)
        bad["value"]["samples"][-1]["yaw_degrees"] = 999
        with self.assertRaisesRegex(TotalCaptureContractError, "pass through"):
            validate_angular_target_history(bad)


if __name__ == "__main__":
    unittest.main()
