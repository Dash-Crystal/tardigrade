from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

from state_replay import state_hash
from state_replay.source1_adapter import CanonicalSource1Reducer, ReductionContext
from state_replay.source1_entities import EntityChange, EntityState, StringTableState
from state_replay.total_capture_contract import (
    ANGULAR_TARGET_DERIVATION,
    COMPLETENESS_DERIVATION,
    GLOBAL_REQUIRED_KINDS,
    POV_STREAM_KINDS,
    STREAM_REQUIRED_FIELDS,
    TOTAL_CAPTURE_MANIFEST_SCHEMA,
    capture_completeness_document,
    incomplete_capture_document,
    seal_document,
)
from state_replay.total_capture_collection_contract import (
    TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA,
    TOTAL_CAPTURE_COLLECTION_SCHEMA,
    TOTAL_CAPTURE_COLLECTION_TERMINAL_SCHEMA,
    epoch_context_entity,
    epoch_stream_id,
)

from counter_strike_render.source1_backend import (
    Source1RenderConfig,
    Source1RenderRefusal,
    MODEL_BINDING_DERIVATION,
    backend_descriptor,
    render_batch,
    render_collection,
    render_session,
    render_snapshot,
)
from tests.test_source1_render_formats import _fixture_bsp, _fixture_model_triplet


def _component(value, provenance="recorded", **extra):
    return {"provenance": provenance, "value": value, **extra}


def _snapshot(*, model=None, animation=None, include_fov=True):
    geometry = {
        "space": "world",
        "vertices": [[4.0, -1.0, -1.0], [4.0, 1.0, -1.0], [4.0, 0.0, 1.0]],
        "triangles": [[0, 1, 2]],
        "color": [201, 93, 41],
    }
    visual_components = (
        {"model": _component(model)}
        if model is not None
        else {"source1_geometry": _component(geometry)}
    )
    if animation is not None:
        visual_components["animation"] = _component(animation)
    camera_value = {
        "origin": [0.0, 0.0, 0.0],
        "yaw_degrees": 0.0,
        "pitch_degrees": 0.0,
    }
    if include_fov:
        camera_value["fov"] = 90.0
    result = {
        "schema": "tardigrade/state-snapshot/v1",
        "stream_id": "match/csgo-legacy",
        "tick": 91,
        "subtick": 2,
        "state_hash": "pending",
        "entities": [
            {
                "id": "camera",
                "generation": 0,
                "class": "camera",
                "components": {
                    "camera": _component(camera_value)
                },
            },
            {
                "id": "fixture-triangle",
                "generation": 0,
                "class": "source1_reference_fixture",
                "components": visual_components,
            },
        ],
    }
    result["state_hash"] = state_hash(result["entities"])
    return result


def _write_model(root: Path) -> None:
    mdl, vvd, vtx = _fixture_model_triplet()
    models = root / "models"
    models.mkdir(parents=True)
    (models / "fixture.mdl").write_bytes(mdl)
    (models / "fixture.vvd").write_bytes(vvd)
    (models / "fixture.dx90.vtx").write_bytes(vtx)


def _vpk_files(entries):
    by_extension = {}
    for path, payload in entries.items():
        directory, filename = path.rsplit("/", 1)
        stem, extension = filename.rsplit(".", 1)
        by_extension.setdefault(extension, {}).setdefault(directory, []).append(
            (stem, payload)
        )
    tree = bytearray()
    for extension in sorted(by_extension):
        tree += extension.encode() + b"\0"
        for directory in sorted(by_extension[extension]):
            tree += directory.encode() + b"\0"
            for stem, payload in sorted(by_extension[extension][directory]):
                tree += stem.encode() + b"\0"
                tree += struct.pack(
                    "<IHHIIH",
                    zlib.crc32(payload) & 0xFFFFFFFF,
                    len(payload),
                    0x7FFF,
                    0,
                    0,
                    0xFFFF,
                )
                tree += payload
            tree += b"\0"
        tree += b"\0"
    tree += b"\0"
    return struct.pack("<III", 0x55AA1234, 1, len(tree)) + bytes(tree)


def _reducer_model_contract():
    """Return the real resource/model documents emitted by the reducer."""
    reducer = CanonicalSource1Reducer()
    decoder = SimpleNamespace(entities={})
    context = ReductionContext(0, 0, 0)
    table = StringTableState(
        5,
        "modelprecache",
        1024,
        False,
        0,
        0,
        {3: ("models/fixture.mdl", b"")},
    )
    table_ops = reducer.on_string_table(
        "create", table, (3,), context, decoder
    )
    resource = copy.deepcopy(table_ops[0]["entity"])
    entity = EntityState(
        0, 0, "CTest", 1, {"baseclass.m_nModelIndex": 3}, True, True
    )
    decoder.entities = {0: entity}
    change = EntityChange("enter", 0, 0, "CTest", 1, {}, True)
    entity_ops = reducer.on_entities((change,), decoder, context)
    binding = copy.deepcopy(
        entity_ops[0]["entity"]["components"]["source1_model"]
    )
    binding["value"]["bodygroups"] = [0]
    return resource, binding


def _snapshot_with_reducer_model():
    resource, binding = _reducer_model_contract()
    snapshot = _snapshot(model={"source1_mdl": "unused.mdl"}, animation=_final_bones())
    components = snapshot["entities"][1]["components"]
    components["source1_model"] = binding
    del components["model"]
    snapshot["entities"].append(resource)
    snapshot["state_hash"] = state_hash(snapshot["entities"])
    return snapshot


def _final_bones():
    return {
        "model_checksum": 0x12345678,
        "matrix_kind": "source1_bone_to_world_skinning_3x4",
        "evaluated_bones": [{
            "name": "root",
            "matrix3x4": [
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }],
    }


def _total_capture_snapshot(pov_ids=("bot-a", "bot-b")):
    streams = []
    completeness = {}
    for kind in sorted(GLOBAL_REQUIRED_KINDS):
        stream_id = f"global:{kind}"
        streams.append({
            "id": stream_id, "kind": kind, "pov_id": None,
            "producer": "renderer-test", "required": True,
            "horizon": {"start_tick": 0, "end_tick": 100},
            "fields": sorted(STREAM_REQUIRED_FIELDS[kind]),
        })
        completeness[stream_id] = {
            "status": "recorded",
            "fields": {
                field: "recorded" for field in STREAM_REQUIRED_FIELDS[kind]
            },
        }
    for pov_id in pov_ids:
        for kind in sorted(POV_STREAM_KINDS):
            stream_id = f"{pov_id}:{kind}"
            streams.append({
                "id": stream_id, "kind": kind, "pov_id": pov_id,
                "producer": "renderer-test", "required": True,
                "horizon": {"start_tick": 0, "end_tick": 100},
                "fields": sorted(STREAM_REQUIRED_FIELDS[kind]),
            })
            completeness[stream_id] = {
                "status": "recorded",
                "fields": {
                    field: "recorded" for field in STREAM_REQUIRED_FIELDS[kind]
                },
            }
    manifest = seal_document({
        "schema": TOTAL_CAPTURE_MANIFEST_SCHEMA,
        "capture_id": "renderer-total-capture",
        "engine": {
            "name": "source1/csgo", "build_id": "legacy-test-build",
            "build_sha256": "1" * 64,
            "content_manifest_sha256": "2" * 64,
        },
        "artifacts": {
            "demo_sha256": "3" * 64,
            "sidecars": [{
                "id": "renderer-state", "sha256": "4" * 64,
                "role": "capture-stream",
            }],
        },
        "producer_closure": {
            "ownership": "monitor-owned-native-demo",
            "capture_session_id": "renderer-session",
            "monitor_sidecar_id": "renderer-state",
            "native_demo": {
                "format": "source1-hl2demo-v4",
                "terminal_command": "dem_stop",
                "terminal_tick": 100,
                "playback_ticks": 100,
                "map_name": "de_renderer_test",
                "server_name": "renderer-test-server",
            },
            "sidecars": {
                "renderer-state": {
                    "terminal_schema": "tardigrade/capture-sidecar-terminal/v1",
                    "terminal_sha256": "8" * 64,
                    "start_tick": 0,
                    "end_tick": 100,
                    "capture_session_id": "renderer-session",
                },
            },
        },
        "tick_rate_hz": 64,
        "checkpoint_interval_ticks": 32,
        "horizon": {"start_tick": 0, "end_tick": 100},
        "scored_povs": list(pov_ids),
        "streams": streams,
        "completeness": completeness,
    }, "manifest_sha256")
    terminals = {item["id"]: (2, "5" * 64) for item in streams}
    closure = capture_completeness_document(
        manifest, closed=True, terminals=terminals
    )
    geometry = copy.deepcopy(_snapshot()["entities"][1])
    geometry["id"] = "scored-triangle"
    entities = [geometry, {
        "id": "total-capture:metadata", "generation": 0,
        "class": "TotalCaptureMetadata",
        "components": {
            "total_capture_manifest": _component(manifest),
            "capture_completeness": _component(
                closure, "derived", derivation=COMPLETENESS_DERIVATION
            ),
        },
    }]
    for index, pov_id in enumerate(pov_ids):
        entities.append({
            "id": f"total-capture:pov:{pov_id}", "generation": 0,
            "class": "CapturedPOV",
            "components": {
                "camera": _component({
                    "pov_id": pov_id, "vantage": "first-person",
                    "origin": [0.0, float(index) * 0.75, 0.0],
                    "yaw_degrees": 0.0, "pitch_degrees": 0.0,
                    "roll_degrees": 0.0, "fov": 90.0,
                    "projection": "perspective", "near": 0.01, "far": 4096.0,
                    "viewport": [64, 40],
                    "view_matrix_4x4": [
                        1, 0, 0, 0,
                        0, 1, 0, -float(index) * 0.75,
                        0, 0, 1, 0,
                        0, 0, 0, 1,
                    ],
                    "projection_matrix_4x4": [
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        1, 0, 0, 0,
                        1, 0, 0, 0,
                    ],
                    "matrix_convention": {
                        "layout": "row-major", "vectors": "column-vectors",
                        "view_handedness": "left-handed",
                        "view_axis_order": ["forward", "right", "up"],
                        "clip_depth_range": "zero-to-one",
                        "ndc_y_direction": "up",
                        "view_transform_direction": "world-to-view",
                    },
                    "clock": {"monotonic_ns": 2000, "engine_tick": 91,
                              "source_sequence": 91},
                    "prediction": {
                        "mode": "recorded-final", "command_number": 91,
                        "predicted_tick": 91, "source_sequence": 91,
                        "prediction_error": [0.0, 0.0, 0.0],
                        "corrected": False,
                    },
                }),
                "diegetic_ui": _component({
                    "pov_id": pov_id, "viewport": [64, 40], "elements": [],
                }),
                "visibility": _component({
                    "pov_id": pov_id,
                    "visible_entity_ids": ["scored-triangle"],
                    "method": "recorded-render-list",
                }),
                "render_history": _component({
                    "pov_id": pov_id, "samples": [],
                }),
                "angular_target_history": _component(
                    {
                        "focus_segment_id": f"focus:{pov_id}",
                        "hid_intervals": [],
                        "anchors": [
                            {
                                "clock": {"monotonic_ns": 1000,
                                          "engine_tick": 90,
                                          "source_sequence": 90},
                                "angle": {"kind": "render_camera",
                                          "yaw_degrees": 0.0,
                                          "pitch_degrees": 0.0,
                                          "roll_degrees": 0.0},
                            },
                            {
                                "clock": {"monotonic_ns": 2000,
                                          "engine_tick": 91,
                                          "source_sequence": 91},
                                "angle": {"kind": "render_camera",
                                          "yaw_degrees": 0.0,
                                          "pitch_degrees": 0.0,
                                          "roll_degrees": 0.0},
                            },
                        ],
                        "samples": [
                            {"monotonic_ns": 1000, "yaw_degrees": 0.0,
                             "pitch_degrees": 0.0, "roll_degrees": 0.0},
                            {"monotonic_ns": 2000, "yaw_degrees": 0.0,
                             "pitch_degrees": 0.0, "roll_degrees": 0.0},
                        ],
                        "source_artifact_sha256s": ["6" * 64],
                    },
                    "derived",
                    derivation=ANGULAR_TARGET_DERIVATION,
                ),
            },
        })
    snapshot = {
        "schema": "tardigrade/state-snapshot/v1",
        "stream_id": "total/csgo", "tick": 91, "subtick": 0,
        "state_hash": "pending", "entities": entities,
    }
    snapshot["state_hash"] = state_hash(entities)
    return snapshot, manifest


def _capture_collection(*, with_salvage=True):
    first, capture_manifest = _total_capture_snapshot(("ego",))
    salvage = incomplete_capture_document(
        "collection-session-1", "recording stopped before dem_stop",
        {"partial-demo": "a" * 64},
    )
    second_status = "crashed" if with_salvage else "complete"
    second_start = 250 if with_salvage else 200
    epochs = [
        {
            "ordinal": 0,
            "process_epoch_id": "process-7",
            "map_epoch_id": "map-de_dust2-4",
            "capture_epoch_id": "demo-segment-11",
            "map_name": "de_renderer_test",
            "status": "complete",
            "host_horizon": {
                "start_monotonic_ns": 100,
                "end_monotonic_ns": 200,
            },
            "reset": {
                "kind": "session-start",
                "previous_process_epoch_id": None,
                "previous_map_epoch_id": None,
                "previous_capture_epoch_id": None,
                "reason": "capture request began",
            },
            "capture_manifest_sha256": capture_manifest["manifest_sha256"],
            "salvage_sha256": None,
            "native_demo_sha256": capture_manifest["artifacts"]["demo_sha256"],
            "demo_playback_ns": 100,
        },
        {
            "ordinal": 1,
            "process_epoch_id": "process-7",
            "map_epoch_id": "map-de_dust2-4",
            "capture_epoch_id": "demo-segment-12",
            "map_name": "de_renderer_test",
            "status": second_status,
            "host_horizon": {
                "start_monotonic_ns": second_start,
                "end_monotonic_ns": 400,
            },
            "reset": {
                "kind": "record-rotation",
                "previous_process_epoch_id": "process-7",
                "previous_map_epoch_id": "map-de_dust2-4",
                "previous_capture_epoch_id": "demo-segment-11",
                "reason": "native demo segment rotated",
            },
            "capture_manifest_sha256": (
                None if with_salvage else capture_manifest["manifest_sha256"]
            ),
            "salvage_sha256": salvage["salvage_sha256"] if with_salvage else None,
            "native_demo_sha256": (
                None if with_salvage else capture_manifest["artifacts"]["demo_sha256"]
            ),
            "demo_playback_ns": None if with_salvage else 200,
        },
    ]
    intervals = [{
        "start_monotonic_ns": 100,
        "end_monotonic_ns": 200,
        "state": "target-active",
        "process_epoch_id": "process-7",
        "map_epoch_id": "map-de_dust2-4",
        "capture_epoch_id": "demo-segment-11",
        "reason": None,
    }]
    if with_salvage:
        intervals.append({
            "start_monotonic_ns": 200,
            "end_monotonic_ns": 250,
            "state": "gap",
            "process_epoch_id": None,
            "map_epoch_id": None,
            "capture_epoch_id": None,
            "reason": "demo rotation lost terminal closure",
        })
    intervals.append({
        "start_monotonic_ns": second_start,
        "end_monotonic_ns": 400,
        "state": "target-active",
        "process_epoch_id": "process-7",
        "map_epoch_id": "map-de_dust2-4",
        "capture_epoch_id": "demo-segment-12",
        "reason": None,
    })
    total_active = 250 if with_salvage else 300
    complete_active = 100 if with_salvage else 300
    manifest = seal_document({
        "schema": TOTAL_CAPTURE_COLLECTION_SCHEMA,
        "collection_id": "hour-corpus-1",
        "request": {
            "requested_start_monotonic_ns": 100,
            "required_target_active_ns": 80,
            "close_policy": "target-active-threshold",
            "max_wall_ns": None,
            "target_executable_sha256": "b" * 64,
        },
        "clock": {
            "domain": "host-monotonic-ns",
            "start_monotonic_ns": 100,
            "end_monotonic_ns": 400,
        },
        "outer_artifacts": [
            {
                "id": "renderer-monitor-events",
                "role": "monitor-events",
                "sha256": "c" * 64,
            },
            {
                "id": "renderer-coverage-ledger",
                "role": "coverage-ledger",
                "sha256": "d" * 64,
            },
        ],
        "coverage": {
            "intervals": intervals,
            "target_active_ns": total_active,
            "complete_target_active_ns": complete_active,
            "target_inactive_ns": 0,
            "gap_ns": 50 if with_salvage else 0,
            "corpus_status": "target-met",
            "continuity_status": "gapped" if with_salvage else "gap-free",
        },
        "epochs": epochs,
    }, "collection_manifest_sha256")
    terminal = seal_document({
        "schema": TOTAL_CAPTURE_COLLECTION_TERMINAL_SCHEMA,
        "collection_id": manifest["collection_id"],
        "collection_manifest_sha256": manifest["collection_manifest_sha256"],
        "end_monotonic_ns": 400,
        "corpus_status": "target-met",
        "continuity_status": "gapped" if with_salvage else "gap-free",
        "close_reason": "user-stop",
    }, "terminal_sha256")

    first["stream_id"] = epoch_stream_id(
        manifest["collection_id"], 0, epochs[0]["capture_epoch_id"]
    )
    first["sequence"] = 0
    first["entities"].append(epoch_context_entity(manifest, 0))
    first["state_hash"] = state_hash(first["entities"])
    entries = [{
        "ordinal": 0,
        "process_epoch_id": epochs[0]["process_epoch_id"],
        "map_epoch_id": epochs[0]["map_epoch_id"],
        "capture_epoch_id": epochs[0]["capture_epoch_id"],
        "status": "complete",
        "stream_id": first["stream_id"],
        "frames": [first],
        "salvage": None,
    }]
    if with_salvage:
        entries.append({
            "ordinal": 1,
            "process_epoch_id": epochs[1]["process_epoch_id"],
            "map_epoch_id": epochs[1]["map_epoch_id"],
            "capture_epoch_id": epochs[1]["capture_epoch_id"],
            "status": "crashed",
            "stream_id": None,
            "frames": [],
            "salvage": salvage,
        })
    else:
        second, _ = _total_capture_snapshot(("ego",))
        camera = second["entities"][2]["components"]["camera"]["value"]
        camera["origin"][1] = 0.75
        camera["view_matrix_4x4"][7] = -0.75
        second["stream_id"] = epoch_stream_id(
            manifest["collection_id"], 1, epochs[1]["capture_epoch_id"]
        )
        second["sequence"] = 0
        second["entities"].append(epoch_context_entity(manifest, 1))
        second["state_hash"] = state_hash(second["entities"])
        entries.append({
            "ordinal": 1,
            "process_epoch_id": epochs[1]["process_epoch_id"],
            "map_epoch_id": epochs[1]["map_epoch_id"],
            "capture_epoch_id": epochs[1]["capture_epoch_id"],
            "status": "complete",
            "stream_id": second["stream_id"],
            "frames": [second],
            "salvage": None,
        })
    return {
        "schema": TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA,
        "manifest": manifest,
        "terminal": terminal,
        "epochs": entries,
    }, capture_manifest


class Source1BackendTests(unittest.TestCase):
    def test_descriptor_is_registry_ready_and_honest(self):
        descriptor = backend_descriptor()
        self.assertEqual(descriptor["engine"], "source1")
        self.assertEqual(
            descriptor["entrypoint"],
            "counter_strike_render.source1_backend:render_session",
        )
        self.assertIn("mdl48/49-vvd4-vtx7-lod0-skinning",
                      descriptor["implemented"])
        with self.assertRaises(TypeError):
            descriptor["engine"] = "source2"

    def test_reference_raster_is_nonempty_and_deterministic(self):
        config = Source1RenderConfig(
            width=64, height=40, allow_synthetic_geometry=True
        )
        first = render_snapshot(_snapshot(), config)
        second = render_snapshot(_snapshot(), config)

        self.assertEqual(first.ppm, second.ppm)
        self.assertEqual(first.color_sha256, second.color_sha256)
        self.assertEqual(first.depth_sha256, second.depth_sha256)
        self.assertEqual(first.scene.state_hash, _snapshot()["state_hash"])
        self.assertTrue(first.ppm.startswith(b"P6\n64 40\n255\n"))
        payload = first.ppm.split(b"\n", 3)[3]
        self.assertIn(bytes((201, 93, 41)), payload)

    def test_total_capture_scored_povs_produce_distinct_joined_outputs(self):
        snapshot, manifest = _total_capture_snapshot()
        rendered = render_batch(
            [snapshot],
            Source1RenderConfig(
                width=64, height=40, allow_synthetic_geometry=True
            ),
        )
        self.assertEqual(["bot-a", "bot-b"], [
            item.scene.camera.vantage_id for item in rendered
        ])
        self.assertNotEqual(rendered[0].ppm, rendered[1].ppm)
        self.assertNotEqual(
            rendered[0].scene.vantage_state_hash,
            rendered[1].scene.vantage_state_hash,
        )
        self.assertEqual(
            {snapshot["state_hash"]}, {item.scene.state_hash for item in rendered}
        )
        self.assertEqual(
            manifest["manifest_sha256"],
            rendered[0].scene.capture_manifest_sha256,
        )

    def test_collection_renders_complete_epoch_and_labels_crash_salvage(self):
        collection, _capture_manifest = _capture_collection(with_salvage=True)
        rendered = render_collection(
            collection,
            Source1RenderConfig(
                width=64, height=40, allow_synthetic_geometry=True
            ),
        )
        self.assertEqual(1, len(rendered.frames))
        self.assertFalse(rendered.is_continuous)
        self.assertEqual(
            "demo-segment-11",
            rendered.frames[0].scene.epoch_context["capture_epoch_id"],
        )
        summary = rendered.summary()
        self.assertEqual("gapped", summary["continuity_status"])
        self.assertEqual(
            "discrete-complete-epochs-with-explicit-gaps",
            summary["presentation"],
        )
        self.assertEqual("crashed", summary["salvage_epochs"][0]["status"])
        self.assertEqual(
            "salvage-evidence-only-no-canonical-frames",
            summary["salvage_epochs"][0]["presentation"],
        )

    def test_gap_free_but_target_incomplete_collection_is_not_continuous(self):
        collection, _capture_manifest = _capture_collection(with_salvage=False)
        manifest = collection["manifest"]
        manifest["request"]["required_target_active_ns"] = 350
        manifest["coverage"]["corpus_status"] = "target-not-met"
        manifest = seal_document(manifest, "collection_manifest_sha256")
        collection["manifest"] = manifest
        terminal = collection["terminal"]
        terminal["collection_manifest_sha256"] = manifest[
            "collection_manifest_sha256"
        ]
        terminal["corpus_status"] = "target-not-met"
        collection["terminal"] = seal_document(terminal, "terminal_sha256")
        for ordinal, entry in enumerate(collection["epochs"]):
            snapshot = entry["frames"][0]
            context_index = next(
                index for index, item in enumerate(snapshot["entities"])
                if item["id"] == "total-capture:epoch-context"
            )
            snapshot["entities"][context_index] = epoch_context_entity(
                manifest, ordinal
            )
            snapshot["state_hash"] = state_hash(snapshot["entities"])

        rendered = render_collection(
            collection,
            Source1RenderConfig(
                width=64, height=40, allow_synthetic_geometry=True
            ),
        )
        summary = rendered.summary()
        self.assertEqual("gap-free", summary["continuity_status"])
        self.assertFalse(summary["target_complete"])
        self.assertFalse(summary["continuous"])
        self.assertEqual(
            "discrete-complete-epochs-target-incomplete",
            summary["presentation"],
        )

    def test_collection_authoritative_acceptance_is_scoped_per_complete_epoch(self):
        collection, capture_manifest = _capture_collection(with_salvage=True)
        engine = capture_manifest["engine"]
        rendered = render_collection(
            collection,
            Source1RenderConfig(
                mode="authoritative",
                fidelity_mode="total-capture-authoritative",
                expected_collection_manifest_sha256=collection["manifest"][
                    "collection_manifest_sha256"
                ],
                expected_build_id=engine["build_id"],
                expected_build_sha256=engine["build_sha256"],
                expected_content_manifest_sha256=engine[
                    "content_manifest_sha256"
                ],
                width=64,
                height=40,
                allow_synthetic_geometry=True,
            ),
        )
        self.assertEqual(1, len(rendered.frames))
        self.assertEqual(1, rendered.summary()["rendered_complete_epochs"])
        self.assertFalse(rendered.summary()["continuous"])

    def test_collection_rejects_epoch_context_cross_join(self):
        collection, _capture_manifest = _capture_collection(with_salvage=False)
        second = collection["epochs"][1]["frames"][0]
        context = next(
            item for item in second["entities"]
            if item["id"] == "total-capture:epoch-context"
        )
        context["components"]["total_capture_epoch_context"] = copy.deepcopy(
            collection["epochs"][0]["frames"][0]["entities"][-1]["components"][
                "total_capture_epoch_context"
            ]
        )
        second["state_hash"] = state_hash(second["entities"])
        with self.assertRaisesRegex(
            Source1RenderRefusal, "invalid-total-capture"
        ):
            render_collection(
                collection,
                Source1RenderConfig(
                    width=64, height=40, allow_synthetic_geometry=True
                ),
            )

    def test_collection_rejects_discontinuity_inside_complete_epoch(self):
        collection, _capture_manifest = _capture_collection(with_salvage=False)
        collection["epochs"][1]["frames"][0]["discontinuity"] = {
            "kind": "reset", "reason": "unclosed gap"
        }
        with self.assertRaisesRegex(
            Source1RenderRefusal, "discontinuous-complete-epoch"
        ):
            render_collection(
                collection,
                Source1RenderConfig(
                    width=64, height=40, allow_synthetic_geometry=True
                ),
            )

    def test_total_capture_authoritative_requires_exact_build_join(self):
        snapshot, manifest = _total_capture_snapshot(("ego",))
        engine = manifest["engine"]
        base = dict(
            mode="authoritative",
            fidelity_mode="total-capture-authoritative",
            allow_synthetic_geometry=True,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_build_id=engine["build_id"],
            expected_build_sha256=engine["build_sha256"],
            expected_content_manifest_sha256=engine[
                "content_manifest_sha256"
            ],
            width=64,
            height=40,
        )
        rendered = render_snapshot(snapshot, Source1RenderConfig(**base))
        self.assertEqual("ego", rendered.scene.camera.vantage_id)
        base["expected_build_id"] = "wrong-build"
        with self.assertRaisesRegex(
            Source1RenderRefusal, "total-capture-build-mismatch"
        ):
            render_snapshot(snapshot, Source1RenderConfig(**base))

    def test_total_capture_declared_effects_are_never_silently_dropped(self):
        snapshot, _manifest = _total_capture_snapshot(("ego",))
        snapshot["entities"][0]["components"]["effects"] = _component({
            "instances": [{
                "effect_id": "smoke:1", "kind": "smoke", "active": True,
                "transform_3x4": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                ],
                "resource_content_sha256": "7" * 64,
                "parameters": {"seed": 7},
            }]
        })
        snapshot["state_hash"] = state_hash(snapshot["entities"])
        with self.assertRaisesRegex(Source1RenderRefusal, "effects-unimplemented"):
            render_snapshot(
                snapshot,
                Source1RenderConfig(
                    width=64, height=40, allow_synthetic_geometry=True
                ),
            )

    def test_total_capture_camera_matrix_must_join_recorded_origin(self):
        snapshot, _manifest = _total_capture_snapshot(("ego",))
        camera = snapshot["entities"][2]["components"]["camera"]["value"]
        camera["view_matrix_4x4"][3] = 1.0
        snapshot["state_hash"] = state_hash(snapshot["entities"])
        with self.assertRaisesRegex(
            Source1RenderRefusal, "camera-view-matrix-mismatch"
        ):
            render_snapshot(
                snapshot,
                Source1RenderConfig(
                    width=64, height=40, allow_synthetic_geometry=True
                ),
            )

    def test_authoritative_total_capture_refuses_open_stream_horizon(self):
        snapshot, manifest = _total_capture_snapshot(("ego",))
        metadata = snapshot["entities"][1]
        metadata["components"]["capture_completeness"] = _component(
            capture_completeness_document(manifest, closed=False),
            "derived",
            derivation=COMPLETENESS_DERIVATION,
        )
        snapshot["state_hash"] = state_hash(snapshot["entities"])
        engine = manifest["engine"]
        with self.assertRaisesRegex(
            Source1RenderRefusal, "incomplete-total-capture"
        ):
            render_snapshot(
                snapshot,
                Source1RenderConfig(
                    mode="authoritative",
                    fidelity_mode="total-capture-authoritative",
                    allow_synthetic_geometry=True,
                    expected_manifest_sha256=manifest["manifest_sha256"],
                    expected_build_id=engine["build_id"],
                    expected_build_sha256=engine["build_sha256"],
                    expected_content_manifest_sha256=engine[
                        "content_manifest_sha256"
                    ],
                    width=64,
                    height=40,
                ),
            )

    def test_authoritative_total_capture_refuses_crash_salvage_document(self):
        snapshot, manifest = _total_capture_snapshot(("ego",))
        snapshot["entities"][1]["components"]["total_capture_manifest"] = (
            _component(incomplete_capture_document(
                "renderer-session", "game crashed before dem_stop",
                {"partial-demo": "9" * 64},
            ))
        )
        snapshot["state_hash"] = state_hash(snapshot["entities"])
        engine = manifest["engine"]
        with self.assertRaisesRegex(
            Source1RenderRefusal, "invalid-total-capture"
        ):
            render_snapshot(
                snapshot,
                Source1RenderConfig(
                    mode="authoritative",
                    fidelity_mode="total-capture-authoritative",
                    allow_synthetic_geometry=True,
                    expected_manifest_sha256=manifest["manifest_sha256"],
                    expected_build_id=engine["build_id"],
                    expected_build_sha256=engine["build_sha256"],
                    expected_content_manifest_sha256=engine[
                        "content_manifest_sha256"
                    ],
                    width=64,
                    height=40,
                ),
            )

    def test_canonical_total_capture_labels_open_stream_horizon(self):
        snapshot, manifest = _total_capture_snapshot(("ego",))
        snapshot["entities"][1]["components"]["capture_completeness"] = (
            _component(
                capture_completeness_document(manifest, closed=False),
                "derived",
                derivation=COMPLETENESS_DERIVATION,
            )
        )
        snapshot["state_hash"] = state_hash(snapshot["entities"])
        rendered = render_snapshot(
            snapshot,
            Source1RenderConfig(
                width=64, height=40, allow_synthetic_geometry=True
            ),
        )
        self.assertTrue(any(
            item["code"] == "total-capture-horizon-incomplete"
            for item in rendered.scene.omissions
        ))

    def test_synthetic_geometry_requires_explicit_configuration(self):
        with self.assertRaisesRegex(
            Source1RenderRefusal, "synthetic-geometry-disabled"
        ):
            render_snapshot(_snapshot(), Source1RenderConfig())

    def test_authoritative_mode_refuses_reference_quality_claim(self):
        with self.assertRaisesRegex(
            Source1RenderRefusal, "authoritative-parity-unimplemented"
        ):
            render_snapshot(
                _snapshot(),
                Source1RenderConfig(
                    mode="authoritative", allow_synthetic_geometry=True
                ),
            )

    def test_model_does_not_accept_source2_vmdl_or_substitute_asset(self):
        with self.assertRaisesRegex(
            Source1RenderRefusal, "ambiguous-model-engine"
        ):
            render_snapshot(
                _snapshot(model={"source1_mdl": "characters/ctm.vmdl_c"}),
                Source1RenderConfig(allow_synthetic_geometry=True),
            )

    def test_adapter_visual_evidence_without_model_binding_is_refused(self):
        for component_name, component in (
            (
                "player_state",
                _component(
                    {"baseclass.m_iHealth": 100},
                    "derived",
                    derivation=(
                        "canonical player fields selected from decoded Source 1 netprops"
                    ),
                ),
            ),
            (
                "source1_netprops",
                _component(
                    {"baseclass.m_nModelIndex": 3},
                    "derived",
                    derivation=(
                        "Source 1 sendtable baseline plus ordered property deltas"
                    ),
                    completeness="partial-no-instancebaseline",
                ),
            ),
        ):
            with self.subTest(component=component_name):
                snapshot = _snapshot()
                snapshot["entities"][1]["components"] = {
                    component_name: component
                }
                snapshot["state_hash"] = state_hash(snapshot["entities"])
                with self.assertRaisesRegex(
                    Source1RenderRefusal, "missing-visual-asset"
                ):
                    render_snapshot(snapshot, Source1RenderConfig())

    def test_mdl_vvd_vtx_final_bones_produce_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_model(root)
            snapshot = _snapshot(
                model={
                    "source1_mdl": "models/fixture.mdl",
                    "lod": 0,
                    "bodygroups": [0],
                },
                animation=_final_bones(),
            )
            rendered = render_snapshot(
                snapshot,
                Source1RenderConfig(asset_root=root, width=64, height=40),
            )
            payload = rendered.ppm.split(b"\n", 3)[3]
            self.assertNotEqual(set(payload), {18, 20, 24})
            self.assertIn("mdl:models/fixture.mdl", rendered.scene.meshes[0].source)
            self.assertTrue(any(
                item["code"] == "mdl-materials-unimplemented"
                for item in rendered.scene.omissions
            ))

    def test_total_capture_final_pose_attachments_and_content_join_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_model(root)
            mdl, _vvd, _vtx = _fixture_model_triplet()
            model_hash = hashlib.sha256(mdl).hexdigest()
            snapshot, _manifest = _total_capture_snapshot(("ego",))
            actor_id = "captured:player:1"
            snapshot["entities"].append({
                "id": actor_id, "generation": 0, "class": "CapturedPlayer",
                "components": {
                    "model_binding": _component({
                        "engine": "source1/csgo", "model_id": "opaque-player-1",
                        "model_content_sha256": model_hash,
                        "material_set": "recorded-default", "skin": 0,
                        "bodygroups": [0], "lod": 0,
                    }),
                    "world_transform": _component({
                        "matrix_3x4": [1, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 1, 0],
                    }),
                    "final_pose": _component({
                        "space": "world", "model_content_sha256": model_hash,
                        "skeleton_id": "fixture-root", "pose_sequence": 9,
                        "bones": [{
                            "name": "root", "parent": -1,
                            "matrix_3x4": [1, 0, 0, 4, 0, 1, 0, 0,
                                           0, 0, 1, 0],
                        }],
                        "attachments": [{
                            "name": "muzzle", "bone": "root",
                            "matrix_3x4": [1, 0, 0, 4, 0, 1, 0, 0,
                                           0, 0, 1, 0],
                        }],
                    }),
                },
            })
            viewmodel_id = "captured:viewmodel:ego"
            snapshot["entities"].append({
                "id": viewmodel_id, "generation": 0,
                "class": "CapturedViewModel",
                "components": {
                    "model_binding": _component({
                        "engine": "source1/csgo", "model_id": "opaque-viewmodel-1",
                        "model_content_sha256": model_hash,
                        "material_set": "recorded-default", "skin": 0,
                        "bodygroups": [0], "lod": 0,
                    }),
                    "viewmodel": _component({
                        "pov_id": "ego", "handedness": "right",
                        "viewmodel_fov": 54.0, "projection": "perspective",
                        "near": 0.01, "far": 128.0,
                        "projection_matrix_4x4": [
                            0, 1, 0, 0, 0, 0, 1, 0,
                            1, 0, 0, 0, 1, 0, 0, 0,
                        ],
                    }),
                    "final_pose": _component({
                        "space": "view", "model_content_sha256": model_hash,
                        "skeleton_id": "fixture-view-root", "pose_sequence": 4,
                        "bones": [{
                            "name": "root", "parent": -1,
                            "matrix_3x4": [1, 0, 0, 2, 0, 1, 0, 0,
                                           0, 0, 1, 0],
                        }],
                        "attachments": [],
                    }),
                },
            })
            pov = snapshot["entities"][2]
            pov["components"]["visibility"]["value"][
                "visible_entity_ids"
            ].append(actor_id)
            pov["components"]["visibility"]["value"][
                "visible_entity_ids"
            ].append(viewmodel_id)
            snapshot["state_hash"] = state_hash(snapshot["entities"])
            rendered = render_snapshot(
                snapshot,
                Source1RenderConfig(
                    asset_root=root,
                    width=64,
                    height=40,
                    allow_synthetic_geometry=True,
                    content_model_bindings=((model_hash, "models/fixture.mdl"),),
                ),
            )
            self.assertTrue(any(
                mesh.entity_id == actor_id for mesh in rendered.scene.meshes
            ))
            self.assertTrue(any(
                mesh.entity_id == viewmodel_id
                and mesh.space == "view"
                and mesh.projection_fov_degrees == 54.0
                for mesh in rendered.scene.meshes
            ))
            self.assertTrue(any(
                item["code"] == "captured-attachments-consumed"
                and item["count"] == 1
                for item in rendered.scene.omissions
            ))

    def test_absent_demo_lod_uses_named_reference_lod0_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_model(root)
            snapshot = _snapshot(
                model={"source1_mdl": "models/fixture.mdl", "bodygroups": [0]},
                animation=_final_bones(),
            )
            rendered = render_snapshot(
                snapshot, Source1RenderConfig(asset_root=root, width=32, height=24)
            )
            self.assertTrue(any(
                item["code"] == "reference-model-lod-policy"
                and item["value"] == 0
                for item in rendered.scene.omissions
            ))

    def test_nested_unavailable_demo_lod_uses_reference_lod0_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_model(root)
            snapshot = _snapshot(
                model={
                    "source1_mdl": "models/fixture.mdl",
                    "bodygroups": [0],
                    "lod": {
                        "provenance": "unavailable",
                        "reason": "render LOD is not entity state",
                    },
                },
                animation=_final_bones(),
            )
            rendered = render_snapshot(
                snapshot, Source1RenderConfig(asset_root=root, width=32, height=24)
            )
            policy = next(
                item for item in rendered.scene.omissions
                if item["code"] == "reference-model-lod-policy"
            )
            self.assertEqual(policy["provenance"], "renderer-policy")
            self.assertEqual(policy["reason"], "render LOD is not entity state")

    def test_exact_modelprecache_join_derivation_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_model(root)
            snapshot = _snapshot_with_reducer_model()
            rendered = render_snapshot(
                snapshot, Source1RenderConfig(asset_root=root, width=32, height=24)
            )
            self.assertTrue(rendered.scene.meshes)

    def test_player_state_with_rendered_model_does_not_fall_through(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_model(root)
            snapshot = _snapshot_with_reducer_model()
            snapshot["entities"][1]["components"]["player_state"] = _component(
                {"baseclass.m_iHealth": 100},
                "derived",
                derivation=(
                    "canonical player fields selected from decoded Source 1 netprops"
                ),
            )
            snapshot["state_hash"] = state_hash(snapshot["entities"])
            rendered = render_snapshot(
                snapshot,
                Source1RenderConfig(asset_root=root, width=32, height=24),
            )
            self.assertTrue(rendered.scene.meshes)

    def test_generic_or_malformed_derived_model_binding_is_refused(self):
        for derivation, mutation in (
            ("guessed from entity class", None),
            (MODEL_BINDING_DERIVATION, "bad-hash"),
        ):
            with self.subTest(derivation=derivation, mutation=mutation):
                snapshot = _snapshot_with_reducer_model()
                binding = snapshot["entities"][1]["components"]["source1_model"]
                binding["derivation"] = derivation
                if mutation:
                    binding["value"]["source_entry_sha256"] = mutation
                snapshot["state_hash"] = state_hash(snapshot["entities"])
                with self.assertRaisesRegex(
                    Source1RenderRefusal,
                    "untrusted-model-derivation|invalid-model-join",
                ):
                    render_snapshot(snapshot, Source1RenderConfig())

    def test_derived_model_binding_is_joined_to_same_frame_modelprecache(self):
        mutations = (
            "absent",
            "revision",
            "path",
            "user-data",
            "duplicate",
            "aggregate-derivation",
            "field-origin",
            "entry-provenance",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                snapshot = _snapshot_with_reducer_model()
                table = snapshot["entities"].pop()
                component = table["components"]["source1_string_table"]
                value = component["value"]
                if mutation == "revision":
                    value["revision"] = 8
                elif mutation == "path":
                    value["server_entries"][0]["string"] = "models/other.mdl"
                elif mutation == "user-data":
                    value["server_entries"][0]["user_data_base64"] = "eA=="
                elif mutation == "duplicate":
                    value["server_entries"].append(
                        copy.deepcopy(value["server_entries"][0])
                    )
                elif mutation == "aggregate-derivation":
                    component["derivation"] = "generic canonicalization"
                elif mutation == "field-origin":
                    value["field_origins"]["server_entries"][
                        "entry_content_provenance"
                    ] = "guessed"
                elif mutation == "entry-provenance":
                    value["server_entries"][0]["content_provenance"] = "derived"
                if mutation != "absent":
                    snapshot["entities"].append(table)
                snapshot["state_hash"] = state_hash(snapshot["entities"])
                with self.assertRaisesRegex(
                    Source1RenderRefusal, "model-join-mismatch"
                ):
                    render_snapshot(snapshot, Source1RenderConfig())

    def test_vpk_backed_mdl_vvd_vtx_are_crc_verified_and_rendered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mdl, vvd, vtx = _fixture_model_triplet()
            archive = root / "pak01_dir.vpk"
            archive.write_bytes(_vpk_files({
                "models/fixture.mdl": mdl,
                "models/fixture.vvd": vvd,
                "models/fixture.dx90.vtx": vtx,
            }))
            snapshot = _snapshot(
                model={"source1_mdl": "MODELS\\FIXTURE.MDL", "bodygroups": [0]},
                animation=_final_bones(),
            )
            rendered = render_snapshot(
                snapshot,
                Source1RenderConfig(
                    asset_root=root,
                    vpk_paths=(archive,),
                    width=32,
                    height=24,
                ),
            )
            self.assertTrue(rendered.scene.meshes)

    def test_vpk_backed_bsp_uses_only_configured_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "maps_dir.vpk"
            archive.write_bytes(_vpk_files({"maps/fixture.bsp": _fixture_bsp()}))
            snapshot = _snapshot()
            snapshot["entities"] = snapshot["entities"][:1]
            snapshot["state_hash"] = state_hash(snapshot["entities"])
            rendered = render_snapshot(
                snapshot,
                Source1RenderConfig(
                    asset_root=root,
                    vpk_paths=(archive,),
                    map_asset="maps/fixture.bsp",
                    width=32,
                    height=24,
                ),
            )
            self.assertEqual(rendered.scene.meshes[0].entity_id, "worldspawn")
            self.assertIn("vpk:", rendered.scene.meshes[0].source)

    def test_sequence_cycle_without_final_bones_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_model(root)
            snapshot = _snapshot(
                model={"source1_mdl": "models/fixture.mdl", "lod": 0},
                animation={"sequence": 1, "cycle": 0.5},
            )
            with self.assertRaisesRegex(
                Source1RenderRefusal, "unevaluated-animation-state"
            ):
                render_snapshot(snapshot, Source1RenderConfig(asset_root=root))

    def test_nested_unavailable_fov_uses_explicit_canonical_profile_value(self):
        snapshot = _snapshot(include_fov=False)
        camera = snapshot["entities"][0]["components"]["camera"]["value"]
        camera["fov"] = {
            "provenance": "unavailable",
            "reason": "democmdinfo has no projection FOV",
        }
        snapshot["state_hash"] = state_hash(snapshot["entities"])
        rendered = render_snapshot(
            snapshot,
            Source1RenderConfig(
                allow_synthetic_geometry=True,
                reference_fov_degrees=101.0,
            ),
        )
        self.assertEqual(rendered.scene.camera.fov_degrees, 101.0)
        self.assertEqual(rendered.scene.camera.fov_provenance, "profile-derived")
        self.assertEqual(rendered.scene.omissions[0]["code"],
                         "camera-fov-not-recorded")

    def test_recorded_fov_wins_over_reference_fov(self):
        rendered = render_snapshot(
            _snapshot(),
            Source1RenderConfig(
                allow_synthetic_geometry=True,
                reference_fov_degrees=110.0,
            ),
        )
        self.assertEqual(rendered.scene.camera.fov_degrees, 90.0)
        self.assertEqual(rendered.scene.camera.fov_provenance, "recorded")

    def test_state_hash_mismatch_is_refused_before_pixels(self):
        snapshot = _snapshot()
        snapshot["state_hash"] = "0" * 64
        with self.assertRaisesRegex(Source1RenderRefusal, "state-hash-mismatch"):
            render_snapshot(
                snapshot,
                Source1RenderConfig(allow_synthetic_geometry=True),
            )

    def test_out_of_pvs_geometry_contributes_no_pixels(self):
        visible = _snapshot()
        visible_entity = visible["entities"][1]
        visible_entity["components"]["visibility"] = _component({"in_pvs": True})
        visible["state_hash"] = state_hash(visible["entities"])
        with_hidden = copy.deepcopy(visible)
        hidden = copy.deepcopy(visible_entity)
        hidden["id"] = "hidden-nearer-triangle"
        hidden["components"]["visibility"] = _component({"in_pvs": False})
        hidden_geometry = hidden["components"]["source1_geometry"]["value"]
        hidden_geometry["vertices"] = [
            [2.0, -1.0, -1.0], [2.0, 1.0, -1.0], [2.0, 0.0, 1.0]
        ]
        hidden_geometry["color"] = [3, 250, 5]
        with_hidden["entities"].append(hidden)
        with_hidden["state_hash"] = state_hash(with_hidden["entities"])
        config = Source1RenderConfig(
            width=64, height=40, allow_synthetic_geometry=True
        )
        baseline = render_snapshot(visible, config)
        rendered = render_snapshot(with_hidden, config)
        self.assertEqual(rendered.ppm, baseline.ppm)
        self.assertTrue(any(
            item["code"] == "entity-outside-pvs"
            for item in rendered.scene.omissions
        ))

    def test_mixed_stream_batch_refuses_before_output_assignment(self):
        first = _snapshot()
        second = copy.deepcopy(first)
        second["stream_id"] = "another/csgo-stream"
        with self.assertRaisesRegex(Source1RenderRefusal, "mixed-stream-batch"):
            render_batch(
                [first, second],
                Source1RenderConfig(allow_synthetic_geometry=True),
            )

    def test_render_session_writes_ppm_and_state_hash_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "snapshot.json"
            source.write_text(json.dumps(_snapshot()))
            result = render_session(
                [
                    "--snapshot-json", str(source),
                    "--scene-out", "scene.json",
                    "--frame-out", "frame.ppm",
                    "--width", "48",
                    "--height", "32",
                    "--allow-synthetic-geometry",
                ],
                cwd=root,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "frame.ppm").read_bytes().startswith(b"P6"))
            scene = json.loads((root / "scene.json").read_text())
            depth = root / "frame.depth.u64le"
            self.assertTrue(depth.is_file())
            self.assertEqual(
                hashlib.sha256(depth.read_bytes()).hexdigest(),
                scene["frames"][0]["depth_sha256"],
            )
            self.assertEqual(
                Path(scene["frames"][0]["depth_output"]), depth.resolve()
            )
            self.assertEqual(
                scene["frames"][0]["state_hash"],
                _snapshot()["state_hash"],
            )
            self.assertEqual(scene["frames"][0]["scene"]["schema"],
                             "tardigrade/source1-reference-scene/v1")

    def test_render_session_writes_collision_safe_scored_pov_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, _manifest = _total_capture_snapshot()
            source = root / "snapshot.json"
            source.write_text(json.dumps(snapshot))
            result = render_session(
                [
                    "--snapshot-json", str(source),
                    "--scene-out", "scene.json",
                    "--frame-out", "frames",
                    "--width", "64",
                    "--height", "40",
                    "--allow-synthetic-geometry",
                ],
                cwd=root,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            scene = json.loads((root / "scene.json").read_text())
            self.assertEqual(2, len(scene["frames"]))
            self.assertEqual(
                {"bot-a", "bot-b"},
                {item["vantage_id"] for item in scene["frames"]},
            )
            outputs = [Path(item["output"]) for item in scene["frames"]]
            depths = [Path(item["depth_output"]) for item in scene["frames"]]
            self.assertEqual(2, len(set(outputs)))
            self.assertEqual(2, len(set(depths)))
            self.assertTrue(all(path.is_file() for path in outputs + depths))
            self.assertNotEqual(
                scene["frames"][0]["vantage_state_hash"],
                scene["frames"][1]["vantage_state_hash"],
            )
            self.assertNotEqual(outputs[0].read_bytes(), outputs[1].read_bytes())

    def test_render_session_qualifies_outputs_by_capture_epoch_and_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection, _capture_manifest = _capture_collection(
                with_salvage=False
            )
            source = root / "collection.json"
            source.write_text(json.dumps(collection))
            result = render_session(
                [
                    "--snapshot-json", str(source),
                    "--scene-out", "scene.json",
                    "--frame-out", "frames",
                    "--width", "64",
                    "--height", "40",
                    "--allow-synthetic-geometry",
                ],
                cwd=root,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            scene = json.loads((root / "scene.json").read_text())
            self.assertEqual(2, len(scene["frames"]))
            self.assertTrue(scene["collection"]["continuous"])
            self.assertEqual(
                {"demo-segment-11", "demo-segment-12"},
                {
                    item["epoch_context"]["capture_epoch_id"]
                    for item in scene["frames"]
                },
            )
            outputs = [Path(item["output"]) for item in scene["frames"]]
            self.assertEqual(2, len(set(outputs)))
            self.assertTrue(all(path.is_file() for path in outputs))
            for output in outputs:
                relative = output.relative_to((root / "frames").resolve())
                self.assertTrue(str(relative).startswith("collection-"))
                self.assertIn("/process-", str(relative))
                self.assertIn("/map-", str(relative))
            self.assertNotEqual(outputs[0].read_bytes(), outputs[1].read_bytes())

    def test_render_session_returns_failure_without_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "snapshot.json"
            source.write_text(json.dumps(_snapshot()))
            result = render_session(
                [
                    "--snapshot-json", str(source),
                    "--scene-out", "scene.json",
                    "--frame-out", "frame.ppm",
                ],
                cwd=root,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("synthetic-geometry-disabled", result.stderr)
            self.assertFalse((root / "frame.ppm").exists())
            self.assertFalse((root / "scene.json").exists())


if __name__ == "__main__":
    unittest.main()
