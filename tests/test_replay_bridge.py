from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from counter_strike_render.replay_bridge import (
    AuthoritativeStateUnavailable,
    LegacyRendererUnsupported,
    SnapshotValidationError,
    StateContinuityError,
    StateToRenderBridge,
    snapshots_to_render_batch,
    stage_legacy_inputs,
)


def component(provenance="recorded", value=None, **extra):
    item = {"provenance": provenance, **extra}
    if provenance != "unavailable" or value is not None:
        item["value"] = value
    return item


def entity(entity_id, generation=0, entity_class="prop", **components):
    return {
        "id": entity_id,
        "generation": generation,
        "class": entity_class,
        "components": components,
    }


def snapshot(tick, entities, **extra):
    return {
        "schema": "tardigrade/state-snapshot/v1",
        "stream_id": "match/pov-1",
        "tick": tick,
        "subtick": 0,
        "state_hash": f"state-{tick}",
        "entities": entities,
        **extra,
    }


class ReplayBridgeTests(unittest.TestCase):
    def test_subtick_is_a_nonnegative_integer_ordinal(self):
        marker = entity(1, marker=component(value={"visible": True}))
        self.assertEqual(
            StateToRenderBridge().convert(snapshot(1, [marker], subtick=7)).subtick,
            7,
        )
        for invalid in (0.25, -1, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    SnapshotValidationError, "subtick must be an integer >= 0"
                ):
                    StateToRenderBridge().convert(
                        snapshot(1, [marker], subtick=invalid)
                    )

    def test_canonical_batch_routes_records_and_preserves_provenance(self):
        camera = entity(
            "camera",
            entity_class="camera",
            camera=component(value={"fov": 90, "world_transform": [1, 2, 3]}),
        )
        player = entity(
            7,
            entity_class="CCSPlayerPawn",
            player=component(value={"health": 100}),
            animation=component(
                "derived",
                {"evaluated_bones": [[1, 0, 0, 0]]},
                derivation="canonical locomotion evaluator v2",
            ),
        )
        prop = entity(
            "crate",
            dynamic_physics=component(
                "unavailable", reason="not present in the SourceTV stream"
            ),
        )

        batch = snapshots_to_render_batch(
            [snapshot(10, [camera, player, prop])], mode="canonical"
        )

        self.assertEqual(len(batch.frames), 1)
        self.assertEqual([item.entity_id for item in batch.cameras], ["camera"])
        self.assertEqual([item.entity_id for item in batch.players], [7])
        self.assertEqual([item.entity_id for item in batch.animations], [7])
        self.assertEqual([item.entity_id for item in batch.dynamic_physics], ["crate"])
        self.assertEqual(batch.animations[0].primary.provenance, "derived")
        self.assertEqual(batch.dynamic_physics[0].primary.provenance, "unavailable")
        self.assertIsNone(batch.dynamic_physics[0].primary.value)

    def test_authoritative_accepts_recorded_physics_and_evaluated_bones(self):
        player = entity(
            1,
            entity_class="CCSPlayerPawn",
            player=component(value={"health": 82}),
            animation=component(value={"evaluated_bones": {"pelvis": [1, 0, 0]}}),
        )
        prop = entity(
            2,
            dynamic_physics=component(
                value={"dynamic": True, "world_transform": [1, 0, 0, 0]}
            ),
        )

        frame = StateToRenderBridge("authoritative").convert(
            snapshot(1, [player, prop])
        )

        self.assertEqual(frame.mode.value, "authoritative")
        self.assertEqual(len(frame.players), 1)
        self.assertEqual(len(frame.dynamic_physics), 1)

    def test_authoritative_accepts_complete_recorded_animation_graph(self):
        player = entity(
            1,
            player=component(value={"health": 100}),
            animation=component(
                value={
                    "graph_state": {
                        "complete": True,
                        "asset": "characters/models/ctm.vmdl",
                        "active_nodes": [4, 11],
                        "parameters": {"speed": 0.0},
                    }
                }
            ),
        )
        StateToRenderBridge("authoritative").convert(snapshot(1, [player]))

    def test_authoritative_refuses_missing_or_derived_world_transform(self):
        missing = entity(
            "barrel", dynamic_physics=component(value={"dynamic": True})
        )
        with self.assertRaisesRegex(
            AuthoritativeStateUnavailable, "recorded world_transform"
        ):
            StateToRenderBridge("authoritative").convert(snapshot(1, [missing]))

        derived = entity(
            "barrel",
            dynamic_physics=component(
                value={
                    "dynamic": True,
                    "world_transform": component(
                        "derived", [1, 0, 0, 0], derivation="interpolation"
                    ),
                }
            ),
        )
        with self.assertRaisesRegex(
            AuthoritativeStateUnavailable, "recorded world_transform"
        ):
            StateToRenderBridge("authoritative").convert(snapshot(1, [derived]))

    def test_authoritative_refuses_player_without_complete_pose(self):
        no_animation = entity(1, player=component(value={"health": 100}))
        with self.assertRaisesRegex(
            AuthoritativeStateUnavailable, "requires an animation component"
        ):
            StateToRenderBridge("authoritative").convert(
                snapshot(1, [no_animation])
            )

        incomplete = entity(
            1,
            player=component(value={"health": 100}),
            animation=component(value={"graph_state": {"complete": False}}),
        )
        with self.assertRaisesRegex(
            AuthoritativeStateUnavailable, "complete=true"
        ):
            StateToRenderBridge("authoritative").convert(snapshot(1, [incomplete]))

    def test_strict_provenance_contract(self):
        unavailable_without_reason = entity(
            1, animation=component("unavailable")
        )
        with self.assertRaisesRegex(SnapshotValidationError, "reason is required"):
            StateToRenderBridge().convert(snapshot(1, [unavailable_without_reason]))

        derived_without_method = entity(
            1, animation=component("derived", {"evaluated_bones": [[1]]})
        )
        with self.assertRaisesRegex(
            SnapshotValidationError, "derivation is required"
        ):
            StateToRenderBridge().convert(snapshot(1, [derived_without_method]))

    def test_order_hash_chain_and_generation_are_validated(self):
        bridge = StateToRenderBridge()
        bridge.convert(snapshot(10, [entity(1, player=component(value={}))]))

        with self.assertRaisesRegex(StateContinuityError, "did not advance"):
            bridge.convert(snapshot(9, [entity(1, player=component(value={}))]))

        # Failed batches and converts do not alter the last successful cursor.
        bridge.convert(
            snapshot(
                11,
                [entity(1, player=component(value={}))],
                previous_state_hash="state-10",
            )
        )
        with self.assertRaisesRegex(StateContinuityError, "preceding state_hash"):
            bridge.convert(
                snapshot(
                    12,
                    [entity(1, player=component(value={}))],
                    previous_state_hash="wrong",
                )
            )

    def test_reappearing_identity_requires_generation_increment(self):
        bridge = StateToRenderBridge()
        prop = lambda generation=0: entity(
            "prop", generation=generation, marker=component(value={"visible": True})
        )
        bridge.convert(snapshot(1, [prop()]))
        bridge.convert(snapshot(2, []))
        with self.assertRaisesRegex(StateContinuityError, "reuses inactive generation"):
            bridge.convert(snapshot(3, [prop()]))

        bridge.convert(snapshot(3, [prop(1)]))

    def test_explicit_discontinuity_allows_seek_and_resets_identity_history(self):
        bridge = StateToRenderBridge()
        prop = lambda generation: entity(
            "prop", generation=generation, marker=component(value={"visible": True})
        )
        bridge.convert(snapshot(100, [prop(9)]))

        frame = bridge.convert(
            snapshot(
                2,
                [prop(0)],
                discontinuity={"kind": "seek", "reason": "checkpoint seek"},
            )
        )

        self.assertEqual(frame.tick, 2)
        self.assertEqual(frame.discontinuity.kind, "seek")

    def test_batch_is_transactional(self):
        bridge = StateToRenderBridge()
        marker = entity("marker", marker=component(value={"visible": False}))
        with self.assertRaises(StateContinuityError):
            bridge.batch([snapshot(4, [marker]), snapshot(3, [marker])])

        # Tick 3 remains legal because the failed batch did not retain tick 4.
        self.assertEqual(bridge.convert(snapshot(3, [marker])).tick, 3)

    def test_failed_authoritative_convert_is_transactional(self):
        bridge = StateToRenderBridge("authoritative")
        bad = entity(
            "crate", dynamic_physics=component(value={"dynamic": True})
        )
        with self.assertRaises(AuthoritativeStateUnavailable):
            bridge.convert(snapshot(8, [bad]))

        good = entity(
            "crate",
            dynamic_physics=component(
                value={"dynamic": True, "world_transform": [1, 0, 0, 0]}
            ),
        )
        self.assertEqual(bridge.convert(snapshot(7, [good])).tick, 7)


def camera_component(**overrides):
    values = {
        "x": 10.0,
        "y": 20.0,
        "z": 30.0,
        "eye_z": 94.0,
        "yaw_degrees": 45.0,
        "pitch_degrees": -5.0,
        "fov": 90.0,
        "fire": False,
        "is_alive": True,
        "weapon": "weapon_ak47",
    }
    values.update(overrides)
    return component(value=values)


def player_entity(entity_id=2, *, animation=None):
    components = {
        "player": component(
            value={
                "is_alive": True,
                "team": "t",
                "fire": True,
                "weapon": "weapon_glock",
            }
        ),
        "transform": component(value={"x": 1, "y": 2, "z": 3, "yaw": 180}),
    }
    if animation is not None:
        components["animation"] = animation
    return entity(entity_id, entity_class="CCSPlayerPawn", **components)


class LegacyStagingTests(unittest.TestCase):
    def test_canonical_staging_writes_current_schemas_and_gap_manifest(self):
        camera = entity(
            1,
            entity_class="camera",
            camera=camera_component(),
            effects=component(value={"flash_alpha": 0.8}),
        )
        # Use a nested provenance envelope for FOV; unlike a bare value it
        # must survive as a reported canonical downgrade.
        camera["components"]["camera"]["value"]["fov"] = component(
            "derived", 73.0, derivation="recorded vertical-fov conversion"
        )
        player = player_entity(
            animation=component(
                "derived",
                {"evaluated_bones": [[1, 0, 0, 0]]},
                derivation="legacy clip selector",
            )
        )
        batch = snapshots_to_render_batch(
            [snapshot(4, [camera, player])], mode="canonical"
        )

        with tempfile.TemporaryDirectory() as temporary:
            result = stage_legacy_inputs(
                batch, temporary, 128, ego_entity_id=1
            )
            camera_doc = json.loads(result.camera_json.read_text())
            player_doc = json.loads(result.playermodel_json.read_text())
            manifest = json.loads(result.gap_manifest_json.read_text())

            self.assertEqual(
                camera_doc["schema"], "iji/cs2-demo-ego-camera-path/v1"
            )
            self.assertEqual(camera_doc["tick_rate"], 128)
            self.assertEqual(camera_doc["steamid"], 1)
            self.assertEqual(camera_doc["ticks"][0]["fov"], 73.0)
            self.assertEqual(camera_doc["ticks"][0]["active_weapon_id"], "weapon_ak47")
            self.assertEqual(
                player_doc["schema"], "iji/cs2-demo-playermodels/v1"
            )
            self.assertEqual(player_doc["ticks"]["4"][0]["x"], 1.0)
            self.assertEqual(
                player_doc["ticks"]["4"][0]["active_weapon_id"],
                "weapon_glock",
            )
            codes = {item["code"] for item in manifest["gaps"]}
            self.assertIn("derived-field", codes)
            self.assertIn("unsupported-world-pose", codes)
            self.assertIn("unsupported-field", codes)
            self.assertFalse(manifest["authoritative_refusal"])
            self.assertFalse(list(Path(temporary).glob(".*.tmp")))

    def test_authoritative_staging_refuses_complete_pose_and_physics_state(self):
        camera = entity(1, entity_class="camera", camera=camera_component())
        player = player_entity(
            animation=component(value={"evaluated_bones": {"pelvis": [1, 0, 0]}})
        )
        prop = entity(
            3,
            dynamic_physics=component(
                value={"dynamic": True, "world_transform": [1, 0, 0, 0]}
            ),
        )
        batch = snapshots_to_render_batch(
            [snapshot(5, [camera, player, prop])], mode="authoritative"
        )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(LegacyRendererUnsupported) as raised:
                stage_legacy_inputs(batch, temporary, 128, ego_entity_id=1)

            manifest = raised.exception.manifest
            self.assertTrue(manifest["authoritative_refusal"])
            details = " ".join(item["detail"] for item in manifest["gaps"])
            self.assertIn("evaluated bones reached the bridge", details)
            self.assertIn("dynamic physics state reached the bridge", details)
            self.assertTrue((Path(temporary) / "legacy_gap_manifest.json").exists())
            self.assertFalse((Path(temporary) / "camera.json").exists())
            self.assertFalse((Path(temporary) / "playermodels.json").exists())

    def test_authoritative_camera_only_batch_can_reach_legacy_consumer(self):
        camera = entity(1, entity_class="camera", camera=camera_component())
        batch = snapshots_to_render_batch(
            [snapshot(1, [camera])], mode="authoritative"
        )

        with tempfile.TemporaryDirectory() as temporary:
            result = stage_legacy_inputs(batch, temporary, 64, ego_entity_id=1)
            self.assertEqual(result.gap_manifest["gaps"], [])
            self.assertTrue(result.camera_json.exists())
            self.assertTrue(result.playermodel_json.exists())

    def test_canonical_missing_required_camera_field_is_never_defaulted(self):
        camera = entity(
            1,
            entity_class="camera",
            camera=component(
                value={
                    "x": 1,
                    "y": 2,
                    "z": 3,
                    "eye_z": 67,
                    "yaw_degrees": 0,
                    # pitch deliberately unavailable: no zero-angle invention.
                    "fire": False,
                    "is_alive": True,
                    "weapon": "knife",
                    "fov": 90,
                }
            ),
        )
        batch = snapshots_to_render_batch([snapshot(1, [camera])])

        with tempfile.TemporaryDirectory() as temporary:
            result = stage_legacy_inputs(batch, temporary, 128, ego_entity_id=1)
            camera_doc = json.loads(result.camera_json.read_text())
            self.assertEqual(camera_doc["ticks"], [])
            codes = {item["code"] for item in result.gap_manifest["gaps"]}
            self.assertIn("missing-legacy-field", codes)
            self.assertIn("unstageable-camera-row", codes)


if __name__ == "__main__":
    unittest.main()
