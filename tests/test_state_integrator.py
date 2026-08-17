from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay import ConflictError, StateIntegrator  # noqa: E402


def entity(entity_id="pawn:7", generation=0, x=0, nullable=None):
    return {
        "id": entity_id,
        "generation": generation,
        "class": "CCSPlayerPawn",
        "components": {
            "transform": {"x": x, "y": nullable},
            "inventory": {"ammo": 10},
        },
    }


class IntegratorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = {
            "schema": "tardigrade/state-manifest/v1",
            "stream_id": "match/subjective-client",
            "tick_rate_hz": 128,
            "source": {"kind": "cs2-demo-plus-input-sidecar"},
        }
        initial = StateIntegrator.make_checkpoint(
            self.manifest["stream_id"], 0, 0, 0, [entity()])
        self.integrator = StateIntegrator.create(
            self.root, self.manifest, initial)

    def tearDown(self):
        self.integrator.close()
        self.temp.cleanup()

    def tx(self, sequence, tick, ops, base_hash=None, **extra):
        return {
            "schema": "tardigrade/state-transaction/v1",
            "stream_id": self.manifest["stream_id"],
            "sequence": sequence,
            "tick": tick,
            "subtick": 0,
            "base_hash": (base_hash if base_hash is not None
                          else self.integrator.current_snapshot()["state_hash"]),
            "ops": ops,
            **extra,
        }

    @staticmethod
    def by_id(snapshot, entity_id="pawn:7"):
        return next(e for e in snapshot["entities"] if e["id"] == entity_id)

    def test_seek_equals_linear_hash_and_checkpoint_converges(self):
        for i in range(1, 4):
            self.integrator.ingest(self.tx(i, i, [{
                "op": "update", "id": "pawn:7", "generation": 0,
                "components": {"transform": {"x": i}},
            }]))
        cp = self.integrator.checkpoint()
        self.assertEqual(cp["state_hash"], self.integrator.state_at(3)["state_hash"])
        for i in range(4, 7):
            self.integrator.ingest(self.tx(i, i, [{
                "op": "update", "id": "pawn:7", "generation": 0,
                "components": {"transform": {"x": i}},
            }]))
        linear = self.integrator.current_snapshot()
        sought = self.integrator.state_at(6)
        self.assertEqual(linear["state_hash"], sought["state_hash"])
        self.assertEqual(6, self.by_id(sought)["components"]["transform"]["x"])

    def test_lifecycle_generation_reuse_and_stale_refusal(self):
        self.integrator.ingest(self.tx(1, 1, [
            {"op": "destroy", "id": "pawn:7", "generation": 0},
            {"op": "create", "entity": entity(generation=1, x=50)},
        ]))
        self.assertEqual(1, self.by_id(
            self.integrator.current_snapshot())["generation"])
        before = self.integrator.current_snapshot()
        with self.assertRaises(ConflictError):
            self.integrator.ingest(self.tx(2, 2, [{
                "op": "update", "id": "pawn:7", "generation": 0,
                "components": {"transform": {"x": 1}},
            }]))
        self.assertEqual(before, self.integrator.current_snapshot())

    def test_null_is_value_and_absent_is_unchanged(self):
        self.integrator.ingest(self.tx(1, 1, [{
            "op": "update", "id": "pawn:7", "generation": 0,
            "components": {"transform": {"x": None}},
        }]))
        transform = self.by_id(
            self.integrator.current_snapshot())["components"]["transform"]
        self.assertIsNone(transform["x"])
        self.assertIsNone(transform["y"])
        self.assertEqual(10, self.by_id(
            self.integrator.current_snapshot())["components"]["inventory"]["ammo"])

    def test_renderer_provenance_component_envelopes_are_retained(self):
        envelope = {
            "provenance": "recorded",
            "value": {"active_node": 17, "cycle": 0.25},
        }
        unavailable = {
            "provenance": "unavailable",
            "reason": "not present in the source capture",
        }
        self.integrator.ingest(self.tx(1, 1, [{
            "op": "update", "id": "pawn:7", "generation": 0,
            "components": {"animation": envelope, "ragdoll": unavailable},
        }]))
        components = self.by_id(
            self.integrator.current_snapshot())["components"]
        self.assertEqual(envelope, components["animation"])
        self.assertEqual(unavailable, components["ragdoll"])

    def test_discontinuity_requires_complete_checkpoint_and_preserves_old_seek(self):
        self.integrator.ingest(self.tx(1, 1, [{
            "op": "update", "id": "pawn:7", "generation": 0,
            "components": {"transform": {"x": 1}},
        }]))
        discontinuity = StateIntegrator.make_checkpoint(
            self.manifest["stream_id"], 100, 0, 2,
            [entity(generation=3, x=900)], discontinuity="gap",
            reason="capture transport resumed from a full sample")
        self.integrator.checkpoint(discontinuity)
        self.assertEqual("gap", self.integrator.current_snapshot()[
            "discontinuity"]["kind"])
        self.assertEqual("gap", self.integrator.state_at(100)[
            "discontinuity"]["kind"])
        self.assertEqual(1, self.by_id(self.integrator.state_at(1))[
            "components"]["transform"]["x"])
        self.assertEqual(900, self.by_id(self.integrator.state_at(100))[
            "components"]["transform"]["x"])
        with self.assertRaises(ConflictError):
            self.integrator.ingest(self.tx(2, 101, [{
                "op": "update", "id": "pawn:7", "generation": 3,
                "components": {"transform": {"x": 901}},
            }]))

    def test_action_context_has_pre_and_post_transaction_state(self):
        self.integrator.ingest(self.tx(1, 10, [
            {"op": "update", "id": "pawn:7", "generation": 0,
             "components": {"transform": {"x": 8}}},
            {"op": "action", "action_id": "cmd:41/fire", "kind": "attack",
             "actor": {"id": "pawn:7", "generation": 0},
             "payload": {"button": "mouse1", "poll_sequence": 41}},
            {"op": "update", "id": "pawn:7", "generation": 0,
             "components": {"inventory": {"ammo": 9}}},
        ]))
        context = self.integrator.action_context("cmd:41/fire")
        pre, post = self.by_id(context["pre"]), self.by_id(context["post"])
        self.assertEqual(8, pre["components"]["transform"]["x"])
        self.assertEqual(10, pre["components"]["inventory"]["ammo"])
        self.assertEqual(9, post["components"]["inventory"]["ammo"])
        self.assertNotEqual(context["pre"]["state_hash"],
                            context["post"]["state_hash"])

    def test_order_base_and_post_hash_corruption_are_refused_atomically(self):
        initial = self.integrator.current_snapshot()
        with self.assertRaises(ConflictError):
            self.integrator.ingest(self.tx(2, 1, [{
                "op": "update", "id": "pawn:7", "generation": 0,
                "components": {"transform": {"x": 1}},
            }]))
        with self.assertRaises(ConflictError):
            self.integrator.ingest(self.tx(1, 1, [{
                "op": "update", "id": "pawn:7", "generation": 0,
                "components": {"transform": {"x": 1}},
            }], base_hash="0" * 64))
        with self.assertRaises(ConflictError):
            self.integrator.ingest(self.tx(1, 1, [{
                "op": "update", "id": "pawn:7", "generation": 0,
                "components": {"transform": {"x": 1}},
            }], post_hash="f" * 64))
        self.assertEqual(initial, self.integrator.current_snapshot())

    def test_persistence_reopen(self):
        self.integrator.ingest(self.tx(1, 7, [{
            "op": "update", "id": "pawn:7", "generation": 0,
            "components": {"transform": {"x": 77}},
        }]))
        expected = self.integrator.current_snapshot()
        self.integrator.close()
        self.integrator = StateIntegrator.open(
            self.root, self.manifest["stream_id"])
        self.assertEqual(expected, self.integrator.current_snapshot())
        self.assertEqual(expected["state_hash"],
                         self.integrator.state_at(7)["state_hash"])

    def test_batch_preserves_request_order_and_duplicates(self):
        for i in range(1, 5):
            self.integrator.ingest(self.tx(i, i, [{
                "op": "update", "id": "pawn:7", "generation": 0,
                "components": {"transform": {"x": i}},
            }]))
        batch = self.integrator.batch_states([
            {"tick": 4, "subtick": 0}, {"tick": 2, "subtick": 0},
            {"tick": 4, "subtick": 0},
        ])
        self.assertEqual([4, 2, 4], [self.by_id(s)["components"][
            "transform"]["x"] for s in batch])
        self.assertEqual(batch[0], batch[2])

    def test_batch_ingest_is_atomic(self):
        first = self.tx(1, 1, [{
            "op": "update", "id": "pawn:7", "generation": 0,
            "components": {"transform": {"x": 1}},
        }])
        # second base hash cannot be known from the first document and is bad.
        second = dict(self.tx(2, 2, [{
            "op": "update", "id": "pawn:7", "generation": 0,
            "components": {"transform": {"x": 2}},
        }]))
        with self.assertRaises(ConflictError):
            self.integrator.ingest_many([first, second])
        self.assertEqual(0, self.integrator.stats()["transactions"])


if __name__ == "__main__":
    unittest.main()
