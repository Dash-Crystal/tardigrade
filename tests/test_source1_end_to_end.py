from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from counter_strike_render.source1_backend import (
    Source1RenderConfig,
    Source1RenderRefusal,
    render_snapshot,
)
from state_replay import StateIntegrator
from state_replay.source1_adapter import Source1ReplayAdapter
from state_replay.source1_demo import Source1DemoReader

try:
    from .test_source1_demo import (
        command,
        entity_enter,
        header,
        net,
        packet_entities,
        send_tables,
    )
    from .test_source1_render_formats import _fixture_bsp
except ImportError:  # unittest discovery imports test modules as top-level names.
    from test_source1_demo import (  # type: ignore[no-redef]
        command,
        entity_enter,
        header,
        net,
        packet_entities,
        send_tables,
    )
    from test_source1_render_formats import _fixture_bsp  # type: ignore[no-redef]


class Source1IngestRenderEndToEnd(unittest.TestCase):
    def test_demo_packet_integrates_then_renders_same_hashed_state(self):
        # Keep this fixture's edict deliberately non-visual.  The joined proof
        # renders the recorded camera plus BSP world without silently dropping
        # an actor that should have a model.  The companion test below proves
        # adapter-native player evidence fails closed.
        nonvisual_tables = send_tables().replace(b"m_iHealth", b"m_iWidget")
        demo = (
            header()
            + command(6, 0, nonvisual_tables)
            + command(2, 1, net(26, packet_entities(entity_enter())))
            + command(7, 2)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path = root / "journal"
            Source1ReplayAdapter().ingest(
                Source1DemoReader.from_bytes(demo), journal_path, "e2e/source1"
            )
            journal = StateIntegrator.open(journal_path, "e2e/source1")
            try:
                snapshot = journal.current_snapshot()
            finally:
                journal.close()

            map_path = root / "fixture.bsp"
            map_path.write_bytes(_fixture_bsp())
            rendered = render_snapshot(
                snapshot,
                Source1RenderConfig(
                    asset_root=root,
                    map_path=map_path,
                    width=48,
                    height=32,
                    reference_fov_degrees=90.0,
                ),
            )

            self.assertEqual(snapshot["state_hash"], rendered.scene.state_hash)
            self.assertEqual(len(rendered.depth_u64le), 48 * 32 * 8)
            self.assertIn(bytes((92, 104, 112)), rendered.ppm)
            self.assertEqual(
                "profile-derived", rendered.scene.camera.fov_provenance
            )

    def test_ingested_player_without_model_binding_is_not_silently_omitted(self):
        demo = (
            header()
            + command(6, 0, send_tables())
            + command(2, 1, net(26, packet_entities(entity_enter())))
            + command(7, 2)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path = root / "journal"
            Source1ReplayAdapter().ingest(
                Source1DemoReader.from_bytes(demo), journal_path, "e2e/player"
            )
            journal = StateIntegrator.open(journal_path, "e2e/player")
            try:
                snapshot = journal.current_snapshot()
            finally:
                journal.close()
            map_path = root / "fixture.bsp"
            map_path.write_bytes(_fixture_bsp())
            with self.assertRaisesRegex(
                Source1RenderRefusal, "missing-visual-asset"
            ):
                render_snapshot(
                    snapshot,
                    Source1RenderConfig(
                        asset_root=root,
                        map_path=map_path,
                        width=48,
                        height=32,
                        reference_fov_degrees=90.0,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
