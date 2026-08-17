"""Focused service-boundary tests; no torch, GPU, or renderer execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cs_render_server as server  # noqa: E402
import cs_render_service as service  # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_profile_resolves_asset_root_relative_to_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            assets = root / "assets"
            assets.mkdir()
            world = assets / "world.pt"
            world.write_bytes(b"pack")
            profile = root / "profile.json"
            profile.write_text(json.dumps({
                "game_variant": "cs2",
                "asset_root": "assets",
                "flags": [["--world", "world.pt"],
                          ["--lod-weld", "derived"],
                          ["--animated-shadows"]],
            }))

            self.assertEqual(service.load_profile(profile), [
                "--world", str(world), "--lod-weld", "derived",
                "--animated-shadows",
            ])

    def test_binary_world_pack_gets_backwards_migration_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            world = Path(td) / "world.pt"
            world.write_bytes(b"not json")
            with self.assertRaisesRegex(ValueError, "old world-pack argument"):
                service.load_profile(world)

    def test_missing_profile_asset_is_a_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td) / "profile.json"
            profile.write_text(json.dumps({
                "game_variant": "cs2",
                "flags": [["--world", "missing.pt"]],
            }))
            with self.assertRaisesRegex(service.RenderRefusal,
                                        "canonical profile names assets"):
                service.load_profile(profile)


class RenderBoundaryTests(unittest.TestCase):
    @staticmethod
    def _camera_path():
        return {
            "n_ticks": 1,
            "n_future_steered": 0,
            "max_foreshadow_blend": 0.0,
            "ticks": [{"tick": 0}],
        }

    def test_render_view_constructs_atomic_argv_and_requires_mp4(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)

            def fake_run(argv, **kwargs):
                view_dir = Path(argv[argv.index("--session-out") + 1])
                (view_dir / "p2.mp4").write_bytes(b"video")
                return subprocess.CompletedProcess(argv, 0, "ok", "")

            flags = ["--world", "/assets/world.pt", "--lod-weld", "derived"]
            with mock.patch.object(service, "build_view_path",
                                   return_value=self._camera_path()), \
                    mock.patch.object(service.subprocess, "run",
                                      side_effect=fake_run) as run:
                result = service.render_view(
                    object(), "/recording.tard", "ego", [2], 0, None, 16,
                    640, 360, flags, str(out))

            argv = run.call_args.args[0]
            start = argv.index("--world")
            self.assertEqual(argv[start:start + 4], flags)
            self.assertEqual(run.call_args.kwargs["cwd"], service._REPO_ROOT)
            self.assertEqual(result["outputs"], ["p2.mp4"])

    def test_render_view_refuses_nonzero_renderer_exit(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(service, "build_view_path",
                                  return_value=self._camera_path()), \
                mock.patch.object(
                    service.subprocess, "run",
                    return_value=subprocess.CompletedProcess(
                        [], 9, "", "renderer exploded")):
            with self.assertRaisesRegex(service.RenderRefusal,
                                        "exit code 9[\\s\\S]*renderer exploded"):
                service.render_view(object(), "/r.tard", "ego", [0], 0,
                                    None, 16, 640, 360, [], td)

    def test_render_view_refuses_zero_exit_without_output(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(service, "build_view_path",
                                  return_value=self._camera_path()), \
                mock.patch.object(
                    service.subprocess, "run",
                    return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaisesRegex(service.RenderRefusal,
                                        "did not produce.*p0.mp4"):
                service.render_view(object(), "/r.tard", "ego", [0], 0,
                                    None, 16, 640, 360, [], td)

    def test_render_view_does_not_accept_a_stale_output(self):
        with tempfile.TemporaryDirectory() as td:
            view_dir = Path(td) / "ego"
            view_dir.mkdir()
            (view_dir / "p0.mp4").write_bytes(b"old video")
            with mock.patch.object(service, "build_view_path",
                                   return_value=self._camera_path()), \
                    mock.patch.object(
                        service.subprocess, "run",
                        return_value=subprocess.CompletedProcess([], 0, "", "")):
                with self.assertRaisesRegex(service.RenderRefusal, "stale"):
                    service.render_view(object(), "/r.tard", "ego", [0], 0,
                                        None, 16, 640, 360, [], td)

    def test_render_match_loads_profile_once_for_all_views(self):
        class Recording:
            match_id = "m"
            players = [object(), object()]

        with tempfile.TemporaryDirectory() as td:
            recording = Path(td) / "recording.tard"
            recording.write_bytes(b"wire")
            profile = Path(td) / "profile.json"
            profile.write_text(json.dumps({
                "game_variant": "cs2", "flags": [],
            }))
            with mock.patch.object(service, "load_profile",
                                   return_value=["--world", "/w.pt"]) as load, \
                    mock.patch.object(service, "TardigradeV21Recording",
                                      return_value=Recording()), \
                    mock.patch.object(service, "render_view",
                                      return_value={"rc": 0, "players": [],
                                                    "outputs": []}) as render:
                service.render_match(str(recording), str(profile), td,
                                     views=("ego", "gow"), players=[0])

            load.assert_called_once_with(str(profile), "cs2")
            self.assertEqual(render.call_count, 2)
            for call in render.call_args_list:
                self.assertEqual(call.args[9], ["--world", "/w.pt"])


class RequestTests(unittest.TestCase):
    def test_world_field_has_explicit_migration_error(self):
        with self.assertRaisesRegex(ValueError, "'world'.*no longer accepted"):
            server.normalize_request({"logstream": "/r", "world": "/w"})

    def test_unknown_fields_and_invalid_views_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td) / "profile.json"
            profile.write_text(json.dumps({"game_variant": "cs2", "flags": []}))
            base = {"logstream": "/r", "profile": str(profile)}
            with self.assertRaisesRegex(ValueError, "unknown request fields"):
                server.normalize_request({**base, "surprise": True})
            with self.assertRaisesRegex(ValueError, "unknown views"):
                server.normalize_request({**base, "views": ["ego", "escape"]})

    def test_job_identity_is_stable_and_output_stays_under_root(self):
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td) / "profile.json"
            profile.write_text(json.dumps({"game_variant": "cs2", "flags": []}))
            request_a = server.normalize_request({
                "logstream": "/r", "profile": str(profile), "views": ["ego"],
            })
            request_b = server.normalize_request({
                "views": ["ego"], "profile": str(profile), "logstream": "/r",
            })
            self.assertEqual(server.job_identity(request_a),
                             server.job_identity(request_b))
            output = server.job_output_dir(td, server.job_identity(request_a))
            self.assertEqual(os.path.commonpath((os.path.realpath(td), output)),
                             os.path.realpath(td))


if __name__ == "__main__":
    unittest.main()
