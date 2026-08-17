from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cs_render_server as server  # noqa: E402
import cs_render_service as service  # noqa: E402
from variant_config import (  # noqa: E402
    VariantConfigurationError,
    get_variant,
    resolve_profile_request_variant,
)
from state_replay.integrator import state_hash  # noqa: E402

CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "variant_capture_session", ROOT / "scripts/cs2_capture_session.py"
)
assert CAPTURE_SPEC and CAPTURE_SPEC.loader
capture = importlib.util.module_from_spec(CAPTURE_SPEC)
sys.modules[CAPTURE_SPEC.name] = capture
CAPTURE_SPEC.loader.exec_module(capture)


def write_profile(root: Path, variant: str, **values) -> Path:
    profile = root / f"{variant}.profile.json"
    profile.write_text(json.dumps({
        "game_variant": variant,
        "asset_root": ".",
        "flags": [],
        **values,
    }))
    return profile


class RegistryTests(unittest.TestCase):
    def test_closed_registry_has_exact_backends_and_no_macos_cs2_binary(self):
        cs2 = get_variant("cs2")
        legacy = get_variant("csgo_legacy")
        self.assertEqual(cs2.renderer.kind, "python_subprocess")
        self.assertEqual(cs2.renderer.module, "counter_strike_render.gpu_render")
        self.assertEqual(cs2.client_executables_macos, ())
        self.assertEqual(legacy.renderer.kind, "python_callable")
        self.assertEqual(
            f"{legacy.renderer.module}:{legacy.renderer.callable}",
            "counter_strike_render.source1_backend:render_session",
        )
        self.assertIn("csgo_osx64", legacy.foreground_names_macos)
        with self.assertRaises(VariantConfigurationError):
            get_variant("whichever-imports")

    def test_request_may_inherit_profile_but_mismatch_is_rejected(self):
        self.assertEqual(
            resolve_profile_request_variant("csgo_legacy", None).id,
            "csgo_legacy",
        )
        with self.assertRaisesRegex(VariantConfigurationError, "mismatch"):
            resolve_profile_request_variant("csgo_legacy", "cs2")


class RequestSelectionTests(unittest.TestCase):
    def test_profile_variant_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td) / "profile.json"
            profile.write_text(json.dumps({"flags": []}))
            with self.assertRaisesRegex(ValueError, "profile game_variant is required"):
                service.profile_game_variant(profile)

    def test_cs2_and_legacy_use_disjoint_input_contracts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cs2_profile = write_profile(root, "cs2")
            legacy_profile = write_profile(root, "csgo_legacy")
            cs2 = server.normalize_request({
                "profile": str(cs2_profile), "logstream": "/recording.tard",
            })
            legacy = server.normalize_request({
                "profile": str(legacy_profile), "snapshot": "/state.json",
            })
            self.assertEqual(cs2["game_variant"], "cs2")
            self.assertNotIn("snapshot", cs2)
            self.assertEqual(legacy["game_variant"], "csgo_legacy")
            self.assertNotIn("logstream", legacy)
            with self.assertRaisesRegex(ValueError, "does not accept logstream"):
                server.normalize_request({
                    "profile": str(legacy_profile),
                    "game_variant": "csgo_legacy",
                    "logstream": "/wrong.tard",
                    "snapshot": "/state.json",
                })
            with self.assertRaisesRegex(ValueError, "mismatch"):
                server.normalize_request({
                    "profile": str(legacy_profile),
                    "game_variant": "cs2",
                    "logstream": "/wrong.tard",
                })


class LegacyReferenceEndToEndTests(unittest.TestCase):
    @staticmethod
    def snapshot() -> dict:
        def recorded(value):
            return {"provenance": "recorded", "value": value}

        entities = [
            {
                "id": "camera", "generation": 0, "class": "camera",
                "components": {"camera": recorded({
                    "origin": [0, 0, 0], "yaw_degrees": 0,
                    "pitch_degrees": 0, "fov": 90,
                })},
            },
            {
                "id": "triangle", "generation": 0, "class": "prop",
                "components": {"source1_geometry": recorded({
                    "space": "world",
                    "vertices": [[8, -2, -2], [8, 2, -2], [8, 0, 2]],
                    "triangles": [[0, 1, 2]], "color": [200, 90, 40],
                })},
            },
        ]
        return {
            "schema": "tardigrade/state-snapshot/v1",
            "stream_id": "legacy-bot-1",
            "tick": 10,
            "subtick": 0,
            "state_hash": state_hash(entities),
            "entities": entities,
        }

    def test_explicit_legacy_profile_reaches_callable_and_produces_reference_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = write_profile(
                root, "csgo_legacy", mode="canonical",
                allow_synthetic_geometry=True,
            )
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps(self.snapshot()))
            out = root / "out"
            result = service.render_snapshot(
                snapshot, profile, out, width=32, height=24,
            )
            self.assertEqual(result["game_variant"], "csgo_legacy")
            self.assertEqual(result["mode"], "canonical")
            self.assertEqual(result["fidelity"], "deterministic-reference")
            self.assertEqual((out / "reference.ppm").read_bytes()[:2], b"P6")
            scene = json.loads((out / "scene.json").read_text())
            self.assertEqual(
                scene["frames"][0]["state_hash"], self.snapshot()["state_hash"]
            )
            self.assertEqual(len(scene["frames"][0]["color_sha256"]), 64)
            self.assertEqual(len(scene["frames"][0]["depth_sha256"]), 64)

    def test_synthetic_geometry_is_not_an_implicit_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = write_profile(root, "csgo_legacy", mode="canonical")
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps(self.snapshot()))
            with self.assertRaisesRegex(service.RenderRefusal,
                                        "synthetic-geometry-disabled"):
                service.render_snapshot(snapshot, profile, root / "out", 16, 16)

    def test_legacy_profile_resolves_explicit_vpk_search_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vpk = root / "pak01_dir.vpk"
            vpk.write_bytes(b"fixture")
            profile = write_profile(
                root,
                "csgo_legacy",
                mode="canonical",
                map_asset="maps/de_dust2.bsp",
                vpk_paths=[vpk.name],
                reference_fov_degrees=90,
            )
            config = service.load_profile_configuration(
                profile, "csgo_legacy"
            )
            self.assertEqual("maps/de_dust2.bsp", config["map_asset"])
            self.assertEqual([str(vpk)], config["vpk_paths"])
            self.assertEqual(90.0, config["reference_fov_degrees"])

    def test_service_rejects_backend_pixels_joined_to_wrong_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = write_profile(
                root, "csgo_legacy", mode="canonical",
                allow_synthetic_geometry=True,
            )
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps(self.snapshot()))

            def mismatched_backend(_variant, argv):
                scene_path = Path(argv[argv.index("--scene-out") + 1])
                frame_path = Path(argv[argv.index("--frame-out") + 1])
                depth_path = Path(argv[argv.index("--depth-out") + 1])
                ppm = b"P6\n1 1\n255\n\x00\x00\x00"
                frame_path.write_bytes(ppm)
                depth = b"\xff" * (8 * 8 * 8)
                depth_path.write_bytes(depth)
                scene_path.write_text(json.dumps({
                    "schema": "tardigrade/source1-reference-render/v1",
                    "frames": [{
                        "state_hash": "a-different-state",
                        "color_sha256": "0" * 64,
                        "depth_sha256": "1" * 64,
                        "depth_bytes": len(depth),
                        "depth_encoding": {"scalar": "uint64-le"},
                        "width": 8,
                        "height": 8,
                        "output": str(root / "some-other-frame.ppm"),
                        "depth_output": str(depth_path),
                    }],
                }))
                return subprocess.CompletedProcess(argv, 0, "", "")

            with mock.patch.object(
                    service, "_invoke_renderer", side_effect=mismatched_backend):
                with self.assertRaisesRegex(service.RenderRefusal,
                                            "do not bind.*state snapshot"):
                    service.render_snapshot(
                        snapshot, profile, root / "out", width=8, height=8,
                    )


class CaptureVariantTests(unittest.TestCase):
    def test_legacy_defaults_select_exact_macos_lifecycle_names(self):
        legacy = get_variant("csgo_legacy")
        self.assertEqual(
            legacy.foreground_names_macos,
            ("csgo_osx64", "Counter-Strike: Global Offensive"),
        )
        self.assertEqual(
            legacy.foreground_bundle_ids_macos, ("com.valvesoftware.csgo",)
        )
        detected = capture.detect_client(
            {"install_dir_name": "Counter-Strike Global Offensive"},
            "csgo_legacy",
        )
        self.assertTrue(any(path.endswith("csgo_osx64")
                            for path in detected["runnable_client_candidates"]))

    def test_supervisor_refuses_native_macos_cs2_assumption(self):
        with mock.patch.object(platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(SystemExit, "no native macOS client"):
                capture.run([
                    "--consent", "--game-variant", "cs2",
                    "--device", "0x046d:0xb041", "--demo-dir", "/tmp",
                ])


if __name__ == "__main__":
    unittest.main()
