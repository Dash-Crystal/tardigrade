"""Strictly dispatch CS2 TARD jobs or legacy CS:GO state-snapshot jobs.

Views:
  ego   -- naive egocentric: the player's own eye path, verbatim (what a plain
           demo replay shows). No third-person transform.
  gow   -- retrocausal Gears-of-War third-person (demo_camera GOW).
  sm64  -- retrocausal Mario-64/Lakitu spring-orbit (demo_camera SM64).

For CS2, one logstream -> per (player, view) a renderer-ready camera path -> the ported
renderer in SESSION mode (one pack load per view) -> per-player videos.
demo_match_export.build_timeline_from_tardigrade does the wire READ; ego reuses
its focus_rows directly. This module is the engine; cs_render_server wraps it in
a boring REST endpoint. Legacy CS:GO never enters that path: it consumes only
normalized ``tardigrade/state-snapshot/v1`` JSON and emits a deterministic
Source 1 reference scene plus P6 pixels.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys

# Direct execution from the checkout must obey the same import contract as
# the historical flat deployment directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKOUT = os.path.dirname(_HERE)
for _module_dir in (os.path.join(_CHECKOUT, "src"),
                    os.path.join(_CHECKOUT, "src", "gpu_render_libs")):
    if os.path.isdir(_module_dir) and _module_dir not in sys.path:
        sys.path.insert(0, _module_dir)

import demo_camera as dc
from demo_match_export import build_timeline_from_tardigrade
from tardigrade_v21 import TardigradeV21Recording
from variant_config import GameVariant, get_variant, resolve_profile_request_variant


def _locate_core():
    """Find iji_model/counter_strike_render/gpu_render.py + the dir that holds
    iji_model (the cwd so its absolute imports resolve). Robust to both the
    repo layout (harness/gpu_render/) and a flat deploy dir."""
    rel = os.path.join("src", "counter_strike_render", "gpu_render.py")
    for base in (_HERE, os.path.dirname(_HERE),
                 os.path.dirname(os.path.dirname(_HERE))):
        cand = os.path.join(base, rel)
        if os.path.isfile(cand):
            return cand, base
    raise FileNotFoundError(f"gpu_render.py not found near {_HERE}")


_CORE, _REPO_ROOT = _locate_core()
VIEWS = ("ego", "gow", "sm64")


class RenderRefusal(RuntimeError):
    """A render was not produced; callers must not consume partial output."""


_PROFILE_PATH_FLAGS = {
    "--world", "--fam-side", "--irr-npy", "--lightmap", "--cube-probes",
    "--probe-npz", "--skyvis", "--lights", "--vpaint", "--ibl-cube",
    "--content-root", "--playermodels", "--playermodel-bundles",
    "--playermodel-clips", "--playermodel-weapon-map",
    "--playermodel-weapons", "--viewmodel-bundles", "--weapon-bundles",
}
_PATH_SUFFIXES = (
    ".json", ".npy", ".npz", ".pt", ".bin", ".glb", ".gltf",
    ".vpk", ".png", ".exr",
)


# ---------------------------------------------------------------------------
# Canonical render profile -- OWNER RULING (2026-08-11): there is no
# configuration that renders without the task's assets. Every render this
# service performs uses the full-stack profile below; a request chooses WHAT
# to render (logstream, players, views, ticks, resolution) and never how
# little. Missing profile assets are a REFUSAL naming the gap, not a fallback.
# ---------------------------------------------------------------------------

def _read_profile_document(profile_path):
    profile_path = os.path.abspath(os.fspath(profile_path))
    try:
        with open(profile_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"{profile_path} is not a render profile JSON document. The "
            "old world-pack argument is no longer accepted: create a "
            "profile containing game_variant, asset_root, and flags, then "
            "pass its path."
        ) from exc
    except OSError as exc:
        raise ValueError(f"cannot read render profile {profile_path}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("flags"), list):
        raise ValueError(
            f"invalid render profile {profile_path}: expected an object with "
            "an explicit 'game_variant' and a 'flags' list")
    # Validate here, even for the server's lightweight inference read.  A
    # profile without this field is never silently treated as CS2.
    get_variant(doc.get("game_variant"), source="profile game_variant")
    return profile_path, doc


def profile_game_variant(profile_path):
    """Return the canonical explicit variant embedded in a profile."""
    _, doc = _read_profile_document(profile_path)
    return get_variant(doc["game_variant"], source="profile game_variant").id


def load_profile(profile_path, expected_variant=None):
    """Load the deployment's render profile: {"asset_root": dir, "flags":
    [[flag, value|null], ...]}. Relative values resolve under asset_root.
    Every flag value that looks like a path MUST exist -- else refuse."""
    profile_path, doc = _read_profile_document(profile_path)
    variant = resolve_profile_request_variant(
        doc["game_variant"], expected_variant
    )
    profile_dir = os.path.dirname(profile_path)
    root_value = doc.get("asset_root", profile_dir)
    if not isinstance(root_value, (str, os.PathLike)):
        raise ValueError(
            f"invalid render profile {profile_path}: asset_root must be a "
            "path string")
    root_value = os.fspath(root_value)
    root = (root_value if os.path.isabs(root_value)
            else os.path.join(profile_dir, root_value))
    root = os.path.abspath(root)
    flags, missing = [], []
    for entry in doc["flags"]:
        if not isinstance(entry, list) or not 1 <= len(entry) <= 2:
            raise ValueError(
                f"invalid render profile {profile_path}: every flags entry "
                "must be [flag] or [flag, value]")
        flag, value = entry[0], (entry[1] if len(entry) > 1 else None)
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise ValueError(
                f"invalid render profile {profile_path}: flag {flag!r} must "
                "be a string beginning with '--'")
        if value is None:
            flags.append(str(flag))
            continue
        sval = str(value)
        resolved = sval if os.path.isabs(sval) else os.path.join(root, sval)
        path_like = (flag in _PROFILE_PATH_FLAGS or "/" in sval or
                     sval.startswith(".") or
                     sval.lower().endswith(_PATH_SUFFIXES))
        if path_like:
            # path-like: must exist
            if not os.path.exists(resolved):
                missing.append(f"{flag} -> {resolved}")
                continue
            flags.extend((str(flag), resolved))
        else:
            flags.extend((str(flag), sval))   # plain value (numbers, modes)
    if missing:
        raise RenderRefusal(
            "REFUSING to serve renders: the canonical profile names assets "
            "that are not present on this node:\n  " + "\n  ".join(missing) +
            "\nThere is no reduced-asset render mode. Stage the assets or "
            "fix the profile.")
    return flags


def load_profile_configuration(profile_path, expected_variant=None):
    """Load flags plus variant-specific, resolved profile configuration."""
    profile_path, doc = _read_profile_document(profile_path)
    variant = resolve_profile_request_variant(
        doc["game_variant"], expected_variant
    )
    flags = load_profile(profile_path, variant.id)
    root_value = doc.get("asset_root", os.path.dirname(profile_path))
    root_value = os.fspath(root_value)
    asset_root = os.path.abspath(
        root_value if os.path.isabs(root_value)
        else os.path.join(os.path.dirname(profile_path), root_value)
    )
    map_path = doc.get("map")
    if map_path is not None:
        if not isinstance(map_path, str) or not map_path.strip():
            raise ValueError(f"invalid render profile {profile_path}: map must be a path string")
        map_path = os.path.abspath(
            map_path if os.path.isabs(map_path)
            else os.path.join(asset_root, map_path)
        )
        if not os.path.isfile(map_path):
            raise RenderRefusal(
                f"REFUSING to serve renders: profile map is absent: {map_path}"
            )
    map_asset = doc.get("map_asset")
    if map_asset is not None:
        if (not isinstance(map_asset, str) or not map_asset.strip()
                or os.path.isabs(map_asset)
                or not map_asset.casefold().endswith(".bsp")):
            raise ValueError(
                f"invalid render profile {profile_path}: map_asset must be a "
                "relative Source 1 .bsp asset path"
            )
        if map_path is not None:
            raise ValueError(
                f"invalid render profile {profile_path}: configure either "
                "map or map_asset, not both"
            )
    raw_vpks = doc.get("vpk_paths", [])
    if (isinstance(raw_vpks, (str, bytes, dict))
            or not isinstance(raw_vpks, list)
            or any(not isinstance(item, str) or not item.strip()
                   for item in raw_vpks)):
        raise ValueError(
            f"invalid render profile {profile_path}: vpk_paths must be an "
            "array of explicit _dir.vpk paths"
        )
    vpk_paths = []
    for item in raw_vpks:
        resolved = os.path.abspath(
            item if os.path.isabs(item) else os.path.join(asset_root, item)
        )
        if not resolved.casefold().endswith("_dir.vpk"):
            raise ValueError(
                f"invalid render profile {profile_path}: VPK must name an "
                f"explicit _dir.vpk: {item!r}"
            )
        if not os.path.isfile(resolved):
            raise RenderRefusal(
                "REFUSING to serve legacy CS:GO renders: configured VPK is "
                f"absent: {resolved}"
            )
        vpk_paths.append(resolved)
    if variant.id == "csgo_legacy" and not os.path.isdir(asset_root):
        raise RenderRefusal(
            "REFUSING to serve legacy CS:GO renders: profile asset_root "
            f"is not a directory: {asset_root}"
        )
    mode = doc.get("mode", "canonical")
    if mode not in ("canonical", "authoritative"):
        raise ValueError(
            f"invalid render profile {profile_path}: mode must be canonical or authoritative"
        )
    allow_synthetic = doc.get("allow_synthetic_geometry", False)
    if not isinstance(allow_synthetic, bool):
        raise ValueError(
            f"invalid render profile {profile_path}: allow_synthetic_geometry must be boolean"
        )
    reference_fov = doc.get("reference_fov_degrees")
    if reference_fov is not None:
        if (isinstance(reference_fov, bool)
                or not isinstance(reference_fov, (int, float))
                or not math.isfinite(float(reference_fov))
                or not 1.0 <= float(reference_fov) < 179.0):
            raise ValueError(
                f"invalid render profile {profile_path}: "
                "reference_fov_degrees must be a finite number in [1, 179)"
            )
        if variant.id != "csgo_legacy":
            raise ValueError(
                "reference_fov_degrees is only valid for a legacy CS:GO "
                "canonical render profile"
            )
        if mode != "canonical":
            raise ValueError(
                "reference_fov_degrees is profile-derived and forbidden in "
                "authoritative mode"
            )
        reference_fov = float(reference_fov)
    if variant.id == "csgo_legacy" and flags:
        raise ValueError(
            "legacy CS:GO profiles configure asset_root, map, mode, and "
            "allow_synthetic_geometry, reference_fov_degrees, map_asset, and "
            "vpk_paths directly; "
            "Source 2 flags are refused"
        )
    return {
        "path": profile_path,
        "game_variant": variant,
        "asset_root": asset_root,
        "map": map_path,
        "map_asset": map_asset,
        "vpk_paths": vpk_paths,
        "flags": flags,
        "mode": mode,
        "allow_synthetic_geometry": allow_synthetic,
        "reference_fov_degrees": reference_fov,
    }


def _ego_path(tl, game_variant="cs2"):
    """Naive egocentric camera path: the focus player's own eye, verbatim."""
    ticks = []
    for r in tl.focus_rows:
        ticks.append({
            "tick": int(r["tick"]),
            "x": r["x"], "y": r["y"], "z": r["z"], "eye_z": r["eye_z"],
            "yaw_degrees": r["yaw_degrees"], "pitch_degrees": r["pitch_degrees"],
            "fire": bool(r.get("fire")),
            "active_weapon_name": r.get("active_weapon_name"),
            "is_alive": bool(r.get("is_alive", True)),
            "uses_future_sample": False,  # ego never looks ahead
        })
    schema_variant = "cs2" if game_variant == "cs2" else "csgo-legacy"
    return {"schema": f"iji/{schema_variant}-demo-ego-camera-path/v1",
            "game_variant": game_variant, "variant": "ego",
            "tick_rate": tl.tick_rate, "n_ticks": len(ticks),
            "n_future_steered": 0, "max_foreshadow_blend": 0.0, "ticks": ticks}


def build_view_path(recording, player_index, view, tick_begin, tick_end,
                    stride, native_rate=128, game_variant="cs2"):
    tl = build_timeline_from_tardigrade(recording, player_index,
                                        tick_begin, tick_end, stride)
    tl.tick_rate = native_rate
    if view == "ego":
        return _ego_path(tl, game_variant)
    path = dc.compute_camera_path(tl, dc.VARIANTS[view])
    path["game_variant"] = game_variant
    return path


def _invoke_renderer(variant: GameVariant, backend_args):
    """Invoke exactly the backend registered for ``variant``.

    Source 1 exposes a callable because it is an adapter rather than another
    copy of the Source 2 command.  Its stable boundary is
    ``render_session(argv: list[str], cwd: str)`` and it must return a
    subprocess.CompletedProcess-like value with returncode/stdout/stderr.
    """
    backend = variant.renderer
    if backend.kind == "python_subprocess":
        if variant.id != "cs2" or backend.module != "counter_strike_render.gpu_render":
            raise RenderRefusal(
                f"invalid subprocess backend registration for {variant.id!r}")
        return subprocess.run(
            [sys.executable, _CORE, *backend_args], cwd=_REPO_ROOT,
            capture_output=True, text=True,
        )
    if backend.kind != "python_callable":
        raise RenderRefusal(
            f"unsupported renderer backend kind {backend.kind!r} for {variant.id!r}")
    try:
        module = importlib.import_module(backend.module)
        function = getattr(module, backend.callable)
    except (ImportError, AttributeError) as exc:
        raise RenderRefusal(
            f"renderer backend for {variant.id!r} is unavailable: "
            f"{backend.module}:{backend.callable}: {exc}"
        ) from exc
    result = function(list(backend_args), cwd=_REPO_ROOT)
    if not all(hasattr(result, field) for field in ("returncode", "stdout", "stderr")):
        raise RenderRefusal(
            f"renderer backend {backend.module}:{backend.callable} violated "
            "the render_session result contract"
        )
    return result


def _nonempty_ppm(path):
    if not os.path.isfile(path) or os.path.getsize(path) <= len(b"P6\n1 1\n255\n"):
        return False
    with open(path, "rb") as fh:
        return fh.read(2) == b"P6"


def render_view(recording, recording_path, view, players, tick_begin, tick_end,
                stride, width, height, profile_flags, out_dir,
                game_variant="cs2"):
    """Build every player's camera path for one view and render them in ONE
    session (one pack load) -> per-player videos."""
    if game_variant != "cs2":
        raise RenderRefusal(
            "TARD camera/session rendering is a CS2 backend contract; legacy "
            "CS:GO must use render_snapshot with normalized state JSON"
        )
    view_dir = os.path.join(out_dir, view)
    os.makedirs(view_dir, exist_ok=True)
    agents, manifest = [], []
    for pi in players:
        path = build_view_path(
            recording, pi, view, tick_begin, tick_end, stride,
            game_variant=game_variant,
        )
        if path["n_ticks"] == 0:
            continue
        cj = os.path.join(view_dir, f"p{pi}.camera.json")
        with open(cj, "w") as fh:
            json.dump(path, fh)
        agents.append({"id": f"p{pi}", "camera_json": cj,
                       "tick_begin": 0, "tick_end": path["n_ticks"]})
        manifest.append({"player": pi, "n_ticks": path["n_ticks"],
                         "n_future_steered": path["n_future_steered"],
                         "max_foreshadow_blend": round(path["max_foreshadow_blend"], 3)})
    if not agents:
        return {"view": view, "players": [], "note": "no renderable players"}
    session = os.path.join(view_dir, "session.json")
    with open(session, "w") as fh:
        json.dump({"game_variant": game_variant, "agents": agents}, fh)
    cj0 = agents[0]["camera_json"]
    # Canonical profile FIRST (world, irradiance, lightmap, probes, fam_side,
    # post chain, ...); request-scoped flags after. Nothing here may reduce
    # the asset surface -- the unlit escape is deleted at both layers.
    backend_args = [*profile_flags,
                    "--session", session, "--session-out", view_dir,
                    "--camera-json", cj0, "--tick-begin", "0", "--tick-end", "1",
                    "--out", os.path.join(view_dir, "u.mp4"),
                    "--width", str(width), "--height", str(height), "--fps", "32"]
    expected = [f"{agent['id']}.mp4" for agent in agents]
    before = {}
    for name in expected:
        path = os.path.join(view_dir, name)
        if os.path.isfile(path):
            stat = os.stat(path)
            before[name] = (stat.st_mtime_ns, stat.st_size)
    variant = get_variant(game_variant)
    proc = _invoke_renderer(variant, backend_args)
    if proc.returncode:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RenderRefusal(
            f"renderer refused view {view!r} with exit code "
            f"{proc.returncode}:\n{tail}")

    # The current session sink streams one MP4 per agent. Older comments in
    # gpu_render called these tensors; accepting success merely because the
    # subprocess returned zero hid both that contract change and empty jobs.
    missing = []
    for name in expected:
        path = os.path.join(view_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            missing.append(name)
            continue
        stat = os.stat(path)
        if before.get(name) == (stat.st_mtime_ns, stat.st_size):
            missing.append(f"{name} (stale from an earlier invocation)")
    if missing:
        raise RenderRefusal(
            f"renderer exited successfully for view {view!r} but did not "
            f"produce the required non-empty session outputs: "
            f"{', '.join(missing)}")
    return {"view": view, "game_variant": variant.id,
            "renderer_backend": variant.renderer.descriptor(),
            "rc": 0, "players": manifest,
            "outputs": expected, "output_dir": view_dir}


def _validate_render_options(views, players, tick_begin, tick_end, stride,
                             width, height):
    if isinstance(views, str) or not isinstance(views, (list, tuple)):
        raise ValueError("views must be a list of view names")
    views = tuple(views)
    if not views:
        raise ValueError("views must contain at least one view")
    if len(set(views)) != len(views):
        raise ValueError("views must not contain duplicates")
    unknown = [view for view in views if view not in VIEWS]
    if unknown:
        raise ValueError(
            f"unknown views {unknown!r}; supported views are {list(VIEWS)!r}")
    if players != "all":
        if not isinstance(players, (list, tuple)) or not players:
            raise ValueError("players must be 'all' or a non-empty list")
        if any(isinstance(pi, bool) or not isinstance(pi, int) or pi < 0
               for pi in players):
            raise ValueError("player indices must be non-negative integers")
        if len(set(players)) != len(players):
            raise ValueError("players must not contain duplicates")
        players = list(players)
    for name, value, lower, upper in (
            ("tick_begin", tick_begin, 0, None),
            ("stride", stride, 1, None),
            ("width", width, 1, 16384),
            ("height", height, 1, 16384)):
        if (isinstance(value, bool) or not isinstance(value, int) or
                value < lower or (upper is not None and value > upper)):
            suffix = f" and <= {upper}" if upper is not None else ""
            raise ValueError(f"{name} must be an integer >= {lower}{suffix}")
    if (tick_end is not None and
            (isinstance(tick_end, bool) or not isinstance(tick_end, int) or
             tick_end <= tick_begin)):
        raise ValueError("tick_end must be null or an integer > tick_begin")
    return views, players


def render_match(recording_path, profile_path, out_dir, views=VIEWS,
                 players="all", tick_begin=0, tick_end=None, stride=16,
                 width=640, height=360, game_variant=None):
    views, players = _validate_render_options(
        views, players, tick_begin, tick_end, stride, width, height)
    if not isinstance(recording_path, (str, os.PathLike)):
        raise ValueError("recording path must be a filesystem path")
    if not isinstance(profile_path, (str, os.PathLike)):
        raise ValueError("profile path must be a filesystem path")
    recording_path = os.path.abspath(os.fspath(recording_path))
    profile_path = os.path.abspath(os.fspath(profile_path))
    # Resolve once so every view uses exactly the same asset surface even if
    # a profile is edited while a multi-view job is running.
    selected_variant = resolve_profile_request_variant(
        profile_game_variant(profile_path), game_variant
    )
    if selected_variant.id != "cs2":
        raise RenderRefusal(
            "legacy CS:GO cannot consume a TARD/logstream request; submit a "
            "normalized state-snapshot JSON request via 'snapshot'"
        )
    profile_flags = load_profile(profile_path, selected_variant.id)
    with open(recording_path, "rb") as fh:
        data = fh.read()
    rec = TardigradeV21Recording(data)
    if players == "all":
        players = list(range(len(rec.players)))
    elif any(pi >= len(rec.players) for pi in players):
        raise ValueError(
            f"player index outside recording range 0..{len(rec.players) - 1}")
    out_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    result = {"match_id": rec.match_id, "recording": recording_path,
              "profile": profile_path,
              "game_variant": selected_variant.id,
              "renderer_backend": selected_variant.renderer.descriptor(),
              "n_players": len(rec.players), "stride": stride,
              "resolution": [width, height], "views": {}}
    for view in views:
        r = render_view(rec, recording_path, view, players, tick_begin,
                        tick_end, stride, width, height, profile_flags, out_dir,
                        selected_variant.id)
        result["views"][view] = r
        print(f"[{view}] rc={r.get('rc')} "
              f"players={len(r.get('players', []))} "
              f"outputs={len(r.get('outputs', []))}", flush=True)
    with open(os.path.join(out_dir, "match_render.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def render_snapshot(snapshot_path, profile_path, out_dir, width=640, height=360,
                    game_variant="csgo_legacy"):
    """Render one normalized Source 1 state snapshot through its exact backend."""
    selected = get_variant(game_variant, source="request game_variant")
    if selected.id != "csgo_legacy":
        raise RenderRefusal(
            "snapshot rendering is only the csgo_legacy Source 1 contract"
        )
    config = load_profile_configuration(profile_path, selected.id)
    if not isinstance(snapshot_path, (str, os.PathLike)):
        raise ValueError("snapshot path must be a filesystem path")
    snapshot_path = os.path.abspath(os.fspath(snapshot_path))
    try:
        with open(snapshot_path, encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"snapshot must be readable normalized state JSON: {snapshot_path}: {exc}"
        ) from exc
    if not isinstance(snapshot, dict):
        raise ValueError("normalized state snapshot must be a JSON object")
    # The canonical integrator snapshot is intentionally engine-neutral and
    # has a closed top-level schema.  Variant provenance belongs to the
    # profile/request envelope, not inside this state document.
    documents = snapshot.get("frames") if "frames" in snapshot else [snapshot]
    if (isinstance(documents, (str, bytes, dict))
            or not isinstance(documents, list) or not documents
            or any(not isinstance(item, dict) or
                   item.get("schema") != "tardigrade/state-snapshot/v1"
                   for item in documents)):
        raise ValueError(
            "snapshot input must be one tardigrade/state-snapshot/v1 object "
            "or a non-empty {frames: [...]} batch of them"
        )
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 16384:
        raise ValueError("width must be an integer in 1..16384")
    if isinstance(height, bool) or not isinstance(height, int) or not 1 <= height <= 16384:
        raise ValueError("height must be an integer in 1..16384")
    os.makedirs(out_dir, exist_ok=True)
    scene_out = os.path.join(out_dir, "scene.json")
    frame_out = os.path.join(
        out_dir, "reference.ppm" if len(documents) == 1 else "reference-frames"
    )
    depth_out = os.path.join(
        out_dir,
        "reference.depth.u64le" if len(documents) == 1 else "reference-depth",
    )
    backend_args = [
        "--snapshot-json", snapshot_path,
        "--asset-root", config["asset_root"],
        "--scene-out", scene_out,
        "--frame-out", frame_out,
        "--depth-out", depth_out,
        "--width", str(width), "--height", str(height),
        "--mode", config["mode"],
    ]
    if config["map"]:
        backend_args.extend(("--map", config["map"]))
    if config["map_asset"]:
        backend_args.extend(("--map-asset", config["map_asset"]))
    for vpk_path in config["vpk_paths"]:
        backend_args.extend(("--vpk", vpk_path))
    if config["allow_synthetic_geometry"]:
        backend_args.append("--allow-synthetic-geometry")
    if config["reference_fov_degrees"] is not None:
        backend_args.extend((
            "--reference-fov-degrees", str(config["reference_fov_degrees"]),
        ))
    proc = _invoke_renderer(selected, backend_args)
    if proc.returncode:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RenderRefusal(
            f"legacy CS:GO renderer exited with {proc.returncode}:\n{tail}"
        )
    if not os.path.isfile(scene_out) or os.path.getsize(scene_out) == 0:
        raise RenderRefusal(
            "legacy CS:GO renderer returned success without non-empty scene.json"
        )
    if len(documents) == 1:
        frame_paths = [frame_out]
        depth_paths = [depth_out]
    else:
        frame_paths = sorted(
            os.path.join(frame_out, name) for name in os.listdir(frame_out)
            if name.endswith(".ppm")
        ) if os.path.isdir(frame_out) else []
        depth_paths = sorted(
            os.path.join(depth_out, name) for name in os.listdir(depth_out)
            if name.endswith(".depth.u64le")
        ) if os.path.isdir(depth_out) else []
    invalid_frames = [path for path in frame_paths if not _nonempty_ppm(path)]
    if (len(frame_paths) != len(documents) or invalid_frames
            or len(depth_paths) != len(documents)):
        raise RenderRefusal(
            "legacy CS:GO renderer returned success without the exact set "
            "of P6 reference frames and uint64 depth artifacts"
        )
    try:
        with open(scene_out, encoding="utf-8") as fh:
            scene_manifest = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RenderRefusal(f"legacy scene manifest is invalid: {exc}") from exc
    scene_frames = scene_manifest.get("frames", []) if isinstance(scene_manifest, dict) else []
    if (not isinstance(scene_manifest, dict)
            or scene_manifest.get("schema") != "tardigrade/source1-reference-render/v1"
            or len(scene_frames) != len(documents)
            or any(not item.get("state_hash") or not item.get("color_sha256")
                   or not item.get("depth_sha256") for item in scene_frames
                   if isinstance(item, dict))
            or any(not isinstance(item, dict) for item in scene_frames)):
        raise RenderRefusal(
            "legacy scene manifest omitted state, color, or depth hash provenance"
        )
    for document, frame_record, frame_path, depth_path in zip(
            documents, scene_frames, frame_paths, depth_paths, strict=True):
        with open(frame_path, "rb") as fh:
            color_hash = hashlib.sha256(fh.read()).hexdigest()
        try:
            with open(depth_path, "rb") as fh:
                depth_bytes = fh.read()
        except OSError as exc:
            raise RenderRefusal(
                f"legacy depth artifact is unreadable: {depth_path}: {exc}"
            ) from exc
        depth_hash = hashlib.sha256(depth_bytes).hexdigest()
        reported_output = frame_record.get("output")
        reported_depth_output = frame_record.get("depth_output")
        output_matches = (
            isinstance(reported_output, str)
            and os.path.realpath(os.path.abspath(reported_output))
            == os.path.realpath(os.path.abspath(frame_path))
        )
        depth_output_matches = (
            isinstance(reported_depth_output, str)
            and os.path.realpath(os.path.abspath(reported_depth_output))
            == os.path.realpath(os.path.abspath(depth_path))
        )
        if (not output_matches
                or not depth_output_matches
                or frame_record["state_hash"] != document.get("state_hash")
                or frame_record["color_sha256"] != color_hash
                or frame_record["depth_sha256"] != depth_hash
                or frame_record.get("depth_bytes") != width * height * 8
                or len(depth_bytes) != width * height * 8
                or frame_record.get("width") != width
                or frame_record.get("height") != height
                or frame_record.get("depth_encoding", {}).get("scalar")
                != "uint64-le"):
            raise RenderRefusal(
                "legacy reference output hashes do not bind the requested "
                "state snapshot to the emitted pixels and depth"
            )
    result = {
        "game_variant": selected.id,
        "renderer_backend": selected.renderer.descriptor(),
        "snapshot": snapshot_path,
        "profile": config["path"],
        "resolution": [width, height],
        "mode": config["mode"],
        "fidelity": "deterministic-reference",
        "allow_synthetic_geometry": config["allow_synthetic_geometry"],
        "reference_fov_degrees": config["reference_fov_degrees"],
        "map_asset": config["map_asset"],
        "vpk_paths": config["vpk_paths"],
        "outputs": [os.path.basename(scene_out)] + [
            os.path.relpath(path, out_dir) for path in frame_paths
        ] + [os.path.relpath(path, out_dir) for path in depth_paths],
        "output_dir": os.path.abspath(out_dir),
    }
    manifest_path = os.path.join(out_dir, "snapshot_render.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logstream")
    ap.add_argument("--snapshot")
    ap.add_argument("--profile")
    ap.add_argument("--game-variant", choices=("cs2", "csgo_legacy"))
    ap.add_argument("--world", help=argparse.SUPPRESS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="ego,gow,sm64")
    ap.add_argument("--players", default="all")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--tick-end", type=int, default=None)
    a = ap.parse_args()
    if a.world:
        ap.error("--world is no longer accepted by the render service; put "
                 "--world and every other required asset flag in a canonical "
                 "profile JSON and pass --profile PROFILE.json")
    if not a.profile:
        ap.error("--profile PROFILE.json is required")
    selected = resolve_profile_request_variant(
        profile_game_variant(a.profile), a.game_variant
    )
    if selected.id == "cs2":
        if not a.logstream or a.snapshot:
            ap.error("cs2 requires --logstream and does not accept --snapshot")
        pl = "all" if a.players == "all" else [int(x) for x in a.players.split(",")]
        render_match(a.logstream, a.profile, a.out, tuple(a.views.split(",")), pl,
                     0, a.tick_end, a.stride, a.width, a.height, selected.id)
    else:
        if not a.snapshot or a.logstream:
            ap.error("csgo_legacy requires --snapshot and does not accept --logstream")
        render_snapshot(a.snapshot, a.profile, a.out, a.width, a.height, selected.id)
