"""Render a demofile (cs-tard21/tard3 logstream) through three view matrices.

Views:
  ego   -- naive egocentric: the player's own eye path, verbatim (what a plain
           demo replay shows). No third-person transform.
  gow   -- retrocausal Gears-of-War third-person (demo_camera GOW).
  sm64  -- retrocausal Mario-64/Lakitu spring-orbit (demo_camera SM64).

One logstream -> per (player, view) a renderer-ready camera path -> the ported
renderer in SESSION mode (one pack load per view) -> per-player frame tensors.
demo_match_export.build_timeline_from_tardigrade does the wire READ; ego reuses
its focus_rows directly. This module is the engine; cs_render_server wraps it in
a boring REST endpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import demo_camera as dc
from demo_match_export import build_timeline_from_tardigrade
from tardigrade_v21 import TardigradeV21Recording

_HERE = os.path.dirname(os.path.abspath(__file__))


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


# ---------------------------------------------------------------------------
# Canonical render profile -- OWNER RULING (2026-08-11): there is no
# configuration that renders without the task's assets. Every render this
# service performs uses the full-stack profile below; a request chooses WHAT
# to render (logstream, players, views, ticks, resolution) and never how
# little. Missing profile assets are a REFUSAL naming the gap, not a fallback.
# ---------------------------------------------------------------------------

def load_profile(profile_path):
    """Load the deployment's render profile: {"asset_root": dir, "flags":
    [[flag, value|null], ...]}. Relative values resolve under asset_root.
    Every flag value that looks like a path MUST exist -- else refuse."""
    doc = json.loads(open(profile_path).read())
    root = doc.get("asset_root", os.path.dirname(os.path.abspath(profile_path)))
    flags, missing = [], []
    for entry in doc["flags"]:
        flag, value = entry[0], (entry[1] if len(entry) > 1 else None)
        if value is None:
            flags.append(str(flag))
            continue
        sval = str(value)
        resolved = sval if os.path.isabs(sval) else os.path.join(root, sval)
        if ("/" in sval or sval.startswith(".")) or os.path.exists(resolved):
            # path-like: must exist
            if not os.path.exists(resolved):
                missing.append(f"{flag} -> {resolved}")
                continue
            flags.extend((str(flag), resolved))
        else:
            flags.extend((str(flag), sval))   # plain value (numbers, modes)
    if missing:
        raise SystemExit(
            "REFUSING to serve renders: the canonical profile names assets "
            "that are not present on this node:\n  " + "\n  ".join(missing) +
            "\nThere is no reduced-asset render mode. Stage the assets or "
            "fix the profile.")
    return flags


def _ego_path(tl):
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
    return {"schema": "iji/cs2-demo-ego-camera-path/v1", "variant": "ego",
            "tick_rate": tl.tick_rate, "n_ticks": len(ticks),
            "n_future_steered": 0, "max_foreshadow_blend": 0.0, "ticks": ticks}


def build_view_path(recording, player_index, view, tick_begin, tick_end,
                    stride, native_rate=128):
    tl = build_timeline_from_tardigrade(recording, player_index,
                                        tick_begin, tick_end, stride)
    tl.tick_rate = native_rate
    if view == "ego":
        return _ego_path(tl)
    return dc.compute_camera_path(tl, dc.VARIANTS[view])


def render_view(recording, recording_path, view, players, tick_begin, tick_end,
                stride, width, height, profile_flags, out_dir):
    """Build every player's camera path for one view and render them in ONE
    session (one pack load) -> per-player frame tensors."""
    view_dir = os.path.join(out_dir, view)
    os.makedirs(view_dir, exist_ok=True)
    agents, manifest = [], []
    for pi in players:
        path = build_view_path(recording, pi, view, tick_begin, tick_end, stride)
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
        json.dump({"agents": agents}, fh)
    cj0 = agents[0]["camera_json"]
    # Canonical profile FIRST (world, irradiance, lightmap, probes, fam_side,
    # post chain, ...); request-scoped flags after. Nothing here may reduce
    # the asset surface -- the unlit escape is deleted at both layers.
    argv = [sys.executable, _CORE, *profile_flags,
            "--session", session, "--session-out", view_dir,
            "--camera-json", cj0, "--tick-begin", "0", "--tick-end", "1",
            "--out", os.path.join(view_dir, "u.mp4"),
            "--width", str(width), "--height", str(height), "--fps", "32"]
    repo_root = os.path.dirname(os.path.dirname(_HERE))
    proc = subprocess.run(argv, cwd=repo_root, capture_output=True, text=True)
    tensors = sorted(f for f in os.listdir(view_dir) if f.endswith(".pt"))
    return {"view": view, "rc": proc.returncode, "players": manifest,
            "tensors": tensors, "tensor_dir": view_dir,
            "log_tail": proc.stdout[-400:] if proc.returncode else ""}


def render_match(recording_path, world_pack, out_dir, views=VIEWS,
                 players="all", tick_begin=0, tick_end=None, stride=16,
                 width=640, height=360):
    data = open(recording_path, "rb").read()
    rec = TardigradeV21Recording(data)
    if players == "all":
        players = list(range(len(rec.players)))
    os.makedirs(out_dir, exist_ok=True)
    result = {"match_id": rec.match_id, "recording": recording_path,
              "n_players": len(rec.players), "stride": stride,
              "resolution": [width, height], "views": {}}
    for view in views:
        r = render_view(rec, recording_path, view, players, tick_begin,
                        tick_end, stride, width, height, world_pack, out_dir)
        result["views"][view] = r
        print(f"[{view}] rc={r.get('rc')} "
              f"players={len(r.get('players', []))} "
              f"tensors={len(r.get('tensors', []))}", flush=True)
    with open(os.path.join(out_dir, "match_render.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logstream", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="ego,gow,sm64")
    ap.add_argument("--players", default="all")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--tick-end", type=int, default=None)
    a = ap.parse_args()
    pl = "all" if a.players == "all" else [int(x) for x in a.players.split(",")]
    render_match(a.logstream, a.world, a.out, tuple(a.views.split(",")), pl,
                 0, a.tick_end, a.stride, a.width, a.height)
