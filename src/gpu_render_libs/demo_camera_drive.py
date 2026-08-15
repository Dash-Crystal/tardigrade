"""Drive the retrocausal demo cameras from a players-all match dump.

Builds a Timeline per focus player (focus_rows + enemies_by_tick from the other
team) out of players-all.json, runs demo_camera.compute_camera_path for the
requested variant(s), and writes renderer-ready camera-path JSONs. This is the
integration glue between the match trajectory data and the validated renderer;
demo_match_export.export_match_cameras is the same shape over a tardigrade
recording.
"""
import json
import os
import sys

import demo_camera as dc

EYE_STAND, EYE_DUCK = 64.0, 46.0


def _focus_rows(player_rows):
    out = []
    for r in player_rows:
        z = float(r["z"])
        duck = float(r.get("duck", 0.0) or 0.0)
        eye = EYE_STAND - (EYE_STAND - EYE_DUCK) * duck
        out.append({
            "tick": int(r["tick"]),
            "x": float(r["x"]), "y": float(r["y"]), "z": z,
            "eye_z": z + eye,
            "yaw_degrees": float(r["yaw"]),
            "pitch_degrees": float(r.get("pitch", 0.0) or 0.0),  # no pitch in dump
            "fire": bool(r.get("fire")),
            "active_weapon_name": r.get("weapon"),
            "is_alive": bool(r.get("alive", True)),
        })
    return out


def _enemies_by_tick(players, focus_index, focus_team):
    by_tick = {}
    for p in players:
        if p["index"] == focus_index or p.get("team") == focus_team:
            continue
        for r in p["rows"]:
            if not r.get("alive", True):
                continue
            by_tick.setdefault(int(r["tick"]), []).append({
                "index": p["index"],
                "x": float(r["x"]), "y": float(r["y"]),
                "z": float(r["z"]) + EYE_STAND * 0.5,
            })
    return by_tick


def build_timeline(match, focus_index, tick_rate):
    players = match["players"]
    focus = next(p for p in players if p["index"] == focus_index)
    tl = dc.Timeline(
        tick_rate=tick_rate,
        focus_rows=_focus_rows(focus["rows"]),
        enemies_by_tick=_enemies_by_tick(players, focus_index, focus.get("team")),
        map_name=match.get("map_name", "de_inferno"),
        demo=match.get("demo", "players-all"),
        steamid=focus_index,
    )
    return tl


def emit(match, focus_index, variant_name, tick_rate, out_path):
    tl = build_timeline(match, focus_index, tick_rate)
    path = dc.compute_camera_path(tl, dc.VARIANTS[variant_name])
    # Renderer's _load_agent_rows reads ticks[] + tick_rate; keep both.
    path.setdefault("tick_rate", tick_rate)
    with open(out_path, "w") as fh:
        json.dump(path, fh)
    return path


def main(argv):
    match_path, out_dir = argv[1], argv[2]
    variants = argv[3].split(",") if len(argv) > 3 else ["gow", "sm64"]
    players = argv[4] if len(argv) > 4 else "0"
    match = json.load(open(match_path))
    # Native tick rate: the rows carry NATIVE tick values (0,4,8,... at
    # tick_step 4), so dt = step/native_rate. Using the decimated rate here
    # made dt 4x too large and diverged the snap-omega spring (omega*dt >> 2).
    tick_rate = int(match.get("native_tick_rate", 128))
    os.makedirs(out_dir, exist_ok=True)
    idxs = ([p["index"] for p in match["players"]] if players == "all"
            else [int(x) for x in players.split(",")])
    manifest = []
    for pi in idxs:
        for v in variants:
            op = os.path.join(out_dir, f"p{pi}_{v}.camera.json")
            path = emit(match, pi, v, tick_rate, op)
            manifest.append({
                "player": pi, "variant": v, "path": op,
                "n_ticks": path["n_ticks"],
                "n_future_steered": path["n_future_steered"],
                "max_foreshadow_blend": round(path["max_foreshadow_blend"], 3),
            })
            print(f"  p{pi} {v}: {path['n_ticks']} ticks, "
                  f"{path['n_future_steered']} future-steered, "
                  f"max_blend {path['max_foreshadow_blend']:.3f} -> {op}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest)} camera paths to {out_dir}")


if __name__ == "__main__":
    main(sys.argv)
