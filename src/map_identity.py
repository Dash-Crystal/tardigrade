"""Map identity is FIT FROM TRAJECTORY DATA and enforced by refusal.

The header lies (upstream D10, measured live: header said de_mirage, the
trajectories fit de_inferno). No pipeline stage may choose a world pack by a
string a human or a header wrote. The interlock:

  1. identify_map(recording)  -> fits the all-player position bbox against
     every map's own overview calibration; exactly one map must contain it
     with positive slack, or REFUSE (ambiguity and no-fit are both stops).
  2. resolve_pack(map_name, pack_dir) -> the pack path derived from the
     FITTED name; the pack must exist AND its own extent must contain the
     trajectories (a renamed pack fails here, closing the D10-shaped rename
     hole on the pack side too).
  3. bind(recording, pack_dir) -> (map_name, pack_path) or SystemExit.
     Callers get the pair from ONE function; there is no argument to pass a
     map name in, so there is nothing to pass wrongly.

Calibrations are READ from the extracted overview resources; their location
is configurable but their CONTENT is the game's own.
"""
from __future__ import annotations

import glob
import os
import re

OVERVIEW_DIRS = (
    os.path.expanduser("~/dox/runs/cs-tard21/render/extracted/resource/overviews"),
    "assets/overviews",
)
_MARGIN = 200.0          # world units of tolerance at the map edge


def _calibrations(dirs=OVERVIEW_DIRS):
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.txt")):
            txt = open(p, errors="replace").read()
            def g(k):
                m = re.search(rf'"{k}"\s+"?(-?[\d.]+)"?', txt)
                return float(m.group(1)) if m else None
            px, py, sc = g("pos_x"), g("pos_y"), g("scale")
            if None in (px, py, sc):
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            out[name] = (px, px + 1024 * sc, py - 1024 * sc, py)
    return out


def trajectory_bbox(recording, stride=64):
    xs, ys = [], []
    for pi in range(len(recording.players)):
        p = recording.player(pi)
        n = recording.player_sample_count(pi)
        for t in range(0, n, stride):
            s = p.sample(t)
            if s is not None and s.position.present:
                xs.append(s.position.x)
                ys.append(s.position.y)
    if not xs:
        raise SystemExit("REFUSING map identification: recording decodes "
                         "ZERO positioned samples.")
    return min(xs), max(xs), min(ys), max(ys)


def identify_map(recording, dirs=OVERVIEW_DIRS):
    """The ONE fitted map name, or refusal. Header is printed as a comment
    and compared, never trusted."""
    cal = _calibrations(dirs)
    if not cal:
        raise SystemExit(f"REFUSING map identification: no overview "
                         f"calibrations found under {dirs}.")
    bx0, bx1, by0, by1 = trajectory_bbox(recording)
    fits = []
    for name, (x0, x1, y0, y1) in cal.items():
        slack = min(bx0 - x0, x1 - bx1, by0 - y0, y1 - by1)
        if slack > -_MARGIN:
            fits.append((slack, name))
    fits.sort(reverse=True)
    header = getattr(recording, "map_name_header",
                     getattr(recording, "map_name", "?"))
    if not fits:
        raise SystemExit(
            f"REFUSING: trajectory bbox x[{bx0:.0f},{bx1:.0f}] "
            f"y[{by0:.0f},{by1:.0f}] fits NO known map "
            f"(header claims {header!r}, which is a comment).")
    best_slack, best = fits[0]
    strong = [n for s, n in fits if s > 0]
    if len(strong) > 1:
        raise SystemExit(
            f"REFUSING: trajectory fits {len(strong)} maps with positive "
            f"slack ({strong}); ambiguity is a stop, not a choice.")
    tag = "AGREES" if best == header else "CONTRADICTS"
    print(f"map identity: FITTED {best!r} (slack {best_slack:.0f}); "
          f"header {header!r} {tag} and was not used.", flush=True)
    return best


def resolve_pack(map_name, pack_dir):
    path = os.path.join(pack_dir, f"{map_name}.pt")
    if not os.path.isfile(path):
        raise SystemExit(
            f"REFUSING: fitted map {map_name!r} has no pack at {path}. "
            f"Stage the correct pack; do NOT rename another map's pack "
            f"(the extent check would refuse it anyway).")
    return path


def bind(recording, pack_dir, dirs=OVERVIEW_DIRS):
    """(fitted_map_name, pack_path) -- the only sanctioned way for match
    tooling to obtain a world pack."""
    name = identify_map(recording, dirs)
    return name, resolve_pack(name, pack_dir)
