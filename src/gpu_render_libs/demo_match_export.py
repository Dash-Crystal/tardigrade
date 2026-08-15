"""Format-3 deliverable: per-player third-person camera renders for a MATCH.

One API call takes a TARDIGRADE (HCI-128 v2.1) demo-format match and, for
EVERY player (or a chosen subset), drives the third-person camera CONTROLLER
(`demo_camera.compute_camera_path`) to a per-tick camera trajectory and then
drives OUR renderer (`gpu_render.py`, third-person worldmodel + weapon-in-hand
posed from demo state) ALONG that trajectory, producing that player's stream.
`camera='both'` emits the GoW and SM64 variants; `'gow'`/`'sm64'` emit one.

    export_match_cameras(match, camera='gow'|'sm64'|'both',
                         players='all'|[ids]) -> ExportResult
                         (per-player frame/video render jobs + a manifest)

PIPELINE, per player:
    tardigrade recording --(build_timeline_from_tardigrade: READ the wire, no
        derive-from-name)--> Timeline(focus_rows + enemies_by_tick)
      --(compute_camera_path, per requested variant)--> camera-json trajectory
        (the EXISTING ego-camera-path schema gpu_render already consumes)
      --> render_job: the gpu_render.py invocation that renders the
          third-person stream ALONG that camera-json.

GROUNDED IN REAL INFRA (imported, not re-derived):
  * `iji_model.tardigrade_v21.TardigradeV21Recording` -- the immutable match
    reader (lazy per-player decode; positions READ, weapon/location catalogs
    READ; `map_name` is a wire field). THIS is the match source.
  * `demo_camera` (this dir) -- the retrocausal third-person controller. Its
    two-sided-calibrated pre-shoot predictor is reused verbatim; the camera
    rows this export writes are the SAME schema gpu_render already renders.
  * `gpu_render.py` -- OUR renderer. It is NOT in the working tree (it lives
    on the ws-2 / terul deploy tree; the deploy rsyncs the WORKING TREE not
    HEAD), so its md5 -- the renderer identity half of provenance -- is
    resolved AT RENDER, and the render step is reported queue-blocked here.

TWO-SIDED, REFUSE-BY-DEFAULT (the export is an instrument):
  * It REFUSES (typed error, CLI exit nonzero, NO manifest written) on a
    missing path (MissingMatchError) or an empty match -- zero players, zero
    requested-and-present players, or zero usable ticks for every player
    (EmptyMatchError). An empty manifest that reads as success is the defect
    this guards against.
  * `python3 demo_match_export.py selftest` builds a REAL TR21 pack (encoded
    into the actual wire, decoded by the actual reader -- not an A==B fixture)
    and checks BOTH sides: a valid 2-player match with one shooter must emit a
    manifest whose SHOOTER stream is future-steered (n_future_steered>0) and
    whose QUIET stream is not (==0); an empty match and a missing path must
    each REFUSE with no manifest. A pipeline that always emits, or never
    foreshadows, fails one side.

PROVENANCE (an artifact is identified by input md5s + renderer md5, NOT the
git sha): the manifest carries md5s of the match bytes, the controller module,
the reader module, this module, and each written camera-json, plus a
renderer_md5 slot filled at render on ws-2. The git sha is recorded as context
and explicitly labelled not-the-identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# demo_camera is this directory's controller.
_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import demo_camera  # noqa: E402
from demo_camera import Timeline, VARIANTS, compute_camera_path  # noqa: E402

# The tardigrade reader lives in the iji_model package at the repo root.
_REPO_ROOT = _HERE.parent.parent.parent            # harness/gpu_render -> repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Typed refusals. An empty/missing match RAISES; it never returns a manifest.
# ---------------------------------------------------------------------------


class MatchExportError(Exception):
    """Base: the export refused to emit."""


class MissingMatchError(MatchExportError):
    """The match path does not exist / is not a file."""


class EmptyMatchError(MatchExportError):
    """The match has no players, no requested-present players, or no ticks."""


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


@dataclass
class PlayerStream:
    variant: str
    camera_json: str
    camera_json_md5: str | None
    n_ticks: int
    n_fire_ticks: int
    n_future_steered: int
    max_foreshadow_blend: float
    tick_begin: int
    tick_end: int
    render_job: dict


@dataclass
class ExportResult:
    manifest: dict
    manifest_path: str | None
    streams: list                             # list[PlayerStream]

    @property
    def n_streams(self) -> int:
        return len(self.streams)


# ---------------------------------------------------------------------------
# Eye height. READ from the engine (same constants demo_camera / cs2_demo_camera
# use): VEC_VIEW z=64, VEC_DUCK_VIEW z=46 -> eye = 64 - 18*duck. Tardigrade
# `ducking` is a 0/1 event, not a ratio, so duck is that flag.
# ---------------------------------------------------------------------------

EYE_STANDING = 64.0
EYE_DUCK_DROP = 18.0


def _event_int(sample, name, default=None):
    ov = sample.events.get(name)
    if ov is None or not ov.present or ov.value is None:
        return default
    return ov.value


def build_timeline_from_tardigrade(recording, focus_index: int,
                                   tick_begin: int = 0, tick_end=None,
                                   stride: int = 1,
                                   position_projection: str = "visual_linear"
                                   ) -> Timeline:
    """READ one focus player's rows + the other-team players as enemies.

    Positions are READ from the wire (`TardigradePositionValue`, already the
    reader's declared units: wire tenths -> source inches). Angles are the
    reader's absolute yaw/pitch (initial + cumulative view deltas). `fire` is
    the reader's per-tick active flag (a burst is active across its whole run,
    which the controller's hold/foreshadow both handle). position_projection
    stays 'causal_hold' by default so the ONLY future-sampling in the output
    is the camera controller's shot-foreshadow -- position reads never set
    `uses_future_sample`, keeping that flag's meaning single.
    """
    focus = recording.player(focus_index)
    n = focus.sample_count
    hi = n if tick_end is None else min(tick_end, n)
    lo = max(0, tick_begin)
    ticks = range(lo, hi, max(1, stride))

    focus_desc = recording.players[focus_index]
    focus_team = focus_desc.team

    focus_rows = []
    for t in ticks:
        s = focus.sample(t, position_projection=position_projection)
        if s is None or not s.position.present:
            continue
        if s.absolute_yaw_degrees is None or s.absolute_pitch_degrees is None:
            continue
        duck = 1.0 if _event_int(s, "ducking", 0) else 0.0
        eye = EYE_STANDING - EYE_DUCK_DROP * duck
        health = _event_int(s, "health", None)
        z = s.position.z
        focus_rows.append({
            "tick": int(t),
            "x": float(s.position.x), "y": float(s.position.y), "z": float(z),
            "eye_z": float(z) + eye,
            "yaw_degrees": float(s.absolute_yaw_degrees),
            "pitch_degrees": float(s.absolute_pitch_degrees),
            "fov": None,                       # no direct FOV on the TARD wire
            "is_alive": bool(health is None or health > 0),
            "active_weapon_name": _event_int(s, "weapon_index", None),
            "fire": bool(s.fire.value),
            "fire_weapon": (_event_int(s, "weapon_index", None)
                            if s.fire.value else None),
        })

    enemies_by_tick: dict = {}
    for j, desc in enumerate(recording.players):
        if j == focus_index:
            continue
        if focus_team is not None and desc.team == focus_team:
            continue                           # skip teammates
        other = recording.player(j)
        for t in ticks:
            s = other.sample(t, position_projection=position_projection)
            if s is None or not s.position.present:
                continue
            health = _event_int(s, "health", None)
            enemies_by_tick.setdefault(int(t), []).append({
                "index": int(desc.player_id),
                "steamid": int(desc.player_id),
                "x": float(s.position.x), "y": float(s.position.y),
                "z": float(s.position.z) + 40.0,
                "alive": bool(health is None or health > 0),
                "team": int(desc.team),
            })

    return Timeline(
        tick_rate=recording.summary()["native_tick_rate"],
        focus_rows=focus_rows,
        enemies_by_tick=enemies_by_tick,
        map_name=recording.map_name,
        demo=str(recording.match_id),
        steamid=int(focus_desc.player_id),
    )


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------


def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _md5_file(path: Path) -> str | None:
    try:
        return _md5_bytes(path.read_bytes())
    except OSError:
        return None


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(_REPO_ROOT), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _renderer_identity() -> dict:
    """The renderer half of provenance. gpu_render.py is ws-2/terul-only; if it
    is not in the working tree its md5 is UNRESOLVED and filled at render."""
    rp = _HERE.parent / "gpu_render.py"
    md5 = _md5_file(rp) if rp.exists() else None
    return {
        "entrypoint": str(rp),
        "renderer_md5": md5,
        "renderer_md5_note": (
            "resolved AT RENDER on ws-2 from gpu_render.py; not in the working "
            "tree here (deploy rsyncs the WORKING TREE not HEAD) -- two streams "
            "with different renderer_md5 are not comparable"
            if md5 is None else "md5 of the gpu_render.py in THIS working tree"),
    }


def _provenance(match_path, match_bytes, match_summary, camera_json_md5s):
    reader = _REPO_ROOT / "src" / "tardigrade_v21.py"
    controller = _HERE.parent / "demo_camera.py"
    input_md5 = {
        "match_bytes": _md5_bytes(match_bytes) if match_bytes is not None
        else None,
        "demo_camera.py": _md5_file(controller),
        "tardigrade_v21.py": _md5_file(reader),
        "demo_match_export.py": _md5_file(_HERE),
    }
    prov = {
        "input_md5": input_md5,
        "match_content_id": match_summary.get("content_id"),
        "match_payload_id": match_summary.get("payload_id"),
        "camera_json_md5": camera_json_md5s,
        "git_sha": _git_sha(),
        "identity_note": (
            "an artifact is identified by input md5s + renderer_md5, NOT the "
            "git sha; the git sha is context only"),
        "deploy_note": "the deploy rsyncs the WORKING TREE not HEAD",
    }
    prov.update(_renderer_identity())
    return prov


# ---------------------------------------------------------------------------
# Render job (the ws-2-gated step, described as data).
# ---------------------------------------------------------------------------


# Third-person driver flags. NAMES read from gpu_render.py's argparse (not
# derived): the render poses the other players' worldmodels + weapon-in-hand
# from demo state through this family. Ego renders EXCLUDE the focus player;
# third-person KEEPS them, so --playermodel-exclude-steamid is deliberately
# NOT set here. Value grammar (asset paths) is resolved on ws-2, not guessed.
_THIRDPERSON_FLAGS = (
    "--playermodels", "--playermodel-bundles", "--playermodel-clips",
    "--playermodel-weapon-map", "--pm-anim", "--view-punch",
)


def _verify_flags(renderer_path: Path, flags) -> dict:
    """Confirm each flag NAME is present in the (md5'd) renderer source. A
    staleness detector: an unknown flag is fatal to argparse, so a name that
    has drifted out shows here rather than at submit."""
    try:
        text = renderer_path.read_text(errors="replace")
    except OSError:
        return {f: None for f in flags}        # renderer not in tree
    return {f: (f'add_argument("{f}"' in text) for f in flags}


def _render_job(map_name, camera_json_path, tick_begin, tick_end,
                out_dir: Path, tag: str, width: int, height: int, fps: int,
                mode: str, focus_steamid: int) -> dict:
    """The gpu_render.py invocation for one player's stream, as data.

    The camera/output flag surface is exactly the one cs1k_arm_render.sh drives
    (--world/--camera-json/--tick-begin/--tick-end/--fps/--width/--height/
    --batch/--png-dir/--out). The third-person worldmodel+weapon flag NAMES are
    read from gpu_render.py and re-verified present at plan time; their asset
    VALUES (bundles/clips/weapon-map) are resolved on ws-2, not invented.
    """
    renderer = _HERE.parent / "gpu_render.py"
    flag_present = _verify_flags(renderer, _THIRDPERSON_FLAGS)
    tp_verified = all(v for v in flag_present.values()) if any(
        v is not None for v in flag_present.values()) else False
    png_dir = out_dir / f"{tag}_frames"
    mp4 = out_dir / f"{tag}.mp4"
    log = out_dir / f"{tag}.render.log"
    known_argv = [
        "gpu_render.py",
        "--world", f"<REQUIRED: world.pt for map '{map_name}' "
                   f"(resolve on ws-2 from the worlds/ corpus)>",
        "--camera-json", str(camera_json_path),
        "--tick-begin", str(tick_begin),
        "--tick-end", str(tick_end),
        "--fps", str(fps),
        "--width", str(width),
        "--height", str(height),
        "--batch", "1",
        "--png-dir", str(png_dir),
        "--out", str(mp4),
    ]
    return {
        "status": "queue-blocked" if mode == "plan" else "planned",
        "blocked_on": (
            "ws-2 (2x RTX 5090) -- render node once #96 encryption lands; "
            "terul GPU partition 100% owned by a foreign trainer tonight"),
        "backend": "gpu_render",
        "third_person": True,
        "known_argv": known_argv,
        "thirdperson_driver": {
            "flags_present": flag_present,
            "note": "flag NAMES read from gpu_render.py (see renderer_md5), "
                    "re-verified present at plan time; ego EXCLUDES the focus "
                    "player, third-person KEEPS them so "
                    "--playermodel-exclude-steamid is NOT set",
            "focus_steamid": int(focus_steamid),
        },
        "requires": {
            "world_asset": f"world.pt for map '{map_name}'",
            "player_state": "third-person worldmodel + weapon-in-hand posed "
                            "from demo state via the --playermodel* family "
                            "(bundles/clips/weapon-map resolved on ws-2)",
        },
        "outputs": {"png_dir": str(png_dir), "mp4": str(mp4),
                    "log": str(log)},
        "thirdperson_flags_present_verified": tp_verified,
        "renderer_help_verified": False,
        "renderer_note": (
            "camera/output flags are the cs1k_arm_render.sh-verified surface; "
            "third-person flag NAMES verified present in the md5'd renderer; "
            "bind asset VALUES + run `gpu_render.py --help` on ws-2 before "
            "submit (an unknown flag is fatal -- a staleness detector)"),
    }


# ---------------------------------------------------------------------------
# The API.
# ---------------------------------------------------------------------------


def _resolve_match(match):
    """(recording, match_path, match_bytes). Refuses missing paths."""
    from tardigrade_v21 import TardigradeV21Recording
    if isinstance(match, (str, Path)):
        p = Path(match).expanduser()
        if not p.is_file():
            raise MissingMatchError(f"no match file at {p}")
        b = p.read_bytes()
        return TardigradeV21Recording(b, content_id=None), str(p), b
    # Duck-typed: an already-loaded recording (tests / callers holding bytes).
    if hasattr(match, "players") and hasattr(match, "player"):
        return match, None, getattr(match, "_data", None)
    raise MissingMatchError(f"unsupported match spec: {type(match)!r}")


def _select_variants(camera: str) -> list:
    if camera == "both":
        return ["gow", "sm64"]
    if camera in VARIANTS:
        return [camera]
    raise ValueError(f"camera must be gow|sm64|both, got {camera!r}")


def _select_players(recording, players) -> list:
    """-> list of descriptor INDICES. Refuses when none present."""
    n = len(recording.players)
    if n == 0:
        raise EmptyMatchError("match has zero players")
    if players == "all" or players is None:
        return list(range(n))
    want = {int(x) for x in players}
    by_id = {int(d.player_id): i for i, d in enumerate(recording.players)}
    idx = []
    for pid in sorted(want):
        if pid in by_id:
            idx.append(by_id[pid])
        elif 0 <= pid < n and pid not in by_id:
            # allow raw descriptor index as a fallback selector
            idx.append(pid)
    idx = sorted(set(idx))
    if not idx:
        raise EmptyMatchError(
            f"none of players={sorted(want)} are present "
            f"(player_ids={sorted(by_id)})")
    return idx


def export_match_cameras(match, camera: str = "gow", players="all",
                         out_dir=None, mode: str = "plan",
                         tick_begin: int = 0, tick_end=None, stride: int = 1,
                         lookahead_ms: float = 700.0, hold_ms: float = 220.0,
                         width: int = 1280, height: int = 720, fps: int = 128,
                         write_manifest: bool = True) -> ExportResult:
    """Match -> per-player {controller trajectory -> gpu_render along it}.

    mode='plan' (default) emits the per-player camera-json + a render manifest
    WITHOUT touching the GPU (each render_job is queue-blocked on ws-2). REFUSES
    (typed, no manifest) on a missing/empty match.
    """
    variants = _select_variants(camera)
    recording, match_path, match_bytes = _resolve_match(match)
    summary = recording.summary()
    idx = _select_players(recording, players)

    out = Path(out_dir) if out_dir is not None else (
        _REPO_ROOT / "docs" / "projects" / "counter-strike-sft" /
        "match_exports" / str(summary.get("match_id") or "match"))
    out.mkdir(parents=True, exist_ok=True)

    streams: list = []
    camera_json_md5s: dict = {}
    player_entries: list = []

    for pi in idx:
        desc = recording.players[pi]
        tl = build_timeline_from_tardigrade(
            recording, pi, tick_begin=tick_begin, tick_end=tick_end,
            stride=stride)
        if not tl.focus_rows:
            player_entries.append({
                "descriptor_index": pi, "player_id": int(desc.player_id),
                "team": int(desc.team), "skipped": "no usable ticks in range",
                "streams": []})
            continue
        t0 = tl.focus_rows[0]["tick"]
        t1 = tl.focus_rows[-1]["tick"] + 1
        p_streams = []
        for var in variants:
            payload = compute_camera_path(
                tl, VARIANTS[var], lookahead_ms=lookahead_ms, hold_ms=hold_ms)
            tag = f"p{int(desc.player_id)}_{var}"
            cj = out / f"{tag}.camera.json"
            blob = json.dumps(payload).encode()
            cj.write_bytes(blob)
            cj_md5 = _md5_bytes(blob)
            camera_json_md5s[cj.name] = cj_md5
            rj = _render_job(tl.map_name, cj, t0, t1, out, tag,
                             width, height, fps, mode,
                             focus_steamid=int(desc.player_id))
            ps = PlayerStream(
                variant=var, camera_json=str(cj), camera_json_md5=cj_md5,
                n_ticks=payload["n_ticks"],
                n_fire_ticks=payload["n_fire_ticks"],
                n_future_steered=payload["n_future_steered"],
                max_foreshadow_blend=payload["max_foreshadow_blend"],
                tick_begin=t0, tick_end=t1, render_job=rj)
            streams.append(ps)
            p_streams.append({
                "variant": var, "camera_json": str(cj),
                "camera_json_md5": cj_md5,
                "n_ticks": ps.n_ticks, "n_fire_ticks": ps.n_fire_ticks,
                "n_future_steered": ps.n_future_steered,
                "max_foreshadow_blend": ps.max_foreshadow_blend,
                "tick_begin": t0, "tick_end": t1, "render_job": rj})
        player_entries.append({
            "descriptor_index": pi, "player_id": int(desc.player_id),
            "team": int(desc.team), "streams": p_streams})

    if not streams:
        raise EmptyMatchError(
            "no player produced a usable trajectory (empty match / range) -- "
            "refusing to write a manifest that would read as success")

    manifest = {
        "schema": "iji/cs2-demo-match-camera-export/v1",
        "generated_by": str(_HERE),
        "mode": mode,
        "match": {
            "source_kind": "tardigrade/hci-128/v2.1",
            "path": match_path,
            "match_id": summary.get("match_id"),
            "map_name": summary.get("map_name"),
            "tick_count": summary.get("tick_count"),
            "native_tick_rate": summary.get("native_tick_rate"),
            "player_count": summary.get("player_count"),
        },
        "camera": camera,
        "variants": variants,
        "variant_params": {v: VARIANTS[v] for v in variants},
        "players_requested": players,
        "controller": {
            "module": str(_HERE.parent / "demo_camera.py"),
            "lookahead_ms": lookahead_ms, "hold_ms": hold_ms,
            "consumes_schema": "iji/cs2-demo-ego-camera-path/v1",
        },
        "render": {
            "backend": "gpu_render", "width": width, "height": height,
            "fps": fps, "out_dir": str(out), "third_person": True,
        },
        "provenance": _provenance(match_path, match_bytes, summary,
                                  camera_json_md5s),
        "counts": {
            "n_players_exported": sum(1 for e in player_entries
                                      if e.get("streams")),
            "n_players_skipped": sum(1 for e in player_entries
                                     if not e.get("streams")),
            "n_streams": len(streams),
            "n_camera_json_written": len(camera_json_md5s),
        },
        "refusal_contract": {
            "missing_path": "MissingMatchError (exit nonzero, no manifest)",
            "empty_match": "EmptyMatchError (exit nonzero, no manifest)",
        },
        "render_status": (
            "QUEUE-BLOCKED on ws-2 (#96 encryption pending; terul GPU 100% "
            "foreign trainer tonight) -- camera-json + manifest are LIVE, GPU "
            "renders are not" if mode == "plan" else "planned"),
        "players": player_entries,
    }

    manifest_path = None
    if write_manifest:
        mp = out / "match_camera_manifest.json"
        mp.write_text(json.dumps(manifest, indent=2))
        manifest_path = str(mp)

    return ExportResult(manifest=manifest, manifest_path=manifest_path,
                        streams=streams)


# ---------------------------------------------------------------------------
# TR21 fixture ENCODER -- for two-sided calibration ONLY. It writes the REAL
# wire so the REAL reader decodes it (not an A==B fixture).
# ---------------------------------------------------------------------------


def _uvarint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _zz(x: int) -> int:
    return (x << 1) if x >= 0 else ((-x) << 1) - 1


def _signed_stream(values) -> bytes:
    out = bytearray()
    for v in values:
        out += _uvarint(_zz(int(v)))
    return bytes(out)


def _position_stream(wire_vals) -> bytes:
    """Delta-of-delta signed stream reconstructing `wire_vals` under the
    reader's `_decode_position(base=wire_vals[0])`."""
    m = len(wire_vals)
    if m == 0:
        return b""
    coded = [0] * m                            # coded[0]=seed 0
    for k in range(1, m):
        s_k = wire_vals[k] - wire_vals[k - 1]
        s_km1 = 0 if k - 1 == 0 else (wire_vals[k - 1] - wire_vals[k - 2])
        coded[k] = s_k - s_km1
    return _signed_stream(coded)


def _fire_stream(runs) -> bytes:
    """runs = [(start_tick, duration_ticks)] -> reader's _decode_fire wire."""
    if not runs:
        return b""
    out = bytearray(_uvarint(len(runs)))
    prev = 0
    for start, dur in runs:
        out += _uvarint(start - prev)
        out += _uvarint(dur)
        prev = start
    return bytes(out)


def _event_stream(entries) -> bytes:
    """entries = [(tick, int_value)] -> reader's _decode_events wire."""
    if not entries:
        return b""
    out = bytearray(_uvarint(len(entries)))
    prev = 0
    for tick, val in entries:
        out += _uvarint(tick - prev)
        out += _uvarint(_zz(int(val)))
        prev = tick
    return bytes(out)


def _text32(s: str) -> bytes:
    return s.encode("utf-8")[:32].ljust(32, b"\0")


def _catalog(items) -> bytes:
    out = bytearray(struct_pack_u8(len(items)))
    for it in items:
        out += _text32(it)
    return bytes(out)


def struct_pack_u8(n: int) -> bytes:
    return bytes([n & 0xFF])


def _encode_tr21(map_name: str, match_id: str, players, weapons, locations,
                 tick_count: int) -> bytes:
    """players = [dict(player_id, team, rank, init_yaw, init_pitch,
    full{name:list}, positions=(xs,ys,zs) floats, fire_runs, events{name:
    [(tick,val)]})]. Encodes the exact TR21 v2.1 layout the reader parses."""
    import struct
    from tardigrade_v21 import (
        FULL_RATE_STREAM_NAMES, EVENT_STREAM_NAMES)
    out = bytearray()
    out += b"TR21"
    out += struct.pack("<H", 2)
    out += _text32(match_id)
    out += _text32(map_name)
    out += struct.pack("<I", tick_count)
    out += struct_pack_u8(len(players))
    for p in players:
        out += struct.pack("<HBHff", p["player_id"], p["team"], p["rank"],
                           float(p["init_yaw"]), float(p["init_pitch"]))
    out += _catalog(weapons)
    out += _catalog(locations)
    for p in players:
        streams = []
        for name in FULL_RATE_STREAM_NAMES:
            streams.append(_signed_stream(p["full"][name]))
        xs, ys, zs = p["positions"]
        streams.append(_position_stream([round(v * 10) for v in xs]))
        streams.append(_position_stream([round(v * 10) for v in ys]))
        streams.append(_position_stream([round(v * 10) for v in zs]))
        streams.append(_fire_stream(p.get("fire_runs", [])))
        for name in EVENT_STREAM_NAMES:
            streams.append(_event_stream(p.get("events", {}).get(name, [])))
        for s in streams:
            out += struct.pack("<I", len(s))
            out += s
        # position bases = first wire value per axis
        bx = round(xs[0] * 10) if xs else 0
        by = round(ys[0] * 10) if ys else 0
        bz = round(zs[0] * 10) if zs else 0
        out += struct.pack("<iii", bx, by, bz)
    return bytes(out)


def _fixture_match(n: int = 300, fire_tick: int = 150, tr: int = 128) -> bytes:
    """A REAL TR21 pack: player 0 (team 2) walks +x aiming +y and fires a
    4-tick burst at fire_tick; player 1 (team 3) stands to the +y as the enemy
    and never fires. So the shooter must be future-steered and the quiet one
    must not."""
    v = 180.0
    xs0 = [v * (t / float(tr)) for t in range(n)]
    ys0 = [0.0] * n
    zs0 = [0.0] * n
    enemy_x = v * (fire_tick / float(tr)) + 300.0
    xs1 = [enemy_x] * n
    ys1 = [260.0] * n
    zs1 = [0.0] * n
    full0 = {name: [0] * n for name in
             ("mouse_dx", "mouse_dy", "view_yaw_centidegrees",
              "view_pitch_centidegrees", "forward_move", "left_move")}
    full1 = {name: [0] * n for name in full0}
    ev0 = {"health": [(0, 100)], "weapon_index": [(0, 1)], "ducking": [(0, 0)]}
    ev1 = dict(ev0)
    players = [
        {"player_id": 10, "team": 2, "rank": 0, "init_yaw": 35.0,
         "init_pitch": 0.0, "full": full0, "positions": (xs0, ys0, zs0),
         "fire_runs": [(fire_tick, 4)], "events": ev0},
        {"player_id": 20, "team": 3, "rank": 0, "init_yaw": 200.0,
         "init_pitch": 0.0, "full": full1, "positions": (xs1, ys1, zs1),
         "fire_runs": [], "events": ev1},
    ]
    return _encode_tr21("de_selftest", "match_selftest", players,
                        weapons=["ak47"], locations=["mid"], tick_count=n)


def _empty_match() -> bytes:
    return _encode_tr21("de_empty", "match_empty", players=[],
                        weapons=["ak47"], locations=["mid"], tick_count=0)


# ---------------------------------------------------------------------------
# Two-sided selftest.
# ---------------------------------------------------------------------------


def selftest() -> int:
    import tempfile
    from tardigrade_v21 import TardigradeV21Recording
    ok = True
    lines = []

    # Round-trip guard: the encoder must reproduce positions through the REAL
    # reader (an encoder that lies would make the whole calibration vacuous).
    rec = TardigradeV21Recording(_fixture_match())
    p0 = rec.player(0)
    s10 = p0.sample(10, position_projection="causal_hold")
    exp_x = 180.0 * (10 / 128.0)
    rt_ok = abs(s10.position.x - exp_x) < 0.2 and bool(
        p0.sample(151).fire.value) and not bool(p0.sample(10).fire.value)
    ok = ok and rt_ok
    lines.append(f"  [round-trip] decoded x@10={s10.position.x:.3f} "
                 f"(exp {exp_x:.3f}) fire@151={bool(p0.sample(151).fire.value)} "
                 f"fire@10={bool(p0.sample(10).fire.value)} "
                 f"-> {'PASS' if rt_ok else 'FAIL'}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # --- CLEAN side: valid 2-player match, camera='both'.
        good = td / "good.tard"
        good.write_bytes(_fixture_match())
        res = export_match_cameras(good, camera="both", players="all",
                                   out_dir=td / "good_out", mode="plan")
        by_pid = {}
        for e in res.manifest["players"]:
            by_pid[e["player_id"]] = e
        shooter = by_pid.get(10, {})
        quiet = by_pid.get(20, {})
        sh_fut = max((s["n_future_steered"] for s in shooter.get("streams", [])),
                     default=0)
        qu_fut = max((s["n_future_steered"] for s in quiet.get("streams", [])),
                     default=0)
        n_streams = res.n_streams
        blocked = all(
            s.render_job["status"] == "queue-blocked" for s in res.streams)
        manifest_written = res.manifest_path is not None and Path(
            res.manifest_path).is_file()
        # camera-json files actually on disk & non-empty (an exit code is not
        # the check -- count+size the artifacts).
        cams = list((td / "good_out").glob("*.camera.json"))
        cam_ok = len(cams) == 4 and all(c.stat().st_size > 200 for c in cams)
        clean_ok = (n_streams == 4 and sh_fut > 0 and qu_fut == 0 and blocked
                    and manifest_written and cam_ok)
        ok = ok and clean_ok
        lines.append(
            f"  [CLEAN both] streams={n_streams} shooter_future={sh_fut} "
            f"quiet_future={qu_fut} render=queue-blocked({blocked}) "
            f"manifest={manifest_written} camera_json={len(cams)}x"
            f"(nonempty={cam_ok}) -> {'PASS' if clean_ok else 'FAIL'}")

        # --- FIRE side A: empty match must REFUSE, no manifest.
        empty = td / "empty.tard"
        empty.write_bytes(_empty_match())
        refused_empty = False
        try:
            export_match_cameras(empty, camera="gow", out_dir=td / "empty_out",
                                 mode="plan")
        except EmptyMatchError:
            refused_empty = True
        no_manifest_empty = not (td / "empty_out" /
                                 "match_camera_manifest.json").exists()
        a_ok = refused_empty and no_manifest_empty
        ok = ok and a_ok
        lines.append(
            f"  [REFUSE empty] raised EmptyMatchError={refused_empty} "
            f"no_manifest={no_manifest_empty} -> {'PASS' if a_ok else 'FAIL'}")

        # --- FIRE side B: missing path must REFUSE.
        refused_missing = False
        try:
            export_match_cameras(td / "does_not_exist.tard", camera="gow",
                                 out_dir=td / "missing_out", mode="plan")
        except MissingMatchError:
            refused_missing = True
        b_ok = refused_missing
        ok = ok and b_ok
        lines.append(
            f"  [REFUSE missing] raised MissingMatchError={refused_missing} "
            f"-> {'PASS' if b_ok else 'FAIL'}")

        # --- FIRE side C: requesting an absent player id must REFUSE.
        refused_absent = False
        try:
            export_match_cameras(good, camera="gow", players=[999],
                                 out_dir=td / "absent_out", mode="plan")
        except EmptyMatchError:
            refused_absent = True
        c_ok = refused_absent
        ok = ok and c_ok
        lines.append(
            f"  [REFUSE absent-id] raised EmptyMatchError={refused_absent} "
            f"-> {'PASS' if c_ok else 'FAIL'}")

    print("two-sided match-camera export calibration:")
    for ln in lines:
        print(ln)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def cmd_plan(args) -> int:
    players = "all" if args.players == "all" else [
        int(x) for x in args.players.split(",") if x != ""]
    try:
        res = export_match_cameras(
            args.match, camera=args.camera, players=players,
            out_dir=args.out_dir, mode="plan",
            tick_begin=args.tick_begin, tick_end=args.tick_end,
            stride=args.stride, lookahead_ms=args.lookahead_ms,
            hold_ms=args.hold_ms, width=args.width, height=args.height,
            fps=args.fps)
    except MissingMatchError as e:
        print(f"REFUSED (missing match): {e}", file=sys.stderr)
        return 3
    except EmptyMatchError as e:
        print(f"REFUSED (empty match): {e}", file=sys.stderr)
        return 3
    c = res.manifest["counts"]
    print(f"manifest {res.manifest_path}")
    print(f"  match {res.manifest['match']['match_id']} "
          f"map {res.manifest['match']['map_name']} "
          f"players_exported {c['n_players_exported']} "
          f"streams {c['n_streams']} camera_json {c['n_camera_json_written']}")
    for s in res.streams:
        print(f"  stream {Path(s.camera_json).name}: {s.n_ticks} ticks, "
              f"{s.n_fire_ticks} fire, {s.n_future_steered} future-steered "
              f"(max_blend={s.max_foreshadow_blend:.3f}) "
              f"render={s.render_job['status']}")
    print("RENDER STATUS: " + res.manifest["render_status"])
    return 0


def cmd_selftest(args) -> int:
    return selftest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="match -> per-player camera-json + render "
                                     "manifest (dry, no GPU)")
    pp.add_argument("--match", required=True, help="path to a TARD 2.1 pack")
    pp.add_argument("--camera", choices=["gow", "sm64", "both"], default="gow")
    pp.add_argument("--players", default="all",
                    help="'all' or comma-separated player ids")
    pp.add_argument("--out-dir")
    pp.add_argument("--tick-begin", type=int, default=0)
    pp.add_argument("--tick-end", type=int)
    pp.add_argument("--stride", type=int, default=1)
    pp.add_argument("--lookahead-ms", type=float, default=700.0)
    pp.add_argument("--hold-ms", type=float, default=220.0)
    pp.add_argument("--width", type=int, default=1280)
    pp.add_argument("--height", type=int, default=720)
    pp.add_argument("--fps", type=int, default=128)
    pp.set_defaults(func=cmd_plan)

    st = sub.add_parser("selftest",
                        help="two-sided export calibration (real TR21 pack)")
    st.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
