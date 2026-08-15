"""Retrocausal third-person demo-export camera CONTROLLER.

A PURE FUNCTION of (demo timeline for a match, focus player) -> a per-tick
camera pose emitted in the EXISTING ego-camera-path JSON schema that
`harness/counter_strike_tard21_blender_ego_v2.py` and gpu_render consume
(the same `{"ticks": [{tick, x, y, eye_z, yaw_degrees, pitch_degrees, fire,
...}]}` shape `harness/cs2_demo_camera.py camera` already writes). The
renderer already poses the third-person player WORLDMODEL + weapon-in-hand
from demo state (anim-pm's stack); THIS module emits ONLY the camera
trajectory, so the camera rows are the CAMERA's position/angles, not the
player's.

WHY THIS IS NOT `cs2_demo_camera.py`. That module reads the .dem into the
FIRST-PERSON (ego) path -- the player's own eye. This one is the
THIRD-PERSON director. It CONSUMES cs2_demo_camera's parsing (its
field-read discipline, `ego_fire_ticks`, the eye-height read) rather than
re-deriving any of it; see `build_timeline_from_demo`.

CONTROL LAW (derived from first principles; no game's code or assets copied
-- "GoW" and "SM64" name camera FEELS realised fresh as parameter sets):

  * Base pose: low, close to the head, third-person over-the-shoulder.
  * TWO directions are estimated per tick and kept SEPARATE:
      (a) a Kalman/EMA-smoothed MOVEMENT+GAZE heading -- a constant-velocity
          Kalman filter on X/Y gives a smoothed velocity; a circular EMA of
          the view yaw gives gaze; they are blended by speed (still -> gaze,
          moving -> movement). The camera BODY trails this heading.
      (b) the player's actual AIM yaw/pitch (the demo view angles). The
          camera SWINGS to align with aim when the player is about to shoot.
  * RETROCAUSAL foreshadowing -- the whole point. This is a REPLAY EXPORT,
    so the FUTURE of the demo is in hand. The pre-shoot predictor reads
    tick t+N (upcoming weapon_fire within `lookahead_ms`) and ramps a
    `foreshadow_blend` UP BEFORE the shot, swinging the body toward aim and
    steering the framing target (angle mostly, depth a little) to bring the
    ABOUT-TO-BE-ENGAGED enemy into frame BEFORE a live camera could know it
    exists. Every tick a future sample touched carries `uses_future_sample`
    True plus the `future_fire_tick` / `framed_enemy_index` that drove it --
    legible only "with wallhacks on", which is exactly why it PROVES we
    render demofiles, not restream video (a restream cannot see t+0.7s).

TWO-SIDED CALIBRATION (`python3 demo_camera.py selftest`): the pre-shoot
predictor must FIRE (blend rises through the pre-fire window) on a timeline
with a known upcoming shot AND stay in movement-follow (blend == 0,
uses_future_sample False) on a known no-fire stretch. A predictor that
always swings measures nothing and FAILS the no-fire side.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Angle helpers. Source convention (READ from cs2_demo_camera's emitted
# `coordinate_convention`): z up; yaw 0 = +x, ccw; pitch POSITIVE = looking
# DOWN. forward = (cos p cos y, cos p sin y, -sin p).
# ---------------------------------------------------------------------------


def forward_from_angles(yaw_deg: float, pitch_deg: float) -> tuple:
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    cp = math.cos(p)
    return (cp * math.cos(y), cp * math.sin(y), -math.sin(p))


def angles_from_direction(dx: float, dy: float, dz: float) -> tuple:
    """(yaw_deg, pitch_deg) that a camera at the origin looks along (dx,dy,dz).

    Inverse of forward_from_angles: pitch positive when the target is BELOW
    (dz < 0), matching the engine's positive-down pitch.
    """
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        return 0.0, 0.0
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.asin(max(-1.0, min(1.0, dz / n))))
    return yaw, pitch


def wrap180(a: float) -> float:
    """Fold a degree delta into (-180, 180] -- the SHORTEST signed turn."""
    return (a + 180.0) % 360.0 - 180.0


def circ_slerp(a_deg: float, b_deg: float, t: float) -> float:
    """Interpolate a_deg -> b_deg the short way; t in [0,1]."""
    return a_deg + wrap180(b_deg - a_deg) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 0.0 if x < edge0 else 1.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Estimators.
# ---------------------------------------------------------------------------


class CVKalman1D:
    """Constant-velocity scalar Kalman filter: state [pos, vel].

    Derived, not fitted: a CV process model (x += v*dt) with white
    acceleration noise `q`, updated by a position measurement with variance
    `r`. Its whole job is a smoothed VELOCITY that a raw finite difference on
    jittery demo positions does not give.
    """

    def __init__(self, q: float, r: float, dt: float):
        self.q = q
        self.r = r
        self.dt = dt
        self.x = [0.0, 0.0]                       # pos, vel
        self.P = [[1e3, 0.0], [0.0, 1e3]]         # large initial uncertainty
        self.started = False

    def step(self, z: float) -> tuple:
        dt = self.dt
        if not self.started:
            self.x = [z, 0.0]
            self.started = True
            return self.x[0], self.x[1]
        # Predict.
        px = self.x[0] + dt * self.x[1]
        pv = self.x[1]
        P = self.P
        # F P F^T
        p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1]
        p01 = P[0][1] + dt * P[1][1]
        p10 = P[1][0] + dt * P[1][1]
        p11 = P[1][1]
        # + Q  (continuous white-accel, discretised)
        q = self.q
        p00 += q * dt ** 3 / 3.0
        p01 += q * dt ** 2 / 2.0
        p10 += q * dt ** 2 / 2.0
        p11 += q * dt
        # Update with measurement of pos (H = [1, 0]).
        s = p00 + self.r
        k0 = p00 / s
        k1 = p10 / s
        y = z - px
        self.x = [px + k0 * y, pv + k1 * y]
        self.P = [[(1 - k0) * p00, (1 - k0) * p01],
                  [p10 - k1 * p00, p11 - k1 * p01]]
        return self.x[0], self.x[1]


class CircEMA:
    """Exponential moving average of an ANGLE (degrees), on the short arc."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.v = None

    def step(self, a_deg: float) -> float:
        if self.v is None:
            self.v = a_deg
        else:
            self.v = self.v + self.alpha * wrap180(a_deg - self.v)
        return self.v


class Spring1D:
    """Critically-damped angular spring toward a moving target (degrees).

    `omega` is the natural frequency (rad/s). Semi-implicit integration on
    the SHORT arc so a snap across +/-180 does not spin the long way. Higher
    omega => snappier (GoW aim-snap); lower => laggier (SM64 float).
    """

    def __init__(self, omega: float, dt: float):
        self.omega = omega
        self.dt = dt
        self.a = None
        self.v = 0.0

    def step(self, target_deg: float, omega: float = None) -> float:
        w = self.omega if omega is None else omega
        dt = self.dt
        if self.a is None:
            self.a = target_deg
            self.v = 0.0
            return self.a
        err = wrap180(target_deg - self.a)
        accel = w * w * err - 2.0 * w * self.v
        self.v += accel * dt
        self.a = self.a + self.v * dt
        return self.a


# ---------------------------------------------------------------------------
# Timeline (the INPUT). Focus rows are cs2_demo_camera ego rows; `enemies`
# is a tick -> [ {x,y,z,index,alive,team} ] lookup for retrocausal framing.
# ---------------------------------------------------------------------------


@dataclass
class Timeline:
    tick_rate: int
    focus_rows: list                       # list[dict] sorted by tick
    enemies_by_tick: dict = field(default_factory=dict)   # tick -> [dict]
    map_name: str = None
    demo: str = None
    steamid: int = None


# ---------------------------------------------------------------------------
# Style variants -- parameter sets over the ONE control law.
# ---------------------------------------------------------------------------

GOW = {
    "name": "gow",
    # over-the-shoulder rig, engine units (inches).
    "boom": 54.0,          # distance behind the head
    "height": 14.0,        # raise above the eye
    "shoulder": 20.0,      # lateral offset (right shoulder)
    "focus_dist": 220.0,   # how far ahead of the head we aim the look target
    # follow feel.
    "follow_omega": 11.0,  # body trails heading -- brisk
    "snap_omega": 46.0,    # body swing when about to shoot -- fast
    "kalman_q": 4.0e4,
    "kalman_r": 9.0,
    "gaze_alpha": 0.35,
    "speed_ref": 130.0,    # units/s where movement fully wins over gaze
    "orbit": 0.35,         # bias of heading toward movement (stay behind)
    "pitch_down": 5.0,     # constant downward tilt
    "frame_weight_max": 0.55,   # angle pull toward the framed enemy
    "depth_gain": 0.15,    # extra look distance when framing (a LITTLE)
    "obstacle_lift": 0.0,  # GoW: tight, no floor lift
    "min_clearance": 8.0,
}

SM64 = {
    "name": "sm64",
    # Lakitu spring-orbit: higher, further, springier, softer.
    "boom": 150.0,
    "height": 62.0,
    "shoulder": 6.0,
    "focus_dist": 180.0,
    "follow_omega": 3.6,   # laggy float
    "snap_omega": 10.0,    # soft aim swing
    "kalman_q": 1.2e4,
    "kalman_r": 16.0,
    "gaze_alpha": 0.16,
    "speed_ref": 110.0,
    "orbit": 0.85,         # strongly orbit to stay behind movement
    "pitch_down": 17.0,
    "frame_weight_max": 0.40,
    "depth_gain": 0.28,
    "obstacle_lift": 1.0,  # cheap floor-clearance lift
    "min_clearance": 22.0,
}

VARIANTS = {"gow": GOW, "sm64": SM64}


# ---------------------------------------------------------------------------
# Retrocausal pre-shoot predictor.
# ---------------------------------------------------------------------------


def _future_fire(focus_rows: list, i: int, lookahead_ticks: int,
                 hold_ticks: int, tick_rate: int) -> tuple:
    """(foreshadow_blend, future_fire_tick, uses_future) at focus_rows[i].

    REFUSE-BY-DEFAULT: with no upcoming shot inside the lookahead window and
    no recent shot inside the hold window, blend is 0.0 and no future sample
    is consumed. `foreshadow_blend` rises as time-to-fire shrinks (the
    retrocausal ramp), and a `hold_ticks` tail keeps it up DURING a burst so
    the body does not un-swing between rounds of one shot.
    """
    t = focus_rows[i]["tick"]
    la_s = lookahead_ticks / float(tick_rate)
    pre_blend = 0.0
    fire_tick = None
    if lookahead_ticks > 0:
        j = i + 1
        while j < len(focus_rows):
            ft = focus_rows[j]["tick"]
            if ft - t > lookahead_ticks:
                break
            if focus_rows[j].get("fire"):
                ttf = (ft - t) / float(tick_rate)
                pre_blend = max(0.0, min(1.0, 1.0 - ttf / max(la_s, 1e-6)))
                fire_tick = ft
                break
            j += 1
    # Active/just-fired hold looking BACKWARD a short tail (and this tick).
    hold_blend = 0.0
    k = i
    while k >= 0:
        bt = focus_rows[k]["tick"]
        if t - bt > hold_ticks:
            break
        if focus_rows[k].get("fire"):
            decay = 1.0 - (t - bt) / float(max(hold_ticks, 1))
            hold_blend = max(hold_blend, decay)
        k -= 1
    blend = max(pre_blend, hold_blend)
    uses_future = fire_tick is not None and pre_blend > 0.0
    return blend, fire_tick, uses_future


def _select_enemy(tl: Timeline, fire_tick: int, aim_yaw: float,
                  fx: float, fy: float):
    """The enemy the focus player is ABOUT to gunbattle at `fire_tick`.

    Read from the FUTURE state: among enemies alive at the fire tick, the one
    whose bearing from the shooter is closest to the shooter's aim yaw at
    that tick (the subject being shot at), tie-broken by nearness. Returns a
    row dict from that tick or None.
    """
    cands = tl.enemies_by_tick.get(fire_tick)
    if not cands:
        return None
    best = None
    best_key = None
    for e in cands:
        if not e.get("alive", True):
            continue
        bearing = math.degrees(math.atan2(e["y"] - fy, e["x"] - fx))
        ang = abs(wrap180(bearing - aim_yaw))
        dist = math.hypot(e["x"] - fx, e["y"] - fy)
        key = (ang, dist)
        if best_key is None or key < best_key:
            best_key = key
            best = e
    return best


# ---------------------------------------------------------------------------
# The controller.
# ---------------------------------------------------------------------------


def compute_camera_path(tl: Timeline, variant: dict,
                        lookahead_ms: float = 700.0,
                        hold_ms: float = 220.0) -> dict:
    """Pure controller: Timeline -> ego-schema camera-path payload.

    No I/O, no demo parsing -- this is the calibratable core.
    """
    rows = tl.focus_rows
    tr = tl.tick_rate
    if not rows:
        return {"ticks": [], "variant": variant["name"], "lookahead_ms":
                lookahead_ms}
    # dt from the median tick step actually present (the demo may be decimated
    # before it reaches us). READ the step; do not assume 1.
    steps = [rows[i + 1]["tick"] - rows[i]["tick"] for i in range(len(rows) - 1)]
    step = sorted(steps)[len(steps) // 2] if steps else 1
    dt = max(1, step) / float(tr)

    kx = CVKalman1D(variant["kalman_q"], variant["kalman_r"], dt)
    ky = CVKalman1D(variant["kalman_q"], variant["kalman_r"], dt)
    gaze = CircEMA(variant["gaze_alpha"])
    body = Spring1D(variant["follow_omega"], dt)

    lookahead_ticks = int(round(lookahead_ms / 1000.0 * tr))
    hold_ticks = int(round(hold_ms / 1000.0 * tr))

    out = []
    n_future = 0
    n_fire = 0
    max_blend = 0.0
    for i, r in enumerate(rows):
        px, py = float(r["x"]), float(r["y"])
        eye_z = float(r.get("eye_z", float(r.get("z", 0.0)) + 64.0))
        _, vx = kx.step(px)
        _, vy = ky.step(py)
        speed = math.hypot(vx, vy)
        aim_yaw = float(r["yaw_degrees"])
        aim_pitch = float(r["pitch_degrees"])
        gaze_yaw = gaze.step(aim_yaw)
        move_yaw = math.degrees(math.atan2(vy, vx)) if speed > 1e-6 else gaze_yaw

        # (a) movement+gaze heading: still -> gaze, moving -> movement. The
        # orbit param biases toward movement (stay behind).
        w_move = smoothstep(0.0, variant["speed_ref"], speed)
        w_move = w_move + (1.0 - w_move) * variant["orbit"] * \
            smoothstep(0.0, variant["speed_ref"] * 0.35, speed)
        heading_yaw = circ_slerp(gaze_yaw, move_yaw, w_move)

        # retrocausal pre-shoot predictor.
        blend, fire_tick, uses_future = _future_fire(
            rows, i, lookahead_ticks, hold_ticks, tr)
        max_blend = max(max_blend, blend)

        # (b) desired BODY yaw: trail heading, swing to aim as blend rises.
        desired_yaw = circ_slerp(heading_yaw, aim_yaw, blend)
        omega = variant["follow_omega"] + \
            (variant["snap_omega"] - variant["follow_omega"]) * blend
        cam_yaw = body.step(desired_yaw, omega)

        # camera rig: behind + up + right of the head.
        fwd = (math.cos(math.radians(cam_yaw)), math.sin(math.radians(cam_yaw)))
        right = (math.sin(math.radians(cam_yaw)), -math.cos(math.radians(cam_yaw)))
        cam_x = px - variant["boom"] * fwd[0] + variant["shoulder"] * right[0]
        cam_y = py - variant["boom"] * fwd[1] + variant["shoulder"] * right[1]
        cam_z = eye_z + variant["height"]
        if variant["obstacle_lift"] > 0.0:
            floor = float(r.get("z", eye_z - 64.0))
            cam_z = max(cam_z, floor + variant["min_clearance"] +
                        variant["height"] * 0.0)

        # look target: ahead of the head along heading, pulled toward the
        # framed enemy (ANGLE mostly, DEPTH a little) when foreshadowing.
        fw = variant["frame_weight_max"] * blend
        depth = variant["focus_dist"] * (1.0 + variant["depth_gain"] * blend)
        look_yaw = circ_slerp(heading_yaw, aim_yaw, blend)
        lf = forward_from_angles(look_yaw, aim_pitch * 0.5 + variant["pitch_down"])
        tgt_x = px + depth * lf[0]
        tgt_y = py + depth * lf[1]
        tgt_z = eye_z + depth * lf[2]
        framed_enemy = None
        if uses_future and fire_tick is not None:
            enemy = _select_enemy(tl, fire_tick, aim_yaw, px, py)
            if enemy is not None:
                framed_enemy = enemy.get("index")
                mx = 0.5 * (px + enemy["x"])
                my = 0.5 * (py + enemy["y"])
                mz = 0.5 * (eye_z + enemy["z"])
                tgt_x = tgt_x + (mx - tgt_x) * fw
                tgt_y = tgt_y + (my - tgt_y) * fw
                tgt_z = tgt_z + (mz - tgt_z) * fw

        yaw_deg, pitch_deg = angles_from_direction(
            tgt_x - cam_x, tgt_y - cam_y, tgt_z - cam_z)
        pitch_deg += variant["pitch_down"] * 0.0   # tilt already in the target

        if uses_future:
            n_future += 1
        if r.get("fire"):
            n_fire += 1
        rec = {
            "tick": int(r["tick"]),
            "x": cam_x, "y": cam_y, "z": float(r.get("z", eye_z - 64.0)),
            "eye_z": cam_z,
            "yaw_degrees": yaw_deg,
            "pitch_degrees": pitch_deg,
            "fov": r.get("fov"),
            # carried straight through so the muzzle-flash / weapon gates the
            # renderer already joins on the camera path keep working.
            "fire": bool(r.get("fire")),
            "fire_weapon": r.get("fire_weapon"),
            "active_weapon_name": r.get("active_weapon_name"),
            "is_alive": bool(r.get("is_alive", True)),
            # retrocausal legibility -- the proof we saw t+N.
            "uses_future_sample": bool(uses_future),
            "foreshadow_blend": blend,
            "future_fire_tick": fire_tick if uses_future else None,
            "framed_enemy_index": framed_enemy,
            # instantaneous headings, kept SEPARATE, for inspection.
            "heading_yaw": heading_yaw,
            "aim_yaw": aim_yaw,
            "body_yaw": cam_yaw,
            "speed": speed,
        }
        out.append(rec)

    return {
        "schema": "iji/cs2-demo-thirdperson-camera-path/v1",
        "consumes_schema": "iji/cs2-demo-ego-camera-path/v1",
        "variant": variant["name"],
        "variant_params": variant,
        "lookahead_ms": lookahead_ms,
        "hold_ms": hold_ms,
        "lookahead_ticks": lookahead_ticks,
        "tick_rate": tr,
        "demo": tl.demo,
        "map_name": tl.map_name,
        "steamid": tl.steamid,
        "coordinate_convention":
            "source-z-up; yaw0=+x ccw; pitch+down; eye_z = camera height",
        "retrocausal_note":
            "uses_future_sample True => this pose was steered by a demo tick "
            "in the future (upcoming weapon_fire within lookahead_ms); a live "
            "restream cannot produce it",
        "n_ticks": len(out),
        "n_fire_ticks": n_fire,
        "n_future_steered": n_future,
        "max_foreshadow_blend": max_blend,
        "ticks": out,
    }


# ---------------------------------------------------------------------------
# Demo adapter -- CONSUMES cs2_demo_camera's parsing; no re-derivation.
# ---------------------------------------------------------------------------


def build_timeline_from_demo(demo: str, player_index: int, tick_rate: int,
                             tick_begin=None, tick_end=None) -> Timeline:
    import sys
    here = Path(__file__).resolve()
    harness = here.parent.parent            # harness/gpu_render -> harness
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    import cs2_demo_camera as ego            # reuse its field-read discipline

    p = ego._parser(demo)
    h = p.parse_header()
    df, _props, _missing = ego._tick_frame(p)
    ids = sorted(df.steamid.dropna().unique())
    sid = int(ids[player_index])
    fire_ticks, fire_lines = ego.ego_fire_ticks(p, sid)
    for ln in fire_lines:
        print(ln)

    def _sub(this_sid):
        s = df[df.steamid == this_sid].sort_values("tick")
        if tick_begin is not None:
            s = s[s.tick >= tick_begin]
        if tick_end is not None:
            s = s[s.tick < tick_end]
        return s

    focus_rows = []
    for row in _sub(sid).itertuples(index=False):
        z = getattr(row, "Z", None)
        if z is None or z != z:
            continue
        duck = getattr(row, "duck_amount", 0.0) or 0.0
        eye = (ego.EYE_HEIGHT_STANDING -
               (ego.EYE_HEIGHT_STANDING - ego.EYE_HEIGHT_DUCKING) * float(duck))
        focus_rows.append({
            "tick": int(row.tick),
            "x": float(row.X), "y": float(row.Y), "z": float(z),
            "eye_z": float(z) + eye,
            "yaw_degrees": float(row.yaw), "pitch_degrees": float(row.pitch),
            "fov": (float(getattr(row, "fov", float("nan")))
                    if getattr(row, "fov", None) == getattr(row, "fov", None)
                    else None),
            "is_alive": bool(getattr(row, "is_alive", False)),
            "active_weapon_name": getattr(row, "active_weapon_name", None),
            "fire": int(row.tick) in fire_ticks,
            "fire_weapon": fire_ticks.get(int(row.tick)),
        })

    # Enemy positions per tick, for retrocausal framing (READ from the same
    # tick frame; teams differ from the focus player's).
    focus_team = None
    fsub = _sub(sid)
    if len(fsub) and "team_num" in df.columns:
        focus_team = fsub.team_num.dropna().iloc[0] if len(
            fsub.team_num.dropna()) else None
    enemies_by_tick = {}
    for other in ids:
        if int(other) == sid:
            continue
        for row in _sub(other).itertuples(index=False):
            z = getattr(row, "Z", None)
            if z is None or z != z:
                continue
            team = getattr(row, "team_num", None)
            if focus_team is not None and team is not None and team == focus_team:
                continue                      # skip teammates
            enemies_by_tick.setdefault(int(row.tick), []).append({
                "index": int(other) % 100000,
                "steamid": int(other),
                "x": float(row.X), "y": float(row.Y), "z": float(z) + 40.0,
                "alive": bool(getattr(row, "is_alive", False)),
                "team": int(team) if team == team and team is not None else None,
            })

    return Timeline(tick_rate=tick_rate, focus_rows=focus_rows,
                    enemies_by_tick=enemies_by_tick,
                    map_name=h.get("map_name"), demo=str(Path(demo).name),
                    steamid=sid)


# ---------------------------------------------------------------------------
# Synthetic timelines for two-sided calibration.
# ---------------------------------------------------------------------------


def _straight_walk(n: int, tick_rate: int, fire_tick=None,
                   enemy_side: float = 260.0):
    """Player walks +x at ~180 u/s (movement heading ~yaw 0).

    If fire_tick is given the player AIMS off-axis toward an enemy standing to
    the +y side (aim yaw ~+35, distinct from the movement heading so the
    body-swing is observable -- a fixture where aim == heading cannot exhibit
    the swing), ONE fire flag lands at fire_tick, and the enemy is alive
    throughout. Otherwise: gaze straight ahead, no fires, no enemies.
    """
    v = 180.0
    aim = 35.0 if fire_tick is not None else 0.0
    rows = []
    enemies = {}
    for k in range(n):
        t = k
        x = v * (t / float(tick_rate))
        rows.append({
            "tick": t, "x": x, "y": 0.0, "z": 0.0, "eye_z": 64.0,
            "yaw_degrees": aim, "pitch_degrees": 0.0, "fov": 90.0,
            "is_alive": True, "active_weapon_name": "ak47",
            "fire": (fire_tick is not None and t == fire_tick),
            "fire_weapon": ("ak47" if (fire_tick is not None and t == fire_tick)
                            else None),
        })
        if fire_tick is not None:
            ex = v * (fire_tick / float(tick_rate)) + 300.0
            enemies[t] = [{"index": 7, "x": ex, "y": enemy_side, "z": 40.0,
                           "alive": True, "team": 3}]
    return Timeline(tick_rate=tick_rate, focus_rows=rows,
                    enemies_by_tick=enemies, map_name="synthetic",
                    demo="selftest", steamid=1)


def selftest(lookahead_ms: float = 700.0) -> int:
    """FIRE on a known pre-fire window; CLEAN on a known no-fire stretch."""
    tr = 64
    la_ticks = int(round(lookahead_ms / 1000.0 * tr))
    ok = True
    lines = []

    # --- Side A: a shot is coming; the predictor must foreshadow it.
    fire_t = 150
    pre = _straight_walk(220, tr, fire_tick=fire_t)
    for var in ("gow", "sm64"):
        path = compute_camera_path(pre, VARIANTS[var], lookahead_ms=lookahead_ms)
        rows = {r["tick"]: r for r in path["ticks"]}
        window = [rows[t] for t in range(fire_t - la_ticks, fire_t) if t in rows]
        blends = [w["foreshadow_blend"] for w in window]
        rose = max(blends) if blends else 0.0
        near = rows.get(fire_t - 1, {}).get("foreshadow_blend", 0.0)
        used = any(w["uses_future_sample"] for w in window)
        framed = any(w["framed_enemy_index"] == 7 for w in window)
        # body must swing toward the enemy/aim: body_yaw departs from the
        # movement heading (~0) as we approach the shot.
        swing = abs(rows.get(fire_t - 1, {}).get("body_yaw", 0.0))
        a_ok = rose > 0.30 and near > 0.80 and used and framed and swing > 1.0
        ok = ok and a_ok
        lines.append(
            f"  [FIRE {var}] max_blend_in_window={rose:.3f} at_t-1={near:.3f} "
            f"uses_future={used} framed_enemy={framed} "
            f"body_swing_deg={swing:.2f} -> {'PASS' if a_ok else 'FAIL'}")

    # --- Side B: nobody fires; the predictor must stay in movement-follow.
    quiet = _straight_walk(220, tr, fire_tick=None)
    for var in ("gow", "sm64"):
        path = compute_camera_path(quiet, VARIANTS[var], lookahead_ms=lookahead_ms)
        blends = [r["foreshadow_blend"] for r in path["ticks"]]
        used = sum(1 for r in path["ticks"] if r["uses_future_sample"])
        framed = sum(1 for r in path["ticks"]
                     if r["framed_enemy_index"] is not None)
        maxb = max(blends) if blends else 0.0
        b_ok = maxb == 0.0 and used == 0 and framed == 0
        ok = ok and b_ok
        lines.append(
            f"  [QUIET {var}] max_blend={maxb:.3f} future_steered={used} "
            f"framed={framed} -> {'PASS' if b_ok else 'FAIL'} "
            f"(an always-swing predictor FAILS here)")

    # --- Side C: lookahead=0 disables foreshadowing (no PRE-fire steer);
    # only the active-fire hold at/after the shot may raise blend.
    pre0 = compute_camera_path(pre, VARIANTS["gow"], lookahead_ms=0.0)
    rows0 = {r["tick"]: r for r in pre0["ticks"]}
    pre_blends0 = [rows0[t]["foreshadow_blend"]
                   for t in range(fire_t - la_ticks, fire_t) if t in rows0]
    c_ok = (max(pre_blends0) if pre_blends0 else 0.0) == 0.0
    ok = ok and c_ok
    lines.append(
        f"  [LA=0 gow] max_pre_fire_blend={max(pre_blends0) if pre_blends0 else 0:.3f}"
        f" -> {'PASS' if c_ok else 'FAIL'} (lookahead is what enables retrocausality)")

    print("two-sided pre-shoot predictor calibration "
          f"(lookahead_ms={lookahead_ms}, lookahead_ticks={la_ticks}):")
    for ln in lines:
        print(ln)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def cmd_path(args) -> int:
    tl = build_timeline_from_demo(
        args.demo, args.player_index, args.tick_rate,
        tick_begin=args.tick_begin, tick_end=args.tick_end)
    variant = VARIANTS[args.variant]
    payload = compute_camera_path(tl, variant, lookahead_ms=args.lookahead_ms,
                                  hold_ms=args.hold_ms)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload))
    print(f"wrote {args.out}: {payload['n_ticks']} ticks, variant "
          f"{payload['variant']}, {payload['n_fire_ticks']} fire ticks, "
          f"{payload['n_future_steered']} future-steered "
          f"(max_blend={payload['max_foreshadow_blend']:.3f}) on "
          f"{payload['map_name']}")
    return 0


def cmd_selftest(args) -> int:
    return selftest(lookahead_ms=args.lookahead_ms)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("path", help="demo -> third-person camera-json")
    pp.add_argument("--demo", required=True)
    pp.add_argument("--player-index", type=int, default=0)
    pp.add_argument("--variant", choices=sorted(VARIANTS), default="gow")
    pp.add_argument("--lookahead-ms", type=float, default=700.0,
                    help="retrocausal window: steer on weapon_fire up to this "
                         "far in the FUTURE (0 disables foreshadowing)")
    pp.add_argument("--hold-ms", type=float, default=220.0)
    pp.add_argument("--tick-begin", type=int)
    pp.add_argument("--tick-end", type=int)
    pp.add_argument("--tick-rate", type=int, default=64)
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=cmd_path)

    st = sub.add_parser("selftest", help="two-sided predictor calibration")
    st.add_argument("--lookahead-ms", type=float, default=700.0)
    st.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
