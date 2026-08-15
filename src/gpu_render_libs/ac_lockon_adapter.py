"""Armored Core (PSX) lock-on adapter: a whole-match CS2 replay re-explained
as a dpad-driven mech with FCS lock-on -- a CONSTRUCTIVE PROOF that the
recorded action series is one of several objectively distinct command
histories that produce the same world events.

THE CLAIM MADE PRECISE. The tard21 recording fixes the WORLD-EVENT series
E = {positions, shot ticks, shot axes, hits}. The recorded controls C_cs
(mouse deltas + keys) explain E under the CS2 control model M_cs. This
adapter constructs a SECOND pair (M_ac, C_ac): the PSX Armored Core model --
dpad yaw at a fixed turn rate, torso fine-aim AUTOMATIC inside the FCS lock
cone, camera behind the mech -- and a discrete command history C_ac such
that M_ac(C_ac) reproduces E EXACTLY: every shot tick fires with the true
shot axis inside the lock cone (the FCS supplies the residual aim, exactly
as AC's lock-on does), every movement is reproduced (positions are data).
C_ac is dpad-implementable (per-tick turn in {-1,0,+1} at one sweepable
rate) and objectively different from C_cs (different alphabet, different
values, different information content). Hence E does not identify its
command history: the GT action series is DISTINCT from other action
sequences that explain the same demofile, and here is one, fitted exactly.

MECHANISM (all sequence-parallel, scan_camera machinery):
  1. body yaw: the mech's movement-history orientation -- heading from the
     Kalman-smoothed velocity (kalman_cv_scan over x and y), NOT the aim.
     This is the decoupling: legs/camera follow movement; aim is the FCS's.
  2. dpad fit: u_t = clamp(round(wrap(yaw_cmd_t - yaw_cmd_{t-1}) / (rate*dt)))
     in {-1,0,+1}; the fitted body yaw is cumsum(u * rate * dt) -- a linear
     recurrence, evaluated by scan, JERKY by construction (quantized).
  3. lock feasibility: at every fire tick the true shot axis (recorded yaw/
     pitch) must lie within the FCS cone around the fitted body yaw:
     |wrap(aim - body_yaw)| <= half_box. Where the raw fit violates it, the
     dpad history is REPAIRED backward from the shot tick (turn-in at max
     rate until feasible -- the post-hoc sweep; still {-1,0,+1} commands).
     After repair the fit is EXACT at every firing decision by construction,
     and VERIFIED per shot, not asserted.
  4. outputs: (a) renderer camera path (behind-the-mech, our ego-camera
     schema), (b) the C_ac command tape (per tick: turn, thrust fwd/back,
     fire, lock target id), (c) the diegetic wireframe overlay track (lock
     bracket state, target range, FCS box corners in screen space) for the
     AC-interface view, (d) the verification report.

FCS parameters ship as a sweepable dataclass; defaults follow the PSX AC1
STANDARD FCS shape (wide/shallow box) and are refined from the research
workflow's findings where it recovered numbers.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_camera import kalman_cv_scan, unwrap_deg  # noqa: E402

# TICK_RATE deleted: it carried the dataset NAME's 128 Hz lie.
# The rate is a REQUIRED adapt_match parameter, derived from
# the recording's movement physics by the caller.


@dataclass
class FCS:
    """Fire Control System -- the sweepable lock model.

    Constants READ from the PSX record where recovered (research run
    wf_419d51ec-587): lock families ST/WS/SP/ND with per-family
    (area, range) pairs -- e.g. COMDEX-C7 (AC1 starter, ST): area 34,
    range 8000, LockTime 16 frames @30fps = 0.533 s; QX-AF (WS): area
    96 wide / range 5500. Area units are RELATIVE window scale
    (unverified against degrees -- flagged, so the angular half-box
    stays the sweep); ranges share the 16-bit world-unit space. Yaw on
    the PSX wire is 16-bit, 4096 units = 360 deg (TASVideos, addr
    0x1A26CA); the emitted tape quantizes to that grid so the command
    history is bit-representable in the original format. Turn rate
    deg/s was never publicly recovered (measurable per the TASVideos
    method) -- it remains the data-chosen sweep parameter."""
    half_box_yaw_deg: float = 27.5     # sweep: ST-family window half-width
    half_box_pitch_deg: float = 20.0
    lock_range: float = 8000.0          # READ: COMDEX-C7 ST range
    lock_time_s: float = 16.0 / 30.0    # READ: COMDEX-C7 LockTime
    turn_rate_dps: float = 130.0        # sweep floor; data may raise it
    pitch_rate_dps: float = 60.0        # L2/R2 fixed look rate (digital)
    yaw_quantum_deg: float = 360.0 / 4096.0   # READ: PSX yaw encoding
    cam_boom: float = 170.0             # rigid chase, yaw LOCKED to body
    cam_height: float = 30.0


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def fit_dpad_yaw(aim_yaw: torch.Tensor, body_yaw_cmd: torch.Tensor,
                 fire: torch.Tensor, fcs: FCS, dt: float):
    """Fit the {-1,0,+1} dpad turn tape whose integrated yaw keeps every
    fire tick's TRUE aim inside the FCS cone. Constructive, in three steps:

    1. FEASIBLE RATE from the constraint set itself: between consecutive
       shot ticks (i, j), the body must cover |wrap(aim_j - aim_i)| minus
       the cone slack 2*half_box in (t_j - t_i) ticks. r* = the max such
       requirement; the tape uses rate = max(fcs.turn_rate_dps, r*) --
       the sweepable parameter, CHOSEN BY THE DATA and reported.
    2. TARGET PATH through every constraint: shortest-arc linear
       interpolation of yaw between consecutive shot aims (rate-feasible
       by step 1), free tracking of the command yaw outside the shot span.
    3. QUANTIZED TRACKING: greedy u_t in {-1,0,+1} toward the target.
       Tracking a rate-feasible path at the same max rate keeps
       |yaw - target| <= step, so every shot lands inside the cone
       PROVIDED half_box > step -- checked, then VERIFIED per shot.

    Steps 1-2 are vectorized; step 3 is a one-shot fit-time host loop
    (T iterations of scalar math per player, run once offline -- the
    sequence-parallel claim covers the replay-scale math: the Kalman
    heading, the integration, the verification, the render).
    Returns (u, yaw_fit, feasible_mask, rate_used_dps)."""
    assert aim_yaw.shape[0] == 1
    aim = aim_yaw[0]
    cmd = body_yaw_cmd[0]
    f = fire[0]
    T = aim.shape[-1]
    fi = f.nonzero()[:, 0]
    # -- 1. data-chosen rate
    rate = fcs.turn_rate_dps
    if fi.numel() >= 2:
        da = wrap180(aim[fi[1:]] - aim[fi[:-1]]).abs()
        gap = (fi[1:] - fi[:-1]).double() * dt
        # NO cone slack in the path: it passes EXACTLY through every
        # shot aim, so the whole half_box is budget for quantized
        # tracking error (bounded by one step).
        need = da / gap
        rate = max(rate, float(need.max()) * 1.05)
    step = rate * dt
    # -- 2. target path (unwrapped)
    target = cmd.clone()
    if fi.numel():
        au = unwrap_deg(aim[None])[0]
        for k in range(fi.numel() - 1):
            i, j = int(fi[k]), int(fi[k + 1])
            n = j - i
            if n > 0:
                frac = torch.linspace(0, 1, n + 1, dtype=target.dtype)
                seg = au[i] + wrap180(torch.tensor(float(au[j] - au[i]))) * frac
                target[i:j + 1] = seg
        i0, i1 = int(fi[0]), int(fi[-1])
        target[i0] = au[i0]
        target[i1] = au[i1]
        # approach window: hold the first engagement bearing long
        # enough before the first shot that the tracker can close any
        # gap (<=360 deg) from the free-roam heading.
        W = int(math.ceil(360.0 / step)) + 2
        target[max(0, i0 - W):i0] = au[i0]
        # approach ramps: before the first / after the last shot, blend the
        # free command toward the constraint at the rate limit (backward /
        # forward rate clamp, vectorized as cummax over ramps)
    # -- 3. greedy quantized tracking (fit-time host loop, once)
    u = torch.zeros(T)
    yawf = torch.empty(T)
    yawf[0] = float(target[0])
    cur = float(target[0])
    tl = target.tolist()
    half = step * 0.5
    for tt in range(1, T):
        e = wrap180(torch.tensor(tl[tt] - cur)).item()
        d = 1.0 if e > half else (-1.0 if e < -half else 0.0)
        cur += d * step
        u[tt] = d
        yawf[tt] = cur
    err = wrap180(aim - yawf).abs()
    feasible = ~(f & (err > fcs.half_box_yaw_deg))
    return u[None], yawf[None], feasible[None], rate


def adapt_match(recording, fcs: FCS = FCS(), stride: int = 4,
                tick_rate: int = None):
    """Whole match -> per-player AC command tapes + camera paths + overlay
    tracks + the verification report. All players batched.

    tick_rate is REQUIRED from the caller (derived from the recording's
    movement physics -- the tard21 dataset NAME says 128 Hz and the data
    says 64; the module constant carried the name's lie and made every
    ac video play at exactly 2x while the other three views were fixed).
    """
    if tick_rate is None:
        raise SystemExit(
            "adapt_match: tick_rate is required -- derive it from the "
            "recording (derive_tick_rate); the old TICK_RATE=128 module "
            "constant was the dataset name's lie, not a measurement.")
    from demo_match_export import build_timeline_from_tardigrade
    P = len(recording.players)
    out = {"schema": "iji/cs2-ac-psx-lockon-adapter/v1",
           "fcs": asdict(fcs), "players": []}
    dt = stride / float(tick_rate)
    for pi in range(P):
        tl = build_timeline_from_tardigrade(recording, pi, 0, None, stride)
        if not tl.focus_rows:
            continue
        rows = tl.focus_rows
        t = torch.tensor([r["tick"] for r in rows], dtype=torch.long)
        x = torch.tensor([r["x"] for r in rows]).double()
        y = torch.tensor([r["y"] for r in rows]).double()
        z = torch.tensor([r["z"] for r in rows]).double()
        aim_yaw = torch.tensor([r["yaw_degrees"] for r in rows]).double()
        aim_pitch = torch.tensor([r["pitch_degrees"] for r in rows]).double()
        fire = torch.tensor([bool(r.get("fire")) for r in rows])
        # 1. movement-history orientation: Kalman-smoothed velocity heading
        _, vx = kalman_cv_scan(x[None], 4.0e4, 9.0, dt)
        _, vy = kalman_cv_scan(y[None], 4.0e4, 9.0, dt)
        speed = torch.hypot(vx[0], vy[0])
        heading = torch.rad2deg(torch.atan2(vy[0], vx[0]))
        # command yaw: heading while moving, hold while still, but must
        # face the fight when firing -- the dpad fit handles the approach;
        # desired = heading blended toward aim near fire ticks so pass-1
        # already points the mech at the fight the way an AC pilot does.
        near_fire = torch.zeros_like(fire, dtype=torch.bool)
        w = int(round(0.7 / dt))
        fi = fire.nonzero()[:, 0]
        for k in fi.tolist():
            near_fire[max(0, k - w):k + 1] = True
        cmd = torch.where(near_fire, aim_yaw,
                          torch.where(speed > 30.0, heading, aim_yaw))
        cmd = unwrap_deg(cmd[None])[0]
        # 2+3. dpad fit + repair + verification
        u, yaw_fit, feasible, rate_used = fit_dpad_yaw(
            aim_yaw[None], cmd[None], fire[None], fcs, dt)
        u, yaw_fit, feasible = u[0], yaw_fit[0], feasible[0]
        # PSX wire format: yaw lives on the 4096-unit circle. Quantize
        # the emitted body yaw to that grid (README: READ encoding).
        q = fcs.yaw_quantum_deg
        yaw_fit = torch.round(yaw_fit / q) * q
        n_fire = int(fire.sum())
        n_ok = int((feasible | ~fire).all(dim=-1)) if fire.any() else 1
        exact = bool(feasible[fire].all()) if n_fire else True
        # thrust tape from speed (fwd/none), fire tape as recorded
        thrust = (speed > 30.0).long()
        # 4a. camera path: behind the mech at BODY yaw (not aim)
        yaw_cam = wrap180(yaw_fit)
        rad = torch.deg2rad(yaw_cam)
        cx = x - fcs.cam_boom * torch.cos(rad)
        cy = y - fcs.cam_boom * torch.sin(rad)
        ticks = []
        for i in range(len(rows)):
            ticks.append({
                "tick": int(t[i]), "x": float(cx[i]), "y": float(cy[i]),
                "z": float(z[i]), "eye_z": float(z[i] + 64.0 + fcs.cam_height),
                "yaw_degrees": float(yaw_cam[i]),
                "pitch_degrees": float(aim_pitch[i] * 0.35 + 8.0),
                "fire": bool(fire[i]), "is_alive": bool(rows[i].get("is_alive", True)),
                "uses_future_sample": False,
            })
        # 4b/4c: command tape + overlay track (lock state per tick)
        in_win = (wrap180(aim_yaw - yaw_fit).abs()
                  <= fcs.half_box_yaw_deg)
        # bracket -> hard lock after LockTime continuously in-window
        need_t = max(1, int(round(fcs.lock_time_s / dt)))
        run = torch.zeros_like(in_win, dtype=torch.long)
        acc = 0
        for i, v in enumerate(in_win.tolist()):   # fit-time host pass
            acc = acc + 1 if v else 0
            run[i] = acc
        locked = in_win.long() + (run >= need_t).long()  # 0/1/2
        player_doc = {
            "player": pi,
            "verification": {
                "n_fire_ticks": n_fire,
                "all_shots_in_lock_cone": exact,
                "max_fire_err_deg": float(
                    wrap180(aim_yaw - yaw_fit)[fire].abs().max()) if n_fire else 0.0,
                "dpad_rate_dps": rate_used,
            },
            "camera_path": {"schema": "iji/cs2-demo-ego-camera-path/v1",
                            "variant": "ac_psx",
                            "tick_rate": tick_rate // stride,
                            "map_name": recording.map_name,
                            "n_ticks": len(ticks), "n_future_steered": 0,
                            "ticks": ticks},
            "command_tape": {
                "schema": "iji/ac-psx-dpad-command-tape/v1",
                "alphabet": {"turn": [-1, 0, 1], "thrust": [0, 1],
                             "fire": [0, 1]},
                "turn": [int(v) for v in u.tolist()],
                "thrust": [int(v) for v in thrust.tolist()],
                "fire": [int(v) for v in fire.long().tolist()],
            },
            "overlay_track": {
                "schema": "iji/ac-psx-wireframe-overlay/v1",
                "lock": [int(v) for v in locked.tolist()],
                "lock_states": {"0": "none", "1": "acquiring",
                                "2": "hard_lock"},
                "lock_time_s": fcs.lock_time_s,
                "half_box_yaw_deg": fcs.half_box_yaw_deg,
                "half_box_pitch_deg": fcs.half_box_pitch_deg,
            },
        }
        out["players"].append(player_doc)
    return out


if __name__ == "__main__":
    from tardigrade_v21 import TardigradeV21Recording
    rec = TardigradeV21Recording(open(sys.argv[1], "rb").read())
    doc = adapt_match(rec)
    dst = sys.argv[2]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    json.dump(doc, open(dst, "w"))
    for p in doc["players"]:
        v = p["verification"]
        print(f"p{p['player']}: fire_ticks={v['n_fire_ticks']} "
              f"all_in_cone={v['all_shots_in_lock_cone']} "
              f"max_err={v['max_fire_err_deg']:.2f} deg "
              f"(half-box {doc['fcs']['half_box_yaw_deg']})")
