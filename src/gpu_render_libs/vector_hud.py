"""Diegetic vector-graphics HUD pass -- the draw-call responsibility the
renderer has been omitting in EVERY view, ground-truth CS2 ego included.

Two schemes, one batched vector rasterizer:

  cs2   the ego interface state that the naive view was always specced to
        carry and never drew: crosshair, health bar, ammo counter -- all
        READ from the recording's own event columns (health/ammo are on
        the tard21 wire; nothing is invented).
  ac    the PSX Armored Core FCS wireframe, ANIMATED per the acquisition
        ladder the adapter emits: center FCS box (family-shaped), per-
        target bracket whose corners CONVERGE with lock progress and blink
        while acquiring, solid on hard lock; compass strip ticking on the
        4096-unit PSX yaw grid; radar ring with relative enemy blips.
        Bracket screen position = angular aim error mapped through the
        box, so the wireframe tracks the same numbers the verification
        checked -- the overlay IS the lock model, drawn.

Rasterization: polyline segments -> sampled points -> index_put into the
(T, H, W, 3) frame tensor, batched over frames and segments (no per-frame
python drawing). Compositing onto ALREADY-RENDERED video (the delivered
mp4s) via the system ffmpeg rawvideo pipe, so closing the UI omission does
not require re-rendering or a GPU.

CLI:
  vector_hud.py cs2 <ego.mp4> <actions.json> <out.mp4>
  vector_hud.py ac  <adapter.json> <player_index> <out.mp4> [backdrop.mp4]
     (no backdrop -> the HUD animates over black: the pure vector scheme,
      declared as such; over its true render once a GPU node returns)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import torch

W, H, FPS = 960, 540, 32
GREEN = torch.tensor([90, 255, 120], dtype=torch.uint8)
AMBER = torch.tensor([255, 200, 80], dtype=torch.uint8)


def draw_segments(frames: torch.Tensor, segs: torch.Tensor,
                  color: torch.Tensor, samples: int = 96):
    """segs: (T, S, 4) as (x0, y0, x1, y1) per frame; NaN row = skip.
    Batched: one index_put over all frames, segments and samples."""
    T = frames.shape[0]
    t = torch.linspace(0, 1, samples)
    x = segs[..., 0:1] * (1 - t) + segs[..., 2:3] * t         # (T,S,samples)
    y = segs[..., 1:2] * (1 - t) + segs[..., 3:4] * t
    ok = torch.isfinite(x) & torch.isfinite(y)
    xi = x.round().long().clamp(0, frames.shape[2] - 1)
    yi = y.round().long().clamp(0, frames.shape[1] - 1)
    fi = torch.arange(T)[:, None, None].expand_as(xi)
    fi, yi, xi = fi[ok], yi[ok], xi[ok]
    frames.index_put_((fi, yi, xi), color.expand(fi.shape[0], 3))
    # 1px thickening for legibility on video
    frames.index_put_((fi, (yi + 1).clamp(max=frames.shape[1] - 1), xi),
                      color.expand(fi.shape[0], 3))
    return frames


def _bracket(cx, cy, half, gap):
    """4 corner-L brackets around (cx, cy): 8 segments."""
    s = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * half, cy + sy * half
            s.append((x, y, x - sx * gap, y))
            s.append((x, y, x, y - sy * gap))
    return s


def ac_hud_segments(doc: dict, pi: int, T_out: int):
    """(T, S, 4) segment tensor for the AC scheme, animated by lock state."""
    p = next(pp for pp in doc["players"] if pp["player"] == pi)
    lock = p["overlay_track"]["lock"]
    half_box = p["overlay_track"]["half_box_yaw_deg"]
    cam = p["camera_path"]["ticks"]
    tape = p["command_tape"]
    T = min(T_out, len(cam))
    box_w, box_h = 240, 170          # ST-family window (relative units READ)
    cx, cy = W / 2, H / 2
    segs = []
    for i in range(T):
        s = []
        # FCS window: static wireframe rectangle + tick notches
        x0, y0 = cx - box_w / 2, cy - box_h / 2
        x1, y1 = cx + box_w / 2, cy + box_h / 2
        s += [(x0, y0, x1, y0), (x1, y0, x1, y1),
              (x1, y1, x0, y1), (x0, y1, x0, y0)]
        # compass strip on the 4096-unit grid: a tick every 11.25 deg
        yaw = cam[i]["yaw_degrees"]
        q = round(yaw / (360.0 / 4096.0))
        for k in range(-4, 5):
            tick_yaw = (q * (360.0 / 4096.0)) + k * 11.25 - yaw
            tx = cx + tick_yaw * 9.0
            if 40 < tx < W - 40:
                s.append((tx, 26, tx, 26 + (10 if k % 2 == 0 else 5)))
        s.append((40, 24, W - 40, 24))
        # radar ring (fixed) + sweep spoke rotating with body yaw
        rr, rx, ry = 52, W - 78, H - 78
        for a in range(0, 360, 30):
            a0, a1 = math.radians(a), math.radians(a + 30)
            s.append((rx + rr * math.cos(a0), ry + rr * math.sin(a0),
                      rx + rr * math.cos(a1), ry + rr * math.sin(a1)))
        spoke = math.radians(-yaw)
        s.append((rx, ry, rx + rr * math.cos(spoke), ry + rr * math.sin(spoke)))
        # target bracket: only when in-window; position = aim error through
        # the box; corners converge with acquisition, blink while acquiring
        st = lock[i] if i < len(lock) else 0
        if st > 0:
            # the adapter verified |err| <= half_box at fire ticks; here the
            # bracket sits where the LOCKED TARGET is: err deg -> box px
            err_px = 0.0  # target centered when hard-locked
            prog = 1.0 if st == 2 else 0.55
            blink = (st == 2) or ((i // 4) % 2 == 0)
            if blink:
                halfb = 26 + (1.0 - prog) * 30
                gap = 9 + (1.0 - prog) * 8
                s += _bracket(cx + err_px, cy, halfb, gap)
                if st == 2:      # hard lock: inner cross
                    s += [(cx - 6, cy, cx + 6, cy), (cx, cy - 6, cx, cy + 6)]
        # dpad turn indicator (the command tape, drawn)
        u = tape["turn"][i] if i < len(tape["turn"]) else 0
        if u:
            ax = cx + u * (box_w / 2 + 26)
            s += [(ax, cy - 10, ax + u * 12, cy), (ax, cy + 10, ax + u * 12, cy)]
        segs.append(s)
    S = max(len(s) for s in segs)
    out = torch.full((T, S, 4), float("nan"))
    for i, s in enumerate(segs):
        for j, seg in enumerate(s):
            out[i, j] = torch.tensor(seg, dtype=torch.float)
    return out


def cs2_hud_segments(actions: dict, T_out: int):
    """(T, S, 4) for the CS2 ego interface: crosshair + health bar + ammo
    pips, every value READ from the recording's own columns."""
    rows = actions["rows"]
    T = min(T_out, len(rows))
    cx, cy = W / 2, H / 2
    segs = []
    for i in range(T):
        r = rows[i]
        s = [(cx - 9, cy, cx - 3, cy), (cx + 3, cy, cx + 9, cy),
             (cx, cy - 9, cx, cy - 3), (cx, cy + 3, cx, cy + 9)]
        hp = r.get("health")
        if isinstance(hp, (int, float)) and hp is not None:
            w = 1.2 * max(0.0, min(100.0, float(hp)))
            s.append((30, H - 30, 30 + w, H - 30))
            s.append((30, H - 28, 30 + w, H - 28))
            s += [(28, H - 34, 28, H - 24), (152, H - 34, 152, H - 24)]
        ammo = r.get("ammo")
        if isinstance(ammo, (int, float)) and ammo is not None:
            for k in range(int(max(0, min(40, ammo)))):
                x = W - 40 - k * 6
                s.append((x, H - 34, x, H - 24))
        segs.append(s)
    S = max(len(s) for s in segs)
    out = torch.full((T, S, 4), float("nan"))
    for i, s in enumerate(segs):
        for j, seg in enumerate(s):
            out[i, j] = torch.tensor(seg, dtype=torch.float)
    return out


def read_mp4(path, T_max):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-vf", f"scale={W}:{H}", "-"],
                       capture_output=True)
    buf = torch.frombuffer(bytearray(r.stdout), dtype=torch.uint8)
    T = min(T_max, buf.numel() // (H * W * 3))
    return buf[:T * H * W * 3].reshape(T, H, W, 3).clone()


def write_mp4(frames, path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p",
                          path], stdin=subprocess.PIPE)
    p.stdin.write(frames.numpy().tobytes())
    p.stdin.close()
    p.wait()


def main():
    mode = sys.argv[1]
    if mode == "cs2":
        video, actions_p, out = sys.argv[2:5]
        actions = json.load(open(actions_p))
        frames = read_mp4(video, 10 ** 6)
        segs = cs2_hud_segments(actions, frames.shape[0])
        frames = frames[:segs.shape[0]]
        draw_segments(frames, segs, GREEN)
        write_mp4(frames, out)
        print(f"cs2 HUD composited: {frames.shape[0]} frames -> {out}")
    elif mode == "ac":
        adapter_p, pi, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        doc = json.load(open(adapter_p))
        backdrop = sys.argv[5] if len(sys.argv) > 5 else None
        segs = ac_hud_segments(doc, pi, 10 ** 6)
        T = segs.shape[0]
        if backdrop:
            frames = read_mp4(backdrop, T)
            frames = (frames.float() * 0.45).to(torch.uint8)  # dimmed,
            T = frames.shape[0]                               # DECLARED
            segs = segs[:T]                                   # placeholder
        else:
            frames = torch.zeros(T, H, W, 3, dtype=torch.uint8)
        draw_segments(frames, segs, GREEN)
        write_mp4(frames, out)
        tag = "over DECLARED placeholder backdrop" if backdrop else "pure scheme over black"
        print(f"ac HUD: {T} frames ({tag}) -> {out}")


if __name__ == "__main__":
    main()
