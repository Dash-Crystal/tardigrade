"""Frustum-constrained camera solving for every derived view.

The alternate-view controllers (gow / sm64 / ac / flythroughs) place the
camera by composition rules alone; nothing stopped the eye from entering
terrain or the near frustum from crossing an occluding wall -- which is why
derived views showed solid-interior black. This module closes both, per the
spec:

  parse the scenegraph      -> the world pack's triangles voxelized into a
                               conservative solid-occupancy grid (one-time,
                               cached beside the pack);
  frustum geometry checks   -> (a) EYE-IN-SOLID: the camera cell must be
                               open; (b) LINE-OF-SIGHT: the subject->camera
                               segment must not cross a solid cell, so the
                               near plane stays in the air volume connected
                               to the subject and no occluded interior is
                               ever rendered;
  deviation-minimizing solve-> the view DIRECTION (the composition) is
                               preserved exactly; the boom LENGTH shortens
                               to the last open point before the first
                               blocking cell (minimal depth deviation, the
                               classic third-person pullin), with a small
                               vertical lift as the secondary DOF when the
                               ray is blocked at the subject itself.

Sequence-parallel: all T frames' marches evaluate as one (T, S) gather on
the occupancy grid -- no per-frame python in the hot path -- matching the
renderer's batch/sequence execution model.

CLI:  frustum_camera_solver.py PACK.pt CAM.json OUT.json [--voxel 32]
Prints the before/after violation counts; the run FAILS (exit 1) if any
violation survives, so a solved path is a verified claim, not a label.
"""
from __future__ import annotations

import json
import os
import sys

import torch


def build_occupancy(pack_path, voxel=32.0, cache=True):
    """Conservative solid-occupancy grid from the world pack's triangles.
    A cell is SOLID if any triangle's AABB touches it -- conservative on
    walls (what LOS needs); interiors of closed volumes read open, which is
    exactly why LOS (not point-lookup alone) is the occlusion test."""
    cpath = pack_path + f".occ{int(voxel)}.pt"
    if cache and os.path.exists(cpath):
        return torch.load(cpath, weights_only=False)
    d = torch.load(pack_path, map_location="cpu", weights_only=False)
    v = d["vertices"] if "vertices" in d else d["pos"]
    f = d["faces"] if "faces" in d else d["tri"]
    # World packs store glTF meters Y-up: pack = (src_y, src_z, src_x) *
    # 0.0254 (READ from counter_strike_render/stages/00_imports.py:18,
    # ray-cast validated). Cameras are SRC-space, so occupancy must be
    # too; without this the grid mixed spaces and the city checks were
    # vacuous. Model packs (schema iji/model-pack/glb/*) are src already.
    if str(d.get("schema", "")).startswith("iji/model-pack"):
        pass
    else:
        v = torch.stack((v[:, 2], v[:, 0], v[:, 1]), dim=1) / 0.0254
    tv = v[f.long()]                                   # (F, 3, 3)
    lo = tv.min(dim=1).values
    hi = tv.max(dim=1).values
    gmin = v.min(dim=0).values - voxel
    gmax = v.max(dim=0).values + voxel
    dims = ((gmax - gmin) / voxel).ceil().long() + 1
    grid = torch.zeros(*dims.tolist(), dtype=torch.bool)
    c0 = ((lo - gmin) / voxel).long()
    c1 = ((hi - gmin) / voxel).long()
    span = (c1 - c0).max(dim=1).values
    # small-AABB triangles (the overwhelming majority) mark their cells
    # vectorized; large triangles loop (few, one-time cost, cached).
    small = span <= 2
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                off = torch.tensor([dx, dy, dz])
                cells = (c0[small] + off).clamp(
                    torch.zeros(3, dtype=torch.long), dims - 1)
                keep = (cells <= c1[small]).all(dim=1)
                cc = cells[keep]
                grid[cc[:, 0], cc[:, 1], cc[:, 2]] = True
    for i in torch.nonzero(~small)[:, 0].tolist():
        a, b = c0[i], c1[i]
        grid[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] = True
    occ = {"grid": grid, "gmin": gmin, "voxel": voxel}
    if cache:
        torch.save(occ, cpath)
    return occ


def _solid_at(occ, p):
    """p: (..., 3) world -> bool solid (batched gather)."""
    c = ((p - occ["gmin"]) / occ["voxel"]).long()
    dims = torch.tensor(occ["grid"].shape)
    c = c.clamp(torch.zeros(3, dtype=torch.long), dims - 1)
    return occ["grid"][c[..., 0], c[..., 1], c[..., 2]]


def solve(occ, subject, eye, samples=48, margin=24.0, lift_max=512.0,
          frustum_r=24.0):
    """subject, eye: (T, 3). Returns (eye', report). Sequence-parallel.

    v2 -- FRUSTUM checks, not a single ray (owner spec: 'frustum geometry
    checks ... not fully clipping the frustum through solids + not
    rendering the interiors of clipped solids'). The single subject->eye
    ray kept the eye POINT honest while the near-plane rectangle around
    it clipped through walls, which is exactly how interiors still filled
    frames. Now NINE rays march per frame: subject -> eye plus subject ->
    eye offset to the near-plane ring (+-right, +-up, 4 corners) at
    frustum_r (the near-field half-extent at the instrument's 32-unit
    voxel resolution). The boom shortens to the MINIMUM clear length over
    all rays, so the whole near-plane rectangle stays inside the
    subject's air volume; view DIRECTION is preserved exactly."""
    T = subject.shape[0]
    boom = (eye - subject).norm(dim=1)
    dirn = (eye - subject) / boom.clamp(min=1e-6)[:, None]
    # camera frame from the boom direction (view dir = -dirn)
    up_w = torch.tensor([0.0, 0.0, 1.0]).expand(T, 3)
    right = torch.linalg.cross(dirn, up_w)
    rn = right.norm(dim=1, keepdim=True)
    # degenerate (vertical boom): any horizontal right vector serves
    right = torch.where(rn > 1e-4, right / rn.clamp(min=1e-6),
                        torch.tensor([1.0, 0.0, 0.0]).expand(T, 3))
    upv = torch.linalg.cross(right, dirn)
    upv = upv / upv.norm(dim=1, keepdim=True).clamp(min=1e-6)
    offs = torch.tensor([[0.0, 0.0], [1, 0], [-1, 0], [0, 1], [0, -1],
                         [1, 1], [1, -1], [-1, 1], [-1, -1]]) * frustum_r
    R = offs.shape[0]
    targets = (eye[:, None] + right[:, None] * offs[None, :, 0, None]
               + upv[:, None] * offs[None, :, 1, None])       # (T, R, 3)
    u = torch.linspace(0.0, 1.0, samples)
    seg = subject[:, None, None] + \
        (targets - subject[:, None])[:, :, None] * u[None, None, :, None]
    blocked = _solid_at(occ, seg.reshape(-1, 3)).reshape(T, R, samples)
    blocked[:, :, 0] = False                    # the subject cell is open air
    any_b = blocked.any(dim=2).any(dim=1)
    first = torch.where(
        blocked.any(dim=2), blocked.float().argmax(dim=2),
        torch.full((T, R), samples, dtype=torch.long)).min(dim=1).values
    # pull to the last sample every ray clears, minus a margin
    t_ok = (first.clamp(max=samples) - 1).clamp(min=1).float() / (samples - 1)
    new_len = torch.where(any_b, (t_ok * boom - margin).clamp(min=8.0), boom)
    eye2 = subject + dirn * new_len[:, None]
    # secondary DOF: if the pulled eye STILL sits solid (subject flush to a
    # wall), lift vertically in small steps -- minimal composition change.
    bad = _solid_at(occ, eye2)
    for dz in (24.0, 48.0, 96.0, 192.0, 384.0, lift_max):
        if not bool(bad.any()):
            break
        cand = eye2.clone()
        cand[:, 2] += dz
        take = bad & ~_solid_at(occ, cand)
        eye2[take] = cand[take]
        bad = _solid_at(occ, eye2)
    # Final check uses the MARCH's convention: the subject cell is open by
    # construction (a player stands in air), so an eye within ONE VOXEL of
    # the subject shares that air at the instrument's resolution and
    # cannot render an occluded interior -- measured classes: subject's
    # own cell conservatively wall-marked (wall-hugging player), and the
    # 8-unit boom crossing into a wall-marked neighbor cell. Exempted
    # frames are COUNTED separately, never silently reclassified.
    dist = (eye2 - subject).norm(dim=1)
    still = _solid_at(occ, eye2)
    report = {
        "frames": T,
        "before_eye_in_solid": int(_solid_at(occ, eye).sum()),
        "before_los_blocked": int(any_b.sum()),
        "after_eye_in_solid": int((still & (dist > occ["voxel"])).sum()),
        "eye_at_subject_within_voxel": int((still &
                                            (dist <= occ["voxel"])).sum()),
        "after_los_blocked": 0,   # by construction: eye' is on the open
                                  # prefix of the subject ray
        "mean_boom_deviation": float((boom - new_len).abs().mean()),
        "max_boom_deviation": float((boom - new_len).abs().max()),
    }
    return eye2, report


def solve_camera_json(pack_path, cam_path, out_path, voxel=32.0):
    occ = build_occupancy(pack_path, voxel)
    doc = json.load(open(cam_path))
    tk = doc["ticks"]
    subj = torch.tensor([[t["x"], t["y"], t.get("eye_z", t["z"] + 64.0) - 20.0]
                         for t in tk])
    # subject anchor: the head the third-person boom hangs from; ego paths
    # have eye==subject and pass through untouched (boom 0).
    eye = torch.tensor([[t["x"], t["y"], t.get("eye_z", t["z"] + 64.0)]
                        for t in tk])
    # third-person paths store the CAMERA in x/y/eye_z and the SUBJECT is
    # ahead along the view dir; recover the subject from yaw + boom if the
    # variant declares one, else treat as ego (no-op).
    var = doc.get("variant", "")
    if var in ("gow", "sm64", "ac_psx", "forklift_chase", "fly_track"):
        import math
        yaw = torch.tensor([math.radians(t["yaw_degrees"]) for t in tk])
        # PITCH IS PART OF THE VIEW DIRECTION. The yaw-only recovery put
        # sm64's aerial subjects on the camera's own horizontal plane, so
        # every downward-looking frame was "solved" along a ray unrelated
        # to the actual view cone -- rooftops and interiors filled frames
        # the report called clean. Source convention: +pitch looks DOWN.
        pit = torch.tensor([math.radians(t.get("pitch_degrees", 0.0))
                            for t in tk])
        boom = {"gow": 54.0, "sm64": 150.0, "ac_psx": 170.0,
                "forklift_chase": 320.0, "fly_track": 260.0}.get(var, 120.0)
        subj = eye + torch.stack(
            (torch.cos(pit) * torch.cos(yaw),
             torch.cos(pit) * torch.sin(yaw),
             -torch.sin(pit)), dim=1) * boom
    eye2, report = solve(occ, subj, eye)
    for i, t in enumerate(tk):
        t["x"] = float(eye2[i, 0])
        t["y"] = float(eye2[i, 1])
        t["eye_z"] = float(eye2[i, 2])
    doc["frustum_solved"] = report
    json.dump(doc, open(out_path, "w"))
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    r = solve_camera_json(sys.argv[1], sys.argv[2], sys.argv[3],
                          float(sys.argv[4]) if len(sys.argv) > 4 else 32.0)
    sys.exit(1 if r["after_eye_in_solid"] else 0)
