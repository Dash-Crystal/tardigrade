"""Author forklift mount/drive/dismount clips -- mediocre by license, exact
by schema.

The clips sidecar schema (READ from agents.clips.pt): per clip {clip name,
frames, duration, additive, blocks:[{skeleton, bone_ids, parents, ref_t
(74,3), ref_q (74,4), t (F,74,3), q (F,74,4), s (F,74)}]}, PARENT-space
local tracks on the shared worldmodel skeleton. Mediocrity policy, stated:
no bone semantics are invented -- the seated pose is the existing crouch
idle (a real engine pose), mount/dismount are timed interpolations between
the real standing and crouching poses (nlerp on q, lerp on t/s). Every
frame of every authored clip is therefore a convex combination of poses the
engine itself shipped, which is as honest as mediocre gets.

Output: overlay sidecar agents.clips.forklift.pt with events
forklift_mount / forklift_drive / forklift_dismount, loadable by the same
reader as the base sidecar (verified structurally at write time).
"""
from __future__ import annotations

import os
import sys

import torch


def _first_nonadditive(clips):
    for c in clips:
        if not c.get("additive") and c["blocks"]:
            return c
    raise SystemExit("no non-additive clip in category")


def _pose(clip, frame=0):
    b = clip["blocks"][0]
    return b["t"][frame], b["q"][frame], b["s"][frame], b


def _interp(pa, pb, F):
    ta, qa, sa, _ = pa
    tb, qb, sb, _ = pb
    ts, qs, ss = [], [], []
    for i in range(F):
        u = i / (F - 1)
        ts.append(ta * (1 - u) + tb * u)
        q = qa * (1 - u) + qb * u          # nlerp: mediocre, monotone
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        qs.append(q)
        ss.append(sa * (1 - u) + sb * u)
    return torch.stack(ts), torch.stack(qs), torch.stack(ss)


def _mk_clip(name, t, q, s, ref_block, fps=30.0):
    F = t.shape[0]
    return {"clip": f"overlay/forklift/{name}.vnmclip",
            "frames": F, "duration": F / fps, "additive": False,
            "blocks": [{"skeleton": ref_block["skeleton"],
                        "bone_ids": ref_block["bone_ids"],
                        "parents": ref_block["parents"],
                        "ref_t": ref_block["ref_t"],
                        "ref_q": ref_block["ref_q"],
                        "pose_space": ref_block["pose_space"],
                        "t": t, "q": q, "s": s}]}


def main(base_sidecar, out_path):
    d = torch.load(base_sidecar, map_location="cpu", weights_only=False)
    stand = _first_nonadditive(d["events"]["idle"])
    crouch = _first_nonadditive(d["events"]["crouch"])
    ps = _pose(stand)
    pc = _pose(crouch)
    t, q, s = _interp(ps, pc, 16)                 # ~0.53 s mount
    mount = _mk_clip("forklift_mount", t, q, s, ps[3])
    td, qd, sd = pc[0][None].repeat(2, 1, 1), pc[1][None].repeat(2, 1, 1), \
        pc[2][None].repeat(2, 1)
    drive = _mk_clip("forklift_drive", td, qd, sd, pc[3])
    t2, q2, s2 = _interp(pc, ps, 16)
    dismount = _mk_clip("forklift_dismount", t2, q2, s2, ps[3])
    out = {"schema": d["schema"], "skeleton": d["skeleton"],
           "scope": "overlay: forklift phase (fl_ maps)",
           "shared_by": d["shared_by"],
           "events": {"forklift_mount": [mount],
                      "forklift_drive": [drive],
                      "forklift_dismount": [dismount]}}
    # structural verification: the overlay must satisfy the reader's shape
    # expectations exactly (same keys, same per-block tensor ranks/widths).
    rb = d["events"]["idle"][0]["blocks"][0]
    for ev, cl in out["events"].items():
        b = cl[0]["blocks"][0]
        assert set(b) == set(rb), f"{ev}: block keys differ"
        assert b["t"].shape[1:] == rb["t"].shape[1:], f"{ev}: t width"
        assert b["q"].shape[1:] == rb["q"].shape[1:], f"{ev}: q width"
        assert b["s"].shape[1:] == rb["s"].shape[1:], f"{ev}: s width"
    torch.save(out, out_path)
    print(f"authored {list(out['events'])} on {out['skeleton'][:40]}... "
          f"-> {out_path} (schema-conformant, verified)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
