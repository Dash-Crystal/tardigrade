#!/usr/bin/env python3
"""Extract the first-person ARMS/GLOVES, skinned to a weapon's idle pose.

WHY. The weapon bundle is weapon-only, and that is measurable rather than
cosmetic: scored against the cs1k corpus' galil-vs-knife region on tick
77159, the drawn Galil covers 0.0761 of the frame at delta>=24 but only
33.5% of its pixels fall inside the corpus band [357, 544, 993, 719]. The
corpus mask is a galil frame differenced against a knife frame, so it
contains the ARMS AND GLOVES in both poses -- 636 px of band against a
240-450 px weapon. The missing geometry is the arms, not a coefficient.

THE ROUTE, all read:
  * agents/models/shared/arms/<glove>/<glove>.vmdl_c ships two meshes,
    `worldmodel` (2,125 verts) and `viewmodel` (5,732). The first-person one
    is `viewmodel` and the NAME is checked, not the index.
  * That mesh carries BLENDINDICES_0 / BLENDWEIGHT_0, so it is SKINNED --
    unlike the weapon, which rides one bone and needed no palette.
  * BLENDINDICES DO NOT INDEX THE SKELETON. They index a per-mesh slice of
    DATA.m_remappingTable, whose slice starts are m_remappingTableStarts.
    The glove has 52 bones, a 104-entry table and starts [0, 52] -- one
    slice per mesh, `worldmodel` then `viewmodel`. Reading the indices
    straight put vertices on the wrong bones and, because the glove's bind
    pose is in PLAYER space (arm_upper_R at z = 61.66 source units) while
    the viewmodel skeleton's pose is camera-relative, the error did not look
    like a scramble: it looked like a 2.06 x 1.91 x 1.60 m bundle straddling
    the eye. The mesh's own m_nMeshIndex picks the slice.
  * Its own skeleton (DATA.m_modelSkeleton) is the BIND pose. 44 of its 52
    bones name-match animation/skeletons/characters/viewmodel.vnmskel.
  * The POSE comes from the same clip the weapon bundle already reads:
    animation/anims/viewmodel/<class>/<class>_<w>/idle_<w>.vnmclip, joined
    to the weapon by m_secondaryAnimations[].m_skeleton. All 50 of its
    arm/hand/finger tracks are static, so every bone's posed transform is
    the range-starts + m_constantRotation rule -- the same rule the weapon's
    `wpn` bone uses, proven there against the weapon skeleton's own bones to
    max|d| 1.2e-05 source units.

  Skinning is linear blend: v' = sum_i w_i * (P_i * inv(B_i) * v), with B the
  glove skeleton's model-space bind and P the viewmodel skeleton's model-space
  pose. Both are rigid (translation + unit quaternion + scale 1), so the
  inverse is exact rather than a matrix solve.

THE EIGHT UNMATCHED BONES are `*_TWIST` and `*_TWIST1` helpers on the upper
and lower arms, and they take their NEAREST POSED ANCESTOR's transform. That
is a stated approximation -- a twist helper exists precisely to differ from
its parent -- and the alternative is worse by two orders of magnitude, which
is why it is not the alternative: holding them at bind leaves their vertices
in the GLOVE model's space (arm_upper_R sits at z = 61.66 source units, i.e.
player-space, above the feet) while every posed vertex is camera-relative.
Measured, with 1,830 of 5,732 vertices weighted to those eight bones: the
bundle came out with a 2.06 x 2.49 x 1.60 m bounding box straddling the eye.
The count of affected vertices and the substituted bone are printed.

WHAT THIS DOES NOT DO. It does not write into the weapon bundle. The
viewmodel pass dispatches two families off `fam` -- 0 weapon, 1 legs -- and
an arms family has no consumer there yet; merging arms triangles into the
weapon bundle would shade them as a weapon skin. This writes its own file so
the draw side can take it when it has somewhere to put it.

Usage:
  python extract_arms.py --game <depot game/> --weapon galil_ar \
      --weapon-skel animation/skeletons/weapons/galil.vnmskel \
      --out-dir <dir>
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "vcs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ash_extract as A          # noqa: E402
import ash_to_world as _a2w      # noqa: E402
import bc_decode as _bc          # noqa: E402
import kv3v5
import mesh_skin                     # noqa: E402
from extract_viewmodel import (S, to_view, _qmul, _qrot, nm_skeleton,   # noqa: E402
                               find_idle_clip, _static_track)
from vm_slots import TEX_EDGE, SLOT_PARAM, UNMAPPED_PARAM   # noqa: E402

# WHICH GLOVE. The corpus does not carry the recording player's glove skin --
# it is a cosmetic item and cs1k's state has no field for it. This is a
# STATED choice of the default-shaped model, not a read one, and it prints
# itself as such. The geometry that matters here is the arm and the hand;
# glove variants differ in the fingers and in the material, not in the limb.
DEFAULT_GLOVE = "agents/models/shared/arms/glove_fullfinger/glove_fullfinger.vmdl_c"
ARMS_MESH = "viewmodel"


def _rigid(t, q):
    return np.asarray(t, np.float64), np.asarray(q, np.float64)


def _inv(t, q):
    qi = np.array([-q[0], -q[1], -q[2], q[3]])
    return -_qrot(qi, t), qi


def _apply(t, q, v):
    return _qrot(q, v) + t


def model_space_from_parents(names, parents, local):
    """Accumulate parent-space (t, q) down the parent chain. READ."""
    out = {}
    order = sorted(range(len(names)), key=lambda i: _depth(parents, i))
    for i in order:
        t, q = local[i]
        p = parents[i]
        if p < 0:
            out[names[i]] = (np.asarray(t, np.float64),
                             np.asarray(q, np.float64))
            continue
        pt, pq = out[names[p]]
        out[names[i]] = (pt + _qrot(pq, np.asarray(t, np.float64)),
                         _qmul(pq, np.asarray(q, np.float64)))
    return out


def _depth(parents, i):
    d = 0
    while parents[i] >= 0:
        i = parents[i]
        d += 1
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--map-vpk", default=None)
    ap.add_argument("--weapon", required=True,
                    help="cs1k enum name; the bundle is arms_<weapon>.pt")
    ap.add_argument("--weapon-skel", required=True,
                    help="animation/skeletons/weapons/<w>.vnmskel -- the "
                         "field that joins a weapon to its idle clip")
    ap.add_argument("--glove", default=DEFAULT_GLOVE)
    ap.add_argument("--mesh", default=ARMS_MESH)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--edge", type=int, default=TEX_EDGE)
    ap.add_argument("--cvar-offset", type=float, nargs=3, default=(0., 0., 0.),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--weapon-bundle", default=None,
                    help="the weapon's own bundle, whose muzzle and placement "
                         "keys this one CARRIES rather than re-derives. "
                         "Defaults to <out-dir>/<weapon>.pt.")
    a = ap.parse_args(argv)

    vpk = a.map_vpk or os.path.join(a.game, "csgo", "pak01_dir.vpk")
    content = A.Content(vpk, a.game, "pak01")
    OFF = np.asarray(a.cvar_offset, np.float64) * S
    print(f"arms for {a.weapon}")
    print(f"  glove {a.glove}  [STATED: the corpus carries no glove skin "
          f"field, so this is a choice of model, not a read]")

    d = content.find(a.glove)
    if d is None:
        raise SystemExit(f"{a.glove}: not in the content mount")
    blocks = A.ordered_blocks(d)
    by = {}
    for t, b in blocks:
        by.setdefault(t, b)
    ctrl = kv3v5.parse(by["CTRL"])
    meshes = {m.get("m_Name"): m for m in ctrl["embedded_meshes"]}
    if a.mesh not in meshes:
        raise SystemExit(f"{a.glove}: no mesh {a.mesh!r}; has {sorted(meshes)}")
    emb = meshes[a.mesh]
    mo = A.load_meshopt()
    attrs, _r, err = A.decode_vertex_buffer(np, mo, emb["m_vertexBuffers"][0],
                                            blocks)
    assert not err, err
    idx, err = A.decode_index_buffer(np, mo, emb["m_indexBuffers"][0], blocks)
    assert not err, err
    for need in ("BLENDINDICES_0", "BLENDWEIGHT_0"):
        if need not in attrs:
            raise SystemExit(
                f"{a.glove}::{a.mesh} has no {need}. This mesh is skinned by "
                f"construction and a rigid fallback would put the whole arm "
                f"on one bone -- refused rather than approximated.")
    # THE REMAP SLICE for this mesh, read from the model rather than assumed
    # to be the identity.
    _dat0 = kv3v5.parse(by["DATA"], strict=False)
    _rt = list(_dat0.get("m_remappingTable") or [])
    _rs = list(_dat0.get("m_remappingTableStarts") or [])
    _mi = int(emb.get("m_nMeshIndex", 0))
    if _rt and _rs:
        _start = _rs[_mi]
        _end = _rs[_mi + 1] if _mi + 1 < len(_rs) else len(_rt)
        REMAP = [int(x) for x in _rt[_start:_end]]
    else:
        REMAP = None
    pm = np.asarray(attrs["POSITION_0"], np.float64)
    nm = np.asarray(attrs["NORMAL_0_decoded"], np.float64)
    uv0 = np.asarray(attrs["TEXCOORD_0"], np.float64)
    bi = np.asarray(attrs["BLENDINDICES_0"]).astype(np.int64)
    bw = np.asarray(attrs["BLENDWEIGHT_0"], np.float64)
    if bw.max() > 1.5:                      # stored as bytes on some meshes
        bw = bw / 255.0
    wsum = bw.sum(1, keepdims=True)
    print(f"  mesh {a.mesh}: {len(pm)} verts, {len(idx)//3} tris, "
          f"{bi.shape[1]} influences/vertex, weight sums "
          f"min {wsum.min():.4f} max {wsum.max():.4f}")
    if REMAP is None:
        print(f"    m_remappingTable ABSENT -- blend indices taken as "
              f"skeleton indices, which is only right if the model ships no "
              f"remap")
    else:
        print(f"    m_remappingTable slice [{_start}:{_end}] for mesh index "
              f"{_mi}: {len(REMAP)} entries, blend index range "
              f"{int(bi.min())}..{int(bi.max())}")
        if int(bi.max()) >= len(REMAP):
            raise SystemExit(
                f"blend index {int(bi.max())} exceeds the {len(REMAP)}-entry "
                f"remap slice -- the slice is wrong and clamping would put "
                f"vertices on a plausible wrong bone")
    bw = bw / np.maximum(wsum, 1e-9)

    # --- the two skeletons ------------------------------------------------
    dat = kv3v5.parse(by["DATA"], strict=False)
    gs = dat["m_modelSkeleton"]
    gnames = list(gs["m_boneName"])
    gpar = list(gs["m_nParent"])
    glocal = [(np.asarray(p, np.float64), np.asarray(q, np.float64))
              for p, q in zip(gs["m_bonePosParent"], gs["m_boneRotParent"])]
    BIND = model_space_from_parents(gnames, gpar, glocal)
    print(f"  glove skeleton: {len(gnames)} bones (bind pose, model space)")

    vm_skel, vlines = nm_skeleton(
        content, "animation/skeletons/characters/viewmodel.vnmskel")
    for ln in vlines:
        print(ln)
    vdat = content.find("animation/skeletons/characters/viewmodel.vnmskel")
    vby = {}
    for t, b in A.ordered_blocks(vdat):
        vby.setdefault(t, b)
    vraw = kv3v5.parse(vby["DATA"])
    vnames = list(vraw["m_boneIDs"])
    vpar = list(vraw["m_parentIndices"])

    # --- the pose, from the weapon's own idle clip ------------------------
    import weapon_ids as _wid
    index = _wid.vpk_index(a.game)
    hits = find_idle_clip(content, index, a.weapon_skel)
    if not hits:
        raise SystemExit(
            f"no viewmodel idle clip names {a.weapon_skel} in "
            f"m_secondaryAnimations -- this weapon ships none in this depot, "
            f"so there is no pose to skin to and a bind-pose arm would be a "
            f"fabricated one")
    clip_path, clip = hits[0]
    tracks = clip["m_trackCompressionSettings"]
    print(f"  pose from {clip_path}  ({clip['m_nNumFrames']} frame(s), "
          f"{len(tracks)} tracks vs {len(vnames)} viewmodel bones)")
    if len(tracks) != len(vnames):
        raise SystemExit("track count != bone count, so track i is not bone i")
    # ANIMATED TRACKS ARE DECODED, not refused. This used to raise -- "their
    # values live in the compressed stream, which this reader does not
    # decode" -- and that was true until vnmclip.py read the format. Three
    # weapons (usp_s, desert_eagle, r8_revolver) ship idles whose ARM tracks
    # animate even where the weapon bone does not, so the refusal cost them
    # their arms entirely. FRAME 0 is a stated choice, the same one the
    # weapon bundle's hold pose makes and for the same reason.
    import vnmclip
    ok, clines = vnmclip.check_layout(clip)
    for ln in clines:
        print(" " + ln)
    if not ok:
        raise SystemExit("the clip's container check FAILED; refusing to "
                         "decode a layout that does not close on itself")
    nonstatic = [vnames[i] for i, t in enumerate(tracks)
                 if not (t["m_bIsTranslationStatic"]
                         and t["m_bIsRotationStatic"])]
    if nonstatic:
        print("  %d of %d tracks are ANIMATED and are DECODED at frame 0 "
              "(stated): %s%s" % (len(nonstatic), len(tracks),
                                  nonstatic[:6],
                                  " ..." if len(nonstatic) > 6 else ""))
    plocal = []
    for i, t in enumerate(tracks):
        if t["m_bIsTranslationStatic"] and t["m_bIsRotationStatic"]:
            tr, q, _sc = _static_track(t)
        else:
            d = vnmclip.decode_track(clip, i, 0)
            tr, q = d["translation"], d["rotation"]
        plocal.append((tr, q))
    POSE = model_space_from_parents(vnames, vpar, plocal)

    # --- skin -------------------------------------------------------------
    matched = [n for n in gnames if n in POSE]
    unmatched = [n for n in gnames if n not in POSE]
    print(f"  bone join: {len(matched)} of {len(gnames)} glove bones are "
          f"posed by the viewmodel skeleton")
    print(f"    HELD AT BIND, no counterpart to pose them: {unmatched}")
    # Per-bone rigid transform P * inv(B). An unmatched bone borrows the
    # POSE of its nearest posed ancestor while keeping its OWN inverse bind,
    # so its vertices land in the posed skeleton's space rather than in the
    # glove model's.
    sub = {}
    for n in gnames:
        if n in POSE:
            continue
        i = gnames.index(n)
        while i >= 0 and gnames[i] not in POSE:
            i = gpar[i]
        if i < 0:
            raise SystemExit(f"bone {n!r} has no posed ancestor at all, so "
                             f"there is no frame to put its vertices in")
        sub[n] = gnames[i]
    if sub:
        for k, v in sub.items():
            print(f"    {k} -> nearest posed ancestor {v} (STATED "
                  f"approximation: a twist helper is not its parent)")
    xf = {}
    for n in gnames:
        bt, bq = BIND[n]
        it, iq = _inv(bt, bq)
        pt, pq = POSE[sub.get(n, n)]
        xf[n] = (pt + _qrot(pq, it), _qmul(pq, iq))

    pos = np.zeros_like(pm)
    nrm = np.zeros_like(nm)
    held = np.zeros(len(pm), bool)
    for k in range(bi.shape[1]):
        w = bw[:, k:k + 1]
        for b in np.unique(bi[:, k]):
            sel = bi[:, k] == b
            if not sel.any():
                continue
            _real = REMAP[int(b)] if REMAP is not None else int(b)
            name = gnames[_real] if _real < len(gnames) else None
            if name is None:
                raise SystemExit(f"blend index {int(b)} -> bone {_real} "
                                 f"exceeds the glove skeleton's "
                                 f"{len(gnames)} bones")
            t, q = xf[name]
            # _qrot broadcasts a (3,) quaternion over an (N,3) block, so the
            # whole bone's vertices transform in one call rather than a
            # per-vertex loop over 5,732 vertices x 4 influences.
            pos[sel] += w[sel] * (_qrot(q, pm[sel]) + t)
            nrm[sel] += w[sel] * _qrot(q, nm[sel])
            if name in sub:
                held |= sel & (bw[:, k] > 0.0)
    print(f"    {int(held.sum())} of {len(pm)} vertices carry weight on a "
          f"substituted bone ({100.0 * held.mean():.2f}%)")

    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    view = to_view(pos) * S + OFF
    nview = to_view(nrm)
    nview /= np.maximum(np.linalg.norm(nview, axis=1, keepdims=True), 1e-12)

    tri = idx.reshape(-1, 3).astype(np.int64)
    tan = np.zeros((len(pm), 3))
    e1 = view[tri[:, 1]] - view[tri[:, 0]]
    e2 = view[tri[:, 2]] - view[tri[:, 0]]
    d1 = uv0[tri[:, 1]] - uv0[tri[:, 0]]
    d2 = uv0[tri[:, 2]] - uv0[tri[:, 0]]
    den = (d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    tt = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / den[:, None]
    for k in range(3):
        np.add.at(tan, tri[:, k], tt)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)

    # --- material pages, by the same mesh -> drawcall -> vmat route --------
    md = kv3v5.parse(blocks[emb["m_nDataBlock"]][1], strict=False)
    mats = []
    for o in md.get("m_sceneObjects") or []:
        for c in o.get("m_drawCalls") or []:
            if c.get("m_material") and c["m_material"] not in mats:
                mats.append(c["m_material"])
    print(f"  materials on {a.mesh}: {mats}")
    pages, prov = {}, {}
    if mats:
        row, err = A.read_material(content, mats[0])
        if err:
            print(f"  material unreadable: {err}")
        else:
            tp = row["texture_params"]
            print(f"    shader {row['shader_name']}")
            for slot, param in SLOT_PARAM.items():
                p = tp.get(param)
                if not p:
                    prov[slot] = f"PARAMETER ABSENT: {param}"
                    print(f"    slot {slot:8s} <- {param}: ABSENT")
                    continue
                raw = content.find(p)
                if raw is None:
                    prov[slot] = "MISSING: " + p
                    print(f"    slot {slot:8s}: MISSING {p}")
                    continue
                os.makedirs(a.out_dir, exist_ok=True)
                tmp = os.path.join(a.out_dir, ".arms_slot.vtex_c")
                with open(tmp, "wb") as fh:
                    fh.write(raw)
                tnm, w, h, blk, fmtnum = _a2w.read_vtex(tmp, want=a.edge,
                                                        allow_lz4=True)
                os.remove(tmp)
                if blk is None:
                    prov[slot] = f"UNDECODED ({tnm}): {p}"
                    print(f"    slot {slot:8s}: {prov[slot]}")
                    continue
                px = _bc.decode(blk, fmtnum, w, h)
                pages[slot] = torch.tensor(px.copy())
                prov[slot] = f"REAL {p} :: {tnm} mip@{w}x{h}"
                print(f"    slot {slot:8s}: {prov[slot]}")
            for slot, (param, why) in UNMAPPED_PARAM.items():
                p = tp.get(param)
                if p:
                    prov[slot] = f"UNMAPPED: {p} -- {why}"
    for extra in mats[1:]:
        print(f"    further draw call, NOT merged: {extra}")

    bundle = {
        "pos": torch.tensor(view, dtype=torch.float32),
        "uv": torch.tensor(np.concatenate([uv0, uv0], axis=1),
                           dtype=torch.float32),
        "nrm": torch.tensor(nview, dtype=torch.float32),
        "tan": torch.tensor(tan, dtype=torch.float32),
        "loc": torch.tensor(pos, dtype=torch.float32),
        "tri": torch.tensor(tri, dtype=torch.int32),
        "fam": torch.zeros(len(tri), dtype=torch.long),
        "is_legs": torch.zeros(len(pm), dtype=torch.bool),
        "source": f"{a.glove} :: {a.mesh}",
        "weapon": a.weapon,
        "pose_clip": clip_path,
        "glove_basis": "STATED: cs1k carries no glove-skin field",
        "bones_posed": len(matched),
        "bones_substituted": sub,
        "verts_on_substituted_bones": int(held.sum()),
        "offset_m": OFF.tolist(),
        "axis_map": "view = (-Ym, +Zm, -Xm) * 0.0254 + cvar offset, after "
                    "linear-blend skinning to the idle clip's static pose",
        "tex_pages": pages,
        "tex_provenance": prov,
        "tex_edge": a.edge,
    }
    # PER-VERTEX BINDINGS, shipped rather than only consumed. This extractor
    # has always READ the blend streams -- it has to, to bake the idle pose --
    # but it kept them to itself, so the bundle arrived as a rigid snapshot
    # and no consumer could move an arm to any other pose. Same read, now
    # written down.
    _skin, _sklines = mesh_skin.read_skin(
        attrs, emb, _dat0, gnames, label=f"{a.glove}::{a.mesh}")
    for _ln in _sklines:
        print(_ln)
    mesh_skin.add_skin(bundle, _skin, gs, len(pm))
    # --- CARRY the weapon's muzzle and placement keys -------------------
    # An arms bundle is what every egocentric clip loads, so a consumer
    # looking for `muzzle_flash_src` looks HERE, and all 37 of these were
    # missing it -- the flash raised KeyError and refused to guess, which
    # is the correct refusal and a real blocker.
    #
    # COPIED, NOT RE-DERIVED. The weapon bundle already put the muzzle
    # point through the same view-space transform with the same cvar offset
    # and the same posed `wpn`, and its own comment gives the reason:
    # two copies of a transform is two things to fix and one of them gets
    # forgotten. This bundle shares that view space exactly, so the value
    # transfers unchanged.
    #
    # bind_translation_src / bind_applied_src are DELIBERATELY NOT carried.
    # They are the WEAPON's rigid bind; this mesh is skinned from the
    # glove's own bind pose through a bone palette, so copying them would
    # plant a number that is wrong for this geometry and looks right.
    CARRY = ("muzzle_flash_src", "muzzle_flash_view", "muzzle_flash_note",
             "nm_skeleton", "wpn_bone_src", "wpn_bone_quat",
             "wpn_pose_basis", "placement_basis", "offset_basis",
             "dxy_basis", "dxy_muzzle", "dxy_landmark", "sequences",
             # the muzzle's CHAIN travels with the muzzle, or a consumer of
             # this bundle recomputes it the mesh's way and gets the old
             # point. mesh_* is deliberately NOT carried: this mesh is
             # skinned and does not use that chain at all.
             "muzzle_bind_subtracted", "muzzle_view_chain")
    wb = a.weapon_bundle or os.path.join(a.out_dir, f"{a.weapon}.pt")
    if os.path.isfile(wb):
        src = torch.load(wb, weights_only=False, map_location="cpu")
        carried = [k for k in CARRY if k in src]
        for k in carried:
            bundle[k] = src[k]
        bundle["muzzle_from"] = os.path.basename(wb)
        bundle["bind_keys_absent_why"] = (
            "bind_translation_src/bind_applied_src are the WEAPON's rigid "
            "bind and this mesh is skinned; carrying them would be a wrong "
            "number that looks right")
        print(f"  carried {len(carried)} weapon key(s) from "
              f"{os.path.basename(wb)}: {carried}")
        missing = [k for k in CARRY if k not in src]
        if missing:
            print(f"    NOT in the weapon bundle, so not carried: {missing}")
    else:
        bundle["muzzle_from"] = None
        print(f"  NO weapon bundle at {wb}: the muzzle and placement keys "
              f"are ABSENT from this arms bundle. A consumer that needs "
              f"muzzle_flash_src will not find it, and that is a visible "
              f"absence rather than a fabricated point.")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"arms_{a.weapon}.pt")
    torch.save(bundle, out)
    print(f"wrote {out}")
    p = view
    print(f"  view-space bbox (metres): min {np.round(p.min(0), 3)}  "
          f"max {np.round(p.max(0), 3)}")
    print(f"  |normal| mean {np.linalg.norm(nview, axis=1).mean():.5f}  "
          f"|tangent| mean {np.linalg.norm(tan, axis=1).mean():.5f}")
    print(f"  triangles {len(tri)}, max index {int(tri.max())} of {len(pm)-1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
