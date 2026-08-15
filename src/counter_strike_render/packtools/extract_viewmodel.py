#!/usr/bin/env python3
"""Extract ANY weapon .vmdl_c into the tuple gpu_render's viewmodel pass takes.

WHY THIS EXISTS. gpu_render.py's viewmodel pass is complete and reachable
-- `weapon_adjustments` :18832 is live and no shading function in the file
is unreachable -- but its only vertex source is `vm_synth_geometry`
:18606, which fabricates boxes. Its own docstring says the fabrication is
the whole of the gap: it submits "through the SAME dispatch, the SAME
side-table columns and the SAME shading functions a packed weapon would
take; only the vertex source differs". This supplies the missing source
from the shipped asset.

The output matches `vm_synth_geometry`'s return contract exactly
(:18616): pos (N,3) view-space METRES, uv (N,4) with xy the surface UV
and zw the STICKER UV set, nrm (N,3), tan (N,3), loc (N,3) model-local in
SOURCE UNITS, tri (T,3) int32, fam (T,) long, is_legs (N,) bool.

`loc` is kept in source units on purpose: csgo_legs_prepass's cone test is
anchored at (0,0,55) source units and csgo_weapon's SFX phase adds
localPos.x*0.01, so metres there would make those terms degenerate rather
than visibly wrong -- the failure that does not look like one.

CONTAINER NOTES, read rather than inferred:
  * A weapon .vmdl_c has NO MBUF block. It carries MVTX/MIDX pairs and
    CTRL.embedded_meshes, unlike the world models the pipeline was built
    on. `read_model` handles both; only its MDAT parse gets in the way.
  * Its MDAT block is REFUSED by kv3v5's strict check ("v5 strings left 4
    bytes of the auxiliary 1-byte buffer unread") and parses clean at
    strict=False. The mesh -> material join lives in MDAT and is now READ
    through it; see vm_slots.py. The strict residual stays open.
  * The model ships two meshes. `body_legacy` has TEXCOORD_0 running to
    [2.01, 3.41] (tiled); `body_hd` is a clean [0.003, 0.997]. HD is the
    one to use and the name is checked, not the index.

Usage:
  python extract_viewmodel.py --vmdl weapons/models/galilar/weapon_rif_galilar.vmdl_c \
      --game <depot game/> --name galil_ar --out-dir <dir>
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
import kv3v5                     # noqa: E402
import mesh_skin                # noqa: E402
from mesh_skin import add_skin  # noqa: E402
from vm_slots import TEX_EDGE, resolve as resolve_slots   # noqa: E402

S = 0.0254
# THE viewmodel_offset CVARS ARE ANOTHER CAPTURE'S USER SETTING, and they are
# STATED ZERO here rather than carried. This constant used to hold gtvm's
# (2.5, 0, -1.5) source units -- one player's viewmodel_offset_x/y/z, read off
# the gtvm capture. It is a real engine input and it is additive on top of the
# placement, but it belongs to the PLAYER, and the cs1k corpus frames this
# bundle is scored against were recorded by other players whose cvars are not
# in the corpus. CS2's own default for all three is 0, so zero is the value
# that is not an assumption about someone else's config. Pass --cvar-offset to
# state a different one when pairing against a capture whose cvars ARE known.
CVAR_OFFSET_DEFAULT = (0.0, 0.0, 0.0)
# THE FORWARD PUSH IS RETIRED. It was a stated 0.301 m stand-in, and its own
# comment named what it was standing in for: "the real placement comes from
# a viewmodel BONE ... not from viewmodel_offset_z. Until that transform is
# read". That transform is now read -- `wpn` in
# animation/skeletons/characters/viewmodel.vnmskel, the bone every entry of
# that skeleton's m_secondarySkeletons attaches a weapon to -- so the
# stand-in is deleted rather than kept beside the thing that replaced it.
# The bundle records what it used under `placement_basis`.

# PER-WEAPON pixel corrections. Only the AK has any, and both of the AK's
# were fitted against a capture; every other weapon gets zeros that are
# STATED, never measured, and print themselves that way.
#
#   MUZZLE   derived from the READ attachment projected through the
#            bundle's own transform onto GT's flash centroid -- 24 firing
#            frames, sd (24.8, 24.1) px, intensity-weighted centroid.
#            Offset at --viewmodel-hfov 68 is (+143.5, +48.5) px at 1080p,
#            computed independently by decal-vfx and by me to the same
#            figures. NO ASSUMED DEPTH: the attachment carries z = 0.9904 m.
#   LANDMARK a silhouette extremum. COMPOSED WITH, not superseded by, the
#            muzzle term: the muzzle offset was measured against the bundle
#            with the landmark correction already in it, so it is a
#            correction FROM that state. Zeroing it moved the weapon the
#            wrong way and left a +59.0 px vertical residual where x had
#            landed to 2.3 px -- what a compose/replace mix-up looks like.
#            It disagrees with the muzzle term in x by 0.10 m, which is what
#            a silhouette extremum should do: sliding along the barrel axis
#            barely changes the outline.
DXY_FITTED = {
    "ak_47": (np.array([0.0998, -0.0338, 0.0]),        # muzzle
              np.array([-0.0032, -0.0822, 0.0])),      # landmark
}

# The muzzle point is READ, per weapon, and no longer a table. It is NOT in
# the .vmdl_c -- that container has no attachment block and 0 byte
# occurrences of "muzzle" across DATA/ASEQ/AGRP/CTRL. It is a BONE in the
# animation skeleton the model names in DATA.m_vecNmSkeletonRefs:
# `animation/skeletons/weapons/<w>.vnmskel`, bone "muzzle".
#
# The AK's shipped constant was [22.671567916870117, 0.2106740027666092,
# -0.6492080092430115]; that skeleton's muzzle bone reads
# (22.671568, 0.210674, -0.649207), i.e. the same point to 4e-07 source
# units. So the hardcoded value is retired rather than trusted, and the
# galil gets its own (18.737152, 0.200659, -0.145835) by the same route.
MUZZLE_BONE = "muzzle"


def to_view(v):
    return np.stack([-v[:, 1], v[:, 2], -v[:, 0]], axis=1)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _qrot(q, v):
    x, y, z, w = q
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def nm_skeleton(content, path):
    """{bone: (translation, quaternion)} model-space, from a .vnmskel_c.

    `m_modelSpaceReferencePose` rows are 8 floats laid out
    [tx, ty, tz, scale, qx, qy, qz, qw]. That layout is CHECKED rather than
    assumed -- the caller asserts every row's last four components are unit
    norm, which they are to 1e-6 across both weapons and the 56-bone
    viewmodel skeleton, and which the [qx,qy,qz,qw,tx,ty,tz,scale] reading
    fails outright.
    """
    import kv3v5
    d = content.find(path)
    if d is None:
        return None, [f"  {path}: not in the content mount"]
    by = {}
    for t, b in A.ordered_blocks(d):
        by.setdefault(t, b)
    m = kv3v5.parse(by["DATA"])
    out, worst = {}, 0.0
    for nm, row in zip(m["m_boneIDs"], m["m_modelSpaceReferencePose"]):
        r = np.asarray(row, np.float64)
        q = r[4:8]
        worst = max(worst, abs(float(np.linalg.norm(q)) - 1.0))
        out[nm] = (r[:3], q)
    lines = [f"  {path}: {len(out)} bones, "
             f"max ||q|-1| {worst:.2e} over the reference pose "
             f"-- the [t,scale,q] field order is CHECKED"]
    if worst > 1e-5:
        lines.append("    REFUSING the layout: quaternions are not unit, so "
                     "the field order read here is wrong")
        return None, lines
    return out, lines


def bind_translation(skel, bone="weapon_offset"):
    """The bone's MODEL-SPACE position, walked up m_nParent. READ.

    Returns (translation, max_rotation_deviation, lines). This is the BIND
    POSE translation, positive, and the bundle SUBTRACTS it. The shipped AK
    constant stored the negation and subtracted that, i.e. it added the
    bind; the sign correction and its size are printed at extraction.
    """
    names = list(skel["m_boneName"])
    parent = list(skel["m_nParent"])
    pos = [np.asarray(p, np.float64) for p in skel["m_bonePosParent"]]
    rot = [np.asarray(r, np.float64) for r in skel["m_boneRotParent"]]
    if bone not in names:
        return None, None, [f"  bone {bone!r} ABSENT; skeleton has {names}"]
    i = names.index(bone)
    chain = []
    while i >= 0:
        chain.append(i)
        i = parent[i]
    chain.reverse()
    t = np.zeros(3)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    dev = 0.0
    for k in chain:
        t = t + _qrot(q, pos[k])
        q = _qmul(q, rot[k])
        dev = max(dev, float(np.abs(rot[k][:3]).max()))
    lines = [f"  bind bone {bone!r} chain "
             f"{' -> '.join(names[k] for k in chain)}",
             f"    model-space bind translation {np.round(t, 6).tolist()} "
             f"source units",
             f"    max |quaternion xyz| along the chain {dev:.3e} "
             f"-- rotation is identity to that"]
    return t, dev, lines


def _static_track(t):
    """(translation, quaternion, scale) for a track marked static.

    THE FORMAT CLAIM, and it is PROVEN below rather than asserted: a track
    whose m_bIsTranslationStatic is set carries that translation in the
    RANGE STARTS of m_translationRangeX/Y/Z, and its rotation in
    m_constantRotation. The proof is that the galil idle clip's 8 secondary
    tracks reproduce animation/skeletons/weapons/galil.vnmskel's own bone
    translations to max|d| 1.2e-05 source units on 7 of 8 bones -- the
    exception being `bolt`, which the idle pose moves 0.2737 units forward,
    i.e. exactly the one bone an idle pose has any business moving.
    """
    return (np.array([t["m_translationRangeX"]["m_flRangeStart"],
                      t["m_translationRangeY"]["m_flRangeStart"],
                      t["m_translationRangeZ"]["m_flRangeStart"]]),
            np.asarray(t["m_constantRotation"], np.float64),
            t["m_scaleRange"]["m_flRangeStart"])


def find_idle_clip(content, index, weapon_skel_path):
    """The idle clip for THIS weapon, joined by skeleton, not by name.

    Name-matching does not work and the failure is silent: the galil's clip
    folder is `rifle_galilar/idle_galilar` and the AK's is `rifle_ak/idle_ak`,
    neither derivable from `weapons/models/ak47/`. Every viewmodel idle clip
    names the weapon skeleton it animates in
    m_secondaryAnimations[].m_skeleton, so that field is the join and the
    match is exact.
    """
    import kv3v5
    hits = []
    for k in sorted(index):
        if not k.endswith(".vnmclip_c"):
            continue
        if "/viewmodel/" not in k or "idle" not in os.path.basename(k):
            continue
        raw = content.find(k[:-2])
        if raw is None:
            continue
        by = {}
        for t, b in A.ordered_blocks(raw):
            by.setdefault(t, b)
        try:
            c = kv3v5.parse(by["DATA"])
        except Exception:                                       # noqa: BLE001
            continue
        for s in c.get("m_secondaryAnimations") or []:
            if s.get("m_skeleton") == weapon_skel_path:
                hits.append((k[:-2], c))
                break
    # PREFER THE UNNUMBERED IDLE. A knife ships `idle_<x>` and `idle2_<x>`,
    # two different held poses, and sorted order puts `idle2_` FIRST because
    # '2' sorts before '_'. Taking hits[0] therefore picked the second idle
    # by alphabetical accident -- the same shape as taking draw call 0. The
    # unnumbered name is the base pose; the alternatives are printed by the
    # caller either way.
    hits.sort(key=lambda h: (0 if os.path.basename(h[0]).startswith("idle_")
                             else 1, h[0]))
    return hits


def hold_pose(content, index, weapon_skel_path, vm_bone_names):
    """The posed `wpn` transform for the first-person hold, READ.

    Returns (translation, quaternion, lines) or (None, None, lines).
    """
    lines = []
    if not weapon_skel_path or not index:
        return None, None, ["  hold pose: no weapon skeleton or no archive "
                            "index -- NOT READ"]
    hits = find_idle_clip(content, index, weapon_skel_path)
    if not hits:
        return None, None, [f"  hold pose: no viewmodel idle clip names "
                            f"{weapon_skel_path} in m_secondaryAnimations "
                            f"-- this weapon ships none in this depot"]
    path, c = hits[0]
    names = [p for p, _ in hits]
    tracks = c.get("m_trackCompressionSettings") or []
    lines.append(f"  hold pose: {len(hits)} idle clip(s) name this weapon; "
                 f"using {path}")
    if len(hits) > 1:
        lines.append(f"    others, NOT used: {names[1:]}")
    lines.append(f"    {c['m_nNumFrames']} frame(s), duration "
                 f"{c['m_flDuration']}, {len(tracks)} primary tracks vs "
                 f"{len(vm_bone_names)} viewmodel bones, "
                 f"m_compressedPoseData {len(c.get('m_compressedPoseData') or b'')} "
                 f"bytes")
    if len(tracks) != len(vm_bone_names):
        lines.append("    REFUSED: track count does not match the viewmodel "
                     "skeleton, so track i is not bone i")
        return None, None, lines
    i = vm_bone_names.index("wpn")
    t = tracks[i]
    if not (t.get("m_bIsTranslationStatic") and t.get("m_bIsRotationStatic")):
        # ANIMATED. Decoded from the compressed stream at FRAME 0, and the
        # frame choice is STATED rather than implied: a 61-frame idle has no
        # single hold pose, and which phase the GT frame caught is not on the
        # wire, so frame 0 is a stated choice like the glove model is.
        import vnmclip
        ok, clines = vnmclip.check_layout(c)
        lines.extend(clines)
        if not ok:
            lines.append("    container check FAILED -- refusing to decode")
            return None, None, lines
        d = vnmclip.decode_track(c, i, 0)
        tr, q = d["translation"], d["rotation"]
        lines.append(f"    `wpn` track is ANIMATED over "
                     f"{c['m_nNumFrames']} frames; DECODED at frame 0 "
                     f"(stated choice): t {np.round(tr, 6).tolist()}, "
                     f"q {np.round(q, 6).tolist()}, "
                     f"||q|-1| {abs(float(np.linalg.norm(q)) - 1.0):.2e}")
        return tr, q, lines
    tr, q, sc = _static_track(t)
    lines.append(f"    `wpn` posed: t {np.round(tr, 6).tolist()} source "
                 f"units, q {np.round(q, 6).tolist()} "
                 f"(|xyz| {float(np.abs(q[:3]).max()):.2e}, "
                 f"{np.degrees(2 * np.arcsin(min(1.0, float(np.linalg.norm(q[:3]))))):.4f} deg "
                 f"from identity), scale {sc}")
    return tr, q, lines


def arms_rig_placement(content, weapon_skel_path, vm_skel, bind_applied,
                       name=None, hold=None, off=None):
    """What the arms rig says the placement is, and what that leaves over.

    THE JOIN, READ, not modelled:
      animation/skeletons/characters/viewmodel.vnmskel carries a bone `wpn`
      and a list `m_secondarySkeletons`, each entry
      {m_attachToBoneID, m_skeleton}. Every one of the 13 entries attaches
      at `wpn`, and the weapon's own animation skeleton is one of them. So
      the weapon's root sits at the viewmodel skeleton's `wpn` bone, and the
      viewmodel skeleton's root is root_motion -- the camera.

    That fixes the placement up to the POSE. The reference pose is what this
    reads; the hold pose is a clip, and this prints which one it used.
    """
    import kv3v5
    out = []
    if vm_skel is None or "wpn" not in vm_skel:
        return None, out + ["  arms rig: viewmodel.vnmskel unreadable or "
                            "has no `wpn` bone -- NOT READ"]
    d = content.find("animation/skeletons/characters/viewmodel.vnmskel")
    by = {}
    for t, b in A.ordered_blocks(d):
        by.setdefault(t, b)
    sec = kv3v5.parse(by["DATA"]).get("m_secondarySkeletons") or []
    attach = sorted({s.get("m_attachToBoneID") for s in sec})
    listed = [s for s in sec if s.get("m_skeleton") == weapon_skel_path]
    out.append(f"  arms rig: viewmodel.vnmskel m_secondarySkeletons "
               f"{len(sec)} entries, all attach at {attach}")
    # That list is PARTIAL and saying so matters: it holds 13 of the 41
    # weapon skeletons, and which 13 does not track which weapons have
    # viewmodel clips -- ak47 is in the list and ships no viewmodel clip in
    # this depot, galil is absent from it and ships a full set. The
    # per-weapon attach that is authoritative is the CLIP's own
    # m_secondaryAnimations[].m_skeleton, printed on the HOLD POSE line.
    out.append(f"    this weapon's {weapon_skel_path} is "
               f"{'LISTED' if listed else 'NOT listed'} among them -- the "
               f"list is PARTIAL, so absence is not evidence; the clip's "
               f"m_secondaryAnimations is the authoritative join")

    t_ref, q_ref = vm_skel["wpn"]
    out.append(f"    bone `wpn` reference pose: t "
               f"{np.round(t_ref, 6).tolist()} source units, "
               f"|quaternion xyz| {float(np.abs(q_ref[:3]).max()):.2e} "
               f"(identity to that)")
    if hold is not None:
        t_wpn = np.asarray(hold, np.float64)
        out.append(f"    USING THE HOLD POSE, not the reference pose: "
                   f"t {np.round(t_wpn, 6).tolist()}, which moves the weapon "
                   f"{np.round(t_wpn - t_ref, 4).tolist()} source units "
                   f"({np.linalg.norm(t_wpn - t_ref) * S:.4f} m) from bind")
    else:
        t_wpn = np.asarray(t_ref, np.float64)
        out.append("    NO hold pose read -- falling back to the REFERENCE "
                   "pose, which is the bind, not what the player sees")

    # THE FALSIFIER, run and printed rather than described. The AK's two
    # fitted pixel vectors were measured against the RETIRED placement --
    # bind translation ADDED (the shipped constant stored its negation and
    # subtracted) plus a stated 0.301 m forward push. The read placement
    # SUBTRACTS the bind, because the weapon's animation skeleton puts
    # `weapon` at the origin and the viewmodel skeleton puts that root at
    # `wpn`, and it replaces the push with `wpn` itself. The difference
    # between the two is the number below; it is what a re-fit of the AK
    # would have to move by, and it is why the fitted vectors are carried
    # forward marked CONTAMINATED rather than trusted.
    read_view = to_view(np.asarray(t_wpn)[None, :])[0] * S
    retired_push = np.array([0.0, 0.0, -0.301])
    two_bind = to_view((2.0 * bind_applied)[None, :])[0] * S
    delta = read_view - retired_push + two_bind
    muzzle_dxy, landmark_dxy = DXY_FITTED.get(
        name, (np.zeros(3), np.zeros(3)))
    OFF = np.zeros(3) if off is None else np.asarray(off, np.float64)
    out.append(f"    READ placement: cvar offset "
               f"{np.round(OFF, 4).tolist()} + wpn "
               f"{np.round(read_view, 4).tolist()} m, bind SUBTRACTED")
    out.append(f"    RETIRED placement: gtvm offset + stated push "
               f"{np.round(retired_push, 4).tolist()} m, bind ADDED")
    out.append(f"    read - retired = {np.round(delta, 4).tolist()} m, "
               f"max|d| {np.abs(delta).max():.4f} m, "
               f"norm {np.linalg.norm(delta):.4f} m")
    if np.any(muzzle_dxy) or np.any(landmark_dxy):
        dxy = muzzle_dxy + landmark_dxy
        out.append(f"    VERDICT: the read does NOT reproduce the fitted "
                   f"DXY {np.round(dxy, 4).tolist()} m -- residual "
                   f"{np.round(delta - dxy, 4).tolist()} m, max|d| "
                   f"{np.abs(delta - dxy).max():.4f} m. The fit was taken "
                   f"against the retired placement, so it is CONTAMINATED "
                   f"by that displacement and needs re-fitting against a "
                   f"capture, not adjusting here.")
    else:
        out.append(f"    VERDICT: no fitted DXY for this weapon, so there "
                   f"is nothing to reproduce; the placement is entirely "
                   f"read + the gtvm cvar offset.")
    return t_wpn, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vmdl", required=True,
                    help="weapons/models/<dir>/weapon_*.vmdl_c")
    ap.add_argument("--game", required=True, help="depot game/ root")
    ap.add_argument("--map-vpk", default=None,
                    help="defaults to <game>/csgo/pak01_dir.vpk")
    ap.add_argument("--mesh", default="body_hd")
    ap.add_argument("--name", required=True,
                    help="cs1k enum name; the bundle is <name>.pt so "
                         "pair_compare --vm-bundles finds it")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--edge", type=int, default=TEX_EDGE)
    ap.add_argument("--cvar-offset", type=float, nargs=3,
                    default=CVAR_OFFSET_DEFAULT,
                    metavar=("X", "Y", "Z"),
                    help="viewmodel_offset_x/y/z in SOURCE UNITS. Default 0, "
                         "which is CS2's own default and the only value that "
                         "is not an assumption about the recording player's "
                         "config.")
    a = ap.parse_args(argv)

    OFF = np.asarray(a.cvar_offset, np.float64) * S
    print(f"  viewmodel_offset cvars {list(a.cvar_offset)} source units -> "
          f"{np.round(OFF, 4).tolist()} m  ["
          + ("STATED zero (CS2 default; the corpus does not carry the "
             "recording player's cvars)" if not any(a.cvar_offset)
             else "STATED from the command line") + "]")
    vpk = a.map_vpk or os.path.join(a.game, "csgo", "pak01_dir.vpk")
    content = A.Content(vpk, a.game, "pak01")
    d = content.find(a.vmdl)
    if d is None:
        raise SystemExit(f"{a.vmdl}: not in the content mount")
    blocks = A.ordered_blocks(d)
    by = {}
    for t, b in blocks:
        by.setdefault(t, b)
    ctrl = kv3v5.parse(by["CTRL"])
    mo = A.load_meshopt()

    print(f"{a.name}  {a.vmdl}")
    print(f"  blocks {[t for t, _ in blocks]}")

    # ---- THE CLAIM TO FALSIFY, printed before anything is fitted --------
    # The weapon file is asked, by name and by field, whether it holds a
    # first-person hold pose. It is ASEQ.m_localSequenceNameArray, and it
    # is printed whatever it says.
    aseq = kv3v5.parse(by["ASEQ"], strict=False)
    seqs = list(aseq.get("m_localSequenceNameArray") or [])
    print(f"  ASEQ.m_localSequenceNameArray ({len(seqs)}): {seqs}")
    hold = [s for s in seqs
            if any(k in s.lower() for k in ("idle", "hold", "deploy", "look"))]
    verdict = str(hold) if hold else (
        "NONE -- the hold pose is not in the weapon file; it is on the arms "
        "side or in the animgraph")
    print(f"    first-person hold/idle/deploy sequence in THIS file: {verdict}")
    print(f"    'muzzle' byte occurrences in DATA/ASEQ/AGRP/CTRL: "
          + ", ".join(f"{t}={by[t].lower().count(b'muzzle')}"
                      for t in ("DATA", "ASEQ", "AGRP", "CTRL")))

    # ---- geometry -------------------------------------------------------
    have = [m.get("m_Name") for m in ctrl["embedded_meshes"]]
    emb = [m for m in ctrl["embedded_meshes"] if m.get("m_Name") == a.mesh]
    if not emb and len(ctrl["embedded_meshes"]) == 1:
        # NOT EVERY WEAPON SHIPS TWO MESHES. The rifles ship body_legacy +
        # body_hd and the HD one is the one to draw; the default knives ship
        # `body_legacy` ALONE. Refusing there would report "no mesh body_hd"
        # for a model whose only mesh is the right one, which is a shape
        # assumption failing rather than an asset problem -- the same class
        # as the knife's two-level directory. One mesh is unambiguous, so it
        # is taken and the substitution is printed.
        emb = ctrl["embedded_meshes"]
        a.mesh = have[0]
        print(f"  mesh: {a.mesh!r} -- this model ships ONE mesh and no "
              f"'body_hd', so there is nothing to choose between")
    if not emb:
        raise SystemExit(f"{a.vmdl}: no mesh {a.mesh!r}; has {have}")
    emb = emb[0]
    vb = emb["m_vertexBuffers"][0]
    attrs, _r, err = A.decode_vertex_buffer(np, mo, vb, blocks)
    assert not err, err
    idx, err = A.decode_index_buffer(np, mo, emb["m_indexBuffers"][0], blocks)
    assert not err, err
    if len(emb["m_vertexBuffers"]) > 1:
        print(f"  NOTE: mesh has {len(emb['m_vertexBuffers'])} vertex "
              f"buffers; buffer 0 is taken and the rest are NOT merged")

    pm = np.asarray(attrs["POSITION_0"], np.float64)          # source units
    nm = np.asarray(attrs["NORMAL_0_decoded"], np.float64)
    uv0 = np.asarray(attrs["TEXCOORD_0"], np.float64)
    if "TEXCOORD_1" in attrs:
        uv1 = np.asarray(attrs["TEXCOORD_1"], np.float64)
    else:
        # THE STICKER UV SET IS ABSENT ON SOME WEAPONS. The contract puts it
        # in uv.zw and the default knives carry only TEXCOORD_0 -- a knife
        # takes no stickers, so there is no second set to read. uv.zw is a
        # COPY of uv.xy here and it is stated rather than left to look like
        # a second set that happens to agree.
        uv1 = uv0.copy()
        print("  TEXCOORD_1 ABSENT: uv.zw is a COPY of uv.xy. This model has "
              "no sticker UV set, so nothing downstream should read zw")
    N = len(pm)

    dat = kv3v5.parse(by["DATA"], strict=False)
    # PER-VERTEX BONE BINDINGS. The weapon mesh is placed rigidly by `wpn`,
    # which is right for where the gun is but makes every moving part -- the
    # bolt, the slide, the hammer -- unable to move, because no vertex
    # declares which bone owns it. The bindings are read here and shipped
    # beside the rigid placement; nothing about the existing placement
    # changes.
    _mskel = dat["m_modelSkeleton"]
    skin, sklines = mesh_skin.read_skin(
        attrs, emb, dat, list(_mskel["m_boneName"]),
        label=f"{a.vmdl}::{a.mesh}")
    for ln in sklines:
        print(ln)
    if skin is not None:
        print("  skin: moving parts -> %s"
              % mesh_skin.moving_part_report(
                  skin, ("bolt", "slide", "hammer", "trigger", "pump",
                         "chamber")))
    bind_t, _dev, blines = bind_translation(dat["m_modelSkeleton"])
    for ln in blines:
        print(ln)
    if bind_t is None:
        bind_t = np.zeros(3)
        print("    STATED zero bind translation (bone absent)")
    # SUBTRACTED, and that is a sign correction. The shipped AK code stored
    # the NEGATION of this translation and subtracted that, i.e. it ADDED the
    # bind. The weapon's own animation skeleton
    # (animation/skeletons/weapons/<w>.vnmskel) puts `weapon` and
    # `weapon_offset` at the ORIGIN, so expressing the mesh in that skeleton's
    # frame subtracts the .vmdl_c bind translation. The difference is 2x this
    # vector and it is printed in the arms-rig block below.
    bind_applied = np.asarray(bind_t, np.float64)
    print(f"    APPLIED as pos - ({np.round(bind_applied, 6).tolist()}) "
          f"-- SUBTRACTED (the shipped AK code added it)")

    # ---- THE ARMS RIG, read and checked against the fit ------------------
    nm_refs = list(dat.get("m_vecNmSkeletonRefs") or [])
    print(f"  DATA.m_vecNmSkeletonRefs: {nm_refs}")
    wskel, wlines = (nm_skeleton(content, nm_refs[0]) if nm_refs
                     else (None, ["  no m_vecNmSkeletonRefs -- no animation "
                                  "skeleton to read"]))
    for ln in wlines:
        print(ln)
    vm_skel, vmlines = nm_skeleton(
        content, "animation/skeletons/characters/viewmodel.vnmskel")
    for ln in vmlines:
        print(ln)
    import weapon_ids as _wid
    index = _wid.vpk_index(a.game)
    hold_t, hold_q, hlines = hold_pose(
        content, index, nm_refs[0] if nm_refs else None,
        list(vm_skel or {}))
    for ln in hlines:
        print(ln)
    t_wpn, arms = arms_rig_placement(
        content, nm_refs[0] if nm_refs else None, vm_skel, bind_applied,
        name=a.name, hold=hold_t, off=OFF)
    for ln in arms:
        print(ln)
    if t_wpn is None:
        raise SystemExit("the `wpn` bone is the placement; without it this "
                         "bundle would carry a stand-in, and the stand-in "
                         "it replaced has been deleted")

    if a.name in DXY_FITTED:
        muzzle_dxy, landmark_dxy = DXY_FITTED[a.name]
        dxy_tag = "FITTED"
        dxy_why = ("muzzle-centroid over 24 firing frames + a silhouette "
                   "extremum, composed")
    else:
        muzzle_dxy = np.zeros(3)
        landmark_dxy = np.zeros(3)
        dxy_tag = "STATED"
        dxy_why = ("no capture has been fitted for this weapon; zeros are "
                   "stated, not measured -- an unregistered silhouette is "
                   "not a band pass")
    print(f"  DXY {dxy_tag}: muzzle {muzzle_dxy.tolist()} landmark "
          f"{landmark_dxy.tolist()} -- {dxy_why}")

    # The posed `wpn` carries a ROTATION as well as a translation and both
    # are applied. It is small -- 0.08 deg for the galil, 2.21 for the AK --
    # and reading it while dropping it would be the same class of miss as
    # not reading it.
    local = pm - bind_applied
    if hold_q is not None:
        local = np.stack([_qrot(hold_q, v) for v in local])
    pos = to_view(local + t_wpn) * S + OFF
    pos += landmark_dxy + muzzle_dxy
    if hold_q is not None:
        nm = np.stack([_qrot(hold_q, v) for v in nm])
    nrm = to_view(nm)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    # tangent: the buffer carries a rotation + bitangent sign, not a vector.
    # Rather than reconstruct it wrongly, derive a per-vertex tangent from the
    # UV gradient over the triangles that use it -- the same construction
    # apply_normal_map() Gram-Schmidts against the normal anyway.
    tri = idx.reshape(-1, 3).astype(np.int64)
    tan = np.zeros((N, 3))
    e1 = pos[tri[:, 1]] - pos[tri[:, 0]]
    e2 = pos[tri[:, 2]] - pos[tri[:, 0]]
    d1 = uv0[tri[:, 1]] - uv0[tri[:, 0]]
    d2 = uv0[tri[:, 2]] - uv0[tri[:, 0]]
    den = (d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / den[:, None]
    for k in range(3):
        np.add.at(tan, tri[:, k], t)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)

    # --- TEXTURE PAGES, with per-slot provenance -------------------------
    # The resolution is a STATED PARAMETER, not a constant. TEXTURE_EDGE = 64
    # became a problem precisely because it was a defensible choice that
    # stopped being visible; this one is recorded next to the pages so a
    # consumer reads it instead of finding it by reading the decode.
    slots, unmapped, slines = resolve_slots(content, a.vmdl, a.mesh)
    for ln in slines:
        print(ln)

    pages, prov = {}, {}
    for slot, vpath in slots.items():
        raw = content.find(vpath)
        if raw is None:
            prov[slot] = "MISSING: " + vpath
            print(f"  slot {slot}: {prov[slot]}")
            continue
        tmp = os.path.join(a.out_dir, ".vm_slot.vtex_c")
        os.makedirs(a.out_dir, exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(raw)
        tnm, w, h, blk, fmtnum = _a2w.read_vtex(tmp, want=a.edge,
                                                allow_lz4=True)
        os.remove(tmp)
        if blk is None:
            prov[slot] = f"UNDECODED ({tnm}): " + vpath
            print(f"  slot {slot}: {prov[slot]}")
            continue
        px = _bc.decode(blk, fmtnum, w, h)
        pages[slot] = torch.tensor(px.copy())
        prov[slot] = (f"REAL {vpath} :: {tnm} mip@{w}x{h} "
                      f"-> page {px.shape[1]}x{px.shape[0]}")
        print(f"  slot {slot}: {prov[slot]}")

    # Real pages the material names that this bundle does NOT bind. Printed,
    # and recorded in the bundle's provenance, so "the asset has it and we do
    # not use it" is a visible state rather than an absence indistinguishable
    # from "the asset does not have it".
    for slot, (vpath, why) in unmapped.items():
        present = content.find(vpath) is not None
        prov[slot] = (f"UNMAPPED (page {'PRESENT' if present else 'ABSENT'} "
                      f"in the archives): {vpath} -- {why}")
        print(f"  slot {slot}: {prov[slot]}")

    bundle = {
        "pos": torch.tensor(pos, dtype=torch.float32),
        "uv": torch.tensor(np.concatenate([uv0, uv1], axis=1),
                           dtype=torch.float32),
        "nrm": torch.tensor(nrm, dtype=torch.float32),
        "tan": torch.tensor(tan, dtype=torch.float32),
        "loc": torch.tensor(pm, dtype=torch.float32),          # SOURCE UNITS
        "tri": torch.tensor(tri, dtype=torch.int32),
        "fam": torch.zeros(len(tri), dtype=torch.long),        # 0 = weapon
        "is_legs": torch.zeros(N, dtype=torch.bool),
        "source": f"{a.vmdl} :: {a.mesh}",
        "weapon": a.name,
        "offset_m": OFF.tolist(),
        "offset_basis": ("READ from a capture" if any(a.cvar_offset)
                         else "STATED zero -- viewmodel_offset_x/y/z are the "
                              "recording player's cvars and the cs1k corpus "
                              "does not carry them"),
        "bind_translation_src": np.asarray(bind_t).tolist(),
        "bind_applied_src": np.asarray(bind_applied).tolist(),
        "nm_skeleton": nm_refs[0] if nm_refs else None,
        "axis_map": "view = (-Ym, +Zm, -Xm) * 0.0254 + offset, after "
                    "subtracting the weapon_offset bind translation and "
                    "adding the viewmodel skeleton's `wpn` bone",
        "wpn_bone_src": np.asarray(t_wpn).tolist(),
        # THE TWO CHAINS, STATED, because they differ in one term and a
        # consumer cannot tell from the numbers alone. `bind_applied_src` is
        # live and correct for the MESH -- recomputing pos from loc with it
        # agrees to 6e-08 m -- and must NOT be applied to the muzzle, whose
        # source is already in the .vnmskel frame. A reader that assumes one
        # chain for both silently gets the superseded muzzle point, which is
        # exactly what happened after the frame fix landed.
        "mesh_bind_subtracted": True,
        "muzzle_bind_subtracted": False,
        "mesh_view_chain": "to_view(qrot(wpn_bone_quat, loc - "
                           "bind_applied_src) + wpn_bone_src) * 0.0254 "
                           "+ offset_m + dxy_muzzle + dxy_landmark",
        "muzzle_view_chain": "to_view(qrot(wpn_bone_quat, muzzle_flash_src) "
                             "+ wpn_bone_src) * 0.0254 + offset_m "
                             "+ dxy_muzzle + dxy_landmark   -- NOTE "
                             "bind_applied_src is NOT subtracted here",
        "wpn_bone_quat": (np.asarray(hold_q).tolist()
                          if hold_q is not None else None),
        # The string says WHICH pose was used, because it is printed on
        # every frame line. It used to read "(reference pose)"
        # unconditionally and appear beside "pose idle clip (hold pose)" in
        # the same line -- two of this bundle's own fields contradicting
        # each other in one sentence.
        "placement_basis": (
            "READ: animation/skeletons/characters/viewmodel.vnmskel bone "
            "`wpn` (" + ("posed by the weapon's idle clip"
                         if hold_t is not None
                         else "REFERENCE pose -- no idle clip was read, so "
                              "the rotation is identity")
            + ") + the viewmodel_offset cvars. The stated 0.301 m forward "
              "push is RETIRED."),
        "wpn_pose_basis": ("idle clip (hold pose)" if hold_t is not None
                           else "reference pose (no idle clip read)"),
        # A CONSUMER MUST BE ABLE TO REFUSE THIS BUNDLE. When the hold pose
        # is not read the fallback is the REFERENCE pose, whose rotation is
        # identity -- fine for a rifle, whose model-local long axis is +X and
        # therefore already points forward, and wrong for anything whose is
        # not. The default knives are exactly that case: model-local extent
        # (3.50, 1.09, 15.18) source units against the galil's
        # (30.81, 2.81, 10.81), so the blade lies along Z and the identity
        # rotation stands it on end. Drawn, it is a 71 px full-height sliver
        # at frame centre -- 0.0397 of frame, lower-right share 0.339 -- which
        # scores as noise rather than as a miss.
        "hold_pose_read": hold_t is not None,
        "long_axis_model": int(np.argmax(pm.max(0) - pm.min(0))),
        "dxy_basis": dxy_tag,
        "dxy_muzzle": muzzle_dxy.tolist(),
        "dxy_landmark": landmark_dxy.tolist(),
        "sequences": seqs,
        "tex_pages": pages,
        "tex_provenance": prov,
        "tex_edge": a.edge,
    }
    add_skin(bundle, skin, _mskel, N)
    src = (wskel or {}).get(MUZZLE_BONE)
    if src is None:
        bundle["muzzle_flash_src"] = None
        bundle["muzzle_flash_view"] = None
        bundle["muzzle_flash_note"] = (
            f"ABSENT: no {MUZZLE_BONE!r} bone in "
            f"{nm_refs[0] if nm_refs else '(no m_vecNmSkeletonRefs)'}")
        print(f"  muzzle point: {bundle['muzzle_flash_note']}")
    else:
        src = np.asarray(src[0], np.float64).tolist()
        bundle["muzzle_flash_src"] = src
        bundle["muzzle_flash_note"] = (
            f"READ: bone {MUZZLE_BONE!r} of {nm_refs[0]}")
        # The SAME point, already carried through the SAME transform as the
        # vertices, so a consumer attaches to a POINT rather than re-deriving
        # a formula. Anything that fixes the placement -- a posed `wpn`
        # instead of the reference one, a different bind handling -- moves
        # the vertices and this point together.
        #
        # STILL IN THE VIEWMODEL PASS'S PRE-vm_transform SPACE: gpu_render
        # applies vm_transform(pos, is_legs) to the vertices at the top of
        # viewmodel_pass. A consumer must put this point through
        # vm_transform too, or the flash will sit still while the weapon
        # moves under those flags.
        # THE BIND IS NOT SUBTRACTED FROM THE MUZZLE, and that is a fix.
        # The mesh comes from the .vmdl_c, whose `weapon` bone sits at the
        # bind translation, so its vertices need it removed. The muzzle
        # comes from the .vnmskel, whose `weapon` bone is at the ORIGIN --
        # it is ALREADY in that frame. Subtracting the bind a second time
        # displaced every muzzle by exactly -bind.
        #
        # MEASURED over all 33 weapons that have a muzzle bone, as distance
        # from the weapon's own principal axis in units of its 95th-pct
        # barrel radius, and as position along that axis:
        #     subtracting the bind   perp/R median 1.08, mean 1.26, max 3.38
        #                            5 of 33 beyond 2R, and the point lands
        #                            MID-BODY (along 0.14 .. 0.65)
        #     not subtracting it     perp/R median 0.18, mean 0.24, max 0.91
        #                            0 of 33 beyond 2R, and the point lands
        #                            at an END of the axis (along ~0 or ~1)
        # A muzzle on the barrel axis at the barrel tip is what a muzzle is.
        #
        # THIS WAS NEVER SIX OUTLIERS. It is every weapon, displaced by its
        # own |bind|, and the six that got noticed are simply the ones whose
        # bind is large enough to clear a detector threshold: m4a4 |bind|
        # 22.05 source units, scar_20 21.97, against ak_47's 4.21. The AK
        # passing acceptance was the small-bind end of one systematic error.
        _mz = np.asarray(src, np.float64)
        bundle["muzzle_flash_view"] = (
            to_view((_qrot(hold_q, _mz) if hold_q is not None
                     else _mz)[None, :] + t_wpn)[0] * S
            + OFF + landmark_dxy + muzzle_dxy).tolist()
        print(f"  muzzle point src {src} -> view "
              f"{np.round(bundle['muzzle_flash_view'], 4).tolist()}")

    # ---- THE BUNDLE MUST RE-DERIVE FROM ITS OWN RECORDED PROVENANCE ----
    # A rebuilt bundle updated `muzzle_flash_view` and left the chain
    # parameters describing the SUPERSEDED derivation. Measured across the
    # shipped set: recomputing the view point from the bundle's own
    # bind_applied_src / wpn_bone_src / wpn_bone_quat / offset_m / dxy_*
    # reproduced the OLD value, off by 0.107 m on the AK and 0.560 m on the
    # m4a4. Both numbers were in the file and they disagreed.
    #
    # That is worse than either being wrong alone. Every field here is
    # written to be READ -- the whole point of recording the chain is that a
    # consumer can re-derive rather than trust -- so a consumer that does
    # exactly what the provenance invites silently gets the answer this
    # bundle was rebuilt to stop producing. A renderer-side fix was ordered
    # on that basis and would have reintroduced the defect.
    #
    # So the invariant is checked HERE, where both halves exist: recompute
    # from the DICT, not from the local variables, because the local ones
    # agreeing proves nothing about what was written.
    def _recheck(bd):
        _src = bd.get("muzzle_flash_src")
        _view = bd.get("muzzle_flash_view")
        if _src is None or _view is None:
            return None
        # THE MUZZLE'S CHAIN, READ FROM THE BUNDLE, not assumed to be the
        # mesh's. This subtracted bind_applied_src unconditionally, which is
        # the MESH's chain -- correct for vertices, which come from the
        # .vmdl_c whose `weapon` bone sits at the bind, and wrong for the
        # muzzle, which comes from the .vnmskel whose `weapon` bone is at the
        # ORIGIN and is already in that frame. With the muzzle frame fixed,
        # this check refused all 33 weapons that have a muzzle bone and
        # passed only the 4 that do not -- a checker encoding one chain for
        # two consumers. `muzzle_bind_subtracted` says which, and older
        # bundles that predate the flag keep the old behaviour so this stays
        # a check rather than a silent reinterpretation.
        _l = np.asarray(_src, np.float64)
        if bd.get("muzzle_bind_subtracted", True):
            _l = _l - np.asarray(bd["bind_applied_src"], np.float64)
        _q = bd.get("wpn_bone_quat")
        if _q is not None:
            _l = _qrot(np.asarray(_q, np.float64), _l)
        _r = (to_view((_l + np.asarray(bd["wpn_bone_src"], np.float64))[None, :])[0]
              * S + np.asarray(bd["offset_m"], np.float64)
              + np.asarray(bd["dxy_landmark"], np.float64)
              + np.asarray(bd["dxy_muzzle"], np.float64))
        return _r, float(np.linalg.norm(_r - np.asarray(_view, np.float64)))

    _chk = _recheck(bundle)
    if _chk is not None:
        _rec, _d = _chk
        # The tolerance is float32 round-trip on a metre-scale point, not a
        # fitted slack: the bundle stores float32 and the recompute is
        # float64, so ~1e-6 m is the arithmetic floor and anything above it
        # is a real disagreement between the two halves.
        if _d > 1e-5:
            raise SystemExit(
                f"REFUSED: this bundle does not re-derive from its own "
                f"recorded provenance.\n"
                f"  muzzle_flash_view  {np.round(bundle['muzzle_flash_view'], 6).tolist()}\n"
                f"  recomputed         {np.round(_rec, 6).tolist()}\n"
                f"  disagreement       {_d:.6f} m\n"
                f"  Both are in the file. Every recorded field exists to be "
                f"READ, so a consumer that re-derives -- which is what the "
                f"provenance invites -- would get the second value while "
                f"the renderer uses the first. Fix whichever half is stale "
                f"before writing the bundle; do not ship them disagreeing.")
        print(f"  provenance self-check: the view point re-derives from the "
              f"recorded chain to {_d:.2e} m")

    # --- SELF-CONSISTENCY, at the producer ------------------------------
    # Recompute both baked results from the bundle's OWN recorded parameters
    # and refuse to write if either disagrees. The muzzle frame fix corrected
    # the baked value while leaving the recorded chain ambiguous, and nothing
    # here would have noticed; a bundle whose stated derivation does not
    # reproduce its stated result is worse than one that omits the
    # derivation, because it invites a consumer to recompute and be wrong.
    def _replay(local, subtract_bind):
        v = np.atleast_2d(np.asarray(local, np.float64))
        if subtract_bind:
            v = v - bind_applied
        if hold_q is not None:
            v = np.stack([_qrot(hold_q, x) for x in v])
        return (to_view(v + t_wpn) * S + OFF + landmark_dxy + muzzle_dxy)
    _dm = float(np.abs(_replay(bundle["loc"].numpy(), True)
                       - bundle["pos"].numpy()).max())
    print(f"  self-check mesh: recompute from the recorded chain vs baked "
          f"max|d| {_dm:.3e} m")
    if _dm > 1e-5:
        raise SystemExit(f"REFUSING to write {a.name}: the mesh does not "
                         f"recompute from its own recorded parameters "
                         f"(max|d| {_dm:.3e} m)")
    if bundle.get("muzzle_flash_view") is not None:
        _dz = float(np.abs(_replay(bundle["muzzle_flash_src"], False)[0]
                           - np.asarray(bundle["muzzle_flash_view"])).max())
        print(f"  self-check muzzle: max|d| {_dz:.3e} m")
        if _dz > 1e-9:
            raise SystemExit(f"REFUSING to write {a.name}: the muzzle does "
                             f"not recompute from its own recorded chain "
                             f"(max|d| {_dz:.3e} m)")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.name}.pt")
    torch.save(bundle, out)
    print(f"wrote {out}")
    for k in ("pos", "uv", "nrm", "tan", "loc", "tri", "fam", "is_legs"):
        v = bundle[k]
        print(f"  {k:8s} {tuple(v.shape)} {v.dtype}")
    p = bundle["pos"].numpy()
    print(f"  view-space bbox (metres): min {np.round(p.min(0), 3)}  "
          f"max {np.round(p.max(0), 3)}")
    print(f"  x>0 is right, y<0 is below eye, z<0 is forward -> "
          f"{'lower-right, in front: PLAUSIBLE' if p[:, 2].max() < 0 else 'CHECK: some geometry behind the eye'}")
    print(f"  |normal| mean "
          f"{np.linalg.norm(bundle['nrm'].numpy(), axis=1).mean():.5f}")
    print(f"  |tangent| mean "
          f"{np.linalg.norm(bundle['tan'].numpy(), axis=1).mean():.5f}")
    print(f"  triangles {len(tri)}, max index {int(tri.max())} of {N - 1}")
    real = sum(1 for v in prov.values() if v.startswith("REAL"))
    print(f"  {real} REAL pages, {len(prov)} slots with provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
