#!/usr/bin/env python3
"""Extract a CHARACTER .vmdl_c into the bundle gpu_render's playermodel pass takes.

WHY THIS EXISTS, and what it is NOT. gpu_render.py carries the whole
csgo_character shading family -- char_normal_decode, char_frame,
char_aniso_*, char_hair_lobe, char_cloth_lobe, char_retro_lobe, and the
--char-* axis flags -- and it has never had a single character VERTEX to
run them on. de_inferno's pack contains no csgo_character material and no
player geometry, so the visible enemies and teammates that occupy real
frames of the ground-truth corpus were simply ABSENT from every render
this project has scored. This supplies the vertex source; the demo supplies
where each one stands.

THE SPACE IS DIFFERENT FROM extract_viewmodel.py's AND THAT IS THE POINT.
A viewmodel is one mesh at one place in VIEW space, so that extractor bakes
the placement into `pos` and emits view-space metres. A playermodel is the
same mesh at N different places in WORLD space, once per entity per tick,
so baking a placement here would be baking one of them. This emits MODEL
space in SOURCE UNITS -- the .vmdl_c's own vertices, untranslated,
unrotated -- and the renderer applies each entity's own (X, Y, Z, yaw) off
the wire.

That difference is recorded in the bundle as `space`, and both loaders
CHECK it: gpu_render's playermodel loader refuses a view-space bundle and
vm_load_geometry refuses a model-space one. The two files have the same
eight-key shape on purpose, which is exactly why a mix-up would otherwise
be silent -- a viewmodel bundle loaded as a playermodel would render 60 cm
of rifle at each enemy's feet and never raise.

NO POSE. The vertices are the BIND POSE. Animation is not on the wire
(Valve deleted DEM_AnimationData at proto 14150), so a posed character
cannot be READ from any .dem and would have to be run through the
animgraph. The bundle says so in `pose_basis` and the renderer prints it
per frame; a T-posed enemy that announces it is bind-pose is a stated
difference, and one that does not is a silent claim to have posed it.

Usage:
  # what character models does this depot actually ship
  python extract_playermodel.py --list --game <depot game/>

  # one CT and one T -- note the tree: agents/, NOT characters/
  python extract_playermodel.py --game <depot game/> --name ct \
      --vmdl agents/models/ctm_sas/ctm_sas.vmdl_c --out-dir <dir>
  python extract_playermodel.py --game <depot game/> --name t \
      --vmdl agents/models/tm_phoenix/tm_phoenix.vmdl_c --out-dir <dir>

USE THE `agents/models/` TREE. This depot ships every one of the 92 player
models at BOTH `characters/models/<m>/<m>.vmdl_c` and
`agents/models/<m>/<m>.vmdl_c`, and they are not the same asset: the
characters/ one is a ~5 KB STUB holding a 24-vertex cube with a
default_orange material, and the agents/ one is the ~550 KB model with the
geometry, the materials and the ANIM/ASEQ/AGRP/PHYS blocks. Both load, both
extract, and only one is a player -- so --list prints each candidate's
vertex count and flags the stubs, and an extraction under
STUB_HEIGHT_UNITS is REFUSED rather than written.

The --vmdl path is REQUIRED and never guessed. CS:GO shipped players under
models/player/ and CS2 ships them under agents/ and characters/; which of
those this depot has is a fact about the depot, so --list asks the archive
index instead of this file carrying a table that rots.
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
from vm_slots import TEX_EDGE, resolve as resolve_slots   # noqa: E402

# The bundle's declared space. gpu_render checks this string exactly.
SPACE = "model-source-units"

# Substrings that make a vmdl_c path a PLAYER character rather than a prop,
# a weapon or a piece of clothing. Used ONLY by --list, which prints what it
# found and lets the operator choose -- nothing downstream reads this and no
# extraction is gated on it, so a wrong guess here costs a listing and not a
# render.
#
# `agents/models/` IS THE RENDERABLE TREE AND IT WAS MISSING HERE. This list
# first held only characters/models/ and models/player/, and that omission
# cost a bundle: READ from the depot, characters/models/ctm_sas/ctm_sas.vmdl_c
# is 4,791 bytes with ONE mesh of 24 vertices and a
# materials/pbr_defaults/default_orange001 material -- a stub. The model with
# the geometry is agents/models/ctm_sas/ctm_sas.vmdl_c at 563,655 bytes, with
# thirdperson_body / thirdperson_default_gloves / firstperson_* / defusekit
# and the ANIM, ASEQ, AGRP and PHYS blocks the stub has none of. Both paths
# exist for all 92 models, so picking the wrong tree produces a bundle that
# loads, rasterises and draws a 10-unit orange cube at every enemy.
CHARACTER_HINTS = ("agents/models/", "characters/models/", "models/player/")

# The mesh a THIRD-PERSON draw wants, by name. An agent .vmdl_c ships the
# first-person arms and sleeves in the same file, and those are the wrong
# geometry for another player -- they are what the local player's own camera
# sees. Selected by name and PRINTED as such; if the model has no mesh by
# this name the first is taken and the fallback says so.
BODY_MESH = "thirdperson_body"

# Below this height the model is a STUB, not a player. A standing CS player
# is ~72 source units (this SAS body reads 78.6) and a crouching one ~54, so
# 32 is under half of either -- it separates "a character I have not seen
# before" from "a 10-unit proxy cube", and it is the check that caught the
# characters/ tree. STATED, printed with the measured height, and escapable
# with --allow-stub rather than silently enforced.
STUB_HEIGHT_UNITS = 32.0


def cmd_list(game):
    """Every character-looking .vmdl_c in the depot, with its mesh sizes.

    The SIZES are the point, not the paths. A listing of names alone is what
    let a 24-vertex stub be chosen over the real model: both trees carry the
    same 92 names and only the vertex count tells them apart. So this opens
    each candidate and prints what geometry it actually holds.
    """
    import weapon_ids as _wid
    index = _wid.vpk_index(game)
    hits = sorted(k for k in index
                  if k.endswith(".vmdl_c")
                  and any(h in k for h in CHARACTER_HINTS))
    print(f"{len(index)} archive entries; {len(hits)} .vmdl_c under "
          f"{list(CHARACTER_HINTS)}")
    if not hits:
        print("  NONE. This depot ships no path matching those prefixes -- "
              "which is a fact about the depot, not about this filter. Widen "
              "it by hand and look; do not assume the models are absent.")
        return 0
    vpk = os.path.join(game, "csgo", "pak01_dir.vpk")
    content = A.Content(vpk, game, "pak01")
    for k in hits:
        raw = content.find(k[:-2] if k.endswith("_c") else k)
        if raw is None:
            print(f"  {k}\n      UNREADABLE from the content mount")
            continue
        try:
            by = {}
            for t, b in A.ordered_blocks(raw):
                by.setdefault(t, b)
            meshes = kv3v5.parse(by["CTRL"])["embedded_meshes"]
        except Exception as exc:                              # noqa: BLE001
            print(f"  {k}\n      CTRL unreadable: {exc}")
            continue
        tot = sum(sum(int(vb.get("m_nElementCount") or 0)
                      for vb in m["m_vertexBuffers"]) for m in meshes)
        tag = "  <- STUB" if tot < 100 else ""
        print(f"  {k}  {len(raw):,} B, {len(meshes)} mesh(es), "
              f"{tot:,} verts{tag}")
        for m in meshes:
            print("      %-32s %s verts across %d buffer(s)"
                  % (m.get("m_Name"),
                     "+".join(str(vb.get("m_nElementCount"))
                              for vb in m["m_vertexBuffers"]),
                     len(m["m_vertexBuffers"])))
    return 0


def to_world_axes(v):
    """Source (x, y, z) -> this renderer's world axes, in SOURCE UNITS.

    gpu_render.py:18 -- `glTF = (src_y, src_z, src_x) * 0.0254`, ray-cast
    validated. The scale is NOT applied here: the renderer places the model
    per entity and multiplies once, at the same site it multiplies the
    entity's own position, so the two cannot drift apart.
    """
    return np.stack([v[:, 1], v[:, 2], v[:, 0]], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true",
                    help="print the depot's character .vmdl_c paths and exit")
    ap.add_argument("--vmdl", default=None,
                    help="characters/models/<dir>/<model>.vmdl_c -- READ from "
                         "--list, never guessed")
    ap.add_argument("--game", required=True, help="depot game/ root")
    ap.add_argument("--map-vpk", default=None,
                    help="defaults to <game>/csgo/pak01_dir.vpk")
    ap.add_argument("--mesh", default=None,
                    help="embedded mesh name. Default: the first one, and "
                         "every name is printed so the default is visible "
                         "rather than assumed.")
    ap.add_argument("--name", default=None,
                    help="bundle basename; the renderer's "
                         "--playermodel-bundles looks for <team>.pt, so `ct` "
                         "and `t` are the two the pass resolves by team")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--edge", type=int, default=TEX_EDGE)
    ap.add_argument("--allow-stub", action="store_true",
                    help=f"write the bundle even if the model is under "
                         f"{STUB_HEIGHT_UNITS:g} source units tall. Without "
                         f"this a stub is REFUSED, because a bundle that "
                         f"loads and draws a 10-unit cube at every enemy is "
                         f"worse than no bundle -- the pass would report "
                         f"DREW.")
    a = ap.parse_args(argv)

    if a.list:
        return cmd_list(a.game)
    for req, why in (("vmdl", "--list first, then pass the path it printed"),
                     ("name", "the bundle basename the renderer resolves by"),
                     ("out_dir", "where the .pt goes")):
        if getattr(a, req) is None:
            raise SystemExit(f"--{req.replace('_', '-')} is required: {why}")

    vpk = a.map_vpk or os.path.join(a.game, "csgo", "pak01_dir.vpk")
    content = A.Content(vpk, a.game, "pak01")
    d = content.find(a.vmdl)
    if d is None:
        raise SystemExit(f"{a.vmdl}: not in the content mount. Run --list.")
    blocks = A.ordered_blocks(d)
    by = {}
    for t, b in blocks:
        by.setdefault(t, b)
    ctrl = kv3v5.parse(by["CTRL"])
    mo = A.load_meshopt()

    print(f"{a.name}  {a.vmdl}")
    print(f"  blocks {[t for t, _ in blocks]}")

    # ---- geometry -------------------------------------------------------
    names = [m.get("m_Name") for m in ctrl["embedded_meshes"]]
    print(f"  embedded meshes ({len(names)}): {names}")
    if not names:
        raise SystemExit(f"{a.vmdl}: no embedded_meshes. A character with no "
                         f"mesh in CTRL keeps its geometry somewhere this "
                         f"reader does not look; find it before bundling an "
                         f"empty draw.")
    if a.mesh is not None:
        sel = [m for m in ctrl["embedded_meshes"] if m.get("m_Name") == a.mesh]
        if not sel:
            raise SystemExit(f"{a.vmdl}: no mesh {a.mesh!r}; has {names}")
        emb = sel[0]
        print(f"  mesh: {a.mesh!r}, selected by name")
    elif BODY_MESH in names:
        emb = ctrl["embedded_meshes"][names.index(BODY_MESH)]
        print(f"  mesh: {BODY_MESH!r}, selected BY NAME -- an agent .vmdl_c "
              f"also ships firstperson_* meshes, and those are what the "
              f"local player's own camera sees, not what another player "
              f"looks like. Others here: "
              f"{[n for n in names if n != BODY_MESH]}")
    else:
        emb = ctrl["embedded_meshes"][0]
        print(f"  mesh: no {BODY_MESH!r} in this model, DEFAULTED to the "
              f"first, {names[0]!r} -- pass --mesh to pick another")
    mesh_name = emb.get("m_Name")

    # ---- EVERY buffer merged, and the DRAW CALLS decide the triangles ---
    # Two things were wrong with taking buffer 0 and one material, and both
    # were READ from ctm_sas thirdperson_body rather than reasoned about:
    #
    #   GEOMETRY  the mesh is 2 vertex buffers, not 1. Buffer 0 is 4,589
    #             verts spanning z 0..78.6 and buffer 1 is a further 5,310
    #             spanning z 32.9..73.7 -- the upper body. A silhouette
    #             missing 5,310 vertices still rasterises and still shades.
    #   MATERIALS the mesh has FOUR, one per draw call: ctm_sas_lenses,
    #             _head_gasmask, _bodylegs and _body. vm_slots.resolve takes
    #             the FIRST, which is the LENSES -- a 4x4 colour page. Bound
    #             to the whole character it would texture 13,504 triangles
    #             of soldier with an eye lens, print "REAL page" provenance
    #             for it, and be entirely wrong.
    #
    # So the triangles come from the DRAW CALLS, which carry m_nStartIndex,
    # m_nIndexCount, m_nBaseVertex and the buffer handles the range applies
    # to. That is also what makes the per-triangle material assignment a
    # READ rather than a guess: each draw call names its own material and
    # its own index range, and `mat_of_tri` is exactly that join.
    vbs = emb["m_vertexBuffers"]
    ibs = emb["m_indexBuffers"]
    P, NRM, UV0, UV1 = [], [], [], []
    vbase, uv1_missing = [], 0
    ATTRS = []
    for ci, vb in enumerate(vbs):
        attrs, _r, err = A.decode_vertex_buffer(np, mo, vb, blocks)
        assert not err, f"vertex buffer {ci}: {err}"
        ATTRS.append(attrs)
        p = np.asarray(attrs["POSITION_0"], np.float64)
        vbase.append(sum(len(x) for x in P))
        P.append(p)
        NRM.append(np.asarray(attrs["NORMAL_0_decoded"], np.float64))
        u0 = np.asarray(attrs["TEXCOORD_0"], np.float64)
        UV0.append(u0)
        if "TEXCOORD_1" in attrs:
            UV1.append(np.asarray(attrs["TEXCOORD_1"], np.float64))
        else:
            UV1.append(u0.copy())
            uv1_missing += 1
        print(f"  vertex buffer {ci}: {len(p)} verts, z "
              f"{p[:, 2].min():.1f}..{p[:, 2].max():.1f} source units")
    IDX = []
    for ci, ib in enumerate(ibs):
        sub, err = A.decode_index_buffer(np, mo, ib, blocks)
        assert not err, f"index buffer {ci}: {err}"
        IDX.append(np.asarray(sub, np.int64))
        print(f"  index buffer {ci}: {len(sub)} indices "
              f"({len(sub) // 3} tris)")
    pm = np.concatenate(P)                                    # source units
    nm = np.concatenate(NRM)
    uv0 = np.concatenate(UV0)
    uv1 = np.concatenate(UV1)
    N = len(pm)
    if uv1_missing:
        print(f"  TEXCOORD_1 ABSENT on {uv1_missing} of {len(vbs)} buffer(s):"
              f" uv.zw is a COPY of uv.xy there. The viewmodel contract puts"
              f" the sticker UV set in zw and a character has none, so "
              f"nothing downstream should read it -- the copy is stated so "
              f"a reader does not mistake it for a second set.")

    md = kv3v5.parse(blocks[emb["m_nDataBlock"]][1], strict=False)
    TRI, MAT_OF_TRI, mat_paths = [], [], []
    for so in md.get("m_sceneObjects") or []:
        for dc in so.get("m_drawCalls") or []:
            mp = dc.get("m_material")
            prim = dc.get("m_nPrimitiveType")
            if prim != "RENDER_PRIM_TRIANGLES":
                print(f"  draw call {mp} is {prim}, NOT triangles -- SKIPPED "
                      f"rather than reinterpreted as a triangle list")
                continue
            ibh = int((dc.get("m_indexBuffer") or {}).get("m_hBuffer", 0))
            vbh = int((dc.get("m_vertexBuffers") or [{}])[0].get(
                "m_hBuffer", 0))
            start = int(dc.get("m_nStartIndex", 0))
            count = int(dc.get("m_nIndexCount", 0))
            basev = int(dc.get("m_nBaseVertex", 0))
            if ibh >= len(IDX) or vbh >= len(vbase):
                print(f"  draw call {mp}: buffer handle out of range "
                      f"(ib {ibh}/{len(IDX)}, vb {vbh}/{len(vbase)}) -- "
                      f"SKIPPED")
                continue
            sub = IDX[ibh][start:start + count]
            if len(sub) < count:
                print(f"  draw call {mp}: index range [{start}, "
                      f"{start + count}) exceeds buffer {ibh} "
                      f"({len(IDX[ibh])}) -- TRUNCATED to {len(sub)}")
            t = sub[:len(sub) - len(sub) % 3].reshape(-1, 3) \
                + vbase[vbh] + basev
            if mp not in mat_paths:
                mat_paths.append(mp)
            TRI.append(t)
            MAT_OF_TRI.append(np.full(len(t), mat_paths.index(mp), np.int64))
            print(f"  draw call -> {mp}: ib{ibh} [{start}, {start + count}) "
                  f"= {len(t)} tris, vb{vbh} base {vbase[vbh] + basev}")
    if not TRI:
        raise SystemExit(f"{a.vmdl} :: {mesh_name}: no triangle draw calls. "
                         f"Falling back to the raw index buffer would drop "
                         f"the material join this bundle exists to carry.")
    tri = np.concatenate(TRI)
    mat_of_tri = np.concatenate(MAT_OF_TRI)
    print(f"  merged -> {N} verts, {len(tri)} tris across "
          f"{len(mat_paths)} material(s)")
    if int(tri.max()) >= N:
        raise SystemExit(f"{a.vmdl} :: {mesh_name}: a draw call indexes "
                         f"vertex {int(tri.max())} of {N} -- the base-vertex "
                         f"join is wrong and would read past the buffer.")

    # NO placement, NO bind subtraction, NO pose. Each is a decision and each
    # is stated rather than omitted:
    #   * placement is per-entity and comes from the demo, not the asset;
    #   * a character's model origin IS its world origin (the feet), so there
    #     is no weapon_offset-style bind bone to subtract;
    #   * the pose is the BIND pose because animation is not on the wire.
    pos = to_world_axes(pm)
    nrm = to_world_axes(nm)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

    # tangent from the UV gradient, the same construction extract_viewmodel
    # uses and for the same reason: the buffer carries a rotation + bitangent
    # sign rather than a vector, and apply_normal_map Gram-Schmidts against
    # the normal anyway.
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

    # --- TEXTURE PAGES, PER MATERIAL, with per-slot provenance -----------
    # vm_slots.resolve() binds ONE material -- the first draw call's -- which
    # is right for a weapon (one mesh, one skin) and wrong for a character.
    # ctm_sas thirdperson_body draws four materials and the first is the eye
    # LENSES at a 4x4 colour page; binding that to all 13,504 triangles would
    # texture a soldier with an eyeball and report REAL provenance for it.
    #
    # So the slot resolution runs PER MATERIAL, against the same SLOT_PARAM
    # map, and the bundle carries one page set per material beside the
    # per-triangle material index read from the draw calls above. Nothing
    # here decides which material "is" the body: all of them are carried and
    # the renderer draws each over its own triangles.
    os.makedirs(a.out_dir, exist_ok=True)

    def _pages_for(mat_path):
        slots, unmapped, slines = resolve_slots(content, a.vmdl, mesh_name,
                                                material=mat_path)
        for ln in slines:
            print("  " + ln.strip() if ln.strip() else ln)
        pages, prov = {}, {}
        for slot, vpath in slots.items():
            raw = content.find(vpath)
            if raw is None:
                prov[slot] = "MISSING: " + vpath
                print(f"    slot {slot}: {prov[slot]}")
                continue
            tmp = os.path.join(a.out_dir, ".pm_slot.vtex_c")
            with open(tmp, "wb") as fh:
                fh.write(raw)
            tnm, w, h, blk, fmtnum = _a2w.read_vtex(tmp, want=a.edge,
                                                    allow_lz4=True)
            os.remove(tmp)
            if blk is None:
                prov[slot] = f"UNDECODED ({tnm}): " + vpath
                print(f"    slot {slot}: {prov[slot]}")
                continue
            px = _bc.decode(blk, fmtnum, w, h)
            pages[slot] = torch.tensor(px.copy())
            prov[slot] = (f"REAL {vpath} :: {tnm} mip@{w}x{h} "
                          f"-> page {px.shape[1]}x{px.shape[0]}")
            print(f"    slot {slot}: {prov[slot]}")
        for slot, (vpath, why) in unmapped.items():
            present = content.find(vpath) is not None
            prov[slot] = (f"UNMAPPED (page "
                          f"{'PRESENT' if present else 'ABSENT'} in the "
                          f"archives): {vpath} -- {why}")
            print(f"    slot {slot}: {prov[slot]}")
        return pages, prov

    def _sampler_for(mp):
        """The material's own SAMPLER STATE, READ, or None with a reason.

        WHY THIS IS RECORDED AND NOT ASSUMED. Every sampler in the renderer
        hardcodes wrap-repeat -- vm_sample takes `uv % 1.0` and so do the
        world taps -- and NOTHING anywhere reads an address mode. The
        material carries one: `g_nTextureAddressModeU` / `...V` are
        m_intParams on the .vmat_c, present on 113 of the 128 materials the
        playermodel pass draws.

        Censused over exactly those 128: every declared value is 0, which
        is WRAP, so the hardcode is CORRECT for this population and the
        fix is worth ZERO pixels today. That is the whole reason to record
        it rather than to go fix sampling: the gap is real but latent, and
        an assumption that happens to hold is indistinguishable from a
        checked one right up until the first material that says clamp.
        Recording the field converts it into a precondition the consumer
        can assert instead of a coincidence it depends on.
        """
        d = content.find(mp if mp.endswith("_c") else mp + "_c")
        if d is None:
            return None, f"{mp}: not in the content mount"
        by = {}
        for t, blk in A.ordered_blocks(d):
            by.setdefault(t, blk)
        if "DATA" not in by:
            return None, f"{mp}: no DATA block"
        dd = kv3v5.parse(by["DATA"], strict=False)
        ip = {str(p.get("m_name")): p.get("m_nValue")
              for p in (dd.get("m_intParams") or [])}
        u = ip.get("g_nTextureAddressModeU")
        v = ip.get("g_nTextureAddressModeV")
        if u is None and v is None:
            return None, (f"{mp} declares neither g_nTextureAddressModeU "
                          f"nor ...V among its {len(ip)} int params")
        return {"address_u": u, "address_v": v}, None

    materials = []
    for mi, mp in enumerate(mat_paths):
        ntri_m = int((mat_of_tri == mi).sum())
        print(f"  material [{mi}] {mp}  ({ntri_m} tris, "
              f"{100.0 * ntri_m / len(tri):.1f}% of the mesh)")
        pg, pv = _pages_for(mp)
        smp, swhy = _sampler_for(mp)
        print(f"    sampler: {smp if smp else 'UNDECLARED -- ' + str(swhy)}")
        materials.append({"path": mp, "pages": pg, "provenance": pv,
                          "triangles": ntri_m,
                          "sampler": smp, "sampler_absent_reason": swhy,
                          "sampler_meaning":
                              "g_nTextureAddressModeU/V READ from the "
                              ".vmat_c m_intParams. Source 2's enum has 0 = "
                              "WRAP. A consumer that hardcodes wrap MUST "
                              "assert this equals 0 rather than assume it: "
                              "the assumption holds on every material "
                              "censused so far and is not thereby checked."})

    # ---- THE STUB REFUSAL, before anything is written ------------------
    # This is the check that caught characters/models/ctm_sas: 24 vertices,
    # 10 source units tall, a default_orange material, and it would have
    # produced a bundle the renderer loads without complaint and reports as
    # DREW. A bad asset that renders is worse than a missing one, because
    # the per-frame line then says the feature is working.
    _h = float(pos[:, 1].max() - pos[:, 1].min())
    if _h < STUB_HEIGHT_UNITS and not a.allow_stub:
        raise SystemExit(
            f"REFUSED: {a.vmdl} :: {mesh_name} is {_h:.2f} source units "
            f"tall ({_h * 0.0254:.3f} m) across {N} vertices. A standing CS "
            f"player is ~72 units and a crouching one ~54; under "
            f"{STUB_HEIGHT_UNITS:g} this is a STUB, not a player.\n"
            f"  The `characters/models/` tree holds 24-vertex proxy cubes "
            f"for these models and `agents/models/` holds the geometry -- "
            f"both trees carry all 92 names, so check which one this path "
            f"is in before anything else.\n"
            f"  Pass --allow-stub if you meant it.")

    bundle = {
        # The eight-key contract, so one loader shape serves both bundles.
        # `pos` is MODEL SPACE, SOURCE UNITS -- see `space`.
        "pos": torch.tensor(pos, dtype=torch.float32),
        "uv": torch.tensor(np.concatenate([uv0, uv1], axis=1),
                           dtype=torch.float32),
        "nrm": torch.tensor(nrm, dtype=torch.float32),
        "tan": torch.tensor(tan, dtype=torch.float32),
        "loc": torch.tensor(pm, dtype=torch.float32),          # SOURCE UNITS
        "tri": torch.tensor(tri, dtype=torch.int32),
        "fam": torch.zeros(len(tri), dtype=torch.long),
        "is_legs": torch.zeros(N, dtype=torch.bool),
        # PER-TRIANGLE MATERIAL, read off the draw calls' own index ranges.
        # This is what lets the renderer cover 100% of the surface instead of
        # the dominant material's share.
        "mat_of_tri": torch.tensor(mat_of_tri, dtype=torch.long),
        "materials": materials,
        "space": SPACE,
        "source": f"{a.vmdl} :: {mesh_name}",
        "playermodel": a.name,
        "axis_map": "world_src = (src_y, src_z, src_x); the renderer yaws "
                    "about source +Z, adds the entity's own (X, Y, Z) and "
                    "multiplies by 0.0254 at that one site",
        "pose_basis": "BIND POSE. Animation is NOT on the wire (Valve "
                      "deleted DEM_AnimationData at proto 14150), so no "
                      ".dem can supply a pose and none was invented.",
        "placement_basis": "NONE baked in. Placement is per entity per tick "
                           "and comes from the demo, not from this asset.",
        "vertex_buffers": len(emb["m_vertexBuffers"]),
        # `pose_basis` says BIND POSE and stays true -- these bindings do not
        # pose anything, they are what a consumer needs in order TO pose it.
        # Without them the bind pose was not a choice but the only reachable
        # state: no vertex declared which bone owned it, so a walk cycle had
        # nowhere to land.
        "skin_note": "per-vertex bone bindings for the WHOLE merged vertex "
                     "set, in the same order as pos/loc",
        # NO singular tex_pages/tex_provenance. A character has no single
        # page set, and leaving one here would be a set some reader binds to
        # the whole model -- the exact defect `materials` exists to end.
        "tex_edge": a.edge,
    }
    # The remap table and the model skeleton live in the MODEL's DATA block,
    # not in the mesh's -- `md` above is the mesh data block, which holds the
    # draw calls. Reading the remap from the wrong block finds nothing and
    # reads as "this model ships no remap", which is how blend indices end up
    # pointing at the wrong bones with no error raised.
    _mdat = kv3v5.parse(by["DATA"], strict=False)
    _mskel = _mdat["m_modelSkeleton"]
    _skin, _sklines = mesh_skin.read_skin(
        ATTRS, emb, _mdat, list(_mskel["m_boneName"]),
        label=f"{a.vmdl}::{mesh_name}")
    for _ln in _sklines:
        print(_ln)
    mesh_skin.add_skin(bundle, _skin, _mskel, N)
    out = os.path.join(a.out_dir, f"{a.name}.pt")
    torch.save(bundle, out)
    print(f"wrote {out}")
    for k in ("pos", "uv", "nrm", "tan", "loc", "tri", "fam", "is_legs"):
        v = bundle[k]
        print(f"  {k:8s} {tuple(v.shape)} {v.dtype}")
    p = bundle["pos"].numpy()
    print(f"  model-space bbox (SOURCE UNITS, world axes): "
          f"min {np.round(p.min(0), 2)}  max {np.round(p.max(0), 2)}")
    # The one sanity read a character bundle has that a weapon does not: a
    # standing player is about 72 source units tall and stands ON its origin.
    # Printed as a READ of the asset, with no correction applied either way.
    hgt = float(p[:, 1].max() - p[:, 1].min())
    foot = float(p[:, 1].min())
    print(f"  height {hgt:.2f} source units ({hgt * 0.0254:.3f} m), lowest "
          f"vertex at y {foot:+.2f} -- a standing CS player is ~72 units and "
          f"stands ON the origin, so a large |lowest| means this model's "
          f"origin is NOT its feet and the renderer will sink or float it")
    print(f"  |normal| mean "
          f"{np.linalg.norm(bundle['nrm'].numpy(), axis=1).mean():.5f}")
    print(f"  |tangent| mean "
          f"{np.linalg.norm(bundle['tan'].numpy(), axis=1).mean():.5f}")
    print(f"  triangles {len(tri)}, max index {int(tri.max())} of {N - 1}")
    for mi, m in enumerate(materials):
        real = sum(1 for v in m["provenance"].values()
                   if v.startswith("REAL"))
        print(f"  material [{mi}] {os.path.basename(m['path'])}: "
              f"{m['triangles']} tris, {real} REAL pages of "
              f"{len(m['provenance'])} slots with provenance")
    cov = {int(v): int((mat_of_tri == v).sum()) for v in set(mat_of_tri)}
    print(f"  material coverage: {len(tri)} tris, "
          f"{100.0 * sum(cov.values()) / len(tri):.1f}% assigned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
