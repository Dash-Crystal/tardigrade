

def playermodel_pass(mvp, eye, world_rgb, world_zfc):
    """Rasterise the frame's other players into the WORLD depth and composite.

    `world_rgb` is the finished world frame (B, height, width, 3) uint8 and
    `world_zfc` is the resolved scene depth in gl_FragCoord.z convention at
    RASTER resolution (B, H, W) -- the same `zfc_o` the decal and sprite
    layers read, so a player is occluded by exactly what those are occluded
    by.

    One rasterisation PER TEAM, because the two teams are different meshes
    with different vertex counts and one shared topology cannot hold both.
    Depth accumulates across teams and starts at the world's, so
    player-behind-player and player-behind-wall are the same comparison.
    """
    B = mvp.shape[0]
    b0 = int(VS_FRAME0[0])
    if not args.playermodels:
        for i in range(B):
            frame_print(f"PLAYERMODELS f{b0 + i:06d} ABSENT: --playermodels not "
                  f"given, so no other players are drawn at all (GAP, not an "
                  f"empty map)", flush=True)
        return world_rgb
    _pm_load()
    per_frame, ticks, reasons = [], [], []
    for i in range(B):
        ents, tick, why = pm_entities_for_frame(b0 + i)
        per_frame.append(ents)
        ticks.append(tick)
        reasons.append(why)
    if not any(per_frame):
        for i in range(B):
            why = reasons[i] or (
                f"tick {ticks[i]}: every row was excluded -- ego "
                f"{sorted(PM_EGO)}, dead players, or a team with no bundle "
                f"({PM_BUNDLE_NOTE})")
            frame_print(f"PLAYERMODELS f{b0 + i:06d} ABSENT: {why}", flush=True)
        return world_rgb

    # ---- THE POSE, CHOSEN BEFORE ANY GEOMETRY IS BUILT ----------------
    # IN FRAME ORDER, and every frame of the batch, not the first one. The
    # viewmodel side learned this the expensive way: reading only the
    # batch's first row lost three of the ak window's four fire ticks
    # because they landed second-in-batch, and the A/B showed 2 of 32
    # frames moving. Here the scripter's per-entity distance accumulator
    # makes it worse than a lost event -- a skipped row is stride the walk
    # cycle never advances by -- so rows are consumed strictly in order.
    for i in range(B):
        pm_script(per_frame[i], ticks[i])
    for i in range(B):
        for e in per_frame[i]:
            if e.get("_ev") is None:
                continue
            _bl = e["_clip"].get("blocks") or []
            frame_print(
                f"PLAYERMODEL ANIM f{b0 + i:06d} "
                f"{e.get('team')}#{e.get('steamid')} event={e['_ev']} "
                f"clip={os.path.basename(e['_clip']['clip'])} "
                f"t={e['_t']:.3f}s of {e['_clip']['duration']:.3f}s "
                f"({e['_clip']['frames']} frames, {len(_bl)} block(s)"
                + (", STATIC" if e.get("_static") else ", animated")
                + f") | {e['_why'][0]}"
                + (f" | {e['_why'][-1]}" if len(e["_why"]) > 1 else ""),
                flush=True)

    out = world_rgb.float() / 255.0
    zbuf = world_zfc.clone()
    acc_rgb = torch.zeros(B, H, W, 3, device=device)
    acc_a = torch.zeros(B, H, W, device=device)
    # PER ENTITY, not per team. "4 of 6 drew" taken from the team's total
    # would count an entity that was placed inside a wall as having drawn,
    # and a headline that cannot distinguish placed-and-occluded from
    # placed-and-visible is the same collapse the viewmodel's coverage
    # counter made. The slot id is recoverable from the triangle id, so
    # this is a read of the raster rather than an estimate.
    ent_px = [{} for _ in range(B)]
    _used = sorted({e.get("_bundle") for f in per_frame for e in f
                    if e.get("_bundle")})
    for team in _used:
        # (index into per_frame[i], entity) so the count lands back on the
        # entity the line names
        sel = [[(j, e) for j, e in enumerate(f)
                if e.get("_bundle") == team]
               for f in per_frame]
        nslots = max(len(s) for s in sel)
        if nslots == 0:
            continue
        b = PM_BUNDLES[team]
        p_src = b["pos"].to(device)
        n_src = b["nrm"].to(device)
        t_src = b["tan"].to(device)
        uv_src = b["uv"].to(device)
        tri1 = b["tri"].to(device).int()
        nv = p_src.shape[0]
        # the shared topology: the mesh's triangles once per slot, offset.
        # BROADCAST, not a python cat-loop: tri1[None] + (arange*nv) is the
        # identical integer result (exact equality, no float reassociation)
        # in one launch instead of nslots+1 -- HOSTLOOP_TOP10 row 5, x9.2.
        tri = (tri1[None]
               + (torch.arange(nslots, device=tri1.device, dtype=tri1.dtype)
                  * nv)[:, None, None]).reshape(-1, 3).contiguous()
        uv4 = uv_src.repeat(nslots, 1)
        ents_of = [[e for _j, e in s] for s in sel]
        # PER ENTITY PER FRAME, which is the whole point: two players in
        # one slot-set are at different phases of different clips, so the
        # pose cannot be hoisted out of either loop. It IS cached on
        # (bundle, clip, t), so ten players idling on the same tick of the
        # same clip cost one skin evaluation and not ten.
        geo = [pm_slot_geom(team, b, ents_of[i], nslots) for i in range(B)]
        # BATCHED placement -- HOSTLOOP_TOP10 row 1. The (bundle, clip, t)
        # pose cache stays exactly where it was (pm_slot_geom above, host
        # dict work); only the 3*B tensor calls collapse to three batched
        # ones with elementwise-identical expressions. Static count for a
        # 48-frame window: ~3,840 slot-loop tensor ops -> ~18 launches.
        posw = pm_place_batch(p_src, ents_of, B, nslots,
                              per_slot_all=[g[0] for g in geo])
        nrmw = pm_rotate_dirs_batch(n_src, ents_of, B, nslots,
                                    per_slot_all=[g[1] for g in geo])
        tanw = pm_rotate_dirs_batch(t_src, ents_of, B, nslots,
                                    per_slot_all=[g[2] for g in geo])
        # WORLD clip through the frame's own mvp -- the same matrix, the
        # same NDC-z convention and therefore the same depth comparison the
        # three world classes made.
        # SITE 2 of 3 -- nvdiffrast raises "tri must have shape [>0, 3]" on
        # an empty triangle buffer, which is how de_mirage_vanity died. A
        # team with no visible member contributes no geometry; that is a
        # valid state and not a malformed mesh. Guarding only the outer
        # torch.cat (site 1) would have fixed 12 of 14 maps and left this
        # one to resurface later looking like a rasteriser bug, which is
        # precisely why all three are being closed together.
        if tri.numel() == 0 or posw.shape[1] == 0:
            frame_print(f"playermodel_pass: team {team!r} skipped -- "
                        f"{int(posw.shape[1])} vertices / "
                        f"{int(tri.shape[0])} triangles after slot "
                        f"selection, so there is nothing to rasterise. An "
                        f"empty team is a valid frame state.")
            continue
        hom = torch.cat([posw, torch.ones_like(posw[..., :1])], dim=-1)
        clip = torch.einsum("bij,bnj->bni", mvp.float(), hom).contiguous()
        rast, _db = dr.rasterize(ctx, clip, tri, (H, W))
        covered = rast[..., 3] > 0
        if not bool(covered.any()):
            continue
        zfc = _frag_depth(rast[..., 2], covered)
        keep = covered & (zfc < zbuf)
        if not bool(keep.any()):
            continue
        uvp, _ = dr.interpolate(uv4[None].expand(B, -1, -1).contiguous(),
                                rast, tri)
        wposp, _ = dr.interpolate(posw.contiguous(), rast, tri)
        nrmp, _ = dr.interpolate(nrmw.contiguous(), rast, tri)
        tanp, _ = dr.interpolate(tanw.contiguous(), rast, tri)
        view = eye[:, None, None, :] - wposp
        view = view / view.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # PER MATERIAL. The mesh's triangles carry a material index read off
        # the draw calls, so each material shades over exactly its own
        # triangles instead of one page set covering a surface it does not
        # draw. Slot id and local triangle id both come out of the same
        # triangle id: the topology is the mesh repeated nslots times.
        mat_of_tri = b["mat_of_tri"].to(device)
        ntri1 = tri1.shape[0]
        tri_i = (rast[..., 3].long() - 1).clamp(min=0)
        matpix = mat_of_tri[tri_i % ntri1]
        for mi in range(len(b["materials"])):
            mk = keep & (matpix == mi)
            if not bool(mk.any()):
                continue
            rgb, a, _k = vm_gt_shade(uvp, nrmp, tanp, view, mk,
                                     tex=pm_pages(team, mi), wpos=wposp)
            acc_rgb = torch.where(mk.unsqueeze(-1), rgb, acc_rgb)
            acc_a = torch.where(mk, a, acc_a)
        zbuf = torch.where(keep, zfc, zbuf)
        # slot id from the triangle id: the topology is the mesh repeated
        # nslots times, so triangle t belongs to slot t // ntri1.
        ntri1 = tri1.shape[0]
        slot = (rast[..., 3].long() - 1).clamp(min=0) // ntri1
        # HOSTLOOP_TOP10 row 3, the REAL storm: int(tensor.sum()) per
        # (frame, entity) was B*nslots device->host syncs per window per
        # team, inside the render path. ONE bincount over (frame*slot)
        # then ONE .tolist(): identical integers, 2 launches + 1 sync.
        _fs = (torch.arange(B, device=slot.device)[:, None, None] * nslots
               + slot.clamp(max=nslots - 1))
        _cnt = torch.bincount(
            _fs[keep].reshape(-1), minlength=B * nslots
        ).reshape(B, nslots).tolist()
        for i in range(B):
            for k, (j, _e) in enumerate(sel[i]):
                ent_px[i][j] = int(_cnt[i][k])

    # ---- THE WEAPON IN THE HAND -------------------------------------
    # A SECOND MESH ON THE SAME DEPTH BUFFER, not a second pass with its
    # own. `zbuf` already carries the world AND every character drawn
    # above, so a rifle is occluded by the wall in front of it, by the
    # player holding it, and by the player standing in front of that one,
    # all through the one comparison the world classes made. Drawing it
    # into its own buffer and compositing after would put every gun in
    # front of everything, which reads as correct until two players
    # overlap.
    for wname in sorted({e.get("_weapon") for f in per_frame for e in f
                         if e.get("_weapon")}):
        wsel = [[(j, e) for j, e in enumerate(f)
                 if e.get("_weapon") == wname] for f in per_frame]
        wslots = max(len(s) for s in wsel)
        if not wslots:
            continue
        wb = PM_WBUNDLES[wname]
        wtri1 = wb["tri"].to(device).int()
        wnv = wb["loc"].shape[0]
        wtri = torch.cat([wtri1 + i * wnv
                          for i in range(wslots)], dim=0).contiguous()
        if wtri.numel() == 0:
            continue
        wuv4 = wb["uv"].to(device).repeat(wslots, 1)
        wents = [[e for _j, e in s] for s in wsel]
        wgeo = [pm_weapon_geom(wname, wb, wents[i], wslots)
                for i in range(B)]
        if not any(g[0] for g in wgeo):
            continue
        # The zero array is the BIND-POSE weapon in its own model frame,
        # which no slot should ever fall back to: an entity with no
        # chosen clip has no socket to hang on, so its slot stays None
        # and pm_place collapses it to a point (zero-area, invisible).
        wzero = torch.zeros((wnv, 3), device=device, dtype=torch.float32)
        wposw = torch.stack([pm_place(wzero, wents[i], wslots,
                                      per_slot=wgeo[i][0])
                             for i in range(B)])
        wnrmw = torch.stack([pm_rotate_dirs(wzero, wents[i], wslots,
                                            per_slot=wgeo[i][1])
                             for i in range(B)])
        wtanw = torch.stack([pm_rotate_dirs(wzero, wents[i], wslots,
                                            per_slot=wgeo[i][2])
                             for i in range(B)])
        whom = torch.cat([wposw, torch.ones_like(wposw[..., :1])], dim=-1)
        wclip = torch.einsum("bij,bnj->bni", mvp.float(), whom).contiguous()
        wrast, _wdb = dr.rasterize(ctx, wclip, wtri, (H, W))
        wcov = wrast[..., 3] > 0
        if not bool(wcov.any()):
            continue
        wzfc = _frag_depth(wrast[..., 2], wcov)
        wkeep = wcov & (wzfc < zbuf)
        if not bool(wkeep.any()):
            continue
        wuvp, _ = dr.interpolate(wuv4[None].expand(B, -1, -1).contiguous(),
                                 wrast, wtri)
        wwpos, _ = dr.interpolate(wposw.contiguous(), wrast, wtri)
        wnrmp, _ = dr.interpolate(wnrmw.contiguous(), wrast, wtri)
        wtanp, _ = dr.interpolate(wtanw.contiguous(), wrast, wtri)
        wview = eye[:, None, None, :] - wwpos
        wview = wview / wview.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        wmat = wb["mat_of_tri"].to(device)
        wntri1 = wtri1.shape[0]
        wmatpix = wmat[(wrast[..., 3].long() - 1).clamp(min=0) % wntri1]
        for mi in range(len(wb["materials"])):
            mk = wkeep & (wmatpix == mi)
            if not bool(mk.any()):
                continue
            rgb, a, _k = vm_gt_shade(wuvp, wnrmp, wtanp, wview, mk,
                                     tex=pm_weapon_pages(wname, mi),
                                     wpos=wwpos)
            acc_rgb = torch.where(mk.unsqueeze(-1), rgb, acc_rgb)
            acc_a = torch.where(mk, a, acc_a)
        zbuf = torch.where(wkeep, wzfc, zbuf)
        PM_WEAPON_PX[0] += int(wkeep.sum())
        wslot = (wrast[..., 3].long() - 1).clamp(min=0) // wntri1
        for i in range(B):
            for k, (j, _e) in enumerate(wsel[i]):
                per_frame[i][j]["_weapon_px"] = int(
                    ((wslot[i] == k) & wkeep[i]).sum())
        if not PM_WEAPON_NOTE:
            PM_WEAPON_NOTE.append(1)
            print(f"PLAYERMODEL WEAPON TIER: {wname!r} is attached RIGIDLY "
                  f"to the rig's posed `wpn` socket -- it moves with the "
                  f"hands, and its OWN bones (bolt, clip, trigger) are NOT "
                  f"driven, because the world clip poses the CHARACTER rig "
                  f"and the weapon-side motion lives in the weapon's own "
                  f"clips which this pass does not sample. A third-person "
                  f"rifle therefore recoils and does not cycle. That is a "
                  f"tier, not the claim that the weapon is animated.",
                  flush=True)
    if float(acc_a.sum()) == 0.0:
        for i in range(B):
            frame_print(f"PLAYERMODELS f{b0 + i:06d} tick {ticks[i]} ABSENT: "
                  f"{len(per_frame[i])} entities placed and NONE reached a "
                  f"pixel -- all outside the frustum or behind world "
                  f"geometry, which is a visibility result and not a "
                  f"missing asset", flush=True)
        return world_rgb
    img = acc_rgb.flip(1)
    al = acc_a.flip(1).unsqueeze(-1)
    if args.supersample > 1:
        img = torch.nn.functional.avg_pool2d(
            img.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        al = torch.nn.functional.avg_pool2d(
            al.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
    if img.shape[1:3] != out.shape[1:3]:
        # Preset FSR: the raster (and every occlusion input) ran at the
        # INTERNAL resolution while the frame was already upscaled by the
        # post chain. The scene's upscaler is FSR; this layer gets
        # bilinear -- a STATED divergence, printed per run, not silent.
        frame_print(
            f"PLAYERMODELS layer {img.shape[2]}x{img.shape[1]} -> frame "
            f"{out.shape[2]}x{out.shape[1]}: raster at FSR-internal res, "
            f"layer upscaled BILINEAR to composite (scene used FSR).",
            flush=True)
        img = torch.nn.functional.interpolate(
            img.permute(0, 3, 1, 2), size=out.shape[1:3],
            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        al = torch.nn.functional.interpolate(
            al.permute(0, 3, 1, 2), size=out.shape[1:3],
            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    before = out
    out = out * (1.0 - al) + img.clamp(0.0, 1.0) * al
    _ts.observe("playermodels.composite", out, before=before, mask=(al > 0),
                reference="world depth test against zfc_o; "
                          + ("SKINNED per-entity clip pose"
                             if PM_CLIPS[0] is not None else "bind pose")
                          + "; gt_direct() over the character's own pages")
    # THE DRIVEN COLUMN, with its COUNTS. `driven` is what the ANIMATION
    # STATE matrix reports, and a bare event name there would say a clip
    # was selected without saying how often or by how many entities --
    # which is the difference between one player twitching and ten
    # walking. It is written from the tally the scripter kept, so it
    # cannot claim an event nothing chose.
    if PM_EVENT_COUNT:
        ANIM_DRIVEN["playermodel"] = ",".join(
            f"{k}:{v}" for k, v in sorted(PM_EVENT_COUNT.items(),
                                          key=lambda kv: -kv[1]))
    for i in range(B):
        vis = sum(1 for v in ent_px[i].values() if v > 0)
        # REQUESTED vs DREW, per entity. With two bundles built almost
        # every entity takes the team fallback, and that must never read
        # as a fact about the demo when it is a fact about which bundles
        # exist.
        where = "; ".join(
            f"{e.get('team')}#{e.get('steamid')} "
            f"({e['x']:.1f},{e['y']:.1f},{e['z']:.1f}) yaw {e['yaw']:.1f} "
            f"agent={e.get('agent') or 'NONE'} drew={e.get('_bundle')}.pt "
            f"[{e.get('_bundle_how')}] "
            + (f"{e['_ev']}@{e['_t']:.2f}s "
               f"{os.path.basename(e['_clip']['clip'])} "
               if e.get("_ev") else "pose BIND ")
            + f"{ent_px[i].get(j, 0)}px "
            # HOLDING WHAT, per entity. An ally rendered empty-handed is
            # the owner-visible failure this closes, so the weapon gets
            # its own term on the line rather than being inferred from
            # the absence of one.
            + (f"wpn={e['_weapon']} {e.get('_weapon_px', 0)}px"
               if e.get("_weapon")
               else f"wpn=NONE ({e.get('_weapon_how', 'not resolved')})")
            for j, e in enumerate(per_frame[i]))
        # THE POSE CLAUSE IS NOW A READ OF WHAT HAPPENED, not a constant
        # string. It said `pose BIND (animation is NOT on the wire)` on
        # every frame ever rendered; the second half of that stays true --
        # no pose is on the wire and none is read from one -- but the
        # first half is now false whenever a clip was chosen, and a line
        # that cannot tell those apart is the reason this took a defect
        # queue entry to notice.
        _drv = sum(1 for e in per_frame[i] if e.get("_ev"))
        _pose = (f"pose DRIVEN for {_drv} of {len(per_frame[i])} "
                 f"({'/'.join(sorted({e['_ev'] for e in per_frame[i] if e.get('_ev')}))}) "
                 f"-- clip INFERRED from demo state columns, no pose read "
                 f"off the wire by this lane (whether one is present is "
                 f"contested, #102); max displacement vs bind "
                 f"{PM_ANIM_MOVED[0] * 1000:.0f} mm"
                 if _drv else
                 f"pose BIND for all {len(per_frame[i])} "
                 + ("(--pm-anim off: the CONTROL arm)"
                    if args.pm_anim == "off"
                    else f"({PM_CLIPS_NOTE[0] or 'no clip chosen'})"))
        frame_print(f"PLAYERMODELS f{b0 + i:06d} tick {ticks[i]} DREW {vis} of "
              f"{len(per_frame[i])} entities, {_pose} :: {where}", flush=True)
    return (out.clamp(0, 1) * 255).to(torch.uint8)
