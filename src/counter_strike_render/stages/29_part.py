

# ── HYPERVISOR GROUND-TRUTH BONE CAPTURE ─────────────────────────────
# When --hv-capture is given, this loader provides per-tick world-space
# bone transforms read directly from CS2 via the hypervisor. These
# replace the clip scripter for pose selection entirely.
HV_BONES = {}       # tick(approx) → [{pos:[x,y,z], bones:[[px,py,pz,qx,qy,qz,qw,s],...]}]
HV_TICKS = []       # sorted tick list for nearest-match
HV_BONE_NAMES = []  # worldmodel skeleton bone names (from agents.clips.pt)
HV_NOTE = []
if args.hv_capture and os.path.exists(args.hv_capture):
    import bisect as _bisect
    _hv_data = []
    with open(args.hv_capture) as _fh:
        for _line in _fh:
            _hv_data.append(json.loads(_line))
    # Index HV frames by a pseudo-tick derived from frame sequence.
    # The HV capture runs at ~1 Hz while the demo is 64 tick/sec.
    # We assign each HV frame a tick range and match by position proximity.
    # Store as frame-index → player list for position matching.
    HV_FRAMES = _hv_data
    print(f"hv-capture: loaded {len(_hv_data)} frames from {args.hv_capture}, "
          f"{sum(f['np'] for f in _hv_data)} total player snapshots", flush=True)
    # Load worldmodel bone names for the palette join
    if args.playermodel_clips:
        _cs_path = args.playermodel_clips
    elif args.playermodel_bundles:
        _cs_path = os.path.join(args.playermodel_bundles, "agents.clips.pt")
    else:
        _cs_path = None
    if _cs_path and os.path.exists(_cs_path):
        _csd = torch.load(_cs_path, map_location="cpu", weights_only=False)
        _first_clip = next(iter(_csd["events"].values()))[0]
        _blk = _first_clip["blocks"][0]
        HV_BONE_NAMES = list(_blk["bone_ids"])
        print(f"hv-capture: worldmodel skeleton has {len(HV_BONE_NAMES)} bones",
              flush=True)
else:
    HV_FRAMES = []


def _hv_match_entity(hv_frame, demo_entity):
    """Find the HV player closest to this demo entity's position."""
    ex, ey, ez = float(demo_entity["x"]), float(demo_entity["y"]), float(demo_entity["z"])
    best, best_d = None, float("inf")
    for p in hv_frame.get("p", []):
        hx, hy, hz = p["pos"]
        d = (hx - ex)**2 + (hy - ey)**2 + (hz - ez)**2
        if d < best_d:
            best_d = d
            best = p
    return best if best_d < 200**2 else None  # within 200 units


def _hv_frame_for_tick(tick):
    """Find the HV capture frame nearest to this demo tick.

    The HV capture runs at ~1 Hz while the demo runs at 64 tick/sec.
    We distribute the HV frames evenly across the demo's tick range
    and pick the nearest one.
    """
    if not HV_FRAMES:
        return None
    # Map demo ticks to HV frame indices using the camera tick range
    # The simplest approach: use the HV frame index as a coarse time proxy
    # and match entities by position, not by tick.
    # We cycle through HV frames based on the demo tick.
    tick_range = TICK_OF_FRAME  # {frame_idx: tick}
    if not tick_range:
        return HV_FRAMES[0] if HV_FRAMES else None
    all_ticks = sorted(tick_range.values())
    if not all_ticks:
        return HV_FRAMES[0]
    t0, t1 = all_ticks[0], all_ticks[-1]
    span = max(t1 - t0, 1)
    frac = max(0.0, min(1.0, (tick - t0) / span))
    idx = min(int(frac * len(HV_FRAMES)), len(HV_FRAMES) - 1)
    return HV_FRAMES[idx]


def pm_pose_arrays_hv(name, b, hv_bones_raw, entity_pos, entity_yaw):
    """Produce (pos, nrm, tan) from HV ground-truth world-space bones.

    hv_bones_raw: list of [px,py,pz,qx,qy,qz,qw,scale] per bone,
                  in Source world space, ABSOLUTE (already world-positioned).

    The skin pipeline needs bone transforms in model space (relative to the
    entity origin), so we undo the entity's world position and yaw to get
    model-space transforms, then run bone_transforms + lbs as usual.
    """
    skin = b.get("skin")
    if skin is None or not hv_bones_raw or not HV_BONE_NAMES:
        return None

    n_hv = min(len(hv_bones_raw), len(HV_BONE_NAMES))

    # Convert HV world-space bones to model-space by removing entity transform
    yaw_rad = math.radians(float(entity_yaw))
    cos_y, sin_y = math.cos(-yaw_rad), math.sin(-yaw_rad)
    ox, oy, oz = float(entity_pos[0]), float(entity_pos[1]), float(entity_pos[2])

    mt = np.zeros((n_hv, 3), dtype=np.float64)
    mq = np.zeros((n_hv, 4), dtype=np.float64)
    mq[:, 3] = 1.0  # identity

    for i in range(n_hv):
        b_data = hv_bones_raw[i]
        # world position → entity-local position (undo translation + yaw)
        wx, wy, wz = b_data[0] - ox, b_data[1] - oy, b_data[2] - oz
        # undo yaw rotation (rotate by -yaw)
        lx = wx * cos_y - wy * sin_y
        ly = wx * sin_y + wy * cos_y
        lz = wz
        mt[i] = [lx, ly, lz]
        # world quaternion → entity-local quaternion (undo yaw)
        # Entity yaw is a rotation about Z axis
        # q_inv_yaw = (0, 0, sin(-yaw/2), cos(-yaw/2))
        half = -yaw_rad / 2.0
        iq = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
        bq = np.array([b_data[3], b_data[4], b_data[5], b_data[6]])
        mq[i] = _skin.qmul(iq, bq)

    # Now run the standard palette join + LBS
    idx_np = skin["index"].cpu().numpy()
    wgt_np = skin["weight"].cpu().numpy().astype(np.float64)
    loc_np = b["loc"].cpu().numpy().astype(np.float64)

    xq, xt, rep = _skin.bone_transforms(
        skin["palette"],
        skin["bind_t"].cpu().numpy().astype(np.float64),
        skin["bind_q"].cpu().numpy().astype(np.float64),
        HV_BONE_NAMES[:n_hv], mt, mq,
        palette_ancestors=skin.get("palette_ancestors"),
        index=idx_np, weight=wgt_np)

    nrm_np = _pm_to_src_axes_np(b["nrm"].cpu().numpy().astype(np.float64))
    tan_np = _pm_to_src_axes_np(b["tan"].cpu().numpy().astype(np.float64))
    P, (Nn, Tt) = _skin.lbs(loc_np, idx_np, wgt_np, xq, xt,
                            dirs=[nrm_np, tan_np])

    dead = (wgt_np.sum(1, keepdims=True) <= 1e-6)
    P = np.where(dead, loc_np, P)
    Nn = np.where(dead, nrm_np, Nn)
    Tt = np.where(dead, tan_np, Tt)

    P = torch.as_tensor(P, device=device)
    Nn = torch.as_tensor(Nn, device=device)
    Tt = torch.as_tensor(Tt, device=device)

    return (_pm_to_world_axes(P).to(torch.float32),
            _pm_to_world_axes(Nn).to(torch.float32),
            _pm_to_world_axes(Tt).to(torch.float32))


def pm_weapon_geom(name, wb, ents, nslots):
    """Per-slot weapon (pos, nrm, tan) in WORLD AXES, or (None, None, None).

    THE CHAIN, and it is the viewmodel's own with ONE TERM SWAPPED:

        world_src = qrot(q_wpn, loc - bind_attach_t) + t_wpn

    extract_viewmodel states it as qrot(wpn_bone_quat, loc -
    bind_applied_src) + wpn_bone_src, and recomputing `pos` that way
    agrees with its baked array to 3.1e-08 m. The swap is (wpn_bone_quat,
    wpn_bone_src) -- the VIEWMODEL rig's hold transform -- for the
    CHARACTER rig's `wpn` bone at this entity's clip time. Same socket,
    different rig, which is exactly what third person is.

    The attach rotation is applied as inverse(bind_attach_q) rather than
    assumed away. It is identity on ak47 and costs nothing to carry, and a
    weapon whose attach bone is authored rotated would otherwise be wrong
    in a way no test here would catch.

    RIGID, AND SAID SO. The weapon's OWN bones -- bolt, clip, trigger --
    are not driven: the world clip poses the CHARACTER rig, and the
    weapon-side motion lives in the weapon's own clips, which this pass
    does not sample. So a third-person rifle recoils WITH the hands and
    its bolt does not cycle. That is a tier, it is printed once, and it is
    not the same claim as "the weapon is animated".
    """
    skin = wb["skin"]
    pal = list(skin["palette"])
    ai = pal.index("weapon_offset")
    bt = skin["bind_t"][ai].numpy().astype(np.float64)
    bq = skin["bind_q"][ai].numpy().astype(np.float64)
    iq = _skin.qinv(bq)
    loc = wb["loc"].cpu().numpy().astype(np.float64)
    nrm = _pm_to_src_axes_np(wb["nrm"].cpu().numpy().astype(np.float64))
    tan = _pm_to_src_axes_np(wb["tan"].cpu().numpy().astype(np.float64))
    pos_l, nrm_l, tan_l = [], [], []
    any_placed = False
    for i in range(nslots):
        e = ents[i] if i < len(ents) else None
        got = None
        if e is not None and e.get("_clip") is not None:
            key = (e["_clip"]["clip"], round(float(e["_t"]), 4))
            wp = PM_WPN_POSE.get(key)
            if wp is not None:
                mq, mt = wp
                # x = pose_wpn o inverse(bind_attach): the SAME form
                # skin_anim.bone_transforms builds per palette bone. It is
                # written out here rather than called, because the join is
                # not by NAME -- the weapon's bone is `weapon_offset` and
                # the rig's socket is `wpn`, two different names for one
                # attachment, and bone_transforms joins by name by design.
                xq = _skin.qmul(mq, iq)
                xt = mt + _skin.qrot(mq, -_skin.qrot(iq, bt))
                XQ = np.broadcast_to(xq, (loc.shape[0], 4))
                got = (_skin.qrot(XQ, loc) + xt,
                       _skin.qrot(XQ, nrm), _skin.qrot(XQ, tan))
        if got is None:
            pos_l.append(None)
            nrm_l.append(None)
            tan_l.append(None)
        else:
            any_placed = True
            pos_l.append(torch.as_tensor(
                _pm_to_world_axes_np(got[0]), device=device,
                dtype=torch.float32))
            nrm_l.append(torch.as_tensor(
                _pm_to_world_axes_np(got[1]), device=device,
                dtype=torch.float32))
            tan_l.append(torch.as_tensor(
                _pm_to_world_axes_np(got[2]), device=device,
                dtype=torch.float32))
    if not any_placed:
        return None, None, None
    return pos_l, nrm_l, tan_l


def pm_script(ents, tick):
    """Choose (event, clip, t) for each entity of ONE frame, and SAY SO.

    PER ENTITY, KEYED ON STEAMID. anim_script.Scripter is per-entity by
    construction -- its distance accumulator, its one-shot latch and its
    previous row are all dicts keyed by whatever key the caller passes --
    so the only thing this has to get right is to pass a STABLE key and to
    consume rows SEQUENTIALLY. Both matter: a key that changed per frame
    would restart every walk cycle at phase 0, and a skipped row would
    lose the distance the entity travelled over it, which IS the phase.

    A frame the entity is absent from (dead, out of the row set, excluded)
    is a gap in that entity's sequence; the scripter's own dt comes from
    the TICK delta carried on the row, so the speed it derives is right
    across the gap rather than scaled by it.
    """
    # HV ground-truth: attach bone data to entities before clip selection
    hv_frame = _hv_frame_for_tick(tick) if HV_FRAMES else None
    cs = pm_clips()
    if cs is None and not hv_frame:
        return
    for e in ents:
        key = str(e.get("steamid") if e.get("steamid") is not None
                  else e.get("name") or e.get("team"))
        e.setdefault("tick", tick)
        # Try HV ground-truth bones first
        if hv_frame is not None:
            hv_p = _hv_match_entity(hv_frame, e)
            if hv_p is not None and hv_p.get("b"):
                e["_hv_bones"] = hv_p["b"]
                e["_ev"] = "hv_ground_truth"
                e["_clip"] = None
                e["_t"] = 0.0
                e["_why"] = ["HV ground-truth bones from hypervisor capture"]
                e["_static"] = False
                PM_EVENT_COUNT["hv"] = PM_EVENT_COUNT.get("hv", 0) + 1
                if not HV_NOTE:
                    HV_NOTE.append(1)
                    print(f"hv-capture: first HV pose applied at tick {tick}, "
                          f"{len(hv_p['b'])} bones", flush=True)
                wnm, whow = pm_weapon_for(e)
                e["_weapon"], e["_weapon_how"] = wnm, whow
                continue
        # Fall back to clip scripter
        if cs is None:
            continue
        prev = PM_PREV_ROW.get(key)
        ev, clip, t, why = PM_SCRIPT[0].choose(key, e, prev, cs)
        PM_PREV_ROW[key] = e
        if ev is None:
            e["_anim_why"] = why
            continue
        wpn_name = e.get("active_weapon_id")
        if wpn_name and hasattr(cs, '_weapon_tag'):
            ducking = e.get("duck_button") or e.get("duck_amount", 0) > 0.5
            better = cs.pick(ev, key=0, weapon=wpn_name, crouch=ducking)
            if better is not None:
                clip = better
        if not cs.moves(clip):
            # SAID ON THE LINE IT DRIVES. The event, the clip name and the
            # advancing t are all true here and all of them read as motion;
            # the one fact that contradicts them is that this clip's tracks
            # do not change across its own frames, so t buys nothing.
            why = list(why) + [
                "⚠️ STATIC CLIP: this clip's tracks are constant across all "
                f"{clip['frames']} of its frames, so the pose is an authored "
                "one HELD -- t advances and the vertices do not. Not a "
                "T-pose and not motion; pick() had no moving absolute "
                "sibling for this event."]
        e["_ev"], e["_clip"], e["_t"], e["_why"] = ev, clip, float(t), why
        e["_static"] = not cs.moves(clip)
        PM_EVENT_COUNT[ev] = PM_EVENT_COUNT.get(ev, 0) + 1
        # WHAT IS IN THE HAND, resolved on the same row that chose the
        # pose, so the two cannot disagree about which tick they describe.
        wnm, whow = pm_weapon_for(e)
        e["_weapon"], e["_weapon_how"] = wnm, whow
        _wk = str(e.get("active_weapon_id"))
        if _wk not in PM_WEAPON_SEEN:
            PM_WEAPON_SEEN[_wk] = whow
            print(f"playermodels: weapon {_wk!r} -> "
                  f"{wnm + '.pt' if wnm else 'NOT DRAWN'} ({whow})",
                  flush=True)


def pm_slot_geom(name, b, ents, nslots):
    """Per-slot (pos, nrm, tan) overrides, or None where an entity is BIND.

    One entry per SLOT, so the caller can keep its single shared topology:
    a slot whose entity has no chosen clip gets None and falls back to the
    bundle's own bind arrays at exactly the site it always used them.
    """
    if PM_CLIPS[0] is None and not HV_FRAMES:
        return None, None, None
    pos, nrm, tan = [], [], []
    any_posed = False
    for i in range(nslots):
        e = ents[i] if i < len(ents) else None
        got = None
        if e is not None and e.get("_hv_bones") is not None:
            got = pm_pose_arrays_hv(
                name, b, e["_hv_bones"],
                [e["x"], e["y"], e["z"]], e["yaw"])
        elif e is not None and e.get("_clip") is not None:
            got = pm_pose_arrays(name, b, e["_clip"], e["_t"])
        if got is None:
            pos.append(None)
            nrm.append(None)
            tan.append(None)
        else:
            any_posed = True
            pos.append(got[0])
            nrm.append(got[1])
            tan.append(got[2])
    if not any_posed:
        return None, None, None
    return pos, nrm, tan


def pm_place(pos_src, ents, nslots, per_slot=None):
    """(nslots*Nv, 3) world-space METRES for one frame's entities.

    Each entity is its own SLOT of a shared topology, so B frames with
    different player counts still rasterise in one call. Unused slots are
    COLLAPSED to a single point, which makes every triangle in them
    zero-area and therefore invisible -- never moved off-screen by a
    magic distance, which is a number that can be wrong.

    The axis chain, and it is the file's own (gpu_render.py:18, ray-cast
    validated): the bundle already holds (src_y, src_z, src_x) in SOURCE
    units, yaw rotates about source +Z, the entity's own (X, Y, Z) is
    added in source units, and 0.0254 is applied ONCE at the end -- at
    the same site and in the same expression as the camera's own
    `eye = [y*S, eye_z*S, x*S]`, so the model and the camera cannot drift
    into different units.

    `per_slot`, when given, is one POSED vertex array per slot in the SAME
    world-axis source-unit frame as `pos_src` -- the skin's output. A slot
    whose entry is None takes the shared bind array, so the animated and
    the unanimated entity go through ONE placement chain and cannot drift
    apart: the pose is the only thing that differs between them.
    """
    nv = pos_src.shape[0]
    out = pos_src.new_zeros((nslots, nv, 3))
    for i in range(nslots):
        if i >= len(ents):
            out[i] = 0.0            # degenerate: every triangle zero-area
            continue
        e = ents[i]
        src = pos_src
        if per_slot is not None and per_slot[i] is not None:
            src = per_slot[i]
        th = math.radians(float(e["yaw"]))
        c, s = math.cos(th), math.sin(th)
        a0, a1, a2 = src[:, 0], src[:, 1], src[:, 2]
        # a0 = src_y, a1 = src_z (up), a2 = src_x
        r0 = a2 * s + a0 * c
        r2 = a2 * c - a0 * s
        out[i, :, 0] = (r0 + float(e["y"])) * S
        out[i, :, 1] = (a1 + float(e["z"])) * S
        out[i, :, 2] = (r2 + float(e["x"])) * S
    return out.reshape(nslots * nv, 3)


def pm_rotate_dirs(dirs_src, ents, nslots, per_slot=None):
    """The same yaw applied to a direction: rotation only, no translation.

    THE POSED NORMAL, NOT THE BIND ONE. A skinned vertex moves and the
    surface it belongs to TURNS with it; leaving the normals at bind while
    the positions animate lights an arm that has swung 90 degrees as
    though it had not moved, and the error is invisible in a silhouette
    check because the silhouette is right. So `per_slot` carries the
    skin's own blended directions and this applies only the entity yaw on
    top, exactly as it does for the bind case.
    """
    nv = dirs_src.shape[0]
    out = dirs_src.new_zeros((nslots, nv, 3))
    for i in range(nslots):
        if i >= len(ents):
            continue
        src = dirs_src
        if per_slot is not None and per_slot[i] is not None:
            src = per_slot[i]
        th = math.radians(float(ents[i]["yaw"]))
        c, s = math.cos(th), math.sin(th)
        a0, a1, a2 = src[:, 0], src[:, 1], src[:, 2]
        out[i, :, 0] = a2 * s + a0 * c
        out[i, :, 1] = a1
        out[i, :, 2] = a2 * c - a0 * s
    return out.reshape(nslots * nv, 3)


def _pm_batch_src(base_src, ents_all, per_slot_all, B, nslots):
    """(B, nslots, nv, 3) source stack + (B, nslots) cos/sin/xyz tensors.

    The batched halves of pm_place/pm_rotate_dirs share this: gather every
    frame's per-slot source (posed skin where the cache produced one, the
    shared bind array otherwise) into ONE stack, and every scalar the loop
    read per entity (yaw trig, translation) into (B, nslots) tensors. Slots
    past a frame's entity count keep zero sources -- the same collapsed
    degenerate-triangle convention as the loop.
    """
    nv = base_src.shape[0]
    dev = base_src.device
    src = base_src.new_zeros((B, nslots, nv, 3))
    cs = torch.zeros((B, nslots, 2), device=dev)
    txyz = torch.zeros((B, nslots, 3), device=dev)
    live = torch.zeros((B, nslots), dtype=torch.bool, device=dev)
    for b in range(B):                       # host bookkeeping only:
        ents = ents_all[b]                   # cache lookups + scalar reads;
        ps = per_slot_all[b] if per_slot_all is not None else None
        for i in range(min(nslots, len(ents))):
            e = ents[i]
            src[b, i] = (ps[i] if ps is not None and ps[i] is not None
                         else base_src)
            th = math.radians(float(e["yaw"]))
            cs[b, i, 0] = math.cos(th)
            cs[b, i, 1] = math.sin(th)
            txyz[b, i, 0] = float(e["x"])
            txyz[b, i, 1] = float(e["y"])
            txyz[b, i, 2] = float(e["z"])
            live[b, i] = True
    return src, cs, txyz, live, nv


def pm_place_batch(pos_src, ents_all, B, nslots, per_slot_all=None):
    """Batched pm_place over a whole window: (B, nslots*nv, 3) in ~6 tensor
    launches instead of B calls x nslots slot-loops (HOSTLOOP_TOP10 row 1).
    ELEMENTWISE-IDENTICAL expressions to pm_place -- r0 = a2*s + a0*c etc.
    broadcast over (B, nslots) -- so equivalence is exact per element."""
    src, cs, txyz, live, nv = _pm_batch_src(pos_src, ents_all, per_slot_all,
                                            B, nslots)
    c = cs[..., 0, None]
    s = cs[..., 1, None]
    a0, a1, a2 = src[..., 0], src[..., 1], src[..., 2]
    r0 = a2 * s + a0 * c
    r2 = a2 * c - a0 * s
    out = torch.stack(((r0 + txyz[..., 1, None]) * S,
                       (a1 + txyz[..., 2, None]) * S,
                       (r2 + txyz[..., 0, None]) * S), dim=-1)
    out = out * live[..., None, None]        # dead slots -> collapsed point
    return out.reshape(B, nslots * nv, 3)


def pm_rotate_dirs_batch(dirs_src, ents_all, B, nslots, per_slot_all=None):
    """Batched pm_rotate_dirs: rotation only, same expressions broadcast."""
    src, cs, _txyz, live, nv = _pm_batch_src(dirs_src, ents_all,
                                             per_slot_all, B, nslots)
    c = cs[..., 0, None]
    s = cs[..., 1, None]
    a0, a1, a2 = src[..., 0], src[..., 1], src[..., 2]
    out = torch.stack((a2 * s + a0 * c, a1, a2 * c - a0 * s), dim=-1)
    out = out * live[..., None, None]
    return out.reshape(B, nslots * nv, 3)


def pm_bundle_name_for(e):
    """(bundle basename, how) for one entity. Never silently substitutes.

    Three tiers, in order, and the caller prints which one fired:
      1. --playermodel-agent-map, the operator's own handle/name -> bundle
      2. the agent value used AS a bundle name, which only works when the
         demo gave a NAME and someone extracted `<name>.pt`
      3. the entity's TEAM -- ct.pt / t.pt -- as a FALLBACK

    Tier 3 is why the frame line reports requested-vs-drew. With only two
    bundles built it is what almost every entity takes, and "every CT is
    ctm_sas" must never read as a fact about the demo when it is a fact
    about which bundles exist.
    """
    a = e.get("agent")
    if a is not None and a in PM_AGENT_MAP:
        return PM_AGENT_MAP[a], f"agent {a} via --playermodel-agent-map"
    if a is not None and _pm_bundle(a) is not None:
        return a, f"agent {a} named a bundle directly"
    t = e.get("team")
    why = ("no agent on the wire" if a is None
           else f"agent {a} has no bundle")
    return t, f"TEAM FALLBACK to {t}.pt -- {why}"


def pm_entities_for_frame(fidx):
    """(entities, reason) for one frame index. Never guesses a tick."""
    tick = TICK_OF_FRAME.get(fidx)
    if tick is None:
        return [], None, (f"frame {fidx} has no tick in TICK_OF_FRAME -- the "
                          f"camera rows carried no `tick`, so no demo row "
                          f"can be joined to it")
    rows = PM_ROWS.get(tick)
    if rows is None:
        return [], tick, (f"tick {tick} has no rows in {PM_SOURCE} -- the "
                          f"entity source does not cover this tick")
    out = []
    for e in rows:
        if e.get("steamid") is not None:
            _sid_e = int(e["steamid"])
            _own = (AGENT_EGO.get(ROW_AGENT[fidx])
                    if 0 <= fidx < len(ROW_AGENT) else None)
            # session: exclude only THIS frame's own subject (per-agent);
            # single-camera: PM_EGO carries the camera json's steamid.
            if _sid_e in PM_EGO or _sid_e == _own:
                continue
        if not e.get("is_alive", True):
            # THE DEAD ARE AN ENTITY CLASS AND THIS PASS DOES NOT DRAW
            # THEM -- an ACQUISITION task, NOT a missing input.
            #
            # ⚠️ THE PREVIOUS VERSION OF THIS NOTE WAS WRONG, and wrong in
            # the direction that closes a question: it said the ragdoll's
            # position after the kill tick is something "no demo column
            # carries", which reads as "the wire does not have it" and
            # retires the work. The wire DOES have it --
            # CBaseAnimGraph.m_RagdollPose (PhysicsRagdollPose_t.
            # m_Transforms), the one transform array that survives across
            # eras -- and #102 established that entity state at 14174 is
            # decodable (30,196 entity updates read by an independent
            # decoder). What is missing is a DECODER we can call:
            # demoparser2's entity path is the broken piece, and
            # demoinfocs-golang is the shown-working alternative. That is
            # a different kind of gap and it belongs on a work list, not
            # in a justification.
            #
            # The pose side is already here: 10 death clips, all absolute,
            # all animated. So the corpse is one decode away, and until
            # that decode exists a corpse posed at its last STANDING X/Y/Z
            # would be drawn where the body is not -- worse than absent
            # and harder to see. That is why the entity is skipped today,
            # and it is the only part of the old reasoning that survives.
            if not PM_DEAD_NOTE:
                PM_DEAD_NOTE.append(1)
                print("PLAYERMODELS (a declared GAP with a known route): "
                      "entities with is_alive=0 are NOT drawn, animated or "
                      "otherwise. "
                      f"{len(PM_CLIPS[0].events.get('death', ())) if PM_CLIPS[0] else 0} "
                      "death clip(s) are decoded and available, and the "
                      "ragdoll's own transforms ARE on the wire "
                      "(CBaseAnimGraph.m_RagdollPose). What is missing is a "
                      "decoder this renderer can call: demoparser2's entity "
                      "path does not return it, demoinfocs-golang is the "
                      "shown-working alternative (#102). Until then a corpse "
                      "would have to be posed at its last STANDING position, "
                      "which draws it where the body is not.", flush=True)
            continue
        nm, how = pm_bundle_name_for(e)
        if _pm_bundle(nm) is None:
            continue
        # carried on the entity so the frame line can print REQUESTED vs
        # DREW without resolving twice
        e = dict(e, _bundle=nm, _bundle_how=how, tick=tick)
        out.append(e)
    return out, tick, None
