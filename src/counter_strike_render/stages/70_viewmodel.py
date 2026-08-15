

def viewmodel_pass(mvp, eye, world_rgb):
    """Rasterise the viewmodel in ITS OWN depth range and composite.

    `world_rgb` is the finished world frame, (B, height, width, 3) uint8.
    The viewmodel is composited over it by COVERAGE: no comparison is made
    against the world's depth buffer at any point, which is the property
    that stops a viewmodel clipping into world geometry.

    Order inside the pass, and it is the pass structure the bytecode
    implies:
      1. csgo_legs_prepass writes viewmodel depth and its constant
         near-black colour, with its own fade and cone test.
      2. csgo_weapon draws against THAT depth buffer -- the weapon is
         occluded by the legs, and neither is occluded by the world.

    --viewmodel-depth-range world collapses (1) and (2) into the world's
    single camera and single range. It is a DIAGNOSTIC for measuring the
    separation, never a default.
    """
    geom = (vm_load_geometry(args.viewmodel_model) if args.viewmodel_model
            else vm_synth_geometry(args.viewmodel_synth))
    B = mvp.shape[0]
    Hh, Ww = H, W
    if geom is None:
        VM_REACH["coverage"] = (0.0, float(B * Hh * Ww))
        _vm_frame_report(B, absent="no viewmodel geometry source -- neither "
                                   "--viewmodel-model nor --viewmodel-synth "
                                   "supplied a mesh")
        return world_rgb
    pos_v, uv4, nrm_v, tan_v, loc_v, tri, fam_of_tri, is_legs = geom
    # ORDER MATTERS: the clip must be CHOSEN before the geometry can be
    # posed with it. The first version reported the choice after the draw
    # and re-posed before it, so VM_ANIM_CLIP was still None and every
    # frame silently rendered the bind pose while the log said `shoot`.
    _vm_anim_report(B)
    # (N,3) -> (B,N,3): one pose per FRAME. vm_transform is rank-agnostic
    # already (it broadcasts the roll/scale/offset and `is_legs` over the
    # leading axis), so only the two einsums below had to learn the batch.
    pos_v = _vm_repose(pos_v, loc_v, tri, fam_of_tri, B)
    pos_v = vm_transform(pos_v, is_legs)
    rng = VM_RANGE if args.viewmodel_depth_range == "own" else WORLD_RANGE
    # The viewmodel is defined in VIEW space, so its clip position needs
    # only this range's projection. The WORLD position -- which the legs
    # fade, the proximity dissolve and the lighting all need -- comes from
    # inverting the view matrix, recovered exactly from mvp: the world
    # pass built mvp = proj @ view with a known, invertible proj.
    Pw = np.linalg.inv(WORLD_RANGE.proj.astype(np.float64))
    view_m = torch.from_numpy(Pw).to(device).float()[None] @ mvp.float()
    Rv = view_m[:, :3, :3]
    tv = view_m[:, :3, 3]
    # wpos = R^T (p_view - t)
    wpos_v = torch.einsum("bji,bnj->bni", Rv, pos_v) \
        - torch.einsum("bji,bj->bi", Rv, tv)[:, None, :]
    nrm_w = torch.einsum("bji,nj->bni", Rv, nrm_v)
    tan_w = torch.einsum("bji,nj->bni", Rv, tan_v)
    # ALREADY (B,N,4): the re-pose varies along the batch dim, so there is
    # nothing to expand -- expanding here is exactly what made every frame
    # in a batch draw the same pose.
    clip = rng.clip_of_view(pos_v).contiguous()
    rast, _db = dr.rasterize(ctx, clip, tri, (Hh, Ww))
    covered = rast[..., 3] > 0
    # THE SECOND DEPTH BUFFER, now written through this range's VIEWPORT
    # DEPTH REMAP rather than as raw NDC z.
    #
    # `rast[..., 2]` is nvdiffrast's NDC z in [-1, 1]. The reference writes
    # the viewmodel into VkViewport minDepth/maxDepth [0, 0.1] -- READ from
    # wallA2 -- so the value that lands in this pass's depth buffer is
    # window depth in that slice, which is what `window_depth()` computes.
    #
    # ⚠️ RETRACTED, and the retraction is measured. This comment used to say
    # the remap "has a consumer" because the weapon/legs depth test below
    # reads this buffer. THAT JUSTIFICATION IS FALSE.
    #
    # impl-viewmodel built the falsifier I should have built (9c5b3b67):
    # three arms, one frame, everything else pinned, and arm C forces the
    # depth test to reject EVERY weapon fragment.
    #
    #     A  as landed          window_depth(rast_z) <= depth_vm
    #     B  the units mismatch rast_z <= depth_vm
    #     C  reject everything  nearer = False
    #
    #     A vs B   0 of 518,400 px changed
    #     A vs C   0 of 518,400 px changed
    #
    # Arm C settles it: forcing the test to reject everything changes no
    # pixel, so it rejects nothing and its result is discarded. The pass
    # issues ONE rasterisation of weapon and legs together and splits per
    # pixel by the frontmost triangle's family, so m_wpn and m_legs are
    # disjoint by construction, `live` is a subset of m_legs, and
    # `w_keep & (nearer | ~live)` reduces to `w_keep` whatever `nearer`
    # holds. At a weapon pixel the comparison is `x <= x` on bit-identical
    # floats -- the rasteriser already did the occlusion and the depth test
    # re-does it against itself.
    #
    # So the remap's only consumer today is the _ts.observe statistic below.
    # That is the inert-term pattern this ticket has spent its time finding
    # in other people's code, arriving in my own landing, and it is left
    # STATED rather than quietly deleted because the code is still correct
    # and still what the reference does -- what is missing is the second
    # draw that would make the test do work.
    #
    # THE STRUCTURAL FIX, which is not a comment change: the reference draws
    # these SEPARATELY. csgo_legs_prepass is its own shader on its own
    # geometry, and csgo_weapon ships S_MODE_DEPTH as its own 26-line
    # depth-only module (record 192; committed with real binding decorations
    # at docs/projects/counter-strike-sft/reference/csgo_weapon/
    # csgo_weapon__mode_depth1_c1024_d2.glsl). TWO draws into one depth
    # buffer is what makes an inter-draw depth test mean anything; one
    # rasterisation cannot. Until the pass splits them, this buffer is
    # written and never read.
    #
    # The guardrail exists and will say when that changes:
    # iji_model/counter_strike_render/vm_depth_merge_falsifier.py goes NON-ZERO the day
    # the families rasterise separately, at which point this conversion
    # starts doing work and must be re-reviewed.
    depth_vm = torch.where(covered, rng.window_depth(rast[..., 2]),
                           torch.full_like(rast[..., 2], 1e9))
    _ts.observe("viewmodel.viewport_depth_remap",
                rng.window_depth(rast[..., 2]),
                before=rast[..., 2], mask=covered,
                reference="wallA2 VkViewport minDepth/maxDepth "
                          "[0, 0.10000000149011612]")
    VM_REACH["coverage"] = (float(covered.sum()), float(covered.numel()))
    if not bool(covered.any()):
        _vm_frame_report(B, absent=f"{VM_BUNDLE_INFO.get('name', 'geometry')} "
                                   f"loaded and rasterised to ZERO covered "
                                   f"pixels -- the placement puts it outside "
                                   f"the {VM_RANGE.hfov:g} deg viewmodel "
                                   f"frustum, not a missing asset")
        return world_rgb
    tri_i = (rast[..., 3].long() - 1).clamp(min=0)
    fam_pix = fam_of_tri[tri_i]
    uvp, _ = dr.interpolate(uv4[None].expand(B, -1, -1).contiguous(),
                            rast, tri)
    # MODEL-LOCAL, in SOURCE UNITS -- the space the legs cone test and the
    # SFX phase term are written in.
    lposp, _ = dr.interpolate(loc_v[None].expand(B, -1, -1).contiguous(),
                              rast, tri)
    wposp, _ = dr.interpolate(wpos_v.contiguous(), rast, tri)
    nrmp, _ = dr.interpolate(nrm_w.contiguous(), rast, tri)
    tanp, _ = dr.interpolate(tan_w.contiguous(), rast, tri)
    # --- the per-material side-table row, through the SAME dispatch ----
    # MAT_FAM / MAT_EXT2 are keyed by material id. The synthesised draw
    # has no packed material, so it takes the first material of its own
    # family if the pack has one and otherwise the family DEFAULTS built
    # by fast_pack_fam.py -- which is why those columns exist. Either way
    # the row is a real ext2 row read through _e2(), not a literal.
    x_row = vm_ext2_row()
    x = x_row.expand(B, Hh, Ww, -1)
    eye_b = eye
    view = eye[:, None, None, :] - wposp
    view = view / view.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cam_fwd = -Rv[:, 2, :][:, None, None, :].expand_as(view)
    cam_up = Rv[:, 1, :][:, None, None, :].expand_as(view)
    sun_dir = _vm_sun_dir().expand_as(view)
    sun_rgb = _vm_sun_rgb()
    ambient = _vm_ambient()
    yy, xx = torch.meshgrid(torch.arange(Hh, device=device,
                                         dtype=torch.float32),
                            torch.arange(Ww, device=device,
                                         dtype=torch.float32),
                            indexing="ij")
    frag_xy = torch.stack([xx, yy], -1)[None].expand(B, -1, -1, -1)
    rgb = torch.zeros(B, Hh, Ww, 3, device=device)
    alpha = torch.zeros(B, Hh, Ww, device=device)
    live = torch.zeros(B, Hh, Ww, dtype=torch.bool, device=device)
    tsec = torch.zeros(B, device=device)
    # ---- FAMILY DISPATCH ---------------------------------------------
    # Two families, selected per pixel by the family id the side table
    # carries -- the same MAT_FAM mechanism the world classes use.
    m_legs = covered & (fam_pix == VM_FAM_LEGS)
    m_wpn = covered & (fam_pix == VM_FAM_WEAPON)
    m_arms = covered & (fam_pix == VM_FAM_ARMS)
    if args.viewmodel_legs_prepass and bool(m_legs.any()):
        l_rgb, l_keep = legs_prepass_shade(wposp, lposp, eye_b, m_legs, x)
        rgb = torch.where(l_keep.unsqueeze(-1), l_rgb, rgb)
        alpha = torch.where(l_keep, torch.ones_like(alpha), alpha)
        live = live | l_keep
        # the legs establish viewmodel depth; a discarded fragment does
        # not, which is the whole point of a screen-door in a depth pass
        depth_vm = torch.where(m_legs & ~l_keep,
                               torch.full_like(depth_vm, 1e9), depth_vm)
    if args.weapon and bool(m_wpn.any()):
        bg = None
        if args.weapon_bgrefract != "off":
            # the ENGINE RENDER TARGET the refraction samples: the world
            # frame this pass composites over, supplied as the bindless
            # 2D handle's contents rather than as a scene-colour sampler.
            bg = torch.cat([world_rgb[0].float() / 255.0,
                            torch.ones_like(world_rgb[0][..., :1].float())],
                           dim=-1)
        w_rgb, w_a, w_keep = weapon_shade(
            uvp, wposp, lposp, nrmp, tanp, view, eye_b, sun_dir, sun_rgb,
            ambient, tsec, frag_xy, cam_up, cam_fwd, m_wpn, x, bg)
        # the weapon tests the VIEWMODEL depth buffer -- the legs occlude
        # it -- and nothing in this pass tests world depth.
        #
        # BOTH SIDES IN WINDOW DEPTH. `depth_vm` is written through the
        # viewport remap above, so comparing raw NDC z against it would be
        # a units mismatch -- the exact class of defect the decal pass's
        # y-origin turned out to be, arriving here as a direct consequence
        # of changing what the buffer holds. The remap is monotonic, so the
        # ORDER is unchanged and this comparison keeps its meaning; it is
        # written explicitly anyway because "monotonic so it does not
        # matter" is how a units mismatch survives review.
        nearer = rng.window_depth(rast[..., 2]) <= depth_vm
        w_keep = w_keep & (nearer | ~live)
        if args.weapon_visibility_stencil == "proxy":
            # player_visibility_stencil_proxy has no world-family
            # counterpart and this renderer has no stencil buffer. The
            # equivalent COVERAGE MASK is the legs' own coverage: the
            # weapon is kept out of the player-occluded region. Stated
            # difference: a coverage mask, not a hardware stencil.
            w_keep = w_keep & ~(m_legs & live)
            _wpn_reach("player_visibility_stencil_proxy", w_keep,
                       int(m_wpn.sum()))
        # The weapon's own shading output, over the pixels it kept. The DoD
        # for this ticket asks for each component's non-identity output, and
        # a coverage count cannot supply it: weapon_shade runs on fallback
        # constants for every slot vm_textures() has not been fed a real page
        # for (12 of 13 today, #28), so "the weapon covered N pixels" is
        # compatible with "the weapon shaded N pixels a flat constant".
        _ts.observe("viewmodel.weapon.shade", w_rgb, before=rgb, mask=w_keep,
                    reference="vcs/nonworld_ref/wp/s0_d2.glsl")
        rgb = torch.where(w_keep.unsqueeze(-1), w_rgb, rgb)
        alpha = torch.where(w_keep, w_a, alpha)
        live = live | w_keep
    elif bool(m_wpn.any()):
        # THE DRAW WITHOUT --weapon. Not a fallback for a failed one: the
        # twelve csgo_weapon axes are SELECTED by --weapon and they are what
        # this branch declines to run, because on a pack with no csgo_weapon
        # material their parameters come from vm_ext2_row's defaulted row
        # and their pages come from the seed-0x57C1 synthesis. Shading a
        # real extracted mesh with those and compositing it would put
        # fabricated skin parameters into a scored frame.
        #
        # So this is the shared lighting over the bundle's own pages, and
        # the two branches are exclusive rather than additive: whichever
        # ran, the weapon family's pixels are shaded exactly once.
        w_rgb, w_a, w_keep = vm_gt_shade(uvp, nrmp, tanp, view, m_wpn,
                                         wpos=wposp)
        _vm_reach("gt_direct_shade", m_wpn, int(covered.sum()))
        # The same viewmodel-depth test the --weapon branch makes, for the
        # same reason and with the same caveat: the pass issues ONE
        # rasterisation, so today this rejects nothing (arm C of 9c5b3b67
        # changed 0 of 518,400 px). It is written because the day the
        # families rasterise separately it starts doing work, and a test
        # that is absent then is a bug nobody looks for.
        nearer = rng.window_depth(rast[..., 2]) <= depth_vm
        w_keep = w_keep & (nearer | ~live)
        _ts.observe("viewmodel.gt_direct.shade", w_rgb, before=rgb,
                    mask=w_keep,
                    reference="gt_direct() at S_SHADER_QUALITY "
                              f"{args.shader_quality}, sun READ from the "
                              "map's light_environment, over the bundle's "
                              "own colour/normal/mask/ao pages")
        rgb = torch.where(w_keep.unsqueeze(-1), w_rgb, rgb)
        alpha = torch.where(w_keep, w_a, alpha)
        live = live | w_keep
    # --- THE ARMS -----------------------------------------------------
    # Their own family, their own pages, and NOT --weapon-gated. The
    # csgo_weapon axes describe a rifle's skin -- tint mask, glitter,
    # stickers, opaque refract -- and none of them is a thing skin or a
    # glove does, so there is no arm of this pass in which the arms should
    # run through weapon_shade. They take the shared lighting over the
    # arms bundle's own colour/normal/ao pages, in the SAME viewmodel depth
    # slice as the weapon, which is what keeps a hand in front of the grip
    # rather than intersecting it.
    if bool(m_arms.any()):
        a_rgb, a_a, a_keep = vm_gt_shade(uvp, nrmp, tanp, view, m_arms,
                                         tex=_VM_ARMS_TEX[0], wpos=wposp)
        nearer_a = rng.window_depth(rast[..., 2]) <= depth_vm
        a_keep = a_keep & (nearer_a | ~live)
        _vm_reach("arms.gt_direct_shade", a_keep, int(covered.sum()))
        _ts.observe("viewmodel.arms.shade", a_rgb, before=rgb, mask=a_keep,
                    reference="gt_direct() over the arms bundle's own "
                              "pages; glove_fullfinger skinned to the "
                              "weapon's idle clip, placement baked by "
                              "extract_arms")
        rgb = torch.where(a_keep.unsqueeze(-1), a_rgb, rgb)
        alpha = torch.where(a_keep, a_a, alpha)
        live = live | a_keep
    if not bool(live.any()):
        VM_REACH["composited"] = (0.0, float(covered.numel()))
        _vm_frame_report(B, absent=f"{VM_BUNDLE_INFO.get('name', 'geometry')} "
                                   f"covered {int(covered.sum())} raster px "
                                   f"and SHADED NONE -- every fragment was "
                                   f"discarded by the family dispatch or the "
                                   f"screen-door")
        return world_rgb
    VM_REACH["composited"] = (float(live.sum()), float(covered.numel()))
    # per family, per frame -- the arms and the weapon are scored separately
    _fam_px = [{k: float((live[i] & (fam_pix[i] == k)).sum())
                for k in (VM_FAM_WEAPON, VM_FAM_LEGS, VM_FAM_ARMS)}
               for i in range(B)]
    if args.vm_selftest_drop_class != "none":
        _dropk = {"weapon": VM_FAM_WEAPON, "arms": VM_FAM_ARMS,
                  "legs": VM_FAM_LEGS}[args.vm_selftest_drop_class]
        print(f"VIEWMODEL SELFTEST: dropping class "
              f"{args.vm_selftest_drop_class} from the composite ON PURPOSE. "
              f"This frame is WRONG by construction and exists only to make "
              f"the class assertion fire.", flush=True)
        live = live & (fam_pix != _dropk)
        _fam_px = [{k: float((live[i] & (fam_pix[i] == k)).sum())
                    for k in (VM_FAM_WEAPON, VM_FAM_LEGS, VM_FAM_ARMS)}
                   for i in range(B)]
    _wm = live & (fam_pix == VM_FAM_WEAPON)
    if bool(_wm.any()):
        _idx = torch.nonzero(_wm[0])
        if _idx.numel():
            # Y IS FLIPPED INTO IMAGE SPACE HERE, and it has to be. The
            # mask lives in RASTER space; the composite below does
            # `img = rgb.flip(1)`, and the flash's predicted pixel is in
            # the flipped, final-image space. Recording the raw raster row
            # would put the two operands of the comparison in two
            # different y conventions -- which is the same wrong-operand
            # error the comment at the print site warns about, committed
            # at the point that comment was written. Caught because the
            # weapon read y 0..275 while the flash predicted y 506: a gun
            # cannot be at the top of a viewmodel frame.
            _H = _wm.shape[1]
            _r0, _r1 = int(_idx[:, 0].min()), int(_idx[:, 0].max())
            _ss = max(1, args.supersample)
            VM_WPN_BBOX.clear()
            VM_WPN_BBOX.update(y0=(_H - 1 - _r1) // _ss,
                               y1=(_H - 1 - _r0) // _ss,
                               x0=int(_idx[:, 1].min()) // _ss,
                               x1=int(_idx[:, 1].max()) // _ss)
    # ACCUMULATE, so the run has a corpus number and not just N frame
    # lines. Summed over frames exactly as CLASS_PX is (#103).
    for _fp in _fam_px:
        for _k, _v in _fp.items():
            VM_CLASS_PX[int(_k)] = VM_CLASS_PX.get(int(_k), 0) + int(_v)
    # ---- AND CLAIM THEM IN THE BALANCE LAW'S INDEPENDENT OPERAND -------
    # class_px_won is the operand recorded at the line that decides the
    # IMAGE, deliberately separate from the attribution map. The viewmodel
    # composites here, so this IS that line for these pixels, and until now
    # the pass reported into neither operand -- which is why csgo_weapon
    # could read 0:0:1 in eight parity fixtures with a gun plainly on screen.
    #
    # THE JOIN IS THE READ, not a guess: VM_FAM_VFX maps each viewmodel
    # family to the .vfx the REFERENCE binds to that geometry, and
    # csgo_weapon.vfx came from weapon_rif_ak47.vmat's own m_shaderName.
    # A viewmodel family whose .vfx this pack's vocabulary does not carry is
    # SKIPPED WITH ITS COUNT rather than dropped silently.
    if FAM_NAMES and bool(live.any()):
        _vm_skipped = {}
        _wfam = torch.full_like(fam_pix, FAM_UNATTRIBUTED)
        for _k in (VM_FAM_WEAPON, VM_FAM_LEGS, VM_FAM_ARMS):
            _nm = VM_FAM_VFX.get(_k, "").split(" (")[0]
            if _nm in FAM_NAMES:
                _wfam = torch.where(fam_pix == _k, FAM_NAMES.index(_nm),
                                    _wfam)
            else:
                _n = int((live & (fam_pix == _k)).sum())
                if _n:
                    _vm_skipped[_nm or "?%d" % _k] = _n
        class_px_won(_wfam, live)
        for _nm, _n in sorted(_vm_skipped.items()):
            print(f"viewmodel: {_n:,} px of family {_nm} NOT claimed -- this "
                  f"pack's fam_names has no id for it, so the census cannot "
                  f"name the class. Counted here so the skip has a number.",
                  flush=True)
    _vm_frame_report(B, covered=covered, live=live, fam_px=_fam_px)
    _vm_class_assert(_fam_px)
    # --- composite: COVERAGE, never a world-depth comparison -----------
    a = (alpha * live.float()).unsqueeze(-1)
    img = rgb.flip(1)
    a = a.flip(1)
    if args.supersample > 1:
        img = torch.nn.functional.avg_pool2d(
            img.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        a = torch.nn.functional.avg_pool2d(
            a.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
    base = world_rgb.float() / 255.0
    out = base * (1.0 - a) + img.clamp(0.0, 1.0) * a
    _ts.observe("viewmodel.composite", out, before=base, mask=(a > 0),
                reference="coverage composite; the viewmodel never tests "
                          "world depth")
    _before_flash = out
    # depth_vm and live are the viewmodel's OWN depth buffer and its drawn
    # coverage, both at supersample resolution -- the operands the flash
    # has to test against and which only exist inside this pass (#87).
    out = _muzzle_flash_over(out, rng, depth_vm, live, fam_pix)
    # The flash is gated on a REAL fire event off the wire (FIRE[] built from
    # the camera path's `fire` column, widened by the 0.05 s lifetime READ
    # from uweapon_muzflsh_ak47_primaryflash.vpcf_c). On a frame outside the
    # window this observes zero pixels and says so, which is the difference
    # between "the flash is off because nobody fired" and "the flash is
    # broken" -- two states a coverage print collapses into one.
    _ts.observe("viewmodel.muzzle_flash", out, before=_before_flash,
                reference="muzzle_flash.py LIFE_DURATION_S 0.05, attachment "
                          "READ from the bundle")
    return (out.clamp(0, 1) * 255).to(torch.uint8)


MZ_INFO = {}
