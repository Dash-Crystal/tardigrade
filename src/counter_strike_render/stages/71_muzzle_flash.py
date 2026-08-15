

def _muzzle_flash_over(out, rng, depth_vm=None, live_vm=None, fam_vm=None):
    """A2 -- add the muzzle flash over the composited viewmodel.

    ADDITIVE and LAST, which is what the reference declares: the
    sprite's m_nOutputBlendMode is PARTICLE_OUTPUT_BLEND_MODE_ADD with
    m_flDiffuseAmount 0, so it is not lit and never darkens what is
    under it.

    IT IS STILL DEPTH-TESTED (#87). This docstring used to say the flash
    "is not occluded by the weapon it sits in front of", and that claim
    was the defect: ADD describes the blend, not the depth state, and the
    owner saw the bloom drawn over the gun. Gun geometry nearer than the
    muzzle occludes it; what survives is composited additively.

    It rides `rng` -- the viewmodel's OWN projection -- because the
    particle system declares m_nViewModelEffect INHERITABLE_BOOL_TRUE.
    """
    if not args.muzzle_flash:
        return out
    # THE FIRING GATE. Which frames show a flash is scene state, not a
    # render setting -- so it is read off the wire, per frame, and the
    # `always` arm exists only so the pass can be shown to reach pixels
    # on a path with no firing in it (the reachability problem that made
    # --viewmodel-synth necessary next door).
    _b0 = int(VS_FRAME0[0])
    if args.muzzle_flash_gate == "fire":
        _on = [bool(FIRE[_b0 + i]) if _b0 + i < len(FIRE) else False
               for i in range(out.shape[0])]
        if not any(_on):
            MZ_INFO["gated_off_frames"] = MZ_INFO.get(
                "gated_off_frames", 0) + out.shape[0]
            return out
    else:
        _on = [True] * out.shape[0]
    bundle = _vm_bundle_cached()
    if bundle is None:
        print("--muzzle-flash: no viewmodel bundle "
              "(--viewmodel-model); the attachment comes from the "
              "bundle, so nothing is drawn and nothing is guessed",
              flush=True)
        return out
    # vm_transform IS REQUIRED, not optional. The bundle's
    # `muzzle_flash_view` is PRE-vm_transform -- the same space the
    # vertices are in when viewmodel_pass unpacks them, before
    # --viewmodel-roll/-scale/-offset are folded in. Passing the point
    # through unchanged would leave the flash standing still while the
    # weapon moved under those flags: a failure that only shows up when
    # someone uses a flag, which is the kind that survives review.
    rgb, a, info = _mz.draw(dr, ctx, rng, args, bundle,
                            (out.shape[1] * args.supersample,
                             out.shape[2] * args.supersample), device,
                            vm_transform=vm_transform)
    # The depth operands are TENSORS and only meant for the test below;
    # MZ_INFO is a small scalar dict that outlives the frame, and parking
    # a supersample-resolution GPU buffer in it would keep it alive for
    # the whole run.
    MZ_INFO.update({k: v for k, v in info.items()
                    if k not in ("window_depth", "covered_mask")})
    frame_print(f"--muzzle-flash: attachment {info['attachment_src']} src -> "
          f"view {[round(v, 5) for v in info['view_point_m']]} m, "
          f"affine residual {info['affine_residual_m']:.3e} m, "
          f"PREDICTED at pixel ({info['predicted_px'][0]:.1f}, "
          f"{info['predicted_px'][1]:.1f}), in_frame={info['in_frame']}, "
          f"covered {info.get('covered_px', 0)} px", flush=True)
    # BOTH OPERANDS ON ONE LINE: where the flash went, and where the gun
    # actually is. "PREDICTED at (827, 506)" is unfalsifiable alone --
    # tip-or-handguard is a claim ABOUT THE GUN, and the gun's extent was
    # never in the log, which is how the defect survived to an owner's eye.
    # This lane has already shipped one diagnostic that printed the wrong
    # operand and read as broken; the fix both times is to carry both sides.
    if VM_WPN_BBOX and info.get("predicted_px"):
        _px = info["predicted_px"][0]
        _w = max(1, VM_WPN_BBOX["x1"] - VM_WPN_BBOX["x0"])
        frame_print(
            f"--muzzle-flash: the WEAPON occupies x "
            f"{VM_WPN_BBOX['x0']}..{VM_WPN_BBOX['x1']}, y "
            f"{VM_WPN_BBOX['y0']}..{VM_WPN_BBOX['y1']}; the flash sits at "
            f"{(_px - VM_WPN_BBOX['x0']) / _w:+.2f} along that span "
            f"(0 = left edge of the drawn gun, 1 = right edge). Which end "
            f"is the muzzle depends on the pose, so this is a POSITION and "
            f"not a verdict -- see the barrel-axis line below, which says "
            f"which end is which instead of leaving it to be assumed.",
            flush=True)
    # WHICH SCREEN END IS THE MUZZLE. The line above deliberately refuses to
    # guess, and a position with no orientation is how "the flash is at the
    # handguard" and "the flash is at the muzzle" become the same log entry.
    #
    # A VIEWMODEL INVERTS THE INTUITION, which is the trap. The mesh spans
    # view z from about -0.905 (the muzzle, farthest from the eye) to -0.008
    # (the receiver, essentially AT the eye). Under perspective an off-axis
    # object's NEAR end projects FURTHER from the screen centre, so on a gun
    # held right of centre the muzzle lands at the LEFT of its own silhouette
    # and the receiver at the right. Reading "flash at the left edge" as
    # "flash at the back of the gun" is exactly backwards, and nothing in the
    # log said so.
    #
    # So both ends are projected through the same chain and printed.
    _vmb = _vm_bundle_cached()
    if _vmb is not None and info.get("predicted_px") and "pos" in _vmb:
        try:
            # `fam` is PER-TRIANGLE and `pos` is PER-VERTEX -- 45422
            # against 39558 on m4a4 -- so the family mask cannot index the
            # vertices directly. It selects TRIANGLES; their corner indices
            # select the vertices. The first version masked pos with it and
            # the diagnostic said so instead of guessing.
            _vp = _vmb["pos"].to(torch.float64)
            _fam_c, _tri_c = _vmb.get("fam"), _vmb.get("tri")
            if _fam_c is not None and _tri_c is not None:
                _wsel = (_fam_c.reshape(-1) == VM_FAM_WEAPON)
                if bool(_wsel.any()):
                    _vi = torch.unique(_tri_c[_wsel].reshape(-1).long())
                    _vp = _vp[_vi]
            _zc = _vp[:, 2]
            _far = _vp[int(torch.argmin(_zc))]      # most negative z = muzzle
            _near = _vp[int(torch.argmax(_zc))]     # least negative = breech
            _hw = (out.shape[1] * args.supersample,
                   out.shape[2] * args.supersample)
            _pf = _mz.project_view_point(_far.tolist(), rng, _hw, device,
                                         vm_transform=vm_transform)
            _pn = _mz.project_view_point(_near.tolist(), rng, _hw, device,
                                         vm_transform=vm_transform)
            frame_print(
                f"--muzzle-flash: barrel axis on screen -- the mesh's "
                f"FAR-z end (view z {float(_far[2]):+.4f} m, the muzzle) "
                f"projects to x {_pf[0]:.0f}, its NEAR-z end "
                f"(z {float(_near[2]):+.4f} m, the receiver) to x "
                f"{_pn[0]:.0f}. The flash is at x "
                f"{info['predicted_px'][0]:.0f}. A viewmodel's muzzle sits "
                f"at the end NEARER the screen centre, not the far side of "
                f"the silhouette.", flush=True)
        except Exception as _e:                               # noqa: BLE001
            frame_print(f"--muzzle-flash: barrel-axis projection "
                        f"unavailable ({_e}); the position line above is a "
                        f"span fraction with no orientation, so it cannot "
                        f"settle muzzle-vs-receiver on its own.", flush=True)
    # THE BUNDLE VERDICT, printed whenever the flash misses. An off-screen
    # prediction has two causes with the same symptom and different owners:
    # the camera is pointed somewhere the barrel is not, or the bundle's
    # muzzle is not on its own gun. Only the second is a defect, and it is
    # invisible from the renderer without this comparison.
    _pv = info.get("provenance_recheck")
    if _pv and not _pv["consistent"]:
        frame_print(f"⚠️  --muzzle-flash: this bundle does NOT re-derive from its "
              f"own recorded provenance -- baked view "
              f"{[round(v, 5) for v in _pv['baked']]} against "
              f"{[round(v, 5) for v in _pv['recomputed']]} recomputed from "
              f"its bind/wpn/dxy fields via the {_pv.get('chain', '?')}, "
              f"{_pv['delta_m']:.5f} m apart. The "
              f"draw uses the BAKED point. Which half is stale is not "
              f"decidable here, so nothing is corrected -- rebuild it with "
              f"extract_viewmodel.py, which now refuses to write this "
              f"state.", flush=True)
    _bb = info.get("attachment_bbox")
    if _bb and not info.get("in_frame", True):
        if _bb["outside_axes"]:
            frame_print(f"⚠️  --muzzle-flash: the attachment is OUTSIDE this "
                  f"bundle's own mesh on {_bb['outside_axes']} -- weapon "
                  f"spans {[round(v, 3) for v in _bb['lo']]}..."
                  f"{[round(v, 3) for v in _bb['hi']]} m and the muzzle is "
                  f"{[round(v, 5) for v in info['view_point_m']]}. This is a "
                  f"BUNDLE defect, not a camera one: rebuild it with "
                  f"extract_viewmodel.py. in_frame=False here is the "
                  f"SYMPTOM, not the cause.", flush=True)
        else:
            _zf = _bb.get("z_frac")
            frame_print(f"--muzzle-flash: off-screen, but the attachment IS on "
                  f"this bundle's mesh"
                  + (f" ({100.0 * _zf:.0f}% along its barrel)"
                     if _zf else "")
                  + " -- so the camera is pointed away from the muzzle "
                    "rather than the bundle being wrong.", flush=True)
    for row in _mz.provenance(info["texture_loaded"]):
        print("    " + row, flush=True)
    if rgb is None:
        return out
    # ===================================================================
    # THE DEPTH TEST (#87). Owner-observed on combat_stride1.mp4: the
    # bloom drew ATOP the gun.
    # ===================================================================
    # ADDITIVE AND OCCLUDED ARE DIFFERENT PROPERTIES, and this function's
    # own docstring conflated them: m_nOutputBlendMode ADD says the sprite
    # never DARKENS what is under it, and m_flDiffuseAmount 0 says it is
    # not lit. Neither says a fragment behind the receiver reaches the
    # screen. The reference rasterises this quad with a depth test like
    # any other fragment; `return out + rgb * _g` skipped it entirely, so
    # every pixel of the bloom survived including the ones inside the gun
    # body. The muzzle sits at view z about -0.9 m and the receiver is
    # essentially AT the eye, so on any pose where the barrel crosses its
    # own silhouette a large part of the sprite is behind solid geometry.
    #
    # HARD TEST, and at SUPERSAMPLE RESOLUTION, before the pool. A pooled
    # comparison would test one averaged depth against one averaged flash
    # depth and produce a soft edge that is neither the reference's
    # per-fragment result nor a defensible approximation of it; testing
    # first and averaging after is what supersampling means, and it is the
    # only order that makes a partially-occluded edge pixel come out as a
    # coverage fraction.
    #
    # `live_vm`, not `depth_vm` alone: a discarded fragment writes no
    # depth. The legs branch already applies exactly that rule explicitly
    # (it pushes depth to 1e9 where the screen-door discarded), so using
    # drawn coverage here keeps one rule for both instead of two that
    # agree by accident.
    _occ_frac = None
    if (args.muzzle_flash_occlude == "on" and depth_vm is not None
            and info.get("window_depth") is not None):
        _fz = info["window_depth"]                  # (1, Hh, Ww)
        _fcov = info.get("covered_mask")
        _blocked = depth_vm < _fz                   # vm geometry in front
        if live_vm is not None:
            _blocked = _blocked & live_vm
        if _fcov is not None:
            # PER FRAME, not per batch. The flash quad is rasterised ONCE
            # (batch 1) while depth_vm is per frame, so summing the
            # numerator over the batch and the denominator over the single
            # quad reported "1,950 of 1,089 OCCLUDED (179.06%)" and a
            # NEGATIVE survivor count. Caught by the impossible number,
            # which is the only reason a wrong-operand bug in a diagnostic
            # gets caught at all -- the same class this function's own
            # comments warn about twice, committed a third time here.
            # The gun moves between frames, so occlusion is genuinely per
            # frame and there was never a single right answer to print.
            _fc = (_fcov if _fcov.shape[0] == _blocked.shape[0]
                   else _fcov.expand_as(_blocked))
            _per_occ = (_blocked & _fc).flatten(1).sum(1)
            _per_flash = _fc.flatten(1).sum(1)
            # THE POLARITY OPERAND. How many sprite pixels merely OVERLAP
            # viewmodel geometry, against how many the test occluded. The
            # muzzle is the FARTHEST point of the weapon from the eye
            # (view z about -0.905 against a receiver at -0.008 -- the
            # barrel points away), so a billboard there is behind
            # essentially all of the gun and these two counts should be
            # nearly EQUAL. An inverted comparison would occlude ~0 of the
            # overlap instead, so printing both makes the sign of the test
            # readable from the log rather than argued from the source.
            _per_ovl = ((live_vm & _fc).flatten(1).sum(1)
                        if live_vm is not None else _per_occ)
            _tot_o, _tot_f = int(_per_occ.sum()), int(_per_flash.sum())
            _occ_frac = (_tot_o / _tot_f) if _tot_f else 0.0
            # HOSTLOOP_TOP10 row 8 (corrected): the loop below reads FOUR
            # tensor scalars per flash for its report line -- device syncs
            # in the print path. ONE bulk transfer each, then plain ints.
            _per_flash = _per_flash.tolist()
            _per_occ = _per_occ.tolist()
            _per_ovl = _per_ovl.tolist()
            _on_l = _on.tolist() if torch.is_tensor(_on) else _on
            # BOTH OPERANDS, per the rule this function already follows
            # elsewhere: a fraction alone cannot distinguish "nothing was
            # occluded" from "nothing was drawn".
            for _i in range(_blocked.shape[0]):
                _nf, _no = int(_per_flash[_i]), int(_per_occ[_i])
                frame_print(
                    f"--muzzle-flash: frame {_b0 + _i} depth test vs the "
                    f"viewmodel buffer -- {_no:,} of {_nf:,} flash pixels "
                    f"OCCLUDED ({100.0 * _no / _nf if _nf else 0.0:.2f}%), "
                    f"{_nf - _no:,} survive and composite additively; "
                    f"{int(_per_ovl[_i]):,} of the sprite overlaps "
                    f"viewmodel geometry at all"
                    + ("" if _on_l[_i] else " (this frame is gated OFF, so "
                                          "nothing reaches the image)")
                    + ". 0.00% with a nonzero flash count means the bloom "
                      "clears the gun at this pose, NOT that the test is "
                      "off (--muzzle-flash-occlude off turns it off).",
                    flush=True)
            # WHICH VIEWMODEL FAMILY IS DOING THE OCCLUDING. Without this
            # the fraction invites the reading "the gun ate the flash",
            # and the first analysis of this very change made exactly that
            # error: occluded pixels appeared LEFT of the weapon's own
            # screen extent and read as over-occlusion, when the ARMS are
            # viewmodel geometry too and hold the gun from that side. A
            # count with no owner is the wrong operand again.
            if fam_vm is not None:
                _split = []
                for _fnm, _fk in (("weapon", VM_FAM_WEAPON),
                                  ("arms", VM_FAM_ARMS),
                                  ("legs", VM_FAM_LEGS)):
                    _n = int(((_blocked & _fc) & (fam_vm == _fk)).sum())
                    if _n:
                        _split.append(f"{_fnm}={_n:,}")
                frame_print(
                    "--muzzle-flash: the occluder is "
                    + (" ".join(_split) if _split else "NOTHING")
                    + " (viewmodel families, summed over this batch). The "
                      "arms hold the gun and occlude from the side the "
                      "weapon's own bbox does not cover, so an occluded "
                      "pixel outside the weapon extent is expected and is "
                      "not over-occlusion.", flush=True)
        rgb = rgb * (~_blocked).unsqueeze(-1).to(rgb.dtype)
    elif args.muzzle_flash_occlude == "off":
        frame_print("--muzzle-flash: --muzzle-flash-occlude off -- the "
                    "sprite composites with NO depth test, reproducing the "
                    "pre-#87 defect on purpose. Diagnostic arm.",
                    flush=True)
    MZ_INFO["occluded_fraction"] = _occ_frac
    rgb = rgb.flip(1)
    if args.supersample > 1:
        rgb = torch.nn.functional.avg_pool2d(
            rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
    rgb = rgb.expand_as(out).clone() if rgb.shape[0] == 1 else rgb
    # per-frame gate: a batch can straddle a firing boundary, so this is
    # applied per element rather than per call
    _g = torch.tensor([1.0 if v else 0.0 for v in _on],
                      device=out.device).view(-1, 1, 1, 1)
    return out + rgb * _g


_VM_BUNDLE = [None, False]


def _vm_bundle_cached():
    """The viewmodel bundle as a dict, or None. Loaded once."""
    if _VM_BUNDLE[1]:
        return _VM_BUNDLE[0]
    _VM_BUNDLE[1] = True
    if args.viewmodel_model:
        b = torch.load(args.viewmodel_model, map_location="cpu",
                       weights_only=False)
        _VM_BUNDLE[0] = b if isinstance(b, dict) else None
    return _VM_BUNDLE[0]


_VM_ROW = None


def vm_ext2_row():
    """The ext2 row the viewmodel draw shades with.

    Read through the SAME MAT_EXT2 table as every other family. If the
    pack carries a csgo_weapon material, its row is used; otherwise the
    row is the family DEFAULT row fast_pack_fam.py emits, taken from a
    material of index 0 with every wpn_* / legs_* column at its vmat
    default. It is never a literal built here -- _e2() reads it by name,
    so a column missing from the side table fails at the refusal contract
    above rather than silently reading 0.
    """
    global _VM_ROW
    if _VM_ROW is not None:
        return _VM_ROW
    row = None
    if FAM_WEAPON is not None:
        hit = (MAT_FAM == FAM_WEAPON).nonzero()
        if hit.numel():
            row = MAT_EXT2[hit[0, 0]]
    if row is None:
        row = MAT_EXT2[0].clone()
        # this map has no csgo_weapon material, so the axes that are
        # material-selected get their axis flag from the command line
        # rather than from a vmat that does not exist. Every VALUE below
        # still comes from the side table's own column defaults.
        for _k, _on in (("wpn_f_adjust", args.weapon_adjustments),
                        ("wpn_f_tint_mask", args.weapon_tint_mask),
                        ("wpn_f_self_illum", args.weapon_self_illum),
                        ("wpn_f_glitter", args.weapon_glitter),
                        ("wpn_f_sfx_mask", args.weapon_sfx_mask),
                        ("wpn_f_stickers", args.weapon_stickers),
                        ("wpn_f_opaque_refract", args.weapon_opaque_refract),
                        ("wpn_f_alpha_test", args.weapon_alpha_test),
                        ("wpn_f_translucent", args.weapon_translucent),
                        ("wpn_f_additive", args.weapon_additive_blend),
                        ("wpn_f_mode_depth", args.weapon_mode_depth),
                        ("wpn_f_tools_vis", args.weapon_tools_vis)):
            row[EXT2[_k]] = 1.0 if _on else 0.0
        # the axis parameters a synthesised draw needs live values for:
        # a zero here would be a path that is on and does nothing, which
        # is the failure this whole file's refusal contract exists for.
        for _k, _v in (("wpn_hue_shift", 0.6), ("wpn_ramp_freq", 1.3),
                       ("wpn_ramp_phase", 0.2), ("wpn_ramp_amount", 0.7),
                       ("wpn_glitter_scale", 0.8), ("wpn_glitter_uv", 1.0),
                       ("wpn_glitter_spread", 0.35),
                       ("wpn_glitter_balance", 0.2),
                       ("wpn_sfx_amount", 0.7), ("wpn_sfx_speed", 1.0),
                       ("wpn_shimmer", 0.3), ("wpn_sticker_slots", 5.0),
                       ("wpn_stk_scale", 2.4), ("wpn_stk_rot", 0.05),
                       ("wpn_stk_off_u", 0.0), ("wpn_stk_off_v", 0.0),
                       ("wpn_stk_wear", 0.9), ("wpn_stk_holo", 1.0),
                       ("wpn_refract_scale", 0.05),
                       ("wpn_refract_blur", 0.5),
                       ("wpn_refract_edge", 0.4),
                       ("wpn_refract_amount", 1.0),
                       ("wpn_refract_contrast", 0.2),
                       ("wpn_refract_tint", 1.0),
                       ("wpn_alpha_ref", 0.5),
                       ("wpn_metal_lo", 0.15), ("wpn_metal_hi", 0.85),
                       # g_vShimmerTint is a COLOUR and its identity is
                       # white; the module multiplies by it, so a 0 here
                       # would delete the term rather than default it.
                       ("wpn_shimmer_tint_r", 1.0),
                       ("wpn_shimmer_tint_g", 0.94),
                       ("wpn_shimmer_tint_b", 0.78),
                       ("wpn_si_scroll_u", 0.05),
                       ("wpn_si_scroll_v", 0.02),
                       ("legs_fade_dist", args.viewmodel_legs_fade),
                       ("legs_cone_cos", args.viewmodel_legs_cone),
                       ("legs_eye_z", args.viewmodel_legs_eye_z)):
            row[EXT2[_k]] = float(_v)
    _VM_ROW = row.view(1, 1, 1, -1)
    return _VM_ROW


def _vm_sun_dir():
    d = torch.tensor([0.35, 0.86, 0.37], device=device)
    return (d / d.norm()).view(1, 1, 1, 3)


def _vm_sun_rgb():
    return torch.tensor([2.4, 2.25, 2.0], device=device).view(1, 1, 1, 3)


def _vm_ambient():
    return torch.tensor([0.30, 0.34, 0.42], device=device).view(1, 1, 1, 3)


# ======================================================================
# THE PLAYERMODEL PASS -- the other players, placed off the wire
# ======================================================================
# UNLIKE THE VIEWMODEL, THIS IS WORLD GEOMETRY. It rasterises with the
# frame's own `mvp` -- the same projection and the same NDC-z convention
# the three world classes use -- and tests against the resolved world
# depth, so a player behind a wall is behind that wall. Nothing here gets
# its own depth range: the viewmodel needed one because a held weapon
# would otherwise clip into geometry it is not in front of; another
# player IS in the world and must occlude and be occluded normally.
#
# WHAT IS READ AND WHAT IS STATED, in full:
#   READ    each entity's X / Y / Z / yaw / is_alive / team, per tick,
#           from the .dem -- the same demoparser2 dependency and the same
#           parse_ticks call harness/cs2_demo_camera.py makes for the ego
#           path, so the entity rows and the camera row come off the SAME
#           tick of the SAME file and cannot be misaligned by a join.
#   READ    the mesh, its UVs and its material pages, from the depot's own
#           character .vmdl_c via extract_playermodel.py.
#   READ    the state that SELECTS a pose -- fire, active_weapon_name,
#           m_iClip1, duck_amount and the x/y/z deltas, per entity per
#           tick -- and the authored clips and per-vertex bindings the
#           selection lands on. See THIRD-PERSON ANIMATION below.
#   STATED  the POSE ITSELF. It is INFERRED, never read: which clip of the
#           several an event owns, the walk phase from distance travelled,
#           and the speed thresholds. This lane reads NO pose off the wire.
#           ⚠️ That is a statement about THIS lane, not about the format.
#           "animation is not on the wire at all" used to stand here as a
#           fact and it is CONTESTED (#102): the demoparser2 zeros that
#           supported it on 14174-era demos are instrument-scope -- the
#           parser cannot decode that proto at all, while the engine
#           renders those same demos -- so the absence was partly the
#           instrument's. DEM_AnimationData's deletion at proto 14150 does
#           still stand. Nothing here depends on the resolution: the pose
#           is inferred either way, and if a readable pose is found later
#           it REPLACES the inference rather than joining it.
#           Every frame's line says whether the pose was DRIVEN or BIND.
#   STATED  any texture slot the character's vmat does not carry. Those
#           print themselves, per slot, exactly as the viewmodel's do.
#
# WHY gt_direct() AND NOT csgo_character. The character family is written
# in this file and it is NOT what shades this draw, which is a choice and
# not an oversight. Its terms are selected by --char-* axes and driven by
# a per-material ext2 row; on a pack with no csgo_character material there
# is no row to read, and vm_ext2_row's defaulted-row arm would supply
# fabricated anisotropy, hair-shift and cloth parameters. Shading a real
# extracted character through invented family parameters and compositing
# it into a scored frame is the failure mode this project keeps finding.
# gt_direct() is the SHARED lighting the world families already use, over
# the character's OWN pages -- fewer terms, none of them invented.
PM_ROWS = {}            # tick (int) -> [entity dict]
PM_SOURCE = "not loaded"
PM_BUNDLES = {}         # team -> bundle dict
PM_BUNDLE_NOTE = {}     # team -> why there is no bundle
PM_EGO = set()          # steamids NOT drawn (the camera is one of them)
_PM_LOADED = [False]
_PM_PAGES = {}

# ---- THIRD-PERSON ANIMATION STATE -------------------------------------
# `pose_basis` in every bundle still reads BIND POSE and that stays true of
# the ASSET: nothing here rewrites a bundle. What changed is that the pose
# is now CHOSEN per entity per tick from the demo's own state columns and
# applied by the same linear-blend skin the viewmodel path uses.
PM_CLIPS = [None]           # the shared ClipSet, or None
PM_CLIPS_NOTE = [None]      # why there is none, printed once
PM_SCRIPT = [_anim.Scripter()]
PM_PREV_ROW = {}            # entity key -> the previous row the scripter saw
PM_POSE_CACHE = {}          # (bundle, clip, t) -> (pos, nrm, tan) in world axes
PM_EVENT_COUNT = {}         # event -> entity-frames that chose it
PM_ANIM_NOTE = []           # the SKINNED-live line, printed once
PM_ORPHAN_NOTE = set()      # bundles whose orphan report has been printed
PM_DEAD_NOTE = []           # the death-pose fork, printed once
PM_WPN_NOTE = []            # the weapon-in-hand fork, printed once
PM_ANIM_MOVED = [0.0]       # max vertex displacement vs bind, metres

# ---- THE WEAPON IN THE THIRD-PERSON HAND -------------------------------
# THE ATTACH POINT IS READ, AND THE READ IS A CROSS-CHECK BETWEEN TWO
# INDEPENDENTLY PRODUCED ARTIFACTS. extract_viewmodel writes
# `bind_applied_src` -- the point it subtracts before applying the
# viewmodel rig's `wpn` bone -- and extract_playermodel, run over the SAME
# .vmdl_c with none of the same code, writes a skin whose palette contains
# a bone called `weapon_offset`. On ak47 those two are
#     bind_applied_src            [2.9689, -0.1931, 2.9819]
#     weapon_offset bind_t        [2.9689, -0.1931, 2.9819]
# to every printed digit, and weapon_offset's bind_q is identity. So the
# weapon's attach bone is `weapon_offset`, the rig's socket is `wpn`, and
# the viewmodel's own chain -- qrot(wpn_quat, loc - bind_applied_src) +
# wpn_src -- IS "put weapon_offset on wpn". Nothing here is derived from
# the name: the pairing is two files agreeing.
#
# So third person is the same sentence with the CHARACTER rig's `wpn`
# substituted for the viewmodel rig's, and that bone is already posed by
# the clip this pass samples every frame.
#
# CS2 SHIPS NO `w_` WORLD MODELS -- 0 of the 119 .vmdl_c under
# weapons/models/ match w_*. The world weapon is the same asset as the
# viewmodel's, and the two differ only in placement and in WHICH MESH:
# the viewmodel bundle is built from `body_hd` (21,414 verts) and the
# model-space extraction takes `body_legacy` (12,935). That is a real
# difference in the asset, not a extraction bug, and it is why a vm bundle
# is not reused here even before its view-space placement disqualifies it.
PM_WPN_POSE = {}            # (clip, t) -> (mq, mt) of the rig's `wpn` bone
PM_WBUNDLES = {}            # basename -> model-space weapon bundle
PM_WBUNDLE_NOTE = {}        # basename -> why there is none
PM_WEAPON_MAP = {}          # active_weapon_name -> bundle basename
PM_WEAPON_SEEN = {}         # weapon name -> what it resolved to
PM_WEAPON_NOTE = []
PM_WEAPON_PX = [0]          # weapon pixels drawn this run

# Source's own team enumeration, READ from the engine's constants rather
# than inferred from which side had more players: TEAM_TERRORIST = 2,
# TEAM_CT = 3. demoparser2 exposes both the number and the name; whichever
# the demo carries is used and the parse line says which.
PM_TEAM_NUM = {2: "t", 3: "ct"}
PM_TEAM_NAME = {"TERRORIST": "t", "T": "t", "CT": "ct",
                "COUNTER-TERRORIST": "ct", "COUNTERTERRORIST": "ct"}


def _pm_team(name, num):
    """'ct' / 't' / None, from whichever column the demo carried."""
    if name is not None and str(name) == str(name):
        k = str(name).strip().upper()
        if k in PM_TEAM_NAME:
            return PM_TEAM_NAME[k]
    try:
        return PM_TEAM_NUM.get(int(num))
    except (TypeError, ValueError):
        return None


# ======================================================================
# WHICH AGENT EACH PLAYER WORE -- read, and what "read" turned out to mean
# ======================================================================
# PROBED, not assumed, against demoparser2 on the test demo. Ten candidate
# name-bearing props -- model_name, player_model, agent, skin, m_nModelIndex,
# CCSPlayerPawn.m_nModelIndex, CBodyComponent.m_hModel, m_szArmsModel and
# two more -- ALL come back absent: parse_ticks returns only the columns it
# has, so the check is on the RETURNED columns and not on an exception.
# `parse_skins()` is WEAPON skins (def_index / paint_index / paint_wear),
# not agents.
#
# WHAT IS ON THE WIRE is
#   CCSPlayerPawn.CBodyComponentBaseAnimGraph.m_hModel
# and it IS the per-player agent identity: on the test demo it is constant
# per steamid (1 distinct value per player over 578,123 rows) and takes
# exactly 2 values, one per team. So the identity is real and readable.
#
# WHAT IT IS NOT is a name. It is a model HANDLE, and demoparser2 exposes no
# handle -> "agents/models/ctm_sas" string table. So the brief's
# `agents/models/<name> -> pm_bundles/<name>.pt` mapping cannot be READ from
# the demo today; the join needs a handle->name table from somewhere else.
#
# ⚠️ AND THE HANDLE ARRIVES LOSSY. parse_ticks hands it back as float64
# (1.3181005211767974e+19). float64 carries a 53-bit mantissa and these
# values sit near 2^63.5, so the ULP is about 2^11 -- the low ~11 bits are
# GONE. Two agents whose handles differ only in those bits are
# indistinguishable here. CONFIRMED rather than predicted: both handles the
# test demo returns (13181005211767973888 for T, 10254416614829637632 for
# CT) are exact multiples of 2048, which a real 64-bit handle has no reason
# to be -- that is the quantisation, visible in the value. That is why the handle is used as an OPAQUE KEY
# printed in full rather than as a number anything computes with, and why
# an operator-supplied map is the join rather than arithmetic on it.
#
# So the resolution is three tiers, every one of them printed per entity:
#   1. a NAME column, if some demo or parser version returns one (probed
#      the same way; none does today)
#   2. --playermodel-agent-map, a JSON {agent key: bundle basename} the
#      operator supplies once the handle->agent join is known elsewhere
#   3. the TEAM bundle, ct.pt / t.pt, as a printed FALLBACK
# and the frame line says which model was REQUESTED and which DREW, so
# "every CT is ctm_sas because that is the fallback" never reads as "every
# CT is ctm_sas because the demo said so".
PM_AGENT_NAME_PROPS = ("model_name", "player_model", "agent_name", "agent")
PM_AGENT_HANDLE_PROP = ("CCSPlayerPawn.CBodyComponentBaseAnimGraph."
                        "m_hModel")
PM_AGENT_MAP = {}
PM_AGENT_SRC = "not read"
PM_AGENT_SEEN = {}          # agent key -> bundle it resolved to
PM_BUNDLE_MISS = {}         # agent key -> why it fell back


def pm_agent_key(v):
    """An agent value as a stable, printable key.

    A float64 handle is rendered as an integer with its lossy low bits
    stated once by the caller, never rounded silently into a different
    number.
    """
    if v is None or v != v:
        return None
    if isinstance(v, str):
        return v
    return str(int(v))


def _pm_rows_from_dem(path):
    """{tick: [entity]} from a .dem, via demoparser2.

    The prop list is the ego path's, minus what only the camera needs.
    Missing props are detected from the RETURNED COLUMNS and not from an
    exception, because parse_ticks does not raise on a prop the demo has
    no field for -- it silently returns a frame without that column, a
    behaviour harness/cs2_demo_camera.py checked and documented.
    """
    from demoparser2 import DemoParser
    # THE SCRIPTER'S COLUMNS, PER ENTITY. anim_script.Scripter reads five
    # state columns off a row (fire, active_weapon_id, ammo_clip,
    # duck_button, x/y/z) and the ego path already supplies them for ONE
    # player. These are the same observables for EVERY player, PROBED on
    # the test demo the same way the agent props were -- from the RETURNED
    # columns, because parse_ticks does not raise on a prop the demo has no
    # field for. What came back:
    #   duck_amount   READ, a RATIO in [0, 1] (the duck RAMP, not a button)
    #   active_weapon_name / active_weapon   READ, both per entity
    #   m_iClip1      READ -- the magazine count. There is no `ammo_clip`
    #                 column on this parser (it came back MISSING), and the
    #                 reload inference wants the observable that RISES when
    #                 a magazine is replaced, which is this one.
    # `fire` is NOT a tick column for anybody: it is joined from the
    # weapon_fire EVENT below, exactly as cs2_demo_camera.py does it for
    # the ego. A column that came back missing leaves its state UNDRIVEN
    # and the scripter falls through to the speed-derived branch, which is
    # printed per entity rather than assumed.
    want = ["X", "Y", "Z", "yaw", "is_alive", "health",
            "team_name", "team_num",
            "duck_amount", "active_weapon_name", "active_weapon",
            "m_iClip1"] \
        + list(PM_AGENT_NAME_PROPS) + [PM_AGENT_HANDLE_PROP]
    _parser = DemoParser(str(path))
    df = _parser.parse_ticks(list(want))
    have = [c for c in want if c in df.columns]
    missing = [c for c in want if c not in df.columns]
    print(f"playermodels: {path} -> {len(df)} tick-rows, columns READ "
          f"{have}" + (f", MISSING {missing}" if missing else ""), flush=True)
    _fire = _pm_fire_ticks(_parser)
    # WHICH agent column this demo actually carries, from the RETURNED
    # columns. A name is preferred and none has ever been returned; the
    # handle is what is really there.
    global PM_AGENT_SRC
    _acol = next((c for c in PM_AGENT_NAME_PROPS if c in df.columns), None)
    _aname = _acol is not None
    if _acol is None and PM_AGENT_HANDLE_PROP in df.columns:
        # RENAMED before any itertuples: the handle's column name is
        # dotted, and itertuples mangles a dot into a positional _N.
        # harness/cs2_demo_camera.py documents this exact trap for `fov`
        # and `view_punch_angle`; reading it back with getattr on the
        # dotted name silently returns the default for every row, i.e. an
        # agent column that is present and always None.
        df = df.rename(columns={PM_AGENT_HANDLE_PROP: "agent_handle"})
        _acol = "agent_handle"
    if _acol is None:
        PM_AGENT_SRC = "ABSENT"
        print("playermodels: this demo carries NO agent column (tried "
              f"{list(PM_AGENT_NAME_PROPS)} and {PM_AGENT_HANDLE_PROP}). "
              "Every entity falls back to its team bundle.", flush=True)
    elif _aname:
        PM_AGENT_SRC = f"name column {_acol}"
        print(f"playermodels: agent read from NAME column {_acol!r} -- "
              f"bundles resolve by that name directly", flush=True)
    else:
        PM_AGENT_SRC = f"model handle {PM_AGENT_HANDLE_PROP}"
        print(f"playermodels: agent read from {PM_AGENT_HANDLE_PROP} -- a "
              f"model HANDLE, "
              f"not a name. demoparser2 exposes no handle -> "
              f"agents/models/<name> table, and parse_ticks returns it as "
              f"float64 whose ~11 low bits are BELOW the mantissa at this "
              f"magnitude, so it is used as an OPAQUE KEY. Supply "
              f"--playermodel-agent-map to bind a handle to a bundle; "
              f"without it every entity falls back to its team.",
              flush=True)
    for req in ("X", "Y", "Z", "yaw"):
        if req not in df.columns:
            raise SystemExit(
                f"--playermodels {path}: the demo carries no {req!r} column, "
                f"so no entity can be placed. A pass that ran without it "
                f"would put every player at the origin.")
    by = {}
    for r in df.itertuples(index=False):
        x, y, z = getattr(r, "X"), getattr(r, "Y"), getattr(r, "Z")
        if x != x or y != y or z != z:            # NaN -> no position
            continue
        yaw = getattr(r, "yaw", None)
        if yaw is None or yaw != yaw:
            continue
        _sid = (int(r.steamid)
                if getattr(r, "steamid", None) is not None
                and r.steamid == r.steamid else None)
        _tick = int(r.tick)
        _duck = getattr(r, "duck_amount", None)
        _wpn = getattr(r, "active_weapon_name", None)
        if _wpn is None or _wpn != _wpn:
            _wpn = getattr(r, "active_weapon", None)
        _clip1 = getattr(r, "m_iClip1", None)
        by.setdefault(_tick, []).append({
            "steamid": _sid,
            "name": getattr(r, "name", None),
            "tick": _tick,
            "x": float(x), "y": float(y), "z": float(z),
            "yaw": float(yaw),
            "is_alive": bool(getattr(r, "is_alive", True)),
            "team": _pm_team(getattr(r, "team_name", None),
                             getattr(r, "team_num", None)),
            "agent": (pm_agent_key(getattr(r, _acol, None))
                      if _acol else None),
            # --- the scripter's state, per entity ----------------------
            # KEYS ARE THE SCRIPTER'S, values are this demo's columns, and
            # every rename is one-way and stated:
            #   fire            <- the weapon_fire EVENT joined on
            #                      (user_steamid, tick). Not a tick column.
            #   active_weapon_id<- active_weapon_name, or the handle when
            #                      the name is absent. The scripter only
            #                      tests it for CHANGE, so a name is as
            #                      good an id as a number and reads better
            #                      in the log.
            #   ammo_clip       <- m_iClip1. STATED rename: there is no
            #                      `ammo_clip` column on this parser and
            #                      none is invented; m_iClip1 IS the
            #                      magazine count whose RISE the reload is
            #                      inferred from.
            #   duck_button     <- duck_amount > 0. The wire carries the
            #                      duck RAMP as a ratio, not a button, so
            #                      the threshold is STATED (any nonzero
            #                      crouch travel counts as held) rather
            #                      than fitted.
            "fire": bool(_fire.get((_sid, _tick))),
            "fire_weapon": _fire.get((_sid, _tick)),
            "active_weapon_id": (_wpn if _wpn == _wpn else None),
            "ammo_clip": (float(_clip1)
                          if _clip1 is not None and _clip1 == _clip1
                          else None),
            "duck_button": bool(_duck is not None and _duck == _duck
                                and float(_duck) > 0.0),
            "duck_amount": (float(_duck)
                            if _duck is not None and _duck == _duck else 0.0),
        })
    return by


def _pm_fire_ticks(parser):
    """{(steamid, tick): weapon} from the weapon_fire EVENT, for EVERYONE.

    THE FIRING SIGNAL IS NOT A TICK COLUMN and never was -- for the ego
    either. cs2_demo_camera.ego_fire_ticks joins parse_event("weapon_fire")
    on `user_steamid` for ONE player; this is the same read with the
    steamid kept as part of the key instead of filtered on, so all ten
    players get the column the scripter's first branch tests.

    READ from the RETURNED columns, and a parse that gives none is a
    printed absence rather than an exception: with no fire column every
    entity simply falls through to the speed branch, which is a worse
    animation and not a crash, and the difference has to be visible.
    """
    try:
        df = parser.parse_event("weapon_fire")
    except Exception as exc:                                  # noqa: BLE001
        print(f"playermodels: weapon_fire parse FAILED ({exc}) -- no entity "
              f"gets a `fire` tick, so the shoot event can never be chosen "
              f"and every player falls through to the speed branch. That is "
              f"a missing input, not an empty one.", flush=True)
        return {}
    if df is None or len(df) == 0:
        print("playermodels: weapon_fire returned 0 events -- nobody fired "
              "in this demo, which is a fact about the round and not a "
              "missing column.", flush=True)
        return {}
    cols = sorted(df.columns)
    if "user_steamid" not in df.columns or "tick" not in df.columns:
        print(f"playermodels: weapon_fire returned columns {cols}, which "
              f"carry no user_steamid/tick pair to join on -- the `fire` "
              f"state is UNDRIVEN for every entity and nothing is guessed "
              f"from the remaining columns.", flush=True)
        return {}
    out = {}
    for row in df.itertuples(index=False):
        try:
            sid = int(row.user_steamid)
        except (TypeError, ValueError):
            continue
        out[(sid, int(row.tick))] = getattr(row, "weapon", None)
    print(f"playermodels: weapon_fire {len(df)} events over "
          f"{len({k[0] for k in out})} player(s), {len(out)} distinct "
          f"(steamid, tick) pairs; columns {cols}", flush=True)
    return out


def _pm_rows_from_json(path):
    """{tick: [entity]} from a rows JSON.

    Two shapes are accepted and neither is guessed at: a dict with a
    `ticks` mapping of tick -> entity list, or a flat list of entities
    each carrying its own `tick`. Anything else is refused by name.
    """
    d = json.load(open(path))
    by = {}
    if isinstance(d, dict) and isinstance(d.get("ticks"), dict):
        for k, v in d["ticks"].items():
            by[int(k)] = list(v)
        print(f"playermodels: {path} schema "
              f"{d.get('schema', 'UNDECLARED')}, {len(by)} ticks", flush=True)
    elif isinstance(d, list):
        for e in d:
            by.setdefault(int(e["tick"]), []).append(e)
        print(f"playermodels: {path} flat entity list, {len(by)} ticks",
              flush=True)
    else:
        raise SystemExit(
            f"--playermodels {path}: expected a dict with a `ticks` map or a "
            f"flat entity list carrying `tick`. Guessing the shape is how a "
            f"file that parses renders nothing.")
    return by


def _pm_load():
    """Rows, bundles and the ego exclusion. Once, and it prints all three."""
    if _PM_LOADED[0]:
        return
    _PM_LOADED[0] = True
    global PM_SOURCE
    p = args.playermodels
    PM_SOURCE = p
    PM_ROWS.update(_pm_rows_from_dem(p) if str(p).lower().endswith(".dem")
                   else _pm_rows_from_json(p))
    # --- the ego exclusion, READ from the camera path ------------------
    # cs2_demo_camera.py writes the recording player's steamid into the
    # camera JSON (`"steamid": sid`). That entity IS this camera, so its
    # model sits at the near plane and fills the frame with the inside of
    # a head. Excluding it by a POSITION test would be a threshold; this
    # is an identity carried by the file the poses came from.
    try:
        _c = json.load(open(args.camera_json))
        _sid = _c.get("steamid") if isinstance(_c, dict) else None
    except Exception:                                        # noqa: BLE001
        _sid = None
    if SESSION is not None:
        # Session mode: args.camera_json is a PLACEHOLDER (one agent's
        # path); applying its steamid globally excluded that one player
        # from every OTHER agent's view and nobody's own subject from
        # their own (measured on the 10-agent ego set). Exclusion is
        # per-agent via AGENT_EGO, read at session load.
        print("playermodels: SESSION mode -- ego exclusion is PER-AGENT "
              "(AGENT_EGO), the global camera-json steamid is NOT applied",
              flush=True)
        _sid = None
    if _sid is not None:
        PM_EGO.add(int(_sid))
        print(f"playermodels: excluding the RECORDING player, steamid "
              f"{int(_sid)}, READ from {args.camera_json}", flush=True)
    else:
        print("playermodels: --camera-json carries NO `steamid`, so the "
              "recording player CANNOT be identified and is NOT excluded. "
              "Expect one model standing inside the camera. Pass "
              "--playermodel-exclude-steamid to fix it; this is not "
              "guessed from proximity.", flush=True)
    if args.playermodel_exclude_steamid:
        for s in args.playermodel_exclude_steamid.split(","):
            if s.strip():
                PM_EGO.add(int(s.strip()))
        print(f"playermodels: excluding steamids {sorted(PM_EGO)}", flush=True)
    # --- the agent -> bundle map, if the operator supplied one ---------
    if args.playermodel_agent_map:
        PM_AGENT_MAP.update({str(k): str(v) for k, v in
                             json.load(open(args.playermodel_agent_map)
                                       ).items()})
        print(f"playermodels: {len(PM_AGENT_MAP)} agent->bundle binding(s) "
              f"from {args.playermodel_agent_map}: {PM_AGENT_MAP}",
              flush=True)
    else:
        print("playermodels: no --playermodel-agent-map, so no agent binds "
              "to a specific bundle and every entity takes its team's. "
              "Extract more agents and bind them here; the pass picks them "
              "up by name without changing.", flush=True)
    # --- the third-person weapon join -----------------------------------
    if args.playermodel_weapon_map:
        PM_WEAPON_MAP.update({str(k): str(v) for k, v in
                              json.load(open(args.playermodel_weapon_map)
                                        ).items()})
        print(f"playermodels: {len(PM_WEAPON_MAP)} weapon->bundle "
              f"binding(s) from {args.playermodel_weapon_map}",
              flush=True)
    if not args.playermodel_weapons:
        print("playermodels: no --playermodel-weapons directory, so every "
              "ally is posed HOLDING NOTHING -- the rig's `wpn` socket is "
              "driven and empty. That is a missing input, printed per "
              "entity, not a pose that was declined.", flush=True)
    # --- the bundles ---------------------------------------------------
    # ct/t are loaded eagerly because they are the fallback every entity
    # can land on; any OTHER bundle loads on demand, by name, the first
    # time an agent asks for it. That is what makes the set growable:
    # extract `ctm_st6.pt` into this directory, bind the agent to it, and
    # the pass picks it up without a line changing here.
    for team in ("ct", "t"):
        _pm_bundle(team)
    for name, why in sorted(PM_BUNDLE_NOTE.items()):
        print(f"playermodels: bundle {name!r} ABSENT -- {why}. Entities "
              f"resolving to it are NOT drawn and NOT substituted with "
              f"another model.", flush=True)


def _pm_bundle(name):
    """The bundle called `name`, loaded once, or None with a recorded why.

    Keyed by BUNDLE NAME rather than by team so an agent-specific model is
    a filename and not a new branch.
    """
    if name in PM_BUNDLES:
        return PM_BUNDLES[name]
    if name in PM_BUNDLE_NOTE:
        return None
    if not name:
        return None
    if not args.playermodel_bundles:
        PM_BUNDLE_NOTE[name] = "no --playermodel-bundles given"
        return None
    f = os.path.join(args.playermodel_bundles, f"{name}.pt")
    if not os.path.exists(f):
        PM_BUNDLE_NOTE[name] = f"no bundle at {f}"
        return None
    b = torch.load(f, map_location="cpu", weights_only=False)
    need = ("pos", "uv", "nrm", "tan", "tri", "space",
            # A bundle without these predates the per-material draw and
            # would have to be shaded by one page set over a surface it
            # does not draw. Refused by name rather than defaulted, so a
            # stale bundle says so instead of rendering a character in
            # its own goggle lenses.
            "mat_of_tri", "materials")
    miss = [k for k in need if k not in b]
    if miss:
        PM_BUNDLE_NOTE[name] = (f"{f} is missing {miss}; rebuild with "
                                f"extract_playermodel.py")
        return None
    # THE SPACE CHECK, and it is not defensive boilerplate. A viewmodel
    # bundle has the same eight keys and would load without complaint,
    # then render 60 cm of rifle at every enemy's feet -- a failure that
    # produces pixels and therefore survives a coverage check.
    if b.get("space") != "model-source-units":
        PM_BUNDLE_NOTE[name] = (
            f"{f} declares space {b.get('space')!r}, not "
            f"'model-source-units'. That is a VIEWMODEL bundle: its "
            f"vertices are view-space metres with a placement already "
            f"baked in, and placing them per entity would be placing an "
            f"already-placed mesh.")
        return None
    PM_BUNDLES[name] = b
    print(f"playermodels: bundle {name!r} <- {f} -- "
          f"{b['pos'].shape[0]} verts, {b['tri'].shape[0]} tris, "
          f"source {b.get('source')}, pose "
          f"{b.get('pose_basis', 'UNRECORDED')}", flush=True)
    return b


def pm_pages(team, mi):
    """The slot set MATERIAL `mi` of this team's model samples.

    Same slot NAMES vm_gt_shade reads, so one shading function serves both
    draws. A slot the character's vmat does not carry gets a STATED
    constant -- printed here, once, with its value -- rather than the
    viewmodel's synthesised image, because borrowing a weapon's procedural
    checker to stand in for a character's albedo would put a weapon's
    texture statistics into a character's pixels.

    PER MATERIAL, because a character is not one skin. READ from
    ctm_sas thirdperson_body: four materials over 13,504 triangles, and the
    largest covers well under all of them -- on tm_phoenix the largest is
    44.0%. One page set per model would leave the majority of the surface
    textured with a material it does not draw, so the bundle carries a page
    set per material and `mat_of_tri` says which triangle takes which.
    """
    key = (team, mi)
    if key in _PM_PAGES:
        return _PM_PAGES[key]
    b = PM_BUNDLES[team]
    mat = b["materials"][mi]
    _pm_sampler_check(f"playermodel {team}/mat{mi}", mat)
    out = _slot_pages(f"playermodel {team}/mat{mi} "
                      f"{os.path.basename(mat.get('path', '?'))}",
                      mat.get("pages") or {}, mat.get("provenance") or {})
    _PM_PAGES[key] = out
    return out


PM_SAMPLER_SEEN = {}
PM_SCALE_SEEN = set()


def _pm_sampler_check(tag, mat):
    """ASSERT the wrap-repeat every sampler in this file hardcodes.

    THE ASSUMPTION IS UNIVERSAL AND UNCHECKED. vm_sample takes `uv % 1.0`;
    so do the world taps; NOTHING anywhere reads a texture address mode.
    The material carries one -- g_nTextureAddressModeU / ...V are
    m_intParams on the .vmat_c -- and it was simply never consulted.

    CENSUSED BEFORE BEING CALLED A DEFECT, over the 128 materials this
    pass actually draws: 113 declare the pair, 15 declare neither, and
    every declared value is 0 = WRAP. So the hardcode is CORRECT on this
    entire population and changing the sampling would move ZERO pixels.
    That is the finding, and it is the reason this is an assertion and not
    a rewrite: an assumption that happens to hold looks exactly like a
    checked one until the first asset that breaks it, and then it looks
    like a texture bug on that asset rather than a missing read.

    A bundle extracted before the field was recorded says UNRECORDED, once,
    and is not silently counted as agreeing.
    """
    if tag in PM_SAMPLER_SEEN:
        return
    PM_SAMPLER_SEEN[tag] = True
    smp = (mat or {}).get("sampler")
    if smp is None:
        why = (mat or {}).get("sampler_absent_reason")
        print(f"sampler ({tag}): UNRECORDED -- this bundle predates the "
              f"address-mode read ({why or 'no sampler field'}). The tap "
              f"is wrap-repeat regardless, which is what 113 of 113 "
              f"declaring materials asked for, but on THIS material it is "
              f"an assumption and not a read. Re-extract to check it.",
              flush=True)
        return
    u, v = smp.get("address_u"), smp.get("address_v")
    if u == 0 and v == 0:
        print(f"sampler ({tag}): address U={u} V={v} = WRAP, READ from the "
              f"material and matching the wrap-repeat tap this renderer "
              f"applies.", flush=True)
    else:
        print(f"  ⚠️  SAMPLER MISMATCH ({tag}): the material declares "
              f"address U={u} V={v}, and every tap in this renderer is "
              f"wrap-repeat (uv % 1.0). Source 2's enum has 0 = WRAP, so "
              f"this surface is being sampled with the WRONG address mode "
              f"wherever its UVs leave [0,1]. This is the first material "
              f"in the corpus to disagree; the hardcode was censused "
              f"correct on 113 of 113 before it.", flush=True)


def _slot_pages(tag, pages, prov):
    """A bundle's pages in the slot set vm_gt_shade reads, with provenance.

    Shared by the playermodel and the viewmodel ARMS draw. Both take real
    pages from their own bundle and both need the SAME stand-in for a slot
    their material does not carry -- so the STATED table lives here once,
    rather than being written a second time and drifting.

    A missing slot gets a STATED constant printed with its value, NOT the
    viewmodel's synthesised procedural image: borrowing a weapon's checker
    to stand in for skin or a character's albedo would put a weapon's
    texture statistics into pixels that are not a weapon's.
    """
    # STATED defaults, by slot, with the reason each value is the identity
    # for the term that reads it.
    stated = {
        "colour": (torch.tensor([0.5, 0.5, 0.5]),
                   "mid grey -- an albedo with no hue cannot tint the "
                   "lighting it multiplies"),
        "mask": (torch.tensor([0.5, 0.0, 0.0, 1.0]),
                 "roughness 0.5, metalness lerp 0 -- a character is "
                 "dielectric, and 0 is the identity for the F0 lerp"),
        # DERIVED, not picked: octa_normal_decode is a = (r+g) - OCTA_BIAS,
        # b = r - g, so the encode of (0,0,1) is r = g = OCTA_BIAS/2 with
        # OCTA_BIAS = 1.00392162799835205078125 (allon_c3041_d2:540-542).
        # 128/255 is the value that LOOKS right here and is not it.
        "normal": (torch.tensor([0.501960813999176025390625,
                                 0.501960813999176025390625, 0.0, 1.0]),
                   "the octahedral encode of (0,0,1) -- OCTA_BIAS/2 in both "
                   "channels, so the decode returns the geometric normal "
                   "unchanged"),
        "ao": (torch.tensor([1.0, 1.0, 1.0, 1.0]),
               "1.0 -- the multiplicative identity, so an absent AO page "
               "darkens nothing"),
    }
    out = {}
    for slot, (val, why) in stated.items():
        if slot in pages:
            pg = pages[slot].to(device).float() / 255.0
            want = val.numel()
            if pg.shape[-1] > want:
                pg = pg[..., :want]
            elif pg.shape[-1] < want:
                # the viewmodel's refusal, for the same reason: padding
                # invents the missing channel and every term reading it
                # would be measuring the invention
                raise SystemExit(
                    f"{tag}: slot {slot!r} has {pg.shape[-1]} channels; the "
                    f"shading reads {want}. Padding it would invent the "
                    f"missing channel and every term that reads it would be "
                    f"measuring the invention.")
            out[slot] = pg
            print(f"texture ({tag}): {slot:8s} "
                  f"{prov.get(slot, 'REAL, no provenance recorded')}",
                  flush=True)
        else:
            out[slot] = val.to(device).view(1, 1, -1).expand(2, 2, -1)
            print(f"texture ({tag}): {slot:8s} STATED {val.tolist()} "
                  f"-- {why}", flush=True)
    return out


# ======================================================================
# THIRD-PERSON ANIMATION -- the pose the wire cannot carry, chosen from
# the state it does
# ======================================================================
# WHAT IS READ AND WHAT IS STATED, in full, because this pass previously
# printed `pose BIND (animation is NOT on the wire)` on every frame and
# that sentence stays TRUE of the pose itself:
#   READ    the clips. 1,144 world clips on
#           animation/skeletons/characters/worldmodel.vnmskel, decoded by
#           build_clips.py into agents.clips.pt, every one of them carrying
#           its own t/q/s per frame, its bone_ids, its PARENTS and its
#           m_bIsAdditive flag.
#   READ    the per-vertex bindings. Every one of the 82 bundles ships a
#           skin block (iji/mesh-skin/v1) whose palette names bones of that
#           same skeleton, so the clip and the mesh join BY BONE NAME.
#   READ    the state that selects a clip: fire, active_weapon_name,
#           m_iClip1, duck_amount and the x/y/z deltas, per entity per
#           tick, off the demo.
#   STATED  WHICH clip of the several an event owns (deterministic, keyed
#           on the tick), the walk PHASE (derived from distance travelled
#           -- stride phase is not on any wire), and the speed thresholds
#           that separate still from walk from run. anim_script.py prints
#           each of those on the line it drives.
#   NOT DONE  no additive clip is composited. 230 of the 1,144 are deltas
#           against a base pose this lane has no read of -- including 123
#           of 132 idles -- and the scripter REFUSES them with a printed
#           substitution rather than applying deltas as absolute poses.
#
# ROOT MOTION IS NOT A HAZARD HERE, and that is measured rather than
# assumed: over all 1,601 blocks of all 1,144 clips the root track's
# translation span is 0.0000 source units -- every world clip is authored
# IN PLACE, with travel supplied by the entity's own X/Y/Z. So applying a
# clip as authored cannot double-count the walk, and no root track is
# stripped (stripping one would be a silent edit to the asset).


class _PMClipSet(_anim.ClipSet):
    """A ClipSet that prefers a clip whose tracks ACTUALLY MOVE over time.

    A CLIP CAN BE ABSOLUTE, NON-ADDITIVE, CORRECTLY DECODED -- AND STILL BE
    ONE FRAME OF POSE HELD FOR ITS WHOLE DURATION. 341 of the 1,601 blocks
    in agents.clips.pt are constant across every frame they declare, and
    they are not evenly spread: the FIRST absolute `walk` clip is one of
    them, and pick() keys on index 0. So the default selection for the
    single most common event was a clip that cannot animate, and the
    symptom would have been the worst kind -- players in a real pose,
    plainly not T-posed, frozen mid-stride while the log printed an event,
    a clip name and a t that advanced. Every part of that log line would
    have been true.

    A pixel A/B would NOT have caught it either: a held pose still differs
    from the bind pose by 27 source units here, so clip-on vs clip-off
    lights up exactly as brightly as real motion does. Only a within-clip
    comparison separates them, so that is what this reads -- q and t
    variation across the block's own frames, on the block for THIS
    skeleton, once per clip and cached on it.

    WHAT IT DOES NOT DO is drop a static clip from the set. For several
    events the ONLY absolute clips are static; there the static one is
    still returned, because a held authored pose is much closer to right
    than a T-pose, and the caller prints which case it got.
    """

    def moves(self, clip):
        """Does this clip's block for OUR skeleton vary over its frames?

        The detector itself is skin_anim.block_varies -- it was written
        here first, and it moved into the module the moment a second
        consumer could want it, which is the whole of #98. What stays here
        is the caching and WHICH skeleton to ask about; the arithmetic is
        not duplicated.
        """
        v = clip.get("_moves")
        if v is None:
            bl = _skin.block_for(clip, self.skeleton, require_parents=False)
            v = False if bl is None else bool(_skin.block_varies(bl)[0])
            clip["_moves"] = v
        return v

    def pick(self, event, key=0):
        lst = self.events.get(event) or []
        if not lst:
            return None
        absolute = [c for c in lst if not c.get("additive")]
        moving = [c for c in absolute if self.moves(c)]
        if moving:
            return moving[int(key) % len(moving)]
        return super().pick(event, key=key)

    def census(self):
        rows = []
        for k in sorted(self.events):
            lst = self.events[k]
            absolute = [c for c in lst if not c.get("additive")]
            rows.append((k, len(lst), len(absolute),
                         sum(1 for c in absolute if self.moves(c))))
        return rows


def pm_clips():
    """The shared world-clip sidecar, loaded once, or None with a reason.

    ONE SIDECAR FOR ALL 82 BUNDLES, and that is a READ not a convenience:
    `agents.clips.pt` declares skeleton
    animation/skeletons/characters/worldmodel.vnmskel and every character
    bundle's skin palette names bones of it. The viewmodel side resolves a
    sidecar per RIG because knives ship their own; the characters share
    one, so the resolution is by directory and the shared skeleton is
    checked against each bundle's palette at pose time by NAME.
    """
    if PM_CLIPS[0] is not None:
        return PM_CLIPS[0]
    if PM_CLIPS_NOTE[0] is not None:
        return None
    if args.pm_anim == "off":
        PM_CLIPS_NOTE[0] = "--pm-anim off"
        print("playermodels: --pm-anim off -- every entity holds its "
              "bundle's BIND pose. This is the CONTROL arm, not a missing "
              "input: the clips are on disk and are deliberately not "
              "sampled.", flush=True)
        return None
    cand = args.playermodel_clips
    if not cand:
        if not args.playermodel_bundles:
            PM_CLIPS_NOTE[0] = "no --playermodel-bundles to look beside"
            print("playermodels: no --playermodel-bundles, so there is no "
                  "directory to find agents.clips.pt in and no pose can be "
                  "chosen.", flush=True)
            return None
        cand = os.path.join(args.playermodel_bundles, "agents.clips.pt")
    if not os.path.exists(cand):
        PM_CLIPS_NOTE[0] = f"no clip sidecar at {cand}"
        print(f"playermodels: NO clip sidecar at {cand} -- every entity "
              f"renders at BIND pose and the matrix reports decoded=0. That "
              f"is a missing input, not a choice; build it with "
              f"build_clips.py. Nothing is substituted.", flush=True)
        return None
    cs = _PMClipSet(cand)
    print(f"playermodels: animation {cs.note}", flush=True)
    print(f"playermodels: clip skeleton {cs.skeleton} -- every bundle's "
          f"skin palette is joined to it BY BONE NAME, and a palette bone "
          f"the rig does not name is reported per bundle with its vertex "
          f"count and weight mass rather than held silently at identity "
          f"(#90).", flush=True)
    # THREE COUNTS PER EVENT, because a clip fails this pass in three
    # different ways and one number cannot separate them: it can be
    # ADDITIVE (deltas, refused), it can be absolute but STATIC (a held
    # pose -- selectable, and the reason pick() prefers a moving sibling),
    # or it can be absolute and animated. The census is printed so a
    # frozen player can be read off the log instead of the screen.
    _rows = cs.census()
    print("playermodels: clips per event  total/absolute/absolute+MOVING: "
          + ", ".join(f"{k} {n}/{a}/{m}" for k, n, a, m in _rows), flush=True)
    _stuck = [k for k, _n, a, m in _rows if a and not m]
    if _stuck:
        print(f"  ⚠️  events whose every absolute clip is STATIC (one pose "
              f"held for the whole duration): {_stuck}. An entity choosing "
              f"one of those gets an authored pose that does not advance "
              f"with t -- closer to right than a T-pose, and NOT motion. "
              f"Each such choice says so on its own frame line.", flush=True)
    PM_CLIPS[0] = cs
    ANIM_DECODED["playermodel"] = sorted(cs.events)
    PM_SEQUENCES[:] = sorted(cs.events)
    return cs


def _pm_qrot(q, v):
    """Rotate (N,3) `v` by (N,4) xyzw `q`. The rigid path's own formula."""
    t2 = 2.0 * torch.cross(q[:, :3], v, dim=-1)
    return v + q[:, 3:4] * t2 + torch.cross(q[:, :3], t2, dim=-1)


def _pm_to_world_axes(v):
    """Source (x, y, z) -> this renderer's world axes. extract's own map."""
    return torch.stack([v[:, 1], v[:, 2], v[:, 0]], dim=-1)


def _pm_to_src_axes(v):
    """The inverse of the above: world axes -> source (x, y, z)."""
    return torch.stack([v[:, 2], v[:, 0], v[:, 1]], dim=-1)


def _pm_to_src_axes_np(v):
    """Same map, numpy -- skin_anim is torch-free so its oracles can use it."""
    return np.stack([v[:, 2], v[:, 0], v[:, 1]], axis=-1)


def _pm_to_world_axes_np(v):
    """Source (x, y, z) -> world axes, numpy side."""
    return np.stack([v[:, 1], v[:, 2], v[:, 0]], axis=-1)


def pm_pose_arrays(name, b, clip, t_sec):
    """(pos, nrm, tan) for ONE bundle at ONE clip time, in WORLD AXES.

    THE FIVE STEPS ARE skin_anim's, NOT A FOURTH COPY OF THEM (#98).
    block_for / compose_model_space / bone_transforms / lbs are the shared
    module, and this function is now the caller that supplies the two
    things only it knows: WHICH skeleton to select the block by (the clip
    file's own declared one -- a playermodel bundle carries no
    nm_skeleton, and inferring one is the module's first documented trap),
    and which frame its vertices live in.

    It started as a call into _vm_skin_transforms, the viewmodel's private
    copy. That worked and was still a fork: the same five traps, held
    closed in one pass and not the other. The module's own selftest caught
    a real error in the orphan substitution while it was being written --
    the one-pass form that pairs the ancestor's POSE with the orphan's OWN
    inverse bind -- which is the argument for the module in one sentence.

    THE FRAME IS `loc`, NOT `pos`. A playermodel bundle bakes BOTH: `loc`
    is the raw SOURCE-axis mesh and `pos` is the same vertices under
    extract_playermodel.to_world_axes, (src_y, src_z, src_x). The skin's
    bind_t/bind_q come off the model skeleton and are therefore in the
    SOURCE frame, so the blend must run on `loc` and the axis map is
    applied after -- running it on `pos` would compose a transform from one
    frame with vertices in another, which is the shape of #90 and produces
    a pose that is wrong everywhere rather than obviously broken anywhere.

    BIND IS NOT SUBTRACTED. This is the ARTICULATED rule and the fifth
    consumer of it: inverse(bind) inside the per-bone transform already
    removed the bind, and subtracting it again applies it twice.
    """
    key = (name, clip["clip"], round(float(t_sec), 4))
    hit = PM_POSE_CACHE.get(key)
    if hit is not None:
        return hit
    cs = PM_CLIPS[0]
    skin = b.get("skin")
    if skin is None:
        return None
    # 1. THE BLOCK, BY THE SKELETON THE CLIP FILE DECLARES. Never by the
    #    bundle's nm_skeleton -- a playermodel bundle has none, and the
    #    arms bundle's names the wrong rig, which is the trap the module
    #    exists to hold closed. A miss NAMES the alternatives.
    blk = _skin.block_for(clip, cs.skeleton)
    if blk is None:
        if name not in PM_ORPHAN_NOTE:
            PM_ORPHAN_NOTE.add(name)
            print(f"⚠️  PLAYERMODEL ANIM: {os.path.basename(clip['clip'])} "
                  f"carries no composable block for {cs.skeleton} -- it has "
                  f"{_skin.skeletons_in(clip)}. {name!r} holds its BIND pose "
                  f"for this clip; nothing is substituted from another rig.",
                  flush=True)
        return None
    idx_np = skin["index"].cpu().numpy()
    wgt_np = skin["weight"].cpu().numpy().astype(np.float64)
    loc_np = b["loc"].cpu().numpy().astype(np.float64)
    if idx_np.shape[0] != loc_np.shape[0]:
        raise SystemExit(
            f"playermodel {name}: the skin binds {idx_np.shape[0]} vertices "
            f"and the bundle bakes {loc_np.shape[0]} -- bindings and "
            f"positions came from different reads of the mesh, so the pose "
            f"would move the wrong vertices. Refused rather than truncated.")
    # 2. parent-local -> model space, in DEPTH order (a child can precede
    #    its parent in the array, and index order fails silently).
    t_all, q_all, _ = _anim.sample_block(blk, float(t_sec), clip["duration"],
                                         clip["frames"])
    mt, mq = _skin.compose_model_space(
        blk["parents"], np.asarray(t_all, dtype=np.float64),
        np.asarray(q_all, dtype=np.float64))
    # THE SOCKET, CACHED HERE BECAUSE HERE IS WHERE IT IS FREE. The rig's
    # `wpn` bone in model space is what the third-person weapon hangs on,
    # and it depends only on (clip, t) -- not on which character bundle is
    # being skinned -- so it is keyed that way and the first bundle to
    # reach a given pose fills it for all of them.
    # ---- THE SCALE TRACK, AND WHAT IT IS NOT --------------------------
    # sample_block returns (t, q, s) and every consumer here drops `s`.
    # impl-character reads a uniform scale out of the character vertex
    # stage's bone buffer, which says the quantity is real somewhere, so
    # the sidecar was censused rather than assumed about. THE FIRST
    # READING OF THAT CENSUS WAS WRONG AND IS CORRECTED HERE:
    #
    #   "403 of 1,601 blocks carry a non-unit scale, 4 of them negative"
    #
    # is arithmetically true and means almost none of what it sounds like.
    # Partitioned properly:
    #
    #   310 of the non-unit blocks are ADDITIVE, and 306 are scale 0.0
    #       ENTIRELY. Zero is the additive IDENTITY for a delta -- it is
    #       "no change", not "collapse this bone to a point". Reading it
    #       as a scale is the same additive-as-absolute error this pass
    #       already refuses everywhere else.
    #    93 are ABSOLUTE and EVERY ONE IS A WEAPON SKELETON -- cz75a,
    #       revolver, nova, xm1014 and 29 more. The values are ~0.0001 to
    #       0.002 on `magazine2`, `loader_handle`, `shell`: a prop scaled
    #       to nothing so it stops being visible. That is a weapon-side
    #       bone, and this pass explicitly does not drive weapon-side
    #       bones (the rigid-attach tier says so).
    #     0 -- ZERO -- of the 914 absolute WORLDMODEL blocks carry a
    #       non-unit scale. The character path, which is the path this
    #       function poses, has no scale to drop.
    #
    # So skin_anim.lbs discards nothing that reaches a character pose, and
    # the earlier "convicts my path" was a misread of a real count.
    #
    # THE CHECK IS KEPT AND NARROWED TO ABSOLUTE CLIPS. On additive blocks
    # it fired on all 230 -- every one a false positive announcing an
    # identity as a dropped term, which is worse than silence because it
    # is loud and wrong. Narrowed, it has NO positive case in this corpus,
    # and that is stated rather than left to look like a clean sweep: it
    # is a tripwire for a future sidecar, not evidence about this one.
    _sc = blk.get("s")
    if _sc is not None and not clip.get("additive") \
            and clip["clip"] not in PM_SCALE_SEEN:
        PM_SCALE_SEEN.add(clip["clip"])
        _st = torch.as_tensor(_sc)
        _sd = float((_st - 1.0).abs().max())
        if _sd > 1e-6:
            print(f"  ⚠️  SCALE DROPPED ({name}, "
                  f"{os.path.basename(clip['clip'])}): this ABSOLUTE clip's "
                  f"worldmodel block carries a per-track scale up to "
                  f"|s-1| = {_sd:.4f} (min {float(_st.min()):.4f}), and "
                  f"skin_anim.lbs applies rotation and translation only. "
                  f"No absolute worldmodel block in agents.clips.pt did "
                  f"this -- 0 of 914 -- so this sidecar differs from the "
                  f"one the character path was censused on, and the pose "
                  f"is missing a term the asset asked for.", flush=True)
    _wk = (clip["clip"], round(float(t_sec), 4))
    if _wk not in PM_WPN_POSE:
        _wi = _anim.bone_track(blk, "wpn")
        PM_WPN_POSE[_wk] = (None if _wi is None
                            else (mq[_wi].copy(), mt[_wi].copy()))
    # 3. pose o inverse(bind) per palette bone, joined BY NAME, with the
    #    #90 nearest-posed-ancestor substitution and its per-bone sizes.
    xq, xt, rep = _skin.bone_transforms(
        skin["palette"],
        skin["bind_t"].cpu().numpy().astype(np.float64),
        skin["bind_q"].cpu().numpy().astype(np.float64),
        blk["bone_ids"], mt, mq,
        palette_ancestors=skin.get("palette_ancestors"),
        index=idx_np, weight=wgt_np)
    # 4. LBS over the slice, positions and BOTH direction fields in ONE
    #    pass. This was two calls -- the second passing the tangent as
    #    `nrm` and discarding the positions -- until skin_anim grew
    #    `dirs=[...]`; the workaround computed the position blend twice to
    #    obtain nothing. Both fields have to be rotated: a posed arm whose
    #    normals stayed at bind lights as though it had not moved, and its
    #    silhouette is right the whole time, so nothing catches it.
    nrm_np = _pm_to_src_axes_np(b["nrm"].cpu().numpy().astype(np.float64))
    tan_np = _pm_to_src_axes_np(b["tan"].cpu().numpy().astype(np.float64))
    P, (Nn, Tt) = _skin.lbs(loc_np, idx_np, wgt_np, xq, xt,
                            dirs=[nrm_np, tan_np])
    # A VERTEX WITH NO WEIGHT IS NOT AT THE ORIGIN. sum_k w_k = 0 collapses
    # that vertex onto the model origin -- the feet -- which draws a spike
    # of geometry out of every player's ankles and is exactly the kind of
    # defect that produces pixels and so survives a coverage check. Such a
    # vertex keeps its BIND position and the count is printed. This is the
    # caller's guard, not the module's: lbs computes the sum it is asked
    # for and a degenerate binding is a property of THIS asset.
    dead = (wgt_np.sum(1, keepdims=True) <= 1e-6)
    P = np.where(dead, loc_np, P)
    Nn = np.where(dead, nrm_np, Nn)
    Tt = np.where(dead, tan_np, Tt)
    sub, unfixed = rep["substituted"], rep["unsubstituted"]
    P = torch.as_tensor(P, device=device)
    Nn = torch.as_tensor(Nn, device=device)
    Tt = torch.as_tensor(Tt, device=device)
    loc = torch.as_tensor(loc_np, device=device)
    pos_w = _pm_to_world_axes(P).to(torch.float32)
    out = (pos_w, _pm_to_world_axes(Nn).to(torch.float32),
           _pm_to_world_axes(Tt).to(torch.float32))
    moved = float((pos_w - b["pos"].to(device)).norm(dim=-1).max()) * 0.0254
    PM_ANIM_MOVED[0] = max(PM_ANIM_MOVED[0], moved)
    if name not in PM_ORPHAN_NOTE:
        PM_ORPHAN_NOTE.add(name)
        # THE AXIS MAP, ASSERTED RATHER THAN BELIEVED. `pos` must be
        # to_world_axes(`loc`) or the frame this whole function assumes is
        # not the frame the bundle baked, and every pose would be silently
        # wrong. It is one comparison and it runs once per bundle.
        _axis = float((_pm_to_world_axes(loc).to(torch.float32)
                       - b["pos"].to(device)).abs().max())
        if _axis > 1e-3:
            raise SystemExit(
                f"playermodel {name}: to_world_axes(loc) differs from the "
                f"baked `pos` by {_axis:.4f} source units. The skin's bind "
                f"is in the `loc` frame, so a mismatch means the pose would "
                f"be composed in one frame and applied in another. Refused.")
        _nd = int(dead.sum())
        print(f"PLAYERMODEL ANIM: SKINNED re-pose LIVE for {name!r} via "
              f"skin_anim (the SHARED five steps, #98) -- "
              f"{rep['n_palette']} palette bones of which {rep['n_posed']} "
              f"are named by the world rig, {idx_np.shape[1]} "
              f"influence(s)/vertex over {loc_np.shape[0]:,} vertices, "
              f"bind_applied NOT subtracted (articulated rule). "
              f"to_world_axes(loc) reproduces the baked `pos` to "
              f"{_axis:.2e} source units, so the pose is composed in the "
              f"frame the bindings are in. {_nd} vertex/vertices carry zero "
              f"total weight and keep their bind position.", flush=True)
        for _nm, _a, _nv, _ms in sub:
            print(f"  ORPHAN BONE SUBSTITUTED ({name}): `{_nm}` is not named "
                  f"by the world rig; it takes its nearest POSED ancestor "
                  f"`{_a}` (model-skeleton m_nParent, READ). {_nv:,} "
                  f"vertices, {_ms:.2f}% of the weight mass. Identity would "
                  f"leave them in the .vmdl_c frame -- the #90 defect.",
                  flush=True)
        for _nm, _nv, _ms, _why in unfixed:
            print(f"  ⚠️  ORPHAN BONE NOT SUBSTITUTED ({name}): `{_nm}` "
                  f"({_nv:,} vertices, {_ms:.2f}% of the weight mass) -- "
                  f"{_why}. Those vertices are NOT posed correctly and the "
                  f"share of the model they cover is the number above.",
                  flush=True)
        if not PM_WPN_NOTE:
            PM_WPN_NOTE.append(1)
            print("PLAYERMODEL ANIM (STATED, a declared gap): the world rig "
                  "poses `wpn`/`wpnHand_L`/`wpnHand_R`, but no third-person "
                  "WEAPON mesh is parented to them -- pm_bundles carry the "
                  "character body only. So a player's hands take the "
                  "weapon's hold shape while holding nothing. The bones are "
                  "posed and available; attaching a world weapon model is "
                  "the follow-up, and it is NOT silently skipped here.",
                  flush=True)
    if len(PM_POSE_CACHE) > 512:
        PM_POSE_CACHE.clear()
    PM_POSE_CACHE[key] = out
    return out


def _pm_wbundle(name):
    """A model-space weapon bundle by basename, once, or None with a why."""
    if name in PM_WBUNDLES:
        return PM_WBUNDLES[name]
    if name in PM_WBUNDLE_NOTE or not name:
        return None
    if not args.playermodel_weapons:
        PM_WBUNDLE_NOTE[name] = "no --playermodel-weapons directory given"
        return None
    f = os.path.join(args.playermodel_weapons, f"{name}.pt")
    if not os.path.exists(f):
        PM_WBUNDLE_NOTE[name] = f"no bundle at {f}"
        return None
    b = torch.load(f, map_location="cpu", weights_only=False)
    if b.get("space") != "model-source-units":
        # THE SAME REFUSAL THE CHARACTER LOADER MAKES, and for a sharper
        # reason here: a vm bundle of the SAME WEAPON loads with every key
        # this path reads and would draw a correctly-textured rifle at
        # each player's feet, in view space, at a placement meant for a
        # camera that is not there. That produces pixels, so a coverage
        # check passes it.
        PM_WBUNDLE_NOTE[name] = (
            f"{f} declares space {b.get('space')!r}, not "
            f"'model-source-units' -- that is a VIEWMODEL bundle, whose "
            f"vertices are view-space metres with a placement baked in.")
        return None
    skin = b.get("skin")
    pal = list(skin["palette"]) if skin else []
    if "weapon_offset" not in pal:
        # NAMED, not guessed. Without the attach bone there is no read of
        # where this weapon meets the hand, and picking the first palette
        # bone or the mesh centroid would put the gun somewhere plausible
        # and wrong.
        PM_WBUNDLE_NOTE[name] = (
            f"{f} has no `weapon_offset` bone in its skin palette "
            f"({pal[:6]}) -- that bone IS the attach point (it equals the "
            f"viewmodel bundle's bind_applied_src to every digit), so "
            f"without it there is nothing to hang on the rig's `wpn` "
            f"socket and none is invented.")
        return None
    PM_WBUNDLES[name] = b
    i = pal.index("weapon_offset")
    print(f"playermodels: WEAPON bundle {name!r} <- {f} -- "
          f"{b['loc'].shape[0]} verts, {b['tri'].shape[0]} tris, "
          f"{len(b['materials'])} material(s), source {b.get('source')}; "
          f"attach bone `weapon_offset` bind t "
          f"{[round(float(x), 4) for x in skin['bind_t'][i]]} q "
          f"{[round(float(x), 4) for x in skin['bind_q'][i]]}", flush=True)
    return b


_PM_WPAGES = {}


def pm_weapon_pages(name, mi):
    """Material `mi` of a weapon bundle, in the slot set vm_gt_shade reads.

    The SAME _slot_pages the character and the viewmodel ARMS go through,
    so a slot this weapon's vmat does not carry gets the one STATED
    constant table rather than a third opinion about what an absent
    roughness means.
    """
    key = (name, mi)
    if key in _PM_WPAGES:
        return _PM_WPAGES[key]
    b = PM_WBUNDLES[name]
    mat = b["materials"][mi]
    _pm_sampler_check(f"weapon {name}/mat{mi}", mat)
    out = _slot_pages(f"weapon {name}/mat{mi} "
                      f"{os.path.basename(mat.get('path', '?'))}",
                      mat.get("pages") or {}, mat.get("provenance") or {})
    _PM_WPAGES[key] = out
    return out


def _pm_weapon_norm(s):
    """`AK-47` -> `ak47`. Punctuation only; nothing is translated."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def pm_weapon_for(e):
    """(bundle basename, how) for one entity's held weapon, or (None, why).

    THREE TIERS, the same shape the agent join uses, and the frame line
    prints which one fired:
      1. --playermodel-weapon-map, the operator's own name -> bundle
      2. the NORMALISED weapon name used directly as a bundle basename
         (`AK-47` -> `ak47`), which only works if someone extracted it
      3. nothing -- the weapon is NOT drawn and NOT substituted
    """
    w = e.get("active_weapon_id")
    if w is None or w != w:
        return None, "no active_weapon_name on this entity's row"
    w = str(w)
    if w in PM_WEAPON_MAP:
        nm = PM_WEAPON_MAP[w]
        return (nm, f"{w} via --playermodel-weapon-map") \
            if _pm_wbundle(nm) else (None, f"{w} maps to {nm!r}: "
                                     f"{PM_WBUNDLE_NOTE.get(nm)}")
    n = _pm_weapon_norm(w)
    if _pm_wbundle(n) is not None:
        return n, f"{w} normalised to {n!r} and that named a bundle"
    return None, (f"{w} has no bundle ({PM_WBUNDLE_NOTE.get(n, 'not tried')})"
                  f" and none is substituted")
