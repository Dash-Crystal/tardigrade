

def weapon_shade(uv4, wpos, lpos, nrm_g, tan_g, view, eye, sun_dir,
                 sun_rgb, ambient, tsec, frag_xy, cam_up, cam_fwd,
                 covered, x, bg):
    """csgo_weapon -- the beauty path, all twelve static axes.

    Reference modules, one per axis, each named by its bytecode record
    (m_nByteCodeDataIdx, never a position -- vcs/README.md rule 2):

      S_ENABLE_ADJUSTMENTS  r1     enable_adjustments1_c1
      S_MODE_TOOLS_VIS      r2     mode_toolvis1_c2
      S_ALPHA_TEST          r4     alpha_test1_c4
      S_TRANSLUCENT         r8     translucent1_c8
      S_ADDITIVE_BLEND      r10    additive_blend1_c24  (16 alone unshipped)
      S_TINT_MASK           r12    tint_mask1_c32
      S_GLITTER             r24    glitter1_c64
      S_ENABLE_SFX_MASK     r0     == the baseline record, byte for byte
      S_SELF_ILLUM          r48    self_illum1_c256
      S_STICKERS            r96    stickers1_c512
      S_MODE_DEPTH          r192   mode_depth1_c1024
      S_OPAQUE_REFRACT      r218   == r0 byte for byte

    Combo ids decode MIXED-RADIX (all twelve axes are binary here, so the
    strides are 1,2,4,...,2048); the decode was checked by reconstructing
    all 672 shipped ids from their per-axis values, 0 mismatches.

    STATED DIFFERENCE, and it is the largest one here: the shared lighting
    is NOT the engine's. csgo_weapon's 2,326-FLOP floor is dominated by
    the same cascade-shadow loop (51 FLOPs/iteration, 27 sampler2DShadow
    taps), the same clustered-light binner pair and the same cubemap
    binner pair the world families use, and 27 of its 44 fetch sites are
    the shadow atlas. This function does NOT re-run that machinery for the
    viewmodel pass -- it lights the weapon with one directional term plus
    an ambient term and a GGX specular lobe. What that omits, precisely:
    the cascade shadow (so the viewmodel is never in shadow), the
    clustered point/spot lights, the IBL cubemap specular and the probe
    volumes. Every csgo_weapon-SPECIFIC term above is transcribed
    literally; the shared lighting is the closest complete form.

    Returns (rgb, alpha, keep).
    """
    tex = vm_textures()
    if bg is not None:
        tex = dict(tex)
        tex["bgrt"] = bg
    uv = uv4[..., :2]
    ntot = int(covered.sum())
    # --- the shared surface set, allon_c3041_d2:526-542 -------------
    C = vm_sample(tex["colour"], uv)
    ao = vm_sample(tex["ao"], uv)[..., 0]
    M = vm_sample(tex["mask"], uv)
    rough = M[..., 0]                             # allon_c3041_d2:535
    # allon_c3041_d2:533 is mix(lo, hi, mask.y), and what it produces is
    # spent as the metalness lerp into F0 -- NOT a roughness remap. The
    # roughness is mask.x at :535. Reading :533 as roughness throws the
    # term away and leaves F0 dielectric everywhere, which is what the
    # first draft of this function did.
    metal = wpn.metal_remap(_e2(x, "wpn_metal_lo"), _e2(x, "wpn_metal_hi"),
                            M[..., 1])
    mask_z = M[..., 2]
    Ntex = vm_sample(tex["normal"], uv)
    nrm_ts = _octa_normal(Ntex[..., 0], Ntex[..., 1])
    vcol = torch.ones_like(C)                     # the vertex colour tint
    # --- S_TINT_MASK (r12) --------------------------------------------
    if args.weapon_tint_mask:
        tm = vm_sample(tex["tint_mask"], uv)[..., 0]
        alb = wpn.vcolor_tint(C, vcol, tm)        # allon_c3041_d2:528-529
        _wpn_reach("S_TINT_MASK", (tm > 0) & covered, ntot)
    else:
        # Without the axis the mask is absent, and mix(C, C*vcol, 1) is
        # the same expression at mask == 1.
        alb = C * vcol
    # --- S_ENABLE_SFX_MASK (r0 -- no bytecode of its own) --------------
    sfx = torch.zeros_like(rough)
    if args.weapon_sfx_mask:
        alb, rough, nrm_ts, uv, sfx = weapon_sfx_mask(
            uv, wpos, lpos, nrm_ts, alb, rough, tsec, x, tex)
        _wpn_reach("S_ENABLE_SFX_MASK", (sfx > 0) & covered, ntot)
    # --- the tangent frame, allon_c3041_d2:562-586 -------------------
    Ng = nrm_g / nrm_g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    T = tan_g - Ng * (Ng * tan_g).sum(-1, keepdim=True)
    T = T / T.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    Bt = torch.cross(Ng, T, dim=-1)
    # --- S_STICKERS (r96) ---------------------------------------------
    stk = torch.zeros_like(rough)
    if args.weapon_stickers:
        alb, rough, nrm_ts, stk = weapon_stickers(
            uv4, view, T, Bt, Ng, nrm_ts, alb, rough, x, tex,
            cam_up, cam_fwd, args.weapon_sticker_slots)
        _wpn_reach("S_STICKERS", (stk > 0) & covered, ntot)
    # --- S_GLITTER (r24) ----------------------------------------------
    glit = torch.zeros_like(alb)
    if args.weapon_glitter:
        glit, alb, rough, metal, nrm_ts = weapon_glitter(
            uv, view, T, Bt, Ng, nrm_ts, alb, rough, metal, mask_z, x, tex,
            per_view_hi=False)
        _wpn_reach("S_GLITTER", (_luma(glit) > 0) & covered, ntot)
    # world-space shading normal from the tangent frame,
    # allon_c3041_d2:586
    Wn = wpn.tangent_frame_normal(T, Bt, Ng, nrm_ts)
    # --- S_ENABLE_ADJUSTMENTS (r1) ------------------------------------
    if args.weapon_adjustments:
        alb = weapon_adjustments(alb, view, Wn, sun_dir, mask_z, x)
        _wpn_reach("S_ENABLE_ADJUSTMENTS", (mask_z > 0) & covered, ntot)
    # --- S_OPAQUE_REFRACT (r218 == r0) + D_USE_BGREFRACT --------------
    refr = torch.zeros_like(rough)
    if args.weapon_opaque_refract:
        # `stk` is the sticker accumulator, which the all-on module folds
        # into the refract weight as (1 - stickerAccum.x) at :5550. The
        # sticker-free module has no such factor, so it is passed in rather
        # than baked into the term. Order matches the reference: the
        # accumulator is established at :5426, the refract reads it at
        # :5550.
        alb, refr = weapon_opaque_refract(frag_xy, view, Wn, cam_up,
                                          cam_fwd, alb, uv, x, tex,
                                          (uv.shape[2], uv.shape[1]),
                                          sticker_a=stk)
        _wpn_reach("S_OPAQUE_REFRACT", (refr > 0) & covered, ntot)
        _wpn_reach(f"D_USE_BGREFRACT={args.weapon_bgrefract}",
                   (refr > 0) & covered, ntot)
    # --- lighting (see the STATED DIFFERENCE above) --------------------
    rough = rough.clamp(0.03, 1.0)
    ndl = (Wn * sun_dir).sum(-1).clamp(min=0.0)
    diff = alb * (ndl.unsqueeze(-1) * sun_rgb + ambient
                  * ao.unsqueeze(-1))
    hv = sun_dir + view
    hv = hv / hv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    a2 = (rough * rough).clamp(min=1e-4) ** 2
    ndh = (Wn * hv).sum(-1).clamp(min=0.0)
    dgg = a2 / (math.pi * ((ndh * ndh) * (a2 - 1.0) + 1.0) ** 2).clamp(
        min=1e-8)
    f0 = 0.039999999105930328369140625 + (alb - 0.04) * metal.unsqueeze(-1)
    spec = f0 * (dgg * ndl).unsqueeze(-1)
    rgb = diff * (1.0 - metal.unsqueeze(-1)) + spec
    # the glitter radiance rides on the diffuse product, UNSOURCED-glsl:1321
    rgb = rgb + diff * glit
    # --- the SFX shimmer ----------------------------------------------
    # reference/csgo_weapon/csgo_weapon__allon_c3041_d2.glsl:6884-6888.
    # Gated by its own UNIFORM (`_5618._m197 > 0`), not by a static axis:
    # S_ENABLE_SFX_MASK emits no bytecode of its own, so combos 0 and 128
    # at d2 are the same SPIR-V (sha256 838cd370e8c6095e...).
    #
    # The expression is `wpn.shimmer`, A/B'd against the decompiled range
    # as csgo_weapon.t11.shimmer. TWO DEFECTS THE REFERENCE CAUGHT when
    # this was written against an uncommitted module:
    #
    #   * `mix(k * dither.y, 1.0, k)` was implemented as
    #     `k*dither + (1-k)`, the mix operands swapped. Still a lerp,
    #     still in [0,1], still smooth -- and the other term entirely.
    #     Facing the viewer (k -> 0) the reference gives ~0 and the
    #     inversion gives 1.0, i.e. the shimmer was brightest exactly
    #     where the module puts none.
    #   * `_5618._m198` is a vec3 TINT. It was read from
    #     `wpn_refract_tint`, a scalar column belonging to a different
    #     term, which cannot carry a colour at all. It now reads the
    #     three wpn_shimmer_tint_{r,g,b} columns.
    #
    # The exponent's mix parameter is `1 + sin(t*20)*0.5`, outside [0,1]
    # for half its period, so the pow EXTRAPOLATES past 6 rather than
    # oscillating between 3 and 6. Transcribed, not tidied.
    shim = _e2(x, "wpn_shimmer")
    if bool((shim > 0).any()):
        ts = tsec.view(-1, 1, 1) if torch.is_tensor(tsec) else float(tsec)
        ts = torch.as_tensor(ts, device=device, dtype=torch.float32)
        tint = torch.stack([_e2(x, "wpn_shimmer_tint_r"),
                            _e2(x, "wpn_shimmer_tint_g"),
                            _e2(x, "wpn_shimmer_tint_b")], dim=-1)
        rgb = wpn.shimmer(rgb, (Wn * view).sum(-1), _vm_dither_at(covered),
                          tint, shim, ts)
        _wpn_reach("sfx_shimmer", (shim > 0) & covered, ntot)
    # --- S_SELF_ILLUM (r48) -------------------------------------------
    if args.weapon_self_illum:
        # WAS an inline substitution: fract-scroll the uv, sample, ADD. The
        # module's :588-589 tints the emissive by _5618._m19 and then mixes
        # toward emissive*albedo by _5618._m21 -- neither the tint nor the
        # albedo re-modulation existed here, so a tinted self-illum read
        # white and a fully albedo-blended one read unblended.
        t = tsec.view(-1, 1, 1, 1) if torch.is_tensor(tsec) else float(tsec)
        su = _e2(x, "wpn_si_scroll_u").unsqueeze(-1)
        sv = _e2(x, "wpn_si_scroll_v").unsqueeze(-1)
        si_uv = wpn.self_illum_uv(uv, torch.cat([su, sv], dim=-1), t)
        emis_tex = vm_sample(tex["emissive"], si_uv)[..., :3]
        si_tint = torch.stack([_e2(x, "wpn_si_tint_r"),
                               _e2(x, "wpn_si_tint_g"),
                               _e2(x, "wpn_si_tint_b")], dim=-1) \
            if _e2_has(x, "wpn_si_tint_r") else torch.ones_like(emis_tex)
        si_blend = _e2(x, "wpn_si_albedo_blend") \
            if _e2_has(x, "wpn_si_albedo_blend") else torch.zeros_like(rough)
        _wpn_defaults_banner(x)
        rgb = rgb + wpn.self_illum(emis_tex, si_tint, C[..., :3], si_blend)
        _wpn_reach("S_SELF_ILLUM", covered, ntot)
    # --- the SFX mask flow (r?, uniform-gated at :451) -----------------
    # allon_c3041_d2:451-477. The gate is a PRODUCT of two uniforms, not a
    # static axis: S_ENABLE_SFX_MASK emits no bytecode of its own.
    _sfx_gate = wpn.sfx_gate(_e2(x, "wpn_shimmer"), _vm_per_view_scale())
    if bool((_sfx_gate > 0).any()):
        _texel_ratio = float(1.0 / max(int(args.vm_tex_size), 1)) \
            if hasattr(args, "vm_tex_size") else 0.001
        _sfx_uv = wpn.sfx_uv_scale(uv, _texel_ratio)
        _ph = tsec if torch.is_tensor(tsec) else \
            torch.as_tensor(float(tsec), device=device)
        _wipe = wpn.sfx_wipe(_sfx_gate, _ph.view(-1, 1, 1),
                             _e2(x, "wpn_sfx_wipe_width")
                             if _e2_has(x, "wpn_sfx_wipe_width")
                             else torch.full_like(rough, 0.25),
                             lpos[..., 2])
        _flow = vm_sample(tex["mask"], _sfx_uv)
        _sfx_uv2 = wpn.sfx_uv_offset(_sfx_uv, _flow[..., :2], _wipe,
                                     _flow[..., 3])
        rgb = rgb + vm_sample(tex["emissive"], _sfx_uv2)[..., :3] \
            * _wipe.unsqueeze(-1)
        _wpn_reach("sfx_mask_flow", (_sfx_gate > 0) & covered, ntot)
    # --- alpha: S_ALPHA_TEST / S_TRANSLUCENT / S_ADDITIVE_BLEND -------
    alpha = torch.ones_like(rough)
    if args.weapon_alpha_test:
        # the interpolated vertex fade is replaced by a computed cutout
        aref = _e2(x, "wpn_alpha_ref")
        alpha = (C[..., 0] * 0.0 + vm_sample(tex["mask"], uv)[..., 3])
        keep_at = alpha >= aref
        covered = covered & keep_at
        _wpn_reach("S_ALPHA_TEST", keep_at & covered, ntot)
    if args.weapon_translucent:
        alpha = alpha * (0.35 + 0.65 * mask_z)
        _wpn_reach("S_TRANSLUCENT", covered, ntot)
    if args.weapon_additive_blend:
        if not args.weapon_translucent:
            raise SystemExit(
                "--weapon-additive-blend without --weapon-translucent is "
                "an unshipped combo: static id 16 ships no module, and the "
                "lowest shipped id carrying S_ADDITIVE_BLEND is 24 = 8+16. "
                "Add --weapon-translucent.")
        # the fog attenuation the axis multiplies into the output alpha
        d = (wpos - eye[:, None, None, :]).norm(dim=-1)
        alpha = alpha * (1.0 - (d / max(VM_RANGE.far, 1e-6)).clamp(0.0, 1.0))
        _wpn_reach("S_ADDITIVE_BLEND", covered, ntot)
    # --- S_MODE_TOOLS_VIS (r2) ----------------------------------------
    if args.weapon_tools_vis:
        rgb = torch.stack([mask_z, rough, ao], dim=-1)
        _wpn_reach("S_MODE_TOOLS_VIS", covered, ntot)
    # --- D_MOUSE_TRACE_COORD ------------------------------------------
    if args.weapon_mouse_trace == "on":
        # MEASURED: adds NO fetch site. It adds an interpolated vec2 at
        # location 1 -- renumbering every later location -- read once as a
        # texture coordinate (UNSOURCED-glsl:646), and imageStore writes of the
        # traced hit's local x/y/z plus a hit flag (UNSOURCED-glsl:6862-6883). It is
        # a CPU-readback side channel, so the reproduction is a readback
        # dict rather than a shading term, and it is NOT folded into rgb.
        cx, cy = uv.shape[2] // 2, uv.shape[1] // 2
        WPN_TRACE.clear()
        WPN_TRACE.update({
            "hit": bool(covered[:, cy, cx].any()),
            "local_x": float(lpos[:, cy, cx, 0].mean()),
            "local_y": float(lpos[:, cy, cx, 1].mean()),
            "local_z": float(lpos[:, cy, cx, 2].mean()),
        })
        _wpn_reach("D_MOUSE_TRACE_COORD", covered, ntot)
    # --- the two screen-doors in every module's tail -------------------
    keep = vm_proximity_dissolve(wpos, covered)
    _wpn_reach("dissolve", keep, ntot)
    keep = vm_screendoor(alpha, keep)
    # --- S_MODE_DEPTH (r192) ------------------------------------------
    if args.weapon_mode_depth:
        # The 26-line depth-only module: the SAME screen-door, then
        # vec4(0,0,0,1). It REPLACES the beauty shade, exactly as the
        # combo does in the engine -- it does not modify it.
        rgb = torch.zeros_like(rgb)
        alpha = torch.ones_like(alpha)
        _wpn_reach("S_MODE_DEPTH", keep, ntot)
    # TWO AXES THAT SELECT NOTHING, and used to print a coverage number.
    #
    # --weapon-baked-lighting and --weapon-specular-cube-static were read at
    # exactly ONE site each: this loop, which then reported them at 100% and
    # 0% of covered pixels. Neither value reaches a single shading decision.
    # A no-op flag is bad; a no-op flag that PRINTS ITS OWN REACH is worse,
    # because the number is read as evidence the axis was exercised -- the
    # earlier arm of this pass printed "D_BAKED_LIGHTING=none 100.0000%"
    # about a term that does nothing.
    #
    # They are NOT deleted: the charter's rule is that a missing input gets
    # built, not gated away, and both are real dynamic axes of the family
    # (D_BAKED_LIGHTING_FROM_{VERTEX_STREAM,PROBE,LIGHTMAP} at bits 0-2 and
    # D_SPECULAR_CUBE_MAP_STATIC at bit 3, 2,880 and 5,120 of 11,520 shipped
    # pairs). What they need is the weapon's baked-lighting and cubemap
    # paths, which this pass does not run -- the same STATED DIFFERENCE as
    # the cascade and cluster machinery. Until then they declare themselves
    # unimplemented at runtime instead of reporting a fraction.
    _wpn_unimpl("D_BAKED_LIGHTING=" + args.weapon_baked_lighting,
                "selects nothing: the viewmodel pass runs no baked-lighting "
                "path, so all four values shade identically")
    _wpn_unimpl("D_SPECULAR_CUBE_MAP_STATIC="
                + ("1" if args.weapon_specular_cube_static else "0"),
                "selects nothing: the viewmodel pass runs no cubemap "
                "specular, so both values shade identically")
    return rgb.clamp(min=0.0), alpha, keep


WPN_TRACE = {}


# ======================================================================
# THE REAL BUNDLE'S OWN DRAW -- gt_direct() over the extractor's pages
# ======================================================================
# WHAT THIS FIXES, and it is a draw that was reaching zero pixels rather
# than a term that was slightly wrong.
#
# The weapon family only shaded under `--weapon` (the `if args.weapon and
# m_wpn.any()` below). pair_compare launches the per-frame viewmodel with
# `--viewmodel --viewmodel-model <bundle>` and NOTHING ELSE (ceeb2152,
# viewmodel_args :133), so on every paired frame this pass loaded the
# bundle, rasterised it, counted its coverage into VM_REACH -- and then
# fell through `if not live.any(): return world_rgb` having composited
# nothing. A coverage counter said the geometry was there and the image
# was untouched, which is the exact pair of states the per-frame print
# below now separates.
#
# `--weapon` is NOT made the default to fix that. It selects the twelve
# csgo_weapon static axes, whose inputs are the SYNTHESISED pages plus a
# defaulted ext2 row (vm_ext2_row's `MAT_EXT2[0].clone()` arm) on any pack
# with no csgo_weapon material -- i.e. turning it on to get pixels would
# have shaded a real mesh with fabricated skin parameters and called it
# the weapon. This is the other path: the SHARED lighting the world
# families already use, over the bundle's OWN pages, with no csgo_weapon
# axis in it.
#
# STATED DIFFERENCE, in full. This is gt_direct() -- the transcribed sun
# BRDF at the selected S_SHADER_QUALITY -- plus the sun colour and
# direction READ from the map's light_environment, times the bundle's ao
# page. What it does NOT carry: the cascade shadow (so the viewmodel is
# never in shadow), the clustered lights, the IBL cube and the probe
# volumes. That is the same omission set weapon_shade's docstring already
# declares, reached through the shared function instead of a second
# hand-written lobe.
_VM_IRR_NOTED = []


def vm_gt_shade(uv4, nrm_g, tan_g, view, covered, tex=None, wpos=None):
    """The bundle's colour/normal/mask/ao pages, lit by gt_direct().

    Returns (rgb, alpha, keep) -- vm_gt_shade is a drop-in for
    weapon_shade's return contract so the composite below does not branch.

    Every page is read at the SLOT NAME vm_textures() publishes, so a
    bundle that supplies a real page gets it and a bundle that does not
    gets the synthesised stand-in AND the provenance line that says so.
    The channel assignments are csgo_weapon's own, cited where
    weapon_shade cites them: colour.rgb albedo (allon_c3041_d2:526),
    mask.x roughness (:535), mask.y the metalness lerp (:533), the
    two-channel octahedral normal (:540-542), ao.x.

    `tex` defaults to the VIEWMODEL slot set. The playermodel pass passes
    its own, built from its own bundle: the shading maths is one function
    so the two draws cannot drift, and the PAGES are per-draw so a
    character is never lit through a weapon's albedo.
    """
    tex = vm_textures() if tex is None else tex
    uv = uv4[..., :2]
    C = vm_sample(tex["colour"], uv)[..., :3]
    ao = vm_sample(tex["ao"], uv)[..., 0]
    M = vm_sample(tex["mask"], uv)
    # mask.x IS the roughness and mask.y is spent as the metalness lerp --
    # the same read weapon_shade's comment records getting backwards on its
    # first draft, made here from the same two channels.
    rough = M[..., 0].clamp(0.03, 1.0)
    metal = M[..., 1].clamp(0.0, 1.0)
    Ntex = vm_sample(tex["normal"], uv)
    nrm_ts = _octa_normal(Ntex[..., 0], Ntex[..., 1])
    # the tangent frame, allon_c3041_d2:562-586 -- the extractor derives
    # `tan` from the UV gradient, so this Gram-Schmidts it against the
    # interpolated normal exactly as the world families do
    Ng = nrm_g / nrm_g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    T = tan_g - Ng * (Ng * tan_g).sum(-1, keepdim=True)
    T = T / T.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    Bt = torch.cross(Ng, T, dim=-1)
    Wn = wpn.tangent_frame_normal(T, Bt, Ng, nrm_ts)
    # F0, in gt_compose's form (:15553) rather than weapon_shade's --
    # dielectric 0.04 lerped toward the albedo by metalness.
    f0 = 0.04 * (1.0 - metal).unsqueeze(-1) \
        + C.clamp(0, 1) * metal.unsqueeze(-1)
    # THE SUN IS THE MAP'S, READ. _vm_sun_dir()/_vm_sun_rgb() are STATED
    # constants that predate the light_environment read; using them here
    # would light the weapon by a different sun than the wall behind it,
    # which is a defect that looks like a shading difference.
    l = sun_dir.view(1, 1, 1, 3).expand_as(view)
    atten = torch.ones_like(rough)
    diff, spec = gt_direct(args.shader_quality, Wn, l, view, f0, rough,
                           SUN_COLOR.view(1, 1, 1, 3), atten)
    ao3 = ao.unsqueeze(-1)
    # --- THE INDIRECT TERM, from the SAME volume the world reads --------
    # The non-world draws had NO baked/indirect term at all: gt_direct
    # returns the read sun plus the ambient addend and nothing else, so
    # when d82ffa5a lifted the world's backdrop by +0.584 mean the
    # viewmodel moved 0.00000 and the playermodels +0.00001. Measured by
    # loopfam over their own masks (42,685 px and 630 px against the
    # backdrop's 67,952) -- the models were not slightly behind, they were
    # on a different branch.
    #
    # Fixed in kind rather than by a constant: sample_volume() at the
    # model's own world position, which is the INPUT d82ffa5a wired for
    # world pixels -- this map's resample of Valve's own bake, not a
    # stand-in. `wpos` is the interpolated world position both passes
    # already compute (viewmodel_pass recovers it by inverting the view
    # matrix; playermodel_pass has it directly).
    irr = None
    if VOL is not None and wpos is not None:
        irr = sample_volume(wpos)
        _gt_value("nonworld.volume_indirect", irr.mean(-1), identity=0.0,
                  note="sample_volume() at the non-world draw's own world "
                       "position -- the same reconstructed irradiance "
                       "volume the world path samples (d82ffa5a), read at "
                       "a different point")
    if not _VM_IRR_NOTED:
        _VM_IRR_NOTED.append(1)
        if irr is None:
            print("GAP non-world indirect: no irradiance volume "
                  "(--irr-vol 0, or --irr-npy absent), so the viewmodel "
                  "and playermodels carry the sun and the ambient addend "
                  "and NO indirect -- which is what left them unmoved "
                  "when the world's backdrop lifted.", flush=True)
        else:
            print("non-world indirect: SAMPLED from the reconstructed "
                  "irradiance volume at the draw's own world position, "
                  "the same input the world path reads. STATED "
                  "DIVERGENCES, unchanged from the world's: magnitude "
                  "only -- no per-volume transform, no fade distance, no "
                  "second occlusion sampler, and NO DIRECTIONALITY, so "
                  "the sample does not vary with the surface normal.",
                  flush=True)
    # csgo_environment_ps.glsl:934 -- (diff + bakedIrr*ao) * albedo*(1-metal),
    # then :940 + spec*ao. TWO changes from what this line used to be, and
    # the second is a correction rather than an addition:
    #   * the indirect addend now exists at all;
    #   * `diff` is no longer multiplied by ao. :934 applies ao to the
    #     BAKED term only -- the direct sun already carries its own
    #     shadow -- and multiplying both was a divergence from the
    #     composition gt_compose uses for world pixels.
    dalb = C * (1.0 - metal).unsqueeze(-1)
    ind = irr * ao3 if irr is not None else 0.0
    rgb = (diff + ind) * dalb + spec * ao3
    alpha = torch.ones_like(rough)
    return rgb.clamp(min=0.0), alpha, covered


# ======================================================================
# THE VIEWMODEL'S THREE FAMILIES, named
# ======================================================================
# The pass dispatched on the literals 0 and 1. A third family cannot be
# added to two literals without the reader having to know which is which,
# and the arms are a third family rather than more weapon: they are SKIN
# AND CLOTH, and shading them through csgo_weapon's axes would run a
# rifle's tint mask, glitter, stickers and opaque-refract over a glove.
VM_FAM_WEAPON = 0
VM_FAM_LEGS = 1
VM_FAM_ARMS = 2
VM_FAM_NAME = {VM_FAM_WEAPON: "weapon", VM_FAM_LEGS: "legs",
               VM_FAM_ARMS: "arms"}

# What the bundle that last loaded actually was, so the per-frame line can
# name it without re-opening the file.
VM_BUNDLE_INFO = {}
_VM_ARMS_TEX = [None]
_VM_ARMS_CACHE = {}


def vm_arms_path(weapon_path):
    """`arms_<weapon>.pt` beside the weapon bundle. Name, not a search."""
    d, base = os.path.split(weapon_path)
    return os.path.join(d, "arms_" + base)


def vm_load_arms(weapon_path):
    """(geometry 8-tuple, note) for the arms that hold THIS weapon, or None.

    The arms are extracted per weapon because they are SKINNED TO THAT
    WEAPON'S IDLE CLIP -- arms_galil_ar.pt records
    `pose_clip: animation/anims/viewmodel/rifle/rifle_galilar/
    idle_galilar.vnmclip` -- so there is no one arms mesh to share and the
    file is named after the weapon it was posed with. A weapon whose arms
    were never extracted draws without them and says so.

    THE `fam` COLUMN IS OVERRIDDEN, and that is the point of this function.
    The bundle ships fam == 0 for all 9,291 of its triangles, i.e. WEAPON,
    because it was written against vm_synth_geometry's two-family contract.
    Concatenating it unchanged would put a glove through weapon_shade or
    through the weapon's page set -- skin shaded as a rifle skin. So the
    column is rewritten to VM_FAM_ARMS here, at the load, and the rewrite
    is PRINTED rather than done quietly: it is this renderer overriding
    what the asset says about itself.
    """
    p = vm_arms_path(weapon_path)
    if not os.path.exists(p):
        # cleared, not left stale: vm_load_geometry runs per batch, and a
        # page set held over from a previous bundle is the kind of thing
        # that only shows up when two weapons render in one process
        _VM_ARMS_TEX[0] = None
        _VM_ARMS_CACHE.clear()
        return None, f"no arms bundle at {os.path.basename(p)}"
    if _VM_ARMS_CACHE.get("path") == p:
        _VM_ARMS_TEX[0] = _VM_ARMS_CACHE["tex"]
        return _VM_ARMS_CACHE["geom"], _VM_ARMS_CACHE["note"]
    b = torch.load(p, map_location="cpu", weights_only=False)
    need = ("pos", "uv", "nrm", "tan", "loc", "tri", "fam", "is_legs")
    miss = [k for k in need if k not in b]
    if miss:
        return None, f"{os.path.basename(p)} is missing {miss}"
    n, t = b["pos"].shape[0], b["tri"].shape[0]
    # THE FAMILY CENSUS, taken from the fam COLUMN rather than from the
    # triangle total. _vm_class_assert's precondition is "this bundle ships
    # weapon geometry", and its first version answered that with
    # VM_BUNDLE_INFO['tris'] -- which is the whole bundle's triangle count.
    # Pass an arms-only bundle as --viewmodel-model and that reads as
    # "ships a weapon", every triangle is family ARMS, weapon pixels are
    # legitimately zero, and the check fires on a bundle doing exactly what
    # it says. Counting the column cannot make that mistake, and it is the
    # same read-the-field rule the rest of this file runs on.
    # The clip names the bundle DECLARES, recorded where the geometry is
    # loaded. The animation row first read _VM_BUNDLE, which only the
    # muzzle-flash path populates -- so a run that drew the viewmodel
    # without --muzzle-flash reported "no animated asset drew this run"
    # while one plainly had. Same class as the PM_REACH slip in the same
    # block: state read from whichever holder was nearest rather than the
    # one the pass actually fills.
    if all(k in b for k in ("bind_applied_src", "wpn_bone_quat",
                            "wpn_bone_src")):
        _VM_BUNDLE_RAW[0] = b        # the WEAPON bundle, not the arms one
    _fam_col = b["fam"].reshape(-1)
    VM_BUNDLE_INFO["fam_tris"] = {
        int(k): int((_fam_col == k).sum())
        for k in (VM_FAM_WEAPON, VM_FAM_LEGS, VM_FAM_ARMS)}
    if int(b["tri"].max()) >= n:
        return None, (f"{os.path.basename(p)}: index {int(b['tri'].max())} "
                      f"addresses vertex {n - 1} at most")
    was = sorted(set(b["fam"].tolist()))
    fam = torch.full((t,), VM_FAM_ARMS, dtype=torch.long)
    _VM_ARMS_TEX[0] = _slot_pages(f"viewmodel arms {b.get('weapon', '?')}",
                                  b.get("tex_pages") or {},
                                  b.get("tex_provenance") or {})
    note = (f"{os.path.basename(p)}  {n} verts, {t} tris, source "
            f"{b.get('source')}, pose {b.get('pose_clip')}")
    print(f"viewmodel arms: {note}", flush=True)
    print(f"viewmodel arms: fam column OVERRIDDEN {was} -> "
          f"[{VM_FAM_ARMS}] (arms). The bundle ships weapon-family ids "
          f"because it predates this family; drawn as shipped it would "
          f"shade skin and cloth through csgo_weapon's axes.", flush=True)
    # The extractor's own STATED gap, carried to the render rather than
    # left in the .pt: bones it could not pose were substituted, and the
    # vertices riding them are in the bind pose while the rest are posed.
    _sub = b.get("bones_substituted")
    if _sub:
        print(f"viewmodel arms: {_sub} bone(s) SUBSTITUTED, "
              f"{b.get('verts_on_substituted_bones', '?')} of {n} vertices "
              f"ride them -- those are bind-pose inside an otherwise posed "
              f"mesh, STATED by the extractor and not corrected here.",
              flush=True)
    geom = (b["pos"].to(device), b["uv"].to(device), b["nrm"].to(device),
            b["tan"].to(device), b["loc"].to(device),
            b["tri"].to(device).int(), fam.to(device),
            b["is_legs"].to(device).bool())
    _VM_ARMS_CACHE.update(path=p, geom=geom, note=note, tex=_VM_ARMS_TEX[0])
    return geom, note


# Set when a class that SHOULD have drawn drew nothing. Read at exit, so a
# corpus run is never killed mid-flight but also can never be called clean.
VM_CLASS_DEFECT = []
# weapon-family pixel bbox of the last drawn frame, for the flash comparison
VM_WPN_BBOX = {}


def _vm_class_assert(fam_px):
    """THE INVISIBLE GUN: arms on screen, weapon at zero, nobody said a word.

    A viewmodel bundle that ships weapon geometry and then composites zero
    weapon pixels while compositing arms is not a dark frame or a pose
    question -- it is a hand holding nothing, and it shipped in
    combat_closing before a human eye caught it. Every number needed to
    catch it was already being computed and printed per frame; what was
    missing was anything that READ them.

    So this is an assertion over the per-family counts, not new
    instrumentation. The precondition is the bundle's own geometry: only a
    bundle that HAS weapon triangles can be guilty of not drawing them, and
    a legs-only or arms-only bundle is silent here rather than falsely
    accused.

    IT DOES NOT EXIT. The consumer is a tardigrade-3-scale corpus render,
    where dying on frame 40,000 of 89M throws away the run and teaches
    everyone to pass a suppress flag. Instead it prints at full volume on
    every offending frame and records the defect, and the process exits
    non-zero at the end -- so a batch completes, its output is usable for
    everything else, and no script can mistake it for a clean render.
    """
    if not fam_px:
        return
    # NOT DURING WARM-UP. startup.warmup_render draws the same frame to
    # compile kernels, so an unguarded assertion counts every defect twice
    # and the summary reports a frame count that does not exist. The same
    # suppression frame_print() already uses, for the same reason.
    if WARMING_UP[0]:
        return
    # PRECONDITION FROM THE CENSUS, never from the triangle total -- see
    # vm_load_geometry(). Absent a census (synthesised geometry), fall back
    # to the totals, which are correct for that path because it builds one
    # family at a time and says which.
    _ft = VM_BUNDLE_INFO.get("fam_tris")
    if _ft:
        has_wpn = _ft.get(VM_FAM_WEAPON, 0) > 0
        has_arms = (_ft.get(VM_FAM_ARMS, 0) > 0
                    or int(VM_BUNDLE_INFO.get("arms_tris", 0) or 0) > 0)
    else:
        has_wpn = int(VM_BUNDLE_INFO.get("tris", 0) or 0) > 0
        has_arms = int(VM_BUNDLE_INFO.get("arms_tris", 0) or 0) > 0
    b0 = int(VS_FRAME0[0])
    for i, fpx in enumerate(fam_px):
        wpn = fpx.get(VM_FAM_WEAPON, 0.0)
        arms = fpx.get(VM_FAM_ARMS, 0.0)
        legs = fpx.get(VM_FAM_LEGS, 0.0)
        if has_wpn and wpn <= 0 and (arms > 0 or legs > 0):
            msg = (f"VIEWMODEL CLASS DEFECT f{b0 + i:06d}: the bundle ships "
                   f"{VM_BUNDLE_INFO.get('tris', 0)} weapon triangles and "
                   f"composited ZERO weapon pixels, while arms/legs "
                   f"composited {arms:.0f}/{legs:.0f}. That is an empty hand "
                   f"holding a drawn arm, not a lighting or pose question. "
                   f"bundle={VM_BUNDLE_INFO.get('name', '?')} "
                   f"path={VM_BUNDLE_INFO.get('path', '?')} "
                   f"placement={VM_BUNDLE_INFO.get('placement', 'UNRECORDED')}")
            print(msg, flush=True)
            VM_CLASS_DEFECT.append(msg)
        elif has_arms and arms <= 0 and wpn > 0:
            # the mirror case, stated separately so the log names WHICH
            # class vanished rather than reporting a generic mismatch
            msg = (f"VIEWMODEL CLASS DEFECT f{b0 + i:06d}: the bundle ships "
                   f"{VM_BUNDLE_INFO.get('arms_tris', 0)} arms triangles and "
                   f"composited ZERO arms pixels, while the weapon "
                   f"composited {wpn:.0f}. A floating gun is the same defect "
                   f"as an empty hand. "
                   f"bundle={VM_BUNDLE_INFO.get('name', '?')}")
            print(msg, flush=True)
            VM_CLASS_DEFECT.append(msg)


def _q_mul1(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz])


def _q_rot1(q, v):
    u, w = q[:3], q[3]
    return (v * (w * w - (u * u).sum()) + 2.0 * u * (v * u).sum()
            + 2.0 * w * torch.cross(u, v, dim=-1))


def _vm_skin_transforms(b, clip, t_sec, dev):
    """Per-palette-bone (xq, xt) = pose o inverse(bind) at time t.

    TRANSCRIBED from skin_pose_check.articulate()/pose_of(), the composition
    the two-sided bake check already PASSES with (IDLE arm reproduces the
    baked pos to median 6.45e-05 m over 65 weapons) -- not re-derived. The
    only difference is the frame: the check poses at integer frames, this
    samples at t through the same SLERP the rigid path uses.

    A palette bone the clip's rig does not name is an ORPHAN, and it takes
    the transform of its NEAREST POSED ANCESTOR on the model skeleton --
    never identity (#90).

    IDENTITY WAS WRONG, AND WRONG IN A WAY THE COMMENT HID. "A missing bone
    shows as did-not-move" is only true if the two rigs share a frame, and
    they do not: `bind` is the .vmdl_c model frame, `pose` is the .vnmskel
    one, and every POSED bone crosses between them inside
    pose o inverse(bind). A bone left at identity does not stay with the
    gun -- it stays in the frame the gun left, i.e. it detaches by the frame
    offset. Measured on m4a4, whose `sight` is the one orphan in the
    212-bundle census (2222 verts, 5.62% of its weight mass): the idle-arm
    residual is 4.64e-01 m on exactly those vertices and 3.47e-04 m on the
    other 37,336. That 0.46 m was on the books as "the idle clip poses the
    rig away from bind"; the exclusion partition convicted the orphan
    instead. The owner saw it as optics floating at the top of frame.

    THE SUBSTITUTION IS EXACT FOR A RIGID ATTACHMENT, not an approximation
    chosen because it looks better. An orphan `o` rigidly parented to a
    posed ancestor `a` satisfies pose_o = pose_a o (bind_a^-1 o bind_o), so

        x_o = pose_o o bind_o^-1 = pose_a o bind_a^-1 = x_a

    -- the ancestor's own composed transform, reused. Nothing new is fitted
    and no bind proximity is consulted: WHICH ancestor is READ from the
    model skeleton's m_nParent (skin.palette_ancestors, written by
    mesh_skin.add_skin / vm_add_parents.py). This is the same substitution
    extract_arms already applies to twist helpers.

    A bundle predating that field, or an orphan with no posed ancestor at
    all, keeps identity -- there is nothing to read -- and both cases are
    NAMED in the returned report so the caller can print the refusal with
    its vertex count and weight mass rather than rendering a silent detach.

    Returns None when the clip carries no block for the bundle's own
    skeleton -- the caller falls back to the rigid single-bone chain.
    """
    skin = b.get("skin")
    if skin is None:
        return None
    skel = b.get("nm_skeleton")
    blk = next((bl for bl in (clip.get("blocks") or [])
                if bl.get("skeleton") == skel and bl.get("parents")
                is not None), None)
    if blk is None:
        return None
    t_all, q_all, _ = _anim.sample_block(blk, t_sec, clip["duration"],
                                         clip["frames"])
    parents = [int(p) for p in blk["parents"]]
    lt = torch.as_tensor(t_all, dtype=torch.float64)
    lq = torch.as_tensor(q_all, dtype=torch.float64)
    n = len(parents)
    mt = torch.zeros((n, 3), dtype=torch.float64)
    mq = torch.zeros((n, 4), dtype=torch.float64)
    depth = []
    for i in range(n):
        d, j = 0, parents[i]
        while j >= 0:
            d += 1
            j = parents[j]
        depth.append(d)
    for i in sorted(range(n), key=lambda k: depth[k]):
        p = parents[i]
        if p < 0:
            mt[i], mq[i] = lt[i], lq[i]
        else:
            mt[i] = mt[p] + _q_rot1(mq[p], lt[i])
            mq[i] = _q_mul1(mq[p], lq[i])
    ids = list(blk["bone_ids"])
    pal = list(skin["palette"])
    bt = skin["bind_t"].to(torch.float64)
    bq = skin["bind_q"].to(torch.float64)
    N = len(pal)
    xq = torch.zeros((N, 4), dtype=torch.float64)
    xq[:, 3] = 1.0
    xt = torch.zeros((N, 3), dtype=torch.float64)
    unposed = []
    for i, name in enumerate(pal):
        if name not in ids:
            unposed.append(name)
            continue
        j = ids.index(name)
        iq = torch.cat([-bq[i, :3], bq[i, 3:4]])
        it = -_q_rot1(iq, bt[i])
        xt[i] = mt[j] + _q_rot1(mq[j], it)
        xq[i] = _q_mul1(mq[j], iq)

    # ---- ORPHANS: nearest POSED ancestor, never identity (#90) ----------
    # `anc` is the model skeleton's m_nParent chain per palette bone, nearest
    # parent first. It is a READ field; a bundle that predates it gets no
    # substitution and says so, because inventing a parent here (by bind
    # proximity, or by "the heaviest bone") would put a scope on whichever
    # bone happened to be closest and would fail silently when it was the
    # wrong one.
    anc = skin.get("palette_ancestors")
    sub = []          # (orphan, ancestor used, verts, weight mass %)
    unfixed = []      # (orphan, why) -- printed, never swallowed
    if anc is not None and len(anc) != N:
        raise SystemExit(
            f"skin.palette_ancestors has {len(anc)} rows for a {N}-bone "
            f"palette -- the chain and the palette came from different "
            f"reads of the model skeleton.")
    if unposed:
        pidx = {nm: k for k, nm in enumerate(pal)}
        _idx = skin["index"].cpu().numpy()
        _wgt = skin["weight"].cpu().numpy()
        _tot = float(_wgt.sum()) or 1.0
        for name in unposed:
            i = pidx[name]
            nv = int((_idx == i).any(1).sum())
            mass = 100.0 * float(_wgt[_idx == i].sum()) / _tot
            if anc is None:
                unfixed.append((name, nv, mass,
                                "this bundle carries no skin.palette_"
                                "ancestors -- re-run vm_add_parents.py over "
                                "it; the bone is held at IDENTITY and its "
                                "vertices stay in the .vmdl_c frame"))
                continue
            chain = [c for c in anc[i] if c in ids and c in pidx]
            if not chain:
                # The ancestor must be BOTH posed by the clip AND in this
                # mesh's palette -- the palette is what carries the bind we
                # composed against. An ancestor that is posed but unbound
                # has no x to copy, so it is not silently used.
                unfixed.append((name, nv, mass,
                                f"no ancestor of it is both posed by this "
                                f"clip and in the palette (chain "
                                f"{anc[i][:6]}) -- held at IDENTITY"))
                continue
            a = pidx[chain[0]]
            xq[i], xt[i] = xq[a], xt[a]
            sub.append((name, chain[0], nv, mass))
    return xq.to(dev), xt.to(dev), unposed, sub, unfixed


VM_SKIN_NOTE = []


def _vm_bone_gap(pts, own):
    """Distance from the `own` vertices' centroid to the REST's box, metres.

    A weapon is one object: its parts touch. So the gap from any part to the
    box around the other parts is a property of the MESH, and it is the same
    number in bind space and in a correct pose -- a rigid part cannot leave
    the body without the body moving with it. That is what makes it a usable
    invariant: it is invariant by construction rather than by threshold.
    """
    if not bool(own.any()) or not bool((~own).any()):
        return None
    c = pts[own].mean(0)
    lo = pts[~own].min(0).values
    hi = pts[~own].max(0).values
    d = torch.clamp(lo - c, min=0.0) + torch.clamp(c - hi, min=0.0)
    return float(d.norm()) * 0.0254


def _vm_hull_invariant(skin, loc_sk, posed, unposed, sub):
    """WARN when a palette bone's verts leave the posed hull (#90's tripwire).

    THE BOUND IS NOT A CONSTANT AND IS NOT FITTED. For every bone the clip
    poses DIRECTLY, this measures how much its gap-to-the-rest changed
    between bind and pose; the largest such change IS this bundle's own
    articulation scale at this frame, and it is what every other bone is
    compared against. A bone that moved further from the body than anything
    the clip actually articulates is reported with both gaps, the change,
    the bound, and the population the bound was taken over.

    Had this existed, #90 would have printed at the first skinned frame
    instead of being found by eye in a shipped clip: m4a4's `sight` leaves
    the body by ~0.46 m while its posed bones' gaps change by ~0.
    """
    pal = list(skin["palette"])
    idx = skin["index"].cpu().numpy()
    wgt = skin["weight"].cpu().numpy()
    tot = float(wgt.sum()) or 1.0
    orphan = set(unposed)
    subbed = {s[0]: s[1] for s in sub}

    rows = []
    for i, nm in enumerate(pal):
        own = torch.as_tensor((idx == i).any(1), device=loc_sk.device)
        gb = _vm_bone_gap(loc_sk, own)
        gp = _vm_bone_gap(posed, own)
        if gb is None or gp is None:
            continue
        rows.append({"name": nm, "gap_bind": gb, "gap_pose": gp,
                     "delta": abs(gp - gb),
                     "verts": int(own.sum()),
                     "mass": 100.0 * float(wgt[idx == i].sum()) / tot,
                     "direct": nm not in orphan})
    if not rows:
        print("VIEWMODEL HULL INVARIANT: NOT RUN -- this palette has no bone "
              "whose vertices can be separated from the rest (1 bone, or "
              "every vertex on every bone). Nothing is asserted about it.",
              flush=True)
        return
    direct = [r for r in rows if r["direct"]]
    if len(direct) < 2:
        print(f"VIEWMODEL HULL INVARIANT: BOUND UNAVAILABLE -- only "
              f"{len(direct)} palette bone(s) are posed directly, so there "
              f"is no articulation population to derive a bound from. Every "
              f"bone's gap is reported instead and NOTHING is refused:",
              flush=True)
        for r in rows:
            print(f"    {r['name']:<18} gap bind {r['gap_bind']:.4f} m -> "
                  f"pose {r['gap_pose']:.4f} m (change {r['delta']:.4f} m), "
                  f"{r['verts']:,} verts, {r['mass']:.2f}% mass", flush=True)
        return
    bound = max(r["delta"] for r in direct)
    over = [r for r in rows if r["delta"] > bound]
    print(f"VIEWMODEL HULL INVARIANT: bound {bound:.3e} m = the largest "
          f"gap-to-the-rest change among the {len(direct)} DIRECTLY POSED "
          f"palette bone(s) -- this bundle's own articulation scale at this "
          f"frame, derived, not fitted. {len(over)} of {len(rows)} bone(s) "
          f"exceed it. COVERS the WEAPON skin only, at the FIRST skinned "
          f"frame: the arms are not re-posed at all and the playermodels "
          f"run a different path, so a zero here is not a statement about "
          f"either.", flush=True)
    for r in over:
        why = (f"substituted onto `{subbed[r['name']]}`"
               if r["name"] in subbed else
               ("ORPHAN held at identity" if not r["direct"] else
                "posed directly by the rig"))
        print(f"  ⚠️  VIEWMODEL HULL: `{r['name']}` ({why}) leaves the rest "
              f"of the mesh -- gap {r['gap_bind']:.4f} m at bind, "
              f"{r['gap_pose']:.4f} m posed, change {r['delta']:.3e} m "
              f"against a bound of {bound:.3e} m "
              f"({r['delta'] / max(bound, 1e-12):.3g}x). "
              f"{r['verts']:,} vertices, {r['mass']:.2f}% of the weight "
              f"mass, will draw detached.", flush=True)


def _vm_repose(pos_v, loc_v, tri, fam_of_tri, nframes=1):
    """PER-FRAME re-pose: (N,3) in, (B,N,3) out -- one pose per batch row.

    The batch used to carry ONE pose, taken from its first row's choice, and
    said so: "<= nframes-1 ticks of smear". At batch 2 that halved the
    visible motion of every event landing second in its batch, which is the
    same class of loss as #81's stride-2 defect one layer down -- the ticks
    were all read, and then averaged away at the geometry.

    Each row is now posed at its OWN (clip, t) from VM_ANIM_SEQ and the
    results are stacked. The output is ALWAYS (B,N,3), including on the
    paths that do not re-pose at all, so nothing downstream has to ask which
    shape it got -- an expand is free and a shape that varies with a flag is
    how the einsum below would silently mean something else.

    IDENTICAL ROWS ARE COMPUTED ONCE. When every row resolves to the same
    (clip, t) -- the common case, since a 64-tick clip does not change event
    every frame -- this poses once and expands, so the per-frame path is
    bit-identical to the old one there rather than merely close.
    """
    seq = VM_ANIM_SEQ[0]
    B = max(1, int(nframes))
    VM_ANIM_SAME[0] = False       # first row of this batch sets, rest max
    if not seq or args.vm_anim == "off":
        return _vm_repose_one(pos_v, loc_v, tri, fam_of_tri,
                              VM_ANIM_CLIP[0], VM_ANIM_T[0]
                              )[None].expand(B, -1, -1)
    seq = list(seq)[:B] + [None] * max(0, B - len(seq))
    uniq = {}
    for s in seq:
        key = (id(s[0]), float(s[1])) if s else None
        uniq.setdefault(key, s)
    posed = {}
    for key, s in uniq.items():
        c, t = (s if s else (VM_ANIM_CLIP[0], VM_ANIM_T[0]))
        posed[key] = _vm_repose_one(pos_v, loc_v, tri, fam_of_tri, c, t)
    VM_ANIM_POSES[0] = len(uniq)
    rowkey = [((id(s[0]), float(s[1])) if s else None) for s in seq]
    return torch.stack([posed[k] for k in rowkey], dim=0)


def _vm_repose_one(pos_v, loc_v, tri, fam_of_tri, _clip=None, _t=None):
    """Re-pose the WEAPON at the scripted clip time, on the rigid `wpn` bone.

    THE CHAIN IS THE BUNDLE'S OWN, with one term swapped. extract_viewmodel
    states it and this reproduces it:

        pos = to_view(qrot(wpn_bone_quat, loc - bind_applied_src)
                      + wpn_bone_src) * 0.0254 + offset_m + dxy_*

    Recomputing `pos` from `loc` that way agrees with the baked array to
    max 3.1e-08 m over m4a4's 39,558 vertices, which is what licenses
    swapping (wpn_bone_quat, wpn_bone_src) -- the HOLD transform -- for the
    same bone sampled from the clip at t. No inversion is needed and none
    is done: `loc` is in the bundle, so the bone-space position is READ
    rather than solved for out of the posed vertices.

    RIGID, AND SAID SO. This moves the whole weapon on one bone. It is real
    recoil -- m4a4's `wpn` translates 2.72 source units across its shoot
    clip -- and it is NOT bolt cycling. The bolt is a separate bone in the
    weapon-side block, and moving its PIXELS needs per-vertex bindings that
    no mesh bundle carries yet (m4a4.pt has only wpn_bone_src/quat; ct.pt
    has none at all). Applying the weapon-side block without those bindings
    would move nothing, so this does not pretend to.

    The arms are untouched for the same reason, and because they are a
    different rig: they need the 56-track viewmodel block skinned, not one
    bone applied rigidly.
    """
    if args.vm_anim == "off":
        return pos_v
    cs = VM_CLIPS[0]
    # THE CLIP AND t ARE ARGUMENTS NOW, one per batch row. They defaulted to
    # the globals while the batch had a single pose; taking them as
    # parameters is what lets the caller pose row k at row k's choice.
    clip = _clip if _clip is not None else VM_ANIM_CLIP[0]
    t_sec = VM_ANIM_T[0] if _t is None else float(_t)
    b = _VM_BUNDLE_RAW[0]
    if cs is None or clip is None or b is None or loc_v is None:
        # NAME THE MISSING INPUT. This was a bare `return pos_v`, so a run
        # printed event/clip/t every frame, rendered a bind pose, and said
        # nothing about why -- the exact silence the ANIM columns were added
        # to end, reintroduced one function lower. Which of the four is
        # absent decides the owner, so all four are reported.
        if not VM_REPOSE_NOTE:
            VM_REPOSE_NOTE.append(1)
            _absent = [n for n, v in (("clip sidecar", cs), ("chosen clip",
                       clip), ("weapon bundle", b), ("loc (bone-space "
                       "positions)", loc_v)) if v is None]
            print(f"⚠️  VIEWMODEL ANIM: re-pose SKIPPED -- missing "
                  f"{_absent}. The event/t lines above are the SCRIPT's "
                  f"choice, not motion; the weapon is at its bind pose.",
                  flush=True)
        return pos_v
    # THE CHAIN CONSTANTS OR NOTHING. _VM_BUNDLE_RAW is written by
    # vm_load_geometry, which vm_load_arms ALSO goes through -- so the last
    # writer can be the arms bundle, which carries no bind_applied_src and
    # no wpn bone. The first version indexed it directly and died with
    # KeyError mid-render. A bundle without the chain cannot be re-posed by
    # this chain; that is a precondition, so it is checked and named rather
    # than assumed from whichever load happened last.
    need = ("bind_applied_src", "wpn_bone_quat", "wpn_bone_src")
    miss = [k for k in need if k not in b]
    if miss:
        if not VM_REPOSE_NOTE:
            VM_REPOSE_NOTE.append(1)
            frame_print(f"VIEWMODEL ANIM: the bundle in hand is missing "
                        f"{miss} -- it is not the weapon bundle whose chain "
                        f"this re-pose inverts (vm_load_arms shares the "
                        f"loader). The weapon holds its bind pose and "
                        f"nothing is guessed.", flush=True)
        return pos_v
    blk = (clip.get("blocks") or [None])[0]
    if blk is None:
        return pos_v
    trk = _anim.bone_track(blk, "wpn")
    if trk is None:
        frame_print("VIEWMODEL ANIM: this clip's viewmodel block has no "
                    "`wpn` track, so the rigid chain has nothing to drive; "
                    "the weapon holds its bind pose.", flush=True)
        return pos_v
    t_all, q_all, _ = _anim.sample_block(blk, t_sec, clip["duration"],
                                         clip["frames"])
    p_t = t_all[trk].to(torch.float64)
    q_t = q_all[trk].to(torch.float64)
    dev = pos_v.device
    bind = torch.tensor(b["bind_applied_src"], dtype=torch.float64,
                        device=dev)
    off = torch.zeros(3, dtype=torch.float64, device=dev)
    for k in ("offset_m", "dxy_muzzle", "dxy_landmark"):
        if k in b:
            off = off + torch.tensor(b[k], dtype=torch.float64, device=dev)
    skx = _vm_skin_transforms(b, clip, t_sec, dev)
    if skx is not None:
        # ARTICULATED: the weapon-side block poses the weapon's OWN bones
        # (bolt, trigger, weapon_offset ...) and the sampled `wpn` places
        # the whole gun -- composition, not duplication: the weapon rig's
        # root rides `wpn`, which is why each .clips.pt carries BOTH
        # blocks. bind_applied_src is NOT subtracted here: this is the
        # FOURTH consumer and it follows the ARTICULATED rule -- pose o
        # inverse(bind) already removed the bind, and subtracting it again
        # applies it twice (the 1780x idle-residual defect, caught by the
        # bake check's own first run).
        xq, xt, unposed, _sub, _unfixed = skx
        skin = b["skin"]
        idx = skin["index"].to(dev).long()
        wgt = skin["weight"].to(dev).to(torch.float64)
        # THE SKIN COVERS THE WEAPON'S OWN VERTICES ONLY. loc_v/pos_v are
        # the MERGED weapon+arms array (arms appended at the weapon's
        # vertex count -- "merged at vertex offset N" in the load log), so
        # the LBS runs on the leading slice the bindings actually index.
        # The tail keeps the rigid-fallback transform and is then masked
        # out by the weapon-family selection below, exactly as the rigid
        # path always treated it. Indexing the full array crashed the
        # first run loudly (21414 vs 27146) -- kept loud: a skin longer
        # than the mesh is refused, never truncated.
        n_sk = idx.shape[0]
        if n_sk > loc_v.shape[0]:
            raise SystemExit(
                f"viewmodel skin indexes {n_sk} vertices but the merged "
                f"mesh has {loc_v.shape[0]} -- bindings and positions came "
                f"from different reads.")
        Lr = loc_v[:n_sk].to(torch.float64)
        vp_w = torch.zeros_like(Lr)
        for _k in range(idx.shape[1]):
            bsel = idx[:, _k]
            qk = xq[bsel]
            t2 = 2.0 * torch.cross(qk[:, :3], Lr, dim=-1)
            rot = Lr + qk[:, 3:4] * t2 + torch.cross(qk[:, :3], t2, dim=-1)
            vp_w = vp_w + wgt[:, _k:_k + 1] * (rot + xt[bsel])
        vp = loc_v.to(torch.float64) - bind
        vp[:n_sk] = vp_w
        if not VM_SKIN_NOTE:
            VM_SKIN_NOTE.append(1)
            print(f"VIEWMODEL ANIM: SKINNED re-pose LIVE -- "
                  f"{len(skin['palette'])} palette bones, "
                  f"{idx.shape[1]} influence(s)/vertex, "
                  f"{len(unposed)} palette bone(s) the rig does not name, "
                  f"bind_applied_src NOT subtracted (articulated rule -- "
                  f"inverse(bind) already removed it). Composition "
                  f"transcribed from skin_pose_check.articulate, which the "
                  f"bake check passes offline.", flush=True)
            # EVERY ORPHAN IS NAMED WITH ITS SIZE (#90). The substitution is
            # a real change to where pixels land, so it is reported per bone
            # with the vertex count and weight mass that price it -- not
            # summarised as "n bones substituted".
            for _nm, _a, _nv, _ms in _sub:
                print(f"  ORPHAN BONE SUBSTITUTED: `{_nm}` is not named by "
                      f"this clip's rig; it takes the transform of its "
                      f"nearest POSED ancestor `{_a}` (model-skeleton "
                      f"m_nParent chain, READ). {_nv:,} vertices, "
                      f"{_ms:.2f}% of the weight mass. Identity would have "
                      f"left them in the .vmdl_c frame -- detached from the "
                      f"gun by the frame offset, which is the #90 defect.",
                      flush=True)
            for _nm, _nv, _ms, _why in _unfixed:
                print(f"  ⚠️  ORPHAN BONE NOT SUBSTITUTED: `{_nm}` "
                      f"({_nv:,} vertices, {_ms:.2f}% of the weight mass) -- "
                      f"{_why}.", flush=True)
            _vm_hull_invariant(skin, Lr, vp_w, unposed, _sub)
    else:
        # RIGID fallback: one bone, the whole gun, bind SUBTRACTED (the
        # mesh rule -- raw `loc` still carries the .vmdl_c bind).
        vp = loc_v.to(torch.float64) - bind
    qv = q_t.to(dev)[None, :3]
    qw = q_t.to(dev)[3]
    tt = 2.0 * torch.cross(qv.expand_as(vp), vp, dim=-1)
    sp = vp + qw * tt + torch.cross(qv.expand_as(vp), tt, dim=-1) + p_t.to(dev)
    new = torch.stack([-sp[:, 1], sp[:, 2], -sp[:, 0]], dim=-1) * 0.0254 + off
    # WEAPON FAMILY ONLY. fam is per-TRIANGLE, so its corners select the
    # vertices -- the same 45422-vs-39558 trap the barrel-axis line hit.
    sel = torch.zeros(pos_v.shape[0], dtype=torch.bool, device=dev)
    wt = tri[fam_of_tri.reshape(-1) == VM_FAM_WEAPON].reshape(-1).long()
    if wt.numel():
        sel[wt] = True
    out = torch.where(sel[:, None], new.to(pos_v.dtype), pos_v)
    # MAX OVER THE BATCH's rows, not the last one written. Each row calls
    # this once, so a plain assignment would report whichever row happened
    # to run last -- and the line that quotes it sits beside the event name,
    # where a reader takes it for the batch's motion.
    _mv = float((out - pos_v).norm(dim=-1).max()) if bool(sel.any()) else 0.0
    VM_ANIM_MOVED[0] = max(VM_ANIM_MOVED[0], _mv) if VM_ANIM_SAME[0] else _mv
    VM_ANIM_SAME[0] = True
    # ===================================================================
    # THE ANIMATION TIER, stated once, per asset, READ not asserted (#87)
    # ===================================================================
    # The owner's "the animations look wrong" covers two different states
    # and a single "the viewmodel is animated" line cannot separate them.
    # `sel` above is WEAPON TRIANGLES ONLY, so whatever tier the weapon
    # runs at, the ARMS are not re-posed at all -- they render at the bake
    # pose every frame. That is a real limitation of this artifact and it
    # belongs in the log next to the clip name, not in a commit message
    # nobody reads while watching the clip.
    if not VM_TIER_NOTE:
        VM_TIER_NOTE.append(1)
        _n_arm = int((fam_of_tri.reshape(-1) == VM_FAM_ARMS).sum())
        print(f"VIEWMODEL TIER: weapon = "
              f"{'SKINNED (palette, per-vertex weights)' if VM_SKIN_NOTE else 'RIGID (one `wpn` bone, whole gun)'}"
              f", {int(sel.sum()):,} vertices re-posed. "
              f"arms = NOT RE-POSED (bake pose held every frame), "
              f"{_n_arm:,} triangles -- the re-pose selects weapon-family "
              f"triangles only, so arms skinning is ABSENT, not "
              f"approximate. Any judgement of 'the animation looks wrong' "
              f"has to be split across those two tiers before it means "
              f"anything.", flush=True)
    # A SAMPLED DELTA THAT MOVES NO VERTEX IS A DEFECT, and it is silent by
    # construction: the log still says event=shoot with a t that advances,
    # the clip is real, the sample is right -- and the image never changes.
    # That is the shape this lane has been caught by repeatedly, so it is
    # asserted here rather than left to an A/B someone may not run.
    _sd = float((p_t.to(dev) - torch.tensor(
        b["wpn_bone_src"], dtype=torch.float64, device=dev)).norm())
    if _sd > 1e-4 and VM_ANIM_MOVED[0] <= 1e-9:
        if not VM_REPOSE_NOTE:
            VM_REPOSE_NOTE.append(1)
            print(f"⚠️  VIEWMODEL ANIM: the sampled `wpn` bone is "
                  f"{_sd * 25.4:.1f} mm from the hold transform, and the "
                  f"re-pose moved NO vertex ({int(sel.sum())} of "
                  f"{pos_v.shape[0]} selected, "
                  f"{int((fam_of_tri.reshape(-1) == VM_FAM_WEAPON).sum())} "
                  f"of {fam_of_tri.reshape(-1).numel()} triangles are "
                  f"weapon-family). The clip, the sample and the chain are "
                  f"all fine; the geometry is not consuming them. Playback "
                  f"is NOT live -- do not read the event/t lines as motion.",
                  flush=True)
    return out


def _vm_anim_report(nframes=1):
    """Choose each batch row's clip from its demo row, and SAY SO.

    The choice is made and printed even though the geometry delta is not
    applied yet, because the two are separate claims and conflating them is
    how a term gets called landed. What this proves is the SCRIPTING: which
    event the columns imply, which authored clip that names, and what t is.
    What it does not yet prove is motion -- ANIM_DRIVEN carries the choice,
    and the ANIMATION STATE matrix reports driven separately from decoded
    for exactly this reason.
    """
    cs = VM_CLIPS[0]
    if cs is None:
        return
    # EVERY ROW OF THE BATCH, not the first one. The first version read
    # rows[VS_FRAME0[0]] alone, so an event landing on any other row of
    # the batch was never seen by the scripter: at batch 2 the ak window's
    # four fire ticks (frames 6/13/19/25) lost the three that fell
    # second-in-batch, verified by A/B -- 2 of 32 frames moved. The
    # scripter consumes rows SEQUENTIALLY (its per-tick state -- event
    # latch, distance -- is only right if no row is skipped), each choice
    # prints with its own frame index, and the batch's single pose takes
    # the FIRST row's choice: never showing a later tick's state early.
    try:
        _b0 = int(VS_FRAME0[0])
    except Exception:                                         # noqa: BLE001
        return
    _first = None
    _diverge = 0
    _seq = [None] * max(1, int(nframes))
    for _k in range(max(1, int(nframes))):
        _i = _b0 + _k
        if not (0 <= _i < len(rows)):
            continue
        row = rows[_i]
        ev, clip, t, why = VM_SCRIPT[0].choose("ego", row, VM_PREV_ROW[0], cs)
        VM_PREV_ROW[0] = row
        if ev is None:
            continue
        ANIM_DRIVEN["viewmodel"] = ev
        _seq[_k] = (clip, t)
        if _first is None:
            _first = (ev, clip, t)
        elif ev != _first[0] or clip is not _first[1]:
            _diverge += 1
        _blocks = [os.path.basename(b.get("skeleton", "?"))
                   for b in (clip.get("blocks") or [])]
        frame_print(
            f"VIEWMODEL ANIM f{_i:06d} event={ev} "
            f"clip={os.path.basename(clip['clip'])} "
            f"t={t:.3f}s of {clip['duration']:.3f}s ({clip['frames']} "
            f"frames) blocks={'+'.join(_blocks)} | {why[0]}"
            + (f" | {why[-1]}" if len(why) > 1 else "")
            # THE TIER, not a hardcoded word. This read "rigid wpn
            # re-pose" unconditionally, so once skinning went live the
            # same log carried "weapon = SKINNED" and "rigid wpn re-pose"
            # two lines apart -- and the second is the one sitting beside
            # the number, which is the one a reader believes. A label that
            # cannot follow the path it describes is how a tier gets
            # misread as a regression.
            + ((f" | {'SKINNED' if VM_SKIN_NOTE else 'rigid'} wpn re-pose "
                f"moved the weapon max {VM_ANIM_MOVED[0] * 1000:.1f} mm")
               if VM_ANIM_MOVED[0] else ""),
            flush=True)
    if _first is not None:
        VM_ANIM_CLIP[0], VM_ANIM_T[0] = _first[1], _first[2]
    # PER-FRAME, NOT PER-BATCH. `_seq` carries one (clip, t) per batch row,
    # so the re-pose below poses each frame at its OWN choice. This retires
    # the one-pose-per-batch smear -- a stated approximation worth up to
    # (nframes - 1) ticks, which at batch 2 silently halved the visible
    # motion of every event that landed second in its batch.
    #
    # A ROW THAT CHOSE NOTHING CARRIES THE PREVIOUS ROW FORWARD, and that is
    # a read of what `choose` returning None means: no NEW event this tick,
    # so the clip already playing continues. It is not "no animation" -- the
    # rows before the FIRST choice are that, and they get None and hold the
    # bind pose exactly as the whole batch used to when no row chose.
    _carried = sum(1 for k in range(len(_seq))
                   if _seq[k] is None and any(_seq[:k]))
    for _k in range(len(_seq)):
        if _seq[_k] is None and _k and _seq[_k - 1] is not None:
            _seq[_k] = _seq[_k - 1]
    VM_ANIM_SEQ[0] = list(_seq)
    if _diverge:
        frame_print(
            f"VIEWMODEL ANIM: {_diverge} of {int(nframes)} frame(s) in this "
            f"batch chose a different event or clip from f{_b0:06d}'s, and "
            f"each is now posed at ITS OWN choice (per-frame re-pose). The "
            f"one-pose-per-batch smear is retired."
            + (f" {_carried} row(s) chose no new event and carry the "
               f"previous row's clip forward." if _carried else ""))


def _vm_frame_report(nframes, covered=None, live=None, absent=None,
                     fam_px=None):
    """One line per frame: which bundle drew, or ABSENT and why.

    The convention is pair_compare's, which already prints the SELECTION
    side of this sentence per frame -- `viewmodel: id 19 mac_10: resolves;
    no bundle -- viewmodel OFF` (ceeb2152). This is the RENDER side, and
    the two are not the same fact: a selection can succeed, the bundle can
    load, the mesh can rasterise, and the pass can still composite nothing
    -- which is what it did on every paired frame until vm_gt_shade above.
    viewmodel_pass has four returns and three of them returned `world_rgb`
    unchanged without distinguishing themselves.
    """
    b0 = int(VS_FRAME0[0])
    for i in range(nframes):
        f = b0 + i
        if absent is not None:
            frame_print(f"VIEWMODEL f{f:06d} ABSENT: {absent}")
            continue
        cov = float(covered[i].sum()) if covered is not None else 0.0
        lit = float(live[i].sum()) if live is not None else 0.0
        # THE ARMS ARE THEIR OWN STATE ON THIS LINE. "the viewmodel drew"
        # is now two assets: character-2 measured the corpus band as
        # mostly arms (73.1% inside band) against a weapon that is not
        # (33.5%), so a line that reports one number for both cannot say
        # which of them reached the band.
        fpx = fam_px[i] if fam_px else None
        if VM_BUNDLE_INFO.get("arms_tris"):
            arms = (f"arms {VM_BUNDLE_INFO['arms_verts']}v/"
                    f"{VM_BUNDLE_INFO['arms_tris']}t "
                    f"{fpx.get(VM_FAM_ARMS, 0):.0f} px"
                    if fpx else
                    f"arms {VM_BUNDLE_INFO['arms_verts']}v/"
                    f"{VM_BUNDLE_INFO['arms_tris']}t")
        else:
            arms = f"arms ABSENT ({VM_BUNDLE_INFO.get('arms', 'not looked for')})"
        wpn_px = (f" weapon {fpx.get(VM_FAM_WEAPON, 0):.0f} px"
                  if fpx else "")
        frame_print(f"VIEWMODEL f{f:06d} DREW "
              f"{VM_BUNDLE_INFO.get('name', '?')} "
              f"bundle {VM_BUNDLE_INFO.get('path', '?')} "
              f"verts {VM_BUNDLE_INFO.get('verts', 0)} "
              f"tris {VM_BUNDLE_INFO.get('tris', 0)} "
              f"covered {cov:.0f} shaded {lit:.0f} raster px "
              f"|{wpn_px} {arms} "
              f"| placement {VM_BUNDLE_INFO.get('placement', 'UNRECORDED')} "
              f"| pose {VM_BUNDLE_INFO.get('pose', 'UNRECORDED')} "
              f"| dxy {VM_BUNDLE_INFO.get('dxy', 'UNRECORDED')} "
              f"{VM_BUNDLE_INFO.get('dxy_vec', '')}", flush=True)


def vm_load_geometry(path):
    """A REAL weapon mesh, in vm_synth_geometry()'s exact return contract.

    The only difference from the synthesised path is where the vertices
    came from -- same dispatch, same side-table columns, same shading
    functions. Built by extract_viewmodel.py from the shipped .vmdl_c.

    Every one of the eight keys is REQUIRED and checked. A bundle missing
    `loc` would still render, because `loc` only feeds csgo_legs_prepass's
    cone test and csgo_weapon's SFX phase -- and both would silently
    degenerate rather than fail, which is the error this refuses to allow
    a partial file to cause.
    """
    b = torch.load(path, map_location="cpu", weights_only=False)
    need = ("pos", "uv", "nrm", "tan", "loc", "tri", "fam", "is_legs")
    miss = [k for k in need if k not in b]
    if miss:
        raise SystemExit(
            f"--viewmodel-model {path} is missing {miss}. Rebuild it with "
            f"extract_viewmodel.py; a partial bundle degenerates the "
            f"legs cone test and the SFX phase instead of failing.")
    n, t = b["pos"].shape[0], b["tri"].shape[0]
    # THE FAMILY CENSUS, taken from the fam COLUMN rather than from the
    # triangle total. _vm_class_assert's precondition is "this bundle ships
    # weapon geometry", and its first version answered that with
    # VM_BUNDLE_INFO['tris'] -- which is the whole bundle's triangle count.
    # Pass an arms-only bundle as --viewmodel-model and that reads as
    # "ships a weapon", every triangle is family ARMS, weapon pixels are
    # legitimately zero, and the check fires on a bundle doing exactly what
    # it says. Counting the column cannot make that mistake, and it is the
    # same read-the-field rule the rest of this file runs on.
    _fam_col = b["fam"].reshape(-1)
    VM_BUNDLE_INFO["fam_tris"] = {
        int(k): int((_fam_col == k).sum())
        for k in (VM_FAM_WEAPON, VM_FAM_LEGS, VM_FAM_ARMS)}
    if int(b["tri"].max()) >= n:
        raise SystemExit(
            f"--viewmodel-model {path}: index {int(b['tri'].max())} "
            f"addresses vertex {n - 1} at most.")
    # THE RIGID RE-POSE'S INPUT, WRITTEN BY THE WEAPON LOADER. The guard
    # that fills _VM_BUNDLE_RAW lived only in vm_load_arms, whose own
    # comment says "the WEAPON bundle, not the arms one" -- and an arms
    # bundle can never satisfy it, because extract_arms DELIBERATELY does
    # not carry bind_applied_src (it is the weapon's rigid bind and would
    # be wrong for skinned glove geometry). So the condition was correct,
    # sat in the one function whose bundles always fail it, and the re-pose
    # was skipped on every run while printing that it had been. Setting it
    # here does not weaken the guard: the same three chain constants are
    # still required, and the arms site is left in place so whichever
    # loader runs last, the holder still contains a bundle this chain can
    # actually re-pose.
    #
    # bind_applied_src, FOURTH CONSUMER, and it follows the MESH rule:
    # SUBTRACTED. The rigid re-pose takes raw `loc`, which still carries
    # the .vmdl_c bind, and swaps the hold (wpn_bone_quat, wpn_bone_src)
    # for the same bone sampled at t -- the identical chain the bake used,
    # one transform different. The muzzle does not subtract it (its source
    # is already in the .vnmskel frame) and neither does articulated LBS
    # (inverse(bind) has already removed it); this one does.
    if all(k in b for k in ("bind_applied_src", "wpn_bone_quat",
                            "wpn_bone_src")):
        _VM_BUNDLE_RAW[0] = b
        print(f"viewmodel: chain constants present in "
              f"{os.path.basename(path)} -- the rigid `wpn` re-pose has "
              f"its bundle (bind_applied_src SUBTRACTED, the mesh rule).",
              flush=True)
    else:
        _miss_chain = [k for k in ("bind_applied_src", "wpn_bone_quat",
                                   "wpn_bone_src") if k not in b]
        print(f"⚠️  viewmodel: {os.path.basename(path)} lacks {_miss_chain}, "
              f"so the rigid `wpn` re-pose has no chain to invert and "
              f"--vm-anim will hold the bind pose. Rebuild with "
              f"extract_viewmodel.py.", flush=True)
    global _VM_TEX, _VM_TEX_OVERRIDE, _VM_TEX_PROV
    _new = b.get("tex_pages") or None
    if _new is not _VM_TEX_OVERRIDE:
        # the synthesised set is cached; a new override must rebuild it,
        # or the first frame would sample pages the provenance line denies
        _VM_TEX = None
    _VM_TEX_OVERRIDE, _VM_TEX_PROV = _new, b.get("tex_provenance")
    # THE PLACEMENT, recorded for the per-frame line rather than re-derived.
    # It is not applied here: extract_viewmodel.py bakes it into `pos` --
    # the mesh minus the .vmdl_c bind translation, through the posed `wpn`
    # rotation, plus the viewmodel skeleton's `wpn` bone, plus the gtvm
    # viewmodel_offset cvars, plus DXY (e3f94b39/e674f178). So `pos` is
    # already view-space metres and the renderer must NOT re-apply any of
    # it. What the renderer owes is saying WHICH placement it drew with,
    # because a bundle carrying a STATED-zero DXY and a bundle carrying the
    # AK's fitted-and-CONTAMINATED pair render identically silently.
    VM_BUNDLE_INFO.clear()
    VM_BUNDLE_INFO.update(
        path=path, name=b.get("weapon") or os.path.basename(path),
        verts=n, tris=t,
        # THE CLIP NAMES GO IN THE UPDATE, NOT BEFORE IT. Written earlier in
        # this function they were silently discarded by the clear() two
        # lines up, and the animation row reported "clips declared: none"
        # for a bundle carrying five. A dict that is cleared after you write
        # to it is indistinguishable from a field that was never read --
        # third state-plumbing slip in this block, all the same shape:
        # reading or writing whichever holder was nearest instead of the one
        # that survives.
        sequences=list(b.get("sequences") or []),
        clips=bool(vm_clips_for(path)),
        placement=b.get("placement_basis", "UNRECORDED -- bundle predates "
                                           "placement_basis"),
        pose=b.get("wpn_pose_basis", "UNRECORDED"),
        dxy=b.get("dxy_basis", "UNRECORDED"),
        dxy_vec=(f"muzzle {b.get('dxy_muzzle')} landmark "
                 f"{b.get('dxy_landmark')}")
        if b.get("dxy_muzzle") is not None else "")
    # --- THE ARMS, merged as their own family --------------------------
    # One rasterisation, three families, split per pixel by the family id --
    # the same mechanism the weapon and legs already use. The arms are in
    # the SAME view space with their placement already baked by the
    # extractor (`axis_map`: view = (-Ym, +Zm, -Xm) * 0.0254 + cvar offset,
    # after skinning to the idle clip), so nothing is re-applied to them
    # here, exactly as nothing is re-applied to the weapon.
    _arms, _anote = vm_load_arms(path)
    VM_BUNDLE_INFO["arms"] = _anote
    if _arms is not None:
        _ap, _au, _an, _at, _al, _atri, _afam, _aleg = _arms
        VM_BUNDLE_INFO["arms_verts"] = int(_ap.shape[0])
        VM_BUNDLE_INFO["arms_tris"] = int(_atri.shape[0])
        b = dict(b)
        b["pos"] = torch.cat([b["pos"].to(device), _ap])
        b["uv"] = torch.cat([b["uv"].to(device), _au])
        b["nrm"] = torch.cat([b["nrm"].to(device), _an])
        b["tan"] = torch.cat([b["tan"].to(device), _at])
        b["loc"] = torch.cat([b["loc"].to(device), _al])
        b["is_legs"] = torch.cat([b["is_legs"].to(device).bool(), _aleg])
        b["tri"] = torch.cat([b["tri"].to(device).int(),
                              _atri + n])          # offset by weapon verts
        b["fam"] = torch.cat([b["fam"].to(device).long(), _afam])
        print(f"viewmodel arms: merged at vertex offset {n}; the pass now "
              f"rasterises {b['pos'].shape[0]} verts / {b['tri'].shape[0]} "
              f"tris across families "
              f"{[VM_FAM_NAME.get(int(v), int(v)) for v in sorted(set(b['fam'].tolist()))]}",
              flush=True)
    else:
        print(f"viewmodel arms: ABSENT -- {_anote}. The weapon draws "
              f"without hands, which is a missing asset and not an empty "
              f"grip.", flush=True)
    print(f"viewmodel model: {path}  {n} verts, {t} tris"
          f"{'  PROVISIONAL PLACEMENT: ' + str(b['provisional']) if b.get('provisional') else ''}",
          flush=True)
    return (b["pos"].to(device), b["uv"].to(device), b["nrm"].to(device),
            b["tan"].to(device), b["loc"].to(device),
            b["tri"].to(device).int(), b["fam"].to(device).long(),
            b["is_legs"].to(device).bool())
