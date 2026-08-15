if FAMILY:
    # --char-side names a side table built by the CURRENT fast_pack_fam.py
    # under a DISTINCT filename, so a character pack is never written over
    # assets/fam_side.pt. It is the same schema (one merged rebuild covers
    # every family), so it simply replaces --fam-side when given.
    if args.char_side:
        if args.fam_side and args.fam_side != args.char_side:
            print(f"--char-side {args.char_side} replaces --fam-side "
                  f"{args.fam_side} for this run (same schema, superset "
                  f"columns)", flush=True)
        args.fam_side = args.char_side
    if not args.fam_side:
        raise SystemExit("the shader-family features need --fam-side "
                         "(build it with fast_pack_fam.py)"
                         + (" or --char-side" if NONWORLD else ""))
    _fs = _fam_side_load()
    FAM_NAMES = list(_fs["fam_names"])
    MAT_FAM = _fs["fam_id"].to(device).long()
    MAT_EXT2 = _fs["ext2"].to(device)
    EXT2 = {k: i for i, k in enumerate(_fs["ext2_keys"])}
    # --- DIAGNOSTIC: zero one named ext2 column ------------------------
    # A bisect over these columns is only interpretable if "off" has one
    # meaning. Unbinding the whole ext2 block and zeroing a single column
    # are different experiments that both get called "the column off",
    # and they can differ by orders of magnitude in pixels touched. This
    # makes the second one explicit, named, and reproducible from argv.
    # Diagnostic only: it never changes a default.
    if args.ext2_zero:
        for _c in args.ext2_zero.split(","):
            _c = _c.strip()
            if _c not in EXT2:
                raise SystemExit(
                    f"--ext2-zero {_c}: no such ext2 column. A silent "
                    f"no-op here would report the column as having no "
                    f"effect, which is the opposite of the truth.")
            _before = MAT_EXT2[:, EXT2[_c]].clone()
            _n = int((_before != 0).sum())
            if _n == 0:
                raise SystemExit(
                    f"--ext2-zero {_c}: the column is ALREADY zero on all "
                    f"{MAT_EXT2.shape[0]} materials, so zeroing it cannot "
                    f"change a pixel. Refusing, because an A/B whose arms "
                    f"are identical reports 'no effect' either way.")
            MAT_EXT2[:, EXT2[_c]] = 0
            print(f"ext2-zero: {_c} forced to 0 on {_n} of "
                  f"{MAT_EXT2.shape[0]} materials (was "
                  f"{sorted(set(_before.tolist()))})", flush=True)
    MAT_GTRANS = _fs["glass_trans"].to(device)
    MAT_CTINT2 = _fs["ctint_raw"].to(device)
    MAT_L1EM = _fs["l1emtint"].to(device)
    T_GDUST = _fs["t_gdust"].to(device).long()
    T_GTINT = _fs["t_gtint"].to(device).long()
    T_L0MASK = _fs["t_l0mask"].to(device).long()
    T_L1COL = _fs["t_l1col"].to(device).long()
    T_L2COL = _fs["t_l2col"].to(device).long()
    # --- csgo_water_fancy texture slots -------------------------------
    # A side table built before this family landed carries none of them.
    # Defaulting to row 0 would sample SOME OTHER MATERIAL'S image and
    # look like a working water shader, so the table is required and its
    # absence is fatal, naming the rebuild command.
    if args.water_fancy or args.water_selftest:
        _wneed = ["t_wwaves", "t_wfoam", "t_wdebris", "t_wdebrisn",
                  "t_weffect"]
        _wmiss = [k for k in _wneed if k not in _fs]
        if _wmiss:
            raise SystemExit(
                f"--fam-side {args.fam_side} predates csgo_water_fancy: "
                f"it has no {', '.join(_wmiss)} table. Rebuild it with\n"
                f"  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
                f"--out {args.fam_side}")
    T_WWAVES = (_fs["t_wwaves"].to(device).long()
                if "t_wwaves" in _fs else None)
    T_WFOAM = (_fs["t_wfoam"].to(device).long()
               if "t_wfoam" in _fs else None)
    T_WDEBRIS = (_fs["t_wdebris"].to(device).long()
                 if "t_wdebris" in _fs else None)
    T_WDEBRISN = (_fs["t_wdebrisn"].to(device).long()
                  if "t_wdebrisn" in _fs else None)
    T_WEFFECT = (_fs["t_weffect"].to(device).long()
                 if "t_weffect" in _fs else None)
    FAM_OVERLAY = FAM_NAMES.index("csgo_static_overlay.vfx")
    FAM_WATER = FAM_NAMES.index("csgo_water_fancy.vfx")
    # The transcribed path's own family id. FAM_WATER is kept because the
    # alphaMode reclassification above and --water's placeholder both
    # still refer to it; FAM_WATER_FANCY is what the SHADING dispatch
    # selects on, so the two can never be confused for one another.
    FAM_WATER_FANCY = FAM_WATER
    # --- THE THREE SMALL FAMILIES ------------------------------------
    # Resolved by NAME out of fam_names, like every other family index
    # here, so a pack that reorders FAMILIES cannot silently point these
    # at the wrong shader. -1 means the pack does not carry the family
    # at all, and every dispatch entry tests for it.
    FAM_EFFECTS = (FAM_NAMES.index("csgo_effects.vfx")
                   if "csgo_effects.vfx" in FAM_NAMES else -1)
    FAM_BLACK_UNLIT = (FAM_NAMES.index("csgo_black_unlit.vfx")
                       if "csgo_black_unlit.vfx" in FAM_NAMES else -1)
    FAM_VERTEXLIT = (FAM_NAMES.index("csgo_vertexlitgeneric.vfx")
                     if "csgo_vertexlitgeneric.vfx" in FAM_NAMES else -1)
    FAM_GLASS = FAM_NAMES.index("csgo_glass.vfx")
    FAM_COMPLEX = FAM_NAMES.index("csgo_complex.vfx")
    FAM_FOLIAGE = FAM_NAMES.index("csgo_foliage.vfx")
    # --- the three families this block adds --------------------------
    # FAM_IMPORTED is the `csgo_imported` SEARCH PATH, not a shader
    # family: csgo/gameinfo.gi lists `Game csgo, Game csgo_imported,
    # Game csgo_core, Game core`, and csgo_imported ships NO
    # shaders_vulkan_dir.vpk at all -- 4 files, one of them a material.
    # That material is `materials/dev/placeholder.vmat`, decompiled (not
    # substring-scanned) out of csgo_imported/pak01_dir.vpk, and its
    # m_shaderName is `csgo_lightmappedgeneric.vfx`. So the search path
    # contributes exactly one material: the engine's dev placeholder,
    # shaded by lightmappedgeneric with its own constants.
    FAM_LIGHTMAPPEDGENERIC = (FAM_NAMES.index("csgo_lightmappedgeneric.vfx")
                              if "csgo_lightmappedgeneric.vfx" in FAM_NAMES
                              else -1)
    FAM_GENERIC = (FAM_NAMES.index("generic.vfx")
                   if "generic.vfx" in FAM_NAMES else -1)
    FAM_IMPORTED = (FAM_NAMES.index(LMG_IMPORTED_FAM)
                    if LMG_IMPORTED_FAM in FAM_NAMES else -1)
    # --- csgo_simple / csgo_simple_3layer_parallax --------------------
    # Selected by MATERIAL FAMILY out of the side table, exactly like the
    # five above; nothing here matches a name at shade time.
    FAM_SIMPLE = FAM_NAMES.index("csgo_simple.vfx")
    FAM_SIMPLE_3LP = FAM_NAMES.index("csgo_simple_3layer_parallax.vfx")
    # --- NON-WORLD families: the viewmodel pass -------------------------
    # These two are not in any map-material scan of de_inferno, so a side
    # table built before they were appended to fast_pack_fam.FAMILIES does
    # not carry them. .index() would raise ValueError with no explanation;
    # name the rebuild instead, and NEVER fall back to -1 -- a family id of
    # -1 matches nothing, which is a feature disabled by construction that
    # reads as enabled.
    FAM_WEAPON = FAM_LEGS_PREPASS = None
    if args.weapon or args.viewmodel or args.viewmodel_legs_prepass:
        _missing_fam = [n for n in ("csgo_weapon.vfx",
                                    "csgo_legs_prepass.vfx")
                        if n not in FAM_NAMES]
        if _missing_fam:
            raise SystemExit(
                f"--fam-side {args.fam_side} predates the non-world "
                f"families: it has no {', '.join(_missing_fam)} entry in "
                f"fam_names. Rebuild it to a DISTINCT file (never over "
                f"assets/fam_side.pt) with\n"
                f"  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
                f"--out assets/fam_side_weapon.pt")
        FAM_WEAPON = FAM_NAMES.index("csgo_weapon.vfx")
        FAM_LEGS_PREPASS = FAM_NAMES.index("csgo_legs_prepass.vfx")
    # --- non-world families -------------------------------------------
    # Guarded .index() with a -1 fallback, NOT the bare .index() above: a
    # side table built before these three families were appended has no
    # entry for them, and the bare form would raise here and take the whole
    # run down for a feature the run never asked for.
    _fx = lambda n: FAM_NAMES.index(n) if n in FAM_NAMES else -1
    FAM_CHARACTER = _fx("csgo_character.vfx")
    FAM_EYEBALL = _fx("csgo_eyeball.vfx")
    FAM_CUSTOMGLOVE = _fx("csgo_customglove.vfx")
    # --- NON-WORLD FAMILIES ------------------------------------------
    # These two are NOT map materials, so a side table built by scanning
    # the map's materials cannot contain them and .index() would raise.
    # They are appended to the family-name list instead, which is why
    # they get their own ids rather than being looked up: a runtime
    # decal and a particle are spawned, not authored into the map.
    #
    # csgo_projected_decals is a DIFFERENT FAMILY from
    # csgo_static_overlay -- separate .vcs, separate 3,048-combo space,
    # and axes with no counterpart there (S_TRIPLANAR_MAPPING,
    # S_CUTOFF_ANGLE, S_PARALLAX, S_BLOOD_AGING, D_MSAA_DEPTH_BUFFER,
    # D_TRANSLUCENT_SCENE_DEPTH). FAM_OVERLAY stays what it is.
    for _nwname in ("csgo_projected_decals.vfx", "spritecard.vfx"):
        if _nwname not in FAM_NAMES:
            FAM_NAMES.append(_nwname)
    FAM_PROJECTED_DECALS = FAM_NAMES.index("csgo_projected_decals.vfx")
    FAM_SPRITECARD = FAM_NAMES.index("spritecard.vfx")
    # A side table built before the transparency axes landed has no column
    # for them. Reading a missing column would either crash deep inside a
    # render loop or -- far worse -- be papered over with a zero default,
    # which is a feature that reads as tested and is off. Fail here, loudly,
    # naming the rebuild command.
    _need2 = []
    if args.translucent:
        _need2.append("f_translucent")
    if args.cubemap_refraction:
        _need2 += ["f_opaque_cube_refract", "refract_min_rough", "reflectance"]
    if args.water_fancy or args.water_selftest:
        # Naming a REPRESENTATIVE of every block the shader reads, not
        # just one column: a partial rebuild that added the axis flags
        # and none of the wave/foam/debris/SSR/caustic parameters would
        # otherwise run with zeros and read as a tested shader.
        _need2 += ["wf_reflection_type", "wf_refraction", "wf_caustics",
                   "wf_interaction", "wf_blur_refraction",
                   "wf_wave_scale_u", "wf_wave_iter", "wf_low_freq",
                   "wf_rough_min", "wf_foam_scale", "wf_debris_scale",
                   "wf_decay_r", "wf_fresnel_exp", "wf_ssr_steps",
                   "wf_refract_limit", "wf_caustic_str", "wf_eff_ripple",
                   "wf_map_uv_min_u", "wf_roughness_x",
                   "wf_simple_sky_r", "has_wwaves", "has_weffect"]
    if SIMPLE_ON:
        _need2 += ["f_simple_metal_tex", "f_simple_ao_tex",
                   "simple_metalness", "has_simple_ao"]
    if S3LP_ON:
        _need2 += ["f_s3lp_tint_mask", "has_s3lp_tint",
                   "f_s3lp_secondary_uv", "f_s3lp_transmissive",
                   "f_s3lp_layer2_cube", "has_s3lp_layer2_cube",
                   "s3lp_refract1", "s3lp_refract2",
                   "s3lp_lod1_min", "s3lp_lod1_max",
                   "s3lp_lod2_min", "s3lp_lod2_max",
                   "s3lp_layer2_fresnel", "has_s3lp_refract1",
                   "has_s3lp_refract2", "has_s3lp_layer2_fresnel",
                   "has_simple_ao", "simple_metalness",
                   "f_simple_metal_tex",
                   # already present since the first side table, reused
                   # rather than duplicated
                   "has_l0mask", "has_l1col", "has_l2col",
                   "layer1_offset", "layer2_offset",
                   "layer1_emissive", "layer2_emissive"]
    # --- csgo_weapon columns -------------------------------------------
    # Same refusal contract as above and for the same reason: reading a
    # missing column as 0 ships an axis that is off by construction and
    # reads as tested.
    if args.weapon:
        _need2 += ["wpn_f_adjust", "wpn_hue_shift", "wpn_ramp_freq",
                   "wpn_ramp_phase", "wpn_ramp_amount",
                   "wpn_f_tint_mask", "wpn_has_tintmask",
                   "wpn_f_self_illum", "wpn_si_scroll_u", "wpn_si_scroll_v",
                   "wpn_f_glitter", "wpn_glitter_scale", "wpn_glitter_uv",
                   "wpn_glitter_spread", "wpn_glitter_balance",
                   "wpn_f_sfx_mask", "wpn_sfx_amount", "wpn_sfx_speed",
                   "wpn_shimmer", "wpn_f_stickers", "wpn_sticker_slots",
                   "wpn_f_opaque_refract", "wpn_refract_scale",
                   "wpn_refract_blur", "wpn_refract_edge",
                   "wpn_refract_amount", "wpn_refract_contrast",
                   "wpn_refract_tint", "wpn_f_alpha_test", "wpn_alpha_ref",
                   "wpn_f_translucent", "wpn_f_additive",
                   "wpn_f_mode_depth", "wpn_f_tools_vis",
                   "wpn_metal_lo", "wpn_metal_hi",
                   "wpn_shimmer_tint_r", "wpn_shimmer_tint_g",
                   "wpn_shimmer_tint_b"]
    if args.viewmodel_legs_prepass:
        _need2 += ["legs_fade_dist", "legs_cone_cos", "legs_eye_z"]
    if args.weapon_stickers:
        _need2 += ["wpn_stk_scale", "wpn_stk_rot", "wpn_stk_off_u",
                   "wpn_stk_off_v", "wpn_stk_wear", "wpn_stk_holo"]
    _miss2 = [k for k in _need2 if k not in EXT2]
    if _miss2:
        raise SystemExit(
            f"--fam-side {args.fam_side} predates these axes: it has no "
            f"{', '.join(_miss2)} column. Rebuild it with\n"
            f"  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
            f"--out {args.fam_side}")
    # --- csgo_lightmappedgeneric / generic.vfx / csgo_imported --------
    # Same refusal, same reason: a zero-filled missing column would be an
    # axis that is off by construction and reads as tested.
    T_L1AO = T_L1NR = None
    LMG_LIVE = False
    if LMG_ON:
        _missL = [k for k in LMG_NEED_COLS if k not in EXT2]
        _missL += [k for k in ("t_l1ao", "t_l1nr") if k not in _fs]
        if LMG_IMPORTED_FAM not in FAM_NAMES:
            _missL.append("fam_names/" + LMG_IMPORTED_FAM)
        if _missL:
            raise SystemExit(
                f"--fam-side {args.fam_side} predates the "
                f"csgo_lightmappedgeneric / generic.vfx / csgo_imported "
                f"block: it is missing {len(_missL)} of its columns, "
                f"first {_missL[:6]}. Rebuild with\n"
                f"  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
                f"--out <a NEW filename, not the shared fam_side.pt>\n"
                f"or pass --lmg-family off to shade those three families "
                f"with the generic composite instead (which is a DIFFERENT "
                f"shader, not the same one with a feature missing).")
        T_L1AO = _fs["t_l1ao"].to(device).long()
        T_L1NR = _fs["t_l1nr"].to(device).long()
        LMG_LIVE = True
    # --- THE THREE SMALL FAMILIES: their columns and texture slots ----
    # Same contract as the blocks above: a side table built before these
    # columns existed CANNOT serve them, and reading a missing column as
    # 0 would ship three families that read as tested and are off. The
    # refusal names the rebuild command.
    T_SF_FXM1 = T_SF_FXM2 = T_SF_FXM3 = T_SF_FXTINT = None
    T_SF_VLTINT = T_SF_VLAO = T_SF_VLMETAL = None
    T_SF_VLDETMASK = T_SF_VLSELFILLUM = None
    if SFAM or args.sf_flop_audit or args.sf_selftest:
        _sf_cols = (
            [f"sf_fx_{k}" for k in (
                "opacity_scale", "color_boost", "fresnel_exp",
                "fresnel_falloff", "fresnel_min", "fresnel_max",
                "feather_dist", "feather_falloff", "fade_dist",
                "fade_falloff", "fade_min", "fade_max", "additive",
                "tint_mask", "depth_feather", "backfaces", "fog_enabled",
                "has_mask1", "has_mask2", "has_mask3", "has_tintmask")]
            + [f"sf_fx_m{i}_{c}" for i in (1, 2, 3)
               for c in ("pu", "pv", "su", "sv")]
            + ["sf_bu_fog_enabled"]
            + [f"sf_vl_{k}" for k in (
                "spec_direct", "spec_indirect", "force_uv2", "decal",
                "detail", "alpha_test", "translucent", "additive",
                "tint_mask", "self_illum", "fog_enabled", "ao_scale",
                "tintmask_bias", "reflectance", "detail_blend",
                "detail_to_full", "decal_mode", "opacity_scale",
                "si_tint_r", "si_tint_g", "si_tint_b", "si_albedo",
                "si_scroll_u", "si_scroll_v", "uv_su", "uv_sv", "uv_ou",
                "uv_ov", "nrm_su", "nrm_sv", "nrm_ou", "nrm_ov",
                "has_ao", "has_metal", "has_detmask", "has_selfillum",
                "has_tintmask", "has_decal", "has_detail")])
        _sf_tex = ("t_sf_fxmask1", "t_sf_fxmask2", "t_sf_fxmask3",
                   "t_sf_fxtint", "t_sf_vltint", "t_sf_vlao",
                   "t_sf_vlmetal", "t_sf_vldetmask", "t_sf_vlselfillum")
        _sf_miss = [k for k in _sf_cols if k not in EXT2] \
            + [k for k in _sf_tex if k not in _fs]
        if _sf_miss:
            raise SystemExit(
                f"--sf-* needs a side table built by the CURRENT "
                f"fast_pack_fam.py; {args.fam_side} is missing "
                f"{len(_sf_miss)} of its columns/slots, first: "
                f"{_sf_miss[:6]}. Rebuild with:\n"
                f"  python fam_analysis/fast_pack_fam.py --glb <world>.glb "
                f"--out <side>.pt")
        T_SF_FXM1 = _fs["t_sf_fxmask1"].to(device).long()
        T_SF_FXM2 = _fs["t_sf_fxmask2"].to(device).long()
        T_SF_FXM3 = _fs["t_sf_fxmask3"].to(device).long()
        T_SF_FXTINT = _fs["t_sf_fxtint"].to(device).long()
        T_SF_VLTINT = _fs["t_sf_vltint"].to(device).long()
        T_SF_VLAO = _fs["t_sf_vlao"].to(device).long()
        T_SF_VLMETAL = _fs["t_sf_vlmetal"].to(device).long()
        T_SF_VLDETMASK = _fs["t_sf_vldetmask"].to(device).long()
        T_SF_VLSELFILLUM = _fs["t_sf_vlselfillum"].to(device).long()
    # --- csgo_simple / csgo_simple_3layer_parallax texture slots -------
    # Same "refuse, naming the rebuild command" rule the block above uses:
    # a missing slot read as 0 would be a black AO map multiplying the
    # whole indirect term, i.e. a feature that is catastrophically ON, not
    # quietly off. Neither failure mode is acceptable silently.
    T_SIMPAO = T_S3TINT = T_L2CUBE = None
    MAT_L2EM = None
    if SIMPLE_ON or S3LP_ON:
        _need_tex2 = [k for k in ("t_simpao", "t_s3tint", "t_l2cube",
                                  "l2emtint") if k not in _fs]
        if _need_tex2:
            raise SystemExit(
                f"--fam-side {args.fam_side} predates csgo_simple / "
                f"csgo_simple_3layer_parallax: it has no "
                f"{', '.join(_need_tex2)}. Rebuild it with\n"
                f"  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
                f"--out {args.fam_side}")
        T_SIMPAO = _fs["t_simpao"].to(device).long()
        T_S3TINT = _fs["t_s3tint"].to(device).long()
        T_L2CUBE = _fs["t_l2cube"].to(device).long()
        MAT_L2EM = _fs["l2emtint"].to(device)
    FAMTEX = _fs["fam_tex"].to(device)
    TF = FAMTEX.shape[1]
    # ==================================================================
    # NON-WORLD FAMILIES: columns, textures, and REACHABILITY
    # ==================================================================
    T_DFALLOFF = T_SSSMASK = T_DECALMASK = None
    T_BLOODMASK = T_BLOODCOL = T_BLOODNRM = T_IRIDTHICK = None
    T_PATCH = T_PATCHB = None
    T_EYEALB = T_EYEMASK = T_TINTMASK_CH = None
    T_GLSURF = T_GLSUB = T_GLDMG = T_GLGRIME = T_GLGRUNGE = None
    T_GLDETAIL = T_GLPATTERN = T_GLTINTID = T_GLLAYERMASK = T_GLNRM = None
    if NONWORLD:
        _need3 = [
            "f_character", "f_eyeball", "f_customglove",
            "f_char_sss", "f_char_vtx_curvature", "ch_curvature_scale",
            "has_diffuse_falloff", "f_char_aniso", "f_char_spherical_aniso",
            "ch_aniso_amount", "f_char_hair", "ch_hair_rough_scale",
            "ch_hair_shift", "ch_hair_transmission", "f_char_cloth",
            "ch_sheen_scale", "ch_sheen_tint", "f_char_irid",
            "ch_irid_strength", "ch_irid_hue_shift", "ch_irid_fresnel",
            "has_irid_thick", "f_char_retro", "f_char_patches",
            "ch_p0_scale", "ch_p1_scale", "ch_p2_scale",
            "f_char_decal", "ch_decal_blend_mode", "has_char_decal",
            "has_blood_mask", "f_char_adjust", "ch_hue_shift",
            "ch_saturation", "ch_brightness", "ch_contrast",
            "f_char_detail", "ch_detail_blend_mode", "ch_rough_bright",
            "ch_rough_contrast", "ch_mask_rough_by_tint", "f_char_eyes",
            "eye_radius", "eye_iris_size", "eye_pupil_size",
            "eye_hue_shift", "eye_saturation", "has_eye_albedo",
            "eye_metalness", "eye_reflectance",
            "f_char_tint_mask", "f_char_alpha_test", "ch_alpha_ref",
            "f_char_translucent", "f_char_additive", "ch_opacity_scale",
            "ch_reflectance", "ch_aa_edge_strength", "ch_ao_masking",
            "ch_invuln_r", "ch_spawn_invuln",
            "f_glove", "f_glove_aniso", "f_glove_tint_id",
            "f_glove_backcompat", "gl_wear_progress", "gl_wear_exponent",
            "gl_sheen_scale", "gl_tint1_r", "has_glove_surface",
        ]
        # The tint-id composite reads eight vec4 tints and a texel step.
        # A side table built before those columns cannot serve the 9-tap
        # path, and reading a missing column as 0 would silently give every
        # id a black premultiplied tint -- so it is a refusal, not a default.
        _need3 = _need3 + ["gl_tintid_texel"] + [
            "gl_tint%d_%s" % (i, c) for i in range(1, 9) for c in "rgba"]
        _need3t = ["t_diffuse_falloff", "t_sssmask", "t_decalmask",
                   "t_bloodmask", "t_bloodcol", "t_bloodnrm", "t_iridthick",
                   "t_patch0", "t_patch0b", "t_patch1", "t_patch1b",
                   "t_patch2", "t_patch2b", "t_eyealbedo", "t_eyemask",
                   "t_char_tintmask",
                   "t_gl_surface", "t_gl_substrate", "t_gl_damage",
                   "t_gl_grime", "t_gl_grunge", "t_gl_detail",
                   "t_gl_pattern", "t_gl_tintid", "t_gl_layermask",
                   "t_gl_normal"]
        _miss3 = ([k for k in _need3 if k not in EXT2]
                  + [k for k in _need3t if k not in _fs])
        if _miss3 or FAM_CHARACTER < 0:
            raise SystemExit(
                "the csgo_character / csgo_eyeball / csgo_customglove axes "
                "need a side table built by the CURRENT fast_pack_fam.py; "
                f"this one is missing {len(_miss3)} of its columns"
                + (" and does not list csgo_character.vfx at all"
                   if FAM_CHARACTER < 0 else "")
                + (f", first: {_miss3[:6]}" if _miss3 else "") + ".\n"
                "  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
                "--out assets/fam_side_nonworld.pt\n"
                "and pass it as --char-side (NEVER over assets/fam_side.pt).")
        _g = lambda k: _fs[k].to(device).long()
        T_DFALLOFF, T_SSSMASK = _g("t_diffuse_falloff"), _g("t_sssmask")
        T_DECALMASK = _g("t_decalmask")
        T_BLOODMASK, T_BLOODCOL = _g("t_bloodmask"), _g("t_bloodcol")
        T_BLOODNRM, T_IRIDTHICK = _g("t_bloodnrm"), _g("t_iridthick")
        T_PATCH = [_g("t_patch0"), _g("t_patch1"), _g("t_patch2")]
        T_PATCHB = [_g("t_patch0b"), _g("t_patch1b"), _g("t_patch2b")]
        T_EYEALB, T_EYEMASK = _g("t_eyealbedo"), _g("t_eyemask")
        T_TINTMASK_CH = _g("t_char_tintmask")
        T_GLSURF, T_GLSUB = _g("t_gl_surface"), _g("t_gl_substrate")
        T_GLDMG, T_GLGRIME = _g("t_gl_damage"), _g("t_gl_grime")
        T_GLGRUNGE, T_GLDETAIL = _g("t_gl_grunge"), _g("t_gl_detail")
        T_GLPATTERN, T_GLTINTID = _g("t_gl_pattern"), _g("t_gl_tintid")
        T_GLLAYERMASK, T_GLNRM = _g("t_gl_layermask"), _g("t_gl_normal")
        # --- REACHABILITY, measured and printed ------------------------
        # These are MODEL materials; a de_inferno MAP export contains none
        # of them, so at --char-assign=family the three paths run on zero
        # pixels. That is a usage fact about this pack, not a capability
        # fact about the shader -- but a run that silently shaded nothing
        # would be indistinguishable from a run whose code never executed,
        # which is the failure this project keeps hitting. So the count is
        # measured here, printed, and --char-assign is named in the message.
        _nch = int((MAT_FAM == FAM_CHARACTER).sum())
        _ney = int((MAT_FAM == FAM_EYEBALL).sum()) if FAM_EYEBALL >= 0 else 0
        _ngl = (int((MAT_FAM == FAM_CUSTOMGLOVE).sum())
                if FAM_CUSTOMGLOVE >= 0 else 0)
        if args.char_assign != "family":
            _src = {"vertexlit": "csgo_vertexlitgeneric.vfx",
                    "complex": "csgo_complex.vfx", "all": None}[
                        args.char_assign]
            if _src is None:
                _pick = torch.ones_like(MAT_FAM, dtype=torch.bool)
            else:
                _pick = (MAT_FAM == _fx(_src)) if _fx(_src) >= 0 else \
                    torch.zeros_like(MAT_FAM, dtype=torch.bool)
            # Split the picked set three ways so all THREE families reach
            # pixels, not just the largest. The split is by material index
            # parity/modulo, which is arbitrary and said to be arbitrary --
            # it is a reachability harness, not a claim about which model
            # uses which shader.
            _idx = torch.arange(MAT_FAM.shape[0], device=device)
            _want = [FAM_CHARACTER]
            if args.eyeball and FAM_EYEBALL >= 0:
                _want.append(FAM_EYEBALL)
            if args.glove and FAM_CUSTOMGLOVE >= 0:
                _want.append(FAM_CUSTOMGLOVE)
            MAT_FAM = MAT_FAM.clone()
            MAT_EXT2 = MAT_EXT2.clone()
            for _j, _f2 in enumerate(_want):
                _m = _pick & ((_idx % len(_want)) == _j)
                MAT_FAM[_m] = _f2
                # The family columns must agree with the reassignment, or
                # every per-material gate below reads 0 and the path is
                # dead in a way that looks like data.
                for _c, _v in (("f_character", FAM_CHARACTER),
                               ("f_eyeball", FAM_EYEBALL),
                               ("f_customglove", FAM_CUSTOMGLOVE)):
                    if _c in EXT2:
                        MAT_EXT2[_m, EXT2[_c]] = 1.0 if _f2 == _v else 0.0
            _nch = int((MAT_FAM == FAM_CHARACTER).sum())
            _ney = (int((MAT_FAM == FAM_EYEBALL).sum())
                    if FAM_EYEBALL >= 0 else 0)
            _ngl = (int((MAT_FAM == FAM_CUSTOMGLOVE).sum())
                    if FAM_CUSTOMGLOVE >= 0 else 0)
            print(f"--char-assign {args.char_assign}: reassigned "
                  f"{int(_pick.sum())} materials across "
                  f"{len(_want)} non-world families", flush=True)
        print(f"non-world families on this pack: csgo_character {_nch} "
              f"materials, csgo_eyeball {_ney}, csgo_customglove {_ngl}",
              flush=True)
        # The sphere origin S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS
        # projects onto. The engine has the draw's object-space origin;
        # a packed world has no per-draw model transform, so it is the
        # centroid of the geometry the family owns. Named as a
        # substitution rather than silently defaulted to (0,0,0), which
        # would put the projection centre outside the mesh.
        _cf = (MAT_FAM[data["face_matid"].to(device).long()]
               == FAM_CHARACTER)
        if bool(_cf.any()):
            CHAR_ORIGIN = vertices[faces0[_cf].reshape(-1)].mean(0) \
                .view(1, 1, 1, 3)
        else:
            CHAR_ORIGIN = vertices.mean(0).view(1, 1, 1, 3)
        print("SUBSTITUTION: S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS "
              "projects about the csgo_character geometry's CENTROID "
              f"{[round(float(v), 3) for v in CHAR_ORIGIN.view(3)]}; the "
              "engine uses the draw's object-space origin, which a packed "
              "world does not carry", flush=True)
        if _nch + _ney + _ngl == 0:
            print("  *** csgo_character / csgo_eyeball / csgo_customglove "
                  "are MODEL materials and this pack is a MAP export, so "
                  "ZERO materials select them and the three shading paths "
                  "will reach ZERO pixels. That is a usage fact about the "
                  "pack, not a capability fact about the shader. Pass "
                  "--char-assign vertexlit (or complex, or all) to run them "
                  "on real geometry. ***", flush=True)
    # --- BATCH-2 side table -------------------------------------------
    # A SEPARATE FILE. It is never written over assets/fam_side.pt: that
    # pack is what every other agent is scoring against, and appending to
    # it in place would move their baseline underneath them. The loader
    # below refuses a table that predates these columns rather than
    # reading a missing one as zero -- a zero default here would ship six
    # families that read as tested and are off.
    B2_T_STICKER = B2_T_STK_N = B2_T_STK_BASE_N = None
    B2_T_HOLOMASK = B2_T_HOLO_LUT = B2_T_FOIL = B2_T_WEAR = None
    B2_T_GLOVE_MASK = B2_T_GLOVE_IDX = B2_T_GLOVE_WEAR = None
    B2_T_GLOVE_ALB = B2_T_GLOVE_NRM = B2_T_GLOVE_MRA = None
    B2_T_SIMPLE_COL = B2_T_SIMPLE_NRM = B2_T_SIMPLE_AO = None
    B2_T_4W_COL = B2_T_4W_NRM = B2_T_4W_AO = B2_T_4W_DETAIL = None
    B2_T_LIQ_COL = B2_T_LIQ_NRM = B2_T_LIQ_RIPPLE = B2_T_LIQ_FOAM = None
    B2_T_CW_WEAR = B2_T_CW_MASK = B2_T_CW_PATTERN = None
    B2_T_CW_GRUNGE = B2_T_CW_FINISH = None
    if B2:
        if not args.b2_side:
            raise SystemExit(
                "--b2-families needs --b2-side, a side table built by\n"
                "  python fam_analysis/fast_pack_fam.py --glb <map>.glb "
                "--out assets/fam_side.pt --b2-out assets/fam_side_b2six.pt\n"
                "It is a DISTINCT file from --fam-side and must never be "
                "written over assets/fam_side.pt.")
        import os as _os2
        if _os2.path.abspath(args.b2_side) == _os2.path.abspath(
                args.fam_side or ""):
            raise SystemExit(
                f"--b2-side and --fam-side are the same file "
                f"({args.b2_side}). The batch-2 table is append-only and "
                f"SEPARATE by construction; pointing them at one file is "
                f"the overwrite this was written to make impossible.")
        _b2 = torch.load(args.b2_side, map_location="cpu",
                         weights_only=False)
        _b2keys = {k: i for i, k in enumerate(_b2["ext2_keys"])}
        _need_b2 = [k for k in B2_EXT2_KEYS if k not in _b2keys]
        if _need_b2:
            raise SystemExit(
                f"--b2-side {args.b2_side} predates the batch-2 columns: "
                f"it is missing {len(_need_b2)} of them, first "
                f"{_need_b2[:6]}. Rebuild it with --b2-out. Reading a "
                f"missing column as 0 would disable a path silently, "
                f"which is worse than not having it.")
        # The batch-2 columns are APPENDED to the same ext2 matrix, so
        # they are addressed through the same _e2() the rest of the file
        # uses and can never be read at the wrong index.
        MAT_EXT2 = torch.cat([MAT_EXT2, _b2["ext2"].to(device)], dim=1)
        for _k in B2_EXT2_KEYS:
            EXT2[_k] = MAT_EXT2.shape[1] - len(B2_EXT2_KEYS) \
                + B2_EXT2_KEYS.index(_k)
        _bt = lambda k: _b2[k].to(device).long()
        B2_T_STICKER = _bt("b2_t_sticker")
        B2_T_STK_N = _bt("b2_t_sticker_n")
        B2_T_STK_BASE_N = _bt("b2_t_stk_base_n")
        B2_T_HOLOMASK = _bt("b2_t_holomask")
        B2_T_HOLO_LUT = _bt("b2_t_holo_lut")
        B2_T_FOIL = _bt("b2_t_foil")
        B2_T_WEAR = _bt("b2_t_wear")
        B2_T_GLOVE_MASK = _bt("b2_t_glove_mask")
        B2_T_GLOVE_IDX = _bt("b2_t_glove_idx")
        B2_T_GLOVE_WEAR = _bt("b2_t_glove_wear")
        B2_T_GLOVE_ALB = _bt("b2_t_glove_alb")       # (M, 4) -- 4 layers
        B2_T_GLOVE_NRM = _bt("b2_t_glove_nrm")
        B2_T_GLOVE_MRA = _bt("b2_t_glove_mra")
        B2_T_SIMPLE_COL = _bt("b2_t_simple_col")
        B2_T_SIMPLE_NRM = _bt("b2_t_simple_nrm")
        B2_T_SIMPLE_AO = _bt("b2_t_simple_ao")
        B2_T_4W_COL = _bt("b2_t_4w_col")             # (M, 4)
        B2_T_4W_NRM = _bt("b2_t_4w_nrm")
        B2_T_4W_AO = _bt("b2_t_4w_ao")
        B2_T_4W_DETAIL = _bt("b2_t_4w_detail")
        B2_T_LIQ_COL = _bt("b2_t_liq_col")
        B2_T_LIQ_NRM = _bt("b2_t_liq_nrm")
        B2_T_LIQ_RIPPLE = _bt("b2_t_liq_ripple")
        B2_T_LIQ_FOAM = _bt("b2_t_liq_foam")
        B2_T_CW_WEAR = _bt("b2_t_cw_wear")
        B2_T_CW_MASK = _bt("b2_t_cw_mask")
        B2_T_CW_PATTERN = _bt("b2_t_cw_pattern")
        B2_T_CW_GRUNGE = _bt("b2_t_cw_grunge")
        B2_T_CW_FINISH = _bt("b2_t_cw_finish")
        # The batch-2 images live in the SAME fam_tex array, appended, so
        # sample_fam() addresses them unchanged.
        _off = FAMTEX.shape[0]
        FAMTEX = torch.cat([FAMTEX, _b2["fam_tex"].to(device)], dim=0)
        for _n in ("B2_T_STICKER", "B2_T_STK_N", "B2_T_STK_BASE_N",
                   "B2_T_HOLOMASK", "B2_T_HOLO_LUT", "B2_T_FOIL",
                   "B2_T_WEAR", "B2_T_GLOVE_MASK", "B2_T_GLOVE_IDX",
                   "B2_T_GLOVE_WEAR", "B2_T_GLOVE_ALB", "B2_T_GLOVE_NRM",
                   "B2_T_GLOVE_MRA", "B2_T_SIMPLE_COL", "B2_T_SIMPLE_NRM",
                   "B2_T_SIMPLE_AO", "B2_T_4W_COL", "B2_T_4W_NRM",
                   "B2_T_4W_AO", "B2_T_4W_DETAIL", "B2_T_LIQ_COL",
                   "B2_T_LIQ_NRM", "B2_T_LIQ_RIPPLE", "B2_T_LIQ_FOAM",
                   "B2_T_CW_WEAR", "B2_T_CW_MASK", "B2_T_CW_PATTERN",
                   "B2_T_CW_GRUNGE", "B2_T_CW_FINISH"):
            globals()[_n] = globals()[_n] + _off
        # FAM_NAMES is APPENDED to, never reordered: an existing id must
        # keep meaning what it meant, or every other agent's --fam-only
        # number silently changes family.
        _b2names = [n for n in _b2["fam_names"] if n not in FAM_NAMES]
        FAM_NAMES = FAM_NAMES + _b2names
        _remap = {i: FAM_NAMES.index(n)
                  for i, n in enumerate(_b2["fam_names"])}
        _bfam = _b2["fam_id"].to(device).long()
        _newfam = torch.zeros_like(_bfam)
        for _a, _b in _remap.items():
            _newfam[_bfam == _a] = _b
        # a material the base table already classified keeps its id; only
        # the six new families are adopted from the batch-2 table.
        _isb2 = torch.zeros_like(_bfam, dtype=torch.bool)
        for _n in _b2names:
            _isb2 |= (_newfam == FAM_NAMES.index(_n))
        MAT_FAM = torch.where(_isb2, _newfam, MAT_FAM)
        print(f"batch-2 side table: {args.b2_side}, "
              f"{_b2['fam_tex'].shape[0]} extra images, "
              f"{len(B2_EXT2_KEYS)} appended columns, "
              f"{len(_b2names)} appended families "
              f"({int(_isb2.sum())} materials adopted)", flush=True)
    # --- GROUND-TRUTH MATERIAL-SURFACE COMBO AXES ---------------------
    # A side table built before these columns existed cannot serve them.
    # Refusing here is the point: the alternative is reading a missing
    # column as 0 and shipping an axis that is disabled by construction,
    # which is exactly how --ibl-lod-scale=1.0 kept mips 2-6 unreachable
    # while every arm above it reported as tested.
    T_DETAIL = T_DNRM1 = T_DNRM2 = T_DECAL = T_ANISO = None
    T_COL3 = T_NRM3 = T_HGT3 = T_MET2 = None
    if GTSURF:
        _need_cols = (
            "f_detail", "detail_blend", "has_detail", "detail_su",
            "detail_sv", "detail_ou", "detail_ov",
            "f_detail_normal", "detail_nrm_strength", "has_detail_nrm1",
            "has_detail_nrm2", "f_tint_mask2", "has_tintmask2",
            "tm_bright", "tm_contrast", "f_metal_tex", "has_metal2",
            "metal_scalar1", "metal_scalar2", "f_self_illum2",
            "has_selfillum2", "si_brightness", "si_scale",
            "si_albedo_factor2", "f_secondary_uv", "uvsel_c1", "uvsel_c2",
            "uvsel_c3", "uvsel_dn", "uvsel_ovl", "uv2_su", "uv2_sv",
            "uv2_ou", "uv2_ov", "f_decal", "has_decal", "decal_blend",
            "decal_su", "decal_sv", "decal_ou", "decal_ov", "f_aniso",
            "has_aniso", "aniso_amount", "aniso_rotation", "f_layer3",
            "has_c3", "has_n3", "has_h3", "h_scale3", "h_zero3",
            "blend_soft3", "cc3_r", "cc3_g", "cc3_b", "cc3_bright",
            "cc3_contrast", "cc3_sat", "f_tex_anim", "anim_method",
            "anim_cells_x", "anim_cells_y", "anim_num_cells",
            "anim_time_per_frame", "anim_time_offset", "anim_seed",
            "scroll_u", "scroll_v", "f_overlay2", "has_overlay2",
            "overlay_mode2", "overlay_tintmask2", "ovl_bright", "ovl_dark",
            "ovl_su", "ovl_sv", "ovl_ou", "ovl_ov", "f_blend_effects",
            "border_offset", "border_soft", "border_spread", "border_rough",
            "f_border_rough", "bevel_strength", "bevel_soft",
            "btint_r", "btint_g", "btint_b",
            "f_blend_effects3", "border_offset3", "border_soft3",
            "border_spread3", "border_rough3", "f_border_rough3",
            "bevel_strength3", "bevel_soft3",
            "btint3_r", "btint3_g", "btint3_b", "f_alpha_test",
            "alpha_ref_gt", "f_alpha_test_layer", "alpha_ref2",
            "f_use_new_blending", "f_enable_vis", "vis_mode",
            "f_material_ref", "mref_r", "mref_g", "mref_b", "mref_bright",
            "mref_contrast", "mref_sat")
        _missing = [k for k in _need_cols if k not in EXT2]
        _need_tex = ("t_detail", "t_dnrm1", "t_dnrm2", "t_decal", "t_aniso",
                     "t_col3", "t_nrm3", "t_hgt3", "t_met2")
        _missing += [k for k in _need_tex if k not in _fs]
        if _missing:
            raise SystemExit(
                "the material-surface combo axes need a fam_side.pt built "
                "by the CURRENT fast_pack_fam.py; this one is missing "
                f"{len(_missing)} of them, first: {_missing[:6]}. Rebuild "
                "with: fast_pack_fam.py --glb <world.glb> --out fam_side.pt")
        T_DETAIL = _fs["t_detail"].to(device).long()
        T_DNRM1 = _fs["t_dnrm1"].to(device).long()
        T_DNRM2 = _fs["t_dnrm2"].to(device).long()
        T_DECAL = _fs["t_decal"].to(device).long()
        T_ANISO = _fs["t_aniso"].to(device).long()
        T_COL3 = _fs["t_col3"].to(device).long()
        T_NRM3 = _fs["t_nrm3"].to(device).long()
        T_HGT3 = _fs["t_hgt3"].to(device).long()
        T_MET2 = _fs["t_met2"].to(device).long()
        _n2 = lambda k: int((MAT_EXT2[:, EXT2[k]] > 0.5).sum())
        print("material-surface combo axes, materials carrying each: "
              f"S_DETAIL_TEXTURE {int((MAT_EXT2[:, EXT2['f_detail']] > 0).sum())}"
              f" (values {sorted(set(MAT_EXT2[:, EXT2['f_detail']].tolist()))}), "
              f"S_DETAIL_NORMAL {_n2('f_detail_normal')}, "
              f"S_TINT_MASK {_n2('f_tint_mask2')}, "
              f"S_METALNESS_TEXTURE {_n2('f_metal_tex')}, "
              f"S_SELF_ILLUM {_n2('f_self_illum2')}, "
              f"S_SECONDARY_UV {_n2('f_secondary_uv')}, "
              f"S_DECAL_TEXTURE {_n2('f_decal')}, "
              f"S_ANISOTROPIC_GLOSS {_n2('f_aniso')}, "
              f"S_ENABLE_LAYER_3 {_n2('f_layer3')}, "
              f"S_PAINT_VERTEX_COLORS {_n2('f_paint_vc')}, "
              f"S_TEXTURE_ANIMATION {_n2('f_tex_anim')}, "
              f"S_SHARED_COLOR_OVERLAY {_n2('f_overlay2')}, "
              f"S_BLEND_MODE {int((MAT_EXT2[:, EXT2['f_blend_mode']] > 0).sum())}"
              f" (values {sorted(set(MAT_EXT2[:, EXT2['f_blend_mode']].tolist()))}), "
              f"S_BLEND_EFFECTS {_n2('f_blend_effects')}, "
              f"S_BLEND_EFFECTS_3 {_n2('f_blend_effects3')}, "
              f"S_USE_NEW_BLENDING {_n2('f_use_new_blending')}, "
              f"S_ENABLE_VISUALIZATIONS {_n2('f_enable_vis')}, "
              f"S_ALPHA_TEST {_n2('f_alpha_test')}, "
              f"S_ALPHA_TEST_LAYER {_n2('f_alpha_test_layer')}, "
              f"S_MATERIAL_REFERENCE {_n2('f_material_ref')}", flush=True)
        print("  (S_BLEND_EFFECTS_2 is carried by the fast_pack2 pack as "
              "f_blend_effects2 and reported in the vmat feature line)",
              flush=True)
        # ACK FOR THE SECONDARY-UV SELECTORS, because a count of materials
        # carrying S_SECONDARY_UV does NOT answer whether gt_uv can act:
        # its pick is `(uvsel_c* >= 2) AND f_secondary_uv`, so a pack can
        # have the combo on and still select TEXCOORD0 on every slot.
        # Printed as counts of the CONJUNCTION, not of either half, which
        # is the distinction that made "--secondary-uv material changes
        # nothing" unreadable for the whole slot-family arc.
        _f_sec = MAT_EXT2[:, EXT2["f_secondary_uv"]]
        _sel_any = 0
        _parts = []
        for _sk in ("uvsel_c1", "uvsel_c2", "uvsel_c3", "uvsel_dn",
                    "uvsel_ovl"):
            _sv2 = MAT_EXT2[:, EXT2[_sk]]
            _n_sel = int((_sv2 > 1.5).sum())
            _n_pick = int(((_sv2 > 1.5) & (_f_sec > 0)).sum())
            _sel_any += _n_sel
            _parts.append(f"{_sk} sel>=2 on {_n_sel}, AND-combo {_n_pick}")
        print("secondary-UV selectors: " + "; ".join(_parts), flush=True)
        if _sel_any == 0:
            print("  --secondary-uv is INERT ON THIS PACK BY DATA: not one "
                  "material selects UV set 2 on any slot, so gt_uv's pick "
                  "is 0 everywhere and uv2_o is multiplied by zero at both "
                  "call sites. A frame identical with and without the flag "
                  "is the CORRECT result here and says nothing about "
                  "whether the term works. NOTE the packer reads these "
                  "from g_nTexCoordSource1/2/3, g_nDetailTexCoordSource "
                  "and g_nSharedColorOverlayTexCoordSource "
                  "(fast_pack_fam.py:1598-1603); if the map's .vmat_c "
                  "spells them g_nUVSet*/g_nDetailUVSet*/"
                  "g_nColorOverlayUVSet the column is zero because the "
                  "LOOKUP missed, not because the material said 0.",
                  flush=True)
    _mid_all = data["face_matid"].to(device).long()
    # Metres per UV unit, per material. The 3-layer parallax windows need
    # a layer offset expressed in UV, and the vmat gives it in SOURCE
    # units; the triangle's own world-area / uv-area ratio is the only
    # thing that converts one to the other without inventing a constant.
    _p = vertices[faces0]
    _t = uvs[faces0]
    _wa = torch.cross(_p[:, 1] - _p[:, 0], _p[:, 2] - _p[:, 0],
                      dim=1).norm(dim=1) * 0.5
    _u1, _u2 = _t[:, 1] - _t[:, 0], _t[:, 2] - _t[:, 0]
    _ua = (_u1[:, 0] * _u2[:, 1] - _u1[:, 1] * _u2[:, 0]).abs() * 0.5
    _ok = (_wa > 1e-9) & (_ua > 1e-12)
    _sc = (_wa / _ua.clamp(min=1e-12)).sqrt()
    _M = MAT_FAM.shape[0]
    _num = torch.zeros(_M, device=device)
    _den = torch.zeros(_M, device=device)
    _num.index_add_(0, _mid_all[_ok], _sc[_ok])
    _den.index_add_(0, _mid_all[_ok], torch.ones(int(_ok.sum()), device=device))
    MAT_UVSCALE = (_num / _den.clamp(min=1)).clamp(min=1e-4)
    del _p, _t, _wa, _u1, _u2, _ua, _ok, _sc, _num, _den
    print(f"family side table:{FAMTEX.shape[0]} extra textures @{TF} "
          f"({FAMTEX.numel()/1e6:.0f} MB VRAM), "
          f"{len(FAM_NAMES)} families", flush=True)
    if args.alpha_ref:
        # Only where the vmat actually supplies a reference; the rest keep
        # whatever the pack decided.
        _ar = _fs["alpha_cut_vmat"].to(device)[_mid_all]
        _hasar = (MAT_EXT2[:, EXT2["has_alpha_ref1"]] > 0.5)[_mid_all]
        _chg = int(((_ar - face_cut0).abs() > 1e-3 & _hasar).sum()
                   if False else ((_ar != face_cut0) & _hasar).sum())
        face_cut0 = torch.where(_hasar, _ar, face_cut0)
        print(f"  alpha-test reference from the vmat on {_chg:,} faces "
              f"({_chg/len(faces0):.3%})", flush=True)
    if args.alpha_test_gt:
        # S_ALPHA_TEST, every family that declares the axis. --alpha-ref
        # only rewrote csgo_environment's g_flAlphaTestReference1; this
        # is F_ALPHA_TEST + g_flAlphaTestReference wherever the material
        # sets them, which is what selects the S_ALPHA_TEST static combo
        # (csgo_complex weight 40, csgo_environment_blend weight 64).
        _at = (MAT_EXT2[:, EXT2["f_alpha_test"]] > 0.5)[_mid_all]
        _ar2 = MAT_EXT2[:, EXT2["alpha_ref_gt"]][_mid_all]
        _chg2 = int(((face_cut0 != _ar2) & _at).sum())
        face_cut0 = torch.where(_at, _ar2, face_cut0)
        # A material that declares F_ALPHA_TEST but whose glTF alphaMode
        # came out OPAQUE is not alpha-tested today at all; the combo
        # says it is.
        _mv = _at & (face_mode0 == 0)
        face_mode0 = torch.where(_mv, torch.ones_like(face_mode0),
                                 face_mode0)
        print(f"  S_ALPHA_TEST: reference changed on {_chg2:,} faces, "
              f"{int(_mv.sum()):,} faces moved OPAQUE->MASK", flush=True)
    if args.alpha_test_layer:
        # S_ALPHA_TEST_LAYER (csgo_environment bit weight 4). The test
        # runs against the BLENDED layer alpha with its own reference;
        # the per-pixel half is in the MASK pass, this only installs the
        # layer-2 reference so the face carries the right cutoff.
        _al = (MAT_EXT2[:, EXT2["f_alpha_test_layer"]] > 0.5)[_mid_all]
        _arl = MAT_EXT2[:, EXT2["alpha_ref2"]][_mid_all]
        _chg3 = int(((face_cut0 != _arl) & _al).sum())
        face_cut0 = torch.where(_al, _arl, face_cut0)
        _mv2 = _al & (face_mode0 == 0)
        face_mode0 = torch.where(_mv2, torch.ones_like(face_mode0),
                                 face_mode0)
        print(f"  S_ALPHA_TEST_LAYER: layer reference on {_chg3:,} faces, "
              f"{int(_mv2.sum()):,} faces moved OPAQUE->MASK", flush=True)
    if args.translucent or args.additive_blend:
        # S_TRANSLUCENT / S_ADDITIVE_BLEND are STATIC combos: a material
        # that sets F_TRANSLUCENT is compiled into the translucent variant
        # and drawn by the blend unit, full stop. If VRF left such a
        # material in the OPAQUE class the axis has nothing to composite
        # and would silently measure nothing, so the class is taken from
        # the flag -- the same reclassification --overlay-blend and --glass
        # already do for their own families.
        _tsel = torch.zeros_like(MAT_FAM, dtype=torch.bool)
        if args.translucent:
            _tsel |= (MAT_EXT2[:, EXT2["f_translucent"]] > 0.5)
        if args.additive_blend:
            _tsel |= (MAT_EXT2[:, EXT2["f_additive"]] > 0.5)
        _was = int((face_mode0[_tsel[_mid_all]] != 2).sum())
        face_mode0 = torch.where(_tsel[_mid_all],
                                 torch.full_like(face_mode0, 2), face_mode0)
        print(f"  S_TRANSLUCENT/S_ADDITIVE_BLEND: {int(_tsel.sum())} "
              f"materials, {_was:,} faces moved into the BLEND class",
              flush=True)
    if args.overlay_blend or args.glass or args.water or args.sf_effects:
        # VRF writes alphaMode=OPAQUE for the csgo_static_overlay
        # materials (a decal that REPLACES its wall) and for csgo_effects,
        # while F_BLEND_MODE says otherwise. The counts used to be
        # HARDCODED here as "all 24 ... 17 translucent, 6 alpha-tested and
        # 1 modulate"; on the pack this run actually loaded they are 20 and
        # 13/6/1, so the comment was a claim about a run that happened
        # somewhere else. Derived and printed instead, because a count that
        # is computed cannot go stale.
        # Reclassifying is a face-CLASS change, so it has to happen before
        # the cluster sort that permutes every face array.
        _amM = _fs["alpha_mode_ovl"].to(device).long()
        _acM = _fs["alpha_cut_ovl"].to(device)
        _ovM = (MAT_FAM == FAM_NAMES.index("csgo_static_overlay.vfx"))
        _bmC = MAT_EXT2[:, EXT2["f_blend_mode"]]
        print("  csgo_static_overlay: %d materials; F_BLEND_MODE %s"
              % (int(_ovM.sum()),
                 " ".join("mode%d=%d" % (_v, int(((_bmC == _v) & _ovM).sum()))
                          for _v in range(7)
                          if int(((_bmC == _v) & _ovM).sum()))), flush=True)
        # Applied PER FAMILY, so --overlay-blend and --glass can be scored
        # apart: a single table would silently reclassify the glass while
        # the overlay feature was being measured.
        _sel = torch.zeros_like(_amM, dtype=torch.bool)
        if args.overlay_blend:
            _sel |= (MAT_FAM == FAM_NAMES.index("csgo_static_overlay.vfx"))
        if args.glass:
            _sel |= (MAT_FAM == FAM_NAMES.index("csgo_glass.vfx"))
            _sel |= (MAT_FAM == FAM_NAMES.index("csgo_effects.vfx"))
        if args.sf_effects and FAM_EFFECTS >= 0:
            # csgo_effects has its own reason to be in the BLEND class --
            # it is an additive particle -- and until now it only got
            # there as a side effect of --glass, so measuring the effects
            # path meant turning the glass path on. Its own gate.
            _sel |= (MAT_FAM == FAM_EFFECTS)
        if args.water:
            _sel |= (MAT_FAM == FAM_WATER)
            _amM = _amM.clone()
            _amM[FAM_WATER == MAT_FAM] = 2
        _am = torch.where(_sel[_mid_all], _amM[_mid_all], face_mode0)
        _ac = torch.where(_sel[_mid_all], _acM[_mid_all], face_cut0)
        _moved = int((_am != face_mode0).sum())
        face_mode0 = _am
        face_cut0 = _ac
        print(f"  alphaMode reclassified on {_moved:,} faces "
              f"({_moved/len(faces0):.3%})", flush=True)

# BLEND-CLASS POPULATION (task #80). This was written when the counter's
# only call site sat in the OPAQUE/LMG pass and the whole blend class was
# invisible to CLASS_PX. That blind spot is GONE: every raster class now
# writes one attribution map, counted once at the composite, and the
# census prints whether it balances. The census is kept because it is
# still the right operand for a different question -- a class whose entire
# population is blend-class cannot be convicted by a 0, since a blend
# contribution that never resolves is shaded by nobody.
BLEND_FACES_BY_FAM = {}
try:
    _bm2 = (face_mode0 == 2)
    if bool(_bm2.any()):
        _bfam = torch.bincount(MAT_FAM[_mid_all][_bm2].reshape(-1).long(),
                               minlength=len(FAM_NAMES))
        BLEND_FACES_BY_FAM = {int(_i): int(_n)
                              for _i, _n in enumerate(_bfam.tolist()) if _n}
        print("  blend-class census (population, not a blind spot): "
              + " ".join(f"{FAM_NAMES[_i]}={_n}"
                         for _i, _n in sorted(BLEND_FACES_BY_FAM.items())),
              flush=True)
except NameError:
    # no pack loaded on this run; the class matrix prints UNAVAILABLE and
    # this census has no domain.
    pass

# --- THE THREE SMALL FAMILIES: resolve each family's COMBO -------------
# A static combo selects a DIFFERENT COMPILED MODULE, so the axes below
# are resolved to Python values once, here, and the shading functions
# branch on them -- which is what the engine does. Per-PIXEL gating on
# the ext2 column still happens inside, so a family whose materials
# disagree about an axis (de_inferno's eight csgo_vertexlitgeneric
# materials do: 1 of 8 sets F_TINT_MASK) gets each material its own arm.
#
# `--sf-<fam>-<axis>` is a STRING with three values and is compared with
# `==` in every one of them. It is never tested for truthiness: "off" is
# a truthy Python string, and an axis flag read that way is ON when the
# user asked for it to be off.
SF_REACH = {"effects": [0.0, 0.0], "black_unlit": [0.0, 0.0],
            "vertexlit": [0.0, 0.0]}
SF_COMBO = {}
if FAMILY and SFAM:
    def _sf_axis(flag, col, fam_idx):
        """Resolve a three-valued axis flag to a Python bool.

        'on'/'off' are explicit. 'material' asks whether ANY material of
        this family sets the flag -- i.e. whether the compiled module
        that declares the axis has to exist at all. The per-pixel value
        still comes from the column, so 'material' never forces an arm on
        a material that did not ask for it.
        """
        if flag == "on":
            return True
        if flag == "off":
            return False
        if flag != "material":
            raise SystemExit(f"unreachable axis value {flag!r}")
        if fam_idx < 0 or col not in EXT2:
            return False
        m = (MAT_FAM == fam_idx)
        return bool(m.any() and (MAT_EXT2[m, EXT2[col]] > 0.5).any())

    SF_COMBO["effects"] = dict(
        additive=_sf_axis(args.sf_effects_additive, "sf_fx_additive",
                          FAM_EFFECTS),
        tint_mask=_sf_axis(args.sf_effects_tint_mask, "sf_fx_tint_mask",
                           FAM_EFFECTS),
        depth_feather=_sf_axis(args.sf_effects_depth_feather,
                               "sf_fx_depth_feather", FAM_EFFECTS),
        backfaces=_sf_axis(args.sf_effects_backfaces, "sf_fx_backfaces",
                           FAM_EFFECTS),
        fog=_sf_axis(args.sf_effects_fog, "sf_fx_fog_enabled", FAM_EFFECTS),
        tools_vis=args.sf_tools_vis,
        baked=args.sf_effects_baked, mboit=args.sf_effects_mboit,
        msaa=bool(args.sf_effects_msaa))
    SF_COMBO["vertexlit"] = dict(
        spec_direct=_sf_axis(args.sf_vertexlit_spec_direct,
                             "sf_vl_spec_direct", FAM_VERTEXLIT),
        spec_indirect=_sf_axis(args.sf_vertexlit_spec_indirect,
                               "sf_vl_spec_indirect", FAM_VERTEXLIT),
        force_uv2=_sf_axis(args.sf_vertexlit_force_uv2, "sf_vl_force_uv2",
                           FAM_VERTEXLIT),
        decal=_sf_axis(args.sf_vertexlit_decal, "sf_vl_decal", FAM_VERTEXLIT),
        detail=_sf_axis(args.sf_vertexlit_detail, "sf_vl_detail",
                        FAM_VERTEXLIT),
        alpha_test=_sf_axis(args.sf_vertexlit_alpha_test, "sf_vl_alpha_test",
                            FAM_VERTEXLIT),
        translucent=_sf_axis(args.sf_vertexlit_translucent,
                             "sf_vl_translucent", FAM_VERTEXLIT),
        additive=_sf_axis(args.sf_vertexlit_additive, "sf_vl_additive",
                          FAM_VERTEXLIT),
        tint_mask=_sf_axis(args.sf_vertexlit_tint_mask, "sf_vl_tint_mask",
                           FAM_VERTEXLIT),
        self_illum=_sf_axis(args.sf_vertexlit_self_illum, "sf_vl_self_illum",
                            FAM_VERTEXLIT),
        fog=True, tools_vis=args.sf_tools_vis,
        baked=args.sf_vertexlit_baked,
        opaque_fade=bool(args.sf_vertexlit_opaque_fade),
        spec_cube_static=bool(args.sf_vertexlit_spec_cube_static),
        shader_quality=int(args.sf_vertexlit_shader_quality))
    SF_COMBO["black_unlit"] = dict(tools_vis=args.sf_tools_vis)

    _E = SF_COMBO["effects"]
    _V = SF_COMBO["vertexlit"]
    _tv = 1 if args.sf_tools_vis != 0 else 0
    # place values from m_nComboIndexValue, read out of the shipped
    # package -- every axis of all three families is binary, so the
    # mixed-radix id and a bitmask agree HERE and only here.
    _e_id = (_tv + 2 * int(_E["additive"]) + 4 * int(_E["tint_mask"])
             + 8 * int(_E["depth_feather"]))
    _v_id = (int(_V["spec_direct"]) + 2 * int(_V["spec_indirect"])
             + 4 * int(_V["force_uv2"]) + 8 * int(_V["decal"])
             + 16 * int(_V["detail"]) + 32 * _tv
             + 64 * int(_V["alpha_test"]) + 128 * int(_V["translucent"])
             + 256 * int(_V["additive"]) + 512 * int(_V["tint_mask"])
             + 1024 * int(_V["self_illum"])
             + 2048 * int(args.sf_vertexlit_shader_quality))
    _dyn = {"none": 0, "vertex-stream": 1, "probe": 2, "lightmap": 4,
            "auto": 4}
    print("small families: csgo_effects static %d dyn %d | "
          "csgo_black_unlit static %d (family has NO dynamic axis) | "
          "csgo_vertexlitgeneric static %d dyn %d"
          % (_e_id, _dyn[args.sf_effects_baked]
             + (8 if args.sf_effects_msaa else 0), _tv, _v_id,
             _dyn[args.sf_vertexlit_baked]
             + (16 if args.sf_vertexlit_opaque_fade else 0)), flush=True)
    _cf = args.sf_cube_fog_source
    if _cf == "probe" and CUBE is None:
        _cf = "probe requested but no --ibl-cube: the fitted fog colour"
    print("  cube-fog texture source: %s; ramp %g..%g source units"
          % (_cf, args.sf_cube_fog_start, args.sf_cube_fog_end), flush=True)
    if args.sf_effects_msaa:
        print("  D_SCENE_DEPTH_MSAA selected; this renderer has 1 sample "
              "per pixel, so the axis selects the same texel", flush=True)
    if args.sf_effects_baked != "lightmap":
        print("  D_BAKED_LIGHTING_* is MEASURED INERT in csgo_effects: "
              "r10_m0 and r10_m1 differ in one `layout(location=)` and "
              "nothing else", flush=True)
    for _fn, _fi in (("csgo_effects.vfx", FAM_EFFECTS),
                     ("csgo_black_unlit.vfx", FAM_BLACK_UNLIT),
                     ("csgo_vertexlitgeneric.vfx", FAM_VERTEXLIT)):
        print("  %-28s %s" % (_fn, ("%d materials" % int((MAT_FAM == _fi).sum()))
                              if _fi >= 0 else "NOT IN THIS PACK"), flush=True)

n_tex, T = textures.shape[0], textures.shape[1]
print(f"world on GPU: {len(vertices):,} verts, {len(faces0):,} tris, "
      f"{n_tex} textures @{T}", flush=True)

# --- PBR surface set ---------------------------------------------------
if args.normal_map:
    args.vertex_normals = True
# The vmat feature set lives in the same aux array and is indexed by the
# same face_matid, so any of it turns the PBR block on.
# The implication runs BEFORE the VMAT predicate that reads
# args.height_blend, not after it. It used to sit one line below, so
# VMAT read the pre-implication value; that was masked only because
# VMAT's own OR-clause names blend_border and bevel directly, and it
# would have stopped being masked the moment either was dropped from
# that clause. Ordering it correctly costs nothing and removes the
# dependence on a coincidence.
if (args.blend_border or args.bevel) and not args.height_blend:
    args.height_blend = True        # both are the F_BLEND_EFFECTS_2 block
    print("--blend-border/--bevel are the F_BLEND_EFFECTS_2 block; "
          "--height-blend turned on with them", flush=True)
VMAT = (args.height_blend or args.overlay or args.vmat_ao or args.tint_mask
        or args.transmissive or args.metalness_tex or args.self_illum
        or args.blend_border or args.bevel or args.backfaces
        or args.vertex_color_mode or args.height_ch1 != "off")
# THE PACK'S OWN DATA IS A PRECONDITION, not just the flags.
#
# impl-loopfam found this: on a pack that SHIPS mat_ao_pages, the pages were
# silently discarded unless an unrelated flag was passed. Same pack, same
# camera, same tree -- adding --fam-side flipped every AO instrument from
# dark to live. --fam-side was only the flag they happened to try; --overlay
# or any other VMAT/FAMILY trigger does it too, because the real gate is
# PBR, and the AO sampling site lives inside `if PBR:`.
#
# That is verbatim the hazard the AO_PAGES banner names one screen below:
# "a pack that ships AO pages which only apply when an unrelated flag is
# passed is a page that silently does nothing." It was written about --ao
# and was immediately true of a different flag -- which is the tell that the
# defect was never about WHICH flag, but about a data-carrying pack having
# no voice in its own precondition.
#
# `--ao` USED TO SIT IN THIS EXPRESSION and was deleted in d92f8bac as a
# fitted stand-in. Deleting it removed the one term by which AO could turn
# its own surface path on. The replacement is the DATA, which is the same
# substitution the whole AO chain has made: the flag said "pretend there is
# AO", the pages ARE the AO.
_HAS_AO_PAGES = data.get("mat_ao_pages") is not None
PBR = (args.vertex_normals or args.specular or args.emissive
       or _HAS_AO_PAGES or VMAT or FAMILY)
if _HAS_AO_PAGES and not (args.vertex_normals or args.specular
                          or args.emissive or VMAT or FAMILY):
    print("PBR: enabled by the PACK -- it carries mat_ao_pages and no flag "
          "would otherwise have turned the surface path on. Before this, "
          "those pages loaded, announced themselves, and were never "
          "sampled.", flush=True)
AUX = data.get("aux")
# ======================================================================
# THE REFERENCE-LAYOUT PAGES -- the pack side of the normal/roughness/AO hole
# ======================================================================
# CONTRACT with the pack, first-writer-names-the-key:
#
#   mat_normal_pages  (L, E, E, C>=3) uint8 or float. Page L in the
#                     REFERENCE layout: .x/.y the octahedral pair, .z the
#                     ROUGHNESS. Indexed by the EXISTING per-material layer
#                     tables mat_nrm1 / mat_nrm2, so layer1/layer2 need no
#                     new index and cannot disagree with the albedo about
#                     which material is which.
#   mat_ao_pages      (L, E, E, C>=1) uint8 or float, .x the occlusion.
#                     Indexed by the existing mat_ao.
#
# Either key absent -> a printed GAP and the current behaviour, never a
# crash and never a substituted page. Present -> the reference decode
# above, and the runtime says which layout each frame actually sampled,
# because "normals are on" and "normals are on AND in the layout the
# reference reads" are the two states this whole block exists to separate.
NRM_PAGES = data.get("mat_normal_pages")
AO_PAGES = data.get("mat_ao_pages")
if NRM_PAGES is not None:
    NRM_PAGES = NRM_PAGES.to(device)
    NRM_EDGE = int(NRM_PAGES.shape[1])
    print(f"GAP CLOSED normal/roughness: mat_normal_pages "
          f"{tuple(NRM_PAGES.shape)} @{NRM_EDGE}, REFERENCE layout "
          f"(.xy octahedral, .z roughness, csgo_environment_ps.glsl:"
          f"458-463) -- decoded by gt_ref_normal_rough, blended on the "
          f"same weights as the albedo", flush=True)
    # A flag that reads and cannot act is the failure #33 was filed for.
    # --rough-src selects the AUX-array fallback, which this page
    # SUPERSEDES: with the key present, `--rough-src const` and
    # `--rough-src normal-alpha` produce the same frame to every digit,
    # measured (tick 77159: 1.563244329438987 / 3.125316892850285 /
    # 0.1333 under both). Say so, rather than let an arm be configured
    # against a lever that is not connected.
    if args.rough_src != "normal-alpha":
        print(f"  --rough-src {args.rough_src} is INERT this run: the "
              f"reference .z roughness above supersedes the aux-array "
              f"source that flag selects. To reach it, use a pack "
              f"without mat_normal_pages.", flush=True)
else:
    NRM_EDGE = 0
    print("GAP normal/roughness: the pack carries no `mat_normal_pages`, "
          "so the reference-layout decode does NOT run. Normals and "
          "roughness fall back to whatever --normal-map / --rough-src "
          "select over the aux array, which reads roughness from ALPHA "
          "where the reference reads .z -- see gt_ref_normal_rough.",
          flush=True)
if AO_PAGES is not None:
    AO_PAGES = AO_PAGES.to(device)
    AO_EDGE = int(AO_PAGES.shape[1])
    print(f"GAP CLOSED ambient occlusion: mat_ao_pages "
          f"{tuple(AO_PAGES.shape)} @{AO_EDGE}, multiplied into the "
          f"INDIRECT term at csgo_environment_ps.glsl:1230 (q0) / "
          f"1588-1592 (q1) -- inside gt_compose's _gt_lm_tint argument, "
          f"i.e. before the tint pow, which is where 3e2f74ae put the "
          f"tint. NOT gated on --ao: that flag selects the AUX-array "
          f"path, and a pack that ships AO pages which only apply when "
          f"an unrelated flag is passed is a page that silently does "
          f"nothing.", flush=True)
else:
    AO_EDGE = 0
    print("GAP ambient occlusion: the pack carries no `mat_ao_pages`; "
          "--vmat-ao over the aux array is unchanged (--ao is "
          "DELETED) and an "
          "absent AO page leaves the indirect term unoccluded.",
          flush=True)
EXT = {}
DUMP = []
if PBR:
    if AUX is None:
        raise SystemExit(
            "the surface path needs a fast_pack_normals.py pack (no `aux` "
            "in this one). Triggered by "
            + ("the pack's own mat_ao_pages"
               if _HAS_AO_PAGES else
               "--vertex-normals/--normal-map/--specular/--emissive or a "
               "vmat/family flag")
            + ". A pack that carries AO pages but no aux array is a pack "
              "built by two different pipelines; rebuild it rather than "
              "dropping one of them.")
    AUX = AUX.to(device)
    if args.aux_mip and args.aux_mip < AUX.shape[1]:
        _k = AUX.shape[1] // args.aux_mip
        AUX = torch.nn.functional.avg_pool2d(
            AUX.permute(0, 3, 1, 2).float(), _k
        ).permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
        print(f"aux prefiltered to {AUX.shape[1]}^2", flush=True)
    TA = AUX.shape[1]
    vnormal = data["vnormal"].to(device)
    vtangent = data["vtangent"].to(device)
    face_matid0 = data["face_matid"].to(device).long()
    MAT_NRM1 = data["mat_nrm1"].to(device).long()
    MAT_NRM2 = data["mat_nrm2"].to(device).long()
    MAT_AO = data["mat_ao"].to(device).long()
    MAT_EM = data["mat_em"].to(device).long()
    MAT_PBR = data["mat_pbr"].to(device)          # metal, rough, bump, emScale
    MAT_EMTINT = data["mat_emtint"].to(device)
    # mat_nrm1 is >= 0 for EVERY material: the packer substitutes a
    # flat sentinel layer for a missing normalTexture, so counting
    # mat_nrm1 >= 0 always reports 100%. mat_has_nrm is the real flag.
    MAT_HAS_NRM = data["mat_has_nrm"].to(device)
    # A vertex whose TANGENT was absent packs as the zero vector; the
    # shader must fall back to the geometric normal there rather than
    # normalising a zero and producing NaNs.
    tan_ok = (vtangent[:, :3].norm(dim=1) > 0.5)
    print(f"aux textures {AUX.shape[0]} @{TA}; tangents on "
          f"{float(tan_ok.float().mean()):.1%} of verts; "
          f"normal maps on {int(MAT_HAS_NRM.sum())}/{len(MAT_NRM1)} "
          f"materials", flush=True)
    if VMAT:
        if "mat_ext" not in data:
            raise SystemExit("the Source 2 vmat feature set needs a "
                             "fast_pack_shaders.py pack (mat_ext missing)")
        MAT_H1 = data["mat_h1"].to(device).long()
        MAT_H2 = data["mat_h2"].to(device).long()
        MAT_AOV = data["mat_aov"].to(device).long()
        MAT_OVL = data["mat_ovl"].to(device).long()
        MAT_TM = data["mat_tm"].to(device).long()
        MAT_TR = data["mat_tr"].to(device).long()
        MAT_MET = data["mat_met"].to(device).long()
        MAT_SI = data["mat_si"].to(device).long()
        MAT_EXT = data["mat_ext"].to(device)
        MAT_BTINT = data["mat_btint"].to(device)
        MAT_CTINT = data["mat_ctint"].to(device)
        # Read the column layout the packer wrote rather than hardcoding
        # it, so adding a slot cannot silently shift every index.
        EXT.update({k: i for i, k in enumerate(data["ext_keys"])})
        _n = lambda k: int((MAT_EXT[:, EXT[k]] > 0.5).sum())
        if args.overlay:
            # The four per-layer overlay booleans replaced the single
            # `overlay_tintmask` column when the composite was read out of
            # combo 258. A pack written before that carries neither, and
            # the failure without this check is a KeyError three thousand
            # lines away that reads like a code bug rather than a stale
            # input. Refuse by name instead.
            _miss = [k for k in ("ovl_l1", "ovl_l2", "ovl_mask1",
                                 "ovl_mask2") if k not in EXT]
            if _miss:
                raise SystemExit(
                    f"--overlay needs a pack built with the overlay slot: "
                    f"mat_ext is missing {_miss}. Rebuild with "
                    f"ash_to_world.py --tex-dir <ash>/textures "
                    f"--overlay-pages.")
            _ovl_real = int((MAT_OVL != 3).sum())
            print(f"overlay: {_n('f_overlay')} materials set the feature, "
                  f"{_ovl_real} point at a REAL page "
                  f"(the rest keep the AUX_HALF sentinel, which the "
                  f"composite treats as the exact identity)", flush=True)
        if args.height_blend or args.height_ch1 != "off":
            # ACK, because --height-blend is the second flag in this lane
            # found to be inert without a pack that carries its pages, and
            # a run that silently does nothing is how "measured worse"
            # got recorded for a term that may never have run. has_h1 /
            # has_h2 are set ONLY where a page actually decoded, so these
            # counts are the answer to "can this act", not "was it asked
            # for".
            _h1n, _h2n = _n("has_h1"), _n("has_h2")
            if not _h1n and not _h2n:
                print(f"--height-blend is INERT this run: the pack carries "
                      f"NO decoded height page (has_h1 0, has_h2 0), so the "
                      f"blend runs against the AUX_WHITE sentinel and "
                      f"height_blend()'s `use` gate is 0 everywhere. Build "
                      f"with ash_to_world --tex-dir <ash>/textures "
                      f"--height-pages.", flush=True)
            else:
                print(f"--height-blend: has_h1 on {_h1n} materials, has_h2 "
                      f"on {_h2n}; the layer-2 deviation applies only where "
                      f"has_h2 AND the face carries a second layer",
                      flush=True)
        print(f"vmat feature set: height2 on {_n('has_h2')} materials, "
              f"F_BLEND_EFFECTS_2 {_n('f_blend_effects2')}, "
              f"overlay {_n('f_overlay')}, vmat AO {_n('has_aov')}, "
              f"tintmask {_n('f_tintmask')}, transmissive {_n('has_tr')}, "
              f"backfaces {_n('f_render_backfaces')}, "
              f"metalness {_n('f_metalness_tex')}, "
              f"selfillum {_n('f_self_illum')}", flush=True)
else:
    TA = 1
    AUX = None
    # --matid-dump asks WHICH MATERIAL OWNS THIS PIXEL. That answer is in
    # the pack (`face_matid`) and depends on no PBR term -- but this
    # branch zeroed it, so a dump taken at bare defaults attributed every
    # world pixel to material 0 and named it after whatever sorts first
    # (`ar_baggage/luggage/baggage_luggage_02.vmat` on de_inferno). Not an
    # empty result: a plausible, non-zero, wrong one, which is worse,
    # because a consumer reads a full-looking table off it. Ungating
    # mid_op alone was not enough and the first fix here was exactly that.
    face_matid0 = (data["face_matid"].to(device).long() if args.matid_dump
                   else torch.zeros(len(faces0), dtype=torch.long,
                                    device=device))

# -----------------------------------------------------------------------
# The psrs cull column, ungated.
#
# `MAT_EXT` / `EXT` are bound only inside `if PBR: ... if VMAT:` above, and
# VMAT is a disjunction of eleven SHADING flags -- height-blend, overlay,
# tintmask, transmissive, metalness, self-illum, blend-border, bevel,
# backfaces, vertex-colour-mode, height-ch1. None of them is the psrs stage.
#
# But `f_render_backfaces` is the input the psrs D_FRONT_FACE_CULL term needs
# (SHADER_CALLFLOW_psrs_stage.md sec 4.1: F_RENDER_BACKFACES = 1 forces
# CullMode 0 under BOTH variants of the axis, i.e. cull nothing). So whether
# that term could resolve its column depended on whether an unrelated shading
# flag happened to be on, and at bare defaults it never could -- the GAP fired
# on every run that did not also ask for, say, tint masks.
#
# That is EXISTENCE controlled by a config axis, which charter rule 2 forbids;
# axes select values. The column is a plain read from the world pack, so it is
# read whenever the pack carries it. The shading slot tables above stay where
# they are: those really are per-flag, and binding them here would change what
# the shading path does.
if globals().get("MAT_EXT") is None and "mat_ext" in data \
        and "ext_keys" in data:
    MAT_EXT = data["mat_ext"].to(device)
    EXT.update({k: i for i, k in enumerate(data["ext_keys"])})
    _nbf = (int((MAT_EXT[:, EXT["f_render_backfaces"]] > 0.5).sum())
            if "f_render_backfaces" in EXT else -1)
    print(f"psrs: mat_ext bound from the world pack independently of the "
          f"vmat shading flags ({MAT_EXT.shape[0]} materials, "
          f"{len(EXT)} columns); f_render_backfaces set on {_nbf} "
          f"(-1 = column absent from this pack)", flush=True)
    if _nbf < 0:
        _gt_note("D_FRONT_FACE_CULL: the world pack carries mat_ext but no "
                 "f_render_backfaces column, so CullMode 0 (cull nothing) "
                 "cannot be selected. Rebuild the pack with fast_pack2.py.")

# The REAL per-face material id, always read when the pack carries it.
#
# `face_matid0` is the pack's array only when PBR is on or --matid-dump was
# asked for; otherwise it is a ZERO ARRAY. A zero array is not a missing
# input -- it is a wrong one, and it joins silently: every face resolves to
# material 0 and every per-material column read through it reports that one
# material's value for the whole map. This file already carries the postmortem
# of exactly that, for --matid-dump ("a plausible, non-zero, wrong one, which
# is worse, because a consumer reads a full-looking table off it").
#
# The psrs cull term joins f_render_backfaces through it, so it needs the real
# one. Kept as a SEPARATE name rather than reassigning face_matid0, because
# the zero default is load-bearing for the shading path's cheap-default
# behaviour and this ticket is not the place to change what that does.
MATID_OF_FACE = (data["face_matid"].to(device).long()
                 if "face_matid" in data else None)

# =======================================================================
# THE VERTEX STAGE
# =======================================================================
# Everything below drives cs2_vertex_stage.py, which is the transcription;
# this block is the wiring -- which vertices each axis reaches, where the
# streams come from, and what was substituted when the pack does not carry
# one.  Substitutions are PRINTED, never silent.
#
# The animation is evaluated ONCE PER CHUNK, not once per frame: the
# rasterizer shares one vertex buffer across a chunk of camera poses
# (`clip = einsum("bij,nj->bni", ...)`), so a per-frame vertex set would
# need (B,N,4).  `--lod-per-frame` sets the chunk to 1 and makes the
# animation per-frame; without it the animation time is quantised to the
# chunk, which is stated rather than hidden.
VS_ON = (args.foliage_animation or args.vertex_animation > 0
         or args.cs_vertex_animation or args.prebaked_vertex_animation
         or args.vertex_break_animation
         or args.compressed_normals == "on"
         or args.vertex_stage_selftest
         # csgo_character's skinning is a VERTEX axis; without this the
         # flag would be accepted and vertex_stage() would return the bind
         # pose unchanged, which is the failure mode this file's own
         # comments call "present, resident and unreachable".
         or args.char_skinning)
# --uv-vertex-transform is GONE, not defaulted-off (#76). The uv2 affine
# was implemented TWICE in series -- here per-VERTEX from a global flag, and
# in gt_uv per-MATERIAL from ext2 -- so enabling the secondary stream would
# have applied it twice on all 456 selecting materials. RULED: the PIXEL
# stage owns it. gt_uv reads the transform per MATERIAL, which one global
# 2x4 cannot express, and it carries the inherit semantics. Deleting rather
# than inerting is the standing rule: a flag still wired to a live code path
# is the next person's silent double-apply.
UVSTAGE_ON = (args.secondary_uv != "off"
              or args.foliage_uv_animation)
if args.tools_shading_complexity and not FAMILY:
    # Fail at startup, not after the first frame: the cost is per shader
    # FAMILY and without the side table there is no family per pixel.
    raise SystemExit("--tools-shading-complexity needs --fam-side "
                     "(build it with fast_pack_fam.py)")
VS_NOISE = None
VS_FAMV = None
VS_PIVOT = None
VS_FPARAM = None
VS_COLOR1 = None
VS_PACKED = None
VS_PREBAKE = None
VS_BREAK_CENT = None
VS_BREAK_SEED = None
# csgo_character skinning: BLENDINDICES / BLENDWEIGHT / draw transform base /
# the (m5 >> 16) & 15 influence count, plus the per-vertex Curvature stream
# S_USE_PER_VERTEX_CURVATURE reads.
VS_SKIN_BI = None
VS_SKIN_BW = None
VS_SKIN_BASE = None
VS_SKIN_NINF = None
VS_SKIN_CURV = None
VS_SKIN_PAL = None
VS_SUBST = []
