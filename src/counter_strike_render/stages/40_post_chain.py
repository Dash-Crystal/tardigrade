

def _post_chain(albedo, normal, fg, shadow=None, irradiance=None,
                amb_map=None, view=None, rough=None, metal=None,
                ao=None, emissive=None, pbr_mask=None,
                indirect=None, lm_ok=None, vol=None, fogin=None,
                transmissive=None, two_sided=None, probe=None, unlit=None,
                skyv=None, ssao=None,
                sh_irr=None, sh_dir=None, sh_spec=None, alights=None,
                dirocc=None, ss_shadow=None, wet_f0=None, wet_scatter=None,
                gt=None, alights_back=None, fam=None,
                # MERGE GAP, stated rather than silently defaulted:
                # these two are accepted and NEVER PASSED. The branch that
                # added them was terminated mid-write before it wired the
                # call site, so csgo_projected_decals and spritecard reach
                # this function as None on every invocation. That is the
                # "present, resident and unreachable" shape catalogued in
                # CHECKS_THAT_CANNOT_FAIL.md - a parameter that exists and
                # is never supplied reads as implemented. Wiring it needs
                # the branch author's call-site work, not a guess here.
                decal_layer=None, sprite_layer=None):
    B = albedo.shape[0]
    if args.gbuffer:
        up_g = normal[..., 1].clamp(-1, 1) * 0.5 + 0.5
        ndl_g = (normal @ sun_dir).abs().clamp(0, 1)
        GBUF.append((albedo[..., :3].clamp(min=0) ** 2.2, normal, fg,
                     shadow if shadow is not None else
                     (irradiance if irradiance is not None else
                      torch.ones_like(fg, dtype=torch.float32))))
    # one-sided lambert: .abs() lit back-faces and destroyed the sun signal
    # CONSUMER-SIDE PROBE. The producer's tilt says the normal was bent;
    # this says what the shading actually received. Two runs at different
    # --normal-strength whose producer tilt differs and whose
    # compose.normal_in does NOT are a drop between the two, and the
    # bisection is then over the code between them rather than over
    # hypotheses.
    _gt_value("compose.normal_in", normal[..., 1], identity=0.0,
              note="the up-component of the normal AS RECEIVED by the "
                   "shading. Pairs with surface.normal_map_tilt_deg: that "
                   "is what the producer made, this is what arrived")
    ndl_raw = normal @ sun_dir
    ndl = ndl_raw.clamp(0, 1)
    if two_sided is not None:
        # F_RENDER_BACKFACES only: a two-sided surface is lit from either
        # side. Applied per-material rather than globally, because doing
        # it globally is exactly the .abs() that destroyed the sun signal.
        # Carries its own coverage (it spans the opaque AND mask classes),
        # so it must NOT be gated by pbr_mask.
        ndl = ndl * (1 - two_sided) + ndl_raw.abs().clamp(0, 1) * two_sided
    # GATE#5 -- THE FITTED HEMISPHERE IS DELETED. The seed is ZERO.
    #
    # There was never a reference term here to re-derive. g_vSunAmbient,
    # the ONLY constant ambient anywhere in the reference's sun path
    # (csgo_environment_ps.glsl:923, csgo_complex_ps.glsl:641), READS as
    # (0, 0, 0) in 717 of 718 block copies across two captures
    # (READ from the capture: 717 of 718 block copies, two captures). AMBIENT_SKY/AMBIENT_GROUND were a
    # stand-in for the BAKED subsystem, which is a different expression
    # reached below -- probe cubes, the irradiance volume, the lightmap.
    #
    # Gate#4 priced the fit at 148.5x (from 91.8x) with the legs
    # DISAGREEING -- level blown out, palette improved -- which is light
    # counted twice: the read sun landed on top of an ambient fitted
    # against a zero sun, exactly as W2 pre-registered when landing the
    # depth-sign fix. Keeping the fit to protect a number is the
    # anti-pattern; the constant dies and what remains is measured.
    #
    # So pixels the baked path does NOT cover now receive no ambient at
    # all, which is what the reference does. If the frame comes back
    # darker than GT, that is not a regression to be tuned away -- it
    # MEASURES what the baked path still underprovides, and the probes
    # print exactly that below.
    ambient = torch.zeros(*normal.shape[:-1], 3, device=normal.device,
                          dtype=torch.float32)
    _gt_value("ambient.seed_after_hemisphere_deletion", ambient, 0.0,
              note="the fitted hemisphere is GONE; everything nonzero "
                   "from here is the baked path. g_vSunAmbient READS "
                   "(0,0,0), so a zero seed is the reference's value, "
                   "not an ablation.")
    # Texture PNGs from VRF are sRGB-encoded. Lighting must happen in
    # LINEAR space: decode, light, tonemap, then re-encode. Skipping the
    # decode while still encoding at the end applies gamma twice and
    # washes the whole frame toward white.
    if args.albedo_only:
        rgb = albedo[..., :3].flip(1)
        sky_here = (torch.zeros_like(sky_row) if args.fam_only is not None
                    else sky_row)
        rgb = torch.where((~fg.flip(1)).unsqueeze(-1),
                          sky_here.expand(B, H, W, 3), rgb)
        if args.supersample > 1:
            rgb = torch.nn.functional.avg_pool2d(
                rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        return (rgb.clamp(0, 1) * 255).to(torch.uint8)
    linear = albedo[..., :3].clamp(min=0) ** 2.2
    if wet_scatter is not None:
        # glsl:1343-1347. The wet block's `_17129` turns the sun's N.L into
        # a wrapped, back-lit lobe -- a thin water film scatters. Applied
        # to the SAME ndl the shadow then gates, exactly as the reference
        # orders it (the remap is at :1343, the shadow product at :1361).
        _s = wet_scatter.clamp(0, 1)
        _nl2 = (normal @ sun_dir)
        _wrapped = (((0.5 + ndl * 0.5) + (1.0 - _nl2.clamp(0, 1)) ** 4)
                    * ((_nl2 + 0.2) * 4.0).clamp(0, 1)) \
            * (1.0 + (_nl2 - 1.0) * (1.0 - _s)).clamp(0, 1)
        ndl = torch.where(_s > 0, ndl + (_wrapped - ndl) * _s, ndl)
    # Sun visibility is collected rather than multiplied in place, because
    # the engine per-view offset-88 target is min()-combined into it
    # (glsl:1322: min(cascadeShadow, RT88.z)), not multiplied. With no
    # offset-88 target this is the same product, in the same order, as
    # before.
    sunvis = None
    if shadow is not None:
        sunvis = shadow
    if irradiance is not None:
        # Baked visibility gates the sun term; analytic ambient+lambert
        # cannot express contact shadows or occlusion, and this is the
        # asset CS2 itself shades with.
        vis = 1.0 - irradiance if args.sun_vis == "invert" else irradiance
        if lm_ok is not None:
            # unlightmapped props have no baked mask; leave them fully lit
            vis = torch.where(lm_ok > LM_OK_T, vis, torch.ones_like(vis))
        sunvis = vis if sunvis is None else sunvis * vis
    if ss_shadow is not None:
        sunvis = ss_shadow if sunvis is None \
            else torch.minimum(sunvis, ss_shadow)
    if sunvis is not None:
        ndl = ndl * sunvis
    amb0 = ambient

    def _atrace(where_):
        """Checksum of `ambient` at a labelled point in the chain.

        THE BOUNDED BISECT, made permanent. The probe cube was measured
        reaching sample_probes() and NOT reaching the frame, and finding
        which of the eight rewrites of `ambient` below eats it is a
        two-probe bracket -- provided the chain can be asked what it holds
        at each step. It could not. This is that instrument, and it stays:
        the same question will be asked of the next term.
        """
        if not args.probe_trace:
            return
        a = ambient
        v = float(a.double().sum()) if torch.is_tensor(a) else float(a)
        print(f"ambient@{where_}: sum {v:.6f} mean "
              f"{float(a.double().mean()) if torch.is_tensor(a) else v:.6f}",
              flush=True)

    _atrace("00_before_probe")
    if probe is not None:
        # The map's own baked ambient cubes. This is the term the single
        # hemispheric AMBIENT_SKY/GROUND pair was standing in for -- and
        # the measured field separates them (slot +Z is strongly blue,
        # slot -Z is warm), which the one-slot stand-in cannot.
        pcube, phit = probe
        pa = probe_ambient(pcube, normal) * args.probe_gain
        if args.probe_mode == "replace":
            # generic_r69m3.glsl:1114-1120, through passes/lighting.py --
            # the reference's front-to-back composite, run for the volumes
            # we have.
            #
            # THE GAP THIS USED TO DECLARE IS CLOSED, and the way it was
            # worded is worth keeping as a warning. It said probe_field.npz
            # carries "no per-volume boxes, no inverse fade distances, no
            # overlap list" -- true of the TABLE as built, and read for a
            # year as a fact about the map. Both were always in the entity
            # lump (edge_fade_dists, box_mins/box_maxs, angles); nobody had
            # asked it for them. sample_probes() now walks the volumes the
            # way the reference does and hands this site the already-
            # composited cube, so the single-volume step below is a
            # one-step composite over an accumulator that is already done.
            _z = torch.zeros_like(pa[..., 0])
            _w = torch.where(phit, torch.ones_like(_z), _z)
            _rgb, _alpha, _done = _lighting.probe_composite_step(
                torch.zeros_like(pa), _z, pa, 1.0, _w)
            _gt_value("probe.composite_alpha", _alpha, 0.0,
                      note="generic:1115. With one volume this is the "
                           "six-face weight through smoothstep; the "
                           "0.99 early-out at :1120 cannot fire on a "
                           "single volume, and will once per-volume "
                           "boxes are threaded.")
            new_amb = _rgb
        else:
            # exposure-neutral: keep the fitted ambient's LEVEL, take only
            # the field's spatial ratio, same convention as --irr-mode ratio
            #
            # AND IT MULTIPLIES BY ZERO WHENEVER THAT LEVEL IS GONE. This
            # mode was written against the fitted hemispheric ambient; GATE#5
            # then DELETED that stand-in, correctly, because it was a fit --
            # and every ratio-mode consumer silently became a no-op, because
            # `amb0 * k` with amb0 == 0 is 0 for any k. The probe field was
            # loaded, arbitrated, sampled and thrown away by a multiply, on
            # every run, with no error and no visible difference. That is why
            # the term measured inert and why two five-frame arms scored it
            # against the reconstruction and found nothing: neither arm had a
            # probe in it.
            #
            # Not silently any more. The renderer says it, once, naming the
            # mode that does reach the image.
            k = (pa.mean(-1, keepdim=True) / PROBE_REF)
            new_amb = amb0 * k.clamp(0, args.probe_clamp)
            if not _PZERO and float(amb0.abs().sum()) == 0.0:
                _PZERO.append(1)
                print("probe field: INERT BY CONSTRUCTION -- --probe-mode "
                      "ratio scales the base ambient, and the base ambient "
                      "is identically zero here (the fitted hemisphere is "
                      "deleted, GATE#5), so the field contributes exactly "
                      "nothing however good it is. Use --probe-mode replace, "
                      "which is the reference's own composite and the only "
                      "mode that carries an absolute radiance.", flush=True)
        ambient = torch.where(phit.unsqueeze(-1), new_amb, ambient)
        amb0 = ambient
    _atrace("01_after_probe")
    if vol is not None:
        # Every pixel, props included. Exposure-neutral by construction:
        # the ratio to the field's own reference level is the only thing
        # taken from it, so this cannot act as a hidden global gain.
        kv = (vol.mean(-1, keepdim=True) / VOL_REF * args.irr_vol_gain)
        ambient = amb0 * kv.clamp(0, args.irr_clamp)
    _atrace("02_after_vol")
    if indirect is not None:
        # CS2's lighting.DiffuseIndirect: the baked bounce irradiance
        # sampled at the lightmap UV. Its correlation with the surface
        # normal's up component is +0.02 -- i.e. none -- which is what an
        # SH L0 / DC term looks like, so treating it as a normal-
        # independent ambient is right and the directional_irradiance
        # companion carries the L1 lobe we are not using.
        if args.irr_mode == "replace":
            new = indirect * args.irr_gain
        else:
            # exposure-neutral: the fitted hemispheric ambient already
            # absorbed the MEAN bounce level, so only the spatial ratio is
            # new information. Replacing outright double-counts the fit.
            k = (indirect.mean(-1, keepdim=True) / IRR_REF)
            new = amb0 * (k * args.irr_gain).clamp(0, args.irr_clamp)
        ambient = torch.where((lm_ok > LM_OK_T).unsqueeze(-1), new, ambient)
    _atrace("03_after_indirect")
    if sh_irr is not None:
        # CS2's own baked environment probes. The fitted hemisphere is a
        # 2-colour lerp on normal.y; this is the real SH-L2 lobe, and the
        # earlier finding that the lightmap indirect correlates +0.02 with
        # normal.y is exactly the statement that the lightmap carries only
        # the DC term and the directional part was missing.
        if args.ibl_sh_mode == "replace":
            ambient = sh_irr
        elif args.ibl_sh_mode == "modulate":
            # NOTE this DISCARDS the lightmap indirect above rather than
            # composing with it, so it is an A/B of two spatial sources.
            k = sh_irr.mean(-1, keepdim=True) / SH_REF
            ambient = amb0 * k.clamp(0, args.irr_clamp)
        else:
            # `dir`: divide the probe's irradiance by its OWN DC term, so
            # what is left has mean 1 over directions per channel and
            # carries no spatial level and no exposure at all -- purely
            # the L1/L2 lobe, multiplied onto whatever the lightmap and
            # the irradiance volume already decided.
            ambient = ambient * sh_dir.clamp(0, args.irr_clamp)
    _atrace("04_after_sh")
    if amb_map is not None:
        # CS2's baked bounce/ambient irradiance replaces the hemispheric
        # stand-in; analytic ambient cannot express interior falloff,
        # which is what "missing_light" was measuring.
        ambient = ambient * (1 - AMB_MIX) + amb_map * AMB_GAIN * AMB_MIX
    _atrace("05_after_ambmap")
    if ao is not None:
        # AO occludes the sky/bounce term only. Multiplying the sun by it
        # too would double-count the baked sun-visibility mask.
        aof = ao.unsqueeze(-1)
        if pbr_mask is not None:
            aof = 1.0 - (1.0 - aof) * pbr_mask.unsqueeze(-1)
        # At strength 1: --ao-strength was the fitted knob on the deleted
        # stand-in, and `ao` here is now the material PAGE, which enters the
        # reference expression unscaled. The term is KEPT -- the page reaches
        # this composition too, on --gt-lighting 0 -- only the fit is gone.
        ambient = ambient * aof
    _atrace("06_after_ao")
    if ssao is not None:
        # CS2 applies SSAO to the INDIRECT term only (it is an ambient
        # obscurance, not a shadow) -- the sun already carries
        # direct_light_shadows. Same convention as the existing --ao.
        ambient = ambient * (1.0 - args.ssao_strength * (1.0 - ssao.unsqueeze(-1)))
        if args.ssao_sun > 0:
            ndl = ndl * ssao.clamp(0, 1) ** args.ssao_sun
    _atrace("07_after_ssao")
    if dirocc is not None:
        # glsl:1674  _20418 = _21714 * max(_12727, 0)
        #      :1675  _20015 = _8745 * _20418          (ambient diffuse)
        #      :1691  _15754 = ... + _13147 * _20418
        #      :1696  _15734 = ... IBL specular ... * _20418
        # i.e. the offset-76 resolve scales the WHOLE indirect term, both
        # halves. `ambient` is this renderer's carrier for both -- the IBL
        # specular below is `ambient * sh_spec * fe` -- so one multiply
        # here reproduces all three sites.
        ambient = ambient * dirocc.unsqueeze(-1)
    if args.debug_lighting == "dirocc":
        _d = (dirocc if dirocc is not None
              else torch.ones_like(fg, dtype=torch.float32))
        rgb = _d.unsqueeze(-1).expand(B, H, W, 3).flip(1).clone()
        rgb = torch.where((~fg.flip(1)).unsqueeze(-1),
                          torch.zeros_like(rgb), rgb)
        if args.supersample > 1:
            rgb = torch.nn.functional.avg_pool2d(
                rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        return (rgb.clamp(0, 1) * 255).to(torch.uint8)
    if args.debug_lighting == "ss-shadow":
        _d = (ss_shadow if ss_shadow is not None
              else torch.ones_like(fg, dtype=torch.float32))
        rgb = _d.unsqueeze(-1).expand(B, H, W, 3).flip(1).clone()
        rgb = torch.where((~fg.flip(1)).unsqueeze(-1),
                          torch.zeros_like(rgb), rgb)
        if args.supersample > 1:
            rgb = torch.nn.functional.avg_pool2d(
                rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        return (rgb.clamp(0, 1) * 255).to(torch.uint8)
    if args.debug_lighting == "ssao" and ssao is not None:
        rgb = ssao.unsqueeze(-1).expand(B, H, W, 3).flip(1).clone()
        rgb = torch.where((~fg.flip(1)).unsqueeze(-1),
                          torch.zeros_like(rgb), rgb)
        if args.supersample > 1:
            rgb = torch.nn.functional.avg_pool2d(
                rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        return (rgb.clamp(0, 1) * 255).to(torch.uint8)
    if skyv is None:
        # STANDALONE bounce chroma: no sky-visibility field involved. This
        # is exactly the --skyvis-frac 0.0 case, which is what the shipped
        # config runs, without the 29 MB field it never samples.
        _bc = _bounce_chroma(vol)
        if _bc is not None:
            ambient = ambient * _bc
    if skyv is not None:
        # Sky/ambient visibility. The ambient slot is a single FITTED
        # hemispheric term that absorbed sky light AND bounce together
        # (measured: our AMBIENT_SKY is (1.00,0.86,0.93), not the
        # (1.00,1.07,1.38) blue of a real sky probe -- it is a blend).
        # Enclosure blocks the SKY half outright; it does not block the
        # bounce half, and GT's interior floor is 0.172, not 0. So only
        # --skyvis-frac of the slot is gated.
        sv = skyv
        if args.skyvis_cos:
            sv = sv * up
        if args.skyvis_renorm:
            sv = sv / SKYVIS_REF
        sv = sv.clamp(0, args.skyvis_clamp).unsqueeze(-1)
        if args.skyvis_noocc:
            sv = torch.ones_like(sv)
        f = args.skyvis_frac
        bounce = _bounce_chroma(vol)
        if bounce is None:
            bounce = torch.ones_like(sv)
        skyc = torch.ones_like(sv)
        if args.skyvis_sky_chroma > 0:
            _sc = torch.tensor([1.00, 1.07, 1.38], device=sv.device)
            _sc = (_sc / _sc.mean()).view(1, 1, 1, 3)
            skyc = 1.0 + args.skyvis_sky_chroma * (_sc - 1.0)
        ambient = ambient * ((1.0 - f) * bounce + f * sv * skyc)
        if args.skyvis_sun > 0:
            ndl = ndl * skyv.clamp(0, 1) ** args.skyvis_sun
    if args.debug_lighting == "skyvis" and skyv is not None:
        rgb = skyv.unsqueeze(-1).expand(B, H, W, 3).clone()
        rgb = rgb.flip(1)
        fg_f0 = fg.flip(1)
        rgb = torch.where((~fg_f0).unsqueeze(-1),
                          torch.zeros_like(rgb), rgb)
        if args.supersample > 1:
            rgb = torch.nn.functional.avg_pool2d(
                rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        return (rgb.clamp(0, 1) * 255).to(torch.uint8)
    if args.light_dump and len(LDUMP) < 1:
        # DIAGNOSTIC: the exact three factors of the diffuse product, so
        # the chroma of what ARRIVES at a pixel can be separated from the
        # chroma of the surface it lands on.
        _k = min(32, linear.shape[0])
        LDUMP.append((
            linear[:_k].half().cpu().numpy(),
            ambient[:_k].expand(_k, *linear.shape[1:]).half().cpu().numpy()
            if ambient.shape[0] == 1 else ambient[:_k].half().cpu().numpy(),
            (SUN_COLOR.view(1, 1, 1, 3)
             * ndl[:_k].unsqueeze(-1)).half().cpu().numpy(),
            fg[:_k].cpu().numpy()))
    # The replacement, probed in the SAME commit as the deletion -- W2's
    # rule, and the cheapest guard against swapping one silent constant
    # for another. `identity` is 0.0 because ambient is ADDED, so the
    # fraction-at-identity IS the fraction of pixels the baked path fails
    # to cover. That number is the finding, not a diagnostic: it is the
    # size of the hole the fitted hemisphere was filling.
    _gt_value("ambient.baked_coverage", ambient, 0.0,
              note="fraction-at-identity = the share of shaded pixels no "
                   "baked source reached. Before gate#5 the fitted "
                   "hemisphere hid this at exactly 0.000 by construction.")
    if alights is not None:
        # EXPOSURE-NEUTRAL give-back. AMBIENT_SKY/GROUND were FITTED on a
        # render carrying no analytic lights, so the fit has already
        # absorbed their average: adding them raw double-counts by
        # construction and any resulting brightening is the fit's, not
        # the term's. Handing the frame-mean added irradiance back off
        # the ambient leaves only the SPATIAL STRUCTURE, which is the
        # only thing the term claims to supply.
        #
        # Taken over the FOREGROUND pixels of EACH FRAME. Sky pixels
        # receive no analytic light, so including them would shrink the
        # give-back by the sky fraction and silently re-introduce a
        # global brightening that varies with how much sky a frame
        # happens to see. Per FRAME rather than per chunk because the
        # chunk size is a memory decision -- averaging across it would
        # make an exposure-neutral arm depend on --batch, which is
        # exactly the kind of invisible coupling that makes two arms
        # incomparable without either of them looking wrong.
        if args.lights_neutral:
            _fw = fg.float().unsqueeze(-1)
            _den = _fw.sum((1, 2), keepdim=True).clamp(min=1.0)
            _mu = (alights * _fw).sum((1, 2), keepdim=True) / _den
            ambient = (ambient - _mu).clamp(min=0.0)
    if args.debug_lighting == "lights":
        _a = (alights if alights is not None
              else torch.zeros(B, H, W, 3, device=albedo.device))
        rgb = _a.flip(1)
        rgb = torch.where((~fg.flip(1)).unsqueeze(-1),
                          torch.zeros_like(rgb), rgb)
        if args.supersample > 1:
            rgb = torch.nn.functional.avg_pool2d(
                rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
        return (rgb.clamp(0, 1) * 255).to(torch.uint8)
    gt_on = bool(args.gt_lighting) and gt is not None
    # ACK PRINT, once, because a flag that acts must say so and a flag
    # that CANNOT act must say that louder. --gt-lighting defaults to 1
    # and is documented as "shade through the ported CS2 combo axes", but
    # its second conjunct is a per-frame INPUT: `gt` is the gt_pack the
    # caller builds, and a run that never builds one shades through the
    # fitted composition while the flag still reads 1. That is the same
    # shape as --rough-src reading a constant alpha and --ibl-cube's
    # undeclared probe-grid precondition, both found in this lane.
    if not _GT_LIGHTING_ACK:
        _GT_LIGHTING_ACK.append(1)
        if args.gt_lighting and gt is None:
            print("--gt-lighting 1 is INERT this run: the ported CS2 combo "
                  "path needs a gt_pack and the caller supplied none, so "
                  "shading falls to the pre-existing fitted composition. "
                  "The flag reads 1 and does not act.", flush=True)
        else:
            print(f"--gt-lighting {args.gt_lighting}: ported CS2 combo path "
                  f"{'ACTIVE' if gt_on else 'off by flag'}", flush=True)
    if gt_on:
        # ---- GROUND-TRUTH COMBO PATH -------------------------------
        # The ported call flow supplies its OWN baked-lighting term, its
        # own sun attenuation, its own direct BRDF and its own specular
        # cubemap, so the fitted `ambient` and the fitted `SUN * ndl`
        # above are not composed on top of it -- that would double-count
        # every one of them. The fitted hemisphere is still reachable
        # through this path: it IS gt_ambient_sh()'s default
        # coefficients, exactly (see the derivation there).
        q = args.shader_quality
        _mode = gt_select_baked(lm_ok)
        b_irr, b_occ = gt_baked(_mode, gt["wpos"], normal,
                                gt.get("lm_u"), gt.get("lm_v"), lm_ok,
                                gt.get("vlit"), q)
        # ---- offset 76: the indirect-term occlusion -------------------
        # csgo_complex_ps.glsl:523 (= env_blend :1239). THIS IS THE
        # AO TERM, and it is not --ssao: the reference resolves a 4-channel
        # low-res directional-occlusion target against the engine ambient
        # basis, with a nearest-depth bilateral selector from the offset-80
        # depth target. dirocc_resolve() is that resolve, instruction for
        # instruction, and it is already computed by the caller -- so this
        # CONSUMES it rather than building a second one.
        #
        # --ssao is the pre-existing fitted stand-in for the same slot.
        # When the real target is present it REPLACES it; keeping both
        # would apply two occlusions to one term.
        #
        # --ssao IS KEPT, AND THE MEASUREMENT IS WHY -- do not delete it by
        # analogy with --ao. Its sibling went as a validated-replacement
        # stand-in (d92f8bac), and the obvious next step was to take this
        # one with it. The paired arms say no. Same form as --ao's, f700 /
        # de_inferno_nrm.pt, md5 of the rendered PNG:
        #
        #   --dirocc 0       without --ssao  b9b5017365ec1bd54e588557304b2868
        #   --dirocc 0       with    --ssao  535f5a96f8483423368ee30f25d387c5
        #   --gt-lighting 0  without --ssao  2d6abd43873829c6657b5b1f560f4ec7
        #   --gt-lighting 0  with    --ssao  fef15b3787048420c30c23579ce358a9
        #
        # DIFFERENT on both branches, where --ao was byte-identical on
        # both. The reason is structural rather than lucky: --ao SAMPLED a
        # pack page that turned out to be a white sentinel, so it
        # multiplied by 1.0 wherever it ran; --ssao COMPUTES a screen-space
        # term from the depth buffer, so it has real output whatever the
        # pack carries. A stand-in that computes is not the same kind of
        # object as a stand-in that reads a constant, and only the second
        # is safe to delete on a replacement argument.
        #
        # What would retire it is the offset-76 target covering its role on
        # a run where --dirocc is OFF, which is not something these arms
        # tested and not something the deletion rule can assume.
        if dirocc is not None:
            gao = dirocc.clamp(min=0)
            _gt_note("AO: offset-76 directional-occlusion target in use "
                     "(--dirocc); --ssao is the fitted stand-in for this "
                     "same slot and is NOT additionally applied (--ao, the "
                     "other stand-in, is DELETED)")
            # ⚠️ THE MATERIAL AO PAGE IS NOT ONE OF THOSE STAND-INS, and
            # replacing gao wholesale DISCARDED it -- the page loaded,
            # sample_pages ran, the multiply landed in the indirect, and
            # this line threw the result away. corpus-2 caught it as
            # byte-identical renders at 21.6% page coverage (4fa3fb86).
            #
            # READ from csgo_environment_ps.glsl:1229 --
            #     vec3 _20418 = vec3(_21714 * _17119).xyz;
            #     vec3 _20015 = _8745 * _20418;      // ambient diffuse
            # the scale on the whole indirect is a PRODUCT of two factors,
            # not one slot with alternatives:
            #   _21714  :793  a screen-space weighted fetch -- the
            #                 offset-76 resolve; :797 sets it to 1.0 when
            #                 that target is absent, which is exactly the
            #                 slot --dirocc/--ssao fill.
            #   _17119  :732/:742  mix(_14713, 1.0, ...) / _14713, and
            #                 _14713 (:514) is the MATERIAL occlusion.
            #
            # So screen-space occlusion and material occlusion MULTIPLY.
            # --ssao and --dirocc really are alternatives for _21714 and
            # the note above is right about them; the AO page is _17119
            # and belongs on the other side of the product. W3's
            # compose.ao-vs-AO-page distinction is this pair, and feeding
            # one into the other's slot is the mix-up this branch made.
            #
            # STATED DIFFERENCE in the SOURCE, not the role: at :514 the
            # reference's material occlusion comes from the albedo texel's
            # ALPHA (_22057.w); ours comes from mat_ao_pages
            # (g_tAmbientOcclusion). Same position in the product, same
            # role, different channel -- because that is what our pack
            # carries.
            #
            # Gated on AO_PAGES, not on `ao is not None`, because `ao`
            # carries two different things: the material page when the
            # pack ships one, and the fitted --ao aux stand-in otherwise.
            # The stand-in IS a _21714 substitute and must keep not
            # double-applying.
            if ao is not None and AO_PAGES is not None \
                    and args.ao_page_product != "off":
                _aop = ao.clamp(min=0)
                if args.ao_page_product == "zero":
                    _aop = torch.zeros_like(_aop)
                gao = gao * _aop
                _gt_note("AO: the material AO page MULTIPLIES the "
                         "offset-76 resolve (glsl:1229 _21714 * _17119); "
                         "it is not a stand-in for that slot and is no "
                         "longer discarded by --dirocc")
                frame_print(f"[ao-page-product] "
                      f"ARM={args.ao_page_product} -- the "
                      f"material AO page "
                      f"{'MULTIPLIES' if args.ao_page_product == 'on' else 'is FORCED TO ZERO in'}"
                      f" the offset-76 resolve; page mean "
                      f"{float(ao.mean()):.6f} min {float(ao.min()):.6f}, "
                      f"gao after mean {float(gao.mean()):.6f}", flush=True)
                _gt_value("compose.ao_page_in_product", _aop,
                          identity=1.0,
                          note="the material AO factor as it ENTERS gao "
                               "-- glsl:1229's _17119 side. compose.ao "
                               "below is the RESULT of the product; these "
                               "are two different terms and W3 "
                               "established the distinction")
            elif ao is not None and AO_PAGES is not None:
                frame_print("[ao-page-product] ARM=off -- the material AO "
                      "page is "
                      "DISCARDED by --dirocc, reproducing the pre-fix "
                      "behaviour. This arm exists to be compared against; "
                      "it is not the reference composition.", flush=True)
                _gt_note("AO: --ao-page-product off, so the material page "
                         "is discarded as it was before the glsl:1229 "
                         "read. DIAGNOSTIC ONLY.")
            elif ao is not None:
                _gt_note("AO: `ao` is the fitted --ao aux stand-in, not a "
                         "material page (the pack ships no mat_ao_pages), "
                         "so it stays OUT of the product -- it substitutes "
                         "for the same screen-space slot --dirocc filled")
        else:
            # :933 -- ssao * aoTex.x; at q=1 a second screen-space AO fetch
            # multiplies in (:561). Absent terms are 1, the shader's value
            # for an unbound descriptor.
            gao = torch.ones_like(fg, dtype=torch.float32)
            if ssao is not None:
                gao = gao * ssao.clamp(min=0)
            if ao is not None:
                gao = gao * ao.clamp(min=0)
            _gt_note("AO: --dirocc off, so the offset-76 target is absent "
                     "and the fitted --ssao/--ao stand-in is carrying the "
                     "slot")
        gsh = gt_csm_shadow(gt["wpos"], gt.get("eye"), q)
        # ---- offset 88: screen-space shadow ---------------------------
        # csgo_complex_ps.glsl:606, and PER_VIEW_RENDER_TARGETS.md: read by
        # ALL SIX shaders, always channel .z, always at
        # gl_FragCoord.xy * invViewport, always MIN()-combined into the
        # cascade shadow -- never multiplied. gt_csm_shadow() previously
        # documented this and skipped it; the target now exists, so the
        # min() is applied here rather than left as a recorded gap.
        if ss_shadow is not None:
            gsh = torch.minimum(gsh, ss_shadow)
        else:
            _gt_note("csm: --ss-shadow off, so the offset-88 target is "
                     "absent; min(cascade, ss.z) is skipped, which is the "
                     "shader's g_bScreenSpaceShadows=0 branch")
        if shadow is not None:
            # the pre-existing single-map --shadow, if the caller also
            # asked for it, folded the same way -- min(), not multiply.
            gsh = torch.minimum(gsh, shadow)
        rgb = gt_compose(linear, normal, view if view is not None
                         else _nrm(normal), gt["wpos"], gt.get("eye"),
                         rough if rough is not None
                         else torch.full_like(fg, args.roughness,
                                              dtype=torch.float32),
                         metal if metal is not None
                         else torch.zeros_like(fg, dtype=torch.float32),
                         q, b_irr, b_occ, gao, gsh,
                         alights_diff=(linear * alights
                                       if alights is not None else None))
        ambient = b_irr
    else:
        rgb = linear * (ambient
                        + SUN_COLOR.view(1, 1, 1, 3) * ndl.unsqueeze(-1))
        if alights is not None:
            # Diffuse only. A specular lobe on 184 point lights is a
            # separate term and adding it silently would confound this
            # measurement.
            rgb = rgb + linear * alights
    # --- csgo_lightmappedgeneric / generic.vfx / csgo_imported --------
    # Applied HERE, after either branch, so it is live at the file's
    # defaults and not only under --gt-lighting (which still defaults to
    # 0 and says so itself). The three families' own composite REPLACES
    # the composite above for their pixels only; every other family's
    # pixels are returned untouched by the torch.where.
    if LMG_LIVE and fam is not None:
        _q = args.shader_quality
        # environment_ps:918/923 via passes/lighting.py -- the same code
        # object the conformance registry scores, so the term is landed
        # rather than transcribed-and-parallel. g_vSunAmbient READS
        # (0,0,0), so the first addend is zero and the not-lit branch is
        # this expression at atten = 0.
        # :902 -- the sun diffuse SHAPE on thin surfaces, through
        # passes/lighting.py. `t` is _13960 from :735,
        # `_10839 * (0.2 + 0.15*w^3)` with w the normal map's alpha; this
        # renderer does not carry that channel separately, so t is ZERO
        # and the term reduces to plain NdotL by its own algebra --
        # mix(ndl, wrapped, 0) == ndl. The function is in the path and its
        # INPUT is missing, which is a different state from the function
        # being absent, and the probe below says which.
        _t_scatter = torch.zeros_like(ndl)
        _bent = normal
        ndl = _lighting.sun_wrap_scatter(
            ndl, (normal * sun_dir.view(1, 1, 1, 3)).sum(-1),
            normal, _bent, sun_dir.view(1, 1, 1, 3), _t_scatter)
        _gt_value("sun.wrap_scatter", ndl, None,
                  note="environment_ps:902. t (the normal-map alpha "
                       "translucency, :735) is not carried by this "
                       "renderer, so t=0 and this reduces to NdotL "
                       "exactly. MISSING INPUT, not an absent term -- when "
                       "the alpha channel is threaded the half-lambert, "
                       "the pow(1-B,4) rim and the 10x bent-normal "
                       "extrapolation all become live.")
        _direct = _lighting.sun_diffuse(
            SUN_AMBIENT.view(1, 1, 1, 3),
            ndl.unsqueeze(-1), SUN_COLOR.view(1, 1, 1, 3),
            torch.ones_like(ndl))
        if alights is not None:
            _direct = _direct + alights          # r0_m3:403-640
        # --- D_BAKED_LIGHTING_FROM_*, --lmg-baked -------------------
        # Four shipped modules of static combo 0, four sources:
        #   dyn 4 LIGHTMAP       r0_m3:240  the sampler2DArray fetch
        #   dyn 1 VERTEX_STREAM  r0_m1      the COLOR1 attribute
        #   dyn 2 PROBE          r0_m2      the probe volume
        #   dyn 0 (none)         r0_m0:635  dot(ambientSH[i], vec4(N,1))
        # 'none' is one of the four, NOT an off switch -- the r0_m0
        # module evaluates a real L1 ambient basis against the normal.
        _lm = b_irr if gt_on else indirect
        _sh = ambient                                     # r0_m0:635
        _mode = args.lmg_baked
        if _mode == "lightmap":
            _baked = _lm if _lm is not None else _sh
            if _lm is None:
                _lmg_note("--lmg-baked lightmap but no lightmap irradiance "
                          "(--irr-npy / --gt-lighting); the shipped dyn-0 "
                          "ambient-SH module runs instead")
        elif _mode == "vertex-stream":
            _vs = gt.get("vlit") if gt is not None else None
            _baked = _vs if _vs is not None else _sh
            if _vs is None:
                _lmg_note("--lmg-baked vertex-stream but no COLOR1 stream; "
                          "the shipped dyn-0 ambient-SH module runs instead")
        elif _mode == "probe":
            _baked = probe_ambient(*probe) * args.probe_gain \
                if probe is not None else _sh
            if probe is None:
                _lmg_note("--lmg-baked probe but no probe field; the "
                          "shipped dyn-0 ambient-SH module runs instead")
        elif _mode == "none":
            _baked = _sh
        else:                                             # auto
            _baked = _sh if _lm is None else _lm
            if _lm is not None and lm_ok is not None:
                _baked = torch.where((lm_ok > LM_OK_T).unsqueeze(-1),
                                     _lm, _sh)
        rgb, _a2, _cnt = lmg_family_shade(
            rgb,
            (albedo[..., 3] if albedo.shape[-1] > 3
             else torch.ones_like(fg, dtype=torch.float32)),
            linear, fam.get("fam"), fam.get("mid"), fam.get("x"),
            fam.get("uv"), fam.get("n_geo"), fam.get("tan4"),
            fam.get("vcol"), _direct, _baked, ssao, ao, fg,
            lm_u=(gt.get("lm_u") if gt is not None else fam.get("lm_u")),
            lm_v=(gt.get("lm_v") if gt is not None else fam.get("lm_v")),
            quality=_q)
        for _k, _v in _cnt.items():
            LMG_REACH[_k] = LMG_REACH.get(_k, 0) + _v
        # RETIRED (#80 option (c)): this was the opaque-side class_px_
        # accumulate. It had two defects that only a single-map design
        # removes rather than patches. Its family map was MAT_FAM[mid_op]
        # -- the OPAQUE class alone -- sampled under `fg`, the UNION
        # coverage, so every MASK-class pixel was credited to whatever
        # opaque material happened to sit behind it and no mask pixel
        # could ever reach its own class. And it lived HERE, inside
        # `if LMG_LIVE and fam is not None`, so with --lmg-family off
        # nothing counted opaque pixels at all: the whole matrix silently
        # depended on an unrelated family flag. The counting now happens
        # once, at the composite, over fam_pix -- see the census block in
        # render_window.
        if LMG_FADE_ON:
            # D_OPAQUE_FADE (dynamic stride 16, module r0_m7 at static
            # combo 0). The snippet is the one opaque_fade() already
            # transcribes byte-for-byte from three other families; it is
            # reused rather than re-typed, and it is applied only to
            # these three families' pixels.
            _on = ((fam.get("fam") == FAM_LIGHTMAPPEDGENERIC)
                   | (fam.get("fam") == FAM_IMPORTED)
                   | (fam.get("fam") == FAM_GENERIC)).float()
            _fade, _keep = opaque_fade(_a2, _on)
            rgb = torch.where((_keep & (_on > 0)).unsqueeze(-1), rgb,
                              torch.where((_on > 0).unsqueeze(-1),
                                          torch.zeros_like(rgb), rgb))
    if transmissive is not None:
        # S_TRANSMISSIVE_BACKFACE_NDOTL, the LIGHTING half. csgo_foliage
        # s32/d0:532 -- inside the per-light body, so it accumulates over
        # the sun AND every clustered light, not the sun alone:
        #     backlight += max(0, -dot(N, L)) * lightColor
        #                  * (g_bShadows ? shadow : 1.0)
        # and s32/d0:825 adds `backlight * transmissiveColor` to the shade.
        #
        # THREE CORRECTIONS against the previous implementation, all of
        # them structural:
        #  1. no `** 2.2`. The reference applies no gamma here at all --
        #     the albedo branch is already linear (it comes through the
        #     material colour transform) and the texture branch is used
        #     raw. The 2.2 is DELETED, not moved behind a flag.
        #  2. the baked irradiance no longer gates it. The reference gates
        #     the backlight by the SHADOW only (`g_bShadows ? shadow : 1`);
        #     a lightmap is a front-side visibility term and multiplying
        #     the back-side term by it was double-counting occlusion.
        #  3. the clustered lights contribute. `alights` is this
        #     renderer's own per-light diffuse accumulation; its
        #     back-facing counterpart is the same sum with the cosine
        #     replaced by max(0, -N.L), which is what --transmissive-lights
        #     enables (default on, because the reference's term is INSIDE
        #     the light loop -- sun-only would be the truncation).
        back = (-ndl_raw).clamp(0, 1)
        if shadow is not None:
            back = back * shadow
        t = transmissive.clamp(min=0)
        rgb = rgb + t * back.unsqueeze(-1) * SUN_COLOR.view(1, 1, 1, 3)
    if view is not None and not gt_on:
        if args.transmissive_lights and alights_back is not None:
            rgb = rgb + t * alights_back
    if view is not None:
        # Cook-Torrance GGX on the sun only (the ambient term has no
        # direction to reflect). Same ndl -- so the baked sun-visibility
        # mask gates the highlight exactly as it gates the diffuse.
        # SKIPPED under --gt-lighting: gt_direct() already ran the
        # shader's own sun BRDF (a DIFFERENT one at each
        # S_SHADER_QUALITY) and gt_spec_cube() already ran the shader's
        # own environment specular, so running these too would add a
        # second sun highlight and a second env term to every pixel.
        h = _nrm(sun_dir.view(1, 1, 1, 3) + view)
        ndh = (normal * h).sum(-1).clamp(0, 1)
        ndv = (normal * view).sum(-1).clamp(1e-4, 1)
        vdh = (view * h).sum(-1).clamp(1e-4, 1)
        # THE SUN LOBE IS THE REFERENCE'S, NOT A TEXTBOOK GGX. What stood
        # here was d = a2/(pi*(...)^2) with a separate Smith-Schlick G and
        # a 0.25 factor -- a plausible reconstruction, and the shader does
        # none of it. csgo_environment_ps.glsl:913-917 carries ONE fused
        # expression whose denominator is ((d*d)*(LdotH^2))*((r*4)+2):
        # a combined normalisation-and-visibility written in ROUGHNESS,
        # with no pi and no Smith term. Substituting the familiar form
        # changes the lobe's magnitude at every roughness.
        #
        # :908 also floors roughness at the sun's own angular size --
        # _5538._m8.w, READ as 0.040000 at byte offset +31120+12, and
        # spent nowhere else. Without it a smooth material gets a
        # point-source delta where the reference gets a disc.
        rough_f = _lighting.sun_roughness_floor(rough, SUN_ANGULAR_W)
        # dot(L,H) == dot(V,H) for a half vector, so vdh IS the shader's X.
        # glsl:1180  _17127 = mix(_5618._m86, 0.035, _9038) -- the wet
        # block raises the dielectric reflectance toward a literal 0.035
        # over the wetness coverage, and _9451 (:1196) carries it into
        # the IBL Fresnel at :1696. The dry value is the material's own;
        # 0.04 is what this renderer used before there was a wet path.
        _f0d = (0.04 if wet_f0 is None else wet_f0.unsqueeze(-1))
        f0 = _f0d * (1 - metal).unsqueeze(-1) + \
            linear.clamp(0, 1) * metal.unsqueeze(-1)
        f = f0 + (1 - f0) * (1 - vdh).unsqueeze(-1) ** 5
        spec = _lighting.sun_specular_ggx(f, rough_f, ndh, vdh,
                                          ndl) * args.spec_gain
        _gt_value("sun.specular_ggx", spec, 0.0,
                  note="environment_ps:913-917, the reference's fused lobe "
                       "with its (r*4+2) denominator -- replaces a "
                       "textbook d/G/0.25 reconstruction. Roughness is "
                       "floored at the sun's angular size 0.040000 (:908) "
                       "before it enters.")
        if pbr_mask is not None:
            spec = spec * pbr_mask.unsqueeze(-1)
        rgb = rgb + spec * SUN_COLOR.view(1, 1, 1, 3)
        if sh_spec is not None:
            # Environment specular. Scaled by the ambient rather than by an
            # absolute probe level so it carries no independent exposure.
            # Unlike the DIFFUSE probe term this is NOT already in the
            # lightmap -- a lightmap bakes irradiance, not reflected
            # radiance -- so there is no double count to avoid here.
            if args.ibl_brdf == "lazarov":
                fe = env_brdf(rough, ndv, f0)
            else:
                fe = (f0 + (1 - f0) * (1 - ndv).unsqueeze(-1) ** 5) \
                    * (1.0 - rough).clamp(0, 1).unsqueeze(-1)
            e = ambient * sh_spec * fe * args.ibl_spec_gain
            if pbr_mask is not None:
                e = e * pbr_mask.unsqueeze(-1)
            rgb = rgb + e
    if emissive is not None:
        e = emissive * args.emissive_gain
        if pbr_mask is not None:
            e = e * pbr_mask.unsqueeze(-1)
        rgb = rgb + e
    if unlit is not None:
        # csgo_black_unlit / csgo_lightmappedgeneric's `black`: CS2 emits
        # these with no lighting at all. They carry no colour texture, so
        # the pack hands them the white sentinel and the lit path turns
        # them into bright walls.
        u3 = unlit.unsqueeze(-1)
        rgb = rgb * (1 - u3) + linear * u3
    if args.debug_lighting == "irr":
        rgb = ambient.expand(B, H, W, 3).clone()
    elif args.debug_lighting == "lmvalid":
        g = (lm_ok if lm_ok is not None
             else torch.zeros_like(fg, dtype=torch.float32))
        rgb = torch.stack([1 - g, g, torch.zeros_like(g)], dim=-1)
    if fogin is not None:
        # in LINEAR space and before the vertical flip, so the fog blend
        # uses the same un-flipped world attributes it was interpolated
        # with. Applied to lit geometry only unless --fog-sky.
        rgb = apply_fog(rgb, fogin, fg)
    rgb = rgb.flip(1)
    fg_f = fg.flip(1)
    if SKY_CUBE is not None:
        # #48 IN THE EXECUTED PATH. Background pixels take the reference's
        # own cube along the view direction, not a screen-locked gradient.
        _m = SKY_MVP[:B] if (SKY_MVP is not None
                             and SKY_MVP.shape[0] >= B) else SKY_MVP
        _sky = sky_pass(_m, H, W)
        _gt_value("sky.cube_radiance", _sky, None,
                  note="#48 far-plane pass; replaces the screen-locked "
                       "SKY_TOP/SKY_HORIZON gradient. Its spread ACROSS "
                       "frames is the property the gradient could not "
                       "have: a fixed image is std 0.00000 at a fixed "
                       "row by construction.")
        rgb = torch.where((~fg_f).unsqueeze(-1), _sky, rgb)
    else:
        rgb = torch.where((~fg_f).unsqueeze(-1),
                          sky_row.expand(B, H, W, 3), rgb)
    if fogin is not None and args.fog_sky:
        # no-geometry pixels have no world position; the distance limit is
        # infinite, i.e. the fog ramp is saturated. DIAGNOSTIC only --
        # Source 2 fogs the skybox through env_cubemap_fog (measured inert
        # on this map), not through env_gradient_fog.
        a = min(args.fog_strength, args.fog_max_opacity)
        rgb = torch.where((~fg_f).unsqueeze(-1),
                          rgb * (1 - a) + FOG_COLOR.view(1, 1, 1, 3) * a, rgb)
    # sky constants are fitted in linear space, same as the lit path
    # No Reinhard: the lighting coefficients were fitted to predict GT
    # linear radiance directly, so an extra tonemap here just re-darkens
    # what the fit already matched.
    # --- FAM_PROJECTED_DECALS then FAM_SPRITECARD -----------------------
    # In LINEAR space, before the tonemap, which is where the engine's
    # forward pass writes them: decals after opaque (they need the
    # resolved depth), particles after that (translucent, and NOT lit by
    # the cluster path -- spritecard declares no shadow and no 3D
    # sampler at all). The layers arrive already built by render_window
    # and are flipped here to match `rgb`, which flipped above.
    if decal_layer is not None or sprite_layer is not None:
        _dl = None if decal_layer is None else (
            decal_layer[0].flip(1), decal_layer[1].flip(1), decal_layer[2])
        _sl = None if sprite_layer is None else (
            sprite_layer[0].flip(1), sprite_layer[1].flip(1))
        rgb = _nw.composite_layers(rgb, _dl, _sl)
        if _dl is not None:
            NW_PIX[0] += float((_dl[1] > 0.01).sum())
        if _sl is not None:
            NW_PIX[1] += float((_sl[1] > 0.01).sum())
        NW_PIX[2] += float(rgb[..., 0].numel())
    if args.hdr_dump is not None:
        HDR_DUMP.append(torch.cat(
            [rgb.detach(), fg_f.unsqueeze(-1).float()], dim=-1).half())
    # --- videocfg_hdr_detail: the STORE into the HDR scene attachment ---
    # This is the last write of the scene pass, so it is where the
    # hardware quantises to whatever format the engine allocated. The
    # tier -> format map is READ (post.video_config.HDR_FORMAT_BY_TIER,
    # from the engine's own RT enumeration across all 30 GT arms); what
    # the axis DOES is mantissa width, not the range clamp -- see
    # post/scene_format.py. -1 -> RGBA16F (10 bits), 3 -> R11G11B10
    # (6/6/5), a 16x coarser quantisation on every scene pixel of
    # preset0, preset1 and every single-axis arm.
    rgb = _hdr_store(rgb)
    rgb = _tone_encode(rgb, fg_f)
    if args.supersample > 1:
        rgb = torch.nn.functional.avg_pool2d(
            rgb.permute(0, 3, 1, 2), args.supersample).permute(0, 2, 3, 1)
    # --- the preset post chain, in the reference's order ---------------
    #     tonemap -> CMAA -> (MSAA resolve) -> FSR
    # The MSAA resolve is upstream of this function (it consumes the
    # multisampled attachment, which no longer exists here); CMAA runs on
    # the resolved display-space image and FSR presents it. At tier 0
    # FSR_OUT_PRE is None and nothing runs, which is the reference's own
    # NATIVE path, not a skip.
    if args.cmaa == "on":
        rgb = _cmaa_pass(rgb)
    if FSR_OUT_PRE is not None:
        rgb = _fsr_upscale(rgb)
    return (rgb * 255).to(torch.uint8)


# =====================================================================
# THE PRESET POST CHAIN: hdr_detail, fsr_detail, cmaa
# =====================================================================
_HDR_STORE_DONE = []


def _hdr_store(rgb):
    """videocfg_hdr_detail: quantise to the HDR scene attachment.

    The clamp alone is inert on any de_inferno value (nothing reaches
    64512), so a renderer that implements only the clamp emits identical
    images at both tiers -- see post/scene_format.py, which is where the
    tier -> (mantissa bits) map and its two-sided validation against
    hardware float16 live."""
    if args.hdr_scene_format == "off":
        return rgb
    out = _po_fmt.quantize_to_format(rgb, args.hdr_scene_format)
    if not _HDR_STORE_DONE:
        _HDR_STORE_DONE.append(1)
        print(_po_fmt.quantize_report(rgb, out, args.hdr_scene_format),
              flush=True)
    return out[..., :rgb.shape[-1]] if out.shape[-1] > rgb.shape[-1] else out


_FSR_DONE = []


def _fsr_upscale(rgb):
    """EASU then RCAS, back to the output raster. The two modules are
    per-frame fullscreen passes taking a single (H, W, C), so the batch
    is walked rather than folded into a leading dimension the shaders do
    not have."""
    ow, oh = FSR_OUT_PRE
    con = (None if args.fsr_sharpness is None
           else _po_fsr.rcas_con(args.fsr_sharpness))
    out = torch.stack([_po_fsr.fsr_pass(rgb[i], ow, oh, con)[..., :3]
                       for i in range(rgb.shape[0])], 0)
    if not _FSR_DONE:
        _FSR_DONE.append(1)
        print(_po_fsr.fsr_report(rgb[0], out[0]), flush=True)
        if con is None:
            print("post/fsr: RCAS NOT RUN -- --fsr-sharpness is unset and "
                  "the tier -> sharpness table has not been READ out of "
                  "CS2. EASU alone is a legal FSR 1.0 configuration; "
                  "running RCAS at an invented sharpness would put a "
                  "fitted float in every pixel.", flush=True)
    return out


_CMAA_DONE = []


def _cmaa_pass(rgb):
    """r_csgo_cmaa_enable, on the DISPLAY-SPACE image -- cs2_cmaa2_cs_edges
    :39 takes sqrt() of its input, which is the cheap gamma of an already
    encoded value."""
    outs = []
    for i in range(rgb.shape[0]):
        e, c = _po_cmaa.compute_edges(rgb[i])
        n, d = _po_cmaa.process_candidates(rgb[i], e, c)
        o = _po_cmaa.deferred_apply(rgb[i], n, d)
        if not _CMAA_DONE:
            _CMAA_DONE.append(1)
            print(_po_cmaa.cmaa2_report(rgb[i], o, e, c), flush=True)
        outs.append(torch.as_tensor(o, dtype=rgb.dtype, device=rgb.device))
    return torch.stack(outs, 0)


def _single_shade(uv_pix, layer, normal, fg):
    albedo = sample_textures(uv_pix, layer)
    return _post_chain(albedo, normal, fg)


single_shade = torch.compile(_single_shade, dynamic=False)
post_chain = _post_chain  # dynamic shadow arg; keep eager


# =======================================================================
# RENDER / TOOLS MODES
# =======================================================================
# S_MODE_TOOLS_VIS, S_MODE_TOOLS_SHADING_COMPLEXITY and D_QUAD_OVERDRAW are
# part of the render stack, not decoration, so they are implemented rather
# than argued away.  Each one says, in its own docstring, what the corpus
# does and does not fix about it.

def _heat(x):
    """A perceptually monotone blue->cyan->green->yellow->red ramp on [0,1].

    Piecewise-linear, so a value read off the image maps back to a number
    without a lookup table.  Not a colour-science claim, a legend.
    """
    x = x.clamp(0, 1)
    r = (x * 4.0 - 1.5).clamp(0, 1)
    g = torch.minimum((x * 4.0 - 0.5).clamp(0, 1),
                      (3.5 - x * 4.0).clamp(0, 1))
    b = (1.5 - x * 4.0).clamp(0, 1)
    return torch.stack([r, g, b], dim=-1)


def _flop_table_for(unit):
    """The FLOP table for the requested unit, as a (n_families,) tensor.

    PER-VERTEX and PER-PIXEL are different units.  This returns exactly one
    of them and the caller labels the image with which; nothing here
    converts between them and no ratio is applied.
    """
    import cs2_vertex_stage as _vsx
    src = (_vsx.VS_FLOPS_PER_VERTEX if unit == "vertex"
           else _vsx.PS_FLOPS_PER_PIXEL)
    tab = torch.zeros(max(len(FAM_NAMES), 1), device=device)
    hit = []
    for i, nm in enumerate(FAM_NAMES):
        if nm in src:
            v = src[nm]
            tab[i] = float(v[2] if isinstance(v, tuple) else v)  # mean
            hit.append(nm)
    return tab, hit


def tools_shading_complexity(fg, mid, unit):
    """`S_MODE_TOOLS_SHADING_COMPLEXITY`.

    The cost per family is the MEASURED FLOP table, not an estimate of this
    renderer's own cost: per-vertex from
    SHADER_CALLFLOW_vertex_stages.md:41-44 (min/median/mean/max per family,
    the mean is used), per-pixel from FLOP_LOGIT_TABLES.md.

    CAVEAT CARRIED WITH THE NUMBERS, printed once per run: only the
    env_blend and complex per-pixel figures are counted on a single method
    and a single loop basis (3,205 vs 1,510, FLOP_LOGIT_TABLES.md:78-83).
    The other per-pixel rows were extrapolated against a since-retracted
    figure and that file says they "must not be quoted as measurements".
    The per-VERTEX table has no such problem -- all four families were
    counted directly from bytecode with 100% container coverage.
    """
    tab, hit = _flop_table_for(unit)
    if not FAMILY:
        raise SystemExit("--tools-shading-complexity needs --fam-side: the "
                         "cost is per shader FAMILY and without the side "
                         "table there is no family per pixel")
    cost = tab[MAT_FAM[mid]]
    top = args.tools_complexity_max or float(tab.max())
    if not TOOLS_NOTE:
        TOOLS_NOTE.append(1)
        print(f"S_MODE_TOOLS_SHADING_COMPLEXITY: unit = FLOPs per "
              f"{'VERTEX' if unit == 'vertex' else 'PIXEL'}, ramp 0..{top:g}, "
              f"families with a table entry: {', '.join(hit)}", flush=True)
        if unit != "vertex":
            print("  CAVEAT (FLOP_LOGIT_TABLES.md:70-83): only env_blend "
                  "3205 and complex 1510 are like-for-like counts; the "
                  "other per-pixel rows are extrapolations against a "
                  "retracted figure and are not measurements.", flush=True)
    img = _heat(cost / max(top, 1e-6))
    return torch.where(fg.unsqueeze(-1), img, torch.zeros_like(img))
