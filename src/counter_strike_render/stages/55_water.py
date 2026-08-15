

def water_fancy_shade(wpos, n_geo, vcol, mid, x, scene_rgb, scene_z,
                      eye_pos, fwd, tsec, amb=None, lm=None, shadow=None,
                      moit_t=None, moit_final=None):
    """csgo_water_fancy r17_m2 -> (rgb, alpha).

    Every S_ axis is a PER-PIXEL gate, so S_REFLECTION_TYPE 0/1/2,
    S_REFRACTION, S_CAUSTICS, S_INTERACTION_EFFECTS and
    S_BLUR_REFRACTION all run at every value the family declares,
    whatever de_inferno's one material happens to select.

    Arguments that stand in for an engine resource:
      scene_rgb  g_tRefractionMap  <- the already-shaded opaque colour
      scene_z    g_tSceneDepth     <- the opaque depth, gl_FragCoord.z
      amb        the constant ambient (_5538._m0), None -> zeros
      lm         the lightmap irradiance (_22452.xyz), None -> zeros
      shadow     cascade x screen-space shadow (:920-930), None -> 1
      moit_t     g_tZerothMoment, moit_final g_tMoitFinal; None -> the
                 shader's own transmittance-1 degeneracy, i.e. no
                 discard and no resolve.
    """
    flat_n, white = _water_defaults()
    B, Hh, Ww = wpos.shape[0], wpos.shape[1], wpos.shape[2]
    # Per-frame time is BATCHED in session mode -- (B,) tensor, one value
    # per frame in the window. float(tsec) crashed on de_inferno (first
    # map with water reached by a batch): keep it a (B,1,1) broadcast so
    # every animation phase term stays sequence-parallel. A python float
    # (single-camera path) reshapes to (1,1,1) and broadcasts identically.
    t = (torch.zeros(B, 1, 1, device=device) if tsec is None else
         torch.as_tensor(tsec, device=device,
                         dtype=torch.float32).reshape(-1, 1, 1))
    lw = torch.tensor([0.2125000059604644775390625,
                       0.7153999805450439453125,
                       0.07209999859333038330078125], device=device)
    zero1 = torch.zeros(B, Hh, Ww, device=device)
    zero3 = torch.zeros(B, Hh, Ww, 3, device=device)
    # WHAT DIFFERS: this renderer lights its BLEND class AFTER
    # compositing, so the shader's two irradiance inputs -- the constant
    # ambient (_5538._m0, :941) and the lightmap (_22452.xyz, :1206) --
    # are set here to SUM TO UNITY when the caller has neither.  Only
    # their split is substituted; every consumer of them below is the
    # reference's, and the value this function returns then goes through
    # the same lighting the rest of the blend class does.
    half = torch.full((B, Hh, Ww, 3), 0.5, device=device)
    amb = half if amb is None else amb
    lm = half if lm is None else lm

    # ---- combo axes ------------------------------------------------
    a_refl = water_axis(x, "wf_reflection_type", WATER_OV["reflection_type"])
    a_refr = water_axis(x, "wf_refraction", WATER_OV["refraction"])
    a_caus = water_axis(x, "wf_caustics", WATER_OV["caustics"])
    a_int = water_axis(x, "wf_interaction", WATER_OV["interaction"])
    a_blur = water_axis(x, "wf_blur_refraction", WATER_OV["blur_refraction"])
    # S_REFLECTION_TYPE is a THREE-VALUE axis, not a flag.  0 emits
    # g_vSimpleSkyReflectionColor (s0/r0:931 reads a CB constant where
    # s4/r1:880 reads the cube); 1 adds the environment cube; 2 adds the
    # SSR march on top of it (:1222-1333).  Reading it as a bitmask is
    # the README §1 trap and would collapse 2 onto 1.
    refl_cube = (a_refl >= 0.5).float()
    refl_ssr = (a_refl >= 1.5).float()

    # ---- :271-276  backface normal flip ----------------------------
    #   if (g_bRenderBackfaceNormals && !g_bDontFlipBackfaceNormals)
    #       N = vNormal * (gl_FrontFacing ? 1 : -1);
    # WHAT DIFFERS: this renderer draws the water plane single-sided, so
    # gl_FrontFacing is 1 on every covered pixel and the branch is the
    # identity here.  Stated rather than silently dropped.
    n_geo = _nrm(n_geo)

    # ---- :296-307  the view basis ----------------------------------
    noise = _water_blue_noise(B, Hh, Ww)
    n_x = noise[..., 0]                                    # :573 _13724
    n_xy = noise[..., :2]                                  # :297 _4130
    d_eye = wpos - eye_pos.view(-1, 1, 1, 3)
    V = _nrm(d_eye)                                        # :301 _25095
    Vn = -V                                                # :302 _3072
    dist = d_eye.norm(dim=-1)                              # :303 _3558
    vz = Vn[..., 2].clamp(min=1e-3)
    # :306  vec3(V.xy / V.z, sqrt(V.z)) -- the screen-space displacement
    # basis every parallax step below rides on.
    disp = torch.cat([Vn[..., :2] / vz.unsqueeze(-1),
                      vz.sqrt().unsqueeze(-1)], dim=-1)
    par = -Vn[..., :2] / (vz + 0.25).unsqueeze(-1)         # :307 _5201

    # screen UV = gl_FragCoord.xy * g_vInvViewportSize      :298 _21412
    yy, xx = torch.meshgrid(torch.arange(Hh, device=device).float(),
                            torch.arange(Ww, device=device).float(),
                            indexing="ij")
    suv_u = ((xx + 0.5) / Ww).unsqueeze(0).expand(B, Hh, Ww)
    suv_v = ((yy + 0.5) / Hh).unsqueeze(0).expand(B, Hh, Ww)
    scene_col = scene_rgb[..., :3]

    # ---- :314-334  S_REFRACTION, first block -----------------------
    #   depth01   = clamp((sceneDepth - near) / (far - near), 0, 1)
    #   waterDepth= max(viewZ(P) - linear(depth01), 0) * 0.01 + bias
    #   bias      = clamp(luma(sceneColour), 0, 0.4) * -0.03      :319
    z_lin = _lin_depth(scene_z)
    own_lin = ((wpos - eye_pos.view(-1, 1, 1, 3))
               * fwd.view(-1, 1, 1, 3)).sum(-1).clamp(min=1e-4)
    below = (z_lin - own_lin).clamp(min=0.0)
    bias = (scene_col * lw).sum(-1).clamp(0.0, 0.4) * (-0.03)
    wdepth = torch.lerp(torch.ones_like(below),              # :333 else
                        below * 0.01 + bias, a_refr)         # :325 _13378
    shore = (wdepth - 0.02).clamp(min=0.0)                   # :335 _9935
    edge_v = wdepth * vz                                     # :336 _5722

    # ---- :337-338  map UV, for the skybox edge fade ----------------
    mn_u = _e2(x, "wf_map_uv_min_u"); mn_v = _e2(x, "wf_map_uv_min_v")
    mx_u = _e2(x, "wf_map_uv_max_u"); mx_v = _e2(x, "wf_map_uv_max_v")
    map_u = (wpos[..., 0] - mn_u) / (mx_u - mn_u + 1e-6)
    map_v = 1.0 - (wpos[..., 1] - mn_v) / (mx_v - mn_v + 1e-6)

    # ---- :346-349  the three vertex-colour min/max ramps ------------
    # Three (min, max) pairs live at material-CB offsets 88/92, 96/100
    # and 104/108 and are selected by vColor .x / .y / .z.  Their
    # CONSUMERS fix which is which: .x scales the wave amplitude and the
    # specular-power ramp (:835), .y feeds the foam coverage (:575), .z
    # thresholds the debris coverage (:589) -- so they are
    # g_flWaterRoughness*, g_flFoam* and g_flDebris* respectively.
    amp = torch.lerp(_e2(x, "wf_rough_min"), _e2(x, "wf_rough_max"),
                     vcol[..., 0]).clamp(min=0.0)            # :346 _24840
    foam_k = torch.lerp(_e2(x, "wf_foam_min"), _e2(x, "wf_foam_max"),
                        vcol[..., 1]).clamp(min=0.0)         # :349 _13154
    debris_k = torch.lerp(_e2(x, "wf_debris_min"), _e2(x, "wf_debris_max"),
                          vcol[..., 2]).clamp(min=0.0)       # :348 _7010

    # ---- :350-353  wave-space UV and its derivative LOD -------------
    plane_off = _e2(x, "wf_plane_offset")
    wuv = (wpos + disp * (0.5 - plane_off).unsqueeze(-1))[..., :2] \
        * 0.0333333350718021392822265625                     # :350 _3386
    lodk = 0.5 * _fwidth_len(wuv).clamp(min=1e-12).pow(
        0.100000001490116119384765625) * _e2(x, "wf_detail_fade")

    # ---- :354-360  S_INTERACTION_EFFECTS: the effects map -----------
    # g_tWaterEffectsMap at the screen-projected coordinate, .yz decoded
    # as (foam, silt) by clamp((t - 0.5) * 2, 0, 1).
    eff0 = _water_effects(torch.stack([suv_u, suv_v], -1),
                          T_WEFFECT[mid], _e2(x, "has_weffect"))
    e_fs = ((eff0[..., 1:3] - 0.5) * 2.0).clamp(0.0, 1.0)    # :355 _20371
    dist_str = _e2(x, "wf_eff_disturb")
    disturb0 = (e_fs[..., 0] + e_fs[..., 1]) * dist_str * a_int   # :359
    disturb_q = disturb0 * 0.25                              # :360 _7242

    # ================================================================
    # :381-413  THE WAVE LOOP -- `%16582`, UNIFORM-bounded by
    # g_nWaveIterations (97 FLOPs, 1 fetch per body).  A rotated,
    # scale-stepping octave sum of g_tWavesNormalHeight whose per-octave
    # weight is a three-band mix of g_flLowFreqWeight /
    # g_flMedFreqWeight / g_flHighFreqWeight.  Three accumulators come
    # out: a parallax offset, a UV scroll, and the normal + height.
    # ================================================================
    n_iter = int(WATER_OV["wave_iterations"]
                 if WATER_OV["wave_iterations"]
                 else float(_e2(x, "wf_wave_iter").max().clamp(min=1.0)))
    n_iter = max(1, min(n_iter, 16))
    sc = torch.stack([_e2(x, "wf_wave_scale_u"),
                      _e2(x, "wf_wave_scale_v")], -1).clamp(min=1e-3)
    ph = _e2(x, "wf_waves_init_dir")
    speed = _e2(x, "wf_waves_speed")
    sharp = _e2(x, "wf_waves_sharp").clamp(min=1e-3)
    w_low = _e2(x, "wf_low_freq")
    w_med = _e2(x, "wf_med_freq")
    w_high = _e2(x, "wf_high_freq")
    par_amt = _e2(x, "wf_waves_height_off")
    nrm_amt = _e2(x, "wf_waves_normal_str")
    acc_par = torch.zeros(B, Hh, Ww, 2, device=device)       # :366 _13137
    acc_uv = torch.zeros(B, Hh, Ww, 2, device=device)        # :370 _17133
    acc_n = torch.zeros(B, Hh, Ww, 2, device=device)         # :368 _17116
    acc_h = zero1.clone()                                    # :369 _17117
    for i in range(n_iter):
        k = (i / max(n_iter - 1.0, 1.0)) * 2.0               # :388 _14123
        k0 = min(max(k, 0.0), 1.0)
        k1 = min(max(k - 1.0, 0.0), 1.0)
        w = torch.lerp(torch.lerp(w_low + disturb0 * 0.05,
                                  w_med + disturb_q, k0),
                       w_high * amp + disturb_q, k1)         # :389 _14126
        # :390  uv = ((waveUV + parallax*3) + scroll) / scale
        #            + (sin,cos)(phase) * (time*speed*0.5) * sqrt(1/scale)
        u = ((wuv + acc_par * 3.0) + acc_uv) / sc
        rot = torch.stack([torch.sin(ph), torch.cos(ph)], -1)
        u = u + rot * ((t * speed) * 0.5).unsqueeze(-1) * (1.0 / sc).sqrt()
        wv = _water_slot(u, T_WWAVES[mid], _e2(x, "has_wwaves"),
                         flat_n)[..., :3] - 0.5              # :390 _19401
        h = (wv[..., 2] * w) * sc.norm(dim=-1) * 0.01        # :391 _8072
        g = wv[..., :2] * 2.0                                # :392 _9388
        # :393  the aspect-ratio correction: an anisotropic wave scale
        # must not tilt the normal toward its long axis.
        ar = torch.stack([torch.clamp(sc[..., 1] / sc[..., 0], max=1.0),
                          torch.clamp(sc[..., 0] / sc[..., 1], max=1.0)], -1)
        gn = g * ar * (w * 0.1).unsqueeze(-1)                # :393 _10224
        acc_par = acc_par + (-par) * (h * par_amt * amp).unsqueeze(-1)
        acc_uv = acc_uv + (gn * nrm_amt.unsqueeze(-1)) * sc \
            * sharp.unsqueeze(-1)                            # :395 _8974
        acc_h = acc_h + h                                    # :396 _9798
        acc_n = acc_n + gn                                   # :397 _23792
        sc = sc * sharp.unsqueeze(-1)                        # :402 _10517
        ph = ph + 3.5 / float(i + 1)                         # :404 _10149
    scroll = acc_uv * 0.1                                    # :414 _9431
    wave_n = acc_n * amp.unsqueeze(-1)                       # :415 _20576
    wave_h = (acc_h * amp) * 60.0                            # :434 _11974

    # ---- :418-433  the reconstructed-floor normal ------------------
    # `-normalize(cross(ddx(floorPos), ddy(floorPos)))` jittered by the
    # blue noise.  Without S_REFRACTION there is no depth read and the
    # else-branch takes N = (0,0,1) and k = g_flEdgeShapeEffect (:431).
    edge_shape = _e2(x, "wf_edge_shape")
    floor_p = wpos + disp * below.unsqueeze(-1)
    fn = -_nrm(torch.cross(
        torch.nn.functional.pad(floor_p[:, :, 1:] - floor_p[:, :, :-1],
                                (0, 0, 0, 1), mode="replicate"),
        torch.nn.functional.pad(floor_p[:, 1:, :] - floor_p[:, :-1, :],
                                (0, 0, 0, 0, 0, 1), mode="replicate"),
        dim=-1))
    fn_xy = fn[..., :2] + (n_xy - 0.5) * 0.05                # :422 _8218
    floor_n = _nrm(torch.cat([fn_xy, fn[..., 2:]], -1))
    floor_n = torch.lerp(
        torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3)
        .expand_as(floor_n), floor_n, a_refr.unsqueeze(-1))
    disp_k = torch.lerp(
        edge_shape,
        edge_shape * (1.2 - floor_n[..., 2]
                      * (1.0 - (edge_v * 8.0).clamp(0, 1))).clamp(0, 1),
        a_refr)                                              # :427 / :432

    # ================================================================
    # :444-498  S_INTERACTION_EFFECTS -- the ripple / foam / silt field.
    # Five world-projected taps of g_tWaterEffectsMap: one at the
    # displaced position and four finite-difference neighbours, out of
    # which a tangent normal is built by
    # `normalize(cross(dx, dy)).xy * vec2(-1, 1)`.  s48/r9 is this axis
    # alone and is exactly five taps of binding 64.
    # ================================================================
    p_eff = wpos + disp * (torch.lerp(zero1, wave_h, disp_k)
                           - plane_off).unsqueeze(-1)
    p_eff = p_eff + torch.cat([wave_n, zero1.unsqueeze(-1)], -1) * (-16.0)
    ev = []
    for off in ((0.0, 0.0), (1.0, 0.0), (0.0, -1.0),
                (0.005, 0.0), (0.0, 0.005)):
        q = p_eff + torch.tensor([off[0], off[1], 0.0], device=device)
        uu = (q[..., 0] - mn_u) / (mx_u - mn_u + 1e-6)
        vv = 1.0 - (q[..., 1] - mn_v) / (mx_v - mn_v + 1e-6)
        ev.append(_water_effects(torch.stack([uu, vv], -1),
                                 T_WEFFECT[mid], _e2(x, "has_weffect")) - 0.5)
    c_fs = (ev[0][..., 1:3] * 2.0).clamp(0.0, 1.0)           # :456 _20373
    n3 = (ev[3][..., 1:3] * 2.0).clamp(0.0, 1.0)
    n4 = (ev[4][..., 1:3] * 2.0).clamp(0.0, 1.0)
    ai = a_int.unsqueeze(-1)
    ripple_n = torch.stack([-(ev[3][..., 0] - ev[0][..., 0]),
                            (ev[4][..., 0] - ev[0][..., 0])], -1) \
        * (ev[0][..., 0].abs() * 4.0).unsqueeze(-1) \
        * _e2(x, "wf_eff_ripple").unsqueeze(-1) * ai          # :475 _4508
    foam_n = torch.stack([-(c_fs[..., 0] - n3[..., 0]),
                          (c_fs[..., 0] - n4[..., 0])], -1) * ai  # :478
    silt_n = torch.stack([-(c_fs[..., 1] - n3[..., 1]),
                          (c_fs[..., 1] - n4[..., 1])], -1) \
        * c_fs[..., 1:2].clamp(min=1e-4).pow(3.5) * ai       # :481 _17122
    silt = c_fs[..., 1] * _e2(x, "wf_eff_silt") * a_int      # :477 _13138
    eff_foam = c_fs[..., 0] * _e2(x, "wf_eff_foam") * a_int  # :479 _17120
    disturb = torch.lerp(disturb_q,
                         ((c_fs[..., 0] + c_fs[..., 1]) * dist_str) * 0.25,
                         a_int)                              # :480 _17121
    wave_h = torch.lerp(wave_h,
                        wave_h + ev[0][..., 0]
                        * _e2(x, "wf_eff_ripple") * 12.0, a_int)  # :476

    # ================================================================
    # :503-558  THE RAIN LOOP -- `%10380`, FIXED trip 2, 137 FLOPs per
    # body, no fetch.  Gated by `g_flRainStrength * perViewRain > 0`
    # (:499).  Each iteration rotates world XY by 0.48*(i+0.5), tiles it,
    # and rings a cosine ripple out of each cell centre with a per-cell
    # random phase.
    # ================================================================
    rain = _e2(x, "wf_rain_strength") * args.water_rain
    rain_off = torch.zeros(B, Hh, Ww, 2, device=device)
    rain_h = zero1.clone()
    if float(rain.max()) > 0.0:
        base_t = t * 6.5                                     # :505 _20688
        r2 = (rain * rain).clamp(min=1e-6)                   # :534 _20376
        for i in range(2):
            a2 = 0.4799999892711639404296875 * (i + 0.5)     # :522
            s2, c2 = math.sin(a2), math.cos(a2)
            tt = base_t + i * 0.3167000114917755126953125    # :526
            k2 = 0.60000002384185791015625 \
                + i * 0.06610000133514404296875
            px = p_eff[..., 0] * 0.119999997317790985107421875
            py = p_eff[..., 1] * 0.119999997317790985107421875
            gx = (c2 * px + s2 * py) * k2 + tt * 0.02        # :527 _10212
            gy = (-s2 * px + c2 * py) * k2 + tt * 0.02
            fx = gx.frac() * 2.0 - 1.0
            fy = gy.frac() * 2.0 - 1.0
            qx = c2 * fx + s2 * fy                           # :528 _10063
            qy = -s2 * fx + c2 * fy
            ln = torch.sqrt(qx * qx + qy * qy).clamp(min=1e-6)
            f1 = (1.0 - ln).clamp(0, 1)                      # :530 _9017
            f2 = (1.0 - ln * 2.0).clamp(0, 1)                # :531 _20212
            c3 = (gx * 0.1).frac().mul(10.0).floor()         # :532 _7135
            seed = tt + (c3 + (gy * 0.1).frac().mul(10.0).floor()
                         * (0.2 + c3)) * 8.0                 # :533 _7042
            ring = ((torch.sin(((f1 * 4.0) + seed) * rain)
                     - (1.0 - 0.2 * r2)) * 2.0).clamp(0, 1) * f1 / r2
            amp2 = torch.lerp(torch.full_like(f1, 2.0),
                              torch.full_like(f1, 0.25),
                              (_fwidth_scalar(f1) * 20.0).clamp(0, 1))
            dirn = torch.stack([qx, qy], -1) / ln.unsqueeze(-1)
            rain_off = rain_off - (dirn
                                   * torch.cos(f1 * 40.0 + t * 65.0
                                               ).unsqueeze(-1)
                                   * f1.pow(2.0).unsqueeze(-1)
                                   * amp2.unsqueeze(-1)) \
                * (ring * 2.0).unsqueeze(-1)                 # :535 _24829
            rain_h = rain_h + ((torch.sin((((f2 * 0.25) + seed * 0.5) + 4.0)
                                          * rain)
                                - (1.0 - r2 * 0.02)) * 2.0).clamp(0, 1) \
                * f2 / r2 * 0.05                             # :536 _23259
    disturb_z = disturb + rain_h * 400.0                     # :544 _15309.z
    p_disp = wpos + torch.cat([rain_off * 0.5,
                               zero1.unsqueeze(-1)], -1)     # :545 _24093

    # ---- :559-600  foam, then debris -------------------------------
    p_foam = p_disp + disp * (torch.lerp(torch.full_like(wave_h, 0.5),
                                         wave_h, disp_k * 0.5)
                              - plane_off).unsqueeze(-1)
    p_foam = p_foam + torch.cat(
        [ripple_n, zero1.unsqueeze(-1)], -1) * (-2.0)        # :560 _3271
    w_t = (t * 3.0) + torch.sin(t * 0.5) * 0.1               # :561 _4520
    wob = torch.stack([torch.sin(w_t + p_eff[..., 1] * 0.07),
                       torch.cos(w_t + p_eff[..., 0] * 0.07)], -1)
    f_scale = _e2(x, "wf_foam_scale").clamp(min=1e-3).unsqueeze(-1)
    fuv0 = p_foam[..., :2] / f_scale                         # :565 _3790
    fuv = (fuv0 + (scroll * _e2(x, "wf_foam_wobble").unsqueeze(-1) * 0.5)
           * (1.0 - foam_k).unsqueeze(-1)) - silt_n / f_scale  # :566 _4842
    amt = (0.05 + disturb_z).unsqueeze(-1)                   # :567 _9037
    mixk = (shore * 10.0).clamp(0, 1).unsqueeze(-1)          # :568 _9083
    f1t = _water_slot(torch.lerp(fuv0, fuv + wob * amt * 0.03, mixk),
                      T_WFOAM[mid], _e2(x, "has_wfoam"), flat_n)
    wob2 = torch.stack([torch.sin(w_t + p_eff[..., 1] * 0.06),
                        torch.cos(w_t + p_eff[..., 0] * 0.06)], -1)
    f2t = _water_slot(torch.lerp(fuv0.flip(-1) * 0.731000006198883056640625,
                                 fuv.flip(-1) * 0.731000006198883056640625
                                 + wob2 * amt * 0.02, mixk),
                      T_WFOAM[mid], _e2(x, "has_wfoam"), flat_n)  # :570
    fh = torch.maximum(f1t[..., 2], f2t[..., 2]) \
        + torch.sin(n_x) * 0.125                             # :574 _4812
    foam_a = ((foam_k * (1.0 + disturb_z * 0.008)
               * (1.0 - (disturb * 2.0).clamp(0, 1))) + eff_foam).clamp(0, 1)
    dz = disturb_z.clamp(min=0.0).pow(1.5)                   # :576 _8031

    d_scale = _e2(x, "wf_debris_scale").clamp(min=1e-3).unsqueeze(-1)
    duv0 = p_foam[..., :2] / d_scale                         # :578 _3839
    dwob = scroll * _e2(x, "wf_debris_wobble").unsqueeze(-1)  # :579 _22564
    # :581-583  the silt normal is reduced to its DOMINANT axis before it
    # warps the debris UV -- the smaller component is zeroed, not scaled.
    sx = silt_n[..., 0].abs()
    sy_ = silt_n[..., 1] * (silt_n[..., 1].abs() > sx).float()
    sn = torch.stack([silt_n[..., 0] * (sx > sy_.abs()).float(), sy_], -1) \
        / d_scale * 400.0                                    # :583 _10787
    duv = (((duv0 + dwob * (1.0 - debris_k).unsqueeze(-1))
            + par * ((1.0 + (torch.sin(disturb_z * 50.0) * 4.0
                             * (0.1 - dz).clamp(0, 1))) * dz).unsqueeze(-1)
            * 0.1)
           + wob * (0.1 + disturb_z).unsqueeze(-1) * 0.02) - sn
    duv = torch.lerp(duv0, duv, (shore * 4.0).clamp(0, 1).unsqueeze(-1))
    deb = _water_slot(duv, T_WDEBRIS[mid], _e2(x, "has_wdebris"), white)
    dh = deb[..., 3] - 0.5                                   # :587 _9406
    dh2 = dh * 2.0                                           # :588 _3542
    thr = 1.0 - (debris_k * (1.39999997615814208984375 - disturb_z
                             / torch.lerp(torch.ones_like(deb[..., 3]),
                                          torch.full_like(deb[..., 3], 0.4),
                                          deb[..., 3])).clamp(0, 1))
    deb_a = ((deb[..., 3] - thr)
             * _e2(x, "wf_debris_edge_sharp")).clamp(0, 1)   # :590 _6806
    open_w = (2.0 * dz - dh2).clamp(min=0.0)                 # :591 _19396
    deb_edge = (1.0 - open_w * 10.0).clamp(0, 1) * deb_a     # :592 _4500
    dn = _water_slot(duv, T_WDEBRISN[mid], _e2(x, "has_wdebrisn"), flat_n)
    dn_xy = torch.stack([dn[..., 0] - 0.5, -(dn[..., 1] - 0.5)], -1) \
        * _e2(x, "wf_debris_normal_str").unsqueeze(-1)       # :595 _11003
    foam_cov = (((foam_a * fh) * 0.25
                 + (foam_a - (1.0 - fh)).clamp(0, 1) * 0.75)
                - deb_edge).clamp(0, 1)                      # :599 _12904
    wave_h = torch.lerp(wave_h, wave_h * 0.5 + dh2, deb_edge)  # :600 _16601

    # ---- :615-650  the composite tangent normal --------------------
    n_ts = wave_n * 2.0 * _e2(x, "wf_spec_normal_mult").unsqueeze(-1) \
        * torch.lerp(torch.ones_like(lodk), torch.full_like(lodk, 2.0),
                     lodk).unsqueeze(-1)                     # :615 _11876
    n_ts = n_ts * (1.0 + (0.2 - wdepth).clamp(0, 1) * 8.0).unsqueeze(-1)
    n_ts = n_ts + dn_xy * deb_edge.unsqueeze(-1) * 1.5       # :623
    pick = (f2t[..., 2] > f1t[..., 2]).unsqueeze(-1).float()
    n_ts = n_ts + torch.lerp(f1t[..., :2] - 0.5, f2t[..., :2] - 0.5,
                             pick) * foam_cov.unsqueeze(-1)  # :627
    n_ts = n_ts + foam_n * foam_cov.unsqueeze(-1) * 0.5      # :631
    n_ts = n_ts + ripple_n * (1.0 - (deb_edge + foam_cov).clamp(0, 1)
                              ).unsqueeze(-1) * 2.0          # :635
    n_ts = n_ts + rain_off * 0.5                             # :639
    n_ts = n_ts * (1.0 + (n_xy - 0.5) * 2.0
                   * _e2(x, "wf_waves_normal_jitter").unsqueeze(-1))  # :644
    nz = (1.0 - (n_ts * n_ts).sum(-1).clamp(0, 1)).clamp(min=0).sqrt()
    n_shade = torch.cat([n_ts, nz.unsqueeze(-1)], -1)        # :648 _25156
    # :649-650  a SECOND, three-times-wider normal, built for the Fresnel
    # and refraction terms only.  It is a distinct vector, not a scale.
    w3 = n_ts * 3.0
    nz3 = (1.0 - (w3 * w3).sum(-1).clamp(0, 1)).clamp(min=0).sqrt()
    n_fres = torch.cat([w3, nz3.unsqueeze(-1)], -1)          # :650 _3653
    # :655-658  near the shore both bend back to the floor normal.
    lo = torch.lerp(torch.full_like(wdepth, 60.0),
                    torch.full_like(wdepth, 120.0), floor_n[..., 2])
    k_sh = ((((1.0 / lo - wdepth) * lo).clamp(0, 1)
             + ((0.025 - wdepth) * 8.0).clamp(0, 1)).clamp(0, 1)
            / (1.0 + dist * 0.002)) * 0.6 * a_refr
    n_shade = _nrm(torch.lerp(n_shade, floor_n, k_sh.unsqueeze(-1)))
    n_fres = _nrm(torch.lerp(n_fres, floor_n, k_sh.unsqueeze(-1)))

    # ---- :665-667  Fresnel and the foam colour ---------------------
    ndv = (Vn * n_fres).sum(-1).clamp(0, 1)                  # :665 _16080
    fres = (1.0 - ndv).clamp(min=0).pow(_e2(x, "wf_fresnel_exp"))
    foam_rgb = torch.stack([_e2(x, "wf_foam_r"), _e2(x, "wf_foam_g"),
                            _e2(x, "wf_foam_b")], -1) \
        * (1.0 + foam_cov * 0.5).unsqueeze(-1)               # :667 _5903

    # ================================================================
    # :673-826  S_REFRACTION (the scene-colour read) and S_CAUSTICS.
    # :675  the refraction offset is the wide normal expressed in the
    # camera's screen basis, jittered by the blue noise and CLAMPED by
    # g_flRefractionLimit; :680 fades it out in shallow water so the
    # shoreline does not tear.
    # ================================================================
    zdown = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 1, 1, 3)
    right = torch.cross(Vn, zdown.expand_as(Vn), dim=-1)
    ro = torch.stack([(n_fres[..., :2] * right[..., :2]).sum(-1),
                      (n_fres[..., :2] * Vn[..., :2]).sum(-1)], -1)
    ro = (ro + (n_xy - 0.5) * 0.002
          * _e2(x, "wf_caustic_dist").unsqueeze(-1)) \
        * torch.minimum(_e2(x, "wf_refract_limit"), wdepth).unsqueeze(-1)
    ro = ro * (wdepth * 10.0).clamp(0, 1).unsqueeze(-1)      # :680 _3433
    refr = _water_tap(scene_col, (suv_u + ro[..., 0]).clamp(0, 1),
                      (suv_v + ro[..., 1]).clamp(0, 1))      # :683 _18433
    # ---- S_BLUR_REFRACTION -----------------------------------------
    # s108/r18:577 -- a FIVE-tap average whose four outer taps step by
    # g_flRefractSampleOffset along +-x / +-y, each tap's refraction
    # offset grown by (1 + k*g_flRefractChromaticSeparation) and each
    # weighted by its own chroma vector, the sum scaled by 0.2.  This is
    # 4 EXTRA fetches of binding 60, which is exactly the 2 -> 6 census
    # step between s12 and s108.
    off = _e2(x, "wf_refract_sample_off")
    chr_ = _e2(x, "wf_refract_chroma")
    one3 = torch.ones(3, device=device)
    blur = refr * torch.lerp(one3, one3, chr_.unsqueeze(-1))
    for k, (dxo, dyo, cw) in enumerate((
            (0.0, 1.0, (0.0, 0.0, 3.0)), (0.0, -1.0, (0.0, 3.0, 0.0)),
            (1.0, 0.0, (2.0, 2.0, 0.0)), (-1.0, 0.0, (2.0, 0.0, 0.0)))):
        g = 1.0 + chr_ * (k + 1)
        c = _water_tap(scene_col,
                       (suv_u + ro[..., 0] * g + dxo * off).clamp(0, 1),
                       (suv_v + ro[..., 1] * g + dyo * off).clamp(0, 1))
        blur = blur + c * torch.lerp(one3, torch.tensor(cw, device=device),
                                     chr_.unsqueeze(-1))
    blur = blur * 0.20000000298023223876953125
    refr = torch.lerp(refr, blur, a_blur.unsqueeze(-1))
    under = refr.clamp(min=0).pow(1.10000002384185791015625) \
        * _e2(x, "wf_underwater_dark").unsqueeze(-1)         # :684 _21971

    # ---- :690-805  S_CAUSTICS --------------------------------------
    # Gated on the receiver's own luminance clearing
    # g_flCausticShadowCutOff (:686), so a shadowed floor grows no
    # caustic.  Then two 3-iteration loops over g_tWavesNormalHeight --
    # `%7953` accumulating a UV warp and `%22416` the intensity -- a
    # screen-space edge sharpen (:797), and a chromatic split (:800).
    cut = _e2(x, "wf_caustic_cutoff")
    c_gate = (((under * lw).sum(-1) - cut) * (2.0 + cut)).clamp(0, 1) * a_caus
    c_sharp = _e2(x, "wf_caustic_sharp")
    c_scale = (_e2(x, "wf_caustic_uvscale")
               * 0.0333333350718021392822265625).unsqueeze(-1)
    fall = _e2(x, "wf_caustic_falloff").clamp(min=1e-3)
    c_t = below / fall                                       # :716 _12082
    c_near = (1.0 - c_t).clamp(0, 1)                         # :717 _20105
    # :728  the caustic UV is the RECEIVER's world position, not the
    # surface's -- caustics live on the floor, which is why
    # g_bUseTriplanarCaustics exists at all.
    recv = wpos + disp * below.unsqueeze(-1)
    cuv0 = recv[..., :2] * c_scale
    c_sc = torch.stack([_e2(x, "wf_wave_scale_u"),
                        _e2(x, "wf_wave_scale_v")], -1).clamp(min=1e-3)
    c_ph = _e2(x, "wf_waves_init_dir")
    warp = torch.zeros(B, Hh, Ww, 2, device=device)
    for i in range(3):                                       # :740-755
        u = (cuv0 + warp) / c_sc
        rot = torch.stack([torch.sin(c_ph), torch.cos(c_ph)], -1)
        u = u + rot * ((t * speed) * 0.5).unsqueeze(-1) * (1.0 / c_sc).sqrt()
        tex = _water_slot(u, T_WWAVES[mid], _e2(x, "has_wwaves"), flat_n)
        warp = warp + ((tex[..., :2] - 0.5) * 0.5
                       * c_sharp.unsqueeze(-1)) \
            * (1.0 + c_sc) * (0.25 + c_t).unsqueeze(-1)      # :746 _15431
        c_sc = c_sc * sharp.unsqueeze(-1)
        c_ph = c_ph + 3.5 / float(i + 1)
    c_sc = torch.stack([_e2(x, "wf_wave_scale_u"),
                        _e2(x, "wf_wave_scale_v")], -1).clamp(min=1e-3)
    c_ph = _e2(x, "wf_waves_init_dir")
    cau = zero3.clone()
    lam = c_sharp * (1.0 - c_t.clamp(0, 1))                  # :774 _19848
    for i in range(3):                                       # :767-784
        k = (i / max(n_iter - 1.0, 1.0)) * 2.0
        u = (cuv0 + warp) / c_sc
        rot = torch.stack([torch.sin(c_ph), torch.cos(c_ph)], -1)
        u = u + rot * ((t * speed) * 0.5).unsqueeze(-1) * (1.0 / c_sc).sqrt()
        tex = _water_slot(u, T_WWAVES[mid], _e2(x, "has_wwaves"), flat_n)
        # :775  pow(height, sharpness*5) IS the caustic sharpening -- a
        # high exponent turns a smooth wave height field into thin
        # bright filaments.
        h3 = tex[..., 2:3].clamp(min=1e-4).pow((lam * 5.0).unsqueeze(-1))
        w = torch.lerp(torch.lerp(w_low + disturb * 0.1, w_med + disturb,
                                  min(max(k, 0.0), 1.0)),
                       w_high * amp + disturb,
                       min(max(k - 1.0, 0.0), 1.0)).clamp(0.1, 0.4)
        cau = cau + ((h3 * w.unsqueeze(-1)) * (1.0 + cau * 2.0)
                     * c_near.unsqueeze(-1) * lam.unsqueeze(-1)) * 2.0
        c_sc = c_sc * sharp.unsqueeze(-1)
        c_ph = c_ph + 3.5 / float(i + 1)
    cx = cau[..., 0]                                         # :799 _13526
    cx = cx + cx / (_fwidth_scalar(cx) * 1000.0 + 0.5)       # :797 _12449
    eff_c = _e2(x, "wf_eff_caustic")
    cau = cau + ((cx.clamp(0, 1) * 4.0) * eff_c
                 - (-cx).clamp(0, 1) * 0.15 * eff_c).unsqueeze(-1)
    c_amt = c_gate * (below * 0.05).clamp(0, 1) * c_near     # :718 _14250
    c_tint = torch.stack([_e2(x, "wf_caustics_tint_r"),
                          _e2(x, "wf_caustics_tint_g"),
                          _e2(x, "wf_caustics_tint_b")], -1)
    # :800  the caustic multiplies the refracted floor, in the SUN's
    # colour, tinted, at g_flCausticsStrength * 0.1.
    c_col = cau.clamp(min=0.001000000047497451305389404296875) \
        .mul(8.0).pow(2.5) * c_amt.unsqueeze(-1) \
        * SUN_COLOR.view(1, 1, 1, 3) * c_tint \
        * _e2(x, "wf_caustic_str").unsqueeze(-1) * 0.1
    lit_floor = torch.lerp(under, under * (1.0 + c_col), a_caus.unsqueeze(-1))

    # ---- :827-843  the water body: decay, opacity, clarity ----------
    dmax = _e2(x, "wf_max_depth").clamp(min=1e-3)
    d_eff = torch.minimum(dmax, wdepth)                      # :827 _6655
    decay = torch.stack([_e2(x, "wf_decay_r"), _e2(x, "wf_decay_g"),
                         _e2(x, "wf_decay_b")], -1)
    absorb = torch.exp((decay - 1.0)
                       * _e2(x, "wf_decay_strength").unsqueeze(-1)
                       * d_eff.unsqueeze(-1))                # :828 _11466
    turb = torch.maximum(_e2(x, "wf_caustic_dist") + silt * 2.0,
                         c_fs[..., 1] * a_int)               # :829 _8790
    surf = foam_a + (c_fs[..., 1] * a_int - 0.5).clamp(0, 1)  # :830 _5577
    opac = (1.0 - torch.exp(-d_eff * turb)) \
        + (surf - n_x.clamp(0, 1) * 0.25) * 0.1              # :831 _4632
    fogc = torch.stack([_e2(x, "wf_fog_r"), _e2(x, "wf_fog_g"),
                        _e2(x, "wf_fog_b")], -1)
    body = torch.lerp(fogc, foam_rgb, (surf * 0.1).unsqueeze(-1)) \
        * torch.lerp(absorb, torch.ones_like(absorb),
                     (turb * 0.04).clamp(0, 1).unsqueeze(-1))  # :832 _5353
    clarity = 1.0 - opac                                     # :837 _9615
    thru = (((1.0 - deb_a) + open_w).clamp(0, 1)
            * (1.0 - foam_cov * 4.0).clamp(0, 1)) * clarity  # :838 _3039

    # ---- :841-845  the lightmap is fetched at a DISPLACED UV --------
    # :843 pushes the lightmap coordinate by the wave height, so the
    # baked light swims with the surface.  WHAT DIFFERS: the lightmap
    # arrives here already sampled by the caller, so the displacement is
    # not applied; only its magnitude is lost, not its presence.

    # ---- :849-946  shadow, then the sun -----------------------------
    # :920-930  the cascade shadow is min()-combined with the
    # screen-space shadow (the offset-88 resource, here at binding 52).
    sh = torch.ones(B, Hh, Ww, device=device) if shadow is None else shadow
    vis = torch.lerp(sh, torch.ones_like(sh), thru * 0.5)    # :937 _22047
    ndl = (n_shade * sun_dir.view(1, 1, 1, 3)).sum(-1)
    direct = amb + (ndl.clamp(min=0.0) * vis).unsqueeze(-1) \
        * SUN_COLOR.view(1, 1, 1, 3)                         # :941 _21710
    direct = torch.where((ndl * vis > 0).unsqueeze(-1), direct, amb)

    # ---- :1206-1221  the surface composite --------------------------
    #   (lights + lightmap) * mix(mix(body*opacity*fogStrength,
    #                                 foam, foamCoverage),
    #                             debris*edge, debrisCoverage)
    fog_str = _e2(x, "wf_fog_strength")
    inner = torch.lerp((body * opac.unsqueeze(-1)) * fog_str.unsqueeze(-1),
                       foam_rgb, foam_cov.unsqueeze(-1))
    d_tint = torch.stack([_e2(x, "wf_debris_tint_r"),
                          _e2(x, "wf_debris_tint_g"),
                          _e2(x, "wf_debris_tint_b")], -1)
    d_col = deb[..., :3] * (0.5 + deb_edge * 0.5).unsqueeze(-1) * d_tint
    rgb = (direct + lm) * torch.lerp(
        inner, d_col, (deb_a - open_w).clamp(0, 1).unsqueeze(-1))  # :1207
    rgb = torch.lerp(rgb, lit_floor * absorb, thru.unsqueeze(-1))  # :1212
    k_amb = (opac * ((1.0 - (deb_a + foam_cov).clamp(0, 1))
                     + open_w).clamp(0, 1)) * (1.0 - fog_str)
    rgb = torch.lerp(rgb, (body * 4.0) * lm, k_amb.unsqueeze(-1))  # :1217

    # ================================================================
    # :1222  S_REFLECTION_TYPE >= 1 -- the environment cube.
    #   lod = sqrt(dot(mix(g_vRoughness, vec2(1), clamp(k,0,0.35)),
    #                  vec2(0.5))) * 6
    #   rgb = cube(-reflect(V,N), lod) * luma(lightmap)
    #             * g_flEnvironmentMapBrightness * g_flLowEndCubeMapIntensity
    # ================================================================
    R = -_water_reflect(V, n_shade)
    r2v = torch.lerp(torch.stack([_e2(x, "wf_roughness_x"),
                                  _e2(x, "wf_roughness_y")], -1),
                     torch.ones(2, device=device),
                     lodk.clamp(0.0, 0.3499999940395355224609375
                                ).unsqueeze(-1))
    lod6 = (r2v * 0.5).sum(-1).clamp(min=0).sqrt() * 6.0
    if CUBE is not None:
        # The reference's LOD is in a 7-level chain (0..6). Ours has
        # len(CUBE) levels, so the reference value is rescaled onto our
        # chain rather than passed raw -- passing 0..6 into a 3-mip chain
        # would clamp every rough pixel onto the last mip and make the
        # middle levels unreachable BY CONSTRUCTION, which is the
        # --ibl-lod-scale failure this file already carries a note about.
        env = cube_sample(probe_of(wpos), R,
                          lod6 * (len(CUBE) - 1) / 6.0)
    else:
        env = SKY_TOP.view(1, 1, 1, 3).expand(B, Hh, Ww, 3)
    env = env * ((lm * lw).sum(-1) * _e2(x, "wf_env_bright")).unsqueeze(-1) \
        * _e2(x, "wf_lowend_cube_int").unsqueeze(-1)
    # S_REFLECTION_TYPE == 0 emits a CONSTANT instead (s0/r0:931 reads a
    # CB vec3 where s4/r1:880 reads the cube).
    simple = torch.stack([_e2(x, "wf_simple_sky_r"),
                          _e2(x, "wf_simple_sky_g"),
                          _e2(x, "wf_simple_sky_b")], -1)
    refl = torch.lerp(simple, env, refl_cube.unsqueeze(-1))

    # ================================================================
    # :1224-1333  S_REFLECTION_TYPE == 2 -- the SSR march (`%23225`,
    # UNIFORM-bounded by g_nSSRMaxForwardSteps, 75 FLOPs + 1 fetch per
    # step).  A geometric ray march whose step grows 1.15x each
    # iteration; a hit is `0 <= dz < thickness*step`; the hit fraction
    # lerps the previous and current UV (:1303).  The resolve is a 4-tap
    # blur of the scene colour plus a bloom boost, faded by pow(k,4) and
    # by how near the top of the screen the hit landed (:1328).
    # ================================================================
    ssr_n = int(WATER_OV["ssr_steps"] if WATER_OV["ssr_steps"]
                else float(_e2(x, "wf_ssr_steps").max().clamp(min=1.0)))
    ssr_n = max(1, min(ssr_n, 64))
    jit = ((n_x - 0.5) * 2.0) * _e2(x, "wf_ssr_jitter")      # :1229 _4714
    thick = _e2(x, "wf_ssr_thick") + jit                     # :1230 _4799
    stepv = ((_e2(x, "wf_ssr_step") + jit) / (1.0 + lodk * 2.0)) \
        * torch.lerp(torch.full_like(ndv, 20.0),
                     torch.ones_like(ndv), ndv)              # :1244 _9277
    # :1231  the SSR ray uses a normal whose XY is 3x widened -- a
    # separate vector again, so the reflection streaks along the wave
    # crests instead of following the shading normal.
    Rv = _nrm(torch.cat([n_shade[..., :2] * 3.0, n_shade[..., 2:]], -1))
    Rr = _nrm(_water_reflect(V, Rv))                         # :1237 _21677
    hit_u, hit_v, _ = _water_project(wpos, eye_pos, fwd)
    prev_u, prev_v = hit_u.clone(), hit_v.clone()
    pos = wpos.clone()
    dz_prev = zero1.clone()
    frac = zero1.clone()
    done = torch.zeros(B, Hh, Ww, dtype=torch.bool, device=device)
    used_n = zero1.clone()
    for i in range(ssr_n):
        stepv = stepv * 1.14999997615814208984375            # :1279 _3353
        pos = torch.where(done.unsqueeze(-1), pos,
                          pos + Rr * stepv.unsqueeze(-1))    # :1280 _5027
        pu, pv, pz = _water_project(pos, eye_pos, fwd)
        sd = _lin_depth(_water_tap(scene_z.unsqueeze(-1),
                                   pu.clamp(0, 1), pv.clamp(0, 1))[..., 0])
        dz = sd - pz                                         # :1288 _3746
        f = (dz / (dz - dz_prev).clamp(min=1e-6)).clamp(0, 1)  # :1289
        hit = (dz >= 0) & (dz < thick * stepv) & (~done)     # :1291-1293
        prev_u = torch.where(done, prev_u, hit_u)
        prev_v = torch.where(done, prev_v, hit_v)
        hit_u = torch.where(done, hit_u, pu)
        hit_v = torch.where(done, hit_v, pv)
        frac = torch.where(hit, f, frac)
        used_n = torch.where(done, used_n, torch.full_like(used_n, i + 1.0))
        done = done | hit
        dz_prev = torch.where(done, dz_prev, dz)
    ssr_u = torch.lerp(hit_u, prev_u, frac).clamp(0, 1)      # :1303 _23824
    ssr_v = torch.lerp(hit_v, prev_v, frac).clamp(0, 1)
    ratio = (used_n - frac) / float(ssr_n)                   # :1315 _5420
    o = ratio * 0.00390625
    e = 0.001953125
    ssr = (_water_tap(scene_col, (ssr_u - o).clamp(0, 1),
                      (ssr_v - o).clamp(0, 1)) * 0.4444443881511688232421875
           + _water_tap(scene_col, (ssr_u + e).clamp(0, 1),
                        (ssr_v - o).clamp(0, 1))
           * 0.22222219407558441162109375
           + _water_tap(scene_col, (ssr_u - o).clamp(0, 1),
                        (ssr_v + e).clamp(0, 1))
           * 0.22222219407558441162109375
           + _water_tap(scene_col, (ssr_u + e).clamp(0, 1),
                        (ssr_v + e).clamp(0, 1))
           * 0.111111097037792205810546875)                  # :1321 _14528
    boost = ((ssr * lw).sum(-1) - _e2(x, "wf_ssr_boost_thresh")).clamp(min=0)
    ssr = (ssr + _nrm(ssr + 0.001000000047497451305389404296875)
           * boost.unsqueeze(-1) * _e2(x, "wf_ssr_boost").unsqueeze(-1)) \
        * _e2(x, "wf_ssr_bright").unsqueeze(-1)              # :1322 _18229
    k_ssr = ((1.0 - ratio.clamp(0, 1).pow(4.0)).clamp(0, 1)
             * (ssr_v * 8.0).clamp(0, 1)) * done.float()     # :1328
    refl = torch.lerp(refl, torch.lerp(refl, ssr, k_ssr.unsqueeze(-1)),
                      refl_ssr.unsqueeze(-1))

    # ---- :834-836 / :1334-1340  sun specular and its bloom boost -----
    gloss = _e2(x, "wf_glossiness")
    n_spec = _nrm(torch.lerp(n_geo, n_fres,
                             (gloss * (1.0 + dist * 0.0005)).unsqueeze(-1)))
    sd_ = (-sun_dir.view(1, 1, 1, 3)
           * _water_reflect(-_nrm(d_eye), n_spec)).sum(-1).clamp(0, 1)
    p_spec = torch.lerp(_e2(x, "wf_spec_power"),
                        _e2(x, "wf_debris_reflectance") * 8.0, deb_a) \
        * torch.lerp(torch.full_like(amp, 2.0), torch.full_like(amp, 0.2),
                     amp.clamp(0, 1))                        # :835 _3579
    spec = (sd_.clamp(min=1e-6).pow(p_spec) * 0.1
            + sd_.clamp(min=1e-6).pow(p_spec * 10.0))        # :836 _16651
    b_th = _e2(x, "wf_bloom_thresh")
    spec = spec + (spec - (1.0 - b_th)).clamp(min=0.0) \
        * _e2(x, "wf_bloom_str")                             # :1336
    spec = spec * torch.lerp(torch.ones_like(deb_a),
                             _e2(x, "wf_debris_reflectance") * 0.05, deb_a)
    f0 = torch.lerp(_e2(x, "wf_reflectance"),
                    _e2(x, "wf_debris_reflectance"), deb_edge)  # :1334
    k_refl = ((fres * (1.0 - f0) + f0)
              * ((1.0 - (deb_a + foam_cov).clamp(0, 1) * 0.75)
                 - foam_cov * 2.0)) * 1.5                    # :1335 _4658
    rgb = rgb + (direct * spec.unsqueeze(-1)) * k_refl.unsqueeze(-1) \
        * SUN_COLOR.view(1, 1, 1, 3)                         # :1336

    # ---- :1341-1396  the oil-slick ramp, then the reflection ---------
    # :1341-1390  a six-segment hue ramp driven by the Fresnel term, the
    # debris height and time; :1396 mixes the reflection through it by
    # g_flDebrisOilyness -- an oil sheen, which is why it rides on the
    # debris and not on the water.
    hue = ((fres * 20.0 + dh * 8.0 + t * 0.1).frac() * 6.0)  # :1341 _22128
    seg = hue.floor()
    fr = hue - seg
    up = 0.75 * fr
    dn2 = 0.75 * (1.0 - fr)
    z0 = torch.zeros_like(hue)
    cc = torch.full_like(hue, 0.75)
    ramp = torch.stack([
        torch.where(seg == 0, cc, torch.where(seg == 1, dn2,
                    torch.where(seg == 2, z0, torch.where(seg == 3, z0,
                                torch.where(seg == 4, up, cc))))),
        torch.where(seg == 0, up, torch.where(seg == 1, cc,
                    torch.where(seg == 2, cc, torch.where(seg == 3, dn2,
                                torch.where(seg == 4, z0, z0))))),
        torch.where(seg == 0, z0, torch.where(seg == 1, z0,
                    torch.where(seg == 2, up, torch.where(seg == 3, cc,
                                torch.where(seg == 4, cc, dn2)))))], -1)
    oily = (((open_w * 20.0).clamp(0, 1) * _e2(x, "wf_debris_oily"))
            / (1.0 + dist * 0.005)) * (1.0 - edge_v * 5.0).clamp(0, 1)
    rgb = torch.lerp(rgb, torch.lerp(refl, refl * ramp, oily.unsqueeze(-1)),
                     k_refl.clamp(0, 1).unsqueeze(-1))       # :1396

    # ---- :1468-1475  the map-edge fade, a DISCARD -------------------
    #   clamp(1 - clamp((max|2*mapUV-1| - (1-fade))/fade, 0, 1), 0, 1)
    #     - blueNoise.x  <  0   ->  discard
    # s0/r0:1008 has this too, so it is NOT an S_REFRACTION-gated block.
    # A discard in a rasteriser that has already composited is alpha 0,
    # which is the same pixel.
    fade_r = _e2(x, "wf_skybox_fade").clamp(min=1e-3)
    d2 = torch.maximum((0.5 - map_u).abs() * 2.0, (0.5 - map_v).abs() * 2.0)
    keep = (1.0 - ((d2 - (1.0 - fade_r)) / fade_r).clamp(0, 1)).clamp(0, 1)
    keep = (keep - n_x >= 0).float()

    # ---- :1477-1484  refraction vs water, and the alpha -------------
    # The reference OUTPUTS the refracted scene where the water is thin,
    # so its own alpha stays 1 and the blend lives in the colour.  This
    # renderer composites the family OVER an already-shaded background,
    # so the identical mix is expressed as an ALPHA: where the shader
    # would have emitted the scene colour unchanged, alpha is 0.  With
    # S_REFRACTION off the reference takes the else-branch (:1483) and
    # emits the water opaque, which is the axis's whole visible content
    # and is why the placeholder read as an olive slab across the street.
    dark = torch.lerp(torch.ones_like(edge_v),
                      torch.full_like(edge_v, 0.60000002384185791015625),
                      (edge_v * 60.0).clamp(0, 1) / (1.0 + dist * 0.002))
    k_water = ((_e2(x, "wf_fog_shadow_str") * d_eff
                + foam_cov.clamp(0, 1)) + (dh2 - 0.5)).clamp(0, 1)
    k_water = torch.lerp(torch.ones_like(k_water), k_water, a_refr)
    rgb = torch.lerp(lit_floor * dark.unsqueeze(-1), rgb,
                     k_water.unsqueeze(-1))                  # :1479
    alpha = k_water * keep

    # ---- :278-284 / :1486-1504  the MBOIT resolve -------------------
    if moit_t is not None:
        tr = torch.exp(-moit_t)                              # :279 _21877
        cov = 1.0 - tr                                       # :280 _4637
        if moit_final is not None:
            rgb = moit_final[..., :3] \
                * (cov / (moit_final[..., 3]
                          + 9.9999997473787516355514526367188e-06)
                   ).unsqueeze(-1) + rgb * tr.unsqueeze(-1)
        # :281  the shader DISCARDS at cov > 0.9999; alpha 0 is that pixel.
        alpha = alpha * (cov <= 0.99989998340606689453125).float()
    return rgb.clamp(min=0.0), alpha.clamp(0, 1)
# THE THREE SMALL FAMILIES: csgo_effects, csgo_black_unlit,
#                           csgo_vertexlitgeneric
#
# Every line below is transcribed from a module pulled out of the shipped
# .vcs with iji_model/counter_strike_render/vcs/ and decompiled with spirv-cross. Each
# block names its module as <family> c<static combo>/r<record>m<module>,
# with the decompiled GLSL line.
#
# The two container traps in iji_model/counter_strike_render/vcs/README.md were both
# checked rather than assumed:
#   * the record comes from `m_nByteCodeDataIdx`. For csgo_vertexlitgeneric
#     it is NOT the array position: static combo 512 is POSITION 63 and
#     RECORD 51, and record 51 is shared with combo 516.
#   * static combo ids are mixed-radix in general. For these three
#     families every axis has m_nMin 0 / m_nMax 1, so the place values
#     ARE powers of two and a bitmask decode happens to agree -- read off
#     m_nComboIndexValue, not carried over from csgo_environment.
#
# WHY THESE THREE ARE WORTH GETTING EXACT. csgo_black_unlit r0_m0 and
# csgo_effects r10_m0..m3 contain no loop region, so their
# SHADER_CALLFLOW_eight_small_families.md figures -- 132 FLOPs / 1 fetch
# and 194 FLOPs / 5 fetches -- are exact per-invocation counts, not
# floors. Everything else in this project reports a floor. `_SFCost`
# prices this transcription with the identical rules
# iji_model/counter_strike_render/vcs/flopcount.py applies to the SPIR-V, so the
# transcription is checkable against a number rather than against a
# reading. `--sf-flop-audit` runs it and exits non-zero on a mismatch.
# =======================================================================

# The reference ledgers, recovered by re-running flopcount.py's own
# classifier instruction by instruction over r0_m0.spv and r10_m1.spv --
# NOT copied out of the document. (The document's csgo_black_unlit
# category split does not match the tool's; both total 132, and the
# totals are what the document's headline claims.)
_SF_COST_REF = {
    "black_unlit": {
        "total": 132, "fetches": 1, "ops": 38,
        "cat": {"matrix products": 28,
                "raw arithmetic (mul/add/sub/div)": 27,
                "mix / lerp": 18, "pow / exp / log": 16,
                "length / sqrt / rsqrt": 12, "clamp / saturate": 12,
                "dot products": 10, "normalize": 7, "max": 2}},
    "effects": {
        "total": 194, "fetches": 5, "ops": 85,
        "cat": {"raw arithmetic (mul/add/sub/div)": 87,
                "pow / exp / log": 28, "length / sqrt / rsqrt": 27,
                "clamp / saturate": 20, "normalize": 14,
                "dot products": 10, "mix / lerp": 6, "max": 2}},
}


class _SFCost:
    """Per-invocation FLOP + fetch ledger for a transcribed module.

    Cost rules are flopcount.py's, verbatim (N = result component count):
    elementwise N, dot 2N-1, normalize 2N+1, mix 3N, clamp 2N, pow 4N,
    length 2N, distance 3N, min/max N, matrix*vector R*(2C-1),
    vector*scalar R.

    Every method WRAPS the arithmetic it prices and returns its value, so
    a term deleted from the transcription stops being counted at the same
    moment it stops being computed. A hand-maintained table of op counts
    drifts silently away from the code it claims to describe and a check
    against it cannot fail; this one can, and `--sf-flop-audit-mutate`
    demonstrates that it does.

    `off=True` makes every method a plain passthrough. That is what the
    render loop uses, so the ledger costs nothing when it is not being
    read.
    """

    CAT = {
        "fmul": "raw arithmetic (mul/add/sub/div)",
        "fadd": "raw arithmetic (mul/add/sub/div)",
        "fsub": "raw arithmetic (mul/add/sub/div)",
        "fdiv": "raw arithmetic (mul/add/sub/div)",
        "fnegate": "raw arithmetic (mul/add/sub/div)",
        "vector*scalar": "raw arithmetic (mul/add/sub/div)",
        "matrix*vector": "matrix products",
        "dot": "dot products", "Normalize": "normalize",
        "FMix": "mix / lerp", "FClamp": "clamp / saturate",
        "Pow": "pow / exp / log", "Length": "length / sqrt / rsqrt",
        "Distance": "length / sqrt / rsqrt", "FMax": "max", "FMin": "min",
    }

    def __init__(self, off=False):
        self.off = off
        self.reset()

    def reset(self):
        self.cat = collections.Counter()
        self.call = collections.Counter()
        self.seq = []
        self.flops = 0
        self.fetches = 0

    def _c(self, name, cost):
        if self.off:
            return
        self.cat[self.CAT[name]] += cost
        self.call[name] += 1
        self.seq.append((name, cost))
        self.flops += cost

    def mul(self, a, b, n):
        self._c("fmul", n); return a * b

    def add(self, a, b, n):
        self._c("fadd", n); return a + b

    def sub(self, a, b, n):
        self._c("fsub", n); return a - b

    def div(self, a, b, n):
        self._c("fdiv", n); return a / b

    def neg(self, a, n):
        self._c("fnegate", n); return -a

    def vscale(self, v, s, r):
        """OpVectorTimesScalar: cost R, the VECTOR's component count."""
        self._c("vector*scalar", r); return v * s

    def matvec(self, m, v, R=4, C=4):
        """row_major matRxC * vecC, cost R*(2C-1). `m` is (..., R, C)."""
        self._c("matrix*vector", R * (2 * C - 1))
        return (m * v.unsqueeze(-2)).sum(-1)

    def dot(self, a, b, n=3):
        self._c("dot", 2 * n - 1); return (a * b).sum(-1)

    def normalize(self, v, n=3, eps=1e-9):
        self._c("Normalize", 2 * n + 1)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)

    def length(self, v, n=3):
        self._c("Length", 2 * n); return v.norm(dim=-1)

    def distance(self, a, b, n=3):
        self._c("Distance", 3 * n); return (a - b).norm(dim=-1)

    def mix(self, a, b, t, n):
        self._c("FMix", 3 * n); return a + (b - a) * t

    def clamp(self, x, lo, hi, n):
        self._c("FClamp", 2 * n); return x.clamp(lo, hi)

    def pow(self, a, b, n=1):
        self._c("Pow", 4 * n); return a.clamp(min=0.0) ** b

    def fmax(self, a, b, n=1):
        self._c("FMax", n)
        return torch.maximum(a, b) if torch.is_tensor(b) else a.clamp(min=b)

    def fmin(self, a, b, n=1):
        self._c("FMin", n)
        return torch.minimum(a, b) if torch.is_tensor(b) else a.clamp(max=b)

    def fetch(self, value):
        if not self.off:
            self.fetches += 1
            self.seq.append(("FETCH", 0))
        return value

    def compare(self, key):
        """(ok, lines) against the reference ledger for `key`."""
        ref = _SF_COST_REF[key]
        ok = (self.flops == ref["total"] and self.fetches == ref["fetches"]
              and len(self.seq) == ref["ops"])
        out = [f"  FLOPs/px           {self.flops:5d}  reference "
               f"{ref['total']:5d}   delta {self.flops - ref['total']:+d}",
               f"  fetch sites        {self.fetches:5d}  reference "
               f"{ref['fetches']:5d}   delta "
               f"{self.fetches - ref['fetches']:+d}",
               f"  priced instructions{len(self.seq):5d}  reference "
               f"{ref['ops']:5d}   delta {len(self.seq) - ref['ops']:+d}"]
        for c in sorted(set(self.cat) | set(ref["cat"])):
            a, b = int(self.cat.get(c, 0)), int(ref["cat"].get(c, 0))
            out.append(f"    {c:36s} {a:5d} vs {b:5d}  "
                       f"{'OK' if a == b else 'DIFF'}")
            ok = ok and (a == b)
        return ok, out


_SF_NULL = _SFCost(off=True)

# Source units per pack unit. Every constant in these families'
# constant buffers is in SOURCE units, same as the --fog block above.
# Bound to the file's own `S` so the two cannot drift apart.
SF_SRC_U = S


# --- the constant buffer the fog blocks read ---------------------------
# _5037._m0.._m9 (per-view CB, byte offsets 368..576) are engine values,
# not shader bytecode, so their CONTENTS are in no .vcs. What IS literal
# is their CONSTRUCTION, and this file already parses the two entities
# they come from. `_sf_fog_cb()` builds the CB out of exactly those keys,
# so nothing here invents a number:
#     ramp  = clamp(m0.xy + m0.zw * vec2(dist, height), 0, 1)
#     m0.x = -start/(end-start)     m0.z =  1/(end-start)
#     m0.y = endH/(endH-startH)     m0.w = -1/(endH-startH)
#     m1.xy = (fogfalloffexponent, fogverticalexponent)
#     m2.xyz = fogcolor linear      m2.w = fogstrength
#     m3.x = fogstart^2             (the `dot(d,d) > m3.x` gate)
def _sf_fog_cb():
    e = max(args.fog_end - args.fog_start, 1e-6)
    eh = args.fog_end_height - args.fog_start_height
    eh = eh if abs(eh) > 1e-6 else 1e-6
    ce = max(args.sf_cube_fog_end - args.sf_cube_fog_start, 1e-6)
    return dict(
        m0x=-args.fog_start / e, m0y=args.fog_end_height / eh,
        m0z=1.0 / e, m0w=-1.0 / eh,
        m1x=args.fog_exponent, m1y=args.fog_vertical_exponent,
        m2w=min(args.fog_strength, args.fog_max_opacity),
        m3x=args.fog_start * args.fog_start,
        # :61 -- the gradient fog's HEIGHT cutoff, `wp.z * m3.z < m3.y`.
        # env_gradient_fog stops at fogendheight, so the ramp's sign
        # picks the half-space and the cutoff is the entity's own key.
        m3y=args.fog_end_height, m3z=1.0,
        # env_cubemap_fog -- the SECOND entity, whose shipped keys the
        # --fog block's header records as start 800 u / end 280000 u.
        m4x=-args.sf_cube_fog_start / ce, m4y=1.0 / ce,
        m4z=args.sf_cube_fog_lod_bias, m4w=args.sf_cube_fog_exponent,
        m5x=args.sf_cube_fog_height_bias, m5y=args.sf_cube_fog_height_scale,
        m5z=args.sf_cube_fog_height_exponent, m5w=args.sf_cube_fog_lod,
        m7x=args.sf_cube_fog_start * args.sf_cube_fog_start,
        m7y=args.sf_cube_fog_height_cutoff, m7z=1.0,
        m7w=args.sf_cube_fog_strength, m8x=args.sf_cube_fog_gain,
    )


def _sf_grad_fog_gated(L, wsrc, hz, cb, base):
    """black_unlit r0_m0:52-77 / vertexlit r51_m3:667-693.

    dot(d,d) gate, a two-axis clamped ramp, two pow()s, a strength
    scale and a mix() of the fog colour into the base. Returns
    (rgb, factor).
    """
    d2 = L.dot(wsrc, wsrc, 3)                                    # :59
    hgate = L.mul(hz, cb["m3z"], 1)                              # :61
    gate = ((d2 > cb["m3x"]) & (hgate < cb["m3y"])).float()      # :59-61
    dist = L.length(wsrc, 3)                                     # :70
    v = torch.stack([dist, hz], -1)
    sc = torch.tensor([cb["m0z"], cb["m0w"]], device=v.device)
    of = torch.tensor([cb["m0x"], cb["m0y"]], device=v.device)
    g = L.clamp(L.add(L.mul(v, sc, 2), of, 2), 0.0, 1.0, 2)      # :70
    px = L.pow(g[..., 0], cb["m1x"])                             # :71
    py = L.pow(g[..., 1], cb["m1y"])                             # :71
    f = L.mul(L.mul(px, py, 1), cb["m2w"], 1) * gate             # :71
    col = FOG_COLOR.view(*([1] * (base.dim() - 1)), 3).expand_as(base)
    return L.mix(base, col, f.unsqueeze(-1), 3), f               # :72


def _sf_cube_fog_gated(L, wsrc, hz, cb, base, wpos):
    """black_unlit r0_m0:82-103 / vertexlit r51_m3:698-722.

    WHAT DIFFERS. The reference fetches a dedicated env_cubemap_fog
    cubemap -- `samplerCube` at set 1/binding 42 on csgo_black_unlit and
    the per-view CB handle at OFFSET 72 on csgo_vertexlitgeneric
    (r51_m3:714, the offset-72 sky handle PER_VIEW_RENDER_TARGETS.md
    pins). That texture is an engine resource, is in no .vcs and is not
    in this project's asset set. The fetch SITE, the direction
    `normalize((m6 * vec4(d,0)).xyz)`, the LOD
    `m5.w * clamp(1 - occ*m4.z)` and the scale `* m8.x` are transcribed
    exactly; the texture CONTENT is substituted -- the nearest baked
    environment probe cube when --ibl-cube supplied one, else the fitted
    sky/fog colour. `--sf-cube-fog-source` chooses and the banner prints
    which ran, so the substitution is never silent.

    `m6` is the world->fog-cube rotation. cube_sample() already carries
    the pack->Source rotation, so here it is the identity; the
    matrix*vector is still ISSUED because the reference issues it and
    the ledger prices what actually runs.
    """
    d2 = L.dot(wsrc, wsrc, 3)                                    # :86
    hgate = L.mul(cb["m7z"], hz, 1)                              # :88
    gate = ((d2 > cb["m7x"]) & (hgate < cb["m7y"])).float()      # :86-88
    dist = L.length(wsrc, 3)                                     # :96
    a = L.add(L.mul(dist, cb["m4y"], 1), cb["m4x"], 1)           # :96
    a = L.clamp(L.pow(L.fmax(a, 0.0, 1), cb["m4w"]), 0.0, 1.0, 1)
    b = L.add(L.mul(hz, cb["m5y"], 1), cb["m5x"], 1)             # :96
    b = L.clamp(L.pow(L.fmax(b, 0.0, 1), cb["m5z"]), 0.0, 1.0, 1)
    occ = L.mul(a, b, 1)                                         # :96
    f = L.mul(L.clamp(occ, 0.0, 1.0, 1), cb["m7w"], 1) * gate    # :97
    v4 = torch.cat([wsrc, torch.zeros_like(wsrc[..., :1])], -1)
    I = torch.eye(4, device=wsrc.device).expand(*wsrc.shape[:-1], 4, 4)
    d = L.normalize(L.matvec(I, v4, 4, 4)[..., :3], 3)           # :98
    k = L.clamp(L.sub(1.0, L.mul(occ, cb["m4z"], 1), 1), 0.0, 1.0, 1)
    lod = L.mul(k, cb["m5w"], 1)                                 # :98
    if args.sf_cube_fog_source == "probe" and CUBE is not None:
        cube = L.fetch(cube_sample(probe_of(wpos), d, lod))      # :98
    else:
        cube = L.fetch(FOG_COLOR.view(*([1] * (base.dim() - 1)), 3)
                       .expand_as(base))
    cube = L.vscale(torch.cat([cube, torch.ones_like(cube[..., :1])], -1),
                    cb["m8x"], 4)[..., :3]                       # :98
    return L.mix(base, cube, f.unsqueeze(-1), 3), f              # :98


# --- csgo_black_unlit --------------------------------------------------
def sf_black_unlit_shade(wpos, eye, fog_on, L=None):
    """csgo_black_unlit c0/r0m0 -- the WHOLE in-game module, 115 lines.

    The family has no colour texture, no lighting, NO dynamic axis at
    all and two static combos. Its entire output is the map's fog
    composited over BLACK, alpha 1:

        rgb = g_bFogEnabled ? cubeFog(gradFog(vec3(0))) : vec3(0)
        out = vec4(rgb, 1.0)                                     (:113)

    This is why `--unlit`, which only forces the albedo to black, is not
    the same feature: it leaves the two `black_simple` faces UNFOGGED,
    so at 6,000 source units they read pure black where CS2 reads them
    at very nearly the fog colour.
    """
    L = L if L is not None else _SF_NULL
    cb = _sf_fog_cb()
    wp = L.add(wpos, torch.zeros_like(wpos), 3)                  # :47
    wsrc = L.sub(wp, eye, 3) / SF_SRC_U                          # :52
    hz = wp[..., 1] / SF_SRC_U
    rgb, _ = _sf_grad_fog_gated(L, wsrc, hz, cb, torch.zeros_like(wp))
    rgb, _ = _sf_cube_fog_gated(L, wsrc, hz, cb, rgb, wp)
    on = fog_on.unsqueeze(-1)
    rgb = rgb * on                                               # :109-112
    return rgb, torch.ones_like(rgb[..., 0])


# --- S_MODE_TOOLS_VIS = 1 ----------------------------------------------
# csgo_black_unlit c1/r1m0:54-693. This is the FIRST module in this
# corpus that actually selects S_MODE_TOOLS_VIS -- `--tools-vis`'s
# docstring further up says no module selecting the axis was decompiled,
# which was true when it was written and is no longer true. The mode is
# an INT at set 1/binding 0 offset 32 and the module is a flat chain of
# `if (mode == K)` RGB overwrites applied on top of the fogged base,
# alpha untouched. Every constant below is the literal in the module.
# csgo_effects c11/r11m2 carries the identical chain against the same CB
# slot, so both families share this one implementation, exactly as they
# share the snippet.
_SF_TV_CONST = {
    10: (0.5, 0.5, 0.5), 11: (0.525, 0.525, 0.525),
    12: (0.5, 0.5, 0.5), 13: (0.04, 0.04, 0.04),
    14: (1.0, 1.0, 1.0), 15: (1.0, 1.0, 1.0),
    17: (0.0, 0.0, 0.0), 18: (0.0, 0.0, 0.0),
    19: (0.5, 0.5, 0.5), 20: (0.5, 0.5, 1.0),
    22: (1.0, 0.5, 0.5), 23: (0.5, 1.0, 0.5),
    25: (1.0, 1.0, 1.0), 65: (1.0, 1.0, 1.0), 66: (1.0, 1.0, 1.0),
    80: (0.0, 0.0, 0.0),
}
_SF_TV_SRGB = (12, 20, 22, 23, 25)      # r1m0 :227/:452/:487/:512/:543
_SF_TV_MODES = tuple(sorted(set(list(_SF_TV_CONST)
                                + [0, 1, 16, 21, 24, 30, 31, 60, 81])))


def _sf_srgb_piecewise(c):
    """r1m0:437-466 -- the module's own sRGB->linear, branch included."""
    return torch.where(c <= 0.04045, c * 0.0773993805,
                       (c * 0.947867334 + 0.052132703).clamp(min=0.0) ** 2.4)


def sf_tools_vis(mode, base, nrm, wpos, eye, tsec):
    """csgo_black_unlit c1/r1m0. `base` is the fogged colour."""
    def c3(t):
        return torch.tensor(t, device=base.device, dtype=base.dtype).view(
            *([1] * (base.dim() - 1)), 3).expand_as(base)

    rgb = base
    if mode == 0:
        return rgb
    if mode == 60:                                               # :106-118
        rgb = c3(tuple(v * 0.5 for v in args.sf_tools_vis_tint))
    elif mode == 1:                                              # :120-142
        n = _nrm(nrm)
        v = _nrm(eye - wpos)
        p = n.clamp(0, 1) ** 2
        m = (-n).clamp(0, 1) ** 2
        a = (p * c3((0.6, 0.4, 1.0))).sum(-1) \
            + (m * c3((0.6, 0.4, 0.2))).sum(-1)
        lam = 0.3 + 0.7 * (n * v).sum(-1).clamp(0, 1)
        r = 2.0 * (v * n).sum(-1, keepdim=True) * n - v
        spec = 0.05 * ((v * r).sum(-1).clamp(0, 1) ** 4.0)
        rgb = (0.5 * a * lam + spec).unsqueeze(-1).expand_as(base)
    elif mode in (21, 24):                                       # :430/:520
        rgb = _sf_srgb_piecewise(_nrm(nrm) * 0.5 + 0.5)
    elif mode == 30:                                             # :560-575
        n = _nrm(nrm)
        dx = torch.zeros_like(n); dx[..., 1:, :] = n[..., 1:, :] - n[..., :-1, :]
        dy = torch.zeros_like(n); dy[..., 1:, :, :] = n[..., 1:, :, :] - n[..., :-1, :, :]
        v = torch.maximum((dx * dx).sum(-1), (dy * dy).sum(-1))
        rgb = (v.clamp(0, 1) ** 0.333).unsqueeze(-1).expand_as(base)
    elif mode == 31:                                             # :56-58/:577
        fn = torch.zeros_like(nrm); fn[..., 1:, :] = (nrm[..., 1:, :] - nrm[..., :-1, :]).abs()
        fp = torch.zeros_like(wpos); fp[..., 1:, :] = (wpos[..., 1:, :] - wpos[..., :-1, :]).abs()
        r = fn.norm(dim=-1) / fp.norm(dim=-1).clamp(min=1e-6)
        rgb = _sf_srgb_piecewise(r.unsqueeze(-1).expand_as(base))
    elif mode == 16:                                             # :400-406
        rgb = c3(tuple(args.sf_tools_vis_tint))
    elif mode == 81:                                             # :353-366
        k = max(args.sf_tools_vis_range, 1e-6)
        v = (min(max(0.0, -k), k) + k) / (k * 2.0)
        rgb = c3((v, v, v))
    elif mode in _SF_TV_CONST:
        rgb = c3(_SF_TV_CONST[mode])
        if mode in _SF_TV_SRGB:
            rgb = (rgb * 0.947867334 + 0.052132703) ** 2.4
        if mode == 13:                                           # :309-320
            rgb = rgb * 0.0773993805
        if mode in (10, 11, 19):                                 # :159/:244
            t = float(tsec) if tsec is not None else 0.0
            fl = 1.0 if (t - 1.5 * int(t / 1.5)) > 0.75 else 0.0
            rgb = rgb + (c3((0.0, 0.0, 1.0)) - rgb) * (0.7 * fl)
    else:
        raise SystemExit(
            f"--sf-tools-vis {mode}: csgo_black_unlit c1/r1m0 tests the "
            f"mode against {list(_SF_TV_MODES)} and nothing else. An "
            "unlisted value falls through the entire chain and emits the "
            "fogged base unchanged, which reads as 'the mode did nothing' "
            "rather than as 'the mode does not exist'.")
    if args.sf_tools_vis_flash:                                  # :660-676
        t = float(tsec) if tsec is not None else 0.0
        h = t * 0.5
        rgb = rgb + (c3((1.0, 0.0, 0.0)) - rgb) * abs((h - int(h)) * 1.6 - 0.8)
    return rgb


# --- csgo_effects ------------------------------------------------------
def sf_effects_shade(uv, mid, x, wpos, eye, nrm, vcol, albedo,
                     scene_depth_fc, view_fwd, tsec, front_facing,
                     SC, L=None):
    """csgo_effects c10/r10m1 -- the WHOLE module, 145 lines, 194 FLOPs.

    Static combo 10 = S_ADDITIVE_BLEND + S_DEPTH_FEATHER, which is what
    materials/effects/smoke/steam_001.vmat selects (F_ADDITIVE_BLEND 1,
    F_DEPTH_FEATHER 1) -- the steam plume, the family's only de_inferno
    material and 100% of its pixels.

    THE BAKED-LIGHTING DYNAMIC AXIS IS INERT HERE, measured not assumed:
    r10_m0 (dyn 0) and r10_m1 (dyn 4, D_BAKED_LIGHTING_FROM_LIGHTMAP)
    differ in EXACTLY ONE LINE -- the `layout(location=)` of the
    vertex-colour varying, 3 against 4. Identical arithmetic, identical
    194 FLOPs, identical 5 fetches. An unlit particle shader declares
    the baked-lighting axis and then ignores it.

    The five fetch sites in the reference's order: g_tColor, the three
    erosion masks g_tMask1/2/3, and a `texelFetch` of the SCENE DEPTH
    target. Four are material textures out of the side table; the fifth
    is this renderer's own opaque depth buffer.

    `SC` is the resolved STATIC combo -- Python bools, because a static
    combo selects a different compiled module, so branching on it here
    is what the engine does, not an optimisation. The ledger therefore
    prices exactly the module that ran.
    """
    L = L if L is not None else _SF_NULL
    cb = _sf_fog_cb()
    # :79-97 two-sided normal. `m2 != 0 && m1 == 0` is F_RENDER_BACKFACES
    # on the beauty pass (m1 is the depth-pass flag, 0 here).
    sgn = torch.where(front_facing, torch.ones_like(nrm[..., :1]),
                      -torch.ones_like(nrm[..., :1])) \
        if (SC["backfaces"] and front_facing is not None) \
        else torch.ones_like(nrm[..., :1])
    nn = L.normalize(L.vscale(nrm, sgn, 3), 3)                   # :92/:103
    wp = L.add(wpos, torch.zeros_like(wpos), 3)                  # :98
    col = L.fetch(albedo)                                        # :99
    rgb = L.mul(col[..., :3], vcol[..., :3], 3)                  # :105
    a_t = L.mul(col[..., 3], _e2(x, "sf_fx_opacity_scale"), 1)   # :104
    a_t = L.mul(vcol[..., 3], a_t, 1)                            # :104
    rgb = L.vscale(rgb, torch.ones_like(rgb[..., :1]), 3)        # :105 (*1.0)
    d = L.sub(wp, eye, 3)                                        # :100
    vdir = L.normalize(d / SF_SRC_U, 3)                          # :103
    vneg = L.neg(vdir, 3)                                        # :103

    # --- S_DEPTH_FEATHER, :103 ---------------------------------------
    # scene world position reconstructed from the depth target, then a
    # ramp on the distance between it and this fragment.
    if SC["depth_feather"]:
        # `scene_depth_fc` is the depth TARGET in the gl_FragCoord.z
        # convention the reference's texelFetch reads, NOT metres.
        #
        # WHY IT IS NOT METRES. The reference builds the scene point as
        #     hit = eye + V * (1 / (lin * dot(viewFwd, V)))
        # with V the UNNORMALISED eye->fragment vector. For `hit` to land
        # on the scene surface that expression forces `lin` to be
        # 1/z_view, so `m0.z * d + m0.w` is an affine map from the depth
        # target to INVERSE view depth -- which is exactly what a
        # hyperbolic depth buffer makes affine, and why the reference can
        # do it in one mul and one add with no divide. The constants come
        # from this frustum: z = p23/(ndc + p22) and ndc = 2*d - 1, so
        # 1/z = (2/p23)*d + (p22 - 1)/p23.
        _p22 = -(PROJ_FAR + PROJ_NEAR) / (PROJ_FAR - PROJ_NEAR)
        _p23 = -2.0 * PROJ_FAR * PROJ_NEAR / (PROJ_FAR - PROJ_NEAR)
        zraw = L.fetch(scene_depth_fc)                            # :103
        lin = L.sub(zraw, 0.0, 1)                                 # :103 (- m2)
        rng = L.sub(1.0, 0.0, 1)                                  # :103 (m3-m2)
        lin = L.div(lin, rng, 1)                                  # :103
        lin = L.clamp(lin, 0.0, 1.0, 1)                           # :103
        dd = L.sub(wp, eye, 3)                                    # :101
        lin = L.mul(lin, 2.0 / _p23, 1)                           # :103 (* m0.z)
        lin = L.add(lin, (_p22 - 1.0) / _p23, 1)                  # :103 (+ m0.w)
        pr = L.dot(view_fwd, dd, 3)                               # :103
        den = L.mul(lin, pr, 1)                                   # :103
        # The reference divides unguarded; a vectorised port evaluates
        # the fetch on every lane including ones the raster never
        # covered, where `den` can be 0. The clamp is a GUARD, not a
        # transcription, and it is the only added operation in this
        # function -- it is priced as the reference's own OpFDiv.
        inv = L.div(torch.ones_like(den),
                    torch.where(den.abs() < 1e-6,
                                torch.full_like(den, 1e-6), den), 1)
        hit = L.vscale(dd, inv.unsqueeze(-1), 3)                  # :103
        hit = L.add(eye, hit, 3)                                  # :103
        fd = L.distance(wp, hit, 3) / SF_SRC_U                    # :103
        fd = L.div(fd, _e2(x, "sf_fx_feather_dist").clamp(min=1e-4), 1)
        fd = L.clamp(fd, 0.0, 1.0, 1)                             # :103
        fd = L.pow(fd, _e2(x, "sf_fx_feather_falloff"))           # :103
    else:
        fd = torch.ones_like(a_t)

    # --- the facing (Fresnel) ramp, :103 ------------------------------
    nd = L.dot(vneg, nn, 3)                                       # :103
    nd = L.clamp(nd, 0.0, 1.0, 1)                                 # :103
    fr = L.pow(nd, _e2(x, "sf_fx_fresnel_exp"))                   # :103
    fr = L.mul(fr, _e2(x, "sf_fx_fresnel_falloff"), 1)            # :103
    fr = L.clamp(fr, 0.0, 1.0, 1)                                 # :103
    fr = L.mix(_e2(x, "sf_fx_fresnel_min"),
               _e2(x, "sf_fx_fresnel_max"), fr, 1)                # :103

    # --- the three scrolling erosion masks, :103 ----------------------
    t = tsec if tsec is not None else torch.zeros_like(wp[..., :1])
    masks = []
    for i, (tex, has) in enumerate(((T_SF_FXM1, "sf_fx_has_mask1"),
                                    (T_SF_FXM2, "sf_fx_has_mask2"),
                                    (T_SF_FXM3, "sf_fx_has_mask3"))):
        pan = L.vscale(torch.stack([_e2(x, f"sf_fx_m{i+1}_pu"),
                                    _e2(x, f"sf_fx_m{i+1}_pv")], -1), t, 2)
        sc = L.mul(uv, torch.stack([_e2(x, f"sf_fx_m{i+1}_su"),
                                    _e2(x, f"sf_fx_m{i+1}_sv")], -1), 2)
        m = L.fetch(sample_fam(L.add(sc, pan, 2), tex[mid]))[..., 0]
        h = _e2(x, has)
        masks.append(m * h + (1.0 - h))

    # --- the distance fade, :103 --------------------------------------
    fv = L.div(d / SF_SRC_U,
               _e2(x, "sf_fx_fade_dist").clamp(min=1e-4).unsqueeze(-1), 3)
    fl = L.clamp(L.length(fv, 3), 0.0, 1.0, 1)                    # :103
    fm = L.mix(_e2(x, "sf_fx_fade_min"),
               _e2(x, "sf_fx_fade_max"), fl, 1)                   # :103
    fm = L.pow(fm, _e2(x, "sf_fx_fade_falloff"))                  # :103

    # --- the product, :103-104 ----------------------------------------
    ero = L.mul(masks[0], masks[1], 1)                            # :103
    ero = L.mul(ero, masks[2], 1)                                 # :103
    ero = L.mul(ero, fr, 1)                                       # :103
    ero = L.mul(ero, fd, 1)                                       # :103
    ero = L.mul(ero, fm, 1)                                       # :103
    a_t = L.mul(a_t, ero, 1)                                      # :104
    rgb = L.vscale(rgb, _e2(x, "sf_fx_color_boost").unsqueeze(-1), 3)  # :105

    # --- S_TINT_MASK (combo 14, r14_m1:106) ---------------------------
    # With the axis OFF the vertex colour tints unconditionally (:105);
    # with it ON the tint is gated by g_tTintMask.x. Both arms reach
    # pixels; neither is a stub.
    if SC["tint_mask"]:
        tm = L.fetch(sample_fam(uv, T_SF_FXTINT[mid]))[..., 0]
        k = (tm * _e2(x, "sf_fx_has_tintmask")).unsqueeze(-1)
        rgb = L.mix(col[..., :3], rgb, k, 3)

    # --- the fog, :112-142 --------------------------------------------
    # The gate here is the per-view ivec4 `_5037._m0.y/.z`, NOT the
    # dot(d,d) test the other two families use, and the cube layer
    # fetches NO cubemap -- it only produces an opacity. That is why
    # csgo_effects has 5 fetch sites and not 6, and the FLOP ledger is
    # what caught it: an earlier draft reused the black_unlit fog block
    # here and the audit came out at 240 / 6.
    dsrc = L.sub(wp, eye, 3) / SF_SRC_U                           # :114
    hz = wp[..., 1] / SF_SRC_U
    if SC["fog"]:
        dist = L.length(dsrc, 3)                                  # :119
        v = torch.stack([dist, hz], -1)
        g = L.clamp(L.add(L.mul(v, torch.tensor(
            [cb["m0z"], cb["m0w"]], device=v.device), 2), torch.tensor(
            [cb["m0x"], cb["m0y"]], device=v.device), 2), 0.0, 1.0, 2)
        fgr = L.mul(L.mul(L.pow(g[..., 0], cb["m1x"]),
                          L.pow(g[..., 1], cb["m1y"]), 1), cb["m2w"], 1)
        if SC["additive"]:
            a_t = L.mul(a_t, L.sub(1.0, fgr, 1), 1)               # :120
        else:
            col_f = FOG_COLOR.view(*([1] * (rgb.dim() - 1)), 3).expand_as(rgb)
            rgb = L.mix(rgb, col_f, fgr.unsqueeze(-1), 3)
        dist2 = L.length(dsrc, 3)                                 # :130
        a = L.add(L.mul(dist2, cb["m4y"], 1), cb["m4x"], 1)       # :130
        a = L.clamp(L.pow(L.fmax(a, 0.0, 1), cb["m4w"]), 0.0, 1.0, 1)
        b = L.add(L.mul(hz, cb["m5y"], 1), cb["m5x"], 1)          # :130
        b = L.clamp(L.pow(L.fmax(b, 0.0, 1), cb["m5z"]), 0.0, 1.0, 1)
        occ = L.mul(a, b, 1)                                      # :130
        fcu = L.mul(L.clamp(occ, 0.0, 1.0, 1), cb["m7w"], 1)      # :130
        if SC["additive"]:
            a_t = L.mul(a_t, L.sub(1.0, fcu, 1), 1)               # :130
        else:
            col_f = FOG_COLOR.view(*([1] * (rgb.dim() - 1)), 3).expand_as(rgb)
            rgb = L.mix(rgb, col_f, fcu.unsqueeze(-1), 3)
    return rgb, a_t.clamp(0, 1)
