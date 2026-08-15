

def char_selftest():
    """Every csgo_character / csgo_eyeball / csgo_customglove path, at ITS
    OWN DEFAULTS, on a synthetic pixel batch. Prints max|delta| per path
    against that path suppressed and returns non-zero if any SELECTED path
    is dead.

    THIS IS A REACHABILITY PROOF, NOT AN IMAGE METRIC. No ground-truth
    frame is read and nothing is scored. The question it answers is the
    only one that has repeatedly gone wrong here: does the code execute and
    change the output at the defaults the run actually uses.

    IT CAN FAIL, and `--char-selftest-inject <name>` makes it fail on
    purpose by breaking one named path -- CHECKS_THAT_CANNOT_FAIL.md's
    rule. Until each name has been injected and seen to fire, a clean run
    is indistinguishable from a self-test that never ran.
    """
    inj = args.char_selftest_inject
    if inj == "all":
        print("--char-selftest-inject names: " + ", ".join(CHAR_INJECTS))
        return 0
    if inj is not None and inj not in CHAR_INJECTS:
        return f"unknown --char-selftest-inject {inj!r}; " \
               f"names: {', '.join(CHAR_INJECTS)}"
    torch.manual_seed(0)
    B, H, W = 1, 32, 32
    dev = device
    uv = torch.rand(B, H, W, 2, device=dev)
    nv = _nrm(torch.randn(B, H, W, 3, device=dev))
    t4 = torch.cat([_nrm(torch.randn(B, H, W, 3, device=dev)),
                    torch.ones(B, H, W, 1, device=dev)], dim=-1)
    nts = _nrm(torch.randn(B, H, W, 3, device=dev).abs()
               * torch.tensor([0.3, 0.3, 1.0], device=dev))
    wp = torch.randn(B, H, W, 3, device=dev)
    eye3 = torch.zeros(B, 1, 1, 3, device=dev)
    view = _nrm(eye3 - wp)
    l = _nrm(torch.tensor([0.4, 0.8, 0.3], device=dev)).view(1, 1, 1, 3) \
        .expand_as(nv)
    rgb = torch.rand(B, H, W, 3, device=dev) * 0.7 + 0.1
    vc = torch.rand(B, H, W, 4, device=dev)
    ao = torch.ones(B, H, W, device=dev)
    mtex = torch.rand(B, H, W, 4, device=dev)
    gloss = torch.rand(B, H, W, 4, device=dev) * 0.6 + 0.2
    curv = torch.rand(B, H, W, device=dev)
    alpha = torch.rand(B, H, W, device=dev)
    ff = torch.ones(B, H, W, device=dev)
    noise = torch.rand(B, H, W, device=dev)
    uvfw = torch.full((B, H, W), 0.01, device=dev)
    org = torch.zeros(1, 1, 1, 3, device=dev)
    # A synthetic material row: every family column ON, every scalar at a
    # value that is NOT its neutral element, so a path that reads the row
    # and does nothing is visible as a zero delta rather than hidden by a
    # neutral parameter.
    K = MAT_EXT2.shape[1]
    x = torch.zeros(B, H, W, K, device=dev)
    _set = lambda k, v: x[..., EXT2[k]].fill_(v) if k in EXT2 else None
    for _k in EXT2:
        if _k.startswith(("f_char", "has_", "f_glove", "f_eyeball",
                          "f_character", "f_customglove")):
            _set(_k, 1.0)
    for _k, _v in (("ch_curvature_scale", 1.4), ("ch_aniso_amount", 0.7),
                   ("ch_hair_rough_scale", 0.6), ("ch_hair_shift", 0.3),
                   ("ch_hair_transmission", 0.8), ("ch_sheen_scale", 0.9),
                   ("ch_sheen_tint", 0.8), ("ch_irid_strength", 0.8),
                   ("ch_irid_hue_shift", 1.1), ("ch_irid_fresnel", 1.3),
                   ("ch_hue_shift", 0.9), ("ch_saturation", 1.6),
                   ("ch_brightness", 1.2), ("ch_contrast", 1.4),
                   ("ch_rough_bright", 1.3), ("ch_rough_contrast", 1.5),
                   ("ch_reflectance", 0.5), ("ch_opacity_scale", 0.7),
                   ("ch_alpha_ref", 0.4), ("ch_aa_edge_strength", 0.8),
                   ("ch_spawn_invuln", 0.6), ("ch_invuln_r", 1.0),
                   ("ch_invuln_g", 0.4), ("ch_invuln_b", 0.2),
                   ("eye_radius", 0.6), ("eye_iris_size", 0.7),
                   ("eye_pupil_size", 0.3), ("eye_hue_shift", 0.8),
                   ("eye_saturation", 1.5), ("eye_metalness", 0.2),
                   ("eye_reflectance", 0.5),
                   ("gl_wear_progress", 0.6), ("gl_wear_exponent", 1.4),
                   ("gl_l1_detail_aniso", 0.6),
                   ("gl_l1_detail_rough_bright", 1.3),
                   ("gl_l1_detail_nrm_contrast", 1.2),
                   ("gl_l1_damage_max", 1.0),
                   ("gl_l1_damage_edge_rough", 1.0),
                   ("gl_l1_curvature_power", 1.0),
                   ("gl_l1_surface_bright", 1.2),
                   ("gl_l1_surface_contrast", 1.3),
                   ("gl_l1_surface_sat", 1.4),
                   ("gl_l1_substrate_bright", 0.8),
                   ("gl_l1_substrate_contrast", 1.1),
                   ("gl_l1_substrate_sat", 0.7),
                   ("gl_tint1_r", 1.4), ("gl_tint1_g", 0.6),
                   ("gl_tint1_b", 0.9), ("gl_output_mode", 0.0)):
        _set(_k, _v)
    for _i in range(3):
        for _s, _v in (("scale", 2.0), ("rot", 0.4), ("squash", 1.3),
                       ("backing_scale", 0.7), ("highlight", 0.0),
                       ("ou", 0.0), ("ov", 0.0)):
            _set("ch_p%d_%s" % (_i, _s), _v)
    mid = torch.zeros(B, H, W, dtype=torch.long, device=dev)
    rows = []

    def _run():
        return character_surface(rgb, alpha, uv, uv, mid, x, nv, t4, nts,
                                 view, wp, eye3, ff, vc, ao, mtex, gloss,
                                 curv, uvfw, 0.5, org, noise)

    def _delta(name, flag, extra=None):
        """max|delta| of the whole surface set with `flag` on vs off."""
        was = getattr(args, flag)
        setattr(args, flag, False)
        off = _run()
        setattr(args, flag, was)
        on = _run()
        d = 0.0
        for a, b in zip(on[:10], off[:10]):
            if torch.is_tensor(a) and torch.is_tensor(b) \
                    and a.shape == b.shape:
                d = max(d, float((a - b).abs().max()))
        # ctx too: the surface half of S_SUBSURFACE_SCATTERING produces
        # ONLY the mask -- the lobe itself is in char_direct_delta -- so
        # without this the row reads 0 for a path that is alive.
        for a, b in zip(on[10], off[10]):
            if torch.is_tensor(a) and torch.is_tensor(b) \
                    and a.shape == b.shape:
                d = max(d, float((a - b).abs().max()))
        rows.append((name, was, d))
        return d

    _delta("S_SUBSURFACE_SCATTERING (surface half)", "char_sss")
    _delta("S_ANISOTROPIC_GLOSS", "char_aniso")
    _delta("S_ANISOTROPIC_HAIR", "char_hair")
    _delta("F_CLOTH_SHADING", "char_cloth")
    _delta("S_IRIDESCENCE", "char_iridescence")
    _delta("S_RETRO_REFLECTIVE", "char_retro_reflective")
    _delta("S_PATCHES", "char_patches")
    _delta("S_DECAL_TEXTURE", "char_decals")
    _delta("F_BLOOD", "char_blood")
    _delta("S_ENABLE_ADJUSTMENTS", "char_adjustments")
    _delta("S_EYEBALLS (in csgo_character)", "char_eyes")
    _delta("g_flSpawnInvulnerability", "char_invulnerability")
    # --- the DIRECT lobes, which is where aniso/hair/cloth/retro/SSS live
    s = _run()
    r2, f0, cl, rt, hr, ctx = s[3], s[6], s[7], s[8], s[9], s[10]
    n = s[2]
    if inj == "aniso-isotropic":
        r2 = torch.stack([r2.mean(-1), r2.mean(-1)], dim=-1)
    if inj == "hair-dead":
        hr = torch.zeros_like(hr)
    if inj == "cloth-zero":
        cl = torch.zeros_like(cl)
    if inj == "retro-zero":
        rt = torch.zeros_like(rt)
    _lut = None
    if args.char_sss:
        _lut = (lambda q: torch.full(q.shape[:-1] + (4,), 0.5, device=dev)) \
            if inj == "sss-flat" else \
            (lambda q: sample_fam(q, T_DFALLOFF[mid]))
    dd, sd = char_direct_delta(r2, n, l, view, f0, ctx, cl, rt, hr, x, _lut)
    rows.append(("direct DIFFUSE delta (SSS lobe shape)", args.char_sss,
                 float(dd.abs().max())))
    rows.append(("direct SPECULAR delta (aniso/hair/cloth/retro lobe "
                 "shape)", True, float(sd.abs().max())))
    # --- the spherical tangent axis --------------------------------------
    ta, ba = char_spherical_tangent(wp, org, n, t4, torch.cross(
        n, t4[..., :3], dim=-1), "uv")
    tb, bb = char_spherical_tangent(wp, org, n, t4, torch.cross(
        n, t4[..., :3], dim=-1), "spherical")
    rows.append(("S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS",
                 args.char_aniso_tangents == "spherical",
                 float((ta - tb).abs().max())))
    # --- the two extra probe-volume binner pairs -------------------------
    _pc = char_probe_ambient_cube(
        n, torch.rand(B, H, W, 3, device=dev),
        torch.rand(B, H, W, 3, device=dev),
        torch.rand(B, H, W, 3, device=dev))
    _po = char_probe_slice_offsets(n)
    rows.append(("probe-volume ambient cube (2 extra binner pairs)", True,
                 float(_pc.abs().max()) * float(_po.abs().max())))
    # --- MBOIT on player geometry ----------------------------------------
    for _nm in (4, 6):
        _m = char_mboit_moments(alpha, torch.rand(B, H, W, device=dev)
                                * 100 + 1, MBOIT_NEAR, MBOIT_FAR, _nm)
        rows.append((f"D_MBOIT_PASS1 ({_nm} moments)", True,
                     float(max(t.abs().max() for t in _m[1:]))))
    # --- csgo_eyeball -----------------------------------------------------
    _r2e = torch.full((B, H, W, 2), 0.5, device=dev)
    _e_on = eyeball_surface(rgb, _r2e, nv, wp, eye3, uv, mid, x,
                            torch.ones(B, H, W, device=dev))
    _xz = x.clone()
    _xz[..., EXT2["f_eyeball"]] = 0.0
    _e_off = eyeball_surface(rgb, _r2e, nv, wp, eye3, uv, mid, _xz,
                             torch.ones(B, H, W, device=dev))
    if inj == "eye-flat":
        _e_on = _e_off
    rows.append(("csgo_eyeball (ray-sphere iris)", args.eyeball,
                 float(max((a - b).abs().max()
                           for a, b in zip(_e_on[:3], _e_off[:3])))))
    # --- csgo_customglove -------------------------------------------------
    _g_on = customglove_surface(rgb, alpha, uv, mid, x, nts, t4, nv)
    _xg = x.clone()
    _xg[..., EXT2["f_glove"]] = 0.0
    _g_off = customglove_surface(rgb, alpha, uv, mid, _xg, nts, t4, nv)
    if inj == "glove-passthrough":
        _g_on = _g_off
    rows.append(("csgo_customglove (finish compositor)", args.glove,
                 float((_g_on[0] - _g_off[0]).abs().max())))
    for _m in (0, 1, 2, 3, 4):
        _xm = x.clone()
        _xm[..., EXT2["gl_output_mode"]] = float(_m)
        _o = customglove_surface(rgb, alpha, uv, mid, _xm, nts, t4, nv)
        rows.append((f"  F_OUTPUT_MODE {_m}", args.glove,
                     float((_o[0] - _g_off[0]).abs().max())))
    # --- the skinning path -------------------------------------------------
    import cs2_vertex_stage as _vsx
    _n = 4096
    _p = torch.rand(_n, 3, device=dev) * torch.tensor([1., 6., 1.],
                                                      device=dev)
    _bi, _bw, _bs, _ni = _vsx.character_skin_binding(_p, args.char_bones)
    _pal = _vsx.character_bone_palette(_p, args.char_bones, 0.3,
                                       amp=0.9, rate=1.7, device=dev)
    if inj == "skin-identity":
        _pal = _vsx.character_bone_palette(_p, args.char_bones, 0.0,
                                           amp=0.0, rate=0.0, device=dev)
    _ps, _, _ = _vsx.character_skinning(_p, None, None, _bi, _bw, _bs, _ni,
                                        _pal)
    rows.append(("csgo_character SKINNING (4-influence LBS)",
                 args.char_skinning, float((_ps - _p).abs().max())))
    _pal0 = _vsx.character_bone_palette(_p, args.char_bones, 0.0, amp=0.0,
                                        rate=0.0, device=dev)
    _p0, _, _ = _vsx.character_skinning(_p, None, None, _bi, _bw, _bs, _ni,
                                        _pal0)
    print("csgo_character / csgo_eyeball / csgo_customglove self-test "
          "(max|delta| of the path against the path suppressed):",
          flush=True)
    dead = []
    for nm, sel, d in rows:
        mark = "" if d > 0 else ("   <-- DEAD" if sel else "   (not selected)")
        print(f"  {'ON ' if sel else 'off'}  {d:12.6g}  {nm}{mark}",
              flush=True)
        if sel and d == 0:
            dead.append(nm)
    print(f"  identity bone palette -> max|dpos| "
          f"{float((_p0 - _p).abs().max()):.3g}  (0 is CORRECT: it means "
          f"bind pose was requested, and the path still ran)", flush=True)
    if inj:
        if dead:
            print(f"INJECTION {inj!r} FIRED: {len(dead)} path(s) went dead "
                  f"-- {', '.join(dead)}. The self-test can fail, so a "
                  f"clean run is a measurement.", flush=True)
            return 0
        print(f"INJECTION {inj!r} DID NOT FIRE. The self-test cannot "
              f"detect a break in that path and must not be trusted for "
              f"it.", flush=True)
        return 2
    if dead:
        return (f"{len(dead)} SELECTED path(s) reached nothing: "
                + ", ".join(dead))
    return 0


if args.char_selftest:
    raise SystemExit(char_selftest())
if args.gt_lighting_selftest:
    raise SystemExit(gt_lighting_selftest())
if args.lfb_reach:
    # fam_loopfam_b was IMPORTED AND NEVER CALLED. Not one `_lfb.` call
    # site existed anywhere in this file, so five transcribed families
    # and all 22 --lfb-* flags were resident and unreachable, and the
    # only thing that noticed was a counter reading 0 -- a detector
    # built and then handed nothing to detect.
    #
    # This harness is the module's OWN answer to the fact that
    # de_inferno has zero materials in all five families (663/663
    # .vmat_c decompiled, a self-summing partition): it runs every entry
    # point over a synthesised surface at every value of every combo
    # axis and fails when an axis moves nothing. It is a reachability
    # proof, NOT a claim these families shade any pixel of this map.
    _lfb_dead, _lfb_rows = _lfb.reach_selftest(
        inject=args.lfb_reach_inject)
    LFB_STATE["axes"] = len(_lfb_rows)
    LFB_STATE["dead"] = int(_lfb_dead)
    print(f"lfb reach: {len(_lfb_rows)} axis arms executed across "
          f"{len(_lfb.SHADERS)} families, {int(_lfb_dead)} dead",
          flush=True)
    raise SystemExit(int(_lfb_dead))

ffmpeg = None
if args.png_dir:
    import os as _os0
    _os0.makedirs(args.png_dir, exist_ok=True)
if SESSION is not None and not args.benchmark and not args.session_out:
    raise SystemExit(
        "REFUSING: --session renders many agents and --out names ONE mp4, so "
        "all ten agents would be muxed into a single video in pose order and "
        "silently look like one very confused player. Pass --session-out DIR "
        "for the per-agent tensor sink, or --benchmark to render and discard.")
if SESSION is not None:
    args.benchmark = True     # the mp4 path is single-camera only
if not args.benchmark:
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_bin = "ffmpeg"
    ffmpeg = subprocess.Popen(
        [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{args.width}x{args.height}",
         "-framerate", str(min(args.fps, 60)), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE)

pinned_bufs = [
    torch.empty((args.batch, args.height, args.width, 3),
                dtype=torch.uint8, pin_memory=True)
    for _ in range(2)
]

SESSION_SINK = None
if SESSION is not None and args.session_out:
    import os as _os1
    _os1.makedirs(args.session_out, exist_ok=True)
    _agl = SESSION["agents"] if isinstance(SESSION, dict) else SESSION
    # One host tensor per agent, sized to its REAL frame count (padding is
    # discarded, never written). Filled by a pinned device->host copy, which
    # the split measured at 0.29 ms/frame against PNG's 76.93.
    SESSION_SINK = []
    # OWNER RULE (2026-08-13): the units of currency are demofiles and
    # demovideos -- rendering means WRITING A RENDERER, not materializing
    # dense uncompressed artifacts. The sink STREAMS each agent's frames
    # into its own ffmpeg mp4 as batches complete (schedule guarantees
    # agent-contiguity, so frames arrive in order); no host frame buffer,
    # no .pt tensors, ever. Under an FSR preset args.width/height hold
    # the INTERNAL raster size (03_part mutates them; display size is in
    # FSR_OUT_PRE) and frames leave the post chain display-sized.
    _sw, _sh = (globals().get("FSR_OUT_PRE") or (args.width, args.height))
    for _ai, _a in enumerate(_agl):
        _n = sum(1 for i in range(len(rows))
                 if ROW_AGENT[i] == _ai and not ROW_PAD[i])
        SESSION_SINK.append(dict(
            id=str(_a.get("id", _ai)), n=_n, written=0, ff=None, path=None))
    print(f"session sink: {len(SESSION_SINK)} agents, "
          f"{sum(d['n'] for d in SESSION_SINK)} frames, STREAMING one mp4 "
          f"per agent at {_sw}x{_sh} (no dense frame materialization)",
          flush=True)

warm = min(args.batch, len(rows))
# THE WARM-UP RENDERS REAL FRAMES AND THEN THROWS THEM AWAY, so every
# per-frame line inside render_window is emitted TWICE for the frames it
# covers -- once here, once for real. This file already knew that about the
# FRAMES (GBUF.clear() below, with its comment) and not about the PRINTS,
# so the reports inherited a duplication the pixels were protected from:
# VIEWPUNCH, PLAYERMODELS, VIEWMODEL, the [ao-page-product] ack, and any
# per-frame line a future lane adds.
#
# Suppressed at the SOURCE rather than deduped per report. A report-side
# guard -- which is what VIEWPUNCH shipped with -- makes each line correct
# and still leaves the next one wrong by default, and "remember to dedupe"
# is the kind of rule this session has watched fail. WARMING_UP is the one
# fact the reports need and none of them has to know why.
WARMING_UP[0] = True
with stage("startup.warmup_render"):
    _ = render_window(mvps[:warm], eyes[:warm], FWD[:warm], TIME_S[:warm])
    torch.cuda.synchronize()
WARMING_UP[0] = False
GBUF.clear()      # the warm-up pass would otherwise duplicate every frame
MATID_DUMP.clear()   # ditto -- and at batch < 48 the duplicates are a
# DIFFERENT subset of poses, so the dump silently misaligns with the
# frames it is meant to index rather than harmlessly repeating them.
if args.probe_rect:
    print(f"probe {args.probe_rect}: matid, coverage, mean blend weight",
          flush=True)
    for _m, _f, _w in PROBE:
        print(f"    matid {_m:4d}  {_f:6.1%}  w={_w:.3f}", flush=True)
if args.rt_reach:
    # Measured on the WARM-UP pass, which shades the same geometry the run
    # does. The point of this table is that a per-view render target whose
    # read never executes reads as tested; here it reads as 0.000%.
    print("engine per-view render-target reach (fraction of SHADED pixels "
          "at which the read executes), set 1 / binding 3:", flush=True)
    for _k in ("off76_dirocc", "off80_depthgather", "off88_ssshadow",
               "off116_ripple_a", "off116_ripple_b"):
        _hit, _tot = RT_REACH[_k]
        print(f"    {_k:22s} {(_hit / _tot if _tot else 0.0):9.4%}"
              f"   ({int(_hit):,} / {int(_tot):,} px)", flush=True)
for _k in RT_REACH:
    RT_REACH[_k] = [0.0, 0.0]
# --- transparency-axis reachability -----------------------------------
# Printed whenever the flag is on, never conditioned on a --*-stats knob.
# A path that affects zero pixels has to SAY so: --ibl-lod-scale defaulted
# to 1.0 on this renderer and made mips 2-6 unreachable by construction,
# and every arm evaluated before that read as tested.
# --- the four newly-wired modules: did they RUN? ----------------------
# A module that executes and reports nothing cannot be told apart from
# one that was never imported, which is exactly how ~6,100 lines of
# transcribed code sat unreachable until now. Reported unconditionally.
print("newly-wired module reach: "
      f"msaa={MSAA_STATE.get('built', 0)} "
      f"gbuffer={GBUF_STATE.get('packed', 0)} "
      f"b3filters={B3_STATE.get('ran', 0)} "
      f"lfb={sum(LFB_STATE.values()) if LFB_STATE else 0}", flush=True)
if MSAA_STATE.get("samples"):
    print(f"  msaa: {MSAA_STATE['samples']}-sample attachment from "
          f"{MSAA_STATE.get('source')}", flush=True)
if MSAA_STATE.get("note"):
    print(f"  msaa: {MSAA_STATE['note']}", flush=True)
if not LFB_STATE:
    # Its 0 does NOT mean the same thing as the other three modules' 0,
    # and the reach line cannot say which -- so it is said here. The
    # five loop-family entry points have NO per-material dispatch: they
    # are reached today only through --lfb-reach, which proves they
    # execute over a synthesised surface. A MAT_FAM dispatch is the
    # missing piece, and it is blocked on two separate things, both
    # named rather than worked around:
    #   1. MAT_FAM exists only when FAMILY is on, i.e. only with a
    #      --fam-side side table, which does not yet exist rebuilt.
    #   2. de_inferno carries ZERO materials in all five families
    #      (663/663 .vmat_c decompiled), so on THIS map such a dispatch
    #      would correctly shade nothing and could not be validated.
    print("  lfb: 0 means NO PER-MATERIAL DISPATCH, not 'reached no "
          "pixel'. The five families execute under --lfb-reach; a "
          "MAT_FAM dispatch needs --fam-side (absent) and a map that "
          "has these families (de_inferno has none of the five).",
          flush=True)
for _k, _st in (("msaa", MSAA_STATE), ("gbuffer", GBUF_STATE),
                ("b3filters", B3_STATE)):
    if _st.get("error"):
        # Loud rather than swallowed: the brief is that code which runs
        # and corrupts the view buffer beats code omitted for looking
        # risky, but a silent exception is neither.
        print(f"  {_k}: RAISED {_st['error']}", flush=True)

if any((args.opaque_fade, args.translucent_clip,
        args.disable_translucent_clip, args.cubemap_refraction,
        args.translucent, args.additive_blend, args.alpha_test_prepass,
        MBOIT_N)):
    print("transparency-axis reach:", flush=True)
if args.opaque_fade:
    _d, _n = FADE_STAT
    print(f"    D_OPAQUE_FADE          amount {args.opaque_fade_amount:g}, "
          f"dithered away {_d/max(_n,1):.4%} of {int(_n):,} candidate px",
          flush=True)
if args.alpha_test_prepass:
    print("    D_ALPHA_TEST_PREPASS   coverage form active on every "
          "MASK-class pixel (it REPLACES the binary test)", flush=True)
if args.translucent_clip and not args.disable_translucent_clip:
    _c, _n = CLIP_STAT
    print(f"    S_MODE_DEPTH clip      threshold "
          f"{args.translucent_clip_threshold:g}, clipped {_c/max(_n,1):.4%} "
          f"of {int(_n):,} glass px", flush=True)
if args.disable_translucent_clip:
    print("    D_DISABLE_TRANSLUCENT_CLIP  clip module is the 9-line "
          "no-op; every glass fragment writes depth", flush=True)
if args.cubemap_refraction:
    _r, _n = REFRACT_STAT
    print(f"    S_OPAQUE_CUBEMAP_REFRACTION  {_r/max(_n,1):.4%} of "
          f"{int(_n):,} BLEND px refracted (eta {1.0/args.refract_ior:.8g})",
          flush=True)
if args.translucent or args.additive_blend:
    _nt = int((MAT_EXT2[:, EXT2["f_translucent"]] > 0.5).sum()) \
        if args.translucent else 0
    _na = int((MAT_EXT2[:, EXT2["f_additive"]] > 0.5).sum()) \
        if args.additive_blend else 0
    print(f"    S_TRANSLUCENT {_nt} materials, S_ADDITIVE_BLEND {_na} "
          f"materials carry the flag", flush=True)
if MBOIT_N:
    print(f"    D_MBOIT ({MBOIT_N} moments)  peel reached "
          f"{MBOIT_STAT[0]} layers (cap {args.mboit_layers}); "
          f"{MBOIT_STAT[2]}/{MBOIT_STAT[1]} chunks hit the cap"
          + ("  <-- TRUNCATED, raise --mboit-layers"
             if MBOIT_STAT[2] else ""), flush=True)
if any(v[1] for v in SF_REACH.values()):
    print("small-family REACH (fraction of covered pixels each path OWNED):",
          flush=True)
    for _k, (_a, _b) in SF_REACH.items():
        if _b:
            print(f"    {_k:14s} {_a / _b:9.5%}  ({int(_a):,} of {int(_b):,} "
                  "covered pixels)", flush=True)
    if not any(v[0] for v in SF_REACH.values()):
        print("    every enabled path owned ZERO pixels -- a family that "
              "is on the map but off the screen in these poses, or a pack "
              "without the family. That is reported, not passed over.",
              flush=True)

if args.sf_flop_audit or args.sf_flop_audit_mutate:
    print("\n--- the exact-cost audit -----------------------------------",
          flush=True)
    print("csgo_black_unlit r0_m0 and csgo_effects r10_m0..m3 contain no "
          "loop region, so 132/1 and 194/5 are EXACT per-invocation "
          "figures. With the 24 S_MODE_DEPTH modules they are the only "
          "exact figures in the corpus; everything else is a floor.",
          flush=True)
    _ok, _lines = sf_flop_audit(mutate=False)
    for _l in _lines:
        print(_l, flush=True)
    print(f"  audit: {'PASS' if _ok else 'FAIL'}", flush=True)
    if args.sf_flop_audit_mutate:
        print("\n  MUTATION (S_DEPTH_FEATHER term deleted) -- this MUST "
              "fail, or the audit is measuring nothing:", flush=True)
        _mok, _mlines = sf_flop_audit(mutate=True)
        for _l in _mlines[:3]:
            print("  " + _l, flush=True)
        print(f"    mutated audit: {'PASS (BAD)' if _mok else 'FAIL (good)'}",
              flush=True)
        if _mok:
            raise SystemExit("the mutated transcription still priced "
                             "identically: the ledger is not wired to the "
                             "code it claims to price")
    if not _ok:
        raise SystemExit(1)

if args.sf_selftest:
    print("\n--- small-family reachability at DEFAULTS -------------------",
          flush=True)
    _rows = sf_selftest()
    _zero = []
    for _n, _d in _rows:
        _mark = ""
        if not _n.startswith("  "):
            _mark = "  <-- ZERO, the path is inert" if _d == 0.0 else ""
            if _d == 0.0:
                _zero.append(_n)
        print(f"    {_n:56s} max|delta| {_d:.6g}{_mark}", flush=True)
    # the parser replay -- see the block header near --sf-effects
    _p2 = argparse.ArgumentParser(add_help=False)
    _seen, _conf = set(), []
    for _a in parser._actions:
        for _o in _a.option_strings:
            if _o in _seen:
                _conf.append(_o)
            _seen.add(_o)
        try:
            _p2._add_action(_a)
        except argparse.ArgumentError as _e:
            _conf.append(str(_e))
    print(f"    parser replay: {len(parser._actions)} actions, "
          f"{len(_conf)} conflicts{'' if not _conf else ': ' + str(_conf)}",
          flush=True)
    if _zero or _conf:
        raise SystemExit(
            f"{len(_zero)} inert path(s) and {len(_conf)} flag conflict(s): "
            f"{_zero[:4]}")
    print("    all small-family paths move pixels at their own defaults, "
          "0 flag conflicts", flush=True)

if args.fam_reach and FAM_HIST:
    _h = torch.stack(FAM_HIST).mean(0)
    print("shader-family reach (fraction of SHADED pixels whose final "
          "owning surface is that family):", flush=True)
    for _i in _h.argsort(descending=True).tolist():
        if float(_h[_i]) <= 0:
            continue
        print(f"    {FAM_NAMES[_i]:34s} {float(_h[_i]):8.4%}", flush=True)
    print(f"    distinct materials covering >=1 opaque pixel: "
          f"{len(MAT_SEEN)} of {len(MAT_FAM)}", flush=True)
    FAM_HIST.clear()
# Run AFTER the frames, so the per-family pixel counts it prints are the
# ones the render actually produced and not a promise about them.
lmg_family_selftest()
if args.simple_reach or SIMPLE_TAKEN:
    # THE REACHABILITY EVIDENCE for the two families. Not "the function
    # exists" and not "the flag defaults to 1" -- the count of pixels
    # that executed each path and each of its combo axes. A path that
    # shipped and never ran shows up here as 0 and there is nowhere for
    # it to hide.
    print("=" * 66, flush=True)
    print("csgo_simple / csgo_simple_3layer_parallax reach "
          "(pixel-executions, summed over every shaded batch):", flush=True)
    if not SIMPLE_TAKEN:
        print("    NOTHING RAN. Either --fam-side was not given, or "
              "--simple-shading/--s3lp-shading were set to 0, or no pose "
              "in this run saw either family. The first two are "
              "configuration; the third is a property of the camera set. "
              "Do not read this as 'implemented'.", flush=True)
    for _k in sorted(SIMPLE_TAKEN):
        print(f"    {_k:38s} {SIMPLE_TAKEN[_k]:12d}", flush=True)
    for _k in ("dispatch.simple_pixels", "dispatch.s3lp_pixels"):
        if SIMPLE_TAKEN.get(_k, 0) == 0:
            print(f"    WARNING {_k} == 0: that family's shading path was "
                  f"never entered in this run.", flush=True)
    # An axis that is FORCED to 1 and still counts 0 is not a bug in the
    # dispatch -- it is the slot gate refusing to sample a fallback
    # sentinel as if it were data. Saying so is the difference between a
    # 0 that means "unreachable" and a 0 that means "this map binds no
    # such texture"; without this line the two are indistinguishable and
    # the second reads as the first.
    for _flag, _key, _slot in (
            (args.s3lp_second_layer_cubemap, "s3lp.second_layer_cubemap",
             "g_tLayer2Cubemap"),
            (args.s3lp_tint_mask, "s3lp.tint_mask", "g_tTintMask"),
            (args.simple_metalness_texture, "simple.metalness_texture",
             "g_tAmbientOcclusion"),
            (args.s3lp_metalness_texture, "s3lp.metalness_texture",
             "g_tAmbientOcclusion")):
        if _flag == "1" and SIMPLE_TAKEN.get(_key, 0) == 0 \
                and SIMPLE_TAKEN.get("dispatch.s3lp_pixels", 0) \
                + SIMPLE_TAKEN.get("dispatch.simple_pixels", 0) > 0:
            print(f"    NOTE {_key} is 0 with the axis FORCED to 1: no "
                  f"visible material of that family binds {_slot}, and the "
                  f"axis is gated on the slot so the fallback sentinel "
                  f"cannot reach the shading. The arithmetic of this axis "
                  f"is UNEXERCISED on this map.", flush=True)
    print("=" * 66, flush=True)
# --- NON-WORLD FAMILY REACH ------------------------------------------
# Printed whenever either pass ran, never conditioned on a --*-stats
# knob. A pass that affects zero pixels has to SAY so: this is the
# --ibl-lod-scale shape, and it is why FAILURE is an exit condition here
# rather than a note.
if args.decal_project == "on" or args.sprite_particles == "on":
    _tot = max(NW_PIX[2], 1.0)
    print("non-world family reach (fraction of OUTPUT pixels the pass "
          "wrote), csgo_core:", flush=True)
    print(f"    FAM_PROJECTED_DECALS  id={FAM_PROJECTED_DECALS:<3d} "
          f"{NW_PIX[0] / _tot:9.4%}   ({int(NW_PIX[0]):,} px)", flush=True)
    print(f"    FAM_SPRITECARD        id={FAM_SPRITECARD:<3d} "
          f"{NW_PIX[1] / _tot:9.4%}   ({int(NW_PIX[1]):,} px)", flush=True)
    if NW_DEPTH_ERR:
        _mx = max(e[0] for e in NW_DEPTH_ERR)
        _n = sum(e[1] for e in NW_DEPTH_ERR)
        _mdn = max(e[2] for e in NW_DEPTH_ERR)
        print(f"    depth-buffer world reconstruction: median|delta| "
              f"{_mdn:.6g}, max|delta| {_mx:.6g} world units over "
              f"{_n:,} covered px (the decal pass's own unprojection vs the "
              f"rasteriser's interpolated position; NOT an identity claim). "
              f"READ THE MEDIAN: the max is set by far-plane and silhouette "
              f"pixels that are ill-conditioned under any convention and it "
              f"barely moves, which is why it could not see the y-origin "
              f"defect that left this pass writing nothing.", flush=True)
    if args.decal_reach or args.sprite_reach:
        print("    per-arm reach:", flush=True)
        for _k, _hit, _t in _nw.reach_rows():
            print(f"      {_k:44s} {_hit / max(_t, 1.0):9.4%}  "
                  f"({int(_hit):,} px)", flush=True)
    _dead = _nw.reach_failures()
    if args.nonworld_selftest_inject:
        print(f"FAULT INJECTED: --nonworld-selftest-inject "
              f"{args.nonworld_selftest_inject}", flush=True)
        _fired = bool(_dead) or (
            args.nonworld_selftest_inject == "decal-dead"
            and NW_PIX[0] <= 0) or (
            args.nonworld_selftest_inject == "sprite-dead"
            and NW_PIX[1] <= 0) or (
            args.nonworld_selftest_inject == "depth-remap"
            and NW_DEPTH_ERR
            # The MEDIAN, not the max. The max is ~723 world units on a
            # CORRECT run (far-plane and silhouette pixels), so a threshold of
            # 1.0 against it fired on every run ever made, injected or not --
            # a fault-injection check that always reports success is the
            # CHECKS_THAT_CANNOT_FAIL class, and it was guarding the very
            # defect it could not see.
            and max(e[2] for e in NW_DEPTH_ERR) > 1.0)
        if _fired:
            print("INJECTION FIRED: this run PROVES the non-world reach "
                  "check can fail.", flush=True)
        else:
            print("INJECTION DID NOT FIRE. The check cannot see this "
                  "path, so its reachability claim does not cover it. "
                  "That is a defect in the CHECK.", flush=True)
    elif _dead:
        print(f"FAIL: {len(_dead)} selected combo arm(s) reached zero "
              f"pixels -- {', '.join(_dead)}", flush=True)
    if args.decal_project == "on" and NW_PIX[0] <= 0 \
            and not args.nonworld_selftest_inject:
        print("FAIL: --decal-project on wrote no pixels. 'Defined but "
              "never called' does not count; use --decal-synth N to put "
              "a projection volume in front of the eye.", flush=True)
    if args.sprite_particles == "on" and NW_PIX[1] <= 0 \
            and not args.nonworld_selftest_inject:
        print("FAIL: --sprite-particles on wrote no pixels. Use "
              "--sprite-synth N.", flush=True)
if args.light_dump and LDUMP:
    import numpy as _np3
    _d = LDUMP[0]
    _np3.savez_compressed(args.light_dump, albedo=_d[0], indirect=_d[1],
                          direct=_d[2], fg=_d[3])
    print(f"light dump -> {args.light_dump} ({_d[0].shape[0]} frames)",
          flush=True)
    LDUMP.clear()
    args.light_dump = None
if args.probe_dump and DUMP:
    import numpy as _np2
    _np2.savez_compressed(
        args.probe_dump,
        mid=_np2.concatenate([d[0] for d in DUMP]),
        w_color0=_np2.concatenate([d[1] for d in DUMP]),
        w_vpaint=(_np2.concatenate([d[2] for d in DUMP])
                  if DUMP[0][2] is not None else _np2.zeros(0)),
        fg=_np2.concatenate([d[3] for d in DUMP]))
    print(f"probe dump -> {args.probe_dump} "
          f"({sum(d[0].shape[0] for d in DUMP)} frames)", flush=True)
    DUMP.clear()
    args.probe_dump = None
if args.gt_axes_reach:
    _n = GT_REACH.pop("_n", 1.0)
    print("material-surface combo-axis reach over the shaded frame:",
          flush=True)
    for _k, _v in sorted(GT_REACH.items(), key=lambda kv: -kv[1]):
        print(f"    {_k:24s} {_v / _n:7.3%} of pixels", flush=True)
    GT_REACH.clear()
if args.matid_reach:
    _n = REACH.pop("_n", 1.0)
    print("vmat feature reach over the shaded frame:", flush=True)
    for _k, _v in sorted(REACH.items(), key=lambda kv: -kv[1]):
        print(f"    {_k:26s} {_v / _n:7.3%} of pixels", flush=True)
    REACH.clear()
HDR_DUMP.clear()
# --- viewmodel-pass and csgo_weapon axis reachability -----------------
# Printed whenever the pass ran, never conditioned on a --*-stats knob.
# de_inferno has no weapon in it, so a pass that covers zero pixels is
# the EXPECTED state without --viewmodel-synth -- and it has to say so,
# or "nothing to render" and "never called" become the same output.
if args.viewmodel or args.viewmodel_reach:
    print(f"viewmodel pass: {WORLD_RANGE!r}", flush=True)
    print(f"viewmodel pass: {VM_RANGE!r}", flush=True)
    _rng_note = ("its own -- composited by coverage, world depth never "
                 "read" if args.viewmodel_depth_range == "own"
                 else "COLLAPSED into the world range (DIAGNOSTIC)")
    print(f"viewmodel pass: depth range in use = "
          f"{args.viewmodel_depth_range} ({_rng_note})", flush=True)
    print("viewmodel pass: geometry source = "
          + (f"--viewmodel-model {args.viewmodel_model}"
             if args.viewmodel_model
             else f"--viewmodel-synth {args.viewmodel_synth}"), flush=True)
    _cov = VM_REACH.get("coverage", (0.0, 0.0))
    _com = VM_REACH.get("composited", (0.0, 0.0))
    print(f"    rasterised coverage  {_cov[0] / max(_cov[1], 1):9.4%}  "
          f"({int(_cov[0]):,} / {int(_cov[1]):,} px)", flush=True)
    print(f"    survived to composite{_com[0] / max(_com[1], 1):9.4%}  "
          f"({int(_com[0]):,} / {int(_com[1]):,} px)", flush=True)
    for _k in ("legs_cone", "legs_faded_out", "legs_kept"):
        if _k in VM_REACH:
            _h, _t = VM_REACH[_k]
            print(f"    csgo_legs_prepass {_k:12s} "
                  f"{_h / max(_t, 1):9.4%}  ({int(_h):,} / {int(_t):,} px)",
                  flush=True)
    if _cov[0] == 0.0:
        print("    csgo_weapon / csgo_legs_prepass reached 0 px. On "
              "de_inferno that is a USAGE fact -- the world pack contains "
              "no weapon material at all -- not a capability fact. Rerun "
              "with --viewmodel-synth both to submit a synthesised draw "
              "through the same dispatch.", flush=True)
if args.weapon_axes_reach:
    print("csgo_weapon combo-axis reach over the viewmodel pass:",
          flush=True)
    if not WPN_REACH:
        print("    no axis recorded -- the pass shaded 0 weapon pixels",
              flush=True)
    for _k, (_h, _t) in sorted(WPN_REACH.items(), key=lambda kv: -kv[1][0]):
        print(f"    {_k:34s} {_h / max(_t, 1):9.4%}  "
              f"({int(_h):,} / {int(_t):,} px)", flush=True)
    if WPN_TRACE:
        print(f"    D_MOUSE_TRACE_COORD readback: {WPN_TRACE}", flush=True)
    if WPN_UNIMPL:
        print("  axes SELECTED BUT NOT IMPLEMENTED -- no fraction is "
              "printed for these, because there is no path to take:",
              flush=True)
        for _k, _why in sorted(WPN_UNIMPL.items()):
            print(f"    {_k:34s} {_why}", flush=True)

start = time.time()
render_time = 0.0
frames_done = 0
pending = None
_PNG_SUBDIRS = set()
if SESSION is not None and args.unclamped_session_batches:
    print("DIAGNOSTIC --unclamped-session-batches: batches MAY span agents. "
          "This reproduces #71 and the frames are WRONG on purpose.",
          flush=True)
    # 71 MEASURED (t71b): the dominant carrier is the chunk-mean-eye LOD
    # distance (:28813) -- solo-vs-crowd max|d| 228 falls to 14 with
    # --lod-per-frame (which costs ~6 ms per rebuild at 640x360). The
    # residual 14 is a SECOND, unbisected batch-global term, so
    # per-frame LOD makes this mode CLOSER to self-contained, not
    # self-contained. Print the pairing state either way.
    if not getattr(args, "lod_per_frame", False):
        print("⚠️  --unclamped-session-batches WITHOUT --lod-per-frame: "
              "every agent in a spanning batch takes LOD from the batch "
              "MEAN eye -- a position no agent occupies (max|d| 228 "
              "measured). Pass --lod-per-frame to remove the dominant "
              "term; a residual max|d|~14 batch-global term remains "
              "either way (#71, unbisected).", flush=True)
    _SCHED = [(b, min(args.batch, len(rows) - b))
              for b in range(0, len(rows), args.batch)]
elif SESSION is not None:
    _SCHED = [(a0 + o, min(args.batch, n - o))
              for (a0, n) in AGENT_SPAN
              for o in range(0, n, args.batch)]
    _spans = sorted({b for _, b in _SCHED})
    print(f"session batch schedule: {len(_SCHED)} batches, sizes {_spans} "
          f"(cap --batch {args.batch}); no batch spans two agents",
          flush=True)
else:
    _SCHED = [(b, min(args.batch, len(rows) - b))
              for b in range(0, len(rows), args.batch)]
for index, (begin, _bsz) in enumerate(_SCHED):
    mvp = mvps[begin:begin + _bsz]
    eye = eyes[begin:begin + _bsz]
    # Absolute frame index, so every time-driven vertex axis advances with
    # the video rather than restarting each batch.
    VS_FRAME0[0] = begin
    t0 = time.time()
    fwd = FWD[begin:begin + _bsz]
    out = render_window(mvp, eye, fwd, TIME_S[begin:begin + _bsz])
    torch.cuda.synchronize()
    render_time += time.time() - t0
    B = out.shape[0]
    if SESSION_SINK is not None:
        # DEVICE-RESIDENT PATH: one contiguous copy per (batch, agent) run
        # straight into the consumer's tensor. No PNG, no encoder, no
        # per-frame python.
        with stage("io.sink_d2h"):
            _b0 = 0
            while _b0 < B:
                _ai = ROW_AGENT[begin + _b0]
                _b1 = _b0
                while _b1 < B and ROW_AGENT[begin + _b1] == _ai:
                    _b1 += 1
                _keep = [k for k in range(_b0, _b1)
                         if not ROW_PAD[begin + k]]
                if _keep:
                    _d = SESSION_SINK[_ai]
                    _f0 = ROW_FIDX[begin + _keep[0]]
                    if _f0 != _d["written"]:
                        raise SystemExit(
                            f"session sink {_d['id']}: frame {_f0} arrived "
                            f"after {_d['written']} written -- out-of-order "
                            f"delivery would silently scramble the video")
                    if _d["ff"] is None:
                        import imageio_ffmpeg
                        import subprocess as _sp
                        _d["path"] = f"{args.session_out}/{_d['id']}.mp4"
                        _d["ff"] = _sp.Popen(
                            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error",
                             "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{_sw}x{_sh}", "-r", str(args.fps),
                             "-i", "-", "-pix_fmt", "yuv420p", _d["path"]],
                            stdin=_sp.PIPE)
                    _ck = out[_keep[0]:_keep[0] + len(_keep)]
                    if _ck.shape[1] != _sh or _ck.shape[2] != _sw:
                        raise SystemExit(
                            f"session sink {_d['id']}: frame "
                            f"{_ck.shape[2]}x{_ck.shape[1]} != declared "
                            f"{_sw}x{_sh} -- a resolution the mux was not "
                            f"opened for")
                    with stage("io.sink_mux"):
                        _cpu = _ck.to(torch.uint8).cpu()
                        # per-agent screen-space HUD, drawn by the
                        # renderer into the demovideo itself (89_hud)
                        _cpu = hud_apply(_cpu, _ai, _f0)
                        _d["ff"].stdin.write(_cpu.numpy().tobytes())
                    _d["written"] += len(_keep)
                    if _d["written"] == _d["n"]:
                        _d["ff"].stdin.close()
                        if _d["ff"].wait() != 0:
                            raise SystemExit(
                                f"session sink {_d['id']}: ffmpeg exited "
                                f"nonzero -- the video is not trustworthy")
                        print(f"session sink wrote {_d['path']}: "
                              f"{_d['n']} frames streamed at agent "
                              f"boundary", flush=True)
                        _d["ff"] = None
                _b0 = _b1
    if args.png_dir:
        from PIL import Image as _PIm
        # Two different costs, and they were one number: a blocking
        # device->host copy of B*H*W*3 bytes, then a single-threaded PIL
        # deflate per frame. Both sit on the critical path between this
        # batch's raster and the next batch's submit.
        with stage("io.png_d2h"):
            cpu = out.cpu().numpy()
        with stage("io.png_encode"):
            for b in range(B):
                if ROW_PAD[begin + b]:
                    continue
                _gi = ROW_FIDX[begin + b] if SESSION is not None \
                    else frames_done + b
                if args.png_every > 1 and _gi % args.png_every:
                    continue
                _sub = (f"/a{ROW_AGENT[begin + b]:02d}"
                        if SESSION is not None else "")
                if _sub and _sub not in _PNG_SUBDIRS:
                    import os as _os2
                    _os2.makedirs(args.png_dir + _sub, exist_ok=True)
                    _PNG_SUBDIRS.add(_sub)
                _PIm.fromarray(cpu[b]).save(
                    f"{args.png_dir}{_sub}/f{_gi:05d}.png")
    if ffmpeg is not None:
        if pending is not None:
            view, event = pending
            with stage("io.mux_wait"):
                event.synchronize()
            with stage("io.mux_write"):
                ffmpeg.stdin.write(view.numpy().tobytes())
        with stage("io.mux_d2h"):
            buf = pinned_bufs[index % 2]
            buf[:B].copy_(out, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            pending = (buf[:B], event)
    frames_done += B

if SESSION_SINK is not None:
    torch.cuda.synchronize()
    for _d in SESSION_SINK:
        if _d["written"] != _d["n"]:
            raise SystemExit(
                f"session sink {_d['id']}: wrote {_d['written']} of "
                f"{_d['n']} frames -- the routing dropped frames, and a "
                f"short video would read as if the agent stopped moving")
        if _d["ff"] is not None:
            raise SystemExit(
                f"session sink {_d['id']}: mux still open at finalize -- "
                f"the agent-boundary close never fired")

if ffmpeg is not None and pending is not None:
    view, event = pending
    with stage("io.mux_wait"):
        event.synchronize()
    with stage("io.mux_write"):
        ffmpeg.stdin.write(view.numpy().tobytes())

if args.hdr_dump is not None:
    import numpy as _np
    import os as _os
    _os.makedirs(args.hdr_dump, exist_ok=True)
    k = 0
    for chunk in HDR_DUMP:
        arr = chunk.cpu().numpy()
        for b in range(arr.shape[0]):
            _np.save(f"{args.hdr_dump}/{k:05d}.npy", arr[b])
            k += 1
    print(f"hdr-dump: wrote {k} frames to {args.hdr_dump}", flush=True)

if args.matid_dump:
    import numpy as _np1
    import os as _os1
    _os1.makedirs(args.matid_dump, exist_ok=True)
    _k = 0
    for _chunk in MATID_DUMP:
        _a = _chunk.numpy().astype(_np1.uint16)
        for _b in range(_a.shape[0]):
            _np1.save(f"{args.matid_dump}/{_k:05d}.npy", _a[_b])
            _k += 1
    print(f"matid-dump: wrote {_k} frames to {args.matid_dump} "
          f"(batch {args.batch}; valid only for frames from this run)",
          flush=True)
    # REFUSE on zero. "wrote 0 frames" followed by exit 0 is a request
    # that could not be satisfied reported as a request that was, and a
    # downstream attribution run then reads an empty directory. A dump
    # that produced nothing is a failure, not an empty result.
    if _k == 0:
        raise SystemExit(
            f"--matid-dump {args.matid_dump} produced ZERO frames. The "
            f"per-pixel owner was never built, so nothing was written; "
            f"this is a failed run, not an empty one.")

if args.gbuffer:
    import numpy as _np
    import os as _os
    _os.makedirs(args.gbuffer, exist_ok=True)
    k = 0
    for alb, nrm, fg_g, vis_g in GBUF:
        for b in range(alb.shape[0]):
            _np.savez_compressed(
                f"{args.gbuffer}/{k:05d}.npz",
                albedo=alb[b].flip(0).cpu().numpy().astype(_np.float32),
                normal=nrm[b].flip(0).cpu().numpy().astype(_np.float32),
                fg=fg_g[b].flip(0).cpu().numpy().astype(_np.uint8),
                vis=vis_g[b].flip(0).cpu().numpy().astype(_np.float32),
                # _DBG_MATID is a single global overwritten per CHUNK, so
                # indexing it by a batch-relative b throws whenever
                # batch > CHUNK. fit_lighting*.py never reads it.
                matid=_np.zeros((), _np.int32))
            k += 1
    print(f"gbuffer: wrote {k} frames to {args.gbuffer}", flush=True)

if args.lights_dump and LDUMP_ROWS:
    import json as _js
    _rows = {}
    for _r in LDUMP_ROWS:            # last write wins (warm-up re-renders)
        _rows[_r["frame"]] = _r
    _rows = [_rows[k] for k in sorted(_rows)]
    _js.dump({"argv": sys.argv, "gain": args.lights_gain,
              "unit": args.lights_unit, "falloff": args.lights_falloff,
              "angular": args.lights_angular,
              "shadow": bool(args.lights_shadow),
              "sabotage": args.lights_sabotage, "frames": _rows},
             open(args.lights_dump, "w"), indent=1)
    _mm = float(np.mean([r["mean_added"] for r in _rows]))
    _cc = float(np.mean([r["coverage"] for r in _rows]))
    print(f"analytic lights: mean added irradiance {_mm:.5f} over "
          f"{len(_rows)} frames, {_cc:.2%} of geometry pixels reached "
          f"-> {args.lights_dump}", flush=True)

if SKYSTAT is not None and SKYSTAT[1] > 0:
    print(f"skyvis: mean over {int(SKYSTAT[1]):,} rendered geometry "
          f"pixels = {SKYSTAT[0]/SKYSTAT[1]:.4f} (field ref "
          f"{SKYVIS_REF:.4f})", flush=True)
elapsed = time.time() - start
# --- transparency-axis reachability -----------------------------------
# Printed whenever the flag is on, never conditioned on a --*-stats knob.
# A path that affects zero pixels has to SAY so: --ibl-lod-scale defaulted
# to 1.0 on this renderer and made mips 2-6 unreachable by construction,
# and every arm evaluated before that read as tested.
if any((args.opaque_fade, args.translucent_clip,
        args.disable_translucent_clip, args.cubemap_refraction,
        args.translucent, args.additive_blend, args.alpha_test_prepass,
        MBOIT_N)):
    print("transparency-axis reach:", flush=True)
if args.opaque_fade:
    _d, _n = FADE_STAT
    print(f"    D_OPAQUE_FADE          amount {args.opaque_fade_amount:g}, "
          f"dithered away {_d/max(_n,1):.4%} of {int(_n):,} candidate px",
          flush=True)
if args.alpha_test_prepass:
    print("    D_ALPHA_TEST_PREPASS   coverage form active on every "
          "MASK-class pixel (it REPLACES the binary test)", flush=True)
if args.translucent_clip and not args.disable_translucent_clip:
    _c, _n = CLIP_STAT
    print(f"    S_MODE_DEPTH clip      threshold "
          f"{args.translucent_clip_threshold:g}, clipped {_c/max(_n,1):.4%} "
          f"of {int(_n):,} glass px", flush=True)
if args.disable_translucent_clip:
    print("    D_DISABLE_TRANSLUCENT_CLIP  clip module is the 9-line "
          "no-op; every glass fragment writes depth", flush=True)
if args.cubemap_refraction:
    _r, _n = REFRACT_STAT
    print(f"    S_OPAQUE_CUBEMAP_REFRACTION  {_r/max(_n,1):.4%} of "
          f"{int(_n):,} BLEND px refracted (eta {1.0/args.refract_ior:.8g})",
          flush=True)
if args.translucent or args.additive_blend:
    _nt = int((MAT_EXT2[:, EXT2["f_translucent"]] > 0.5).sum()) \
        if args.translucent else 0
    _na = int((MAT_EXT2[:, EXT2["f_additive"]] > 0.5).sum()) \
        if args.additive_blend else 0
    print(f"    S_TRANSLUCENT {_nt} materials, S_ADDITIVE_BLEND {_na} "
          f"materials carry the flag", flush=True)
    if args.translucent:
        print(f"      S_TRANSLUCENT       {TL_STAT[0]/max(TL_STAT[2],1):.4%} "
              f"of {int(TL_STAT[2]):,} BLEND px, alpha actually moved on "
              f"{TL_STAT[1]/max(TL_STAT[0],1):.2%} of them", flush=True)
    if args.additive_blend:
        print(f"      S_ADDITIVE_BLEND    {AD_STAT[0]/max(AD_STAT[1],1):.4%} "
              f"of {int(AD_STAT[1]):,} BLEND px rasterised, "
              f"{int(AD_STAT[2]):,} of those survived the depth test with "
              f"alpha>0.01 and were composited dst+src*a", flush=True)
if args.water_fancy:
    _wp, _wt = WATER_REACH
    _rt = (args.water_reflection_type
           if WATER_OV["reflection_type"] is None
           else str(int(WATER_OV["reflection_type"])))
    print(f"    csgo_water_fancy  water_fancy_shade wrote {int(_wp):,} px "
          f"of {int(_wt):,} BLEND px considered "
          f"({_wp / max(_wt, 1):.4%}); "
          f"S_REFLECTION_TYPE={_rt} "
          f"S_REFRACTION={args.water_refraction} "
          f"S_CAUSTICS={args.water_caustics} "
          f"S_INTERACTION_EFFECTS={args.water_interaction} "
          f"S_BLUR_REFRACTION={args.water_blur_refraction}", flush=True)
    if _wp == 0:
        print("      NOTHING WAS SHADED. Either no csgo_water_fancy "
              "material rasterised into the BLEND class in these poses "
              "or --fam-side carries no such material. This run does NOT "
              "verify the path.", flush=True)
if MBOIT_N:
    print(f"    D_MBOIT ({MBOIT_N} moments)  peel reached "
          f"{MBOIT_STAT[0]} layers "
          f"(cap {args.mboit_layers if args.mboit_layers > 0 else 64}); "
          f"{MBOIT_STAT[2]}/{MBOIT_STAT[1]} chunks hit the cap"
          + ("  <-- TRUNCATED, raise --mboit-layers"
             if MBOIT_STAT[2] else ""), flush=True)
depth_range_partition_check()
_ts.report("NON-WORLD TERM OUTPUT STATISTICS")
gt_gap_epilogue()
gt_value_epilogue()
_t_total = time.time() - _T_PROC0
_startup = sum(v for k, v in STAGE.items() if k.startswith("startup."))
_io = sum(v for k, v in STAGE.items() if k.startswith("io."))
_named = dict(STAGE)
_named["render.raster_shade"] = render_time - STAGE.get("render.lod_rebuild", 0.0)
_named["startup.other"] = (start - _T_PROC0) - _startup
_named["unattributed"] = _t_total - (start - _T_PROC0) - render_time - _io
print("STAGE SPLIT (seconds, and % of process wall clock; the fps pair below "
      "brackets only the batch loop, so `startup.*` is invisible to both)",
      flush=True)
print(f"  {'stage':<24} {'seconds':>10} {'% wall':>8} {'ms/frame':>10}",
      flush=True)
for _k in sorted(_named, key=lambda k: -_named[k]):
    _v = _named[_k]
    if _v <= 0.0005:
        continue
    print(f"  {_k:<24} {_v:>10.2f} {100 * _v / _t_total:>7.1f}% "
          f"{1000 * _v / max(frames_done, 1):>10.2f}", flush=True)
print(f"  {'TOTAL (process)':<24} {_t_total:>10.2f} {100.0:>7.1f}% "
      f"{1000 * _t_total / max(frames_done, 1):>10.2f}", flush=True)
print(f"STAGE JSON: {json.dumps({k: round(v, 4) for k, v in _named.items()})}",
      flush=True)
if NONWORLD:
    # REACHABILITY, MEASURED. Counting pixels that took each path is the
    # only thing that distinguishes "the family shaded nothing" from "the
    # code never ran", and this project has shipped the second as the
    # first before (--ibl-lod-scale). A zero is printed as a zero.
    print("non-world family pixels shaded: "
          + ", ".join(f"{k} {v:,}" for k, v in FAM_HIT.items()), flush=True)
    if sum(FAM_HIT.values()) == 0:
        print("  *** every non-world family path reached ZERO pixels. The "
              "code ran; no geometry selected it. On a de_inferno MAP pack "
              "that is expected -- csgo_character / csgo_eyeball / "
              "csgo_customglove are MODEL materials. Re-run with "
              "--char-assign vertexlit to put them on real geometry. ***",
              flush=True)
    if args.char_skinning:
        print(f"  csgo_character skinning ran on "
              f"{int(_fam_mask(_FAM_CHARACTER).sum()):,} vertices, "
              f"{args.char_bones} bones, amp {args.char_bone_amp}, rate "
              f"{args.char_bone_rate}"
              + ("  (amp and rate are both 0, so the palette is the "
                 "identity and the posed mesh IS the bind pose -- the "
                 "path still executed)"
                 if args.char_bone_amp == 0.0 and args.char_bone_rate == 0.0
                 else ""), flush=True)

print(f"VRAM: peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} "
      f"GiB, peak reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB "
      f"@{args.width}x{args.height} ss{args.supersample} batch {args.batch} "
      f"({frames_done} frames)", flush=True)
print(f"{frames_done} frames: pure-render {frames_done/render_time:.0f} fps, "
      f"end-to-end {frames_done/elapsed:.0f} fps "
      f"@{args.width}x{args.height} ss{args.supersample} "
      f"window {args.batch} chunk {1 if args.lod_per_frame else CHUNK} "
      f"occlusion={'on' if args.occlusion else 'off'} "
      f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
      f"lod={'off' if args.no_lod else 'on'} "
      f"mip={args.mip}"
      + (f" aniso={ANISO_C}/{ANISO_A} bias={args.mip_bias:+.2f} "
         f"gamma={args.mip_gamma} uvconv={args.mip_uvconv}"
         if MIP_ON else ""), flush=True)
# CLASS DEFECTS DECIDE THE EXIT CODE, after the frames are safely written.
#
# The ordering is the whole design. Everything above has already produced
# its PNGs and closed its streams, so a corpus render that hit the invisible
# gun still yields every frame it drew -- usable for the terms that were
# fine, re-runnable for the one that was not. What it does NOT yield is a
# zero exit, so no wrapper, sweep script or CI step can record it as a clean
# render. Dying at the defect would have thrown the batch away and taught
# everyone to pass a suppress flag; passing silently is how the empty hand
# shipped in the first place.
class_matrix_report()
_CLASS_DEFECTS = list(VM_CLASS_DEFECT)
if _CLASS_DEFECTS:
    print("=" * 74, flush=True)
    print(f"CLASS DEFECTS: {len(_CLASS_DEFECTS)} frame(s) drew one class of "
          f"an asset and not another it ships. The frames were written; the "
          f"exit code is 3 so nothing can call this run clean.", flush=True)
    for m in _CLASS_DEFECTS[:5]:
        print("  " + m, flush=True)
    if len(_CLASS_DEFECTS) > 5:
        print(f"  ... and {len(_CLASS_DEFECTS) - 5} more", flush=True)
    print("=" * 74, flush=True)
if ffmpeg is not None:
    ffmpeg.stdin.close()
    _rc = ffmpeg.wait()
    raise SystemExit(_rc or (3 if _CLASS_DEFECTS else 0))
if _CLASS_DEFECTS:
    raise SystemExit(3)
