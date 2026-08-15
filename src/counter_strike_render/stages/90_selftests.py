

def gt_lighting_selftest():
    """Prove every path this file adds is REACHABLE AT ITS OWN DEFAULTS.

    Not an image metric; reads no ground-truth frame. Runs each path on a
    synthetic pixel batch spanning the whole world box, the whole normal
    sphere and roughness 0..1, reports per path how many pixels executed
    it and max|delta| against the path suppressed, and EXITS NON-ZERO if
    any path that the current flags select came back dead. A path with 0
    live pixels, or max|delta| == 0 against its own suppression, is
    switched off by a default -- the --ibl-lod-scale 1.0 failure.

    ####################################################################
    # THIS SELF-TEST HAS NEVER BEEN EXECUTED.
    #
    # It was written on a darwin host with no torch and no CUDA. The
    # renderer needs nvdiffrast's CUDA rasteriser, so not one line below
    # has run. Everything it claims to verify was instead verified by
    # static review, and static review is what found the three inert
    # defaults fixed in 15f86a6 -- so the absence of a run is not
    # theoretical.
    #
    # It must be run on the render host (nkcut2, CUDA + nvdiffrast).
    #
    # AND A CLEAN RUN THERE IS NOT ENOUGH. CHECKS_THAT_CANNOT_FAIL.md
    # instance 5: the instrument built to catch this class was itself in
    # this class, three times over, and was trusted only after known
    # defects were injected and all three fired. Until that is done here,
    # a green run and a run that silently did nothing are the same
    # object. The step, in full:
    #
    #   for n in csm-dead pcf-collapse dfg-zero probe-zero \
    #            distfade-full speccube-zero ambientsh-zero lightmap-zero
    #   do
    #     python3 gpu_render.py <canonical args> --gt-lighting-selftest \
    #             --gt-selftest-inject $n
    #     # each MUST exit non-zero and name $n's path in the FAIL rows
    #   done
    #
    # Any name that exits 0 is a path this self-test cannot see, and the
    # reachability claim does not cover it. Delete this banner only when
    # all eight have been seen to fire, and say so in the commit.
    #
    # AND CHECK *WHICH* ROW FIRED, not merely that something did -- an
    # injection tripping an unrelated row is the same failure wearing a
    # green hat. Expected FAIL rows per name:
    #
    #   csm-dead          S_SHADER_QUALITY=0 shadow, =1 shadow,
    #                     9-tap early-out branch
    #   pcf-collapse      9-tap early-out branch  (the q1 shadow row stays
    #                     live: a 4-tap mean is still a shadow)
    #   dfg-zero          S_SHADER_QUALITY=1 env BRDF
    #   probe-zero        baked=probe
    #   distfade-full     S_SHADER_QUALITY=0 shadow, =1 shadow
    #   speccube-zero     all three env specular rows
    #   ambientsh-zero    baked=none
    #   lightmap-zero     baked=lightmap q=0, q=1
    #   cullmode-tie      cullmode-12 front/back fork  (makes CullMode 1
    #                     and 2 agree, i.e. resolves by accident the one
    #                     thing psrs §4.1 says is not determined)
    #   biasfam-collapse  D_DISABLE_DEPTH_BIAS  (collapses the per-family
    #                     bias split to one scalar, which is what the old
    #                     --gt-depth-bias-{const,slope} did and what the
    #                     +5.0 / -2.0 sign difference forbids)
    #
    # The loop must run with the CANONICAL ASSET SET (--irr-npy,
    # --probe-npz, --ibl-cube, --cube-probes). Four of the eight names
    # break a path whose row is n/a when its asset is absent, so an
    # asset-less run would report "INJECTION DID NOT FIRE" for reasons
    # that have nothing to do with the code.
    ####################################################################
    """
    torch.manual_seed(0)
    n = 4096
    lo, hi = vertices.min(0).values, vertices.max(0).values
    wp = (lo + (hi - lo) * torch.rand(n, 3, device=device)
          ).view(1, 1, n, 3)
    nn = _nrm(torch.randn(1, 1, n, 3, device=device))
    vv = _nrm(torch.randn(1, 1, n, 3, device=device))
    rr = torch.rand(1, 1, n, device=device)
    mm = torch.zeros(1, 1, n, device=device)
    alb = torch.rand(1, 1, n, 3, device=device)
    ey = torch.zeros(1, 3, device=device) + (lo + hi) * 0.5
    lu = torch.rand(1, 1, n, device=device)
    lv = torch.rand(1, 1, n, device=device)
    ones = torch.ones(1, 1, n, device=device)
    rows = []

    def mx(a, b):
        return float((a - b).abs().max())

    print("=" * 74, flush=True)
    print("GT LIGHTING REACHABILITY SELF-TEST (no GT frame is read)",
          flush=True)
    print("=" * 74, flush=True)

    # --- D_BAKED_LIGHTING, all four states -----------------------------
    sh_irr_ = gt_ambient_sh(nn)
    hemi = (AMBIENT_GROUND.view(1, 1, 1, 3)
            + (AMBIENT_SKY - AMBIENT_GROUND).view(1, 1, 1, 3)
            * (nn[..., 1].clamp(-1, 1) * 0.5 + 0.5).unsqueeze(-1))
    rows.append(("baked=none (dyn 0 constant tail)",
                 int((sh_irr_.abs().sum(-1) > 0).sum()),
                 f"max|d| vs the hemispheric ambient it re-expresses = "
                 f"{mx(sh_irr_, hemi):.3e} (one ULP; the affine form "
                 f"represents it to rounding)"))
    pr, po, pw = gt_baked_probe(wp, nn)
    rows.append(("baked=probe (D_BAKED_LIGHTING_FROM_PROBE)",
                 int((pw > 0).sum()),
                 f"max|d| vs the constant tail = {mx(pr, sh_irr_):.4f}; "
                 f"volume weight W in [{float(pw.min()):.3f}, "
                 f"{float(pw.max()):.3f}]"))
    for q in (0, 1):
        li, lo_ = gt_baked_lightmap(lu, lv, q)
        if li is None:
            rows.append((f"baked=lightmap q={q}", 0,
                         "NO --irr-npy: no irradiance array to fetch"))
        else:
            li0, _ = gt_baked_lightmap(lu, lv, 0)
            rows.append((f"baked=lightmap q={q} "
                         f"({'4-tap bicubic x2' if q else '1-tap x2'})",
                         int((li.abs().sum(-1) > 0).sum()),
                         f"max|d| vs the constant tail = "
                         f"{mx(li, sh_irr_):.4f}"
                         + ("" if q == 0 else
                            f"; max|d| q1 vs q0 taps = {mx(li, li0):.4f}")))
    if GT_VLIT is not None:
        rows.append(("baked=vertex-stream (COLOR1)", int(len(GT_VLIT)),
                     f"stream mean {float(GT_VLIT.mean()):.4f}, "
                     f"{float((GT_VLIT.sum(1) > 0).float().mean()):.1%} "
                     f"of vertices non-zero"))
    else:
        rows.append(("baked=vertex-stream (COLOR1)", 0,
                     "not selected, so not baked; select "
                     "--baked-lighting vertex-stream to build it"))

    # --- S_SHADER_QUALITY: the shadow filter ---------------------------
    if GT_CSM_DEPTH is not None:
        s0 = gt_csm_shadow(wp, ey, 0)
        s1 = gt_csm_shadow(wp, ey, 1)
        rows.append(("S_SHADER_QUALITY=0 shadow (1 tap x3 filters)",
                     int((s0 < 1.0).sum()),
                     f"shadowed fraction {float((s0 < 1).float().mean()):.1%}"))
        rows.append(("S_SHADER_QUALITY=1 shadow (9 tap x3 filters)",
                     int((s1 < 1.0).sum()),
                     f"max|d| vs q0 = {mx(s1, s0):.4f}; distinct values "
                     f"{int(s1.unique().numel())} vs {int(s0.unique().numel())}"
                     f" -- q1 must have MORE, or the extra taps did nothing"))
        eo = GT_TAKEN.get("pcf9.early_out", 0)
        fu = GT_TAKEN.get("pcf9.full", 0)
        rows.append(("9-tap early-out branch", eo + fu,
                     f"{eo} texels took the 4-tap exit, {fu} ran all 9. "
                     f"live is eo+fu -- the branch being REACHED. A run "
                     f"where one arm is 0 is a scene property, not a "
                     f"defect, but it does mean this run does not cover "
                     f"that arm"))
    else:
        rows.append(("GT cascaded shadow", 0, "GT_CSM_DEPTH is None"))

    # --- S_SHADER_QUALITY: the direct BRDF -----------------------------
    f0 = 0.04 * (1 - mm).unsqueeze(-1) + alb * mm.unsqueeze(-1)
    ld = sun_dir.view(1, 1, 1, 3).expand_as(nn)
    d0, p0 = gt_direct(0, nn, ld, vv, f0, rr, SUN_COLOR.view(1, 1, 1, 3),
                       ones)
    d1, p1 = gt_direct(1, nn, ld, vv, f0, rr, SUN_COLOR.view(1, 1, 1, 3),
                       ones)
    rows.append(("S_SHADER_QUALITY=0 direct BRDF (combined D*V, no F)",
                 int((p0 > 0).any(-1).sum()),
                 f"spec mean {float(p0.mean()):.5f}"))
    rows.append(("S_SHADER_QUALITY=1 direct BRDF (GGX*SmithSchlick*Schlick)",
                 int((p1 > 0).any(-1).sum()),
                 f"max|d| vs q0 spec = {mx(p1, p0):.5f}; diffuse halves "
                 f"agree to {mx(d1, d0):.3e}"))

    # --- S_SHADER_QUALITY: the environment BRDF ------------------------
    ndv = (nn * vv).sum(-1).clamp(0, 1)
    r0, m0_ = gt_env_brdf(0, rr, ndv, f0)
    r1, m1_ = gt_env_brdf(1, rr, ndv, f0)
    rows.append(("S_SHADER_QUALITY=0 env BRDF (Lazarov polynomial)",
                 int((r0.abs().sum(-1) > 0).sum()),
                 f"response mean {float(r0.mean()):.5f}, multiscatter "
                 f"term is 0 by construction (q0 has none)"))
    rows.append(("S_SHADER_QUALITY=1 env BRDF (DFG LUT + multiscatter)",
                 int(((r1.abs().sum(-1) > 0) & (m1_.abs().sum(-1) > 0)
                      ).sum()),
                 f"max|d| LUT vs polynomial = {mx(r1, r0):.5f}; "
                 f"multiscatter mean {float(m1_.mean()):.5f} -- live "
                 f"counts only pixels where BOTH the response and the "
                 f"multiscatter term are non-zero"))

    # --- environment specular: the THREE reachable combos ---------------
    # (q=0), (q=1, axis=0), (q=1, axis=1). The fourth is excluded by the
    # constraint graph and _gt_check_constraints() refuses it, so it is
    # not exercised here either.
    specs = {}
    for lab, q, sc in (("q=0 unconditional (global slice + mag clamp)",
                        0, False),
                       ("q=1 D_SPECULAR_CUBE_MAP_STATIC=0 (binned walk "
                        "+ box projection)", 1, False),
                       ("q=1 D_SPECULAR_CUBE_MAP_STATIC=1 (one static "
                        "cube, walk removed)", 1, True)):
        try:
            e = gt_spec_cube(q, wp, nn, vv, rr, pr, d0, static_cube=sc)
            specs[lab] = e
            rows.append((f"env specular {lab}",
                         int((e.abs().sum(-1) > 0).sum()),
                         f"mean {float(e.mean()):.5f}, max "
                         f"{float(e.max()):.4f}"))
        except Exception as ex:                             # noqa: BLE001
            rows.append((f"env specular {lab}", 0,
                         f"UNREACHABLE: {type(ex).__name__}: {ex}"))
    kk = list(specs)
    if len(kk) == 3:
        # THE LIVENESS VALUE IS THE max|delta|, NOT THE PIXEL COUNT.
        # These two rows used to pass `n` -- a constant -- so the verdict
        # `"OK" if live else ...` could never be anything but OK. Both
        # printed `max|d| = 0.00000` while their own text says the value
        # must be non-zero, i.e. the row stated its pass condition,
        # reported the violation, and marked itself OK. Vacuous today
        # because no cube radiance is loaded, and it would have passed
        # identically with radiance loaded and a genuinely broken axis.
        # Scaled to an integer because the verdict tests truthiness.
        _d01 = mx(specs[kk[0]], specs[kk[1]])
        _d12 = mx(specs[kk[1]], specs[kk[2]])
        rows.append(("  q0 vs q1", int(_d01 > 0),
                     f"max|d| = {_d01:.5f} -- "
                     f"two different algorithms, not one scaled"))
        rows.append(("  q1 axis=0 vs axis=1", int(_d12 > 0),
                     f"max|d| = {_d12:.5f} -- "
                     f"must be non-zero or removing the volume walk "
                     f"changed nothing"))
    rows.append(("  constraint: SPECCUBE=1 at q=0", 0,
                 "REFUSED by _gt_check_constraints() -- the graph "
                 "excludes it, so it is not a state to test"))

    # --- psrs axes ------------------------------------------------------
    rows.append((f"D_FRONT_FACE_CULL={args.front_face_cull} [psrs]",
                 GT_PSRS[0],
                 f"CullMode {'0/2' if args.front_face_cull else '0/1'}: "
                 f"{GT_PSRS[0]:,} kept, {GT_PSRS[1]:,} culled, "
                 f"{GT_PSRS[4]:,} at CullMode 0 (F_RENDER_BACKFACES). "
                 f"live is the KEPT COUNT, so a cull that removed "
                 f"everything fails here. psrs §4.1"))
    # The 1-vs-2 = front-vs-back ambiguity, kept measurable. Rebuild the
    # atlas under the other assignment and report max|delta|: if the two
    # agree, the fork is not a fork and the flag is decorative.
    _was = args.cullmode_12
    _base = GT_CSM_DEPTH.clone() if GT_CSM_DEPTH is not None else None
    _kept = GT_PSRS[0]
    try:
        args.cullmode_12 = "1-front" if _was == "2-front" else "2-front"
        gt_build_csm(GT_SHADOW_CASTERS, " (cullmode-12 fork probe)")
        _alt = GT_CSM_DEPTH.clone() if GT_CSM_DEPTH is not None else None
        _altkept = GT_PSRS[0]
    finally:
        args.cullmode_12 = _was
        gt_build_csm(GT_SHADOW_CASTERS, " (restored)")
    if _base is not None and _alt is not None:
        _fin = (_base < 1e8) | (_alt < 1e8)
        _d = float((_base.clamp(max=1e3) - _alt.clamp(max=1e3)
                    ).abs()[_fin].max()) if bool(_fin.any()) else 0.0
        rows.append(("cullmode-12 front/back fork [NOT DETERMINED]",
                     1 if _d > 0 else 0,
                     f"'{_was}' keeps {_kept:,} faces, the other keeps "
                     f"{_altkept:,}; max|d| between the two depth atlases "
                     f"= {_d:.4g}. Which of CullMode 1/2 is the front face "
                     f"is NOT fixed by any shipped data (psrs §4.1) -- both "
                     f"are implemented and this row exists so the "
                     f"undetermined choice stays visible. live=0 would mean "
                     f"the flag changes nothing and the ambiguity had been "
                     f"resolved by accident"))
    rows.append((f"D_DISABLE_DEPTH_BIAS={args.disable_depth_bias} [psrs]",
                 GT_PSRS[2],
                 f"max|d| to the depth atlas = {GT_PSRS[3]:.3e} over "
                 f"{GT_PSRS[2]:,} texels. Depth-pass slope bias "
                 f"{args.gt_depth_bias_pass:+g} on env/fol/gl (the axis's "
                 f"only effect, psrs §4.2); material bias "
                 f"{tuple(args.gt_depth_bias_material)} on cx/eb/so where "
                 f"F_DEPTH_BIAS, which the axis does NOT touch. Opposite "
                 f"signs, kept apart"))

    # --- the LOD-scale trap, checked explicitly ------------------------
    ls = args.gt_spec_cube_lod_scale
    if ls is None:
        ls = args.ibl_lod_scale
    nm = (len(CUBE) if CUBE is not None else 0)
    rows.append(("spec-cube LOD reach", nm,
                 f"lod = {ls:g} * sqrt(rough) spans [0, {ls:g}] over "
                 f"{nm} resident mips -- "
                 + ("EVERY mip is reachable" if nm and ls >= nm - 1
                    else f"mips {int(ls)+1}..{max(nm-1,0)} are NOT "
                         f"reachable" if nm else "no cube chain loaded")))

    # --- the composition ------------------------------------------------
    for q in (0, 1):
        # lm_tint EXPLICIT: this is the one composition in the file that
        # does not run gt_baked() first -- `pr, po` come from the probe
        # path -- so it must not read GT_LM_TINT_LIVE, which would be
        # whatever the last real frame left there. False is what this
        # call did before the tint moved out of gt_baked_lightmap.
        c = gt_compose(alb, nn, vv, wp, ey, rr, mm, q, pr, po,
                       ones.squeeze(0).squeeze(0).view(1, 1, n),
                       gt_csm_shadow(wp, ey, q)
                       if GT_CSM_DEPTH is not None else ones,
                       lm_tint=False)
        rows.append((f"gt_compose q={q}", int((c.abs().sum(-1) > 0).sum()),
                     f"mean {float(c.mean()):.5f}"))

    # ---- verdict -------------------------------------------------------
    # A check that only prints is a check that cannot fail: its clean path
    # and its failure path produce the same exit status, so nothing ever
    # investigates it. Every row below is classified REQUIRED or
    # INFORMATIONAL, and a dead REQUIRED row exits non-zero.
    #
    # Required-ness is a function of the flags and the assets actually
    # loaded, NOT a fixed list -- a row that is dead because its axis was
    # not selected is not a defect, and calling it one would train the
    # next person to ignore the output.
    has_lm = IRRMAP is not None
    has_cube = (CUBE is not None) or (SH is not None)
    has_probe = (PROBE_ATLAS is not None) and (PROBE_T is not None)
    req_rules = [
        ("baked=none", True),
        ("baked=probe", has_probe),
        ("baked=lightmap", has_lm),
        ("baked=vertex-stream", GT_VLIT is not None),
        ("S_SHADER_QUALITY=0 shadow", GT_CSM_DEPTH is not None),
        ("S_SHADER_QUALITY=1 shadow", GT_CSM_DEPTH is not None),
        ("9-tap early-out", GT_CSM_DEPTH is not None),
        # Reaching this row AT ALL means gt_build_csm() did not run or
        # did not produce an atlas, and it is called unconditionally
        # under --gt-lighting / --gt-lighting-selftest. So it is always
        # required: without the key it would classify as informational
        # and a failed CSM build would print and exit 0.
        ("GT cascaded shadow", True),
        ("S_SHADER_QUALITY=0 direct BRDF", True),
        ("S_SHADER_QUALITY=1 direct BRDF", True),
        ("S_SHADER_QUALITY=0 env BRDF", True),
        ("S_SHADER_QUALITY=1 env BRDF", True),
        ("env specular", has_cube),
        ("spec-cube LOD reach", CUBE is not None),
        ("gt_compose", True),
        ("D_FRONT_FACE_CULL", GT_CSM_DEPTH is not None),
        ("D_DISABLE_DEPTH_BIAS", GT_CSM_DEPTH is not None),
        # REQUIRED: a dead row here means --cullmode-12 changes nothing,
        # i.e. an ambiguity the shipped data does not resolve has been
        # resolved by accident somewhere in the implementation.
        ("cullmode-12 front/back fork", GT_CSM_DEPTH is not None),
    ]

    def _required(nm):
        """Can this row fail the run?

        A sub-row (two leading spaces) is commentary on the row above and
        cannot gate -- but EXEMPTION MUST BE A VISIBLE STATUS, NOT A
        SILENCE. It used to return False here and print `n/a`, which is
        the same glyph an unselected axis prints, so a sub-row reporting
        a real hole was indistinguishable from a row that legitimately
        did not apply. That is how `spec-cube q1: no probe volumes`
        survived: the term ran, found no boxes, said so on every run, and
        rendered as `n/a` beside forty other `n/a`s.

        Now sub-rows return the sentinel `EXEMPT`, which prints as its
        own flag and is counted in the summary. `EXEMPT` means "cannot
        gate the exit status" and never "unnecessary" -- the same
        distinction the conformance runner enforces on its own table.
        """
        if nm.startswith("  "):
            return "EXEMPT"
        for key, need in req_rules:
            if nm.startswith(key):
                return bool(need)
        return False

    w = max(len(r[0]) for r in rows)
    fails = []
    n_exempt_dark = 0
    for name, live, note in rows:
        req = _required(name)
        if live:
            flag = "OK  "
        elif req == "EXEMPT":
            flag = "EXPT"
            n_exempt_dark += 1
        elif req:
            flag = "FAIL"
            fails.append(name)
        else:
            flag = "n/a "
        print(f"{flag} {name:<{w}}  live={live:>9}  {note}", flush=True)
    if n_exempt_dark:
        print(f"     {n_exempt_dark} EXPT rows are not live and CANNOT fail "
              f"this run. Exempt means 'cannot gate', never 'unnecessary' "
              f"-- read them.", flush=True)
    print("=" * 74, flush=True)
    for g in GT_GAPS:
        print(f"GAP  {g}", flush=True)
    if GT_INJECT:
        print(f"FAULT INJECTED: --gt-selftest-inject {GT_INJECT}",
              flush=True)
        if fails:
            print(f"INJECTION FIRED: {len(fails)} required path(s) went "
                  f"dead -- {', '.join(fails)}", flush=True)
            print("This run PROVES the self-test can fail. Exit 1.",
                  flush=True)
            return 1
        print("INJECTION DID NOT FIRE. The self-test cannot see this "
              "path, so its reachability claim does not cover it. "
              "That is a defect in the CHECK, not in the renderer.",
              flush=True)
        return 3
    if fails:
        print(f"FAIL: {len(fails)} required path(s) dead at their own "
              f"defaults -- {', '.join(fails)}", flush=True)
        print("A path selected by the flags but dead at its defaults is "
              "the --ibl-lod-scale failure: it runs, it returns a "
              "number, and the number is the path switched off.",
              flush=True)
        return 1
    print("All required paths live. NOTE: a clean run is not sufficient "
          "on its own -- see the fault-injection loop in this function's "
          "docstring. Until all eight names have been seen to fire, "
          "'no defects found' and 'the check never ran' are the same "
          "output.", flush=True)
    return 0


gt_lighting_banner()
if args.water_selftest:
    raise SystemExit(0 if water_axis_selftest() else 1)
# MANDATORY, every run, before anything renders: a duplicate flag name
# takes the parser down for every invocation and nothing downstream can
# see it. Cheap enough to be unconditional, which is the point.
LMG_DUPS = lmg_parser_replay()
lmg_family_banner()
# impl-ssao-filters: run the SSAO/filter reachability proof every run.
# A dead row means a transcribed path is not being executed at the
# current flags, which is the failure this whole exercise is about, so
# it exits non-zero rather than printing and carrying on.
CS2FILT_DEAD = cs2filt_reachability_selftest()
if CS2FILT_DEAD:
    raise SystemExit(f"cs2filt reachability: {CS2FILT_DEAD} DEAD path(s)")
CHAR_INJECTS = ("sss-flat", "aniso-isotropic", "hair-dead", "cloth-zero",
                "irid-zero", "retro-zero", "patches-clear", "decal-zero",
                "blood-zero", "adjust-identity", "eye-flat",
                "glove-passthrough", "skin-identity")

# Injection name -> the FLAG whose gated branch it suppresses.
#
# THE FIRST VERSION OF THIS MECHANISM COULD NOT FAIL, and the sweep that
# proved it was never run until 2026-08-08: 11 of 12 names reported "DID NOT
# FIRE". The injections were applied to LOCAL VARIABLES feeding the two
# char_direct_delta rows, while every row they are NAMED after is built by
# _delta(), which toggles args.<flag> around character_surface() and re-reads
# the real kernels. Breaking `hr` could not make the S_ANISOTROPIC_HAIR row
# go dead, because that row never saw `hr`.
#
# So the mutant is now the one the instrument is actually about: the axis is
# SELECTED and its branch produces nothing. That is the exact failure this
# self-test exists to catch -- a dead path, a silent stub, a gated-off scope
# cut -- and it is applied at the branch, so the named row collapses to a
# zero delta and reports DEAD. Numeric fidelity is not this test's job; the
# conformance registry measures that separately, with numbers.
CHAR_INJECT_FLAG = {
    "sss-flat": "char_sss", "aniso-isotropic": "char_aniso",
    "hair-dead": "char_hair", "cloth-zero": "char_cloth",
    "irid-zero": "char_iridescence", "retro-zero": "char_retro_reflective",
    "patches-clear": "char_patches", "decal-zero": "char_decals",
    "blood-zero": "char_blood", "adjust-identity": "char_adjustments",
    "eye-flat": "char_eyes", "glove-passthrough": "customglove",
    "skin-identity": "skinning",
}


def _cinj(flag):
    """True when --char-selftest-inject names the branch guarded by `flag`."""
    return CHAR_INJECT_FLAG.get(args.char_selftest_inject) == flag
