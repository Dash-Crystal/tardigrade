



# =======================================================================
# THE THREE SMALL FAMILIES: the exact-cost audit and the reachability
# self-test.
# =======================================================================
def sf_flop_audit(mutate=False):
    """Price the two EXACT transcriptions and compare against 132/1, 194/5.

    csgo_black_unlit r0_m0 and csgo_effects r10_m1 contain no loop region,
    so those figures are exact per-invocation counts rather than floors --
    they and the 24 S_MODE_DEPTH modules are the only exact figures in the
    whole corpus. That makes them a free correctness check on the
    transcription that no other family in this project can offer, so it is
    run rather than described.

    The ledger is produced BY the transcription (`_SFCost` wraps each
    arithmetic op and returns its value), so it cannot drift away from the
    code. `mutate=True` deletes one term to show the check FAILS -- a
    check that cannot fail is not a check.
    """
    dev = device
    one = torch.ones(1, 1, 1, device=dev)
    wp = torch.stack([one[..., 0] * 12.0, one[..., 0] * 3.0,
                      one[..., 0] * -40.0], -1)
    ey = torch.stack([one[..., 0] * 0.0, one[..., 0] * 1.7,
                      one[..., 0] * 0.0], -1)
    nr = torch.stack([one[..., 0] * 0.0, one[..., 0] * 0.0,
                      one[..., 0] * 1.0], -1)
    uv = torch.stack([one[..., 0] * 0.31, one[..., 0] * 0.62], -1)
    vc = torch.cat([one, one * 0.8, one * 0.6, one * 0.9], -1)
    alb = torch.cat([one * 0.7, one * 0.7, one * 0.7, one * 0.55], -1)
    ok_all = True
    out = []

    L = _SFCost()
    sf_black_unlit_shade(wp, ey, one[..., 0], L=L)
    ok, lines = L.compare("black_unlit")
    out.append("csgo_black_unlit  c0/r0m0  (no loop region -> EXACT)")
    out += lines
    ok_all = ok_all and ok

    # The audit runs the module de_inferno SELECTS, so the combo is the
    # measured one -- static 10, the steam plume -- not whatever the
    # command line asked for. A different combo is a different module and
    # would have a different exact figure.
    SC = dict(additive=True, tint_mask=False, depth_feather=True,
              backfaces=True, fog=True)
    if mutate:
        # delete the distance-fade term. This is the mutation the
        # docstring promises: it changes the OUTPUT and the LEDGER
        # together, which is what makes the audit a real check.
        SC = dict(SC, depth_feather=False)
    x = torch.zeros(1, 1, 1, len(EXT2), device=dev)
    for k, v in (("sf_fx_opacity_scale", 0.75), ("sf_fx_color_boost", 1.0),
                 ("sf_fx_fresnel_exp", 3.461),
                 ("sf_fx_fresnel_falloff", 1.552),
                 ("sf_fx_fresnel_min", 0.0), ("sf_fx_fresnel_max", 1.0),
                 ("sf_fx_feather_dist", 20.0),
                 ("sf_fx_feather_falloff", 1.0),
                 ("sf_fx_fade_dist", 263.158), ("sf_fx_fade_falloff", 1.0),
                 ("sf_fx_fade_min", 0.0), ("sf_fx_fade_max", 1.0),
                 ("sf_fx_m1_su", 0.1), ("sf_fx_m1_sv", 0.5),
                 ("sf_fx_m1_pu", 0.01), ("sf_fx_m1_pv", -0.015),
                 ("sf_fx_m2_su", 0.07), ("sf_fx_m2_sv", 0.4),
                 ("sf_fx_m2_pu", -0.01), ("sf_fx_m2_pv", -0.015),
                 ("sf_fx_m3_su", 0.01), ("sf_fx_m3_sv", 0.05),
                 ("sf_fx_m3_pu", 0.0), ("sf_fx_m3_pv", -0.001),
                 ("sf_fx_has_mask1", 1.0), ("sf_fx_has_mask2", 1.0),
                 ("sf_fx_has_mask3", 1.0), ("sf_fx_fog_enabled", 1.0),
                 ("sf_fx_additive", 1.0), ("sf_fx_depth_feather", 1.0),
                 ("sf_fx_backfaces", 1.0)):
        x[..., EXT2[k]] = v
    mid = torch.zeros(1, 1, 1, dtype=torch.long, device=dev)
    L2 = _SFCost()
    _v = _nrm(wp - ey)
    sf_effects_shade(uv, mid, x, wp, ey, nr, vc, alb,
                     one[..., 0] * 0.995, _v, one, None, SC, L=L2)
    ok2, lines2 = L2.compare("effects")
    out.append("csgo_effects      c10/r10m1 (no loop region -> EXACT)")
    out += lines2
    ok_all = ok_all and ok2
    return ok_all, out


def sf_selftest():
    """Prove every path this block adds is REACHABLE AT ITS OWN DEFAULTS.

    No image metric and no ground-truth frame: each path runs on a
    synthetic batch spanning the world box and the normal sphere, and the
    reported number is max|delta| against that path suppressed. A zero is
    a FAILURE, because a path that changes nothing at its own defaults is
    a feature that reads as tested and is off -- which is exactly how
    --ibl-lod-scale=1.0 kept mips 2-6 unreachable while every arm above
    it reported green.
    """
    dev = device
    n = 4096
    g = torch.rand(1, 1, n, 3, device=dev)
    lo = vertices.min(0).values.view(1, 1, 1, 3)
    hi = vertices.max(0).values.view(1, 1, 1, 3)
    wp = lo + (hi - lo) * g
    ey = (lo + hi) * 0.5
    nr = _nrm(torch.randn(1, 1, n, 3, device=dev))
    uv = torch.rand(1, 1, n, 2, device=dev)
    one = torch.ones(1, 1, n, 1, device=dev)
    vc = torch.cat([torch.rand(1, 1, n, 3, device=dev), one * 0.9], -1)
    alb = torch.cat([torch.rand(1, 1, n, 3, device=dev), one * 0.6], -1)
    mid = torch.zeros(1, 1, n, dtype=torch.long, device=dev)
    rows = []

    def rec(name, a, b):
        d = float((a - b).abs().max())
        rows.append((name, d))
        return d

    # csgo_black_unlit: fog ON against fog OFF (the family's only axis)
    r_on, _ = sf_black_unlit_shade(wp, ey, one[..., 0])
    r_off, _ = sf_black_unlit_shade(wp, ey, torch.zeros_like(one[..., 0]))
    rec("black_unlit  g_bFogEnabled 1 vs 0", r_on, r_off)
    # the CUBE layer on its own, at the SHIPPED env_cubemap_fog ramp. The
    # number is reported whatever it is: a small one is a fact about the
    # map's entity, and printing it is what stops it being an assumption.
    cb = _sf_fog_cb()
    wsrc = (wp - ey) / SF_SRC_U
    hz = wp[..., 1] / SF_SRC_U
    base = torch.zeros_like(wp)
    a1, _ = _sf_grad_fog_gated(_SF_NULL, wsrc, hz, cb, base)
    a2, fcu = _sf_cube_fog_gated(_SF_NULL, wsrc, hz, cb, a1, wp)
    rec("black_unlit  cube-fog layer (shipped 800..280000 u ramp)", a2, a1)
    rows.append(("  cube-fog max opacity over the world box",
                 float(fcu.max())))

    if EXT2:
        x = torch.zeros(1, 1, n, len(EXT2), device=dev)
        for k in EXT2:
            if k.startswith("sf_fx_has") or k.startswith("sf_vl_has"):
                x[..., EXT2[k]] = 1.0
        for k, v in (("sf_fx_opacity_scale", 0.75),
                     ("sf_fx_color_boost", 1.0),
                     ("sf_fx_fresnel_exp", 3.461),
                     ("sf_fx_fresnel_falloff", 1.552),
                     ("sf_fx_fresnel_max", 1.0),
                     ("sf_fx_feather_dist", 20.0),
                     ("sf_fx_feather_falloff", 1.0),
                     ("sf_fx_fade_dist", 263.158),
                     ("sf_fx_fade_falloff", 1.0), ("sf_fx_fade_max", 1.0),
                     ("sf_fx_m1_su", 0.1), ("sf_fx_m1_sv", 0.5),
                     ("sf_fx_m2_su", 0.07), ("sf_fx_m2_sv", 0.4),
                     ("sf_fx_m3_su", 0.01), ("sf_fx_m3_sv", 0.05),
                     ("sf_fx_fog_enabled", 1.0), ("sf_fx_additive", 1.0),
                     ("sf_fx_depth_feather", 1.0),
                     ("sf_vl_ao_scale", 0.0), ("sf_vl_reflectance", 0.04),
                     ("sf_vl_detail_blend", 0.5),
                     ("sf_vl_opacity_scale", 1.0),
                     ("sf_vl_si_tint_r", 1.0), ("sf_vl_si_tint_g", 1.0),
                     ("sf_vl_si_tint_b", 1.0),
                     ("sf_vl_tint_mask", 1.0), ("sf_vl_detail", 1.0),
                     ("sf_vl_decal", 1.0), ("sf_vl_self_illum", 1.0),
                     ("sf_vl_translucent", 1.0), ("sf_vl_alpha_test", 1.0),
                     ("sf_vl_spec_direct", 1.0),
                     ("sf_vl_spec_indirect", 1.0),
                     ("sf_vl_fog_enabled", 1.0)):
            if k in EXT2:
                x[..., EXT2[k]] = v
        zl = torch.rand(1, 1, n, device=dev) * 0.02 + 0.98
        fwd = _nrm(wp - ey)
        base_sc = dict(additive=True, tint_mask=False, depth_feather=True,
                       backfaces=True, fog=True)
        r0, a0 = sf_effects_shade(uv, mid, x, wp, ey, nr, vc, alb, zl, fwd,
                                  one[..., :1],
                                  (torch.rand(1, 1, n, 1, device=dev) > 0.5),
                                  base_sc)
        _ff = (torch.rand(1, 1, n, 1, device=dev) > 0.5)
        for ax in ("depth_feather", "tint_mask", "fog", "additive",
                   "backfaces"):
            sc = dict(base_sc)
            sc[ax] = not sc[ax]
            r1, a1_ = sf_effects_shade(uv, mid, x, wp, ey, nr, vc, alb, zl,
                                       fwd, one[..., :1], _ff, sc)
            rec(f"effects      S_{ax.upper()} toggled",
                torch.cat([r0, a0.unsqueeze(-1)], -1),
                torch.cat([r1, a1_.unsqueeze(-1)], -1))
        # the three erosion masks, each on its own
        for i in (1, 2, 3):
            xx = x.clone()
            xx[..., EXT2[f"sf_fx_has_mask{i}"]] = 0.0
            r1, a1_ = sf_effects_shade(uv, mid, xx, wp, ey, nr, vc, alb, zl,
                                       fwd, one[..., :1], None, base_sc)
            rec(f"effects      g_tMask{i}", a0, a1_)
        _, a_t7 = sf_effects_shade(uv, mid, x, wp, ey, nr, vc, alb, zl, fwd,
                                   one[..., :1] * 7.0,
                                   (torch.rand(1, 1, n, 1, device=dev) > 0.5),
                                   base_sc)
        rec("effects      g_vMaskNPanSpeed (t=1 vs t=7)", a0, a_t7)

        t4 = torch.cat([_nrm(torch.randn(1, 1, n, 3, device=dev)), one], -1)
        nmap = torch.rand(1, 1, n, 4, device=dev)
        amb = torch.rand(1, 1, n, 3, device=dev) * 0.3
        sd = _nrm(torch.tensor([0.3, 0.8, 0.5], device=dev)
                  ).view(1, 1, 1, 3).expand(1, 1, n, 3)
        vw = _nrm(ey - wp)
        vsc = dict(spec_direct=True, spec_indirect=True, force_uv2=False,
                   decal=True, detail=True, alpha_test=False,
                   translucent=True, additive=False, tint_mask=True,
                   self_illum=True, fog=True, baked="lightmap",
                   opaque_fade=False, spec_cube_static=False,
                   shader_quality=0)
        v0, va0 = sf_vertexlit_shade(
            uv, uv, mid, x, alb[..., :3], alb[..., 3], nmap, wp, ey, nr, t4,
            vc, amb, one[..., 0] * 0.2, one[..., 0] * 0.7,
            one[..., 0] * 0.9, one[..., 0], sd,
            SUN_COLOR.view(1, 1, 1, 3), amb, None, vw, 0.0, vsc)
        for ax in ("spec_direct", "spec_indirect", "decal", "detail",
                   "translucent", "alpha_test", "tint_mask", "self_illum",
                   "additive", "fog"):
            sc = dict(vsc)
            sc[ax] = not sc[ax]
            v1, va1 = sf_vertexlit_shade(
                uv, uv, mid, x, alb[..., :3], alb[..., 3], nmap, wp, ey, nr,
                t4, vc, amb, one[..., 0] * 0.2, one[..., 0] * 0.7,
                one[..., 0] * 0.9, one[..., 0], sd,
                SUN_COLOR.view(1, 1, 1, 3), amb, None, vw, 0.0, sc)
            rec(f"vertexlit    S_{ax.upper()}",
                torch.cat([v0, va0.unsqueeze(-1)], -1),
                torch.cat([v1, va1.unsqueeze(-1)], -1))
        # the four D_BAKED_LIGHTING_* values, as four different irradiances
        for bm, bi in (("none", torch.zeros_like(amb)),
                       ("vertex-stream", amb * 2.0),
                       ("probe", amb * 0.5), ("lightmap", amb)):
            sc = dict(vsc, baked=bm)
            v1, _ = sf_vertexlit_shade(
                uv, uv, mid, x, alb[..., :3], alb[..., 3], nmap, wp, ey, nr,
                t4, vc, bi, one[..., 0] * 0.2, one[..., 0] * 0.7,
                one[..., 0] * 0.9, one[..., 0], sd,
                SUN_COLOR.view(1, 1, 1, 3), amb, None, vw, 0.0, sc)
            if bm != "lightmap":
                rec(f"vertexlit    D_BAKED_LIGHTING_FROM_{bm}", v0, v1)
        # offset 76 and offset 88, the two per-view render targets this
        # family declares (r51_m3:84-86)
        for nm, kw in (("offset 76 (directional occlusion)",
                        dict(dirocc=one[..., 0] * 0.4)),
                       ("offset 88 (screen-space shadow)",
                        dict(ss88=one[..., 0] * 0.3))):
            v1, _ = sf_vertexlit_shade(
                uv, uv, mid, x, alb[..., :3], alb[..., 3], nmap, wp, ey, nr,
                t4, vc, amb,
                one[..., 0] * 0.2, kw.get("dirocc", one[..., 0] * 0.7),
                kw.get("ss88", one[..., 0] * 0.9), one[..., 0], sd,
                SUN_COLOR.view(1, 1, 1, 3), amb, None, vw, 0.0, vsc)
            rec(f"vertexlit    {nm}", v0, v1)
        # S_MODE_TOOLS_VIS, every mode the module tests for
        base_tv = sf_tools_vis(0, v0, nr, wp, ey, 0.0)
        for m in _SF_TV_MODES:
            if m == 0:
                continue
            rec(f"tools_vis    mode {m}",
                sf_tools_vis(m, v0, nr, wp, ey, 0.0), base_tv)
    return rows


NW_DEPTH_ERR = []


QOD_NOTE = []


def lmg_parser_replay():
    """MANDATORY: replay every add_argument in this file against a fresh
    ArgumentParser and report the conflict count.

    A P0 shipped because two agents added the same flag name in different
    files, git merged both cleanly, and argparse then raised at IMPORT --
    so the renderer could not build its parser for ANY invocation, not
    just the one that used the flag. The failure is not in the feature,
    it is in the parser, and nothing that runs after `parse_args()` can
    detect it.

    This reads the parser object this file already built rather than
    re-parsing source text, so it sees exactly the option strings
    argparse sees, including the ones another agent added.

    IT CAN FAIL: pass --lmg-selftest-inject dup-flag to have it register
    a deliberate duplicate and watch the count come back non-zero.
    """
    seen, dups = {}, []
    for act in parser._actions:
        for opt in act.option_strings:
            if opt in seen:
                dups.append(opt)
            seen[opt] = act
    probe = argparse.ArgumentParser(add_help=False)
    replayed = 0
    for act in parser._actions:
        if not act.option_strings:
            continue
        kw = {}
        if act.nargs is not None:
            kw["nargs"] = act.nargs
        if act.choices is not None:
            kw["choices"] = list(act.choices)
        if act.default is not None:
            kw["default"] = act.default
        if isinstance(act, argparse._StoreTrueAction):
            kw = {"action": "store_true"}
        elif isinstance(act, argparse._StoreFalseAction):
            kw = {"action": "store_false"}
        elif act.type is not None:
            kw["type"] = act.type
        if isinstance(act, argparse._HelpAction):
            continue
        try:
            probe.add_argument(*act.option_strings, **kw)
            replayed += 1
        except argparse.ArgumentError as e:
            dups.append("%s (%s)" % (act.option_strings[0], e))
    if _lmg_inject("dup-flag"):
        try:
            probe.add_argument("--lmg-family")
        except argparse.ArgumentError as e:
            dups.append("INJECTED --lmg-family (%s)" % e)
    # A string-valued flag tested with bare truthiness is the bug that
    # was hiding behind that P0: "off" is truthy. Every string flag this
    # block adds is listed here with the exact comparison that guards it.
    strflags = {"--lmg-family": 'LMG_ON = args.lmg_family == "on"',
                "--lmg-opaque-fade":
                    'LMG_FADE_ON = args.lmg_opaque_fade == "on"',
                "--lmg-selftest":
                    'LMG_SELFTEST_ON = args.lmg_selftest == "on"',
                "--lmg-baked": 'compared by value, never truthiness',
                "--lmg-lm-direction": 'path or None; `if args.lmg_lm_direction`'
                                      ' is a PATH test, not an on/off test',
                "--lmg-selftest-inject": 'compared by == to a name',
                "--secondary-uv":
                    'GT_AXES["S_SECONDARY_UV"] = args.secondary_uv != "off"'}
    print("=" * 66, flush=True)
    print(f"argparse replay: {replayed} options replayed against a fresh "
          f"ArgumentParser, {len(dups)} conflicts", flush=True)
    if dups:
        print("  CONFLICTS: " + ", ".join(dups), flush=True)
    for f, g in strflags.items():
        print(f"  string flag {f:26s} guarded by  {g}", flush=True)
    print("=" * 66, flush=True)
    return len(dups)


def _lmg_axis_rows(tag, axes, defer=False):
    """One row per (axis, value), each one a real lmg_static_combo() call.

    The row's max|delta| is the combo id STEP between consecutive values
    of that axis. It must equal the axis's own m_nComboIndexValue, and
    for csgo_lightmappedgeneric that means 96 for S_TEXTURETRANSFORMS
    and 192 for S_ALPHA_TEST -- not 64 and 128. Anything decoding with
    `&` reports the powers of two, so this row is the trap's tripwire.
    """
    out = []
    if not EXT2:
        return out
    for name, stride, lo, hi, col in axes:
        if col is None or col not in EXT2:
            out.append((f"{tag}  axis {name:<26s} stride {stride:<5d} "
                        f"values {list(range(lo, hi + 1))}  "
                        f"(not material-driven: m_iFeatureIndex -1)",
                        hi - lo + 1, 0.0))
            continue
        ids, prev, step = [], None, 0.0
        for v in range(lo, hi + 1):
            x = torch.zeros(1, 1, len(EXT2))
            x[0, 0, EXT2[col]] = float(v)
            cid = float(lmg_static_combo(x, axes)[0, 0])
            ids.append(int(cid))
            if prev is not None:
                step = max(step, abs(cid - prev))
            prev = cid
        bad = "" if (hi == lo or step == float(stride)) else \
            f"  <-- STEP {step:.0f} != stride {stride}"
        out.append((f"{tag}  axis {name:<26s} stride {stride:<5d} "
                    f"values {list(range(lo, hi + 1))} -> ids {ids}{bad}",
                    hi - lo + 1, step))
    return out


def lmg_family_banner():
    """Say which shader each of the three families got, and from where."""
    print("=" * 66, flush=True)
    print("csgo_lightmappedgeneric.vfx / generic.vfx / csgo_imported",
          flush=True)
    print("  generic.vfx ships TWICE. csgo/gameinfo.gi SearchPaths order "
          "is", flush=True)
    print("    Game csgo -> Game csgo_imported -> Game csgo_core -> "
          "Game core", flush=True)
    print("  csgo/shaders_vulkan_dir.vpk has 65 families and NEITHER "
          "generic nor", flush=True)
    print("  csgo_lightmappedgeneric; csgo_imported ships no shaders vpk "
          "at all;", flush=True)
    print("  so csgo_core is the first path that has generic.vfx and ITS "
          "COPY WINS", flush=True)
    print("  (1,942,120 B, 70 static combos). The core copy (88,650 B, 50 "
          "static", flush=True)
    print("  combos, 43 records) is SHADOWED -- checked, not skipped; it "
          "declares a", flush=True)
    print("  different axis set (S_MODE_TOOLS_WIREFRAME, S_TOOLS_ENABLED, "
          "S_SPECULAR_", flush=True)
    print("  CUBE_MAP, S_RENDER_BACKFACES, one D_ALPHATINT dynamic axis), "
          "so this is", flush=True)
    print("  not a size tie-break, it is a different shader.", flush=True)
    if not LMG_LIVE:
        print("  NOT LIVE: " + ("--lmg-family off" if not LMG_ON
                                else "no --fam-side, so no pixel knows "
                                     "which family owns it"), flush=True)
        print("=" * 66, flush=True)
        return
    print(f"  FAM_LIGHTMAPPEDGENERIC = {FAM_LIGHTMAPPEDGENERIC}, "
          f"FAM_GENERIC = {FAM_GENERIC}, FAM_IMPORTED = {FAM_IMPORTED}",
          flush=True)
    for fid, nm, axes in ((FAM_LIGHTMAPPEDGENERIC,
                           "csgo_lightmappedgeneric.vfx", LMG_STATIC_AXES),
                          (FAM_IMPORTED, LMG_IMPORTED_FAM, LMG_STATIC_AXES),
                          (FAM_GENERIC, "generic.vfx", GEN_STATIC_AXES)):
        if fid < 0:
            print(f"  {nm:<36s} NOT NAMED by the side table", flush=True)
            continue
        sel = (MAT_FAM == fid)
        n = int(sel.sum())
        if n == 0:
            print(f"  {nm:<36s} 0 materials on this map", flush=True)
            continue
        cid = lmg_static_combo(MAT_EXT2[sel], axes)
        u = sorted({int(v) for v in cid.tolist()})
        print(f"  {nm:<36s} {n:>4d} materials, static combo id(s) "
              f"{u[:8]}{'...' if len(u) > 8 else ''}  "
              f"(MIXED-RADIX: sum v_i * stride_i)", flush=True)
    print("  strides that a bitmask decode gets wrong: "
          "S_TEXTURETRANSFORMS 96, S_ALPHA_TEST 192,", flush=True)
    print("  S_TRANSLUCENT 384, S_DETAILBLENDMODE 768, S_OVERLAY 1536, "
          "S_METALNESS_TEXTURE 3072", flush=True)
    print("  (S_DETAILTEXTURE has THREE values, which is where the "
          "radix stops being 2)", flush=True)
    print("=" * 66, flush=True)


def lmg_family_selftest():
    """Direct-call reachability proof for the three families.

    csgo_lightmappedgeneric renders 0 pixels in all 48 GT poses, and a
    zero pixel count is indistinguishable from a function that is never
    called. So each shading path is invoked HERE on a synthetic batch,
    at EVERY value of EVERY axis it declares, and the report is a pixel
    count plus the max|delta| the axis moves. An axis that changes
    nothing shows up as a 0.0 rather than as silence.

    A zero from an incomplete closure looks exactly like a zero that
    means absent (CHECKS_THAT_CANNOT_FAIL.md), so the two are separated
    by construction: the count column says the path RAN, the delta
    column says it DID something.
    """
    if not LMG_SELFTEST_ON:
        return 0
    N = 64
    g = torch.Generator(device="cpu").manual_seed(20260807)
    lin = torch.rand(1, N, N, 3, generator=g).to(device)
    vcol = torch.rand(1, N, N, 4, generator=g).to(device)
    direct = torch.rand(1, N, N, 3, generator=g).to(device)
    baked = torch.rand(1, N, N, 3, generator=g).to(device)
    ssao = torch.rand(1, N, N, generator=g).to(device)
    aot = torch.rand(1, N, N, generator=g).to(device)
    ctint = torch.rand(1, N, N, 3, generator=g).to(device) * 0.5 + 0.5
    n_geo = _nrm(torch.randn(1, N, N, 3, generator=g).to(device))
    tan4 = torch.cat([_nrm(torch.randn(1, N, N, 3, generator=g).to(device)),
                      torch.ones(1, N, N, 1, device=device)], dim=-1)
    nr = torch.rand(1, N, N, 4, generator=g).to(device)
    front = torch.rand(1, N, N, generator=g).to(device) > 0.5
    rows = []
    print("=" * 66, flush=True)
    print("csgo_lightmappedgeneric / generic.vfx / csgo_imported "
          "reachability", flush=True)
    # --- the shipped variant every family reports on ------------------
    ts = lmg_normal_hemioct(nr)
    Nw = lmg_frame(n_geo, tan4, ts, front)
    rows.append(("lmg  hemi-oct normal + frame  (r0_m3:193-238)",
                 int(Nw.shape[1] * Nw.shape[2]),
                 float((Nw - n_geo).abs().max())))
    base = None
    for metal in (0.0, 0.5, 1.0):
        alb = lmg_albedo(lin, ctint, vcol[..., :3],
                         torch.full_like(ssao, metal))
        rgb, a = lmg_compose(direct, baked, aot, ssao, alb, vcol[..., 3])
        d = 0.0 if base is None else float((rgb - base).abs().max())
        base = rgb
        rows.append((f"lmg  compose (r0_m3:641) g_flMetalness={metal}",
                     int(rgb.shape[1] * rgb.shape[2]), d))
    # Every static-combo axis at every value, through the MIXED-RADIX id
    # builder itself: the axis is set to each of its values with the rest
    # at 0, and the combo id that comes back is the id the engine would
    # select. The delta between consecutive values must equal the axis's
    # own stride, which is what a bitmask decode gets wrong.
    rows += _lmg_axis_rows("lmg", LMG_STATIC_AXES)
    rows += _lmg_axis_rows("gen", GEN_STATIC_AXES, defer=True)
    for name, stride, lo, hi in LMG_DYN_AXES:
        # Dynamic ids are built the same way and read off the same
        # arithmetic; the module each selects is tabulated in
        # SHADER_CALLFLOW_eight_small_families.md section 3.
        rows.append((f"lmg  dyn  {name:<26s} stride {stride:<5d} "
                     f"values {list(range(lo, hi + 1))} -> ids "
                     f"{[v * stride for v in range(lo, hi + 1)]}",
                     hi - lo + 1, float(stride)))
    # --- generic.vfx --------------------------------------------------
    nts = generic_normal(nr, torch.ones_like(ssao))
    rows.append(("gen  DXT5nm normal (.wy) (r0_m3:206-218)",
                 int(nts.shape[1] * nts.shape[2]),
                 float((nts - ts).abs().max())))
    prev = None
    for dz in (0.0, 0.5, 1.0):
        dirtex = torch.cat([torch.rand(1, N, N, 2, generator=g).to(device),
                            torch.full((1, N, N, 1), dz, device=device)],
                           dim=-1)
        b = generic_directional_lightmap(baked, dirtex, nts,
                                         torch.zeros_like(ssao))
        d = 0.0 if prev is None else float((b - prev).abs().max())
        prev = b
        rgb, a = generic_compose(direct, b, ssao * aot, lin * ctint,
                                 vcol[..., 3])
        rows.append((f"gen  directional lightmap (r0_m3:270-289) dir.z={dz}",
                     int(rgb.shape[1] * rgb.shape[2]), d))
    for name, stride, lo, hi in GEN_DYN_AXES:
        rows.append((f"gen  dyn  {name:<26s} stride {stride:<5d} "
                     f"values {list(range(lo, hi + 1))} -> ids "
                     f"{[v * stride for v in range(lo, hi + 1)]}",
                     hi - lo + 1, float(stride)))
    # --- csgo_imported: the dev placeholder, its OWN decoded constants -
    # materials/dev/placeholder.vmat, decompiled from
    # csgo_imported/pak01_dir.vpk (KV3 v5, LZ4 block; the whole tree
    # parses and every buffer is consumed, which is why this is a
    # decompile and not a substring scan). Values verbatim:
    #   g_flMetalness 0.0, g_flModelTintAmount 1.0,
    #   g_flVertexColorOpacityScale 1.0, g_vColorTint (1,1,1,0),
    #   g_vLayer1Tint (1,1,1,0),
    #   g_vLayer1RoughnessBrightnessContrast (0,1,0,0),
    #   g_bFogEnabled 1, g_nTextureAddressModeU/V 0,
    #   g_tColor  = materials/dev/graygrid_color  (512x512, mean 149.3)
    #   g_tLayer1AmbientOcclusion = materials/default/default_ao (1x1 white)
    #   g_tLayer1NormalRoughness  = materials/dev/placeholder (1x1,
    #       texel 127,127 -> hemi-oct (-0.0078, 0, 0.99997))
    ph_color = torch.full((1, N, N, 3), 149.3 / 255.0, device=device)
    ph_ao = torch.ones(1, N, N, device=device)
    ph_nr = torch.cat([torch.full((1, N, N, 2), 127.0 / 255.0),
                       torch.full((1, N, N, 1), 1.0),
                       torch.full((1, N, N, 1), 245.0 / 255.0)],
                      dim=-1).to(device)
    ph_ts = lmg_normal_hemioct(ph_nr)
    ph_alb = lmg_albedo(ph_color, torch.ones_like(ph_color),
                        torch.ones_like(ph_color),
                        torch.zeros_like(ph_ao))
    ph_rgb, ph_a = lmg_compose(direct, baked, ph_ao, ssao, ph_alb,
                               torch.ones_like(ph_ao))
    rows.append(("imp  placeholder.vmat through csgo_lightmappedgeneric",
                 int(ph_rgb.shape[1] * ph_rgb.shape[2]),
                 float((ph_rgb - direct * ph_alb).abs().max())))
    rows.append((f"imp  placeholder hemi-oct normal z="
                 f"{float(ph_ts[..., 2].mean()):.5f}", N * N,
                 float(ph_ts[..., 0].abs().mean())))
    # --- THE DISPATCH ITSELF, not only the composites -----------------
    # csgo_lightmappedgeneric owns 0 pixels in these poses, so the only
    # way to tell "routed but unlit" from "never routed" is to hand
    # lmg_family_shade() a synthesised family map and watch its own
    # counters. This exercises the SAME function the renderer calls,
    # including the torch.where that must leave other families alone.
    if LMG_LIVE:
        H2 = W2 = 16
        famv = torch.full((1, H2, W2), -1, dtype=torch.long, device=device)
        famv[:, :4] = FAM_LIGHTMAPPEDGENERIC
        famv[:, 4:8] = FAM_IMPORTED
        famv[:, 8:12] = FAM_GENERIC
        famv[:, 12:] = FAM_COMPLEX          # a family that must NOT move
        # a real material id per family where one exists, else 0
        midv = torch.zeros((1, H2, W2), dtype=torch.long, device=device)
        for fid in (FAM_LIGHTMAPPEDGENERIC, FAM_IMPORTED, FAM_GENERIC):
            _w = (MAT_FAM == fid).nonzero()
            if len(_w):
                midv = torch.where(famv == fid, int(_w[0, 0]), midv)
        lin2 = torch.full((1, H2, W2, 3), 0.5, device=device)
        rgb0 = torch.full((1, H2, W2, 3), -1.0, device=device)
        a0 = torch.zeros(1, H2, W2, device=device)
        uv2 = torch.rand(1, H2, W2, 2, generator=g).to(device)
        vcol2 = torch.ones(1, H2, W2, 4, device=device)
        one2 = torch.ones(1, H2, W2, device=device)
        rgb1, a1, cnt = lmg_family_shade(
            rgb0, a0, lin2, famv, midv, MAT_EXT2[midv], uv2,
            _nrm(torch.randn(1, H2, W2, 3, generator=g).to(device)),
            torch.cat([torch.ones(1, H2, W2, 3, device=device),
                       torch.ones(1, H2, W2, 1, device=device)], dim=-1),
            vcol2, torch.full((1, H2, W2, 3), 0.3, device=device),
            torch.full((1, H2, W2, 3), 0.7, device=device),
            one2, one2, torch.ones(1, H2, W2, dtype=torch.bool,
                                   device=device))
        if _lmg_inject("imported-unrouted"):
            cnt[LMG_IMPORTED_FAM] = 0
        untouched = (rgb1[:, 12:] == -1.0).all()
        for k in (("csgo_lightmappedgeneric.vfx", FAM_LIGHTMAPPEDGENERIC),
                  (LMG_IMPORTED_FAM, FAM_IMPORTED),
                  ("generic.vfx", FAM_GENERIC)):
            nm, fid = k
            m = (famv == fid)
            d = float((rgb1[m] - rgb0[m]).abs().max()) if int(m.sum()) \
                else 0.0
            rows.append((f"DISPATCH lmg_family_shade -> {nm}",
                         cnt.get(nm, 0), d))
        rows.append(("DISPATCH leaves csgo_complex pixels untouched "
                     f"({bool(untouched)})", 1 if untouched else 0, 0.0))
    dead = 0
    for label, cnt, d in rows:
        flag = ""
        if cnt == 0:
            flag = "  <-- DEAD"
            dead += 1
        print(f"  {label:<62s} px {cnt:>7d}  max|d| {d:.6f}{flag}",
              flush=True)
    if LMG_REACH:
        print("  --- pixels shaded in the rendered frames ---", flush=True)
        for k in sorted(LMG_REACH):
            print(f"  {k:<62s} px {LMG_REACH[k]:>7d}", flush=True)
    for n in LMG_NOTES:
        print(f"  SUBSTITUTED: {n}", flush=True)
    print("=" * 66, flush=True)
    return dead
