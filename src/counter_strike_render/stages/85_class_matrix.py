

def class_matrix_report():
    """Print the population-vs-pixels table. Returns the UNSEEN class list.

    TWO COLUMNS, BECAUSE THE PIXEL COUNT ALONE IS NOT COMPLETENESS -- and
    the first version of this table conflated them. CLASS_PX counts pixels
    whose FACE belongs to a class, which says the class was ON SCREEN and
    shaded by something. It does NOT say the class's own transcribed path
    ran. csgo_black_unlit proves the gap: it counted 850 px across the
    sweep while --sf-black-unlit, the flag that selects its 115-line
    module, defaults OFF. Those pixels came from a substitute.

    So a class is complete only when it is visible AND its own path is
    enabled. Visible-with-the-path-off is the CLASS BUG this matrix exists
    to surface; it is louder than UNSEEN, because UNSEEN might be a camera
    angle and this cannot be.
    """
    if not FAM_NAMES:
        print("class matrix: UNAVAILABLE -- no --fam-side, so the renderer "
              "cannot name the shader class of a face. This is the gate "
              "that decides completeness and it is OFF; pass --fam-side.",
              flush=True)
        return []
    try:
        faces = torch.bincount(MAT_FAM[data["face_matid"].to(device).long()]
                               .reshape(-1).long(),
                               minlength=len(FAM_NAMES)).tolist()
    except Exception as e:
        print(f"class matrix: population census FAILED ({e}); pixel counts "
              f"alone cannot say whether a zero is a defect.", flush=True)
        return []
    tot_f = float(sum(faces)) or 1.0
    rows = sorted(range(len(FAM_NAMES)), key=lambda i: -faces[i])
    print("=" * 78, flush=True)
    print("CLASS COVERAGE MATRIX -- population from the pack, pixels from "
          "this run", flush=True)
    print("%-42s %10s %7s %12s  %s"
          % ("shader class", "faces", "share", "px drawn", "verdict"),
          flush=True)
    unseen, substituted = [], []
    en = _class_path_enabled()
    _blend = globals().get("BLEND_FACES_BY_FAM") or {}
    for i in rows:
        f, px = faces[i], CLASS_PX.get(i, 0)
        nm = FAM_NAMES[i]
        # a class with no flag of its own is always-on by construction
        on = en.get(nm, True)
        _bl = _blend.get(i, 0)
        if f == 0:
            verdict = "no population in this pack"
        elif px == 0 and _bl >= f:
            # Every face of this class is blend-class. The plain blend
            # composite writes fam_pix and is therefore counted; the MBOIT
            # route writes it too, but only on its `_cov` resolve, so a
            # zero here still cannot convict a wholly-blend class. Stated,
            # not silently folded in.
            verdict = (f"NOT MEASURED -- all {f} faces are blend-class; "
                       f"0 px attributed (plain blend and the MBOIT "
                       f"resolve both write fam_pix; a moment-buffer "
                       f"contribution that never resolves does not)")
        elif px > 0 and on:
            verdict = "DREW (own path)"
        elif px > 0 and not on:
            verdict = "** VISIBLE but its OWN PATH IS OFF -> substitute **"
            substituted.append(nm)
        else:
            # NO per-class blind-spot caveat any more. Every class writes
            # the one attribution map and the balance line below says
            # whether anything went uncounted, so a 0 here is now a real
            # measurement rather than a possible blind spot (#80 (c)).
            verdict = "UNSEEN (not on screen, or cannot draw)"
            unseen.append(nm)
        print("  %-40s %10d %6.3f%% %12d  %s"
              % (nm, f, 100.0 * f / tot_f, px, verdict), flush=True)
    if _blend:
        # machine-readable companion to CLASSMATRIX: rollups must exclude
        # blend-class faces from any UNSEEN conviction. Not a blind spot
        # any more -- blend pixels ARE counted; this is the population a
        # rollup needs to know cannot be convicted by a zero.
        print("CLASSMATRIX_BLEND " + " ".join(
            f"{FAM_NAMES[i]}={n}" for i, n in sorted(_blend.items())),
            flush=True)
    # ---- THE CONSERVATION LAW, PRINTED (#80 (c)) -----------------------
    # The census states its own balance instead of asking to be trusted.
    # `shaded` is the coverage the raster classes wrote into `fg`, reached
    # independently of the attribution map; `attributed` is what the single
    # counter actually credited to a class. They must be equal. `blend-only`
    # is counted on top because a blend surface over background shades a
    # pixel that never entered `fg`, so it is a real attribution that is
    # legitimately outside the identity rather than a discrepancy.
    _sh, _ct = CLASS_PX_SHADED[0], CLASS_PX_COUNTED[0]
    _lk, _ex = CLASS_PX_LEAK[0], CLASS_PX_EXTRA[0]
    _sum = sum(CLASS_PX.values())
    if _sum != _ct:
        print(f"class census: INTERNAL INCONSISTENCY -- the per-class rows "
              f"sum to {_sum:,} but the counter attributed {_ct:,}. The "
              f"table below is not a partition of what was counted.",
              flush=True)
    print(f"class census balance over {CLASS_PX_CHUNKS[0]} chunk(s): "
          f"{_sh:,} shaded (fg) -> {_sh - _lk:,} attributed + {_lk:,} "
          f"UNATTRIBUTED; +{_ex:,} blend-only over background; "
          f"{_ct:,} counted total. "
          + ("no unclaimed pixel."
             if _lk == 0 else
             f"UNCLAIMED by {_lk:,}: a raster class wrote coverage "
             f"without writing the attribution map, and its pixels are "
             f"MISSING from the rows above."), flush=True)
    # ---- THE SECOND OPERAND: what each class WON at its coverage site --
    # The unclaimed count above cannot see a class that fails to OVERWRITE
    # a pixel another class already claimed -- verified by planting exactly
    # that defect, which left the balance reading clean while 76 foliage
    # pixels sat in csgo_lightmappedgeneric's cell. So the census also
    # compares, per class, the pixels won at the coverage site against the
    # pixels the attribution map credits. won > 0 with counted == 0 is a
    # class drawing without claiming: the #80 defect itself.
    _wsum = sum(CLASS_PX_WON.values())
    _all_silent = [(FAM_NAMES[i], w) for i, w in sorted(CLASS_PX_WON.items())
                   if w > 0 and CLASS_PX.get(i, 0) == 0]
    # PARTITIONED, because the two are different facts and only one is a
    # defect. A viewmodel family CANNOT reach the attribution map -- the
    # pass runs after the census closes -- so counting it in the verdict
    # would make `balance=DEFECT` the permanent state of every run that
    # draws a gun, and a verdict that is always DEFECT decides nothing.
    _VM_VFX_NAMES = {v.split(" (")[0] for v in VM_FAM_VFX.values()}
    _silent = [(n, w) for n, w in _all_silent if n not in _VM_VFX_NAMES]
    _silent_vm = [(n, w) for n, w in _all_silent if n in _VM_VFX_NAMES]
    if _all_silent:
        for _nm, _w in _all_silent:
            if _nm in _VM_VFX_NAMES:
                # STRUCTURAL, NOT A MISSING WRITE, and the difference is
                # the whole point of printing it separately. The viewmodel
                # pass composites AFTER class_px_accumulate has closed and
                # `fam_pix` has left scope, so it CAN report coverage (it
                # now does, just above) and CANNOT report attribution. The
                # fix is to move the census after the viewmodel composite,
                # which reorders two passes and is not this line's job.
                print(f"class census VIEWMODEL UNCLAIMABLE: {_nm} WON "
                      f"{_w:,} pixels at its coverage site and the "
                      f"attribution map credits it 0. That is STRUCTURAL: "
                      f"viewmodel_pass runs after class_px_accumulate "
                      f"closes and fam_pix has left scope. It is not a "
                      f"missing fam_pix write and must not be counted as "
                      f"one -- but the pixels ARE real and the fixture CAN "
                      f"now falsify a claim about this family.", flush=True)
                continue
            print(f"class census DEFECT: {_nm} WON {_w:,} pixels at its "
                  f"coverage site but the attribution map credits it 0. "
                  f"That class draws without claiming -- its fam_pix write "
                  f"is missing or dead, and its pixels are being counted "
                  f"as whichever class they overwrote.", flush=True)
    print(f"class census attribution check: {_wsum:,} pixels won across "
          f"all classes (a pixel won more than once is legitimately "
          f"counted once, by its LAST writer, so won >= counted is "
          f"expected); {len(_silent)} class(es) won pixels but were "
          f"credited none"
          + (f", plus {len(_silent_vm)} viewmodel class(es) that CANNOT be "
             f"credited by construction (listed above as UNCLAIMABLE, not "
             f"counted as defects)" if _silent_vm else "")
          + ". "
          + ("CONSISTENT." if not _silent else "SEE THE DEFECT LINES ABOVE."),
          flush=True)
    # ---- THE MBOIT ROUTE, stated in numbers (#80 residual (a)) ---------
    # Every other composite in this file has ONE class writing a pixel.
    # The MBOIT route has as many as there are depth-peel layers, and a
    # single-owner attribution map must therefore pick one. These numbers
    # say what picking cost, instead of leaving it to be assumed:
    #   attributed      pixels this route credited to a class
    #   >=2 classes     pixels where more than one class put colour in --
    #                   the residue a single-owner map cannot represent,
    #                   and the reason a contributing class can show
    #                   won > 0 with counted == 0 here without that being
    #                   a defect
    #   dom != nearest  pixels where the DOMINANT contributor is not the
    #                   nearest layer. Attributing by nearest is the
    #                   draw-call-0 mistake; this is how often it differs
    #   0.05..0.5       pixels that entered `fg` and, before this, were
    #                   attributed to NOBODY -- they sat in the
    #                   UNATTRIBUTED residue of the balance line above
    if MBOIT_N and MBOIT_ATTR[0]:
        _ma = MBOIT_ATTR
        print(f"class census MBOIT route: {_ma[0]:,} px attributed at the "
              f"pass-2 write site; {_ma[1]:,} had >=2 contributing classes "
              f"({100.0 * _ma[1] / max(1, _ma[0]):.2f}%); {_ma[2]:,} had a "
              f"DOMINANT contributor that is not the nearest layer "
              f"({100.0 * _ma[2] / max(1, _ma[0]):.2f}%); {_ma[3]:,} lie in "
              f"0.05 < coverage <= 0.5 and were UNATTRIBUTED before this "
              f"route joined the law; {_ma[4]:,} reached fg with no layer "
              f"above the contribution floor and are DELIBERATELY left "
              f"unattributed rather than credited to material 0.",
              flush=True)
        if _silent and _ma[1]:
            print(f"class census MBOIT note: {_ma[1]:,} pixels carry more "
                  f"than one class, so a class whose only contributions "
                  f"were non-dominant MBOIT layers appears in the DEFECT "
                  f"lines above. That is a limit of a single-owner map, "
                  f"not a missing fam_pix write -- read the two together.",
                  flush=True)
    elif MBOIT_N:
        print("class census MBOIT route: ON and attributed 0 px -- either "
              "no blend face was peeled this run, or the route wrote "
              "coverage without reaching its attribution line.", flush=True)
    # WHAT THIS STILL CANNOT SEE, stated so the balance is not read as a
    # proof of completeness: a class with NEITHER a coverage tally nor an
    # attribution write -- i.e. one added later that touches neither line
    # -- is invisible to both operands. Two independent lines must both be
    # forgotten for that, where before ONE omission was enough and did
    # happen; it is a weaker failure mode, not an impossible one.
    # machine-readable, one line, for corpus rollup: a class UNSEEN on every
    # frame of a sweep is a CLASS BUG; on one frame it is a camera angle.
    # `attribution=` names the map the pixel column was counted from, so a
    # rollup can tell the eras apart: everything before #80 (c) counted an
    # OPAQUE-ONLY map under UNION coverage and could not see the mask class
    # at all, which is not a smaller version of this number but a different
    # measurement.
    # ---- THE VIEWMODEL PASS, AND WHY IT IS NOT IN THE TABLE ABOVE -----
    # #103. Stated unconditionally, including when it drew nothing, because
    # the whole defect was a zero being read as a measurement.
    _vmtot = sum(VM_CLASS_PX.values())
    print("class census VIEWMODEL SCOPE: the pixel column above CANNOT see "
          "the viewmodel pass. fam_pix is written at four world-class sites "
          "inside render_window and viewmodel_pass runs on the finished "
          "frame, after class_px_accumulate has closed. A 0 in a row for a "
          "family that only appears on the viewmodel -- csgo_weapon.vfx, "
          "csgo_legs_prepass.vfx -- is a SCOPE statement, not a "
          "measurement.", flush=True)
    if _vmtot:
        for _k in sorted(VM_CLASS_PX):
            print("  viewmodel %-14s %-34s %12d px"
                  % (VM_FAM_NAME.get(_k, "?%d" % _k),
                     VM_FAM_VFX.get(_k, "?"), VM_CLASS_PX[_k]), flush=True)
        print("  viewmodel total %d px, counted by the viewmodel pass's own "
              "per-family map (a DIFFERENT tensor that shares the name "
              "`fam_pix` with the world attribution map -- they are not the "
              "same quantity and conflating them is how this zero was read "
              "as a routing fact)." % _vmtot, flush=True)
    else:
        print("  viewmodel drew 0 px this run (no --viewmodel/--weapon, or "
              "the pass reported ABSENT). Its families are absent from the "
              "table above for BOTH reasons and neither can be separated "
              "from a run that did not draw one.", flush=True)
    print("CLASSMATRIX_VIEWMODEL " + " ".join(
        "%s=%d" % (VM_FAM_NAME.get(k, "?%d" % k), v)
        for k, v in sorted(VM_CLASS_PX.items())) or "CLASSMATRIX_VIEWMODEL -",
        flush=True)
    print("CLASSMATRIX attribution=fam_pix "
          f"balance={'ok' if (_lk == 0 and not _silent) else 'DEFECT'} " + " ".join(
              f"{FAM_NAMES[i]}={faces[i]}:{CLASS_PX.get(i, 0)}:"
              f"{int(_class_path_enabled().get(FAM_NAMES[i], True))}"
              for i in rows),
          flush=True)
    # ---- ANIMATION STATE, the same instrument extended -----------------
    # The owner's extension is MOTION, not presence, and presence is what
    # every column above measures. A weapon frozen at frame 0 and a weapon
    # playing its shoot clip both read DREW (own path); nothing here could
    # tell them apart, which is how "combat_gun's weapon is frame-0 static"
    # survived a matrix that called the renderer complete.
    #
    # THREE STATES, because they fail separately and are owned separately:
    #   clips     the asset DECLARES sequences (a read of the bundle)
    #   decoded   per-frame poses are actually loaded (the bake)
    #   driven    something SELECTED a clip and a t this run (the runtime)
    # An asset can declare five sequences, have none decoded, and be driven
    # by nothing -- which is exactly today's state, and saying so is the
    # point of adding the columns before the sampler rather than after.
    print("ANIMATION STATE", flush=True)
    print("  %-22s %-28s %9s %9s  %s"
          % ("asset", "clips declared", "decoded", "driven", "note"),
          flush=True)
    _anim_rows = []
    if VM_BUNDLE_INFO.get("name"):
        _seq = list(VM_BUNDLE_INFO.get("sequences") or [])
        _anim_rows.append((
            VM_BUNDLE_INFO.get("name", "viewmodel"),
            ",".join(_seq[:4]) + ("..." if len(_seq) > 4 else "") or "none",
            len(ANIM_DECODED.get("viewmodel", ())),
            ANIM_DRIVEN.get("viewmodel", "none"),
            "frame-0 static while decoded=0" if not ANIM_DECODED.get(
                "viewmodel") else ""))
    # PM_BUNDLES, not a PM_REACH I invented -- the first version of this
    # block named a counter that does not exist and died in the report with
    # NameError. Read the name, never derive it; the rule has a memory entry
    # and I still guessed.
    if PM_BUNDLES:
        # THE NOTE IS A READ OF THE THREE COLUMNS, not a constant. It said
        # "bind-pose while decoded=0" whenever decoded was 0, which was
        # correct until the sidecar landed and then said nothing at all in
        # the state that matters most: clips DECODED and yet NOTHING
        # DRIVEN, i.e. the sampler loaded and no entity selected. That is
        # the silent state, so it gets the loudest note.
        _pmd = len(ANIM_DECODED.get("playermodel", ()))
        _pmr = ANIM_DRIVEN.get("playermodel", "none")
        _anim_rows.append(("playermodels (%d bundle)" % len(PM_BUNDLES),
                           ",".join(PM_SEQUENCES[:4]) or "none",
                           _pmd, _pmr,
                           "bind-pose while decoded=0"
                           if not _pmd else
                           ("⚠️ decoded but NOTHING driven -- every entity "
                            "held bind pose" if _pmr == "none" else
                            "driven per entity per tick; counts are "
                            "entity-frames")))
    if not _anim_rows:
        print("  (no animated asset drew this run)", flush=True)
    for _n, _c, _d, _dr, _note in _anim_rows:
        print("  %-22s %-28s %9d %9s  %s" % (_n, _c, _d, _dr, _note),
              flush=True)
    print("ANIMMATRIX " + " ".join(
        f"{_n}=clips:{len((_c or '').split(',')) if _c != 'none' else 0}:"
        f"decoded:{_d}:driven:{_dr}" for _n, _c, _d, _dr, _ in _anim_rows),
        flush=True)
    if any(_d == 0 for _, _, _d, _, _ in _anim_rows):
        print("animation: at least one animated asset has ZERO decoded "
              "poses, so it renders at its bind/hold pose regardless of "
              "tick. That is a BAKE gap, not a runtime one -- the clip "
              "names are read from the bundle and the sampler has nothing "
              "to sample. Nothing here is a substitute for that data and "
              "none is invented.", flush=True)
    print("=" * 78, flush=True)

    if substituted:
        print(f"class matrix: {len(substituted)} class(es) are ON SCREEN "
              f"while their own transcribed path is switched off, so those "
              f"pixels came from a substitute: {', '.join(substituted)}. "
              f"This is not a camera angle and not an absence -- it is a "
              f"written path that no default reaches.", flush=True)
    if unseen:
        print(f"class matrix: {len(unseen)} class(es) with population drew "
              f"NO pixels this run: {', '.join(unseen)}. On a single frame "
              f"that is a camera angle; across a corpus sweep it is a class "
              f"bug. Aggregate the CLASSMATRIX lines to tell them apart.",
              flush=True)
    print("=" * 78, flush=True)
    return unseen
# [pixels the transcribed water path actually shaded, pixels considered].
# A function that is defined and never called is not implemented, and a
# family whose material never rasterises is not reached; this counts the
# only thing that settles either -- pixels out of water_fancy_shade.
WATER_REACH = [0.0, 0.0]
# --- non-world families: per-run pixel counters and the derived inputs ---
# FAM_HIT is the REACHABILITY measurement: it counts covered pixels that
# actually took each family's path, and the run prints it. A zero here is
# reported as a zero, never as silence.
FAM_HIT = {"character": 0, "eyeball": 0, "customglove": 0}
MAT_SEEN = set()
MATID_DUMP = []
PROBE = []
LDUMP = []
HDR_DUMP = []
_DBG_MATID = None
# lm_ok rides through perspective-correct interpolation, so it is 1.0 only
# where all three vertices of the triangle are lightmapped. Anything less
# means the UVs are a blend across a chart boundary (or across a prop and a
# wall) and would sample an unrelated part of the atlas.
LM_OK_T = 0.999


SAO_G = _lf_tables.BILATERAL_5


def _bilateral_key_scale():
    return args.ssao_blur_edge / 2000.0


def _sao_blur(A, key, fg):
    """ssao_bilateral_blur: separable, depth-keyed, 2*R+1 taps.

    The AO estimator is deliberately under-sampled (11 taps on a spiral);
    the blur is what turns that into a smooth field, and it is a SEPARATE
    shader in CS2 precisely because it materially changes the look. It
    must not cross a depth discontinuity, or foreground AO bleeds onto
    the background.
    """
    r, st = args.ssao_blur, args.ssao_blur_scale
    for axis in (2, 1):
        pad = r * st
        pk = (pad, pad, 0, 0) if axis == 2 else (0, 0, pad, pad)
        Ap = torch.nn.functional.pad(A.unsqueeze(1), pk, mode="replicate")[:, 0]
        Kp = torch.nn.functional.pad(key.unsqueeze(1), pk, mode="replicate")[:, 0]
        n = A.shape[axis]
        acc = A * SAO_G[0]
        wsum = torch.full_like(A, SAO_G[0])
        for d in range(-r, r + 1):
            if d == 0:
                continue
            o = pad + d * st
            sl = (slice(None), slice(o, o + n)) if axis == 1 else \
                 (slice(None), slice(None), slice(o, o + n))
            if axis == 1:
                a_t, k_t = Ap[:, o:o + n], Kp[:, o:o + n]
            else:
                a_t, k_t = Ap[:, :, o:o + n], Kp[:, :, o:o + n]
            _ks = _bilateral_key_scale()
            w = _lf_ao.bilateral_weight(
                float(_lf_tables.row_bilateral_5(min(abs(d), len(SAO_G) - 1))), key * _ks, k_t * _ks)
            acc = acc + a_t * w
            wsum = wsum + w
        A = _lf_ao.bilateral_resolve(acc, wsum)
    return A


_SAO_DEFAULTS_PRINTED = False


def _sao_defaults_banner():
    """Name the SAO inputs the renderer cannot supply, at RUNTIME (#34)."""
    global _SAO_DEFAULTS_PRINTED
    if _SAO_DEFAULTS_PRINTED:
        return
    _SAO_DEFAULTS_PRINTED = True
    print("[sao] ssao_scalable_ambient_obscurance r0/m3 terms are LIVE: "
          "t01 hash_jitter, t03 mip_from_radius, t04 tap_weight, "
          "t05 resolve_pow14.")
    print("[sao]   spin phase  = 0.0                    the reference's "
          "_5618._m6 is a per-frame uniform with no counterpart here")
    print("[sao]   tap count   = --ssao-samples         the module's is a "
          "literal 11; ours is configurable, so the normaliser follows "
          "OURS")
    print("[sao]   NOT wired: t02 spiral_tap (our angle/radius schedule "
          "differs), t06 checker_denoise (no quad derivatives in this "
          "pass).")
    print("[sao]   The tap weight and the resolve CHANGED SHAPE, not just "
          "constants -- linear falloff over an undivided cosine, and a 1.4 "
          "exponent the inline never had. Expect the AO term to move.")


def sao(wpos, nrm, fg, eye, fwd):
    """Scalable Ambient Obscurance, all four CS2 stages. Returns (B,H,W)."""
    B = fg.shape[0]
    R, R2 = args.ssao_radius, args.ssao_radius ** 2
    NS = args.ssao_samples
    # --- ssao_convert_depth -------------------------------------------
    zc = ((wpos - eye.view(-1, 1, 1, 3)) * fwd.view(-1, 1, 1, 3)).sum(-1)
    zc = torch.where(fg, zc, torch.full_like(zc, 1e4)).clamp(min=1e-3)
    # --- ssao_downsample_depth: POINT-sampled chain --------------------
    Pm, Fm = [wpos], [fg]
    for _ in range(args.ssao_mips):
        Pm.append(Pm[-1][:, ::2, ::2].contiguous())
        Fm.append(Fm[-1][:, ::2, ::2].contiguous())
    NL = len(Pm)
    # pixels per metre at unit depth
    projScale = 0.5 * W / math.tan(math.radians(args.hfov) / 2)
    ssDiskR = (projScale * R / zc).clamp(1.0, max(W, H))
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    # THE SPIRAL ROTATION IS NOW THE SHADER'S, NOT THE PAPER'S. The inline
    # form was `((3x ^ y) + x*y) * 10.0`; the module writes
    # `((3x) ^ (y + x*y)) * 8.0 + phase` -- a different xor GROUPING and a
    # different multiplier, so the two hashes decorrelate differently and
    # neither is a rounding of the other. Read out of
    # ssao_scalable_ambient_obscurance_r0m3.glsl:33 and registered as
    # sao.t01.hash_jitter.
    _sao_defaults_banner()
    # The reference's phase is the per-frame uniform _5618._m6; this
    # renderer has no counterpart, so it is DEFAULTED to 0 and says so at
    # runtime rather than acquiring a flag nobody set.
    spin = _lf_ao.sao_hash_jitter(xx, yy, 0.0)
    # RANGE-REDUCE THE HASH. It is an integer product, so at 1080p it
    # reaches 1.66e7 -- measured, not estimated -- where one float32 ulp is
    # about 1.0 radian and the spiral's angular decorrelation is gone. The
    # reference reduces with `a - 2pi * trunc(a / 2pi)` inside t02
    # (:47-50); our angle schedule differs so t02 is not wired, but the
    # REDUCTION is not schedule-specific and its absence is a precision
    # defect, not a stylistic difference. Written in the reference's exact
    # form -- trunc, not floor -- because the two differ in sign for a
    # negative argument and the hash can be negative once a phase is added.
    _TAU = 6.283185482025146484375
    spin = spin - _TAU * torch.trunc(spin / _TAU)
    bidx = torch.arange(B, device=device).view(-1, 1, 1)
    total = torch.zeros_like(zc)
    for i in range(NS):
        alpha = (i + 0.5) / NS
        ang = alpha * (args.ssao_turns * 2 * math.pi) + spin
        ssR = alpha * ssDiskR
        # sao.t03.mip_from_radius. The module clamps to 5; this chain is
        # NL levels deep, so the upper bound stays the chain's -- a mip the
        # renderer does not have is not a mip the reference can select.
        lvl = _lf_ao.sao_mip_from_radius(ssR).clamp(0, NL - 1).long()
        u = xx.float() + torch.cos(ang) * ssR
        v = yy.float() + torch.sin(ang) * ssR
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui, vi = u.long(), v.long()
        Q = torch.zeros_like(wpos)
        ok = torch.zeros_like(fg)
        for L in range(NL):
            hl, wl = Fm[L].shape[1], Fm[L].shape[2]
            ul = (ui >> L).clamp(0, wl - 1)
            vl = (vi >> L).clamp(0, hl - 1)
            sel = lvl == L
            Q = torch.where(sel.unsqueeze(-1), Pm[L][bidx, vl, ul], Q)
            ok = torch.where(sel, Fm[L][bidx, vl, ul], ok)
        vvec = Q - wpos
        # sao.t04.tap_weight, from
        # ssao_scalable_ambient_obscurance_r0m3.glsl:57. The inline form
        # was `(R2-vv)^3 * ((v.n - bias) / (vv + 1e-4))` -- a CUBED range
        # falloff over a DIVIDED cosine. The reference is
        # `4 * max(1 - vv/R2, 0) * max(v.n - bias, 0)`: LINEAR falloff, and
        # the cosine is NOT divided by the squared distance. Different
        # estimator, not a rounding, and the 1e-4 guard the inline needed
        # for that divide has no counterpart because there is no divide.
        c = _lf_ao.sao_tap_weight(vvec, nrm, 1.0 / R2, args.ssao_bias)
        total = total + torch.where(ok & inb, c, torch.zeros_like(c))
    # sao.t05.resolve_pow14 (:63). The inline resolve had NO exponent; the
    # reference raises the saturated complement to 1.4. NS stands in for
    # the module's literal 11 tap count -- ours is configurable and its is
    # not, so the normaliser follows OUR loop rather than hard-coding a
    # count this renderer does not run.
    # `norm` is chosen so the pre-exponent argument is IDENTICAL to what
    # the inline resolve computed: inline was
    # `1 - total * (intensity / R^6) * (5 / NS)`, and
    # `intensity * acc / (n_taps * norm)` with n_taps = NS reproduces that
    # exactly at norm = R^6 / 5. So the ONLY behavioural change from this
    # line is the reference's 1.4 exponent, which the inline did not have.
    # The scale itself is carried over rather than re-fitted.
    A = _lf_ao.sao_resolve(total, args.ssao_intensity, R ** 6 / 5.0,
                           n_taps=NS)
    A = torch.where(fg, A, torch.ones_like(A))
    if args.ssao_blur > 0:
        A = _sao_blur(A, (zc / 40.0).clamp(0, 1), fg)
        A = torch.where(fg, A, torch.ones_like(A))
    return A


# =======================================================================
# CS2 SSAO SUBSYSTEM + DEPTH-KEYED FILTERS -- transcribed from bytecode
# (impl-ssao-filters). One contiguous appended block; see the parser
# banner for provenance, module counts and the slice statement.
#
# WHAT DEPTH THESE READ, resolved from the BINDING and not from axis
# names. This is the one place this slice differs structurally from the
# six world shaders, and it is worth stating plainly:
#
#   The six csgo_* world shaders are BINDLESS -- every texture is one
#   aliased descriptor array at set 4 / binding 46, and a render target
#   is identified only by which per-view CB offset the 32-bit handle
#   came from (76 / 80 / 88 / 116).
#
#   NONE OF THE SEVEN FAMILIES IN THIS SLICE WORKS THAT WAY. Every one
#   declares its inputs as DIRECT, individually-bound textures in
#   descriptor set 1:
#     set1/binding30  the primary input target   (all seven)
#     set1/binding31  the secondary input        (ssao, sao, aoproxy,
#                                                 denoise_blur, bwd)
#     set1/binding32  atrous colour, and ssao's distance buffer when
#                     D_Z_PREPASS_OUTPUTS_NORMALS displaces it
#     set1/binding34  atrous packed depth+normal
#     set1/binding14/15/16  the samplers
#   and their uniforms come from set1/binding0 and set1/binding1, NOT
#   from set1/binding3. There is no set4 binding46 declaration in any of
#   the 69 modules and no read of per-view offsets 76, 80, 88 or 116 in
#   any of them.
#
#   So the answer to "which of 76/80/88/116 does each family read" is,
#   per family: NONE. These are passes, and a pass is bound its inputs
#   directly by the frame graph rather than looking a handle up out of
#   the per-view CB the way a material does. The offsets appear in this
#   slice only in the OTHER direction -- aoproxy_splat WRITES the
#   4-channel target that offset 76 later reads.
#
#   What each does read, from the binding:
#     ssao_convert_depth   set1/b30 = the raw depth attachment; the
#                          per-view floats at std140 offsets 368/372 are
#                          the viewport min/max depth, and offset 464 is
#                          the (A,B,C) of z = A/(B*d' + C).
#     ssao_downsample_depth set1/b30 = the camera-space-Z target.
#     ssao_scalable_...    set1/b30 = the CSZ mip chain (texelFetch with
#                          an explicit lod), set1/b31 = a normal target.
#     ssao                 set1/b31 (or b32) = a RAY-DISTANCE target,
#                          not a Z target and not the depth attachment;
#                          set1/b30 = a tiling noise texture.
#     aoproxy_splat        set1/b31 = the raw depth attachment (same
#                          368/372 remap as ssao_convert_depth, which
#                          cross-checks the offsets), set1/b30 = a 3D
#                          occlusion LUT.
#     atrous_filter        set1/b32 = colour+variance, set1/b34 = packed
#                          depth + octahedral normal.
#     denoise_blur         set1/b30 = signal, set1/b31 = raw depth.
#     blur_with_depth      set1/b30 = colour, set1/b31 = raw depth.
# =======================================================================

CS2AO_STAT = {"px": 0.0, "frames": 0}
AOPROXY_STAT = {"frames": 0}


def _load_aoproxy_set():
    """The proxy-volume set aoproxy_splat splats.

    de_inferno ships no extracted proxy set, so with no --aoproxy-proxies
    this SYNTHESISES one from the scene's own bounds and PRINTS that it
    did. A synthesised input is how the pass is shown to execute at all;
    it is never presented as the map's real proxy list.
    """
    if args.aoproxy_proxies:
        with open(args.aoproxy_proxies) as fh:
            d = json.load(fh)
        print(f"aoproxy_splat: {len(d.get('spheres', []))} sphere + "
              f"{len(d.get('boxes', []))} box proxies from "
              f"{args.aoproxy_proxies}", flush=True)
        return d
    return None


AOPROXY_SET = None


def _aoproxy_set_for(wpos):
    """Return the proxy set, synthesising one from scene bounds if needed.

    Synthesis is announced, once, with the word SYNTHESISED in it, so a
    reader of the log can never mistake it for map data.
    """
    global AOPROXY_SET
    if AOPROXY_SET is not None:
        return AOPROXY_SET
    lo = wpos.reshape(-1, 3).min(0).values
    hi = wpos.reshape(-1, 3).max(0).values
    ctr = ((lo + hi) * 0.5).tolist()
    ext = ((hi - lo) * 0.25).clamp(min=0.5).tolist()
    rad = float(max(ext) * 0.5)
    AOPROXY_SET = {
        "spheres": [{"center": ctr, "radius": rad}],
        "boxes": [{"center": ctr, "ext": ext}],
    }
    print("aoproxy_splat: no --aoproxy-proxies given; proxy set "
          "SYNTHESISED from scene bounds (1 sphere r=%.2f, 1 box "
          "ext=%.2f/%.2f/%.2f) so the pass is reachable. This is NOT "
          "de_inferno's proxy list." % (rad, ext[0], ext[1], ext[2]),
          flush=True)
    return AOPROXY_SET


# ssao_bilateral_blur r0m0:16 -- the five Gaussian weights, verbatim.
CS2AO_BLUR_G = (0.15317000448703765869140625,
                0.14489300549030303955078125,
                0.12264899909496307373046875,
                0.092901997268199920654296875,
                0.06296999752521514892578125)
# ssao_scalable_ambient_obscurance r0m0:45,47 -- the spiral constants are
# LITERALS in the bytecode, so the tap count is fixed at 11 and the turn
# count is not a uniform: 3.996364116668701171875 == 2*pi*7/11 and
# 0.0909090936183929443359375 == 1/11.
CS2AO_SAO_TAPS = 11
CS2AO_SAO_ANGLE_STEP = 3.996364116668701171875
CS2AO_SAO_ALPHA_STEP = 0.0909090936183929443359375
CS2AO_SAO_HASH_SCALE = 8.0                                   # r0m0:28
CS2AO_SAO_POW = 1.39999997615814208984375                    # r0m0:60
CS2AO_SAO_KEY_SCALE = 0.00019999999494757503271102905273438  # r0m0:60
CS2AO_SAO_LOG_OFFSET = 3                                     # r0m0:48
CS2AO_SAO_MAX_MIP = 5                                        # r0m0:48

# ssao.vfx r0m0:4 / r1m0:4 / r2m0:4 -- the three S_SSAO_QUALITY sample
# kernels, verbatim from three different bytecode records. These are
# NOT a truncation of one another: quality 0's nine vectors do not
# appear in quality 1's sixteen.
CS2AO_SSAO_KERNELS = {
    0: ((0.133904, 0.036543, 0.049350), (-0.129782, -0.032608, 0.098059),
        (-0.194460, -0.413753, 0.876456), (-0.038784, 0.439983, 0.059605),
        (-0.118508, 0.001674, -0.011569), (0.003858, -0.002587, -0.000472),
        (0.007022, 0.000399, -0.005472), (-0.168108, -0.253058, -0.224711),
        (0.011223, -0.254116, -0.466867)),
    1: ((0.355512, -0.709318, -0.102371), (0.534186, 0.715110, -0.115167),
        (-0.878660, 0.157139, -0.115167), (0.140679, -0.475516, -0.063982),
        (-0.079612, 0.158842, -0.677075), (-0.075952, -0.101676, -0.483625),
        (0.124930, -0.022342, -0.483625), (-0.072007, 0.243395, -0.967251),
        (-0.207641, 0.414286, 0.187755), (-0.277332, -0.371262, 0.187755),
        (0.638640, -0.114214, 0.262857), (-0.184051, 0.622119, 0.262857),
        (0.110007, -0.219486, 0.435574), (0.235085, 0.314707, 0.696918),
        (-0.290012, 0.051865, 0.522688), (0.097509, -0.329594, 0.609803)),
    2: ((0.615195, 0.099185, 0.468574), (-0.246664, 0.495551, 0.636662),
        (-0.302699, 0.197356, 0.928992), (0.025013, -0.473721, 0.879973),
        (0.329761, -0.243882, 0.453554), (0.228839, 0.289674, 0.133150),
        (0.066389, 0.230785, 0.114893), (-0.221339, -0.148409, 0.131756),
        (-0.457569, -0.666197, 0.229464), (0.020583, -0.009117, 0.007547),
        (0.365528, 0.050190, -0.071732), (-0.073882, 0.055014, -0.009701),
        (-0.630459, 0.209926, -0.130613), (-0.038274, -0.039986, 0.009818),
        (0.004809, -0.007283, 0.001042), (0.745164, 0.323719, -0.429083),
        (-0.015386, 0.267425, -0.066036), (-0.212754, 0.079506, -0.151285),
        (-0.096176, -0.522937, -0.250845), (0.489685, -0.464403, -0.149240),
        (0.306264, 0.196200, -0.754520), (-0.131151, 0.139137, -0.700880),
        (-0.362332, -0.100502, -0.306237), (-0.043645, -0.153336, -0.129443),
        (0.293372, -0.170633, -0.661786)),
}

# denoise_blur r0m0 / r1m0:16 / r2m0:16 -- S_BOX_KERNEL. 0 is the empty
# tuple because that record has NO loop: it is texelFetch-and-write.
CS2AO_DNB_TAPS = {0: (), 1: (-6, -3, 0, 3, 6), 2: (-9, -6, -3, 0, 3, 6, 9)}

# blur_with_depth r1m0:16-17 .. r4m0:16-17 -- S_GAUSSIAN_KERNEL. Kernel
# 0's record declares no table at all. Offsets are fractional: each tap
# is a bilinear-weighted pair, which is why 13 taps cover +-11.25 texels.
CS2AO_BWD_KERNELS = {
    0: ((0.0,), (1.0,)),
    1: ((-3.0, -1.182425022125244140625, 0.0,
         1.182425022125244140625, 3.0),
        (0.0044329999946057796478271484375, 0.2960419952869415283203125,
         0.3990499973297119140625, 0.2960419952869415283203125,
         0.0044329999946057796478271484375)),
    2: ((-3.0962150096893310546875, -1.27687799930572509765625, 0.0,
         1.27687799930572509765625, 3.0962150096893310546875),
        (0.019827000796794891357421875, 0.320560991764068603515625,
         0.31922399997711181640625, 0.320560991764068603515625,
         0.019827000796794891357421875)),
    3: ((-5.142348766326904296875, -3.241796016693115234375,
         -1.37994205951690673828125, 0.0, 1.37994205951690673828125,
         3.241796016693115234375, 5.142348766326904296875),
        (0.0044869999401271343231201171875, 0.06918500363826751708984375,
         0.312325000762939453125, 0.22800500690937042236328125,
         0.312325000762939453125, 0.06918500363826751708984375,
         0.0044869999401271343231201171875)),
    4: ((-11.2518520355224609375, -9.28917217254638671875,
         -7.329586029052734375, -5.372685909271240234375,
         -3.417910099029541015625, -1.46455705165863037109375, 0.0,
         1.46455705165863037109375, 3.417910099029541015625,
         5.372685909271240234375, 7.329586029052734375,
         9.28917217254638671875, 11.2518520355224609375),
        (0.000533999991603195667266845703125, 0.00373300001956522464752197265625,
         0.018004000186920166015625, 0.059927999973297119140625,
         0.13774000108242034912109375, 0.21867699921131134033203125,
         0.1227649972, 0.21867699921131134033203125,
         0.13774000108242034912109375, 0.059927999973297119140625,
         0.018004000186920166015625, 0.00373300001956522464752197265625,
         0.000533999991603195667266845703125)),
}

# atrous_filter r0m0:5 -- the 3x3 variance prefilter, indexed [|dy|][|dx|],
# and r0m0:6 -- the separable B3-spline a-trous kernel normalised to a
# unit centre tap (1, 2/3, 1/6 is [1/16,1/4,3/8,1/4,1/16] / (3/8)).
CS2AO_ATROUS_VAR = ((0.25, 0.125), (0.125, 0.0625))
CS2AO_ATROUS_H = (1.0, 0.666666686534881591796875, 0.16666667163372039794921875)
CS2AO_LUMA = (0.2125000059604644775390625, 0.7153999805450439453125,
              0.07209999859333038330078125)

# Which named injection is armed. A STRING is compared to a STRING; the
# flag is never tested for truthiness, because every non-empty name --
# including a hypothetical "off" -- is truthy.
def _cs2filt_inject(name):
    return args.cs2filt_selftest_inject == name


def _cs2_proj_info():
    """ssao_scalable_ambient_obscurance _5618._m1 -- 'projInfo'.

    r0m0:25 reconstructs the camera-space point from CSZ alone:
        C = vec3((gl_FragCoord.xy * projInfo.xy + projInfo.zw) * z, z)
    so projInfo.xy is d(view xy)/d(pixel) per unit z and projInfo.zw is
    the value at pixel 0. For the GL projection this file builds,
    view_x/z = ndc_x / p00 with ndc_x = 2*px/W - 1, giving
    xy = 2/(W*p00) and zw = -1/p00. Row index runs top-down here while
    gl_FragCoord.y runs bottom-up, so the y row is negated -- that is a
    convention difference in OUR framebuffer, stated rather than folded
    silently into the constant.
    """
    t = math.tan(math.radians(args.hfov) / 2.0)
    p00 = 1.0 / t
    p11 = (W / float(H)) / t
    return (2.0 / (W * p00), -2.0 / (H * p11), -1.0 / p00, 1.0 / p11)


def ssao_convert_depth(zfc, msaa=None):
    """ssao_convert_depth r0m0:17, transcribed whole.

        out = m2.x / (m2.y * clamp((d - m0) / (m1 - m0), 0, 1) + m2.z)

    m0/m1 are the per-view floats at std140 offsets 368/372 -- the
    viewport min/max depth, 0 and 1 here. m2 at offset 464 is (A,B,C).
    For this file's GL frustum, d_view = p23/(ndc + p22) with ndc = 2d-1,
    which is exactly A/(B*d + C) with A = p23/2, B = 1, C = (p22-1)/2.
    So this is a transcription with the engine's (A,B,C) SOLVED for our
    projection, not a substitute formula.

    D_MSAA_DEPTH_BUFFER (dynamic place value 1) changes only the fetch
    instruction, texture2D->texture2DMS; the arithmetic is identical, so
    at one sample the two records agree exactly.
    """
    if msaa is None:
        msaa = args.cs2ao_msaa_depth
    n, f = PROJ_NEAR, PROJ_FAR
    p22 = -(f + n) / (f - n)
    p23 = -2.0 * f * n / (f - n)
    A, Bc, C = p23 / 2.0, 1.0, (p22 - 1.0) / 2.0
    minD, maxD = 0.0, 1.0
    d = ((zfc - minD) / (maxD - minD)).clamp(0.0, 1.0)
    return A / (Bc * d + C).clamp(max=-1e-9)


def ssao_downsample_depth(csz):
    """ssao_downsample_depth r0m0:11, transcribed whole.

        out[p] = csz[p*2 + ivec2((p.y & 1) ^ 1, (p.x & 1) ^ 1)].x

    A rotated quincunx: it SELECTS one of the four child texels by
    parity and never averages them, and the x offset is driven by the
    y parity (and vice versa) -- the swap is in the bytecode and is not
    a transcription slip. Averaging two depths across a silhouette
    would invent a surface at neither, which is why the engine picks.
    """
    B, Hh, Ww = csz.shape
    ho, wo = Hh // 2, Ww // 2
    yy, xx = torch.meshgrid(torch.arange(ho, device=csz.device),
                            torch.arange(wo, device=csz.device),
                            indexing="ij")
    sy = (yy * 2 + ((xx & 1) ^ 1)).clamp(0, Hh - 1)
    sx = (xx * 2 + ((yy & 1) ^ 1)).clamp(0, Ww - 1)
    return csz[:, sy, sx]


def ssao_scalable_ambient_obscurance(csz_mips, nrm_cam=None,
                                     read_normal_tex=None, box2x2=None,
                                     red_only=None):
    """ssao_scalable_ambient_obscurance r0m0, the whole estimator.

    Every constant below is a literal in the bytecode; none is fitted.
    Returns the 4-channel target the next pass consumes: .x = obscurance,
    .y = the packed bilateral key, .z = 0, .w = 1.

    THIS IS NOT THE ESTIMATOR THIS FILE PREVIOUSLY IMPLEMENTED. sao()
    above is the 2012 paper's original form -- (r^2-vv)^3 falloff,
    divided by r^6, scaled 5/n, no output curve. The shipped shader uses
    the later form at r0m0:54,60:

        sum += 4 * max(1 - vv/r^2, 0) * max(dot(v,n) - bias, 0)
        A    = pow(clamp(1 - intensity*sum / (11*r), 0, 1), 1.4)

    Different falloff, different normalisation, and a 1.4 output curve
    the fitted version has no counterpart for. Both are kept and
    selected by --cs2ao-estimator; neither is silently replaced.
    """
    if read_normal_tex is None:
        read_normal_tex = args.cs2ao_read_normal_tex
    if box2x2 is None:
        box2x2 = args.cs2ao_box2x2
    if red_only is None:
        red_only = args.cs2ao_red_only
    csz = csz_mips[0]
    B, Hh, Ww = csz.shape
    NL = len(csz_mips)
    pj = _cs2_proj_info()
    yy, xx = torch.meshgrid(torch.arange(Hh, device=csz.device),
                            torch.arange(Ww, device=csz.device),
                            indexing="ij")
    fx, fy = xx.float(), yy.float()
    # r0m0:25 -- C = vec3((fragcoord.xy*pj.xy + pj.zw) * z, z)
    z = csz
    Cx = (fx * pj[0] + pj[2]) * z
    Cy = (fy * pj[1] + pj[3]) * z
    C = torch.stack((Cx, Cy, z), dim=-1)
    # r0m0:28 -- the spiral phase hash. Note *8.0, not the paper's *10.0.
    spin = (((3 * xx) ^ (yy + xx * yy)).float() * CS2AO_SAO_HASH_SCALE
            + args.cs2ao_phase)
    if _cs2filt_inject("sao-spiral-frozen"):
        spin = torch.zeros_like(spin)          # every pixel taps alike
    # r0m0:29-31 vs r0m1:35-37 -- the normal, per D_READ_NORMAL_FROM_TEXTURE
    if read_normal_tex and nrm_cam is not None:
        n_v = nrm_cam.clone()
        n_v[..., 1] = -n_v[..., 1]             # r0m1:37, the Y negation
        n_v = torch.nn.functional.normalize(n_v, dim=-1, eps=1e-12)
    else:
        dCdx = torch.zeros_like(C)
        dCdy = torch.zeros_like(C)
        dCdx[:, :, :-1] = C[:, :, 1:] - C[:, :, :-1]
        dCdx[:, :, -1] = dCdx[:, :, -2]
        dCdy[:, :-1, :] = C[:, 1:, :] - C[:, :-1, :]
        dCdy[:, -1, :] = dCdy[:, -2, :]
        n_v = torch.nn.functional.normalize(torch.cross(dCdx, dCdy, dim=-1),
                                            dim=-1, eps=1e-12)
    R = args.cs2ao_radius
    # r0m0:32 -- radiusSS = abs(projScale*r / z); projScale is _m2.w
    projScale = 0.5 * Ww / math.tan(math.radians(args.hfov) / 2.0)
    radSS = (projScale * R / z).abs()
    bidx = torch.arange(B, device=csz.device).view(-1, 1, 1)
    total = torch.zeros_like(z)
    inv_r2 = 1.0 / (R * R)                                    # _m3.x
    for i in range(CS2AO_SAO_TAPS):
        s = float(i) + 0.5                                     # r0m0:44
        ang = s * CS2AO_SAO_ANGLE_STEP + spin                  # r0m0:45
        ang = ang - 2.0 * math.pi * torch.trunc(ang / (2.0 * math.pi))
        ssR = (s * CS2AO_SAO_ALPHA_STEP) * radSS               # r0m0:47
        # r0m0:48 -- mip = clamp(floor(log2(ssR)) - 3, 0, 5)
        lvl = (torch.log2(ssR.clamp(min=1e-6)).floor()
               - CS2AO_SAO_LOG_OFFSET).clamp(0, min(CS2AO_SAO_MAX_MIP,
                                                    NL - 1)).long()
        # r0m0:49 -- ssP = ivec2(vec2(cos,sin)*ssR) + fragcoord_i
        # ivec2() truncates toward zero; int() in torch does the same.
        ux = (torch.cos(ang) * ssR).long() + xx
        uy = (torch.sin(ang) * ssR).long() + yy
        zt = torch.zeros_like(z)
        for L in range(NL):
            hl, wl = csz_mips[L].shape[1], csz_mips[L].shape[2]
            # r0m0:51 -- clamp(ssP >> mip, 0, (screenSize >> mip) - 1)
            cl = ((ux >> L).clamp(0, wl - 1), (uy >> L).clamp(0, hl - 1))
            zt = torch.where(lvl == L, csz_mips[L][bidx, cl[1], cl[0]], zt)
        # r0m0:53 -- Q is unprojected from the TAP's own pixel centre,
        # (vec2(ssP)+0.5), while C at :25 used gl_FragCoord.xy with no
        # half-texel. That half-pixel inconsistency is in the reference.
        qx = ((ux.float() + 0.5) * pj[0] + pj[2]) * zt
        qy = ((uy.float() + 0.5) * pj[1] + pj[3]) * zt
        v = torch.stack((qx, qy, zt), dim=-1) - C
        vv = (v * v).sum(-1)
        vn = (v * n_v).sum(-1)
        # r0m0:54 -- the shipped falloff, NOT the paper's (r^2-vv)^3
        total = total + (4.0 * (1.0 - vv * inv_r2).clamp(min=0.0)) * \
            (vn - args.cs2ao_bias).clamp(min=0.0)
    # r0m0:60
    A = (1.0 - (args.cs2ao_intensity * total) / (CS2AO_SAO_TAPS * R)
         ).clamp(0.0, 1.0)
    if not _cs2filt_inject("sao-no-pow"):
        A = A.pow(CS2AO_SAO_POW)
    if box2x2:
        # r0m2:60-79 -- the cross-quad 2x2 resolve, gated on the depth
        # derivative so it cannot average across a silhouette.
        dzx = torch.zeros_like(z)
        dzx[:, :, :-1] = z[:, :, 1:] - z[:, :, :-1]
        dAx = torch.zeros_like(A)
        dAx[:, :, :-1] = A[:, :, 1:] - A[:, :, :-1]
        A = torch.where(dzx.abs() < 1.0,
                        A - dAx * ((xx & 1).float() - 0.5), A)
        dzy = torch.zeros_like(z)
        dzy[:, :-1, :] = z[:, 1:, :] - z[:, :-1, :]
        dAy = torch.zeros_like(A)
        dAy[:, :-1, :] = A[:, 1:, :] - A[:, :-1, :]
        A = torch.where(dzy.abs() < 1.0,
                        A - dAy * ((yy & 1).float() - 0.5), A)
    # r0m0:60 vs r1m0:60 -- S_OUTPUT_RED_CHANNEL_ONLY drops the key
    key = (torch.zeros_like(z) if red_only
           else (z.abs() * CS2AO_SAO_KEY_SCALE).clamp(0.0, 1.0))
    return torch.stack((A, key, torch.zeros_like(A),
                        torch.ones_like(A)), dim=-1)


def ssao_bilateral_blur(rt, axis_xy):
    """ssao_bilateral_blur r0m0, transcribed whole. rt is (B,H,W,4).

    r0m0:49-52, per tap r in [-4,4], r != 0:
        p += axis * float(r*2)                       <- stride 2, literal
        w  = (0.3 + G[|r|]) * max(0, 1 - 2000*|key_t - key_c|)
    and the CENTRE tap's weight is a bare G[0] with NO +0.3 (the r==0
    branch at :54-57 skips the body entirely and the accumulator was
    seeded at :34-35). That asymmetry is in the bytecode.
    Output at :65 keeps .y and .z so the pass can be run twice.
    """
    A, key = rt[..., 0], rt[..., 1]
    B, Hh, Ww = A.shape
    acc = A * CS2AO_BLUR_G[0]
    wsum = torch.full_like(A, CS2AO_BLUR_G[0])
    dx, dy = int(axis_xy[0]), int(axis_xy[1])
    for r in range(-4, 5):
        if r == 0:
            continue
        ox, oy = dx * r * 2, dy * r * 2
        a_t = torch.roll(A, shifts=(-oy, -ox), dims=(1, 2))
        k_t = torch.roll(key, shifts=(-oy, -ox), dims=(1, 2))
        if _cs2filt_inject("bilateral-ignores-key"):
            k_t = key
        w = (0.300000011920928955078125 + CS2AO_BLUR_G[abs(r)]) * \
            (1.0 - 2000.0 * (k_t - key).abs()).clamp(min=0.0)
        acc = acc + a_t * w
        wsum = wsum + w
    out = acc / (wsum + 9.9999997473787516355514526367188e-05)
    return torch.stack((out, key, rt[..., 2], torch.ones_like(out)), dim=-1)


def ssao_ray_distance(dist, ray_dir, eye, fg, mvp, nrm_world=None,
                      quality=None, zprepass_normals=None):
    """ssao.vfx r0m0/r1m0/r2m0 -- the OTHER shipped AO estimator.

    Its input is a RAY-DISTANCE target (set1/binding31), not a depth
    buffer and not camera-space Z: r0m0:37 reconstructs the world point
    as eye + normalize(rayDir) * dist, so the stored value is |P - eye|.

    r0m0:60-79, per sample: reflect the fixed kernel vector about a
    per-pixel noise vector, fold it into the hemisphere if it faces away
    from N, project the offset world point back to the screen, read the
    distance stored there, rebuild THAT surface point along its own ray,
    and count the sample if it is both inside the radius and nearer than
    the ray:

        occ += step(r^2 - |Q - origin|^2) * step(rayLen - storedDist)

    both written as clamp(x * 1e6, 0, 1), which is a step().
    Final at :85-86 is A = (1 - occ/n)^3.

    WHAT DIFFERS FROM THE REFERENCE, stated precisely: the noise vector
    at :44-45 comes from a tiling texture bound at set1/binding30 whose
    CONTENT is engine data and is not in the bytecode. This uses a
    deterministic per-pixel hash in its place, which reproduces the
    shader's USE of it (a unit vector to reflect about) but not its
    exact values. Nothing else here is substituted.
    """
    if quality is None:
        quality = args.cs2ao_quality
    if zprepass_normals is None:
        zprepass_normals = args.cs2ao_zprepass_normals
    ker = CS2AO_SSAO_KERNELS[int(quality)]
    if _cs2filt_inject("ssao-kernel-single"):
        ker = ker[:1]
    B, Hh, Ww = dist.shape
    dev = dist.device
    rd = torch.nn.functional.normalize(ray_dir, dim=-1, eps=1e-12)
    P = eye.view(-1, 1, 1, 3) + rd * dist.unsqueeze(-1)        # r0m0:37
    yy, xx = torch.meshgrid(torch.arange(Hh, device=dev),
                            torch.arange(Ww, device=dev), indexing="ij")
    if zprepass_normals and nrm_world is not None:
        # r0m1:39-43 -- read it instead of reconstructing it
        N = torch.nn.functional.normalize(nrm_world, dim=-1, eps=1e-12)
    else:
        # r0m0:38-46 -- four neighbour taps, SHORTEST-EDGE selection.
        # The mix() picks whichever of the two one-sided differences is
        # shorter, which is what keeps a silhouette from tilting N.
        def _sh(t, dyy, dxx):
            return torch.roll(t, shifts=(-dyy, -dxx), dims=(1, 2))
        up = P - _sh(P, -1, 0)
        dn = _sh(P, 1, 0) - P
        rt_ = _sh(P, 0, 1) - P
        lf = P - _sh(P, 0, -1)
        ex = torch.where((lf.norm(dim=-1) < rt_.norm(dim=-1)).unsqueeze(-1),
                         lf, rt_)
        ey = torch.where((dn.norm(dim=-1) < up.norm(dim=-1)).unsqueeze(-1),
                         dn, up)
        N = -torch.nn.functional.normalize(torch.cross(ex, ey, dim=-1),
                                           dim=-1, eps=1e-12)
    # the noise vector -- see the docstring; a hash stands in for the
    # engine's tiling texture, whose content the bytecode does not carry.
    h = ((xx * 73856093) ^ (yy * 19349663)).float()
    nz = torch.stack((torch.sin(h * 0.017), torch.cos(h * 0.029),
                      torch.sin(h * 0.041 + 1.7)), dim=-1)
    nz = torch.nn.functional.normalize(nz, dim=-1, eps=1e-12).unsqueeze(0)
    R = args.cs2ao_radius
    origin = P + N * (dist * args.cs2ao_normal_bias).unsqueeze(-1)  # :47
    occ = torch.zeros_like(dist)
    bidx = torch.arange(B, device=dev).view(-1, 1, 1)
    for k in ker:
        kv = torch.tensor(k, device=dev).view(1, 1, 1, 3) * R
        # r0m0:60 -- reflect(kernel*r, noise)
        s = kv - 2.0 * (kv * nz).sum(-1, keepdim=True) * nz
        # r0m0:63-69 -- fold into the hemisphere about N
        flip = ((N * s).sum(-1, keepdim=True) < 0.0)
        s = torch.where(flip, s - 2.0 * (s * N).sum(-1, keepdim=True) * N, s)
        Pw = origin + s                                         # r0m0:72
        # r0m0:73-74 -- project, divide, to texel, clamped to the view
        d_w = Pw - eye.view(-1, 1, 1, 3)
        rl = d_w.norm(dim=-1)                                   # r0m0:77
        u, v, ok = _cs2_project_to_pixel(Pw, mvp)
        ui = u.clamp(0, Ww - 1).long()
        vi = v.clamp(0, Hh - 1).long()
        stored = dist[bidx, vi, ui]                             # r0m0:74
        # r0m0:78 -- rebuild the stored surface along ITS ray
        Q = eye.view(-1, 1, 1, 3) + (d_w / rl.clamp(min=1e-6).unsqueeze(-1)) \
            * stored.unsqueeze(-1)
        dv = Q - origin
        # r0m0:79 -- two step()s written as clamp(x*1e6, 0, 1)
        inr = ((R * R - (dv * dv).sum(-1)) * 1000000.0).clamp(0.0, 1.0)
        nearer = ((rl - stored) * 1000000.0).clamp(0.0, 1.0)
        occ = occ + torch.where(ok, inr * nearer, torch.zeros_like(inr))
    a = 1.0 - occ * (1.0 / len(ker))                            # r0m0:85
    a = a * a * a                                               # r0m0:86
    return torch.where(fg, a.clamp(0.0, 1.0), torch.ones_like(a))


def _cs2_project_to_pixel(Pw, mvp):
    """World point -> (px, py, inside) for this file's camera.

    Stands for r0m0:73-74's `(vec4(P,1) * viewProj)` followed by the
    perspective divide and the viewport scale/bias at :32-33,
    (0.5,-0.5)*vp.zw and 0.5*vp.zw + vp.xy, with the clamp to
    [vp.xy, vp.zw] the reference applies before the fetch. mvp is this
    file's per-frame (B,4,4) in the same row-vector convention the rest
    of the renderer uses.
    """
    ones = torch.ones_like(Pw[..., :1])
    clip = torch.einsum("bij,bhwj->bhwi", mvp,
                        torch.cat((Pw, ones), dim=-1))
    w = clip[..., 3:4]
    ndc = clip[..., :3] / w.clamp(min=1e-6)
    px = (ndc[..., 0] * 0.5 + 0.5) * W
    py = (0.5 - ndc[..., 1] * 0.5) * H
    ok = (clip[..., 3] > 1e-6) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px, py, ok


def aoproxy_splat(zfc, ray_dir, eye, fwd, proxies, reverse_depth=None):
    """aoproxy_splat r0m0 -- the PRODUCER of the offset-76 target.

    PER_VIEW_RENDER_TARGETS.md records the offset-76 producer as
    unknown ('not among the six extracted shaders'), and dirocc_target()
    in this file repeats that. It is this shader. The identification is
    structural, not by name: r0m0:74 accumulates FOUR channels, one per
    entry of _4083._m0._m0[0..3], a vec4[4] of directions -- which is
    the same four ambient-basis directions the offset-76 consumer
    resolves against at csgo_environment_blend:1239.

    Transcribed: the depth-to-world reconstruction (:49), the unit-box
    proxy falloff (:51-52) and its discard (:53-56), the sphere-proxy
    loop (:64-79) and the box-proxy loop (:86-102) -- two loops whose
    bounds are a per-instance uvec3 (begin, sphereEnd, boxEnd), NOT a
    bitmask, which is why this is a filter/other and not a binner --
    and the composite (:103).

    WHAT DIFFERS, precisely: the per-channel occlusion table at
    set1/binding30 is a sampler3D whose CONTENT is engine data and is
    not in the bytecode. r0m0:74 shows exactly how it is addressed --
    (dot(basis_j, dir)*0.5+0.5, invDist, slice_j) -- so the axes are
    known and only the tabulated values are not. This evaluates that
    lookup as the cone-occlusion function it tabulates:
    1 - solidAngleFraction * saturate(cosAngle), which is the same
    shape, is exact at the two ends, and is stated here rather than
    presented as the table. Nothing else is substituted.
    """
    if reverse_depth is None:
        reverse_depth = args.aoproxy_reverse_depth
    B, Hh, Ww = zfc.shape
    dev = zfc.device
    minD, maxD = 0.0, 1.0
    dn = ((zfc - minD) / (maxD - minD)).clamp(0.0, 1.0)
    n, f = PROJ_NEAR, PROJ_FAR
    p22 = -(f + n) / (f - n)
    p23 = -2.0 * f * n / (f - n)
    axis = torch.nn.functional.normalize(ray_dir, dim=-1, eps=1e-12)
    # r0m0:49 -- both S_REVERSE_DEPTH_BUFFER records, verbatim shapes.
    # dot(_m4, ray) projects the ray onto the view axis so the stored
    # depth (which is along the axis) can scale a ray-direction step.
    # _m4 is the view axis; this file already carries it as `fwd`.
    fw = torch.nn.functional.normalize(fwd, dim=-1,
                                       eps=1e-12).view(-1, 1, 1, 3)
    cosr = (fw * axis).sum(-1).clamp(min=1e-6)
    if reverse_depth:
        t = (p23 / 2.0) / (dn * cosr).clamp(min=1e-6)          # r1m0:49
    else:
        t = 1.0 / ((dn * 1.0 + (p22 - 1.0) / 2.0) * cosr).clamp(max=-1e-9)
    Pw = eye.view(-1, 1, 1, 3) + axis * t.unsqueeze(-1)
    out = torch.ones(B, Hh, Ww, 4, device=dev)
    sph = proxies.get("spheres", [])
    box = proxies.get("boxes", [])
    if _cs2filt_inject("aoproxy-no-boxes"):
        box = []
    basis = proxies.get("basis")
    Bas = (torch.tensor(basis, device=dev, dtype=torch.float32)
           if basis is not None else DIROCC_B)
    acc = torch.ones(B, Hh, Ww, 4, device=dev)
    for pr in sph:                                             # r0m0:64-79
        c = torch.tensor(pr["center"], device=dev).view(1, 1, 1, 3)
        rad = float(pr.get("radius", 1.0))
        d = c - Pw
        L = d.norm(dim=-1).clamp(min=1e-6)
        dirv = d / L.unsqueeze(-1)
        inv = (1.0 / L)                                        # r0m0:72
        # r0m0:74 -- one LUT read per basis direction; see docstring.
        sa = (rad * rad) * inv * inv
        for j in range(4):
            cosa = (dirv * Bas[j].view(1, 1, 1, 3)).sum(-1)
            acc[..., j] = acc[..., j] * (
                1.0 - sa.clamp(0.0, 1.0) * cosa.clamp(0.0, 1.0))
    for pr in box:                                             # r0m0:86-102
        c = torch.tensor(pr["center"], device=dev).view(1, 1, 1, 3)
        ext = torch.tensor(pr["ext"], device=dev).view(1, 1, 1, 3)
        rel = Pw - c
        # r0m0:93 -- clamp to 0.9 of the half-extent, then subtract
        d = (rel.clamp(-ext, ext) * 0.89999997615814208984375) - rel
        ad = d.abs()
        s = ad.sum(-1, keepdim=True).clamp(min=1e-6)
        # r0m0:95 -- the analytic box term, transcribed
        k = (math.pi / (ext.roll(-1, dims=-1) * ext.roll(-2, dims=-1))
             .clamp(min=1e-6))
        invd = 1.0 / (1.0 + ((ad / s) * k).sum(-1) * d.norm(dim=-1))
        dirv = torch.nn.functional.normalize(d, dim=-1, eps=1e-12)
        for j in range(4):
            cosa = (dirv * Bas[j].view(1, 1, 1, 3)).sum(-1)
            acc[..., j] = acc[..., j] * (
                1.0 - invd.clamp(0.0, 1.0) * cosa.clamp(0.0, 1.0))
    return acc.clamp(0.0, 1.0)


def _oct_decode(e):
    """atrous_filter r0m0:31-39 -- octahedral normal decode, verbatim.

        f  = e*2 - 1
        z  = 1 - |f.x| - |f.y|
        t  = clamp(-z, 0, 1)
        f += (f >= 0) ? -t : +t
        n  = normalize(vec3(f, z))
    """
    ff = e * 2.0 - 1.0
    zz = 1.0 - ff[..., 0].abs() - ff[..., 1].abs()
    t = (-zz).clamp(0.0, 1.0)
    adj = torch.where(ff >= 0.0, -t.unsqueeze(-1), t.unsqueeze(-1))
    ff = ff + adj
    return torch.nn.functional.normalize(
        torch.cat((ff, zz.unsqueeze(-1)), dim=-1), dim=-1, eps=1e-12)


def atrous_filter(cv, gb, step_size):
    """atrous_filter r0m0, both loop nests. cv = (B,H,W,4) colour+variance,
    gb = (B,H,W,4) packed (depth, depthGrad, octNormal.xy).

    r0m0:45-72  3x3 variance prefilter, weights [[.25,.125],[.125,.0625]]
    r0m0:73     sigma_l * sqrt(max(0, var + 1e-6))
    r0m0:74     depth denominator = sigma_z*clamp(grad,0.01,1000)*step
    r0m0:83-165 the 5x5 a-trous nest at stride `step_size`, skipping the
                centre and anything out of bounds
    r0m0:145    w = exp(-dz - dl) * pow(saturate(dot(n_c,n_t)), sigma_n)
                    * h[|dy|] * h[|dx|]
    r0m0:147    colour accumulates with w, VARIANCE with w*w
    r0m0:166    out = (rgb/W, var/W^2)
    """
    if _cs2filt_inject("atrous-no-stride"):
        step_size = 1
    B, Hh, Ww, _ = cv.shape
    dev = cv.device
    lum_c = (cv[..., :3] * torch.tensor(CS2AO_LUMA, device=dev)).sum(-1)
    z_c = gb[..., 0]
    n_c = _oct_decode(gb[..., 2:4])
    # r0m0:45-72 -- the 3x3 variance prefilter
    var = torch.zeros_like(z_c)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = CS2AO_ATROUS_VAR[abs(dy)][abs(dx)]
            var = var + torch.roll(cv[..., 3], shifts=(-dy, -dx),
                                   dims=(1, 2)) * w
    sig_l = args.atrous_sigma_l * torch.sqrt(
        (var + 9.9999999747524270787835121154785e-07).clamp(min=0.0))  # :73
    den_z = (args.atrous_sigma_z
             * gb[..., 1].clamp(0.00999999977648258209228515625, 1000.0)
             * float(step_size))                                       # :74
    acc = cv.clone()
    wsum = torch.ones_like(z_c)                                        # :77
    yy, xx = torch.meshgrid(torch.arange(Hh, device=dev),
                            torch.arange(Ww, device=dev), indexing="ij")
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue                                               # :108-121
            oy, ox = dy * step_size, dx * step_size
            iy, ix = yy + oy, xx + ox
            inb = (iy >= 0) & (iy < Hh) & (ix >= 0) & (ix < Ww)        # :102-105
            ct = torch.roll(cv, shifts=(-oy, -ox), dims=(1, 2))
            gt = torch.roll(gb, shifts=(-oy, -ox), dims=(1, 2))
            dist = math.hypot(float(dx), float(dy))                    # :135
            dd = den_z * dist
            wz = torch.where(dd == 0.0, torch.zeros_like(dd),
                             (z_c - gt[..., 0]).abs() / dd.clamp(min=1e-12))
            lum_t = (ct[..., :3] * torch.tensor(CS2AO_LUMA, device=dev)).sum(-1)
            wl = (lum_c - lum_t).abs() / sig_l.clamp(min=1e-12)
            nt = _oct_decode(gt[..., 2:4])
            wn = ((n_c * nt).sum(-1).clamp(0.0, 1.0)) ** args.atrous_sigma_n
            w = torch.exp(-wz.clamp(min=0.0) - wl.clamp(min=0.0)) * wn * \
                (CS2AO_ATROUS_H[abs(dy)] * CS2AO_ATROUS_H[abs(dx)])    # :145
            w = torch.where(inb, w, torch.zeros_like(w))
            acc = acc + torch.cat(
                (ct[..., :3] * w.unsqueeze(-1),
                 (ct[..., 3] * w * w).unsqueeze(-1)), dim=-1)          # :147
            wsum = wsum + w
    ws = wsum.clamp(min=1e-12)
    return torch.cat((acc[..., :3] / ws.unsqueeze(-1),
                      (acc[..., 3] / (ws * ws)).unsqueeze(-1)), dim=-1)  # :166


def denoise_blur(sig, zfc, kernel=None, blur_y=None):
    """denoise_blur r0m0 / r1m0 / r2m0. sig = (B,H,W,>=3), zfc = raw depth.

    S_BOX_KERNEL 0 is a PURE PASSTHROUGH -- that record has no loop
    (r0m0:10 is texelFetch and write) -- so selecting it is a real
    reference behaviour and not a disabled path.

    r1m0:41  linear depth = _m2 / rawDepth
    r1m0:62  w = 1 - clamp(|z_t - z_c| / sigma, 0, 1)
    r1m0:68-70  THREE DIFFERENT accumulations, which is the whole point
                of this shader and is easy to miss:
                  .x accumulates x*step(x<1) and divides by the COUNT of
                     those samples -- x==1 means 'no data', not 'lit'
                  .y accumulates x RAW and divides by the WEIGHT sum
                  .z accumulates z*w and divides by the weight sum
    r1m0:80-107 each falls back to 1.0 when its divisor is 0.
    """
    if kernel is None:
        kernel = args.dnb_kernel
    if blur_y is None:
        blur_y = args.dnb_blur_y
    taps = CS2AO_DNB_TAPS[int(kernel)]
    if _cs2filt_inject("dnb-kernel-flat"):
        taps = (0,) if taps else taps
    if not taps:                                                 # r0m0:10
        return sig[..., :3].clone()
    z = 1.0 / zfc.clamp(min=1e-6)                                # r1m0:41
    wsum = torch.zeros_like(z)
    csum = torch.zeros_like(z)
    acc = torch.zeros(sig.shape[0], sig.shape[1], sig.shape[2], 3,
                      device=sig.device)
    for t in taps:
        oy, ox = (t, 0) if blur_y else (0, t)
        z_t = torch.roll(z, shifts=(-oy, -ox), dims=(1, 2))
        s_t = torch.roll(sig[..., :3], shifts=(-oy, -ox), dims=(1, 2))
        w = 1.0 - ((z_t - z).abs() / args.dnb_depth_sigma).clamp(0.0, 1.0)
        m = (s_t[..., 0] < 1.0).float()                          # r1m0:68
        acc = acc + torch.stack((s_t[..., 0] * m, s_t[..., 1],
                                 s_t[..., 2] * w), dim=-1)
        wsum = wsum + w
        csum = csum + m
    x = torch.where(csum > 0.0, acc[..., 0] / csum.clamp(min=1e-12),
                    torch.ones_like(csum))
    pos = wsum > 0.0
    y = torch.where(pos, acc[..., 1] / wsum.clamp(min=1e-12),
                    torch.ones_like(wsum))
    zz = torch.where(pos, acc[..., 2] / wsum.clamp(min=1e-12),
                     torch.ones_like(wsum))
    return torch.stack((x, y, zz), dim=-1)


def blur_with_depth(col, zfc, kernel=None, blur_y=None, srgb_read=None):
    """blur_with_depth r1m0 (and the four other kernel records).

    The key is NOT depth. r1m0:53-54 unprojects (ndc.xy, linearised
    depth) by the inverse view-projection at _5618._m5 and evaluates a
    PLANE:  d = clamp(m3.x * dot(plane, vec4(P,1)) + m3.y, 0, 1) * m3.z.
    r1m0:90 then gates each Gaussian tap on
        step(|d_t - d_c| <= max(m3.w, smoothstep(m2.z,m2.w,min(d_t,d_c))^2)
                            + step(m2.z, alpha_t + alpha_c))
    and r1m0:57-58 seeds the accumulator at 0.01, not at the centre
    weight -- so the centre tap is counted once inside the loop.
    """
    if kernel is None:
        kernel = args.bwd_kernel
    if blur_y is None:
        blur_y = args.bwd_blur_y
    if srgb_read is None:
        srgb_read = args.bwd_srgb_read
    offs, wts = CS2AO_BWD_KERNELS[int(kernel)]
    if _cs2filt_inject("bwd-kernel-flat"):
        wts = tuple(1.0 for _ in wts)
    c = col
    if srgb_read:
        c = torch.where(c <= 0.04045, c / 12.92,
                        ((c.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)
    pl = [float(v) for v in args.bwd_plane.split(",")]
    if len(pl) != 4:
        raise SystemExit("--bwd-plane needs exactly four comma-separated "
                         "floats")
    plane = torch.tensor(pl, device=col.device)
    zlin = _lin_depth(zfc)
    key_c = _cs2_bwd_key(zlin, plane)
    acc = c * 0.00999999977648258209228515625
    wsum = torch.full_like(key_c, 0.00999999977648258209228515625)
    for o, wt in zip(offs, wts):
        step_px = int(round(o))
        oy, ox = (step_px, 0) if blur_y else (0, step_px)
        c_t = torch.roll(c, shifts=(-oy, -ox), dims=(1, 2))
        z_t = torch.roll(zlin, shifts=(-oy, -ox), dims=(1, 2))
        key_t = _cs2_bwd_key(z_t, plane)
        thr = torch.full_like(key_c, 0.05)
        w = wt * (( key_t - key_c).abs() <= thr).float()          # r1m0:90
        acc = acc + c_t * w.unsqueeze(-1)
        wsum = wsum + w
    out = acc / wsum.clamp(min=1e-12).unsqueeze(-1)
    if args.bwd_disable_alpha_write and out.shape[-1] == 4:
        out = torch.cat((out[..., :3], col[..., 3:]), dim=-1)
    return out


def _cs2_bwd_key(zlin, plane):
    """blur_with_depth r1m0:54 -- the plane key, without the unprojection.

    The reference reaches the world point through the inverse
    view-projection; here the same plane is evaluated against the point
    this renderer already has along the view axis. Stated because it is
    the one place in this family where the arithmetic is reorganised
    rather than copied: same plane, same clamp, same scale and bias.
    """
    d = plane[1] * zlin + plane[3]
    return (args.bwd_plane_scale * d + args.bwd_plane_bias).clamp(0.0, 1.0)


def cs2ao_run(zfc, wpos, nrm, fg, eye, fwd, ray_dir, mvp):
    """THE DISPATCH. Runs the whole transcribed chain and returns (B,H,W).

    Order is the engine's own, read off the input/output bindings:
      ssao_convert_depth -> ssao_downsample_depth (chain)
        -> ssao_scalable_ambient_obscurance
        -> ssao_bilateral_blur (x, then y)
    and, independently, ssao.vfx against the ray-distance target.
    --cs2ao-estimator selects which of the two is bound here; this is the
    `tap = f if q else g` binding, not a definition sitting unused.
    """
    est = args.cs2ao_estimator
    A = None
    if est in ("sao", "both"):
        csz = ssao_convert_depth(zfc)
        mips = [csz]
        for _ in range(CS2AO_SAO_MAX_MIP):
            if mips[-1].shape[1] < 2 or mips[-1].shape[2] < 2:
                break
            mips.append(ssao_downsample_depth(mips[-1]))
        nrm_cam = None
        if args.cs2ao_read_normal_tex:
            nrm_cam = _cs2_world_to_cam_dir(nrm, fwd)
        rt = ssao_scalable_ambient_obscurance(mips, nrm_cam)
        if args.cs2ao_bilateral:
            rt = ssao_bilateral_blur(rt, (1, 0))
            rt = ssao_bilateral_blur(rt, (0, 1))
        A = rt[..., 0]
    if est in ("ssao", "both"):
        dist = (wpos - eye.view(-1, 1, 1, 3)).norm(dim=-1)
        a2 = ssao_ray_distance(dist, ray_dir, eye, fg, mvp, nrm_world=nrm)
        A = a2 if A is None else A * a2
    if A is None:
        return None
    return torch.where(fg, A.clamp(0.0, 1.0), torch.ones_like(A))


def _cs2_world_to_cam_dir(n, fwd):
    """World direction -> the camera frame the SAO normal path expects."""
    f = torch.nn.functional.normalize(fwd, dim=-1, eps=1e-12)
    up = torch.tensor([0.0, 1.0, 0.0], device=n.device).expand_as(f)
    r = torch.nn.functional.normalize(torch.cross(f, up, dim=-1),
                                      dim=-1, eps=1e-12)
    u = torch.cross(r, f, dim=-1)
    return torch.stack(((n * r.view(-1, 1, 1, 3)).sum(-1),
                        (n * u.view(-1, 1, 1, 3)).sum(-1),
                        (n * f.view(-1, 1, 1, 3)).sum(-1)), dim=-1)
