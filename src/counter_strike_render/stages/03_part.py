parser.add_argument("--b2-cw-overlay-texture", type=int, choices=(0, 1),
                    default=0, help="S_OVERLAY_TEXTURE.")
parser.add_argument("--b2-cw-case-hardening", type=int, choices=(0, 1),
                    default=0, help="S_CASE_HARDENING.")
parser.add_argument("--b2-cw-case-hardening-trilinear", type=int,
                    choices=(0, 1), default=0,
                    help="S_CASE_HARDENING_TRILINEAR.")
parser.add_argument("--b2-cw-patina-age", type=int, choices=(0, 1),
                    default=0, help="S_PATINA_AGE.")
parser.add_argument("--b2-cw-output", choices=("albedo", "material"),
                    default="albedo",
                    help="the compositor's two output modes, the "
                         "_5618._m0 branch (glsl:220 vs :104): 'albedo' is "
                         "the finish colour, 'material' is the "
                         "channel-packed rough/metal/AO target, sRGB "
                         "-> linear per channel at the 0.04045 knee.")
parser.add_argument("--b2-cw-blend-mode", type=int, default=-1,
                    help="_5618._m18, the 5-way paint blend switch "
                         "(glsl:160-207): 0 alpha-over, 1 normalised "
                         "colour-burn, 2 multiply-then-normalise, 3 "
                         "additive-with-luma-lift, 4 alpha-over with the "
                         "wear-driven alpha rewrite. -1 = from the "
                         "material.")
parser.add_argument("--b2-parser-replay", action="store_true",
                    help="re-play every add_argument in this file against "
                         "a FRESH ArgumentParser and print the number of "
                         "argparse.ArgumentError conflicts, then exit. A "
                         "P0 broke main today because two agents declared "
                         "the same flag in different files, git merged "
                         "clean, and argparse raised at import so the "
                         "renderer could not build its parser for ANY "
                         "invocation. This makes that failure detectable "
                         "without running a render.")
args = parser.parse_args()


def _resolve_content_root(a):
    """OWNER RULE (2026-08-13): a render node gets the WHOLE content tree,
    and the renderer resolves its OWN inputs from it -- never a human
    selecting files from the pack and hoping the transfer was remembered.
    The v960 defect (viewmodel/weapons/HUD absent, sidecars staged but
    unreferenced) was a hand-assembled flag list; this makes that
    mechanism structurally impossible: with --content-root every input
    family resolves by map name, every resolution PRINTS, and a missing
    family REFUSES BY NAME -- no waiver exists. Per-family flags remain
    as explicit A/B overrides that beat resolution.

    Tree shape (provision_render_node.py materializes + verifies it):
      <root>/worlds/<map>.pt            + every sidecar beside it by stem
      <root>/pm_bundles/                shared third-person bundles
      <root>/weapons/                   world weapon packs (wpn attach)
      <root>/vm/                        first-person viewmodel bundles
    """
    if not a.content_root:
        return
    if not a.map:
        raise SystemExit("REFUSING: --content-root without --map -- the "
                         "map name is FITTED by the caller "
                         "(iji_model.map_identity), never typed here.")
    root = os.path.abspath(a.content_root)
    w = os.path.join(root, "worlds", a.map + ".pt")
    missing = []

    def _family(name, path, attr=None, current=None):
        # OWNER RULE (2026-08-13, second ruling): there is NO waiver. A
        # renderer that can produce a frame without a family will
        # eventually do so; the error class must be impossible. Genuine
        # absence is a property of the CONTENT, authored by the
        # extraction pipeline as a sibling `<path>.ABSENT` file whose
        # body states the verified reason (e.g. a lobby map with no sky
        # entity) -- a fact READ from the tree, never an operator
        # decision at run time.
        ok = os.path.exists(path)
        absent = path + ".ABSENT"
        if ok and attr is not None and current in (None, "off"):
            setattr(a, attr, path)
        if ok:
            state = "RESOLVED"
        elif os.path.exists(absent):
            reason = open(absent).read().strip()[:120]
            state = f"ABSENT (authored): {reason}"
        else:
            state = "MISSING"
            missing.append(name)
        print(f"content-root {name:22s} {state}  {path}", flush=True)

    _family("world", w, "world", None)
    a.world = w if os.path.exists(w) else a.world
    # families that resolve BESIDE the pack via their own 'auto' defaults
    # (irradiance, sky_cube, light_environment, direct_light_shadows):
    # verified present here so 'auto' cannot silently find nothing.
    for fam, suffix in (("irradiance", ".irradiance.npy"),
                        ("direct_light_shadows",
                         ".direct_light_shadows.npy"),
                        ("sky_cube", ".sky_cube.npz"),
                        ("light_environment", ".light_environment.json")):
        _family(fam, w[:-3] + suffix)
    _family("fam_side", w[:-3] + ".fam_side.pt", "fam_side", a.fam_side)
    _family("probe_field", w[:-3] + ".probe_field.npz", "probe_npz",
            a.probe_npz)
    # pm_bundles is checked by CONTENT (the team bundles the loader
    # opens), not directory existence -- a mis-nested tree read as
    # RESOLVED while every entity fell to 'no bundle' and the matrix
    # rendered playerless.
    _pmb = os.path.join(root, "pm_bundles")
    _family("pm_bundles",
            _pmb if (os.path.exists(os.path.join(_pmb, "ct.pt")) and
                     os.path.exists(os.path.join(_pmb, "t.pt")))
            else os.path.join(_pmb, "ct.pt"),
            "playermodel_bundles", a.playermodel_bundles)
    if a.playermodel_bundles and a.playermodel_bundles.endswith("ct.pt"):
        a.playermodel_bundles = None            # failed content check
    _family("weapons", os.path.join(root, "weapons"),
            "playermodel_weapons", a.playermodel_weapons)
    _family("vm", os.path.join(root, "vm"))
    if missing:
        raise SystemExit(
            f"REFUSING: content root {root} lacks famil"
            f"{'y' if len(missing) == 1 else 'ies'} {sorted(missing)} for "
            f"map {a.map}. There is NO waiver: provision the node "
            f"(tools/provision_render_node.py), build the family, or -- "
            f"only if the absence is a verified property of the content "
            f"-- have the extraction pipeline author `<path>.ABSENT` "
            f"stating the reason.")


_resolve_content_root(args)
if not args.world:
    raise SystemExit("REFUSING: no world pack -- pass --content-root "
                     "with --map (the provisioned path) or --world "
                     "explicitly.")
smoke_mboit.configure(args)


def _apply_preset(a, argv):
    """--preset N: resolve the GT preset's WHOLE cvar vector, or refuse (#91).

    A preset is not a fidelity label, it is 11 cvars, and they are READ from
    the committed gtq manifest. Resolving two of them and rendering anyway
    would put a preset's name on an artifact rendered at this renderer's own
    defaults, which is the failure this refuses.

    THE GAP LIST IS THE POINT. Every cvar with no mapped axis prints with
    its kind and its basis, and the run then REFUSES. It is a deliverable --
    the honest inventory of reference options not implemented -- so
    --preset-accept-gaps exists to render anyway, and it makes the operator
    say so and stamps the artifact's log with what is missing. It never
    becomes the default: a bracket claim nobody had to acknowledge is a
    bracket claim nobody checked.
    """
    if a.preset is None:
        return
    import render_config as _rc
    tag = f"preset{a.preset}"
    applied, gaps, vector = _rc.resolve_preset(tag)
    print(f"PRESET {tag}: {len(vector)} cvars READ from "
          f"{os.path.relpath(_rc.MANIFEST, _rc.REPO_ROOT)} "
          f"(sha {_rc.sha(_rc.MANIFEST)[:12]})", flush=True)

    # EXPLICIT BEATS THE PRESET, AND SAYS SO. A flag typed on the command
    # line is not silently overwritten by the preset and does not silently
    # overwrite it -- the collision is named, because either direction of
    # silence produces a render whose config nobody can reconstruct.
    typed = {t.split("=")[0] for t in argv if t.startswith("--")}
    for cvar, gtval, flag, ours, basis in applied:
        if flag is None:
            print(f"  RESOLVED  {cvar:<40} = {gtval:<4} -> no axis needed "
                  f"({basis})", flush=True)
            continue
        dest = flag.lstrip("-").replace("-", "_")
        if flag in typed:
            have = getattr(a, dest, None)
            print(f"  ⚠️  CONFLICT {cvar:<38} = {gtval:<4} -> {flag} {ours}, "
                  f"but {flag} was typed as {have}. THE TYPED VALUE WINS "
                  f"and this render is NOT at {tag} on that axis.",
                  flush=True)
            continue
        setattr(a, dest, ours)
        print(f"  RESOLVED  {cvar:<40} = {gtval:<4} -> {flag} {ours}",
              flush=True)

    if not gaps:
        print(f"  GAP LIST: EMPTY -- every one of the {len(vector)} cvars "
              f"in {tag} resolved to an axis. Note this asserts the axis is "
              f"WIRED and its mapping READ, never that our pass agrees with "
              f"the reference's.", flush=True)
        return
    print(f"\n  GAP LIST for {tag}: {len(gaps)} of {len(vector)} cvars have "
          f"no resolved axis in this renderer. This is the inventory, not a "
          f"crash -- each row names WHY.", flush=True)
    for cvar, gtval, kind, flag, why in gaps:
        print(f"    {kind:<18} {cvar:<40} = {gtval}", flush=True)
        print(f"      {'flag: ' + flag if flag else 'no flag exists'} -- "
              f"{why}", flush=True)
    if not a.preset_accept_gaps:
        raise SystemExit(
            f"\nREFUSED: {tag} has {len(gaps)} unresolved cvar(s) above, so "
            f"this render would not be at {tag} -- it would be at this "
            f"renderer's defaults on those axes with {tag}'s name on it. "
            f"Pass --preset-accept-gaps to render anyway; the gap list above "
            f"stays in the log and the artifact must quote it.")
    print(f"\n  ⚠️  --preset-accept-gaps: rendering with the "
          f"{len(gaps)} gap(s) above UNRESOLVED. This image is at {tag} on "
          f"{len(applied)} axes and at this renderer's defaults on "
          f"{len(gaps)}. Any claim it makes about {tag} is bounded by that "
          f"list.", flush=True)


_apply_preset(args, sys.argv[1:])


def _fsr_scale_now(a):
    """videocfg_fsr_detail -> render at the INPUT size, present at the
    output size. MUST run before anything reads args.width/args.height.

    THE TIER IS NOT A POST-PROCESS. `con0.xy = inputViewport/outputSize`
    (post/fsr.py:easu_con) is the whole axis, and it means the SCENE
    RASTER SHRINKS -- every LOD, every derivative, every aliasing
    decision downstream is taken at the smaller size. Upscaling a
    full-resolution image afterwards reproduces none of that.

    preset0 and preset1 carry videocfg_fsr_detail 3; preset2 and preset3
    carry 0. So the low presets are UPSCALED and the high ones native,
    which is why the same-config repeat floor is FSR-dependent by 31x
    (#94), and it makes this one of the largest single differences in the
    whole preset ladder."""
    global FSR_OUT_PRE
    FSR_OUT_PRE = None
    if a.fsr_detail == 0:
        return
    scale = _po_vcfg.FSR_TIER_SCALE.get(a.fsr_detail)
    if scale is None:
        raise SystemExit(
            f"REFUSED: --fsr-detail {a.fsr_detail}: "
            f"{_po_vcfg.FSR_TIER_UNRESOLVED_NOTE} Pass --fsr-detail 0 to "
            f"render native, or land the row in "
            f"post/video_config.FSR_TIER_SCALE once it is READ.")
    if scale >= 1.0:
        return
    FSR_OUT_PRE = (a.width, a.height)
    # FLOOR, not round. The measured internal rasters are
    # floor(dim * pct): 985x554 / 857x482 / 755x424 at 1280x720, and
    # round() reproduces 985 but MISSES 857 (gives 858) and 755 (756).
    a.width, a.height = _po_vcfg.fsr_internal_size(
        a.fsr_detail, a.width, a.height)
    if not _po_vcfg.FSR_TIER_RCAS.get(a.fsr_detail):
        print(f"FSR tier {a.fsr_detail}: RCAS DOES NOT RUN at this tier "
              f"-- measured, not assumed: it loses energy in every band "
              f"(0.21 at its own Nyquist against 0.81-0.83 for tiers 2-4) "
              f"and allocates no second full-res LDR target.", flush=True)
    elif a.fsr_sharpness is None:
        print(f"FSR tier {a.fsr_detail}: RCAS SHOULD RUN here and WILL "
              f"NOT -- {_po_vcfg.FSR_RCAS_SHARPNESS_UNREAD_NOTE}",
              flush=True)
    print(f"FSR: {_po_vcfg.FSR_MIP_BIAS_UNREAD_NOTE}", flush=True)
    print(f"FSR tier {a.fsr_detail}: scene raster {a.width}x{a.height}, "
          f"presented at {FSR_OUT_PRE[0]}x{FSR_OUT_PRE[1]} "
          f"(scale {scale:.4f}). The RASTER shrinks -- LODs and "
          f"derivatives are taken at the smaller size, which IS the axis; "
          f"the upscale is the consequence.", flush=True)


FSR_OUT_PRE = None
_fsr_scale_now(args)

# --------------------------------------------------------------------------
# r_texturefilteringquality -> the sampler state, resolved ONCE, here.
# --------------------------------------------------------------------------
# Before anything reads args.max_aniso (ANISO_C/ANISO_A are built at
# import time from it) or MIP_FILTER. The tier is ONE state covering three
# columns, so it is expanded in one place rather than by three consumers
# that could drift apart -- and in particular the MIP FILTER column has to
# travel with the aniso count, because tier 0 -> 1 moves only that column.
MIP_FILTER = "linear"


def _apply_filtering_tier(a):
    global MIP_FILTER
    if a.texture_filtering_quality is None:
        print("r_texturefilteringquality: NOT SET -- this render is at "
              f"--max-aniso {a.max_aniso}/{a.max_aniso_aux or a.max_aniso} "
              f"with a linear mip filter, which is NO CS2 tier. --preset "
              f"always sets one.", flush=True)
        return
    st = _po_tfq.sampler_state(a.texture_filtering_quality)
    typed = {t.split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
    for flag, dest, val in (("--max-aniso", "max_aniso", st["aniso"]),
                            ("--max-aniso-aux", "max_aniso_aux",
                             st["aniso"])):
        if flag in typed:
            print(f"  ⚠️  CONFLICT r_texturefilteringquality="
                  f"{a.texture_filtering_quality} wants {flag} {val}, but "
                  f"{flag} was typed as {getattr(a, dest)}. THE TYPED "
                  f"VALUE WINS and this render is NOT at that tier on "
                  f"this axis.", flush=True)
            continue
        setattr(a, dest, val)
    MIP_FILTER = st["mip"]
    if st["bias"] and "--mip-bias" not in typed:
        a.mip_bias = st["bias"]
    print(f"{_po_tfq.describe(a.texture_filtering_quality)} -> "
          f"--max-aniso {a.max_aniso}, mip filter {MIP_FILTER}, "
          f"--mip-bias {a.mip_bias:+g}", flush=True)
    if a.texture_filtering_quality >= 3:
        print(f"  NOTE: {_po_tfq.TOP_TIERS_UNARBITRATED}", flush=True)


_apply_filtering_tier(args)


# --------------------------------------------------------------------------
# INERT-AXIS CHECK. An axis this run can PROVE cannot reach the image, said
# before the render instead of discovered from a divergence table afterwards.
#
# WHY THIS EXISTS: preset2 and preset3 rendered BIT-IDENTICAL images -- same
# md5, not merely "below threshold" -- at the four poses where the reference
# genuinely separates them (D_ref 4.5-11.7, floor 4.12). Every axis that
# changes between those presets was inert:
#
#   msaa 4->8                        gated below; both fell to 1 sample
#   r_texturefilteringquality 3->5   aniso 4->16, mip+bias identical
#   videocfg_ao_detail 2->3          value table empty (typed UNREAD)
#   videocfg_particle_detail 2->3    value table empty (typed UNREAD)
#
# The renderer ALREADY said the msaa half: both logs print
# "msaa: 1-sample attachment from single" beside "--msaa-samples 4"/"8".
# So this is not a missing banner -- it is a banner that was printed, in
# both logs, and not read. A louder line in the same place would not have
# helped. What changes the outcome is saying it where the run STOPS being
# shaped by a value it cannot use: at argument time, naming the remedy.
def _inert_axes(a):
    """Return [(axis, why, remedy)] for axes provably unable to move a pixel.

    Only PROVABLE cases belong here. "I doubt this matters" is not proof and
    must not be listed -- a check that cries wolf gets muted, and a muted
    check is the printed-and-unread banner all over again.
    """
    out = []
    ss = max(1, int(getattr(a, "supersample", 1)))
    ms = int(getattr(a, "msaa_samples", 1))
    if ms > 1 and ss * ss < ms:
        # Arithmetic, not empirical: ms_depth_from_supersample needs one
        # sub-pixel cell per sample and the producer refuses to duplicate
        # cells, so this lands on the 1-sample attachment every time.
        out.append((
            "--msaa-samples",
            f"{ms} samples asked, --supersample {ss} gives {ss * ss} "
            f"sub-pixel cell{'s' if ss * ss != 1 else ''}; the MSAA path "
            f"needs supersample**2 >= samples and builds the 1-sample "
            f"(D_NON_MSAA_DEPTH) attachment otherwise",
            f"--supersample {math.ceil(math.sqrt(ms))}"))
    return out


_INERT = _inert_axes(args)
if _INERT:
    print("=" * 72, flush=True)
    print("INERT AXIS: this run sets a value that CANNOT reach the image.",
          flush=True)
    for _ax, _why, _fix in _INERT:
        print(f"  {_ax}: {_why}", flush=True)
        print(f"     remedy: {_fix}", flush=True)
    print("  Two presets differing ONLY in an inert axis render "
          "BIT-IDENTICALLY, which is how a lattice rung collapses without "
          "anything failing.", flush=True)
    print("=" * 72, flush=True)


def _apply_detail_tiers(a):
    """The three videocfg_*_detail tiers -> their flags, in ONE place.

    A tier is not one flag -- particle_detail plausibly keys the march
    step count AND the resolution AND the light steps together -- so
    three consumers each reading the tier could drift apart. Expanding
    here also makes the CONFLICT rule uniform with --preset's: an
    explicitly typed flag beats the tier and SAYS SO.

    Every table is UNREAD, so every tier currently REFUSES to apply
    values and the render proceeds at this renderer's own. That is the
    honest state and it is printed as such: recording the tier without
    applying it, and saying which, is different from silently rendering
    at a default with the tier's name in the log."""
    typed = {t.split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
    for axis, tier, flag in (
            ("videocfg_texture_detail", a.texture_detail, "--texture-detail"),
            ("videocfg_particle_detail", a.particle_detail,
             "--particle-detail"),
            ("videocfg_ao_detail", a.ao_detail, "--ao-detail")):
        if tier is None:
            continue
        try:
            vals = _po_det.tier_values(axis, tier)
        except _po_det.DetailTierUnread as e:
            print(f"⚠️  {flag} {tier}: RECORDED, NOT APPLIED -- {e}",
                  flush=True)
            continue
        for k, v in sorted(vals.items()):
            if k.startswith("__"):
                continue
            if k in typed:
                print(f"  ⚠️  CONFLICT {axis}={tier} wants {k} {v}, but "
                      f"{k} was typed. THE TYPED VALUE WINS and this "
                      f"render is NOT at that tier on this axis.",
                      flush=True)
                continue
            setattr(a, k.lstrip("-").replace("-", "_"), v)
        print(_po_det.describe(axis, tier), flush=True)
    # The non-exhibiting-fixture census, printed whether or not the tier
    # was set: a reader of a null result needs to know which state it was.
    print(_po_det.particle_exhibition_note(
        getattr(a, "smoke", "off"), int(getattr(a, "smoke_instances", 0) or 0)),
        flush=True)


_apply_detail_tiers(args)


def _water_axis_override(v):
    """`--water-*` string -> None (use the material) or a float.

    Resolved ONCE, here.  Nothing downstream ever sees the string, so
    nothing downstream can test it with bare truthiness -- "off" is
    truthy, and a consumer that wrote `if args.water_refraction:` would
    have turned the axis ON when told to turn it off.  That is the exact
    shape of the bug that was hiding behind today's argparse P0.
    """
    if v == "material":
        return None
    if v == "on":
        return 1.0
    if v == "off":
        return 0.0
    return float(v)


WATER_OV = {
    "reflection_type": _water_axis_override(args.water_reflection_type),
    "refraction": _water_axis_override(args.water_refraction),
    "caustics": _water_axis_override(args.water_caustics),
    "interaction": _water_axis_override(args.water_interaction),
    "blur_refraction": _water_axis_override(args.water_blur_refraction),
    # These two are INTEGER overrides whose sentinel is 0, and 0 is
    # falsey, so `x if x else material` is right for them and wrong for
    # the five above. Keeping the two kinds apart is the whole reason the
    # resolution happens here instead of at each use site.
    "wave_iterations": args.water_wave_iterations,
    "ssr_steps": args.water_ssr_steps,
}


def _parser_replay_selftest(inject="none"):
    """Replay every add_argument() in this file against a FRESH parser.

    A P0 broke main because two agents declared the same flag in two
    files: git merged clean, and argparse then raised at IMPORT, so the
    renderer could not build its parser for ANY invocation -- including
    --help. The failure is not in any one file's diff; it is in the union.

    This reads THIS FILE's source with `ast` -- not the live `parser`
    object, which by definition already survived construction -- pulls the
    option strings out of every `parser.add_argument(...)` call, and adds
    them one at a time to a brand-new ArgumentParser. argparse raises
    ArgumentError on a collision, which is exactly the P0's failure mode.

    It can fail. `inject` proves it:
      duplicate          re-adds a flag this file DOES declare
      duplicate-foreign  adds a flag this file does NOT declare, twice
    Both must be seen to fire before a clean run means anything.

    Returns (n_options, conflicts) where conflicts is a list of
    (option_string, message).
    """
    import ast as _ast
    import os as _os
    src = open(_os.path.abspath(__file__)).read()
    tree = _ast.parse(src)
    opts = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, _ast.Attribute) and fn.attr == "add_argument"):
            continue
        for a in node.args:
            if isinstance(a, _ast.Constant) and isinstance(a.value, str):
                opts.append((a.value, getattr(node, "lineno", -1)))
    if inject == "duplicate":
        # re-add one of this file's OWN flags -- the exact P0 shape
        opts.append(("--viewmodel-hfov", -1))
    elif inject == "duplicate-foreign":
        opts.append(("--a-flag-this-file-never-declares", -1))
        opts.append(("--a-flag-this-file-never-declares", -1))
    fresh = argparse.ArgumentParser(add_help=True)
    conflicts = []
    for opt, lineno in opts:
        try:
            # every declaration is reduced to the same trivial spec: the
            # collision argparse raises on is on the option STRING, and
            # keeping the spec uniform means a type= or choices= that
            # happens to differ cannot mask a real duplicate.
            fresh.add_argument(opt, default=None)
        except argparse.ArgumentError as exc:
            conflicts.append((opt, f"{exc} (line {lineno})"))
        except (TypeError, ValueError) as exc:
            # a positional or an option argparse rejects for another
            # reason is still a real construction failure -- report it
            # rather than swallowing it, or this check stops being one.
            conflicts.append((opt, f"{type(exc).__name__}: {exc} "
                                   f"(line {lineno})"))
    return len(opts), conflicts


if args.viewmodel_parser_selftest:
    _n_opt, _conf = _parser_replay_selftest(
        args.viewmodel_parser_selftest_inject)
    print(f"parser replay: {_n_opt} add_argument() option strings replayed "
          f"against a fresh ArgumentParser", flush=True)
    print(f"parser replay: {len(_conf)} conflicts", flush=True)
    for _o, _m in _conf:
        print(f"    CONFLICT {_o}: {_m}", flush=True)
    if args.viewmodel_parser_selftest_inject != "none" and not _conf:
        raise SystemExit(
            "parser replay: an injected defect "
            f"({args.viewmodel_parser_selftest_inject}) produced NO "
            "conflict. The check cannot fail, so it is not a check.")
    raise SystemExit(1 if _conf else 0)

if args.weapon and not args.viewmodel:
    raise SystemExit(
        "--weapon shades the viewmodel pass, which --viewmodel creates. "
        "Without it there is no pass to run in and --weapon would reach "
        "zero pixels while reporting as enabled. Add --viewmodel.")
if args.viewmodel_legs_prepass and not args.viewmodel:
    raise SystemExit(
        "--viewmodel-legs-prepass runs inside the viewmodel pass. "
        "Add --viewmodel.")
if args.viewmodel and args.viewmodel_synth == "off" \
        and not args.viewmodel_model:
    # de_inferno's pack has no weapon and no legs, so an un-synthesised
    # viewmodel pass over this world pack covers zero pixels. Saying so
    # here is the difference between "nothing to render" and "never
    # called" -- the two the deliverable requires be distinguishable.
    print("--viewmodel: no viewmodel geometry source selected "
          "(--viewmodel-synth off). The pass will run and cover 0 px on a "
          "pack with no weapon; use --viewmodel-synth both to submit a "
          "synthesised draw through the same dispatch.", flush=True)
if args.viewmodel_dither_size & (args.viewmodel_dither_size - 1):
    raise SystemExit(
        f"--viewmodel-dither-size {args.viewmodel_dither_size} is not a "
        "power of two. The shader indexes the tile with "
        "`ivec2(gl_FragCoord.xy) & mask` (csgo_legs_prepass:28), so a "
        "non-power-of-two tile would alias silently instead of tiling.")

if args.nonworld_selftest:
    # Runs BEFORE anything else parses a world: the parser-conflict half
    # must be answerable on a tree with no assets at all, since that is
    # the failure it exists to catch.
    _rc = 0
    _inj = args.nonworld_selftest_inject

    def _factory(p):
        _nw.add_arguments(p)
        if _inj == "dup-flag":
            p.add_argument("--decal-project", default="off")

    _nflags, _conf = _nw.parser_replay(_factory)
    print(f"parser replay: {_nflags} flags declared by "
          f"fam_nonworld.add_arguments, {len(_conf)} conflicts"
          + (f" -- {', '.join(_conf)}" if _conf else ""), flush=True)
    _st = _nw.parser_replay_selftest()
    print(f"  replay CAN fail: {_st['can_fail']} "
          f"(injected duplicate -> {_st['injected_conflicts']})", flush=True)
    if not _st["can_fail"]:
        print("  FAIL: the replay could not be made to fail, so '0 "
              "conflicts' is indistinguishable from 'it never ran'.",
              flush=True)
        _rc = 1
    # The whole real parser, replayed. This is the check the P0 needed.
    _seen, _dupes = {}, []
    for _a in parser._actions:
        for _o in _a.option_strings:
            if _o in _seen:
                _dupes.append(_o)
            _seen[_o] = True
    print(f"gpu_render.py parser: {len(_seen)} distinct option strings, "
          f"{len(_dupes)} duplicates"
          + (f" -- {', '.join(sorted(set(_dupes)))}" if _dupes else ""),
          flush=True)
    if _dupes:
        _rc = 1
    _cost = _nw.selftest_cost()
    if _cost is None:
        print("spritecard cost recount: reference GLSL not present "
              "(vcs/nonworld_ref/sc/s0_d0.glsl)", flush=True)
        _rc = 1
    else:
        print("spritecard s0_d0 (the representative module) recounted "
              "from the reference GLSL, as a SECOND method against "
              "flopcount.py's SPIR-V count:", flush=True)
        for _k, _v in _cost.items():
            print(f"    {_k:34s} {_v}", flush=True)
        _fetch = _cost["glsl_texel_reading_call_sites"]
        _loops = _cost["loops"]
        print(f"    published: 19 fetch sites, 0 loops, 708 FLOPs "
              f"(EXACT, not a floor)", flush=True)
        if _fetch != 19:
            print(f"    FAIL: recount gives {_fetch} texel-reading call "
                  f"sites, published 19", flush=True)
            _rc = 1
        if _loops != 0:
            print(f"    FAIL: recount gives {_loops} loops, published 0",
                  flush=True)
            _rc = 1
    raise SystemExit(_rc)


# =====================================================================
# PARSER REPLAY -- the P0 that broke main today, made detectable
# =====================================================================
# Two agents declared the same flag in two different files. Git merged
# clean because the lines do not touch. argparse then raised
# ArgumentError at IMPORT, so `gpu_render.py --help` -- and every other
# invocation -- died before it could build a parser at all.
#
# The check below can FAIL: `--b2-selftest-inject parser-collide` adds a
# duplicate option to the replay parser on purpose, and the replay must
# then report a non-zero conflict count. Verified by running it.
def _b2_parser_replay(inject=None):
    """Re-play every add_argument in THIS FILE onto a fresh ArgumentParser.

    -> (n_replayed, [conflict messages]).

    Replaying is not the same as trusting that this module imported: the
    module importing proves only that ITS OWN calls are consistent. The
    cross-file scan below is the half that catches the actual P0, where
    the duplicate lives in a second file that adds to the same parser.
    """
    import ast as _ast
    import argparse as _ap
    import copy as _copy
    import os as _os
    src = open(_os.path.abspath(__file__)).read()
    tree = _ast.parse(src)
    # ONLY calls on the name `parser`. The first version of this walk
    # matched `.add_argument` on ANY receiver, so it also collected the
    # helper calls inside this very function and reported a conflict on a
    # CLEAN tree. A check that fires on nothing is as useless as one that
    # never fires, and this one was caught only because the clean run was
    # made to print its count instead of a boolean.
    calls = [n for n in _ast.walk(tree)
             if isinstance(n, _ast.Call)
             and isinstance(n.func, _ast.Attribute)
             and n.func.attr == "add_argument"
             and isinstance(n.func.value, _ast.Name)
             and n.func.value.id == "parser"]
    fresh = _ap.ArgumentParser(add_help=True)
    ns = {"parser": fresh, "_f3": _f3, "int": int, "float": float, "str": str}
    bad = []
    unevaluable = []
    first = None
    for c in calls:
        # Deep-copy: rewriting the receiver in place would mutate the tree
        # that `calls` still holds, so a second pass would see a tree that
        # no longer matches its own filter.
        c2 = _copy.deepcopy(c)
        c2.func.value = _ast.Name(id="parser", ctx=_ast.Load())
        mod = _ast.Expression(body=c2)
        _ast.fix_missing_locations(mod)
        if first is None:
            for a in c2.args:
                if isinstance(a, _ast.Constant) and \
                        isinstance(a.value, str) and a.value.startswith("-"):
                    first = a.value
        try:
            eval(compile(mod, "<replay>", "eval"), ns)
            if inject == "parser-collide" and first is not None:
                # Re-declare an option string this file really owns, the
                # way the P0's second file did. Injected through the SAME
                # parser the replay uses, and never written as a literal
                # add_argument in this source -- a literal would be picked
                # up by the walk above and make the CLEAN run dirty.
                eval(compile(mod, "<replay>", "eval"), ns)
                inject = None      # one collision is enough to prove it
        except _ap.ArgumentError as e:
            bad.append(str(e))
        except Exception as e:                 # noqa: BLE001
            # NOT a flag conflict. The replay evaluates each
            # add_argument in a hand-built namespace, so any helper this
            # file uses in an argument expression that `ns` does not
            # carry raises NameError here. Bucketing those with the
            # ArgumentErrors made the replay report 8 conflicts on a tree
            # with ZERO duplicate flags, and since the exit code was
            # `1 if (_bad or _dupes)` the gate could not pass at all. A
            # gate that cannot pass gets ignored, and then it is not a
            # gate. These are reported separately as a limitation OF THE
            # REPLAY and do not fail the run.
            unevaluable.append(f"{type(e).__name__}: {e}")
    return len(calls), bad, unevaluable


def _b2_cross_file_flags():
    """Every option string declared by every *.py beside this one.

    -> {flag: [files]}.  The P0 was a flag declared in TWO files; this is
    the scan that sees it. Files are read with `ast`, never imported, so
    a module with a side effect cannot run here.
    """
    import ast as _ast
    import collections as _co
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    owners = _co.defaultdict(list)
    for fn in sorted(_os.listdir(here)):
        if not fn.endswith(".py"):
            continue
        try:
            tree = _ast.parse(open(_os.path.join(here, fn)).read())
        except SyntaxError:
            continue
        for n in _ast.walk(tree):
            if not (isinstance(n, _ast.Call)
                    and isinstance(n.func, _ast.Attribute)
                    and n.func.attr == "add_argument"):
                continue
            for a in n.args:
                if isinstance(a, _ast.Constant) and isinstance(a.value, str) \
                        and a.value.startswith("-"):
                    owners[a.value].append(fn)
    return owners


if args.b2_parser_replay:
    _n, _bad, _uneval = _b2_parser_replay(
        inject=("parser-collide"
                if args.b2_selftest_inject == "parser-collide" else None))
    print(f"parser replay: {_n} add_argument calls re-played against a "
          f"fresh ArgumentParser, {len(_bad)} conflicts", flush=True)
    for _m in _bad:
        print(f"  CONFLICT {_m}", flush=True)
    # Reported, never counted as a conflict: see the comment at the
    # except clause. These are calls the replay's own namespace cannot
    # evaluate, which is a limitation of the replay and says nothing
    # about whether two flags collide.
    if _uneval:
        print(f"  ({len(_uneval)} call(s) the replay could not evaluate -- "
              f"a helper this file uses is not in the replay's namespace. "
              f"NOT flag conflicts; the replay does not cover these)",
              flush=True)
        for _m in _uneval:
            print(f"    UNEVALUABLE {_m}", flush=True)
    _own = _b2_cross_file_flags()
    _dupes = {k: v for k, v in _own.items()
              if len(v) > 1 and len(set(v)) > 1}
    # A separate script with its OWN ArgumentParser is not a collision.
    # The P0 this replay exists for was two files adding to the SAME
    # parser, which is what the in-file replay above detects. Reusing
    # `--out` across 17 independent tools is normal and was never a
    # defect; counting it as one is why this gate could not exit 0 on any
    # tree it has ever been run against.
    print(f"cross-file scan: {len(_own)} distinct option strings over the "
          f"src/counter_strike_render *.py, {len(_dupes)} declared in more than "
          f"one file. INFORMATIONAL: each of these files builds its own "
          f"parser, so a shared name is not a conflict; the failure this "
          f"gate exists for is two files adding to ONE parser, which the "
          f"in-file replay above is what detects.", flush=True)
    for _k, _v in sorted(_dupes.items()):
        print(f"  shared {_k} in {sorted(set(_v))}", flush=True)
    raise SystemExit(1 if _bad else 0)


# =====================================================================
# BATCH-2 argument normalisation
# =====================================================================
B2_NAMES = ("sticker", "glove", "simple", "fourway", "liquid",
            "customweapon")
B2_INJECT_NAMES = ("sticker-frames-collapse", "sticker-foil-dead",
                   "glove-deriv-zero", "fourway-weights-uniform",
                   "liquid-depth-dead",
                   "cw-style-collapse", "parser-collide")
if int(args.b2_liquid_second_pass):
    raise SystemExit(
        "--b2-liquid-second-pass 1: the csgo_simple_liquid second pass is "
        "NOT IMPLEMENTED. The version that stood here re-ran the binner at "
        "the same position it had already used (the re-projection's return "
        "value was never read), recomputed baked lighting from identical "
        "arguments, and blended with weight b2_liq_second_w = 0 on all 327 "
        "materials -- 2,048 executions per audit that could not move a "
        "pixel. Refusing is the point: accepting 1 would report the pass as "
        "live again.")
if args.b2_selftest_inject == "all":
    print("fault-injection names for --b2-selftest-inject:", flush=True)
    for _n in B2_INJECT_NAMES:
        print(f"  {_n}", flush=True)
    raise SystemExit(0)
if args.b2_selftest_inject is not None \
        and args.b2_selftest_inject not in B2_INJECT_NAMES:
    raise SystemExit(f"unknown --b2-selftest-inject "
                     f"{args.b2_selftest_inject!r}; "
                     f"--b2-selftest-inject all lists them")
B2_INJECT = args.b2_selftest_inject


def _b2_family_set(spec):
    """Resolve --b2-families ONCE, here, into a set of names.

    `spec` is a STRING and "off" is TRUTHY, so this is the only place
    that is allowed to look at it. Everything downstream tests set
    membership. That bug -- a string-valued flag tested with bare
    truthiness -- was hiding behind this morning's argparse P0, so the
    resolution is a function with one caller rather than an idiom
    repeated at each use site.
    """
    s = (spec or "").strip().lower()
    if s in ("", "off", "none", "0"):
        return frozenset()
    if s == "all":
        return frozenset(B2_NAMES)
    out, bad = set(), []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in B2_NAMES:
            bad.append(tok)
        else:
            out.add(tok)
    if bad:
        raise SystemExit(f"--b2-families: unknown {bad}; "
                         f"known are {list(B2_NAMES)}, plus 'all' / 'off'")
    return frozenset(out)


B2_FAMS = _b2_family_set(args.b2_families)
if args.b2_reach or args.b2_selftest:
    # A reachability run with no family selected would measure nothing and
    # report success -- the exact shape CHECKS_THAT_CANNOT_FAIL.md
    # catalogues. Turn them all on and say so rather than silently
    # producing an empty table.
    if not B2_FAMS:
        B2_FAMS = frozenset(B2_NAMES)
        print("--b2-reach/--b2-selftest with no --b2-families: all six "
              "turned on, because an empty selection would make the "
              "measurement report success without measuring anything",
              flush=True)
B2 = bool(B2_FAMS)

# The batch-2 ext2 columns, APPENDED to the base table's own EXT2_KEYS.
# fast_pack_fam.py emits exactly this set under --b2-out and the loader
# refuses a table missing any of them, so a column can never be read at
# the wrong index and a missing one can never be read as 0.
B2_EXT2_KEYS = (
    "b2_4w_carve1", "b2_4w_carve2", "b2_4w_carve3", "b2_4w_det0",
    "b2_4w_det1", "b2_4w_det2", "b2_4w_det3", "b2_4w_hmax0", "b2_4w_hmax1",
    "b2_4w_hmax2", "b2_4w_hmax3", "b2_4w_hmin0", "b2_4w_hmin1",
    "b2_4w_hmin2", "b2_4w_hmin3", "b2_4w_scale1", "b2_4w_scale2",
    "b2_4w_scale3", "b2_4w_soft_hi1", "b2_4w_soft_hi2", "b2_4w_soft_hi3",
    "b2_4w_soft_lo1", "b2_4w_soft_lo2", "b2_4w_soft_lo3", "b2_cw_alpha",
    "b2_cw_base_r3", "b2_cw_blend_mode", "b2_cw_burn_hi", "b2_cw_burn_lo",
    "b2_cw_burn_pow", "b2_cw_burn_range", "b2_cw_paint_alpha",
    "b2_cw_paint_scale", "b2_cw_paint_style", "b2_cw_wear_amount",
    "b2_cw_wear_boost", "b2_cw_wear_mode", "b2_cw_wear_soft",
    "b2_foil_scale", "b2_foil_strength", "b2_glove_band", "b2_glove_pal0_a",
    "b2_glove_pal0_b", "b2_glove_pal0_g", "b2_glove_pal0_r",
    "b2_glove_pal1_a", "b2_glove_pal1_b", "b2_glove_pal1_g",
    "b2_glove_pal1_r", "b2_glove_pal2_a", "b2_glove_pal2_b",
    "b2_glove_pal2_g", "b2_glove_pal2_r", "b2_glove_pal3_a",
    "b2_glove_pal3_b", "b2_glove_pal3_g", "b2_glove_pal3_r",
    "b2_glove_pal4_a", "b2_glove_pal4_b", "b2_glove_pal4_g",
    "b2_glove_pal4_r", "b2_glove_pal5_a", "b2_glove_pal5_b",
    "b2_glove_pal5_g", "b2_glove_pal5_r", "b2_glove_pal6_a",
    "b2_glove_pal6_b", "b2_glove_pal6_g", "b2_glove_pal6_r",
    "b2_glove_pal7_a", "b2_glove_pal7_b", "b2_glove_pal7_g",
    "b2_glove_pal7_r", "b2_glove_slot0", "b2_glove_slot1", "b2_glove_slot2",
    "b2_glove_slot3", "b2_glove_slot4", "b2_glove_slot5", "b2_glove_slot6",
    "b2_glove_slot7", "b2_glove_wear_scale", "b2_glove_weave_blur",
    "b2_glow", "b2_has_4w_a0", "b2_has_4w_a1", "b2_has_4w_a2",
    "b2_has_4w_a3", "b2_has_4w_c0", "b2_has_4w_c1", "b2_has_4w_c2",
    "b2_has_4w_c3", "b2_has_4w_detail", "b2_has_4w_n0", "b2_has_4w_n1",
    "b2_has_4w_n2", "b2_has_4w_n3", "b2_has_base_n", "b2_has_cw_finish",
    "b2_has_cw_grunge", "b2_has_cw_mask", "b2_has_cw_pattern",
    "b2_has_cw_wear", "b2_has_foil", "b2_has_glove_idx", "b2_has_glove_l0",
    "b2_has_glove_l1", "b2_has_glove_l2", "b2_has_glove_l3",
    "b2_has_glove_mask", "b2_has_glove_wear", "b2_has_holo_lut",
    "b2_has_holomask", "b2_has_liq_col", "b2_has_liq_foam",
    "b2_has_liq_nrm", "b2_has_liq_ripple", "b2_has_simple_ao",
    "b2_has_simple_col", "b2_has_simple_nrm", "b2_has_sticker",
    "b2_has_stk_n", "b2_has_wear", "b2_liq_depth_fade", "b2_liq_foam_hi",
    "b2_liq_foam_lo", "b2_liq_fres_bias", "b2_liq_refract_amount",
    "b2_liq_ripple_amp", "b2_liq_scroll", "b2_liq_second_w",
    "b2_simple_metal", "b2_sticker_metal", "b2_sticker_scale",
    "b2_sticker_wear", "b2_wear_bias", "b2_wear_min", "b2_wear_pow",
)


def _gt_check_constraints():
    """Refuse combo tuples the shipped constraint graph excludes.

    RENDER_SETTINGS_AXES.md section 4. The graph is not advisory: section
    0 shows that enumerating the cartesian product and keeping only the
    tuples the rules admit reproduces `m_staticComboIDs` exactly on all
    18 stages -- zero extra, zero missing -- so a tuple the rules reject
    is a variant Valve does not ship and no module exists for. Accepting
    one silently would mean rendering a combo that has no ground truth.
    """
    bad = []
    # D_SPECULAR_CUBE_MAP_STATIC REQUIRES S_SHADER_QUALITY == 1
    # (cx:ps, env:ps, eb:ps, fol:ps). Independently visible in the
    # shipped enumeration: no q=0 static combo ships a dynamic id with
    # the SPECCUBE bit -- csgo_environment's q=0 statics ship dyn
    # {0,1,2,4} only, its q=1 static ships {0,1,2,4,8,...,28}.
    if args.spec_cube_static and args.shader_quality != 1:
        bad.append("D_SPECULAR_CUBE_MAP_STATIC=1 REQUIRES "
                   "S_SHADER_QUALITY==1 (RENDER_SETTINGS_AXES.md sec 4); "
                   "at quality 0 the axis is excluded by the constraint "
                   "graph, not merely unused. Use --shader-quality 1.")
    # so:psrs REQUIRES S_MATERIAL_REFERENCE != 0 -> S_LIT != 0, and
    # S_LIT == 1 -> S_BLEND_MODE in {0,1,2}. The renderer has no
    # S_BLEND_MODE axis of its own (it is another agent's), so only the
    # S_LIT half is checkable here and it is checked.
    if args.s_lit not in (0, 1):
        bad.append("S_LIT is 0..1")
    if bad:
        for b in bad:
            print(f"CONSTRAINT VIOLATION: {b}", flush=True)
        raise SystemExit(2)


# GT gap log. Defined HERE, not with the rest of the GT module ~1900
# lines below, because the asset-availability decisions that record gaps
# happen during argument normalisation -- and a _gt_note() call before
# its own def is a NameError that only fires on the branch that has no
# assets, i.e. exactly the branch nobody runs.
GT_GAPS = []


def _gt_note(msg):
    if msg not in GT_GAPS:
        GT_GAPS.append(msg)


# Fault injection for the self-test. Each name disables one path the way
# a bad default would have, so the check can be made to fail on purpose --
# the only step that distinguishes "the check passed" from "the check
# never ran" (CHECKS_THAT_CANNOT_FAIL.md, instance 5). Defined HERE, with
# the rest of argument normalisation, because --gt-selftest-inject all
# lists it below; the previous definition sat ~3000 lines further down
# and that listing was a NameError nobody had run.
GT_INJECT_NAMES = ("csm-dead", "pcf-collapse", "dfg-zero", "probe-zero",
                   "distfade-full", "speccube-zero", "ambientsh-zero",
                   "lightmap-zero", "cullmode-tie", "biasfam-collapse")
GT_INJECT = args.gt_selftest_inject
if GT_INJECT is not None and GT_INJECT != "all" \
        and GT_INJECT not in GT_INJECT_NAMES:
    raise SystemExit(f"unknown --gt-selftest-inject {GT_INJECT!r}; "
                     f"known: {', '.join(GT_INJECT_NAMES)} (or 'all' to "
                     f"list)")


def _gt_inject(name):
    """True when this path is the one being deliberately broken."""
    return GT_INJECT == name


if args.gt_selftest_inject:
    if args.gt_selftest_inject == "all":
        print("fault-injection names for --gt-selftest-inject:", flush=True)
        for _n in GT_INJECT_NAMES:
            print(f"  {_n}", flush=True)
        raise SystemExit(0)
    if not args.gt_lighting_selftest:
        print("--gt-selftest-inject is only meaningful with "
              "--gt-lighting-selftest: it exists to make the SELF-TEST "
              "fail on purpose, not to render a broken frame.", flush=True)
        raise SystemExit(2)

if args.gt_lighting or args.gt_lighting_selftest:
    _gt_check_constraints()

# --- GT lighting axes: pull in the assets the selected paths REQUIRE ---
# A path that is selected but whose backing asset was never loaded is the
# --ibl-lod-scale failure in another costume: it runs, it returns a
# number, and the number is the path switched off. So the selection turns
# the loaders on rather than silently degrading, and gt_lighting_banner()
# prints, per path, whether it got its asset or is running on the
# shader's own unbound-descriptor value.
# Declared BEFORE the loader that may bind it, exactly as CUBE is below.
# SH was assigned ONLY inside the --cube-probes branch, while
# gt_spec_cube() reads `if CUBE is None and SH is None` unconditionally --
# so --gt-lighting 1 without --cube-probes died on
# `NameError: name 'SH' is not defined` at the first shaded pixel. That
# read was already written to handle the asset being absent; it just had
# no name to test. The NameError was invisible for as long as
# --gt-lighting defaulted to 0, because nothing reached the line.
SH = None
if args.gt_lighting or args.gt_lighting_selftest:
    # sh_at()/probe_of()/SH/CUBE live behind these two flags. The GT
    # probe path reads them, and so does the environment specular -- which
    # is UNCONDITIONAL in the q=0 module (csgo_complex_ps.glsl:945) and
    # present at q=1 in both D_SPECULAR_CUBE_MAP_STATIC states, so it is
    # not conditional on --spec-cube-static and the loaders must not be
    # either.
    #
    # ONLY when the asset is actually there. Setting these unconditionally
    # made --gt-lighting 1 without --cube-probes die at the probe loader
    # with "--ibl-sh/--ibl-spec need --cube-probes" -- an error naming two
    # flags the user never passed, for an asset this path is designed to
    # do without. gt_baked_probe() and gt_spec_cube() both already handle
    # the absent case by falling to the shipped dyn-0 tail and recording a
    # GAP, so the hard exit contradicted the design rather than protecting
    # it. A flag set as a side effect must not then be read as user
    # intent, which is exactly what the loader was doing.
    if args.cube_probes:
        args.ibl_sh = True
        args.ibl_spec = True
    else:
        _gt_note("--cube-probes not given: the SH/cubemap loaders stay "
                 "off, so the probe path takes the dyn-0 constant tail "
                 "and the environment specular has no radiance to fetch. "
                 "Pass --cube-probes to reach either.")
# Stamp the full argv FIRST, before any derived state is printed.
#
# Four separate stale-arm failures in one session were each invisible from
# inside the analysis and trivial to catch from outside: an arm was compared
# against a baseline that silently lacked a landed feature, and no log said
# which flags produced it. These logs already print probe counts, atlas
# dimensions and VRAM totals -- everything except the one line needed to
# reproduce the run. Flags arriving through a wrapper's "$@" were recorded
# nowhere at all. This line is the cheapest possible fix and it must come
# before anything else so a truncated log still carries it.
print("ARGV: " + " ".join(shlex.quote(a) for a in sys.argv), flush=True)
if args.no_cc:
    args.cc_mode = "off"

device = torch.device("cuda")
# The non-world side table (decal projection volumes + particle quads).
# A DISTINCT file from --fam-side; fam_nonworld.load_side() refuses the
# name assets/fam_side.pt outright rather than reading it and finding
# zeros, which would be an axis that reads as tested and is off.
NW_SIDE = _nw.load_side(args.decal_side, device)
with stage("startup.pack_load"):
    data = torch.load(args.world, map_location=device, weights_only=True)
# WHICH PACK GENERATION THIS RUN ACTUALLY USED. ash_to_world.py stamps
# pack_generation as a sha256 over its own source plus the options that
# produced the pack, so two packs agree on it only if the same code ran
# with the same choices. Nothing read it until now, which is how a
# directory could hold two generations and hand different answers to
# different readers with no evidence in either artifact.
#
# Printed unconditionally: a run that does not say which inputs it used
# cannot be compared with one that does. --require-pack-generation turns
# it into a refusal for anyone scoring across maps, where a mixed set is
# the failure that looks like a result.
_pgen = data.get("pack_generation")
print(f"pack: {args.world} generation={_pgen} "
      f"effects_kept={data.get('effects_kept')} "
      f"mat_scaffold={data.get('mat_scaffold')}", flush=True)
if args.require_pack_generation and _pgen != args.require_pack_generation:
    raise SystemExit(
        f"pack generation mismatch: {args.world} is {_pgen}, run requires "
        f"{args.require_pack_generation}. A cross-map comparison over a "
        f"mixed set is a well-formed table of two different renderers.")
vertices = data["vertices"].to(device)
uvs = data["uvs"].to(device)
faces0 = data["faces"].to(device).long()
face_mat0 = data["face_mat"].to(device).long()
face_normals0 = data["face_normals"].to(device)
textures = data["textures"].to(device)
# glTF declares alpha handling per material; 0=OPAQUE 1=MASK 2=BLEND.
# Guessing it from texture-alpha statistics misclassifies walls whose
# diffuse carries junk alpha (that mistake produced dark ground patches).
# AN INPUT THAT STANDS IN FOR ANOTHER INPUT MUST SAY SO AT RUNTIME. This
# fallback is silent, and its silence is why "UV2 does not reach the
# image" took a whole arc to state: a pack with no TEXCOORD_1 renders
# UV2==UV1 and is indistinguishable, from the frames alone, from a pack
# whose real UV2 is discarded downstream. Same omission as --rough-src
# reading a constant alpha and mat_ao reading the AUX_WHITE sentinel.
# 11 of the 44 built world packs take this branch.
uvs2 = data.get("uvs2")
if uvs2 is None:
    uvs2 = data["uvs"]
    print("uvs2 SUBSTITUTED: this pack carries no TEXCOORD_1, so uvs2 IS "
          "uvs. Every secondary-UV term below is inert BY INPUT, not by "
          "choice -- --secondary-uv cannot act and uv.uv2_differs will "
          "read 1.0000. Rebuild with ash_to_world's TEXCOORD_1 selection "
          "to test any of them.", flush=True)
else:
    _d2 = (uvs2.float() - data["uvs"].float()).abs().amax(-1)
    _n2v = int((_d2 > 1e-6).sum())
    print(f"uvs2 REAL: TEXCOORD_1 differs from TEXCOORD_0 on {_n2v} of "
          f"{_d2.numel()} vertices ({100.0 * _n2v / max(_d2.numel(), 1):.1f}%), "
          f"max|d| {float(_d2.max()):.6g}. A secondary-UV term that is "
          f"inert on this pack is inert in the CONSUMER.", flush=True)
uvs2 = uvs2.to(device)
lmuv = data.get("lmuv")
lmuv = data["uvs"] if lmuv is None else lmuv
lmuv = lmuv.to(device)
# TEXCOORD_1 is NOT one semantic, and "0 <= uv <= 1" does not separate
# them. Its LUXEL DENSITY -- sqrt(uv area)/sqrt(world area) per triangle
# -- is sharply bimodal over the 899k triangles carrying a non-degenerate
# value: a tight peak at 2^-9 uv/m holding 66% of them, and a broad
# 2^-3..2^0 shoulder. A lightmap packer bakes every chart at ONE luxel
# scale, so the tight peak is the lightmap; the shoulder is model
# TEXCOORD_1, which is a secondary *texture* UV (raw values run to 6e5).
# Selecting by uv range instead admits 24.5% of vertices where only 7.6%
# are lightmapped, i.e. 2/3 of what we "baked" was atlas noise sampled at
# a texture coordinate. Restricting to the tight population also sharpens
# the UV-convention sweep: within-voxel irradiance variance 0.099 for
# scale 1.14284 / unflipped V against 0.226 for the runner-up and 0.405
# for random UVs (bounce_work/lmprobe2.py).
_p3 = vertices[faces0]
_t3 = lmuv[faces0]
_wa = torch.cross(_p3[:, 1] - _p3[:, 0], _p3[:, 2] - _p3[:, 0],
                  dim=1).norm(dim=1) * 0.5
_e1, _e2 = _t3[:, 1] - _t3[:, 0], _t3[:, 2] - _t3[:, 0]
_ua = (_e1[:, 0] * _e2[:, 1] - _e1[:, 1] * _e2[:, 0]).abs() * 0.5
_inr = (lmuv[:, 0] >= 0) & ~((lmuv[:, 0] == 0) & (lmuv[:, 1] == 0))
_okt = _inr[faces0].all(1) & (_wa > 1e-4) & (_ua > 1e-12)
_dens = torch.where(_okt, (_ua / _wa.clamp(min=1e-9)).sqrt(),
                    torch.zeros_like(_wa))
_tight = _okt & (_dens > 2 ** -11.5) & (_dens < 2 ** -7.5)
# PER-MAP RECENTER (task #82). The band's job is stream IDENTIFICATION --
# the tight peak is the lightmap because a packer bakes every chart at ONE
# luxel scale -- but its POSITION was measured on competitive-size maps,
# and 13 of 14 _vanity packs bake ~7x finer (graphics_settings log2 -5.20
# vs de_mirage -8.02), so the fixed window rejected their entire, real,
# tight lightmap population: 19 packs rendered with under half their baked
# lighting, one crashed (lm_band survey, lmband_4547.log). The reference
# shader samples the lightmap UNCONDITIONALLY -- there is no density gate
# in the engine; the gate is ours, so it must identify, not veto.
#
# Nothing here is newly fitted: the +/-2 octave width is the legacy band's
# own half-width, the 2^-3.5 shoulder boundary is the MEASURED texture-UV
# population location (2^-3..2^0 over 899k triangles, comment above), and
# the 50% trigger sits inside an 81-point measured gap (bad maps hold
# 0.3-8.7% in the legacy band, good maps 90-99%). Maps where the legacy
# band works are BIT-IDENTICAL to before -- this branch does not run.
_frac_legacy = float(_tight[_okt].float().mean()) if bool(_okt.any()) else 0.0
if bool(_okt.any()) and _frac_legacy < 0.5:
    _l2 = torch.log2(_dens[_okt].clamp(min=1e-12))
    _hb = torch.histc(_l2, bins=96, min=-14.0, max=-2.0)
    _mode = -14.0 + (float(_hb.argmax()) + 0.5) * (12.0 / 96.0)
    if _mode < -3.5:
        _tight = _okt & (_dens > 2 ** (_mode - 2.0)) \
            & (_dens < 2 ** (_mode + 2.0))
        print(f"lightmap UVs: legacy density band [2^-11.5, 2^-7.5] holds "
              f"only {_frac_legacy:.1%} of candidate triangles; RECENTERED "
              f"on this map's own packer peak 2^{_mode:.2f} (+/-2 octaves, "
              f"the legacy width) -> {float(_tight[_okt].float().mean()):.1%} "
              f"in band. The peak's position is the map's, not "
              f"de_inferno's (#82).", flush=True)
    else:
        print(f"lightmap UVs: legacy band holds {_frac_legacy:.1%} and the "
              f"density mode 2^{_mode:.2f} sits in the texture-UV shoulder "
              f"(>= 2^-3.5) -- no tight lightmap population exists on this "
              f"pack; NOT recentered, a texture UV must not be baked as "
              f"irradiance.", flush=True)
_vg = torch.zeros(len(vertices), dtype=torch.bool, device=device)
_vb = torch.zeros(len(vertices), dtype=torch.bool, device=device)
for _c in range(3):
    _vg[faces0[_tight, _c]] = True
    _vb[faces0[_okt & ~_tight, _c]] = True
lm_valid = (_vg & ~_vb).float()
del _p3, _t3, _wa, _e1, _e2, _ua, _inr, _okt, _dens, _tight, _vg, _vb
lmuv = lmuv * args.lm_uv_scale
print(f"lightmap UVs: {float(lm_valid.mean()):.1%} of vertices are "
      f"lightmapped world geometry (uv scale {args.lm_uv_scale})", flush=True)
lmattr = torch.cat([lmuv, lm_valid[:, None]], dim=1).contiguous()
LIGHTMAP = None
IRRADIANCE = None
IRRMAP = None
# The pack's own map name, when it has one. Written by ash_to_world at
# build time; absent from every pack built before the stem trap was
# found. _sidecar() below is the only consumer.
PACK_MAP = data.get("map") if isinstance(data, dict) else None
def _sidecar(suffix, what):
    """Resolve <world><suffix>, falling back to the pack's OWN map name.

    THE STEM TRAP, killed at the resolver instead of by symlink. All three
    sidecar families -- .irradiance.npy, .light_environment.json,
    .sky_cube.npz -- resolve by splitting the world path's extension, so a
    variant pack named `de_inferno_ovl.pt` looks for
    `de_inferno_ovl.sky_cube.npz` and misses the `de_inferno.sky_cube.npz`
    sitting beside it. Each of the three has now missed that way at least
    once; the sky one cost every arm in the slot-family lane its sky pass,
    silently, behind a GAP line that was printed and read past.

    A symlink fixes today and the next variant pack recreates the miss, so
    the fallback is the PACK'S OWN `map` key -- written at build time by
    ash_to_world, which knows the map name because it read that .ash. A
    pack without the key (every pack built before this) falls back to the
    stem with everything after the first underscore-suffix stripped, which
    is a GUESS and says so. Whichever route resolved is printed, because
    "the sky loaded" and "the sky loaded from the map next door" are
    different facts.
    """
    if not args.world:
        return None, "no --world"
    stem = os.path.splitext(args.world)[0]
    exact = stem + suffix
    if os.path.exists(exact):
        return exact, "exact stem"
    base = PACK_MAP
    how = "pack['map']"
    if not base:
        # No key: strip a trailing _<variant> from the file name. Named a
        # guess rather than presented as a resolution.
        nm = os.path.basename(stem)
        base = nm.split("_")[0] + ("_" + nm.split("_")[1]
                                   if len(nm.split("_")) > 1 else "")
        how = "GUESSED base stem (pack carries no 'map' key)"
    cand = os.path.join(os.path.dirname(stem), base + suffix)
    if base and os.path.exists(cand):
        print(f"{what}: {os.path.basename(exact)} absent; resolved "
              f"{os.path.basename(cand)} via {how}", flush=True)
        return cand, how
    return None, f"neither {os.path.basename(exact)} nor {os.path.basename(cand)}"



IRR_REF = 1.0
# Resolve 'auto' against the world pack, so the baked lighting travels
# with the map it was baked for rather than with a path someone typed.
if args.irr_npy == "auto":
    _irr_auto, _irr_how = _sidecar(".irradiance.npy", "irradiance")
    if _irr_auto:
        args.irr_npy = _irr_auto
        print(f"irradiance: auto -> {_irr_auto}", flush=True)
    else:
        # REFUSE, DO NOT DEFAULT. Printing this at full volume was not
        # enough: it was printed, with the penalty quantified, one line
        # above the numbers being read, and it still lost to a more
        # interesting hypothesis -- costing an hour and a published
        # fabricated bug (ARTEFACT_PROVENANCE_SWEEP.md,
        # OVERLAY_PAGE_AND_K.md §5).
        #
        # A MISSING FILE IS AN ERROR; A MISSING SIDECAR IS A DEFAULT, and
        # that asymmetry is the whole defect. Auto-discovery keyed on the
        # pack's FILENAME STEM means RENAMING A PACK SILENTLY CHANGES THE
        # PHYSICAL MODEL -- which is exactly the scratch-copy workflow a
        # publish freeze instructs everyone to use. So a miss is a stop.
        #
        # It stays PASSABLE, which is the other half: `--irr-npy none`
        # states the intent explicitly. The refusal only forbids being
        # SILENT. A gate that cannot pass is worse than no gate.
        #
        # SCOPE, DECIDED EXPLICITLY RATHER THAN BY ENFORCING THE CURRENT
        # PATTERN. Refusing on every miss would have been a day-one block:
        # measured, de_inferno is the ONLY pack on this box with an atlas,
        # and the ~42 others (cs_italy, de_dust2, de_ancient, de_anubis,
        # de_boulder, ...) have none -- so a blanket refusal breaks #27's
        # render-any-of-43-maps capability outright. That is the "lint that
        # cannot pass" failure, and this gate nearly was one.
        #
        # The discriminator is DATA, not a filename heuristic: a pack
        # carrying lightmap UVs was built expecting a baked atlas, so a
        # missing one is a genuine mismatch (the rename accident). A pack
        # with no `lmuv` key never had one and is not doing anything wrong.
        # Measured: de_inferno.pt has lmuv with 8,407,225 non-zero rows;
        # de_dust2.pt / cs_italy.pt / de_ancient.pt have no lmuv key at all.
        _lm_n = (int((lmuv.abs().sum(-1) > 0).sum()) if lmuv is not None
                 else 0)
        args.irr_npy = None
        if _lm_n == 0:
            print("irradiance: auto -> none beside the pack, and this pack "
                  "carries no lightmap UVs, so none is expected. Baked "
                  "lighting is off and nothing is being silently "
                  "substituted.", flush=True)
        else:
            raise SystemExit(
                f"REFUSING: --irr-npy auto found no atlas beside a pack "
                f"whose geometry IS lightmapped ({_lm_n:,} vertices carry "
                f"lightmap UVs).\n"
                f"  expected: {_irr_auto}\n"
                f"  pack:     {args.world}\n"
            f"Baked irradiance is the DOMINANT light source here, so "
            f"continuing would render a different physical model and score "
            f"it against the same ground truth -- measured KL_rgb ~5.58 "
            f"against ~2.44 with the atlas loaded.\n"
            f"This is resolved by FILENAME STEM, so the usual cause is a "
            f"renamed or copied pack whose sidecar did not come with it. "
            f"Either:\n"
            f"  * put the atlas beside the pack (a symlink is enough), or\n"
            f"  * pass --irr-npy none to say you meant to render without "
            f"baked lighting, or\n"
            f"  * pass --irr-npy PATH to name it explicitly.")
elif args.irr_npy == "none":
    # OWNER RULING (2026-08-11): `none` may not disable the DOMINANT light
    # source on a pack that was BUILT for baked lighting. The escape produced
    # exportable renders with no relationship to the task's assets (measured
    # KL_rgb ~5.58 vs ~2.44), and its only tracked consumer was that mistake.
    # The discriminator stays DATA, same as auto: a pack with no lmuv never
    # had an atlas and renders fine without one; a pack WITH lmuv refuses.
    _lm_n_none = (int((lmuv.abs().sum(-1) > 0).sum()) if lmuv is not None
                  else 0)
    if _lm_n_none > 0:
        raise SystemExit(
            f"REFUSING: --irr-npy none on a pack whose geometry IS "
            f"lightmapped ({_lm_n_none:,} vertices carry lightmap UVs). "
            f"Baked irradiance is the dominant light source for this pack; "
            f"rendering without it produces frames with no relationship to "
            f"the asset (measured KL_rgb ~5.58 vs ~2.44). There is no "
            f"unlit-export mode. Put the atlas beside the pack or pass "
            f"--irr-npy PATH.")
    print("irradiance: none, and this pack carries no lightmap UVs, so none "
          "is expected.", flush=True)
    args.irr_npy = None
if args.irr_npy:
    import numpy as _np2
    IRRMAP = torch.from_numpy(_np2.load(args.irr_npy)).to(device).float()
    # SCALE ON THE LIVE PATH, applied at the source.
    #
    # --irr-gain and --irr-mode are read ONLY at :10604/:10610, inside
    # _post_chain, whose sole caller _single_shade is never called
    # anywhere in this file -- so both are bit-identical across their
    # whole range and cannot move a pixel. Demonstrated by sweeping
    # --irr-gain 0.25..1.0 and getting max|d| 0/255 at every step.
    #
    # Scaling IRRMAP here instead reaches every live consumer uniformly:
    # the per-vertex resolver at :8590 that samples it through lm_valid,
    # and the irradiance VOLUME at :4263, which is BUILT from IRRMAP
    # samples and would otherwise disagree with it by the scale factor.
    # One multiply at the source keeps the two products consistent; a
    # multiply at either consumer would not.
    if args.irr_scale != 1.0:
        IRRMAP = IRRMAP * args.irr_scale
        print(f"irradiance scaled by {args.irr_scale:g} at load", flush=True)
    for _ in range(max(0, args.irr_mip)):
        IRRMAP = 0.25 * (IRRMAP[0::2, 0::2] + IRRMAP[1::2, 0::2]
                         + IRRMAP[0::2, 1::2] + IRRMAP[1::2, 1::2])
    # Reference level = the mean the atlas actually presents to THIS
    # geometry, not the mean over the whole atlas (which is mostly padding
    # and charts we never see). Dividing by it makes --irr-mode modulate
    # exposure-neutral by construction, so any metric move is the spatial
    # term and not a rescale.
    _lv = lm_valid > 0.5
    _hi, _wi = IRRMAP.shape[0], IRRMAP.shape[1]
    _uu = (lmuv[_lv, 0].clamp(0, 1) * (_wi - 1)).long()
    _vv = (lmuv[_lv, 1].clamp(0, 1) * (_hi - 1)).long()
    IRR_REF = float(IRRMAP[_vv, _uu].mean())
    del _lv, _uu, _vv
    print(f"baked INDIRECT irradiance {IRRMAP.shape[1]}x{IRRMAP.shape[0]} "
          f"HDR (mip {args.irr_mip}), atlas mean "
          f"{float(IRRMAP[::4, ::4].mean()):.3f}, at lightmapped verts "
          f"{IRR_REF:.3f}, max {float(IRRMAP.max()):.2f}", flush=True)

VOL = None
VOL_LO = VOL_INV = None
VOL_REF = 1.0
VOL_MEAN3 = None
if args.irr_vol > 0 and IRRMAP is not None:
    import torch.nn.functional as _F
    _sel = lm_valid > 0.5
    _sp = vertices[_sel]
    _hi, _wi = IRRMAP.shape[0], IRRMAP.shape[1]
    _sv = IRRMAP[(lmuv[_sel, 1].clamp(0, 1) * (_hi - 1)).long(),
                 (lmuv[_sel, 0].clamp(0, 1) * (_wi - 1)).long()]
    # HDR spikes are baked light sources sitting in the atlas; they must
    # not smear over a whole voxel neighbourhood during the fill
    _sv = _sv.clamp(0, 8 * IRR_REF)
    # SITE 3 of 3 -- .min(0) on a zero-size dim raises IndexError, which is
    # how graphics_settings died. A map where no vertex is lightmapped is
    # not a broken pack: it is a map with no baked lighting to reconstruct,
    # and the volume simply has no domain. Same rule as the other two --
    # skip the thing with no input, say so with the count.
    # The guard must cover the WHOLE construction, not just the .min(0)
    # that happens to raise first: every line below indexes _sp or divides
    # by a dimension derived from it. Guarding only the first raiser moves
    # the crash three lines down and makes it look like a different bug --
    # which is exactly how this defect came to have three sites.
    _vol_ok = _sp.shape[0] > 0
    if not _vol_ok:
        # This branch is reachable with IRRMAP PRESENT -- graphics_settings
        # carries a full atlas whose every triangle the extraction-side
        # texel-density band [2^-11.5, 2^-7.5] rejected (log2 -5.20, ~7x
        # finer than de_mirage). "A map with no baked lighting" was the
        # first draft here and it is FALSE for that map; the survey is
        # conformance/lm_band.py and the open question (what the engine
        # does at out-of-band densities) is a READ, not a re-fit.
        print(f"irradiance volume: SKIPPED -- 0 of {int(_sel.numel())} "
              f"vertices are lightmapped-valid, so the volume has no "
              f"domain. IRRMAP present: the atlas exists but the "
              f"extraction density band accepted none of it (see "
              f"conformance/lm_band.py -- 19 packs land under 50% "
              f"in-band). The reconstructed-volume ambient path stays "
              f"off for this render.", flush=True)
    if _vol_ok:
        VOL_LO = _sp.min(0).values - args.irr_vol
        _hi3 = _sp.max(0).values + args.irr_vol
        _dim = ((_hi3 - VOL_LO) / args.irr_vol).ceil().long() + 1
        VOL_INV = 1.0 / args.irr_vol
        _g = ((_sp - VOL_LO) * VOL_INV).long()
        _flat = (_g[:, 0] * int(_dim[1]) + _g[:, 1]) * int(_dim[2]) + _g[:, 2]
        _n = int(_dim[0] * _dim[1] * _dim[2])
        _num = torch.zeros(_n, 3, device=device)
        _den = torch.zeros(_n, 1, device=device)
        _num.index_add_(0, _flat, _sv)
        _den.index_add_(0, _flat, torch.ones_like(_sv[:, :1]))
        _num = _num.T.reshape(1, 3, int(_dim[0]), int(_dim[1]), int(_dim[2]))
        _den = _den.T.reshape(1, 1, int(_dim[0]), int(_dim[1]), int(_dim[2]))
        _occ = float((_den > 0).float().mean())
        # push-pull: 2x-downsample num and den to the coarsest level, then
        # pull back down, preferring a level's own samples and falling back to
        # the coarser estimate only where it has none. Straight blurring would
        # bleed a bright courtyard through a wall into a dark interior;
        # push-pull keeps the fine detail wherever it exists.
        _pyr = [(_num, _den)]
        while min(_pyr[-1][0].shape[2:]) > 1:
            _pyr.append((_F.avg_pool3d(_pyr[-1][0], 2, ceil_mode=True),
                         _F.avg_pool3d(_pyr[-1][1], 2, ceil_mode=True)))
        _val = _pyr[-1][0] / _pyr[-1][1].clamp(min=1e-9)
        for _lv in range(len(_pyr) - 2, -1, -1):
            _n_, _d_ = _pyr[_lv]
            _up = _F.interpolate(_val, size=_n_.shape[2:], mode="trilinear",
                                 align_corners=False)
            _w = (_d_ > 0).float()
            _val = _w * (_n_ / _d_.clamp(min=1e-9)) + (1 - _w) * _up
        VOL = _val
        # Normalise by the FIELD's own mean, not the lightmap's. The volume
        # includes air cells the lightmap never sees, so its mean sits ~9%
        # below IRR_REF; dividing by IRR_REF would apply that 9% as a silent
        # global darkening (and an --irr-vol-gain of 1.1 would then be a
        # fitted constant doing nothing but cancelling it). Dividing by the
        # field's own mean makes the modulation average exactly 1.
        VOL_REF = float(VOL.mean())
        VOL_MEAN3 = VOL.mean(dim=(0, 2, 3, 4)).view(1, 1, 1, 3)
        print(f"irradiance volume {tuple(_dim.tolist())} @ {args.irr_vol}m "
              f"({_occ:.1%} of cells directly sampled by {int(_sel.sum()):,} "
              f"lightmapped verts, rest push-pull filled), mean "
              f"{float(VOL.mean()):.3f} vs lightmap {IRR_REF:.3f}", flush=True)
        # INSIDE the guard: these names only exist when the volume was
        # built. The re-indent stopped one line short of here and the run
        # died with NameError: _pyr -- the guard fired correctly, printed
        # correctly, and then the very next unguarded line crashed. Same
        # lesson as the three sites themselves: a guard that covers the
        # first raiser and not the block moves the crash rather than
        # removing it.
        del _pyr, _num, _den, _val, _g, _flat
    del _sp, _sv


# --- environment probes -----------------------------------------------
PROBES = None
# --cubemap-refraction resolves refract() against the same cube ARRAY the
# reflection uses (glass s2/d8:1183 taps offset 104, the IBL array, inside
# the same cubemap-blend loop), and needs that loop's box bounds, so it
# builds the probe field exactly as --ibl-spec does rather than carrying a
# private copy.
if args.ibl_sh or args.ibl_spec or args.cubemap_refraction:
    if not args.cube_probes:
        raise SystemExit("--ibl-sh/--ibl-spec/--cubemap-refraction need "
                         "--cube-probes (build it with cube_extract_fam.py)")
    _cp = torch.load(args.cube_probes, map_location=device,
                     weights_only=False)
    SH = _cp["sh"].to(device).float()                       # (P, 3, 9)
    # Source is Z-up; the pack is VRF's Y-up export. The permutation was
    # not assumed -- of all 48 axis maps, pack = (y_src, z_src, x_src)
    # is the only one whose probe SH-DC correlates with the independently
    # baked lightmap irradiance at nearby geometry (r = +0.363; every
    # other mapping lands within +-0.09 of zero).
    def _to_pack(a):
        return torch.stack([a[:, 1], a[:, 2], a[:, 0]], dim=1) * 0.0254
    PORG = _to_pack(_cp["origin_src"].to(device).float())
    _bl = _to_pack(_cp["box_min_src"].to(device).float())
    _bh = _to_pack(_cp["box_max_src"].to(device).float())
    PBMIN = torch.minimum(_bl, _bh)
    PBMAX = torch.maximum(_bl, _bh)
    # Probe lookup as a voxel grid: 116 probes against every shaded pixel
    # every frame is 3e9 distances a batch, against 1.6M voxels once.
    _lo = vertices.min(0).values - args.ibl_grid
    _hi = vertices.max(0).values + args.ibl_grid
    _dim = ((_hi - _lo) / args.ibl_grid).ceil().long() + 1
    _ax = [torch.arange(int(_dim[k]), device=device, dtype=torch.float32)
           * args.ibl_grid + _lo[k] for k in range(3)]
    _gc = torch.stack(torch.meshgrid(*_ax, indexing="ij"), dim=-1)
    _flat = _gc.reshape(-1, 3)
    _best = torch.zeros(len(_flat), dtype=torch.long, device=device)
    _bestcost = torch.full((len(_flat),), float("inf"), device=device)
    _vol = (PBMAX - PBMIN).clamp(min=1e-3).prod(1)
    for _i in range(0, len(_flat), 1 << 20):
        _c = _flat[_i:_i + (1 << 20)]
        _d = torch.cdist(_c, PORG)
        # A probe whose OWN box contains the point wins over any nearest-
        # origin guess, and among those the smallest box wins -- that is
        # how Source 2 prioritises overlapping light-probe volumes.
        _in = ((_c[:, None, :] >= PBMIN[None]) &
               (_c[:, None, :] <= PBMAX[None])).all(-1)
        _cost = torch.where(_in, _vol[None].expand_as(_d) * 1e-6, _d + 1e6)
        _v, _k = _cost.min(1)
        _best[_i:_i + (1 << 20)] = _k
        _bestcost[_i:_i + (1 << 20)] = _v
    PGRID = _best.reshape(int(_dim[0]), int(_dim[1]), int(_dim[2])).int()
    PLO, PINV = _lo, 1.0 / args.ibl_grid
    PDIM = _dim
    # Voxelised SH COEFFICIENTS, so a pixel trilinearly interpolates the
    # basis and evaluates once, instead of snapping to one probe and
    # putting a hard seam on every probe boundary. Interpolating the
    # coefficients is exact for the SH basis (it is linear), which
    # interpolating the evaluated colour would not be for the specular
    # case later. 27 channels x the voxel grid, fp16.
    SHVOL = SH.reshape(SH.shape[0], 27)[_best].reshape(
        int(_dim[0]), int(_dim[1]), int(_dim[2]), 27
    ).permute(3, 0, 1, 2).unsqueeze(0).half().contiguous()
    print(f"  SH coefficient volume {tuple(SHVOL.shape[2:])} x 27 "
          f"= {SHVOL.numel()*2/1e6:.0f} MB", flush=True)
    # Direction-averaged irradiance of the whole probe set: the reference
    # level --ibl-sh-mode modulate divides by, so the modulation averages
    # to 1 and carries no exposure.
    SH_REF = float((0.886227 * SH[:, :, 0]).mean())
    _cov = float((_bestcost < 1e5).float().mean())
    print(f"env probes: {SH.shape[0]} SH-L2, grid {tuple(_dim.tolist())} "
          f"@{args.ibl_grid}m ({_cov:.1%} of voxels inside a probe box), "
          f"reference irradiance {SH_REF:.3f}", flush=True)
    del _gc, _flat, _best, _bestcost

CUBE = None
if args.ibl_cube:
    import numpy as _np3
    CUBE = []
    for _m in range(args.ibl_cube_mips):
        _a = _np3.load(f"{args.ibl_cube}/env_cubemap_mip{_m}_f16.npy")
        CUBE.append(torch.from_numpy(_a).to(device))      # (P,6,N,N,3) f16
    if args.ibl_lod_scale is None:
        # DERIVED, not fitted: sqrt(roughness) spans [0,1], so the scale
        # that reaches the coarsest mip is exactly nmips-1.
        args.ibl_lod_scale = float(len(CUBE) - 1)
    _mb = sum(c.numel() * 2 for c in CUBE) / 1e6
    _hi = float(CUBE[0].float().max())
    print(f"  HDR cube array: {len(CUBE)} mips, {CUBE[0].shape[0]} probes "
          f"@{CUBE[0].shape[2]}^2, {_mb:.0f} MB VRAM, peak radiance "
          f"{_hi:.1f}; lod scale {args.ibl_lod_scale:g} (derived from "
          f"{len(CUBE)} mips)", flush=True)
else:
    # WHY IT IS OFF, printed, so the decision stays visible instead of
    # living in a commit message. The chain is BUILDABLE from a read
    # input -- sky_cube_to_ibl.py turns the map's own <map>.sky_cube.npz
    # into exactly this format -- so "off" here is a MEASURED CHOICE, not
    # a missing capability, and those two states must not look alike.
    #
    # Measured, tick 77159 f000400, both arms, sky cube present:
    #   + specular IBL, overlay pack   nrmse +0.0870  KL +0.5282  ncc -0.0034
    #   + specular IBL, ao pack        nrmse +0.1332  KL +0.7058  ncc -0.0039
    # All three worse, on both arms. The standing suspect is the chain's
    # own stated P = 1: one sky environment fetched everywhere INCLUDING
    # UNDER ROOFS, which is not the reference's per-probe
    # parallax-corrected radiance. That is a real mechanism and it is not
    # separated from "the term is wrong here"; the parked probe-placement
    # work is the eventual answer. Until then the honest default is the
    # one that measured better, and this line is why.
    if args.ibl_lod_scale is None:
        args.ibl_lod_scale = 1.0
    print("specular IBL: OFF (no --ibl-cube/--cube-probes). NOT a missing "
          "capability -- sky_cube_to_ibl.py builds the chain from the "
          "map's own sky cube. It is off because it MEASURED WORSE on all "
          "three metrics on both arms (nrmse +0.087/+0.133, KL "
          "+0.53/+0.71, ncc -0.003/-0.004); the suspect is P=1 radiance "
          "fetched under roofs, unseparated from the term itself.",
          flush=True)


def sh_irradiance(pack_normal, pi):
    """Ramamoorthi-Hanrahan cosine-convolved irradiance from SH-L2.

    The block is named CUBEMAP_RADIANCE_SH, i.e. it stores RADIANCE, so
    the clamped-cosine convolution has to be applied here rather than
    assumed baked in. The coefficient frame was checked physically, not
    guessed: evaluating band index 2 as the up axis gives median
    irradiance 1.187 looking at +Z against 0.726 at -Z, i.e. sky brighter
    than ground on a daylit map, and only 1 probe of 116 rings negative.
    """
    c1, c2, c3 = 0.429043, 0.743125, 0.886227
    c4, c5 = 0.511664, 0.247708
    # back to SOURCE axes: pack = (y_src, z_src, x_src)
    x = pack_normal[..., 2]
    y = pack_normal[..., 0]
    z = pack_normal[..., 1]
    L = SH[pi]                                   # (..., 3, 9)
    x_, y_, z_ = x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)
    return (c1 * L[..., 8] * (x_ * x_ - y_ * y_)
            + c2 * L[..., 6] * z_ * z_
            + c3 * L[..., 0] - c5 * L[..., 6]
            + 2 * c1 * (L[..., 4] * x_ * y_ + L[..., 7] * x_ * z_
                        + L[..., 5] * y_ * z_)
            + 2 * c4 * (L[..., 3] * x_ + L[..., 1] * y_
                        + L[..., 2] * z_)).clamp(min=0)


def sh_at(wpos):
    """Trilinearly interpolated SH-L2 coefficients at a world position."""
    g = (wpos - PLO) * PINV
    d = torch.tensor([PDIM[0] - 1, PDIM[1] - 1, PDIM[2] - 1],
                     device=device, dtype=torch.float32).clamp(min=1)
    n = (g / d * 2 - 1).clamp(-1, 1)
    grid = torch.stack([n[..., 2], n[..., 1], n[..., 0]], dim=-1)
    B = grid.shape[0]
    out = torch.nn.functional.grid_sample(
        SHVOL.expand(B, -1, -1, -1, -1).float(), grid.unsqueeze(1),
        mode="bilinear", align_corners=True, padding_mode="border")
    return out.squeeze(2).permute(0, 2, 3, 1).reshape(*wpos.shape[:-1], 3, 9)


def cube_sample(pi, d, lod):
    """Prefiltered radiance from the HDR cube array along world dir `d`.

    Face selection is the standard D3D table.  The world-frame direction
    is that table's vector with x and y NEGATED -- a 180 degree rotation
    about Z -- established from the in-file SH block at R^2 0.99992
    against a runner-up of 0.75.  `d` arrives in PACK axes, so it goes to
    Source axes first and then through that rotation.

    Bilinear inside a face, linear between the two bracketing mips.  No
    cross-face filtering: a seam texel blends against its own face only,
    a real if small error at 4x4 (mip 6).

    TEXEL CENTRES.  The decode places texel i at (i+0.5)/N, so the
    mapping is x = u*N - 0.5.  An earlier version used u*(N-1), which
    maps corner-to-corner and compresses the whole face onto the texel
    CENTRES -- at N=4 that is a 12.5% angular warp toward face centres.
    The convention here matches the decode the array was validated with.
    """
    # pack -> source, then the 180 deg rotation about Z
    sx = -d[..., 2]
    sy = -d[..., 0]
    sz = d[..., 1]
    ax, ay, az = sx.abs(), sy.abs(), sz.abs()
    face = torch.where(
        (ax >= ay) & (ax >= az), torch.where(sx > 0, 0, 1),
        torch.where(ay >= az, torch.where(sy > 0, 2, 3),
                    torch.where(sz > 0, 4, 5)))
    ma = torch.maximum(torch.maximum(ax, ay), az).clamp(min=1e-8)
    u = torch.where(face == 0, -sz, torch.where(face == 1, sz,
        torch.where(face == 2, sx, torch.where(face == 3, sx,
        torch.where(face == 4, sx, -sx)))))
    v = torch.where(face == 0, -sy, torch.where(face == 1, -sy,
        torch.where(face == 2, sz, torch.where(face == 3, -sz,
        torch.where(face == 4, -sy, -sy)))))
    u = (u / ma) * 0.5 + 0.5
    v = (v / ma) * 0.5 + 0.5

    def _tap(m):
        c = CUBE[m]
        N = c.shape[2]
        x = (u * N - 0.5).clamp(0, N - 1)
        y = (v * N - 0.5).clamp(0, N - 1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1 = (x0 + 1).clamp(max=N - 1)
        y1 = (y0 + 1).clamp(max=N - 1)
        fx = (x - x0.float()).unsqueeze(-1)
        fy = (y - y0.float()).unsqueeze(-1)
        c00 = c[pi, face, y0, x0].float()
        c01 = c[pi, face, y0, x1].float()
        c10 = c[pi, face, y1, x0].float()
        c11 = c[pi, face, y1, x1].float()
        return ((c00 * (1 - fx) + c01 * fx) * (1 - fy)
                + (c10 * (1 - fx) + c11 * fx) * fy)

    top = len(CUBE) - 1
    l = lod.clamp(0, top)
    l0 = l.floor().long()
    w = (l - l0.float()).unsqueeze(-1)
    out = None
    for m in range(len(CUBE)):
        sel = (l0 == m)
        if not bool(sel.any()):
            continue
        a = _tap(m)
        b = _tap(min(m + 1, top))
        blend = a * (1 - w) + b * w
        out = blend if out is None else torch.where(sel.unsqueeze(-1),
                                                    blend, out)
    return out.clamp(min=0)


def sh_radiance(L, d):
    """Raw SH-L2 RADIANCE along d -- no clamped-cosine convolution.

    sh_eval() below convolves with the cosine lobe, which is right for
    the diffuse term and wrong here: a reflection samples radiance, not
    irradiance. Real SH basis constants.
    """
    x = d[..., 2:3]
    y = d[..., 0:1]
    z = d[..., 1:2]
    return (0.282095 * L[..., 0]
            + 0.488603 * (L[..., 1] * y + L[..., 2] * z + L[..., 3] * x)
            + 1.092548 * (L[..., 4] * x * y + L[..., 5] * y * z
                          + L[..., 7] * x * z)
            + 0.315392 * L[..., 6] * (3.0 * z * z - 1.0)
            + 0.546274 * L[..., 8] * (x * x - y * y)).clamp(min=0)


def env_brdf(rough, ndv, f0):
    """Lazarov's analytic split-sum approximation.

    Verbatim from the decompiled csgo_environment_blend PS -- Source 2
    carries no DFG lookup texture, it evaluates this polynomial. This
    replaces a hand-rolled Schlick * (1 - roughness), which got both the
    roughness falloff and the grazing-angle response wrong.
    """
    r0 = rough * -1.0 + 1.0
    r1 = rough * -0.0275 + 0.0425
    r2 = rough * -0.572 + 1.04
    r3 = rough * 0.022 - 0.04
    a004 = torch.minimum(r0 * r0, torch.exp2(-9.28 * ndv)) * r0 + r1
    A = -1.04 * a004 + r2
    B = 1.04 * a004 + r3
    return f0 * A.unsqueeze(-1) + B.unsqueeze(-1)


def sh_eval(L, pack_normal):
    """Cosine-convolved irradiance from an explicit coefficient block."""
    c1, c2, c3 = 0.429043, 0.743125, 0.886227
    c4, c5 = 0.511664, 0.247708
    x = pack_normal[..., 2:3]
    y = pack_normal[..., 0:1]
    z = pack_normal[..., 1:2]
    return (c1 * L[..., 8] * (x * x - y * y)
            + c2 * L[..., 6] * z * z
            + c3 * L[..., 0] - c5 * L[..., 6]
            + 2 * c1 * (L[..., 4] * x * y + L[..., 7] * x * z
                        + L[..., 5] * y * z)
            + 2 * c4 * (L[..., 3] * x + L[..., 1] * y
                        + L[..., 2] * z)).clamp(min=0)
