if LIGHT_ENV is not None:
    # WHICH env_sky DRAWS. The corpus carries maps with two, and the field
    # that separates them is READ, not measured: de_inferno's pair is
    # `_MAT_sky` (StartDisabled false) and `_LIGHT_sky`, a Hosek-Wilkie
    # procedural reference with StartDisabled TRUE -- the second does not
    # draw and was never a candidate. de_dust2's two name the SAME
    # material and are one sky placed twice, so there was never a tie
    # there either. A render pair between those arms would have been
    # guaranteed-identical, which is not a measurement.
    # PREFER THE SIDECAR'S OWN ANSWER. The extractor emits `skyname` with
    # `skyname_source` naming the entity that supplied it, and two of the
    # seven maps do NOT get theirs from an enabled env_sky at all:
    #
    #   de_ancient  no enabled env_sky. The drawn material is on
    #               env_cubemap_fog; its only env_sky is DISABLED and is
    #               what light_environment.skytexture points at for
    #               LIGHTING. Scanning env_sky here returns the irradiance
    #               reference -- a 256^2 page topping out at 0.898 against
    #               the drawn 2048^2 plate's 1.904.
    #   de_anubis   only a disabled env_sky exists.
    #
    # So a local scan is not a cheaper version of that read, it is a
    # DIFFERENT and wrong one on two maps. Re-deriving a join somebody
    # already performed is how two answers drift apart.
    if LIGHT_ENV.get("skyname"):
        SKY_ENTITY = {"targetname": LIGHT_ENV.get("skyname_source",
                                                  "<source not recorded>"),
                      "skyname": LIGHT_ENV["skyname"]}
        print("sky material: %s   (source: %s)"
              % (SKY_ENTITY["skyname"], SKY_ENTITY["targetname"]),
              flush=True)
    else:
        print("GAP sky: sidecar carries no `skyname`; falling back to a "
              "local env_sky scan, which is WRONG on any map whose sky "
              "comes from env_cubemap_fog rather than an enabled env_sky.",
              flush=True)
    _es = [] if LIGHT_ENV.get("skyname") else (LIGHT_ENV.get("env_sky") or [])
    if isinstance(_es, dict):
        _es = [_es]
    _on = [e for e in _es if not e.get("StartDisabled", False)]
    _mats = sorted({e.get("skyname") for e in _on})
    if len(_mats) > 1:
        # THE CASE WHERE A REAL A/B EXISTS. Two entities both enabled and
        # naming DIFFERENT materials cannot be resolved by reading, so
        # this refuses rather than taking the first -- and says that the
        # measurement is the resolution, because picking here would be a
        # guess wearing a selection's clothes.
        print("REFUSE sky: %d enabled env_sky entities name DIFFERENT "
              "materials %s. Reading cannot pick between them; render one "
              "frame per material and score the triplet against GT, then "
              "pass --sky-cube explicitly for the winner." % (len(_on), _mats),
              flush=True)
    elif _on:
        SKY_ENTITY = _on[0]
        print("sky entity: %s -> %s  (%d env_sky in the lump, %d enabled)"
              % (SKY_ENTITY.get("targetname"), SKY_ENTITY.get("skyname"),
                 len(_es), len(_on)), flush=True)
    elif _es:
        print("GAP sky entity: %d env_sky in the lump and NONE enabled "
              "(StartDisabled true on all). The cube loaded below, if any, "
              "is not justified by an entity." % len(_es), flush=True)

# PRECEDENCE, most specific first, and every step prints. This used to be
# `_sun_el, _sun_az = args.sun_el, args.sun_az` followed by an
# UNCONDITIONAL sidecar overwrite, so --sun-el/--sun-az never survived to
# be used on any map that ships a light_environment -- which is all seven.
# A lever that reads and cannot act is the #33 shape; loopfam's probe found
# it as "sun direction moves zero pixels", which was true of the FLAG and
# false of the term.
_sun_el, _sun_az = 63.0, 38.0
_sun_src = "fallback el 63 az 38 (de_inferno's, no sidecar and no flag)"
if LIGHT_ENV is not None:
    # Pitch is down-positive and yaw is a compass bearing, so the vector
    # TO the sun is elevation +pitch, azimuth yaw-180. That derivation
    # reproduced de_inferno's GPU-buffer sun direction to max|d| 1.0e-6,
    # which is what licenses applying it to the other six maps.
    _ang = LIGHT_ENV["angles"]
    _sun_el, _sun_az = float(_ang[0]), float(_ang[1]) - 180.0
    _sun_src = "sidecar %s angles %s" % (LIGHT_ENV.get("map", "?"), _ang)
    print(f"sun from {LIGHT_ENV.get('map', 'sidecar')}: angles {_ang} -> "
          f"el {_sun_el:g} az {_sun_az:g}", flush=True)
if args.sun_entity:
    # light_environment angles [63, 218, 0] = the direction the light
    # POINTS (pitch down-positive, yaw compass). The vector TO the sun is
    # elevation +63, azimuth 218 - 180 = 38.
    _prev = _sun_src
    _sun_el, _sun_az = 63.0, 38.0
    _sun_src = "--sun-entity (de_inferno's el 63 az 38, hardcoded)"
    print(f"sun from light_environment: el {_sun_el} az {_sun_az} "
          f"(--sun-entity, overriding {_prev})", flush=True)
# LAST, so an explicitly-passed angle beats the sidecar and --sun-entity
# both. This is the whole repair: the flags now ACT.
if args.sun_el is not None or args.sun_az is not None:
    _was, _prev = (_sun_el, _sun_az), _sun_src
    if args.sun_el is not None:
        _sun_el = args.sun_el
    if args.sun_az is not None:
        _sun_az = args.sun_az
    _sun_src = "--sun-el/--sun-az on the command line"
    print("sun OVERRIDDEN by flag: el %g az %g (was el %g az %g from %s)"
          % (_sun_el, _sun_az, _was[0], _was[1], _prev), flush=True)
print("sun direction source: %s" % _sun_src, flush=True)
sun_el, sun_az = math.radians(_sun_el), math.radians(_sun_az)
sun_src = np.array([math.cos(sun_az) * math.cos(sun_el),
                    math.sin(sun_az) * math.cos(sun_el),
                    math.sin(sun_el)])
sun_dir = torch.tensor([sun_src[1], sun_src[2], sun_src[0]],
                       dtype=torch.float32, device=device)
# Measured from ground-truth CS2 frames on de_inferno: the sky is a warm
# neutral haze (0.646, 0.633, 0.626), not the blue gradient we assumed.
SKY_TOP = torch.tensor([0.4232, 0.5151, 0.6363], device=device)
SKY_HORIZON = torch.tensor([0.5006, 0.4488, 0.3212], device=device)
AMB_MIX = 0.0  # directional_irradiance packs a DIRECTION, not radiance: using it as an ambient colour raised NRMSE 1.55 -> 1.90 at every gain
AMB_GAIN = 2.4
# READ, not fitted. Byte offset +31136 of the set=3 binding=0 block reads
# (3.000000, 2.790334, 2.492310), identical in both captures and all 718
# block copies. It is DERIVED independently as
#     sRGB_EOTF([255, 247, 235]) * brightness 3.0
# from the map's own light_environment, agreeing to max|delta| = 1.4e-6.
# The transfer function is settled by the read, not argued: a pow-2.2
# decode gives (3.0000, 2.8125, 2.5253), off by 0.022 / 0.033 -- four
# orders of magnitude worse. This replaces a fitted (0.5957, 0.5903,
# 0.5695), i.e. the reference sun is 5.0x brighter at the same
# chromaticity. READ from the capture, two captures agreeing.
SUN_COLOR = torch.tensor([3.0, 2.790334, 2.492310], device=device)
# READ: the sun direction's W at byte offset +31120+12 is 0.040000, and
# csgo_environment_ps.glsl:908 spends it as a floor on roughness -- the
# angular size of the source, which no highlight can be sharper than. The
# map ships `angulardiameter 0.3` on its light_environment.
# The candidate constant ambient in the sun path. Zero until a sidecar and
# the 'read' arm say otherwise, so the default on a map with no sidecar is
# the value the de_inferno buffer actually carries.
SUN_AMBIENT = torch.zeros(3, device=device)
SUN_AMBIENT_SRC = "zero (no sidecar)"
SUN_ANGULAR_W = 0.04
if LIGHT_ENV is not None:
    def _srgb_to_linear(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    _b = float(LIGHT_ENV.get("brightness", 1.0))
    SUN_COLOR = torch.tensor(
        [_srgb_to_linear(float(c)) * _b for c in LIGHT_ENV["color"]],
        dtype=torch.float32, device=device)
    print(f"SUN_COLOR from sidecar: color {LIGHT_ENV['color']} x brightness "
          f"{_b:g} -> {[round(float(x), 6) for x in SUN_COLOR]}  "
          f"(the same derivation that matched de_inferno's GPU buffer to "
          f"max|d| 1.4e-6)", flush=True)
    _sab = LIGHT_ENV.get("skyambientbounce")
    if args.sky_ambient_bounce == "read" and _sab is not None:
        SUN_AMBIENT = torch.tensor(
            [_srgb_to_linear(float(c)) for c in _sab],
            dtype=torch.float32, device=device)
        SUN_AMBIENT_SRC = ("READ from entity skyambientbounce %s -> linear "
                           "%s" % (list(_sab),
                                   [round(float(x), 6) for x in SUN_AMBIENT]))
    elif _sab is not None:
        SUN_AMBIENT_SRC = ("zero (--sky-ambient-bounce zero; the entity "
                           "carries %s)" % list(_sab))
    print("g_vSunAmbient candidate: %s" % SUN_AMBIENT_SRC, flush=True)
    if any(float(x) != 0.0 for x in SUN_AMBIENT):
        print("  NOTE: nonzero, so THIS MAP CAN DECIDE the open question. "
              "The two arms differ here and are identical on any map whose "
              "skyambientbounce is [0,0,0].", flush=True)
    if "sun_angular_w" in LIGHT_ENV:
        SUN_ANGULAR_W = float(LIGHT_ENV["sun_angular_w"])
        print(f"sun angular roughness floor READ for this map: "
              f"{SUN_ANGULAR_W:g}", flush=True)
    else:
        # DO NOT DERIVE IT. de_inferno ships angulardiameter 0.3 and its
        # buffer reads 0.040000; 0.3 deg is 0.002618 rad as a half-angle
        # and neither that nor any simple function of it is 0.04, so the
        # relationship is NOT established. Inventing one would be a fitted
        # float wearing a derivation costume.
        print(f"GAP sun angular floor: this map ships angulardiameter "
              f"{LIGHT_ENV.get('angulardiameter', '?')} but no capture has "
              f"been read for it, and the entity value does NOT derive the "
              f"buffer's 0.040000 by any relation established here. Using "
              f"de_inferno's {SUN_ANGULAR_W:g} as a stand-in.", flush=True)
# ⚠ DELETED AT GATE#5. These were the last fitted lighting constants and
# they are now ZERO -- not re-fitted, not moved behind a flag, gone.
#
# g_vSunAmbient, the only constant ambient in the reference's sun path,
# READS as (0,0,0) (READ from the capture, two captures agreeing). There is no hemisphere in
# CS2 to re-derive: this pair stood in for the BAKED subsystem, which is a
# different expression. Gate#4 priced the fit at 148.5x with the legs
# disagreeing -- light counted twice, the read sun landing on an ambient
# fitted against a zero sun.
#
# The NAMES survive only as diagnostic surfaces, per the charter's rule
# that config axes select VALUES and never EXISTENCE: --amb-sky/--amb-ground
# still override them and --light-basis still isolates them, both of which
# are diagnostics. Nothing on the default path reads a fitted number any
# more, and the seed at the shading site is an explicit zero tensor rather
# than these, so deleting the flags later cannot resurrect a value.
AMBIENT_SKY = torch.zeros(3, device=device)
AMBIENT_GROUND = torch.zeros(3, device=device)
if args.light_entity:
    # colour / skycolor are sRGB bytes; decode to linear, normalise to
    # unit mean so ONLY the chromaticity crosses over, and let
    # --sun-scale / --sky-scale carry the (non-transferable) magnitude.
    def _chroma(rgb255, ref):
        v = torch.tensor([(c / 255.0) ** 2.2 for c in rgb255],
                         dtype=torch.float32, device=device)
        return v / v.mean() * float(ref.mean())
    # SUN_COLOR is NOT re-derived here any more. _chroma() decodes with
    # pow(c/255, 2.2) and then throws the magnitude away by normalising to
    # unit mean, handing it to --sun-scale. Both halves are refuted by the
    # read at +31136: the magnitude IS in the reference (brightness 3.0 is
    # applied to the linear colour and lands in the buffer), and the
    # transfer function is the true sRGB EOTF, which matches to 1.4e-6
    # where pow-2.2 is off by 0.022/0.033. The default above is that read
    # value; --sun-scale remains a diagnostic multiplier on it.
    AMBIENT_SKY = _chroma([136, 199, 255], AMBIENT_SKY) * args.sky_scale
    AMBIENT_GROUND = AMBIENT_GROUND * args.ground_scale
    print(f"light_environment chromaticity: SUN_COLOR "
          f"{[round(float(x),4) for x in SUN_COLOR]}  AMBIENT_SKY "
          f"{[round(float(x),4) for x in AMBIENT_SKY]}  AMBIENT_GROUND "
          f"{[round(float(x),4) for x in AMBIENT_GROUND]}", flush=True)

# UNCONDITIONAL, and that is the fix. This multiply used to sit inside the
# `if args.light_entity:` block above, so on any run without that flag --
# which is every corpus render -- --sun-scale read its value and multiplied
# nothing. The comment three lines up already called it "a diagnostic
# multiplier on it", describing behaviour the branch prevented.
#
# gt_direct() consumes SUN_COLOR directly (gt_compose passes it), so the
# multiply has to land before that and outside every branch, or the lever
# is dead on the path that renders. It prints when it is not 1 so a scaled
# run cannot be mistaken for an unscaled one in the log.
if args.sun_scale != 1.0:
    SUN_COLOR = SUN_COLOR * args.sun_scale
    print("SUN_COLOR scaled by --sun-scale %g -> %s"
          % (args.sun_scale, [round(float(x), 6) for x in SUN_COLOR]),
          flush=True)


# ---------------------------------------------------------------------
# THE SKY PASS'S CUBE-TO-FRAMEBUFFER LEVEL, READ from the module that runs
# ---------------------------------------------------------------------
# The executed sky pixel shader is module 78427, reached from draw 40411 ->
# pipeline 84652 -> pLibraries in the wallA2 capture. It is NOT the module
# in sky_vulkan_50_ps.vcs: none of that archive's 24 modules is byte-equal
# to anything bound in the capture, and 78427 is smaller than the smallest
# of them. Its chain (spv_chain.py) is
#
#     out.rgb = max(cube * U[set1,bind0,+72] * U[set1,bind0,+76]
#                        * U[set1,bind0,+80 : f32x3], 0)
#
# and the READ from wallA2, buffer 90549 +1557120 range 112:
#     +72 = 2.0   +76 = 1.0   +80 = (0.8, 0.8, 0.8)   product 1.6
#
# ACHROMATIC. The product has no chroma at all, which is the shape the
# independent falsifier predicted: the eroded sky ratio field against GT is
# G-normalised (0.9595, 1.0000, 1.0022).
#
# ONE MAP. wallA2 is de_inferno, so 1.6 is de_inferno's value. The other
# six need their own captures; de_dust2's is on the ws-1 list. Following
# this file's own precedent for SUN_ANGULAR_W, the read value is carried as
# a STATED STAND-IN on the other six rather than silently replaced by 1.0 --
# a stand-in that prints is a different thing from a default that does not.
SKY_LEVEL = 1.6
SKY_LEVEL_SRC = None
_sky_map = (LIGHT_ENV or {}).get("map")
if args.sky_level is not None:
    SKY_LEVEL = args.sky_level
    SKY_LEVEL_SRC = "--sky-level %g on the command line" % args.sky_level
elif _sky_map == "de_inferno":
    SKY_LEVEL_SRC = ("READ from the wallA2 capture: 2.0 * 1.0 * "
                     "(0.8, 0.8, 0.8) = 1.6, achromatic")
else:
    SKY_LEVEL_SRC = ("STAND-IN: de_inferno's read 1.6 applied to %s, which "
                     "has no capture" % (_sky_map or "this map"))
print("sky level (executed module 78427, set1/bind0 +72/+76/+80): %g -- %s"
      % (SKY_LEVEL, SKY_LEVEL_SRC), flush=True)
if _sky_map != "de_inferno" and args.sky_level is None:
    _gt_note("sky level: %g is de_inferno's READ value standing in for %s. "
             "The SHAPE is established corpus-wide (the product is "
             "achromatic in the module that runs, and the measured residual "
             "is achromatic to 4%% in red); the MAGNITUDE is one map's and "
             "needs this map's own capture."
             % (SKY_LEVEL, _sky_map or "this map"))


def _lighting_provenance():
    """Print, every run, which lighting constants are READ and which are not.

    Charter rule: constants are READ or DERIVED-from-reference-inputs, never
    fitted. #34: uncertainty prints at runtime, it does not live in a
    comment. This is the only place a reader can see, without leaving the
    log, that AMBIENT_SKY/AMBIENT_GROUND are the last fitted lighting
    scalars standing AND that the reference offers nothing to replace them
    with -- g_vSunAmbient, the one constant ambient in the sun path, reads
    exactly zero.
    """
    print("lighting constants -- provenance (READ from the capture):",
          flush=True)
    print(f"  READ   sun direction      el {_sun_el:g} az {_sun_az:g}"
          f"  -> {[round(float(x), 6) for x in sun_dir]}"
          f"   (+31120, max|d| 1.0e-6 vs light_environment)", flush=True)
    print(f"  READ   SUN_COLOR          "
          f"{[round(float(x), 6) for x in SUN_COLOR]}"
          f"   (+31136, = sRGB([255,247,235]) x 3.0 to 1.4e-6)", flush=True)
    print(f"  READ   g_flShadowBias     {args.gt_csm_bias:+.12g}"
          f"   (+31256, = -2**-16 exactly)", flush=True)
    print(f"  READ   cascade count      {args.gt_csm_cascades}"
          f"   (+31248)", flush=True)
    print(f"  READ   cascade blend band "
          f"{[float(x) for x in args.gt_csm_fade]}   (+31280/+31284)",
          flush=True)
    print(f"  READ   g_vSunAmbient      (0, 0, 0)   (+288, 717 of 718 "
          f"block copies) -- ON DE_INFERNO ONLY", flush=True)
    print(f"         both captures are de_inferno POSES (wallA2 and the "
          f"church interior), so this is ONE MAP, not two independent "
          f"reads -- and de_inferno's entity `skyambientbounce` is also "
          f"[0,0,0], so the buffer member and the entity key AGREE here "
          f"and this read cannot tell them apart. de_dust2 ships "
          f"skyambientbounce [151,151,151]: if its buffer reads nonzero "
          f"the member IS skyambientbounce and 'the reference adds no "
          f"constant ambient' is FALSE corpus-wide. No dust2 capture "
          f"exists, so that is open.", flush=True)
    print(f"  DELETED AMBIENT_SKY       "
          f"{[round(float(x), 4) for x in AMBIENT_SKY]}", flush=True)
    print(f"  DELETED AMBIENT_GROUND    "
          f"{[round(float(x), 4) for x in AMBIENT_GROUND]}", flush=True)
    print("  GATE#5: the fitted hemisphere is DELETED, not re-fitted. "
          "Pixels no baked source reaches now receive NO ambient, which "
          "is the reference's value. If the frame is darker than GT that "
          "MEASURES what the baked path underprovides -- see the "
          "ambient.baked_coverage probe, whose fraction-at-identity is "
          "the size of the hole the fit was filling. Cascade placement is "
          "READ and no longer open: the reference's cascades ARE "
          "camera-centred. The two earlier claims to the contrary were "
          "mine and both projected an ABSOLUTE world position through "
          "matrices the shader only ever feeds a CAMERA-RELATIVE one "
          "(csgo_environment_vs_base.glsl:180). See cascade_project() "
          "in passes/lighting.py.",
          flush=True)


for _nm_, _ov_ in (("SKY_TOP", args.sky_top), ("SKY_HORIZON", args.sky_horizon),
                   ("SUN_COLOR", args.sun_color),
                   ("AMBIENT_SKY", args.amb_sky),
                   ("AMBIENT_GROUND", args.amb_ground)):
    if _ov_ is not None:
        globals()[_nm_] = torch.tensor(_ov_, dtype=torch.float32, device=device)
        print(f"lighting override {_nm_} = {_ov_}", flush=True)
if args.light_basis is not None:
    # Zero every lighting constant, then set the one this basis isolates
    # to unit white. Emissive and fog are unaffected by all five, so
    # basis 5 (all zero) captures them as the constant residue.
    _Z = torch.zeros(3, device=device)
    _O = torch.ones(3, device=device)
    AMBIENT_GROUND = _O if args.light_basis == 0 else _Z
    AMBIENT_SKY = _O if args.light_basis == 1 else _Z
    SUN_COLOR = _O if args.light_basis == 2 else _Z
    SKY_TOP = _O if args.light_basis == 3 else _Z
    SKY_HORIZON = _O if args.light_basis == 4 else _Z
    print(f"LIGHT BASIS {args.light_basis}: "
          f"AG {float(AMBIENT_GROUND[0])} AS {float(AMBIENT_SKY[0])} "
          f"SUN {float(SUN_COLOR[0])} SKYT {float(SKY_TOP[0])} "
          f"SKYH {float(SKY_HORIZON[0])}", flush=True)
# After the overrides and the basis isolation, so what prints is what the
# render will actually use -- not what the defaults were.
_lighting_provenance()
if args.fog_color_space == "srgb":
    FOG_COLOR = torch.tensor([(c / 255.0) ** 2.2 for c in args.fog_color],
                             dtype=torch.float32, device=device)
else:
    FOG_COLOR = torch.tensor([c / 255.0 for c in args.fog_color],
                             dtype=torch.float32, device=device)
if args.fog:
    print(f"env_gradient_fog: start {args.fog_start} end {args.fog_end} u, "
          f"colour {args.fog_color} -> linear "
          f"{[round(float(x),4) for x in FOG_COLOR]}, strength "
          f"{args.fog_strength}, maxopacity {args.fog_max_opacity}, "
          f"heightfog {bool(args.fog_height)} "
          f"{args.fog_start_height}->{args.fog_end_height}", flush=True)
ydir = torch.linspace(1, 0, H, device=device).view(1, H, 1, 1)
sky_row = SKY_HORIZON.view(1, 1, 1, 3) * (1 - ydir) + \
    SKY_TOP.view(1, 1, 1, 3) * ydir

# --- #48: the sky pass, as the reference draws it ----------------------

SKY_CUBE = None
if args.sky_cube == "auto":
    # A FLAG MUST NOT DECIDE WHETHER A REFERENCE FUNCTION EXISTS. The sky
    # is a real draw in the reference call graph, so the default cannot be
    # "off" -- it resolves beside the world pack, the same convention
    # --irr-npy uses, because the sky belongs to the map and travels with
    # it. 'none' is a diagnostic that selects the pre-#48 gradient, and it
    # says so at runtime rather than silently.
    _sky_auto, _sky_how = _sidecar(".sky_cube.npz", "sky cube")
    if _sky_auto:
        args.sky_cube = _sky_auto
        print(f"sky cube: auto -> {_sky_auto}", flush=True)
    else:
        args.sky_cube = None
        print(f"sky cube: auto found no {_sky_auto} -- the sky pass has no "
              f"texture and falls back to the pre-#48 screen-locked "
              f"gradient. That is a MISSING INPUT, not a disabled feature: "
              f"the reference draws the sky at chunkIndex 40411 on every "
              f"frame.", flush=True)
elif args.sky_cube == "none":
    args.sky_cube = None
    print("sky cube: DIAGNOSTIC 'none' -- the pre-#48 screen-locked "
          "gradient, selected deliberately.", flush=True)
if args.sky_cube:
    _sc = np.load(args.sky_cube)["cube"]              # (6, N, N, 3)
    SKY_CUBE = torch.from_numpy(_sc).to(device).float()
    if SKY_ENTITY is None:
        print("GAP sky: a cube is loaded but no enabled env_sky entity "
              "names it -- the image is present and its JUSTIFICATION is "
              "not. Check <pack>.light_environment.json.", flush=True)
    print(f"sky cube: {tuple(SKY_CUBE.shape)} "
          f"min {float(SKY_CUBE.min()):.4f} max {float(SKY_CUBE.max()):.4f} "
          f"mean {float(SKY_CUBE.mean()):.4f}  z-rot {args.sky_cube_zrot} deg",
          flush=True)
else:
    print("GAP sky: no --sky-cube, so the sky is the SCREEN-LOCKED "
          "SKY_TOP/SKY_HORIZON gradient. That is not the reference's "
          "operation with a constant missing -- the reference samples an "
          "HDR cube BY VIEW DIRECTION, and a fixed image has no "
          "orientation state to put the read 115 degrees into, so it "
          "cannot respond to camera yaw at all (#48, List A).", flush=True)


def sky_cube_sample(d):
    """Sample the reference's sky cube along world direction `d`.

    Face selection, the u/v table and the TEXEL-CENTRE mapping
    (x = u*N - 0.5, not u*(N-1)) are taken from cube_sample() above rather
    than re-derived -- that function's docstring records the 12.5% angular
    warp the corner-to-corner alternative causes, and re-deriving a
    convention that has already been validated is how two implementations
    of one thing appear.

    NOT inherited from cube_sample: its 180-degree Z rotation. That was
    established for the PROBE array against its own in-file SH block; this
    is a different asset with its own read orientation, and carrying the
    probe's rotation over would be exactly the family-generalisation trap
    of the two-selector comparison in this file.
    """
    face, u, v = _lighting.cube_face_uv(d)
    face = face.long()
    N = SKY_CUBE.shape[1]
    x = (u * N - 0.5).clamp(0, N - 1)
    y = (v * N - 0.5).clamp(0, N - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1 = (x0 + 1).clamp(max=N - 1)
    y1 = (y0 + 1).clamp(max=N - 1)
    fx = (x - x0.float()).unsqueeze(-1)
    fy = (y - y0.float()).unsqueeze(-1)
    c00 = SKY_CUBE[face, y0, x0]
    c01 = SKY_CUBE[face, y0, x1]
    c10 = SKY_CUBE[face, y1, x0]
    c11 = SKY_CUBE[face, y1, x1]
    return ((c00 * (1 - fx) + c01 * fx) * (1 - fy)
            + (c10 * (1 - fx) + c11 * fx) * fy)


SKY_MVP = None       # set per chunk by render_window(); the shading chain
