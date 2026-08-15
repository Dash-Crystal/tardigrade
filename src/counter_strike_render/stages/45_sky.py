                     # it feeds does not receive a camera, and threading one
                     # through a 20-parameter signature to reach one call
                     # site is a worse change than a named module slot.


def sky_pass(mvp_b, Hs, Ws):
    """#48's draw: a far-plane pass sampling the cube by view direction.

    The reference's whole uniform state for chunkIndex 40411 is ONE
    matrix -- a pure Z-rotation of 115 degrees, unit scale, NO
    translation. The absence of translation is the semantic content: the
    sky does not move with the camera, so the operation is a DIRECTION
    lookup and nothing else.

    Rays come from inverting the renderer's own mvp and unprojecting the
    near and far planes, rather than rebuilding a camera basis. The
    column-vector convention is READ off this file's own use sites
    (`einsum("bij,nj->bni", mvp, ...)`), not assumed -- getting a
    convention from the code that already works beats getting it from
    which one makes the picture look right.

    THE ONE INFERRED STEP, stated rather than buried: the read rotation is
    about SOURCE Z (up). This renderer's pack basis is
    glTF = (src_y, src_z, src_x) (line 18), so source Z is pack Y and the
    rotation is applied about pack Y here. That mapping is documented and
    ray-cast validated at the top of this file; it is not re-derived, but
    it IS a step between the read constant and this code, and if the sky
    comes out rotated it is the first thing to check.
    """
    inv = torch.linalg.inv(mvp_b.float())
    ny = torch.linspace(1.0, -1.0, Hs, device=mvp_b.device)
    nx = torch.linspace(-1.0, 1.0, Ws, device=mvp_b.device)
    gy, gx = torch.meshgrid(ny, nx, indexing="ij")
    one = torch.ones_like(gx)
    near = torch.stack([gx, gy, torch.zeros_like(gx), one], -1)
    far = torch.stack([gx, gy, one, one], -1)
    p0 = torch.einsum("bij,hwj->bhwi", inv, near)
    p1 = torch.einsum("bij,hwj->bhwi", inv, far)
    p0 = p0[..., :3] / p0[..., 3:4].clamp(min=1e-12)
    p1 = p1[..., :3] / p1[..., 3:4].clamp(min=1e-12)
    d = torch.nn.functional.normalize(p1 - p0, dim=-1, eps=1e-12)
    # Through passes/lighting.py so the registry scores the expression the
    # renderer runs. A whole-pass term would have compared this function
    # against itself on synthetic inputs; these two have reference lines.
    d = _lighting.sky_zrot(d, args.sky_cube_zrot)
    return sky_cube_sample(d) * SKY_LEVEL

# --- engine per-view render targets: setup -----------------------------
# projection() is called with its defaults, so these ARE the frustum the
# MVPs were built from. Kept as named constants because the offset-88
# producer and the offset-80 gather both need to move between the
# gl_FragCoord.z convention and metres.
PROJ_NEAR, PROJ_FAR = 0.05, 800.0
# _5037._m15 (offset 264) and _5037._m14 (offset 256): fragcoord -> the
# low-res target's texel index, and that index -> UV. glsl:1211 reads
#   uv = floor(gl_FragCoord.xy * _m15) * _m14 + _m14 * 0.5
# so _m15 = 1/scale and _m14 = 1/(low-res dimension) exactly.
DIROCC_S = max(1, int(args.dirocc_scale))
DIROCC_W = (W + DIROCC_S - 1) // DIROCC_S
DIROCC_H = (H + DIROCC_S - 1) // DIROCC_S
RTV_M15 = 1.0 / DIROCC_S
RTV_M14 = torch.tensor([1.0 / DIROCC_W, 1.0 / DIROCC_H], device=device)
SS_S = max(1, int(args.ss_shadow_scale))
SS_W = (W + SS_S - 1) // SS_S
SS_H = (H + SS_S - 1) // SS_S
# _4459._m3.xy (set 1 binding 1, offset 352): the inverse viewport the
# offset-88 read multiplies gl_FragCoord.xy by. glsl:1322.
RTV_INVVP = torch.tensor([1.0 / W, 1.0 / H], device=device)

# The four engine ambient-basis directions, _5538._m1._m0[0..3].xyz.
# glsl:1238 dots them against the world NORMAL and glsl:1237 dots
# normalize(vec3(b.xy, 0.25)) against -N. The 0.25 is a literal, so the
# bytecode fixes the CONSTRUCTION of the tilted vector but not the
# directions themselves -- those live in an engine constant buffer this
# extraction does not reach. Source world space is Z-up; this renderer is
# glTF (src_y, src_z, src_x), so a Source (sx, sy, sz) is (sy, sz, sx).
_az = [math.radians(float(a)) for a in args.dirocc_basis.split(",")]
if len(_az) != 4:
    raise SystemExit("--dirocc-basis needs exactly four azimuths")
_bsrc = np.stack([np.array([math.cos(a), math.sin(a), args.dirocc_tilt])
                  for a in _az]).astype(np.float32)
_bsrc /= np.linalg.norm(_bsrc, axis=1, keepdims=True)
# world-space basis, our axis order
DIROCC_B = torch.tensor(_bsrc[:, [1, 2, 0]], device=device)          # (4,3)
# normalize(vec3(b.xy, 0.25)) -- b.xy is SOURCE xy, i.e. our (z, x)
_btilt = np.stack([np.array([b[0], b[1], 0.25]) for b in _bsrc]).astype(np.float32)
_btilt /= np.linalg.norm(_btilt, axis=1, keepdims=True)
DIROCC_BT = torch.tensor(_btilt[:, [1, 2, 0]], device=device)        # (4,3)
DIROCC_VW = torch.tensor([float(v) for v in
                          args.dirocc_view_weights.split(",")], device=device)
if DIROCC_VW.numel() != 4:
    raise SystemExit("--dirocc-view-weights needs exactly four values")

# _4459._m0 (set 1 binding 1, offset 300): the time the two wind-scrolled
# ripple layers are offset by, glsl:1072/1081. Taken from the camera
# path's own tick so it is deterministic per frame and shares no state
# with wall-clock.
# ONE engine clock. The reference exposes a single time uniform (draw-data
# buffer offset 300, confirmed identical in all three vertex listings), read
# by BOTH the weather ripple scroll (glsl:1072/1081) and the texture/UV
# animation (csgo_complex_vs_max.glsl:260,:279). Two agents each derived a
# clock; merging them as two globals would give one reference value two
# different rates. TIME_S is that clock; --weather-tickrate is retained as a
# scroll-RATE scale applied at the consumer, not as a second time base.
TIMES = TIME_S
WEATHER_TINT = torch.tensor([float(v) for v in args.weather_tint.split(",")],
                            device=device)

RIPPLE = None
if args.weather:
    if args.weather_ripple:
        import imageio.v2 as _iio
        _r = _iio.imread(args.weather_ripple).astype(np.float32) / 255.0
        if _r.ndim == 2:
            _r = np.stack([_r] * 4, axis=-1)
        if _r.shape[-1] == 3:
            _r = np.concatenate([_r, np.ones_like(_r[..., :1])], axis=-1)
        RIPPLE = torch.tensor(_r[..., :4], device=device)
    else:
        # A STAND-IN, and it is one on purpose: the offset-116 texture is
        # engine-supplied and none of the six shaders carries its content.
        # What the shader tells us about it is the LAYOUT, and that is what
        # this reproduces -- .xy a tangent-space normal in 0..1 (glsl:1073
        # and :1082 both decode xy*2-1), .z a ring/height term summed into
        # the wetness alpha at glsl:1090, .w a coverage noise mixed against
        # the literal 0.5 at glsl:1044. Value noise on a fixed seed, so the
        # frame is a function of the arguments and nothing else.
        _g = np.random.default_rng(0x116)
        _N = 256
        _lo = _g.random((16, 16, 4)).astype(np.float32)
        _yy, _xx = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
        _fy, _fx = _yy / (_N / 16.0), _xx / (_N / 16.0)
        _iy, _ix = np.floor(_fy).astype(int) % 16, np.floor(_fx).astype(int) % 16
        _ty = (_fy - np.floor(_fy))[..., None]
        _tx = (_fx - np.floor(_fx))[..., None]
        _ty = _ty * _ty * (3 - 2 * _ty)
        _tx = _tx * _tx * (3 - 2 * _tx)
        _a = _lo[_iy, _ix] * (1 - _tx) + _lo[_iy, (_ix + 1) % 16] * _tx
        _b = _lo[(_iy + 1) % 16, _ix] * (1 - _tx) \
            + _lo[(_iy + 1) % 16, (_ix + 1) % 16] * _tx
        _v = _a * (1 - _ty) + _b * _ty
        # .xy centred on 0.5 so the decoded normal is flat on average
        _v[..., 0:2] = 0.5 + (_v[..., 0:2] - 0.5) * 0.5
        RIPPLE = torch.tensor(_v, device=device)
    print(f"per-view offset 116 (weather ripple): {tuple(RIPPLE.shape)} "
          f"{'from ' + args.weather_ripple if args.weather_ripple else 'PROCEDURAL STAND-IN'}",
          flush=True)
# --- g_tWaterEffectsMap, set 1 / binding 64 ------------------------------
# csgo_water_fancy's S_INTERACTION_EFFECTS axis reads this at five
# world-projected taps (glsl :354, :450, :455, :468, :470) and one more
# inside the caustic block (:788). It is an ENGINE per-view target in the
# same class as offsets 76 / 80 / 88 / 116: no material binds it -- the
# fountain's .vmat_c has no such TextureParam -- and no .vcs carries its
# content. What the shader fixes is the LAYOUT, and that is what this
# reproduces:
#   .x  a signed ripple height, consumed as `t.x - 0.5` and differenced
#       across four neighbours into a tangent normal (:475)
#   .y  foam,  .z  silt, both decoded `clamp((t - 0.5) * 2, 0, 1)` (:355)
#   .w  unread
# ALWAYS BUILT, never None. A None here would make S_INTERACTION_EFFECTS
# -- an axis this map's one de_inferno material DOES select -- a silent
# no-op that still reports as running, which is the exact failure mode
# --ibl-lod-scale had.
WEFFECT = None
if True:
    if args.water_effects_map:
        import imageio.v2 as _iio2
        _w = _iio2.imread(args.water_effects_map).astype(np.float32) / 255.0
        if _w.ndim == 2:
            _w = np.stack([_w] * 4, axis=-1)
        if _w.shape[-1] == 3:
            _w = np.concatenate([_w, np.ones_like(_w[..., :1])], axis=-1)
        WEFFECT = torch.tensor(_w[..., :4], device=device,
                               dtype=torch.float32)
    else:
        _gw = np.random.default_rng(0x64)
        _NW = 256
        _low = _gw.random((32, 32, 4)).astype(np.float32)
        _yw, _xw = np.meshgrid(np.arange(_NW), np.arange(_NW), indexing="ij")
        _fyw, _fxw = _yw / (_NW / 32.0), _xw / (_NW / 32.0)
        _iyw = np.floor(_fyw).astype(int) % 32
        _ixw = np.floor(_fxw).astype(int) % 32
        _tyw = (_fyw - np.floor(_fyw))[..., None]
        _txw = (_fxw - np.floor(_fxw))[..., None]
        _tyw = _tyw * _tyw * (3 - 2 * _tyw)
        _txw = _txw * _txw * (3 - 2 * _txw)
        _aw = _low[_iyw, _ixw] * (1 - _txw) \
            + _low[_iyw, (_ixw + 1) % 32] * _txw
        _bw = _low[(_iyw + 1) % 32, _ixw] * (1 - _txw) \
            + _low[(_iyw + 1) % 32, (_ixw + 1) % 32] * _txw
        _vw = _aw * (1 - _tyw) + _bw * _tyw
        # A disturbance field is mostly CALM: the ripple channel is
        # centred on 0.5 (zero displacement) and the foam/silt channels
        # sit just under 0.5 so most of the surface decodes to zero and
        # only the disturbed patches clear the threshold.
        _vw[..., 0] = 0.5 + (_vw[..., 0] - 0.5) * 0.6
        _vw[..., 1:3] = 0.35 + _vw[..., 1:3] * 0.45
        # float32 explicitly: the numpy interpolation above promotes to
        # float64 through np.arange, and a float64 tensor reaching
        # torch.lerp raises rather than silently costing precision.
        WEFFECT = torch.tensor(_vw, device=device,
                               dtype=torch.float32)
    if args.water_fancy or args.water_selftest:
        print(f"g_tWaterEffectsMap (set 1/binding 64): "
              f"{tuple(WEFFECT.shape)} "
              + ("from " + args.water_effects_map
                 if args.water_effects_map else "PROCEDURAL STAND-IN")
              + f"; ripple .x in [{float(WEFFECT[..., 0].min()):.3f}, "
              f"{float(WEFFECT[..., 0].max()):.3f}], decoded foam .y in "
              f"[{float(((WEFFECT[..., 1] - 0.5) * 2).clamp(0, 1).min()):.3f}"
              f", {float(((WEFFECT[..., 1] - 0.5) * 2).clamp(0, 1).max()):.3f}]",
              flush=True)


# per-target executed-read counters for --rt-reach; [read px, shaded px]
RT_REACH = {k: [0.0, 0.0] for k in ("off76_dirocc", "off80_depthgather",
                                    "off88_ssshadow", "off116_ripple_a",
                                    "off116_ripple_b")}
if args.dirocc or args.ss_shadow or args.weather:
    print(f"engine per-view render targets (set 1 binding 3): "
          f"off76/80 dirocc {'on' if args.dirocc else 'OFF'} "
          f"@{DIROCC_W}x{DIROCC_H} (scale {DIROCC_S}), "
          f"off88 ss-shadow {'on' if args.ss_shadow else 'OFF'} "
          f"@{SS_W}x{SS_H}, "
          f"off116 weather {'on' if args.weather else 'OFF'} "
          f"(wetness {args.weather_wetness}, rain {args.weather_rain}, "
          f"wind {args.weather_wind})", flush=True)
    if args.weather and args.weather_wetness <= 0.0:
        print("  NOTE off116: per-view wetness is 0, so the shader's own "
              "gate skips the glsl:1072/1081 ripple-layer reads and keeps "
              "the glsl:1014 read. Use --rt-reach to measure it.",
              flush=True)

# --- volumetric smoke + MBOIT resolve: the pass's own resources --------
# STRING FLAG, never bare truthiness: `if args.smoke:` is TRUE for "off".
SMOKE_ON = args.smoke != "off"
SMOKE_TEX = None
SMOKE_VOLS = {}
SMOKE_EVENTS = []
SMOKE_EVENT_META = {}
# The hook the viewmodel pass fills with its own colour/depth layer so smoke
# can composite against the weapon instead of the world. NOTHING FILLS IT YET
# -- the viewmodel layer producer is task #62's lane -- so both .get() calls
# below return None, and smoke_mboit's viewmodel arm answers that by appending
# NOTE viewmodel_pass_no_layer and returning None (smoke_mboit.py:2139). That
# is deliberate: the arm reports itself as NOT RUN rather than quietly
# compositing against the world and reading as if a viewmodel had been there.
# Referenced at the smoke_pass call below; it was read there and defined
# nowhere, which is a NameError for every --smoke run that reaches it.
SMOKE_VIEWMODEL = {}
# TICK_OF_FRAME is defined and filled far above, beside the scope state --
# rebinding it here would discard every tick the camera loader had joined.
if SMOKE_ON and args.smoke == "demo":
    if not args.smoke_grenades:
        raise SystemExit(
            "REFUSED: --smoke demo needs --smoke-grenades <json>. There is "
            "deliberately no fall back to synth: synthesised smoke under a "
            "demo flag renders a cloud that is not where the demo put it, "
            "and nothing in the frame says so.")
    SMOKE_EVENTS, SMOKE_EVENT_META = smoke_mboit.load_demo_grenades(
        args.smoke_grenades)
    print("smoke: %d smokegrenade_detonate from %s (%s), %d unmatched "
          "expire, mean lifetime %.0f ticks"
          % (SMOKE_EVENT_META["n"], SMOKE_EVENT_META.get("demo"),
             SMOKE_EVENT_META.get("map_name"),
             SMOKE_EVENT_META["unmatched"], SMOKE_EVENT_META["mean_life"]),
          flush=True)
if SMOKE_ON:
    if args.smoke_side:
        _st = torch.load(args.smoke_side, map_location="cpu",
                         weights_only=False)
        SMOKE_TEX = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in _st.items()}
        _need = ("noise3", "shape3", "ramp", "jitter", "scope", "envcube")
        _miss = [k for k in _need if k not in SMOKE_TEX]
        if _miss:
            raise SystemExit(
                f"--smoke-side {args.smoke_side} is missing {_miss}; "
                "rebuild it with fast_pack_fam.py --out-smoke")
        print(f"smoke side table {args.smoke_side}: "
              f"noise3 {tuple(SMOKE_TEX['noise3'].shape)}, "
              f"shape3 {tuple(SMOKE_TEX['shape3'].shape)}, "
              f"ramp {tuple(SMOKE_TEX['ramp'].shape)}", flush=True)
    else:
        # Synthesised, and it says so. The world pack carries no 3D
        # volumes at all, so without this the two sampler3D sites the
        # march is built around would have nothing to read -- which is a
        # missing feature, not a default worth having.
        SMOKE_TEX = smoke_mboit.synth_textures(device)
        print("smoke data dependencies: SYNTHESISED (no --smoke-side). "
              "The 3D noise and shape volumes, the colour ramp, the "
              "jitter tile, the scope mask and the ambient cube are "
              "generated on a fixed seed; build the real ones with "
              "fast_pack_fam.py --out-smoke.", flush=True)
    if args.smoke == "pack" and not args.smoke_side:
        raise SystemExit("--smoke pack needs --smoke-side")
    print(f"smoke: D_SMOKE_INSTANCES={args.smoke_instances} (measured "
          f"range 1..6), D_SMOKE_QUALITY={args.smoke_quality}, "
          f"D_SMOKE_FULLRES={args.smoke_fullres}, march steps "
          f"{args.smoke_march_steps if args.smoke_quality == '1' else args.smoke_march_steps_lowq}"
          f" (a UNIFORM of ours -- the enumeration gives no ceiling for "
          f"this loop), resolve={args.mboit_resolve}", flush=True)
if args.mboit_resolve != "off" and not MBOIT_N:
    raise SystemExit(
        "--mboit-resolve consumes the buffers D_MBOIT_PASS1/PASS2 write, "
        "so it needs --mboit 4 or --mboit 6. It is the RESOLVE half of "
        "the MBOIT structure, not a second MBOIT.")


def segment_gather(starts, lengths):
    total = int(lengths.sum())
    return torch.repeat_interleave(
        starts - lengths.cumsum(0) + lengths, lengths
    ) + torch.arange(total, device=device)


def cluster_visibility(mvp_batch):
    planes = torch.stack([
        mvp_batch[:, 3] + mvp_batch[:, 0], mvp_batch[:, 3] - mvp_batch[:, 0],
        mvp_batch[:, 3] + mvp_batch[:, 1], mvp_batch[:, 3] - mvp_batch[:, 1],
        mvp_batch[:, 3] + mvp_batch[:, 2], mvp_batch[:, 3] - mvp_batch[:, 2],
    ], dim=1)
    norm = planes[..., :3].norm(dim=-1).clamp(min=1e-9)
    d = torch.einsum("bpk,ck->bpc", planes, centers_h) / norm[..., None]
    return ~(d < -radii[None, None, :]).any(dim=1)          # (B, C)


def occlusion_refine(mvp, visible):
    """visible (B, C) -> mask with Hi-Z-occluded clusters removed."""
    B = mvp.shape[0]
    clip = torch.einsum("bij,nj->bni", mvp,
                        vertices_h[occ_used]).contiguous()
    rast, _ = dr.rasterize(ctx, clip, occluder_faces, (HIZ_H, HIZ_W))
    depth = torch.where(rast[..., 3] > 0, rast[..., 2],
                        torch.ones_like(rast[..., 2]))       # (B, h, w)
    # max-mip pyramid
    pyramid = [depth.unsqueeze(1)]
    while pyramid[-1].shape[-1] > 1:
        pyramid.append(torch.nn.functional.max_pool2d(
            pyramid[-1], 2, ceil_mode=True))
    # project cluster corners
    cc = torch.einsum("bij,ckj->bcki", mvp, corners_h)       # (B, C, 8, 4)
    w = cc[..., 3]
    safe = (w > 1e-4).all(dim=2)                             # (B, C)
    ndc = cc[..., :3] / w.clamp(min=1e-4).unsqueeze(-1)
    x = (ndc[..., 0] * 0.5 + 0.5) * (HIZ_W - 1)
    y = (ndc[..., 1] * 0.5 + 0.5) * (HIZ_H - 1)
    x0 = x.min(dim=2).values.clamp(0, HIZ_W - 1)
    x1 = x.max(dim=2).values.clamp(0, HIZ_W - 1)
    y0 = y.min(dim=2).values.clamp(0, HIZ_H - 1)
    y1 = y.max(dim=2).values.clamp(0, HIZ_H - 1)
    zmin = ndc[..., 2].min(dim=2).values                     # nearest point
    span = torch.maximum(x1 - x0, y1 - y0).clamp(min=1.0)
    level = span.log2().ceil().long().clamp(0, len(pyramid) - 1)
    # Sample every mip unconditionally (tiny gathers, no host syncs) and
    # select the conservative level per cluster afterwards. At level l a
    # 2x2 corner sample fully covers any bbox with span <= 2^l texels.
    bidx = torch.arange(B, device=device)[:, None].expand_as(zmin)
    occ_depth = torch.zeros_like(zmin)
    for lv, p in enumerate(pyramid):
        scale = 2 ** lv
        pl = p[:, 0]
        hh, ww = pl.shape[-2], pl.shape[-1]
        bx0 = (x0 / scale).long().clamp(0, ww - 1)
        bx1 = (x1 / scale).long().clamp(0, ww - 1)
        by0 = (y0 / scale).long().clamp(0, hh - 1)
        by1 = (y1 / scale).long().clamp(0, hh - 1)
        m = torch.maximum(
            torch.maximum(pl[bidx, by0, bx0], pl[bidx, by0, bx1]),
            torch.maximum(pl[bidx, by1, bx0], pl[bidx, by1, bx1]))
        occ_depth = torch.where(level == lv, m, occ_depth)
    occluded = zmin > occ_depth + 1e-4
    return visible & ~(occluded & safe)
