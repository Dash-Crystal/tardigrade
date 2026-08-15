

def scope_pass(scene_rgb, fov_deg, base_fov_deg=90.0):
    """The whole subsystem: FOV-driven projection scale, lens mask,
    vignette, black surround.

    `scene_rgb` must already be the MAGNIFIED re-render -- the scale this
    returns is what the caller feeds its projection. Composing a scope over
    an unmagnified frame produces the right mask and the wrong pixels.
    """
    global _SCOPE_PRINTED
    B, H, W, _ = scene_rgb.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=scene_rgb.device),
                            torch.arange(W, device=scene_rgb.device),
                            indexing="ij")
    frag = torch.stack([xx, yy], -1).to(scene_rgb.dtype).expand(B, H, W, 2)
    r = scope_lens_r(frag, H, W)
    scale = scope_fov_scale(fov_deg, base_fov_deg)
    if not _SCOPE_PRINTED:
        _SCOPE_PRINTED = True
        print("[scope] AWP scope pass LIVE. fov=%s -> projection scale "
              "%.4fx; lens radius %.3f x frame height; vignette "
              "1-%.3f*smoothstep(%.2f,1,r); surround %.1f."
              % (fov_deg, scale, SCOPE_RADIUS, SCOPE_VIG_DEPTH,
                 SCOPE_VIG_START, SCOPE_SURROUND))
        print("[scope]   Constants MEASURED from "
              "eval_figures/cs1k_subsystem_tranche_16.png, not fitted to a "
              "metric. Worst residual 0.046 at r 0.75.")
        print("[scope]   scene_rgb is assumed ALREADY re-rendered at the "
              "scoped FOV; this pass does not magnify.")
    return scope_compose(scene_rgb, r), scale


# =======================================================================
# The last eleven: the post chain, the MSAA resolves, the lightmap
# bicubic tap builder and the probe-volume weight.
# =======================================================================
def lf_post_chain(scene_rgb, bloom_exposed, a, b):
    """post_process: Hable with the vpost uniforms, the mode-0 bloom curve
    and the screen blend, plus the TRUE sRGB encode.

    The renderer already tonemaps and encodes; these are the decompiled
    expressions for the same three steps, called here so the frame runs
    them rather than the hand-written copies beside them.
    """
    tm = _po_chain.hable_uniform(scene_rgb)
    bc = _po_chain.bloom_curve_mode0(bloom_exposed)
    sb = _po_chain.screen_blend(a, b)
    return tm, bc, sb, _po_tone.srgb_encode(tm)


def lf_msaa_resolves(samples, depth_samples, exposure_scale):
    """msaa_resolve: the flat-weight colour resolve, the MAX depth resolve
    (a depth attachment resolves by max, not by average -- averaging two
    depths produces a surface that is at neither), and the Karis-weighted
    tonemapped blend."""
    return (_po_msaa.resolve_msaa(samples),
            _po_msaa.resolve_depth_max(depth_samples),
            _po_msaa.resolve_msaa_tonemap_blend(samples, exposure_scale))


def lf_env_bicubic(uv, texel, wt, ht):
    """csgo_environment q1 lightmap: the flat-indexed bicubic tap builder.

    texel_coord_flat and nearest_index_flat are the point-sample path;
    bspline4_weights_flat and bspline4_tap_uvs_flat are the four bilinear
    taps that stand in for sixteen cubic ones.
    """
    w = _sh_env2.bspline4_weights_flat(uv, texel)
    # w packs (i.xy, h.xyzw, g0.xy, g1.xy); the tap builder wants the first
    # two and the next four back as vectors, which is the chain the pass
    # runs -- calling them with independent arguments is what my first
    # version did and the shapes refused it.
    i = w[..., 0:2]
    h = w[..., 2:6]
    return (_sh_env2.texel_coord_flat(uv[..., 0], uv[..., 1], wt, ht),
            _sh_env2.nearest_index_flat(uv[..., 0], uv[..., 1], wt, ht),
            w, _sh_env2.bspline4_tap_uvs_flat(i, h, texel))


def lf_probe_volume_weight(p_local, box_min, box_max, inv_fade):
    """The probe volume's inset weight -- the MIN over six faces, so a
    probe in a corner is not doubly attenuated."""
    return _lighting.probe_volume_weight(p_local, box_min, box_max, inv_fade)


def _bloom(rgb, fg):
    """vpost bloom: soft-knee threshold, 5 equally-weighted blur levels
    (m_flBlurWeight = [0.2]*5, m_vBlurTint all white), BLOOM_BLEND_ADD."""
    lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722],
                              device=rgb.device)).sum(-1, keepdim=True)
    t0 = args.bloom_threshold
    t1 = t0 + max(args.bloom_threshold_width, 1e-4)
    k = ((lum - t0) / (t1 - t0)).clamp(0, 1)
    k = k * k * (3 - 2 * k)                        # smoothstep knee
    # the sky gets its own strength (m_flSkyboxBloomStrength 0.1557)
    sscale = 0.1557 / max(args.bloom_strength, 1e-6)
    k = torch.where(fg.unsqueeze(-1), k, k * sscale)
    bright = (rgb * k).permute(0, 3, 1, 2)
    acc = torch.zeros_like(bright)
    cur = bright
    for _ in range(5):
        cur = _blur5(torch.nn.functional.avg_pool2d(cur, 2))
        acc = acc + 0.2 * torch.nn.functional.interpolate(
            cur, size=bright.shape[-2:], mode="bilinear", align_corners=False)
    return rgb + args.bloom_strength * acc.permute(0, 2, 3, 1)


def _bloom_curve0(b):
    """D_BLOOM_MODE 0, verbatim from the shader:
        clamp((b / (b + 0.187)) * 1.035, 0, 1) * strength
    A Reinhard-shaped curve on the bloom alone -- NOT the scene's Hable
    curve and NOT sRGB-encoded. The composite is still the screen blend.
    Both D_BLOOM_MODE variants screen, so the earlier 'is the blend
    combo-selected?' question is closed: it never is."""
    return ((b / (b + 0.187)) * 1.035).clamp(0, 1) * args.bloom_mode0_strength


def _bloom_only(rgb, fg):
    """the bloom contribution alone, for the engine order where it is
    tonemapped and encoded separately before compositing."""
    return _bloom(rgb, fg) - rgb


def _apply_cc(srgb):
    """trilinear fetch from the 32^3 volume; x=R y=G b=B, [B][G][R] order."""
    n = CC_LUT.shape[0]
    src = srgb.clamp(0, 1)
    # cs2_post_process_ps.glsl:124 -- (c*(N-1) + 0.5)/N, the half-texel
    # inset, unconditionally. There is no other convention the reference
    # takes; see the deleted --lut-texel flag above.
    c = (src * (n - 1) + 0.5).clamp(0, n - 1)
    i0 = c.floor()
    f = (c - i0).unsqueeze(-1)
    i0 = i0.long()
    i1 = (i0 + 1).clamp(max=n - 1)
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg_, fb = f[..., 0, :], f[..., 1, :], f[..., 2, :]
    def L(b, g, r):
        return CC_LUT[b, g, r]
    c00 = L(b0, g0, r0) * (1 - fr) + L(b0, g0, r1) * fr
    c01 = L(b0, g1, r0) * (1 - fr) + L(b0, g1, r1) * fr
    c10 = L(b1, g0, r0) * (1 - fr) + L(b1, g0, r1) * fr
    c11 = L(b1, g1, r0) * (1 - fr) + L(b1, g1, r1) * fr
    c0 = c00 * (1 - fg_) + c01 * fg_
    c1 = c10 * (1 - fg_) + c11 * fg_
    out = c0 * (1 - fb) + c1 * fb
    if args.cc_amount != 1.0:
        out = out * args.cc_amount + src * (1 - args.cc_amount)
    return out


def _srgb_encode(x):
    """TRUE sRGB, as post_process_vulkan_50_ps does it (decompiled): the
    linear segment below 0.0031308 then pow(1/2.4)*1.055 - 0.055. NOT
    pow(1/2.2) -- the shader's exponent is 0.41666666 = 1/2.4."""
    x = x.clamp(min=0)
    return torch.where(x <= 0.0031308, x * 12.92,
                       x.clamp(min=1e-8) ** (1 / 2.4) * 1.055 - 0.055)


def _srgb_decode(x):
    """The shader decodes back to linear after the LUT (step 8)."""
    return _lf_filters.srgb_to_linear(x.clamp(0, 1))


def _encode(x):
    return _srgb_encode(x) if args.srgb_encode == "true" \
        else x.clamp(0, 1) ** (1 / 2.2)


def _tonemap(x):
    """Hable with the vpost's coefficients. The shader pre-scales by a
    hardcoded 2.8 and clamps, then normalises by a CPU-side uniform (m8)
    rather than evaluating the curve at the white point inline; m8 is
    1/F(W) for the vpost's WhitePoint, which is what _TM_WHITE holds."""
    if args.tonemap != "filmic":
        return x.clamp(0, 1)
    if args.tm_prescale != 1.0:
        x = (x * args.tm_prescale).clamp(max=args.tm_clamp)
    return (_hable(x.clamp(min=0)) / _TM_WHITE).clamp(0, 1)


def _tone_encode(rgb, fg):
    """linear HDR -> display.

    Two orders are available. 'legacy' is what this file did before the
    shader was decompiled: bloom added in LINEAR before the tonemap, a
    pow(1/2.2) encode, and an edge-convention LUT fetch. 'engine' is the
    order read out of post_process_vulkan_50_ps:
        tonemap -> TRUE sRGB encode -> bloom (itself tonemapped and
        encoded) composited with a SCREEN blend -> LUT on the encoded
        value with the half-texel convention -> sRGB decode -> final
        pow(1/2.2).
    Every prior tonemap verdict in this project was measured on 'legacy',
    which has the bloom on the wrong side of the curve and in the wrong
    blend, so those results do not carry over.
    """
    if (args.tonemap == "off" and not args.bloom and CC_LUT is None
            and args.srgb_encode != "true"):
        return rgb.clamp(0, 1) ** (1 / 2.2)
    rgb = rgb * PRE_EXPOSURE
    if args.auto_exposure:
        lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722],
                                  device=rgb.device)).sum(-1)
        avg = lum.mean(dim=(1, 2)).clamp(min=1e-5)
        e = (args.ae_key / avg).clamp(args.min_exposure, args.max_exposure)
        rgb = rgb * e.view(-1, 1, 1, 1)

    if args.post_order == "legacy":
        if args.bloom:
            rgb = _bloom(rgb, fg)
        srgb = _encode(_tonemap(rgb))
        return _apply_cc(srgb) if CC_LUT is not None else srgb

    # --- engine order ------------------------------------------------
    bl = _bloom_only(rgb, fg) if args.bloom else None
    srgb = _encode(_tonemap(rgb))
    if bl is not None:
        # the shader runs the bloom through the SAME curve and encode,
        # then screens it over the scene: (s + b) - s*b
        b = (_bloom_curve0(bl.clamp(min=0))
             if args.bloom_curve == "reinhard"
             else _encode(_tonemap(bl)))
        srgb = (srgb + b) - srgb * b if args.bloom_blend == "screen" \
            else (srgb + b).clamp(0, 1)
    if CC_LUT is not None:
        srgb = _apply_cc(srgb)
    if args.post_decode:
        # steps 8 + 11: back to linear after the LUT, then the shader's
        # adjustable final gamma pow(c, m5/2.2)
        srgb = _srgb_decode(srgb).clamp(min=1e-8) ** (args.final_gamma / 2.2)
    return srgb.clamp(0, 1)




_FOGSTAT = []


def fog_opacity(fogin):
    """The scalar `a` apply_fog() blends with, without applying it.

    S_ADDITIVE_BLEND needs the SAME number apply_fog computes, because the
    axis is precisely that an additive surface consumes it differently:
    csgo_complex s240/d2:909 multiplies the output ALPHA by (1 - fog)
    while s80/d2:926 mixes the output COLOUR toward the fog colour. Two
    consumers, one quantity -- so it is factored out rather than
    re-derived, which is how the two could silently drift apart.
    """
    d = fogin[..., 0]
    t = ((d - args.fog_start) / max(args.fog_end - args.fog_start, 1e-6)
         ).clamp(0, 1)
    if args.fog_exponent != 1.0:
        t = t ** args.fog_exponent
    if args.fog_height:
        hz = ((args.fog_end_height - fogin[..., 1])
              / max(args.fog_end_height - args.fog_start_height, 1e-6)
              ).clamp(0, 1)
        if args.fog_vertical_exponent != 1.0:
            hz = hz ** args.fog_vertical_exponent
        t = t * hz
    return (t * args.fog_strength).clamp(0, args.fog_max_opacity)


def apply_fog(rgb, fogin, fg):
    """env_gradient_fog, evaluated exactly as the entity specifies.

    Source's gradient fog is a radial distance blend toward fogcolor,
    modulated by a vertical gradient that is FULL at fogstartheight and
    absent at fogendheight. Both exponents are 1.0 here, so both ramps
    are linear. fogin[...,0] is |eye - fragment| and fogin[...,1] is the
    fragment's world height, both in SOURCE UNITS.
    """
    d = fogin[..., 0]
    t = ((d - args.fog_start) / max(args.fog_end - args.fog_start, 1e-6)
         ).clamp(0, 1)
    if args.fog_exponent != 1.0:
        t = t ** args.fog_exponent
    if args.fog_height:
        hz = ((args.fog_end_height - fogin[..., 1])
              / max(args.fog_end_height - args.fog_start_height, 1e-6)
              ).clamp(0, 1)
        if args.fog_vertical_exponent != 1.0:
            hz = hz ** args.fog_vertical_exponent
        t = t * hz
    a = (t * args.fog_strength).clamp(0, args.fog_max_opacity).unsqueeze(-1)
    if args.fog_stats and not _FOGSTAT:
        _FOGSTAT.append(1)
        m = fg & (d > 0)
        dd = d[m]
        q = torch.tensor([0.5, 0.9, 0.99], device=d.device)
        print(f"fog stats: geometry-pixel distance (source u) median "
              f"{dd.median():.0f} p90 {dd.quantile(0.9):.0f} p99 "
              f"{dd.quantile(0.99):.0f} max {dd.max():.0f}; opacity mean "
              f"{a[..., 0][m].mean():.4f} p90 {a[..., 0][m].quantile(0.9):.4f} "
              f"max {a[..., 0][m].max():.4f}", flush=True)
        del q
    a = a * fg.unsqueeze(-1).float()      # geometry pixels only
    return rgb * (1 - a) + FOG_COLOR.view(1, 1, 1, 3) * a


# --- baked ambient-cube field ----------------------------------------
PROBE_ATLAS = None
PROBE_T = None
PROBE_REF = 1.0
PROBE_BAND = 140
# WHY THIS ONE IS OFF BY DEFAULT WHILE ITS THREE SIBLING SIDECARS ARE ON.
#
# The INPUT is validated: 43 atlases, 1,745 volumes, every depth == 6*band,
# the builder zero-delta against the hand-built reference, and the resolver
# below finds one for every map in the corpus. What is NOT validated is our
# CONSUMPTION of it. The de_inferno five-frame paired arm (866c3e38) scored
# the field mixed-small-negative against the reconstructed volume it is
# supposed to beat: nrmse worse 4/5, ncc worse 3/5, no consistent sign.
#
# The precedent this follows is specular IBL, from this same lane: a
# validated input with an unvalidated consumption that measures level-or-
# worse is left OFF by default with the reason printed -- not defaulted ON
# with a caveat printed. The distinction from the terms that DID go default
# (page B, a -35% all-three win) is the one the validated-replacement door
# requires, and it is simply that this one did not win.
#
# The suspect is right below: the overlap arbitration is a HEURISTIC we
# invented, not a rule we read. The default flips the day it is READ from
# csgo_complex_ps:425-454 (per-volume transform + fade, already transcribed
# by corpus-4) and the price comes back positive.
#
# Resolution itself is unchanged and shared: 'auto' resolves against the
# world pack so the baked field travels with the map it was baked for
# rather than with a path someone typed. Same resolver as --irr-npy /
# --gt-lm-occlusion, so the stem trap that cost three sidecar families
# their inputs (4fa3fb86) is fixed once, in _sidecar, and inherited here
# rather than re-implemented a fourth time.
if args.probe_npz == "auto":
    _pb_auto, _pb_how = _sidecar(".probe_field.npz", "probe field")
    if _pb_auto:
        args.probe_npz = _pb_auto
        print(f"probe field: auto -> {_pb_auto} (via {_pb_how})", flush=True)
    else:
        # THE WEAKER REFUSAL, deliberately, and the asymmetry with
        # --irr-npy is the point. An absent irradiance sidecar leaves the
        # baked term with no substitute, so that one REFUSES. An absent
        # probe atlas leaves the ambient cube falling back to the
        # reconstructed irradiance VOLUME -- a defined path with its own
        # printed provenance, not a stand-in pretending to be the field.
        # Degrading to a defined state is renderable; degrading to a
        # fabricated one is not, and only the second earns a refusal.
        args.probe_npz = None
        print(f"GAP probe field: no <stem>.probe_field.npz for this pack "
              f"({_pb_how}), so the ambient cube comes from the "
              f"reconstructed irradiance volume instead. That is a defined "
              f"fallback with its own provenance line, NOT the baked field "
              f"-- a map that should have an atlas and does not is a "
              f"build gap, not a render setting.", flush=True)
elif args.probe_npz in ("off", "none"):
    args.probe_npz = None
    # The DEFAULT branch, so it states why rather than just that. A GAP-
    # shaped line would be wrong -- nothing is missing here, the asset
    # resolves everywhere; what is missing is a consumption that beats the
    # fallback.
    print("probe field: OFF (default). The baked atlas is a real "
          "directional asset and resolves for every map in the corpus, but "
          "it loses mixed-small to the reconstructed irradiance volume on "
          "the de_inferno 5-frame arm (nrmse 4/5 worse, ncc 3/5 worse, all "
          "|d nrmse| <= 0.0073) -- suspected CONSUMPTION defect, the "
          "overlap arbitration is a HEURISTIC we invented rather than one "
          "read from csgo_complex_ps. The ambient cube takes the volume "
          "path. Enable with --probe-npz auto.", flush=True)
if args.probe_npz:
    _pz = np.load(args.probe_npz, allow_pickle=True)
    PROBE_ATLAS = torch.tensor(_pz["atlas"].astype(np.float32), device=device)
    _t = _pz["table"]
    # ORDER IS THE SHADER'S: ascending volume index. csgo_complex_ps:421
    # walks the cluster's volume bitmask with findLSB, clearing the lowest
    # set bit each pass, so volumes are visited in index order and the
    # (1 - accumulated) factor makes the EARLIEST contributor dominate.
    # Order is therefore load-bearing, not cosmetic.
    #
    # This replaces `np.lexsort((-_t[:, 13], _t[:, 12]))` -- sort by
    # indoor_outdoor_level, tie-break by smallest volume, take the last
    # match. That rule was invented. The shader does not read
    # indoor_outdoor_level in this block at all.
    if _t.shape[1] >= 32:
        PROBE_T = torch.tensor(_t[np.argsort(_t[:, 14], kind="stable")],
                               device=device)
    else:
        # An atlas built before the read has no edge_fade_dists, no local
        # bounds and no yaw, so the accumulation cannot be evaluated at all.
        # REFUSE: rendering it through the old pick would silently produce
        # the pre-read image from a flag that now promises the read one.
        raise SystemExit(
            f"{args.probe_npz} has a {_t.shape[1]}-column table; the read "
            "arbitration needs 32 (edge_fade_dists, local bounds, yaw, "
            "scales). Rebuild it with probe_atlas_build.py -- the fields "
            "are in the map's entity lump and always have been.")
    PROBE_BAND = int(_pz["band"])
    PROBE_REF = float(_pz["atlas"].astype(np.float32).mean())
    print(f"probe field: {PROBE_T.shape[0]} volumes, atlas "
          f"{tuple(PROBE_ATLAS.shape)}, band {PROBE_BAND}, ref {PROBE_REF:.4f}",
          flush=True)
    # ENABLED BY REQUEST, against the measurement, so the renderer says so
    # every time rather than leaving that in a doc. Paired arm, de_inferno
    # five frames, --probe-npz auto vs off, one renderer, one node, one
    # srun: nrmse worse on 4 of 5 (max +0.0073, <=0.9% rel), ncc worse on
    # 3 of 5 (max -0.0170), kl_rgb worse on 3 of 5. No metric moves
    # consistently in either direction. ncc moving AT ALL means the field
    # changes lighting STRUCTURE, not just level -- so this is a live term
    # reaching the image, not an inert load, and it is currently a
    # wash-to-slightly-negative one. The atlas is the baked truth and the
    # volume is the reconstruction; that the reconstruction scores level
    # with it indicts our CONSUMPTION of the field, starting with the
    # arbitration immediately below.
    print("probe field: UNPRICED-POSITIVE -- enabled by request, and the "
          "de_inferno 5-frame paired arm scores it neutral-to-slightly-"
          "worse than the volume path it replaces (nrmse 4/5 worse, ncc "
          "3/5 worse, all |d nrmse| <= 0.0073). This is NOT the default; "
          "omit --probe-npz for the scored-better arm.", flush=True)
_PSTAT = []
_PZERO = []


def probe_volume_t(p, i):
    """csgo_complex_ps:426-428 -- volume i's boundary weight at each point.

    Returns (t, local, bmin, bmax) with p and local in SOURCE units. t is 0
    outside the volume and ramps to 1 across `edge_fade_dists` from each
    face; the shader takes the min over all six faces, so a point near any
    face is faded by that face.

    THE FADE IS READ, NOT FITTED. `_m3` in the shader is the reciprocal of
    the entity's `edge_fade_dists`, which is per-axis and which this table
    now carries. It replaces `--gt-probe-fade`, a float that expressed the
    fade as a fraction of the box size -- a different quantity that happens
    to have the same units. The distinction matters: fade width is absolute
    (8 units is 8 units in a broom cupboard and in a courtyard), so any
    single fraction is wrong for every box but one.

    A ZERO fade is a HARD edge, not an absent one. The shader divides, so
    zero becomes an infinite slope and the clamp saturates the moment the
    point is inside. 39 of de_inferno's 116 volumes are authored that way,
    and reading zero as "no fade term" instead of "instant fade" would give
    them weight 0 and drop them from the accumulation entirely.
    """
    row = PROBE_T[i]
    org, yaw = row[25:28], row[28]
    d = p - org
    if float(yaw.abs()) > 1e-6:
        # World -> volume local: Rz(-yaw). 14 of de_inferno's 116 volumes
        # carry a yaw, and the axis-aligned world_mins/world_maxs in cols
        # 0:6 are origin+box_mins -- correct only at yaw 0. Those 14 have
        # had wrong bounds since the table was first built.
        a = float(yaw) * math.pi / 180.0
        ca, sa = math.cos(a), math.sin(a)
        local = torch.stack([d[:, 0] * ca + d[:, 1] * sa,
                             -d[:, 0] * sa + d[:, 1] * ca,
                             d[:, 2]], dim=-1)
    else:
        local = d
    bmin, bmax = row[19:22], row[22:25]
    fade = row[16:19]
    hard = fade <= 1e-6
    inv = torch.where(hard, torch.ones_like(fade), 1.0 / fade.clamp(min=1e-6))
    a1 = torch.where(hard, (local >= bmin).float(),
                     ((local - bmin) * inv).clamp(0, 1))
    b1 = torch.where(hard, (local <= bmax).float(),
                     ((bmax - local) * inv).clamp(0, 1))
    t = torch.minimum(a1.min(-1).values, b1.min(-1).values)
    return t, local, bmin, bmax


def sample_probes(wpos):
    """Ambient cube at each world position, from the map's probe field.

    wpos is glTF metres; the volumes are in SOURCE units, and
    glTF = (src_y, src_z, src_x) * S, so src = (z, x, y) / S.
    Returns (B,H,W,6,3): the six axial radiances, slot order
    (+X, +Y, +Z, -X, -Y, -Z) in SOURCE axes.

    OVERLAP IS ACCUMULATED, NOT ARBITRATED, and that is the correction this
    function exists to record. It used to pick ONE volume per point --
    highest indoor_outdoor_level, tie-broken by smallest world volume -- a
    rule nobody read anywhere. csgo_complex_ps:421-454 walks the volume list
    in ASCENDING INDEX order (findLSB over a bitmask, lowest bit first) and
    BLENDS: each volume contributes smoothstep(t) * (1 - accumulated), the
    remaining-transmittance factor making earlier volumes dominate, and the
    walk stops once accumulation passes 0.99. indoor_outdoor_level is never
    read in that block at all.

    So the old rule was wrong in KIND, not in tuning. It produced a hard
    seam wherever two volumes met -- every point on one side taking one
    volume's cube whole -- where the reference crossfades over the authored
    edge_fade_dists. No amount of re-ranking gets a crossfade out of a pick.
    """
    p = torch.stack([wpos[..., 2], wpos[..., 0], wpos[..., 1]], dim=-1) / S
    sh = p.shape[:-1]
    p = p.reshape(-1, 3)
    D, Hh, Ww = PROBE_ATLAS.shape[0], PROBE_ATLAS.shape[1], PROBE_ATLAS.shape[2]
    cube = torch.zeros(p.shape[0], 6, 3, device=p.device)
    W = torch.zeros(p.shape[0], device=p.device)
    nvol = 0
    for i in range(PROBE_T.shape[0]):
        t, local, bmin, bmax = probe_volume_t(p, i)
        live = (t > 0) & (W <= 0.99)             # :429 skip, :455 early-out
        if not bool(live.any()):
            continue
        nvol += 1
        # :440 -- ((t*t) * ((-2)*t + 3)) * (1 - W), smoothstep's polynomial
        # written out exactly as the shader writes it.
        dw = ((t * t) * ((-2.0) * t + 3.0)) * (1.0 - W)
        dw = torch.where(live, dw, torch.zeros_like(dw))
        size, at = PROBE_T[i, 6:9], PROBE_T[i, 9:12]
        # grid coordinate in the volume's OWN frame, so the yawed volumes
        # address their probes correctly rather than off-axis
        c = ((local - bmin) / (bmax - bmin).clamp(min=1e-6)).clamp(0, 1) * size

        def fetch(idx):
            ix = (at[0] + idx[:, 0]).long().clamp(0, Ww - 1)
            iy = (at[1] + idx[:, 1]).long().clamp(0, Hh - 1)
            iz = (at[2] + idx[:, 2]).long()
            out = []
            for d6 in range(6):
                out.append(PROBE_ATLAS[(d6 * PROBE_BAND + iz).clamp(0, D - 1),
                                       iy, ix])
            return torch.stack(out, dim=-2)                 # (P,6,3)

        if args.probe_filter == "nearest":
            vc = fetch((c - 0.5).round().clamp(min=0))
        else:
            # trilinear over the probe grid; texel centres sit at +0.5
            f = (c - 0.5).clamp(min=0)
            i0 = f.floor()
            lim = (size - 1).clamp(min=0)
            frac = f - i0
            vc = 0
            for k in range(8):
                off = torch.tensor([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1],
                                   dtype=torch.float32, device=p.device)
                wgt = 1.0
                for a in range(3):
                    wa = frac[:, a].view(-1, 1, 1)
                    wgt = wgt * (wa if off[a] > 0 else (1.0 - wa))
                vc = vc + wgt * fetch(torch.minimum(i0 + off, lim))
        cube = cube + vc * dw.view(-1, 1, 1)
        W = W + dw
    hit = W > 0
    # UNCONDITIONAL, ONCE. This used to sit behind --probe-stats, and that
    # is how an afternoon went into asking "which of the two probe
    # consumers is actually in the measured path" -- a question the renderer
    # could have answered for free on every run. A term that reaches the
    # image and says nothing is exactly the failure this project keeps
    # paying for; the flag stays for the heavier per-volume dump, but WHICH
    # PATH RAN is provenance, not a diagnostic.
    if not _PSTAT:
        _PSTAT.append(1)
        _multi = float(((W > 0) & (W < 0.99)).float().mean())
        print(f"probe path: sample_probes (read arbitration, "
              f"{PROBE_T.shape[0]} volumes in index order)", flush=True)
        print(f"probe stats: {float(hit.float().mean()):.1%} of shaded px fall "
              f"inside a volume; cube lum mean "
              f"{float(cube[hit].mean()) if bool(hit.any()) else 0:.4f} max "
              f"{float(cube.max()):.3f}; volumes contributing {nvol}/"
              f"{PROBE_T.shape[0]}; mean accumulated weight "
              f"{float(W[hit].mean()) if bool(hit.any()) else 0:.4f}; "
              f"{_multi:.1%} of px are in a FADE region (weight under 0.99 "
              f"-- these are the ones the old pick got as a hard seam)",
              flush=True)
    return cube.reshape(*sh, 6, 3), hit.reshape(*sh)


def probe_ambient(cube, normal):
    """Standard ambient-cube evaluation, n^2-weighted over the three axes.

    normal is glTF; source axes are (z, x, y) of it.
    """
    ns = torch.stack([normal[..., 2], normal[..., 0], normal[..., 1]], dim=-1)
    w = ns * ns
    out = 0
    for a in range(3):
        pos = cube[..., a, :]
        neg = cube[..., a + 3, :]
        sel = torch.where((ns[..., a] > 0).unsqueeze(-1), pos, neg)
        out = out + w[..., a].unsqueeze(-1) * sel
    return out


# =====================================================================
# GROUND-TRUTH LIGHTING / SHADOW COMBO AXES  --  ported call flows
# =====================================================================
# Every function below names the reference file and line range it was
# transcribed from.  Where a uniform or a texture the shader reads is not
# in any committed artefact, the function says so in its docstring and
# GT_GAPS collects it for the startup banner -- the computation is still
# the shader's, only its input is generated.

GT_TAKEN = {}          # path name -> pixels that took it (reachability)
# GT_GAPS / _gt_note are defined up with argument normalisation, because
# the first gap is recorded there.

# GT_INJECT_NAMES / GT_INJECT / _gt_inject are defined up with argument
# normalisation, because --gt-selftest-inject all lists them there.
def _gt_count(name, mask):
    """Record how many pixels actually executed a path.

    NECESSARY AND NOT SUFFICIENT. This answers "did the code run", which
    is arm 1 of the two-arm rule. It cannot answer "did the value reach
    the image", and the two are not the same question: `_gt_count(
    "speccube.q0_unconditional", ones)` below reports 100% of pixels on
    every run ever made, while the value that branch returned was
    EXACTLY ZERO on all of them, because its asset had never been built.
    A term can execute on every pixel and contribute nothing.

    That is the defect COMPLETENESS_BY_FLOPS.md raises against itself --
    every `live` in its table, its most-used label at ~54% by FLOPs, was
    justified from a flag default -- and it is the shape of
    the cascade depth-sign fix, where a cascade term was `live` for months while
    its value was 0.0000 on 100.0% of 518,400 samples.

    `_gt_value()` below is arm 2. Use both.
    """
    n = int(mask.sum()) if mask is not None else 0
    GT_TAKEN[name] = GT_TAKEN.get(name, 0) + n
    return n


# Arm 2 of the two-arm rule: what the term was WORTH, not whether it ran.
GT_VALUE = {}


def _gt_value(name, value, identity=None, note=None):
    """Record a term's output statistics so inertness is visible DATA.

    `identity` is the value at which the term contributes nothing to its
    consumer -- 0.0 for something added, 1.0 for something multiplied.
    Give it and the fraction of samples sitting exactly there is
    reported; a term at fraction-at-identity 1.000 ran and changed
    nothing, and the epilogue says so in those words.

    Not behind a flag, and cheap: one reduction per term per call. A
    diagnostic you have to enable is a diagnostic nobody reads, and the
    two failures this exists to catch were both invisible precisely
    because nothing printed unless asked.
    """
    if value is None:
        return value
    v = value.detach().float()
    if v.numel() == 0:
        return value
    e = GT_VALUE.setdefault(name, {"n": 0, "sum": 0.0, "min": float("inf"),
                                   "max": float("-inf"), "idf": 0.0,
                                   "identity": identity, "note": note})
    e["n"] += 1
    e["sum"] += float(v.mean())
    e["min"] = min(e["min"], float(v.min()))
    e["max"] = max(e["max"], float(v.max()))
    if identity is not None:
        e["idf"] += float((v == float(identity)).float().mean())
    return value


def gt_value_json(path):
    """Emit the term-value table as json, for the gate's precondition.

    W6 asked for this so `gate.py` can refuse a run in which any
    instrumented term is inert -- the charter's form-1 precondition
    ("the term demonstrably REACHES PIXELS") checked mechanically
    instead of per-term by hand. Four inert terms were found in one day,
    so the class is common rather than incidental.

    `frac_at_identity` is what mean-and-range cannot give you: a term
    can have a healthy mean and still contribute nothing, which is
    exactly what `sun.baked_occlusion` (constant 1.0) and the
    pre-asset environment specular (constant 0.0) did.

    `frac_reaching_output` IS DELIBERATELY ABSENT and that is not an
    oversight. W6 named it as the field that matters, and it is the one
    field this instrument cannot honestly produce: a term's value
    reaching the framebuffer is a property of its CONSUMER, not of the
    term, and `_gt_value` sees only the value. `atlas.indirect` has
    mean 0.5119 on 99.94% of pixels and reaches nothing -- no statistic
    of that tensor could have said so; it took changing the term and
    watching the frame not move. Emitting a guessed
    `frac_reaching_output` would put a number nobody measured into a
    gate precondition, which is the failure this whole instrument
    exists to stop. What CAN be automated is the experiment that found
    it -- perturb a term, re-render, diff -- and that is a harness, not
    a field. Recorded here as the reason the key is missing.
    """
    import json as _json
    out = {}
    for k, e in GT_VALUE.items():
        n = max(1, e["n"])
        out[k] = {
            "mean": e["sum"] / n,
            "min": e["min"],
            "max": e["max"],
            "frac_at_identity": (None if e["identity"] is None
                                 else e["idf"] / n),
            "identity": e["identity"],
            "batches": e["n"],
            "constant": e["max"] == e["min"],
            "inert": (e["identity"] is not None and e["idf"] / n > 0.9999),
            "note": e["note"] or "",
        }
    doc = {
        "schema": "iji/term-values/v1",
        "frac_reaching_output": None,
        "frac_reaching_output_absent_because":
            "a term's value reaching the framebuffer is a property of "
            "its CONSUMER, not of the term. _gt_value sees only the "
            "value. atlas.indirect has mean 0.5119 on 99.94% of pixels "
            "and reaches nothing; no statistic of that tensor says so. "
            "It took perturbing the term and watching the frame not "
            "move. A guessed value here would be a number nobody "
            "measured inside a gate precondition.",
        "terms": out,
    }
    with open(path, "w") as f:
        _json.dump(doc, f, indent=1, sort_keys=True)
    n_inert = sum(1 for v in out.values() if v["inert"])
    print(f"TERM VALUES json -> {path} ({len(out)} terms, {n_inert} inert)",
          flush=True)


def gt_value_epilogue():
    """Print every instrumented term's mean, range and inert fraction."""
    if args.term_values_json:
        gt_value_json(args.term_values_json)
    if not GT_VALUE:
        return
    print("=" * 66, flush=True)
    print(f"TERM VALUES ({len(GT_VALUE)} instrumented). _gt_count answers "
          f"'did it run'; this answers 'was it worth anything'. A term at "
          f"at-identity 1.0000 reached every pixel and changed none of "
          f"them.", flush=True)
    for k in sorted(GT_VALUE):
        e = GT_VALUE[k]
        m = e["sum"] / max(1, e["n"])
        idf = e["idf"] / max(1, e["n"])
        ids = "" if e["identity"] is None else \
            f"  at-identity({e['identity']:g}) {idf:.4f}"
        flag = ""
        if e["identity"] is not None and idf > 0.9999:
            flag = "   <== INERT: runs on every pixel, changes nothing"
        elif e["max"] == e["min"]:
            flag = "   <== CONSTANT across every sample this run"
        print(f"  {k:<32s} mean {m: .6g}  "
              f"[{e['min']: .4g}, {e['max']: .4g}]{ids}{flag}", flush=True)
        if e["note"]:
            print(f"      {e['note']}", flush=True)


# ---------------------------------------------------------------------
# D_BAKED_LIGHTING = none  --  the constant-buffer tail
# csgo_complex_ps.glsl:480-489 / csgo_environment_ps.glsl equivalent:
#     if (accumW < 0.99)
#         irr = accum + vec3(dot(g_vAmbientCube[0], vec4(n,1)),
#                            dot(g_vAmbientCube[1], vec4(n,1)),
#                            dot(g_vAmbientCube[2], vec4(n,1)))
#               * (1.0 - accumW);
# This is the FOURTH baked-lighting state, not an off switch: dynamic id
# 0 is a shipped module in every static combo of every family.
# ---------------------------------------------------------------------
if args.gt_ambient_sh is not None:
    GT_AMB_SH = torch.tensor(args.gt_ambient_sh, dtype=torch.float32,
                             device=device).view(3, 4)
else:
    # DERIVED, not fixed. The pre-existing hemispheric ambient is
    #     ground + (sky - ground) * (n.y * 0.5 + 0.5)
    # which is affine in n, and vec4(A, B) . vec4(n, 1) is the general
    # affine form, so the shader's own expression represents it to
    # rounding:
    #     A = (0, (sky-ground)*0.5, 0)      B = (sky+ground)*0.5
    # Quantified rather than called identical: swept over n.y in [-1, 1]
    # at 2001 points in float64, max|delta| = 1.11e-16, i.e. one ULP.
    # --gt-lighting-selftest reprints it at the renderer's own precision.
    # A fixed default here would have silently replaced the fitted
    # hemisphere with somebody else's constants.
    _gs = (AMBIENT_SKY - AMBIENT_GROUND) * 0.5
    _gb = (AMBIENT_SKY + AMBIENT_GROUND) * 0.5
    GT_AMB_SH = torch.stack([
        torch.stack([torch.zeros_like(_gs[c]), _gs[c],
                     torch.zeros_like(_gs[c]), _gb[c]])
        for c in range(3)], dim=0)


def gt_ambient_sh(normal):
    """irradiance = vec3(dot(g_vAmbientCube[c], vec4(n, 1))), c in 0..2.

    csgo_complex_ps.glsl:483-484 -- three dot4 against vec4(normal, 1).
    That is 3 x (2N-1) = 21 FLOPs, and it is exactly the 21 FLOPs that
    D_BAKED_LIGHTING_FROM_VERTEX_STREAM removes in every row of
    SHADER_CALLFLOW_csgo_environment.md:551-566 (2370->2349, 2564->2543,
    2426->2405, 2620->2599, 2760->2739) -- which is how we know the
    vertex-stream path REPLACES this term rather than adding to it.
    """
    if _gt_inject("ambientsh-zero"):
        return torch.zeros_like(normal)
    nh = torch.cat([normal, torch.ones_like(normal[..., :1])], dim=-1)
    return torch.stack([(nh * GT_AMB_SH[c]).sum(-1) for c in range(3)],
                       dim=-1)


# ---------------------------------------------------------------------
# D_BAKED_LIGHTING_FROM_PROBE
# csgo_complex_ps.glsl:302-489.  Two binned loops over the light-probe
# volume list; the second one is the one that fetches.
# ---------------------------------------------------------------------
_PGT = []


def _probe_gt_note(which, irr, w, sel=None):
    """Say ONCE, on the gt path, whether the probe volumes carry anything.

    The same instrument as sample_probes()'s banner, on the other consumer,
    added for the same reason: this term was believed live on this path for
    a long time on the strength of a combo-table row that only proves the
    MODULE was selected, never that the volumes reached the pixel. A
    selected module with an empty accumulator renders exactly like no
    module at all.
    """
    if _PGT or PROBE_T is None:
        return
    _PGT.append(1)
    hit = w > 0
    frac = float(hit.float().mean())
    lvl = float(irr[hit].mean()) if bool(hit.any()) else 0.0
    _selfrac = ("" if sel is None else
                f"; the engine split ROUTES {float(sel.float().mean()):.2%} "
                f"of px to the probe (the rest take the lightmap) -- this, "
                f"not the coverage above, bounds what the term can do here")
    print(f"probe path: gt_baked_probe (mode {which}, "
          f"{PROBE_T.shape[0]} volumes) -- {frac:.1%} of px inside a "
          f"volume, mean irradiance {lvl:.4f}, mean accumulated weight "
          f"{float(w[hit].mean()) if bool(hit.any()) else 0.0:.4f}"
          + _selfrac
          + ("" if frac > 0 else "  <-- EMPTY: the module is selected and "
             "the volumes reach nothing, which renders identically to not "
             "running it"), flush=True)


def gt_baked_probe(wpos, normal):
    """Ambient-cube irradiance volume with the shader's soft volume blend.

    Ported from csgo_complex_ps.glsl:378-489.  Per volume i, in order:

      p      = M_i * vec4(wpos, 1)                              (:425)
      a      = clamp((p - boxMin_i) * fade_i, 0, 1)             (:426)
      b      = clamp((boxMax_i - p) * fade_i, 0, 1)             (:427)
      t      = min(min3(a), min3(b))                            (:428)
      if t == 0: skip                                           (:429)
      dw     = t*t*(3 - 2*t) * (1 - W)                          (:440)
      W     += dw                                               (:441)
      slice  = mix(vec3(0,1/6,2/6), vec3(3/6,4/6,5/6), step(n,0))  (:451)
      n2     = n * n                                            (:452)
      irr   += (T(uvw + (0,0,slice.x)) * n2.x
              + T(uvw + (0,0,slice.y)) * n2.y
              + T(uvw + (0,0,slice.z)) * n2.z) * scale_i * dw   (:453)
      occ   += T_occ(uvw2) * dw                                 (:454)
      if W > 0.99: break                                        (:455)

    The 3-slice-of-6 selection with n*n weights IS the ambient cube, and
    probe_ambient() above already computes exactly that inner sum; what
    this adds is the shader's *volume* arithmetic, which the pre-existing
    hard "highest indoor_outdoor_level wins" select did not have.

    BOTH GENERATED CONSTANTS HAVE NOW BEEN READ, and this docstring's
    "NOT IN THE ASSET" was true of the npz and false of the map/engine:

      * fade_i (_m3, :344) is edge_fade_dists, an authored per-axis field
        on every light_probe_volume entity. READ; the builder carries it
        and probe_volume_t() uses it. The old 1/(--gt-probe-fade * extent)
        expressed fade as a FRACTION of the box, which is a different
        quantity in the same units.

      * scale_i (_m4.xyz, :453) READS EXACTLY 1.0. It lives in the
        atlas-entry uniform block (set 1 binding 2, 400 records), which is
        engine descriptor state rather than map data, so it took a capture
        read: wallA2/interior, uniform buffer of range 57600 = 400 * 144,
        row_major frame (_m3 @76, _m4 @80, _m5 @96, _m6 @108, _m7 @112).
        1.0 on all 248 populated records across two buffers, 0.0 on the
        unpopulated tail, with _m3/_m6 corroborating as adjacent bindless
        descriptor indices (9502..10218).

        SO "Held at 1.0" WAS CORRECT, and this is the good outcome for a
        generated constant: the read agrees with it. What the read also
        does is KILL the hypothesis it was run to test -- our probe
        irradiance is ~1.63x the atlas reference (1.3300 vs 0.8180) and a
        missing per-volume scale is NOT the explanation. That discrepancy
        is now unexplained rather than attributed, which is the honest
        state and a better one than a plausible wrong cause.
      * the SECOND sampler3D (_m6, :454), the per-light baked occlusion
        vec4. No such volume is decoded anywhere in this repo, so occ is
        the shader's own unbound-descriptor value, 0 -- which makes
        1 - dot(occ, mask) equal 1, i.e. the sun is gated by the cascade
        shadow alone. This is reported by the banner, not swallowed.
    """
    if PROBE_ATLAS is None or PROBE_T is None:
        z = torch.zeros(*wpos.shape[:-1], 4, device=wpos.device)
        # THE dyn-0 TAIL IS AN IDENTICAL ZERO, AND IT IS THE WHOLE INDIRECT
        # TERM FOR EVERY NON-LIGHTMAPPED PIXEL.
        #
        # gt_ambient_sh() evaluates GT_AMB_SH, which is derived from
        # AMBIENT_SKY and AMBIENT_GROUND -- and both are `torch.zeros(3)`
        # at :9771-9772 on every path. --light-entity does not rescue it
        # either: _chroma() renormalises to `float(ref.mean())`, and the
        # ref IS the zero tensor, so the product is zero again. The
        # hemisphere those two constants used to carry was deleted on the
        # read that g_vSunAmbient is (0,0,0) in 717 of 718 block copies,
        # which is correct for the SUN'S ambient and was never a
        # replacement for the baked bounce.
        #
        # So the chain for a pixel outside the lightmap was: gt_baked
        # "auto" -> lm_ok fails -> probe -> no --probe-npz -> here -> 0.
        # Three hops, each individually defensible, ending in black. On
        # de_mirage that is 70.1% of the surface (29.9% lightmapped), and
        # it is why quadrupling irradiance and octupling the sun left the
        # backdrop 98-99% black: neither lever touches this path. Same
        # single fault reaches the backdrop, the viewmodel and the
        # playermodels, which is why they were three symptoms.
        #
        # THE VOLUME IS ALREADY BUILT. --irr-vol reconstructs an indirect
        # field from the map's own lightmapped vertices (1.1% of cells
        # sampled directly, the rest push-pull filled) and, until now,
        # only gt_bake_vertex_stream() -- an opt-in axis -- ever read it.
        # It is a READ input, not a fitted one: every value in it comes
        # from Valve's own bake, resampled. Sampling it at the shaded
        # point is what the reference does with probes, on the field this
        # repo actually holds.
        if VOL is not None:
            _gt_note(
                "probe: --probe-npz not given, so the indirect for "
                "non-lightmapped geometry is SAMPLED FROM THE "
                "RECONSTRUCTED IRRADIANCE VOLUME (--irr-vol) at the "
                "shaded point. That volume is built from this map's own "
                "lightmapped vertices by push-pull fill, so it is a "
                "resample of Valve's bake, not a fitted stand-in. WHAT "
                "IT IS NOT: the reference's probe volumes carry a "
                "per-volume transform, fade distance, scale and a second "
                "occlusion sampler3D (csgo_complex_ps.glsl:425-454); "
                "none of those exist here, so this is the ambient "
                "magnitude without the volume arithmetic and without "
                "directionality -- the sample does not vary with the "
                "shading normal. THAT LAST PART IS FIXABLE TODAY AND NOT "
                "BY WRITING CODE: de_inferno's own ambient-cube atlas is "
                "on disk at .scratch-tone/probe_field.npz (116 volumes, "
                "six genuinely different faces, 4.1x up-vs-down), and "
                "--probe-npz runs the full six-face path. Applying the "
                "reference's normal^2 basis to THIS volume instead would "
                "be provably inert -- the volume is push-pull filled from "
                "lightmapped vertices and carries one RGB per cell, so "
                "all six slices would be equal and sum(n_i^2) == 1 makes "
                "the weighted sum the identity. Directionality needs the "
                "directional ASSET, not the basis expression.")
            irr_v = sample_volume(wpos)
            _gt_value("probe.volume_indirect", irr_v.mean(-1), identity=0.0,
                      note="the indirect handed to every non-lightmapped "
                           "pixel. at-identity 1.0000 means the volume is "
                           "built but reads zero here, which is a "
                           "push-pull coverage question, not a wiring "
                           "one; before this landed the value was an "
                           "EXACT zero with no row at all.")
            return irr_v, z, torch.zeros_like(wpos[..., 0])
        _gt_note("probe: --probe-npz not given AND no irradiance volume "
                 "(--irr-vol 0, or --irr-npy absent), so "
                 "D_BAKED_LIGHTING_FROM_PROBE falls to the dyn-0 tail "
                 "gt_ambient_sh -- which on this build evaluates to an "
                 "IDENTICAL ZERO, because AMBIENT_SKY and AMBIENT_GROUND "
                 "are both zeros(3) at :9771. Every non-lightmapped pixel "
                 "therefore receives NO indirect light at all. This is a "
                 "real module reading a real (0,0,0), not a stub, and it "
                 "is the wrong source for baked bounce: build the volume "
                 "with --irr-vol, or bind probes with --probe-npz.")
        _gt_value("probe.dyn0_tail", gt_ambient_sh(normal).mean(-1),
                  identity=0.0,
                  note="the dyn-0 ambient handed to non-lightmapped "
                       "pixels. at-identity 1.0000 = they are receiving "
                       "exactly zero indirect light.")
        return gt_ambient_sh(normal), z, torch.zeros_like(wpos[..., 0])
    _gt_note("probe: the second sampler3D (per-light baked occlusion, "
             "csgo_complex_ps.glsl:454) has no decoded asset in this repo; "
             "occ = 0, the shader's unbound value")
    # source units, matching sample_probes()
    p = torch.stack([wpos[..., 2], wpos[..., 0], wpos[..., 1]], dim=-1) / S
    sh = p.shape[:-1]
    p = p.reshape(-1, 3)
    ns = torch.stack([normal[..., 2], normal[..., 0], normal[..., 1]],
                     dim=-1).reshape(-1, 3)
    n2 = ns * ns
    D, Hh, Ww = (PROBE_ATLAS.shape[0], PROBE_ATLAS.shape[1],
                 PROBE_ATLAS.shape[2])
    irr = torch.zeros_like(p)
    W = torch.zeros(p.shape[0], device=p.device)
    for i in range(PROBE_T.shape[0]):
        size, at = PROBE_T[i, 6:9], PROBE_T[i, 9:12]
        if args.gt_probe_blend:
            # THE SAME READ AS sample_probes, from the same helper, because
            # these are two consumers of ONE quantity and having them agree
            # by construction is worth more than either being locally tidy.
            # This site used to compute the fade as a FRACTION of the box
            # (--gt-probe-fade * box-size); the entity's edge_fade_dists is
            # an absolute per-axis width, so the two agreed on no volume
            # except by accident.
            t, local, bmin, bmax = probe_volume_t(p, i)
            lo, hi = bmin, bmax
            p_local = local
        else:
            # DIAGNOSTIC hard select: inside = 1, outside = 0.
            lo, hi = PROBE_T[i, 0:3], PROBE_T[i, 3:6]
            p_local = p
            t = (((p >= lo) & (p < hi)).all(-1)).float()
        live = (t > 0) & (W <= 0.99)          # :429 and :455
        if not bool(live.any()):
            continue
        # :440 -- t*t*(3-2t) is smoothstep's polynomial written out, and
        # the shader writes it out: ((t*t) * ((-2)*t + 3)) * (1 - W).
        dw = ((t * t) * ((-2.0) * t + 3.0)) * (1.0 - W)
        dw = torch.where(live, dw, torch.zeros_like(dw))
        if _gt_inject("probe-zero"):
            dw = torch.zeros_like(dw)
        # texel coordinate inside this volume's atlas block
        c = ((p_local - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1) * size
        f = (c - 0.5).clamp(min=0)
        i0 = f.floor()
        lim = (size - 1).clamp(min=0)
        frac = f - i0

        def _fetch(idx):
            ix = (at[0] + idx[:, 0]).long().clamp(0, Ww - 1)
            iy = (at[1] + idx[:, 1]).long().clamp(0, Hh - 1)
            iz = (at[2] + idx[:, 2]).long()
            out = []
            for d in range(6):
                out.append(PROBE_ATLAS[(d * PROBE_BAND + iz).clamp(0, D - 1),
                                       iy, ix])
            return torch.stack(out, dim=-2)                 # (P,6,3)

        if args.probe_filter == "nearest":
            cube = _fetch((c - 0.5).round().clamp(min=0))
        else:
            cube = 0
            for k in range(8):
                off = torch.tensor([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1],
                                   dtype=torch.float32, device=p.device)
                wgt = 1.0
                for ax in range(3):
                    wa = frac[:, ax].view(-1, 1, 1)
                    wgt = wgt * (wa if off[ax] > 0 else (1.0 - wa))
                cube = cube + wgt * _fetch(torch.minimum(i0 + off, lim))
        # :451-453 -- pick the +axis or -axis slice by step(n, 0), weight
        # by n*n, sum the three. scale_i (_m4.xyz) held at 1.0.
        acc = 0
        for ax in range(3):
            sel = torch.where((ns[:, ax] > 0).unsqueeze(-1),
                              cube[:, ax, :], cube[:, ax + 3, :])
            acc = acc + n2[:, ax].unsqueeze(-1) * sel
        irr = irr + acc * dw.unsqueeze(-1)
        W = W + dw
    # :480-489 -- whatever weight is left over goes to the constant tail.
    rem = (1.0 - W).clamp(min=0.0)
    irr = irr.reshape(*sh, 3) + gt_ambient_sh(normal) * rem.reshape(
        *sh, 1)
    occ = torch.zeros(*sh, 4, device=wpos.device)
    return irr, occ, W.reshape(*sh)


# ---------------------------------------------------------------------
# D_BAKED_LIGHTING_FROM_LIGHTMAP
# csgo_environment_ps.glsl:756-758 (the two sampler2DArray fetches),
# :888 and :1067 (how the second one is consumed), :1227-1240 (how the
# first one is consumed).
# ---------------------------------------------------------------------
def _gt_tex2d(tex, u, v):
    """One hardware-bilinear sampler2DArray tap, clamped.

    TEXEL CENTRES, `x = u*W - 0.5`. Not `u*(W - 1)`, which is what this
    function did until 2026-08-08 and which is the SAME defect
    `cube_sample()` above records finding and fixing in the cube decode:
    corner-to-corner mapping compresses the whole page onto the texel
    CENTRES. On the 64-wide lightmap pages this pack ships that is a
    half-texel shift plus a 64/63 scale, applied to the baked
    irradiance, which is the dominant light in every frame.

    It also made the q=1 filter incoherent rather than merely wrong.
    `csgo_environment_ps_shaderquality1.glsl:844` computes its B-spline
    offsets in TEXEL-CENTRE space (`uv/texel - 0.5` with
    `texel = 1/textureSize`), so feeding those coordinates to a
    corner-to-corner sampler converts twice. Fixing the filter's
    addressing alone measured WORSE (ncc 0.3998 -> 0.3511) for exactly
    that reason; the two conventions have to agree, and the reference's
    is the one both of them have to agree with.
    """
    ht, wt = tex.shape[0], tex.shape[1]
    x, y = env_shading.texel_coord(u, v, wt, ht)
    x, y = x.clamp(0, wt - 1), y.clamp(0, ht - 1)
    x0, y0 = x.floor(), y.floor()
    fx, fy = (x - x0), (y - y0)
    x0 = x0.long().clamp(0, wt - 1)
    y0 = y0.long().clamp(0, ht - 1)
    x1 = (x0 + 1).clamp(max=wt - 1)
    y1 = (y0 + 1).clamp(max=ht - 1)
    g00 = tex[y0, x0]
    if g00.dim() > fx.dim():
        fx, fy = fx.unsqueeze(-1), fy.unsqueeze(-1)
    return ((g00 * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
            + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)


def _gt_bicubic4(tex, u, v):
    """The q=1 lightmap filter. DELEGATES to the importable module.

    Body moved to `iji_model/counter_strike_render/shading/environment.py`
    (`bicubic4`), which the conformance registry A/Bs against
    `csgo_environment_ps_shaderquality1.glsl:844-862` directly. This
    function is now the renderer's adapter to it: page-size to texel,
    (u, v) to a vec2, and `_gt_tex2d` as the bilinear tap.

    WHY IT MOVED, and it is not tidiness. `gpu_render.py` parses argv at
    import, so nothing defined in it can be imported by a test -- and an
    expression no test can reach is an expression whose conformance is
    an assertion. This one in particular: `Q1_LOCALISATION.md` measured
    the old body at **83.8% of the MSE of our q0->q1 delta and 100% of
    the ncc regression**, which made it the single largest contributor
    to defect #57, the gate-blocking one.

    WHAT THE OLD BODY GOT WRONG, now that the q1 reference is extracted
    and committed (it was not when that body was written -- its own
    docstring said "no q=1 csgo_environment GLSL is committed, so the
    exact tap pattern could not be read", and the kernel was chosen from
    a fetch COUNT):

      * the weight polynomials were RIGHT -- algebraically identical to
        Q1:849-852. The kernel family was never the problem.
      * the ADDRESSING was wrong. It used `u * (wt - 1)` and stepped by
        `1/(wt - 1)`: corner-to-corner. The reference uses `uv/texel`
        with `texel = 1/textureSize`, i.e. texel CENTRES. On a 64-wide
        page that is a 1/64 shift plus a 64/63 scale. `cube_sample()`
        above records finding and fixing this exact error in the cube
        decode; it was still live here.
      * the second `- 0.5` at Q1:862 was absent, moving every tap a half
        texel.
      * the reference uses `textureGrad` with the derivatives of the
        ORIGINAL uv at all four sites, so the taps share one mip; ours
        took four plain bilinear taps.

    STILL WRONG, AND NOT FIXABLE HERE: four of the six extra q1 fetches
    are not a filter at all. The reference binds a THIRD page and
    redistributes the irradiance by `dot(bakedDirection, normal)`
    (Q1:818-831), so at q1 its lightmap responds to the normal map. We
    have no direction page. `environment.lightmap()` carries that as a
    returned note rather than a silent default, and it is an ASSET hole
    of the same class as the cube array.
    """
    ht, wt = tex.shape[0], tex.shape[1]
    uv = torch.stack([u, v], dim=-1)
    texel = torch.tensor([1.0 / wt, 1.0 / ht], device=uv.device,
                         dtype=uv.dtype)

    def _sample(c):
        return _gt_tex2d(tex, c[..., 0], c[..., 1])

    return env_shading.bicubic4(_sample, uv, texel)


# Set by gt_baked() on EVERY call, and read by gt_compose(). It carries
# one bit -- "the lightmap module ran and --gt-lm-tint is on" -- from the
# baked-source selection down to the composite, because the shader's tint
# site is downstream of the AO multiply and therefore downstream of where
# the baked source is chosen. gt_baked() is the single entry point for
# every baked mode, so the flag cannot go stale on a path that skips the
# lightmap; the one caller that composes without going through gt_baked
# (gt_lighting_selftest, on probe irradiance) passes lm_tint explicitly.
GT_LM_TINT_LIVE = False

# Resolve 'auto' against the world pack, so the baked occlusion page
# travels with the map it was baked for. Same convention as --irr-npy and
# --sky-cube; the REFUSAL is deliberately weaker than --irr-npy's and the
# reason is in the difference between the two absences. With no irradiance
# the renderer substitutes a different physical model, so that one exits.
# With no occlusion page, `1 - dot(0, mask)` is 1, which is exactly what
# the reference computes when the descriptor is unbound -- a state the
# shader defines, not a stand-in we invented. So this says so at full
# volume and in the GAP banner (which prints BEFORE shading, per #43) and
# renders on.
if args.gt_lm_occlusion == "auto":
    _occ_auto = (os.path.splitext(args.world)[0] + ".direct_light_shadows.npy"
                 if args.world else None)
    if _occ_auto and os.path.exists(_occ_auto):
        args.gt_lm_occlusion = _occ_auto
        print("lm occlusion: auto -> %s" % _occ_auto, flush=True)
    else:
        args.gt_lm_occlusion = None
        # The discriminator is DATA, not the filename: a pack carrying
        # lightmap UVs was built expecting the page, and one without never
        # had it. Same test --irr-npy uses.
        _lm_n = (int((lmuv.abs().sum(-1) > 0).sum()) if lmuv is not None
                 else 0)
        if _lm_n == 0:
            print("lm occlusion: auto -> none beside the pack, and this "
                  "pack carries no lightmap UVs, so none is expected.",
                  flush=True)
        else:
            print("lm occlusion: auto found NO page beside a pack whose "
                  "geometry IS lightmapped (%s vertices carry lightmap "
                  "UVs).\n  expected: %s\nsun.baked_occlusion will pin at "
                  "exactly 1 -- the shader's unbound value, so this is a "
                  "DEFINED render and not a silent substitution, but it is "
                  "a measurably WORSE one: on de_dust2 f2512 the page moves "
                  "nrmse 1.4954 -> 0.9707 and ncc 0.2567 -> 0.3377. "
                  "Extract it with lightmap_extract.py --entry "
                  "lightmaps/direct_light_shadows.vtex_c, or pass "
                  "--gt-lm-occlusion none to say you meant to render "
                  "without it." % (f"{_lm_n:,}", _occ_auto), flush=True)
            _gt_note("lm occlusion: NO page beside a lightmapped pack; "
                     "sun.baked_occlusion is pinned at 1 (the shader's "
                     "unbound value). Measured cost on de_dust2 f2512: "
                     "nrmse 0.9707 -> 1.4954, ncc 0.3377 -> 0.2567")
elif args.gt_lm_occlusion == "none":
    print("lm occlusion: DISABLED by --gt-lm-occlusion none", flush=True)
    args.gt_lm_occlusion = None

GT_LM_OCC = None
if args.gt_lm_occlusion:
    _occ_np = np.load(args.gt_lm_occlusion)
    GT_LM_OCC = torch.from_numpy(_occ_np).to(device).float()
    # THE /255 IS THE CODEC'S, NOT A CHOICE. The page ships as BC7 on
    # de_dust2 and ATI2N on de_inferno; both store UNORM channels, and
    # UNORM is defined as byte/255 -> [0,1]. lightmap_extract.py writes the
    # decoded BYTES, so the sampler's normalisation has to be applied here
    # or `1 - dot(occ, mask)` evaluates to about -254 instead of a
    # visibility in [0,1]. Keyed on the array's own dtype rather than
    # assumed, so a float sidecar is left alone.
    if _occ_np.dtype == np.uint8:
        GT_LM_OCC = GT_LM_OCC / 255.0
        print("GT lightmap occlusion: uint8 source, UNORM-normalised /255 "
              "(the texture format's own definition, not a fitted scale)",
              flush=True)
    else:
        print("GT lightmap occlusion: dtype %s, taken as already normalised"
              % _occ_np.dtype, flush=True)
    print(f"GT lightmap occlusion array {tuple(GT_LM_OCC.shape)} "
          f"range [{float(GT_LM_OCC.min()):.4f}, {float(GT_LM_OCC.max()):.4f}]"
          f" mean {float(GT_LM_OCC.mean()):.4f}", flush=True)

# The THIRD lightmap array, q1 only (_4565._m3, Q1:817). Normalised to
# [0,1] here because Q1:818 reads it as a texture fetch -- `tex.xy*2-1`
# only recovers a direction if the fetch already delivered [0,1], which
# is what a UNORM8 sampler does and what an integer .npy does NOT.
# Divide-by-255 is therefore part of reading the module correctly, not a
# convenience.
GT_LM_DIR = None
if args.gt_lm_direction:
    _d = np.load(args.gt_lm_direction)
    GT_LM_DIR = torch.from_numpy(_d).to(device).float()
    if _d.dtype == np.uint8:
        GT_LM_DIR = GT_LM_DIR / 255.0
    _ch = [float(GT_LM_DIR[..., i].mean())
           for i in range(GT_LM_DIR.shape[-1])]
    print(f"GT lightmap DIRECTION array {tuple(GT_LM_DIR.shape)} "
          f"({_d.dtype}), per-channel mean "
          f"{' '.join(f'{c:.4f}' for c in _ch)}. Consumed per Q1:818-831: "
          f".xy = hemispherical disc direction, .z = the confidence in "
          f"clamp(z + bias, 0, 1). THE MODULE DECLARES THAT; the channel "
          f"means are printed as corroboration and were not used to infer "
          f"it.", flush=True)
    del _d

# generic.vfx (csgo_core) r0_m3:270 fetches a THIRD lightmap
# sampler2DArray that csgo_lightmappedgeneric does not have: xy is the
# dominant baked direction, z the baked AO level. Loaded here beside the
# other two so all three lightmap arrays are visibly in one place.
GEN_LMDIR = None
if args.lmg_lm_direction:
    GEN_LMDIR = torch.from_numpy(
        np.load(args.lmg_lm_direction)).to(device).float()
    print(f"generic.vfx directional lightmap array "
          f"{tuple(GEN_LMDIR.shape)}", flush=True)


def gen_lm_direction(u, v, quality):
    """The direction+AO array at the lightmap UV, or the shader's own
    degenerate value.

    With no array the encoded direction is (0.5, 0.5) -> d == 0 -> z ==
    1, which makes generic_directional_lightmap() return
    A + (irr - A) * max(0, n.z) with A = irr*clamp(0 + bias). That is
    the reference's own isotropic case, reached through the reference's
    own arithmetic, not a bypass -- but it IS a substitution and it is
    printed rather than taken silently.
    """
    if GEN_LMDIR is None or u is None or v is None:
        _lmg_note("generic.vfx: no --lmg-lm-direction, so the third "
                  "lightmap array (r0_m3:270) reads (0.5,0.5,0), the "
                  "encoded zero direction; the directional "
                  "reconstruction degenerates to the isotropic lightmap")
        return None
    tap = _gt_bicubic4 if quality else _gt_tex2d
    return tap(GEN_LMDIR, u, v).float()


def gt_baked_lightmap(u, v, quality):
    """The two co-registered lightmap sampler2DArray fetches.

    csgo_environment_ps.glsl:756-758 --

        vec3 uvw = vec3(vLightmapUV * g_vLightmapScale, 0.0);
        vec4 lm0 = texture(sampler2DArray(A0, s), uvw);   // irradiance
        vec4 lm1 = texture(sampler2DArray(A1, s), uvw);   // occlusion

    Both at the SAME coordinate and at the literal array slice 0.0
    (SHADER_CALLFLOW_csgo_static_overlay.md:244-256: "the array slice is
    the literal constant 0.0. The shader never computes a slice").

    lm0.xyz is consumed as the baked irradiance directly, with NO normal
    dependence (:1227 `vec3 _8745 = _21760.xyz;` then :1230, :1240,
    :1251). lm1 is consumed only as a per-light occlusion vector, never
    as a colour: :888 `1.0 - dot(lm1, g_vSunOcclusionMask)` and :1067
    `1.0 - dot(lm1, light[i].m10)`.

    ASSET MAPPING: lm0 is --irr-npy (irradiance.vtex, CS2's
    lighting.DiffuseIndirect); lm1 is --gt-lm-occlusion. With no lm1 the
    occlusion is 0, the shader's unbound value, so 1 - dot(.) is 1.
    --lightmap's direct_light_shadows is a SCALAR the pre-existing path
    multiplies into ndl; it is not lm1's vec4 and is not silently reused
    as one.
    """
    if IRRMAP is None:
        _gt_note("lightmap: --irr-npy not given; "
                 "D_BAKED_LIGHTING_FROM_LIGHTMAP has no irradiance array "
                 "to fetch")
        return None, None
    if u is None or v is None:
        # No lightmap UV set was interpolated for this pass, so the
        # coordinate the two fetches share (:756) does not exist. The
        # caller falls back to the shipped dyn-0 module, which is a real
        # variant; it does NOT quietly return an unlit surface.
        _gt_note("lightmap: no lightmap UV set on this geometry; the "
                 "lightmap fetches have no coordinate and the shipped "
                 "dyn-0 module runs instead")
        return None, None
    # ⚠️ THE BICUBIC IS ON THE OCCLUSION PAGE ONLY, AND IT IS GATED (#57).
    # ------------------------------------------------------------------
    # This used to be `tap = _gt_bicubic4 if quality else _gt_tex2d`,
    # applied to BOTH pages. The q=1 shader does not do that:
    #
    #   :809  irradiance  _4565._m1 (offset 16)  -- plain texture()
    #   :817  direction   _4565._m2 (offset 20)  -- plain texture(), q1-only
    #   :839  occlusion   _4565._m3 (offset 24)  -- textureGrad x4 B-spline,
    #                                               inside `if (_21710)`
    #   :866  occlusion   ...                    -- plain texture() otherwise
    #
    # and offset 24 is the SAME page q=0 calls `_m2`. So the reference
    # bicubics one page, conditionally, and bilinear-taps the irradiance
    # at BOTH qualities -- while we were low-passing the DOMINANT light
    # term at q=1 and not at q=0. That is a q=1-only loss on every
    # lightmapped pixel, and lightmapped is 95 of 153 materials here.
    # It is suspect #2 behind #57, after the DFG transposition.
    #
    # THE GATE, :277-286, is `_4565._m0.z > _5037._m18` (a lightmap
    # DENSITY against a threshold) AND `min(fwidth(lmUV.z),
    # fwidth(lmUV.w)) > 0.1`. The second half needs screen-space
    # derivatives of the lightmap UV's SECOND pair, which this pack does
    # not carry -- so --gt-lm-bicubic exposes the choice and the banner
    # says the precondition is unevaluated rather than implying it held.
    irr = _gt_tex2d(IRRMAP, u, v).float()
    if _gt_inject("lightmap-zero"):
        irr = torch.zeros_like(irr)
    if GT_LM_OCC is not None:
        occ_bicubic = bool(quality) and args.gt_lm_bicubic == "on"
        occ = (_gt_bicubic4 if occ_bicubic else _gt_tex2d)(
            GT_LM_OCC, u, v).float()
        _gt_note(f"lightmap: occlusion page filter is "
                 f"{'B-spline bicubic (:839)' if occ_bicubic else 'bilinear (:866)'}"
                 f" at q={quality}; the irradiance page is bilinear at BOTH "
                 f"qualities (:809). The :277-286 density/fwidth gate is "
                 f"NOT evaluated -- this pack carries no second lightmap "
                 f"UV pair to take fwidth of.")
    else:
        _gt_note("lightmap: --gt-lm-occlusion not given; the second "
                 "sampler2DArray (csgo_environment_ps.glsl:758) reads 0, "
                 "the shader's unbound value, so 1 - dot(occ, mask) = 1")
        occ = torch.zeros(*irr.shape[:-1], 4, device=irr.device)
    # THE TINT IS NOT APPLIED HERE. It used to be, and the comment that
    # stood here said in its own words that the shader applies it AFTER
    # the AO multiply -- which is a different value, because pow is not
    # linear: pow(irr * ao) != pow(irr) * ao. gt_compose() applies it at
    # the position both files put it; see _gt_lm_tint(). This function
    # only records that the lightmap module ran, which is the condition
    # the tint was scoped to before and still is.
    global GT_LM_TINT_LIVE
    GT_LM_TINT_LIVE = bool(args.gt_lm_tint)
    return irr, occ


# ---------------------------------------------------------------------
# D_BAKED_LIGHTING_FROM_VERTEX_STREAM
# COLOR1 `PerVertexLighting`.  SHADER_CALLFLOW_vertex_stages.md:55-71:
# "the data does arrive, just not through a sampler ... baked lighting
# under D_BAKED_LIGHTING_FROM_VERTEX_STREAM, which is a *vertex stream*
# -- an attribute -- not a texture", and :165-168: COLOR0 is
# VertexPaintTintColor, COLOR1 is PerVertexLighting, "supplying one where
# the other is expected is a silent error".
# ---------------------------------------------------------------------
def gt_bake_vertex_stream():
    """Build the COLOR1 attribute this world pack does not carry.

    The glTF pack exposes COLOR_0 only (`vcolor`), which is
    VertexPaintTintColor -- the WRONG semantic, and the one the vertex
    inventory calls a silent error to substitute. --vertex-color-mode
    drives that COLOR_0 tint and is left alone.

    So the stream is baked here from the same quantity Valve's baker
    reads: the baked irradiance evaluated AT THE VERTEX. Lightmapped
    vertices take it from the irradiance atlas at their own lightmap UV;
    every other vertex takes it from the reconstructed irradiance volume
    if --irr-vol built one, else from the constant tail. Then it is
    interpolated across the triangle by the rasteriser, which is what
    makes this axis structurally different from the lightmap path
    (0 pixel-rate fetches, and 21 FLOPs cheaper than dyn 0 because it
    replaces the three dot4 of gt_ambient_sh -- exactly the delta in
    SHADER_CALLFLOW_csgo_environment.md:551-566).

    WHAT DIFFERS: CS2 bakes COLOR1 offline per vertex against the full
    GI solution. This bakes it against the same solution's *lightmap*
    projection, which is the only form of it this repo holds. The
    interpolation, the pixel-rate cost and the frequency content are the
    shader's; the values are a resample of Valve's bake, not a read of
    Valve's COLOR1 stream.
    """
    if args.gt_vertex_lighting:
        vs = torch.from_numpy(
            np.load(args.gt_vertex_lighting)).to(device).float()
        assert vs.shape[0] == len(vertices), (
            f"--gt-vertex-lighting has {vs.shape[0]} rows, world has "
            f"{len(vertices)} vertices")
        print(f"GT COLOR1 PerVertexLighting: {tuple(vs.shape)} from "
              f"{args.gt_vertex_lighting}", flush=True)
        return vs[:, :3].contiguous()
    out = torch.zeros(len(vertices), 3, device=device)
    got = torch.zeros(len(vertices), dtype=torch.bool, device=device)
    if IRRMAP is not None:
        sel = lm_valid > 0.5
        hh, ww = IRRMAP.shape[0], IRRMAP.shape[1]
        uu = (lmuv[sel, 0].clamp(0, 1) * (ww - 1)).long()
        vv = (lmuv[sel, 1].clamp(0, 1) * (hh - 1)).long()
        out[sel] = IRRMAP[vv, uu].float()
        got |= sel
    if VOL is not None:
        need = ~got
        if bool(need.any()):
            out[need] = sample_volume(
                vertices[need].view(1, 1, -1, 3)).view(-1, 3)
            got |= need
    if not bool(got.all()):
        # the shipped dyn-0 tail, evaluated on the vertex normal
        need = ~got
        vn = vnormals[need] if "vnormals" in globals() else None
        if vn is None:
            vn = torch.zeros(int(need.sum()), 3, device=device)
            vn[:, 1] = 1.0
        out[need] = gt_ambient_sh(vn)
        _gt_note(f"vertex-stream: {int(need.sum())} of {len(vertices)} "
                 f"vertices had no baked source and took the dyn-0 "
                 f"constant tail")
    print(f"GT COLOR1 PerVertexLighting BAKED at load: "
          f"{float((out.mean(1) > 0).float().mean()):.1%} of "
          f"{len(vertices)} vertices non-zero, mean "
          f"{float(out.mean()):.4f}", flush=True)
    return out.contiguous()


# ---------------------------------------------------------------------
# S_SHADER_QUALITY -- the shadow filter
# q=0: csgo_complex_ps.glsl:570 / :584 / :885  -- one tap each.
# q=1: csgo_complex_ps_shaderquality1.glsl:634+649, 669+684, 997+1012
#      -- three NINE-tap kernels, each emitted as a 4-tap group and a
#      5-tap group, with an early exit between them.
# ---------------------------------------------------------------------
def _gt_shadow_tap(depth, u, v, ref):
    """One sampler2DShadow DrefExplicitLod: 1 where the ref is in front."""
    ht, wt = depth.shape[0], depth.shape[1]
    x = (u * (wt - 1)).round().long().clamp(0, wt - 1)
    y = (v * (ht - 1)).round().long().clamp(0, ht - 1)
    return (ref <= depth[y, x]).float()


def gt_pcf(depth, u, v, ref, quality):
    """The shader's shadow filter at the selected S_SHADER_QUALITY.

    q=0 -- one tap:
        textureLod(sampler2DShadow(atlas, s), vec3(uv, ref), 0.0)

    q=1 -- shaderquality1.glsl:634-649, transcribed:
        // 4-tap group: the four corners, offsets from two vec4 uniforms
        c = dot(vec4(T(uv + (m2.z, m3.z)), T(uv + (m2.y, m3.z)),
                     T(uv + (m2.z, m3.y)), T(uv + (m2.y, m3.y))),
                vec4(0.25));
        if (c == 0.0 || c == 1.0) return c;                      // :635-648
        // 5-tap group: the four edges and the centre
        return c * (m0.w * 4.0)
             + dot(vec4(T(uv + (m2.z, 0)), T(uv + (m2.y, 0)),
                        T(uv + (0, m3.y)), T(uv + (0, m3.z))), m1.xxxx)
             + T(uv) * m1.y;                                     // :649

    c * (m0.w * 4) is the four corner taps re-weighted from 1/4 each to
    m0.w each, so the kernel weights are corner=m0.w, edge=m1.x,
    centre=m1.y -- a separable 3x3. m2.y/m2.z and m3.y/m3.z are the two x
    and two y offsets; the edge taps pair each with 0, which is only
    consistent with a {-texel, 0, +texel} grid.

    NOT IN THE ASSET: the three weights and the two offsets are uniforms
    no committed artefact carries. --gt-pcf-weights defaults to the
    separable binomial (1-2-1)x(1-2-1)/16 = corner 1/16, edge 2/16,
    centre 4/16, which is the unique 3x3 separable kernel that sums to 1
    over these three tap classes; the offsets are +/- one texel.

    The early exit is NOT a pure optimisation and is not treated as one:
    a 4-corner mean and a 9-tap binomial mean are different estimators,
    so returning the 4-tap value when it is exactly 0 or exactly 1
    changes the result on those texels as well as the cost. It is ported
    because it is in the call flow, and --gt-pcf-early-out 0 forces all
    nine taps for anyone who wants to see the difference.
    """
    if not quality:
        return _gt_shadow_tap(depth, u, v, ref)
    wc, we, wk = args.gt_pcf_weights
    sx = 1.0 / max(depth.shape[1] - 1, 1)
    sy = 1.0 / max(depth.shape[0] - 1, 1)
    # 4-tap corner group (:634)
    c = 0.25 * (_gt_shadow_tap(depth, u - sx, v - sy, ref)
                + _gt_shadow_tap(depth, u + sx, v - sy, ref)
                + _gt_shadow_tap(depth, u - sx, v + sy, ref)
                + _gt_shadow_tap(depth, u + sx, v + sy, ref))
    # 5-tap group (:649)
    e = (_gt_shadow_tap(depth, u - sx, v, ref)
         + _gt_shadow_tap(depth, u + sx, v, ref)
         + _gt_shadow_tap(depth, u, v + sy, ref)
         + _gt_shadow_tap(depth, u, v - sy, ref))
    full = c * (wc * 4.0) + e * we + _gt_shadow_tap(depth, u, v, ref) * wk
    if _gt_inject("pcf-collapse"):
        return c
    if not args.gt_pcf_early_out:
        return full
    done = (c == 0.0) | (c == 1.0)                              # :635-648
    _gt_count("pcf9.early_out", done)
    _gt_count("pcf9.full", ~done)
    return torch.where(done, c, full)


GBUF = []
SKYSTAT = [0.0, 0.0] if args.skyvis else None
# Reachability counters for the transparency axes. Every one of them can
# turn out to affect zero pixels for a reason that is NOT "the flag is
# off" -- no material carries the feature, the peel found one layer, the
# clip never fires -- and a silent zero is indistinguishable from a
# disabled path. These are printed at the end of the run, unconditionally
# whenever the flag is on, so "no delta" can be told from "no pixels".
FADE_STAT = [0.0, 0.0]          # discarded, considered
CLIP_STAT = [0.0, 0.0]          # clipped glass px, glass px
MBOIT_STAT = [0, 0, 0]          # max layers, chunks, chunks that hit the cap
# [attributed, multi-class pixels, dominant != nearest, below the old 0.5
# resolve threshold]. The MBOIT route is the one composite in this file
# where MORE THAN ONE class writes colour into a pixel, so the single-owner
# attribution map has to choose, and these four numbers say what the choice
# cost, plus a fifth: pixels that reached `fg` with NO layer above the
# contribution floor, which stay UNATTRIBUTED on purpose rather than being
# credited to whatever material id the tracker happened to start at.
MBOIT_ATTR = [0, 0, 0, 0, 0]
REFRACT_STAT = [0.0, 0.0]       # refracting px, blend px
TL_STAT = [0.0, 0.0, 0.0]       # translucent px, px whose alpha MOVED, blend px
AD_STAT = [0.0, 0.0, 0.0]       # additive px, blend px, px that SURVIVED the depth test with alpha>0.01
REACH = {}
GT_REACH = {}
FAM_HIST = []
# --- THE CLASS COVERAGE MATRIX ------------------------------------------
# One table, at the end of the run, answering the only question that says
# whether the renderer is complete: for every shader class the map's
# geometry actually USES, did pixels of that class reach the frame?
#
# The counters this consolidates already existed -- REACH, LMG_REACH,
# FAM_HIT, WATER_REACH, VM_REACH -- each printed in its own format, in its
# own place, with no verdict and no denominator. That is how a class with
# real population can draw nothing and read as normal: nobody was holding
# the two halves together. Population comes from the PACK (face census via
# MAT_FAM), pixels come from the RUN, and the verdict is the join.
#
# THE HONEST VERDICT IS THREE-VALUED, and the middle one is the point.
# A class that draws nothing in ONE camera view is not necessarily broken
# -- it may simply not be on screen -- so a single frame cannot separate
# "absent from view" from "cannot draw". It is reported as UNSEEN, not as a
# defect, and the machine-readable line lets a corpus sweep turn "UNSEEN in
# every frame of the corpus" into the CLASS BUG it then is. Calling a
# one-frame zero a defect would cry wolf on every indoor frame; calling it
# a pass would be the silence this exists to end.
CLASS_PX = {}
# per-asset animation state, filled by the bake loader and the runtime
# scripter as those land; empty means the stage has not run, which the
# matrix reports as a zero rather than as silence.
ANIM_DECODED = {}
ANIM_DRIVEN = {}
PM_SEQUENCES = []
VM_CLIPS = [None]        # the loaded ClipSet, or None
VM_SCRIPT = [_anim.Scripter()]
VM_PREV_ROW = [None]
VM_ANIM_CLIP = [None]
VM_ANIM_T = [0.0]
# One (clip, t) per BATCH ROW, written by _vm_anim_report and consumed by
# _vm_repose. The batch used to hold a single pose; this is what makes the
# re-pose per-frame.
VM_ANIM_SEQ = [None]
VM_ANIM_POSES = [1]     # distinct poses computed for the last batch
VM_ANIM_SAME = [False]  # inside one batch's stack of rows
VM_ANIM_MOVED = [0.0]
VM_TIER_NOTE = []                 # the animation-tier line prints once
VM_REPOSE_NOTE = []
_VM_BUNDLE_RAW = [None]
CAM_ROW = []      # this frame's demo row, set by the camera walk
