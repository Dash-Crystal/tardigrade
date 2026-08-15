

def probe_of(wpos):
    """World position -> probe index, via the precomputed voxel grid.

    A SINGLE-ENVIRONMENT CUBE HAS NO GRID TO LOOK UP, and before this
    guard `--ibl-cube` without `--cube-probes` died thirty thousand lines
    in with `NameError: name 'PLO' is not defined` -- a precondition that
    was real, undeclared, and reported as a code fault rather than a
    missing input. The sky-cube chain is one environment for the whole
    map (P = 1), so index 0 is not a fallback, it is the only answer
    there is; the approximation lives in the CUBE having one probe, and
    sky_cube_to_ibl.py's header is where that is argued.
    """
    if "PGRID" not in globals():
        if CUBE is not None and CUBE[0].shape[0] == 1:
            return torch.zeros(wpos.shape[:-1], dtype=torch.long,
                               device=wpos.device)
        raise SystemExit(
            "probe_of() needs either --cube-probes (a probe grid) or an "
            "--ibl-cube with exactly one environment. This cube has "
            f"{CUBE[0].shape[0] if CUBE is not None else 0}.")
    g = ((wpos - PLO) * PINV).long()
    g0 = g[..., 0].clamp(0, PGRID.shape[0] - 1)
    g1 = g[..., 1].clamp(0, PGRID.shape[1] - 1)
    g2 = g[..., 2].clamp(0, PGRID.shape[2] - 1)
    return PGRID[g0, g1, g2].long()


def sample_volume(wpos):
    """Trilinear lookup of the reconstructed indirect field, (B,H,W,3)."""
    g = (wpos - VOL_LO) * VOL_INV
    d = torch.tensor([VOL.shape[2] - 1, VOL.shape[3] - 1, VOL.shape[4] - 1],
                     device=device, dtype=torch.float32).clamp(min=1)
    # grid_sample wants (x=last axis .. z=first axis) in [-1,1]
    n = (g / d * 2 - 1).clamp(-1, 1)
    grid = torch.stack([n[..., 2], n[..., 1], n[..., 0]], dim=-1)
    B = grid.shape[0]
    out = torch.nn.functional.grid_sample(
        VOL.expand(B, -1, -1, -1, -1), grid.unsqueeze(1),
        mode="bilinear", align_corners=True, padding_mode="border")
    return out.squeeze(2).permute(0, 2, 3, 1)
SKYVIS = None
SKYVIS_LO = SKYVIS_INV = None
SKYVIS_REF = 1.0
if args.skyvis:
    import numpy as _np3
    _z = _np3.load(args.skyvis)
    SKYVIS = torch.from_numpy(_z["field"]).to(device).float()
    SKYVIS_LO = torch.from_numpy(_z["lo"]).to(device).float()
    SKYVIS_INV = 1.0 / float(_z["vox"])
    if args.skyvis_renorm and args.skyvis_ref <= 0:
        # Silently falling back to the npz's global mean here divided the
        # ambient by ~3x too much and produced an 18% too-dark render
        # whose only symptom was a printed number. Refuse instead.
        raise SystemExit(
            f"--skyvis-renorm 1 requires an explicit --skyvis-ref.\n"
            f"  The npz's baked ref ({float(_z['ref']):.4f}) is the "
            f"field's GLOBAL mean over the whole map and is NOT the "
            f"renormaliser.\n"
            f"  You want the mean over the RENDERED pixels of THIS pose "
            f"set. Get it by running once with --skyvis-renorm 0 and "
            f"reading the 'skyvis: mean over N rendered geometry pixels' "
            f"line, then pass that as --skyvis-ref.")
    SKYVIS_REF = args.skyvis_ref if args.skyvis_ref > 0 else float(_z["ref"])
    for _ax in args.skyvis_flip:
        SKYVIS = SKYVIS.flip({"x": 2, "y": 3, "z": 4}[_ax])
    print(f"sky visibility field {tuple(SKYVIS.shape[2:])} @ {float(_z['vox'])}m, "
          f"{int(_z['dirs'])} dirs / {float(_z['rng'])}m range, ref "
          f"{SKYVIS_REF:.4f} (npz {float(_z['ref']):.4f}), frac "
          f"{args.skyvis_frac}, renorm {args.skyvis_renorm}", flush=True)


def sample_skyvis(wpos):
    """Trilinear lookup of the sky-visibility field, (B,H,W)."""
    g = (wpos - SKYVIS_LO) * SKYVIS_INV
    d = torch.tensor([SKYVIS.shape[2] - 1, SKYVIS.shape[3] - 1,
                      SKYVIS.shape[4] - 1], device=device,
                     dtype=torch.float32).clamp(min=1)
    n = (g / d * 2 - 1).clamp(-1, 1)
    grid = torch.stack([n[..., 2], n[..., 1], n[..., 0]], dim=-1)
    B = grid.shape[0]
    out = torch.nn.functional.grid_sample(
        SKYVIS.expand(B, -1, -1, -1, -1), grid.unsqueeze(1),
        mode="bilinear", align_corners=True, padding_mode="border")
    return out.squeeze(2).squeeze(1)


# ======================================================================
# CLUSTERED ANALYTIC LIGHTS
# ======================================================================
LIGHTS = None
LOCC = None
LDUMP_ROWS = []


def _light_frame(d):
    """Orthonormal (dir, tangent, bitangent) per light, for barn doors."""
    up = torch.zeros_like(d)
    # pick the world axis least parallel to dir, so the cross is stable
    ax = d.abs().argmin(dim=1)
    up[torch.arange(d.shape[0], device=d.device), ax] = 1.0
    t = torch.cross(up, d, dim=1)
    t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-6)
    b = torch.cross(d, t, dim=1)
    return t, b


if args.lights:
    import json as _json
    _ents = _json.load(open(args.lights))
    _an = [e for e in _ents if e.get("cls") != "light_environment"]
    if args.lights_select == "direct1":
        _sel = [e for e in _an if str(e.get("directlight")) == "1"]
    elif args.lights_select == "direct":
        _sel = [e for e in _an if str(e.get("directlight")) != "0"]
    else:
        _sel = _an
    if not _sel:
        raise SystemExit("--lights selected zero entities")
    import numpy as _np

    def _col(k, dflt=0.0):
        return _np.array([float(e.get(k, dflt) or 0.0) for e in _sel],
                         _np.float64)

    _pos = _np.array([e["pos"] for e in _sel], _np.float64)
    _dir = _np.array([e["dir"] for e in _sel], _np.float64)
    _dir = _dir / _np.maximum(_np.linalg.norm(_dir, axis=1, keepdims=True),
                              1e-9)
    _rgb = _np.array([e["color"] for e in _sel], _np.float64)
    _units = _np.array([str(e.get("brightness_units", "1")) for e in _sel])
    _nits, _cd = _col("brightness_nits"), _col("brightness_candelas")
    _lm, _leg = _col("brightness_lumens"), _col("brightness_legacy")
    _bev = _col("brightness")
    if args.lights_unit == "candelas":
        _I = _cd
    elif args.lights_unit == "nits":
        _I = _nits
    elif args.lights_unit == "lumens":
        _I = _lm / (4.0 * math.pi)
    elif args.lights_unit == "legacy":
        _I = _leg
    elif args.lights_unit == "ev":
        _I = _np.exp2(_bev)
    else:
        # PER-LIGHT selector. The mapping units==1 -> nits, units==0 ->
        # 2^brightness is an INFERENCE from the two groups' distributions
        # (units==0: nits med 8.5, brightness med -1.87, i.e. an
        # EV-shaped exponent; units==1: nits med 352.5, brightness med
        # +0.07). It is not read from the engine and is here to be
        # falsified against the global choices, not assumed.
        _I = _np.where(_units == "1", _nits, _np.exp2(_bev))
    _I = _I * _col("brightnessscale", 1.0)
    # UNIT NORMALISATION, and it is the step that makes the brightness-
    # unit question ANSWERABLE.
    #
    # Every unit choice is in absolute photometric units while this
    # renderer's lighting is a fit in arbitrary ones, so a constant is
    # required either way. Left raw, the four fields differ in MEDIAN by
    # up to ~180x, so swapping --lights-unit would swap global exposure
    # and the A/B would measure the gain, not the unit. Each field is
    # therefore divided by ITS OWN median and multiplied by the CANDELAS
    # median, which
    #   (a) leaves --lights-unit candelas EXACTLY the raw candelas the
    #       lost implementation used, so --lights-gain 0.10 still means
    #       what the recorded numbers meant, and
    #   (b) makes every other unit equal-energy to it, so a unit A/B is
    #       a comparison of how brightness is DISTRIBUTED ACROSS LIGHTS
    #       -- the only thing the unit question is actually about.
    _cdm = float(_np.median(_cd[_cd > 0])) if (_cd > 0).any() else 1.0
    _med = float(_np.median(_I[_I > 0])) if (_I > 0).any() else 1.0
    _I = _I / max(_med, 1e-12) * _cdm
    _cls = _np.array([e["cls"] for e in _sel])
    _rng = _col("range_m", 1.0)
    _rng = _np.maximum(_rng, 0.05)
    _ia, _oa = _col("inner_angle", 180.0), _col("outer_angle", 180.0)
    _sk = _col("skirt", 0.0)
    _sp = _np.array([e.get("size_params", [0, 0, 0]) for e in _sel],
                    _np.float64)
    _kind = _np.where(_cls == "light_rect", 1,
                      _np.where(_cls == "light_barn", 2, 0)).astype(_np.int64)
    if args.lights_sabotage == "shuffle":
        # Same energy, same colours, same falloff, same count -- only the
        # PLACEMENT destroyed. A metric that cannot tell this apart from
        # the real arm cannot see the term at all.
        _g = _np.random.default_rng(20260807)
        _pos = _pos[_g.permutation(_pos.shape[0])]

    def _t(a, dt=torch.float32):
        return torch.as_tensor(a, dtype=dt, device=device)

    _half = 0.5 if args.lights_cone_angle == "full" else 1.0
    LIGHTS = {
        "pos": _t(_pos), "dir": _t(_dir), "rgb": _t(_rgb),
        "I": _t(_I), "rng": _t(_rng), "kind": _t(_kind, torch.long),
        "cos_in": _t(_np.cos(_np.radians(_np.minimum(_ia * _half, 90.0)))),
        "cos_out": _t(_np.cos(_np.radians(_np.minimum(_oa * _half, 90.0)))),
        "omni": _t((_oa >= 179.9).astype(_np.float64)),
        "skirt": _t(_sk),
        "barn": _t(_np.radians(_np.clip(_sp[:, :2], 1.0, 89.0))),
    }
    LIGHTS["tan"], LIGHTS["bit"] = _light_frame(LIGHTS["dir"])
    _ew = _I * _rgb.mean(1)
    _ew = _ew / max(_ew.sum(), 1e-12)
    print(f"analytic lights: {len(_sel)} of {len(_an)} "
          f"(select={args.lights_select}: omni2 {int((_kind==0).sum())} "
          f"rect {int((_kind==1).sum())} barn {int((_kind==2).sum())}), "
          f"unit={args.lights_unit} (own median {_med:.4g} -> candelas "
          f"median {_cdm:.4g}), falloff={args.lights_falloff}, "
          f"angular={args.lights_angular}, gain={args.lights_gain}, "
          f"sabotage={args.lights_sabotage}", flush=True)
    print(f"  range m: med {float(_np.median(_rng)):.2f} max "
          f"{float(_rng.max()):.2f} | energy-weighted chroma "
          f"{(_ew[:,None]*_rgb).sum(0)[0]:.3f} "
          f"{(_ew[:,None]*_rgb).sum(0)[1]:.3f} "
          f"{(_ew[:,None]*_rgb).sum(0)[2]:.3f}", flush=True)

if args.lights_shadow:
    import numpy as _np2
    _z2 = _np2.load(args.lights_shadow)
    _dm = _z2["dim"].astype(int)
    _bits = torch.from_numpy(
        _np2.unpackbits(_z2["occ"])[:int(_dm[0]) * int(_dm[1]) * int(_dm[2])]
    ).to(device).bool().view(int(_dm[0]), int(_dm[1]), int(_dm[2]))
    LOCC = {"occ": _bits, "lo": torch.from_numpy(_z2["lo"]).to(device).float(),
            "vox": float(_z2["vox"]),
            "dim": torch.tensor([int(_dm[0]) - 1, int(_dm[1]) - 1,
                                 int(_dm[2]) - 1], device=device)}
    print(f"light occupancy grid {tuple(_dm)} @ {LOCC['vox']}m, "
          f"{float(_bits.float().mean()):.2%} solid "
          f"({int(_bits.sum()):,} cells), bias "
          f"{args.lights_shadow_bias} vox / (N.L), step "
          f"{args.lights_shadow_step} vox, max "
          f"{args.lights_shadow_maxsteps} steps", flush=True)

if args.lights_enclosure is not None:
    # Per-light sky visibility, marched from the light itself. Same
    # estimator build_skyvis.py uses for surfaces, on 184 points instead
    # of millions, so it costs nothing and shares the field's semantics.
    if LOCC is None or LIGHTS is None:
        raise SystemExit("--lights-enclosure requires --lights and "
                         "--lights-shadow")
    _n = args.lights_enclosure_dirs
    _ii = torch.arange(_n, device=device, dtype=torch.float64) + 0.5
    _y = 1.0 - _ii / _n
    _r = (1 - _y * _y).clamp(min=0).sqrt()
    _ph = _ii * math.pi * (3.0 - math.sqrt(5.0))
    _dv = torch.stack([_r * torch.cos(_ph), _y, _r * torch.sin(_ph)],
                      1).float()
    _dv = _dv / _dv.norm(dim=1, keepdim=True)
    _P = LIGHTS["pos"]
    _step = 0.6 * LOCC["vox"]
    _ns = int(math.ceil(12.0 / _step))
    _acc = torch.zeros(_P.shape[0], device=device)
    for _k in range(_n):
        _al = torch.ones(_P.shape[0], dtype=torch.bool, device=device)
        _pos = _P.clone()
        for _s in range(_ns):
            _pos = _pos + _dv[_k].view(1, 3) * _step
            _g = ((_pos - LOCC["lo"]) / LOCC["vox"]).long()
            _ob = ((_g < 0) | (_g > LOCC["dim"].view(1, 3))).any(1)
            _g = torch.minimum(torch.clamp(_g, min=0),
                               LOCC["dim"].view(1, 3))
            _al = _al & ~(LOCC["occ"][_g[:, 0], _g[:, 1], _g[:, 2]] & ~_ob)
            if not bool(_al.any()):
                break
        _acc += _al.float()
    _sv = _acc / _n
    _keep = ((_sv >= args.lights_enclosure[0])
             & (_sv <= args.lights_enclosure[1]))
    _ki = _keep.nonzero().squeeze(1)
    print(f"light enclosure gate [{args.lights_enclosure[0]:.2f},"
          f"{args.lights_enclosure[1]:.2f}]: {int(_keep.sum())} of "
          f"{_P.shape[0]} lights kept; own sky visibility p10 "
          f"{float(_sv.quantile(0.1)):.3f} med {float(_sv.median()):.3f} "
          f"p90 {float(_sv.quantile(0.9)):.3f}", flush=True)
    if int(_keep.sum()) == 0:
        raise SystemExit("--lights-enclosure kept zero lights")
    for _k2 in ("pos", "dir", "rgb", "I", "rng", "kind", "cos_in",
                "cos_out", "omni", "skirt", "barn", "tan", "bit"):
        LIGHTS[_k2] = LIGHTS[_k2][_ki]


def _light_distance_falloff(d, rng):
    """RECONSTRUCTED. See the provenance note on --lights."""
    f = args.lights_falloff
    x = (d / rng).clamp(min=0)
    if f == "invsq-win":
        w = (1.0 - x.pow(4)).clamp(0, 1) ** 2
        return w / d.clamp(min=0.05).pow(2)
    if f == "invsq-clip":
        return (x < 1.0).float() / d.clamp(min=0.05).pow(2)
    if f == "invsq-soft":
        # softened near field: no singularity, same 1/d^2 tail
        w = (1.0 - x.pow(4)).clamp(0, 1) ** 2
        return w / (d + 1.0).pow(2)
    if f == "linear":
        return (1.0 - x).clamp(0, 1)
    if f == "smooth2":
        return (1.0 - x).clamp(0, 1) ** 2
    return torch.exp(-3.0 * x) * (x < 1.0).float()


def _light_angular(L, i):
    """Angular gate for light index i, given unit vector L from light to
    surface. RECONSTRUCTED throughout -- barn doors especially."""
    if args.lights_angular == "none":
        return torch.ones_like(L[..., 0])
    kind = int(LIGHTS["kind"][i])
    d = LIGHTS["dir"][i].view(*([1] * (L.dim() - 1)), 3)
    c = (L * d).sum(-1)
    if kind == 0:
        if args.lights_angular in ("cone", "cone-barn") and \
                float(LIGHTS["omni"][i]) < 0.5:
            ci = float(LIGHTS["cos_in"][i])
            co = float(LIGHTS["cos_out"][i])
            # skirt widens the penumbra rather than the cone
            sk = float(LIGHTS["skirt"][i])
            lo = co - sk * max(ci - co, 1e-3)
            t = ((c - lo) / max(ci - lo, 1e-4)).clamp(0, 1)
            return t * t * (3.0 - 2.0 * t)
        return torch.ones_like(c)
    if kind == 1:
        # light_rect: an area emitter, one-sided along its normal.
        return c.clamp(min=0)
    # light_barn: two independent angular gates in the light's own
    # tangent frame, softened by `skirt`. size_params[0:2] read as
    # DEGREES; they could equally be metres and that is not determinable
    # from the dump, which is exactly why --lights-angular cosine exists
    # as the ablation.
    if args.lights_angular != "cone-barn":
        return c.clamp(min=0)
    t = LIGHTS["tan"][i].view(*([1] * (L.dim() - 1)), 3)
    b = LIGHTS["bit"][i].view(*([1] * (L.dim() - 1)), 3)
    cz = c.clamp(min=1e-4)
    ax = torch.atan2((L * t).sum(-1).abs(), cz)
    ay = torch.atan2((L * b).sum(-1).abs(), cz)
    hx, hy = float(LIGHTS["barn"][i, 0]), float(LIGHTS["barn"][i, 1])
    sk = max(float(LIGHTS["skirt"][i]), 1e-3)
    gx = (1.0 - (ax - hx * (1 - sk)) / max(hx * sk, 1e-4)).clamp(0, 1)
    gy = (1.0 - (ay - hy * (1 - sk)) / max(hy * sk, 1e-4)).clamp(0, 1)
    return (c > 0).float() * gx * gy


def _march_occ(p0, p1):
    """1 = unoccluded, 0 = occluded. Fixed-step DDA through the packed
    occupancy grid from p0 to p1, both (N,3). FAIL-OPEN past the step
    cap: the cap can only under-shadow, never invent shadow."""
    v = p1 - p0
    dist = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    u = v / dist
    step = args.lights_shadow_step * LOCC["vox"]
    n = min(args.lights_shadow_maxsteps,
            int(math.ceil(float(dist.max()) / step)) + 1)
    alive = torch.ones(p0.shape[0], dtype=torch.bool, device=p0.device)
    lo, inv, dm = LOCC["lo"], 1.0 / LOCC["vox"], LOCC["dim"]
    occ = LOCC["occ"]
    for s in range(1, n + 1):
        t = s * step
        pos = p0 + u * t
        # stop one step short of the light itself
        inside = (t < (dist.squeeze(-1) - step))
        if not bool(inside.any()):
            break
        g = ((pos - lo) * inv).long()
        oob = ((g < 0) | (g > dm.view(1, 3))).any(-1)
        g = torch.minimum(torch.clamp(g, min=0), dm.view(1, 3))
        hit = occ[g[:, 0], g[:, 1], g[:, 2]] & (~oob) & inside
        alive = alive & (~hit)
        if not bool(alive.any()):
            break
    return alive.float()


def analytic_lights(wpos, nrm, mvp, want_back=False):
    """Sum of the analytic light entities' irradiance at every pixel.

    Returns (B,H,W,3) in the same arbitrary linear units the fitted
    ambient and sun use, i.e. it is added alongside SUN_COLOR*ndl and
    then multiplied by albedo.

    Culling is a per-CHUNK frustum test on each light's bounding sphere:
    only lights whose sphere intersects some frame's frustum are
    evaluated. Shadowing marches only pixels whose UNSHADOWED
    contribution already exceeds --lights-min-contrib, so the cost is
    the pixel-light pairs that matter rather than the 184 x frames
    product.
    """
    B, H, W = wpos.shape[:3]
    out = torch.zeros(B, H, W, 3, device=wpos.device)
    # S_TRANSMISSIVE_BACKFACE_NDOTL lives INSIDE this loop in the
    # reference (csgo_foliage s32/d0:532 sits in the same per-light body
    # as the diffuse term), so the back-facing sum is accumulated here in
    # the same pass rather than by a second call: the visibility march is
    # per light-pixel pair and running it twice would be both slower and,
    # worse, gated differently.
    back = torch.zeros(B, H, W, 3, device=wpos.device) if want_back else None
    P, R = LIGHTS["pos"], LIGHTS["rng"]
    # Gribb-Hartmann planes from each frame's MVP; a light is kept if it
    # is inside every plane by more than -range for at least one frame.
    m = mvp                                    # (B,4,4), row-vector conv
    planes = torch.stack([m[:, 3] + m[:, 0], m[:, 3] - m[:, 0],
                          m[:, 3] + m[:, 1], m[:, 3] - m[:, 1],
                          m[:, 3] + m[:, 2], m[:, 3] - m[:, 2]], dim=1)
    nl = planes[..., :3].norm(dim=-1, keepdim=True).clamp(min=1e-9)
    planes = planes / nl
    sd = (torch.einsum("bpk,nk->bpn", planes[..., :3], P)
          + planes[..., 3].unsqueeze(-1))          # (B,6,N)
    keep = (sd > -R.view(1, 1, -1)).all(dim=1).any(dim=0)
    sel = keep.nonzero().squeeze(1)
    if sel.numel() == 0:
        return (out, back) if want_back else out
    ndl_eps = 1e-3
    n_march = 0
    for i in sel.tolist():
        lp = P[i].view(1, 1, 1, 3)
        v = lp - wpos
        d = v.norm(dim=-1)
        near = d < float(R[i])
        if not bool(near.any()):
            continue
        Ldir = v / d.unsqueeze(-1).clamp(min=1e-6)
        ndl_signed = (nrm * Ldir).sum(-1)
        ndl = ndl_signed.clamp(min=0)
        att = _light_distance_falloff(d, float(R[i]))
        ang = _light_angular(-Ldir, i)
        common = (att * ang * float(LIGHTS["I"][i]) * near.float()
                  * args.lights_gain)
        s = ndl * common
        sb = (-ndl_signed).clamp(min=0) * common if want_back else None
        # One march covers both hemispheres of this light: gating the
        # march on the FRONT term alone would leave every back-lit pixel
        # unshadowed, which is a different feature, not a cheaper one.
        m_hit = (s if sb is None else torch.maximum(s, sb)) \
            > args.lights_min_contrib
        if not bool(m_hit.any()):
            continue
        if LOCC is not None:
            idx = m_hit.nonzero(as_tuple=False)
            n_march += idx.shape[0]
            p = wpos[idx[:, 0], idx[:, 1], idx[:, 2]]
            nn = nrm[idx[:, 0], idx[:, 1], idx[:, 2]]
            # |N.L|, not the clamped front-side cosine: the 1/(N.L) shell
            # correction below is a geometric statement about the angle
            # the ray leaves at, and a back-lit pixel leaves at the same
            # angle on the other side.
            nd = ndl_signed[idx[:, 0], idx[:, 1],
                            idx[:, 2]].abs().clamp(min=ndl_eps)
            # Conservative voxelisation inflates every surface to a
            # half-voxel shell, so at grazing incidence the start point
            # is INSIDE its own shell and a fixed normal offset cannot
            # lift it out. The offset a ray needs along the normal to
            # clear a shell of thickness h before travelling h along L
            # grows as 1/(N.L); that is the whole correction.
            off = (args.lights_shadow_bias * LOCC["vox"]
                   / nd.clamp(min=0.05)).clamp(max=8.0 * LOCC["vox"])
            # chunked purely so a light that reaches half the frame
            # across a 32-frame chunk cannot spike peak VRAM; the march
            # is independent per ray, so this changes nothing it returns
            p = p + nn * off.unsqueeze(-1)
            lpn = P[i].view(1, 3)
            vis = torch.cat([
                _march_occ(p[k:k + 4_000_000],
                           lpn.expand(min(4_000_000, p.shape[0] - k), 3))
                for k in range(0, p.shape[0], 4_000_000)])
            sv = torch.zeros_like(s)
            sv[idx[:, 0], idx[:, 1], idx[:, 2]] = vis
            s = s * sv
            if sb is not None:
                sb = sb * sv
        out = out + s.unsqueeze(-1) * LIGHTS["rgb"][i].view(1, 1, 1, 3)
        if sb is not None:
            back = back + sb.unsqueeze(-1) * LIGHTS["rgb"][i].view(1, 1, 1, 3)
    if args.lights_sabotage == "sign":
        out = -out
        if back is not None:
            back = -back
    return (out, back) if want_back else out


if args.lightmap:
    from PIL import Image as _Im
    import numpy as _np
    # direct_light_shadows channel 0 is CS2's BAKED SUN VISIBILITY
    # (mean 0.856, p05 0 / p95 1 -- a lit/shadowed mask at 8192^2).
    # directional_irradiance packs a light DIRECTION, not radiance, so it
    # is not a drop-in colour term and is left alone for now.
    _Im.MAX_IMAGE_PIXELS = None
    _p = f"{args.lightmap}/direct_light_shadows.png"
    _a = _np.asarray(_Im.open(_p), _np.float32)[..., 0] / 255.0
    LIGHTMAP = torch.from_numpy(_a).to(device)
    _q = f"{args.lightmap}/directional_irradiance.png"
    _b = _np.asarray(_Im.open(_q).convert("RGB"), _np.float32) / 255.0
    IRRADIANCE = torch.from_numpy(_b).to(device)
    print(f"baked irradiance {IRRADIANCE.shape[1]}x{IRRADIANCE.shape[0]}",
          flush=True)
    print(f"baked sun shadows {LIGHTMAP.shape[1]}x{LIGHTMAP.shape[0]}, "
          f"lit fraction {float((LIGHTMAP>0.5).float().mean()):.1%}",
          flush=True)
vcolor = data.get("vcolor")
vcolor = (torch.ones(len(vertices), 4, device=device) if vcolor is None
          else vcolor.to(device))
MTINT = None
if args.model_tint_table:
    _mt = torch.load(args.model_tint_table, map_location="cpu",
                     weights_only=True)
    MTINT = _mt["blend_off" if args.model_tint == "blend-off"
                else "model_tint"].to(device)
    print(f"model-tint gate: baseColorFactor suppressed on "
          f"{int((MTINT[:, 0] == 0).sum())} layer-1 and "
          f"{int((MTINT[:, 1] == 0).sum())} layer-2 slots", flush=True)
elif args.model_tint != "always":
    raise SystemExit("--model-tint flag needs --model-tint-table")

VPAINT = None
if args.vpaint:
    _vp = torch.load(args.vpaint, map_location="cpu", weights_only=True)
    if len(_vp["vpaint"]) != len(vertices):
        raise SystemExit(f"vpaint sidecar has {len(_vp['vpaint']):,} verts, "
                         f"pack has {len(vertices):,}; not index-aligned")
    VPAINT = _vp["vpaint"].to(device).float()
    print(f"vertex paint (_TEXCOORD_4.x) on {len(VPAINT):,} verts: "
          f"mean {float(VPAINT.mean()):.4f}, "
          f"{float((VPAINT == 0).float().mean()):.1%} at 0, "
          f"{float(((VPAINT > 0) & (VPAINT < 1)).float().mean()):.1%} partial",
          flush=True)
elif data.get("vpaint") is not None:
    # The pack carries the weight, so no sidecar is needed. Preferred over
    # --vpaint precisely BECAUSE it cannot be forgotten: a sidecar the
    # canonical invocation omits is loaded-or-not by accident, and that is
    # how every blended surface came to be shaded on COLOR_0 -- a stream
    # this file's own --blend-channel help already documents as the wrong
    # one. Packed data that travels with the geometry has no such failure.
    _vp = data["vpaint"]
    if len(_vp) != len(vertices):
        raise SystemExit(f"pack vpaint has {len(_vp):,} verts, geometry has "
                         f"{len(vertices):,}; not index-aligned")
    VPAINT = _vp.to(device).float()
    print(f"vertex paint (TEXCOORD_4.x) FROM PACK on {len(VPAINT):,} verts: "
          f"mean {float(VPAINT.mean()):.4f}, "
          f"{float((VPAINT == 0).float().mean()):.1%} at 0, "
          f"{float(((VPAINT > 0) & (VPAINT < 1)).float().mean()):.1%} partial",
          flush=True)
elif args.blend_weight == "vpaint":
    # AN EXPLICIT REQUEST IS HONOURED OR REFUSED. A DEFAULT MUST NOT BRICK
    # THE TOOL. Those are different obligations and collapsing them took
    # main down: the default moved to vpaint on the strength of a
    # measurement, the published packs do not carry the key yet, and every
    # agent running render_gt.sh against ~/worlds got exit 1 and no frames.
    #
    # So: if the operator typed --blend-weight vpaint, they get the refusal
    # -- silently substituting a stream this file documents as the WRONG
    # one would be the silent-wrong-render class we keep closing. If it is
    # merely the default and the pack predates it, fall back and SAY SO,
    # loudly, every run, naming the pack and the remedy. That is a stated
    # divergence, not a silent one.
    if "--blend-weight" in sys.argv:
        raise SystemExit(
            "--blend-weight vpaint needs --vpaint or a pack that carries a "
            "vpaint key. This pack has neither. Re-pack with a current "
            "ash_to_world.py (it binds TEXCOORD_4.x), or pass --vpaint, or "
            "ask for --blend-weight color0 explicitly.")
    args.blend_weight = "color0"
    print("=" * 70, flush=True)
    print("WARNING: this pack carries no `vpaint` key, so the layer-2 blend "
          "weight\n         falls back to COLOR_0 -- a per-vertex TINT, "
          "which is NOT the\n         weight. Measured cost on pose 0 of "
          "de_inferno, lit, vs GT:\n         ncc -0.0449 -> -0.1468, ground "
          "-0.1614 -> -0.3033, MSE\n         0.04658 -> 0.05550. Blended "
          "ground is wrong in this render.\n         Fix: re-pack with a "
          "current ash_to_world.py, which binds\n         TEXCOORD_4.x and "
          "ships it in the pack.", flush=True)
    print("=" * 70, flush=True)
# EACH OF THESE FOUR DEFAULTS INDEPENDENTLY. They used to be gated as a
# group on face_mat2 alone, which cost twice in opposite directions.
#
# A packer supplying only the members it had data for left the others None
# and the run died on `face_sec_0.to(device)` -- an absence test on one
# member cannot speak for four.
#
# Then, worse and silently: a packer that DID supply all four had to
# materialise the constants this branch would have built for free.
# face_params is 15 floats per face, so on de_boulder that is 1,173 MB of
# all-ones -- the largest tensor in the pack, a third of it. And the world
# is loaded with torch.load(map_location=device), so every byte of it is
# resident in VRAM whether the run reads the key or not. That is 1.19 GB
# of VRAM per big map spent transmitting a default, and every check we had
# passed it, because all-ones is a perfectly valid tensor.
#
# Defaulting per key lets a pack ship face_mat2/face_has2 -- which are
# genuinely per-material -- and omit the two that are constant.
_n = len(faces0)
_D2 = (("face_mat2", lambda: torch.zeros(_n, dtype=torch.int32)),
       ("face_has2", lambda: torch.zeros(_n, dtype=torch.int8)),
       ("face_secondary", lambda: torch.zeros(_n, dtype=torch.int8)),
       ("face_params", lambda: torch.ones(_n, 15)))
_g2 = {}
for _k2, _mk in _D2:
    _v2 = data.get(_k2)
    _g2[_k2] = _mk() if _v2 is None else _v2
_miss = [k for k, _ in _D2 if data.get(k) is None]
if _miss:
    print(f"world pack omits {', '.join(_miss)}; defaulted per key "
          f"({_n:,} faces)", flush=True)
face_mat2_0 = _g2["face_mat2"].to(device).long()
face_has2_0 = _g2["face_has2"].to(device).float()
face_sec_0 = _g2["face_secondary"].to(device).float()
face_params_0 = _g2["face_params"].to(device)
_fm = data.get("face_alpha_mode")
face_mode0 = (torch.zeros(len(faces0), dtype=torch.long, device=device)
              if _fm is None else _fm.to(device).long())
_fc = data.get("face_alpha_cutoff")
face_cut0 = (torch.full((len(faces0),), 0.5, device=device)
             if _fc is None else _fc.to(device))

# --- Source 2 shader-family side table ---------------------------------
# --- GROUND-TRUTH MATERIAL-SURFACE COMBO AXES --------------------------
# Every one of these rides on the fast_pack_fam.py side table, so they
# turn FAMILY on exactly the way the older family features do. They are
# listed once, here, so a new axis cannot be added to the parser and then
# silently never reach the shading path.
GT_AXES = {
    "S_DETAIL_TEXTURE": args.detail_texture,
    "S_DETAIL_NORMAL": args.detail_normal,
    # `!= "off"`, never the bare value: --secondary-uv is a STRING flag and
    # "off" is truthy, so the bare form made GTSURF -- and through it
    # FAMILY -- unconditionally true, which turned "--fam-side is required"
    # into a hard error for every invocation of this file, including the
    # ones that never asked for a family. That is the exact outcome the
    # comment on the LMG entry below says must not happen. Every other one
    # of this flag's ~20 uses already compares it by value.
    "S_SECONDARY_UV": args.secondary_uv != "off",
    "S_DECAL_TEXTURE": args.decal_texture,
    "S_ANISOTROPIC_GLOSS": args.aniso_gloss,
    "S_ENABLE_LAYER_3": args.layer3,
    "S_TEXTURE_ANIMATION": args.texture_animation,
    "S_BLEND_EFFECTS": args.blend_effects,
    "S_BLEND_EFFECTS_2": args.blend_effects_2,
    "S_BLEND_EFFECTS_3": args.blend_effects_3,
    "S_USE_NEW_BLENDING": args.use_new_blending,
    "S_ENABLE_VISUALIZATIONS": args.enable_visualizations,
    "S_ALPHA_TEST": args.alpha_test_gt,
    "S_ALPHA_TEST_LAYER": args.alpha_test_layer,
    "S_MATERIAL_REFERENCE": args.material_reference,
    # These four already had a flag; the flag now drives the GT form and
    # reads the side table's parameters, so they join the same gate.
    "S_TINT_MASK": args.tint_mask,
    "S_METALNESS_TEXTURE": args.metalness_tex,
    "S_SELF_ILLUM": args.self_illum,
    "S_SHARED_COLOR_OVERLAY": args.overlay,
    "S_PAINT_VERTEX_COLORS": args.paint_vertex_colors,
    "S_BLEND_MODE": args.overlay_blend,
}
# DE-ALIASED. GTSURF used to be exactly `any(GT_AXES.values())`, so the
# only way to reach it was to ask for an unrelated surface feature --
# `--overlay` turned on the GT-surface path for the WHOLE FRAME as a side
# effect, and an arm meant to price the overlay priced both. Same class as
# --vmat-ao riding --normal-map. `auto` keeps the historical coupling so
# no existing command line changes meaning; `on` and `off` make the axis
# addressable on its own, which is what pricing it requires.
#
# WHAT GTSURF IS, read off the dispatch rather than assumed: it is not a
# shading MODE, it is "the EXT2 side table is loaded". :26930 is the whole
# mechanism -- `x_gt = MAT_EXT2[mid_op] if GTSURF else None` -- and every
# gt_* transcription takes x_gt. With it off those functions cannot read
# their material parameters, so they do not run; they are ABSENT, not
# substituted. Nearly every `if GTSURF:` interior is further gated by its
# own feature flag, so turning GTSURF on alone changes nothing by itself.
# The one exception on the overlay path is :27033, which swaps
# gt_tint_mask_value() for tint_mask_value() whenever `--tint-mask or
# --overlay` -- that swap is unconditional and is the remaining candidate
# for the A->C move; see SLOT_FAMILY_PRICES.md for the three candidates
# already refuted.
#
# THE DEFAULT IS NOW ON WHENEVER THE SIDE TABLE IS PRESENT, signed off
# 2026-08-09 on IDENTITY grounds and not on a metric column. GTSURF off
# does not select an alternative shading model -- it leaves the ported
# reference functions without their parameters, so they do not run. A
# default that silently deletes transcribed reference functions is the
# opposite of what this renderer is for, whatever any score says.
#
# THE SCORE SAYS SOMETHING MIXED AND IT IS PUBLISHED HERE RATHER THAN
# OMITTED. Five frames of the de_inferno pair, de-aliased, one variable
# (SLOT_FAMILY_PRICES.md): nrmse ON better 5 of 5, -0.0215 to -0.3133,
# mean -0.1882; KL better 3 of 5 and flat; **ncc WORSE on 4 of 5**,
# -0.0017 to -0.0134. ncc is contrast- and brightness-invariant, so that
# pattern reads as the level improving while the structure degrades
# slightly. The default is not claimed to win on ncc. It is claimed to be
# the honest one.
#
# A run that loads no side table is unaffected: there is nothing to bind,
# so `auto` still resolves OFF and nothing changed for it.
_GTSURF_AUTO = any(GT_AXES.values()) or args.gt_axes_reach
_GTSURF_SIDE = bool(args.fam_side or args.char_side)
GTSURF = (_GTSURF_AUTO or _GTSURF_SIDE) if args.gt_surface == "auto" \
    else (args.gt_surface == "on")
if args.gt_surface == "auto" and _GTSURF_SIDE and not _GTSURF_AUTO:
    print("--gt-surface auto: ON because a side table is present and the "
          "GT-surface path's parameters can therefore be bound. OFF would "
          "leave the ported reference functions without inputs, i.e. "
          "delete them; that is why this is the default despite ncc "
          "measuring worse on 4 of 5 frames (SLOT_FAMILY_PRICES.md).",
          flush=True)
if args.gt_surface != "auto" and GTSURF != _GTSURF_AUTO:
    print(f"--gt-surface {args.gt_surface}: GT-surface path forced "
          f"{'ON' if GTSURF else 'OFF'}, against the {_GTSURF_AUTO} the "
          f"selected axes would have implied. This is the de-aliasing "
          f"lever: with it OFF and --overlay on, the overlay composites "
          f"from the world pack's mat_ext instead of the side table's "
          f"EXT2, which is the same read expression either way.",
          flush=True)
# --- combo-graph constraints, applied where they are DECLARED ----------
# RENDER_SETTINGS_AXES.md §4 is the shipped rule set, validated by set
# equality against the shipped combo IDs. The flag-level ones are applied
# here so a run cannot be configured into a variant the shader does not
# ship; the material-level ones are per-pixel gates in the shading.
if args.blend_effects_3 and not args.layer3:
    # eb:ps REQUIRES S_BLEND_EFFECTS_3 != 0 -> S_ENABLE_LAYER_3 != 0.
    args.layer3 = True
    GT_AXES["S_ENABLE_LAYER_3"] = True
    print("S_BLEND_EFFECTS_3 requires S_ENABLE_LAYER_3 (eb:ps rule); "
          "--layer3 turned on with it", flush=True)
if args.blend_effects_2:
    # The axis is the whole block; the individual terms stay available
    # so each can still be scored on its own.
    _b2on = [f for f, v in (("--blend-border", args.blend_border),
                            ("--bevel", args.bevel)) if not v]
    args.blend_border = True
    args.bevel = True
    # PRINTED, because the block below promises "Each implication is
    # printed, so a run's log says which flags it turned on for itself"
    # and this implication -- and the blend_border/bevel -> height_blend
    # one it feeds, further down -- were the two that stayed silent. A
    # file that states its own contract and then violates it for two
    # cases teaches the reader to distrust the contract.
    if _b2on:
        print(f"S_BLEND_EFFECTS_2 is the whole block; "
              f"{', '.join(_b2on)} turned on with it "
              f"(and --height-blend below, which they imply)", flush=True)
# --- REACHABILITY: an axis that lands inside another flag's block -----
# Several of these axes only exist inside a surface term the renderer
# computes conditionally -- a detail NORMAL only means something once a
# normal map is being applied, a metalness only once there is a specular
# lobe to give an F0 to. Shipping them without the enclosing flag would
# reproduce the --ibl-lod-scale failure exactly: the arm runs, reports,
# and never executes the code. Each implication below is printed, so a
# run's log says which flags it turned on for itself.
_IMPLIES = [
    (args.detail_normal, ("normal_map", "vertex_normals"),
     "S_DETAIL_NORMAL composites onto the tangent-space normal"),
    (args.blend_effects_2, ("normal_map", "vertex_normals"),
     "S_BLEND_EFFECTS_2's bevel bends the normal-mapped normal"),
    (args.aniso_gloss, ("specular", "normal_map", "vertex_normals"),
     "S_ANISOTROPIC_GLOSS splits a specular roughness"),
    (args.metalness_tex, ("specular",),
     "S_METALNESS_TEXTURE only reaches the shading through the F0"),
    (args.blend_effects, ("overlay_blend",),
     "S_BLEND_EFFECTS is csgo_static_overlay's, and those faces are "
     "drawn OPAQUE until F_BLEND_MODE reclassifies them"),
    # csgo_simple / csgo_simple_3layer_parallax. Both shaders resolve a
    # roughness, a metalness and an F0 and then hand them to the SAME
    # composition the other families use, so the surface they write has
    # to exist: `rough` and `metal` are only assigned inside
    # `if args.specular:` and the roughness itself lives in the normal
    # map's blue channel, which is only sampled inside
    # `if args.normal_map:`. Without these three the two families would
    # compute an albedo and drop every other term they measured --
    # exactly the --ibl-lod-scale shape (resident, never read).
    (bool(args.simple_shading) and bool(args.fam_side),
     ("specular", "normal_map", "vertex_normals"),
     "csgo_simple's roughness is the normal map's BLUE channel and its "
     "metalness only reaches the shading through an F0"),
    (bool(args.s3lp_shading) and bool(args.fam_side),
     ("specular", "normal_map", "vertex_normals"),
     "csgo_simple_3layer_parallax needs the tangent frame for its "
     "parallax and the normal map's blue channel for its layer mips"),
]
for _want, _flags, _why in _IMPLIES:
    if not _want:
        continue
    _added = [f for f in _flags if not getattr(args, f)]
    for f in _added:
        setattr(args, f, True)
    if _added:
        print(f"{_why}; turned on --"
              + ", --".join(f.replace("_", "-") for f in _added),
              flush=True)
if args.use_new_blending and not args.height_blend:
    # S_USE_NEW_BLENDING selects a TRANSITION FUNCTION, so it only means
    # anything inside height_blend(). Leaving --height-blend off would
    # make the axis unreachable at its own default -- the --ibl-lod-scale
    # failure exactly -- so it is turned on with it and said out loud.
    args.height_blend = True
    print("S_USE_NEW_BLENDING is a blend transition function; "
          "--height-blend turned on with it", flush=True)
_FAM_SIDE_CACHE = []


def _fam_side_load():
    """The side table, read ONCE. Two consumers now: the class-population
    pre-pass below and the FAMILY block further down. 7.6 MB is cheap
    per run and not cheap across a tardigrade-3 corpus."""
    if not _FAM_SIDE_CACHE:
        _FAM_SIDE_CACHE.append(torch.load(args.fam_side, map_location="cpu",
                                          weights_only=False))
    return _FAM_SIDE_CACHE[0]


MBOIT_N = 0 if args.mboit == "off" else int(args.mboit)
if MBOIT_N:
    # RULE #34: uncertainty prints where the render happens, never only in a
    # comment or a --help string nobody reads at run time. Both of these
    # constants were PUBLISHED PAPER VALUES until the binary read, and the
    # read that replaced them carries a caveat of its own. A run that
    # depends on them says so.
    print(f"MBOIT constants: bias={args.mboit_bias:g} "
          f"overestimation={args.mboit_overestimation:g}. Exact as read: "
          f"4.999999873689376e-06 (5e-6f) and 0.009999999776482582 (0.01f). "
          f"Both READ from the FLOAT ConVar registration in libclient.so "
          f"(r_csgo_mboit_bias / r_csgo_mboit_overestimation), CORROBORATED "
          f"BY TWO INDEPENDENT FILTERS: the movss/xmm0 instruction shape, "
          f"and the call target -- the ConVar constructor at va 0xc84900, "
          f"552 calls, against 9,582-18,432 for the generic helpers. The "
          f"two filters partition IDENTICALLY over the 23 resolvable cvars: "
          f"same three, no more, no fewer. See option_surface.py "
          f"CONVAR_CTOR_VA.", flush=True)
    print(f"MBOIT constants: they REPLACED the published Muenstermann "
          f"values 6e-05 / 0.25 -- 12x and 25x. Those were PAPER numbers "
          f"standing in for engine constants this port could not decode, "
          f"and NEITHER happened to match its read, so this is a real "
          f"change to the resolve and not a rename of the same value. "
          f"THE HONEST CEILING, and it is one step below 'the engine's "
          f"default is X': 0xc84900 is almost certainly the FLOAT ConVar "
          f"template instantiation, and the argument at xmm0 is not PROVEN "
          f"to be the default rather than another float parameter -- a "
          f"stripped build gives no signature. Numbers from this run "
          f"inherit that.", flush=True)

# ---- CLASS ENABLEMENT FROM PACK POPULATION ---------------------------
# THE ORDERING IS THE WHOLE FIX, so it is stated before the code.
#
# csgo_vertexlitgeneric, csgo_effects, csgo_black_unlit and csgo_water_fancy
# each have a fully transcribed path in this file, and each was reachable
# only by a flag nobody passes. The coverage matrix caught black_unlit
# drawing 850 px through a SUBSTITUTE while its own 115-line module sat
# switched off. A class whose path is written and whose default never
# reaches it is not a missing feature; it is a break/unbreak ritual.
#
# THE TEMPTING ONE-LINER IS WRONG. Defaulting these flags to True makes the
# FAMILY predicate below unconditionally true, and this file already carries
# that scar in its own words: "FAMILY -- unconditionally true, which turned
# '--fam-side is required' into a hard error for every invocation of this
# file, including the ones that never asked for a family." Every bare render
# in the corpus would break.
#
# So the class turns on because the PACK CONTAINS IT. That reads a census,
# which needs the side table, which used to be loaded behind FAMILY, which
# is computed from these very flags -- the cycle. It breaks here because the
# census needs NOTHING from FAMILY: the world pack is already loaded, and
# the side table is a 7.6 MB read that can happen on its own. Populations
# first, enablement from populations, and only THEN does FAMILY evaluate --
# over flags that already reflect the map.
#
# A flag remains the manual override IN BOTH DIRECTIONS. Passing it forces
# on; passing --no-<flag> is not offered, so an explicit off is expressed by
# not passing it on a pack without population -- and any override that acts
# is printed. Without --fam-side there is no census, nothing is enabled, and
# bare renders behave exactly as they did.
_CLASS_ENABLE = {
    "csgo_vertexlitgeneric.vfx": "sf_vertexlit",
    "csgo_effects.vfx": "sf_effects",
    "csgo_black_unlit.vfx": "sf_black_unlit",
    "csgo_water_fancy.vfx": "water_fancy",
}
CLASS_FACES = {}
if args.fam_side and "face_matid" in data:
    try:
        _pfs = _fam_side_load()
        _pnames = list(_pfs["fam_names"])
        _pfam = _pfs["fam_id"].to(device).long()
        _pcnt = torch.bincount(_pfam[data["face_matid"].to(device).long()]
                               .reshape(-1),
                               minlength=len(_pnames)).tolist()
        CLASS_FACES = {n: int(_pcnt[i]) for i, n in enumerate(_pnames)}
    except Exception as _e:
        print(f"class enablement: census FAILED ({_e}); the class flags keep "
              f"their command-line values and a written path may stay "
              f"unreachable. This is a GAP, not a default.", flush=True)
# --class-off, resolved ONCE against the same table the enablement loop
# walks, so a name that no class carries is an error here rather than a
# silent no-op that reads as "the arm ran and made no difference".
_CLASS_OFF = set()
if args.class_off:
    _by_flag = {v: k for k, v in _CLASS_ENABLE.items()}
    for _t in args.class_off.split(","):
        _t = _t.strip()
        if not _t:
            continue
        _key = _t.replace("-", "_")
        if _t in _CLASS_ENABLE:
            _CLASS_OFF.add(_t)
        elif _key in _by_flag:
            _CLASS_OFF.add(_by_flag[_key])
        else:
            raise SystemExit(
                "--class-off %r: no such class. Known family names are %s; "
                "known flag names are %s. A silent no-op here would report "
                "the substitute arm as having been run when it was not."
                % (_t, sorted(_CLASS_ENABLE), sorted(_CLASS_ENABLE.values())))
for _cn, _attr in _CLASS_ENABLE.items():
    _pop = CLASS_FACES.get(_cn)
    _cur = getattr(args, _attr, None)
    if _cn in _CLASS_OFF:
        # WINS OVER BOTH the population default and an explicit --sf-<x>,
        # because it is the only way to say "off" and a flag that can be
        # overruled by the thing it exists to overrule is not a flag.
        setattr(args, _attr, False)
        print(f"class {_cn}: OFF by explicit --class-off, overriding a pack "
              f"population of "
              f"{'unknown' if _pop is None else _pop} faces"
              + (" AND an explicit --%s" % _attr.replace('_', '-')
                 if _cur else "")
              + f". Its pixels will be shaded by a SUBSTITUTE and the class "
              f"matrix will report them as '** VISIBLE but its OWN PATH IS "
              f"OFF **'. That verdict was previously unreachable from the "
              f"CLI on any map that could produce it.", flush=True)
        continue
    if _cur:
        print(f"class {_cn}: ON by explicit --{_attr.replace('_', '-')} "
              f"(pack population {'unknown' if _pop is None else _pop} "
              f"faces)", flush=True)
        continue
    if _pop:
        setattr(args, _attr, True)
        print(f"class {_cn}: ON because the pack contains {_pop} faces of "
              f"it -- its transcribed path is what shades them. Previously "
              f"this defaulted OFF and those pixels came from a substitute.",
              flush=True)
    elif _pop == 0:
        setattr(args, _attr, False)
    else:
        setattr(args, _attr, False)
        if args.fam_side:
            print(f"class {_cn}: OFF -- population unknown (no census), so "
                  f"nothing is assumed.", flush=True)

# `--gt-surface on` needs MAT_EXT2, which lives in the side table, so it
# joins the FAMILY predicate rather than being on in name only.
FAMILY = any((B2, args.fam_reach, args.fam_only is not None,
              # GTSURF, however it resolved, needs MAT_EXT2 bound or it is
              # on in name only -- the failure this whole axis exists to
              # stop being invisible.
              GTSURF,
              args.alpha_ref, args.rough_remap,
              args.overlay_blend, args.paint_vertex_colors,
              args.glass, args.parallax3, args.mask_normals, args.foliage,
              args.unlit, args.complex_tint, args.water,
              args.depth_bias > 0, GTSURF,
              # the transparency/refraction axes are all per-material
              # csgo_complex / csgo_glass / csgo_foliage features, so they
              # all need the family side table
              args.water_fancy, args.water_selftest,
              args.translucent, args.additive_blend, args.cubemap_refraction,
              args.translucent_clip, args.disable_translucent_clip,
              args.opaque_fade, args.alpha_test_prepass, args.mode_depth,
              MBOIT_N > 0,
              # csgo_lightmappedgeneric / generic.vfx / csgo_imported are
              # ON whenever the side table exists. They are NOT added to
              # this predicate as a bare `args.lmg_family == "on"`,
              # because that would make --fam-side mandatory for every
              # invocation of this file, including the ones that never
              # wanted a family. Given the table, they run at defaults.
              args.fam_side is not None and args.lmg_family == "on",
              # the three small families are per-material by construction
              args.sf_effects, args.sf_black_unlit, args.sf_vertexlit,
              args.sf_flop_audit, args.sf_flop_audit_mutate,
              args.sf_selftest,
              # ON, so they are reachable at the defaults -- but they are
              # ANDed with --fam-side rather than added bare, because a
              # bare default-1 entry here would make FAMILY unconditionally
              # true and turn "--fam-side is required" into a hard error
              # for every invocation that never asked for a family
              # feature. Default-on where the side table is present,
              # silent where it is not; both halves are stated in
              # --simple-shading's help.
              bool(args.simple_shading) and bool(args.fam_side),
              bool(args.s3lp_shading) and bool(args.fam_side),
              args.simple_reach and bool(args.fam_side),
              # csgo_weapon / csgo_legs_prepass are per-material families
              # like every other, so the viewmodel pass needs MAT_FAM and
              # MAT_EXT2 exactly as the world families do. Leaving them out
              # of this tuple would leave MAT_FAM unbound and the feature
              # would NameError only in the configuration nobody runs.
              args.weapon, args.viewmodel, args.viewmodel_legs_prepass,
              args.weapon_axes_reach, args.viewmodel_reach,
              ))
LMG_ON = args.lmg_family == "on"     # never `if args.lmg_family:` -- "off"
                                     # is a true string
LMG_FADE_ON = args.lmg_opaque_fade == "on"
LMG_SELFTEST_ON = args.lmg_selftest == "on"
# The family name fast_pack_fam.py writes for the csgo_imported search
# path's one material. Kept in ONE place so the packer and the renderer
# can never disagree about the spelling.
LMG_IMPORTED_FAM = "csgo_imported/dev_placeholder.vmat"
# Columns the three transcribed shaders read. Named here so the missing-
# column refusal below can list them.
LMG_NEED_COLS = (
    # csgo_lightmappedgeneric static-combo axes, resolved through
    # m_iFeatureIndex against the family's OWN feature array (the ps
    # stage's S_SPECULAR_DIRECT carries m_iFeatureIndex 9 and position 9
    # of the features .vcs is F_SPECULAR_DIRECT, and so on for all 11
    # material-driven axes) -- never by string-matching a name.
    "f_lmg_specular_direct", "f_lmg_specular_indirect", "f_lmg_layers",
    "f_lmg_fancy_blending", "f_lmg_detailtexture", "f_lmg_texturetransforms",
    "f_lmg_alpha_test", "f_lmg_translucent", "f_lmg_detailblendmode",
    "f_lmg_overlay", "f_lmg_metalness_texture", "f_lmg_tint_mask",
    "f_lmg_addbumpmaps", "f_lmg_no_spec_at_full_rough",
    "f_lmg_render_backfaces", "f_lmg_dont_flip_backface_normals",
    "f_lmg_do_not_cast_shadows", "f_lmg_additive_blend",
    # generic.vfx (csgo_core) static-combo axes, same method
    "f_gen_specular", "f_gen_unlit", "f_gen_alpha_test",
    "f_gen_translucent", "f_gen_self_illum", "f_gen_tint_mask",
    "f_gen_overlay", "f_gen_additive", "f_gen_render_backfaces",
    "f_gen_dont_flip_backface_normals",
    # uniforms all three read
    "lmg_metalness", "lmg_model_tint_amount", "lmg_vcol_opacity_scale",
    "lmg_fog_enabled", "lmg_addr_mode_u", "lmg_addr_mode_v",
    "lmg_l1tint_r", "lmg_l1tint_g", "lmg_l1tint_b",
    "lmg_rough_bright", "lmg_rough_contrast", "lmg_normalmap_strength",
    "has_l1ao", "has_l1nrmrough", "is_dev_placeholder")
SFAM = any((args.sf_effects, args.sf_black_unlit, args.sf_vertexlit))
SIMPLE_ON = bool(args.simple_shading)
S3LP_ON = bool(args.s3lp_shading)
EXT2, FAM_NAMES = {}, []
# [decal px, particle px, total px] -- filled by the composite in
# _post_chain. This is the reachability evidence for both families:
# "defined but never called" cannot produce a non-zero here.
NW_PIX = [0.0, 0.0, 0.0]
# Non-world family ids. -1 until the family table exists; the two passes
# below are driven by their own flags and by the non-world side table,
# not by a map-material scan, so they must not depend on --fam-side
# being present. They are re-bound to real ids in the FAMILY block.
FAM_PROJECTED_DECALS = -1
FAM_SPRITECARD = -1
NW_SIDE = None            # the non-world side table, loaded once
NW_TRANS_Z = [None]       # translucent-scene depth target, per chunk
FAMTEX = None
# Defined unconditionally so a reader of this file cannot mistake "the
# name does not exist" for "the family is not on the map". -1 means the
# side table does not name that family; LMG_LIVE means the three
# transcribed shaders have their data and will run.
FAM_LIGHTMAPPEDGENERIC = FAM_GENERIC = FAM_IMPORTED = -1
T_L1AO = T_L1NR = None
LMG_LIVE = False
# --- non-world families: one gate, listed once ------------------------
# Every csgo_character / csgo_eyeball / csgo_customglove axis rides on the
# side table, so they turn FAMILY on the same way the world axes do. They
# are collected here so a new axis cannot be added to the parser and then
# silently never reach the shading path.
CHAR_AXES = {
    # S_USE_PER_VERTEX_CURVATURE and
    # S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS are deliberately NOT in
    # this literal. Their flags default to "auto" and are resolved from
    # the block below, whose predicate is any() over THIS dict -- listing
    # them here would let the two of them decide their own answer, which
    # is how they came to be on unconditionally in the first place.
    "S_SUBSURFACE_SCATTERING": args.char_sss,
    "S_ANISOTROPIC_GLOSS": args.char_aniso,
    "S_ANISOTROPIC_HAIR": args.char_hair,
    "F_CLOTH_SHADING": args.char_cloth,
    "S_IRIDESCENCE": args.char_iridescence,
    "S_RETRO_REFLECTIVE": args.char_retro_reflective,
    "S_PATCHES": args.char_patches,
    "S_DECAL_TEXTURE": args.char_decals,
    "S_SUPPORTS_DECALS": args.char_supports_decals,
    "F_BLOOD": args.char_blood,
    "S_ENABLE_ADJUSTMENTS": args.char_adjustments,
    "F_DETAIL_TEXTURE": args.char_detail,
    "S_EYEBALLS": args.char_eyes,
    "S_TINT_MASK": args.char_tint_mask,
    "S_ALPHA_TEST": args.char_alpha_test,
    "S_TRANSLUCENT": args.char_translucent,
    "S_ADDITIVE_BLEND": args.char_additive,
    "S_MODE_DEPTH": args.char_mode_depth,
    "S_MODE_TOOLS_VIS": args.char_tools_vis != 0,
    "D_BAKED_LIGHTING_*": args.char_baked_lighting != "inherit",
    "F_INVULNERABILITY": args.char_invulnerability,
    "player_visibility_stencil_proxy": args.char_visibility_stencil,
    "csgo_eyeball": args.eyeball,
    "csgo_eyeball:parallax": args.eyeball_parallax,
    "D_SPECULAR_CUBE_MAP_STATIC": args.eyeball_cubemap_static,
    "csgo_customglove": args.glove,
    "csgo_customglove:S_ANISOTROPIC_GLOSS": args.glove_aniso,
    "csgo_customglove:S_TINT_ID": args.glove_tint_id,
    "csgo_customglove:S_BACKWARDS_COMPATIBILITY": args.glove_backcompat,
    "csgo_customglove:g_fWearProgress": args.glove_wear >= 0.0,
}
# Resolve the two "auto" axes against the axes that were REALLY asked for.
# Both used to default straight to their on-value, and because CHAR_AXES
# feeds NONWORLD which feeds FAMILY, that made --fam-side a hard error for
# every invocation of this file -- a renderer that could not start without
# an asset most runs do not need. Defaulting them off instead would have
# been the other half of the same mistake: it would silently drop
# curvature and spherical tangents from character rendering, where they
# ARE the shipped default. So they follow the character path: on when one
# is selected, off when none is, and the substitution is printed either
# way rather than inferred from a missing log line. Same shape as the
# --char auto-enable immediately below.
# An EXPLICIT non-off value for either of the two counts as selecting a
# character path in its own right. Without this the pair are invisible to
# their own predicate: `--char-curvature derivative` alone would print
# "no csgo_character path is selected" and then demand --fam-side four
# lines later, which is two statements of opposite sign in one log.
_char_real = (args.char or args.eyeball or args.glove or args.char_skinning
              or args.char_selftest or any(CHAR_AXES.values())
              or args.char_curvature not in ("auto", "off")
              or args.char_aniso_tangents not in ("auto", "uv"))
for _cflag, _cdest, _con, _coff in (
        ("--char-curvature", "char_curvature", "vertex", "off"),
        ("--char-aniso-tangents", "char_aniso_tangents", "spherical", "uv")):
    if getattr(args, _cdest) == "auto":
        setattr(args, _cdest, _con if _char_real else _coff)
        _cwhy = ("a csgo_character path is selected"
                 if _char_real else
                 "no csgo_character path is selected, so this axis stays "
                 "off and does not pull in --fam-side")
        print(f"{_cflag} auto -> {getattr(args, _cdest)} ({_cwhy})",
              flush=True)
CHAR_AXES["S_USE_PER_VERTEX_CURVATURE"] = args.char_curvature != "off"
CHAR_AXES["S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS"] = (
    args.char_aniso_tangents == "spherical")
NONWORLD = (args.char or args.eyeball or args.glove or args.char_skinning
            or args.char_selftest or any(CHAR_AXES.values()))
if NONWORLD and not args.char:
    # Any non-world axis implies its family's surface path; without it the
    # axis is present, resident and unreachable -- the --ibl-lod-scale
    # failure exactly. Said out loud rather than done silently.
    _why = [k for k, v in CHAR_AXES.items() if v][:3]
    # The condition is NONWORLD itself, which is the outer test. It used
    # to re-enumerate a narrower list of --char-* flags that OMITTED the
    # two axes resolved twenty lines above -- S_USE_PER_VERTEX_CURVATURE
    # and S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS. So
    # `--char-curvature vertex` alone made NONWORLD (and FAMILY) true,
    # forcing --fam-side to load, while args.char stayed False and the
    # consumer -- gated on `NONWORLD and args.char` -- never ran. The
    # axis was resident and never read, which is the failure the comment
    # directly above this one names. Re-enumerating a predicate that
    # already exists is how the two drifted apart; not re-enumerating it
    # is the fix.
    args.char = True
    print(f"a csgo_character axis is selected ({', '.join(_why)}); "
          f"--char turned on with it", flush=True)
FAMILY = FAMILY or NONWORLD
EXT2, FAM_NAMES = {}, []
FAMTEX = None
# Declared BEFORE the block that assigns them: a `= None` placed after the
# assignment would silently blank it, which is exactly the class of bug
# this file keeps catching.
CHAR_CURV_PIX = None
CHAR_ORIGIN = None
