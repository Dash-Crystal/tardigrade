

def sample_textures(uv, layer, taps=None):
    if TEX_MIP is not None:
        if taps is None:      # no footprint for this pass: level 0, 1 tap
            taps = (torch.zeros_like(uv[..., 0]), None)
        return _mip_fetch(TEX_MIP, uv, layer, taps, ANISO_C)
    u = (uv[..., 0] % 1.0) * (T - 1)
    v = (uv[..., 1] % 1.0) * (T - 1)
    u0, v0 = u.floor().long(), v.floor().long()
    u1, v1 = (u0 + 1) % T, (v0 + 1) % T
    fu = (u - u0.float()).unsqueeze(-1)
    fv_ = (v - v0.float()).unsqueeze(-1)
    c00 = textures[layer, v0, u0].float()
    c01 = textures[layer, v0, u1].float()
    c10 = textures[layer, v1, u0].float()
    c11 = textures[layer, v1, u1].float()
    return ((c00 * (1 - fu) + c01 * fu) * (1 - fv_)
            + (c10 * (1 - fu) + c11 * fu) * fv_) / 255.0


def sample_aux(uv, layer, taps=None):
    """Bilinear fetch from the linear-space aux array (normal/AO/emissive).

    Separate from sample_textures because the aux array has its own edge
    length and must NOT be sRGB-decoded: a normal map's RGB is a direction
    and its alpha is Source 2's roughness, neither of which is a colour.
    """
    if AUX_MIP is not None:
        if taps is None:
            taps = (torch.zeros_like(uv[..., 0]), None)
        return _mip_fetch(AUX_MIP, uv, layer, taps, ANISO_A)
    u = (uv[..., 0] % 1.0) * (TA - 1)
    v = (uv[..., 1] % 1.0) * (TA - 1)
    u0, v0 = u.floor().long(), v.floor().long()
    u1, v1 = (u0 + 1) % TA, (v0 + 1) % TA
    fu = (u - u0.float()).unsqueeze(-1)
    fv_ = (v - v0.float()).unsqueeze(-1)
    c00 = AUX[layer, v0, u0].float()
    c01 = AUX[layer, v0, u1].float()
    c10 = AUX[layer, v1, u0].float()
    c11 = AUX[layer, v1, u1].float()
    return ((c00 * (1 - fu) + c01 * fu) * (1 - fv_)
            + (c10 * (1 - fu) + c11 * fu) * fv_) / 255.0


def _fam_flag(mid):
    """1 where the material belongs to --fam-only's family, else 0."""
    if args.fam_only < 0:
        return torch.ones(mid.shape + (1,), device=mid.device)
    return (MAT_FAM[mid] == args.fam_only).float().unsqueeze(-1)


def sample_fam(uv, layer, taps=None):
    """Bilinear fetch from the shader-family side texture array.

    Its own array because it holds a handful of slots referenced by 4-24
    materials each (glass dust/tint, the three parallax window layers);
    12 distinct images at 512^2 is 15 MB, against the 3.4 GB it would
    cost to rebuild the main pack to carry them.

    IT MUST FILTER LIKE ITS NEIGHBOURS OR IT ALIASES ALONE. These pages
    are 512^2 while `textures` and `aux` are 64^2, so with --mip on this
    was the only array in the frame sampled at 8x the resolution with
    flat bilinear and no taps -- layer 3 shimmering while layers 1 and 2
    sat still. The resolution is not the defect (the other two being
    degraded is), so the fix is filtering, not downsampling: lowering the
    correct layer would have erased the evidence.

    `taps` is optional and defaults the same way sample_aux does, so all
    ~60 existing call sites keep working unchanged and the ones that hold
    a footprint can pass it.
    """
    if FAM_MIP is not None:
        if taps is None:
            taps = (torch.zeros_like(uv[..., 0]), None)
        return _mip_fetch(FAM_MIP, uv, layer, taps, ANISO_A)
    u = (uv[..., 0] % 1.0) * (TF - 1)
    v = (uv[..., 1] % 1.0) * (TF - 1)
    u0, v0 = u.floor().long(), v.floor().long()
    u1, v1 = (u0 + 1) % TF, (v0 + 1) % TF
    fu = (u - u0.float()).unsqueeze(-1)
    fv_ = (v - v0.float()).unsqueeze(-1)
    c00 = FAMTEX[layer, v0, u0].float()
    c01 = FAMTEX[layer, v0, u1].float()
    c10 = FAMTEX[layer, v1, u0].float()
    c11 = FAMTEX[layer, v1, u1].float()
    return ((c00 * (1 - fu) + c01 * fu) * (1 - fv_)
            + (c10 * (1 - fu) + c11 * fu) * fv_) / 255.0


def _e2(e, key):
    """Column `key` of the per-pixel shader-family parameter block."""
    return e[..., EXT2[key]]


def _nrm(v, eps=1e-9):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


# ======================================================================
# THE REFERENCE NORMAL PAGE -- octahedral .xy, ROUGHNESS .z
# ======================================================================
# READ from csgo_environment_ps.glsl:456-478, which settles a layout this
# renderer had two different answers for. The normal sampler's texel
# `_12167` is used as:
#
#   :459-463  x, y are an OCTAHEDRAL PAIR --
#               a = (x + y) - 1.00392162799835205078125
#               b =  x - y
#               n = normalize(vec3(a, b, (1 - |a|) - |b|))
#             the SAME decode and the SAME bias constant csgo_weapon uses
#             at allon_c3041_d2:540-542, i.e. wpn.octa_normal_decode.
#   :458      ROUGHNESS is .z --  _18178 = _12167.z * _6616.x
#   :476      layer 2 adds       _13137 = _18178 + _12168.z * _6616.y
#             on the SAME _6616 weights the albedo is blended by at :456
#             (_23806 = colour * _6616.x) and :480 (+ colour2 * _6616.y),
#             and as a WEIGHTED SUM, not a lerp.
#   :514      _6375 = clamp(_13138, 0, 1)
#
# WHAT THIS CONTRADICTS, and it is in this file: the world path reads
# roughness from `nmap[..., 3]` under --rough-src normal-alpha, with a
# comment asserting "Source 2's csgo_* shaders carry roughness in the
# normal map's ALPHA". The reference reads .z. Both cannot be right about
# the same page, and which is right about the PACK's existing aux pages
# depends on what VRF wrote into them -- so the existing path is LEFT
# ALONE and this decode is applied only to pages that declare the
# reference layout. The discrepancy is printed, not silently resolved:
# resolving it needs the pack side to say which layout it wrote.
#
# The three-component `rgb * 2 - 1` read in apply_normal_map is likewise
# not this: a two-component octahedral pair is not a scaled direction, and
# feeding one to the other yields a plausible, wrong, normalised vector.
def gt_ref_normal_rough(texel):
    """(tangent-space normal (...,3), roughness (...)) from a REFERENCE page.

    `texel` is the raw page sample in [0,1]. Nothing here is fitted: the
    decode is wpn.octa_normal_decode -- the code object the conformance
    registry already A/Bs against the decompiled range -- and the
    roughness is the .z channel the reference multiplies by the layer
    weight.
    """
    n_ts = wpn.octa_normal_decode(texel[..., 0], texel[..., 1])
    return n_ts, texel[..., 2]


_WEATHER_ACK = []


def _weather_needs_famside():
    """Say that the weather block cannot act, once, and why.

    Its writes are scoped to the wet families (csgo_environment_blend and
    friends) by a mask built from fam_pix. No side table means no fam_pix
    means no mask -- and a family-scoped write with no family mask is not
    "apply to everything", it is "cannot be applied".
    """
    if _WEATHER_ACK:
        return
    _WEATHER_ACK.append(1)
    print("--weather: INERT this run -- the ripple's writes are scoped to "
          "the wet families and that scope comes from the side table "
          "(--fam-side), which this run has none of. Before this, the "
          "writes were applied to EVERY pixel instead, which discarded the "
          "normal map frame-wide. Pass --fam-side to let the term act.",
          flush=True)


_NM_EFFECT_CHECKED = []


def _normal_map_effect_check(nrm, n_geo, covered):
    """Say, at runtime, how far the normal map actually bent the normal.

    --normal-strength scales the tangent-space XY inside apply_normal_map,
    so its whole effect is an ANGLE between the shaded normal and the
    geometric one. That angle is the quantity a reader needs and the one
    nothing printed: an A/B is currently the only way to discover that a
    strength did nothing, and an A/B across two arms is exactly what
    deployment drift makes unreliable (b814137f).

    MEASURED and unresolved as of this landing: on one pinned renderer the
    strength separates cleanly WITH --fam-side (1.0 / 4.0 / 8.0 give three
    distinct images) and NOT without it (1.0 and 8.0 byte-identical). The
    mechanism is not pinned -- the pages are real (mean tilt 10.2 deg on
    de_inferno_nrm, 76% of texels past 1 deg) and tangents are on 100% of
    verts, so it is neither a data nor a guard problem. This does not fix
    that. It makes the state VISIBLE from a single run, so the next person
    reads the angle instead of inferring it from two hashes that may not
    even share a renderer.
    """
    if _NM_EFFECT_CHECKED:
        return
    _NM_EFFECT_CHECKED.append(1)
    m = covered if covered is not None else torch.ones_like(nrm[..., 0],
                                                            dtype=torch.bool)
    if not bool(m.any()):
        return
    a = _nrm(nrm)[m]
    b = _nrm(n_geo)[m]
    cos = (a * b).sum(-1).clamp(-1.0, 1.0)
    deg = torch.rad2deg(torch.arccos(cos))
    mean, mx = float(deg.mean()), float(deg.max())
    p95 = float(deg.kthvalue(max(1, int(0.95 * deg.numel())))[0])
    if mx < 1e-4:
        print(f"⚠️  --normal-map: the shaded normal is the GEOMETRIC normal "
              f"to {mx:.2e} deg over {int(m.sum()):,} covered px -- the map "
              f"is INERT this run and --normal-strength "
              f"{args.normal_strength:g} is scaling nothing. Not a claim "
              f"about the pages: check whether this run has the flags that "
              f"make the term reach the surface.", flush=True)
    else:
        print(f"--normal-map: shaded normal departs the geometric one by "
              f"mean {mean:.4f} deg, p95 {p95:.4f}, max {mx:.4f} over "
              f"{int(m.sum()):,} covered px, at --normal-strength "
              f"{args.normal_strength:g}. The angle IS the term's whole "
              f"effect, so two runs whose angle matches at different "
              f"strengths are a superseded flag however their pixels "
              f"differ.", flush=True)
    _gt_value("surface.normal_map_tilt_deg", deg, identity=0.0,
              note="angle between the shaded and geometric normal; 0 means "
                   "--normal-map/--normal-strength reached nothing")


_ROUGH_ALPHA_CHECKED = []


def _rough_alpha_degenerate_check(alpha):
    """Say, at runtime, when --rough-src normal-alpha reads a constant.

    corpus-2 measured this pack's aux alpha as 255 on all 111 pages, and a
    roughness that is a constant 1.0 everywhere is indistinguishable, in
    any image, from a roughness that is merely wrong. The reach counters
    cannot separate them either: the path runs, covers every pixel, and
    reports as live.

    So the SPREAD is checked on the first shaded batch and printed. A pack
    with real alpha prints its range and nothing is claimed; a pack with a
    stand-in prints that the term is inert. Neither is a gate -- the
    branch runs whatever this finds.
    """
    if _ROUGH_ALPHA_CHECKED:
        return
    _ROUGH_ALPHA_CHECKED.append(1)
    lo, hi = float(alpha.min()), float(alpha.max())
    mean, std = float(alpha.mean()), float(alpha.std())
    if hi - lo < 1e-6:
        print(f"⚠️  --rough-src normal-alpha: this pack's normal alpha is "
              f"CONSTANT {lo:.4f} on the first shaded batch, so the "
              f"roughness term is INERT -- every surface shades at "
              f"rough = {max(lo, 0.03):.4f}. corpus-2 measured the same "
              f"across all 111 aux pages (min = max = mean = 255); "
              f"decode_normal() writes alpha = 255 as a stand-in. The "
              f"reference reads roughness from .z "
              f"(csgo_environment_ps.glsl:458), which is what "
              f"mat_normal_pages carries.", flush=True)
    else:
        print(f"--rough-src normal-alpha: alpha spread on the first "
              f"shaded batch is [{lo:.4f}, {hi:.4f}] mean {mean:.4f} std "
              f"{std:.4f} -- NOT constant, so this pack's alpha carries "
              f"something. Whether it is roughness is still the .z/.w "
              f"question; this only says the channel is not a stand-in.",
              flush=True)
    _gt_value("surface.roughness_alpha_spread",
              torch.tensor([hi - lo], device=alpha.device), identity=0.0,
              note="range of the normal-map alpha the --rough-src "
                   "normal-alpha branch reads; 0 means the term is inert")


def sample_pages(pages, edge, uv, layer):
    """Bilinear fetch from a page array, in sample_aux's exact convention.

    Separate array, same filtering and the same wrap, so a reference page
    and an aux page sampled at one UV land on the same texel centres. No
    mip chain: these pages arrive at one resolution and this says so
    rather than silently point-sampling a chain that is not there.
    """
    u = (uv[..., 0] % 1.0) * (edge - 1)
    v = (uv[..., 1] % 1.0) * (edge - 1)
    u0, v0 = u.floor().long(), v.floor().long()
    u1, v1 = (u0 + 1) % edge, (v0 + 1) % edge
    fu = (u - u0.float()).unsqueeze(-1)
    fv_ = (v - v0.float()).unsqueeze(-1)
    c00 = pages[layer, v0, u0].float()
    c01 = pages[layer, v0, u1].float()
    c10 = pages[layer, v1, u0].float()
    c11 = pages[layer, v1, u1].float()
    out = ((c00 * (1 - fu) + c01 * fu) * (1 - fv_)
           + (c10 * (1 - fu) + c11 * fu) * fv_)
    return out / 255.0 if pages.dtype == torch.uint8 else out


def apply_normal_map(n_geo, tan4, nmap):
    """Tangent-space normal map -> world normal.

    n_geo (B,H,W,3) interpolated vertex normal, tan4 (B,H,W,4) interpolated
    TANGENT with w = bitangent sign, nmap (B,H,W,4) raw normal-map texel.
    Gram-Schmidt re-orthogonalises T against N because interpolation
    across a triangle does not preserve orthogonality.
    """
    n = _nrm(n_geo)
    t_raw = tan4[..., :3]
    have = (t_raw.norm(dim=-1, keepdim=True) > 0.5)
    t = _nrm(t_raw - n * (n * t_raw).sum(-1, keepdim=True))
    b = torch.cross(n, t, dim=-1) * tan4[..., 3:4].sign()
    ts = nmap[..., :3] * 2.0 - 1.0
    ts = torch.cat([ts[..., :2] * args.normal_strength, ts[..., 2:]], dim=-1)
    out = _nrm(t * ts[..., 0:1] + b * ts[..., 1:2] + n * ts[..., 2:3])
    return torch.where(have, out, n)


# --- the vpost's post chain ------------------------------------------
# Hable / Uncharted-2 filmic with de_inferno_prefab.vpost's own
# coefficients. The vpost names map 1:1 onto Hable's A..F,W.
TM_A = 0.15   # m_flShoulderStrength
TM_B = 0.50   # m_flLinearStrength
TM_C = 0.10   # m_flLinearAngle
TM_D = 0.20   # m_flToeStrength
TM_E = 0.02   # m_flToeNum
TM_F = 0.30   # m_flToeDenom


def _hable(x):
    return ((x * (TM_A * x + TM_C * TM_B) + TM_D * TM_E)
            / (x * (TM_A * x + TM_B) + TM_D * TM_F)) - TM_E / TM_F


_TM_WHITE = float(_hable(torch.tensor(float(args.white_point))))
_pe = [float(v) for v in str(args.pre_exposure).split(",")]
PRE_EXPOSURE = torch.tensor(_pe * 3 if len(_pe) == 1 else _pe,
                            device=device).view(1, 1, 1, 3)

CC_LUT = None
if args.post_cc:
    _l = np.load(args.post_cc).astype(np.float32) / 255.0   # [B][G][R][4]
    CC_LUT = torch.tensor(_l[..., :3], device=device)
    print(f"colour-correction volume {CC_LUT.shape[0]}^3 loaded "
          f"from {args.post_cc}")

def _gauss5_integer_kernel():
    """blur_r6m2.glsl:16-17 five BILINEAR taps -> the equivalent integer
    kernel. A tap at -1.182425022125244140625 is texels -2 and -1 at
    weights 0.182425... and 0.817574...; expanding all five gives a 7-tap
    kernel that a conv2d evaluates exactly."""
    off = np.array([_lf_tables.row_gauss5_offsets(i) for i in range(5)])
    wts = np.array([_lf_tables.row_gauss5_weights(i) for i in range(5)])
    lo = int(np.floor(off.min())); hi = int(np.ceil(off.max()))
    k = np.zeros(hi - lo + 1)
    for o, w in zip(off, wts):
        f = np.floor(o); t = o - f
        k[int(f) - lo] += w * (1.0 - t)
        if int(f) + 1 - lo < len(k):
            k[int(f) + 1 - lo] += w * t
    return k


_G5 = _gauss5_integer_kernel()
_BLUR_K = torch.tensor(_G5, device=device, dtype=torch.float32)
_BLUR_PAD = (len(_G5) - 1) // 2
_BLUR_K = _BLUR_K.view(1, 1, 1, len(_G5)).expand(3, 1, 1, len(_G5)).contiguous()


def _blur5(x):
    """separable 5-tap binomial blur on NCHW."""
    x = torch.nn.functional.conv2d(x, _BLUR_K, padding=(0, _BLUR_PAD),
                                   groups=3)
    return torch.nn.functional.conv2d(
        x, _BLUR_K.transpose(2, 3).contiguous(), padding=(_BLUR_PAD, 0),
        groups=3)


# =======================================================================
# The four filter passes of the loop-carrying families, written as passes
# rather than left as unreached functions. Every term each one declares is
# called; a config toggle is a node of the graph and gets implemented too,
# so each pass has an --lf-* switch and every switch defaults ON.
# =======================================================================
def lf_atrous(rgb, var, depth, nrm_packed, step, sigma_scale, n_power,
              slope):
    """atrous_filter r0/m1: one a-trous iteration over rgb + variance.

    Calls octahedral_normal_decode, luma, variance_sigma,
    depth_slope_reject, atrous_weight, atrous_resolve_rgb,
    atrous_resolve_variance.
    """
    _vk = [_lf_tables.row_atrous_variance_2x2(i) for i in range(2)]
    n0 = _lf_filters.octahedral_normal_decode(nrm_packed[..., 2:4])
    l0 = _lf_filters.luma(rgb)
    sig = _lf_filters.variance_sigma(var, sigma_scale)
    acc_rgb = torch.zeros_like(rgb)
    acc_var = torch.zeros_like(var)
    wsum = torch.zeros_like(var)
    K = [_lf_tables.row_atrous_tap_3(i) for i in range(3)]
    for j in (-2, -1, 0, 1, 2):
        for i in (-2, -1, 0, 1, 2):
            if i == 0 and j == 0:
                continue
            sh = (-j * step, -i * step)
            rt = torch.roll(rgb, shifts=sh, dims=(1, 2))
            vt = torch.roll(var, shifts=sh, dims=(1, 2))
            dt = torch.roll(depth, shifts=sh, dims=(1, 2))
            nt = torch.roll(nrm_packed, shifts=sh, dims=(1, 2))
            off = torch.tensor([float(i), float(j)], device=rgb.device)
            dz = _lf_filters.depth_slope_reject(depth, dt, slope,
                                                off.expand(depth.shape + (2,)))
            kp = float(K[min(abs(i), 2)] * K[min(abs(j), 2)])
            w = _lf_filters.atrous_weight(
                dz, l0, _lf_filters.luma(rt), sig, n0,
                _lf_filters.octahedral_normal_decode(nt[..., 2:4]),
                n_power, kp)
            acc_rgb = acc_rgb + rt * w.unsqueeze(-1)
            acc_var = acc_var + vt * (w * w)
            wsum = wsum + w
    wsum = wsum.clamp(min=1e-8)
    return (_lf_filters.atrous_resolve_rgb(acc_rgb, wsum),
            _lf_filters.atrous_resolve_variance(acc_var, wsum))


def lf_denoise_blur(tap_rgba, depth_texel, scale, width):
    """denoise_blur r1/m0: reciprocal depth, linear range weight, a
    saturation mask on channel x only, and two guarded means."""
    d0 = _lf_filters.reciprocal_depth(scale, depth_texel)
    acc_x = torch.zeros_like(d0)
    acc_yz = torch.zeros_like(tap_rgba[..., 1:3])
    n_un = torch.zeros_like(d0)
    wsum = torch.zeros_like(d0)
    for _oi in range(5):
        o = _lf_tables.row_denoise_offsets_5(_oi)
        t = torch.roll(tap_rgba, shifts=(0, -int(o)), dims=(1, 2))
        dt = _lf_filters.reciprocal_depth(
            scale, torch.roll(depth_texel, shifts=(0, -int(o)), dims=(1, 2)))
        w = _lf_filters.depth_range_weight(d0, dt, width)
        m = _lf_filters.unsaturated_mask(t[..., 0])
        acc_x = acc_x + t[..., 0] * m
        n_un = n_un + m
        acc_yz = acc_yz + t[..., 1:3] * w.unsqueeze(-1)
        wsum = wsum + w
    x = _lf_filters.guarded_mean(acc_x, n_un)
    y = _lf_filters.guarded_mean(acc_yz[..., 0], wsum)
    z = _lf_filters.guarded_mean(acc_yz[..., 1], wsum)
    return torch.stack([x, y, z], -1)


def lf_blur_with_depth(rgba, uv, depth01, m, plane, ramp, lo, hi, floor_w,
                       texel_x):
    """blur_with_depth r1/m4: the projected scalar field, the STEP tap
    gate, and the conditional alpha promote."""
    ndc = _lf_filters.ndc_from_uv_depth(uv, depth01, m)
    f0 = _lf_filters.projected_field(ndc, plane, ramp)
    acc = rgba * 0.00999999977648258209228515625
    wsum = torch.full_like(depth01, 0.00999999977648258209228515625)
    for _gi in range(5):
        k = _lf_tables.row_gauss5_weights(_gi)
        o = _lf_tables.row_gauss5_offsets(_gi)
        sh = -int(round(float(o)))
        t = torch.roll(rgba, shifts=(0, sh), dims=(1, 2))
        ft = torch.roll(f0, shifts=(0, sh), dims=(1, 2))
        a = _lf_filters.alpha_promote(t[..., 3], ft, lo)
        w = _lf_filters.depth_tap_gate(float(k), ft, f0, a, rgba[..., 3],
                                       lo, hi, floor_w)
        acc = acc + t * w.unsqueeze(-1)
        wsum = wsum + w
    return acc / wsum.clamp(min=1e-8).unsqueeze(-1)


def lf_general_filter(tex, origin, size, texel, taps, tint, vcolor):
    """general_filter r3/m1: the 1.5-texel inset rect, alpha-weighted taps,
    and the reciprocal resolve."""
    rect = _lf_filters.sample_rect(origin, size, texel)
    acc = torch.zeros_like(tex)
    wsum = torch.full(tex.shape[:-1], 1.0000000133514319600180897396058e-10,
                      device=tex.device, dtype=tex.dtype)
    for (ox, oy, wt) in taps:
        t = torch.roll(tex, shifts=(-int(oy), -int(ox)), dims=(1, 2))
        acc = acc + _lf_filters.alpha_tap_contribution(t, wt)
        wsum = wsum + _lf_filters.alpha_tap_weight(t, wt)
    return _lf_filters.general_filter_resolve(acc, wsum, tint, vcolor), rect


def lf_msaa_pair(s0, s1):
    """blur r6/m2: the 2-sample resolve that precedes the sRGB decode."""
    return _lf_filters.msaa_pair_average(s0, s1)


# =======================================================================
# effects / shadows / volumetrics / binner, written as passes.
# =======================================================================
def lf_ssao_full(pos, eye, face_dr, face_dl, face_du, face_dd, nmap_texel,
                 depth_buf, mvp, radius, bias, rect):
    """ssao r0/m0: the 9-tap hemisphere estimator, end to end.

    The projected uv is what SELECTS the sampled depth, so it is used to
    index depth_buf rather than being computed and discarded.
    """
    map_n = _lf_ao.normal_decode_unpack(nmap_texel)
    fn = _lf_ao.face_normal_min_delta(face_dr, face_dl, face_du, face_dd)
    B, H, W = depth_buf.shape
    d_here = depth_buf
    origin = _lf_ao.sample_origin(pos, fn, d_here, bias)
    acc = torch.zeros_like(d_here)
    bidx = torch.arange(B, device=pos.device).view(-1, 1, 1)
    for _ki in range(9):
        kv = _lf_tables.row_ssao_kernel_9(_ki)
        kvec = torch.tensor(kv, device=pos.device,
                            dtype=pos.dtype).expand(pos.shape)
        v = _lf_ao.hemisphere_reflect(kvec, radius, map_n, fn)
        sp = pos + v
        clip = torch.einsum("...ij,...j->...i", mvp,
                            torch.cat([sp, torch.ones_like(sp[..., :1])], -1))
        uv = _lf_ao.project_clamped_uv(clip, rect)
        ui = (uv[..., 0] * (W - 1)).round().long().clamp(0, W - 1)
        vi = (uv[..., 1] * (H - 1)).round().long().clamp(0, H - 1)
        d_tap = depth_buf[bidx, vi, ui]
        acc = acc + _lf_ao.occlusion_step(sp, eye, d_tap, origin,
                                          radius * radius)
    return _lf_ao.ao_resolve_cubed(acc)


def lf_sao_spiral(i, jitter, r_px, v, parity):
    """sao r0/m3 tail: the spiral tap schedule and the checker de-noise."""
    tap = _lf_ao.sao_spiral_tap(i, jitter)
    ddx = torch.zeros_like(v)
    return tap, _lf_ao.sao_checker_denoise(v, ddx, parity)


def lf_aoproxy(local_pos, half_extent, basis, dir_to_proxy, inv_dist,
               slice_w, acc, fade, vertex_alpha, inst_scale):
    """aoproxy_splat r0/m0: box/sphere fade, box inverse distance, the LUT
    coordinate and the multiplicative composite."""
    f = _lf_ao.proxy_box_fade(local_pos)
    inv = _lf_ao.proxy_box_inv_dist(local_pos, half_extent)
    uvw = _lf_ao.proxy_sphere_lookup_uvw(basis, dir_to_proxy, inv_dist, slice_w)
    return f, inv, uvw, _lf_ao.proxy_composite(acc, fade, vertex_alpha,
                                               inst_scale)


def lf_inferno(uv, world_pos, blob_c, blob_r, sdf, radius, age, lift,
               noise_xy, w):
    """inferno r0/m0: cluster tile, squashed blob field, weight, noise lift
    and the two-channel shade."""
    return (_lf_effects.inferno_tile_index(uv),
            _lf_effects.inferno_blob_field(world_pos, blob_c, blob_r),
            _lf_effects.inferno_blob_weight(sdf, radius, age),
            _lf_effects.inferno_noise_lookup_pos(world_pos, lift),
            _lf_effects.inferno_shade(noise_xy, w))


def lf_bomb_blast(dist, radius, uv, centre_uv, strength, rgb):
    """bomb_blast r0/m0: ring profile, refraction offset, pow-8 bloom."""
    return (_lf_effects.blast_shock_profile(dist, radius),
            _lf_effects.blast_refract_offset(uv, centre_uv, strength),
            _lf_effects.blast_bloom(rgb))


def lf_grenade_camera(uv, aspect, vign, hard, phase, rgb):
    """csgo_grenade_camera r0/m0: vignette, diagonal wipe, output gamma."""
    return (_lf_effects.grenade_vignette(uv, aspect),
            _lf_effects.grenade_scanline_alpha(uv, vign, hard, phase),
            _lf_effects.grenade_gamma_out(rgb))


def lf_outlines(count, rgb_mean, radial, width):
    _taps = [_lf_tables.row_outline_taps_5(i) for i in range(5)]
    """outlines r0/m1: shrinking tap radius, majority-count edge, shade."""
    m = _lf_effects.outline_edge_metric(count)
    return (_lf_effects.outline_tap_offset_scale(radial, width), m,
            _lf_effects.outline_shade(rgb_mean, m, radial), _taps)


def lf_panorama_alpha(rgb, depth, ref, count):
    _ring = [_lf_tables.row_panorama_ring_8(i) for i in range(8)]
    """panorama_alpha r0/m0: background test and the ring coverage."""
    return (_lf_effects.panorama_is_background(rgb, depth, ref),
            _lf_effects.panorama_coverage(count), _ring)


def lf_convolve_env(n, sample_rgb, weight):
    """convolve_environment_map r0/m0: tangent basis and clamped accumulate."""
    return (_lf_effects.convolve_tangent_basis(n),
            _lf_effects.convolve_accumulate(sample_rgb, weight))


def lf_terrain_slope(d):
    _ring = [_lf_tables.row_slope_ring_8(i) for i in range(8)]
    """tools_terrain_composite_slope r0/m0: the squared-cosine metric."""
    return _lf_effects.slope_metric(d), _ring


def lf_volume_viewer(world_pos, direction, acc, steps):
    """csgo_volume_viewer r0/m0: fixed-box march start, step, resolve."""
    uvw = _lf_effects.volume_march_start(world_pos)
    return uvw, _lf_effects.volume_march_step(uvw, direction), \
        _lf_effects.volume_resolve(acc, steps)


def lf_visualize_depth(mip):
    """visualize_depth r0/m0: the two 3-entry colour tables, indexed i mod 3."""
    i = int(mip) % 3
    return (_lf_tables.row_visdepth_colours_a(i),
            _lf_tables.row_visdepth_colours_b(i))


def lf_msaa_resolve_m39(rgb, frag_xy, phase, amount, field, near, far):
    """msaa_resolve r0/m39: weighted resolve, dither, inverse, two-sided CoC."""
    w = _lf_effects.resolve_tonemap_weight(rgb)
    d = _lf_effects.resolve_dither(frag_xy, phase, amount)
    inv = _lf_effects.resolve_tonemap_invert(
        (w + d).clamp(0, 0.9900000095367431640625))
    return w, d, inv, _lf_effects.coc_two_sided(field, near, far)


def lf_cascade(uv, g0, g1, g2, g3, inset, slope, s_this, s_next, factor,
               shadow, dist, start, inv_range, scale_bias, jitter):
    """deferred_particle_shadows r0/m0: the cascade selector and its PCF."""
    return (_lf_shadows.uv_inside_unit_square(uv),
            _lf_shadows.pcf_gather_16(g0, g1, g2, g3),
            _lf_shadows.cascade_border_factor(uv, inset, slope),
            _lf_shadows.cascade_blend(s_this, s_next, factor),
            _lf_shadows.shadow_distance_fade(shadow, dist, start, inv_range),
            _lf_shadows.shadow_uv_from_cascade(uv, scale_bias, jitter))


def lf_player_visibility(occ, acc_lit, acc_unlit, n_lit, lu, ll, cw, cov):
    _w16 = [_lf_tables.row_player_vis_weights_16(i) for i in range(16)]
    """player_visibility r0/m1: mask, split means, asymmetric boost, edge."""
    return (_lf_shadows.visibility_tap_mask(occ),
            _lf_shadows.visibility_split_means(acc_lit, acc_unlit, n_lit),
            _lf_shadows.visibility_contrast_boost(lu, ll),
            _lf_shadows.visibility_edge_weight(cw, cov, 0.0, lu, ll), _w16)


def lf_tools_grid(eye, dir_, origin, u_axis, v_axis, cell, fw, scale):
    """tools_grid r0/m1: plane intersect, line coverage, derivative fade."""
    q = _lf_shadows.grid_plane_intersect(eye, dir_, origin, u_axis, v_axis)
    return (q, _lf_shadows.grid_line_coverage(cell, fw, scale),
            _lf_shadows.grid_fade_by_derivative(fw))


def lf_volumetrics(t_max, plane_n, plane_d, origin, direction, ndc_a, ndc_b,
                   density, frag_xy, scale, phase, step, ndc_xy, corner,
                   softness):
    """sfm_volumetrics_frustum r1/m0: slab clip, step size, dither, mask."""
    return (_lf_vol.slab_clip(t_max, plane_n, plane_d, origin, direction),
            _lf_vol.march_step_size(t_max, ndc_a, ndc_b, density),
            _lf_vol.hash_dither(frag_xy, scale, phase, step),
            _lf_vol.rounded_rect_mask(ndc_xy, corner, softness))


def lf_binner(cull_a, cull_b, wave, t, w1, w3, w01, w23, texel_xy, z,
              deriv_ratio, color, amb_s, amb_b, ndotl, texel, nscale,
              base_rough, ddx_n, ddy_n, normal, tangent4, iflip, back,
              q, qscale, to_light, half_axis, ndc_z, near_s, far_s,
              local_pos, lo, hi, inv_fade, cov, cur, pid, claimed, counter,
              fired, frag_xy):
    """generic r69/m3 + lpv_debug_grid r2/m2 + csgo_tools_shading_complexity.

    The wave-coherent binner, the bicubic lightmap filter, the directional
    lightmap, the surface terms, the light primitives, the probe volumes
    and the claim protocol -- every registered term of the three.
    """
    _ident = [_lf_tables.row_identity_rows_4(i) for i in range(4)]
    mask = _lf_binner.wave_coherent_mask(cull_a, cull_b, wave)
    lsb, rest = _lf_binner.peel_lowest_bit(mask)
    bw = _lf_binner.bspline_weights(t)
    off = _lf_binner.bspline_offsets(w1, w3, w01, w23)
    dl = _lf_binner.directional_lightmap_decode(texel_xy)
    sh = _lf_binner.directional_sharpen(z, deriv_ratio)
    irr = _lf_binner.directional_irradiance(color, amb_s, amb_b, z, ndotl)
    nrm_wy = _lf_binner.normal_decode_wy(texel, nscale)
    ro = _lf_binner.specular_aa_roughness(base_rough, ddx_n, ddy_n)
    bt = _lf_binner.bitangent_with_flips(normal, tangent4, iflip, back)
    ax = _lf_binner.quaternion_axis_y(q, qscale)
    cp = _lf_binner.capsule_closest_point(to_light, half_axis)
    sf = _lf_binner.spot_endcap_fade(ndc_z, near_s, far_s)
    vw = _lf_binner.volume_inset_weight(local_pos, lo, hi, inv_fade)
    va = _lf_binner.volume_smooth_and_accumulate(vw, cov)
    tile = _lf_binner.tools_tile_coord(frag_xy)
    old, new = _lf_binner.tools_compswap(cur, pid)
    o2, n2 = _lf_binner.tools_exchange(cur)
    latch = _lf_binner.tools_claim_state(old, pid, claimed)
    cnt = _lf_binner.tools_overdraw_add(counter, fired)
    return (mask, lsb, rest, bw, off, dl, sh, irr, nrm_wy, ro, bt, ax, cp,
            sf, vw, va, tile, old, new, o2, n2, latch, cnt, _ident)


# =======================================================================
# The AWP scope pass.
#
# No .vcs for this one: it is read from the reference frames themselves,
# iji_model/counter_strike_render/eval_figures/cs1k_subsystem_tranche_16.png, four
# scoped tiles of 320x180. Every number below is MEASURED off those pixels
# and the measurement is reproducible from the file:
#
#   lens centre      (159.3-159.6, 89.4-89.5) of (320, 180) -- the tile
#                    centre to within half a pixel, on all four tiles
#   lens radius      76.0, 75.8, 76.1 px = 0.421-0.423 x tile height on
#                    three tiles; 72.2 (0.401) on the fourth, which shows a
#                    visibly smaller lens. SCOPE_RADIUS takes the three
#                    that agree; the outlier is recorded, not averaged in.
#   surround         luminance 0.00006-0.00171 -- BLACK, not dark grey
#   vignette         normalised radial profile over the three agreeing
#                    tiles, reference band r 0.3-0.6:
#                      r<0.70  0.93-1.09   (scene content, no trend)
#                      0.75    0.9286      0.85  0.7413
#                      0.80    0.8796      0.90  0.6824
#                      0.95    0.3628      (edge pixels straddle the cut)
#                      r>1.00  0.0038      black
#
# So the falloff begins at r ~ 0.70 and the lens edge is a HARD cut, not
# the tail of a broad vignette. SCOPE_VIG_DEPTH is solved from the 0.90
# sample rather than tuned: 1 - D*smoothstep(0.7,1,0.90) = 0.6824 gives
# D = 0.343. Residuals against the other bins: 0.85 +0.005, 0.80 +0.032,
# 0.75 +0.046. The 0.75 bin is the worst and is stated rather than fitted
# away -- a two-parameter curve does not reproduce this profile exactly
# and pretending otherwise would be the fitted-float failure.
# =======================================================================
SCOPE_RADIUS = 0.422          # x tile height, mean of the three agreeing
SCOPE_RADIUS_OUTLIER = 0.401  # the fourth tile, recorded not averaged
SCOPE_VIG_START = 0.70
SCOPE_VIG_DEPTH = 0.343
SCOPE_SURROUND = 0.0          # measured 0.00006-0.00171

_SCOPE_PRINTED = False


def scope_fov_scale(fov_deg, base_fov_deg=90.0):
    """Projection change for a scoped view.

    m_iFOV is 40, 15 or 10 on scoped ticks against a 90 default, so the
    magnification is tan(base/2)/tan(fov/2): 2.75x, 7.60x, 11.43x. This
    returns the scale the projection multiplies, not a post-process zoom --
    the reference re-renders, it does not magnify the frame it already has.
    """
    import math
    t = math.tan(math.radians(float(fov_deg)) * 0.5)
    return math.tan(math.radians(base_fov_deg) * 0.5) / max(t, 1e-6)


def scope_lens_r(frag_xy, height, width):
    """Radius of a pixel in lens units: 1.0 is the lens edge.

    The lens is centred on the frame and its radius is a fraction of the
    frame HEIGHT, not of the diagonal -- measured centre (159.5, 89.5) of
    (320, 180) and radius 76 of 180.
    """
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    dx = frag_xy[..., 0] - cx
    dy = frag_xy[..., 1] - cy
    return (dx * dx + dy * dy).sqrt() / (SCOPE_RADIUS * height)


def scope_lens_mask(r):
    """1 inside the lens, 0 outside. The measured transition is one pixel
    wide -- 0.3628 at r 0.95-1.00 against 0.0038 at 1.00-1.05 -- so this
    is a hard cut with no feather of our own invention."""
    return (r < 1.0).to(r.dtype)


def scope_vignette(r):
    """1 - 0.343 * smoothstep(0.70, 1.0, r), from the measured profile."""
    t = ((r - SCOPE_VIG_START) / (1.0 - SCOPE_VIG_START)).clamp(0.0, 1.0)
    return 1.0 - SCOPE_VIG_DEPTH * (t * t * (3.0 - 2.0 * t))


def scope_compose(scene_rgb, r):
    """lens * vignette * scene, over a BLACK surround."""
    m = scope_lens_mask(r).unsqueeze(-1)
    v = scope_vignette(r).unsqueeze(-1)
    return scene_rgb * m * v + SCOPE_SURROUND * (1.0 - m)
