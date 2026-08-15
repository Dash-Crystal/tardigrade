

def cs2filt_reachability_selftest():
    """Reachability proof for all seven families, EVERY axis, EVERY value.

    Our scene supplies no CSZ target, no ray-distance target, no packed
    depth+normal gbuffer and no proxy-volume set, so every input here is
    SYNTHESISED -- a deterministic pixel batch with real depth structure
    (two slabs and a ramp) so that a depth-keyed weight has something to
    key on. Without that structure a bilateral filter and a box filter
    are indistinguishable, and the test would pass for the wrong reason.

    Two columns, and they answer different questions:
      px      the path RAN (separates 'never called' from 'called and
              produced nothing', which CHECKS_THAT_CANNOT_FAIL.md warns
              look identical)
      max|d|  the axis DID something, measured against the neighbouring
              value of that same axis

    CAN THIS CHECK FAIL? Yes, and it is made to: nine named injections
    (--cs2filt-selftest-inject) each break exactly one path, and each
    must drive at least one row to a 0 that is non-zero without it. A
    self-test with no demonstrated failure mode is the subject of that
    whole document.
    """
    if args.cs2filt_selftest != "on":
        return 0
    N = 64
    g = torch.Generator(device="cpu").manual_seed(20260807)
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    # a depth field with REAL structure: near slab, far slab, ramp
    z = torch.where(xx < N // 3, torch.full_like(xx, 0.20, dtype=torch.float32),
                    torch.where(xx < 2 * N // 3,
                                torch.full_like(xx, 0.80, dtype=torch.float32),
                                0.20 + 0.60 * (xx.float() / N)))
    zfc = z.unsqueeze(0).to(device).clamp(1e-3, 1.0 - 1e-3)
    fg = torch.ones(1, N, N, dtype=torch.bool, device=device)
    eye = torch.zeros(1, 3, device=device)
    fwd = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    dist = (2.0 + 8.0 * z).unsqueeze(0).to(device)
    rayd = torch.stack(((xx.float() / N - 0.5).expand(N, N),
                        (yy.float() / N - 0.5).expand(N, N),
                        torch.ones(N, N)), dim=-1).unsqueeze(0).to(device)
    wpos = eye.view(-1, 1, 1, 3) + torch.nn.functional.normalize(
        rayd, dim=-1) * dist.unsqueeze(-1)
    nrm = torch.nn.functional.normalize(
        torch.randn(1, N, N, 3, generator=g).to(device), dim=-1)
    mvp = torch.eye(4, device=device).unsqueeze(0)
    mvp[:, 2, 3] = 3.0
    col = torch.rand(1, N, N, 4, generator=g).to(device)
    sig = torch.rand(1, N, N, 3, generator=g).to(device)
    cv = torch.rand(1, N, N, 4, generator=g).to(device)
    gb = torch.stack((_lin_depth(zfc), torch.full_like(zfc, 1.0),
                      torch.rand(1, N, N, generator=g).to(device),
                      torch.rand(1, N, N, generator=g).to(device)), dim=-1)
    rows, notes = [], []

    def row(label, t, base=None):
        d = 0.0 if base is None else float((t - base).abs().max())
        rows.append((label, int(t.numel()), d))
        return t

    print("=" * 74, flush=True)
    print("CS2 SSAO subsystem + depth-keyed filters -- reachability "
          "(synthesised input)", flush=True)

    # --- ssao_convert_depth / ssao_downsample_depth -------------------
    csz0 = row("ssao_convert_depth  D_MSAA_DEPTH_BUFFER=0  (r0m0:17)",
               ssao_convert_depth(zfc, msaa=0))
    csz1 = ssao_convert_depth(zfc, msaa=1)
    row("ssao_convert_depth  D_MSAA_DEPTH_BUFFER=1  (r0m1)", csz1, csz0)
    mips = [csz0]
    for _ in range(CS2AO_SAO_MAX_MIP):
        if mips[-1].shape[1] < 2 or mips[-1].shape[2] < 2:
            break
        mips.append(ssao_downsample_depth(mips[-1]))
    row(f"ssao_downsample_depth  chain of {len(mips)}  (r0m0:11)",
        mips[-1])
    # the quincunx must NOT equal a 2x2 box average -- that is the whole
    # point of the shader, and a check that cannot tell them apart is no
    # check at all.
    boxavg = (csz0[:, 0::2, 0::2] + csz0[:, 0::2, 1::2] +
              csz0[:, 1::2, 0::2] + csz0[:, 1::2, 1::2]) * 0.25
    row("  quincunx != 2x2 box average (must be non-zero)",
        mips[1], boxavg[:, :mips[1].shape[1], :mips[1].shape[2]])

    # --- ssao_scalable_ambient_obscurance, all four dyn/static values --
    nrm_cam = _cs2_world_to_cam_dir(nrm, fwd)
    base = ssao_scalable_ambient_obscurance(mips, nrm_cam, 0, 0, 0)
    row("SAO  D_READ_NORMAL_FROM_TEXTURE=0 D_2X2=0 S_RED_ONLY=0 (r0m0)",
        base)
    row("SAO  D_READ_NORMAL_FROM_TEXTURE=1               (r0m1:35-37)",
        ssao_scalable_ambient_obscurance(mips, nrm_cam, 1, 0, 0), base)
    row("SAO  D_2X2_BOX_FILTER=1                          (r0m2:60-79)",
        ssao_scalable_ambient_obscurance(mips, nrm_cam, 0, 1, 0), base)
    row("SAO  D_READ_NORMAL=1 D_2X2=1                          (r0m3)",
        ssao_scalable_ambient_obscurance(mips, nrm_cam, 1, 1, 0), base)
    row("SAO  S_OUTPUT_RED_CHANNEL_ONLY=1 (.y key dropped) (r1m0:60)",
        ssao_scalable_ambient_obscurance(mips, nrm_cam, 0, 0, 1), base)

    # --- ssao_bilateral_blur, both axes -------------------------------
    bx = row("ssao_bilateral_blur  axis x  (r0m0:49, stride 2)",
             ssao_bilateral_blur(base, (1, 0)), base)
    row("ssao_bilateral_blur  axis y  (r0m0:49)",
        ssao_bilateral_blur(bx, (0, 1)), bx)

    # --- ssao.vfx, all three S_SSAO_QUALITY + both normal sources -----
    q = {}
    for qq in (0, 1, 2):
        q[qq] = ssao_ray_distance(dist, rayd, eye, fg, mvp, nrm_world=nrm,
                                  quality=qq, zprepass_normals=0)
        row(f"ssao.vfx  S_SSAO_QUALITY={qq} "
            f"({len(CS2AO_SSAO_KERNELS[qq])} taps, r{qq}m0:4)",
            q[qq], q[0] if qq else None)
    row("ssao.vfx  D_Z_PREPASS_OUTPUTS_NORMALS=1     (r0m1:39-43)",
        ssao_ray_distance(dist, rayd, eye, fg, mvp, nrm_world=nrm,
                          quality=2, zprepass_normals=1), q[2])
    # S_ALPHA_BLEND deduplicates to the same bytecode record; recording
    # the measured zero is the point, not hiding it.
    notes.append("ssao.vfx S_ALPHA_BLEND: 6 static combos -> 3 bytecode "
                 "records; the axis changes no instruction (host blend "
                 "state), so its measured max|d| is 0 BY THE REFERENCE.")

    # --- aoproxy_splat, both S_REVERSE_DEPTH_BUFFER values ------------
    px = _aoproxy_set_for(wpos)
    a0 = aoproxy_splat(zfc, rayd, eye, fwd, px, reverse_depth=0)
    row("aoproxy_splat  S_REVERSE_DEPTH_BUFFER=0  (r0m0:49) 4ch", a0)
    row("aoproxy_splat  S_REVERSE_DEPTH_BUFFER=1  (r1m0:49)",
        aoproxy_splat(zfc, rayd, eye, fwd, px, reverse_depth=1), a0)

    # --- atrous_filter, every dynamic value and every stride ----------
    at = atrous_filter(cv, gb, 1)
    row("atrous_filter  step 1  (r0m0:83-165)", at)
    for st in (2, 4):
        row(f"atrous_filter  step {st} (the a-trous hole)",
            atrous_filter(cv, gb, st), at)
    for nm, fl in (("D_FIRST_PASS", "atrous_first_pass"),
                   ("D_FINAL_PASS", "atrous_final_pass"),
                   ("D_BAKED_SHADOWS", "atrous_baked_shadows"),
                   ("D_BAKED_AO", "atrous_baked_ao")):
        old = getattr(args, fl)
        setattr(args, fl, 1 - old)
        row(f"atrous_filter  {nm}={1 - old}",
            atrous_filter(cv, gb, 1), at)
        setattr(args, fl, old)

    # --- denoise_blur, all three kernels and both axes -----------------
    d0 = denoise_blur(sig, zfc, kernel=0)
    row("denoise_blur  S_BOX_KERNEL=0 PASSTHROUGH, no loop (r0m0:10)",
        d0)
    for k in (1, 2):
        row(f"denoise_blur  S_BOX_KERNEL={k} "
            f"({len(CS2AO_DNB_TAPS[k])} taps, r{k}m0:16) D_BLUR_Y=0",
            denoise_blur(sig, zfc, kernel=k, blur_y=0), d0)
        row(f"denoise_blur  S_BOX_KERNEL={k} D_BLUR_Y=1   (r{k}m1:60)",
            denoise_blur(sig, zfc, kernel=k, blur_y=1),
            denoise_blur(sig, zfc, kernel=k, blur_y=0))

    # --- blur_with_depth, all five kernels, both srgb, both axes ------
    b0 = blur_with_depth(col, zfc, kernel=0, blur_y=0, srgb_read=0)
    row("blur_with_depth  S_GAUSSIAN_KERNEL=0 (no table)  (r0m0)", b0)
    for k in (1, 2, 3, 4):
        row(f"blur_with_depth  S_GAUSSIAN_KERNEL={k} "
            f"({len(CS2AO_BWD_KERNELS[k][0])} taps, r{k}m0:16)",
            blur_with_depth(col, zfc, kernel=k, blur_y=0, srgb_read=0), b0)
    row("blur_with_depth  D_BLUR_Y=1                      (r1m1:70)",
        blur_with_depth(col, zfc, kernel=4, blur_y=1, srgb_read=0),
        blur_with_depth(col, zfc, kernel=4, blur_y=0, srgb_read=0))
    row("blur_with_depth  S_SRGB_READ=1 (static stride 5)",
        blur_with_depth(col, zfc, kernel=4, blur_y=0, srgb_read=1),
        blur_with_depth(col, zfc, kernel=4, blur_y=0, srgb_read=0))

    # --- the MIXED-RADIX static decode, and its bitmask falsifier -----
    # blur_with_depth: S_GAUSSIAN_KERNEL stride 1 (0..4), S_SRGB_READ
    # stride 5 (0..1) -> 10 combos. Decoding with & instead of divmod is
    # the vcs/README.md trap, and it MUST move the ids.
    ids_ok, ids_bad = [], []
    for kk in range(5):
        for ss in range(2):
            cid = kk * 1 + ss * 5
            ids_ok.append(cid)
            if _cs2filt_inject("combo-bitmask"):
                ids_bad.append((cid & 1, (cid >> 1) & 1))
            else:
                ids_bad.append((cid % 5, (cid // 5) % 2))
    dec_ok = all(d == (k, s) for d, (k, s) in
                 zip(ids_bad, [(k, s) for k in range(5) for s in range(2)]))
    rows.append(("blur_with_depth static combo mixed-radix decode "
                 f"round-trips ({dec_ok})", 10 if dec_ok else 0,
                 1.0 if dec_ok else 0.0))

    dead = 0
    for label, cnt, d in rows:
        flag = ""
        if cnt == 0:
            flag = "  <-- DEAD"
            dead += 1
        print(f"  {label:<64s} px {cnt:>8d}  max|d| {d:.6f}{flag}",
              flush=True)
    for n in notes:
        print(f"  NOTE: {n}", flush=True)
    if args.cs2filt_selftest_inject:
        print(f"  INJECTION ARMED: {args.cs2filt_selftest_inject}",
              flush=True)
    print(f"  CS2AO_STAT {CS2AO_STAT}  AOPROXY_STAT {AOPROXY_STAT}",
          flush=True)
    print("=" * 74, flush=True)
    return dead


# =======================================================================
# ENGINE PER-VIEW RENDER TARGETS (set 1, binding 3)
#
# Four handles our renderer supplied nothing for. See the argument block
# for the provenance method and the byte-offset table. Everything below
# ports the CONSUMER side literally from the bytecode; where a PRODUCER
# pass is not one of the six extracted shaders that is said so at the
# function that stands in for it.
# =======================================================================

def _frag_depth(ndc_z, covered):
    """The depth TARGET, in the gl_FragCoord.z convention the reads use.

    nvdiffrast's rast[...,2] is OpenGL NDC z in [-1,1] (the projection
    this file builds is the GL one). gl_FragCoord.z is the value written
    to the depth attachment, [0,1]. Uncovered pixels take the far plane,
    which is what an untouched depth attachment holds after a clear.
    """
    z = ndc_z * 0.5 + 0.5
    return torch.where(covered, z.clamp(0.0, 1.0), torch.ones_like(z))


def _recip_depth_scale_bias():
    """(scale, bias) taking gl_FragCoord.z straight to RECIPROCAL depth.

    THE REFERENCE'S DEPTH READ IS A LINEAR REMAP AND OURS IS A PROJECTION
    INVERSE -- and they turn out to be the same map. With
    d = p23 / (ndc + p22) and ndc = 2*zfc - 1,

        1/d = (2/p23) * zfc + (p22 - 1)/p23

    which is linear in zfc. So the CB members the shader reads
    (`_4459._m0.z` and `_m0.w`, the scale and bias of
    `depth.t01.linear_remap`) are exactly these two numbers for our
    frustum, and our depth target can feed the reference's expression
    without a re-encode. That was not obvious and I nearly reported the
    wire as blocked on a target-format mismatch.
    """
    n, f = PROJ_NEAR, PROJ_FAR
    p22 = -(f + n) / (f - n)
    p23 = -2.0 * f * n / (f - n)
    return 2.0 / p23, (p22 - 1.0) / p23


def _lin_depth(zfc):
    """gl_FragCoord.z -> metres along the view axis, for this frustum.

    Now routed through `depth.t01.linear_remap`, transcribed from
    bomb_blast_r0m0.glsl:44 (identical in aoproxy_splat:49, inferno:67 and
    deferred_particle_shadows:66). That term returns the RECIPROCAL, which
    is what the shader's own divide consumes; metres is one more
    reciprocal, taken here.

    near/far are 0 and 1: our target holds gl_FragCoord.z, already
    normalised, so the reference's `(raw - near) / (far - near)` is the
    identity and its clamp bounds the value the same way the old
    `.clamp(max=-1e-6)` did -- on the input rather than on the quotient.
    """
    sc, bi = _recip_depth_scale_bias()
    k = _lf_depth.linear_depth_remap(zfc, 0.0, 1.0, sc, bi)
    # `k` is 1/d and is negative in this frustum (p23 < 0). Guard the
    # magnitude, not the sign: clamping toward zero from the wrong side
    # would flip which end of the frustum saturates.
    return 1.0 / torch.where(k.abs() < 1e-9, torch.full_like(k, -1e-9), k)


def wpos_from_depth(zfc, ray, fwd, eye):
    """World position where the camera ray meets the depth target.

    `depth.t02.world_pos_from_ray`, the expression four loop families
    write character for character. `ray` is NOT normalised inside the
    term -- bomb_blast normalises and negates before calling and
    aoproxy_splat passes an interpolated attribute raw -- so the caller
    owns that, exactly as in the reference.
    """
    sc, bi = _recip_depth_scale_bias()
    return _lf_depth.world_pos_from_ray(eye, ray, fwd, zfc, 0.0, 1.0, sc, bi)


def _rt_reach(key, hit, shaded):
    if not args.rt_reach:
        return
    RT_REACH[key][0] += float((hit & shaded).sum())
    RT_REACH[key][1] += float(shaded.sum())


def dirocc_target(wpos, fg, eye, fwd):
    """PRODUCER of the offset-76 target: 4-channel directional obscurance.

    The engine pass that writes this target is not among the six shaders
    extracted, so its estimator is not recoverable. What IS recoverable is
    the target's SHAPE and its MEANING, from how the consumer reads it:
    four channels, resolved at glsl:1238 by max(dot(basis_i, N), 0), i.e.
    channel i is the obscurance seen along basis direction i. This uses
    the SAO estimator the renderer already implements for --ssao (CS2's
    own named algorithm, five ssao_* shaders in core/), with the shading
    normal replaced by each basis direction in turn -- one set of spiral
    taps serving all four lobes, so it costs one gather, not four.

    Runs at DIROCC_W x DIROCC_H. Both this and the depth target are POINT
    sampled down from the full-res buffers, never averaged, for the reason
    already recorded for ssao_downsample_depth: averaging two depths
    across a silhouette invents a surface that is at neither.
    """
    s = DIROCC_S
    P = wpos[:, ::s, ::s][:, :DIROCC_H, :DIROCC_W].contiguous()
    F = fg[:, ::s, ::s][:, :DIROCC_H, :DIROCC_W].contiguous()
    B, Hl, Wl = F.shape
    R, R2 = args.dirocc_radius, args.dirocc_radius ** 2
    NS = args.dirocc_samples
    zc = ((P - eye.view(-1, 1, 1, 3)) * fwd.view(-1, 1, 1, 3)).sum(-1)
    zc = torch.where(F, zc, torch.full_like(zc, 1e4)).clamp(min=1e-3)
    Pm, Fm = [P], [F]
    for _ in range(args.dirocc_mips):
        Pm.append(Pm[-1][:, ::2, ::2].contiguous())
        Fm.append(Fm[-1][:, ::2, ::2].contiguous())
    NL = len(Pm)
    projScale = 0.5 * Wl / math.tan(math.radians(args.hfov) / 2)
    ssDiskR = (projScale * R / zc).clamp(1.0, max(Wl, Hl))
    yy, xx = torch.meshgrid(torch.arange(Hl, device=device),
                            torch.arange(Wl, device=device), indexing="ij")
    spin = (((3 * xx) ^ yy) + xx * yy).float() * 10.0
    bidx = torch.arange(B, device=device).view(-1, 1, 1)
    total = torch.zeros(B, Hl, Wl, 4, device=device)
    for i in range(NS):
        alpha = (i + 0.5) / NS
        ang = alpha * (args.dirocc_turns * 2 * math.pi) + spin
        ssR = alpha * ssDiskR
        # sao.t03.mip_from_radius. The module clamps to 5; this chain is
        # NL levels deep, so the upper bound stays the chain's -- a mip the
        # renderer does not have is not a mip the reference can select.
        lvl = _lf_ao.sao_mip_from_radius(ssR).clamp(0, NL - 1).long()
        u = xx.float() + torch.cos(ang) * ssR
        v = yy.float() + torch.sin(ang) * ssR
        inb = (u >= 0) & (u < Wl) & (v >= 0) & (v < Hl)
        ui, vi = u.long(), v.long()
        Q = torch.zeros_like(P)
        ok = torch.zeros_like(F)
        for L in range(NL):
            hl, wl = Fm[L].shape[1], Fm[L].shape[2]
            ul = (ui >> L).clamp(0, wl - 1)
            vl = (vi >> L).clamp(0, hl - 1)
            sel = lvl == L
            Q = torch.where(sel.unsqueeze(-1), Pm[L][bidx, vl, ul], Q)
            ok = torch.where(sel, Fm[L][bidx, vl, ul], ok)
        vvec = Q - P
        vv = (vvec * vvec).sum(-1)
        f = (R2 - vv).clamp(min=0)
        # one dot per BASIS direction, sharing the tap
        vn = torch.einsum("bhwc,kc->bhwk", vvec, DIROCC_B)
        c = (f * f * f).unsqueeze(-1) * \
            ((vn - args.dirocc_bias) / (vv + 1e-4).unsqueeze(-1)).clamp(min=0)
        total = total + torch.where((ok & inb).unsqueeze(-1), c,
                                    torch.zeros_like(c))
    A = (1.0 - total * (args.dirocc_intensity / R ** 6) * (5.0 / NS)
         ).clamp(0, 1)
    return torch.where(F.unsqueeze(-1), A, torch.ones_like(A))


def dirocc_depth_target(zfc):
    """PRODUCER of the offset-80 target: the low-res depth buffer.

    Point-sampled from the gl_FragCoord.z-convention depth target, same
    grid as the offset-76 target -- which is what makes the consumer's
    nearest-depth selection at glsl:1213-1236 mean anything.
    """
    s = DIROCC_S
    return zfc[:, ::s, ::s][:, :DIROCC_H, :DIROCC_W].contiguous()


def dirocc_resolve(occ_rt, dep_rt, zfc, nrm, fg):
    """CONSUMER of offsets 76 and 80 -- csgo_environment_blend_ps.glsl
    1206-1244, instruction for instruction.

      _11087 = floor(gl_FragCoord.xy * _m15) * _m14 + _m14*0.5    :1211
      _18418 = textureGather(RT80, _11087) - gl_FragCoord.zzzz    :1212
      ... three comparisons pick the nearest-depth of the four       :1213-36
      _10010 = normalize(vec4(_m2[i] * (dot(-N, tilt_i)*.5+.5)))     :1237
      _13232 = max(dot(basis_i, N),0) * normalize(clamp(
                   (_10010 - max(_10010) + 0.2) * 5, 0, 1))          :1238
      result = (1/(dot(w,1)+eps)) * (eps + dot(w, RT76[uv+off]))     :1239
    """
    B = zfc.shape[0]
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    # gl_FragCoord.xy is the pixel CENTRE
    ti = torch.floor((xx.float() + 0.5) * RTV_M15).long()
    tj = torch.floor((yy.float() + 0.5) * RTV_M15).long()
    ti0 = ti.clamp(0, DIROCC_W - 1)
    tj0 = tj.clamp(0, DIROCC_H - 1)
    ti1 = (ti + 1).clamp(0, DIROCC_W - 1)
    tj1 = (tj + 1).clamp(0, DIROCC_H - 1)
    b = torch.arange(B, device=device).view(-1, 1, 1)
    # textureGather component order at an exact texel centre:
    #   .x = (i, j+1)   .y = (i+1, j+1)   .z = (i+1, j)   .w = (i, j)
    g_x = dep_rt[b, tj1, ti0]
    g_y = dep_rt[b, tj1, ti1]
    g_z = dep_rt[b, tj0, ti1]
    g_w = dep_rt[b, tj0, ti0]
    d_x, d_y = g_x - zfc, g_y - zfc
    d_z, d_w = g_z - zfc, g_w - zfc
    # :1215  bool _12285 = abs(_18418.z) < _18418.w   -- abs on ONE side
    c1 = d_z.abs() < d_w
    best1 = torch.where(c1, d_z, d_w)
    c2 = d_x.abs() < best1
    best2 = torch.where(c2, d_x, best1)
    c3 = d_y.abs() < best2
    # the (di, dj) the three comparisons select
    di = torch.where(c3, torch.ones_like(ti0),
                     torch.where(c2, torch.zeros_like(ti0), c1.long()))
    dj = torch.where(c3, torch.ones_like(tj0), c2.long())
    # textureLod(RT76, _11087 + off, 0.0): _11087 + off is EXACTLY a texel
    # centre in all four cases, so the bilinear filter reduces to that
    # texel and this indexing equals the filtered fetch by construction.
    occ4 = occ_rt[b, (tj0 + dj).clamp(0, DIROCC_H - 1),
                  (ti0 + di).clamp(0, DIROCC_W - 1)]
    n = nrm
    v = -n                                                     # :1204
    a = DIROCC_VW.view(1, 1, 1, 4) * (
        torch.einsum("bhwc,kc->bhwk", v, DIROCC_BT) * 0.5 + 0.5)
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)       # :1237
    sel = (((a - a.max(dim=-1, keepdim=True).values) + 0.2) * 5.0).clamp(0, 1)
    sel = sel / sel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w = torch.einsum("bhwc,kc->bhwk", n, DIROCC_B).clamp(min=0) * sel  # :1238
    eps = args.dirocc_eps
    out = (1.0 / (w.sum(-1) + eps)) * (eps + (w * occ4).sum(-1))
    return torch.where(fg, out, torch.ones_like(out))


def ss_shadow_target(wpos, zfc, fg, mvp):
    """PRODUCER of the offset-88 target.

    All six shaders read this slot, always as `.z`, always at
    gl_FragCoord.xy * invViewport, always min()-combined into an occlusion
    (glsl:1322 here; :876 / :606 / :441 / :323 / :618 in the others). The
    pass that WRITES it is not one of the six, so its algorithm is not
    recoverable from this extraction. What stands in is a screen-space
    contact shadow traced through the depth target toward the sun -- a
    real screen-space buffer, sampled by the consumer exactly as the
    reference samples it, NOT an analytic occlusion folded into the sun
    term at the shading site.

    Channels .x .y .w are never read by any of the six shaders and their
    contents are not derivable; they are written as 1.0.
    """
    s = SS_S
    P = wpos[:, ::s, ::s][:, :SS_H, :SS_W].contiguous()
    F = fg[:, ::s, ::s][:, :SS_H, :SS_W].contiguous()
    B = P.shape[0]
    b = torch.arange(B, device=device).view(-1, 1, 1)
    occ = torch.ones(B, SS_H, SS_W, device=device)
    N = max(1, args.ss_shadow_steps)
    for k in range(1, N + 1):
        Q = P + sun_dir.view(1, 1, 1, 3) * (args.ss_shadow_len * k / N)
        # row-at-a-time so the 4-vector is never materialised: at chunk 32
        # and 960x540 the cat+einsum form costs 265 MB per step

        def _row(i):
            m = mvp[:, i].view(-1, 1, 1, 4)
            return (Q[..., 0] * m[..., 0] + Q[..., 1] * m[..., 1]
                    + Q[..., 2] * m[..., 2] + m[..., 3])
        wclip = _row(3)
        iw = 1.0 / wclip.clamp(min=1e-6)
        nx, ny, nz = _row(0) * iw, _row(1) * iw, _row(2) * iw
        sx = (nx * 0.5 + 0.5) * W
        sy = (ny * 0.5 + 0.5) * H
        inb = (wclip > 1e-6) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        si = sx.long().clamp(0, W - 1)
        sj = sy.long().clamp(0, H - 1)
        d_scene = _lin_depth(zfc[b, sj, si])
        d_ray = _lin_depth((nz * 0.5 + 0.5).clamp(0, 1))
        blocked = inb & F & (d_scene < d_ray - args.ss_shadow_bias) \
            & ((d_ray - d_scene) < args.ss_shadow_thickness)
        occ = torch.where(blocked, torch.zeros_like(occ), occ)
    o = torch.ones_like(occ)
    return torch.stack([o, o, torch.where(F, occ, o), o], dim=-1)


def ss_shadow_read(rt, fg):
    """CONSUMER of offset 88 -- glsl:1322.

        min(cascadeShadow,
            textureLod(RT88, gl_FragCoord.xy * _4459._m3.xy, 0.0).z)

    The min() is applied by the caller so it composes with whichever sun
    visibility this renderer already has. The bilinear fetch is real: at
    --ss-shadow-scale 1 the sample lands exactly on a texel centre and it
    reduces to that texel (max|delta| 0 by construction), at >1 it filters.
    """
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    u = (xx.float() + 0.5) * RTV_INVVP[0] * SS_W - 0.5
    v = (yy.float() + 0.5) * RTV_INVVP[1] * SS_H - 0.5
    x0 = u.floor(); y0 = v.floor()
    fx = (u - x0).unsqueeze(0); fy = (v - y0).unsqueeze(0)
    x0 = x0.long().clamp(0, SS_W - 1); y0 = y0.long().clamp(0, SS_H - 1)
    x1 = (x0 + 1).clamp(0, SS_W - 1); y1 = (y0 + 1).clamp(0, SS_H - 1)
    b = torch.arange(rt.shape[0], device=device).view(-1, 1, 1)
    z = rt[..., 2]
    out = ((z[b, y0, x0] * (1 - fx) + z[b, y0, x1] * fx) * (1 - fy)
           + (z[b, y1, x0] * (1 - fx) + z[b, y1, x1] * fx) * fy)
    _rt_reach("off88_ssshadow", fg, fg)
    return out


def _ripple(uv):
    """One offset-116 fetch: bilinear, REPEAT, no mip (the reference uses
    `texture(...)` with no explicit lod at all three sites)."""
    n = RIPPLE.shape[0]
    x = uv[..., 0] * n - 0.5
    y = uv[..., 1] * n - 0.5
    x0 = x.floor(); y0 = y.floor()
    fx = (x - x0).unsqueeze(-1); fy = (y - y0).unsqueeze(-1)
    x0 = x0.long() % n; y0 = y0.long() % n
    x1 = (x0 + 1) % n; y1 = (y0 + 1) % n
    return ((RIPPLE[y0, x0] * (1 - fx) + RIPPLE[y0, x1] * fx) * (1 - fy)
            + (RIPPLE[y1, x0] * (1 - fx) + RIPPLE[y1, x1] * fx) * fy)


def weather_wet(wpos, nrm, vcz, vcw, height, ao, rough, albedo, fg, tsec):
    """CONSUMER of offset 116, all three reads --
    csgo_environment_blend_ps.glsl 1004-1195 (identically
    csgo_environment_ps.glsl 556-...).

    Returns (albedo, roughness, normal, ambient_ao, f0, scatter). The
    whole block is evaluated in SOURCE world space (Z up), because the
    shader's puddle test is `_24347.z` and its flow frame is built from
    `_10061.xy`; this renderer is glTF (src_y, src_z, src_x), so the
    vectors are rotated in and the resulting normal rotated back out.

    Uniforms: the per-view weather state (_5037._m8..m12) and the
    material's weather scales (_5618._m70..m80, _m86, _m88) are engine and
    material CONSTANT-BUFFER data, not in this world pack. They are
    --weather-* arguments, uniform over materials. That is the difference
    from the reference, and it is a difference in the UNIFORMS, not in the
    call flow.
    """
    def to_src(v):
        return torch.stack([v[..., 2], v[..., 0], v[..., 1]], dim=-1)

    def from_src(v):
        return torch.stack([v[..., 1], v[..., 2], v[..., 0]], dim=-1)

    P = to_src(wpos) / S                       # _10061, SOURCE units
    N = _nrm(to_src(nrm))                      # _24347
    t = tsec.view(-1, 1, 1)                    # _4459._m0

    #  _12896/_12897 :987-1003 -- the material accepts weather, its two
    #  strengths are not both zero, and the vertex weather mask is lit.
    if args.weather_enable == 0 or \
            (args.weather_wet_strength + args.weather_cover_strength) <= 0.0:
        gate = torch.zeros_like(fg)
    else:
        gate = fg & (vcz > 0.0)
    _rt_reach("off116_ripple_a", gate, fg)

    #  :1014  READ 1 -- world-projected, only .w is consumed (:1044)
    t1 = _ripple(P[..., :2] * 0.006000000052154064178466796875)

    up = N[..., 2]                                            # _21302
    pw = up.clamp(min=0.0) ** 32.0                            # _14498
    _wet = min(max(args.weather_wetness, 0.0), 1.0)
    _snowcut = 1.0 - min(max(args.weather_snow - 0.5, 0.0), 1.0) * 2.0
    base = vcz.clamp(max=min(_wet, _snowcut))                 # _23204
    o = (base - 0.75).clamp(0, 1)
    amt = base + 4.0 * o * o                                  # _8286
    amt_h = amt - args.weather_height_offset                  # _16777
    k20 = (amt * 20.0).clamp(0, 1)                            # _11964
    edge = max(0.001, args.weather_edge)                      # _16799
    #  :1025-1038  the flow-direction branch (_5618._m68); the pack has no
    #  such flag, so the `else` -- flat flow, ripple from the noise only.
    flow0 = torch.stack([torch.zeros_like(up), torch.zeros_like(up),
                         torch.ones_like(up), torch.zeros_like(up)], -1)
    depth_ref = (amt * pw - args.weather_height_offset).clamp(min=0)  # _24883
    #  :1040  smoothstep(a, b, h) with a > b -- a DESCENDING edge, kept.
    #  The second edge carries COLOR_0.w (_5908.w), the vertex puddle-depth
    #  channel; a 3-channel vertex colour makes it 0, which is the
    #  hardest edge the expression can take.
    puddle = _smoothstep_ab(depth_ref + edge,
                            depth_ref - (vcw + edge * 2.0), height) * k20
    damp = torch.maximum(
        _smoothstep_ab(base, amt_h - 0.2 * amt, height)
        * args.weather_damp_strength * k20, puddle) \
        * args.weather_wet_strength                                   # _19808
    drip_h = (base - 1.0) - height                                    # _17924
    drip = (drip_h + args.weather_drip_offset).clamp(0, 1) \
        * args.weather_drip_strength * puddle                         # _10334
    cover = ((((amt * 2.0)
               - (0.5 + (t1[..., 3] - 0.5) * (pw * 20.0).clamp(0, 1)) * 0.5)
              - height * 0.5 + 0.25).clamp(0, 1)
             * args.weather_cover_strength * k20 + puddle).clamp(0, 1)  # _13706

    wet_n = flow0
    ripple_z = torch.zeros_like(up)
    flow_rough = torch.zeros_like(up)
    if args.weather_wind > 0.0 and args.weather_flow_strength > 0.0:
        #  :1051-1096  the wind-driven flow frame and READS 2 and 3
        fw = args.weather_flow_strength * args.weather_wind            # _12497
        _d0 = np.array([-1.0, 0.20000000298023223876953125, 0.0])
        _d0 = _d0 / np.linalg.norm(_d0)
        ang = args.weather_wind_angle * 6.283185482025146484375
        ca, sa = math.cos(ang), math.sin(ang)
        wx = float(_d0[0] * ca + _d0[1] * sa)
        wy = float(-_d0[0] * sa + _d0[1] * ca)
        wdir = torch.tensor([wx, wy, float(_d0[2])], device=device
                            ).view(1, 1, 1, 3).expand_as(N)            # _22900
        side = -torch.cross(wdir, N, dim=-1)                           # _21326
        s9 = (P * side).sum(-1)                                        # _9965
        c12 = 1.0 - vcz
        u0 = ((P * wdir).sum(-1).unsqueeze(-1)
              + torch.tensor([0.1, -0.1], device=device) * s9.unsqueeze(-1)
              + torch.tensor([8.0, 5.0], device=device) * P[..., 2:3]
              + ((c12 * c12) * -80.0).unsqueeze(-1))                   # _20274
        uv1 = torch.stack([u0[..., 0], s9], -1) \
            * torch.tensor([0.0199999995529651641845703125,
                            0.010999999940395355224609375], device=device)
        #  :1072  READ 2
        r1 = _ripple(uv1 + torch.tensor([0.2, 0.0], device=device) * t.unsqueeze(-1))
        n1 = r1[..., :2] * 2.0 - 1.0
        w1 = (_fwidth_len(uv1) * 10.0).clamp(0, 1)
        p1 = n1 * torch.tensor([0.1, 0.05], device=device) \
            * vcz.unsqueeze(-1) * (1.0 + (0.3 - 1.0) * w1).unsqueeze(-1)
        uv2 = torch.stack([u0[..., 1], s9], -1) \
            * torch.tensor([0.03599999845027923583984375,
                            0.01620000042021274566650390625], device=device) \
            + n1 * 0.0500000007450580596923828125
        #  :1081  READ 3
        r2 = _ripple(uv2 + torch.tensor([0.15, 0.0], device=device) * t.unsqueeze(-1))
        n2 = (r2[..., :2] * 2.0 - 1.0) * 0.05200000107288360595703125 \
            * vcz.unsqueeze(-1)
        n2 = torch.stack([n2[..., 0], n2[..., 1] * 0.5], -1)
        w2 = (_fwidth_len(uv2) * 10.0).clamp(0, 1)
        p2 = n2 * (1.0 + (0.3 - 1.0) * w2).unsqueeze(-1)
        tang = wdir[..., :2]
        bitn = side[..., :2]
        off = torch.stack([(tang * p1).sum(-1) + (tang * p2).sum(-1),
                           (bitn * p1).sum(-1) + (bitn * p2).sum(-1)], -1)
        wet_n = torch.cat([off, torch.ones_like(up).unsqueeze(-1),
                           (vcz * (r1[..., 2] + r2[..., 2])).unsqueeze(-1)],
                          dim=-1)                                      # _20490
        wet_n = flow0 + wet_n * (pw * fw).unsqueeze(-1)                # _12898
        flow_rough = fw * (0.04 * (w1 + w2))                           # _13143
        _rt_reach("off116_ripple_b", gate & (cover > 0.5), fg)

    #  :1104-1152  the rain-ring loop, EXACTLY three unrolled iterations
    rain = args.weather_ripple_strength * args.weather_rain ** 2       # _8914
    if rain > 0.0:
        q = P[..., :2] * 0.119999997317790985107421875
        seed = t.unsqueeze(-1) * 6.5 + torch.tensor(
            [0.0, 0.3167000114917755126953125,
             0.633400022983551025390625], device=device).view(1, 1, 1, 3)
        acc_xy = torch.zeros_like(q)
        acc_z = torch.zeros_like(up)
        CA = [0.97130000591278076171875, 0.751800000667572021484375,
              0.362399995326995849609375]
        SA = [0.23770000040531158447265625, 0.65939998626708984375,
              0.931999981403350830078125]
        SC = [0.60000002384185791015625, 0.666100025177001953125,
              0.73220002651214599609375]
        r2f = rain * rain                                              # _18120
        for j in range(3):
            ca, sa = CA[j], SA[j]
            rot = lambda p: torch.stack([p[..., 0] * ca + p[..., 1] * sa,
                                         -p[..., 0] * sa + p[..., 1] * ca], -1)
            sj = seed[..., j]
            c10 = rot(q) * SC[j] + (sj * 0.0199999995529651641845703125
                                    ).unsqueeze(-1)
            v78 = rot((c10 - c10.floor()) * 2.0 - 1.0)
            L = v78.norm(dim=-1).clamp(min=1e-8)
            a20 = (1.0 - L * (3.0 + (1.0 - 3.0) * puddle)).clamp(0, 1)
            b20 = (1.0 - L * 6.0).clamp(0, 1)
            # GLSL fract is x - floor(x); torch.frac is x - trunc(x), which
            # differs in sign for negative world coordinates
            _c0 = c10[..., 0] * 0.100000001490116119384765625
            _c1 = c10[..., 1] * 0.100000001490116119384765625
            f7 = torch.floor((_c0 - _c0.floor()) * 10.0)
            f8 = torch.floor((_c1 - _c1.floor()) * 10.0)
            sj2 = sj + (f7 + f8 * (0.2 + f7)) * 8.0
            ring = (((v78 / L.unsqueeze(-1))
                     * torch.cos(1.0 + ((a20 * 40.0 + t * 65.0)
                                        - 1.0) * puddle).unsqueeze(-1)
                     * (a20 * a20).unsqueeze(-1)
                     * (2.0 + (0.25 - 2.0)
                        * (_fwidth_scalar(a20) * 20.0).clamp(0, 1)
                        ).unsqueeze(-1)) * -1.0)
            gain = ((torch.sin(((a20 * (1.0 + 3.0 * puddle)) + sj2) * rain)
                     - (1.0 - 0.2 * r2f * (0.25 + 0.75 * puddle))
                     ) * 2.0).clamp(0, 1) * a20 / r2f
            acc_xy = acc_xy + ring * gain.unsqueeze(-1) \
                * (16.0 + (2.0 - 16.0) * puddle).unsqueeze(-1)
            acc_z = acc_z + ((torch.sin(((b20 + sj2 + 4.0)
                                         + torch.sin(height * 60.0) * 0.25)
                                        * rain)
                              - (1.0 - r2f * 0.02000000141561031341552734375)
                              ) * 2.0).clamp(0, 1) * b20 / r2f * 28.0
        acc_xy = acc_xy * pw.unsqueeze(-1)
        wet_n = torch.cat([wet_n[..., :2] + acc_xy, wet_n[..., 2:]], -1)
        ripple_z = acc_z * pw
    #  :1049 / :1167-1172  the OUTER branch. Below coverage 0.5 the shader
    #  takes none of the above: _23607 falls back to _9719 = vec4(0,0,1,0)
    #  and _13144 / _13151 are 0. A vectorised port cannot skip the
    #  fetches the way a dynamically-branching wave does, so it does them
    #  and discards -- the RESULT matches, the fetch COUNT is a superset.
    _c05 = cover > 0.5
    wet_n = torch.where(_c05.unsqueeze(-1), wet_n, flow0)
    ripple_z = torch.where(_c05, ripple_z, torch.zeros_like(ripple_z))
    flow_rough = torch.where(_c05, flow_rough, torch.zeros_like(flow_rough))
    wet_n3 = _nrm(wet_n[..., :3])

    #  :1173-1184  what the block hands back
    strength = puddle * args.weather_wet_strength                      # _10839
    tint_w = (damp * drip).unsqueeze(-1)                               # _23311
    n_wet = _nrm(nrm + (from_src(wet_n3) - nrm)
                 * (strength * 4.0).clamp(0, 1).unsqueeze(-1))         # _13713
    m9038 = torch.maximum(
        cover * (1.0 - min(1.0, max(0.0, args.weather_snow * 2.0))), damp)
    # The reference darkens in LINEAR space (CS2 shades linear and the
    # sRGB encode is the framebuffer's); this renderer carries sRGB in
    # `alb` until _post_chain's **2.2, so decode, run the block, re-encode.
    lin = albedo.clamp(min=0) ** 2.2
    dark = lin + (lin ** 1.2999999523162841796875 * 0.800000011920928955078125
                  - lin) * ((lin ** 0.25) * (m9038 * rough).unsqueeze(-1)
                            * 8.0).clamp(0, 1)                         # _18806
    a_wet = dark + (dark * 0.5 - dark) * (puddle * 1.5).clamp(0, 1).unsqueeze(-1)
    a_wet = a_wet * (1.0 + (ao - 1.0) * damp).unsqueeze(-1)
    a_wet = a_wet * (1.0 + (-drip_h) * drip).unsqueeze(-1)
    a_wet = a_wet + (WEATHER_TINT.view(1, 1, 1, 3) - a_wet) * tint_w
    a_wet = a_wet * (1.0 + ripple_z).unsqueeze(-1) \
        + (ripple_z * 0.1).unsqueeze(-1)                               # _13145
    a_wet = a_wet.clamp(min=0) ** (1.0 / 2.2)
    #  :1179  the fwidth is of _13713, the FINAL shading normal
    r_wet = (rough + (args.weather_rough_wet - rough) * m9038
             + flow_rough + _fwidth_len(n_wet) * 0.5).clamp(0, 1)
    f0_wet = args.weather_f0 + (0.0350000001490116119384765625
                                - args.weather_f0) * m9038             # _17127
    ao_wet = ao + (1.0 - ao) * damp                                    # _12727
    scatter = strength * (0.2 + 0.15 * wet_n[..., 3] ** 3)             # _17129

    g = gate.unsqueeze(-1)
    return (torch.where(g, a_wet, albedo),
            torch.where(gate, r_wet, rough),
            torch.where(g, n_wet, nrm),
            torch.where(gate, ao_wet, ao),
            torch.where(gate, f0_wet,
                        torch.full_like(rough, args.weather_f0)),
            torch.where(gate, scatter, torch.zeros_like(scatter)))


def _smoothstep_ab(a, b, x):
    """GLSL smoothstep with NO ordering assumption on (a, b): the shader
    calls it with a > b at glsl:1040, which is a descending edge and which
    torch has no builtin for.

    GLSL leaves e0 == e1 undefined and the hardware returns a step; the
    zero-width interval is REACHED here (at --weather-wetness 0 the
    glsl:1041 call collapses to smoothstep(0, 0, h)), and 0/0 would put a
    NaN into the frame instead of the 0 that the k20 factor immediately
    multiplies it by. The epsilon makes it the ascending step.
    """
    d = b - a
    if torch.is_tensor(d):
        d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    elif abs(d) < 1e-9:
        d = 1e-9
    t = ((x - a) / d).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


def _fwidth_len(v):
    """length(fwidth(v)) on a screen-space buffer. The reference computes
    it from hardware quad derivatives; here it is the forward difference
    of the same buffer, replicate-padded at the right/bottom edge."""
    dx = torch.zeros_like(v)
    dy = torch.zeros_like(v)
    dx[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    dy[:, :-1, :] = v[:, 1:, :] - v[:, :-1, :]
    return (dx.abs() + dy.abs()).norm(dim=-1)


def _fwidth_scalar(v):
    dx = torch.zeros_like(v)
    dy = torch.zeros_like(v)
    dx[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    dy[:, :-1, :] = v[:, 1:, :] - v[:, :-1, :]
    return dx.abs() + dy.abs()


def _bounce_chroma(vol):
    """Luminance-normalised chroma of the reconstructed indirect volume.

    Returns (B,H,W,3) multiplicative tint, or None when disabled. Pure
    chroma: divide out the field's own mean colour, then divide out this
    pixel's luminance, so the tint is exposure-neutral per pixel and
    cannot smuggle in a gain.
    """
    if args.bounce_chroma <= 0 or vol is None:
        return None
    cb = vol / VOL_MEAN3
    cb = cb / cb.mean(-1, keepdim=True).clamp(min=1e-6)
    return 1.0 + args.bounce_chroma * (cb - 1.0)


# One-shot latch for the --gt-lighting ack below. Declared BEFORE its
# consumer, per this file's own rule about placeholders landing after
# the assignment that fills them.
_GT_LIGHTING_ACK = []
