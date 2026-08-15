

def alpha_test_prepass(a_eff, ref):
    """D_ALPHA_TEST_PREPASS -> (coverage, keep mask).

    csgo_foliage s10/d1:44-52, isolated against s10/d0 which is the same
    module with the binary test:

        OFF:  if (a_eff < ref) discard;   out.a = msaa ? a_eff : 1
        ON:   float cov = msaa
                  ? clamp(0.5 + (a_eff-ref)/max(fwidth(a_eff),1e-6), 0, 1)
                  : (a_eff - ref);
              if ((cov - 0.001) < 0.0) discard;
              out.a = msaa ? cov : 1

    The MSAA branch is the one taken whenever the per-view alpha-to-
    coverage flag is set; this renderer has no multisample target, so the
    derivative-normalised form is the one implemented -- it is also the
    only one of the two that differs from the plain test, so choosing the
    other would make the axis a no-op.

    fwidth is |dF/dx| + |dF/dy| on the 2x2 quad; here it is the same
    quantity taken with one-pixel forward differences over the screen
    buffer, which is what a 2x2 quad derivative reduces to away from
    triangle edges. AT a silhouette the quad derivative sees only the
    covered lanes and this sees the neighbour's value, so the coverage
    ramp is one pixel wider there.
    """
    dx = torch.zeros_like(a_eff)
    dy = torch.zeros_like(a_eff)
    dx[..., :, :-1] = (a_eff[..., :, 1:] - a_eff[..., :, :-1]).abs()
    dy[..., :-1, :] = (a_eff[..., 1:, :] - a_eff[..., :-1, :]).abs()
    fw = (dx + dy).clamp(min=9.9999999747524270787835121154785e-07)
    cov = (0.5 + (a_eff - ref) / fw).clamp(0, 1)
    return cov, (cov - 0.001) >= 0.0


def translucent_clip(rgb, ndv, alpha):
    """S_MODE_DEPTH on csgo_glass -> keep mask (does this pane WRITE DEPTH?).

    csgo_glass s1/d0, the whole 153-line module, verbatim:

        ndv   = max(0, dot(-normalize(wpos-eye), normalize(N)))
        col   = g_tColor.rgb * mix(1, vColor.rgb, g_flModelTintAmount)
        fres  = 1 - mix(0.04, 1.0, pow(1-ndv, 5))
        gamma = 1 / max(0.01, sqrt(1 - 0.4444444*(1 - ndv*ndv)))
        rgb   = fres * pow(col, gamma)
        a     = clamp(g_tTrans.a*vColor.a,0,1)*(remap.y-remap.x) + remap.x
        rgb  *= (1 - a)
        ... gradient fog and cubemap fog multiply rgb by (1-fog) ...
        if (0.7 < dot(rgb, vec3(0.2125, 0.7154, 0.0721))) discard;
        out = vec4(0);

    i.e. a pane that still TRANSMITS more than 70% luminance writes no
    depth and therefore does not occlude what is behind it. 0.4444444 is
    (1/1.5)^2 -- the same IOR 1.5152 the refract() constant 0.66 encodes --
    so pow(col, 1/cos_transmitted) is Beer-Lambert path length through the
    pane at the refracted angle, not a curve.

    D_DISABLE_TRANSLUCENT_CLIP=1 replaces this entire module with
    `out = vec4(0)` and NO discard (glass s1/d1, 288 bytes, 9 lines) --
    every fragment writes depth. That is --disable-translucent-clip, and
    it is checked by the caller, not here.

    WHAT DIFFERS: the two fog multiplies are applied by the caller from
    this renderer's own fog chain when --fog is on, because the fog
    parameters live in the same per-view CB either way; with --fog off the
    clip sees unfogged luminance, which is the F=0 case of the same
    expression.
    """
    f = 1.0 - (0.04 + (1.0 - 0.04) * (1.0 - ndv).clamp(0, 1) ** 5.0)
    cos_t = (1.0 - 0.4444444477558135986328125
             * (1.0 - ndv * ndv)).clamp(min=0.0).sqrt().clamp(min=0.01)
    lit = f.unsqueeze(-1) * rgb.clamp(min=0) ** (1.0 / cos_t).unsqueeze(-1)
    lit = lit * (1.0 - alpha).unsqueeze(-1)
    lum = (lit * torch.tensor([0.2125000059604644775390625,
                               0.7153999805450439453125,
                               0.07209999859333038330078125],
                              device=device)).sum(-1)
    return lum <= args.translucent_clip_threshold


def cubemap_refraction(base_rgb, wpos, nrm, view, rough, amb, on, x):
    """S_OPAQUE_CUBEMAP_REFRACTION -> (rgb, alpha).

    csgo_glass s2/d8:1114-1191, the only block the S_OPAQUE_CUBEMAP_
    REFRACTION=0 module does not have (the axis adds 3 fetch sites,
    18 -> 21, and the shader's only `Refract`):

        rough_r = mix(g_flRefractMinRoughness, 1.0, dot(rough.xy, vec2(.5)))
        R       = refract(I, N, 0.66)                    // eta, IOR 1.5152
        lod     = g_flCubeLodScale * sqrt(rough_r)
        D       = normalize(mix(R, R, ...))              // both arms are R
        for each cubemap whose box contains the pixel (first hit wins):
            p  = C.worldToLocal * vec4(wpos,1)
            d  = (C.worldToLocal * vec4(D,0)).xyz
            t  = |min3(max((C.max-p)/d, (C.min-p)/d))|   // box exit
            dir= mix(p + d*t, d, vec3(rough_r))
            col+= textureLod(cubeArray[C.idx], vec4(dir, C.slice), lod).rgb
                  * C.tint * (1 - w);   nrm += C.norm * (1 - w);   w = 1
        env   = col / w                                  // w starts at 0.01
        scale = min(luminance(ambientCube) / dot(vec4(R,1), nrm/w),
                    max(rough_r*g_vCubeNorm.x + g_vCubeNorm.y, 1.0))
        out   = fogAdd + base * (env * scale)
        out.a = 1.0                                      // <- OPAQUE

    The alpha-1.0 is the axis's name: a refracting pane is not composited
    at all, it shows the refracted environment instead.

    WHAT DIFFERS, precisely:
      * The cubemap-blend loop is a cluster loop over every cubemap whose
        cluster covers the pixel; this renderer's probe field is already
        reduced to ONE winning probe per voxel by PGRID (built exactly by
        the engine's own priority: a probe whose box contains the point
        beats a nearest-origin guess, smallest box first). So the loop is
        run at N=1 and the `(1-w)` weights collapse -- which is what the
        shipped loop does too, because it sets w=1 and breaks on the first
        hit. The `w` initialiser 0.01 is kept, so a pixel inside NO probe
        box gets the same x100 the engine gives it.
      * `C.norm` (the per-cubemap normalisation vec4) and g_vCubeNorm are
        not in the cube-probe pack, so `scale` is taken as the second arm
        of the min(), max(rough_r*0 + 1, 1) = 1.0, i.e. unnormalised. That
        is the arm the min picks whenever the first exceeds 1.
      * Face-cube slice selection is cube_sample()'s, so C.slice is the
        probe index rather than an array layer.
    """
    rr = (_e2(x, "refract_min_rough")
          + (1.0 - _e2(x, "refract_min_rough")) * rough).clamp(0, 1)
    eta = 1.0 / max(args.refract_ior, 1e-6)
    I = -view                                   # incident, toward surface
    ndi = (nrm * I).sum(-1, keepdim=True)
    k = 1.0 - eta * eta * (1.0 - ndi * ndi)
    R = torch.where(k >= 0.0,
                    eta * I - (eta * ndi + k.clamp(min=0).sqrt()) * nrm,
                    torch.zeros_like(I))
    R = _nrm(R)
    pi = probe_of(wpos)
    lo, hi = PBMIN[pi], PBMAX[pi]
    d = R
    inv = torch.where(d.abs() > 1e-6, 1.0 / d, torch.full_like(d, 1e6))
    t = torch.maximum((hi - wpos) * inv, (lo - wpos) * inv)
    thit = t.min(dim=-1, keepdim=True).values.abs()
    hitp = wpos + d * thit
    # mix(parallax-corrected hit point, raw direction, rough_r) -- the
    # rougher the pane, the less the box correction is trusted.
    dirv = _nrm(hitp - PORG[pi]) * (1 - rr.unsqueeze(-1)) + d * rr.unsqueeze(-1)
    lod = rr.sqrt() * args.ibl_lod_scale
    w = 0.01
    inbox = ((wpos >= lo) & (wpos <= hi)).all(-1)
    w = torch.where(inbox, torch.ones_like(rr), torch.full_like(rr, 0.01))
    env = cube_sample(pi, _nrm(dirv), lod) * torch.where(
        inbox, torch.ones_like(rr), torch.zeros_like(rr)).unsqueeze(-1)
    env = env / w.unsqueeze(-1)
    scale = torch.ones_like(rr)
    if amb is not None:
        lum = (amb * torch.tensor([0.2125000059604644775390625,
                                   0.7153999805450439453125,
                                   0.07209999859333038330078125],
                                  device=device)).sum(-1)
        # dot(vec4(R,1), norm) with the unavailable per-cubemap norm vec4
        # taken as (0,0,0,1) -> 1, so the first arm of the min is the
        # ambient luminance itself.
        scale = torch.minimum(lum, torch.ones_like(lum))
    o3 = on.unsqueeze(-1)
    out = base_rgb * (1 - o3) + (base_rgb * env * scale.unsqueeze(-1)) * o3
    return out, on


# --- D_MBOIT_PASS1 / D_MBOIT_PASS2 / D_MBOIT_4_MOMENTS -----------------
# Moment-based order-independent transparency. Two passes, both real:
#   PASS 1  csgo_complex s80/d34 (6 moments) / s80/d162 (4 moments),
#           csgo_glass s0/d16 and s0/d80. 237-302 GLSL lines, 0-1 fetch
#           sites, no lighting at all: it writes MOMENTS and shades nothing.
#   PASS 2  csgo_complex s80/d66 / s80/d194, csgo_glass s0/d32 / s0/d96.
#           The full shade, then a resolve against pass 1's targets.
#
# Our rasteriser returns one fragment per pixel, so PASS 1 could not see
# the second and third layer of glass -- which is the entire point of an
# order-independent method. It IS expressible: nvdiffrast ships
# dr.DepthPeeler, a multi-layer rasteriser, and both passes below iterate
# its layers. They are separate invocations over the same peel, not one
# fused pass: mboit_pass1 runs to completion and its three moment buffers
# are closed before mboit_transmittance reads any of them.

MBOIT_BIAS4 = (0.0, 0.375, 0.0, 0.375)
MBOIT_BIAS6 = (0.0, 0.4799999892711639404296875, 0.0,
               0.4510000050067901611328125, 0.0, 0.449999988079071044921875)


def mboit_warp_depth(zlin):
    """The warped depth both passes key on.

    complex s80/d34:290 and s80/d66:979, identical expression:
        z = ((log(dot(planeN, wpos - planeOrigin)) - logNear)
             / (logFar - logNear)) * 2 - 1
    dot(planeN, wpos - eye) is the view-space linear depth, so this is a
    log-warp of linear depth onto [-1, 1]. logNear/logFar are per-view
    constants; this renderer's own projection near/far are used, which is
    the same quantity computed from the same camera rather than a fit.
    """
    ln = math.log(max(MBOIT_NEAR, 1e-6))
    lf = math.log(max(MBOIT_FAR, MBOIT_NEAR * 1.0001))
    return ((zlin.clamp(min=1e-6).log() - ln) / (lf - ln)) * 2.0 - 1.0


def mboit_absorbance(alpha):
    """b = -log(1 - clamp(alpha, 1e-5, 0.9999)), complex s80/d34:291.

    csgo_glass feeds `1 - luminance(transmitted rgb)` into the same
    expression (glass s0/d16:*), and csgo_complex under S_ADDITIVE_BLEND
    folds it to the constant 1.0013630344474222511053085327148e-05 --
    which is exactly -log(1 - half(1e-5)), i.e. an additive surface writes
    the floor absorbance and occludes nothing. The caller supplies whichever
    alpha its class has; the additive case falls out of alpha == 0 here
    with max|delta| 1.4e-9 against that folded constant.
    """
    return -(1.0 - alpha.clamp(9.9999997473787516355514526367188e-06,
                               0.99989998340606689453125)).log()


def mboit_pass1(alpha, zlin, keep, n_moments):
    """PASS 1: accumulate moments. Shades nothing. Returns the moment set.

    complex s80/d34:293-295 (6 moments, THREE render targets):
        RT0 = vec4(b, 0, 0, 0)
        RT1 = vec4(vec2(z, z2) * b, 0, 0)
        RT2 = vec4(z2*z, z4, z4*z, z4*z2) * b
    complex s80/d162:1011-1012 (4 moments, TWO render targets):
        RT0 = vec4(b, 0, 0, 0)
        RT1 = vec4(z, z2, z2*z, z2*z2) * b
    The engine's blend state on those targets is additive, so a layer's
    contribution ADDS; that is the sum below over peeled layers.
    """
    b = mboit_absorbance(alpha) * keep.float()
    z = mboit_warp_depth(zlin)
    z2 = z * z
    if n_moments == 4:
        return b, torch.stack([z * b, z2 * b, z2 * z * b, z2 * z2 * b], -1)
    z4 = z2 * z2
    return (b,
            torch.stack([z * b, z2 * b], -1),
            torch.stack([z2 * z * b, z4 * b, z4 * z * b, z4 * z2 * b], -1))


def _mboit_t4(b0, m, zw):
    """4-power-moment transmittance. complex s80/d194:988-1017, verbatim.

    Cholesky on the 2x2 Hankel system, then the two-root quadrature that
    Muenstermann et al. call computeTransmittanceAtDepthFrom4PowerMoments.
    """
    ov = args.mboit_overestimation
    bias = torch.tensor(MBOIT_BIAS4, device=b0.device)
    mm = m / b0.unsqueeze(-1).clamp(min=1e-12)
    mm = mm + (bias - mm) * args.mboit_bias
    b1, b2, b3, b4 = mm[..., 0], mm[..., 1], mm[..., 2], mm[..., 3]
    nb1 = -b1
    d1 = nb1 * b2 + b3
    inv1 = 1.0 / _nz(nb1 * b1 + b2)
    c1 = d1 * inv1
    zb = zw - b1
    c2 = ((zw * zw) - (b2 + c1 * zb)) / _nz(-d1 * c1 + (-b2 * b2 + b4))
    e2 = c2
    e1 = zb * inv1 - c1 * c2
    root = ((e1 * e1) * 0.25 / _nz(c2 * c2)
            - (1.0 - (e1 * b1 + e2 * b2)) / _nz(c2))
    s = root.clamp(min=0).sqrt()
    h = (e1 / _nz(c2)) * (-0.5)
    z1, z2 = h - s, h + s
    o1 = (z1 < zw).float()
    a1 = (o1 - ov) / _nz(z1 - zw)
    a2 = (((z2 < zw).float() - o1) / _nz(z2 - z1) - a1) / _nz(z2 - zw)
    p1 = a1 - a2 * z1
    p0 = ov - p1 * zw
    poly = p0 + b1 * (p1 - a2 * zw) + b2 * a2
    return (-b0 * poly).exp().clamp(0, 1)


def _nz(x, eps=1e-12):
    """Divide-guard that preserves sign, so a near-singular Cholesky pivot
    saturates instead of producing a NaN that would poison the frame."""
    return torch.where(x.abs() < eps, torch.full_like(x, eps) * x.sign()
                       + (x == 0).float() * eps, x)


def _mboit_t6(b0, m12, m3456, zw):
    """6-power-moment transmittance. complex s80/d66:995-1071, verbatim.

    Three-root variant: the same Cholesky, then the trigonometric cubic
    solve (atan2/cos/sin at line 1052-1055) for the three knots.
    """
    ov = args.mboit_overestimation
    inv0 = 1.0 / b0.clamp(min=1e-12)
    odd = torch.stack([m12[..., 0], m3456[..., 0], m3456[..., 2]], -1) \
        * inv0.unsqueeze(-1)
    even = torch.stack([m12[..., 1], m3456[..., 1], m3456[..., 3]], -1) \
        * inv0.unsqueeze(-1)
    bias = torch.tensor(MBOIT_BIAS6, device=b0.device)
    bm = torch.stack([odd[..., 0], even[..., 0], odd[..., 1],
                      even[..., 1], odd[..., 2], even[..., 2]], -1)
    bm = bm + (bias - bm) * args.mboit_bias
    b = [bm[..., i] for i in range(6)]
    i0 = 1.0 / _nz(-b[0] * b[0] + b[1])
    d1 = -b[0] * b[1] + b[2]
    l21 = d1 * i0
    d2 = -d1 * l21 + (-b[1] * b[1] + b[3])
    i1 = 1.0 / _nz(d2)
    d3 = -b[0] * b[2] + b[3]
    l31 = d3 * i0
    d4 = -d1 * l31 + (-b[1] * b[2] + b[4])
    l32 = d4 * i1
    i2 = 1.0 / _nz(-b[2] * b[2] + b[5] - (d3 * l31 + d4 * l32))
    zz = zw * zw
    c0 = zw - b[0]
    c1 = zz - (b[1] + l21 * c0)
    c2 = (zz * zw) - (b[2] + l31 * c0 + l32 * c1)
    s2 = c2 * i2
    s1 = c1 * i1 - l32 * s2
    s0 = c0 * i0 - (l21 * s1 + l31 * s2)
    # solve the cubic s0 + s1 x + s2 x^2 + x^3, normalised by s2
    q = torch.stack([1.0 - (b[0] * s0 + b[1] * s1 + b[2] * s2),
                     s0, s1, s2], -1) / _nz(s2).unsqueeze(-1)
    p0, p1, p2 = q[..., 0], q[..., 1], q[..., 2]
    # s80/d66:1044-1055. q1 = (t0/t2)/3, q2 = (t1/t2)/3 -- the depressed
    # cubic's coefficients -- then the trigonometric three-root solve.
    q1, q2 = p1 / 3.0, p2 / 3.0
    pp = -q2 * q2 + q1
    qq = -q1 * q2 + p0
    ang = torch.atan2(((4.0 * pp) * (q2 * p0 - q1 * q1)
                       - qq * qq).clamp(min=0).sqrt(),
                      2.0 * q2 * pp - qq) / 3.0
    ca, sa = ang.cos(), ang.sin()
    r = 2.0 * (-pp).clamp(min=0).sqrt()
    z1 = r * ca - q2
    z2 = r * (-0.5 * ca - 0.866025388240814208984375 * sa) - q2
    z3 = r * (-0.5 * ca + 0.866025388240814208984375 * sa) - q2
    o1 = (z1 <= zw).float()
    o2 = (z2 <= zw).float()
    o3 = (z3 <= zw).float()
    a1 = (o1 - ov) / _nz(z1 - zw)
    a2 = (o2 - o1) / _nz(z2 - z1)
    a3 = (a2 - a1) / _nz(z2 - zw)
    a4 = (((o3 - o2) / _nz(z3 - z2) - a2) / _nz(z3 - z1) - a3) / _nz(z3 - zw)
    e2 = -a4 * z2 + a3
    e1 = a4 * (-z1) + e2
    e0 = e2 * (-z1) + a1
    # s80/d66:1069 `_25088 = 1.0 - _19289`, and _19289 is the UNSHIFTED
    # warp (line 979) while zw = _19289 - 1 (line 980). So v = -zw, not
    # 1 - zw: the 6-moment module keeps both forms of the warped depth
    # alive and this line is the only consumer of the unshifted one.
    v = -zw
    poly = ((e0 * v + ov) * 1.0 + (e1 * v + e0) * b[0]
            + (a4 * v + e1) * b[1] + a4 * b[2])
    return (-b0 * poly).exp().clamp(0, 1)


def mboit_transmittance(mom, zlin, n_moments):
    """PASS 2's resolve: the transmittance in front of this fragment.

    complex s80/d66:983-1072 and s80/d194:980-1018. The early-out
        if ((b0 - 0.001000500284135341644287109375) < 0.0) return 1.0
    is the shipped constant, not a tolerance chosen here.
    """
    b0 = mom[0]
    zw = mboit_warp_depth(zlin)
    if n_moments == 4:
        t = _mboit_t4(b0, mom[1], zw)
    else:
        t = _mboit_t6(b0, mom[1], mom[2], zw)
    t = torch.nan_to_num(t, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.where(b0 - 0.001000500284135341644287109375 < 0.0,
                       torch.ones_like(t), t)


# --- the smoke subsystem's tie into the EXISTING MBOIT ------------------
# D_MBOIT_PASS2 on `smoke_volume` resolves the smoke against the moment
# buffers the per-material D_MBOIT_PASS1 already wrote. That resolve is
# the power-moment reconstruction directly above, and there is exactly ONE
# implementation of it in this tree -- so smoke_mboit.py CALLS it rather
# than carrying a second copy. Without this line the module falls back to
# the zeroth-order form and SAYS so in its reach report; with it, the
# smoke and the materials resolve through the same function.
smoke_mboit.TRANSMITTANCE = mboit_transmittance


def parallax3(rgb, uv, mid, x, n_geo, tan4, view, uvscale):
    """csgo_simple_3layer_parallax (9 materials: the apartment windows).

    Three coplanar layers seen through one pane: g_tColor is the glass
    itself, g_tLayer1Color sits g_flLayer1Offset behind it and
    g_tLayer2Color g_flLayer2Offset behind that (-90 to -359 SOURCE
    units, i.e. 2.3-9.1 m of fake room). g_tLayer0Mask says where the
    glass is transparent enough to see through.

    The UV shift for a layer at depth d is (V.xy / V.z) * d expressed in
    UV units, so it needs metres-per-UV -- carried per face as `uvscale`
    from the triangle's own world/UV area ratio. Without it the offset
    would have to be an arbitrary constant, which is a fit, not a shader.
    """
    n = _nrm(n_geo)
    t_raw = tan4[..., :3]
    t = _nrm(t_raw - n * (n * t_raw).sum(-1, keepdim=True))
    b = torch.cross(n, t, dim=-1) * tan4[..., 3:4].sign()
    # view in tangent space; z is the component along the surface normal
    vx = (view * t).sum(-1)
    vy = (view * b).sum(-1)
    vz = (view * n).sum(-1).clamp(min=0.15)
    S_ = 0.0254                       # source units -> metres
    inv = 1.0 / uvscale.clamp(min=1e-4)
    on = _e2(x, "has_l1col")

    def _layer(depth_src, tex):
        d = depth_src.abs() * S_ * inv       # metres -> UV units
        return sample_fam(torch.stack([uv[..., 0] - vx / vz * d,
                                       uv[..., 1] - vy / vz * d],
                                      dim=-1), tex)

    l1 = _layer(_e2(x, "layer1_offset"), T_L1COL[mid])
    l2 = _layer(_e2(x, "layer2_offset"), T_L2COL[mid])
    m0 = sample_fam(uv, T_L0MASK[mid])
    # layer 2 furthest back, layer 1 over it, glass over both.
    interior = l2[..., :3] * (1 - l1[..., 3:]) + l1[..., :3] * l1[..., 3:]
    emis = (_e2(x, "layer1_emissive").unsqueeze(-1) * MAT_L1EM[mid]
            * l1[..., :3] * l1[..., 3:])
    # g_tLayer0Mask: where the pane is see-through, the interior shows.
    see = (1.0 - m0[..., 0:1]) * on.unsqueeze(-1)
    out = rgb * (1 - see) + (interior + emis) * see
    return out.clamp(0, 8)


# =======================================================================
# csgo_simple.vfx  and  csgo_simple_3layer_parallax.vfx
# =======================================================================
# Transcribed from the SPIR-V this session extracted from
# csgo/pak01_dir.vpk, decompiled with spirv-cross and committed as
#   docs/projects/counter-strike-sft/csgo_simple_ps.glsl
#       = csgo_simple record 0 (static combo 0), dynamic 128
#         (D_BAKED_LIGHTING_FROM_LIGHTMAP)
#   docs/projects/counter-strike-sft/csgo_simple_3layer_parallax_ps.glsl
#       = 3lp record 4 (static combo 4, S_TINT_MASK), dynamic 16
#         (D_BAKED_LIGHTING_FROM_LIGHTMAP)
# Every `:NNN` below is a line of the corresponding file.
#
# WHAT THESE TWO FAMILIES SHARE WITH THE ONES ALREADY PORTED, verbatim:
# the ambient-basis directional-occlusion resolve at per-view CB offsets
# 76/80, the min()-combined screen-space shadow at 88, the cascade
# shadow walk, the clustered analytic-light loop, the Lazarov split-sum
# env BRDF, the roughness-driven cube LOD and the two-stage gradient +
# aerial fog are all IDENTICAL instruction sequences to
# csgo_complex/csgo_environment. They are therefore taken through the
# existing gt_* helpers rather than re-implemented; the diff between
# csgo_simple's :266-304/:305-393/:698-726 and csgo_complex's own is
# constant-buffer member NUMBERING only.
#
# WHAT IS SPECIFIC TO THESE TWO, and is what these functions carry:
#   csgo_simple  -- albedo * vColor with NO tint mask, NO detail, NO
#     overlay, NO secondary UV and NO layer blending; roughness in the
#     normal map's BLUE channel floored by the normal derivative;
#     metalness either a uniform or g_tAmbientOcclusion.w; AO from
#     g_tAmbientOcclusion.x.
#   3lp -- three coplanar layers at three parallax depths behind one
#     pane, per-layer emissives, a view-angle exponent on layer 2, a
#     tint mask, and a metalness that is itself masked by the pane.
#
# NEITHER of these two per-view CBs declares a member at offset 116, so
# NEITHER family reads the weather ripple map -- checked against the
# std140 layout the decompiler emitted (csgo_simple_ps.glsl:76-99: the
# members run 16,32,48,72,76,80,88,104,192,... and 116 is absent), not
# assumed from PER_VIEW_RENDER_TARGETS.md's table, which enumerates six
# OTHER shaders and says nothing about these. 76, 80 and 88 ARE all three
# declared and all three consumed, so these families take the dirocc
# resolve and the screen-space shadow like csgo_complex and unlike
# csgo_static_overlay / csgo_glass.

SIMPLE_TAKEN = {}


def _simple_count(name, mask, gate=None):
    """Reach counter for the two simple families. --simple-reach prints it.

    `gate` is the FAMILY MASK and is not optional in practice. Both
    shading functions are branch-free and are evaluated over the whole
    tile, so a count taken without the gate counts every pixel in the
    batch and is guaranteed to be large no matter what -- a number that
    cannot be small is not evidence. The first run of this
    instrumentation reported `simple.shaded 12,441,600` (the entire
    tile) next to `dispatch.simple_pixels 44,508` (the truth), and
    `s3lp.secondary_uv 4,951,297` for an axis that is set on ZERO of the
    nine materials -- it was counting csgo_environment_blend's
    F_SECONDARY_UV through the shared ext2 block. Both are fixed by
    ANDing the gate in here.
    """
    if mask is None:
        n = 0
    else:
        if gate is not None:
            mask = mask & (gate > 0.5 if gate.dtype != torch.bool else gate)
        n = int(mask.sum())
    SIMPLE_TAKEN[name] = SIMPLE_TAKEN.get(name, 0) + n
    return n


def _axis3(flag, per_material):
    """A STRING-valued axis flag -> a per-pixel 0/1 tensor.

    `flag` is one of "auto" / "0" / "1". It is compared to those strings
    EXPLICITLY and never tested for truthiness: "0" is a non-empty string
    and therefore truthy, and `if args.simple_metalness_texture:` would
    have silently forced the axis ON at every setting including the one
    that names itself off. That bug shipped once behind the argparse P0
    and is the reason this helper exists instead of an inline test.
    """
    if flag == "0":
        return torch.zeros_like(per_material)
    if flag == "1":
        return torch.ones_like(per_material)
    if flag != "auto":
        raise SystemExit(f"axis flag {flag!r} is not one of auto/0/1")
    return per_material


def simple_surface(rgb, x, mid, uv, vc, nmap, n_geo, fmask):
    """csgo_simple.vfx -- the whole material stage of r0/d128.

    Returns (albedo, rough, metal, ao, alpha) as (B,H,W,3), (B,H,W),
    (B,H,W), (B,H,W), (B,H,W). Branch-free: every gate is a multiply, so
    the caller blends the result in over the family mask.

    :204  g_tColor                sampled at TEXCOORD.xy  (arrives as `rgb`)
    :205  g_tAmbientOcclusion     .x = AO, .w = metalness when
                                  S_METALNESS_TEXTURE = 1
    :206-211 g_tNormal            Source 2's own decode, NOT 2*rgb-1:
              nx = (r + g) - 256/255 ;  ny = r - g ;  nz = 1 - |nx| - |ny|
    :253  albedo   = g_tColor.rgb * vColor.rgb
    :254  F0       = mix(vec3(g_flReflectance), albedo, metalness)
    :261  rough    = max(g_tNormal.z,
                         clamp(max(|dFdx n|^2, |dFdy n|^2),0,1) ^ (1/3))
    :711  diffuse albedo = albedo * (1 - metalness)
    :710  the indirect multiplier is dirOcc * g_tAmbientOcclusion.x
    output alpha = vColor.a (:713)

    S_AMBIENT_OCCLUSION_TEXTURE is a NO-OP IN THE PIXEL SHADER. Static
    combos 0 and 2 share bytecode record 0 and combos 1 and 3 share
    record 1 (m_nByteCodeDataIdx, 2 of 12 combos agree with their array
    position -- the walk is wrong here, which is trap 2 of
    iji_model/counter_strike_render/vcs/README.md firing on live data). The AO texture
    is sampled unconditionally either way; the axis only changes which
    image the descriptor points at. That is reported rather than
    implemented as a branch, because implementing a branch would be
    inventing one.

    WHAT DIFFERS FROM THE REFERENCE. The derivative floor on :261 uses
    dFdx/dFdy of the GEOMETRIC normal. nvdiffrast rasterises to a tensor
    with no quad derivatives, so the floor is computed from the same
    normal by finite differences over the pixel grid, which is the same
    quantity to the same order but is NOT the hardware's 2x2 quad
    difference; at a silhouette the two disagree by up to the full
    normal delta across one pixel.
    """
    # --- the surface normal's own derivative floor (:257-261) ---------
    n = _nrm(n_geo)
    dx = torch.zeros_like(n)
    dy = torch.zeros_like(n)
    dx[:, :, 1:, :] = n[:, :, 1:, :] - n[:, :, :-1, :]
    dy[:, 1:, :, :] = n[:, 1:, :, :] - n[:, :-1, :, :]
    dfloor = ((dx * dx).sum(-1).maximum((dy * dy).sum(-1))
              ).clamp(0, 1) ** 0.333000004291534423828125
    rough = torch.maximum(nmap[..., 2], dfloor).clamp(0.03, 1.0)   # :261

    # --- g_tAmbientOcclusion: .x is AO, .w is metalness ---------------
    aotex = sample_fam(uv, T_SIMPAO[mid])
    have_ao = _e2(x, "has_simple_ao")
    ao = aotex[..., 0] * have_ao + (1.0 - have_ao)                 # :710
    if args.simple_ao_texture == "0":
        # not a shader state -- a measurement switch, so the term's
        # contribution can be read off. Said out loud in the help.
        ao = torch.ones_like(ao)

    # --- S_METALNESS_TEXTURE (place value 1) --------------------------
    mtex = _axis3(args.simple_metalness_texture, _e2(x, "f_simple_metal_tex"))
    # the .w read is gated by the SLOT, so the WHITE fallback row (alpha
    # 255) can never arrive as "fully metal" on a material with no map.
    mtex = mtex * have_ao
    metal = (_e2(x, "simple_metalness") * (1 - mtex)
             + aotex[..., 3] * mtex).clamp(0, 1)                   # r1 diff

    albedo = rgb * vc[..., :3]                                     # :253
    alpha = vc[..., 3]                                             # :713
    _simple_count("simple.shaded", fmask, fmask)
    _simple_count("simple.metalness_texture", mtex > 0.5, fmask)
    _simple_count("simple.ao_texture_bound", have_ao > 0.5, fmask)
    # `have_ao` is returned so the CALLER can tell "this family's AO slot
    # said 1.0" from "this family has no AO slot and 1.0 is the identity
    # standing in for it". Those are the same float and different facts,
    # and folding the second over an AO the caller already had is how a
    # real page gets replaced by a stand-in.
    return albedo, rough, metal, ao, alpha, have_ao
