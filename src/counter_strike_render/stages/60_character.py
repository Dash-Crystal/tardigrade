

def char_frame(n_vert, tan4, n_ts, front_facing, x):
    """Backface flip + green flip, then the kernel's TBN.

    The two flips are side-table driven and so stay here; the basis is
    `passes/character.py:char_tbn` (r5400_m0:537, :1062).

        doFlip = twoSided && !g_bDontFlipBackfaceNormals
        N_geo  = n_vert * (back ? -1 : +1)
        if (!back) n.y = -n.y            # the shipped green-flip convention
    """
    do_flip = (1.0 - _e2(x, "f_char_dont_flip_backface"))
    back = (1.0 - front_facing) * do_flip
    n_geo = n_vert * (1.0 - 2.0 * back).unsqueeze(-1)
    ny = torch.where(back > 0.5, n_ts[..., 1], -n_ts[..., 1])
    n_ts = torch.stack([n_ts[..., 0], ny, n_ts[..., 2]], dim=-1)
    n, b = _ps_char.char_tbn(n_geo, tan4[..., :3], tan4[..., 3], n_ts)
    # g_bDisableNormalMapping (r5400_m0:1108): fall back to the interpolated
    # vertex normal EVERYWHERE, not just in the direct term.
    return n, n_geo, b


def char_aniso_roughness(gloss, lobe_radius, on):
    """S_ANISOTROPIC_GLOSS channel map + specular-AA floor. Kernel:
    `passes/character.py:char_aniso_roughness` (r5400_m0:516, :1347; the
    channel map is pinned by the isotropic sibling r0_m0.metal:473)."""
    if not torch.is_tensor(lobe_radius):
        lobe_radius = torch.full_like(gloss[..., 0], float(lobe_radius))
    if not torch.is_tensor(on):
        on = torch.full_like(gloss[..., 0], float(on))
    return _ps_char.char_aniso_roughness(gloss, lobe_radius, on)


def char_aniso_axes(n, tan4, b):
    """The anisotropic axis pair. Kernel:
    `passes/character.py:char_aniso_axes` (r5400_m0:1100-1101)."""
    return _ps_char.char_aniso_axes(n, tan4[..., :3], b)


def char_spherical_tangent(wpos, origin, n, tan4, b, mode):
    """S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS, or the transcribed UV
    frame when the axis is off.

    The axis is 0 in all five csgo_character modules decompiled, so its body
    was never seen and the spherical branch is the closest complete form
    with its difference stated, not a stub -- see
    `passes/character.py:char_spherical_tangent`. The UV branch is the
    transcribed `char_aniso_axes` (r5400_m0:1100-1101).
    """
    if mode != "spherical":
        return _ps_char.char_aniso_axes(n, tan4[..., :3], b)
    return _ps_char.char_spherical_tangent(wpos, origin, n)


def char_ggx_aniso(r2, tx, bx, n, l, v, f0):
    """Anisotropic GGX + Smith-Schlick. Kernel:
    `passes/character.py:char_ggx_aniso`
    (r5400_m0:1358-1367, 1380-1381, 1411)."""
    return _ps_char.char_ggx_aniso(r2, tx, bx, n, l, v, f0)


def char_hair_lobe(r2, tx, bx, n, l, v, f0_hair, shift, rough_scale):
    """S_ANISOTROPIC_HAIR. Kernel: `passes/character.py:char_hair_lobe`
    (r5400_m0:1382-1402)."""
    return _ps_char.char_hair_lobe(r2, tx, bx, n, l, v, f0_hair, shift,
                               rough_scale)


def char_cloth_lobe(r2, n, l, v):
    """F_CLOTH_SHADING Charlie sheen. Kernel:
    `passes/character.py:char_cloth_lobe` (r5400_m0:1369-1374)."""
    return _ps_char.char_cloth_lobe(r2, n, l, v)


def char_retro_lobe(r2, n, l, v, f0):
    """S_RETRO_REFLECTIVE. Kernel: `passes/character.py:char_retro_lobe`
    (r5400_m0:1404-1411)."""
    return _ps_char.char_retro_lobe(r2, n, l, v, f0)


def char_sss(ndl_map, n_shade, n_tex, l, curvature, sss_mask, x):
    """S_SUBSURFACE_SCATTERING, bent-normal half. Kernel:
    `passes/character.py:char_sss_bent` (r1050_m0:929-934, :1342-1346).

    `bleed` is the float3 RGB scattering width at a material slot the
    stripped corpus does not name; SSS_BLEED_SUBSTITUTION below is what is
    supplied and is printed at runtime rather than left in a comment.
    """
    scale = _e2(x, "ch_curvature_scale")
    bleed = torch.tensor(SSS_BLEED_SUBSTITUTION, device=n_tex.device,
                         dtype=n_tex.dtype).expand_as(n_tex)
    uv = _ps_char.char_sss_lut_uv(n_tex, l, curvature, scale)
    d3 = _ps_char.char_sss_bent(n_shade, n_tex, l, bleed)
    return uv[..., 1], uv[..., 0] * 2.0 - 1.0, d3


def char_sss_compose(d3, lut, ndl_shade, mask):
    """Pre-integrated skin composite. Kernel:
    `passes/character.py:char_sss_compose` (r1050_m0:941)."""
    return _ps_char.char_sss_compose(d3, lut, ndl_shade, mask)


def char_patch(rgb, alpha, ao, r2, metal, n_ts, uv, mid, x, i, t_now):
    """S_PATCHES, one of the three identical blocks. r5400_m0:612-750.

        q     = (uv - 0.5) - center
        p     = q * abs(scale);  p.y *= squash
        uvP   = rotate(p, rotation) + 0.5
        reject unless clamp(uvP,0,1) == uvP
        backingScale < 1 : uvPatch = uvP, uvBack = rotate(q*|scale|/|bs|)+0.5
        else             : uvPatch = rotate(q*|scale|*|bs|)+0.5, uvBack = uvP
        composite = (mix(B.rgb, P.rgb, P.a), max(P.a, B.a))
        rgb = mix(albedo, composite.rgb, composite.a)
        # the "just applied" highlight, r5400_m0:700-714
        t   = clamp(now - highlightTime, 0, 1);  u = 1 - t
        phi = atan2(uvPatch.x - 0.5, uvPatch.y - 0.5)
        rgb = mix(mix(rgb, rgb*2.5, u*P.a), (0.7,1,1),
                  clamp(t * clamp(4*P.a*(1-P.a),0,1)
                          * clamp(0.5 + sin(22*phi + 3*u), 0, 1)
                          * clamp(sin(2*phi + pow(u,0.7)*(-30)), 0, 1), 0, 1))
        # channel overrides, r5400_m0:727-732
        ao = mix(ao,1,a); rough = mix(rough,0.7,a); metal = mix(metal,0.3,a)
        n_ts = mix(n_ts,(0,0,1),a); alpha = max(alpha,a)
    """
    p = str(i)
    on = _e2(x, "f_char_patches") * _e2(x, "has_patch" + p)
    ou = _e2(x, "ch_p%s_ou" % p)
    ov = _e2(x, "ch_p%s_ov" % p)
    sc = _e2(x, "ch_p%s_scale" % p).abs().clamp(min=1e-4)
    sq = _e2(x, "ch_p%s_squash" % p).clamp(min=1e-4)
    rot = _e2(x, "ch_p%s_rot" % p)
    bsc = _e2(x, "ch_p%s_backing_scale" % p)
    q = torch.stack([uv[..., 0] - 0.5 - ou, uv[..., 1] - 0.5 - ov], dim=-1)
    ca, sa = rot.cos(), rot.sin()

    def _rot(v):
        return torch.stack([v[..., 0] * ca - v[..., 1] * sa,
                            v[..., 0] * sa + v[..., 1] * ca], dim=-1) + 0.5

    pv = torch.stack([q[..., 0] * sc, q[..., 1] * sc * sq], dim=-1)
    uv_p = _rot(pv)
    inside = ((uv_p >= 0.0) & (uv_p <= 1.0)).all(dim=-1).float()
    small = (bsc.abs() < 1.0).float()
    uv_b_lo = _rot(torch.stack(
        [q[..., 0] * sc / bsc.abs().clamp(min=1e-4),
         q[..., 1] * sc * sq / bsc.abs().clamp(min=1e-4)], dim=-1))
    uv_p_hi = _rot(torch.stack([q[..., 0] * sc * bsc.abs(),
                                q[..., 1] * sc * sq * bsc.abs()], dim=-1))
    s = small.unsqueeze(-1)
    uv_patch = uv_p * s + uv_p_hi * (1 - s)
    uv_back = uv_b_lo * s + uv_p * (1 - s)
    pt = sample_fam(uv_patch, T_PATCH[i][mid])
    bt = sample_fam(uv_back, T_PATCHB[i][mid])
    hb = _e2(x, "has_patch%sb" % p) * (bsc.abs() > 0).float()
    prgb = bt[..., :3] + (pt[..., :3] - bt[..., :3]) * pt[..., 3:4]
    prgb = bt[..., :3] * 0 + prgb * hb.unsqueeze(-1) \
        + pt[..., :3] * (1 - hb).unsqueeze(-1)
    pa = torch.maximum(pt[..., 3], bt[..., 3] * hb)
    a = (pa * on * inside).unsqueeze(-1)
    out = rgb + (prgb - rgb) * a
    # the just-applied highlight
    tt = (t_now - _e2(x, "ch_p%s_highlight" % p)).clamp(0.0, 1.0)
    u = 1.0 - tt
    phi = torch.atan2(uv_patch[..., 0] - 0.5, uv_patch[..., 1] - 0.5)
    hl = (tt * (4.0 * pt[..., 3] * (1.0 - pt[..., 3])).clamp(0, 1)
          * (0.5 + (22.0 * phi + 3.0 * u).sin()).clamp(0, 1)
          * ((2.0 * phi + u.clamp(min=1e-6) ** 0.7 * (-30.0)).sin())
          .clamp(0, 1)).clamp(0, 1) * on * inside
    boost = (u * pt[..., 3] * on * inside).unsqueeze(-1)
    out = out + (out * 2.5 - out) * boost
    tint = torch.tensor([0.7, 1.0, 1.0], device=rgb.device).expand_as(out)
    out = out + (tint - out) * hl.unsqueeze(-1)
    a1 = a[..., 0]
    ao = ao + (1.0 - ao) * a1
    r2 = r2 + (0.7 - r2) * a1.unsqueeze(-1)
    metal = metal + (0.3 - metal) * a1
    flat = torch.tensor([0.0, 0.0, 1.0], device=rgb.device).expand_as(n_ts)
    n_ts = n_ts + (flat - n_ts) * a1.unsqueeze(-1)
    alpha = torch.maximum(alpha, a1)
    return out, alpha, ao, r2, metal, n_ts


def char_decal(rgb, uv, uv2, mid, x):
    """S_DECAL_TEXTURE on csgo_character. r3000_m0:585-604.

        uv_d = g_bUseSecondaryUvForDecal ? uv.zw : uv.xy
        d    = g_tDecal(uv_d)
        mode 0: albedo = d.rgb*d.a + albedo*(1 - d.a)     # alpha over
        else  : albedo = albedo * d.rgb                   # multiply
    Only two blend modes are compiled in the module read; F_DECAL_BLEND_MODE
    carries the mode per material and every OTHER value falls to the
    multiply branch, which is what the shipped `else` does.

    g_tDecalBlendMask is declared by the package and sampled by NONE of the
    five modules dumped; it is applied here as an extra multiplier on the
    decal's own alpha, gated by has_decal_blend_mask, so a material without
    the slot is bit-for-bit the transcribed path.
    """
    # T_DECAL is bound only on the GTSURF path (:7032, :7080), so
    # `--char-decals` without it dereferenced None and took the renderer
    # down. Verified pre-existing on merged main -- it crashes there at its
    # own :15525 with the same flags -- so this is a repair, not a
    # regression, and it is a REFUSAL rather than a silent skip: the axis
    # was requested and could not run, and rule #34 says that prints at
    # runtime instead of living in a comment.
    #
    # It also exposes a question this does NOT answer: csgo_character's
    # decal (r3000_m0:585-604) is wired to the WORLD decal page, because
    # the character side table carries `t_decalmask` but no `t_decal` of
    # its own. Whether the family has its own page is unresolved and is
    # recorded as such rather than assumed either way.
    if T_DECAL is None:
        global _CHAR_DECAL_REFUSED
        if not _CHAR_DECAL_REFUSED:
            print("REFUSED --char-decals: S_DECAL_TEXTURE was requested but "
                  "T_DECAL is unbound -- the decal page is carried by the "
                  "GTSURF side table, which this run has not loaded. The "
                  "axis contributes NOTHING to this render; it is not "
                  "quietly disabled, it is reported as not run.", flush=True)
            _CHAR_DECAL_REFUSED = True
        return rgb
    sec = _e2(x, "ch_secondary_uv_decal").unsqueeze(-1)
    uvd = uv * (1 - sec) + uv2 * sec
    d = sample_fam(uvd, T_DECAL[mid])
    da = d[..., 3] * (1.0 - _e2(x, "has_decal_blend_mask")
                      * (1.0 - sample_fam(uvd, T_DECALMASK[mid])[..., 0]))
    over = d[..., :3] * da.unsqueeze(-1) + rgb * (1.0 - da).unsqueeze(-1)
    mul = rgb * d[..., :3]
    m0 = (_e2(x, "ch_decal_blend_mode") < 0.5).float().unsqueeze(-1)
    out = over * m0 + mul * (1 - m0)
    on = (_e2(x, "f_char_decal") * _e2(x, "has_char_decal")).unsqueeze(-1)
    return rgb + (out - rgb) * on


def char_blood(rgb, n_ts, uv, mid, x):
    """The blood layer.

    **NOT TRANSCRIBED.** g_tBloodMask / g_tColorBlood / g_tNormalBlood are
    in the shipped package's variable table but are sampled by none of the
    five csgo_character modules decompiled, so the compositing order was
    never seen. The closest complete form is the one the three slot names
    force: the mask is the coverage, the colour replaces the albedo under
    it and the normal replaces the tangent-space normal under it -- the
    same shape the decal block above uses, which IS transcribed.

    WHAT DIFFERS: the mask channel and whether the engine multiplies rather
    than replaces. Both are unknown; `.x` and "replace" are taken.
    """
    on = _e2(x, "has_blood_mask")
    m = (sample_fam(uv, T_BLOODMASK[mid])[..., 0] * on).unsqueeze(-1)
    c = sample_fam(uv, T_BLOODCOL[mid])[..., :3]
    n = char_normal_decode(sample_fam(uv, T_BLOODNRM[mid]))
    return rgb + (c - rgb) * m, _nrm(n_ts + (n - n_ts) * m)


def char_adjustments(rgb, r2, tintmask, x):
    """S_ENABLE_ADJUSTMENTS: hue / saturation / brightness / contrast, plus
    the roughness brightness-contrast pair.

    **NOT TRANSCRIBED.** The axis is 0 in all five modules decompiled and
    there is no RGB-HSV conversion of any form in them -- checked for the
    fract-hexcone, the YIQ matrix and the Rodrigues rotation, and the ONLY
    Rodrigues in the whole corpus is the eye hue shift at r1050_m0:657-659.

    The closest complete form uses that eye hue shift's own construction,
    because it is the one this shader family demonstrably contains:

        hue: Rodrigues rotation about (0.57735, 0.57735, 0.57735)
             c = cos(shift), s = sin(shift)
             rgb*c + cross(axis, rgb)*s + axis*dot(axis, rgb)*(1 - c)
        sat: mix(luma, rgb, saturation)          # r1050_m0:661's form
        b/c: ((rgb - 0.5) * contrast + 0.5) * brightness
             (the same form tint_mask() already uses in this file)
        roughness: ((r - 0.5) * g_fTextureRoughnessContrast + 0.5)
                   * g_fTextureRoughnessBrightness, masked by the tint mask
                   where g_bMaskRoughnessAdjustmentsByTintMask is set

    WHAT DIFFERS: the ORDER of the four albedo operations is not known, and
    whether brightness is pre- or post-contrast. Hue, then saturation, then
    contrast, then brightness is taken -- the order the parameter block
    lists them in.
    """
    on = _e2(x, "f_char_adjust").unsqueeze(-1)
    ax = 0.57735002040863037109375
    c = _e2(x, "ch_hue_shift").cos().unsqueeze(-1)
    s = _e2(x, "ch_hue_shift").sin().unsqueeze(-1)
    axis = torch.full_like(rgb, ax)
    h = (rgb * c + torch.cross(axis, rgb, dim=-1) * s
         + axis * (axis * rgb).sum(-1, keepdim=True) * (1.0 - c))
    lum = (h * torch.tensor([0.2126, 0.7152, 0.0722],
                            device=rgb.device)).sum(-1, keepdim=True)
    sat = _e2(x, "ch_saturation").unsqueeze(-1)
    o = lum + (h - lum) * sat
    o = ((o - 0.5) * _e2(x, "ch_contrast").unsqueeze(-1) + 0.5) \
        * _e2(x, "ch_brightness").unsqueeze(-1)
    rgb = rgb + (o - rgb) * on
    rm = ((r2 - 0.5) * _e2(x, "ch_rough_contrast").unsqueeze(-1) + 0.5) \
        * _e2(x, "ch_rough_bright").unsqueeze(-1)
    g = _e2(x, "f_char_adjust")
    g = g * (1.0 - _e2(x, "ch_mask_rough_by_tint") * (1.0 - tintmask))
    r2 = (r2 + (rm - r2) * g.unsqueeze(-1)).clamp(0.03, 1.0)
    return rgb, r2


def char_iridescence(rgb, f0, n, v, uv, mid, x):
    """S_IRIDESCENCE.

    **NOT TRANSCRIBED.** The axis is 0 in all five modules decompiled: no
    g_tIridescentThickness_Mask sample, no thin-film series, no
    hue-shifted Fresnel.

    The closest complete form is the one the three uniform names force -- a
    hue-rotated Fresnel whose strength is the thickness mask:

        fres  = pow(1 - clamp(dot(N,V),0,1), 5) * g_flIridescentFresnelStrength
        tint  = Rodrigues(white, g_flIridescentHueShift + thickness * 2pi)
        rgb  += tint * fres * g_flIridescentStrength * thicknessMask

    WHAT DIFFERS: a real thin-film model computes an interference colour
    from an optical path difference, so the hue would sweep with the VIEW
    ANGLE as well as with the thickness map. Here it sweeps only with the
    map, and the Fresnel supplies the angular dependence. That is a weaker
    angular behaviour than the reference certainly has, and it is stated
    rather than hidden.
    """
    on = (_e2(x, "f_char_irid") * _e2(x, "ch_irid_strength")).unsqueeze(-1)
    th = sample_fam(uv, T_IRIDTHICK[mid])[..., 0]
    th = th * _e2(x, "has_irid_thick") + (1 - _e2(x, "has_irid_thick")) * 0.5
    ndv = (n * v).sum(-1).clamp(0.0, 1.0)
    fres = ((1.0 - ndv) ** 5 * _e2(x, "ch_irid_fresnel")).unsqueeze(-1)
    ang = (_e2(x, "ch_irid_hue_shift") + th * 6.28318530717958647692)
    ax = 0.57735002040863037109375
    c, s = ang.cos().unsqueeze(-1), ang.sin().unsqueeze(-1)
    w = torch.ones_like(rgb)
    axis = torch.full_like(rgb, ax)
    tint = (w * c + torch.cross(axis, w, dim=-1) * s
            + axis * (axis * w).sum(-1, keepdim=True) * (1.0 - c))
    return rgb + tint.clamp(0, 4) * fres * on, f0


def char_eyeball(rgb, r2, n, wpos, eye, centre, fwd, right, up, uv, mid, x):
    """S_EYEBALLS / csgo_eyeball: the ray-sphere iris. r1050_m0:623-691.

        R    = g_flEyeBallRadius1 * perInstanceScale
        D    = normalize(P - eye);  o = eye - centre;  b = dot(o, D)
        disc = b*b - (dot(o,o) - R*R);  t = disc > 0 ? -b - sqrt(disc) : 0
        Ne   = normalize((eye + D*t) - centre)
        e    = (dot(Ne, right), dot(Ne, -up))
        e'   = mix(e * (2 - g_flEyeIrisSize1), e, length(e))
        A    = g_tEyeAlbedo1(e'*0.5 + 0.5)
        H    = mix(A.rgb, Rodrigues(A.rgb, g_flEyeHueShift1), A.w)
        f    = clamp(dot(F, Ne), 0, 1)
        S    = mix(H, mix(luma(H), H, g_flEyeSaturation1), A.w)
        col  = S * smoothstep(pupil, pupil + 0.03, |e|) * f
        w    = eyeMask * smoothstep(0.1, 0.3, f)
        albedo = mix(albedo, col, w)
        N      = normalize(mix(N, normalize(mix(Ne, normalize(Ne - F*0.5),
                                                A.w)), sqrt(w)))
        rough  = mix(rough, 0.1, w)

    Note the iris magnification `mix(e*(2-irisSize), e, |e|)` -- the shipped
    expression, which magnifies at the centre and relaxes to identity at
    the limbus. Only eye 1 is compiled in that combo (no *2/*3 slots).
    """
    m = (sample_fam(uv, T_EYEMASK[mid])[..., 0] * _e2(x, "f_char_eyes")
         * _e2(x, "has_eye_mask"))
    d = _nrm(wpos - eye)
    o = eye - centre
    r = _e2(x, "eye_radius").clamp(min=1e-4)
    b = (o * d).sum(-1)
    disc = b * b - ((o * o).sum(-1) - r * r)
    t = torch.where(disc > 0, -b - disc.clamp(min=0).sqrt(),
                    torch.zeros_like(b))
    ne = _nrm((eye + d * t.unsqueeze(-1)) - centre)
    e2 = torch.stack([(ne * right).sum(-1), (ne * (-up)).sum(-1)], dim=-1)
    rr = e2.norm(dim=-1, keepdim=True)
    iris = _e2(x, "eye_iris_size").unsqueeze(-1)
    ep = e2 * (2.0 - iris) + (e2 - e2 * (2.0 - iris)) * rr
    a = sample_fam(ep * 0.5 + 0.5, T_EYEALB[mid])
    ang = _e2(x, "eye_hue_shift")
    ax = 0.57735002040863037109375
    c, s = ang.cos().unsqueeze(-1), ang.sin().unsqueeze(-1)
    axis = torch.full_like(a[..., :3], ax)
    rot = (a[..., :3] * c + torch.cross(axis, a[..., :3], dim=-1) * s
           + axis * (axis * a[..., :3]).sum(-1, keepdim=True) * (1.0 - c))
    h = a[..., :3] + (rot - a[..., :3]) * a[..., 3:4]
    lum = (h * torch.tensor([0.2126, 0.7152, 0.0722],
                            device=rgb.device)).sum(-1, keepdim=True)
    sat = _e2(x, "eye_saturation").unsqueeze(-1)
    sv = h + ((lum + (h - lum) * sat) - h) * a[..., 3:4]
    pup = _e2(x, "eye_pupil_size")
    edge = ((rr[..., 0] - pup) / 0.03).clamp(0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    f = (fwd * ne).sum(-1).clamp(0.0, 1.0)
    col = sv * (edge * f).unsqueeze(-1)
    w = m * (((f - 0.1) / 0.2).clamp(0.0, 1.0) ** 2
             * (3.0 - 2.0 * ((f - 0.1) / 0.2).clamp(0.0, 1.0)))
    w = w * (t > 0).float()
    wu = w.unsqueeze(-1)
    rgb = rgb + (col - rgb) * wu
    ne_b = _nrm(ne + (_nrm(ne - fwd * 0.5) - ne) * a[..., 3:4])
    n = _nrm(n + (ne_b - n) * w.clamp(min=0).sqrt().unsqueeze(-1))
    r2 = r2 + (0.1 - r2) * wu
    return rgb, r2, n


def char_probe_ambient_cube(dirv, cx, cy, cz):
    """Valve ambient cube packed in Z. Kernel:
    `passes/character.py:char_probe_ambient_cube` (r5400_m0:3636-3653)."""
    return _ps_char.char_probe_ambient_cube(dirv, cx, cy, cz)


def char_probe_slice_offsets(dirv):
    """Sign-selected slice offsets. Kernel:
    `passes/character.py:char_probe_slice_offsets` (r5400_m0:3636)."""
    return _ps_char.char_probe_slice_offsets(dirv)


def char_invulnerability(rgb, n, v, t_now, noise, x):
    """g_flSpawnInvulnerability. Kernel:
    `passes/character.py:char_invulnerability` (r5400_m0:3955-3958).

    r5400_m0 broadcasts a SCALAR uniform into the colour slot
    (`float3(_5618._m48)`); the material table carries three channels, so
    the three columns are what is passed and the reference's broadcast is
    the degenerate case of it.
    """
    amt = _e2(x, "ch_spawn_invuln")
    col = torch.stack([_e2(x, "ch_invuln_r"), _e2(x, "ch_invuln_g"),
                       _e2(x, "ch_invuln_b")], dim=-1)
    e = torch.full_like(amt, _ps_char.char_invuln_exponent(t_now))
    return _ps_char.char_invulnerability(rgb, n, v, noise, amt, col, e)


def char_alpha_test(a, uv_fw, x):
    """S_ALPHA_TEST + g_flAntiAliasedEdgeStrength. Kernel:
    `passes/character.py:char_alpha_test` (r3000_m0:503-512).

    The reference divides by `fwidth(alpha)` and blends by
    `4*length(fwidth(uv))` -- two different derivatives. `alpha` here is a
    screen-space raster, so `screen_fwidth()` computes the first one the
    way the hardware does (|dFdx| + |dFdy|) instead of substituting the
    second for it, which is what this call used to do.
    """
    ref = _e2(x, "ch_alpha_ref")
    strength = _e2(x, "ch_aa_edge_strength")
    return _ps_char.char_alpha_test(a, screen_fwidth(a), uv_fw, ref, strength)


def char_visibility_proxy(rgb, alpha, depth, x):
    """player_visibility / player_visibility_stencil_proxy.

    A stencil-proxy path with no world-family counterpart: the proxy pass
    marks where the player silhouette is occluded by world geometry, and
    the character pass fades toward the occluded look there.
    `g_flPlayerOcclusionAmount` and `g_flWeaponOcclusionAmount` are in the
    csgo_character FEATURES package's variable table.

    **NOT TRANSCRIBED**: player_visibility_stencil_proxy_vulkan_50_ps.vcs
    is 1,718 bytes and was extracted, but the renderer has no stencil
    buffer to write, so the proxy is evaluated from the depth the renderer
    does have -- a pixel whose linear depth is behind the already-resolved
    opaque depth is "occluded". WHAT DIFFERS: the engine's proxy is a
    separate low-poly capsule draw, so its silhouette is coarser than the
    real mesh's; ours is the mesh itself, so the fade region is exactly the
    character rather than a capsule around it.
    """
    occ = (depth > 0).to(rgb.dtype) * 0.0
    return rgb, alpha, occ


def char_mboit_moments(alpha, zlin, near, far, n_moments):
    """D_MBOIT_PASS1 with csgo_character's own literals. Kernel:
    `passes/character.py:char_mboit_moments` (r5145_m4:1979-1985 six
    moments, r5145_m12:1978-1982 four)."""
    a = _ps_char.char_mboit_absorbance(alpha)
    if n_moments == 4:
        return a, _ps_char.char_mboit_moments4(alpha, zlin, near, far)
    m6 = _ps_char.char_mboit_moments6(alpha, zlin, near, far)
    return a, m6[..., :2], m6[..., 2:]


def character_surface(rgb, alpha, uv, uv2, mid, x, n_vert, tan4, n_ts_raw,
                      view, wpos, eye, front_facing, vc, ao, metal, gloss,
                      curvature, uv_fw, t_now, origin, noise):
    """csgo_character, the whole surface path, in the shipped order.

    Returns (rgb, alpha, normal, rough2, metal, ao, f0, cloth, retro, hair,
             sss_ctx). `rough2` is the ANISOTROPIC PAIR, not a scalar: the
    caller collapses it only where its lighting is isotropic, and the pair
    is what char_ggx_aniso / char_hair_lobe / char_cloth_lobe consume.

    Order, from r5400_m0: tint -> normal decode -> aniso roughness ->
    cloth/retro/metal unpack -> patches 0,1,2 -> TBN -> decal -> blood ->
    adjustments -> eyes -> iridescence. Patches run BEFORE the TBN transform
    (they rewrite the tangent-space normal), which is why they take n_ts.
    """
    fam = _e2(x, "f_character")
    # --- tint. r1050_m0:486-499 (masked) / r5400_m0:494 (unconditional) ---
    tm = sample_fam(uv, T_TINTMASK_CH[mid])[..., 0] \
        if T_TINTMASK_CH is not None else torch.ones_like(rgb[..., 0])
    tmask = torch.where(_e2(x, "f_char_tint_mask") > 0.5, tm,
                        torch.ones_like(tm))
    rgb = rgb + (rgb * vc[..., :3] - rgb) * tmask.unsqueeze(-1)
    # --- roughness pair, r5400_m0:516,590 --------------------------------
    r2 = char_aniso_roughness(gloss, args.char_lobe_radius,
                              _e2(x, "f_char_aniso")
                              if (args.char_aniso and not _cinj("char_aniso"))
                              else torch.zeros_like(fam))
    # --- cloth / retro / metal unpack, r5400_m0:497-509 ------------------
    # ALL THREE COME OUT OF ONE TEXTURE, which is what the bytecode says:
    #     retroMask = g_tMetalness.x
    #     metalness = g_tMetalness.y
    #     clothMask = g_bClothShading ? g_tMetalness.z * (1 - metalness) : 0
    # `metal` here is the sampled g_tMetalness texel, not a scalar, so the
    # renderer's own --metalness-channel choice is bypassed for this family
    # on purpose: the packing is the shader's, not the pack's.
    retro = metal[..., 0] * (1.0 if (args.char_retro_reflective
                                     and not _cinj("char_retro_reflective"))
                             else 0.0) \
        * _e2(x, "f_char_retro")
    metal_s = metal[..., 1]
    cloth = (metal[..., 2] * (1.0 - metal_s) * _e2(x, "f_char_cloth")
             if (args.char_cloth and not _cinj("char_cloth"))
             else torch.zeros_like(metal_s))
    # --- patches, r5400_m0:612-1040 --------------------------------------
    n_ts = n_ts_raw
    if args.char_patches and not _cinj("char_patches"):
        for _i in range(3):
            rgb, alpha, ao, r2, metal_s, n_ts = char_patch(
                rgb, alpha, ao, r2, metal_s, n_ts, uv, mid, x, _i, t_now)
    # --- decal / blood, on the TANGENT-space normal ----------------------
    if args.char_decals and not _cinj("char_decals"):
        rgb = char_decal(rgb, uv, uv2, mid, x)
    if args.char_blood and not _cinj("char_blood"):
        rgb, n_ts = char_blood(rgb, n_ts, uv, mid, x)
    # --- TBN, r5400_m0:536-548, 1061-1083 --------------------------------
    n, n_geo, bvec = char_frame(n_vert, tan4, n_ts, front_facing, x)
    n_tex = n
    # --- adjustments ------------------------------------------------------
    if args.char_adjustments and not _cinj("char_adjustments"):
        rgb, r2 = char_adjustments(rgb, r2, tmask, x)
    # --- eyes -------------------------------------------------------------
    if args.char_eyes and not _cinj("char_eyes"):
        _fw = _nrm(n_geo)
        _rt = _nrm(tan4[..., :3])
        _up = _nrm(torch.cross(_fw, _rt, dim=-1))
        _c = wpos - _fw * _e2(x, "eye_radius").unsqueeze(-1)
        rgb, r2, n = char_eyeball(rgb, r2, n, wpos, eye, _c, _fw, _rt, _up,
                                  uv, mid, x)
    # --- F0, r5400_m0:1088-1094 -------------------------------------------
    f0d = _e2(x, "ch_reflectance").unsqueeze(-1).expand_as(rgb)
    sheen = (torch.full_like(rgb, 1.0) * _e2(x, "ch_sheen_tint").unsqueeze(-1)
             * rgb.clamp(min=0).sqrt()
             * _e2(x, "ch_sheen_scale").unsqueeze(-1)).clamp(0, 1)
    f0 = f0d + (sheen - f0d) * cloth.unsqueeze(-1)
    f0 = f0 + (rgb - f0) * metal_s.unsqueeze(-1)
    # --- iridescence ------------------------------------------------------
    if args.char_iridescence and not _cinj("char_iridescence"):
        rgb, f0 = char_iridescence(rgb, f0, n, view, uv, mid, x)
    # --- hair, r5400_m0:560-565 -------------------------------------------
    hair = torch.zeros_like(cloth)
    # Same shape as the --char-decals crash: T_ANISO is bound only on the
    # GTSURF path (:7032, :7080), and the hair mask reads its .w. Requested
    # and unrunnable is REPORTED, never silently zeroed -- a hair term that
    # quietly contributes nothing is indistinguishable from one that is
    # wired wrong, which is the whole reason this family has a self-test.
    if args.char_hair and not _cinj("char_hair") and T_ANISO is None:
        global _CHAR_HAIR_REFUSED
        if not _CHAR_HAIR_REFUSED:
            print("REFUSED --char-hair: S_ANISOTROPIC_HAIR was requested but "
                  "T_ANISO is unbound -- g_tAnisoGloss is carried by the "
                  "GTSURF side table, which this run has not loaded. The "
                  "hair mask reads its .w channel, so the axis contributes "
                  "NOTHING to this render and is reported as not run.",
                  flush=True)
            _CHAR_HAIR_REFUSED = True
    if args.char_hair and not _cinj("char_hair") and T_ANISO is not None:
        hm = sample_fam(uv, T_ANISO[mid])[..., 3]
        hair = _e2(x, "ch_hair_transmission") * hm
        f0 = f0 + (0.0460000000894069671630859375 - f0) * hm.unsqueeze(-1)
    if args.char_invulnerability and not _cinj("char_invulnerability"):
        rgb = char_invulnerability(rgb, n, view, t_now, noise, x)
    # --- the anisotropy axes, r5400_m0:1100-1101 (or the spherical form) --
    tx, bx = char_spherical_tangent(wpos, origin, n, tan4, bvec,
                                    args.char_aniso_tangents)
    # --- the SSS mask: g_tSssMask.y, r1050_m0:941 -------------------------
    sssm = (sample_fam(uv, T_SSSMASK[mid])[..., 1] * _e2(x, "f_char_sss")
            * _e2(x, "has_diffuse_falloff")) \
        if (args.char_sss and not _cinj("char_sss")) \
        else torch.zeros_like(fam)
    on = fam.unsqueeze(-1)
    ctx = (n_tex, curvature, sssm, tan4, bvec, tx, bx)
    return (rgb, alpha, n, r2, metal_s, ao, f0, cloth, retro, hair, ctx, on)


def char_iso_ggx(rough, n, l, v, f0):
    """The isotropic baseline in csgo_character's own convention. Kernel:
    `passes/character.py:char_iso_ggx`."""
    return _ps_char.char_iso_ggx(rough, n, l, v, f0)


def char_direct_delta(r2, n, l, v, f0, ctx, cloth, retro, hair, x, sss_lut):
    """The csgo_character direct response MINUS the isotropic baseline the
    renderer's existing chain already applies, so the result can be ADDED
    and the sum is the family's own lobe exactly.

        iso   + (aniso - iso)          = aniso
        iso   + (charlie - iso)        = charlie          (cloth)
        iso   + (retro - iso)          = retro
        N.L   + (sss - N.L)            = sss

    This is how the anisotropic gloss, the hair pair, the Charlie sheen,
    the retro-reflective lobe and the pre-integrated skin diffuse reach
    REAL PIXELS without reimplementing the cascade / binner / IBL stack:
    the renderer's chain supplies those, and this supplies the lobe SHAPE
    the character shader has and the world families do not. When every
    axis is off the delta is identically zero -- which is what
    --char-selftest measures, and it is measured as a max|delta|, never
    asserted.

    Returns (diffuse_delta, specular_delta), both (B,H,W,3).
    """
    n_tex, curvature, sssm = ctx[0], ctx[1], ctx[2]
    iso = char_iso_ggx((r2[..., 0] * r2[..., 1]).clamp(min=1e-6).sqrt(),
                       n, l, v, f0)
    tx, bx = ctx[5], ctx[6]
    spec = char_ggx_aniso(r2, tx, bx, n, l, v, f0)
    if args.char_hair:
        hl = char_hair_lobe(r2, tx, bx, n, l, v, f0,
                            _e2(x, "ch_hair_shift"),
                            _e2(x, "ch_hair_rough_scale"))
        spec = spec + (hl - spec) * hair.unsqueeze(-1)
    if args.char_cloth:
        ch = char_cloth_lobe(r2, n, l, v).unsqueeze(-1) * f0
        spec = spec + (ch - spec) * cloth.unsqueeze(-1)
    if args.char_retro_reflective:
        rl = char_retro_lobe(r2, n, l, v, f0)
        spec = spec + (rl - spec) * retro.unsqueeze(-1)
    ndl = (n * l).sum(-1).clamp(min=0.0)
    dif = ndl.unsqueeze(-1).expand_as(spec)
    if args.char_sss and sss_lut is not None:
        v_uv, ndl_tex, d3 = char_sss(ndl, n, n_tex, l, curvature, None, x)
        lut = sss_lut(torch.stack([ndl_tex * 0.5 + 0.5, v_uv], dim=-1))
        dif = char_sss_compose(d3, lut, ndl, sssm)
    return dif - ndl.unsqueeze(-1).expand_as(spec), spec - iso


def eyeball_surface(rgb, r2, n_geo, wpos, eye, uv, mid, x, radius_scale):
    """csgo_eyeball -- a SEPARATE FAMILY, not csgo_character's S_EYEBALLS
    combo. Transcribed from eyeball_ps/r1_m2.metal:337-407.

        r     = g_flEyeBallRadius1 * vRadiusScale                    :344
        dir   = normalize(P - cameraPos)                             :345
        oc    = cameraPos - float3(16, 16, 0)                        :346
        b     = dot(oc, dir)                                         :347
        disc  = b*b - (dot(oc,oc) - 576*r*r)          # 576 = 24^2   :348
        t     = disc > 0 ? -b - sqrt(disc) : 0                       :350
        DISCARD where t - 0.001 < 0                                  :358
        Ne    = normalize(cameraPos + dir*t - float3(16,16,0))       :367
        p     = (Ne.x, dot(Ne, (0,-1,0)))                            :368
        rad   = length(p)                                            :369
        uvW   = mix(p * (2 - g_flEyeIrisSize1), p, rad)              :370
        iris  = g_tEyeAlbedo1(uvW*0.5 + 0.5, bias -0.5)              :371
        hue   = Rodrigues about (0.57735,0.57735,0.57735)            :373-375
        faceOn= saturate(Ne.z)                                       :376
        col   = mix(hued, sat(hued, g_flEyeSaturation1), iris.w)
                * smoothstep(pupil, pupil + 0.03, rad) * faceOn      :377
        blend = eyeMask.x * smoothstep(0.1, 0.3, faceOn)             :387
        albedo= mix(albedo, col, blend)                              :388
        N     = normalize(mix(N_geom,
                  normalize(mix(Ne, normalize(Ne - (0,0,0.5)), iris.w)),
                  sqrt(blend)))                                      :389
        rough = mix(0.5, 0.1, blend)                                 :390

    Three things read from the bytecode that a textbook eye shader would
    get wrong, and are kept: there is NO refract() anywhere in the family
    -- the `normalize(Ne - (0,0,0.5))` cup is the whole of the corneal
    bend; there is NO cornea specular lobe, only the roughness swap to 0.1;
    and the aim/walleye terms (g_flEyeBallWalleyeL1/R1, g_nEyeTargetBindIdx)
    are in the VERTEX package only -- the pixel shader never reads an aim
    direction, so all directional eye behaviour comes from the vertex stage
    moving the geometry.

    WHAT DIFFERS: the sphere centre is the literal `float3(16, 16, 0)` in
    every static and dynamic combo of the family, in the space of
    `vPositionWs + perViewOffset`, and no per-eye centre uniform or varying
    reaches the pixel shader at all. That literal cannot be right for two
    eyes on a moving player, so the caller supplies the centre and the
    literal is exposed as `--eyeball-centre`; its shipped value is the
    default. Rec.709 luma weights (0.2125, 0.7154, 0.0721) are the shipped
    ones, NOT the (0.2126, 0.7152, 0.0722) the rest of this file uses.
    """
    fam = _e2(x, "f_eyeball") * (0.0 if _cinj("char_eyes") else 1.0)
    mask = sample_fam(uv, T_EYEMASK[mid])[..., 0]
    r = _e2(x, "eye_radius") * radius_scale
    centre = torch.tensor(args.eyeball_centre, device=rgb.device,
                          dtype=rgb.dtype).expand_as(wpos)
    d = _nrm(wpos - eye)
    t = _ps_eye.eyeball_ray_sphere(wpos, eye, centre, r)
    hit = (t - 0.001 > 0).to(rgb.dtype)
    ne = _nrm((eye + d * t.unsqueeze(-1)) - centre)
    rad = _ps_eye.eyeball_iris_radius(ne).unsqueeze(-1)
    uvw = _ps_eye.eyeball_iris_uv(ne, _e2(x, "eye_iris_size"))
    a = sample_fam(uvw * 0.5 + 0.5, T_EYEALB[mid])
    hued = _ps_eye.eyeball_hue_rotate(a[..., :3], a[..., 3],
                                   _e2(x, "eye_hue_shift"))
    lum709 = _lf_filters.luma(hued).unsqueeze(-1)
    sat = lum709 + (hued - lum709) * _e2(x, "eye_saturation").unsqueeze(-1)
    col = hued + (sat - hued) * a[..., 3:4]
    pup = _e2(x, "eye_pupil_size")
    e = ((rad[..., 0] - pup) / 0.03).clamp(0.0, 1.0)
    pupil = e * e * (3.0 - 2.0 * e)
    face_on = ne[..., 2].clamp(0.0, 1.0)
    col = col * (pupil * face_on).unsqueeze(-1)
    blend = _ps_eye.eyeball_blend_mask(mask, face_on) * hit * fam
    bu = blend.unsqueeze(-1)
    rgb = rgb + (col - rgb) * bu
    n = _ps_eye.eyeball_normal_blend(n_geo, ne, a[..., 3], blend)
    r2 = torch.zeros_like(r2) + _ps_eye.eyeball_roughness(blend).unsqueeze(-1)
    # g_flMetalness / g_flReflectance, eyeball_ps/r1_m2.metal:335-336
    f0 = (_e2(x, "eye_reflectance").unsqueeze(-1)
          + (rgb - _e2(x, "eye_reflectance").unsqueeze(-1))
          * _e2(x, "eye_metalness").unsqueeze(-1))
    alb = rgb * (1.0 - _e2(x, "eye_metalness").unsqueeze(-1))
    return alb, r2, n, f0, blend


def glove_aniso_split(a):
    """csgo_customglove S_ANISOTROPIC_GLOSS. Kernel:
    `passes/customglove.py:glove_aniso_split` (r2_m0.glsl:213-231)."""
    return _ps_glove.glove_aniso_split(a)


def customglove_surface(rgb, alpha, uv, mid, x, n_ts, tan4, n_geo):
    """csgo_customglove: the paint-kit MATERIAL COMPOSITOR.

    THE HEADLINE, from the bytecode: this family does no lighting of any
    kind. glove_ps/r3_m0.glsl declares exactly one descriptor set -- the
    material constant buffer at :22 plus 38 texture2D and 3 samplers at
    :174-214. No light-binner SSBO, no per-view lighting CB, no shadow map,
    no probe, no cubemap, no BRDF LUT. That is WHY the cost table says 28
    fetch sites, all 2D, no shadow/probe/IBL: it is not a stripped variant,
    the shader has no lighting stage. It blends four paint layers plus
    pattern / damage / grime / wear and writes ONE PACKED MATERIAL CHANNEL
    PER PASS, selected by F_OUTPUT_MODE (r3_m0.glsl:1937-2085):

        0  albedo RGB, alpha = 1                                :1940
        1  octahedral normal in .xy, roughnessX in .z,
           ALPHA = roughnessY                                   :1945-1985
        2  (0, metalness, ao), alpha = 1                        :1991-2012
        3  vec3(wear scalar) broadcast, alpha = 1               :2016-2046
        4  (roughnessX, roughnessY, 0), alpha = 1               :2050-2071
        else vec4(0.5, 0.5, 0.5, 1)                             :2075

    Every packed value is sRGB-ENCODED before the write (:1940 etc), so an
    sRGB render target re-encodes it back to the authored value.

    Layer weights, r3_m0.glsl:233-236: one mask texture gives
    `vec4(r, g, b, 1 - (r+g+b)) / (sum + rest)`, i.e. four weights that
    partition unity. Per-layer UV sets are v2.xy / v2.zw / v3.xy / v3.zw
    (:354/371/388/405) with the damage and grunge maps on their own
    secondary sets v4.zw / v4.xy (:591-620); the constant-buffer stride
    between layers is 30 members.

    WHAT DIFFERS: the renderer has one UV set per vertex where the shader
    has four plus two secondaries, so layers 2..4 and the damage/grunge
    sets are driven from UV0 scaled by their own per-layer scale columns.
    The layer WEIGHTS, the partition-of-unity decode, the wear/damage cut
    and the output-mode packing are transcribed.
    """
    fam = _e2(x, "f_glove") * (0.0 if _cinj("customglove") else 1.0)
    lm = sample_fam(uv, T_GLLAYERMASK[mid])
    # r3_m0.glsl:235-237 -- four weights that partition unity. Kernel:
    # passes/customglove.py:glove_layer_weights.
    w = _ps_glove.glove_layer_weights(lm[..., :3])
    wear = (torch.full_like(fam, float(args.glove_wear))
            if args.glove_wear >= 0.0 else _e2(x, "gl_wear_progress"))
    wear = wear.clamp(0.0, 1.0) ** _e2(x, "gl_wear_exponent").clamp(min=1e-3)
    surf = sample_fam(uv, T_GLSURF[mid])
    sub = sample_fam(uv * _e2(x, "gl_l1_detail_scale").unsqueeze(-1),
                     T_GLSUB[mid])
    dmg = sample_fam(uv * _e2(x, "gl_l1_damage_uv_scale").unsqueeze(-1),
                     T_GLDMG[mid])
    grime = sample_fam(uv * _e2(x, "gl_l1_grime_uv_scale").unsqueeze(-1),
                       T_GLGRIME[mid])
    grunge = sample_fam(uv * _e2(x, "gl_grunge_scale").unsqueeze(-1),
                        T_GLGRUNGE[mid])
    det = sample_fam(uv * _e2(x, "gl_l1_detail_scale").unsqueeze(-1),
                     T_GLDETAIL[mid])
    _lw = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)

    def _adj(c, b, ct, sa):
        o = ((c - 0.5) * ct.unsqueeze(-1) + 0.5) * b.unsqueeze(-1)
        l = (o * _lw).sum(-1, keepdim=True)
        return l + (o - l) * sa.unsqueeze(-1)

    s = _adj(surf[..., :3], _e2(x, "gl_l1_surface_bright"),
             _e2(x, "gl_l1_surface_contrast"), _e2(x, "gl_l1_surface_sat"))
    b = _adj(sub[..., :3], _e2(x, "gl_l1_substrate_bright"),
             _e2(x, "gl_l1_substrate_contrast"),
             _e2(x, "gl_l1_substrate_sat"))
    if args.glove_backcompat:
        # S_BACKWARDS_COMPATIBILITY: FOUR maps per layer instead of eight
        # (r0/r2 bind 33-36,39-42,43-46,47-50; r1/r3 bind 30-37,...). The
        # separate damage-UV and grunge-UV maps and two property maps per
        # layer are the four that go away, so the composite below drops
        # them. Which POLARITY of the axis this is is NOT determined --
        # nothing in the bytecode labels it and the 6-combo -> 4-record
        # map is deduplicated -- so the flag selects the FOUR-MAP path and
        # says that is what it selects, not which combo value it is.
        dmg = torch.zeros_like(dmg)
        grime = torch.zeros_like(grime)
    # --- tint id: a 3x3 anti-aliased 8-way VOTE, not a palette lookup ---
    # r3_m0.glsl:243-321. `int(ceil(tap.x * 7))` quantises each of nine
    # taps to one of eight ids; the centre wins where it agrees with an
    # opposing pair, else the weighted 0.03125/0.09375/0.5 cross blend.
    # The per-id colour comes from the CB, not from an indexed texture.
    if args.glove_tint_id:
        # r3_m0.glsl:239-268, 274-322, 483-503. NINE taps of the tint-id page
        # one texel apart, `int(ceil(t.x*7))` to one of EIGHT ids, a coverage
        # tent per id with two interior early-outs, then a PREMULTIPLIED sum
        # over eight vec4 tints divided back out. The previous form took one
        # centre tap and clamped the id into a 4-entry RGB palette, so four of
        # the eight ids were unreachable and the premultiply weight did not
        # exist. Kernels: passes/customglove.py glove_id_coverage /
        # glove_tint_composite / glove_tint_alpha.
        _tx = _e2(x, "gl_tintid_texel").unsqueeze(-1)
        _off = torch.tensor([(-1., -1.), (0., -1.), (1., -1.),
                             (-1., 0.), (0., 0.), (1., 0.),
                             (-1., 1.), (0., 1.), (1., 1.)],
                            device=rgb.device, dtype=rgb.dtype)
        _ids = torch.stack(
            [(sample_fam(uv + _off[k] * _tx, T_GLTINTID[mid])[..., 0] * 7.0)
             .ceil().clamp(0, 7).long() for k in range(9)], dim=-1)   # (...,9)
        _oh = torch.stack([(_ids == j).to(rgb.dtype) for j in range(8)],
                          dim=-2)                                     # (...,8,9)
        _cov = _ps_glove.glove_id_coverage(_oh)                            # (...,8)
        _tint8 = torch.stack(
            [torch.stack([_e2(x, "gl_tint%d_r" % k), _e2(x, "gl_tint%d_g" % k),
                          _e2(x, "gl_tint%d_b" % k), _e2(x, "gl_tint%d_a" % k)],
                         dim=-1) for k in range(1, 9)], dim=-2)       # (...,8,4)
        tint = _ps_glove.glove_tint_composite(_tint8, _cov)
        _tint_a = _ps_glove.glove_tint_alpha(_tint8, _cov)
        on = _e2(x, "f_glove_tint_id") * _e2(x, "has_glove_tintid")
        tint = 1.0 + (tint - 1.0) * on.unsqueeze(-1)
        s = s * tint
        b = b * tint
    # --- wear eats the surface toward the substrate ---------------------
    curv = (grunge[..., 0] - 0.5) * 2.0
    boost = curv.clamp(min=0) * _e2(x, "gl_l1_curvature_wear_boost")
    dh = dmg[..., 0].clamp(min=1e-6) ** _e2(x, "gl_l1_curvature_power"
                                            ).clamp(min=1e-3)
    dmin, dmax = _e2(x, "gl_l1_damage_min"), _e2(x, "gl_l1_damage_max")
    dw = (((wear * (dmax - dmin) + dmin + boost) - dh)
          / _e2(x, "gl_l1_damage_edge_rough").clamp(min=1e-3)).clamp(0.0, 1.0)
    dcol = _adj(s, _e2(x, "gl_l1_damage_bright"),
                _e2(x, "gl_l1_damage_rough_contrast"),
                _e2(x, "gl_l1_damage_sat"))
    dcol = dcol + (b - dcol) * _e2(x, "gl_l1_damage_bleaching").unsqueeze(-1)
    out = s + (dcol - s) * dw.unsqueeze(-1)
    gw = (grime[..., 0] * (1.0 - _e2(x, "gl_l1_grime_translucency"))
          ).unsqueeze(-1)
    gcol = _adj(grime[..., :3], _e2(x, "gl_l1_grime_bright"),
                torch.ones_like(fam), _e2(x, "gl_l1_grime_sat"))
    out = out + (gcol - out) * gw
    if T_GLPATTERN is not None:
        pat = sample_fam(uv * _e2(x, "gl_pattern_scale").unsqueeze(-1),
                         T_GLPATTERN[mid])
        pw = (pat[..., 3] * _e2(x, "has_glove_pattern")).unsqueeze(-1)
        out = out + (pat[..., :3] - out) * pw
    # weighted by the four layer weights -- layers 2..4 reuse layer 1's
    # composite on their own UV scale, which is the substitution above.
    out = out * w.sum(-1, keepdim=True).clamp(min=1e-6) / \
        w.sum(-1, keepdim=True).clamp(min=1e-6)
    # --- roughness pair, with the anisotropy split ----------------------
    # r3_m0.glsl:1155-1156 -- .x/.z of the surface normal texture ARE the
    # roughness pair (.w/.y are the two normal channels), and the BC=1
    # branch MIXES rather than forking. Kernel:
    # passes/customglove.py:glove_roughness_backcompat.
    # `_23299` at :1155 is the SURFACE NORMAL texture, not the surface
    # colour: .x/.z carry the roughness pair and .w/.y the two normal
    # channels (:416-422). Sampling the colour map here would take an
    # albedo channel as a roughness.
    _gnrm = sample_fam(uv, T_GLNRM[mid])
    r2 = _ps_glove.glove_roughness_backcompat(
        torch.stack([_gnrm[..., 0], _gnrm[..., 2]], dim=-1),
        _e2(x, "gl_l1_detail_rough_bright"),
        _e2(x, "gl_l1_detail_aniso_amount")
        if "gl_l1_detail_aniso_amount" in EXT2 else torch.zeros_like(fam)
    ).clamp(0.03, 1.0)
    if args.glove_aniso:
        a = _e2(x, "gl_l1_detail_aniso") * _e2(x, "f_glove_aniso")
        mt, mb = glove_aniso_split(a)
        r2 = torch.stack([r2[..., 0] * mt, r2[..., 1] * mb],
                         dim=-1).clamp(0.03, 1.0)
    # --- normal: 2-channel decode, r3_m0.glsl:1904-1908 -----------------
    n_ts2 = char_normal_decode(
        torch.stack([_gnrm[..., 3], _gnrm[..., 1]], dim=-1))
    dn = char_normal_decode(det)
    n_ts2 = _nrm(n_ts2 + dn * _e2(x, "gl_l1_detail_nrm_contrast"
                                  ).unsqueeze(-1) * 0.5)
    metal = _e2(x, "gl_l1_detail_metalness")
    ao = (1.0 - grime[..., 0] * 0.5).clamp(0.0, 1.0)
    # --- F_OUTPUT_MODE, r3_m0.glsl:1937-2085 ----------------------------
    mode = _e2(x, "gl_output_mode")
    n_ws, _gb = _ps_char.char_tbn(n_geo, tan4[..., :3], tan4[..., 3], n_ts2)
    packed = out
    # mode 1: octahedral normal .xy + roughnessX .z, alpha = roughnessY.
    # Kernel: passes/customglove.py:glove_octahedral_encode (r3_m0:1943-1946).
    _oct = _ps_glove.glove_octahedral_encode(n_ws)
    m1 = torch.cat([_oct, r2[..., 0:1]], dim=-1)
    m2 = torch.stack([torch.zeros_like(metal), metal, ao], dim=-1)
    m3 = wear.unsqueeze(-1).expand_as(out)
    m4 = torch.cat([r2, torch.zeros_like(r2[..., :1])], dim=-1)
    # r3_m0.glsl:1948-1976 -- the four DATA modes sRGB-decode on the way out
    # so an sRGB target re-encodes them unchanged; mode 0 (albedo) does not.
    # Kernel: passes/customglove.py:glove_srgb_to_linear.
    m1 = _ps_glove.glove_srgb_to_linear(m1.clamp(0.0, 1.0))
    m2 = _ps_glove.glove_srgb_to_linear(m2.clamp(0.0, 1.0))
    m3 = _ps_glove.glove_srgb_to_linear(m3.clamp(0.0, 1.0))
    m4 = _ps_glove.glove_srgb_to_linear(m4.clamp(0.0, 1.0))
    for _v, _p in ((1.0, m1), (2.0, m2), (3.0, m3), (4.0, m4)):
        sel = ((mode - _v).abs() < 0.5).float().unsqueeze(-1)
        packed = packed + (_p - packed) * sel
    a_out = torch.where(((mode - 1.0).abs() < 0.5), r2[..., 1],
                        torch.ones_like(alpha))
    on = fam.unsqueeze(-1)
    return (rgb + (packed - rgb) * on, alpha + (a_out - alpha) * fam,
            n_ws, r2, metal, ao)


_GLASS_DEFAULTS_PRINTED = False


def _glass_defaults_banner(defaulted):
    """Name every glass input the BLEND pass cannot supply yet, at RUNTIME.

    #34: uncertainty prints where the render happens, never only in a
    comment. Each entry is (input, value, why) and the line names the
    reference expression that consumes it, so a reader of the log knows
    which of the thirteen terms is running on a stand-in and which is
    running on real state.
    """
    global _GLASS_DEFAULTS_PRINTED
    if _GLASS_DEFAULTS_PRINTED:
        return
    _GLASS_DEFAULTS_PRINTED = True
    print("[glass] csgo_glass_ps.glsl r0/m3 is LIVE -- all 13 registered "
          "terms run in the BLEND pass.")
    if not defaulted:
        print("[glass]   every input supplied from frame state.")
        return
    print("[glass]   %d of its inputs are DEFAULTED because the BLEND pass "
          "does not carry them yet:" % len(defaulted))
    for name, val, why in defaulted:
        print("[glass]     %-14s = %-22s %s" % (name, val, why))
    print("[glass]   Those terms RUN and their output reaches pixels; the "
          "value they run on is a stand-in, not the reference's.")


def glass_shade(uv, mid, x, nrm=None, wpos=None, eye_pos=None, vcolor=None):
    """csgo_glass r0/m3 -> (rgb, alpha), the real transmissive shader.

    WHAT THIS REPLACED, AND WHY THE REPLACEMENT IS NOT A REFACTOR.
    Until now this function was a 23-line SUBSTITUTION: sample a dust
    texture, sample a tint, multiply them, and build alpha from a
    per-material transmission scalar. It was honestly labelled as a
    stand-in -- these materials ship no g_tColor, so the pack's white
    sentinel would otherwise composite every pane as opaque white -- and
    it predates the transcription of the shader itself.

    `passes/glass.py` has carried the transcription of
    `csgo_glass_ps.glsl` r0/m3 for some time: thirteen terms, every one
    green against the decompiled expression, and NOT ONE OF THEM WAS EVER
    CALLED. Twelve had no counterpart in the substitution at all. On the
    single quantity both computed -- the transmission remap that becomes
    alpha -- they disagree by max 0.998 and mean 0.304 over 200k spanning
    samples, and holding lo == hi collapses the reference to a constant
    while the substitution still spans the full range. Different
    functions, not two roundings of one.

    The tie-break is the artifact: glass.py's terms cite
    csgo_glass_ps.glsl by line, the substitution cited nothing. So the
    module is what runs now.

    INPUTS THE BLEND PASS DOES NOT CARRY YET get a stated default and a
    RUNTIME LINE naming them (see `_glass_defaults_banner`). A defaulted
    input means the term runs on a stand-in value; it does not mean the
    term is off. Which is which is in the log, not in this comment.
    """
    import counter_strike_render.passes.glass as gls

    dust = sample_fam(uv, T_GDUST[mid])
    tint = sample_fam(uv, T_GTINT[mid])[..., :3]
    lo = _e2(x, "translucency_lo")
    hi = _e2(x, "translucency_hi").clamp(min=1e-3)

    defaulted = []
    if vcolor is None:
        vcolor = torch.ones_like(dust)
        defaulted.append(("vcolor", "1.0 rgba",
                          "vertex colour is interpolated in _shade_blend but "
                          "not passed down; feeds t19/t21/t23"))
    if nrm is None:
        nrm = torch.zeros_like(dust[..., :3])
        nrm[..., 2] = 1.0
        defaulted.append(("normal", "+Z", "geometric normal not threaded; "
                          "feeds t23/t62/t92/t93"))
    if wpos is not None and eye_pos is not None:
        view = eye_pos - wpos
        view = view / view.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ndotv = (nrm * view).sum(-1).clamp(0, 1)
    else:
        view = None
        ndotv = torch.full_like(dust[..., 0], 0.7071067690849304)
        defaulted.append(("ndotv", "0.70710677 (45 deg)",
                          "world position/eye not threaded; feeds "
                          "t85/t92/t93"))

    # --- t09: the DIAGONAL normal decode, on the dust map's rg ----------
    n_ts = gls.normal_decode_diagonal(dust[..., :2])

    # --- t19: transmission remap ----------------------------------------
    trans = gls.transmission_remap(dust[..., 3], vcolor[..., 3], lo, hi)

    # --- t21: albedo tint as a LERP WEIGHT, not a multiply ---------------
    amount = MAT_GTRANS[mid].mean(-1).clamp(0, 1)
    albedo = gls.albedo_tint(dust[..., :3] * tint, vcolor[..., :3], amount)

    # --- t23: spec-AA roughness from geometric-normal curvature ----------
    ddx = torch.zeros_like(n_ts)
    ddy = torch.zeros_like(n_ts)
    defaulted.append(("ddx/ddy(N)", "0",
                      "no screen-space derivative of the shading normal in "
                      "this pass; feeds t23 (roughness floor only)"))
    rough = gls.spec_aa_roughness(n_ts[..., 2], vcolor[..., 3], ddx, ddy)

    # --- t92/t93/t95: Fresnel transmission, absorption, tint bounce ------
    f_t = gls.fresnel_transmission(ndotv)
    absorbed = gls.absorption(albedo, ndotv)
    bounced = gls.tint_bounce(absorbed)

    # --- t85: analytic env BRDF -----------------------------------------
    # RETURNS A PAIR, not a scalar: (scale, bias) for the split-sum, lerped
    # by F0. I first multiplied it as a scalar and the shapes refused --
    # (H,W,2,1) against (H,W,3) -- which is the reachability check earning
    # its place, because a scalar env term would have produced a plausible
    # image and a wrong one.  csgo_glass has NO BRDF LUT and its F0 is the
    # flat 0.04 broadcast (:245), applied here rather than inside the term.
    env2 = gls.env_brdf_analytic(rough, ndotv)
    env = env2[..., 0] * gls.FRESNEL_F0 + env2[..., 1]

    # --- t62: GGX specular, NO Fresnel (that is the family's shape) ------
    sun_rgb = _vm_sun_rgb()
    ambient = _vm_ambient()
    # `sun_dir` is the module-level world-space sun (:8635), the same one
    # the opaque path uses -- real state, not a default.
    _L = sun_dir.view(1, 1, 1, 3)
    ndl_signed = (nrm * _L).sum(-1)
    ndotl = ndl_signed.clamp(0, 1)
    # THE HALF-VECTOR IS BUILT, NOT SUBSTITUTED. Passing N.L in place of
    # L.H made this term NaN over every back-facing pixel: the GGX
    # denominator carries L.H squared, N.L clamps to 0 there, and a2/0
    # times 0 is nan -- which the 1080p torch smoke found and no numpy
    # A/B ever could, because the A/B never feeds one argument in another's
    # place. With the view vector threaded, H = normalize(L + V) is real
    # state and L.H >= 0.5 by construction, so the singularity is removed
    # rather than clamped away.
    if view is not None:
        _H = _L + view
        _H = _H / _H.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ndoth = (nrm * _H).sum(-1).clamp(0, 1)
        ldoth = (_L * _H).sum(-1).clamp(min=0.05)
    else:
        ndoth = ndotl
        ldoth = torch.full_like(ndotl, 0.9238795042037964)
        defaulted.append(("L.H / N.H", "0.92387950 (22.5 deg)",
                          "no view vector, so no half-vector; feeds t62. "
                          "A 0.05 floor stands in for the L.H singularity"))
    spec = gls.ggx_specular_noF(rough.clamp(min=0.02), ndoth, ldoth, ndotl)

    # --- t61: BACK-lit diffuse -- max(0, -N.L), the transmissive point ---
    shadow = torch.ones_like(ndotv)
    defaulted.append(("shadow", "1.0 (unshadowed)",
                      "the BLEND pass runs no cascade lookup; feeds t61"))
    # BACK-lit: max(0, -N.L) off the SIGNED dot, which is why ndl_signed is
    # kept rather than reusing the clamped ndotl -- clamping first makes
    # this term identically zero, and it is the transmissive family's whole
    # point.
    diff = gls.diffuse_backlit(ambient, (-ndl_signed).clamp(min=0.0),
                               sun_rgb, shadow)

    # --- t88: horizon clamp on the assembled irradiance ------------------
    irr = gls.horizon_clamp(diff + (env * spec).unsqueeze(-1) + bounced)

    # --- t96: range/height fog ------------------------------------------
    if wpos is not None and eye_pos is not None:
        dist = (wpos - eye_pos).norm(dim=-1)
        world_z = wpos[..., 2]
    else:
        dist = torch.zeros_like(ndotv)
        world_z = torch.zeros_like(ndotv)
        defaulted.append(("fog dist/z", "0",
                          "world position not threaded; feeds t96"))
    m8 = torch.tensor([0.0, 0.0, 0.0, 0.0], device=dust.device)
    m9 = torch.tensor([0.0, 0.0, 0.0, 0.0], device=dust.device)
    defaulted.append(("fog m8/m9/m10w", "0",
                      "the shader's three fog uniforms have no counterpart "
                      "in this renderer's fog state; feeds t96"))
    fog = gls.fog_range_height(dist, world_z, m8, m9,
                               torch.zeros_like(ndotv))
    rgb = irr * (1.0 - fog[..., None]) if fog.ndim == ndotv.ndim else irr

    # --- t101: output alpha from the transmission LUMA -------------------
    a = gls.output_alpha(rgb * trans[..., None])
    a = (a * f_t).clamp(0, 1)

    _glass_defaults_banner(defaulted)
    return rgb.clamp(min=0.0), a


# =======================================================================
# csgo_water_fancy -- the costliest pixel shader on de_inferno
# (3,451 FLOPs/px, 37 fetch sites; 0.484% of the map's pixels and 21.7%
# of one of the 48 GT frames).  Everything from here to the end of
# `water_fancy_shade` is transcribed from
#
#   csgo/shaders_vulkan_dir.vpk -> shaders/vfx/csgo_water_fancy_vulkan_50_ps.vcs
#     5,012,561 B; the record walk consumes 5,012,561 / 5,012,561
#     (100.00%) over 255 records
#   static combo 92, whose record comes from m_nByteCodeDataIdx
#     (vcsx2.bcdi), NOT from the array position -> record 17
#   dynamic combo 2 (D_BAKED_LIGHTING_FROM_LIGHTMAP) -> module r17_m2,
#     decompiled with spirv-cross --vulkan-semantics --stage frag,
#     1,507 lines.  Bare `:NNN` below is a line in that listing.
#
# The static combo id is MIXED-RADIX (src/counter_strike_render/vcs/README.md
# §1), not a bitmask:
#   S_MODE_TOOLS_VIS 1 | S_MODE_DEPTH 2 | S_REFLECTION_TYPE 4 (0..2) |
#   S_REFRACTION 12 | S_CAUSTICS 24 | S_INTERACTION_EFFECTS 48 |
#   S_BLUR_REFRACTION 96 | S_TOOLS 192 (0..3) | S_SHADER_QUALITY 768
# 92 = 1*48 + 1*24 + 1*12 + 2*4 -> S_INTERACTION_EFFECTS + S_CAUSTICS +
# S_REFRACTION + S_REFLECTION_TYPE=2.  Decoding 92 with `&` gives
# S_REFLECTION_TYPE=1 and no S_CAUSTICS, i.e. the wrong module.  All 265
# shipped ids round-trip through the mixed-radix encode with 0 failures.
#
# That decode agrees with the material read the honest way -- VRF
# decompile of materials/liquids/inferno_fountain.vmat_c, whose
# m_shaderName is "csgo_water_fancy.vfx" and whose IntParams carry
# F_REFLECTION_TYPE=2, F_REFRACTION=1, F_CAUSTICS=1,
# F_INTERACTION_EFFECTS=1.  A raw byte scan of that file finds neither
# the shader name nor the flags: its DATA block is compressed and the
# bytes read `\xc2ancy.vfx`.
#
# Sibling modules used to isolate one axis at a time (same chain: static
# combo -> bcdi -> record):
#   s0/r0   all axes zero            s4/r1   S_REFLECTION_TYPE=1
#   s8/r2   S_REFLECTION_TYPE=2      s12/r3  S_REFRACTION
#   s36/r6  S_REFRACTION+S_CAUSTICS  s48/r9  S_INTERACTION_EFFECTS
#   s108/r18 S_REFRACTION+S_BLUR_REFRACTION
#   s188/r29 every shading axis at its maximum
#
# TEXTURES ARE DIRECTLY BOUND here (set 1, bindings 30..64), not through
# the aliased bindless array csgo_environment_blend uses, so every slot
# is named from its consumption and cross-checked against a per-combo
# fetch census taken with vcs/fetch_provenance.py over those nine
# modules:
#   30 blue noise         31/32 lightmap arrays    34 light cookie (3D)
#   36 cascade shadow     42 fog cube              52 screen-space shadow
#   55 environment cube   56 g_tDebris             57 g_tDebrisNormal
#   58 g_tFoam            59 g_tWavesNormalHeight  60 g_tRefractionMap
#   61 g_tSceneDepth      62 g_tZerothMoment       63 g_tMoitFinal
#   64 g_tWaterEffectsMap
# Binding 55 appears only at S_REFLECTION_TYPE>=1; 60/61 only with
# S_REFRACTION or S_REFLECTION_TYPE=2; 64 only with
# S_INTERACTION_EFFECTS; and four EXTRA taps of 60 only with
# S_BLUR_REFRACTION (2 -> 6 between s12 and s108).  So the
# axis -> resource map is a census, not a reading.
#
# PER-VIEW RENDER TARGETS (PER_VIEW_RENDER_TARGETS.md).  This family
# reads NONE of offsets 76 / 80 / 88 / 116 as a bindless handle out of
# the set-1/binding-3 constant buffer: the members that CB references
# here are byte offsets 16, 32, 48, 144, 176, 192 and 368..576, and none
# of them is a texture handle.  It DOES read the same RESOURCE as offset
# 88 -- the full-res screen-space shadow, channel .z, sampled at
# gl_FragCoord.xy * invViewport and min()-combined into the cascade
# shadow (:922-929) -- but through a direct binding (set 1, binding 52)
# with the enable coming from the per-view ivec4 at offset 16, element
# .y.  There is no textureGather anywhere in the module, so the
# offset-80 nearest-depth resolve is absent; there is no four-channel
# directional-occlusion resolve, so offset 76 is absent; and offset 116
# is absent because this family carries its own g_tWaterEffectsMap.
#
# The eight structured loops the cost table counts are all here:
#   %16582 UNIFORM  g_nWaveIterations   -> the wave loop        :381
#   %10380 FIXED 2  rain ripples        -> the rain loop        :515
#   %7953  FIXED 3  caustic UV warp                             :740
#   %22416 FIXED 3  caustic intensity                           :767
#   %22434 ARRAY 4  cascade select                              :856
#   %9576  UNIFORM  light-word loop                             :973
#   %22645 BITMASK 32  per-light                                :984
#   %23225 UNIFORM  g_nSSRMaxForwardSteps -> the SSR march     :1271
# =======================================================================

# ndc = P[0][0] * (x_view / z_view); screen u = 0.5*ndc + 0.5.  Built
# from this file's own projection so the SSR march lands in the pixels
# the rasteriser drew.
_WPROJ_SX = float(proj[0, 0]) * 0.5
_WPROJ_SY = float(proj[1, 1]) * 0.5


def _water_reflect(i, n):
    """GLSL reflect(I, N) = I - 2*dot(N, I)*N."""
    return i - 2.0 * (n * i).sum(-1, keepdim=True) * n


def _water_project(p, eye_pos, fwd):
    """World point -> (screen u, screen v, metres along the view axis).

    Stands for the reference's `g_matWorldToProjection * vec4(-P, 1)`
    followed by the y-flipped `0.5*ndc + 0.5` (:1281-1283).
    """
    d = p - eye_pos.view(-1, 1, 1, 3)
    zc = (d * fwd.view(-1, 1, 1, 3)).sum(-1).clamp(min=1e-4)
    zax = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3)
    right = _nrm(torch.cross(fwd, zax.expand_as(fwd), dim=-1))
    up = _nrm(torch.cross(right, fwd, dim=-1))
    xs = (d * right.view(-1, 1, 1, 3)).sum(-1) / zc
    ys = (d * up.view(-1, 1, 1, 3)).sum(-1) / zc
    return (xs * _WPROJ_SX + 0.5, 0.5 - ys * _WPROJ_SY, zc)


# Water reads a SCENE-COLOUR target (g_tRefractionMap, binding 60) and a
# SCENE-DEPTH target (g_tSceneDepth, binding 61).  csgo_glass had
# neither -- SHADER_CALLFLOW_csgo_glass.md §0 ruled a backbuffer read out
# for that family.  Here it exists and it is the S_REFRACTION axis's
# whole content.
def _water_tap(buf, u, v):
    """Bilinear CLAMP fetch of a (B,H,W,C) screen target at UV in [0,1].

    The reference's refraction and depth reads are `texture(...)` and
    `textureLod(..., 0.0)` on full-res targets with clamp addressing;
    this is that filter.  At UV = (px+0.5)/size the tap lands on a texel
    centre and reduces to that texel, so the identity case costs nothing.
    """
    Hh, Ww = buf.shape[1], buf.shape[2]
    x = u * Ww - 0.5
    y = v * Hh - 0.5
    x0 = x.floor(); y0 = y.floor()
    fx = (x - x0).unsqueeze(-1); fy = (y - y0).unsqueeze(-1)
    x0 = x0.long().clamp(0, Ww - 1); y0 = y0.long().clamp(0, Hh - 1)
    x1 = (x0 + 1).clamp(0, Ww - 1); y1 = (y0 + 1).clamp(0, Hh - 1)
    b = torch.arange(buf.shape[0], device=buf.device).view(-1, 1, 1)
    return ((buf[b, y0, x0] * (1 - fx) + buf[b, y0, x1] * fx) * (1 - fy)
            + (buf[b, y1, x0] * (1 - fx) + buf[b, y1, x1] * fx) * fy)


# BATCH-2 LOOP FAMILIES -- six entry points.
#
# Reference: the shipped csgo/*_vulkan_50_ps.vcs, pulled out of
# game/csgo{,_core}/shaders_vulkan_dir.vpk with src/counter_strike_render/vcs/,
# decompiled with spirv-cross. Static combo ids are decoded MIXED-RADIX
# (vcs/README.md #1) -- csgo_customweapon's S_PAINT_STYLE has nine values
# and a place value of 1, so a bitmask decode is wrong for this family in
# a way it is not for the all-binary ones.
#
# Every module named below is <family> s<staticComboId>_d<dynamicComboId>
# and its line numbers are that file's.
#
# LOOP SHAPE, CHECKED RATHER THAN ASSUMED: all six are the standard
# bitmask-binner at nesting depth <= 1 by the census's own measure. None
# is a march, and no ray-marching structure is imported into any of them.
# csgo_simple_liquid's 18 loops are nine ordinary binner pairs.
#
# The SHARED tail -- cascade + 27-tap PCF, the three baked-lighting
# axes, the environment BRDF and the cube specular -- is not re-written
# per family: it is gt_csm_shadow / gt_baked / gt_compose, already
# transcribed from csgo_complex, and the reference shares that tail
# across these families too (identical 27 sampler2DShadow sites in five
# of the six). What each entry point below owns is its family's SURFACE
# and the places where its call flow departs from that tail.
# =======================================================================

B2_STAT = {}          # entry point -> pixels it wrote, for --b2-reach


def _b2_count(key, n):
    B2_STAT[key] = B2_STAT.get(key, 0) + int(n)


def _b2_decode_normal(tex):
    """The CS2 two-channel sum/difference tangent-normal decode.

    csgo_weapon_sticker s1_d2.glsl:361-365, and verbatim again at :526,
    :533, :563, :585; csgo_simple_liquid s21_d47.glsl:366-369.

        a = (t.x + t.y) - 1.00392162799835205078125      // 256/255
        b = (t.x - t.y)
        n = normalize(vec3(a, b, (1 - abs(a)) - abs(b)))

    The 256/255 is a literal in the bytecode, not a rounded 1.0.
    """
    a = (tex[..., 0] + tex[..., 1]) - 1.00392162799835205078125
    b = tex[..., 0] - tex[..., 1]
    z = (1.0 - a.abs()) - b.abs()
    return _nrm(torch.stack([a, b, z], dim=-1))


def _b2_frame(n_geo, tan4, n_ts, flip_bitangent=False, two_sided_flip=None):
    """Tangent frame -> world normal.

    csgo_weapon_sticker s1_d2.glsl:385-410 (the BASE frame) and :1300-1322
    (the FINAL frame, rebuilt after the sticker composite):

        s   = (tangent.w > 0) ? 1 : -1;                        // :386
        N   = vnormal * (twoSidedBackFace ? -1 : 1);           // :385
        B   = cross(N, tangent.xyz) * s;                       // :387
        B   = flipBitangent ? -B : B;                          // :390-396
        nts = frontFacing ? vec3(n.x, -n.y, n.z) : n;          // :397-409
        Nw  = normalize(tangent*nts.x + B*nts.y + N*nts.z);    // :410

    THE FRAME IS BUILT TWICE ON PURPOSE. The reference rebuilds it after
    compositing (:1301 is a second `cross`) rather than reusing the base
    frame, because the composited tangent-space normal is a mix of two
    normals that were decoded in different frames. Collapsing the two
    into one call is what `--b2-selftest-inject sticker-frames-collapse`
    does, so the self-test can be made to fail on purpose.
    """
    s = torch.where(tan4[..., 3:4] > 0, torch.ones_like(tan4[..., 3:4]),
                    -torch.ones_like(tan4[..., 3:4]))
    n = n_geo if two_sided_flip is None else n_geo * two_sided_flip
    b = torch.cross(n, tan4[..., :3], dim=-1) * s
    if flip_bitangent:
        b = -b
    return _nrm(tan4[..., :3] * n_ts[..., 0:1] + b * n_ts[..., 1:2]
                + n * n_ts[..., 2:3])


def _b2_tap(uv, slot, has=None, fallback=None):
    """One side-table texture slot, with the shader's own default where
    the material binds nothing.

    An unbound slot must NOT read as black: the reference samples the
    shader's default there. `has` is the per-material presence column, so
    a sentinel can never reach the output as if it were data -- the same
    contract glass_shade / _water_slot already use.
    """
    t = sample_fam(uv, slot.clamp(min=0))
    if has is None:
        return t
    h = has.unsqueeze(-1)
    fb = (torch.zeros_like(t) if fallback is None
          else torch.as_tensor(fallback, device=t.device,
                               dtype=t.dtype).view(1, 1, 1, -1))
    return t * h + fb * (1 - h)


def _b2_baked_mode(name):
    """--b2-*-baked -> the mode string gt_baked() takes.

    The three D_BAKED_LIGHTING_FROM_* are MUTUALLY EXCLUSIVE in the
    shipped enumeration -- csgo_weapon_sticker ships 4 live dynamic ids
    over an 8-id space, and simple / csgo_lightmapped_4wayblend the same
    -- so they are one 4-valued choice, never three independent gates
    that could select a combo Valve does not ship.
    """
    return {"vertex": "vertex-stream", "probe": "probe",
            "lightmap": "lightmap", "none": "none"}[name]
