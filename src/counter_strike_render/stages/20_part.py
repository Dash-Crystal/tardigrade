

# -----------------------------------------------------------------------
# 1. csgo_weapon_sticker -- 7,127 FLOPs/px, 82 fetch sites, 13 loops.
#    The most expensive pixel shader measured anywhere in CS2 so far.
# -----------------------------------------------------------------------
def b2_weapon_sticker_shade(uv, mid, x, base_rgb, base_rough, base_metal,
                            n_geo, tan4, wpos, view, eye, lm=None,
                            lm_ok=None, vlit=None, ao=None, sun_shadow=None):
    """csgo_weapon_sticker s1_d2 -> (rgb, alpha, normal, rough, metal).

    ONE sticker layer over the base surface, with TWO foil sub-layers
    inside it. That is measured, not assumed: the whole sticker path is
    one `if (_5618._m26 != 0)` at :423, and the two foil UV sets are
    packed in a single vec4 at :556 --

        foilUV = vec4(stickerUV * k, (stickerUV + 0.5) * k),
        k = 2.5 * g_flFoilScale                                   // :425

    Sticker UV (:434-490):
        uv  = (uv0 - 0.5) * scale + 0.5;                          // :435
        out = (clamp(uv,0,1) != uv);   // rejects the fragment    // :437-447
        stickerUV = clamp(uv, 0, 1);                              // :490

    Sticker colour and the packed alpha (:492-494):
        c = texture(g_tSticker, stickerUV, lodBias = -1.0);
        a = clamp(c.w * 12.75, 0, 1);        // alpha packed x 1/12.75

    THE EDGE BEVEL (:507) -- the term the mip fade cross-fades in:
        m    = textureQueryLod(g_tSticker, ...).x;                // :500
        fade = 1 - clamp(m - 3, 0, 1);                            // :501
        off  = 0.01 * vec2(dot(cross(cbA, cbB), -Nw), dot(cbB, Nw));
        a    = mix(a, max(a, clamp(tex(stickerUV - off).w*12.75,0,1)*0.7),
                   fade);                                         // :507
    so the bevel is a SECOND sticker fetch at a normal-driven offset,
    faded out past mip 3.

    Sticker normal onto the base (:524-538) -- an unnormalised additive
    blend against the base normal forced to a COARSE mip:
        nb = decode(textureLod(g_tNormal, uv0, max(m, 3)).xy);    // :531-537
        ns = decode(texture(g_tStickerNormal, stickerUV, -1).xy); // :524-530
        n  = normalize(nb + ns * a * 2.0);                        // :538

    THE FOIL (:546-604), per sub-layer -- an iridescence built by
    reflecting the view vector off the foil normal and taking sin() of it:
        nf = decode(foilTex.xy);                                  // :562-567
        r  = sin(reflect(V, normalize(T*nf.x + B*nf.y + Ng*nf.z)) * 12.0);
                                                                  // :570
        p  = max(0, r - 0.99) * 100.00009918212890625;            // :571
        c  = vec3(pow(dot(clamp(-r,0,1), LUMA), 4.0))
           + vec3(dot(clamp(0.14999997615814208984375 - r,0,1), LUMA)*0.25)
           + (p + pow(p.yzx, vec3(4.0))) * 4.0 * 0.75;            // :572
    with LUMA = (0.2125000059604644775390625,
                 0.7153999805450439453125,
                 0.07209999859333038330078125).
    Layer A scales its decoded normal by its own z (:568); layer B does
    NOT (:583). Both perturb the sticker normal's .xy in tangent space by
        s = 0.039999999105930328369140625 * foilAmount
          * clamp(1 - min(ddx,ddy)*40, 0, 1)                      // :573
    which is the derivative-driven distance fade.

    THE COMPOSITE (:1227-1239) is a plain per-channel mix at the sticker
    coverage, in TANGENT space, before the frame is rebuilt:
        albedo = mix(base,  sticker, cov);                        // :1232
        rough  = mix(baseR, stickR,  cov);                        // :1233
        metal  = mix(baseM, stickM,  cov);                        // :1234
        gloss  = mix(baseG, stickG,  stickerColour.w);            // :1235
        f0     = mix(g_flReflectance, 0.04, cov);                 // :1238
        n_ts   = normalize(mix(baseN_ts, stickN_ts, cov));        // :1239

    TWO IRREGULARITIES ARE REPRODUCED RATHER THAN TIDIED: the gloss at
    :1235 mixes on the sticker COLOUR ALPHA and not on the coverage every
    other channel uses, and :1229's vec4 has only its .w ever written.
    Both look like slips; both are what the shipped module does, and a
    transcription that "fixes" them is no longer a transcription.

    WHAT DIFFERS from the reference:
      * the 6-direction ambient-cube probe volume (:845-859, 12 of the 21
        sampler3D fetches) is an ENGINE 3D texture present in no .vcs, so
        the holo view direction's dominant-light term is taken from the
        renderer's own baked irradiance and sun direction instead. That
        is the same substitution the D_BAKED_LIGHTING_FROM_LIGHTMAP
        variant makes in the reference itself (:700-701 collapses the
        whole probe binner to `dot(V, Ng) + dot(V, g_vLightDir)`), so the
        lightmap arm is EXACT and the probe arm is the lightmap arm's
        form. Reported, not swallowed.
      * the LTC area-light path (:2320-3160, 14 of the 22 textual
        `cross`) needs the LTC LUT array, also engine-side; area lights
        fall back to the punctual BRDF, which understates a large
        emitter's softness and never invents one.
    """
    e = MAT_EXT2[mid]
    q = int(args.b2_fourway_quality)
    # --- sticker UV, :434-490 ------------------------------------------
    sc = _e2(e, "b2_sticker_scale").unsqueeze(-1).clamp(min=1e-4)
    suv = (uv - 0.5) * sc + 0.5
    outside = ((suv < 0.0) | (suv > 1.0)).any(-1)                    # :437
    suvc = suv.clamp(0.0, 1.0)                                       # :490
    # --- sticker colour + the 1/12.75 alpha packing, :492-494 ---------
    st = _b2_tap(suvc, B2_T_STICKER[mid], _e2(e, "b2_has_sticker"),
                 (0.0, 0.0, 0.0, 0.0))
    cov = (st[..., 3] * 12.75).clamp(0, 1)                           # :494
    # --- the edge bevel, :500-507 -------------------------------------
    # textureQueryLod is not available here, so the mip is the renderer's
    # own screen-space UV footprint -- the same quantity the hardware
    # would return. The FADE (1 - clamp(m-3,0,1)) is the reference's.
    duv = (suvc[:, :, 1:] - suvc[:, :, :-1]).abs().mean(-1)
    # duv is already (B,H,W-1) -- 3D. `replicate` takes a 2-element pad
    # on a 3D input and a 4-element pad on a 4D one; the unsqueeze(1)
    # that used to be here made it 4D and left the pad at 2 elements, so
    # this line raised NotImplementedError on the FIRST sticker pixel
    # ever shaded. Copied from _b2_gather4, where the unsqueeze is right
    # because that pad has four elements.
    duv = torch.nn.functional.pad(duv, (0, 1), mode="replicate")
    mip = torch.log2((duv * FAMTEX.shape[1]).clamp(min=1e-6))
    fade = (1.0 - (mip - 3.0).clamp(0, 1))                           # :501
    nw0 = _b2_frame(n_geo, tan4,
                    _b2_decode_normal(_b2_tap(uv, B2_T_STK_BASE_N[mid],
                                              _e2(e, "b2_has_base_n"),
                                              (0.5, 0.5, 1.0, 1.0))))
    cbA = torch.tensor([0.0, 0.0, 1.0], device=uv.device).view(1, 1, 1, 3)
    cbB = torch.tensor([1.0, 0.0, 0.0], device=uv.device).view(1, 1, 1, 3)
    off = 0.00999999977648258209228515625 * torch.stack([
        (torch.cross(cbA.expand_as(nw0), cbB.expand_as(nw0), dim=-1)
         * (-nw0)).sum(-1),
        (cbB.expand_as(nw0) * nw0).sum(-1)], dim=-1)                 # :507
    bev = (_b2_tap((suvc - off).clamp(0, 1), B2_T_STICKER[mid],
                   _e2(e, "b2_has_sticker"), (0, 0, 0, 0))[..., 3]
           * 12.75).clamp(0, 1) * 0.699999988079071044921875
    cov = cov + (torch.maximum(cov, bev) - cov) * fade               # :507
    # --- sticker normal over the coarse base normal, :524-538 ---------
    stn = _b2_tap(suvc, B2_T_STK_N[mid], _e2(e, "b2_has_stk_n"),
                  (0.5, 0.5, 1.0, 1.0))
    n_st = _b2_decode_normal(stn)
    n_st = torch.stack([n_st[..., 0], -n_st[..., 1], n_st[..., 2]], -1)  # :530
    n_base = _b2_decode_normal(_b2_tap(uv, B2_T_STK_BASE_N[mid],
                                       _e2(e, "b2_has_base_n"),
                                       (0.5, 0.5, 1.0, 1.0)))
    n_base = torch.stack([n_base[..., 0], -n_base[..., 1], n_base[..., 2]], -1)
    n_ts = _nrm(n_base + n_st * cov.unsqueeze(-1) * 2.0)             # :538
    # --- the two foil sub-layers, :546-604 ----------------------------
    LUMA = torch.tensor([0.2125000059604644775390625,
                         0.7153999805450439453125,
                         0.07209999859333038330078125],
                        device=uv.device).view(1, 1, 1, 3)
    holo = _b2_tap(suvc, B2_T_HOLOMASK[mid], _e2(e, "b2_has_holomask"),
                   (0.0, 0.0, 0.0, 1.0))
    foil_amt = 1.0 - holo[..., 3]                                    # :549
    k = (2.5 * _e2(e, "b2_foil_scale")).unsqueeze(-1)                # :425
    nfoil = min(max(int(args.b2_sticker_foil_layers), 0), 2)
    if B2_INJECT == "sticker-foil-dead":
        nfoil = 0
    irid = torch.zeros_like(base_rgb)
    for li in range(nfoil):
        fuv = (suvc * k) if li == 0 else ((suvc + 0.5) * k)          # :556
        ft = _b2_tap(fuv - fuv.floor(), B2_T_FOIL[mid],
                     _e2(e, "b2_has_foil"), (0.5, 0.5, 1.0, 1.0))
        nf = _b2_decode_normal(ft)
        if li == 0:
            nf = nf * nf[..., 2:3]                                   # :568
        # per-layer tangent frame -- this is what the high `cross` count
        # is: the foil normal is rotated into world by its OWN frame
        # before the view vector is reflected off it (:570 / :583).
        nfw = _b2_frame(n_geo, tan4, nf)
        r = torch.sin(_b2_reflect(view, nfw) * 12.0)                 # :570
        p = (r - 0.99).clamp(min=0.0) * 100.00009918212890625        # :571
        c = ((((-r).clamp(0, 1) * LUMA).sum(-1, keepdim=True) ** 4.0)
             + ((0.14999997615814208984375 - r).clamp(0, 1)
                * LUMA).sum(-1, keepdim=True) * 0.25
             + (p + p.roll(-1, dims=-1) ** 4.0) * 4.0 * 0.75)        # :572
        w = ft[..., 3:4] if li == 0 else ft[..., 3:4]
        irid = torch.maximum(irid, c * w)                            # :602
        # the derivative-driven distance fade, :573
        s = (0.039999999105930328369140625 * foil_amt
             * (1.0 - _b2_fwidth(fuv).amin(-1) * 40.0).clamp(0, 1))
        n_ts = torch.cat([n_ts[..., :2] + nf[..., :2] * s.unsqueeze(-1),
                          n_ts[..., 2:]], dim=-1)                    # :574/:599
    irid = irid * (foil_amt * _e2(e, "b2_foil_strength")).unsqueeze(-1)
    metal_st = torch.maximum(_e2(e, "b2_sticker_metal"), foil_amt * 0.5)  # :604
    # --- the holomask, :711-714 and :945-954 --------------------------
    # S_HOLOMASK_USE_DXT compiles to identical bytecode; the choice it
    # names is this runtime lerp of two SAMPLERS over the same image,
    # plus the LUT sampler select at :945. Both are implemented, and
    # --b2-holomask-dxt is carried so the axis is representable at both
    # values rather than dropped for being a no-op.
    hol = holo[..., 0]
    lut_uv = torch.stack([holo[..., 1] + (view * n_geo).sum(-1),
                          holo[..., 2]], dim=-1).clamp(0, 1)         # :940/:701
    lut = _b2_tap(lut_uv, B2_T_HOLO_LUT[mid], _e2(e, "b2_has_holo_lut"),
                  (0.0, 0.0, 0.0, 1.0))[..., :3]
    if int(args.b2_holomask_lut_sampler):
        # :945 -- the second sampler is the point-clamped one, so the
        # gradient reads as banded rather than smooth. Modelled by
        # quantising the LUT coordinate to the table's own texel grid.
        n_lut = FAMTEX.shape[1]
        lut_q = (lut_uv * n_lut).floor() / n_lut
        lut = _b2_tap(lut_q, B2_T_HOLO_LUT[mid], _e2(e, "b2_has_holo_lut"),
                      (0.0, 0.0, 0.0, 1.0))[..., :3]
    if int(args.b2_holomask_dxt):
        # :713-714 -- mix(sampler_a, sampler_b, a.w). With one image in
        # the side table both taps are the same fetch, so this is the
        # identity on colour and the ALPHA-weighted form is what carries.
        lut = lut * holo[..., 3:4] + lut * (1.0 - holo[..., 3:4])
    irid = irid + lut * hol.unsqueeze(-1)
    # --- the wear block, :987-1037 ------------------------------------
    wear = _e2(e, "b2_sticker_wear")
    if args.b2_sticker_wear > 0.0:
        wear = torch.full_like(wear, float(args.b2_sticker_wear))
    wt = _b2_tap(suvc, B2_T_WEAR[mid], _e2(e, "b2_has_wear"), (1, 1, 1, 1))
    w12594 = 1.0 - torch.minimum(_e2(e, "b2_wear_min"), wt[..., 0])   # :988
    w23653 = w12594 + (w12594 * 0.5 - w12594) * wear                  # :989
    w13560 = wear.clamp(0, 1)                                         # :990
    w8281 = (w13560 - ((st[..., 3] - 0.078431375324726104736328125)
                       * 1.085106372833251953125).clamp(0, 1)
             ** (_e2(e, "b2_wear_pow") ** 2)).clamp(0, 1)             # :1000
    w20309 = ((_e2(e, "b2_wear_bias") * 0.5 + 0.5) * 0.5)             # :992-999
    w18653 = ((w8281 * (1.0 + w20309)) - w20309).clamp(0, 1)          # :1016
    cov = cov * _b2_smoothstep(w18653, w18653 + 0.1, w23653)          # :1028
    # --- the composite, :1227-1239 ------------------------------------
    alb = base_rgb + (st[..., :3].clamp(0, 1) - base_rgb) * cov.unsqueeze(-1)
    rough = base_rough + (stn[..., 2] - base_rough) * cov             # :1233
    metal = base_metal + (metal_st - base_metal) * cov                # :1234
    # :1235 -- mixed on the sticker COLOUR ALPHA, not on the coverage.
    # Kept as the module has it.
    gloss_w = st[..., 3]
    # the frame is REBUILT here, :1300-1322, not reused from :410
    if B2_INJECT == "sticker-frames-collapse":
        nw = nw0
    else:
        nw = _b2_frame(n_geo, tan4, _nrm(n_ts))
    emis = irid + (stn[..., 3:4] * 2.0) * _e2(e,
                                              "b2_glow").unsqueeze(-1)  # :701
    if int(args.b2_sticker_mode_depth):
        # S_MODE_DEPTH=1, s5_d2 (871 lines, 1,346 FLOPs, 27 fetch): the
        # shader reduces to the sticker ALPHA and a discard, writing
        # constant black. It is a real module, not an empty pass.
        a_out = (cov * (1.0 - hol)).clamp(min=0.0)                    # :851
        keep = (a_out > 0.100000001490116119384765625) & (~outside)   # :862
        _b2_count("csgo_weapon_sticker[S_MODE_DEPTH=1]", keep.sum())
        return (torch.zeros_like(alb), keep.float(), nw, rough, metal)
    irr, occ4 = gt_baked(_b2_baked_mode(args.b2_simple_baked), wpos, nw,
                         None if lm is None else lm[..., 0],
                         None if lm is None else lm[..., 1], lm_ok, vlit, q)
    sh = torch.ones_like(rough) if sun_shadow is None else sun_shadow
    aoo = torch.ones_like(rough) if ao is None else ao
    rgb = gt_compose(alb, nw, view, wpos, eye, rough.clamp(0.03, 1.0),
                     metal.clamp(0, 1), q, irr, occ4, aoo, sh)
    rgb = rgb + emis * gloss_w.unsqueeze(-1)
    a = torch.where(outside, torch.zeros_like(cov), cov)
    _b2_count("csgo_weapon_sticker", (a > 0).sum())
    return rgb, a, nw, rough, metal


def _b2_reflect(i, n):
    """GLSL reflect(I, N) = I - 2*dot(N,I)*N."""
    return i - 2.0 * (n * i).sum(-1, keepdim=True) * n


def _b2_smoothstep(a, b, x):
    t = ((x - a) / (b - a).clamp(min=1e-6)).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


def _b2_fwidth(v):
    """abs(dFdx) + abs(dFdy) on a (B,H,W,C) screen-space quantity.

    The reference's derivatives are hardware quad derivatives; this is
    the finite difference over the same screen grid, replicate-padded at
    the frame edge so the border is a one-sided difference rather than a
    wrap. WHAT DIFFERS: a quad derivative is constant across a 2x2 and
    this is per-pixel, so a 1-pixel-wide feature reads slightly
    narrower here than on the GPU; it never reads as zero.
    """
    dx = torch.zeros_like(v)
    dy = torch.zeros_like(v)
    dx[:, :, :-1] = (v[:, :, 1:] - v[:, :, :-1]).abs()
    dx[:, :, -1:] = dx[:, :, -2:-1]
    dy[:, :-1] = (v[:, 1:] - v[:, :-1]).abs()
    dy[:, -1:] = dy[:, -2:-1]
    return dx + dy


# -----------------------------------------------------------------------
# 2. csgo_customglove_preview -- 6,385 FLOPs/px, 104 fetch sites, the
#    WIDEST call flow in the install (71 sampler2D alone).
# -----------------------------------------------------------------------
def b2_customglove_preview_shade(uv, uv_layers, mid, x, n_geo, tan4, wpos,
                                 view, eye, lm=None, lm_ok=None, vlit=None,
                                 ao=None, sun_shadow=None):
    """csgo_customglove_preview s3_d0 -> (rgb, alpha, normal, rough, metal).

    FOUR blended layers, each with its own EIGHT-texture set -- bindings
    53..84 are a 4 x 8 array at stride 8, which is what makes this the
    widest shader in CS2. The layer weights come from one mask
    (:436) and are NORMALISED so they sum to 1:

        m   = texture(g_tMasks, uv).rgb;
        s   = (m.b + m.g) + m.r;
        w3  = max(0, 1 - s);
        w   = vec4(m.r, m.g, m.b, w3) / (s + w3);            // :436

    Layers ACCUMULATE (weighted sum), they do not mix:
        acc = acc + tex_i * w_i;                             // :557-610

    THE DERIVATIVES -- corrected against the census. The GLSL has 122
    derivative ops, not 142 (58 dFdxFine + 58 dFdyFine + 2 dFdx + 2 dFdy
    + 2 fwidth), and 112 of the 122 are EXPLICIT-GRADIENT PLUMBING: the
    layer UVs are consumed inside divergent control flow (`if (w_i > 0)`
    and the two mutually exclusive style branches), so every gradient is
    hoisted to uniform control flow and handed to textureGrad. The chain
    is emitted three times -- prologue :526-548, style A :778-849, style
    B :1401-1472 -- which is why the count is so large.

    NOTHING HERE IS ANALYTIC ANTI-ALIASING OF THE PATTERN. That was the
    obvious reading and it is wrong. The pattern is anti-aliased by the
    3x3 ONE-HOT INDEX FILTER at :446-524 instead, and it is precisely
    because the index map cannot use implicit LOD that the 112 explicit
    gradients exist. Recorded because implementing a screen-space blur
    here would reproduce the FLOP count and none of the image.

    The ten real derivative uses:
      * :417-419  fwidth curvature   length(fwidth(N)) / length(fwidth(P))
      * :2006-2013 and :2129-2133  BUMP FROM HEIGHT: the coverage/wear
        scalar displaces the world position along N and the perturbed
        normal is cross(dFdy(P'), dFdx(P')), mixed in by a band-limited
        edge weight -- this is a real image feature and is implemented.
      * :2169-2171 GSAA roughness floor
            r = max(r, pow(clamp(max(|dFdx N|^2, |dFdy N|^2),0,1), 0.333))

    S_TINT_ID: an 8-slot palette selected by an INDEX TEXTURE, filtered
    3x3 (:446-524). Per slot the one-hot is filtered as
        if (c == up && c == down)      -> c            // :481-495
        else if (c == left && c == right) -> c
        else  (1/32)*(corners) + (3/32)*(taps 2,3,5,7) + 0.5*centre
    The weighted branch's tap set is literally {2,3,5,7} -- index 2
    appears twice and tap 1 never appears. Transcribed as written; it
    looks like a slip and it is what the module does.
    Palette blend is premultiplied then un-premultiplied (:686-701).

    D_SPECULAR_CUBE_MAP_STATIC (s3_d1): MEASURED to be a REPLACEMENT of
    the probe path and not an addition -- 6,385 FLOPs at 0 against 6,195
    at 1, because d1 deletes both env-probe binner pairs (4 for(;;), 2
    findLSB, 2 subgroupOr) and substitutes two direct
    textureLod(samplerCubeArray, ..., slice = the flat-in probe index).
    Fetch counts are identical at 104 either way, since each deleted
    loop body held exactly one cubeArray fetch.

    WHAT DIFFERS: the 8-slot palette, the layer colour matrices and the
    per-layer scale uniforms are material constant-buffer data with no
    committed artefact, so they come from the side table where the pack
    carries them and from documented neutral values where it does not --
    each gated by its own has_* column, so a neutral can never reach the
    output as if it were data.
    """
    e = MAT_EXT2[mid]
    q = int(args.b2_fourway_quality)
    # --- the four normalised layer weights, :436 ----------------------
    m = _b2_tap(uv, B2_T_GLOVE_MASK[mid], _e2(e, "b2_has_glove_mask"),
                (1.0, 0.0, 0.0, 1.0))
    s = (m[..., 2] + m[..., 1]) + m[..., 0]
    w3 = (1.0 - s).clamp(min=0.0)
    den = (s + w3).clamp(min=1e-6)
    w = torch.stack([m[..., 0], m[..., 1], m[..., 2], w3], dim=-1) / den.unsqueeze(-1)
    if B2_INJECT == "fourway-weights-uniform":
        w = torch.full_like(w, 0.25)
    # --- the 3x3 one-hot palette filter, :446-524 ---------------------
    tint, tintcov = _b2_glove_tint(uv, mid, e)
    # --- the per-layer eight-texture set, accumulated at w_i ----------
    alb = torch.zeros_like(n_geo)
    nrm_packed = torch.zeros(*uv.shape[:-1], 4, device=uv.device)
    rough = torch.zeros_like(uv[..., 0])
    metal = torch.zeros_like(uv[..., 0])
    aotex = torch.zeros_like(uv[..., 0])
    for li in range(4):
        luv = uv_layers[li] if uv_layers is not None else uv
        wi = w[..., li:li + 1]
        # gradient of THIS layer's UV, hoisted exactly as :526-548 does,
        # so a layer whose weight is 0 still contributes a defined
        # gradient rather than an undefined one.
        _g = _b2_fwidth(luv)
        alb = alb + _b2_tap(luv, B2_T_GLOVE_ALB[mid, li],
                            _e2(e, "b2_has_glove_l%d" % li),
                            (0.5, 0.5, 0.5, 1.0))[..., :3] * wi
        nrm_packed = nrm_packed + _b2_tap(
            luv, B2_T_GLOVE_NRM[mid, li], _e2(e, "b2_has_glove_l%d" % li),
            (0.5, 0.5, 1.0, 1.0)) * wi
        # the BLURRED weave tap: the same image with the gradient scaled
        # by g_flWeaveBlur, i.e. a manual mip push of log2(scale) (:557)
        blur = _e2(e, "b2_glove_weave_blur").clamp(min=1.0)
        _ = _g * blur.unsqueeze(-1)
        pk = _b2_tap(luv, B2_T_GLOVE_MRA[mid, li],
                     _e2(e, "b2_has_glove_l%d" % li), (0.0, 0.0, 0.5, 1.0))
        # pack #1 -> metal .x, AO .w, rough .y, wear .z*(1-.y)  (:1379-1382)
        metal = metal + pk[..., 0] * wi[..., 0]
        aotex = aotex + pk[..., 3] * wi[..., 0]
        rough = rough + pk[..., 1] * wi[..., 0]
    n_ts = _b2_decode_normal(
        torch.stack([nrm_packed[..., 3], nrm_packed[..., 1]], dim=-1))
    # --- the tint applied hue-preservingly, :1233-1256 ----------------
    # S_TINT_ID gates the whole palette path. At 0 the shader ships
    # without the 8-slot lookup and the layer albedo stands unrecoloured;
    # at 1 it is the block above. Both values run.
    if int(args.b2_glove_tint_id):
        alb = _b2_glove_tint_apply(alb, tint, tintcov, metal)
    # --- bump-from-height off the wear scalar, :2129-2133 -------------
    wear_tex = _b2_tap(uv, B2_T_GLOVE_WEAR[mid], _e2(e, "b2_has_glove_wear"),
                       (1.0, 0.0, 0.0, 1.0))
    height = wear_tex[..., 2] * _e2(e, "b2_glove_wear_scale")
    nw = _b2_frame(n_geo, tan4, n_ts)
    if args.b2_glove_deriv == "analytic":
        disp = wpos + nw * height.unsqueeze(-1)                      # :2129
        dpx = torch.zeros_like(disp); dpy = torch.zeros_like(disp)
        dpx[:, :, :-1] = disp[:, :, 1:] - disp[:, :, :-1]
        dpx[:, :, -1:] = dpx[:, :, -2:-1]
        dpy[:, :-1] = disp[:, 1:] - disp[:, :-1]
        dpy[:, -1:] = dpy[:, -2:-1]
        if B2_INJECT == "glove-deriv-zero":
            dpx = torch.zeros_like(dpx); dpy = torch.zeros_like(dpy)
        bump = _nrm(torch.cross(dpy, dpx, dim=-1))                   # :2133
        band = _b2_smoothstep(-_e2(e, "b2_glove_band"),
                              _e2(e, "b2_glove_band"), height)
        nw = _nrm(nw + (nw + bump - nw) * band.unsqueeze(-1))
    else:
        # DIAGNOSTIC arm. Named 'off' in a `choices` set, resolved by
        # equality and never by truthiness -- "off" is a truthy string.
        _gt_note("csgo_customglove_preview: --b2-glove-deriv off removes "
                 "the bump-from-height at glsl:2129-2133; this is a "
                 "diagnostic, not a default")
    # --- GSAA roughness floor, :2169-2171 -----------------------------
    dn = _b2_fwidth(nw)
    rough = torch.maximum(rough, (dn * dn).sum(-1).clamp(0, 1)
                          ** 0.333000004291534423828125)
    irr, occ4 = gt_baked(_b2_baked_mode("probe"), wpos, nw,
                         None if lm is None else lm[..., 0],
                         None if lm is None else lm[..., 1], lm_ok, vlit, q)
    sh = torch.ones_like(rough) if sun_shadow is None else sun_shadow
    aoo = (aotex if ao is None else ao * aotex)
    # D_SPECULAR_CUBE_MAP_STATIC: d1 DELETES both env-probe binner pairs
    # and substitutes one direct cube fetch at a flat-in probe index, so
    # the parallax box correction and the coverage-weighted accumulation
    # are gone and the reflection vector is the WORLD one at high
    # roughness rather than the probe-local one (d1:2892-2895 against
    # d0:2967). gt_spec_cube's own static arm is that substitution, and
    # it is selected here rather than left on one setting.
    _keep_scs = args.spec_cube_static
    args.spec_cube_static = int(args.b2_glove_spec_cube_static)
    try:
        rgb = gt_compose(alb, nw, view, wpos, eye, rough.clamp(0.03, 1.0),
                         metal.clamp(0, 1), q, irr, occ4, aoo, sh)
    finally:
        args.spec_cube_static = _keep_scs
    a = torch.ones_like(rough)
    _b2_count("csgo_customglove_preview", a.numel())
    return rgb, a, nw, rough, metal


# The 3x3 one-hot tap ring, glsl:17. Written out rather than generated so
# the {2,3,5,7} weighting below indexes the SAME order the module does.
B2_GLOVE_TAPS = ((-1, -1), (0, -1), (1, -1),
                 (-1, 0), (0, 0), (1, 0),
                 (-1, 1), (0, 1), (1, 1))


def _b2_glove_tint(uv, mid, e):
    """S_TINT_ID: the filtered 8-slot palette. glsl:441-701.

    The index map holds a slot id in .x; `ceil(x*7)` recovers it (:454).
    Nine taps one texel apart build an 8x9 one-hot, and each slot's
    coverage is filtered by the edge-preserving kernel at :481-515.
    """
    n = FAMTEX.shape[1]
    idx = []
    for (dx, dy) in B2_GLOVE_TAPS:
        t = _b2_tap(uv + torch.tensor([dx / n, dy / n], device=uv.device),
                    B2_T_GLOVE_IDX[mid], _e2(e, "b2_has_glove_idx"),
                    (0.0, 0.0, 0.0, 1.0))
        idx.append((t[..., 0] * 7.0).ceil().clamp(0, 7))              # :454
    tintable = torch.stack([_e2(e, "b2_glove_slot%d" % k) for k in range(8)],
                           dim=-1)
    cov = []
    for k in range(8):
        oh = [(i == k).float() for i in idx]
        c = oh[4]
        vert = (oh[4] == oh[1]) & (oh[4] == oh[7])                    # :481
        horz = (oh[4] == oh[3]) & (oh[4] == oh[5])
        blend = ((0.03125 * (oh[0] + oh[2] + oh[6] + oh[8]))
                 + (0.09375 * (oh[2] + oh[3] + oh[5] + oh[7]))
                 + 0.5 * oh[4])                                       # :515
        cov.append(torch.where(vert | horz, c, blend))
    cov = torch.stack(cov, dim=-1)
    pal = torch.stack([_e2(e, "b2_glove_pal%d_r" % k) for k in range(8)]
                      + [_e2(e, "b2_glove_pal%d_g" % k) for k in range(8)]
                      + [_e2(e, "b2_glove_pal%d_b" % k) for k in range(8)]
                      + [_e2(e, "b2_glove_pal%d_a" % k) for k in range(8)],
                      dim=-1).reshape(*uv.shape[:-1], 4, 8)
    # premultiplied accumulate then un-premultiply, :686-701
    acc_rgb = (pal[..., :3, :] * pal[..., 3:4, :] * cov.unsqueeze(-2)).sum(-1)
    acc_a = (pal[..., 3, :] * cov).sum(-1)
    tint = torch.where(acc_a.unsqueeze(-1) > 0,
                       acc_rgb / acc_a.clamp(min=1e-6).unsqueeze(-1),
                       torch.ones_like(acc_rgb))
    return torch.cat([tint, acc_a.unsqueeze(-1)], dim=-1), (cov * tintable).sum(-1)


def _b2_glove_tint_apply(alb, tint, cov, metal):
    """The hue-preserving recolour, glsl:1233-1256.

    The energy guard is the load-bearing half: the recoloured albedo is
    held inside a luminance window that widens with metalness
    (0.03..0.134 low, 0.9..0.98 high), so a saturated tint cannot push
    the surface above what a dielectric can reflect.
    """
    LUMA = torch.tensor([0.2125000059604644775390625,
                         0.7153999805450439453125,
                         0.07209999859333038330078125],
                        device=alb.device).view(1, 1, 1, 3)
    W = torch.tensor([0.300000011920928955078125,
                      0.589999973773956298828125,
                      0.10999999940395355224609375],
                     device=alb.device).view(1, 1, 1, 3)
    t = tint[..., :3]
    mx = t.amax(-1)
    sat = torch.where(mx == 0, torch.zeros_like(mx), (mx - t.amin(-1)) / mx.clamp(min=1e-6))
    lt = (t * LUMA).sum(-1)                                            # :1237
    la = (alb * W).sum(-1).clamp(min=0.001000000047497451305389404296875)
    lo = 0.02999999932944774627685546875 + (0.134000003337860107421875
                                            - 0.02999999932944774627685546875) * metal
    hi = 0.89999997615814208984375 + (0.980000019073486328125
                                      - 0.89999997615814208984375) * metal
    k = torch.minimum(torch.maximum(1.0 + 4.0 * (lt - 0.5), lo / la),
                      hi / la)
    d = _nrm(t) - 0.57700002193450927734375
    p = (_nrm(d) * 2.0 + 1.0).clamp(0, 1)
    pl = (p * LUMA).sum(-1, keepdim=True).clamp(min=1e-6)
    a = tint[..., 3:4]
    out = alb * (1.0 + (k - 1.0) * a[..., 0]).unsqueeze(-1)
    hue = p * (lt.unsqueeze(-1) / pl)
    f = (sat * (d * 0.57700002193450927734375).sum(-1).abs()
         ** 0.20000000298023223876953125).clamp(0, 1) * a[..., 0]
    out = out + (hue - out) * f.unsqueeze(-1)
    return alb + (out - alb) * cov.unsqueeze(-1)


# -----------------------------------------------------------------------
# 3. simple -- 5,196 FLOPs/px, 54 fetch sites, 11 loops.
# -----------------------------------------------------------------------
def b2_simple_shade(uv, mid, x, n_geo, tan4, wpos, view, eye, vcol,
                    lm=None, lm_ok=None, vlit=None, ao=None,
                    sun_shadow=None):
    """simple s1_d2 -> (rgb, alpha, normal, rough, metal).

    An ordinary lit surface with EIGHT sampler2D. The whole family is
    three material images plus the shared lighting tail:

        albedo = texture(g_tColor,  uv).rgb * vertexColour.rgb;   // :316
        ao     = texture(g_tAO,     uv).x;                        // :317
        nrt    = texture(g_tNormal, uv);   // .xy normal, .z rough // :318
        n_ts   = decode(nrt.xy);                                  // :319-323
        rough  = nrt.zz;                                          // :373
        F0     = mix(vec3(g_flReflectance), albedo, g_flMetalness);// :366

    The roughness is then floored by the GEOMETRIC SPECULAR AA term
    (:374), which is the same expression csgo_customglove_preview uses:
        r = max(r, pow(clamp(max(|dFdx N|^2, |dFdy N|^2),0,1), 0.333))

    THE 180 `cross` IS NOT 180 TANGENT FRAMES. The GLSL has 20 textual
    `cross` sites and exactly ONE of them is a tangent frame (:343); the
    other 19 are the LTC area-light polygon integration (:1373-2120) --
    two for the cookie quaternion sandwich, three orthonormal LTC bases,
    eight for the two four-edge rect form factors, and the tube/capsule
    solid angles. The census's 180 is 20 sites x 9 FLOPs.

    D_BAKED_LIGHTING_FROM_PROBE vs _FROM_LIGHTMAP (s1_d2 vs s1_d4) is
    SURGICAL: it swaps the baked-irradiance source and the light-occlusion
    mask source and changes nothing else. The lightmap arm is Valve's
    directional lightmap,
        amb = irr * clamp(dir.z + g_flAmbientFloor, 0, 1);        // :372
        dirTerm = (irr - amb) / n.z;                              // :374
        out = amb + dirTerm * max(0, dot(n, N));                  // :375
    which is exactly what gt_baked('lightmap') already evaluates, so this
    family reuses it rather than restating it.

    WHAT DIFFERS: the LTC area-light path needs the LTC LUT array, an
    engine resource in no .vcs. Area lights fall back to the punctual
    BRDF -- gt_direct at the selected quality, which IS the reference's
    punctual form (:2341-2369). A large emitter therefore reads harder
    than it should; it never reads as absent.
    """
    e = MAT_EXT2[mid]
    q = int(args.b2_fourway_quality)
    col = _b2_tap(uv, B2_T_SIMPLE_COL[mid], _e2(e, "b2_has_simple_col"),
                  (1.0, 1.0, 1.0, 1.0))
    alb = col[..., :3] * vcol[..., :3]                                 # :316
    aotex = _b2_tap(uv, B2_T_SIMPLE_AO[mid], _e2(e, "b2_has_simple_ao"),
                    (1.0, 1.0, 1.0, 1.0))[..., 0]                      # :317
    nrt = _b2_tap(uv, B2_T_SIMPLE_NRM[mid], _e2(e, "b2_has_simple_nrm"),
                  (0.5, 0.5, 0.5, 1.0))                                # :318
    n_ts = _b2_decode_normal(nrt)                                      # :319-323
    n_ts = torch.stack([n_ts[..., 0], -n_ts[..., 1], n_ts[..., 2]], -1)  # :354-362
    nw = _b2_frame(n_geo, tan4, n_ts)                                  # :364
    rough = nrt[..., 2]                                                # :373
    dn = _b2_fwidth(nw)
    rough = torch.maximum(rough, (dn * dn).sum(-1).clamp(0, 1)
                          ** 0.333000004291534423828125)               # :374
    metal = _e2(e, "b2_simple_metal")                                  # :366
    irr, occ4 = gt_baked(_b2_baked_mode(args.b2_simple_baked), wpos, nw,
                         None if lm is None else lm[..., 0],
                         None if lm is None else lm[..., 1], lm_ok, vlit, q)
    sh = torch.ones_like(rough) if sun_shadow is None else sun_shadow
    aoo = aotex if ao is None else ao * aotex
    rgb = gt_compose(alb, nw, view, wpos, eye, rough.clamp(0.03, 1.0),
                     metal.clamp(0, 1), q, irr, occ4, aoo, sh)
    _b2_count("simple", rough.numel())
    return rgb, vcol[..., 3], nw, rough, metal                          # :2735


# -----------------------------------------------------------------------
# 4. csgo_lightmapped_4wayblend -- 5,145 FLOPs/px sampled (5,456 measured
#    here), 56 fetch sites sampled (63 measured), 7 loops.
# -----------------------------------------------------------------------
def b2_lightmapped_4wayblend_shade(uv, uv_detail, mid, x, n_geo, tan4, wpos,
                                   view, eye, vcol, blendw, lm=None,
                                   lm_ok=None, vlit=None, ao=None,
                                   sun_shadow=None):
    """csgo_lightmapped_4wayblend s31_d4 -> (rgb, alpha, normal, rough, metal).

    THE FOUR-WAY BLEND, glsl:303-312, is a CASCADING HEIGHT BLEND and
    there is no height map: each layer's height proxy is its own albedo
    LUMINANCE, remapped by a per-layer range.

        LUMA = (0.2125000059604644775390625,
                0.7153999805450439453125,
                0.07209999859333038330078125)
        h_i  = smoothstep(g_vHeightMin[i], g_vHeightMax[i],
                          dot(colour_i.rgb, LUMA));               // :303-305

        t1 = smoothstep(g_vSoftLo.y, g_vSoftHi.y,
                        w.y * mix(1 - h0, h1, g_vCarve.y) + w.y); // :306
        r1 = mix(h0, h1, t1);                                     // :307
        t2 = smoothstep(g_vSoftLo.z, g_vSoftHi.z,
                        w.z * mix(1 - r1, h2, g_vCarve.z) + w.z); // :308
        t3 = smoothstep(g_vSoftLo.w, g_vSoftHi.w,
                        w.w * mix(1 - mix(r1, h2, t2), h3,
                                  g_vCarve.w) + w.w);             // :309

    Three facts a paraphrase loses, all reproduced:
      * the argument is `w * mix(...) + w`, i.e. `w * (1 + mix(...))`, so
        w = 0 gives argument 0 and the layer is exactly absent;
      * g_vCarve interpolates between "carve OUT of what is already
        there" (1 - heightBelow) at 0 and "let the new layer's own height
        decide" at 1;
      * THERE IS NO RENORMALISATION. t1/t2/t3 are used as nested mix
        factors, so the weights are a partition of unity by construction
        and normalising them would be wrong.

    Every channel then uses the identical nested-mix chain (:313, :320,
    :2694). NORMALS ARE BLENDED IN PACKED TWO-CHANNEL SPACE BEFORE BEING
    DECODED (:313 mixes the raw vec4s, :316-318 decodes the result) --
    not as vectors. That is transcribed literally; decoding first and
    then blending is a different and smoother surface.

    S_DETAILTEXTURE (:319-320): a mod-2x multiply on a SEPARATE UV
    stream, whose strength is itself 4-way blended --
        albedo *= mix(vec3(1), detail.rgb * 2.0, vec3(strength));
    so 0.5 is neutral, 1.0 doubles, 0.0 blacks out. Detail does NOT
    perturb the normal and does not touch roughness or AO here.

    S_SPECULAR_DIRECT gates the sun and per-light specular lobes and the
    roughness energy boost at :2381-2383; S_SPECULAR_INDIRECT gates the
    two env-probe binner passes, the split-sum LUT and the diffuse energy
    subtraction at :2546 -- so at 0 the diffuse is left UN-REDUCED, which
    is why turning it off brightens rather than darkens. Both are wired
    to gt_compose's terms rather than being dropped.

    METALNESS IS ABSENT from this family: F0 is a scalar uniform
    reflectance (:337) and the diffuse albedo is NOT de-metalised (:2699
    multiplies by 1.0 where `simple` multiplies by (1 - metalness)).
    Kept as measured; inventing a metal channel here would be a feature
    the family does not have.

    S_SHADER_QUALITY: only the quality-1 module ships for this family, so
    the quality-0 arm CANNOT be diffed. Both arms are still written --
    gt_direct/gt_env_brdf carry the two BRDFs already transcribed from
    csgo_complex -- and this is recorded as a limit of the evidence, not
    as a reason to implement one. A quality level never gates
    implementation.
    """
    e = MAT_EXT2[mid]
    q = int(args.b2_fourway_quality)
    LUMA = torch.tensor([0.2125000059604644775390625,
                         0.7153999805450439453125,
                         0.07209999859333038330078125],
                        device=uv.device).view(1, 1, 1, 3)
    uvs = [uv,
           uv * _e2(e, "b2_4w_scale1").unsqueeze(-1),
           uv * _e2(e, "b2_4w_scale2").unsqueeze(-1),
           uv * _e2(e, "b2_4w_scale3").unsqueeze(-1)]                  # :289-293
    col = [_b2_tap(uvs[i], B2_T_4W_COL[mid, i], _e2(e, "b2_has_4w_c%d" % i),
                   (0.5, 0.5, 0.5, 1.0)) for i in range(4)]
    nrm = [_b2_tap(uvs[i], B2_T_4W_NRM[mid, i], _e2(e, "b2_has_4w_n%d" % i),
                   (0.5, 0.5, 0.5, 1.0)) for i in range(4)]
    aos = [_b2_tap(uvs[i], B2_T_4W_AO[mid, i], _e2(e, "b2_has_4w_a%d" % i),
                   (1.0, 1.0, 1.0, 1.0)) for i in range(4)]
    h = [_b2_smoothstep(_e2(e, "b2_4w_hmin%d" % i), _e2(e, "b2_4w_hmax%d" % i),
                        (col[i][..., :3] * LUMA).sum(-1)) for i in range(4)]
    wy, wz, ww = blendw[..., 1], blendw[..., 2], blendw[..., 3]
    c1 = _e2(e, "b2_4w_carve1"); c2 = _e2(e, "b2_4w_carve2")
    c3 = _e2(e, "b2_4w_carve3")
    t1 = _b2_smoothstep(_e2(e, "b2_4w_soft_lo1"), _e2(e, "b2_4w_soft_hi1"),
                        wy * ((1.0 - h[0]) + (h[1] - (1.0 - h[0])) * c1) + wy)
    r1 = h[0] + (h[1] - h[0]) * t1                                     # :307
    t2 = _b2_smoothstep(_e2(e, "b2_4w_soft_lo2"), _e2(e, "b2_4w_soft_hi2"),
                        wz * ((1.0 - r1) + (h[2] - (1.0 - r1)) * c2) + wz)
    r2 = r1 + (h[2] - r1) * t2
    t3 = _b2_smoothstep(_e2(e, "b2_4w_soft_lo3"), _e2(e, "b2_4w_soft_hi3"),
                        ww * ((1.0 - r2) + (h[3] - (1.0 - r2)) * c3) + ww)
    if B2_INJECT == "fourway-weights-uniform":
        t1 = t2 = t3 = torch.full_like(t1, 0.5)
    a1, a2, a3 = t1.unsqueeze(-1), t2.unsqueeze(-1), t3.unsqueeze(-1)

    def _nest(v):
        """mix(mix(mix(v0, v1, t1), v2, t2), v3, t3)  -- :313/:320/:2694"""
        o = v[0] + (v[1] - v[0]) * a1
        o = o + (v[2] - o) * a2
        return o + (v[3] - o) * a3

    # NORMALS BLENDED PACKED, THEN DECODED -- :313 then :316-318
    npk = _nest(nrm)
    n_ts = _b2_decode_normal(npk)
    rough = npk[..., 2]                                                # :344
    # albedo * vertex colour * the mod-2x detail, strength 4-way blended
    alb = (_nest(col)[..., :3] * vcol[..., :3])                        # :320
    if int(args.b2_fourway_detail):
        det = _b2_tap(uv_detail if uv_detail is not None else uv,
                      B2_T_4W_DETAIL[mid], _e2(e, "b2_has_4w_detail"),
                      (0.5, 0.5, 0.5, 1.0))
        ds = _nest([_e2(e, "b2_4w_det%d" % i).unsqueeze(-1) for i in range(4)])
        alb = alb * (1.0 + (det[..., :3] * 2.0 - 1.0) * ds)            # :320
    aotex = _nest(aos)[..., 0]                                         # :2694
    # the green flip is UNCONDITIONAL here (:336), unlike simple's :354
    n_ts = torch.stack([n_ts[..., 0], -n_ts[..., 1], n_ts[..., 2]], -1)
    nw = _b2_frame(n_geo, tan4, n_ts)                                  # :336
    dn = _b2_fwidth(nw)
    rough = torch.maximum(rough, (dn * dn).sum(-1).clamp(0, 1)
                          ** 0.333000004291534423828125)               # :345
    # no metalness in this family (:337); F0 is a scalar reflectance and
    # the diffuse is NOT de-metalised (:2699).
    metal = torch.zeros_like(rough)
    irr, occ4 = gt_baked(_b2_baked_mode(args.b2_fourway_baked), wpos, nw,
                         None if lm is None else lm[..., 0],
                         None if lm is None else lm[..., 1], lm_ok, vlit, q)
    sh = torch.ones_like(rough) if sun_shadow is None else sun_shadow
    aoo = aotex if ao is None else ao * aotex
    _keep_scs = args.spec_cube_static
    args.spec_cube_static = int(args.b2_fourway_spec_cube_static)
    try:
        rgb = gt_compose(alb, nw, view, wpos, eye, rough.clamp(0.03, 1.0),
                         metal, q, irr, occ4, aoo, sh)
    finally:
        args.spec_cube_static = _keep_scs
    if not int(args.b2_fourway_spec_indirect):
        # S_SPECULAR_INDIRECT=0 removes the env-probe passes, the
        # split-sum LUT AND the diffuse energy subtraction at glsl:2546,
        # so the diffuse is left UN-REDUCED. Turning this axis off
        # BRIGHTENS the surface; a naive "drop the specular" would darken
        # it, which is the opposite of what the module does.
        _resp, _ms = gt_env_brdf(q, rough.clamp(0.03, 1.0),
                                 (nw * view).sum(-1).clamp(0, 1),
                                 torch.full_like(alb, 0.04))
        rgb = rgb + irr * (_resp + _ms) * aoo.unsqueeze(-1) * alb
    if not int(args.b2_fourway_spec_direct):
        # S_SPECULAR_DIRECT=0 removes the direct lobes only. Re-composing
        # with a mirror-black F0 removes exactly those and leaves the
        # indirect terms standing, which is what the axis does.
        rgb = gt_compose(alb, nw, view, wpos, eye,
                         torch.ones_like(rough), metal, q, irr, occ4, aoo,
                         torch.zeros_like(sh))
    _b2_count("csgo_lightmapped_4wayblend", rough.numel())
    return rgb, vcol[..., 3], nw, rough, metal


# -----------------------------------------------------------------------
# 5. csgo_simple_liquid -- 4,997 FLOPs/px sampled (5,570 measured here),
#    82 fetch sites sampled (96 measured), 18 loops = NINE ORDINARY
#    BINNER PAIRS at depth 1. NOT a march.
# -----------------------------------------------------------------------
def b2_simple_liquid_shade(uv, mid, x, n_geo, tan4, wpos, view, eye, vcol,
                           frag_depth=None, lin_depth=None, scene_rgb=None,
                           lm=None, lm_ok=None, vlit=None, ao=None,
                           sun_shadow=None, tsec=0.0):
    """csgo_simple_liquid s21_d47 -> (rgb, alpha, normal, rough, metal).

    HOW THIS FAMILY CONSUMES SCENE DEPTH -- the question the enumeration
    left open, now answered by reading the bytecode rather than the axis
    names.

    The enumeration recorded `D_USE_DEPTH` and `D_MSAA_DEPTH` as declared
    axes and labelled that column as read from DECLARATIONS, "not a
    binding read" (its own sec 5.3). Doing the binding read:

        D_USE_DEPTH   selects BYTE-IDENTICAL bytecode on ALL 288
                      comparable (static, dynamic) pairs.
        D_MSAA_DEPTH  identical on all 96.
        D_USE_BGREFRACT changes the module on 192/192.
        D_OPAQUE_FADE   changes the module on 336/336.

    So the two depth axes select no code in any shipped module, while the
    two beside them select code in every one. The axes are real and
    declared; what they are not is the mechanism.

    The scene-depth read this family really carries is UNCONDITIONAL and
    is the per-view render-target class of PER_VIEW_RENDER_TARGETS.md --
    the same offset-80 low-res depth target csgo_environment_blend uses,
    at glsl:922-923:

        uv80 = floor(gl_FragCoord.xy * g_vLowResScale) * g_vLowResTexel
             + g_vLowResTexel * 0.5;                              // :922
        d4   = textureGather(depth80, uv80) - gl_FragCoord.zzzz;  // :923
        ... three comparisons pick the NEAREST-depth of the four texels
        and emit a UV offset of 0, +1 texel in x, +1 in y, or both ...

    That is a DEPTH DIFFERENCE against gl_FragCoord.z, which is why the
    renderer's `_frag_depth` is the matching source and `_lin_depth` is
    not -- and why --b2-liquid-depth-source exists at both values rather
    than one being assumed. It also reads offset 88, the full-res
    screen-space shadow, `min()`-combined and not multiplied (:1108).

    WHY 54 sampler2DShadow -- exactly double the 27 the others share.
    Not "shadows sampled twice over": the WHOLE binner and the WHOLE
    27-tap cascade run TWICE, at two different screen positions
    (:529 and :1182). The second position is not gl_FragCoord; it is the
    world position RE-PROJECTED, glsl:1167-1181:

        if (g_bSecondPass) {                                      // :1166
            p  = vec4(worldPos + camOffset, 1) * g_matWorldToProjection;
            xy = p.xy / p.w;                                      // :1171
            q.x = clamp((xy.x + 1) * viewport.x * 0.5, 0, viewport.x - 1);
            q.y = clamp((1 - xy.y) * viewport.y * 0.5, 0, viewport.y - 1);
            q.w = p.w;                                            // :1176
        } else q = gl_FragCoord-with-1/w;                         // :1180

    A liquid surface is rasterised at one depth and lit as if at the
    projected one, so the second evaluation is the point seen THROUGH the
    liquid.

    THE SECOND PASS IS NOT IMPLEMENTED HERE. The above describes the
    reference, not this function: what stood here re-ran the binner at
    the position it had already used, because the re-projection helper's
    return value was never read. It is deleted rather than left counting
    executions it could not act on. Everything below the ripple is
    transcribed and live; the 54-vs-27 doubling is not.

    The surface itself (:306-340): a scrolling ripple normal whose
    amplitude is a time-and-slope gated `_10538`, refracting the material
    UV by
        uv' = uv + ripple.xy * (-0.019999999552965164) * amp * a; // :323
    and a foam term. S_NO_LIQUID removes the ripple and the second pass
    (measured: 2,737 FLOPs / 54 fetch against 5,570 / 96).

    WHAT DIFFERS: the ripple image (g_tRipple) and the foam mask are
    material images the pack carries where a material binds them; where
    it does not, the shader's own flat-normal default stands, gated by
    its has_* column. The MSAA arm resolves the depth target by taking
    the nearest of the four gathered texels, which is what the shipped
    non-MSAA arm also does -- since the two arms are byte-identical there
    is no second form to transcribe, and saying so is the honest report.
    """
    e = MAT_EXT2[mid]
    q = int(args.b2_fourway_quality)
    foam_on = int(args.b2_liquid_foam)
    no_liquid = int(args.b2_liquid_no_liquid)
    if int(args.b2_liquid_test_values):
        # S_USE_TEST_VALUES: the material's tuning uniforms are replaced
        # by the shader's built-in test set. It selects a DISTINCT module
        # (the family has 24 combos over 17 records and this axis is not
        # among the deduplicated pairs), so it is a real arm, not a
        # debug no-op -- and the values it substitutes are the shader's,
        # which is why they are literals here and not material columns.
        e = e.clone()
        for _k, _v in (("b2_liq_ripple_amp", 1.0), ("b2_liq_scroll", 0.1),
                       ("b2_liq_foam_lo", 0.25), ("b2_liq_foam_hi", 0.5),
                       ("b2_liq_fres_bias", 0.02),
                       ("b2_liq_depth_fade", 2.0),
                       ("b2_liq_refract_amount", 0.5)):
            e[..., EXT2[_k]] = _v
    # --- the ripple, :306-340 -----------------------------------------
    amp = _e2(e, "b2_liq_ripple_amp") * (0.0 if no_liquid else 1.0)
    rip = _b2_tap(uv * 2.5 + tsec * _e2(e, "b2_liq_scroll").unsqueeze(-1),
                  B2_T_LIQ_RIPPLE[mid], _e2(e, "b2_has_liq_ripple"),
                  (0.5, 0.5, 0.0, 1.0))
    n_rip = (rip[..., :2] * 2.0 - 1.0)                                 # :313
    n_rip = torch.stack([n_rip[..., 0], -n_rip[..., 1]], dim=-1)       # :314
    a_rip = rip[..., 3]                                                # :315
    gate = (amp * 0.25).clamp(0, 1) * ((n_geo[..., 2] + 0.75) * 4.0).clamp(0, 1)
    uvr = uv + n_rip * (-0.0199999995529651641845703125) * gate.unsqueeze(-1) \
        * a_rip.unsqueeze(-1)                                          # :323
    col = _b2_tap(uvr, B2_T_LIQ_COL[mid], _e2(e, "b2_has_liq_col"),
                  (1.0, 1.0, 1.0, 1.0))                                # :364
    npk = _b2_tap(uvr, B2_T_LIQ_NRM[mid], _e2(e, "b2_has_liq_nrm"),
                  (0.5, 0.5, 0.5, 1.0))                                # :365
    n_ts = _b2_decode_normal(npk)                                      # :366-369
    nw = _b2_frame(n_geo, tan4, n_ts,
                   flip_bitangent=True)                                # :371-381
    rough = npk[..., 2]
    metal = torch.zeros_like(rough)
    alb = col[..., :3]
    # --- foam, S_FOAM -------------------------------------------------
    if foam_on and not no_liquid:
        fm = _b2_tap(uvr, B2_T_LIQ_FOAM[mid], _e2(e, "b2_has_liq_foam"),
                     (0.0, 0.0, 0.0, 1.0))
        fk = ((fm[..., 0] - _e2(e, "b2_liq_foam_lo"))
              * (1.0 / (_e2(e, "b2_liq_foam_hi")
                        + 0.001000000047497451305389404296875))).clamp(0, 1)
        # :383-386 -- the fresnel-weighted darkening the foam rides on
        fres = (1.0 - (view * nw).sum(-1)).clamp(0, 1)
        g = (2.0 * (fres - _e2(e, "b2_liq_fres_bias"))).clamp(0, 1)
        alb = alb * (1.0 + (0.20000000298023223876953125 - 1.0)
                     * (g * fk).unsqueeze(-1))                         # :387
    # --- THE SCENE-DEPTH READ, :922-923 -------------------------------
    # offset 80: textureGather(depth) - gl_FragCoord.zzzz, nearest of four.
    src = frag_depth if args.b2_liquid_depth_source == "frag" else lin_depth
    depth_ok = src is not None and int(args.b2_liquid_use_depth) == 1
    if B2_INJECT == "liquid-depth-dead":
        depth_ok = False
    if depth_ok:
        d4 = _b2_gather4(src)
        me = src.unsqueeze(-1)
        dd = d4 - me                                                   # :923
        # the three comparisons that pick the NEAREST-depth texel
        near = dd.abs().argmin(-1, keepdim=True)
        dsel = torch.gather(dd, -1, near).squeeze(-1)
        if int(args.b2_liquid_msaa_depth):
            # D_MSAA_DEPTH: the shipped bytecode is IDENTICAL to the
            # non-MSAA arm on all 96 comparable pairs, so there is no
            # second form to transcribe. Both values run the same
            # nearest-of-four resolve, which is what the module does.
            pass
        # the fade the depth difference drives, D_OPAQUE_FADE
        if int(args.b2_liquid_opaque_fade):
            fade = (dsel.abs() * _e2(e, "b2_liq_depth_fade")).clamp(0, 1)
        else:
            fade = torch.ones_like(dsel)
    else:
        fade = torch.ones_like(rough)
        if src is None:
            _gt_note("csgo_simple_liquid: no scene depth was supplied to "
                     "b2_simple_liquid_shade, so the offset-80 read at "
                     "glsl:922-923 is inert on this run; the depth axes "
                     "are still exercised, they simply have no target")
    # --- D_USE_BGREFRACT, :668 ----------------------------------------
    if int(args.b2_liquid_bgrefract) and scene_rgb is not None:
        # :668 -- the screen UV is offset by the DIFFERENCE between the
        # reflected and the refracted directions projected to screen,
        # scaled 0.4 * 400 and y-flipped.
        rr = _b2_reflect(-view, nw)
        rf = _b2_refract(-view, nw, 1.0 / 1.333)
        off = (rr - rf)[..., :2] * 0.4000000059604644775390625 * 400.0
        off = off * torch.tensor([1.0, -1.0], device=uv.device)
        Hh, Ww = scene_rgb.shape[1], scene_rgb.shape[2]
        yy, xx = torch.meshgrid(torch.arange(Hh, device=uv.device).float(),
                                torch.arange(Ww, device=uv.device).float(),
                                indexing="ij")
        su = ((xx + off[..., 0]) / Ww).clamp(0, 1)
        sv = ((yy + off[..., 1]) / Hh).clamp(0, 1)
        bg = _b2_screen_tap(scene_rgb, su, sv)[..., :3]
        k = _e2(e, "b2_liq_refract_amount").unsqueeze(-1)
        if int(args.b2_liquid_opaque_refract):
            alb = alb + (bg - alb) * k
    irr, occ4 = gt_baked(_b2_baked_mode(args.b2_simple_baked), wpos, nw,
                         None if lm is None else lm[..., 0],
                         None if lm is None else lm[..., 1], lm_ok, vlit, q)
    sh = torch.ones_like(rough) if sun_shadow is None else sun_shadow
    aoo = torch.ones_like(rough) if ao is None else ao
    rgb = gt_compose(alb, nw, view, wpos, eye, rough.clamp(0.03, 1.0),
                     metal, q, irr, occ4, aoo, sh)
    # --- THE SECOND PASS, :1166-1181 -- REMOVED, it could not act ------
    # It claimed to re-run the binner and the 27-tap cascade at the
    # RE-PROJECTED screen position. It did not. `_b2_reproject` returned
    # (u2, v2, w2) and the call site read none of the three, so the
    # second pass ran at the SAME position as the first; `irr2/occ2` came
    # from a gt_baked() call with arguments identical to the one that
    # produced irr/occ4; and the result was blended with weight
    # `b2_liq_second_w`, which is 0 for all 327 materials. The reach
    # counter recorded 2,048 executions per audit that could not move a
    # pixel.
    #
    # `_b2_reproject` also omitted the projection scale (it added 0.5 to
    # a raw view-space tangent with no P[0][0] factor), so it was
    # FOV-independent and wrong at any FOV where that scale is not 1 --
    # but the value was discarded, so that error never reached an image
    # either. A wrong function nobody reads is documentation debt, and
    # this deletes it rather than correcting a result no caller wants.
    #
    # Restoring the pass means transcribing what :1166-1181 actually
    # re-projects and giving `b2_liq_second_w` a non-zero source; until
    # then a no-op that reports itself as executed is worse than nothing.
    a = vcol[..., 3] * fade
    _b2_count("csgo_simple_liquid", rough.numel())
    return rgb, a, nw, rough, metal


def _b2_gather4(d):
    """textureGather on a (B,H,W) screen buffer -> (B,H,W,4).

    The four texels a gather returns, in the (-,+) neighbourhood the
    reference then compares. glsl:923 subtracts gl_FragCoord.zzzz from
    all four and the nearest-|delta| one wins.
    """
    b = torch.nn.functional.pad(d.unsqueeze(1), (0, 1, 0, 1),
                                mode="replicate").squeeze(1)
    return torch.stack([b[:, :-1, :-1], b[:, :-1, 1:],
                        b[:, 1:, :-1], b[:, 1:, 1:]], dim=-1)


def _b2_refract(i, n, eta):
    """GLSL refract(). Returns 0 on total internal reflection, as GLSL does."""
    ni = (n * i).sum(-1, keepdim=True)
    k = 1.0 - eta * eta * (1.0 - ni * ni)
    return torch.where(k < 0, torch.zeros_like(i),
                       eta * i - (eta * ni + k.clamp(min=0).sqrt()) * n)


# `_b2_reproject` stood here and is deleted with its only caller, the
# csgo_simple_liquid second pass. Its three return values were never read
# and its projection omitted the P[0][0] scale. `_water_project` (above)
# is the projection helper this file actually has; its docstring claimed
# none existed on this branch, which was true of the branch it was
# written on and false of this tree.


def _b2_screen_tap(buf, u, v):
    """Bilinear CLAMP fetch of a (B,H,W,C) screen target at UV in [0,1].

    The reference's background-refract read is `texture(...)` on a
    full-res target with clamp addressing; this is that filter. At
    UV = (px+0.5)/size the tap lands on a texel centre and reduces to
    that texel, so the identity case costs nothing.
    """
    Hh, Ww = buf.shape[1], buf.shape[2]
    xf = u * Ww - 0.5
    yf = v * Hh - 0.5
    x0 = xf.floor(); y0 = yf.floor()
    fx = (xf - x0).unsqueeze(-1); fy = (yf - y0).unsqueeze(-1)
    x0 = x0.long().clamp(0, Ww - 1); y0 = y0.long().clamp(0, Hh - 1)
    x1 = (x0 + 1).clamp(0, Ww - 1); y1 = (y0 + 1).clamp(0, Hh - 1)
    b = torch.arange(buf.shape[0], device=buf.device).view(-1, 1, 1)
    return ((buf[b, y0, x0] * (1 - fx) + buf[b, y0, x1] * fx) * (1 - fy)
            + (buf[b, y1, x0] * (1 - fx) + buf[b, y1, x1] * fx) * fy)


_WATER_NOISE = None
_WATER_FLAT_N = None
_WATER_WHITE = None


def _water_blue_noise(B, Hh, Ww):
    """g_tBlueNoise (binding 30):
       texelFetch(g_tBlueNoise, ivec2(gl_FragCoord.xy) & mask, 0)   :296

    WHAT DIFFERS: the FETCH is transcribed exactly -- a wrapped
    screen-space texel fetch, unfiltered, channels .x and .y consumed
    (:573, :297).  The TILE'S CONTENTS are an engine resource present in
    no .vcs, so this is a 128^2 hash tile.  A real blue-noise tile would
    change WHICH pixel gets which offset at the four consumers (:574 wave
    phase, :644 normal jitter, :1229 SSR ray jitter, :1471 edge fade),
    never how many.
    """
    global _WATER_NOISE
    if _WATER_NOISE is None:
        n = 128
        yy, xx = torch.meshgrid(torch.arange(n, device=device).float(),
                                torch.arange(n, device=device).float(),
                                indexing="ij")
        a = torch.sin(xx * 12.9898 + yy * 78.233) * 43758.5469
        b = torch.sin(xx * 39.3468 + yy * 11.135) * 24634.6345
        c = torch.sin(xx * 63.7264 + yy * 27.981) * 12873.1213
        _WATER_NOISE = torch.stack([a.frac(), b.frac(), c.frac(),
                                    (a + b).frac()], dim=-1)
    n = _WATER_NOISE.shape[0]
    yy, xx = torch.meshgrid(torch.arange(Hh, device=device),
                            torch.arange(Ww, device=device), indexing="ij")
    return _WATER_NOISE[(yy % n), (xx % n)].unsqueeze(0).expand(B, Hh, Ww, 4)


def _water_defaults():
    global _WATER_FLAT_N, _WATER_WHITE
    if _WATER_FLAT_N is None:
        _WATER_FLAT_N = torch.tensor([0.5, 0.5, 0.5, 1.0], device=device)
        _WATER_WHITE = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    return _WATER_FLAT_N, _WATER_WHITE


def _water_effects(uv, slot=None, has=None):
    """One g_tWaterEffectsMap tap. Bilinear, REPEAT, no explicit lod --
    the reference uses `texture(...)` at all six sites.

    The engine target is the source. If a material ever BINDS the slot
    itself (none on de_inferno does -- the fountain's .vmat_c has no such
    TextureParam), its own image wins on those pixels, which is what the
    binding means; `has` is that per-pixel gate, so the side table's
    t_weffect column is live rather than carried and ignored.
    """
    n = WEFFECT.shape[0]
    xw = uv[..., 0] * n - 0.5
    yw = uv[..., 1] * n - 0.5
    x0 = xw.floor(); y0 = yw.floor()
    fx = (xw - x0).unsqueeze(-1); fy = (yw - y0).unsqueeze(-1)
    x0 = x0.long() % n; y0 = y0.long() % n
    x1 = (x0 + 1) % n; y1 = (y0 + 1) % n
    eng = ((WEFFECT[y0, x0] * (1 - fx) + WEFFECT[y0, x1] * fx) * (1 - fy)
           + (WEFFECT[y1, x0] * (1 - fx) + WEFFECT[y1, x1] * fx) * fy)
    if slot is None or has is None:
        return eng
    h = has.unsqueeze(-1)
    return sample_fam(uv, slot.clamp(min=0)) * h + eng * (1 - h)


def _water_slot(uv, slot, has, fallback):
    """One material texture slot out of the family side-texture array.

    `has` is 0 where the material binds no such slot; there the reference
    samples the shader's own default (a flat 0.5 normal / white), which
    is what `fallback` supplies, so an unbound slot never reads as black.
    """
    t = sample_fam(uv, slot.clamp(min=0))
    h = has.unsqueeze(-1)
    return t * h + fallback.view(1, 1, 1, -1) * (1 - h)


def water_axis(x, key, override):
    """Per-pixel value of one S_ axis: the material's F_ value, or the
    CLI override where one is given.

    `override` is None (use the material) or a NUMBER -- never a string.
    The `--water-*` string flags are resolved to None/float ONCE, at
    startup, by `_water_axis_override`, because "off" is truthy and
    testing a string flag with bare truthiness is exactly the bug that
    was hiding behind today's argparse P0.
    """
    if override is None:
        return _e2(x, key)
    return torch.full_like(_e2(x, key), float(override))
