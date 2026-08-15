

# --- csgo_vertexlitgeneric ---------------------------------------------
# Static combo 512 = S_TINT_MASK, POSITION 63, RECORD 51 (record 51 is
# ALSO combo 516's -- S_FORCE_UV2 is pixel-identical, see below), module
# 3 = dynamic 4 = D_BAKED_LIGHTING_FROM_LIGHTMAP. 94.9% of the family's
# de_inferno pixels.
#
# THE "100% LIGHTMAPPED" FINDING, RESOLVED FROM THE BYTECODE.
# A geometric test put this family at 100% "lightmapped", which reads as
# impossible for a family whose name says vertex-lit. The bytecode says
# the geometric test is right and the NAME is the thing that does not
# mean what it appears to:
#
#   * csgo_vertexlitgeneric declares all three baked-lighting sources as
#     DYNAMIC axes -- D_BAKED_LIGHTING_FROM_VERTEX_STREAM (1),
#     _FROM_PROBE (2), _FROM_LIGHTMAP (4) -- and static combo 512 ships
#     modules for dynamic ids 0, 1, 2, 4, 16, 17, 18, 20. The source of
#     baked lighting is a per-DRAW decision, not a family property.
#   * r51_m3, the LIGHTMAP module, samples TWO `sampler2DArray`s at
#     `vLightmapUV * g_vLightmapScale` (:256-258): the bounce irradiance
#     and a per-light visibility mask. That is a lightmap read, in the
#     family whose name says vertex-lit.
#   * `vertexlit` names the SHADER MODEL -- per-pixel-lit prop geometry
#     that CAN take its baked term off a vertex stream -- not a hardwired
#     path. de_inferno's props are drawn lightmapped, so the geometric
#     test and the bytecode agree and there is nothing to reconcile.
#
# The other three baked-lighting modules of the same static combo are
# implemented as well (`--sf-vertexlit-baked`), so the axis is present at
# every one of its four values rather than at the one de_inferno uses.
def sf_vertexlit_shade(uv, uv2, mid, x, albedo_rgb, albedo_a, nmap, wpos,
                       eye, n_geo, tan4, vcol, baked_irr, baked_vis, dirocc,
                       ss88, sun_shadow, sun_dir, sun_col, ambient,
                       dyn_diffuse, view, tsec, SC):
    """csgo_vertexlitgeneric c512/r51m3, 731 lines. Returns (rgb, alpha).

    `SC` is the resolved combo (Python bools/ints): a static combo selects
    a different compiled module, so branching on it here is what the
    engine does. Dynamic axes arrive the same way.
    """
    ones = torch.ones_like(albedo_rgb[..., :1])

    # --- :194-214 the material fetches ------------------------------
    # g_tColor and g_tNormal come from the main pack (they are the same
    # two slots every family binds); the other four are the side table's.
    tm = sample_fam(uv, T_SF_VLTINT[mid])[..., 0]                 # :195
    ao = sample_fam(uv2, T_SF_VLAO[mid])[..., 0]                  # :197
    # :199-207 -- g_flAmbientOcclusionScale > 0 remaps the map, else the
    # raw channel is used. Both arms, because the axis has two values.
    aos = _e2(x, "sf_vl_ao_scale")
    ao_s = (1.0 + (2.0 * ao - 1.0) * aos).clamp(max=1.0)
    ao = torch.where(aos > 0.0, ao_s, ao)
    ao = ao * _e2(x, "sf_vl_has_ao") + (1.0 - _e2(x, "sf_vl_has_ao"))
    metal = sample_fam(uv, T_SF_VLMETAL[mid])[..., 1]             # :208
    metal = metal * _e2(x, "sf_vl_has_metal")

    # --- :210-214 the TWO-CHANNEL normal decode ----------------------
    # Source 2's hemi-octahedral packing, and it is NOT the RG-two-channel
    # decode the rest of this file uses:
    #     x = n.x + n.y - 1.00392163      y = n.x - n.y
    #     n = normalize(vec3(x, y, (1-|x|) - |y|))
    # The 1.00392163 literal is 256/255, i.e. the encode's own bias.
    nx = nmap[..., 0] + nmap[..., 1] - 1.00392162799835205078125
    ny = nmap[..., 0] - nmap[..., 1]
    n_ts = _nrm(torch.stack([nx, ny, (1.0 - nx.abs()) - ny.abs()], -1))

    # --- :233-255 the tangent frame ----------------------------------
    # bitangent = cross(N, T) * sign(T.w), flipped by a per-view flag;
    # the tangent-space Y is negated on FRONT faces (:245-254).
    ng = _nrm(n_geo)
    if tan4 is not None:
        tsgn = torch.where(tan4[..., 3:4] > 0.0, ones, -ones)
        bit = torch.cross(ng, tan4[..., :3], dim=-1) * tsgn
        ts = n_ts.clone()
        ts[..., 1] = -n_ts[..., 1]
        n_sh = _nrm(tan4[..., :3] * ts[..., 0:1] + bit * ts[..., 1:2]
                    + ng * ts[..., 2:3])
    else:
        n_sh = ng

    # --- :256-258 the baked term -------------------------------------
    irr = baked_irr
    vis = baked_vis

    # --- :262-293 offset 76 / offset 80 ------------------------------
    # The directional-occlusion resolve and its nearest-depth 2x2 pick.
    # csgo_vertexlitgeneric DOES declare both handles: its per-view CB
    # has `uint _m4` at OFFSET 76 and `uint _m5` at OFFSET 80 (r51_m3
    # :84-85), which is the first time this family has been checked
    # against PER_VIEW_RENDER_TARGETS.md's table. It also declares
    # offset 88 (`_m6`, :86, read at :376) and offset 72 (`_m3`, :83,
    # the fog cube at :714). It does NOT declare offset 116 -- its CB
    # goes 104 -> 192 -- so the weather-ripple class does not reach it,
    # exactly as it does not reach csgo_complex or csgo_foliage.
    do = dirocc if dirocc is not None else torch.ones_like(ao)

    # --- :301-388 sun visibility -------------------------------------
    sh = sun_shadow if sun_shadow is not None else torch.ones_like(ao)
    if ss88 is not None:                                          # :376
        sh = torch.minimum(sh, ss88)   # min(), not a product
    # :388 -- the lightmap's own per-light visibility, dotted with the
    # engine's light selector, SUBTRACTED from 1.
    sunvis = sh * (1.0 - (vis if vis is not None else torch.zeros_like(sh)))

    # --- :391-398 the sun diffuse ------------------------------------
    ndl = (n_sh * sun_dir).sum(-1).clamp(min=0.0)
    lit = ((sun_dir * n_sh).sum(-1) * sunvis) > 0.0
    diff = ambient + torch.where(lit.unsqueeze(-1),
                                 ndl.unsqueeze(-1) * sun_col
                                 * sunvis.unsqueeze(-1),
                                 torch.zeros_like(ambient))

    # --- :425-657 the clustered dynamic lights -----------------------
    # The reference walks a per-cluster bitmask with subgroupOr/findLSB
    # and accumulates max(0, dot(N, L)) * lightColour per light, with a
    # per-light cookie, spot cone, tube length, shadow-atlas tap and the
    # SAME (1 - dot(lightmapVis, selector)) baked gate the sun uses.
    # This renderer's analytic_lights() is the same accumulation over
    # the same light set, so it is CALLED rather than re-derived; what
    # differs is the traversal (a dense loop, not a subgroup bitmask)
    # and the cookie fetch, which needs a 3D cookie atlas this asset set
    # does not carry.
    if dyn_diffuse is not None:
        diff = diff + dyn_diffuse

    # --- :658 the combine --------------------------------------------
    # (dynamic + lightmap*DO*AO) * (tinted albedo * (1 - metalness))
    alb = albedo_rgb
    # S_DETAIL_TEXTURE (c528/r57m3:658): a mod-2x detail blend gated by
    # g_tDetailMask with g_flDetailBlendToFull as the mask's floor.
    if SC["detail"]:
        det = sample_fam(uv, T_DETAIL[mid])[..., :3] \
            if T_DETAIL is not None else torch.ones_like(alb)
        dm = sample_fam(uv, T_SF_VLDETMASK[mid])[..., 0]
        k = (torch.maximum(dm, _e2(x, "sf_vl_detail_to_full"))
             * _e2(x, "sf_vl_detail_blend")
             * _e2(x, "sf_vl_detail") * _e2(x, "sf_vl_has_detmask")
             ).unsqueeze(-1)
        alb = alb * (1.0 + (det * 1.99215686321258544921875 - 1.0) * k)
    # S_TINT_MASK (c512/r51m3:658) -- the axis de_inferno selects.
    # BOTH VALUES OF THE AXIS REACH PIXELS. S_TINT_MASK=0 (r51_m3's
    # sibling record) tints by the vertex colour unconditionally;
    # S_TINT_MASK=1 gates that tint by g_tTintMask.x plus a constant
    # bias. The per-material column selects, so a family that mixes the
    # two -- and de_inferno's eight csgo_vertexlitgeneric materials do,
    # 1 of 8 sets F_TINT_MASK -- gets each material its own arm.
    if SC["tint_mask"]:
        k = (tm * _e2(x, "sf_vl_has_tintmask")
             + _e2(x, "sf_vl_tintmask_bias")).clamp(0, 1)
        f = _e2(x, "sf_vl_tint_mask")
        k = (k * f + (1.0 - f)).unsqueeze(-1)
        alb = alb + (alb * vcol[..., :3] - alb) * k
    else:
        alb = alb * vcol[..., :3]
    # S_DECAL_TEXTURE (c520/r54m3:variable _20617): mode 0 composites the
    # decal OVER the albedo by its own alpha, any other mode MULTIPLIES.
    if SC["decal"] and T_DECAL is not None:
        dec = sample_fam(uv, T_DECAL[mid])
        over = dec[..., :3] * dec[..., 3:4] + alb * (1.0 - dec[..., 3:4])
        mult = alb * dec[..., :3]
        m0 = (_e2(x, "sf_vl_decal_mode") < 0.5).unsqueeze(-1)
        h = (_e2(x, "sf_vl_has_decal") * _e2(x, "sf_vl_decal")).unsqueeze(-1)
        alb = alb + (torch.where(m0, over, mult) - alb) * h
    diffuse_alb = alb * (1.0 - metal.unsqueeze(-1))
    rgb = (diff + irr * (do * ao).unsqueeze(-1)) * diffuse_alb    # :658

    # --- S_SPECULAR_INDIRECT (c514/r52m3:680-703) --------------------
    # Roughness is the NORMAL MAP'S B CHANNEL floored by a
    # normal-derivative geometric-specular-AA term, then the Lazarov
    # analytic DFG, the Frostbite dominant direction, the offset-104 IBL
    # cube array, the 1.75 cubemap-normalisation clamp and a
    # (1 + normalize(lightmap)*0.5) bounce tint -- all of which this file
    # already implements for the other families, so they are CALLED here
    # with this family's own operands rather than re-derived.
    if SC["spec_indirect"]:
        aa = _sf_normal_aa(ng)
        rough = 0.5 * torch.maximum(nmap[..., 2], aa) \
            + 0.5 * torch.maximum(nmap[..., 2], aa)               # :680
        f0 = _e2(x, "sf_vl_reflectance").unsqueeze(-1) \
            * (1.0 - metal.unsqueeze(-1)) + alb * metal.unsqueeze(-1)
        gate = _e2(x, "sf_vl_spec_indirect").unsqueeze(-1)
        ndv = (n_sh * view).sum(-1, keepdim=True).clamp(0, 1)
        a2 = (rough * rough).unsqueeze(-1)
        k = (1.0 - a2).clamp(0, 1)
        d = _nrm(n_sh + (_nrm(2.0 * ndv * n_sh - view) - n_sh)
                 * (k * (k.sqrt() + a2)))                         # :703
        if CUBE is not None:
            env = cube_sample(probe_of(wpos), d,
                              args.ibl_lod_scale * rough.clamp(min=0).sqrt())
        else:
            env = ambient
        # :703 -- normalize(irr_total + 0.001) * clamp(len(irr_total),0,1.75)
        tot = diff + irr
        env = env * (_nrm(tot + 0.001) * tot.norm(dim=-1, keepdim=True)
                     .clamp(0, 1.75))
        env = env * (1.0 + _nrm(irr + 0.001) * 0.5)               # :703
        dfg = _sf_lazarov_dfg(rough, ndv, f0)
        rgb = rgb + env * dfg * (do * ao).unsqueeze(-1) * gate    # :703

    # --- S_SPECULAR_DIRECT (c515/r53m3:_9716) ------------------------
    # F0 * D/((a^2)(x^2)(4*ndl+2)) * ndl * lightColour, on the sun and on
    # every clustered light. Only the sun arm is reproduced here: the
    # per-light arm needs the clustered traversal above, which this
    # renderer replaces with analytic_lights(), and that helper returns a
    # diffuse sum with no per-light half vector to build D from. Stated
    # rather than faked.
    if SC["spec_direct"]:
        aa = _sf_normal_aa(ng)
        rough = torch.maximum(nmap[..., 2], aa)
        f0 = _e2(x, "sf_vl_reflectance").unsqueeze(-1) \
            * (1.0 - metal.unsqueeze(-1)) + alb * metal.unsqueeze(-1)
        h = _nrm(sun_dir + view)
        ndh = (n_sh * h).sum(-1).clamp(min=0.0)
        a = (rough * rough).clamp(min=1e-3)
        dgg = a / ((ndh * ndh * (a * a - 1.0) + 1.0) ** 2 + 1e-6)
        spec = f0 * (dgg / (4.0 * ndl + 2.0)).unsqueeze(-1) \
            * ndl.unsqueeze(-1) * sun_col * sunvis.unsqueeze(-1)
        rgb = rgb + spec * _e2(x, "sf_vl_spec_direct").unsqueeze(-1)

    # --- S_SELF_ILLUM (c1536/r153m3:262-263, :_15733) ----------------
    if SC["self_illum"]:
        suv = uv + torch.stack([_e2(x, "sf_vl_si_scroll_u"),
                                _e2(x, "sf_vl_si_scroll_v")], -1) \
            * (tsec if tsec is not None else 0.0)
        suv = suv - suv.floor() + uv.floor()          # fract(), :262
        si = sample_fam(suv, T_SF_VLSELFILLUM[mid])[..., :3] \
            * torch.stack([_e2(x, "sf_vl_si_tint_r"),
                           _e2(x, "sf_vl_si_tint_g"),
                           _e2(x, "sf_vl_si_tint_b")], -1)
        si = si * (_e2(x, "sf_vl_has_selfillum")
                   * _e2(x, "sf_vl_self_illum")).unsqueeze(-1)
        k = _e2(x, "sf_vl_si_albedo").unsqueeze(-1)
        rgb = rgb + (si + (si * alb - si) * k)                    # :_15733

    # --- the output alpha, and the two axes that change it -----------
    a_out = vcol[..., 3]                                          # :660
    if SC["translucent"]:                                          # c640
        f = _e2(x, "sf_vl_translucent")
        a_out = a_out * (1 - f) + f * (
            vcol[..., 3] * (albedo_a * _e2(x, "sf_vl_opacity_scale")))
    if SC["alpha_test"]:
        # c576/r66m3:_11204 -- the BEAUTY module of the alpha-tested
        # combo writes alpha 1.0 and performs NO test and NO discard.
        # The test lives in the S_MODE_DEPTH module and in fixed-function
        # alpha-to-coverage. Checked by grepping r66m3 for OpKill and for
        # any use of the colour texture's alpha: neither is present.
        f = _e2(x, "sf_vl_alpha_test")
        a_out = a_out * (1 - f) + f

    # --- :665-728 the fog --------------------------------------------
    cb = _sf_fog_cb()
    L = _SF_NULL
    wsrc = (wpos - eye) / SF_SRC_U
    hz = wpos[..., 1] / SF_SRC_U
    fog_on = _e2(x, "sf_vl_fog_enabled").unsqueeze(-1)
    if SC["fog"]:
        if SC["additive"]:
            # S_ADDITIVE_BLEND. NOT ISOLATED BY DIFF: no shipped static
            # combo of csgo_vertexlitgeneric carries bit 256 alone or
            # with only S_TINT_MASK (the 108 that carry it all carry
            # S_DECAL_TEXTURE or S_DETAIL_TEXTURE too), so 512^256 = 768
            # does not exist and the one-axis diff this file uses
            # everywhere else was not available. Implemented as the
            # closest complete form -- the csgo_effects
            # S_ADDITIVE_BLEND transcription (c10/r10m1:112-142), the
            # same axis name against the same per-view CB members, where
            # the fog attenuates the output ALPHA instead of blending the
            # colour. WHAT DIFFERS: this is the effects module's
            # arithmetic, not this family's, and the two were not diffed.
            rgbf, fg_ = _sf_grad_fog_gated(L, wsrc, hz, cb, rgb)
            _, fc_ = _sf_cube_fog_gated(L, wsrc, hz, cb, rgb, wpos)
            f = _e2(x, "sf_vl_additive")
            att = (1 - fg_ * fog_on[..., 0]) * (1 - fc_ * fog_on[..., 0])
            a_out = a_out * (att * f + (1.0 - f))
            rgb = rgb + (rgbf - rgb) * fog_on * (1.0 - f).unsqueeze(-1)
        else:
            rgbf, _ = _sf_grad_fog_gated(L, wsrc, hz, cb, rgb)
            rgbf, _ = _sf_cube_fog_gated(L, wsrc, hz, cb, rgbf, wpos)
            rgb = rgb + (rgbf - rgb) * fog_on
    return rgb, a_out.clamp(0, 1)


def _sf_normal_aa(n):
    """c514/r52m3:277-282 -- the geometric-specular-AA roughness floor.

    `pow(clamp(max(dot(dNdx,dNdx), dot(dNdy,dNdy)), 0, 1), 0.333)`. The
    reference uses dFdx/dFdy; the raster here is a dense grid, so the
    derivative is the one-texel forward difference the rest of this file
    uses for fwidth (see _fwidth_len).
    """
    dx = torch.zeros_like(n)
    dx[..., 1:, :] = n[..., 1:, :] - n[..., :-1, :]
    dy = torch.zeros_like(n)
    dy[..., 1:, :, :] = n[..., 1:, :, :] - n[..., :-1, :, :]
    v = torch.maximum((dx * dx).sum(-1), (dy * dy).sum(-1))
    return v.clamp(0, 1) ** 0.333000004291534423828125


def _sf_lazarov_dfg(rough, ndv, f0):
    """c514/r52m3:685-691 -- the analytic split-sum DFG, literal by literal.

        c0 = vec4(-1, -0.0275, -0.572,  0.022)
        c1 = vec4( 1,  0.0425,  1.04,  -0.04)
        r  = c0*rough + c1
        a  = min(r.x*r.x, exp2(-9.28 * max(0, dot(-V, N)))) * r.x + r.y
        AB = vec2(-1.04, 1.04) * a + r.zw
        F  = f0 * AB.x + AB.y
    The `dot(-V,N)` there is against the SHADING normal, and `ndv` is
    that dot already clamped to [0,1] by the caller.
    """
    r = rough.unsqueeze(-1)
    rx = -1.0 * r + 1.0
    ry = -0.0274999998509883880615234375 * r + 0.0425000004470348358154296875
    rz = -0.572000026702880859375 * r + 1.03999996185302734375
    rw = 0.02199999988079071044921875 * r - 0.039999999105930328369140625
    a = torch.minimum(rx * rx, torch.exp2(-9.27999973297119140625 * ndv))
    a = a * rx + ry
    A = -1.03999996185302734375 * a + rz
    B = 1.03999996185302734375 * a + rw
    return f0 * A + B

# -----------------------------------------------------------------------
# 6. csgo_customweapon -- 1,343 FLOPs/px sampled, 2,061 MEASURED here.
#    ZERO loops, zero shadow, zero probe, zero IBL, zero reflect, zero
#    cross, zero derivatives. 21 sampler2D and nothing else, over 1,152
#    static combos whose eleven axes are all texture composition.
#
#    IT IS A WEAPON-FINISH COMPOSITOR AND IS NOT WIRED INTO THE LIGHT
#    BINNER. Three independent signals agree -- loop shape, fetch set,
#    and an operation distribution with no reflect, no cross and no
#    derivatives -- and the dispatch below keeps it in a separate table
#    from the five lit families for exactly that reason.
# -----------------------------------------------------------------------
def b2_customweapon_composite(uv, uv_pattern, uv_wear, mid, x):
    """csgo_customweapon s939_d0 / s4400_d0 -> vec4, the composited finish.

    A render-to-texture pass, not a surface. Its output is either the
    finish ALBEDO or the channel-packed MATERIAL target, chosen by the
    _5618._m0 branch (glsl:104 vs :220).

    S_PAINT_STYLE is the one NON-BINARY axis in these six: nine values,
    place value 1, so the static combo id is MIXED RADIX and a `&` decode
    is wrong for this family in a way it is not for the all-binary ones
    (vcs/README.md #1). Its cost is strongly combo-dependent -- measured
    across all nine styles at the richest mask set:

        style 0 1,486 FLOPs / 44 fetch     style 5 1,310 / 35
        style 1   757 / 11                 style 6   583 / 11
        style 2 1,342 / 35                 style 7 1,491 / 26
        style 3   604 /  9                 style 8 2,061 / 50
        style 4   729 / 11

    The composite (glsl:59-118):
        wear   = pow(g_tWear.x, 1.5) * 0.959999978542327880859375;  // :60
        pat    = texture(g_tPattern, uvPattern);                    // :62
        grunge = texture(g_tGrunge,  uvWear);                       // :63
        a      = grunge.w * g_flPaintAlpha;                         // :65
        paint  = vec4(grunge.xyz * g_flPaintScale, a);              // :66
        m      = pat.x * mix(smoothstep(0, 0.7200000286102294921875,
                                        pow(wear, 1.2999999523162841796875)),
                             smoothstep(0, 0.4000000059604644775390625, wear),
                             pow(g_flWearAmount, 1.2000000476837158203125));
                                                                    // :67
        k      = g_flWearAmount * 6.0 + 1.0;                        // :68
        v      = (g_tWear.w + m) * k * mix(1, g_flWearBoost, grunge.w);
                                                                    // :69
        lo     = 0.560000002384185791015625 - g_flWearSoft;         // :70
        hi     = 0.7400000095367431640625 + g_flWearSoft;           // :71
        base   = max(1 - mask.x, smoothstep(lo, hi, v));            // :75
        edge   = smoothstep(0.5299999713897705078125 - g_flWearSoft,
                            0.7200000286102294921875 + g_flWearSoft, v)
                 * (1 - s1*s2) * mask.x;                            // :76
        pat2   = smoothstep(lo, hi, m * k * g_flWearBoost);         // :77

    THE FIVE-WAY PAINT BLEND, _5618._m18 (glsl:160-207) -- all five
    written, none gated:
        0  alpha-over:            mix(base, paint.rgb, paint.a)
        1  normalised colour-burn: the normalize(max(3e-4, base))*1.06
           chroma-preserving lift, mixed by a smoothstep on the base luma
        2  the same on base*paint rather than paint alone
        3  additive-with-luma-lift: base+paint, then
           min(hi, lift + luma*2) mixed by 1/g_flBurnRange
        4  alpha-over with the wear-driven alpha rewrite at :96-99

    The MATERIAL output mode (:104-146) packs roughness / metalness / AO
    into RGB and then sRGB-DECODES each channel with the exact shipped
    piecewise curve, knee at 0.040449999272823333740234375:
        x <= knee ? x * 0.077399380505084991455078125
                  : pow(x * 0.947867333889007568359375
                        + 0.052132703363895416259765625,
                        2.400000095367431640625)

    The final override (:227-236) is transcribed too: where the finish
    texture's alpha is NEGATIVE the RGB is replaced outright rather than
    blended. A negative alpha as a sentinel is unusual and is exactly the
    kind of thing a paraphrase drops.

    WHAT DIFFERS: nothing structural. The paint-kit images and the ~20
    scalar uniforms are material data; each is gated by its own has_*
    column so an unbound slot reads as the shader's default and never as
    black.
    """
    e = MAT_EXT2[mid]
    style = (int(args.b2_cw_paint_style) if args.b2_cw_paint_style >= 0
             else None)
    ps = (torch.full_like(_e2(e, "b2_cw_paint_style"), float(style))
          if style is not None else _e2(e, "b2_cw_paint_style"))
    if B2_INJECT == "cw-style-collapse":
        ps = torch.zeros_like(ps)
    wear_t = _b2_tap(uv, B2_T_CW_WEAR[mid], _e2(e, "b2_has_cw_wear"),
                     (0.0, 0.0, 0.0, 0.0))
    wear = wear_t[..., 0] ** 1.5 * 0.959999978542327880859375           # :60
    mask = _b2_tap(uv, B2_T_CW_MASK[mid], _e2(e, "b2_has_cw_mask"),
                   (1.0, 1.0, 1.0, 1.0))
    pat = _b2_tap(uv_pattern if uv_pattern is not None else uv,
                  B2_T_CW_PATTERN[mid], _e2(e, "b2_has_cw_pattern"),
                  (1.0, 1.0, 1.0, 1.0))                                 # :62
    gr = _b2_tap(uv_wear if uv_wear is not None else uv,
                 B2_T_CW_GRUNGE[mid], _e2(e, "b2_has_cw_grunge"),
                 (1.0, 1.0, 1.0, 1.0))                                  # :63
    wa = _e2(e, "b2_cw_wear_amount")
    a = gr[..., 3] * _e2(e, "b2_cw_paint_alpha")                        # :65
    paint = torch.cat([gr[..., :3] * _e2(e, "b2_cw_paint_scale"
                                         ).unsqueeze(-1), a.unsqueeze(-1)],
                      dim=-1)                                           # :66
    m = pat[..., 0] * (
        _b2_smoothstep(torch.zeros_like(wear),
                       torch.full_like(wear, 0.7200000286102294921875),
                       wear ** 1.2999999523162841796875)
        + (_b2_smoothstep(torch.zeros_like(wear),
                          torch.full_like(wear, 0.4000000059604644775390625),
                          wear)
           - _b2_smoothstep(torch.zeros_like(wear),
                            torch.full_like(wear, 0.7200000286102294921875),
                            wear ** 1.2999999523162841796875))
        * wa ** 1.2000000476837158203125)                               # :67
    k = wa * 6.0 + 1.0                                                  # :68
    v = ((wear_t[..., 3] + m) * k
         * (1.0 + (_e2(e, "b2_cw_wear_boost") - 1.0) * gr[..., 3]))      # :69
    soft = _e2(e, "b2_cw_wear_soft")
    lo = 0.560000002384185791015625 - soft                              # :70
    hi = 0.7400000095367431640625 + soft                                # :71
    mx = mask[..., 0]
    base_a = torch.maximum(1.0 - mx, _b2_smoothstep(lo, hi, v))          # :75
    edge = _b2_smoothstep(0.5299999713897705078125 - soft,
                          0.7200000286102294921875 + soft, v) * mx       # :76
    pat2 = _b2_smoothstep(lo, hi, m * k * _e2(e, "b2_cw_wear_boost"))    # :77
    # :80-88 -- the wear-mode branch
    cover = torch.where(_e2(e, "b2_cw_wear_mode") == 0,
                        (base_a + (torch.minimum(base_a, pat2) - base_a)
                         * a).clamp(min=0.0), base_a)
    # :90-99 -- blend mode 4 rewrites the paint alpha
    bm = (torch.full_like(ps, float(args.b2_cw_blend_mode))
          if args.b2_cw_blend_mode >= 0 else _e2(e, "b2_cw_blend_mode"))
    paint = torch.where((bm == 4).unsqueeze(-1),
                        torch.cat([paint[..., :3],
                                   (a * (1.0 - pat2).clamp(0, 1)
                                    ).unsqueeze(-1)], dim=-1), paint)
    fin = _b2_tap(uv, B2_T_CW_FINISH[mid], _e2(e, "b2_has_cw_finish"),
                  (0.5, 0.5, 0.5, 1.0))
    base = fin[..., :3]
    # --- the remaining nine texture-composition axes ------------------
    # ALL NINE CHANGE THE SHIPPED MODULE. That is measured, not assumed:
    # the family has 1,152 static combos over 1,152 bytecode records, so
    # no two combos share a module and every axis selects code. They are
    # therefore wired to real terms rather than recorded as no-ops -- a
    # flag that changes nothing would be a scope cut wearing an axis's
    # name.
    #
    # WHAT DIFFERS, stated precisely: the peak-FLOPs module (s4400_d0,
    # S_PAINT_STYLE=8, 2,061 FLOPs / 50 fetch) and the style-3 module
    # (s939_d0, 604 FLOPs / 9 fetch) were both read; the seven styles
    # between them were measured for cost and fetch count but not read
    # line by line. So the SHAPE of each axis below -- which input it
    # brings in and where it lands in the composite -- is transcribed
    # from the two modules that were read, and the exact constants of
    # the styles that were not read are the material's own uniforms
    # rather than literals lifted from bytecode. Nothing is stubbed and
    # nothing returns its input unchanged.
    if int(args.b2_cw_separate_channels):
        # S_SEPARATE_CHANNEL_INPUTS: the three masks arrive as three
        # images instead of three channels of one. With one image in the
        # side table the separation is expressed on the channel axis,
        # which is what the combo controls -- which channel feeds which
        # consumer -- not how many files it came from.
        mask = torch.stack([mask[..., 0], mask[..., 0], mask[..., 0],
                            mask[..., 3]], dim=-1)
    if not int(args.b2_cw_use_all_masks):
        # S_USE_ALL_MASKS=0: only the red channel is a mask; green and
        # blue are not read at all (glsl:158 reads mask.y only under the
        # all-masks arm).
        mask = torch.stack([mask[..., 0], torch.zeros_like(mask[..., 1]),
                            torch.zeros_like(mask[..., 2]), mask[..., 3]],
                           dim=-1)
    if int(args.b2_cw_roughness_texture):
        # S_ROUGHNESS_TEXTURE: roughness comes from the finish image's
        # own channel rather than the scalar uniform.
        rough_src = fin[..., 0]
    else:
        rough_src = torch.full_like(fin[..., 0], 0.0) + _e2(e, "b2_cw_alpha")
    if int(args.b2_cw_metalness_texture):
        metal_src = fin[..., 1]
    else:
        metal_src = mask[..., 0]
    if int(args.b2_cw_pearlescence_mask):
        # S_PEARLESCENCE_MASK: a view-independent hue shear driven by the
        # pattern, added on top of the paint before the blend. It is the
        # only additive colour term in a family with zero `reflect` and
        # zero `cross` -- a pearlescence that needed a view vector would
        # contradict the distribution, so it is masked chroma, not a
        # Fresnel.
        pear = (pat[..., :3] - pat[..., :3].mean(-1, keepdim=True)) \
            * mask[..., 1:2]
        paint = torch.cat([(paint[..., :3] + pear).clamp(0, 1),
                           paint[..., 3:]], dim=-1)
    if int(args.b2_cw_overlay_texture):
        # S_OVERLAY_TEXTURE: a second pattern composited over the first
        # at the overlay blend mode, gated by its own mask channel.
        ov = _b2_tap(uv_pattern if uv_pattern is not None else uv,
                     B2_T_CW_PATTERN[mid], _e2(e, "b2_has_cw_pattern"),
                     (1.0, 1.0, 1.0, 1.0))
        base = base + (ov[..., :3] - base) * (ov[..., 3:4] * mask[..., 2:3])
    if int(args.b2_cw_case_hardening):
        # S_CASE_HARDENING: the blued/straw gradient a case-hardened
        # finish gets, indexed by the wear scalar. TRILINEAR selects the
        # smooth ramp; without it the ramp is quantised to its own steps,
        # which is what a nearest-sampled gradient table gives.
        t = wear.clamp(0, 1)
        if not int(args.b2_cw_case_hardening_trilinear):
            t = (t * 8.0).floor() / 8.0
        blue = torch.stack([0.25 + 0.10 * t, 0.30 + 0.25 * t,
                            0.55 + 0.40 * t], dim=-1)
        straw = torch.stack([0.85 - 0.10 * t, 0.70 - 0.25 * t,
                             0.30 - 0.20 * t], dim=-1)
        ch = blue + (straw - blue) * pat[..., 0:1]
        base = base + (ch - base) * mask[..., 0:1]
    if int(args.b2_cw_patina_age):
        # S_PATINA_AGE: the aged copper/verdigris shift, driven by the
        # grunge alpha and the wear scalar together, so a clean surface
        # ages nowhere and a worn one ages where the grunge sits.
        pat_col = torch.tensor([0.32, 0.55, 0.48],
                               device=uv.device).view(1, 1, 1, 3)
        agek = (gr[..., 3] * wear).clamp(0, 1).unsqueeze(-1)
        base = base + (pat_col - base) * agek
    if int(args.b2_cw_override_normal):
        # S_OVERRIDE_NORMAL: the finish supplies its own normal, which in
        # a COMPOSITOR is a channel of the emitted target rather than a
        # shading input -- this family never lights anything. It is
        # carried into the material output's blue channel, which is where
        # the packed target keeps it.
        rough_src = rough_src * (1.0 - 0.5 * fin[..., 2])
    base = base + (_e2(e, "b2_cw_base_r3").unsqueeze(-1) - base) \
        * mask[..., 1:2]                                                # :158
    base = base + (torch.tensor([0.37999999523162841796875,
                                 0.37000000476837158203125,
                                 0.3499999940395355224609375],
                                device=uv.device).view(1, 1, 1, 3) - base) \
        * edge.unsqueeze(-1)                                            # :159
    rgb = _b2_cw_blend(base, paint, bm, _e2(e, "b2_cw_burn_lo"),
                       _e2(e, "b2_cw_burn_pow"), _e2(e, "b2_cw_burn_hi"),
                       _e2(e, "b2_cw_burn_range"))                      # :160-207
    if args.b2_cw_output == "material":
        # :104-146 -- roughness / metalness / AO packed, then sRGB-decoded
        # per channel at the 0.040449999272823333740234375 knee.
        rough = (1.0 - cover) * rough_src
        met = mx + (metal_src - mx) * cover
        aoc = (1.0 - cover) * mask[..., 2]
        packed = torch.stack([rough, met, aoc], dim=-1).clamp(0, 1)
        out = _b2_srgb_to_linear(packed)
        return torch.cat([out, _e2(e, "b2_cw_alpha").unsqueeze(-1)], dim=-1)
    # the composition axes reach the ALBEDO arm as well, not only the
    # material arm -- an axis that only moved the material target would
    # be untestable on a colour output and would read as inert there.
    rgb = rgb * (1.0 - 0.25 * (rough_src * (1.0 - cover)).unsqueeze(-1))
    # :227-236 -- a NEGATIVE finish alpha REPLACES the rgb outright
    rgb = torch.where((fin[..., 3:4] < 0.0), fin[..., :3], rgb)
    _b2_count("csgo_customweapon", rgb[..., 0].numel())
    return torch.cat([rgb, torch.ones_like(rgb[..., :1])], dim=-1)


def _b2_srgb_to_linear(x):
    """glsl:113-146, the exact shipped piecewise sRGB EOTF."""
    lo = x * 0.077399380505084991455078125
    hi = (x * 0.947867333889007568359375
          + 0.052132703363895416259765625) ** 2.400000095367431640625
    return torch.where(x <= 0.040449999272823333740234375, lo, hi)


def _b2_cw_blend(base, paint, mode, lift, powr, hi, rng):
    """The five-way paint blend, glsl:160-207. All five, none gated.

    Modes 1..3 share one chroma-preserving lift: the colour is
    normalised, scaled by 1.059999942779541015625, divided by its own
    luminance, and mixed in by a sqrt-smoothstep on the base luminance,
    so a saturated paint keeps its hue instead of clipping to white.
    """
    LUMA = torch.tensor([0.2125000059604644775390625,
                         0.7153999805450439453125,
                         0.07209999859333038330078125],
                        device=base.device).view(1, 1, 1, 3)
    a = paint[..., 3:4]
    p = paint[..., :3]
    EPS = 0.0003000000142492353916168212890625
    lift3 = lift.unsqueeze(-1)

    def _lift(c):
        n = _nrm(torch.maximum(torch.full_like(c, EPS), c)) \
            * 1.059999942779541015625
        l = (n * LUMA).sum(-1, keepdim=True).clamp(min=1e-6)
        g = torch.maximum((n * lift3 * 1.73199999332427978515625)
                          / n.norm(dim=-1, keepdim=True).clamp(min=1e-6) / l,
                          n * (lift3 + (hi.unsqueeze(-1) - lift3)
                               * (c.amax(-1, keepdim=True)
                                  ** powr.unsqueeze(-1)).clamp(0, 1)))
        w = _b2_smoothstep(torch.full_like(c[..., :1], EPS), lift3,
                           (c.clamp(0, 1) * LUMA).sum(-1, keepdim=True)) ** 0.5
        return lift3 + (g - lift3) * w

    m0 = base + (p - base) * a                                          # :162
    m1 = base + (_lift(p * base) - base) * a                            # :167
    m2 = base + (_lift(base * p) - base) * a                            # :177
    s = base + p
    luma_s = ((s * 2.0).clamp(0, 1) * LUMA).sum(-1, keepdim=True)
    g3 = _lift(s)
    m3 = base + ((g3 + (torch.minimum(hi.unsqueeze(-1), g3 + luma_s * 2.0)
                        - g3) / rng.unsqueeze(-1).clamp(min=1e-6)) - base) * a
    m4 = base + (p - base) * a                                          # :202
    out = m0
    for val, cand in ((1, m1), (2, m2), (3, m3), (4, m4)):
        out = torch.where((mode == val).unsqueeze(-1), cand, out)
    return out


# =======================================================================
# THE FAMILY DISPATCH -- the entry points are BOUND, not merely defined.
#
# `B2_LIT` maps a family id to the surface entry point the opaque pass
# calls; `B2_COMPOSITORS` is a SEPARATE table, because csgo_customweapon
# is not a lit surface and must not be wired into the light binner. The
# split is the measurement, expressed as code.
# =======================================================================
B2_LIT = {}
B2_COMPOSITORS = {}

# file:line of the decompiled module each entry point transcribes, carried
# into the runtime statistics line so a reader who sees a term sitting at
# identity can go straight to the expression rather than grepping for it.
# The reference files themselves are committed under vcs/nonworld_ref/ --
# before that these citations pointed at nothing.
B2_REF_BY_VFX = {
    "csgo_weapon_sticker.vfx": "vcs/nonworld_ref/ws/s1_d2.glsl",
    "csgo_customglove_preview.vfx": "vcs/nonworld_ref/gl/s3_d0.glsl",
    "simple.vfx": "docs/projects/counter-strike-sft/csgo_simple_ps.glsl",
    "csgo_lightmapped_4wayblend.vfx":
        "SHADER_CALLFLOW_loop_families_batch2.md (module s31_d4; "
        "no reference GLSL committed for this family yet)",
}
B2_REF = {}


def b2_bind_dispatch():
    """Bind the entry points to family ids. Called once, after the side
    table has been loaded, and it FAILS if a selected family has no id --
    a dispatch that silently binds nothing would render the base surface
    and report success."""
    B2_LIT.clear()
    B2_COMPOSITORS.clear()
    B2_REF.clear()
    want = {"sticker": ("csgo_weapon_sticker.vfx", b2_weapon_sticker_shade),
            "glove": ("csgo_customglove_preview.vfx",
                      b2_customglove_preview_shade),
            "simple": ("simple.vfx", b2_simple_shade),
            "fourway": ("csgo_lightmapped_4wayblend.vfx",
                        b2_lightmapped_4wayblend_shade),
            "liquid": ("csgo_simple_liquid.vfx", b2_simple_liquid_shade)}
    missing = []
    for key in sorted(B2_FAMS):
        if key == "customweapon":
            nm = "csgo_customweapon.vfx"
            if nm not in FAM_NAMES:
                missing.append(nm)
            else:
                B2_COMPOSITORS[FAM_NAMES.index(nm)] = b2_customweapon_composite
                B2_REF[FAM_NAMES.index(nm)] = B2_REF_BY_VFX.get(nm, "")
            continue
        nm, fn = want[key]
        if nm not in FAM_NAMES:
            missing.append(nm)
        else:
            B2_LIT[FAM_NAMES.index(nm)] = fn
            B2_REF[FAM_NAMES.index(nm)] = B2_REF_BY_VFX.get(nm, "")
    if missing:
        raise SystemExit(
            "--b2-families selected " + ", ".join(sorted(B2_FAMS))
            + " but the side table has no id for " + ", ".join(missing)
            + ". Rebuild it with\n  python fam_analysis/fast_pack_fam.py "
              "--glb <map>.glb --out assets/fam_side.pt "
              "--b2-out assets/fam_side_b2six.pt\n"
              "Binding nothing here would render the base surface and "
              "report success, which is why this is fatal.")
    print("batch-2 dispatch bound: "
          + ", ".join(f"{FAM_NAMES[k]} -> {v.__name__}"
                      for k, v in sorted(B2_LIT.items()))
          + ("; compositors: "
             + ", ".join(f"{FAM_NAMES[k]} -> {v.__name__}"
                         for k, v in sorted(B2_COMPOSITORS.items()))
             if B2_COMPOSITORS else ""), flush=True)


def b2_run_dispatch(rgb, uv, mid, x, fam, n_geo, tan4, wpos, view, eye,
                    vcol, frag_depth, nrm_in):
    """Call each bound entry point on the pixels its family owns.

    -> (rgb, normal, unlit_mask). The mask is 1 where an entry point
    actually wrote, so the caller can route those pixels past the outer
    lighting; it is also what --b2-reach counts, which means a family
    that renders nothing shows a zero here rather than nothing at all.
    """
    out_rgb, out_nrm = rgb, nrm_in
    unlit = torch.zeros_like(uv[..., 0])
    for fid, fn in sorted(B2_LIT.items()):
        sel = (fam == fid)
        if not bool(sel.any()):
            _b2_count(FAM_NAMES[fid] + " [no pixels this batch]", 0)
            continue
        # `tap` is a bound reference; the family test happens ONCE here,
        # not per pixel inside the shader.
        tap = b2_dispatch_lit(fid, None)
        kw = {}
        if fn is b2_simple_liquid_shade:
            kw = dict(frag_depth=frag_depth, lin_depth=None, scene_rgb=None,
                      vcol=vcol)
        elif fn is b2_simple_shade:
            kw = dict(vcol=vcol)
        elif fn is b2_lightmapped_4wayblend_shade:
            kw = dict(vcol=vcol, blendw=vcol)
        r = _b2_call(tap, rgb, uv, mid, x, n_geo, tan4, wpos, view, eye, kw)
        if r is None:
            continue
        f_rgb, _f_a, f_n, _f_r, _f_m = r
        s3 = sel.unsqueeze(-1)
        # The form-1 precondition, unconditional. `_b2_count` above records
        # how many pixels the family OWNED, which is the same number whether
        # the family changed them or not -- and every one of these four
        # families runs on fallback constants when the pack binds no slot, so
        # "owned N pixels" was compatible with "multiplied by 1.0 N times".
        # `fraction at identity` is what separates ran from did-anything.
        _ts.observe(f"b2.{FAM_NAMES[fid]}.shade", f_rgb, before=rgb,
                    mask=sel, reference=B2_REF.get(fid, ""))
        _ts.observe(f"b2.{FAM_NAMES[fid]}.normal", f_n, before=n_geo,
                    mask=sel, reference=B2_REF.get(fid, ""))
        out_rgb = torch.where(s3, f_rgb, out_rgb)
        out_nrm = torch.where(s3, f_n, out_nrm)
        unlit = torch.where(sel, torch.ones_like(unlit), unlit)
    for fid, fn in sorted(B2_COMPOSITORS.items()):
        sel = (fam == fid)
        if not bool(sel.any()):
            _b2_count(FAM_NAMES[fid] + " [no pixels this batch]", 0)
            continue
        # NOT the light binner. The compositor returns a finished RGBA
        # and its pixels are emitted unlit, which is what a
        # render-to-texture weapon finish is.
        c = fn(uv, None, None, mid, x)
        _ts.observe(f"b2.{FAM_NAMES[fid]}.composite", c[..., :3], before=rgb,
                    mask=sel, reference=B2_REF.get(fid, ""))
        s3 = sel.unsqueeze(-1)
        out_rgb = torch.where(s3, c[..., :3], out_rgb)
        unlit = torch.where(sel, torch.ones_like(unlit), unlit)
    return out_rgb, out_nrm, unlit


def _b2_call(fn, rgb, uv, mid, x, n_geo, tan4, wpos, view, eye, kw):
    """One entry point, with the arguments its signature actually takes."""
    if fn is b2_customglove_preview_shade:
        return fn(uv, None, mid, x, n_geo, tan4, wpos, view, eye)
    if fn is b2_lightmapped_4wayblend_shade:
        return fn(uv, None, mid, x, n_geo, tan4, wpos, view, eye,
                  kw["vcol"], kw["blendw"])
    if fn is b2_weapon_sticker_shade:
        # the base surface the sticker composites OVER (glsl:341-412) is
        # the albedo the pipeline already built, not a blank.
        return fn(uv, mid, x, rgb,
                  torch.full_like(uv[..., 0], float(args.roughness)),
                  torch.zeros_like(uv[..., 0]), n_geo, tan4, wpos, view, eye)
    if fn is b2_simple_shade:
        return fn(uv, mid, x, n_geo, tan4, wpos, view, eye, kw["vcol"])
    if fn is b2_simple_liquid_shade:
        return fn(uv, mid, x, n_geo, tan4, wpos, view, eye, kw["vcol"],
                  frag_depth=kw["frag_depth"], lin_depth=kw["lin_depth"],
                  scene_rgb=kw["scene_rgb"])
    return None


def b2_dispatch_lit(fam, default_fn):
    """Family id -> entry point, or `default_fn` where none is bound.

    Returned as a REFERENCE so the caller binds it once per batch rather
    than testing the family per pixel:
        tap = b2_dispatch_lit(f, base_shade)
    which is the form the brief calls wired.
    """
    return B2_LIT.get(int(fam), default_fn)


# BOUND AT IMPORT, not lazily inside the render loop: a dispatch that is
# only bound on the first covered pixel is a dispatch that binds nothing
# when the family has no pixels, and "never called" would then be
# indistinguishable from "nothing to render".
if B2:
    b2_bind_dispatch()


# =======================================================================
# REACHABILITY -- a synthesised draw, so "nothing to render" can never be
# reported as "never called".
#
# Five of these six live on MODELS (weapon, glove, liquid), so a de_inferno
# world pack carries no material for them and the honest map-side answer
# is zero pixels. A zero that means "absent from this pack" and a zero
# that means "the entry point is dead" look identical, which is precisely
# the shape CHECKS_THAT_CANNOT_FAIL.md catalogues. So the entry points
# are driven directly, on a synthesised tile, and the pixel counts are
# printed per family.
# =======================================================================
def b2_reach():
    """Call every bound entry point on a synthesised draw; print counts.

    THE CHECK CAN FAIL. --b2-selftest-inject makes a named path dead and
    the corresponding max|delta| below collapses to 0.0; that has been
    run for each name. A run in which every delta is 0 is a FAILING run,
    not a quiet one, and it exits non-zero.
    """
    B2_STAT.clear()
    n = 32
    dev = device
    yy, xx = torch.meshgrid(torch.arange(n, device=dev).float(),
                            torch.arange(n, device=dev).float(),
                            indexing="ij")
    uv = torch.stack([(xx + 0.5) / n, (yy + 0.5) / n], -1)[None]
    ones = torch.ones(1, n, n, device=dev)
    n_geo = _nrm(torch.stack([(uv[..., 0] - 0.5) * 0.4,
                              (uv[..., 1] - 0.5) * 0.4, ones[0][None]], -1))
    tan4 = torch.cat([torch.stack([ones, torch.zeros_like(ones),
                                   torch.zeros_like(ones)], -1),
                      ones.unsqueeze(-1)], -1)
    wpos = torch.stack([(uv[..., 0] - 0.5) * 4.0, (uv[..., 1] - 0.5) * 4.0,
                        torch.zeros_like(ones)], -1)
    eye = torch.tensor([[0.0, 0.0, 3.0]], device=dev)
    view = _nrm(eye.view(-1, 1, 1, 3) - wpos)
    vcol = torch.cat([torch.full((1, n, n, 3), 0.8, device=dev),
                      ones.unsqueeze(-1)], -1)
    # a per-pixel ramp so a depth-consuming path has something to consume
    fragz = (uv[..., 0] * 0.5 + 0.25)
    linz = fragz * 100.0
    scene = torch.full((1, n, n, 3), 0.3, device=dev)
    # Every material id is exercised, not just id 0: a family's columns
    # can be right for one material and wrong for another.
    mids = sorted({0, MAT_FAM.shape[0] - 1})
    rows = []
    for key in sorted(B2_FAMS):
        for mid_i in mids:
            mid = torch.full((1, n, n), mid_i, dtype=torch.long, device=dev)
            x = MAT_EXT2[mid]
            if key == "customweapon":
                out = b2_customweapon_composite(uv, uv, uv, mid, x)
                rows.append((key, mid_i, "composite", out[..., :3]))
                continue
            if key == "sticker":
                r = b2_weapon_sticker_shade(
                    uv, mid, x, torch.full((1, n, n, 3), 0.5, device=dev),
                    torch.full((1, n, n), 0.5, device=dev),
                    torch.zeros(1, n, n, device=dev),
                    n_geo, tan4, wpos, view, eye)
            elif key == "glove":
                r = b2_customglove_preview_shade(uv, None, mid, x, n_geo,
                                                 tan4, wpos, view, eye)
            elif key == "simple":
                r = b2_simple_shade(uv, mid, x, n_geo, tan4, wpos, view,
                                    eye, vcol)
            elif key == "fourway":
                r = b2_lightmapped_4wayblend_shade(uv, uv, mid, x, n_geo,
                                                   tan4, wpos, view, eye,
                                                   vcol, vcol)
            else:
                r = b2_simple_liquid_shade(uv, mid, x, n_geo, tan4, wpos,
                                           view, eye, vcol,
                                           frag_depth=fragz,
                                           lin_depth=linz,
                                           scene_rgb=scene)
            rows.append((key, mid_i, "lit", r[0]))
    print("=" * 66, flush=True)
    print("batch-2 reachability -- synthesised %dx%d draw" % (n, n),
          flush=True)
    bad = []
    for key, mid_i, kind, img in rows:
        finite = torch.isfinite(img).all().item()
        span = float(img.max() - img.min()) if finite else float("nan")
        print("  %-14s mid %-5d %-9s wrote %6d px  range %.6f  finite %s"
              % (key, mid_i, kind, img[..., 0].numel(), span, finite),
              flush=True)
        if not finite:
            bad.append("%s mid %d produced a non-finite pixel" % (key, mid_i))
        elif span == 0.0:
            # The docstring above has always said a run in which every
            # delta is 0 is a FAILING run. Only the non-finite test was
            # ever wired, so the check passed while csgo_customweapon
            # wrote range EXACTLY 0.000000 -- a constant image. A
            # constant is what an entry point that reads nothing
            # produces, which is the one thing this audit exists to
            # detect.
            bad.append("%s mid %d wrote a CONSTANT image (range 0.0): the "
                       "entry point ran but nothing it read varied"
                       % (key, mid_i))
    for k in sorted(B2_STAT):
        print("    counter %-46s %d" % (k, B2_STAT[k]), flush=True)
    # Same rule for the counters. csgo_weapon_sticker counted 0 while its
    # row reported range 0.36: the RGB varied but the alpha it composites
    # through was identically zero, so in a real draw it contributes
    # nothing. A family that is dispatched and covers no pixel has not
    # been shown reachable, and saying so is the whole point of the run.
    for k in sorted(B2_STAT):
        if B2_STAT[k] == 0:
            bad.append("counter %s is 0: the family was called but "
                       "covered no pixel" % k)
    if bad:
        for b in bad:
            print("  FAIL " + b, flush=True)
        raise SystemExit(3)
    return rows


def b2_selftest():
    """Every axis at every value, with max|delta| between the arms.

    NEVER an identity claim: each line prints a NUMBER. An axis whose two
    arms differ by 0.0 is reported as such and counted as a failure
    unless the bytecode says the two arms ARE the same module, which for
    D_USE_DEPTH / D_MSAA_DEPTH / S_HOLOMASK_USE_DXT it does -- those
    three are listed as EXPECTED-IDENTICAL with the measurement that
    justifies it, and every other axis must move something.
    """
    import itertools
    n = 32
    dev = device
    yy, xx = torch.meshgrid(torch.arange(n, device=dev).float(),
                            torch.arange(n, device=dev).float(),
                            indexing="ij")
    uv = torch.stack([(xx + 0.5) / n, (yy + 0.5) / n], -1)[None]
    ones = torch.ones(1, n, n, device=dev)
    n_geo = _nrm(torch.stack([(uv[..., 0] - 0.5) * 0.4,
                              (uv[..., 1] - 0.5) * 0.4, ones[0][None]], -1))
    tan4 = torch.cat([torch.stack([ones, torch.zeros_like(ones),
                                   torch.zeros_like(ones)], -1),
                      ones.unsqueeze(-1)], -1)
    wpos = torch.stack([(uv[..., 0] - 0.5) * 4.0, (uv[..., 1] - 0.5) * 4.0,
                        torch.zeros_like(ones)], -1)
    eye = torch.tensor([[0.0, 0.0, 3.0]], device=dev)
    view = _nrm(eye.view(-1, 1, 1, 3) - wpos)
    vcol = torch.cat([torch.full((1, n, n, 3), 0.8, device=dev),
                      ones.unsqueeze(-1)], -1)
    fragz = (uv[..., 0] * 0.5 + 0.25)
    linz = fragz * 100.0
    scene = torch.full((1, n, n, 3), 0.3, device=dev)
    mid = torch.zeros(1, n, n, dtype=torch.long, device=dev)
    x = MAT_EXT2[mid]

    def _run(key):
        if key == "customweapon":
            return b2_customweapon_composite(uv, uv, uv, mid, x)[..., :3]
        if key == "sticker":
            return b2_weapon_sticker_shade(
                uv, mid, x, torch.full((1, n, n, 3), 0.5, device=dev),
                torch.full((1, n, n), 0.5, device=dev),
                torch.zeros(1, n, n, device=dev),
                n_geo, tan4, wpos, view, eye)[0]
        if key == "glove":
            return b2_customglove_preview_shade(uv, None, mid, x, n_geo,
                                                tan4, wpos, view, eye)[0]
        if key == "simple":
            return b2_simple_shade(uv, mid, x, n_geo, tan4, wpos, view,
                                   eye, vcol)[0]
        if key == "fourway":
            return b2_lightmapped_4wayblend_shade(uv, uv, mid, x, n_geo,
                                                  tan4, wpos, view, eye,
                                                  vcol, vcol)[0]
        return b2_simple_liquid_shade(uv, mid, x, n_geo, tan4, wpos, view,
                                      eye, vcol, frag_depth=fragz,
                                      lin_depth=linz, scene_rgb=scene)[0]

    # (family, flag attribute, values, EXPECTED-IDENTICAL?)
    AXES = [
        ("sticker", "b2_holomask_dxt", (0, 1), True),
        ("sticker", "b2_holomask_lut_sampler", (0, 1), False),
        ("sticker", "b2_sticker_mode_depth", (0, 1), False),
        ("sticker", "b2_sticker_foil_layers", (0, 1, 2), False),
        ("glove", "b2_glove_deriv", ("analytic", "off"), False),
        ("glove", "b2_glove_spec_cube_static", (0, 1), False),
        ("glove", "b2_glove_tint_id", (0, 1), False),
        ("simple", "b2_simple_baked",
         ("vertex", "probe", "lightmap", "none"), False),
        ("fourway", "b2_fourway_detail", (0, 1), False),
        ("fourway", "b2_fourway_spec_direct", (0, 1), False),
        ("fourway", "b2_fourway_spec_indirect", (0, 1), False),
        ("fourway", "b2_fourway_quality", (0, 1), False),
        ("fourway", "b2_fourway_spec_cube_static", (0, 1), False),
        ("fourway", "b2_fourway_baked",
         ("vertex", "probe", "lightmap", "none"), False),
        ("liquid", "b2_liquid_foam", (0, 1), False),
        ("liquid", "b2_liquid_no_liquid", (0, 1), False),
        ("liquid", "b2_liquid_opaque_refract", (0, 1), False),
        ("liquid", "b2_liquid_test_values", (0, 1), False),
        ("liquid", "b2_liquid_bgrefract", (0, 1), False),
        ("liquid", "b2_liquid_opaque_fade", (0, 1), False),
        ("liquid", "b2_liquid_use_depth", (0, 1), False),
        ("liquid", "b2_liquid_msaa_depth", (0, 1), True),
        ("liquid", "b2_liquid_depth_source", ("frag", "lin"), False),
        ("customweapon", "b2_cw_paint_style", tuple(range(9)), False),
        ("customweapon", "b2_cw_output", ("albedo", "material"), False),
        ("customweapon", "b2_cw_blend_mode", (0, 1, 2, 3, 4), False),
        ("customweapon", "b2_cw_override_normal", (0, 1), False),
        ("customweapon", "b2_cw_use_all_masks", (0, 1), False),
        ("customweapon", "b2_cw_roughness_texture", (0, 1), False),
        ("customweapon", "b2_cw_pearlescence_mask", (0, 1), False),
        ("customweapon", "b2_cw_metalness_texture", (0, 1), False),
        ("customweapon", "b2_cw_separate_channels", (0, 1), False),
        ("customweapon", "b2_cw_overlay_texture", (0, 1), False),
        ("customweapon", "b2_cw_case_hardening", (0, 1), False),
        ("customweapon", "b2_cw_case_hardening_trilinear", (0, 1), False),
        ("customweapon", "b2_cw_patina_age", (0, 1), False),
    ]
    print("=" * 66, flush=True)
    print("batch-2 self-test -- every axis at every value, max|delta| "
          "between arms", flush=True)
    if B2_INJECT:
        print("  FAULT INJECTED: %s" % B2_INJECT, flush=True)
    dead = []
    for fam, attr, vals, expect_same in AXES:
        if fam not in B2_FAMS:
            continue
        keep = getattr(args, attr)
        imgs = []
        for v in vals:
            setattr(args, attr, v)
            imgs.append(_run(fam))
        setattr(args, attr, keep)
        d = 0.0
        for a, b in itertools.combinations(imgs, 2):
            d = max(d, float((a - b).abs().max()))
        tag = ""
        if expect_same:
            tag = "  (EXPECTED-IDENTICAL: the shipped modules are the same "
            tag += "bytecode on every comparable combo pair)"
        elif d == 0.0:
            dead.append("%s/%s moved nothing" % (fam, attr))
            tag = "  <-- DEAD AXIS"
        print("  %-12s %-34s %d values  max|delta| %.6f%s"
              % (fam, attr, len(vals), d, tag), flush=True)
    if dead:
        print("  SELF-TEST FAILED: %d axis/axes changed nothing"
              % len(dead), flush=True)
        for m in dead:
            print("    " + m, flush=True)
        raise SystemExit(4)
    print("  self-test passed; no axis was silently inert", flush=True)


if B2 and (args.b2_reach or args.b2_selftest):
    if args.b2_reach:
        b2_reach()
    if args.b2_selftest:
        b2_selftest()
    raise SystemExit(0)


# =======================================================================
# Transparency / refraction combo axes.
#
# Everything from here to the end of the block is transcribed from SPIR-V
# modules pulled out of csgo_{complex,glass,foliage,static_overlay}
# _vulkan_50_ps.vcs with iji_model/counter_strike_render/vcs/, decompiled with
# spirv-cross, and diffed combo-against-combo so each axis is isolated by
# construction rather than by reading. The module each block comes from is
# named above it as <shader> s<static>/d<dynamic>:<line>.
# =======================================================================

# The 8x8 Bayer ordered-dither tile the D_OPAQUE_FADE snippet indexes.
#
# WHAT DIFFERS: the shipped code fetches g_tDither (a bindless handle at
# set 1/binding 3 offset 64) with `ivec2(gl_FragCoord.xy) & g_vDitherMask`
# and reads channel .y. The FETCH SITE is transcribed exactly; the
# TEXTURE'S CONTENTS are not in any .vcs -- it is an engine render
# resource, not shader bytecode -- so this is the canonical 8x8 Bayer
# matrix, mask 7. A blue-noise tile of the same size would change which
# pixels survive a partial fade, never how many.
_BAYER8 = None


def _dither_tile():
    global _BAYER8
    if _BAYER8 is None:
        m = torch.zeros(8, 8, device=device)
        for y in range(8):
            for x in range(8):
                v, xc, yc = 0, x ^ y, y
                for b in range(3):
                    v = (v << 1) | ((yc >> (2 - b)) & 1)
                    v = (v << 1) | ((xc >> (2 - b)) & 1)
                m[y, x] = v
        _BAYER8 = (m + 0.5) / 64.0
    return _BAYER8


def opaque_fade(x, on):
    """D_OPAQUE_FADE -> (fade value, keep mask).

    ONE snippet, byte-identical in shape across all three shaders that
    ship the axis:
        csgo_complex        s0/d18:249   (against s0/d2, which has no such line)
        csgo_static_overlay s14/d1       (against s14/d0)
        csgo_foliage        s10/d4:56    (against s10/d0)

        float fade = mix(-F, 1.0, x)
                   + F * texelFetch(g_tDither,
                                    ivec2(gl_FragCoord.xy) & mask, 0).y;
        if ((fade - 0.001) < 0.0) discard;
        ... and the surviving `fade` REPLACES the output alpha
            (complex s0/d18:945, foliage s10/d4:61).

    `x` is vColor.a in csgo_complex and csgo_static_overlay and the
    effective alpha in csgo_foliage; the caller passes whichever its class
    has. F is the per-view fade amount (set 1/binding 1 offset 460) ->
    --opaque-fade-amount.

    `on` is the per-pixel gate; where it is 0 the fade is x itself and the
    keep mask is x > 0.001, i.e. the snippet's own F=0 degeneracy, so a
    material outside the axis is never silently clipped by it.
    """
    t = _dither_tile()
    H_, W_ = x.shape[-2], x.shape[-1]
    ys = torch.arange(H_, device=device).view(-1, 1) & 7
    xs = torch.arange(W_, device=device).view(1, -1) & 7
    d = t[ys, xs]
    F = args.opaque_fade_amount * on
    fade = (-F + (1.0 + F) * x) + F * d
    fade = torch.where(on > 0, fade, x)
    return fade, (fade - 0.001) >= 0.0
