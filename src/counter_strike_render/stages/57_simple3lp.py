

def simple3lp(rgb, x, mid, uv, uv2, vc, nmap, n_geo, tan4, wpos, eye,
              uvscale, fmask):
    """csgo_simple_3layer_parallax.vfx -- the whole material stage of r4/d16.

    Returns (albedo, rough, metal, ao, emissive, alpha, trans_mask).

    THIS IS NOT env_blend's THREE-LAYER PATH AND DOES NOT SHARE A LINE
    WITH IT. `--parallax3` and `--layer3` above are, respectively, an
    earlier guess at this family and csgo_environment_blend's
    S_ENABLE_LAYER_3 height blend. The three constructions are different
    in kind:
      * gt_layer3()  blends a THIRD MATERIAL LAYER by a HEIGHT-derived
        weight in the SAME plane -- no view vector enters it at all.
      * parallax3()  offsets by (V.xy / V.z) * depth, SUBTRACTS the
        offset, has no view-angle exponent, no per-layer refraction, no
        per-layer mip, composites l2-over-l1 by l1's alpha and then mixes
        the RESULT over the incoming rgb by (1 - mask).
      * this function offsets by tangentViewDir * (depth *
        pow(NdotV, 0.7) / dot(V, N)), ADDS it, adds a tangent-normal
        refraction term, picks a mip per layer, composites
        layer1-over-layer2 by LAYER 1's alpha and then mixes
        g_tColor over that by g_tLayer0Mask.x -- the mask selects the
        PANE, and its polarity is the opposite of parallax3()'s.
    Nothing here is derived from the env_blend work; it is a transcription
    of this family's own bytecode.

    :222  V        = normalize(wpos - eye)          (points AWAY from eye)
    :223  invNdV   = 1.0 / dot(V, Nvertex)          (negative on front faces)
    :224  ndv      = clamp(dot(-V, Nvertex), 0, 1)
    :225  k        = clamp(pow(ndv, 0.7), 0, 1)
    :226-227 vts   = vec2(dot(V, T), dot(V, cross(Nvertex,T)*sign(tan.w)))
    :228-233 nTS   = Source 2 normal decode of g_tNormal
    :276  rough0   = g_tNormal.z
    :277  layer0   = texture(g_tColor, uv)
    :278  layer1   = texture(g_tLayer1Color,
                             nTS.xy*refract1 + uv + vts*(off1*k*invNdV),
                             lod = mix(lod1min, lod1max, rough0))
    :279  mask     = texture(g_tLayer0Mask, uv).x
    :281  layer2   = texture(g_tLayer2Color,
                             nTS.xy*refract2 + uv + vts*(off2*k*invNdV),
                             lod = mix(lod2min, lod2max, rough0))
    :285  layers   = mix(mix(layer2, layer1, layer1.a), layer0, mask)
    :288  metal    = g_flMetalness * mask       <- masked by the PANE
    :289  tint     = mix(vec3(1), vColor.rgb, tintMask.x)   [S_TINT_MASK=1]
                     vColor.rgb                             [S_TINT_MASK=0]
    :290  albedo   = layers * tint
    :291  F0       = mix(vec3(g_flReflectance), albedo, metal)
    :749  emissive = mix(layer2 * cube? * clamp(ndv^e2,0,1)
                                * g_flLayer2EmissiveLevel
                                * g_vLayer2EmissiveTint,
                         layer1 * g_flLayer1EmissiveLevel
                                * g_vLayer1EmissiveTint,
                         layer1.a) * tint * (1 - mask)
    :746 (r16)  cube = textureLod(g_tLayer2Cubemap,
                                  vec3(nTS.xy,0)*refract2 + V,
                                  mix(lod2min, lod2max, rough0))

    WHAT DIFFERS FROM THE REFERENCE, precisely:

    1. THE DEPTH UNIT. `off1`/`off2` are g_flLayer1Offset /
       g_flLayer2Offset, which the de_inferno assets set to -90.411 or
       -120 and -358.904 or 0 SOURCE units. The pixel shader multiplies
       them straight into a UV, so either the tangent the VERTEX stage
       writes carries a UV-per-world scale in its length, or the offsets
       are pre-divided upstream. NO VERTEX STAGE WAS EXTRACTED for this
       family (SHADER_CALLFLOW_eight_small_families.md section 6 says so
       for all eight), so the PS alone cannot settle it. The conversion
       used is source-units -> metres (--s3lp-depth-scale, 0.0254) times
       the per-face metres-per-UV already carried in MAT_UVSCALE. That is
       a physical reading, not a fit, and --s3lp-depth-scale 1.0 takes
       the offsets as raw UV instead so the two readings are one flag
       apart.
    2. THE PER-LAYER MIP. :278/:281 pass an explicit LOD of
       mix(lodMin, lodMax, rough0). sample_fam() is a single bilinear
       tap with no mip chain, so the LOD is COMPUTED and REPORTED (it
       drives nothing). No de_inferno material sets either endpoint, so
       the shipped value is the .vfx default, which is not in the PS.
    3. refract1 / refract2 / the layer-2 exponent are read by the module
       and set by NO de_inferno material -- see --s3lp-refract1's help.
       The fallbacks are declared, not measured.
    4. S_TRANSMISSIVE_BACKFACE_NDOTL: see the comment at the term itself.
       The sun's backface N.L is carried; the clustered lights' half of
       the accumulator, the dedicated transmission map (bound by no
       de_inferno material) and the backface-specific cascade depth bias
       are not.
    """
    B_ = _e2(x, "has_l1col")            # both inner layers ship together
    n_v = _nrm(n_geo)
    t_raw = tan4[..., :3]
    t = _nrm(t_raw - n_v * (n_v * t_raw).sum(-1, keepdim=True))
    sgn = torch.where(tan4[..., 3:4] > 0, torch.ones_like(tan4[..., 3:4]),
                      -torch.ones_like(tan4[..., 3:4]))
    b = torch.cross(n_v, t, dim=-1) * sgn                          # :227
    v = _nrm(wpos - eye)                                           # :222
    dvn = (v * n_v).sum(-1)
    # 1/dot(V,N) is unbounded at grazing incidence; the shader's own
    # divide is unguarded, but a NaN here would poison the whole frame,
    # so the magnitude is floored at the same 0.15 the pre-existing
    # parallax3 used and the SIGN is preserved -- the sign is what makes
    # the offset point INTO the surface.
    inv_ndv = torch.where(dvn.abs() < 0.15,
                          torch.sign(dvn).clamp(min=-1.0, max=1.0)
                          * 0.15, dvn)
    inv_ndv = torch.where(inv_ndv == 0, torch.full_like(inv_ndv, -0.15),
                          inv_ndv)
    inv_ndv = 1.0 / inv_ndv                                        # :223
    ndv = (-(v * n_v).sum(-1)).clamp(0, 1)                         # :224
    k = (ndv ** 0.699999988079071044921875).clamp(0, 1)            # :225
    vts_x = (v * t).sum(-1)                                        # :227
    vts_y = (v * b).sum(-1)

    nts = gt_decode_normal(nmap)                                   # :228-233
    rough0 = nmap[..., 2]                                          # :276

    # --- S_SECONDARY_UV: three independent selectors ------------------
    sec = _axis3(args.s3lp_secondary_uv, _e2(x, "f_s3lp_secondary_uv"))
    sec1 = sec.unsqueeze(-1)
    uv_m = uv * (1 - sec1) + uv2 * sec1        # normal map + the layers
    uv_k = uv * (1 - sec1) + uv2 * sec1        # g_tLayer0Mask
    uv_t = uv * (1 - sec1) + uv2 * sec1        # g_tTintMask
    # The three selectors are three DIFFERENT constant-buffer members in
    # the module (the S_SECONDARY_UV=1 record emits three separate
    # `sel != 0 ? TEXCOORD.zw : TEXCOORD.xy` blocks). de_inferno sets
    # F_SECONDARY_UV on 0 of 9 materials so nothing distinguishes them
    # here; they are kept as three names so a pack that does set them
    # differently is one column-read away, not a rewrite.

    inv_uv = 1.0 / uvscale.clamp(min=1e-4)
    dscale = args.s3lp_depth_scale * inv_uv

    def _layer(off_key, refr_key, has_refr_key, refr_flag_default,
               lo_key, hi_key, tex):
        d = _e2(x, off_key) * dscale                       # depth in UV
        hr = _e2(x, has_refr_key)
        refr = _e2(x, refr_key) * hr + refr_flag_default * (1 - hr)
        du = nts[..., 0] * refr + (uv_m[..., 0] + vts_x * (d * k * inv_ndv))
        dv = nts[..., 1] * refr + (uv_m[..., 1] + vts_y * (d * k * inv_ndv))
        lod = _e2(x, lo_key) + (_e2(x, hi_key) - _e2(x, lo_key)) * rough0
        return sample_fam(torch.stack([du, dv], dim=-1), tex), lod

    l1, lod1 = _layer("layer1_offset", "s3lp_refract1", "has_s3lp_refract1",
                      args.s3lp_refract1, "s3lp_lod1_min", "s3lp_lod1_max",
                      T_L1COL[mid])                                # :278
    l2, lod2 = _layer("layer2_offset", "s3lp_refract2", "has_s3lp_refract2",
                      args.s3lp_refract2, "s3lp_lod2_min", "s3lp_lod2_max",
                      T_L2COL[mid])                                # :281
    _simple_count("s3lp.layer_mip_requested", (lod1 + lod2) > 0, fmask)

    mask = sample_fam(uv_k, T_L0MASK[mid])[..., 0:1]               # :279
    mask = mask * _e2(x, "has_l0mask").unsqueeze(-1) \
        + (1 - _e2(x, "has_l0mask").unsqueeze(-1))
    a1 = l1[..., 3:4]                                              # :283
    inner = l2[..., :3] * (1 - a1) + l1[..., :3] * a1              # :285
    layers = inner * (1 - mask) + rgb * mask                       # :285

    # --- S_TINT_MASK (place value 4; de_inferno's own combo) ----------
    tm_on = _axis3(args.s3lp_tint_mask, _e2(x, "f_s3lp_tint_mask")) \
        * _e2(x, "has_s3lp_tint")
    tmv = sample_fam(uv_t, T_S3TINT[mid])[..., 0]
    tm = tmv * tm_on + (1.0 - tm_on)          # S_TINT_MASK=0 -> tint whole
    tint = 1.0 + (vc[..., :3] - 1.0) * tm.unsqueeze(-1)            # :289
    albedo = layers * tint                                         # :290

    # --- S_METALNESS_TEXTURE, then masked by the pane (:288) ----------
    aotex = sample_fam(uv, T_SIMPAO[mid])
    have_ao = _e2(x, "has_simple_ao")
    ao = aotex[..., 0] * have_ao + (1.0 - have_ao)
    mtex = _axis3(args.s3lp_metalness_texture,
                  _e2(x, "f_simple_metal_tex")) * have_ao
    metal = ((_e2(x, "simple_metalness") * (1 - mtex)
              + aotex[..., 3] * mtex) * mask[..., 0]).clamp(0, 1)

    # --- roughness: same construction as csgo_simple (:298) -----------
    dx = torch.zeros_like(n_v)
    dy = torch.zeros_like(n_v)
    dx[:, :, 1:, :] = n_v[:, :, 1:, :] - n_v[:, :, :-1, :]
    dy[:, 1:, :, :] = n_v[:, 1:, :, :] - n_v[:, :-1, :, :]
    dfloor = ((dx * dx).sum(-1).maximum((dy * dy).sum(-1))
              ).clamp(0, 1) ** 0.333000004291534423828125
    rough = torch.maximum(rough0, dfloor).clamp(0.03, 1.0)

    # --- the per-layer emissive (:749) --------------------------------
    hf = _e2(x, "has_s3lp_layer2_fresnel")
    e2exp = _e2(x, "s3lp_layer2_fresnel") * hf \
        + args.s3lp_layer2_fresnel * (1 - hf)
    fres = (ndv.clamp(min=1e-6) ** e2exp).clamp(0, 1)
    cube_on = _axis3(args.s3lp_second_layer_cubemap,
                     _e2(x, "f_s3lp_layer2_cube")) \
        * _e2(x, "has_s3lp_layer2_cube")
    # :746 -- the fetch direction is the VIEW vector perturbed by the
    # tangent normal, NOT a reflection. Our fam_tex array is 2D, so the
    # cube is sampled as a lat-long map of the same direction; that is
    # the projection difference, and it is the only one.
    _hr2 = _e2(x, "has_s3lp_refract2")
    _refr2 = _e2(x, "s3lp_refract2") * _hr2 + args.s3lp_refract2 * (1 - _hr2)
    cdir = _nrm(v + torch.stack([nts[..., 0], nts[..., 1],
                                 torch.zeros_like(nts[..., 0])], dim=-1)
                * _refr2.unsqueeze(-1))
    cuv = torch.stack([torch.atan2(cdir[..., 1], -cdir[..., 0])
                       * 0.15915493667125701904296875 + 0.5,
                       torch.acos(cdir[..., 2].clamp(-1, 1))
                       * 0.3183098733425140380859375], dim=-1)
    cube = sample_fam(cuv, T_L2CUBE[mid])[..., :3]
    cube = cube * cube_on.unsqueeze(-1) + (1 - cube_on.unsqueeze(-1))
    e2 = (l2[..., :3] * cube * fres.unsqueeze(-1)
          * _e2(x, "layer2_emissive").unsqueeze(-1) * MAT_L2EM[mid])
    e1 = (l1[..., :3] * _e2(x, "layer1_emissive").unsqueeze(-1)
          * MAT_L1EM[mid])
    emis = (e2 * (1 - a1) + e1 * a1) * tint * (1 - mask)            # :749

    # --- S_TRANSMISSIVE_BACKFACE_NDOTL (place value 8) ----------------
    trans = _axis3(args.s3lp_transmissive, _e2(x, "f_s3lp_transmissive"))
    # The S_TRANSMISSIVE_BACKFACE_NDOTL module accumulates
    # max(0, -dot(N, L)) * lightColour * atten over the sun AND every
    # clustered local light, multiplies the total by a transmission map
    # and adds it to the output UNLIT. The sun half is carried here. The
    # WHOLE axis, and what is and is not carried:
    #   * carried: the sun's backface N.L, against the SHADING normal
    #     (the module uses _13532, the tangent-frame normal, not the
    #     vertex one), added through `emis` so it is not multiplied by
    #     the lighting -- which is what "unlit" means in this renderer.
    #   * NOT carried, and stated rather than hidden: (a) the per-light
    #     half of the accumulator, because the clustered light loop is
    #     the renderer's analytic_lights() and it does not return a
    #     backface term; (b) the transmission MAP -- the module samples
    #     a dedicated 2D map for it, no de_inferno material binds one,
    #     and an unbound descriptor reads white, so white is what is
    #     used, consistent with the convention the rest of this file
    #     already applies; (c) the module ALSO swaps the cascade depth
    #     bias for a backface-specific one, and gt_csm_shadow() takes a
    #     single bias.
    _nsh = _nrm(t * nts[..., 0:1] - b * nts[..., 1:2] + n_v * nts[..., 2:3])
    _back = (-(_nsh * sun_dir.view(1, 1, 1, 3)).sum(-1)).clamp(min=0)
    trans_rgb = (_back.unsqueeze(-1) * SUN_COLOR.view(1, 1, 1, 3)
                 * trans.unsqueeze(-1))

    _simple_count("s3lp.shaded", B_ > 0.5, fmask)
    _simple_count("s3lp.transmissive_lit", (trans > 0.5) & (_back > 0),
                  fmask)
    _simple_count("s3lp.tint_mask", tm_on > 0.5, fmask)
    _simple_count("s3lp.metalness_texture", mtex > 0.5, fmask)
    _simple_count("s3lp.secondary_uv", sec > 0.5, fmask)
    _simple_count("s3lp.second_layer_cubemap", cube_on > 0.5, fmask)
    _simple_count("s3lp.transmissive", trans > 0.5, fmask)
    _simple_count("s3lp.layer1_emissive", _e2(x, "layer1_emissive") > 0,
                  fmask)
    _simple_count("s3lp.layer2_emissive", _e2(x, "layer2_emissive") > 0,
                  fmask)
    _simple_count("s3lp.pane_opaque", mask[..., 0] > 0.99, fmask)
    _simple_count("s3lp.interior_visible", mask[..., 0] < 0.99, fmask)
    _simple_count("s3lp.parallax_offset_nonzero",
                  ((_e2(x, "layer1_offset").abs()
                    + _e2(x, "layer2_offset").abs()) > 0), fmask)
    # the transmissive term is ADDITIVE and UNLIT, so it rides out on the
    # same channel the layer emissives do rather than being returned for
    # a caller that would have to remember to apply it.
    return albedo, rough, metal, ao, emis + trans_rgb, vc[..., 3], trans


def simple_dispatch(fam_pix, alb, rough, metal, ao, emis, x, mid, uv, uv2,
                    vc, nmap, n_geo, tan4, wpos, eye, uvscale):
    """THE FAMILY DISPATCH for csgo_simple / csgo_simple_3layer_parallax.

    Selected by MATERIAL FAMILY out of the side table's `fam_id`
    (fam_pix == FAM_SIMPLE / FAM_SIMPLE_3LP), never by name and never by
    a texture-presence heuristic. Returns the same tuple it was given,
    with each family's pixels replaced.

    Both families are 100% OPAQUE on de_inferno -- all 18 csgo_simple and
    all 9 csgo_simple_3layer_parallax materials carry alphaMode OPAQUE in
    the packed glTF and none of them is reclassified by
    fast_pack_fam.py's F_BLEND_MODE rule -- so calling this from the
    opaque class reaches 100% of both families' pixels rather than some
    fraction of them. That is a checked property of the asset, not an
    assumption: --simple-reach prints the per-family shaded-pixel count
    and it is 0 if this is wrong.
    """
    out_alb, out_r, out_m, out_ao, out_e = alb, rough, metal, ao, emis
    if SIMPLE_ON:
        m = (fam_pix == FAM_SIMPLE)
        if bool(m.any()):
            a, r, mt, o, _al, _hao = simple_surface(alb, x, mid, uv, vc,
                                                    nmap, n_geo, m)
            f = m.unsqueeze(-1).float()
            out_alb = out_alb * (1 - f) + a * f
            mf = m.float()
            if out_r is not None:
                out_r = out_r * (1 - mf) + r * mf
            if out_m is not None:
                out_m = out_m * (1 - mf) + mt * mf
            if out_ao is not None:
                # ONLY WHERE THIS FAMILY ACTUALLY HAS AN AO SLOT.
                # csgo_simple reads its own g_tAmbientOcclusion (:205,
                # :710) and substitutes 1.0 where the material binds none
                # -- correct in isolation, wrong as a fold: on
                # de_inferno f700 simple.ao_texture_bound is 0 over all
                # 147,264 of this family's pixels, so an unconditional
                # blend overwrote the incoming AO with a constant 1.0 on
                # ~16% of the frame. That includes the mat_ao page.
                #
                # Where the family's slot IS bound it still wins: it is
                # the more specific read of the same physical slot. Where
                # it is NOT bound, 1.0 is an identity standing in for an
                # absent texture, and an identity must not displace a
                # value the caller already has.
                _keep = _hao.clamp(0, 1)
                out_ao = out_ao * (1 - mf * _keep) + o * (mf * _keep)
                _simple_count("simple.ao_slot_displaced_page",
                              (_hao <= 0.5), m)
            _simple_count("dispatch.simple_pixels", m)
    if S3LP_ON:
        m = (fam_pix == FAM_SIMPLE_3LP)
        if bool(m.any()):
            a, r, mt, o, e, _al, _tr = simple3lp(
                alb, x, mid, uv, uv2, vc, nmap, n_geo, tan4, wpos, eye,
                uvscale, m)
            f = m.unsqueeze(-1).float()
            out_alb = out_alb * (1 - f) + a * f
            mf = m.float()
            if out_r is not None:
                out_r = out_r * (1 - mf) + r * mf
            if out_m is not None:
                out_m = out_m * (1 - mf) + mt * mf
            if out_ao is not None:
                out_ao = out_ao * (1 - mf) + o * mf
            if out_e is None:
                out_e = torch.zeros_like(out_alb)
            out_e = out_e * (1 - f) + e * f
            _simple_count("dispatch.s3lp_pixels", m)
    return out_alb, out_r, out_m, out_ao, out_e


# =======================================================================
# GROUND-TRUTH MATERIAL-SURFACE COMBO AXES
# =======================================================================
# Transcribed from docs/projects/counter-strike-sft/. Where a permutation
# with the axis ENABLED was decompiled the port is line-for-line and the
# lines are cited; where only the axis-OFF permutation is committed the
# closest complete form is implemented and the difference is written down
# in the docstring. Nothing here is a stub that returns its input.

def _uv_transform(uv, su, sv, ou, ov):
    """One texture slot's affine UV transform.

    csgo_environment_blend_vs_max.glsl:343-478 and
    csgo_complex_vs_max.glsl:250-251 both build every slot's coordinate as
    `u' = dot(uv, rowU.xy) + rowU.w`, `v' = dot(uv, rowV.xy) + rowV.w`.
    The de_inferno materials only ever populate the diagonal of that 2x2
    (a scale), so the off-diagonal terms are not carried through the pack
    -- a rotated UV transform would need two more columns per slot and no
    material in this export asks for one. That is the ONE difference from
    the reference here and it is a data limit, not a maths simplification.
    """
    return torch.stack([uv[..., 0] * su + ou, uv[..., 1] * sv + ov], dim=-1)


def gt_uv(uv0, uv1, sel, x, su, sv, ou, ov):
    """S_SECONDARY_UV: pick the slot's coordinate stream, then transform.

    csgo_environment_blend_vs_max.glsl:343-350 --
        `mix(uv0, uv1, bvec2(sel == 2))` then the affine transform,
    with `sel == 0` meaning "inherit whatever the previous slot used"
    (lines 330-341 resolve a -1 selector to the previous layer's).
    Here "inherit" is uv0, because uv0 is what every earlier slot in this
    renderer already uses.
    """
    # THE `* f_secondary_uv` CO-FACTOR IS GONE, on a read of the reference
    # rather than a preference. All three compiled variants of this vertex
    # stage guard the construct on the per-material UNIFORM alone --
    # `if (_5618._m0 > 0)` at csgo_environment_blend_vs_max.glsl:344,
    # _vs_base.glsl:162 and _vs_gameplay_max.glsl:335 -- and select the
    # stream with `_m0 == 2` on the next line. No static combo appears in
    # the guard in any of the three.
    #
    # It was not a harmless belt-and-braces. Across all 43 built maps, 78
    # slots select UV set 2 and NOT ONE of them sits on a material that
    # also sets F_SECONDARY_UV, so the product was zero on every slot that
    # asked for the secondary stream -- a second, independent zero waiting
    # behind the packer's key-name miss. Fixing only the key name would
    # still have rendered identical frames, and the natural reading of
    # that would have been "the packer fix did not work".
    #
    # STILL DIVERGENT, deliberately and NOT silently: the reference runs
    # its affine whenever `_m0 > 0`, i.e. for `_m0 == 1` (TEXCOORD0) too,
    # while this keeps the transform riding with the STREAM. The pack
    # carries one secondary scale/offset per MATERIAL, not one per slot,
    # so applying it on the `_m0 == 1` path would put the secondary
    # stream's transform on the primary stream for the majority of
    # materials (124 of 221 on de_cache read uvsel_c1 == 1). That is a
    # separate change with its own price and it is not made here.
    pick = (_e2(x, sel) > 1.5).float().unsqueeze(-1)
    alt = _uv_transform(uv1, _e2(x, su), _e2(x, sv),
                        _e2(x, ou), _e2(x, ov))
    # The transform rides with the STREAM here, not with the slot: the
    # pack carries one secondary scale/offset per material, not one per
    # slot, so a slot that stayed on TEXCOORD0 keeps its untransformed
    # coordinate rather than being handed the secondary stream's scale.
    # In the reference every slot has its own transform (four vec4 per
    # layer); that is the difference, and it is a pack-width limit.
    return uv0 * (1 - pick) + alt * pick


def gt_texture_animation(uv, x, t):
    """S_TEXTURE_ANIMATION -- csgo_complex_vs_max.glsl:252-282, ported.

        uv  = uv * cellSize                                    (line 252)
        m == 0: frame = uint((t + timeOffset) / timePerFrame) % numCells
        m == 1: frame = uint(fract(sin(dot(vec2(seed),
                          vec2(12.9898, 78.233))) * 43758.546875)
                          * float(numCells))
        m == 2: frame = seed                                   (lines 256-274)
        uv += vec2(frame % cellsX, uint(frame * cellSize.x)) * cellSize
                                                               (line 275)
        uv += scrollSpeed * t                                  (line 279)

    `cellSize` is `_5618._m10.xy`, used both as the pre-scale at line 252
    and as the cell stride at line 275, so it is 1/cells per axis and
    `uint(float(frame) * cellSize.x)` is the row index floor(frame/cellsX).

    The scroll at line 279 is UNCONDITIONAL in the reference -- it is in
    the base module too (csgo_complex_vs_base.glsl:158) -- so it is not
    gated by the animation flag inside this function; it rides on the
    same --texture-animation switch only because that is where a caller
    would look for it.
    """
    on = _e2(x, "f_tex_anim")
    cx = _e2(x, "anim_cells_x").clamp(min=1.0)
    cy = _e2(x, "anim_cells_y").clamp(min=1.0)
    csx, csy = 1.0 / cx, 1.0 / cy
    n = _e2(x, "anim_num_cells").clamp(min=1.0)
    m = _e2(x, "anim_method")
    seed = _e2(x, "anim_seed")
    tpf = _e2(x, "anim_time_per_frame").clamp(min=1e-4)
    toff = _e2(x, "anim_time_offset")
    f_seq = torch.floor((t + toff) / tpf) % n
    f_rnd = torch.floor(
        torch.frac(torch.sin(seed * 12.98980045318603515625
                             + seed * 78.233001708984375)
                   * 43758.546875) * n)
    frame = torch.where(m < 0.5, f_seq,
                        torch.where(m < 1.5, f_rnd, seed))
    col = torch.remainder(frame, cx)
    rowi = torch.floor(frame * csx)
    u = uv[..., 0] * csx + col * csx
    v = uv[..., 1] * csy + rowi * csy
    anim = torch.stack([u, v], dim=-1)
    on4 = on.unsqueeze(-1)
    out = uv + (anim - uv) * on4
    # line 279: the scroll, which the reference applies to every material.
    # `t` is (frames, 1, 1) so it broadcasts against the (frames, H, W)
    # scalars above; the scroll is (frames, H, W, 2) and needs the extra
    # trailing axis or the frame axis would line up against H.
    out = out + torch.stack([_e2(x, "scroll_u"), _e2(x, "scroll_v")],
                            dim=-1) * t.unsqueeze(-1)
    # The cell scale shrinks the UV footprint by the same factor, so the
    # mip footprint has to shrink with it or an animated sprite sheet
    # would be filtered as if it still spanned the whole texture.
    duv = torch.stack([1.0 + (csx - 1.0) * on, 1.0 + (csy - 1.0) * on],
                      dim=-1)
    return out, duv


def gt_detail_texture(rgb, det, x):
    """S_DETAIL_TEXTURE -- RANGE 0..4, the VALUE is the blend mode.

    `csgo_complex_ps.glsl` is static combo 40 (`S_ALPHA_TEST` only) and
    `csgo_complex_ps_shaderquality1.glsl` is 5160, so NEITHER committed
    permutation has this axis on and no decompiled module shows the five
    blend expressions. What IS established (SHADER_CALLFLOW_csgo_complex
    .md §1): the axis is `m_iFeatureIndex` 11 = `F_DETAIL_TEXTURE`, bit
    weight 2, range 0..4, and exactly one de_inferno material selects it
    (static id 1282 = `S_DETAIL_TEXTURE=1` + `S_METALNESS_TEXTURE`);
    values 2..4 are not realised by any de_inferno material.

    So the four non-zero modes are implemented as the four Source detail
    composites, in the order Source 2's own detail enum uses:
        1 mod2x     base * detail * 2
        2 multiply  base * detail
        3 overlay   screen above mid-grey, multiply below
        4 additive  base + detail
    each lerped by `g_flDetailBlendFactor`. **The value -> mode
    correspondence is NOT established by the archives** -- only that the
    axis is 5-valued and that value 1 is the one de_inferno uses.
    --detail-mode overrides it so the assignment can be tested rather
    than assumed.
    """
    v = (torch.full_like(_e2(x, "f_detail"), float(args.detail_mode))
         if args.detail_mode >= 0 else _e2(x, "f_detail"))
    on = (_e2(x, "has_detail") * (v > 0.5).float()
          * _e2(x, "detail_blend")).unsqueeze(-1)
    d = det[..., :3]
    mod2x = rgb * d * 2.0
    mult = rgb * d
    over = torch.where(d > 0.5, 1.0 - 2.0 * (1.0 - rgb) * (1.0 - d),
                       2.0 * rgb * d)
    add = rgb + d
    vv = v.unsqueeze(-1)
    out = torch.where(vv < 1.5, mod2x,
                      torch.where(vv < 2.5, mult,
                                  torch.where(vv < 3.5, over, add)))
    return (rgb + (out - rgb) * on).clamp(0, 8)


def gt_decode_normal(tex):
    """Source 2's own tangent-normal decode, not `2*rgb-1`.

    csgo_environment_blend_ps.glsl:534-538:
        nx = (r + g) - 1.00392162799835205078125
        ny =  r - g
        nz = 1 - |nx| - |ny|
        n  = normalize(vec3(nx, ny, nz))
    The 1.00392... is 256/255, i.e. the encode is exact on 8-bit texels.
    Used here for the DETAIL normal slots, which are this agent's axis;
    the base normal map keeps whatever decode apply_normal_map() uses.
    """
    r, g = tex[..., 0], tex[..., 1]
    nx = (r + g) - 1.00392162799835205078125
    ny = r - g
    nz = (1.0 - nx.abs()) - ny.abs()
    return _nrm(torch.stack([nx, ny, nz], dim=-1))


def gt_detail_normal(n_ts, det, x, w=None, det2=None):
    """S_DETAIL_NORMAL -- per-layer g_tNormalDetail composited on the base.

    The axis is `F_DETAIL_NORMAL` (csgo_environment `m_iFeatureIndex` 5,
    bit weight 1; csgo_environment_blend bit weight 1) and the per-layer
    slots are named in SHADER_TERM_LIST.md §"Per-layer textures"
    (`TextureNormalDetail`, x3). `F_DETAIL_NORMAL` is 0/151 on
    csgo_environment and no committed permutation has `S_DETAIL_NORMAL`
    set (SHADER_CALLFLOW_csgo_environment.md §3), so the COMPOSITION
    OPERATOR is not readable from any decompiled module.

    Implemented as the partial-derivative blend, which is the one
    composition that reproduces `detail = flat` as an exact identity:
        n = normalize(vec3(base.xy * det.z + det.xy * base.z,
                           base.z * det.z))
    scaled by g_flDetailNormalStrength. The layer-2 detail normal blends
    onto the layer-1 one on the same weight the colour used, so bump and
    albedo cannot disagree about where the transition is.
    """
    d = gt_decode_normal(det)
    if det2 is not None and w is not None:
        d = _nrm(d + (gt_decode_normal(det2) - d) * w.unsqueeze(-1))
    s = (_e2(x, "detail_nrm_strength") * _e2(x, "f_detail_normal")
         * torch.maximum(_e2(x, "has_detail_nrm1"),
                         _e2(x, "has_detail_nrm2"))).unsqueeze(-1)
    d = _nrm(torch.cat([d[..., :2] * s, d[..., 2:]], dim=-1))
    comb = _nrm(torch.cat([n_ts[..., :2] * d[..., 2:]
                           + d[..., :2] * n_ts[..., 2:],
                           n_ts[..., 2:] * d[..., 2:]], dim=-1))
    return _nrm(n_ts + (comb - n_ts) * s)


def gt_tint_mask_value(tm_tex, h_tex, x, e):
    """S_TINT_MASK's mask, from whichever slot the material actually has.

    csgo_environment_blend_ps.glsl:593 is the exact remap, and it reads
    the mask out of the HEIGHT texture's GREEN channel:
        `clamp(((g_fTintMaskContrast1 * (h.y - 0.5)) + 0.5)
               * g_fTintMaskBrightness1, 0, 1)`
    csgo_complex binds a real `g_tTintMask` instead (its `S_TINT_MASK` is
    `m_iFeatureIndex` 8, bit weight 640, on 8 of the 67 materials). Both
    are the same remap over a different texel, so --tint-mask-src auto
    takes g_tTintMask where the material binds one and height.g where it
    does not, and neither source is ever read where the material binds
    nothing.
    """
    b = _e2(x, "tm_bright")
    c = _e2(x, "tm_contrast")
    have_tm = _e2(x, "has_tintmask2")
    src_tm = (((tm_tex[..., 0] - 0.5) * c + 0.5) * b).clamp(0, 1)
    src_h = (((h_tex[..., 1] - 0.5) * c + 0.5) * b).clamp(0, 1)
    if args.tint_mask_src == "tintmask-tex":
        return src_tm * have_tm
    if args.tint_mask_src == "height-g":
        return src_h * (_ext(e, "has_h1") if e is not None else 1.0)
    use = have_tm
    have_h = _ext(e, "has_h1") if e is not None else torch.zeros_like(use)
    return src_tm * use + src_h * (1 - use) * have_h


def gt_vertex_color_tint(rgb, vc, tm, e):
    """The vertex-colour tint, GATED BY THE TINT MASK.

    csgo_environment_blend_ps.glsl:600-604, ported line for line:
        n   = normalize(max(vColor.rgb, 0.001))
        Lc  = dot(rgb, (0.2125, 0.7154, 0.0721))
        Ln  = dot(n,   (0.2125, 0.7154, 0.0721))
        mx  = max(vColor.r, max(vColor.g, vColor.b))
        rgb = clamp(mix(rgb, n * min(Lc/Ln, 3*Lc*mx),
                        (vColor.w * tintMask) * float(vertexColorMode != 0)),
                    0, 1)
    i.e. the tint replaces the HUE while preserving the albedo's own
    luminance, and it only applies where the tint mask lets it. This is
    the term --vertex-color-mode's help text points at ("glsl:604 shows
    g_nVertexColorMode gates the vertex-COLOUR tint") and that nothing
    implemented; --vertex-color-mode itself inverts a blend weight and is
    a different thing, so it is left alone.
    """
    LW = torch.tensor([0.2125000059604644775390625,
                       0.7153999805450439453125,
                       0.07209999859333038330078125], device=rgb.device)
    n = _nrm(vc[..., :3].clamp(min=0.001000000047497451305389404296875))
    lc = (rgb * LW).sum(-1, keepdim=True)
    ln = (n * LW).sum(-1, keepdim=True).clamp(min=1e-6)
    mx = vc[..., :3].max(-1, keepdim=True).values
    tinted = n * torch.minimum(lc / ln, 3.0 * lc * mx)
    mode = ((_ext(e, "vcmode1") != 0).float() if e is not None
            else torch.ones_like(tm))
    a = (vc[..., 3:4] * tm.unsqueeze(-1) * mode.unsqueeze(-1)).clamp(0, 1)
    return (rgb + (tinted - rgb) * a).clamp(0, 1)


def gt_material_reference(rgb, x):
    """S_MATERIAL_REFERENCE -- a second colour transform, per pixel.

    csgo_environment `m_iFeatureIndex` 7, bit weight 2, set on 51 of the
    151 de_inferno materials (SHADER_CALLFLOW_csgo_environment.md §3).
    No permutation with it on is committed as GLSL, but its cost is
    measured: r2/m3 against r0/m3 is +28 matrix FLOPs (exactly one
    mat4-by-vec4), +9 mix FLOPs (three vec3 lerps), +6 selects, and
    IDENTICAL fetch counts -- 24 sites, same sampler breakdown (that
    document's §4/§5 tables). A feature that adds one colour matrix and
    three lerps and reads no new texture is a second colour transform of
    a value already in registers, and the blend shader shows the exact
    shape at csgo_environment_blend_ps.glsl:594, where the tint is
    `vec4(color.rgb, 1) * mat4`.

    So: apply the referenced material's tint/brightness/contrast/
    saturation as a second colour_correct() and select it where
    F_MATERIAL_REFERENCE is set. **This is a structural inference from
    the FLOP delta, not a read of the bytecode**; the fetch-count
    invariance is what rules out "it samples a referenced texture".
    """
    on = _e2(x, "f_material_ref").unsqueeze(-1)
    tint = torch.stack([_e2(x, "mref_r"), _e2(x, "mref_g"),
                        _e2(x, "mref_b")], dim=-1)
    ref = color_correct(rgb, tint, _e2(x, "mref_bright"),
                        _e2(x, "mref_contrast"), _e2(x, "mref_sat"))
    return rgb * (1 - on) + ref * on


def gt_decal(rgb, dec, x):
    """S_DECAL_TEXTURE -- g_tDecal over the albedo.

    csgo_complex `m_iFeatureIndex` 21, bit weight 2560. Not realised by
    any de_inferno material (SHADER_CALLFLOW_csgo_complex.md §1) and no
    permutation with it on was decompiled, so the composite is the
    standard one: `mix(rgb, decal.rgb, decal.a * g_flDecalBlendFactor)`
    on the decal's own UV. **The composite is not read from bytecode.**
    """
    on = (_e2(x, "f_decal") * _e2(x, "has_decal")
          * _e2(x, "decal_blend") * dec[..., 3]).unsqueeze(-1).clamp(0, 1)
    return rgb * (1 - on) + dec[..., :3] * on


def gt_aniso_gloss(rough, gloss, x, nrm, tan4, view):
    """S_ANISOTROPIC_GLOSS -> (roughness_t, roughness_b, tangent).

    csgo_complex `m_iFeatureIndex` 9, bit weight 20; not realised by any
    de_inferno material and not decompiled with the axis on. Implemented
    as the standard anisotropic split: the gloss map's R rotates the
    tangent within the tangent plane and its G scales the anisotropy,
    times g_flAnisotropicGlossAmount, giving
        alpha_t = rough * sqrt(1 + a),  alpha_b = rough * sqrt(1 - a)
    which preserves rough at a = 0 exactly, so a material with the axis
    off is untouched. **The map's channel assignment is not read from
    bytecode.**
    """
    on = _e2(x, "f_aniso") * _e2(x, "has_aniso")
    a = (gloss[..., 1] * _e2(x, "aniso_amount") * on).clamp(-0.98, 0.98)
    ang = (gloss[..., 0] * 2.0 - 1.0) * math.pi + _e2(x, "aniso_rotation")
    n = _nrm(nrm)
    t_raw = tan4[..., :3]
    t = _nrm(t_raw - n * (n * t_raw).sum(-1, keepdim=True))
    b = torch.cross(n, t, dim=-1) * tan4[..., 3:4].sign()
    ca, sa = ang.cos().unsqueeze(-1), ang.sin().unsqueeze(-1)
    tan = _nrm(t * ca + b * sa)
    return (rough * (1 + a).clamp(min=1e-3).sqrt(),
            rough * (1 - a).clamp(min=1e-3).sqrt(), tan)


def gt_overlay_layer_k(mode, tintmask, on, tmask, w):
    """glsl:1188 -- `k`, from the EXT2 columns, for the GT-surface branch.

    THIS REPLACES gt_shared_overlay_mode(), WHICH WAS BUILT ON A
    MISREADING and is deleted rather than kept behind a flag. That
    function dispatched four composites on `g_nColorOverlayMode`,
    reasoning that combo 0 could not be read so the mode must select the
    blend. Combo 258 is now extracted and the shader has exactly ONE
    composite; `g_nColorOverlayMode` selects a LAYER, not a blend --
    `0=All Layers,1=Layer1,2=Layer2,3=Layer3`, the declared enum string
    in the shader's own variable table. There is no mode-0 `rgb*o`, no
    mode-2 `rgb*o*2`, and no mode-3 replace anywhere in the shader.

        L_n    = (mode == 0) || (mode == n)
        mask_n = (tintMask == 0) || (tintMask == n)
        k      = (mask1 ? tm : 1) * L1 * (1-w) + (mask2 ? tm : 1) * L2 * w

    `tintMask == 4` is `4=Unmasked` and satisfies neither branch, so it
    gates nothing -- the opposite of what the old call site did with it.
    """
    l1 = ((mode < 0.5) | ((mode > 0.5) & (mode < 1.5))).float()
    l2 = ((mode < 0.5) | ((mode > 1.5) & (mode < 2.5))).float()
    m1 = ((tintmask < 0.5) | ((tintmask > 0.5) & (tintmask < 1.5))).float()
    m2 = ((tintmask < 0.5) | ((tintmask > 1.5) & (tintmask < 2.5))).float()
    tm = torch.ones_like(on) if tmask is None else tmask
    wl = torch.zeros_like(on) if w is None else w
    return on * ((1.0 - m1 + m1 * tm) * l1 * (1.0 - wl)
                 + (1.0 - m2 + m2 * tm) * l2 * wl)


def gt_blend_effects(rgb, rough, nrm, n_geo, band, x, btint, seam):
    """S_BLEND_EFFECTS: the border tint / border roughness / bevel set.

    `--blend-border` and `--bevel` already implement this for the 1->2
    seam, whose parameters carry the `_2` suffix; this is the SAME block
    for the unsuffixed seam (`F_BLEND_EFFECTS`, 5/31 de_inferno
    materials) and the 2->3 seam (`F_BLEND_EFFECTS_3`), which had no
    implementation at all. `seam` is "" or "3" and selects the suffix.

    The band shaping is identical to blend_border()'s, which is where
    that form was established; only the parameter set differs, so the
    two seams cannot disagree about what a border is.
    """
    s = seam
    on = _e2(x, f"f_blend_effects{s}")
    off = _e2(x, f"border_offset{s}")
    spread = _e2(x, f"border_spread{s}").clamp(min=1e-3)
    bsoft = _e2(x, f"border_soft{s}").clamp(min=1e-3)
    xx = (band - off).abs() / (spread + bsoft)
    m = (1.0 - xx).clamp(0, 1)
    m = m * m * (3 - 2 * m) * on * band
    rgb = rgb * (1 - m.unsqueeze(-1)) + rgb * btint * m.unsqueeze(-1)
    if rough is not None:
        br = _e2(x, f"f_border_rough{s}") * on * band
        rough = (rough + (_e2(x, f"border_rough{s}") - rough) * br
                 ).clamp(0.03, 1.0)
    if nrm is not None and n_geo is not None:
        bs = (_e2(x, f"bevel_strength{s}") * on * band
              / _e2(x, f"bevel_soft{s}").clamp(min=1e-3))
        nrm = _nrm(nrm + _nrm(n_geo) * bs.unsqueeze(-1))
    return rgb, rough, nrm


def gt_layer3(rgb, c3, h12, h3, w3, x):
    """S_ENABLE_LAYER_3 -> (rgb, layer-3 weight, its transition band).

    csgo_environment_blend static bit weight 4 (FOG_AND_COMBO_MAPPING.md
    §3). The weight construction IS in the committed combo-0 module --
    csgo_environment_blend_ps.glsl:260-262 builds a THREE-component
    vector from COLOR_0 whether or not layer 3 is compiled in:

        w2 = clamp(vIn0.x * 1.1 - 0.05, 0, 1)
        w3 = clamp(vIn0.y * 1.1 - 0.05, 0, 1)
        bary = normalize(vec3(clamp((1 - w2) - w3, 0, 1), w2, w3))

    so COLOR_0.y is layer 3's weight and the same 5% dead zone applies to
    it. That is ported exactly by the caller. What is NOT in any
    committed module is the 3-way mix itself, because combo 0 has the
    axis off. It is implemented SEQUENTIALLY -- layer 3 blended over the
    1->2 result with its own g_flHeightMapScale3 / ZeroPoint3 /
    BlendSoftness3 -- because every parameter in the feature set is named
    for a SEAM rather than a layer (`F_BLEND_EFFECTS_2` is the 1->2 seam,
    `_3` the 2->3 seam; `g_flBlendSoftness2/3` likewise), which is only
    consistent with a chain of two 2-layer blends.

    F_ENABLE_LAYER_3 is 0/31 on de_inferno (FOG_AND_COMBO_MAPPING.md §4),
    so on this map the feature reaches no pixel; it is implemented
    because the axis exists, not because it is expected to move a metric.
    """
    f3 = _e2(x, "f_layer3") * _e2(x, "has_c3")
    w = (w3 * f3).clamp(0, 1)
    s3 = _e2(x, "h_scale3")
    z3 = _e2(x, "h_zero3")
    soft = _e2(x, "blend_soft3").clamp(min=1e-3)
    if args.height_blend:
        a1 = (1 - w) + h12 * args.height_contrast
        a2 = w + (h3 - z3) * s3
        m = torch.maximum(a1, a2) - soft
        b1 = (a1 - m).clamp(min=0)
        b2 = (a2 - m).clamp(min=0)
        w = torch.where(f3 > 0.5, b2 / (b1 + b2).clamp(min=1e-6), w)
    tint = torch.stack([_e2(x, "cc3_r"), _e2(x, "cc3_g"),
                        _e2(x, "cc3_b")], dim=-1)
    # No per-instance model-tint factor is applied to layer 3: MTINT is a
    # per-LAYER 0/1 gate on glTF's baseColorFactor and the pack carries
    # only two layers of it, so there is nothing to gate layer 3 with.
    # Multiplying the colour by the two-layer gate would have blacked the
    # layer out. That is the difference from layers 1 and 2 and it is a
    # pack-width limit, not a shading decision.
    col3 = color_correct(c3[..., :3], tint, _e2(x, "cc3_bright"),
                         _e2(x, "cc3_contrast"), _e2(x, "cc3_sat"))
    w4 = w.unsqueeze(-1)
    band = (4.0 * w * (1.0 - w)).clamp(0, 1) * f3
    return rgb * (1 - w4) + col3 * w4, w, band


def gt_composite(base, rgb, a, mode, opacity):
    """S_BLEND_MODE, all 7 values -- the composite for each.

    `S_BLEND_MODE` is csgo_static_overlay's bit-weight-2 axis with RANGE
    0..6 (SHADER_CALLFLOW_csgo_static_overlay.md §"Combo axes"). Values 1
    and 2 are pinned by material evidence (the 17 with mode 1 are exactly
    the 17 carrying g_flOpacityScale; the 6 with mode 2 are exactly the 6
    carrying g_flAlphaTestReference). That document's §7.2 states plainly
    that the meaning of 3 vs 4 vs 5 vs 6 is NOT established, so:

        0 opaque         replace
        1 translucent    src.a over dst          (pinned)
        2 alpha-tested   replace where it passed (pinned)
        3 modulate       dst * src               (its one material is
                                                  `top_grime_2`, a grime
                                                  pass, so multiply)
        4 modulate2x     dst * src * 2
        5 additive       dst + src * a
        6 premultiplied  dst * (1 - a) + src

    3..6 are the remaining Source blend states in their conventional
    order. **Not established from the archives**; implemented so the axis
    is complete and so a wrong assignment is a visible, testable
    difference instead of a missing branch.
    """
    a = (a * opacity.unsqueeze(-1)).clamp(0, 1)
    m = mode.unsqueeze(-1)
    over = base * (1 - a) + rgb * a
    modu = base * (1 - a + a * rgb)
    mod2 = base * (1 - a + a * rgb * 2.0)
    addi = base + rgb * a
    prem = base * (1 - a) + rgb
    # MODE 0 MUST NOT WRITE OUTSIDE COVERAGE. It used to return `rgb`
    # unconditionally while every other branch multiplies by `a` or
    # (1-a), so a mode-0 material replaced the whole blend-raster
    # FOOTPRINT rather than the alpha-passing area. Measured: forcing the
    # 20 csgo_static_overlay materials onto this branch changed 99.9994%
    # of f00738 and took KL 1.8546 -> 7.4050, ncc 0.1820 -> 0.0409, CV
    # ratio 0.3544 -> 0.0502 -- the frame going nearly uniform.
    #
    # DELIBERATELY MINIMAL: `a > 0` changes ONLY the a == 0 case and
    # leaves every covered pixel exactly as before. Whether "opaque"
    # should hard-replace or blend at PARTIAL coverage is a question
    # about CS2's mode 0 that the stripped SPIR-V cannot answer, so it is
    # left open rather than guessed -- an opaque replace inside coverage
    # is what the docstring says and what this preserves.
    out = torch.where(m < 0.5, torch.where(a > 0, rgb, base),
                      torch.where(m < 2.5, over,
                                  torch.where(m < 3.5, modu,
                                              torch.where(m < 4.5, mod2,
                                                          torch.where(m < 5.5, addi, prem)))))
    return out


def shade_albedo(uv1, l1, l2, has2, params, blend_w,
                 taps=None, mgate=None, uv_l2=None):
    """Two-layer Source blend: layer1 -> layer2 by per-vertex weight.

    365 of 923 de_inferno materials are csgo_environment_blend; glTF's
    single baseColorTexture is layer 1 only, so ignoring layer 2 renders
    e.g. the terracotta road as grey stone.

    `uv2` AND `sec` ARE GONE, and their absence is the point. Both were
    accepted and neither was ever read in this body: layer 2 samples
    `uv_l2` when gt_uv supplies one and `uv1` otherwise, and the UV-set
    selection is entirely gt_uv's. The call site read
    `shade_albedo(uv_o, uv2_o, ...)` and was named in the uv2_o reach
    trace as one of THREE consumers to instrument -- it was not a
    consumer at all, and a parameter that looks like a wire is worse than
    no wire, because it makes the trace stop at the wrong place. Deleted
    rather than left with a comment, so the signature cannot lie again.
    """
    factor = params[..., 0:3]
    # Per-layer model-tint gate: 1 keeps the factor, 0 makes it neutral.
    f1 = f2 = factor
    if mgate is not None:
        f1 = 1.0 + (factor - 1.0) * mgate[..., 0:1]
        f2 = 1.0 + (factor - 1.0) * mgate[..., 1:2]
    t1, b1, c1, s1 = params[..., 3:6], params[..., 6], params[..., 7], params[..., 8]
    t2, b2, c2, s2 = params[..., 9:12], params[..., 12], params[..., 13], params[..., 14]
    a1 = sample_textures(uv1, l1, taps)
    col1 = color_correct(a1[..., :3] * f1, t1, b1, c1, s1)
    # Layer 2 shares layer 1's tiling UV unless S_SECONDARY_UV gave it
    # its own. Reading F_SECONDARY_UV as "sample layer 2 with the raw
    # TEXCOORD_2" produced a flat crop (a navy floor, a bright green
    # wall) because TEXCOORD_1/2 are 0..1 lightmap-atlas coordinates
    # here; csgo_environment_blend_vs_max.glsl:343-478 shows why that
    # reading was wrong -- the selector picks the STREAM and the slot's
    # own affine transform is what maps it back to tiling space. gt_uv()
    # supplies the transformed coordinate; without --secondary-uv the
    # layer keeps uv1 exactly as before.
    uvB = uv1 if uv_l2 is None else uv_l2
    a2 = sample_textures(uvB, l2, taps)
    col2 = color_correct(a2[..., :3] * f2, t2, b2, c2, s2)
    w = (blend_w * has2).clamp(0, 1).unsqueeze(-1)
    rgb = col1 * (1 - w) + col2 * w
    return torch.cat([rgb, a1[..., 3:]], dim=-1)


def water_axis_selftest():
    """Run water_fancy_shade at EVERY value of EVERY axis the family
    declares and report what each one moves.

    Absence on de_inferno is a USAGE fact.  The family declares
    S_REFLECTION_TYPE 0/1/2, S_REFRACTION, S_CAUSTICS,
    S_INTERACTION_EFFECTS and S_BLUR_REFRACTION; de_inferno's one
    material selects 2/1/1/1/0, so S_REFLECTION_TYPE 0 and 1 and
    S_BLUR_REFRACTION 1 are never selected by the map and would go
    unexercised by any render of it.  This drives them directly.

    It reports max|delta| against the material's own setting.  A row
    whose max|delta| is 0.0 means the axis moved NOTHING and is a
    FAILURE, not a pass -- which is the check's own falsifier: setting an
    axis to the value the material already has must print 0, and every
    other row must not.
    """
    if not FAMILY:
        raise SystemExit("--water-selftest needs --fam-side")
    sel = (MAT_FAM == FAM_WATER_FANCY).nonzero().flatten()
    if not len(sel):
        raise SystemExit(
            "--water-selftest found no csgo_water_fancy material in "
            f"{args.fam_side}. The family is not in the pack, so nothing "
            "the self-test printed would be about this shader.")
    mid = sel[:1].view(1, 1, 1).expand(1, 24, 32).contiguous()
    x = MAT_EXT2[mid]
    # A synthetic plane at a plausible fountain scale, viewed at a
    # grazing angle so the Fresnel, the refraction offset and the SSR
    # march are all in their non-degenerate range.
    yy, xx = torch.meshgrid(torch.arange(24, device=device).float(),
                            torch.arange(32, device=device).float(),
                            indexing="ij")
    wpos = torch.stack([xx * 0.25 - 4.0, yy * 0.25 - 3.0,
                        torch.zeros_like(xx)], -1).unsqueeze(0)
    n_geo = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3) \
        .expand(1, 24, 32, 3).contiguous()
    vcol = torch.full((1, 24, 32, 4), 0.5, device=device)
    scene = (torch.stack([xx / 32.0, yy / 24.0,
                          (xx + yy) / 56.0], -1).unsqueeze(0) * 0.7 + 0.15)
    # gl_FragCoord.z of a surface behind the water plane, varying across
    # the tile so the depth-difference terms are not constant.
    zf = (0.55 + (xx + yy) / 400.0).unsqueeze(0)
    eye_p = torch.tensor([[0.0, -6.0, 1.4]], device=device)
    fwdv = _nrm(torch.tensor([[0.0, 1.0, -0.22]], device=device))

    def run():
        return water_fancy_shade(wpos, n_geo, vcol, mid, x, scene, zf,
                                 eye_p, fwdv, 3.0)

    saved = dict(WATER_OV)
    base_rgb, base_a = run()
    print("csgo_water_fancy combo-axis self-test  (base = the material's "
          "own combo, static 92)", flush=True)
    print(f"  base: rgb range [{float(base_rgb.min()):.5f}, "
          f"{float(base_rgb.max()):.5f}]  alpha range "
          f"[{float(base_a.min()):.5f}, {float(base_a.max()):.5f}]",
          flush=True)
    print("  %-34s %-12s %-12s %s" % ("axis = value", "max|d rgb|",
                                      "max|d a|", "verdict"), flush=True)
    ok = True
    rows = []
    for key, vals in (("reflection_type", (0.0, 1.0, 2.0)),
                      ("refraction", (0.0, 1.0)),
                      ("caustics", (0.0, 1.0)),
                      ("interaction", (0.0, 1.0)),
                      ("blur_refraction", (0.0, 1.0))):
        own = float(MAT_EXT2[sel[0], EXT2["wf_" + key]])
        for v in vals:
            WATER_OV[key] = v
            try:
                r, aa = run()
            finally:
                WATER_OV[key] = saved[key]
            dr_ = float((r - base_rgb).abs().max())
            da = float((aa - base_a).abs().max())
            same = abs(v - own) < 1e-6
            # The falsifier: the value the material ALREADY has must move
            # nothing; every other value must move something.
            good = (dr_ + da == 0.0) if same else (dr_ + da > 0.0)
            ok = ok and good
            rows.append((key, v, dr_, da, same, good))
            print("  %-34s %-12.6g %-12.6g %s"
                  % (f"S_{key.upper()} = {v:g}"
                     + ("  (the material's own)" if same else ""),
                     dr_, da,
                     "ok" if good else
                     ("FAIL: identical to base" if not same
                      else "FAIL: moved, but is the base")), flush=True)
    # The two UNIFORM-bounded loop counts are axes too: the shader reads
    # them from the material and the cost table calls both loops
    # unbounded, so a trip count that changes nothing means the loop body
    # is inert.
    for key, lo, hi in (("wave_iterations", 1, 8), ("ssr_steps", 1, 48)):
        WATER_OV[key] = lo
        try:
            r_lo, a_lo = run()
        finally:
            WATER_OV[key] = saved[key]
        WATER_OV[key] = hi
        try:
            r_hi, a_hi = run()
        finally:
            WATER_OV[key] = saved[key]
        d = float((r_hi - r_lo).abs().max()) + float((a_hi - a_lo).abs().max())
        ok = ok and d > 0.0
        print("  %-34s %-12.6g %-12s %s"
              % (f"trip count {lo} vs {hi}", d, "-",
                 "ok" if d > 0 else "FAIL: the loop body is inert"),
              flush=True)
    print("  water-fancy self-test: " + ("PASS" if ok else "FAIL"),
          flush=True)
    return ok


def _raster(clip, faces_sub, mat_sub, norm_sub, uv_attr, rast_faces):
    """Rasterize one alpha class; returns (rast, fg, uv_pix, mat, norm, uv_da).

    uv_da is nvdiffrast's own screen-space UV derivative quad
    (du/dx, du/dy, dv/dx, dv/dy), produced by asking the rasterizer for the
    barycentric derivative buffer (grad_db) and interpolating the UV
    attribute with diff_attrs. It is None when mipmapping is off so the
    rasterizer can skip the db buffer entirely.
    """
    # SITE 4 of the empty-visible-set family (task #27's rule, one more
    # consumer). The mask and blend CALLERS carry population guards; the
    # opaque caller ran unconditionally, and a foreign camera on
    # de_mirage_vanity left its opaque class with zero faces after
    # visibility selection -- nvdiffrast raises "tri must have shape
    # [>0, 3]". Guarded HERE, in the shared function, so every caller is
    # covered at once: guarding only the raiser that fired moves the
    # crash to the next unguarded caller and makes it look like a
    # different bug, which is exactly how this family got to four sites.
    if rast_faces.numel() == 0:
        B_ = clip.shape[0]
        # print(), NOT frame_print(): frame_print de-duplicates by frame and
        # suppresses the warm-up render, which is exactly the render this
        # guard first fires on -- so the one diagnostic that explains the
        # crash was silent at the only moment it mattered. A rare state that
        # prints nothing is indistinguishable from a state that never
        # happened, and this family already cost four sites of that.
        print(f"_raster skipped: 0 faces in this alpha class after "
              f"visibility selection -- an empty class is a valid "
              f"frame state; the pass contributes nothing.", flush=True)
        return (torch.zeros(B_, H, W, 4, device=clip.device),
                torch.zeros(B_, H, W, dtype=torch.bool, device=clip.device),
                torch.zeros(B_, H, W, uv_attr.shape[-1], device=clip.device),
                torch.zeros(B_, H, W, dtype=mat_sub.dtype,
                            device=clip.device),
                torch.zeros(B_, H, W, norm_sub.shape[-1]
                            if norm_sub.dim() > 1 else 3, device=clip.device),
                None)
    want_db = MIP_ON
    rast, rast_db = dr.rasterize(ctx, clip, rast_faces, (H, W),
                                 grad_db=want_db)
    tri = rast[..., 3].long()
    idx = (tri - 1).clamp(min=0)
    if want_db:
        uv_pix, uv_da = dr.interpolate(uv_attr[None].contiguous(), rast,
                                       rast_faces, rast_db=rast_db,
                                       diff_attrs="all")
    else:
        uv_pix, _ = dr.interpolate(uv_attr[None].contiguous(), rast,
                                   rast_faces)
        uv_da = None
    return rast, tri > 0, uv_pix, mat_sub[idx], norm_sub[idx], uv_da


def _interp(attr, rast, faces):
    """dr.interpolate, with the SAME empty-class rule _raster carries.

    SITE 4b of the empty-visible-set family (#27). _raster's guard covers
    the rasteriser; it does not cover the ~18 places that afterwards
    interpolate a SECOND attribute against the same face list, and
    nvdiffrast refuses `tri` with shape [0, 3] in interpolate exactly as
    it does in rasterize. Guarding only the raiser that fired is how this
    family reached four sites, so the rule is applied to the shared entry
    point rather than to one call.

    An empty class covers no pixel, so every downstream read of the
    result is masked off by an all-False `fg`; zeros is the value that
    masking already treats as absence, not a stand-in for a real one.
    """
    if faces.numel() == 0:
        return (torch.zeros(rast.shape[0], rast.shape[1], rast.shape[2],
                            attr.shape[-1], device=rast.device,
                            dtype=rast.dtype), None)
    return dr.interpolate(attr, rast, faces)


# ======================================================================
# THE VIEWMODEL PASS -- a second camera and a SECOND DEPTH RANGE
# ======================================================================
# Everything above this line shares ONE camera and ONE depth range: `proj`
# (near 0.05, far 800.0, --hfov) and the single (B,H,W) NDC-z tensor
# `depth` that the OPAQUE / MASK / BLEND classes test and write.
#
# A first-person viewmodel is not a model placed near that camera. The
# evidence in the bytecode is direct:
#
#   1. csgo_legs_prepass exists AT ALL, as its own single-combo shader in
#      a different archive (csgo_core), carrying its own fade uniform
#      g_flFirstpersonLegsFade at set 1 / binding 0 / offset 8. A term
#      that only makes sense for first-person legs, in a shader that only
#      draws first-person legs, is a dedicated pass.
#   2. That shader writes a CONSTANT near-black colour, vec4(0.01, 0.01,
#      0.01, 1.0) (csgo_legs_prepass__only_d0:32). A pass whose only output
#      is its own discard is a pass that exists to establish DEPTH.
#   3. csgo_weapon ships S_MODE_DEPTH (stride 1024, record 192) as a
#      separate 26-line depth-only module with the same screen-door
#      discard and the same vec4(0,0,0,1) constant output. The weapon has
#      its own depth-establishing combo, in the same shape as the legs.
#   4. Both depth modules index the SAME dither tile with
#      `ivec2(gl_FragCoord.xy) & mask` and discard on
#      `fma(alpha, 2.0, -1.5) + dither.y < 0` -- one shared stochastic
#      alpha, used by both members of one pass.
#
# A viewmodel drawn into the world's depth range would clip into world
# geometry the moment the player stood against a wall, and at 0.05..800 a
# hand held 30 cm from the eye gets almost no depth precision. So this
# pass gets its own projection, its own near/far, its own depth buffer,
# and composites over the world by COVERAGE -- it never compares against
# world depth. That last property is the one that stops it clipping.
#
# What had to be ADDED to express this: the renderer had exactly one
# `proj`, one hardcoded (0.05, 800.0) repeated at three sites, and one
# `depth` tensor. `DepthRange` below makes the range a named object
# instead of three loose constants, the world's range becomes an explicit
# instance rather than an implicit default, and the viewmodel gets a
# second one. `viewmodel_pass()` owns its own depth tensor for its whole
# lifetime and returns (rgb, coverage) rather than writing into `depth`.


# ======================================================================
# CITATION QUARANTINE -- read this before trusting any `glsl:NNN` below
# ======================================================================
# db11cafd's line citations were taken against a decompiled module that was
# never committed and no longer exists. The module is gone, so the numbers
# cannot be checked, and they are not merely stale -- the two ranges that
# HAVE been re-read landed ~44 lines from the terms they claimed, and one
# whole path (the refraction) was transcribed from a combo that provably
# does not contain it.
#
# So citations in this region are in one of two states, and they say which:
#
#   `allon_c3041_d2:NNN`, `csgo_legs_prepass__only_d0:NNN`,
#   `mode_depth1_c1024_d2:NNN`, `allon_c3041_bgr1_d34:NNN`
#       RE-READ against the committed file under
#       docs/projects/counter-strike-sft/reference/csgo_weapon/. Most also
#       carry a registered conformance term.
#
#   `UNSOURCED-glsl:NNN`
#       INHERITED from the dead module. The offset is unknown. Do NOT
#       transcribe against one of these, and do not cite one as evidence:
#       re-read the range in the committed module first, then re-cite.
#
# The remaining UNSOURCED ranges are the sticker slots, the glitter kernel,
# the hue rotation, the rainbow ramp, the second SFX block and the
# mouse-trace readback. Each is a term still reported ABSENT.
# ======================================================================


class DepthRange:
    """One camera's projection and depth range, named.

    There are two. Making the WORLD one an explicit instance is the point:
    before this the world's near/far was a default argument on
    projection() plus the literals 0.05/800.0 repeated at MBOIT_NEAR/FAR
    and PROJ_NEAR/FAR, which is a single range that nothing could ask a
    question about. A second range cannot be added to that shape without
    first giving the first one a name.
    """

    __slots__ = ("name", "near", "far", "hfov", "proj",
                 "min_depth", "max_depth")

    def __init__(self, name, near, far, hfov,
                 min_depth=0.0, max_depth=1.0):
        self.name = name
        self.near = float(near)
        self.far = float(far)
        self.hfov = float(hfov)
        self.proj = projection(args.width, args.height, hfov, near, far)
        # The VIEWPORT depth remap -- VkViewport.minDepth / .maxDepth. This
        # is a different mechanism from near/far and composes with it: the
        # projection decides where a point lands in the clip volume, the
        # viewport decides what slice of the depth buffer that volume is
        # written into.
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def clip_of_view(self, p_view):
        """view-space points -> clip in THIS range. (N,3) or (B,N,3).

        RANK-AGNOSTIC ON PURPOSE. This indexed `p_view[:, :1]` and
        concatenated on dim=1, which is the same thing only while the input
        is 2-D: handed a batched (B,N,3) it would have taken a (B,1,3)
        slice and appended it along the VERTEX axis, producing a wrong
        shape rather than an error. Per-frame viewmodel posing makes
        (B,N,3) the normal case, so the ellipsis form is the one that
        cannot silently mean something else.
        """
        P = torch.as_tensor(self.proj, device=p_view.device,
                            dtype=p_view.dtype)
        h = torch.cat([p_view, torch.ones_like(p_view[..., :1])], dim=-1)
        return h @ P.T

    def window_depth(self, ndc_z):
        """NDC z -> window depth, through THIS range's viewport remap.

        Vulkan: z_win = minDepth + z_ndc01 * (maxDepth - minDepth), and this
        file's projection is the GL one, so z_ndc01 = z_ndc * 0.5 + 0.5.
        With the default [0,1] viewport this is exactly `_frag_depth`'s
        existing mapping and nothing moves.
        """
        z01 = ndc_z * 0.5 + 0.5
        return self.min_depth + z01 * (self.max_depth - self.min_depth)

    def __repr__(self):
        return (f"DepthRange({self.name}: hfov {self.hfov:g} deg, "
                f"near {self.near:g} m, far {self.far:g} m, "
                f"viewport depth [{self.min_depth:g}, {self.max_depth:g}])")


# The world's range, stated rather than implied. Its numbers are the ones
# already in this file (projection() defaults / MBOIT_NEAR / PROJ_NEAR).
# Viewport depth [0, 0.94999998807907104]: READ from wallA2 -- the world-opaque
# class, 392 of 605 draws. [0, 1] is a different 161-draw class the reference
# reserves for other passes; our world geometry was writing into it.
WORLD_MAX_DEPTH = 0.94999998807907104   # READ, float32-exact, wallA2
WORLD_RANGE = DepthRange("world", 0.05, 800.0, args.hfov,
                         max_depth=WORLD_MAX_DEPTH)

# THE VIEWMODEL'S RANGE, AS THE REFERENCE EXPRESSES IT -- and this replaced a
# mechanism, not a pair of numbers.
#
# This renderer used to compress the viewmodel by giving it a near/far in
# METRES (--viewmodel-near 0.01, --viewmodel-far 4.0), i.e. a second frustum
# scaled to "the few metres a held weapon occupies". That was a plausible
# reconstruction and it is not what CS2 does.
#
# READ from the wallA2 capture: the frame uses exactly five viewport depth
# ranges, and the viewmodel's is a VkViewport minDepth/maxDepth of
# [0, 0.10000000149011612] -- the front tenth of the depth buffer. Same
# frustum as the world; a different slice of the buffer written into. The
# identification of that group as the viewmodel rests on four independent
# lines (W4_PROVENANCE_AUDIT.md sec 5-6): 20 of 20 vkCmdDrawIndexed with zero
# indirect where the world is 71% indirect; W1's binding-0 block-size
# cross-tab putting the 57600 B model region there and the 640/1456 B world
# material blocks at [0.95,1]; W1's world-validated join covering [0.95,1] at
# world-class hit rates; and STANDARD Z, read from the frame's own
# view-projection (472 draw-bindings, z(100u) +0.935 -> z(1e6u) +1.000), which
# puts the near end of the buffer at 0.
#
# The metre-based flags are DELETED rather than kept alongside: a validated
# replacement leaves one mechanism, and two knobs that both claim to control
# viewmodel depth is how a run silently uses the one nobody meant.
VM_MIN_DEPTH = 0.0
VM_MAX_DEPTH = 0.10000000149011612      # READ, float32-exact, wallA2
VM_RANGE = DepthRange("viewmodel", WORLD_RANGE.near, WORLD_RANGE.far,
                      args.viewmodel_hfov,
                      min_depth=VM_MIN_DEPTH, max_depth=VM_MAX_DEPTH)

# THE REFERENCE'S VIEWPORT DEPTH-RANGE PARTITION, as a conformance check.
#
# READ from wallA2: the frame uses exactly FIVE VkViewport minDepth/maxDepth
# ranges, and how the 605 draws divide between them is a structural property
# of the frame that our renderer should reproduce. It is registered as its own
# check because it is the evidence the viewmodel identification rests on, and
# a claim that carries a ticket's config deletion should be re-checkable
# rather than remembered.
#
# Ours cannot match the COUNTS -- we do not issue Vulkan draws and our world
# is one batched raster, not 392 indirect ones. What is comparable is the SET
# of ranges the renderer writes depth into, which is the mechanism. So the
# check is on the set, and the counts are carried as the reference's own
# figures for a reader, explicitly not as our target.
REFERENCE_DEPTH_RANGES = {
    (0.0, 0.94999998807907104): ("world opaque", 392),
    (0.0, 1.0): ("full range", 161),
    (0.94999998807907104, 1.0): ("far slice / 3D skybox", 31),
    (0.0, 0.10000000149011612): ("VIEWMODEL", 20),
    (1.0, 1.0): ("sky, degenerate (ticket #48)", 1),
}


def depth_range_partition_check():
    """Which viewport depth ranges did THIS run actually write into?"""
    ours = {}
    for r in (WORLD_RANGE, VM_RANGE):
        ours[(r.min_depth, r.max_depth)] = r.name
    print("\nVIEWPORT DEPTH-RANGE PARTITION (reference: wallA2, 605 draws, "
          "5 ranges)", flush=True)
    for k, (what, n) in sorted(REFERENCE_DEPTH_RANGES.items()):
        mine = ours.get(k)
        mark = f"<- ours: {mine}" if mine else ""
        print(f"  [{k[0]:.10g}, {k[1]:.10g}]  {what:28s} "
              f"{n:>4d} reference draws  {mark}", flush=True)
    extra = [k for k in ours if k not in REFERENCE_DEPTH_RANGES]
    for k in extra:
        print(f"  [{k[0]:.10g}, {k[1]:.10g}]  OURS ONLY ({ours[k]}) -- not a "
              f"range the reference frame uses", flush=True)
    if extra:
        _gt_note(f"viewport depth ranges: this run writes {len(extra)} range"
                 f"(s) the reference frame does not use: {extra}")
    return ours


VM_REACH = {}
# THE VIEWMODEL'S OWN PIXEL CENSUS, accumulated over the run.
#
# WHY IT HAS TO EXIST SEPARATELY (#103). CLASSMATRIX's pixel column is
# counted from render_window's `fam_pix` attribution map, and that map is
# written at exactly four sites -- the opaque, mask, MBOIT and blend world
# classes. `viewmodel_pass` is called on the FINISHED uint8 frame, after
# class_px_accumulate has already closed the census, and it writes neither
# operand. So no viewmodel pixel of ANY family can appear in that table.
#
# That made csgo_weapon.vfx read 0:0:1 in all eight parity fixtures and the
# zero was taken as evidence that the AK viewmodel is shaded by
# csgo_vertexlitgeneric instead. It is not evidence of anything: the AK's
# own materials NAME csgo_weapon.vfx (read from weapon_rif_ak47.vmat and
# v_models/rif_ak47/ak47.vmat), the bundle's fam column is {0: 26133} which
# IS VM_FAM_WEAPON, and csgo_weapon has no world population on that map at
# all (3 of 49 packs carry it and de_mirage is not one). A structural zero
# read as a routing fact -- the same instrument-scope shape as the
# four-class census that nearly disposed of csgo_foliage.
#
# The viewmodel pass already computes exactly this per frame (`_fam_px`).
# It was thrown away after the per-frame line. Here it accumulates, and
# class_matrix_report prints it BESIDE the world table with the scope
# stated, so a zero in either can no longer be read as the other's answer.
VM_CLASS_PX = {}
# VM family id -> the .vfx the reference binds to that geometry. READ:
# weapons/models/ak47/materials/weapon_rif_ak47.vmat and
# materials/models/weapons/v_models/rif_ak47/ak47.vmat both resolve
# m_shaderName to csgo_weapon.vfx.
VM_FAM_VFX = {0: "csgo_weapon.vfx", 1: "csgo_legs_prepass.vfx",
              2: "csgo_weapon.vfx (arms geometry)"}
WPN_REACH = {}


def _vm_reach(key, mask, total):
    """Record what fraction of the pass's covered pixels took a path."""
    h, t = VM_REACH.get(key, (0.0, 0.0))
    VM_REACH[key] = (h + float(mask.sum()), t + float(total))


def _wpn_reach(key, mask, total):
    h, t = WPN_REACH.get(key, (0.0, 0.0))
    WPN_REACH[key] = (h + float(mask.sum()), t + float(total))


WPN_UNIMPL = {}


def _wpn_unimpl(key, why):
    """Declare an axis that is SELECTED but selects nothing.

    A reach counter answers "what fraction of pixels took this path". For an
    axis no shading decision reads, that question has no answer, and any
    number printed against it -- 100% or 0% -- is read as one. This records
    the axis and its reason so the report can say so in words, which is the
    runtime-uncertainty rule (#34) applied to a flag rather than a constant.
    """
    WPN_UNIMPL[key] = why


# --- the screen-door dither tile --------------------------------------
# csgo_legs_prepass__only_d0:20 declares it as a DEDICATED `texture2D` at
# set 1 / binding 30 -- and that module did NOT need the plain-uniform
# spirv-cross fallback, so its decorations are the real ones. It reads it
# with
#     texelFetch(tex, ivec2(gl_FragCoord.xy) & mask, 0).y
# Every csgo_weapon module reads the same construct but through the
# BINDLESS array at set 4 / binding 46, indexed by a uint member of the
# set-1/binding-3 block (mode_depth1_c1024_d2:12-19,
# allon_c3041_d2:6888, :6932, :6941). Both take channel .y and both mask
# with a power-of-two `&`, so the tile is a power-of-two tile and .y is
# the channel that matters.
#
# STATED DIFFERENCE: the engine's tile is a shipped blue-noise asset that
# is not in any archive we opened. This synthesises a tile of the same
# role, size and channel layout. It is not that asset and is not claimed
# to be; what it reproduces is the screen-door's STRUCTURE -- a per-pixel
# threshold in [0,1) tiled over the framebuffer.
_VM_DITHER = None


def vm_dither_tile():
    """(n, n) float in [0,1) -- the .y channel of the screen-door tile."""
    global _VM_DITHER
    if _VM_DITHER is not None:
        return _VM_DITHER
    n = args.viewmodel_dither_size
    if args.viewmodel_dither == "bayer":
        # recursive Bayer, exact for any power-of-two n
        m = torch.zeros((1, 1), dtype=torch.float32)
        k = 1
        while k < n:
            m = torch.cat([torch.cat([4 * m, 4 * m + 2], dim=1),
                           torch.cat([4 * m + 3, 4 * m + 1], dim=1)], dim=0)
            k *= 2
        _VM_DITHER = (m / float(n * n)).to(device)
    elif args.viewmodel_dither == "hash":
        yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n),
                                indexing="ij")
        h = (xx * 1103515245 + yy * 12345 + 2654435761) % 65536
        _VM_DITHER = (h.float() / 65536.0).to(device)
    else:  # "white"
        g = torch.Generator().manual_seed(0x5EED)
        _VM_DITHER = torch.rand((n, n), generator=g).to(device)
    return _VM_DITHER


def _vm_dither_at(covered):
    """The tile indexed the way both depth modules index it.

    mode_depth1_c1024_d2:19 and csgo_legs_prepass__only_d0:28 both spell
    `texelFetch(tile, ivec2(gl_FragCoord.xy) & mask, 0).y` -- a power-of-two
    AND, not a modulo, and channel .y. Three call sites had this inline; one
    definition means a change to the indexing cannot land in two of them.
    """
    tile = vm_dither_tile()
    n = tile.shape[0]
    B, Hh, Ww = covered.shape
    yy = torch.arange(Hh, device=device) & (n - 1)
    xx = torch.arange(Ww, device=device) & (n - 1)
    return tile[yy][:, xx][None].expand(B, Hh, Ww)


def vm_screendoor(alpha, covered):
    """The shared stochastic-alpha discard, transcribed.

    csgo_legs_prepass__only_d0:28, csgo_weapon mode_depth1_c1024_d2:19-25
    and the tail of every csgo_weapon beauty module
    (allon_c3041_d2:6938-6944) all spell the SAME test -- word for word,
    checked against all three committed files:

        if (alpha < 1.0)
            if (fma(alpha, 2.0, -1.5) + dither.y < 0.0) discard;

    Keep-probability over a uniform dither is therefore
    clamp(2*alpha - 0.5, 0, 1): alpha <= 0.25 is always discarded,
    alpha >= 0.75 always kept. Returns the KEEP mask.
    """
    return covered & wpn.screendoor_keep(alpha, _vm_dither_at(covered))


def vm_proximity_dissolve(wpos, covered):
    """The proximity screen-door in the tail of EVERY csgo_weapon module.

    reference/csgo_weapon/csgo_weapon__allon_c3041_d2.glsl:6928-6936,
    verbatim shape:

        if (g_vDissolve.w > 0.0) {
            f = clamp((smoothstep(6.0, 2.0, distance(g_vDissolve.xyz, P))
                       - 1.0) * -1.0, 0.0, 1.0);
            t = mix(f * 0.51 + 0.5, f * 0.91 + 0.1,
                    smoothstep(0.1, 1.0, g_vDissolve.w));
            if (dither.y - t < 0.0) discard;
        }

    smoothstep(6, 2, d) descends, so f is 0 within 2 units of the origin
    and 1 beyond 6; the threshold therefore rises with distance and the
    fragment dissolves AWAY from the origin. Returns the KEEP mask.

    The origin is engine data (a per-draw constant), so it arrives as
    --weapon-dissolve-origin; the shader's own `if (w > 0)` is honoured,
    which is why a strength of 0 leaves this term entirely alone rather
    than being a flag that quietly disables a path.
    """
    if args.weapon_dissolve_strength <= 0.0:
        return covered
    o = torch.tensor(args.weapon_dissolve_origin
                     if args.weapon_dissolve_origin is not None
                     else (0.0, 0.0, 0.0), device=device)
    thr = wpn.dissolve_threshold((wpos - o).norm(dim=-1),
                                 float(args.weapon_dissolve_strength))
    return covered & wpn.dissolve_keep(_vm_dither_at(covered), thr)
