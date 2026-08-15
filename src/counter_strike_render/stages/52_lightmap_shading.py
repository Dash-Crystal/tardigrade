

def lmg_frame(n_geo, tan4, ts, front=None, flip_bitangent=None):
    """Tangent frame exactly as r0_m3:216-238.

        nf  = vNormal * (backface ? -1 : 1)                        :216
        bt  = cross(nf, vTangent.xyz) * sign(vTangent.w)           :217
        bt  = g_bFlipBitangent ? -bt : bt                          :219-226
        ts.y = frontFacing ? -ts.y : ts.y                          :227-237
        N   = normalize(T*ts.x + bt*ts.y + nf*ts.z)                :238

    Note the ORDER: the bitangent is cross(NORMAL, TANGENT), not
    cross(TANGENT, NORMAL); and the y flip is applied on FRONT faces,
    not back ones. Both are the reference's, and both invert the bump
    if guessed the other way round.
    """
    n = _nrm(n_geo)
    if front is None:
        front = torch.ones_like(n[..., 0], dtype=torch.bool)
    f1 = front.unsqueeze(-1)
    nf = n * torch.where(f1, torch.ones_like(n[..., :1]),
                         -torch.ones_like(n[..., :1]))
    t_raw = tan4[..., :3]
    have = (t_raw.norm(dim=-1, keepdim=True) > 0.5)
    t = _nrm(t_raw - nf * (nf * t_raw).sum(-1, keepdim=True))
    bt = torch.cross(nf, t, dim=-1) * tan4[..., 3:4].sign()
    if flip_bitangent is not None:
        bt = bt * torch.where(flip_bitangent.unsqueeze(-1) > 0.5,
                              -torch.ones_like(bt[..., :1]),
                              torch.ones_like(bt[..., :1]))
    tsy = torch.where(front, -ts[..., 1], ts[..., 1])
    out = _nrm(t * ts[..., 0:1] + bt * tsy.unsqueeze(-1) + nf * ts[..., 2:3])
    return torch.where(have, out, n)


def lmg_albedo(color, ctint, vcol, metal):
    """The albedo product, r0_m3:641 (the right-hand factor).

        (texColor.xyz * g_vColorTint.xyz) * vColor.xyz * (1 - g_flMetalness)

    `1 - g_flMetalness` is the same diffuse-albedo factor csgo_complex
    writes as `albedo*(1 - metal)` at :934, which is how the uniform at
    set 1 / binding 0 / offset 160 is identified -- it enters exactly
    where a metalness would and nowhere else.
    """
    if _lmg_inject("lmg-metal-ignored"):
        metal = torch.zeros_like(metal)
    return color * ctint * vcol * (1.0 - metal).unsqueeze(-1)


def lmg_compose(direct, baked, ao_tex, ssao, albedo, vcol_a):
    """r0_m3:641-646, the whole output of csgo_lightmappedgeneric at
    static combo 0.

        vec3 rgb = (direct + lm0.xyz * (ssao * aoTex.x)) * albedo;   :641
        out      = vec4(rgb, vColor.w);                              :643-646

    `direct` is :374-381 (a constant ambient plus the sun's
    max(0,N.L)*sunColor*atten) plus the clustered-light accumulator
    :403-640. `lm0` is the first lightmap sampler2DArray (:240).
    S_SPECULAR_DIRECT and S_SPECULAR_INDIRECT are both 0 at combo 0, so
    there is no specular term in this module at all -- that is the
    shipped variant, not an omission.
    """
    if _lmg_inject("lmg-albedo-passthrough"):
        return albedo, vcol_a
    occ = (ssao * ao_tex).unsqueeze(-1)
    return (direct + baked * occ) * albedo, vcol_a


# ---------------------------------------------------------------------
# generic.vfx  (csgo_core), static combo 0, record 0
# module r0_m3 = D_BAKED_LIGHTING_FROM_LIGHTMAP (dyn 4), 896 GLSL lines
# ---------------------------------------------------------------------
def generic_normal(nm, strength):
    """generic r0_m3:206-218.

        vec2 t  = texture(g_tNormal, uv).wy * 2.0 - 1.0;
        vec3 n  = vec3(t.x, t.y, sqrt(clamp(1 - dot(t,t), 0, 1)));
        n.y     = -t.y;
        n.xy   *= g_flNormalMapStrength;
        n       = normalize(n);

    DXT5nm: the x channel comes from ALPHA and y from GREEN, the blue
    channel is reconstructed. This is a different encoding from
    csgo_lightmappedgeneric's hemi-oct in the same session's other
    shader, and the y sign flip is inside the decode here but outside it
    there.
    """
    t = nm[..., [3, 1]] * 2.0 - 1.0
    z = (1.0 - (t * t).sum(-1)).clamp(0.0, 1.0).sqrt()
    n = torch.stack([t[..., 0], -t[..., 1], z], dim=-1)
    xy = n[..., :2] * strength.unsqueeze(-1)
    return _nrm(torch.cat([xy, n[..., 2:3]], dim=-1))


def generic_directional_lightmap(irr, dirtex, n_ts, ao_bias):
    """generic r0_m3:270-289 -- the term csgo_lightmappedgeneric has no
    counterpart for at all.

        vec2 d  = dirTex.xy * 2.0 - 1.0;
        d      *= 0.996190845966339111328125
                  / max(0.996190845966339111328125, length(d));      :275
        float z = sqrt(1 - dot(d,d));                                :280
        vec3  L = normalize(vec3(d, z * sharpen));                   :281
        vec3  A = irr * clamp(dirTex.z + g_flLightmapAOBias, 0, 1);   :282
        out     = A + ((irr - A) / L.z) * max(0, dot(L, nTangent));   :284

    The clamp at :275 is 254.03/255: it keeps the encoded direction off
    the unit circle so the sqrt at :280 cannot produce a zero z and the
    divide at :284 cannot blow up. Transcribed as the reference writes
    it rather than as a normalize(), because the two differ exactly in
    the case the constant exists to handle.

    `sharpen` at :281 is a derivative-driven term
    (smoothstep(0.1, 0.01, |fwidth(n)| / |fwidth(P)|) * 0.8 selecting
    between 1.0 and mix(0.1, 2.0, ...)); it is applied by the caller
    through `n_ts` where the derivatives exist and is 1.0 otherwise,
    which is the reference's own value where the smoothstep saturates.
    """
    if _lmg_inject("gen-direction-ignored"):
        return irr
    K = 0.996190845966339111328125
    d = dirtex[..., :2] * 2.0 - 1.0
    ln = d.norm(dim=-1, keepdim=True)
    d = d * (K / torch.maximum(torch.full_like(ln, K), ln))
    z = (1.0 - (d * d).sum(-1)).clamp(min=0.0).sqrt()
    L = _nrm(torch.cat([d, z.unsqueeze(-1)], dim=-1))
    A = irr * dirtex[..., 2:3].add(ao_bias.unsqueeze(-1)).clamp(0.0, 1.0)
    ndl = (L * n_ts).sum(-1).clamp(min=0.0).unsqueeze(-1)
    return A + ((irr - A) / L[..., 2:3].clamp(min=1e-4)) * ndl


def generic_compose(direct, baked, occ, albedo, vcol_a):
    """generic r0_m3:823-828.

        vec3 rgb = (direct + bakedDirectional * occ) * albedo;       :823
        out      = vec4(rgb, vColor.w);                              :825-828

    `occ` is :369-377 -- the SSAO estimate multiplied by a screen-space
    shadow buffer fetch, a single scalar. Note what is NOT here and IS
    in csgo_lightmappedgeneric: no `(1 - metalness)` factor. generic.vfx
    declares no metalness feature at all (its 10 F_ features are
    SPECULAR, UNLIT, ALPHA_TEST, TRANSLUCENT, ADDITIVE_BLEND, OVERLAY,
    RENDER_BACKFACES, DONT_FLIP_BACKFACE_NORMALS, SELF_ILLUM,
    TINT_MASK), so borrowing lightmappedgeneric's albedo product here
    would introduce a factor the shader does not have.
    """
    return (direct + baked * occ.unsqueeze(-1)) * albedo, vcol_a


def lmg_family_shade(rgb, alpha, linear, fam, mid, x, uv, n_geo, tan4,
                     vcol, direct, baked, ssao, ao, fg,
                     lm_u=None, lm_v=None, quality=0):
    """Route each pixel to the shader its MATERIAL FAMILY selects.

    Returns (rgb, alpha, counts). Pixels of any other family come back
    exactly as they went in -- this is a masked replacement, not a
    global change of composite.
    """
    counts = {}
    if not LMG_LIVE or fam is None:
        return rgb, alpha, counts
    sel_l = (fam == FAM_LIGHTMAPPEDGENERIC) & fg
    sel_i = (fam == FAM_IMPORTED) & fg
    sel_g = (fam == FAM_GENERIC) & fg
    counts = {"csgo_lightmappedgeneric.vfx": int(sel_l.sum()),
              LMG_IMPORTED_FAM: int(sel_i.sum()),
              "generic.vfx": int(sel_g.sum())}
    if not (sel_l.any() or sel_i.any() or sel_g.any()):
        return rgb, alpha, counts
    ctint = MAT_CTINT2[mid]
    metal = _e2(x, "lmg_metalness")
    vc = vcol[..., :3] if vcol is not None else torch.ones_like(linear)
    va = (vcol[..., 3] if vcol is not None
          else torch.ones_like(linear[..., 0]))
    # g_flVertexColorOpacityScale (r0_m3 reads vColor.w straight into the
    # output alpha; the scale is the material's own multiplier on it)
    va = va * _e2(x, "lmg_vcol_opacity_scale")
    # An unbound descriptor reads as 1 in the shader, which is what an
    # absent --ao / --ssao means here. Shaped from `fg`, because `ao`
    # itself may be the thing that is missing.
    one = torch.ones_like(fg, dtype=linear.dtype)
    ao_t = one if ao is None else ao
    ss = one if ssao is None else ssao
    if T_L1AO is not None:
        # g_tLayer1AmbientOcclusion.x -- the ONE channel r0_m3:641 reads
        _t = sample_fam(uv, T_L1AO[mid])[..., 0]
        ao_t = torch.where(_e2(x, "has_l1ao") > 0.5, _t, ao_t)
    # --- csgo_lightmappedgeneric AND the csgo_imported placeholder ----
    # They run the SAME module: placeholder.vmat's m_shaderName is
    # csgo_lightmappedgeneric.vfx. They stay separate constants because
    # they are different MATERIAL sets with different uniforms and
    # different textures, and collapsing them would lose the ability to
    # say which of the two a pixel came from.
    sel_lmg = sel_l | sel_i
    if sel_lmg.any():
        alb = lmg_albedo(linear, ctint, vc, metal)
        r2, a2 = lmg_compose(direct, baked, ao_t, ss, alb, va)
        m3 = sel_lmg.unsqueeze(-1)
        rgb = torch.where(m3, r2, rgb)
        alpha = torch.where(sel_lmg, a2, alpha)
    # --- generic.vfx (csgo_core) --------------------------------------
    if sel_g.any():
        b = baked
        dirtex = gen_lm_direction(lm_u, lm_v, quality)
        if dirtex is None:
            # the shader's own encoded-zero direction (see gen_lm_direction)
            dirtex = torch.cat(
                [torch.full_like(baked[..., :2], 0.5),
                 torch.zeros_like(baked[..., :1])], dim=-1)
        # generic.vfx reconstructs against the TANGENT-space normal
        # (:284 dots the decoded direction with the tangent normal, not
        # the world one), so the tangent normal is what is handed over.
        if T_L1NR is not None:
            nts = generic_normal(sample_fam(uv, T_L1NR[mid]),
                                 _e2(x, "lmg_normalmap_strength"))
        else:
            nts = torch.zeros_like(baked)
            nts[..., 2] = 1.0
        b = generic_directional_lightmap(baked, dirtex[..., :3], nts,
                                         torch.zeros_like(metal))
        alb = linear * ctint * vc
        r2, a2 = generic_compose(direct, b, ss * ao_t, alb, va)
        m3 = sel_g.unsqueeze(-1)
        rgb = torch.where(m3, r2, rgb)
        alpha = torch.where(sel_g, a2, alpha)
    return rgb, alpha, counts


def gt_lighting_banner():
    """Say out loud which paths are live and which asset each one got."""
    print("=" * 66, flush=True)
    if not args.gt_lighting:
        print("GT lighting axes: AVAILABLE, NOT SELECTED "
              "(--gt-lighting 1 to shade through them). The pre-existing "
              "composition is running unchanged.", flush=True)
        print("This is now an EXPLICIT CHOICE, not the default: the "
              "default flipped to 1 at integration, which is what the "
              "merge blocker that stood here demanded. You are opting "
              "OUT of every ported axis and shading through the "
              "pre-existing fitted composition instead. Nothing below "
              "this line was measured with the axes live.", flush=True)
        print("=" * 66, flush=True)
        return
    print(f"GT lighting axes ACTIVE", flush=True)
    print(f"  D_BAKED_LIGHTING       : {args.baked_lighting}", flush=True)
    print(f"  S_SHADER_QUALITY       : {args.shader_quality}", flush=True)
    print(f"  S_LIT                  : {args.s_lit}", flush=True)
    print(f"  S_ANIMATED_SHADOWS     : {args.animated_shadows}", flush=True)
    print(f"  D_SPECULAR_CUBE_MAP_STATIC : {args.spec_cube_static}"
          + ("  (axis does not exist at S_SHADER_QUALITY=0; the q=0 "
             "module's env specular is unconditional)"
             if args.shader_quality == 0 else ""), flush=True)
    print(f"  D_FRONT_FACE_CULL      : {args.front_face_cull}   [psrs]  "
          f"CullMode {'0/2' if args.front_face_cull else '0/1'}; "
          f"1-vs-2 front/back UNDETERMINED, running "
          f"--cullmode-12 {args.cullmode_12}", flush=True)
    print(f"  D_DISABLE_DEPTH_BIAS   : {args.disable_depth_bias}   [psrs]  "
          f"depth-pass slope {args.gt_depth_bias_pass:+g} (env/fol/gl); "
          f"material {tuple(args.gt_depth_bias_material)} (cx/eb/so, axis "
          f"has no effect there)", flush=True)
    _o76 = ("WIRED into the whole indirect term" if args.dirocc
            else "OFF -- fitted --ssao/--ao carries the slot")
    _o88 = ("WIRED, min() into the cascade" if args.ss_shadow else "OFF")
    print(f"  offset 76 (--dirocc)   : {_o76}", flush=True)
    print(f"  offset 88 (--ss-shadow): {_o88}", flush=True)
    print(f"  sun cascades           : {args.gt_csm_cascades} @ "
          f"{args.gt_csm}px, PCF "
          f"{'9-tap x3' if args.shader_quality else '1-tap x3'}", flush=True)
    # THE q1-ONLY AXES, PRINTED. Both of these were invisible in every
    # render before #57: one was silently transposed and one was applied
    # to the wrong pages, and a banner that lists only the flags that
    # exist cannot show that. So each prints WHAT IT RESOLVED TO, not
    # that it is on.
    print(f"  lightmap page filters  : irradiance BILINEAR (:809), "
          f"direction BILINEAR (:817), occlusion "
          f"{'BICUBIC (:839)' if (args.shader_quality and args.gt_lm_bicubic == 'on') else 'BILINEAR (:866)'}"
          f"  -- the reference bicubics ONE page and its :277-286 "
          f"density/fwidth gate is NOT evaluated here (no second lightmap "
          f"UV pair)", flush=True)
    print(f"  env BRDF               : "
          f"{'DFG LUT + multiscatter, table axes [sqrt(1-NdotV), roughness]' if args.shader_quality else 'Lazarov polynomial, no multiscatter'}",
          flush=True)
    print(f"  soft terminator (q1)   : {args.gt_soft_terminator}  -- "
          f"shaderquality1.glsl:1113-1119; its selector _5037._m2.w is in "
          f"NO committed artefact, so neither state is READ", flush=True)
    # --- the preset post chain -----------------------------------------
    print(f"  hdr scene attachment   : {args.hdr_scene_format}"
          + ("   (NO attachment store emulated -- this render is at "
             "float32 scene colour, which is no CS2 tier)"
             if args.hdr_scene_format == "off" else
             f"   mantissa {_po_fmt.FORMAT_MANTISSA_BITS[args.hdr_scene_format]}"),
          flush=True)
    print(f"  FSR                    : tier {args.fsr_detail}"
          + (f", scene raster {args.width}x{args.height} -> "
             f"{FSR_OUT_PRE[0]}x{FSR_OUT_PRE[1]}, RCAS "
             + ("sharpness " + str(args.fsr_sharpness)
                if args.fsr_sharpness is not None else "NOT RUN (unread)")
             if FSR_OUT_PRE is not None else "  NATIVE, no upscale pass"),
          flush=True)
    print(f"  texture filtering      : "
          + (_po_tfq.describe(args.texture_filtering_quality)
             if args.texture_filtering_quality is not None
             else f"tier NOT SET -- aniso {ANISO_C}/{ANISO_A}, mip filter "
                  f"{MIP_FILTER}; that combination is NO CS2 tier"),
          flush=True)
    print(f"  CMAA2                  : {args.cmaa}  -- "
          f"r_csgo_cmaa_enable is 0 in all four presets, so this axis "
          f"contributes nothing to the preset ladder", flush=True)
    for g in GT_GAPS:
        print(f"  GAP  {g}", flush=True)
    # This banner runs BEFORE the first frame is shaded, so the list above
    # can only contain notes raised during setup. Every _gt_note() from
    # inside the shading path is raised after this point and would never
    # be seen. gt_gap_epilogue() prints those at the end of the run;
    # GT_GAPS_AT_BANNER is where the two lists divide.
    global GT_GAPS_AT_BANNER
    GT_GAPS_AT_BANNER = len(GT_GAPS)
    gt_uncertainty_banner()
    print("=" * 66, flush=True)


GT_GAPS_AT_BANNER = None


def gt_gap_epilogue():
    """Print the GAPs the banner could not have known about.

    THE BANNER IS PRINTED BEFORE ANYTHING IS SHADED. Every _gt_note()
    raised inside the shading path -- the AO stand-in, the missing
    screen-space shadow buffer, the sun-ambient constant held at 0 --
    lands in GT_GAPS after the banner has already printed, so a
    production render could never show them. --gt-lighting-selftest could,
    because it prints its own list at the end, which is exactly why the
    two disagreed and why the difference read as "that code did not run".
    It ran; the printer went first.

    A diagnostic that cannot report the thing it exists to report is the
    failure this file has a document about. This is the second half of
    the list, labelled as such rather than merged into the first, because
    WHEN a gap was discovered is information: a setup gap is a
    configuration you chose, a shading gap is one the frame found.
    """
    if GT_GAPS_AT_BANNER is None or len(GT_GAPS) <= GT_GAPS_AT_BANNER:
        return
    print("=" * 66, flush=True)
    print(f"GT gaps raised DURING SHADING ({len(GT_GAPS) - GT_GAPS_AT_BANNER}"
          f" of {len(GT_GAPS)} total). The banner above was printed before "
          f"the first frame and could not have listed these.", flush=True)
    for g in GT_GAPS[GT_GAPS_AT_BANNER:]:
        print(f"  GAP  {g}", flush=True)
    print("=" * 66, flush=True)


# =====================================================================
# WHAT THIS RUN IS NOT SURE OF -- printed, not commented
# =====================================================================
# Every term below already carried an accurate "not established from the
# archives" note IN ITS SOURCE COMMENT, and none of it reached stdout. A
# caveat only a code reader sees is not a caveat: an operator reads the
# banner, cites the number, and never learns which parts of it are
# convention. A run should be able to say what it is unsure of without
# anyone opening the file.
#
# ESTABLISHED entries are here for the opposite reason: D_FRONT_FACE_CULL
# and D_DISABLE_DEPTH_BIAS were unestablished when written and are not
# any more, and the supersession is dated with its basis rather than
# quietly edited. Naming the basis in the same place as the open
# questions is what stops the retirement from reading as an omission.
GT_UNESTABLISHED = (
    ("S_BLEND_MODE 3-6",
     "the remaining Source blend states in their CONVENTIONAL order; "
     "§7.2 states the meaning of 3 vs 4 vs 5 vs 6 is not established"),
    ("S_DETAIL_TEXTURE",
     "the value -> mode assignment is NOT established; only that the "
     "axis is 5-valued and that de_inferno uses value 1"),
    ("S_DECAL_TEXTURE",
     "no permutation with it on was decompiled, so the composite is the "
     "STANDARD one and is not read from bytecode"),
    ("S_USE_NEW_BLENDING",
     "combo 0 is old-blending, so the exact new-blending expression is "
     "NOT readable from any committed permutation; this is the closest "
     "complete form"),
    ("S_ENABLE_VISUALIZATIONS",
     "the enum's LABELLING is not in the archives; what is established "
     "is that the axis exists at eb:ps feature index 21"),
)
GT_ESTABLISHED_LATE = (
    ("D_FRONT_FACE_CULL / D_DISABLE_DEPTH_BIAS",
     "transcribed from SHADER_CALLFLOW_psrs_stage.md §4.1-4.3 (opcode "
     "disassembly). These were 'no bytecode exists' when first written; "
     "that is superseded, and the open part is below"),
)


def gt_uncertainty_banner():
    """Print what is convention and what is transcribed, every run."""
    print("  --- NOT ESTABLISHED FROM THE ARCHIVES (implemented so the "
          "axis is complete and a wrong assignment is visible) ---",
          flush=True)
    for _name, _why in GT_UNESTABLISHED:
        print(f"  UNESTABLISHED  {_name}: {_why}", flush=True)
    for _name, _why in GT_ESTABLISHED_LATE:
        print(f"  BASIS          {_name}: {_why}", flush=True)
    # The one sub-question inside an otherwise-established axis. It was
    # already printed at the point of use; it is repeated here so the
    # banner is a complete list rather than a partial one.
    print(f"  OPEN           CullMode 1 vs 2 = front vs back is NOT "
          f"fixed by any shipped data (psrs §4.1); this run used "
          f"--cullmode-12 {args.cullmode_12}", flush=True)


def color_correct(c, tint, bright, contrast, sat):
    """Source 2 per-layer colour correction (csgo_environment_blend)."""
    if args.cc_mode == "off":
        return c.clamp(0, 1)
    if args.cc_mode == "sat":
        # Tint + SATURATION only, no brightness/contrast.
        #
        # The standing belief was that VRF pre-bakes brightness/contrast/
        # saturation into the exported PNG, so the whole chain is dropped.
        # For saturation that is measurably false: 101 materials with
        # geometry declare g_fTextureColorSaturation1 = 0.0 (full
        # desaturation) and their exported textures average saturation
        # 0.2188 -- HIGHER than the 97 materials declaring 1.0 (0.1451).
        # A pre-bake could not produce that ordering in any direction.
        #
        # Brightness is excluded because its magnitudes (up to 8.0, 4.0 on
        # 19 materials) would multiply an albedo this renderer keeps in
        # 0..1, and its evidence is weaker: higher declared brightness
        # correlates with a DARKER texture, which says "applied at shade
        # time" but does not pin the operator.
        c = c * tint
        lum = (c * torch.tensor([0.299, 0.587, 0.114], device=c.device)
               ).sum(-1, keepdim=True)
        return (lum + (c - lum) * sat.unsqueeze(-1)).clamp(0, 1)
    if args.cc_mode == "tint":
        # CORRECTION (this session): the reason recorded here was that VRF
        # pre-bakes brightness/contrast/saturation into the exported PNG.
        # That is measurably FALSE. 101 materials with geometry declare
        # g_fTextureColorSaturation1 = 0.0 -- full desaturation -- and
        # their exported textures average saturation 0.2188, HIGHER than
        # the 97 declaring 1.0 (0.1451); no pre-bake produces that
        # ordering in either direction. Brightness inverts the same way
        # (declared 4.0 -> texture mean 0.491, declared 0.65 -> 0.692).
        #
        # Keeping cc off is still right, but for a different reason: the
        # sun/ambient/sky constants were FITTED to predict GT radiance
        # from this albedo, so the fit already absorbed whatever the
        # correction would contribute and re-applying it double-corrects.
        # The evidence is that the damage scales with how much of the
        # chain you apply -- tint 1.3826, sat 1.3831, full 1.4885 -- which
        # is not how a merely wrong feature behaves. See --light-basis and
        # refit_fam.py, which re-solve the lighting per configuration so a
        # feature is scored against a fit that has NOT compensated for its
        # absence.
        return (c * tint).clamp(0, 1)
    c = c * tint
    c = (c - 0.5) * contrast.unsqueeze(-1) + 0.5
    c = c * bright.unsqueeze(-1)
    lum = (c * torch.tensor([0.299, 0.587, 0.114], device=c.device)
           ).sum(-1, keepdim=True)
    return (lum + (c - lum) * sat.unsqueeze(-1)).clamp(0, 1)


def _ext(e, key):
    """Column `key` of the per-pixel material-parameter block."""
    return e[..., EXT[key]]


def height_blend(h1, h2, w, e, x=None):
    """Source 2 height blend -> (effective layer-2 weight, transition band).

    A linear lerp is the wrong TRANSITION FUNCTION, not merely a wrongly
    parameterised one: CS2 lets the locally higher of the two layers win
    outright over a narrow softness ramp, which is what produces the hard
    interlocking cobble/dirt edges. Everywhere the vertex paint is fully
    0 or 1 all three modes below agree with the lerp exactly, so only the
    ~21% of blend faces carrying partial paint can move.

    threshold    the height DIFFERENCE sets a local threshold that the
                 vertex weight must cross, ramped over g_flBlendSoftness2.
                 w=0 and w=1 are preserved exactly.
    hlerp        the standard height-lerp (max of height+weight, minus a
                 softness window). Does NOT preserve w=0/w=1 when the
                 height maps disagree strongly.
    hlerp-scaled as hlerp but with g_flHeightMapScale1/2 applied; those
                 reach 4.0, which swamps the vertex weight entirely.

    Also returns `band` in 0..1, peaking on the transition itself, which
    the F_BLEND_EFFECTS_2 border tint and bevel are applied along.
    """
    s1 = _ext(e, "h_scale1")
    s2 = _ext(e, "h_scale2")
    z1 = _ext(e, "h_zero1")
    z2 = _ext(e, "h_zero2")
    soft = _ext(e, "blend_soft2").clamp(min=1e-3)
    if args.height_mode == "threshold":
        # 0.5 +- half the height difference: a local threshold in the
        # same 0..1 units as the vertex weight.
        d = ((h1 - z1) - (h2 - z2)) * args.height_contrast
        thr = (0.5 + 0.5 * d).clamp(0.0, 1.0)
        lo, hi = thr - soft, thr + soft
        w_eff = ((w - lo) / (hi - lo).clamp(min=1e-4)).clamp(0, 1)
        w_eff = w_eff * w_eff * (3 - 2 * w_eff)          # smoothstep
        # A fully painted vertex must stay fully painted; the ramp above
        # can otherwise bleed layer 2 into w=0 ground.
        w_eff = torch.where(w <= 0.0, torch.zeros_like(w_eff), w_eff)
        w_eff = torch.where(w >= 1.0, torch.ones_like(w_eff), w_eff)
    else:
        if args.height_mode == "hlerp-scaled":
            a1 = (1 - w) + (h1 - z1) * s1
            a2 = w + (h2 - z2) * s2
        else:
            a1 = (1 - w) + (h1 - z1) * args.height_contrast
            a2 = w + (h2 - z2) * args.height_contrast
        m = torch.maximum(a1, a2) - soft
        b1 = (a1 - m).clamp(min=0)
        b2 = (a2 - m).clamp(min=0)
        w_eff = b2 / (b1 + b2).clamp(min=1e-6)
    if args.use_new_blending and x is not None:
        # S_USE_NEW_BLENDING (csgo_environment_blend m_iFeatureIndex 16).
        # The OLD transition is the softness-window height lerp above.
        # The NEW one, which is what every Source 2 "new blending"
        # material authored after the change uses, drives the window off
        # the height DIFFERENCE and re-centres it on the vertex weight,
        # so a fully painted vertex stays fully painted at any softness:
        #     d  = (h1 - z1)*s1 - (h2 - z2)*s2
        #     t  = clamp((w - 0.5) * 2, -1, 1)
        #     wn = smoothstep(d - soft, d + soft, t)
        # It is 0 on all 43 de_inferno materials (RENDER_SETTINGS_AXES.md
        # line 172), so it changes nothing on this map unless a material
        # sets it. **The exact new-blending expression is not readable
        # from any committed permutation** -- combo 0 is old-blending --
        # so this is the closest complete form: it agrees with the old
        # one at soft -> 0 and preserves w = 0 and w = 1 exactly.
        d = ((h1 - z1) * s1 - (h2 - z2) * s2) * args.height_contrast
        t = ((w - 0.5) * 2.0).clamp(-1, 1)
        lo, hi = d - soft, d + soft
        wn = ((t - lo) / (hi - lo).clamp(min=1e-4)).clamp(0, 1)
        wn = wn * wn * (3 - 2 * wn)
        wn = torch.where(w <= 0.0, torch.zeros_like(wn), wn)
        wn = torch.where(w >= 1.0, torch.ones_like(wn), wn)
        use_new = _e2(x, "f_use_new_blending")
        w_eff = w_eff + (wn - w_eff) * use_new
    # The transition band: 1 where the blend is genuinely partial.
    band = (4.0 * w_eff * (1.0 - w_eff)).clamp(0, 1)
    return w_eff, band


def blend_border(rgb, band, e, btint):
    """F_BLEND_EFFECTS_2 border: a tinted band along the blend seam.

    g_vBorderTint2 is a light grey on most of de_inferno's ground
    materials (0.47-0.91), i.e. this is CS2 depositing pale mortar/dust
    along the cobble-to-dirt seam. Omitting it is one reason a partially
    blended ground reads as flat brown.
    """
    off = _ext(e, "border_offset2")
    spread = _ext(e, "border_spread2").clamp(min=1e-3)
    bsoft = _ext(e, "border_soft2").clamp(min=1e-3)
    # Centre the band at border_offset2 within the transition, width from
    # spread, edge falloff from softness.
    x = (band - off).abs() / (spread + bsoft)
    m = (1.0 - x).clamp(0, 1)
    m = m * m * (3 - 2 * m)
    m = m * _ext(e, "f_blend_effects2") * band
    return rgb * (1 - m.unsqueeze(-1)) + rgb * btint * m.unsqueeze(-1)


def overlay_composite(rgb, ovl, bright, dark, k):
    """g_tSharedColorOverlay, transcribed from the shader that runs it.

    `csgo_environment_blend_ps_c258_d0.glsl:1211-1217` -- static combo
    258, the ONLY combo carrying `S_SHARED_COLOR_OVERLAY`, extracted and
    pinned 2026-08-09 (`vcs/eb_combo.py`, sha256 `d44253733fcc7aa0…`).
    Every earlier form in this file was written against combo 0, where
    the feature is compiled OUT, so there was nothing there to read:

        vec3 s = (texel * 2.0 - 1.0).xyz;
        vec3 f = max(0, (1 - pow(1 - max(0,s), B)) * B
                      + (pow(1 + min(0,s), D) - 1) * D
                      + 1);
        rgb = rgb * mix(vec3(1.0), f, vec3(k));

    B = g_flOverlayBrightnessContrast (block byte 1104), D =
    g_flOverlayDarknessContrast (1100); the names are joined to the
    offsets out of the shader's own m_allVars, not from `_mN` ordinals,
    which spirv-cross renumbers per combo.

    IT IS A MULTIPLY BY A FACTOR AROUND 1.0. The previous implementation
    was a Photoshop "overlay" -- screen above mid-grey, multiply below --
    with B and D as lerp weights. Same two numbers, different function:
    here each of them is an EXPONENT as well as a scale, each appearing
    twice in its own half. Endpoints, evaluated rather than described:
    s = 0 gives f = 1 exactly (so a mid-grey page is the identity, which
    is why AUX_HALF is the right sentinel), s = +1 gives 1 + B, s = -1
    gives 1 - D. On inferno_plaster_facade_01_basic (B 0.20, D 0.65) the
    reachable albedo scale is [0.35, 1.20].

    `k` is the caller's, because the two callers build it from different
    column sets; see the read in docs/.../SHARED_COLOR_OVERLAY.md.
    """
    s = ovl[..., :3] * 2.0 - 1.0
    b = bright.unsqueeze(-1) if bright.dim() == rgb.dim() - 1 else bright
    d = dark.unsqueeze(-1) if dark.dim() == rgb.dim() - 1 else dark
    up = (1.0 - (1.0 - s.clamp(min=0.0)) ** b) * b
    dn = ((1.0 + s.clamp(max=0.0)) ** d - 1.0) * d
    f = (up + dn + 1.0).clamp(min=0.0)
    kk = k.unsqueeze(-1) if k.dim() == rgb.dim() - 1 else k
    return (rgb * (1.0 + (f - 1.0) * kk)).clamp(0, 1)


def shared_overlay(rgb, ovl, e, tmask=None, w=None):
    """The combo-258 composite, driven by the world pack's mat_ext columns.

    `k` is glsl:1188 -- per layer, the layer's own enable times its blend
    weight, times its tint mask only where the material asks for one:

        k = (mask1 ? tm : 1) * L1 * (1-w) + (mask2 ? tm : 1) * L2 * w

    L1/L2 and mask1/mask2 come from `g_nColorOverlayMode` and
    `g_nColorOverlayTintMask`, which are not uniforms at all but
    expression sources: `g_bColorOverlayLayerN = (mode == 0) || (mode ==
    N)` and `g_bColorOverlayMaskLayerN = (tintMask == 0) || (tintMask ==
    N)`. The packer folds those enums into the four ovl_* columns, so
    this reads booleans rather than re-deriving an enum per pixel.

    WHAT THIS CORRECTS, beyond the composite. `g_nColorOverlayTintMask =
    4` is `4=Unmasked` -- the declared enum string in the shader's
    variable table -- and the old code read the value 4 as "gate the
    overlay by the tint mask", which is the opposite instruction, on 8 of
    de_inferno's 11 overlay materials.

    `w` is the layer blend weight; a caller with no layer-2 weight passes
    None and gets the layer-1 term alone.
    """
    on = _ext(e, "f_overlay") * args.overlay_gain
    tm = tmask if tmask is not None else torch.ones_like(on)
    m1 = (_ext(e, "ovl_mask1") > 0.5).float()
    m2 = (_ext(e, "ovl_mask2") > 0.5).float()
    wl = torch.zeros_like(on) if w is None else w
    k = on * ((1.0 - m1 + m1 * tm) * _ext(e, "ovl_l1") * (1.0 - wl)
              + (1.0 - m2 + m2 * tm) * _ext(e, "ovl_l2") * wl)
    return overlay_composite(rgb, ovl, _ext(e, "ovl_bright_contrast"),
                             _ext(e, "ovl_dark_contrast"), k)


def tint_mask_value(tm, e, w):
    """g_tTintMask remapped by g_fTintMaskBrightness/Contrast, per layer.

    Returns the 0..1 mask only; what multiplies it is the caller's
    business (the colour tint for F_TINT_MASK, the overlay strength for
    g_nColorOverlayTintMask).
    """
    b = _ext(e, "tm_bright1") * (1 - w) + _ext(e, "tm_bright2") * w
    c = _ext(e, "tm_contrast1") * (1 - w) + _ext(e, "tm_contrast2") * w
    return (((tm[..., 0] - 0.5) * c + 0.5) * b).clamp(0, 1)


def transmissive_term(uv, mid, e, alb, taps=None, vc=None):
    """S_TRANSMISSIVE_BACKFACE_NDOTL -- the TRANSMITTED COLOUR only.

    Ported from csgo_foliage s32/d0 lines 323-343 (the static-combo pair
    s0/d0 -> s32/d0 isolates this axis exactly); csgo_complex s10280/d2
    carries the identical block.  The shipped code is

        vec4 tex   = texture(g_tTransmissiveColor, uv);
        vec3 tcol;
        if (g_bUseAlbedoForTransmissive != 0)
            tcol = linearAlbedo;                       // line 266, ALREADY
                                                       // the decoded albedo
        else {
            vec3 t = tex.rgb;
            if (g_bTransmissiveColorTransform != 0)
                t = (vec4(t,1) * g_matTransmissive).xyz;
            tcol = t * vColor.rgb;
        }

    and it is added, at line 825, as `rgb += backlight * tcol`, where
    `backlight` is accumulated per light by transmissive_backlight() below.

    THREE THINGS THIS CHANGED, each of which the previous implementation
    had structurally wrong rather than merely differently parameterised:

      1. `use albedo` REPLACES the transmissive texture; it does not
         multiply it. The old form was tex*(1-u) + albedo*tex*u, so on the
         63 of 67 materials that set F_USE_ALBEDO_FOR_TRANSMISSIVE it
         squared the transmission through a 1x1 constant.
      2. The non-albedo branch multiplies by the VERTEX COLOUR, which was
         absent entirely.
      3. The old chain applied `** 2.2` to the result in _post_chain. The
         reference decodes the albedo branch through the material colour
         transform (already linear here, since shade_albedo has decoded)
         and leaves the texture branch alone. The 2.2 is gone, not moved.

    WHAT STILL DIFFERS, precisely: g_matTransmissive, the 4x4 colour
    transform on the texture branch, is not in the pack (the glb carries
    no such matrix), so that branch is the identity -- exact for every
    de_inferno material, because all 67 that set the axis also set
    F_USE_ALBEDO_FOR_TRANSMISSIVE or leave the transform unset.
    """
    use_alb = _ext(e, "f_albedo_for_transmissive").unsqueeze(-1)
    tr_tex = sample_aux(uv, MAT_TR[mid], taps)[..., :3]
    if vc is not None:
        tr_tex = tr_tex * vc[..., :3]
    tr = tr_tex * (1 - use_alb) + alb[..., :3] * use_alb
    return (tr * _ext(e, "f_transmissive_ndotl").unsqueeze(-1)
            * args.transmissive_gain)




def paint_vertex_colors(rgb, vc, x):
    """F_PAINT_VERTEX_COLORS (18 materials, all of them decals).

    On these materials COLOR_0 is not a blend weight at all -- it is a
    per-vertex TINT the level designer painted onto the decal instance.
    All 18 are csgo_static_overlay (12) or csgo_complex (6).

    A primitive with no COLOR_0 packs as (0,0,0), which as a tint would
    turn the decal black, so an all-zero vertex colour is read as "not
    painted" and left at white. That is the packer's sentinel, not a
    fitted guard: fast_pack_shaders defaults COLOR_0 to 0.0 precisely so
    an unpainted primitive means "layer 1 only".
    """
    on = _e2(x, "f_paint_vc").unsqueeze(-1)
    c = vc[..., :3]
    if GTSURF:
        # csgo_complex_vs_max.glsl:289-324 is the axis with
        # S_PAINT_VERTEX_COLORS on, and it settles two details this got
        # wrong. The sentinel is a FOUR-component zero test --
        #     allzero = c.x==0 && c.y==0 && c.z==0 && c.w==0
        #     out = allzero ? tinted : tinted * COLOR_0
        # -- so a decal painted pure black with alpha 1 is PAINTED, not
        # treated as unpainted, which the RGB-only test got backwards.
        # And the multiply is over all four components, alpha included.
        painted = ((vc.abs().sum(-1, keepdim=True)) > 0.0).float()
        tint = c * painted + (1.0 - painted)
        return rgb * (1 - on) + rgb * tint * on
    painted = (c.sum(-1, keepdim=True) > 1e-3).float()
    tint = c * painted + (1.0 - painted)
    return rgb * (1 - on) + rgb * tint * on


def unlit_tint(rgb, x, ctint, e=None):
    """g_vColorTint applied WITHOUT the F_TINT_MASK gate.

    tint_mask() only applies g_vColorTint where F_TINT_MASK is set. Two
    csgo_static_overlay materials carry a non-white tint and NO mask, so
    their tint is currently dropped on the floor. Materials that do have
    a mask are left to tint_mask() so the two features cannot double up.
    """
    if e is not None:
        gate = 1.0 - (_ext(e, "f_tintmask") > 0.5).float()
    else:
        gate = torch.ones_like(rgb[..., 0])
    g = gate.unsqueeze(-1)
    return rgb * (1 - g) + rgb * ctint * g


# =======================================================================
# NON-WORLD FAMILIES: csgo_character / csgo_eyeball / csgo_customglove
# =======================================================================
#
# PROVENANCE.  Everything from here to the end of the block is transcribed
# from SPIR-V pulled out of `csgo_character_vulkan_50_ps.vcs` and
# `csgo_customglove_vulkan_50_{ps,vs}.vcs` in
# **csgo_core/shaders_vulkan_dir.vpk** and `csgo_eyeball_vulkan_50_ps.vcs`
# in `csgo/shaders_vulkan_dir.vpk`, decompiled with spirv-cross (GLSL where
# the block layout is std140-expressible, MSL 2.3 where it is not -- three
# of the character's constant buffers are not expressible in ANY GLSL
# packing, which is why the character listings are Metal and the glove's
# are GLSL).
#
# Reference modules, picked by MEASUREMENT (a FLOP/fetch scan over a
# 40-record sample of the 5,788 bytecode records), not by hoping record 0
# is representative:
#   r5400_m0  aniso gloss + aniso hair + cloth + 3 patches + retro-reflective
#   r3000_m0  the same minus retro, plus S_ALPHA_TEST and S_DECAL_TEXTURE
#   r1050_m0  aniso + hair + cloth + S_TINT_MASK + S_EYEBALLS + S_SUBSURFACE
#   r5145_m4  D_MBOIT_PASS1 with D_MBOIT_4_MOMENTS=0 (6 power moments)
#   r5145_m12 D_MBOIT_PASS1 with D_MBOIT_4_MOMENTS=1 (4 power moments)
# Line numbers below are in those files.
#
# WHAT IS NOT TRANSCRIBED, and is said here rather than buried:
#   * S_SPHERICAL_PROJECTED_ANISOTROPIC_TANGENTS is 0 in every module
#     dumped, so its body was never seen. `char_spherical_tangent()` is the
#     closest complete form, and its docstring says exactly what differs.
#   * S_IRIDESCENCE and S_ENABLE_ADJUSTMENTS are likewise 0 in every module
#     dumped. Same treatment, same disclosure.
#   * The blood pair (g_tBloodMask / g_tColorBlood / g_tNormalBlood) is
#     declared by the package's variable table but sampled by none of the
#     modules dumped; `char_blood()` is a closest complete form.
# None of the three is a stub that returns its input: each computes the
# effect its uniforms name, and each is gated by its own has_* column so a
# material that does not carry the slot is untouched.


# The RGB scattering widths csgo_character's SSS reads from a material slot
# the stripped corpus does not name (r1050_m0:934-940 index `_5618._m17`).
# Red-deepest is the standard ordering and is what is supplied; it is a
# SUBSTITUTED INPUT, printed at runtime by --char-banner, not a read.
SSS_BLEED_SUBSTITUTION = (1.0, 0.3, 0.2)

# printed once per process, not once per pixel batch
_CHAR_DECAL_REFUSED = False
_CHAR_HAIR_REFUSED = False
_CHAR_GTSURF_REFUSED = False


def screen_fwidth(a):
    """`fwidth(a)` = |dFdx(a)| + |dFdy(a)| on a screen-space raster.

    The hardware evaluates this over a 2x2 quad; at 1x rasterization the
    quad difference and a full-resolution finite difference are the same
    construction, and this renderer rasterizes the character pass at 1x.
    Edges replicate, so a one-pixel-wide sliver reports the derivative of
    its interior rather than a zero that would turn the alpha test hard.
    """
    if a.dim() < 3:
        return torch.zeros_like(a)
    dx = torch.zeros_like(a)
    dy = torch.zeros_like(a)
    dx[..., :, :-1] = (a[..., :, 1:] - a[..., :, :-1]).abs()
    dx[..., :, -1] = dx[..., :, -2] if a.shape[-1] > 1 else 0.0
    dy[..., :-1, :] = (a[..., 1:, :] - a[..., :-1, :]).abs()
    dy[..., -1, :] = dy[..., -2, :] if a.shape[-2] > 1 else 0.0
    return dx + dy


def char_normal_decode(tex):
    """2-channel diamond normal decode. Kernel:
    `iji_model/counter_strike_render/passes/character.py:char_normal_decode`, which cites
    csgo_character_ps_r5400_m0.metal:513-515."""
    return _ps_char.char_normal_decode(tex)
