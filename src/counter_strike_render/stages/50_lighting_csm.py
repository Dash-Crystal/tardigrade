

def gt_build_csm(alpha_cut=None, tag=""):
    """Rasterise the sun depth atlas the cascade loop reads.

    THE SHADER SIDE of the cascade axis is the loop at :538-566: it
    indexes g_matWorldToCascade[i] (a mat4[4]) and g_vCascadeExtent[i] (a
    vec4) by the loop counter, which is why the trip count is provably in
    1..4 (SHADER_CALLFLOW_csgo_complex.md:301) -- and those two uniforms
    are engine-side. No committed artefact carries them, so the cascade
    frusta are GENERATED here: C nested world-centred ortho volumes with
    radii from the practical split scheme at --gt-csm-split. What is
    ported verbatim is how the shader CONSUMES them, in gt_csm_shadow().

    S_ANIMATED_SHADOWS (csgo_foliage vs+ps;
    SHADER_CALLFLOW_vertex_stages.md:207 puts it beside S_MODE_DEPTH,
    S_VERTEX_ANIMATION and S_FOLIAGE_ANIMATION on the VERTEX stage) is
    the axis that decides whether the depth/shadow pass runs the same
    vertex animation as the beauty pass. It has two observable halves and
    this renderer can only exercise one of them, so both are stated:

      * caster SILHOUETTE -- at 1 the depth pass honours the caster's
        alpha cutoff, so an alpha-tested leaf or fence casts its cut-out
        shape instead of its quad. IMPLEMENTED and observable: it is the
        alpha_cut argument, and it changes the depth atlas.
      * caster POSE -- at 1 the depth pass is rebuilt from the current
        frame's animated vertices. IMPLEMENTED as a per-frame rebuild
        hook, but this world pack has no time-varying vertex data at all
        (the geometry is one static buffer and only the camera moves), so
        on THIS asset the rebuild returns the same atlas every frame. It
        is not a stub -- the rebuild runs -- but it cannot differ here,
        and saying otherwise would be a lie.
    """
    global GT_CSM, GT_CSM_DEPTH
    lo, hi = vertices.min(0).values, vertices.max(0).values
    center = (lo + hi) * 0.5
    radius = float((hi - lo).norm()) * 0.5 + 1.0
    z = sun_dir / sun_dir.norm()
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if abs(float(z @ up)) > 0.95:
        up = torch.tensor([1.0, 0.0, 0.0], device=device)
    x = torch.cross(up, z, dim=0)
    x = x / x.norm()
    y = torch.cross(z, x, dim=0)
    view = torch.eye(4, device=device)
    view[0, :3], view[1, :3], view[2, :3] = x, y, z
    view[:3, 3] = -(view[:3, :3] @ (center - z * radius))
    C = max(1, min(4, args.gt_csm_cascades))
    lam = args.gt_csm_split
    near = radius / 64.0
    mats, exts = [], []
    for i in range(C):
        f = (i + 1) / C
        # practical split scheme: lambda-blend of the log and uniform
        # splits, the standard generator for the extents the shader reads
        r_log = near * (radius / near) ** f
        r_uni = near + (radius - near) * f
        r = lam * r_log + (1.0 - lam) * r_uni
        o = torch.eye(4, device=device)
        o[0, 0] = 1.0 / r
        o[1, 1] = 1.0 / r
        # THIS SIGN IS WRONG AND IS KNOWINGLY LEFT WRONG. See
        # the cascade depth-sign fix. gt_csm_shadow forms
        # ref = clamp(p.z + g_flShadowBias, 0, 1) -- the shader's [0,1]
        # depth range -- but a negative z scale puts every cascade depth in
        # [-0.8, 0], so clamp() pins every receiver's reference to 0,
        # `lit <=> 0 <= depth` degenerates into a pure COVERAGE test, and
        # the shadow term measures 0.0000 on 100.0% of 518,400 samples:
        # the sun's direct term is multiplied by zero in every frame this
        # renderer has ever produced. It also inverts the z-buffer, since
        # nvdiffrast keeps the SMALLEST z, which here is the surface
        # FARTHEST from the sun rather than the caster.
        #
        # LANDED 2026-08-08 (#38 reopened). The sign is now +1.0.
        #
        # It was previously left at -1.0 because flipping it REGRESSES all
        # three metrics on holdout66 (nrmse 1.2713 -> 1.3243, KL_rgb
        # 3.2713 -> 3.3512, exposure bias 0.6188 -> 0.7167). That is still
        # true, and it is not a reason to keep a known-wrong sign. Two
        # things had to be separated before it could land:
        #
        #   1. The regression's CAUSE is a second charter violation, not
        #      evidence for the old sign. `the cascade depth-sign fix` section 5
        #      states it: the ambient hemisphere was FITTED against a zero
        #      sun, so it already contains the sun's contribution and
        #      restoring the sun double-counts the light. The charter's
        #      standing rule is that constants are READ or DERIVED from
        #      reference inputs and never fitted, so the correct sequence
        #      is to land the reference-correct sign and let the fitted
        #      ambient FAIL LOUDLY, not to keep a coverage test in place
        #      of a depth test so a fitted constant keeps looking right.
        #      Re-deriving the ambient is #61/W3's, and this makes it
        #      measurable instead of theoretical.
        #   2. "Metrics got worse" is not an argument this warp accepts
        #      for leaving the reference unimplemented. The gate passes
        #      when the computation matches; nobody descends a metric
        #      toward it, and nobody holds a computation back to protect
        #      one either.
        #
        # The value is READ, not chosen: `csgo_environment_ps.glsl`:840
        # and :854 clamp the receiver reference to [0, 1] before the
        # sampler2DShadow compare, so the reference's stored cascade
        # depth is in [0, 1] and a negative z scale cannot be what the
        # module expects. +1.0 puts our cascade z in [0, 0.8].
        #
        # It also fixes a second-order consequence: nvdiffrast keeps the
        # SMALLEST z, which under the negative scale was the surface
        # FARTHEST from the sun rather than the caster -- so the atlas was
        # not merely offset, it held the wrong surface.
        o[2, 2] = 1.0 / (2.5 * radius)
        mats.append(o @ view)
        # g_vCascadeExtent[i]. The ortho above maps this cascade's own
        # box to clip space, so the extent the shader tests against is
        # exactly 1 -- the world-space radius r is already in the matrix.
        # Kept as a per-cascade value, and gt_csm_shadow() reads it, so
        # a non-unit extent stays expressible.
        exts.append(1.0)
    GT_CSM = list(zip(mats, exts))
    if args.gt_csm_distance_fade is None:
        # DERIVED FROM THE SCENE, for the reason in the flag's help: the
        # ramp has to land inside the world or the term is identically 0
        # and the :602 fade never runs. 0.8r -> 1.0r.
        # READ, not derived. Byte offsets +31288 and +31292 of the
        # set=3 binding=0 block hold (-19.0, +0.008219). Those are per
        # SOURCE UNIT; this renderer's world is source * 0.0254, so the
        # slope converts and the offset does not.
        #
        # It was previously HELD on the argument that the pair is
        # calibrated to the reference's cascade extents, which we did not
        # reproduce. That argument is dead twice over: the cascades ARE
        # camera-centred (read at vs_base:180) and ours are built that
        # way, and "the reference had this configured differently" is not
        # a reason to keep a generated constant in place of a read one.
        _slope = 0.008219 / S
        args.gt_csm_distance_fade = [-19.0, _slope]
        _k0, _k1 = 19.0 / _slope, 20.0 / _slope
        print(f"GT CSM distance fade READ (+31288/+31292): "
              f"(-19.0, {0.008219:g}) per source unit -> "
              f"(-19.0, {_slope:.6f}) per metre; fully shadowed to "
              f"{_k0:.1f} m, fully lit past {_k1:.1f} m, against a world "
              f"radius of {radius:.1f} m", flush=True)
    res = max(16, int(args.gt_csm))
    sel = alpha_cut if alpha_cut is not None else \
        torch.ones(len(faces0), dtype=torch.bool, device=device)
    # ---- D_FRONT_FACE_CULL (psrs, all six families) ------------------
    # SHADER_CALLFLOW_psrs_stage.md §4.1. The axis selects between two
    # mutually exclusive CullMode variables, both writing render-state
    # field id 1 and differing in one immediate:
    #
    #     D_FRONT_FACE_CULL == 0  ->  CullMode = F_RENDER_BACKFACES ? 0 : 1
    #     D_FRONT_FACE_CULL == 1  ->  CullMode = F_RENDER_BACKFACES ? 0 : 2
    #
    # Liveness is exact by set equality in all six families (240/240 cx,
    # 48/48 env, 96/96 eb, 48/48 fol, 192/192 gl, 32/32 so), and over all
    # 656 0->1 pairs the axis swaps these two variables and NOTHING else.
    #
    # TWO MECHANISMS, and they are not the same one:
    #   F_RENDER_BACKFACES  per-MATERIAL feature -> CullMode 0, cull nothing
    #   D_FRONT_FACE_CULL   per-DRAW dynamic axis -> which side is culled,
    #                       without changing the material
    # The pre-existing --backfaces conflates them; it stays as the
    # per-material lighting two-sidedness it already drives, and the cull
    # is computed here from the material feature instead of from it.
    cull = torch.full((len(faces0),), 2 if args.front_face_cull else 1,
                      dtype=torch.long, device=device)
    _rb_src = "held at 0 (no side table)"
    _g = globals()
    # MATID_OF_FACE, not face_matid0: the latter is a ZERO ARRAY unless PBR
    # or --matid-dump is on, and joining a per-material column through zeros
    # reports material 0's value for every face on the map -- a full-looking
    # wrong answer rather than a visible absence.
    _fmid = _g.get("MATID_OF_FACE")
    if _fmid is None:
        _fmid = _g.get("face_matid0")
    if _g.get("MAT_EXT") is not None \
            and "f_render_backfaces" in _g.get("EXT", {}) \
            and _fmid is not None:
        _rb = (_g["MAT_EXT"][_fmid][
            ..., _g["EXT"]["f_render_backfaces"]] > 0.5)
        cull = torch.where(_rb, torch.zeros_like(cull), cull)
        _rb_src = (f"{int(_rb.sum()):,} of {len(_rb):,} faces"
                   f" (join via {'MATID_OF_FACE' if _g.get('MATID_OF_FACE') is not None else 'face_matid0'})")
    else:
        # `f_render_backfaces` IS in the world pack's ext_keys, unlike
        # f_depth_bias -- so this lookup is against the right table and
        # the cause is that MAT_EXT is unbound, which happens whenever
        # the VMAT block did not run. Name that, rather than reporting a
        # missing column.
        #
        # The material counts that stood here ("1 of 13 csgo_complex and
        # 1 of 2 csgo_foliage") were hardcoded, and when the column IS
        # bound the run reports 0 materials -- the text was quoting a
        # different pack generation than the run it described. Removed
        # rather than re-derived: in this branch there is by definition
        # no bound column to count from, and a count that cannot be
        # measured here is exactly the kind of claim that should not be
        # made here.
        _why = ("no --fam-side/vmat material block is bound (MAT_EXT is "
                "absent), so the column cannot be resolved"
                if _g.get("MAT_EXT") is None else
                "the loaded material block has no f_render_backfaces "
                "column")
        _gt_note(f"D_FRONT_FACE_CULL: {_why}, so CullMode 0 (cull "
                 f"nothing) is never selected and every face is culled "
                 f"by side. Pass the vmat feature flags to bind it and "
                 f"this run will print the measured face count.")
    fn = torch.cross(vertices[faces0[:, 1]] - vertices[faces0[:, 0]],
                     vertices[faces0[:, 2]] - vertices[faces0[:, 0]], dim=1)
    facing = (fn * sun_dir.view(1, 3)).sum(1)
    # WHICH OF 1/2 IS FRONT IS NOT DETERMINED BY THE SHIPPED DATA.
    # §4.1: the packages fix the partition {0} vs {1,2} and the pairing
    # (axis=0 <-> 1, axis=1 <-> 2), but not which of 1/2 is the front
    # face; the doc's own 2->FRONT assignment is the one step in it that
    # leans on the axis's declared name. So the assignment is a FLAG here,
    # not a constant: --cullmode-12 flips it, both settings are reachable,
    # and gt_lighting_selftest() checks the two produce different depth
    # atlases so the ambiguity stays a live fork rather than a buried
    # choice.
    _two_is_front = (args.cullmode_12 == "2-front")
    keep1 = (facing > 0) if _two_is_front else (facing < 0)
    keep2 = (facing < 0) if _two_is_front else (facing > 0)
    if _gt_inject("cullmode-tie"):
        # make CullMode 1 and 2 agree -- i.e. resolve the undetermined
        # front/back question by accident, which is the defect the fork
        # row exists to catch
        keep2 = keep1
    keep_face = torch.where(cull == 0, torch.ones_like(keep1),
                            torch.where(cull == 1, keep1, keep2))
    sel = sel & keep_face
    GT_PSRS[0] = int(sel.sum())
    GT_PSRS[1] = int(len(faces0) - sel.sum())
    GT_PSRS[4] = int((cull == 0).sum())
    print(f"D_FRONT_FACE_CULL={args.front_face_cull}: CullMode "
          f"{'0/2' if args.front_face_cull else '0/1'} -- "
          f"{int((cull == 0).sum()):,} faces CullMode 0 (F_RENDER_BACKFACES, "
          f"{_rb_src}), depth pass keeps {int(sel.sum()):,} of "
          f"{len(faces0):,}", flush=True)
    print(f"  1-vs-2 = front-vs-back is NOT FIXED by the shipped data "
          f"(psrs §4.1); running --cullmode-12 {args.cullmode_12}",
          flush=True)
    f_use = faces0[sel].int().contiguous()
    # #27, the SHADOW-PASS instance of the same zero-geometry class.
    #
    # The audit for #27 looked for empty-list cat/stack sites and this one is
    # not that: `depths` is appended once per `range(C)` and cannot be empty.
    # The hazard here is one step EARLIER and one the empty-list search would
    # have walked straight past -- `f_use` can be zero-face after the cull and
    # keep_face masks, and nvdiffrast raises on a zero-triangle rasterize
    # before the stack is ever reached.
    #
    # Same root cause as the main pass (a view, here the SUN's, containing no
    # geometry), same remedy: render it as what it is rather than dying. A sun
    # that sees no caster casts no shadow, so every cascade is the far value
    # and gt_csm_shadow's taps all read "unoccluded", which is the correct
    # depth atlas for that state and not a substituted one.
    if f_use.numel() == 0:
        print("occupancy: the CSM depth pass sees ZERO caster faces "
              "(sun view empty); every cascade stores the far value and "
              "the frame is unshadowed, which is the correct render of "
              "that state", flush=True)
        _gt_note("CSM: no caster geometry in the sun's view for this frame; "
                 "cascades are all-far and the sun term is unoccluded "
                 "everywhere. Reported so an unshadowed frame is DATA "
                 "rather than an unexplained absence.")
        GT_CSM_DEPTH = torch.full((C, res, res), 1e9, device=device)
        return GT_CSM_DEPTH
    depths = []
    for i in range(C):
        clip = (mats[i] @ vertices_h.T).T.contiguous()[None]
        rast, _ = dr.rasterize(ctx, clip, f_use, (res, res))
        cov = rast[0, ..., 3] > 0
        if alpha_cut is not None:
            # S_ANIMATED_SHADOWS=1 also runs the beauty pass's ALPHA TEST
            # in the depth pass, so an alpha-tested leaf casts its cut-out
            # and not its quad. Interpolate the caster's own UV, fetch the
            # albedo alpha at shadow-map rate, and reject below the
            # material's cutoff exactly as the main pass does.
            uv_s, _ = dr.interpolate(uvs[None].contiguous(), rast, f_use)
            fi = (rast[0, ..., 3].long() - 1).clamp(min=0)
            lay = face_mat0[sel][fi]
            cut = face_cut0[sel][fi]
            am = sample_textures(uv_s[0], lay)[..., 3]
            keep = (face_mode0[sel][fi] != 1) | (am >= cut)
            cov = cov & keep
        z = rast[0, ..., 2]
        # ---- the depth bias, per family (psrs §4.2 / §4.3) ------------
        # TWO distinct, non-interchangeable mechanisms with OPPOSITE SIGN.
        # A single scalar knob cannot express both, which is what the old
        # --gt-depth-bias-{const,slope} stand-ins were doing:
        #
        # (a) MATERIAL bias -- cx, eb, so. Three rasterizer fields gated
        #     on the material feature F_DEPTH_BIAS, ALWAYS live:
        #        DepthBias           = F_DEPTH_BIAS ? -64.0          : 0
        #        SlopeScaleDepthBias = F_DEPTH_BIAS ? -2.0           : 0
        #        DepthBiasClamp      = F_DEPTH_BIAS ? -5.0000002e-04 : 0
        #     D_DISABLE_DEPTH_BIAS changes NOTHING here (0 of 240 cx,
        #     0 of 96 eb, 0 of 32 so).
        #
        # (b) DEPTH-PASS bias -- env, fol, gl. No expression, a constant,
        #     and DepthBias/DepthBiasClamp are not declared at all:
        #        SlopeScaleDepthBias = 5.0
        #        live iff (S_MODE_DEPTH == 1 and D_DISABLE_DEPTH_BIAS == 0)
        #     This IS the depth pass, so S_MODE_DEPTH == 1 holds and the
        #     axis is its per-draw off switch.
        #
        # -2.0 for material decal bias against +5.0 for depth-pass bias:
        # different fields, different passes, opposite polarity. Kept
        # apart rather than normalised.
        fi_b = (rast[0, ..., 3].long() - 1).clamp(min=0)
        gx = torch.zeros_like(z)
        gy = torch.zeros_like(z)
        gx[:, 1:] = (z[:, 1:] - z[:, :-1]).abs()
        gy[1:, :] = (z[1:, :] - z[:-1, :]).abs()
        slope = torch.maximum(gx, gy)
        unit = 1.0 / (1 << 24)          # D3D r for a 24-bit depth buffer
        dz = torch.zeros_like(z)
        fam_t = GT_FAM_OF_FACE[sel][fi_b] if GT_FAM_OF_FACE is not None \
            else None
        if _gt_inject("biasfam-collapse"):
            # collapse the per-family split back to one scalar -- the
            # exact thing --gt-depth-bias-{const,slope} used to do, and
            # what the sign difference between +5.0 and -2.0 forbids
            fam_t = None
        if fam_t is not None:
            # (b) env / fol / gl -- the depth-pass slope bias
            m_b = torch.zeros_like(z, dtype=torch.bool)
            for _f in GT_FAM_DEPTHPASS:
                m_b |= (fam_t == _f)
            if not args.disable_depth_bias:
                dz = torch.where(m_b, args.gt_depth_bias_pass * slope, dz)
            # (a) cx / eb / so -- the material bias, gated on F_DEPTH_BIAS
            m_a = torch.zeros_like(z, dtype=torch.bool)
            for _f in GT_FAM_MATBIAS:
                m_a |= (fam_t == _f)
            if GT_FDEPTHBIAS_OF_FACE is not None:
                m_a = m_a & GT_FDEPTHBIAS_OF_FACE[sel][fi_b]
            db, ss, cl = args.gt_depth_bias_material
            mat_dz = (db * unit + ss * slope)
            # DepthBiasClamp: negative clamp is a lower bound (D3D)
            mat_dz = mat_dz.clamp(min=cl) if cl < 0 else mat_dz.clamp(max=cl)
            dz = torch.where(m_a, mat_dz, dz)
        else:
            _gt_note("depth bias: no per-face family, so the psrs "
                     "per-family split cannot be applied; the depth-pass "
                     "slope bias is used for every face")
            if not args.disable_depth_bias:
                dz = args.gt_depth_bias_pass * slope
        GT_PSRS[2] += int(((dz != 0) & cov).sum())
        GT_PSRS[3] = max(GT_PSRS[3],
                         float(dz[cov].abs().max()) if bool(cov.any())
                         else 0.0)
        z = z + dz
        depths.append(torch.where(cov, z, torch.full_like(z, 1e9)))
    GT_CSM_DEPTH = torch.stack(depths)
    # ---- THE COMPARISON CONVENTION, PRINTED (#34) ---------------------
    # gt_csm_shadow() forms ref = clamp(p.z + gt_csm_bias, 0, 1) and a tap
    # returns lit iff ref <= depth. That is only a DEPTH test if the
    # stored depths share the [0,1] range the clamp assumes. The ortho
    # built above uses o[2,2] = -1/(2.5r) with no z offset, so cascade z
    # is NEGATIVE over the whole world -- and clamp(negative, 0, 1) is 0
    # for EVERY receiver, which turns the depth test into a pure COVERAGE
    # test and makes every gt_csm_bias <= 0 produce the identical term.
    # That is not an inference to leave in a comment: it is the reason a
    # two-sided all-lit/all-shadowed fault came back bit-identical, so
    # the range is measured and printed on every build.
    for _i, _d in enumerate(depths):
        _c = _d < 1e8
        if not bool(_c.any()):
            print(f"  cascade {_i}: EMPTY -- no caster rasterised, so "
                  f"every receiver reads 1e9 and is lit unconditionally",
                  flush=True)
            continue
        _lo, _hi = float(_d[_c].min()), float(_d[_c].max())
        _neg = float((_d[_c] < 0).float().mean()) * 100.0
        # ref = clamp(p.z + bias, 0, 1) can never exceed 1, so ANY covered
        # texel storing depth > 1 satisfies ref <= depth unconditionally
        # and is lit no matter what occludes it. That is a shadow LEAK, and
        # it is the other half of the range mismatch: fixing only the sign
        # leaves it in place.
        _over = float((_d[_c] > 1.0).float().mean()) * 100.0
        print(f"  cascade {_i} stored depth over covered texels: "
              f"[{_lo:+.4f}, {_hi:+.4f}], {_neg:.1f}% negative, "
              f"{_over:.1f}% ABOVE 1.0 (unconditionally lit: ref is "
              f"clamped to <=1)", flush=True)
        if _neg > 0.0:
            _gt_note(
                f"cascade {_i}: {_neg:.1f}% of covered texels store a "
                f"NEGATIVE depth, but gt_csm_shadow clamps the receiver "
                f"reference to [0,1]. Every gt_csm_bias <= 0 (the default "
                f"is {args.gt_csm_bias:+g}) therefore yields ref == 0, and "
                f"lit <=> (0 <= depth) is a COVERAGE test, not a depth "
                f"test. all-lit and the baseline are the same render.")
    _db, _ss, _cl = args.gt_depth_bias_material
    print(f"D_DISABLE_DEPTH_BIAS={args.disable_depth_bias}: depth-pass "
          f"SlopeScaleDepthBias {args.gt_depth_bias_pass:+g} on env/fol/gl "
          f"{'DISABLED by the axis' if args.disable_depth_bias else 'live'}"
          f"; material bias ({_db:+g}, {_ss:+g}, {_cl:+.7g}) on cx/eb/so "
          f"where F_DEPTH_BIAS -- the axis does not touch those "
          f"(psrs §4.2: 0 of 240/96/32)", flush=True)
    cov = [float((d < 1e8).float().mean()) for d in depths]
    print(f"GT CSM{tag}: {C} cascades @ {res}x{res}, radii "
          f"{[round(float(1.0 / m[0,0]), 1) for m in mats]} m, coverage "
          f"{['%.1f%%' % (c*100) for c in cov]}", flush=True)


GT_SHADOW_CASTERS = None
if args.gt_lighting or args.gt_lighting_selftest:
    if args.animated_shadows:
        # S_ANIMATED_SHADOWS=1: the caster set honours the beauty pass's
        # alpha class. MASK (alpha-tested) geometry keeps its cut-out
        # silhouette; BLEND (decals) casts nothing, which is what
        # F_DO_NOT_CAST_SHADOWS does to them in the material.
        # face_mode0 is per-face and rides the same `order` permutation
        # faces0 does (see the cluster sort), so they stay aligned. BLEND
        # faces (decals) are dropped from the caster set outright; MASK
        # faces stay and are alpha-tested per shadow texel in
        # gt_build_csm().
        GT_SHADOW_CASTERS = face_mode0 != 2
        print(f"S_ANIMATED_SHADOWS=1 caster set: "
              f"{int(GT_SHADOW_CASTERS.sum()):,} of {len(faces0):,} faces "
              f"({int((face_mode0 == 2).sum()):,} BLEND dropped, "
              f"{int((face_mode0 == 1).sum()):,} MASK alpha-tested per "
              f"shadow texel)", flush=True)
    gt_build_csm(GT_SHADOW_CASTERS,
                 " (S_ANIMATED_SHADOWS=1)" if args.animated_shadows else "")

GT_VLIT = None
if (args.gt_lighting or args.gt_lighting_selftest) and \
        args.baked_lighting in ("vertex-stream",):
    # Only built when the axis is SELECTED: the bake is a full pass over
    # every vertex and the attribute is dead weight otherwise. Selecting
    # the axis always builds it -- there is no path where
    # --baked-lighting vertex-stream runs without a stream.
    GT_VLIT = gt_bake_vertex_stream()


_CSM_DIAG_DONE = False
_CSM_DIAG_N = 0


def gt_csm_shadow(wpos, eye, quality):
    """The cascade loop and everything downstream of it, transcribed.

    csgo_complex_ps.glsl:531-617 (q=0); the q=1 file is the same control
    flow with gt_pcf() at quality 1 in place of each single tap
    (shaderquality1.glsl:605-715).

      for (i = 0; i < g_nCascades; ++i) {                        // :540
          p = vec4(wpos, 1) * g_matWorldToCascade[i];            // :547
          if (max(abs(p.x), abs(p.y)) < g_vCascadeExtent[i]) {   // :549
              f  = vec2(1) - clamp(abs(p.xy)*m18 + vec2(m17),0,1);// :553
              uv = p.xy * g_vCascadeAtlas[i].zw
                        + g_vCascadeAtlas[i].xy;                 // :554
              fade = clamp(f.x * f.y, 0, 1);                     // :558
              found = i; break;                                  // :561
          }
      }
      s = found >= 0 ? T(uv, clamp(p.z + g_flShadowBias, 0, 1)) : 1.0;
                                                                 // :570
      if (fade < 1 && found < g_nCascades - 1)                   // :573-576
          s = mix(T(next cascade), s, fade);                     // :590
      s = mix(s, 1.0,
              clamp(distance(wpos, g_vEyePos) * m20 + m19, 0, 1));// :602
      if (g_bScreenSpaceShadows) s = min(s, SSS(fragcoord).z);   // :606

    The `break` at :561 makes the FIRST cascade that contains the point
    win, which is the tightest one because the cascades are nested; the
    vectorised form below reproduces that by taking the lowest index
    whose extent test passes.

    NOT IN THE ASSET: g_vCascadeAtlas (each cascade has its own depth
    surface here, so the atlas transform is the identity), and the
    screen-space shadow buffer at :606, which is a per-view render target
    this renderer never produces -- with no buffer, min(s, 1) = s, the
    shader's own value when g_bScreenSpaceShadows is 0.
    """
    if GT_CSM is None or GT_CSM_DEPTH is None or _gt_inject("csm-dead"):
        return torch.ones(*wpos.shape[:-1], device=wpos.device)
    _gt_note("csm: the screen-space shadow buffer "
             "(csgo_complex_ps.glsl:606) is a per-view render target this "
             "renderer does not produce; min(s, ssshadow.z) is skipped, "
             "which is the shader's g_bScreenSpaceShadows=0 branch")
    wh = torch.cat([wpos, torch.ones_like(wpos[..., :1])], dim=-1)
    C = len(GT_CSM)
    found = torch.full(wpos.shape[:-1], -1, dtype=torch.long,
                       device=wpos.device)
    uvz = torch.zeros(*wpos.shape[:-1], 3, device=wpos.device)
    fade = torch.ones(wpos.shape[:-1], device=wpos.device)
    m17, m18 = args.gt_csm_fade
    for i in range(C - 1, -1, -1):
        # descending, so the LOWEST passing index ends up written last --
        # the shader's `break` on the first hit.
        # :819 and :820 through passes/lighting.py. NOTE, in the same
        # spirit as the sky.cube_face_uv row: these two now agree with the
        # reference EXPRESSION while consuming cascade matrices WE
        # generate, so a passing term says the algebra matches and says
        # nothing about the matrices. That gap is characterised rather
        # than open -- §14 established the reference's cascades are
        # camera-centred and ours are built to be -- but it is not closed,
        # and the term's value is regression until the builder matches.
        _M = GT_CSM[i][0].t()
        p = _lighting.cascade_project(wh, _M[0], _M[1], _M[2], _M[3])
        inside = _lighting.cascade_inside(p[..., 0], p[..., 1], GT_CSM[i][1])
        # :553-558 -- lifted to passes/lighting.py so the conformance
        # registry compares the DECOMPILED expression against this same
        # code object rather than against a retyped copy of itself.
        fi = _lighting.cascade_fade(p[..., 0], p[..., 1], m17, m18)
        found = torch.where(inside, torch.full_like(found, i), found)
        # environment_ps:824 via passes/lighting.py. Our cascades each own
        # a depth surface, so the reference's atlas sub-rect is the
        # identity here -- centre 0.5, half-extent 0.5, which reproduces
        # the *0.5+0.5 this line used to inline. Routing it through the
        # registered term means the expression the harness scores is the
        # expression the renderer runs; when the real sub-rects land
        # (the read sub-rects are exact 1280px pixel rectangles in a 4352x5248 atlas) only
        # the constant below changes.
        _sub = torch.tensor([0.5, 0.5, 0.5, 0.5], device=p.device)
        _uv = _lighting.cascade_atlas_uv(p[..., :2], _sub)
        uvz = torch.where(inside.unsqueeze(-1),
                          torch.stack([_uv[..., 0], _uv[..., 1],
                                       p[..., 2]], dim=-1), uvz)
        fade = torch.where(inside, fi, fade)
    hit = found >= 0
    _gt_count("csm.in_cascade", hit)
    idx = found.clamp(min=0)
    # :570 / environment_ps:840 -- lifted, see passes/lighting.py
    ref = _lighting.shadow_compare_ref(uvz[..., 2], args.gt_csm_bias)
    s = torch.ones_like(fade)
    for i in range(C):
        m = idx == i
        if not bool((m & hit).any()):
            continue
        si = gt_pcf(GT_CSM_DEPTH[i], uvz[..., 0], uvz[..., 1], ref, quality)
        s = torch.where(m & hit, si, s)
    # :573-590 -- blend into the NEXT cascade across the fade band
    blend = hit & (fade < 1.0) & (found < C - 1)
    _gt_count("csm.cascade_blend", blend)
    if bool(blend.any()):
        s2 = torch.ones_like(s)
        for i in range(C - 1):
            m = blend & (idx == i)
            if not bool(m.any()):
                continue
            p = torch.einsum("ij,...j->...i", GT_CSM[i + 1][0], wh)
            r2 = _lighting.shadow_compare_ref(p[..., 2], args.gt_csm_bias)
            sn = gt_pcf(GT_CSM_DEPTH[i + 1], p[..., 0] * 0.5 + 0.5,
                        p[..., 1] * 0.5 + 0.5, r2, quality)
            s2 = torch.where(m, sn, s2)
        s = torch.where(blend, s2 + (s - s2) * fade, s)
    s = torch.where(hit, s, torch.ones_like(s))
    # :602 -- distance fade to fully lit
    d = (wpos - eye.view(-1, 1, 1, 3)).norm(dim=-1) if eye is not None \
        else torch.zeros_like(s)
    k = (d * args.gt_csm_distance_fade[1]
         + args.gt_csm_distance_fade[0]).clamp(0, 1)
    if _gt_inject("distfade-full"):
        k = torch.ones_like(k)
    _gt_count("csm.distance_faded", k > 0)
    out = s + (1.0 - s) * k
    # What the sun term is actually MULTIPLIED BY, once, on the first
    # shaded batch. The atlas-side print above establishes the convention
    # mismatch; this establishes its CONSEQUENCE in screen space, which is
    # the part an atlas-coverage percentage can only suggest.
    global _CSM_DIAG_DONE, _CSM_DIAG_N
    if not _CSM_DIAG_DONE:
        _CSM_DIAG_DONE = True
        print(f"  CSM shadow term, first shaded batch ({out.numel():,} "
              f"samples): mean {float(out.mean()):.4f}, "
              f"{100.0 * float((out < 0.01).float().mean()):.1f}% fully "
              f"shadowed, {100.0 * float((out > 0.99).float().mean()):.1f}% "
              f"fully lit, {100.0 * float(hit.float().mean()):.1f}% inside "
              f"a cascade", flush=True)
    # PER-CALL, so the term can be split by pose rather than reported
    # frame-wide. A frame-wide "38.4% lit" cannot distinguish "the sun
    # lights the exteriors it should" from "the sun is also indoors",
    # and those differ by whether the caster set contains the roofs.
    # One line per call, parsed by pose downstream.
    _CSM_DIAG_N += 1
    print(f"CSMTERM {_CSM_DIAG_N} mean {float(out.mean()):.6f} "
          f"lit {float((out > 0.99).float().mean()):.6f} "
          f"shadowed {float((out < 0.01).float().mean()):.6f}", flush=True)
    return out


# =====================================================================
# S_LIT / direct light BRDF -- the OTHER half of S_SHADER_QUALITY
# =====================================================================
def gt_direct(quality, n, l, v, f0, rough, light_rgb, atten, mat_ao=None):
    """Sun / local-light response at the selected S_SHADER_QUALITY.

    These are two different BRDFs, not one scaled. Both transcribed.

    q=0, csgo_complex_ps.glsl:625-636 --
        NdotL = max(0, dot(n, l));
        a     = max(rough, g_vSunDir.w);           // light softness
        h     = normalize(l + v);
        nh    = dot(h, n);   lh = max(0, dot(l, h));
        a2 = a*a;  a4 = a2*a2;
        den   = nh*nh*(a4 - 1) + 1;
        spec  = f0 * (a4 / ((den*den) * (lh*lh) * (a*4 + 2))) * NdotL;
        diff  = g_vSunAmbient + NdotL * (g_vSunColour * shadow);
    There is NO Fresnel and no separate visibility term: the (4a+2) and
    the lh*lh in the denominator ARE the combined D*V approximation.

    q=1, shaderquality1.glsl:728-743 --
        t   = a2 / (nh*nh*(a2*a2 - 1) + 1);        // GGX D, factored
        k   = (a + 1)^2 * 0.125;                   // Disney's k
        DV  = (t*t) / (4 * (NdotL*(1-k) + k) * (max(0,dot(n,v))*(1-k) + k));
        F   = f0 + (1 - f0)*pow(max(1e-6, 1 - max(0, dot(l, h))), 5);
        spec = F * DV * NdotL;
    i.e. GGX D x Smith-Schlick visibility x Schlick Fresnel, full
    Cook-Torrance. Neither variant carries the 1/pi.

    The diffuse half is the same at both qualities; the ROLES of the two
    accumulators swap between the two files (q=0 puts specular in the
    slot q=1 puts diffuse in, :635-636 against :742-743, and q=1 swaps
    them back at :1051-1052), which is bookkeeping, not a difference, and
    is why this function returns them by name.
    """
    ndl = (n * l).sum(-1).clamp(min=0)
    a = rough.clamp(min=1e-3)
    h = _nrm(l + v)
    nh = (h * n).sum(-1)
    lh = (l * h).sum(-1).clamp(min=0)
    a2 = a * a
    if quality == 0:
        a4 = a2 * a2
        den = (nh * nh * (a4 - 1.0) + 1.0)
        dv = a4 / ((den * den) * (lh * lh) * (a * 4.0 + 2.0)).clamp(min=1e-8)
        spec = f0 * (dv * ndl).unsqueeze(-1)
    else:
        t = a2 / (nh * nh * (a2 * a2 - 1.0) + 1.0).clamp(min=1e-8)
        k = (a + 1.0) * (a + 1.0) * 0.125
        ik = 1.0 - k
        dv = (t * t) / (4.0 * (ndl * ik + k).clamp(min=1e-4)
                        * ((n * v).sum(-1).clamp(min=0) * ik + k
                           ).clamp(min=1e-4))
        fr = f0 + (1.0 - f0) * (1.0 - lh).clamp(min=1e-6).unsqueeze(-1) ** 5
        spec = fr * (dv * ndl).unsqueeze(-1)
    # --- q1-ONLY: the SOFT TERMINATOR, shaderquality1.glsl:1113-1119 ----
    # A term q0 has no counterpart for at all, and one this renderer did
    # not have on either side:
    #     s = smoothstep(0, 1, clamp(dot(n,l) / sqrt(max(1 - matAO, 1e-5)),
    #                                0, 1));
    #     spec *= s;  diff *= s;
    # It widens the sun's terminator in proportion to the material's own
    # ambient occlusion, so an occluded surface turns away from the sun
    # gradually instead of at NdotL = 0. Applied to BOTH halves.
    #
    # THE SELECTOR IS UNREAD. The shader gates it on
    # `notEqual(_5037._m2, ivec4(0)).w` -- a feature-flag component q0
    # never reads and that is not in any committed artefact, so
    # --gt-soft-terminator carries the choice and DEFAULTS OFF. Off is
    # not "unimplemented": the expression is here, it is exercised by the
    # conformance registry, and the banner prints that the reference's
    # own selector for it could not be read.
    #
    # matAO is the composer's occlusion, which is ssao * aoTex * matAO;
    # the shader's `_17119` is the MATERIAL scalar alone. Ours is the
    # tighter quantity (never larger), so this runs a terminator at least
    # as soft as the reference's, and that difference is stated here
    # rather than being papered over by calling both of them "ao".
    if quality and mat_ao is not None and args.gt_soft_terminator == "on":
        s = ((n * l).sum(-1)
             / (1.0 - mat_ao).clamp(min=1e-5).sqrt()).clamp(0, 1)
        s = s * s * (3.0 - 2.0 * s)                      # smoothstep(0,1,x)
        _gt_value("sun.soft_terminator", s.unsqueeze(-1), identity=1.0,
                  note="shaderquality1.glsl:1113-1119, q1-only. Its "
                       "selector _5037._m2.w is NOT in any committed "
                       "artefact; --gt-soft-terminator carries it and "
                       "defaults off.")
        spec = spec * s.unsqueeze(-1)
        soft = s.unsqueeze(-1)
    else:
        soft = None
    # THE ADDEND, at the position the shader defines for it. The buffer
    # member is identified by its DECLARED BYTE OFFSET, not its ordinal:
    # csgo_environment_ps.glsl:155 `layout(offset = 288) vec4 _m3`, the
    # same 288 that csgo_complex_ps.glsl spells `_m4` and
    # shaderquality1.glsl spells `_m9`. Sun colour is the neighbouring
    # `layout(offset = 31136) vec4 _m9` (:161), which is `light_rgb`.
    #
    # It is added OUTSIDE the light-colour multiply, which the previous
    # comment here got wrong: it wrote the slot as `g_vSunAmbient +
    # NdotL`, i.e. inside, and all three files add it outside.
    #
    #   env :918   _24880 = _5538._m3.xyz + (_18223.xyz * _16808);
    #   env :923   _24880 = _5538._m3.xyz;              // not-lit branch
    #   complex:636 _24878 = _5538._m4.xyz + (vec3(_11797) * _16808);
    #   sq1   :1125 _9719  = _5538._m9.xyz + (_24880.xyz * _16808);
    #
    # with _16808 = sunColour * shadow in every one. So the addend is
    # unattenuated and unshadowed, and the not-lit branch is this same
    # expression at atten = 0 -- which is why no branch is needed here:
    # ndl is already clamped at 0 and atten is already 0 there.
    #
    # Placement matters by a factor of the sun colour: de_dust2's
    # SUN_COLOR reads [2.5, 1.826152, 1.272203], so `(amb + ndl) * lc`
    # would have shipped the addend 2.5x too bright in red and 1.27x in
    # blue -- a hue shift as well as a level shift, and therefore not
    # something the ncc could have caught.
    #
    # SUN_AMBIENT is module-scope and carries the --sky-ambient-bounce
    # arm; the expression is passes/lighting.py:sun_diffuse, the code
    # object the conformance registry scores, called rather than
    # re-transcribed. THIS IS THE --gt-lighting BRANCH (gt_compose ->
    # gt_direct); the separate wire in the LMG_LIVE block below the
    # gt_on/else split feeds the same function for the three
    # lightmappedgeneric families' pixels only.
    lc = light_rgb * atten.unsqueeze(-1)
    amb = SUN_AMBIENT.view(*((1,) * ndl.ndim), 3)
    _gt_value("sun.ambient_addend", amb, identity=0.0,
              note="csgo_environment_ps.glsl:918 first addend, the member "
                   "declared at BYTE OFFSET 288. Carries the "
                   "--sky-ambient-bounce arm: `read` puts the map's own "
                   "skyambientbounce here, `zero` holds it at 0. Added "
                   "after the sun-colour multiply and before the albedo "
                   "multiply, so a non-zero value lifts every lit AND "
                   "every unlit pixel of the world families")
    diff = _lighting.sun_diffuse(amb, ndl.unsqueeze(-1), light_rgb, atten)
    if soft is not None:
        # :1118-1119 multiplies BOTH halves. The specular half was
        # multiplied at the branch; the diffuse half is here, because
        # sun_diffuse() is the registry-scored code object and is called
        # rather than re-transcribed.
        diff = diff * soft
    return diff, spec * lc


# ---------------------------------------------------------------------
# environment BRDF -- the third face of S_SHADER_QUALITY
# ---------------------------------------------------------------------
GT_DFG_LUT = None


def _gt_build_dfg_lut(n=64, samples=256):
    """The sampler2DArray the q=1 path reads instead of the polynomial.

    shaderquality1.glsl:1170 --
        vec4 lut = textureLod(sampler2DArray(g_tDFG, s),
                              vec3(vec2(rough, sqrt(1 - max(0, dot(v, n))))
                                   * 0.984375 + 0.0078125, 1.0), 0.0);
        specResponse = mix(lut.xxx, lut.yyy, f0);
    mix(x, y, f0) with the q=0 packing vec2(B, A+B) gives B + f0*A, the
    same split-sum form the q=0 polynomial produces -- so lut.x = B and
    lut.y = A + B, and the second coordinate is sqrt(1 - NdotV), not
    NdotV. The 0.984375/0.0078125 remap is the standard half-texel inset
    for a 64-wide table (1 - 1/64, 1/128).

    WHAT DIFFERS: Valve's texture is not in this repo. The TABLE is built
    here by numerically integrating the same split-sum DFG with GGX
    importance sampling, so the contents are mine and the lookup, the
    coordinates, the packing and every consumer are the shader's. A LUT
    and a polynomial fit of the same integral do not agree to zero; the
    q=0 Lazarov polynomial is the published fit of this integral, and
    --gt-lighting-selftest prints max|delta| between the two so the
    disagreement is a number and not an adjective.

    ⚠️ THE AXIS ORDER IS THE TABLE'S IDENTITY, AND IT WAS TRANSPOSED (#57).
    ------------------------------------------------------------------
    `_gt_tex2d(tex, u, v)` indexes `tex[y, x]` with x from `u` and y from
    `v`, and `gt_env_brdf` passes `u = rough`, `v = sqrt(1 - NdotV)`. This
    builder used to `meshgrid(r, ndv, indexing="ij")`, returning
    `[rough, ndv, 2]` -- so the roughness coordinate addressed the NdotV
    AXIS and vice versa, and the `sqrt(1 - NdotV)` warp was applied to an
    axis parameterised by raw NdotV. Both axes crossed at once.

    Measured effect at f0 = 0.04, same integrator, same tap: a head-on
    smooth surface (rough 0.10, NdotV 0.95) read 0.4888 where the correct
    orientation reads 0.0400 -- 12x over-bright; a grazing one (rough
    0.10, NdotV 0.10) read 0.0447 against 0.6002 -- 13x under. And since
    q=1 ALONE feeds this back as `1 - E` on the diffuse (gt_compose), it
    darkened the dominant baked term by up to 23% on exactly the pixels
    q=0 leaves alone. That is a q=1-ONLY regression by construction, and
    it is the mechanism behind #57's "our q1 loses to our q0".

    Nothing caught it because the coordinate expression was unit-tested
    (csgo_environment_terms.py) while the TABLE IT ADDRESSES was not, and
    the lighting selftest printed a delta it never asserted on. So this
    function now runs its own ORIENTATION CHECK at build time and prints
    it -- see `_gt_dfg_orientation_check`. A transposed table is loud.

    Axis order, stated once so it cannot drift again:
        axis 0 (y, addressed by `v`) = t, and NdotV = 1 - t*t
        axis 1 (x, addressed by `u`) = roughness
    """
    # ⚠️ AND THE SAMPLE POSITIONS ARE SET BY THE SHADER'S OWN REMAP.
    # The lookup is `c * 0.984375 + 0.0078125` = `c*(63/64) + 0.5/64`, and
    # `_gt_tex2d` resolves that to `x = c*(N-1) = 63c`. So texel j holds
    # the integrand at c = j/63 -- NOT at (j + 0.5)/64, which is what this
    # builder used and which is a DIFFERENT convention (texel centres in
    # [0,1]). The two disagree by up to half a texel plus a 64/63 scale,
    # applied to both axes. Same class as the cube-decode and _gt_tex2d
    # corner-to-corner defects: the sampler's convention and the table's
    # convention have to be the same one, and the reference's is it.
    r = torch.arange(n, device=device, dtype=torch.float32) / (n - 1)
    t = torch.arange(n, device=device, dtype=torch.float32) / (n - 1)
    # T is the SECOND texture coordinate the shader builds, so the table's
    # rows are spaced in t and NdotV is DERIVED from it. Spacing the rows
    # in NdotV instead would put the table's samples in the wrong places
    # even once the transposition is fixed.
    T_, R = torch.meshgrid(t, r, indexing="ij")
    V = (1.0 - T_ * T_).clamp(1e-3, 1.0)
    a = (R * R).clamp(min=1e-4)
    vx = (1.0 - V * V).clamp(min=0).sqrt()
    A = torch.zeros_like(R)
    B = torch.zeros_like(R)
    for i in range(samples):
        # Hammersley
        u1 = (i + 0.5) / samples
        b = i
        u2 = 0.0
        f = 0.5
        while b:
            u2 += f * (b & 1)
            b >>= 1
            f *= 0.5
        phi = 2.0 * math.pi * u1
        ct = ((1.0 - u2) / (1.0 + (a * a - 1.0) * u2)).clamp(0, 1).sqrt()
        st = (1.0 - ct * ct).clamp(min=0).sqrt()
        hx, hy, hz = st * math.cos(phi), st * math.sin(phi), ct
        vh = vx * hx + V * hz
        lx, ly, lz = 2.0 * vh * hx - vx, 2.0 * vh * hy, 2.0 * vh * hz - V
        nl = lz.clamp(min=0)
        nh = hz.clamp(min=0)
        vhc = vh.clamp(min=0)
        k = a * a * 0.5
        g = (V / (V * (1 - k) + k).clamp(min=1e-6)) \
            * (nl / (nl * (1 - k) + k).clamp(min=1e-6))
        gv = torch.where(nl > 0, g * vhc / (nh * V).clamp(min=1e-6),
                         torch.zeros_like(g))
        fc = (1.0 - vhc) ** 5
        A = A + gv * (1.0 - fc)
        B = B + gv * fc
    A, B = A / samples, B / samples
    # packed as (B, A+B) -- see the docstring
    return torch.stack([B, A + B], dim=-1).contiguous()


# The two probes the orientation check evaluates. They are chosen because
# the two orientations DISAGREE BY AN ORDER OF MAGNITUDE there and in
# opposite directions, so one number cannot be right by accident: a
# transposed table reads high at the smooth head-on probe and low at the
# smooth grazing one. A probe near the middle of the domain would pass
# under either orientation, which is how the defect survived.
_GT_DFG_PROBES = ((0.10, 0.95), (0.10, 0.10))


def _gt_dfg_orientation_check(lut):
    """Print the built table's response against the q=0 polynomial at the
    two probes, THROUGH THE REAL LOOKUP PATH.

    This is a two-sided detector (`calibrate-the-detector-two-sided`): it
    must agree at both probes with the correct orientation and disagree
    at both with the transposed one. It prints rather than asserts,
    because the LUT and the Lazarov fit are two approximations of one
    integral and do not agree to zero -- but a factor of ten is not a fit
    disagreement, it is a transposition, and the ratio is on the line."""
    rows = []
    worst = 0.0
    for rough, ndv in _GT_DFG_PROBES:
        rq = torch.tensor([rough], device=lut.device)
        nq = torch.tensor([ndv], device=lut.device)
        u = rq * 0.984375 + 0.0078125
        v = (1.0 - nq).clamp(min=0).sqrt() * 0.984375 + 0.0078125
        tap = _gt_tex2d(lut, u, v)
        f0 = torch.full_like(rq, 0.04)
        ours = float(tap[..., 0] + (tap[..., 1] - tap[..., 0]) * f0)
        poly = float(env_brdf(rq, nq, f0).reshape(-1)[0])
        ratio = ours / poly if abs(poly) > 1e-9 else float("inf")
        worst = max(worst, ratio, 1.0 / max(ratio, 1e-9))
        rows.append(f"rough {rough:.2f} NdotV {ndv:.2f}: LUT {ours:.4f} "
                    f"vs q0 polynomial {poly:.4f} (x{ratio:.2f})")
    verdict = ("ORIENTATION OK" if worst < 2.0 else
               "⚠️  ORIENTATION SUSPECT -- a >2x disagreement at BOTH "
               "probes in OPPOSITE directions is the transposed-axis "
               "signature that caused #57")
    print(f"GT DFG LUT orientation check ({verdict}, worst ratio "
          f"{worst:.2f}x):", flush=True)
    for r in rows:
        print(f"    {r}", flush=True)


def gt_env_brdf(quality, rough, ndv, f0):
    """Split-sum environment BRDF at the selected S_SHADER_QUALITY.

    q=0, csgo_complex_ps.glsl:927-931 --
        vec4 r = vec4(-1, -0.0275, -0.572, 0.022)*rough
               + vec4( 1,  0.0425,  1.04, -0.04);
        float a004 = min(r.x*r.x, exp2(-9.28*max(0,dot(-viewFromEye, n))))
                   * r.x + r.y;
        vec2 AB = vec2(-1.04, 1.04)*a004 + r.zw;
        vec2 k  = vec2(AB.y, AB.x + AB.y);
        response = mix(k.xxx, k.yyy, f0);          // == AB.y + f0*AB.x
    env_brdf() already in this file is that polynomial with the same
    constants and returns f0*A + B, so this axis WIRES THROUGH the
    existing implementation rather than adding a second copy.

    q=1 replaces it with the LUT, plus a multiple-scattering term the
    q=0 file has no counterpart for (shaderquality1.glsl:1172-1174):
        E   = 1 - lut.y;
        Fss = f0 + (1 - f0)*0.047619;              // 1/21
        ms  = (response * Fss / (1 - Fss*E)) * E;
    The caller adds `response + ms`, and the q=1 diffuse is scaled by
    1 - (response + ms) (:1176) -- an energy-conservation factor the q=0
    file does not apply at all.
    """
    global GT_DFG_LUT
    if quality == 0:
        return env_brdf(rough, ndv, f0), torch.zeros_like(f0)
    if GT_DFG_LUT is None:
        GT_DFG_LUT = _gt_build_dfg_lut()
        print(f"GT DFG LUT built {tuple(GT_DFG_LUT.shape)} "
              f"= [sqrt(1-NdotV), roughness, (B, A+B)] "
              f"(S_SHADER_QUALITY=1 sampler2DArray, slice 1)", flush=True)
        _gt_dfg_orientation_check(GT_DFG_LUT)
    u = rough.clamp(0, 1) * 0.984375 + 0.0078125
    v = (1.0 - ndv.clamp(0, 1)).clamp(min=0).sqrt() * 0.984375 + 0.0078125
    lut = _gt_tex2d(GT_DFG_LUT, u, v)
    if _gt_inject("dfg-zero"):
        lut = torch.zeros_like(lut)
    lx, ly = lut[..., 0:1], lut[..., 1:2]
    resp = lx + (ly - lx) * f0                       # mix(lut.xxx, lut.yyy, f0)
    e = 1.0 - ly
    fss = f0 + (1.0 - f0) * 0.0476190485060215
    ms = (resp * fss / (1.0 - fss * e).clamp(min=1e-4)) * e
    return resp, ms


# ---------------------------------------------------------------------
# D_SPECULAR_CUBE_MAP_STATIC -- two different algorithms, one per quality
# ---------------------------------------------------------------------
def gt_spec_dir(n, view, rough):
    """The GGX dominant direction both qualities use.

    csgo_complex_ps.glsl:945 / shaderquality1.glsl:1067 --
        float a  = rough*rough;
        float k  = clamp(1 - a, 0, 1);
        vec3  d  = normalize(mix(n, reflect(viewFromEye, n),
                                 vec3(k * (sqrt(k) + a))));
    `view` here points TOWARD the eye, so reflect() takes -view.
    """
    a = (rough * rough).unsqueeze(-1)
    k = (1.0 - a).clamp(0, 1)
    t = (k * (k.sqrt() + a)).clamp(0, 1)
    ndv = (n * view).sum(-1, keepdim=True)
    mir = _nrm(2.0 * ndv * n - view)
    return _nrm(n + (mir - n) * t)


def gt_spec_cube(quality, wpos, n, view, rough, baked_irr, direct_diff,
                 static_cube=None):
    """Environment specular. THREE call flows, one per shipped combo.

    Which one runs is fixed by the constraint graph, not chosen freely:
    D_SPECULAR_CUBE_MAP_STATIC REQUIRES S_SHADER_QUALITY == 1
    (RENDER_SETTINGS_AXES.md sec 4), so the three reachable states are
    (q=0), (q=1, axis=0) and (q=1, axis=1) -- and _gt_check_constraints()
    refuses the fourth rather than quietly rendering it.

    --- q=0 : csgo_complex_ps.glsl:922-945, UNCONDITIONAL --------------
    There is no D_SPECULAR_CUBE_MAP_STATIC at quality 0, so this term is
    not behind that axis and is not labelled as if it were. One cubearray
    fetch at slice 0, then a chroma-preserving magnitude clamp against
    the LOCAL light level:
        vec3 env = textureLod(samplerCubeArray(g_tEnv, s),
                              vec4(dominantDir, 0.0),
                              g_flEnvMapLodScale * sqrt(rough)).xyz;
        vec3 L   = directDiffuse + bakedIrradiance;
        env *= normalize(L + 0.001) * clamp(length(L), 0, 1.75);
        env *= vec3(1) + normalize(bakedIrradiance + 0.001) * 0.5;

    --- q=1, D_SPECULAR_CUBE_MAP_STATIC = 0 ---------------------------
    shaderquality1.glsl:1068-1187 -- a BINNED loop over cubemap volumes
    with BOX PROJECTION (parallax correction) and a luminance-ratio
    renormalisation. Neither exists at q=0.
        pB = M_i * vec4(wpos, 1);   dB = M_i * vec4(dominantDir, 0);
        t   = min3(max((boxMax - pB)/dB, (boxMin - pB)/dB));    // :1140
        w   = t*t*(3-2t) * (1 - W);                             // :1141
        dir = mix(pB + dB*abs(t), dB, vec3(rough));             // :1143
        cube += T(slice_i, dir, lodScale*sqrt(rough)) * tint_i * w;
        avg  += g_vCubeAvg[i] * w;                              // :1144
      then :1187
        k = min(luminance(bakedIrr) / dot(vec4(n,1), avg/W),
                max(rough*x + y, 1.0));
        out += (cube/W) * k;

    --- q=1, D_SPECULAR_CUBE_MAP_STATIC = 1 ---------------------------
    The axis REMOVES the volume walk: one static cubemap, no loop. This
    is read off the shipped enumeration rather than assumed --
    SHADER_CALLFLOW_csgo_environment.md line 668 vs 676 is loops 6 -> 4
    at FLOPs 2975 -> 2896, and line 667 vs 675 is 10 -> 8 at 2965 ->
    2895; the walk is exactly two loops (outer word scan + inner bitmask)
    and the fetch count is unchanged at 1 cubearray site, which is what
    replacing a loop of one-site iterations with a single site does.
    Everything downstream of the walk is retained, with W = 1 and the two
    accumulators holding the single cube's own values -- so the same
    luminance-ratio renormalisation runs, and the parallax box of that
    one cube still applies.

    NOT IN THE ASSET at q=1: g_vCubeAvg, the per-cube vec4 whose
    dot(vec4(n,1), .) is the level the cube was baked under. Built here
    from each probe's own SH -- L1 luminance lobe as .xyz, DC luminance
    as .w -- which is exactly the affine form that dot expects. tint_i
    (_m6) is held at 1.0.
    """
    lodscale = args.gt_spec_cube_lod_scale
    if lodscale is None:
        lodscale = args.ibl_lod_scale
    d = gt_spec_dir(n, view, rough)
    if CUBE is None and SH is None:
        _gt_note("spec-cube: neither --ibl-cube nor --cube-probes given; "
                 "the environment specular has no radiance to fetch")
        return torch.zeros_like(baked_irr)
    pi = probe_of(wpos)
    lod = lodscale * rough.clamp(min=0).sqrt()
    if quality == 0:
        _gt_count("speccube.q0_unconditional",
                  torch.ones_like(rough, dtype=torch.bool))
        env = cube_sample(pi, d, lod) if CUBE is not None \
            else sh_radiance(sh_at(wpos), d)
        L = direct_diff + baked_irr
        ln = L.norm(dim=-1, keepdim=True)
        env = env * (L + 1e-3) / (L + 1e-3).norm(dim=-1, keepdim=True) \
            * ln.clamp(0, args.gt_spec_cube_clamp)
        env = env * (1.0 + (baked_irr + 1e-3)
                     / (baked_irr + 1e-3).norm(dim=-1, keepdim=True) * 0.5)
        if _gt_inject("speccube-zero"):
            return torch.zeros_like(env)
        _gt_value("env.spec_cube_q0", env, identity=0.0,
                  note="csgo_environment_ps.glsl:1251 / csgo_complex_ps"
                       ".glsl:945. UNCONDITIONAL in the module -- behind "
                       "no axis at either quality. Its at-identity "
                       "fraction was 1.0000 on every render this project "
                       "made before the cube array was built, while "
                       "_gt_count reported the branch taken on 100% of "
                       "pixels.")
        return env
    # ---- q=1 ---------------------------------------------------------
    # The box the shader intersects is the cubemap volume's own box; the
    # probe volumes are the only boxes this repo decodes, so they are it.
    # THE BOXES. shaderquality1.glsl:1140 intersects "the cubemap
    # volume's own box", and the 116 `env_combined_light_probe_volume`
    # entities that place the cube array ARE those volumes -- their
    # box_mins/box_maxs come in with the array itself and are already
    # loaded as PBMIN/PBMAX whenever --cube-probes is given.
    #
    # This used to read PROBE_T exclusively, which comes from a
    # DIFFERENT asset (--probe-npz) carrying an ambient-cube atlas.
    #
    # ⚠️ "an ambient-cube atlas we do not have" is what this comment said
    # until 2026-08-10, and it was FALSE. The atlas is on disk at
    # .scratch-tone/probe_field.npz -- 116 volumes, atlas (840,208,160,3)
    # fp16, band 140 -- a VRF export of de_inferno's own
    # maps/de_inferno/lightmaps/env_light_probe_volume_atlas.vtex_c (READ
    # from dec_probe_atlas.py's first line, not inferred from its extents).
    # It is genuinely DIRECTIONAL: face means over (+X,+Y,+Z,-X,-Y,-Z) are
    # 1.006 / 0.995 / 1.125 / 0.759 / 0.749 / 0.275, a 4.1x up-vs-down
    # ratio, with a median per-texel relative spread of 0.85. Passing
    # --probe-npz makes the whole six-face path run today; nothing needed
    # writing. Same shape as the four answers that were already in
    # comments nobody read.
    #
    # So the box projection was gated on an asset it does not
    # need, and with the cube array loaded and no --probe-npz the q1
    # walk was skipped entirely: `--spec-cube-static` toggled between
    # two identical images and the kernel oracle's
    # "q1 axis=0 vs axis=1" row read max|d| = 0.00000 while its own text
    # said the value must be non-zero. Coupling to the wrong asset, not
    # a missing one -- and the boxes were in memory the whole time.
    #
    # PROBE_T still wins when it exists, but NOT for the reason this
    # comment used to give. It said PROBE_T "carries the engine's own
    # overlap arbitration (indoor/outdoor level, then smallest volume)".
    # That rule was never the engine's -- csgo_complex_ps:421-454 walks the
    # volumes in index order and BLENDS, and never reads
    # indoor_outdoor_level at all. PROBE_T wins here because it is the
    # authored volume list; the boxes are the same boxes either way.
    #
    # NOTE THE COUPLING, because it cost a whole measurement: the rows are
    # taken as [:, 0:6] and accumulated IN ORDER by the walk below, so
    # PROBE_T's row ORDER changes this term's output. When the load-time
    # sort changed from the invented key to the shader's index order, the
    # de_inferno five-frame numbers moved -- and they moved HERE, in the
    # specular cube bounds, not in the probe arbitration those arms were
    # believed to be measuring.
    _boxes = None
    _boxsrc = ""
    if PROBE_T is not None:
        _boxes = PROBE_T[:, 0:6]
        _boxsrc = "--probe-npz (with the engine's overlap arbitration)"
    elif PROBE_T is None and 'PBMIN' in globals() and PBMIN is not None:
        _boxes = torch.cat([PBMIN, PBMAX], dim=1)
        _boxsrc = ("--cube-probes: the 116 env_combined_light_probe_volume "
                   "boxes that place the cube array, i.e. the volumes "
                   "shaderquality1.glsl:1140 intersects. No overlap "
                   "arbitration -- the soft blend below is the only "
                   "resolution")
    if _boxes is not None:
        _gt_note(f"spec-cube q1: box projection from {_boxsrc}")
        p_src = torch.stack([wpos[..., 2], wpos[..., 0], wpos[..., 1]],
                            dim=-1) / S
        d_src = torch.stack([d[..., 2], d[..., 0], d[..., 1]], dim=-1)
        flat = p_src.reshape(-1, 3)
        if static_cube:
            # D_SPECULAR_CUBE_MAP_STATIC=1: no walk. One volume -- the
            # one containing the point -- with weight 1.
            _gt_count("speccube.q1_static",
                      torch.ones_like(rough, dtype=torch.bool))
            best = torch.full(flat.shape[:1], -1, dtype=torch.long,
                              device=flat.device)
            for i in range(_boxes.shape[0]):
                ins = ((flat >= _boxes[i, 0:3])
                       & (flat < _boxes[i, 3:6])).all(-1)
                best = torch.where(ins, torch.full_like(best, i), best)
            bi = best.clamp(min=0)
            lo = _boxes[bi, 0:3].reshape(*wpos.shape[:-1], 3)
            hi = _boxes[bi, 3:6].reshape(*wpos.shape[:-1], 3)
        else:
            # D_SPECULAR_CUBE_MAP_STATIC=0: the binned walk, with the
            # shader's own soft weight -- same t*t*(3-2t)*(1-W) form as
            # the probe loop, seeded at 0.01 (:1074) and exiting at
            # W > 0.99 (:1145).
            W = torch.full(flat.shape[:1], 0.01, device=flat.device)
            acc_lo = torch.zeros_like(flat)
            acc_hi = torch.zeros_like(flat)
            acc_w = torch.zeros(flat.shape[0], device=flat.device)
            for i in range(_boxes.shape[0]):
                blo, bhi = _boxes[i, 0:3], _boxes[i, 3:6]
                fade = 1.0 / (args.gt_probe_fade
                              * (bhi - blo).clamp(min=1e-4))
                a = ((flat - blo) * fade).clamp(0, 1)
                b = ((bhi - flat) * fade).clamp(0, 1)
                t = torch.minimum(a.min(-1).values, b.min(-1).values)
                live = (t > 0) & (W <= 0.99)
                if not bool(live.any()):
                    continue
                dw = ((t * t) * ((-2.0) * t + 3.0)) * (1.0 - W)
                dw = torch.where(live, dw, torch.zeros_like(dw))
                acc_lo = acc_lo + blo * dw.unsqueeze(-1)
                acc_hi = acc_hi + bhi * dw.unsqueeze(-1)
                acc_w = acc_w + dw
                W = W + dw
            wn = acc_w.clamp(min=1e-4).unsqueeze(-1)
            lo = (acc_lo / wn).reshape(*wpos.shape[:-1], 3)
            hi = (acc_hi / wn).reshape(*wpos.shape[:-1], 3)
            _gt_count("speccube.q1_binned_walk",
                      (acc_w > 0).reshape(*wpos.shape[:-1]))
        c = (lo + hi) * 0.5
        h = (hi - lo) * 0.5 * args.gt_parallax_box
        lo, hi = c - h, c + h
        ds = torch.where(d_src.abs() < 1e-5,
                         torch.full_like(d_src, 1e-5), d_src)
        t = torch.maximum((hi - p_src) / ds, (lo - p_src) / ds)
        t = t.min(-1, keepdim=True).values
        hitp = p_src + d_src * t.abs()                          # :1140-1143
        r3 = rough.unsqueeze(-1)
        dir_src = _nrm(hitp * (1.0 - r3) + d_src * r3)
        d = torch.stack([dir_src[..., 1], dir_src[..., 2], dir_src[..., 0]],
                        dim=-1)
    else:
        _gt_note("spec-cube q1: no probe volumes, so the box projection "
                 "(shaderquality1.glsl:1140) has no box; the dominant "
                 "direction is used unprojected")
    env = cube_sample(pi, d, lod) if CUBE is not None \
        else sh_radiance(sh_at(wpos), d)
    lum = torch.tensor(_lf_filters.LUMA_709, device=wpos.device,
                       dtype=torch.float32)
    if SH is not None:
        Ll = (SH[pi] * lum.view(3, 1)).sum(-2)                  # (...,9)
        avg_w = 0.886227 * Ll[..., 0]
        avg_xyz = 2.0 * 0.511664 * torch.stack(
            [Ll[..., 3], Ll[..., 1], Ll[..., 2]], dim=-1)
        den = ((n * avg_xyz).sum(-1) + avg_w).clamp(min=1e-4)
    else:
        den = torch.ones_like(rough)
    ceil_ = (rough * args.gt_spec_cube_boost[0]
             + args.gt_spec_cube_boost[1]).clamp(min=1.0)
    k = torch.minimum((baked_irr * lum).sum(-1) / den, ceil_)   # :1187
    if _gt_inject("speccube-zero"):
        return torch.zeros_like(env)
    return env * k.unsqueeze(-1)


# =====================================================================
# The composition -- how the shader assembles the terms it just computed
# csgo_complex_ps.glsl:618-945 (q=0), shaderquality1.glsl:721-1187 (q=1)
# =====================================================================
def gt_select_baked(lm_ok):
    """Which D_BAKED_LIGHTING_FROM_* each pixel takes.

    MUTUAL EXCLUSIVITY IS VERIFIED, NOT ASSUMED. csgo_environment_ps
    ships 16 dynamic combo ids (SHADER_CALLFLOW_csgo_environment.md:
    551-566): 0,1,2,4,8,9,10,12,16,17,18,20,24,25,26,28. Bits 1, 2 and 4
    are VERTEX_STREAM, PROBE and LIGHTMAP; every one of those 16 ids sets
    AT MOST ONE of them, and the set factors exactly as
    {none, VTX, PROBE, LM} x D_OPAQUE_FADE(8) x D_SPECULAR_CUBE_MAP_STATIC(16).
    Ids 3, 5, 6 and 7 -- the two-bit combinations -- are not shipped in
    any static combo. So the axis is a 4-way choice, and 'none' is one of
    the four, not an off switch.

    'auto' reproduces the engine's own split, which is why
    csgo_environment's primary variant is dyn 4 and csgo_complex's is
    dyn 2: lightmapped world geometry takes the lightmap, everything
    else takes the probe.
    """
    m = args.baked_lighting
    if m != "auto":
        return m
    if lm_ok is None or IRRMAP is None:
        return "probe"
    return "auto"


def _gt_lm_directional(irr, u, v, normal, quality):
    """Q1:818-831 -- make the lightmap respond to the normal map.

    q1 binds a THIRD lightmap array and splits the baked irradiance into
    an ambient floor scaled by the direction page's confidence channel
    plus a directional remainder redistributed by
    `dot(bakedDirection, shadingNormal)`. q0 has no counterpart and uses
    the lightmap flat (:756-758, :1227-1228). For the 95 of 153
    materials on D_BAKED_LIGHTING_FROM_LIGHTMAP that is the largest
    behavioural difference between the two quality levels, and it is the
    half of defect #57 that was an ASSET hole rather than a maths one.

    THE JOIN IS MADE AT THE MODULE'S DECLARATION, not at the page's
    statistics. W7's extraction deliberately declined to name the
    encoding from its channel histogram (R,G near neutral, B biased --
    hemi-oct-SHAPED, but shaped is not read). Q1:818-828 says .xy is a
    hemispherical disc direction and Q1:829 says .z is the confidence;
    that is a declaration and it is what this consumes. The channel
    means are printed at load as corroboration only.

    NOT IMPLEMENTED HERE, and it changes the term's meaning rather than
    its magnitude: the reference expresses the shading normal in the
    LIGHTMAP'S OWN tangent basis (`vec3(dot(n,T), dot(n,B), dot(n,N))`
    at Q1:831), and this passes the world-space normal because no
    per-luxel lightmap tangent frame exists in the pack. So the cosine
    is taken between a world direction and a world normal instead of
    between two tangent-space ones. If the baked direction is stored in
    tangent space, that is WRONG and not approximately right -- and the
    disagreement is a rotation, which no scale factor repairs. Printed
    as an uncertainty at runtime, not left in this docstring, and it is
    the first thing to settle once a lightmap tangent frame exists.
    """
    if quality == 0:
        return irr
    if GT_LM_DIR is None or u is None:
        # ABSENCE MUST PRINT. I told W3 this state was "self-describing"
        # and it was not: this returned silently, and the GAP written
        # for this term fires only AFTER the page is bound, describing
        # the tangent-space uncertainty rather than the absence. So a
        # run on a map with no direction page said nothing at all about
        # it -- an unattributable flat frame six weeks later, which is
        # the failure the whole instrument exists to prevent.
        #
        # Not made moot by de_inferno's page landing: the all-43 sweep
        # is still running, and a page existing for SOME maps is exactly
        # the condition under which a missing one goes unnoticed.
        _gt_note(
            "lightmap q1 directional: NO DIRECTION PAGE"
            + (" for this map" if GT_LM_DIR is None else
               " coordinate on this geometry")
            + " -- the reference binds a third sampler2DArray at "
              "Q1:817 and redistributes irradiance by "
              "dot(bakedDirection, normal) at Q1:829-831, so at q1 its "
              "lightmap responds to the normal map. Ours cannot here: "
              "the irradiance is used FLAT, which is the q0 behaviour "
              "running under a q1 label. Pass --gt-lm-direction to bind "
              "it. This is an ASSET absence, not a disabled feature.")
        _gt_value("lm.directional_gain", torch.ones_like(irr[..., 0]),
                  identity=1.0,
                  note="forced to identity because no direction page is "
                       "bound -- the row exists so the term is VISIBLE "
                       "as inert rather than missing from the table")
        return irr
    dpage = _gt_bicubic4(GT_LM_DIR, u, v)
    curv = torch.ones_like(dpage[..., 0])
    d = env_shading.lm_direction_decode(dpage[..., 0:2], curv, curv)
    _gt_value("lm.direction_z", d[..., 2], identity=1.0,
              note="Q1:818-828 decoded direction, z component. 1.0 "
                   "everywhere would mean the page carries no lateral "
                   "signal and the reconstruction is an identity")
    out = env_shading.lm_directional_reconstruct(
        irr, dpage[..., 2], args.gt_lm_direction_bias, d, normal)
    # Gain only where the input is a real positive: a luxel that was
    # already 0 has no ratio, and reporting 0/0 as a NaN would poison the
    # statistic for every luxel that does. Unlit atlas area is most of a
    # lightmap, so this is the common case, not an edge one.
    _num, _den = out.sum(-1), irr.sum(-1)
    _ok = torch.isfinite(_num) & torch.isfinite(_den) & (_den > 1e-6)
    _gt_value("lm.directional_gain",
              torch.where(_ok, _num / _den.clamp(min=1e-6),
                          torch.ones_like(_den)), identity=1.0,
              note="Q1:829-831 out/in, over luxels with non-zero input "
                   "only. At-identity 1.0 would mean the third page is "
                   "bound and changes nothing")
    _bad = int((~torch.isfinite(out)).sum())
    if _bad:
        _gt_note(f"lightmap q1 directional: {_bad} non-finite outputs -- "
                 f"Q1:831 divides by dir.z unguarded and relies on the "
                 f"Q1:822 disc clamp to keep it away from 0; a page whose "
                 f"xy exceeds the clamp radius breaks that")
    _gt_note("lightmap q1 directional: the shading normal is passed in "
             "WORLD space; Q1:831 uses the lightmap's own tangent basis, "
             "which the pack does not carry. If the baked direction is "
             "tangent-space this is a rotation error, not a scale error")
    return out


def gt_baked(mode, wpos, normal, lm_u, lm_v, lm_ok, vlit, quality):
    """Evaluate the selected baked-lighting path. Returns (irr, occ4)."""
    # Cleared for EVERY mode, before any branch, so a mode that never
    # reaches gt_baked_lightmap cannot inherit the previous call's tint.
    # The lightmap branch sets it back on when its fetch succeeds -- and
    # the two early `return None, None` exits there leave it off, which
    # is right: those fall back to the dyn-0 module.
    global GT_LM_TINT_LIVE
    GT_LM_TINT_LIVE = False
    if mode == "none":
        _gt_count("baked.none", torch.ones_like(wpos[..., 0], dtype=torch.bool))
        return (gt_ambient_sh(normal),
                torch.zeros(*wpos.shape[:-1], 4, device=wpos.device))
    if mode == "vertex-stream":
        if vlit is None:
            _gt_note("vertex-stream selected but no COLOR1 attribute was "
                     "interpolated; see gt_bake_vertex_stream()")
            return gt_baked("none", wpos, normal, lm_u, lm_v, lm_ok, None,
                            quality)
        _gt_count("baked.vertex_stream",
                  torch.ones_like(wpos[..., 0], dtype=torch.bool))
        return (vlit,
                torch.zeros(*wpos.shape[:-1], 4, device=wpos.device))
    if mode == "probe":
        irr, occ, _w = gt_baked_probe(wpos, normal)
        _gt_count("baked.probe", torch.ones_like(wpos[..., 0],
                                                 dtype=torch.bool))
        _probe_gt_note("probe", irr, _w)
        return irr, occ
    if mode == "lightmap":
        irr, occ = gt_baked_lightmap(lm_u, lm_v, quality)
        if irr is None:
            return gt_baked("none", wpos, normal, lm_u, lm_v, lm_ok, vlit,
                            quality)
        irr = _gt_lm_directional(irr, lm_u, lm_v, normal, quality)
        _gt_count("baked.lightmap", torch.ones_like(wpos[..., 0],
                                                    dtype=torch.bool))
        return irr, occ
    # auto: per-pixel, the engine's own split
    li, lo_ = gt_baked_lightmap(lm_u, lm_v, quality)
    pi_, po, _w = gt_baked_probe(wpos, normal)
    if li is None:
        return pi_, po
    # 'auto' is the engine's own per-pixel split, so the q1 directional
    # reconstruction has to run on this branch too -- applying it only
    # in the explicit "lightmap" mode would make the term appear and
    # disappear with a flag that is about WHICH baked source is chosen,
    # not about whether the third page is bound.
    li = _gt_lm_directional(li, lm_u, lm_v, normal, quality)
    sel = (lm_ok > LM_OK_T).unsqueeze(-1)
    # The fraction the ENGINE'S OWN SPLIT routes to the probe -- which is a
    # different number from "how many pixels are inside a probe volume",
    # and it is the one that decides whether this term can matter here at
    # all. Reported next to the coverage so the two cannot be confused
    # again: 100% of pixels sit inside a volume, and that says nothing
    # about how many pixels TAKE the volume.
    _probe_gt_note("auto", pi_, _w, sel=~sel[..., 0])
    _gt_count("baked.auto_lightmap", sel[..., 0])
    _gt_count("baked.auto_probe", ~sel[..., 0])
    return (torch.where(sel, li, pi_), torch.where(sel, lo_, po))


def _gt_lm_tint(indirect, live):
    """g_bLightmapTint, at the position both transcribed files put it.

    The gate is `_5618._m6 != 0` (csgo_environment_ps.glsl:525), a member
    identified by DECLARED BYTE OFFSET 60, `layout(offset = 60) int _m6`.
    It is NOT `_5618._m3`, which is offset 28 and a uint used as the
    sampler-state index at :436, :495 and :512 -- the ordinals are close
    and the roles are unrelated, which is the whole reason the offset is
    the identity.

    WHAT IT IS APPLIED TO. In both files the operand is the FINISHED
    indirect term, after the AO multiply and immediately before the sum
    with the direct accumulator:

      q=0  :1227  _8745  = lm0.xyz;                     // irradiance
           :1229  _20418 = _21714 * _17119;             // AO product
           :1230  _20015 = _8745 * _20418;
           :1234  _20579 = mix(_20015, pow(_20015, t), float(bTint));
           :1240  out    = (_16324 + _20579) * albedo;

      q=1  :1588  _22438 = (irr * (1 - (response + multiscatter))) * AO;
           :1592  _19631 = mix(_22438, pow(_22438, t), float(bTint));
           :1598  out    = (_13143 + _19631) * albedo;

    so the q=1 energy-conservation factor is INSIDE the operand too.

    The mix is FOLDED to the pow here, not written out. `_21618` is a
    bool, so `float(_21618)` is exactly 1.0 on the branch that runs the
    mix at all, and mix(a, b, 1.0) is b -- but `a + (b - a) * 1.0` is not
    bit-for-bit `b` in floating point, so spelling the mix out would add
    rounding this function does not otherwise have. If that uniform ever
    turns out to be a float rather than a bool, this is the line that has
    to grow the weight back.

    The environment-specular term reads the UNtinted irradiance (:1228
    `_16991 = _16324 + _8745`, the raw fetch), so gt_spec_cube keeps
    getting baked_irr and not this.
    """
    if not live:
        _gt_value("baked.lm_tint", torch.ones_like(indirect), identity=1.0,
                  note="g_bLightmapTint (_5618._m6, BYTE OFFSET 60) is off "
                       "for this baked source, so the ratio is 1 by "
                       "construction and the pow never ran")
        return indirect
    t = torch.tensor([1.05, 0.95, 0.80], device=indirect.device)
    out = indirect.clamp(min=0) ** t
    _gt_value("baked.lm_tint", out / indirect.clamp(min=1e-6), identity=1.0,
              note="pow(indirect, [1.05, 0.95, 0.80]) over the indirect "
                   "term AFTER the AO multiply (:1230/:1234 at q=0, "
                   ":1588/:1592 at q=1). Reported as the RATIO it applies, "
                   "because the pow's effect depends entirely on how far "
                   "the operand is from 1.0 -- at AO 1 and irradiance 1 it "
                   "is exactly 1 and the term is a no-op no matter how it "
                   "is placed")
    return out
