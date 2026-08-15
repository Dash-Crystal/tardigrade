

def render_window(mvp, eye, fwd, tsec=None):
    """Three alpha-class passes composited by depth.

    OPAQUE fills the frame. MASK (alpha-tested foliage, fences) is
    rejected per-pixel below its cutoff and otherwise wins where it is
    nearer — a single layer, so foliage behind foliage is approximate.
    BLEND (decals, overlays) alpha-composites over the result where it
    is nearer. Classes come from glTF alphaMode, never guessed from
    texture-alpha statistics.
    """
    # #48 needs the camera and the shading chain it feeds does not
    # receive one; threading a camera through a 20-parameter
    # signature to reach one call site is the worse change.
    global SKY_MVP
    SKY_MVP = mvp
    B = mvp.shape[0]
    per_frame_visible = cluster_visibility(mvp)
    outs = []
    # The RESOLVED scene depth, kept per chunk so it survives the loop. The
    # playermodel pass needs it frame-wide -- it is world geometry and must
    # be occluded by the world -- and `zfc_o` was chunk-local, i.e. the one
    # thing the pass could not have reconstructed afterwards from `frame`.
    zouts = []
    # chunk of 1 => eye[c0:c1].mean(dim=0) IS that frame's eye, so the
    # LOD below becomes per-frame and therefore --batch-invariant.
    _ck = 1 if args.lod_per_frame else CHUNK
    for c0 in range(0, B, _ck):
        c1 = min(c0 + _ck, B)
        if VS_ON:
            # One evaluation per CHUNK; `--lod-per-frame` makes the chunk 1
            # and therefore makes the animation per-frame.  The time is the
            # chunk's FIRST frame, so the quantisation only ever lags.
            global VS_VPOS, VS_VPOS_H, VS_VNRM, VS_VTAN
            _t = (VS_FRAME0[0] + c0) * VS_DT
            _p, _n, _tg = vertex_stage(_t)
            VS_VPOS = _p
            VS_VPOS_H = torch.cat([_p, torch.ones_like(_p[:, :1])], dim=1)
            VS_VNRM = _n
            VS_VTAN = _tg
        vis = per_frame_visible[c0:c1]
        if args.occlusion:
            vis = occlusion_refine(mvp[c0:c1], vis)
        visible = vis.any(dim=0)
        idx = visible.nonzero().squeeze(1)
        _t_lod = _time.time()
        if args.no_lod:
            lvl = torch.zeros(len(idx), dtype=torch.long, device=device)
        else:
            dist = (centers[idx] - eye[c0:c1].mean(dim=0)).norm(dim=1)
            lvl = torch.bucketize(dist, lod_dist[1:], right=True)
        parts_f, parts_m, parts_n, parts_mode, parts_cut = [], [], [], [], []
        parts_mat2, parts_has2, parts_sec, parts_params = [], [], [], []
        parts_mid = []
        for lv in range(n_levels):
            sel = idx[lvl == lv]
            if not len(sel):
                continue
            g = segment_gather(level_begin[lv][sel], level_counts[lv][sel])
            parts_f.append(level_faces[lv][g])
            parts_m.append(level_mat[lv][g])
            parts_n.append(level_norm[lv][g])
            parts_mode.append(level_mode[lv][g])
            parts_cut.append(level_cut[lv][g])
            parts_mat2.append(level_mat2[lv][g])
            parts_has2.append(level_has2[lv][g])
            parts_sec.append(level_sec[lv][g])
            parts_params.append(level_params[lv][g])
            parts_mid.append(level_matid[lv][g])
        # A pose whose view contains NO world geometry is a legitimate
        # state in a demo -- a player looking straight up at open sky.
        # Our sky is a screen-space fill and contributes no faces, so
        # every parts_* list is empty and torch.cat() raises. It used to
        # take the whole job with it, which at corpus scale means one
        # rare state kills a multi-hour render. Emit a ZERO-FACE batch
        # instead: the raster produces no fragments, the background and
        # sky composite exactly as they would behind geometry, and the
        # frame is the correct render of a view with nothing in it.
        # Sliced from level_* so dtype and device follow the real
        # tensors rather than being asserted here.
        def _cat0(parts, proto):
            return torch.cat(parts) if parts else proto[:0]
        _empty_view = not parts_f
        sub_mid = _cat0(parts_mid, level_matid[0])
        sub_faces = _cat0(parts_f, level_faces[0])
        sub_mat = _cat0(parts_m, level_mat[0])
        sub_norm = _cat0(parts_n, level_norm[0])
        sub_mode = _cat0(parts_mode, level_mode[0])
        sub_cut = _cat0(parts_cut, level_cut[0])
        sub_mat2 = _cat0(parts_mat2, level_mat2[0])
        sub_has2 = _cat0(parts_has2, level_has2[0])
        sub_sec = _cat0(parts_sec, level_sec[0])
        sub_params = _cat0(parts_params, level_params[0])
        # bucketize is cheap; the cost is the per-level segment_gather plus
        # eleven fancy-index gathers plus ten cats above, i.e. a complete
        # index-buffer and per-face-attribute rebuild, materialised fresh
        # once per CHUNK -- or once per FRAME under --lod-per-frame.
        STAGE["render.lod_rebuild"] = (STAGE.get("render.lod_rebuild", 0.0)
                                       + _time.time() - _t_lod)
        if _empty_view:
            # Occupancy prints so an empty frame arrives as DATA. Without
            # it this fix converts a loud crash into a silent black frame
            # indistinguishable from a dark scene -- the exact trade this
            # renderer's own catalogue warns about.
            print("occupancy: frames %d..%d see ZERO world faces "
                  "(valid empty view, background only)" % (c0, c1 - 1),
                  flush=True)
            # nvdiffrast refuses a zero-vertex batch outright ("pos must
            # have shape [>0, >0, 4]"), so the raster cannot be asked to
            # draw nothing -- it has to be SKIPPED. Emit the background
            # directly, in the shape and dtype the rest of the pipeline
            # already produces, and let the caller composite as usual.
            outs.append(torch.zeros((c1 - c0, args.height, args.width, 3),
                                    dtype=torch.uint8, device=device))
            # The DEPTH side of the same shaped-empty convention. This
            # branch emitted the background frame but appended nothing to
            # zouts, so a window mixing empty and non-empty chunks handed
            # playermodel_pass a SHORT z-buffer (measured: 48 mvp rows vs
            # 32 z rows, p1/p3 of the tard21 match sweep). An empty world
            # has far-plane depth everywhere: entities in it are tested
            # against +inf and drawn over the background, which is the
            # physically correct join. The frame COUNT is recorded here;
            # the tensor is materialised at the cat site with the dims and
            # dtype of the window's REAL z-buffers, so supersampled runs
            # stay consistent by construction.
            zouts.append(c1 - c0)
            continue

        used, remapped = torch.unique(sub_faces.reshape(-1),
                                      return_inverse=True)
        local = remapped.reshape(-1, 3).int()
        # --- the VERTEX stage's texcoord construction -----------------
        # S_SECONDARY_UV and S_FOLIAGE_UV_ANIMATION are vertex-stage terms
        # in the reference, so they are applied to the ATTRIBUTE here, not
        # to the interpolated pixel value.  The two are not equivalent: an
        # affine on a per-vertex UV interpolates differently from the same
        # affine applied after interpolation once the transform is not
        # identity, and the reference does it per vertex.
        uv_attr = uvs[used]
        if UVSTAGE_ON:
            # per-vertex UV-set selector, from the owning material's flag
            _selv = torch.ones(len(used), dtype=torch.long, device=device)
            if args.secondary_uv == "material":
                _sv = torch.zeros(len(vertices), device=device)
                for _c in range(3):
                    _sv[faces0[:, _c]] = face_sec_0
                _selv = (_sv[used] + 1).long()
            elif args.secondary_uv != "off":
                _selv = torch.full((len(used),), int(args.secondary_uv),
                                   dtype=torch.long, device=device)
            # SELECTION here, TRANSFORM in gt_uv (#76). foliage_uv_animation
            # below still composes on top of whatever this returns, and that
            # order is preserved deliberately: its scroll is additive and
            # runs after the stream choice, exactly as before.
            uv_attr = vsx.secondary_uv(uv_attr, uvs2[used], _selv, None)
            if args.foliage_uv_animation:
                _fm = _fam_mask(_FAM_FOLIAGE)[used].unsqueeze(-1)
                _t_uv = (VS_FRAME0[0] + c0) * VS_DT
                _sc = torch.tensor(args.foliage_uv_scroll, device=device)
                uv_attr = torch.where(
                    _fm, vsx.foliage_uv_animation(uv_attr, _sc, _t_uv),
                    uv_attr)
        clip = torch.einsum("bij,nj->bni", mvp[c0:c1],
                            VS_VPOS_H[used]).contiguous()

        m_op = sub_mode == 0
        m_mk = sub_mode == 1
        m_bl = sub_mode == 2
        FAR = 1e9

        uv2_attr = uvs2[used]
        vc_attr = vcolor[used]
        # THE OPAQUE CLASS IS GUARDED LIKE ITS TWO SIBLINGS. mask and
        # blend have carried `if int(m_*.sum()):` since they were
        # written; opaque ran unconditionally because "the opaque class
        # is never empty" was true of every map tested and is not true of
        # de_mirage_vanity from a foreign camera, where the visible set is
        # mask and blend only. _raster's own guard (site 4) returns
        # well-shaped empties, and the very next line hands the SAME (0,3)
        # face list to dr.interpolate, which refuses it -- guarding the
        # rasteriser alone moved the raise one line down.
        if int(m_op.sum()):
            rast_o, fg_o, uv_o, mat_o, nrm_o, uvda_o = _raster(
                clip, sub_faces[m_op], sub_mat[m_op], sub_norm[m_op],
                uv_attr, local[m_op])
            # One footprint per raster pass, reused by every map sampled with
            # this UV: the derivative maths is shared, only the gathers are not.
            tapsC_o = mip_taps(uvda_o, T, ANISO_C, TEX_MIP[3]) \
                if TEX_MIP is not None and uvda_o is not None else None
            tapsA_o = mip_taps(uvda_o, TA, ANISO_A, AUX_MIP[3]) \
                if AUX_MIP is not None and uvda_o is not None else None
            if args.mip_diag and not MIP_DIAG_DONE:
                MIP_DIAG_DONE.append(1)
                _mip_report(uv_o, uvda_o, fg_o)
            depth_o = torch.where(fg_o, rast_o[..., 2],
                                  torch.full_like(rast_o[..., 2], FAR))
            i_o = (rast_o[..., 3].long() - 1).clamp(min=0)
            uv2_o, _ = dr.interpolate(uv2_attr[None].contiguous(), rast_o,
                                      local[m_op])
            # LINK 0 OF THE uv2_o REACH TRACE. Everything below this line can
            # only pass on a difference that exists here, and for the whole
            # slot-family arc nothing printed whether there was one: `uvs2`
            # falls back to `uvs` at :5365 with no announcement, so a pack
            # without TEXCOORD_1 renders UV2==UV1 and looks exactly like a
            # pack whose UV2 is being discarded downstream. Those are
            # different faults with different owners, and this separates them.
            _gt_value("uv.uv2_differs", (uv2_o - uv_o).abs().amax(-1),
                      identity=0.0,
                      note="fraction at identity = pixels where UV2 IS UV1. "
                           "1.0000 means the attribute never reached the "
                           "interpolator (pack has no uvs2, or the gather is "
                           "wrong) and NO downstream selector could have acted; "
                           "below 1.0 means the attribute is here and any "
                           "identical frame is a CONSUMER fault.")
            vc_o, _ = dr.interpolate(vc_attr[None].contiguous(), rast_o,
                                     local[m_op])
            vp_o = None
            if VPAINT is not None:
                vp_o, _ = dr.interpolate(
                    VPAINT[used][:, None][None].contiguous(), rast_o, local[m_op])
                vp_o = vp_o[..., 0]
            # --- Source 2 vmat block: the blend weight is decided BEFORE the
            # albedo so the colour layers, the normal layers and the border
            # tint all use the same transition. Doing it inside shade_albedo
            # would leave the normals blending on a different weight.
            # --matid-dump is a GEOMETRY question -- which material owns this
            # pixel -- and depends on no PBR term, so it must not be gated on
            # PBR. It was: at bare defaults PBR is False, mid_op was None,
            # mid_pix was None, MATID_DUMP stayed empty, and the run printed
            # "matid-dump: wrote 0 frames" and exited 0. A dump request that
            # cannot produce a dump is the "measured zero vs never measured"
            # failure, and the canonical ego01 invocation is exactly the one
            # that hits it.
            mid_op = sub_mid[m_op][i_o] if (PBR or args.matid_dump) else None
            if args.matid_reach and VMAT:
                for _k in ("f_blend_effects2", "f_overlay", "has_aov",
                           "f_tintmask", "f_transmissive_ndotl",
                           "f_render_backfaces", "f_metalness_tex",
                           "f_self_illum", "has_h2"):
                    _m = (MAT_EXT[mid_op][..., EXT[_k]] > 0.5) & fg_o
                    REACH[_k] = REACH.get(_k, 0.0) + float(_m.float().mean())
                REACH["_n"] = REACH.get("_n", 0.0) + 1.0
            w_src = (vp_o if args.blend_weight == "vpaint"
                     else vc_o[..., args.blend_channel])
            if args.blend_weight == "vpaint" and args.vpaint_remap:
                # csgo_environment_blend_ps.glsl:260-261, the FIRST two lines of
                # MainPs -- the raw vertex paint is remapped before anything
                # else touches it:
                #   w2 = clamp(vIn0.x * 1.1 - 0.05, 0, 1)
                #   w3 = clamp(vIn0.y * 1.1 - 0.05, 0, 1)
                # a 5% dead zone at both ends, so paint below 0.0455 is exactly
                # layer 1 and above 0.9545 exactly layer 2.
                w_src = (w_src * 1.10000002384185791 - 0.05).clamp(0, 1)
            w_raw = (w_src * sub_has2[m_op][i_o]).clamp(0, 1)
            if args.probe_dump and PBR:
                DUMP.append((
                    mid_op.to(torch.int16).cpu().numpy(),
                    (vc_o[..., args.blend_channel] * sub_has2[m_op][i_o]
                     ).clamp(0, 1).half().cpu().numpy(),
                    (vp_o * sub_has2[m_op][i_o]).clamp(0, 1).half().cpu().numpy()
                    if vp_o is not None else None,
                    fg_o.cpu().numpy(),
                ))
            if args.probe_rect and PBR and not PROBE:
                _x, _y, _w, _h = (int(v) for v in args.probe_rect.split(","))
                # the shaded buffer is bottom-up; flip y to screen order
                _yy = H - _y - _h
                _mm = mid_op[0, _yy:_yy + _h, _x:_x + _w].reshape(-1)
                _ww = w_raw[0, _yy:_yy + _h, _x:_x + _w].reshape(-1)
                _u, _c = _mm.unique(return_counts=True)
                for _i in _c.argsort(descending=True)[:6].tolist():
                    _sel = _mm == _u[_i]
                    PROBE.append((int(_u[_i]), float(_c[_i]) / _mm.numel(),
                                  float(_ww[_sel].mean())))
            # --- GROUND-TRUTH MATERIAL-SURFACE COMBO AXES ---------------
            # These run BEFORE the albedo because two of them (the UV axes)
            # change what every later fetch samples, and one (layer 3)
            # changes the weight the colour, the normal and the border tint
            # all have to agree on.
            x_gt = MAT_EXT2[mid_op] if GTSURF else None
            uv_l2 = None
            w3_raw = None
            T_NOW = ((tsec[c0:c1] if tsec is not None
                      else torch.zeros(c1 - c0, device=device))
                     .view(-1, 1, 1))
            if GTSURF:
                if args.texture_animation:
                    uv_o, _duv = gt_texture_animation(uv_o, x_gt, T_NOW)
                    if uvda_o is not None:
                        uvda_o = uvda_o * torch.stack(
                            [_duv[..., 0], _duv[..., 0],
                             _duv[..., 1], _duv[..., 1]], dim=-1)
                        tapsC_o = mip_taps(uvda_o, T, ANISO_C, TEX_MIP[3]) \
                            if TEX_MIP is not None else None
                        tapsA_o = mip_taps(uvda_o, TA, ANISO_A, AUX_MIP[3]) \
                            if AUX_MIP is not None else None
                if args.secondary_uv != "off":
                    # LINKS 1 AND 2. Both selectors read the SAME pre-selection
                    # uv_o: `_uv_c0`, not the rebound `uv_o`. Layer 2's
                    # "inherit" branch means "whatever the previous slot used",
                    # and the previous slot is TEXCOORD0 -- feeding it layer 1's
                    # already-selected coordinate would make slot 1's choice
                    # silently propagate into slot 2's inherit. Inert today
                    # because both picks are 0 (see the ack below), which is
                    # exactly why it had to be fixed before the gate opens
                    # rather than after.
                    _uv_c0 = uv_o
                    uv_o = gt_uv(_uv_c0, uv2_o, "uvsel_c1", x_gt,
                                 "uv2_su", "uv2_sv", "uv2_ou", "uv2_ov")
                    uv_l2 = gt_uv(_uv_c0, uv2_o, "uvsel_c2", x_gt,
                                  "uv2_su", "uv2_sv", "uv2_ou", "uv2_ov")
                    # Did the selector MOVE the coordinate it was handed? This
                    # is the link the AO disambiguation would have asked for:
                    # `uv.uv2_differs` says the input is present, these two say
                    # whether either consumer took it.
                    _gt_value("uv.gt_uv_c1_moved", (uv_o - _uv_c0).abs().amax(-1),
                              identity=0.0,
                              note="fraction at identity = pixels where the "
                                   "layer-1 selector kept TEXCOORD0. 1.0000 with "
                                   "uv.uv2_differs < 1.0 means the SELECTOR "
                                   "dropped a UV2 that was present -- an ext2 "
                                   "uvsel_c1/f_secondary_uv question, not a pack "
                                   "question.")
                    _gt_value("uv.gt_uv_c2_moved", (uv_l2 - _uv_c0).abs().amax(-1),
                              identity=0.0,
                              note="same, for layer 2. uv_l2 is the ONLY route "
                                   "by which UV2 reaches an albedo fetch: "
                                   "shade_albedo's `uv2` parameter is never read "
                                   "in its body, so this reading 1.0000 means "
                                   "layer 2 sampled TEXCOORD0.")
                if args.layer3:
                    # csgo_environment_blend_ps.glsl:260-262 -- COLOR_0.y is
                    # layer 3's weight and carries the SAME 5% dead zone the
                    # layer-2 weight does.
                    w3_raw = (vc_o[..., 1] * 1.10000002384185791
                              - 0.05).clamp(0, 1)
                if args.gt_axes_reach:
                    for _k in ("f_detail", "f_detail_normal", "f_tint_mask2",
                               "f_metal_tex", "f_self_illum2",
                               "f_secondary_uv", "f_decal", "f_aniso",
                               "f_layer3", "f_paint_vc", "f_tex_anim",
                               "f_overlay2", "f_blend_mode",
                               "f_blend_effects", "f_blend_effects3",
                               "f_use_new_blending", "f_enable_vis",
                               "f_alpha_test",
                               "f_alpha_test_layer", "f_material_ref"):
                        _m = (x_gt[..., EXT2[_k]] > 0.0) & fg_o
                        GT_REACH[_k] = GT_REACH.get(_k, 0.0) + \
                            float(_m.float().mean())
                    GT_REACH["_n"] = GT_REACH.get("_n", 0.0) + 1.0
            w_eff, band, e_o, tmv = w_raw, None, None, None
            hc1 = None
            hgt_o = None            # _17125, the layer-blended height
            if VMAT:
                e_o = MAT_EXT[mid_op]
                if args.vertex_color_mode:
                    # g_nVertexColorMode<N> says which LAYER COLOR_0 is the
                    # blend weight OF. This renderer has always assumed
                    # layer 2. Mode 2 on layer 1 with no layer-2 mode holds on
                    # exactly three materials -- inferno_stonefloor07_{dirt,
                    # concrete,gravel}_blend, the plaza ground -- and there
                    # COLOR_0 is layer 1's weight, so layer 2 gets its
                    # complement. Read off the flag, not fitted.
                    lay1 = ((_ext(e_o, "vcmode1") > 1.5)
                            & (_ext(e_o, "vcmode2") < 1.5))
                    w_raw = torch.where(lay1,
                                        (1.0 - w_raw) * sub_has2[m_op][i_o],
                                        w_raw)
                    w_eff = w_raw
                if args.height_blend or args.height_ch1 != "off" or args.weather:
                    hs1 = sample_aux(uv_o, MAT_H1[mid_op], tapsA_o)
                    hs2 = sample_aux(uv_o, MAT_H2[mid_op], tapsA_o)
                    h1, h2 = hs1[..., 0], hs2[..., 0]
                    # glsl:969 _17125 = mix(height1, height2, blendWeight) --
                    # the same blended height the puddle depth test reads
                    hgt_o = h1 * (1 - w_raw) + h2 * w_raw
                    # ch1 of the same vtex: a second structured map, blended
                    # on the same weight, neutral where there is no height map.
                    hc1 = hs1[..., 1] * (1 - w_raw) + hs2[..., 1] * w_raw
                    hc1 = torch.where(_ext(e_o, "has_h1") > 0.5, hc1,
                                      torch.ones_like(hc1))
                if args.height_blend:
                    w_h, band = height_blend(h1, h2, w_raw, e_o, x_gt)
                    # Only the materials that actually carry a layer-2 height
                    # map may deviate; the rest keep the linear lerp.
                    use = (_ext(e_o, "has_h2") * sub_has2[m_op][i_o])
                    w_eff = w_raw + (w_h - w_raw) * use
                    band = band * use
                if args.tint_mask or args.overlay:
                    # S_TINT_MASK: the mask comes from whichever slot the
                    # material binds -- g_tTintMask on csgo_complex, the
                    # HEIGHT texture's green channel on
                    # csgo_environment_blend (glsl:593).
                    _tm_raw = sample_aux(uv_o, MAT_TM[mid_op], tapsA_o)
                    # W1's prediction, tested here rather than argued:
                    # csgo_environment_ps reads the tint-mix factor out of a
                    # PACKED material texture's channel (_5618._m12.y at
                    # :517, `clamp(tex0.y, 0, 1)`), and our mat_tm slot is
                    # AUX_WHITE on 327/327 materials. If the sampled mask is
                    # pinned at 1.0 then the FULLY-TINTED branch runs on
                    # every material everywhere -- a whole-frame wrong-branch
                    # defect, not a missing texture. at-identity(1) == 1.0000
                    # confirms it; anything less refutes it.
                    _gt_value("material.tint_mask_raw", _tm_raw, identity=1.0,
                              note="csgo_environment_ps.glsl:517 mix weight "
                                   "(_m12.y). 1.0 everywhere = the tinted "
                                   "branch on 100% of pixels, from an "
                                   "AUX_WHITE slot rather than from a page")
                    tmv = gt_tint_mask_value(
                        _tm_raw,
                        sample_aux(uv_o, MAT_H1[mid_op], tapsA_o),
                        x_gt, e_o) if GTSURF else tint_mask_value(
                            _tm_raw, e_o, w_eff)
                    _gt_value("material.tint_mask_used", tmv, identity=0.0,
                              note="the same weight after the family's own "
                                   "selection. identity 0.0 here = no tint; "
                                   "compare against tint_mask_raw to see "
                                   "whether the selection rescued a pinned "
                                   "input or passed it through")
            alb = shade_albedo(uv_o, mat_o, sub_mat2[m_op][i_o],
                               sub_has2[m_op][i_o],
                               sub_params[m_op][i_o], w_eff, tapsC_o,
                               MTINT[mid_op] if (MTINT is not None
                                                 and args.model_tint != "always"
                                                 and mid_op is not None)
                               else None,
                               uv_l2=uv_l2)
            # --- GROUND-TRUTH MATERIAL-SURFACE COMBO AXES, albedo side ---
            band3, w3_eff = None, None
            if GTSURF:
                rgb_g = alb[..., :3]
                if args.layer3:
                    _h12 = ((h1 - _ext(e_o, "h_zero1")) * (1 - w_eff)
                            + (h2 - _ext(e_o, "h_zero2")) * w_eff) \
                        if (args.height_blend and VMAT) \
                        else torch.zeros_like(w_eff)
                    rgb_g, w3_eff, band3 = gt_layer3(
                        rgb_g, sample_fam(uv_o, T_COL3[mid_op]), _h12,
                        sample_fam(uv_o, T_HGT3[mid_op])[..., 0], w3_raw, x_gt)
                if args.detail_texture:
                    rgb_g = gt_detail_texture(
                        rgb_g,
                        sample_fam(_uv_transform(uv_o, _e2(x_gt, "detail_su"),
                                                 _e2(x_gt, "detail_sv"),
                                                 _e2(x_gt, "detail_ou"),
                                                 _e2(x_gt, "detail_ov")),
                                   T_DETAIL[mid_op]),
                        x_gt)
                if args.decal_texture:
                    rgb_g = gt_decal(
                        rgb_g,
                        sample_fam(_uv_transform(uv_o, _e2(x_gt, "decal_su"),
                                                 _e2(x_gt, "decal_sv"),
                                                 _e2(x_gt, "decal_ou"),
                                                 _e2(x_gt, "decal_ov")),
                                   T_DECAL[mid_op]),
                        x_gt)
                if args.material_reference:
                    rgb_g = gt_material_reference(rgb_g, x_gt)
                alb = torch.cat([rgb_g, alb[..., 3:]], dim=-1)
            if VMAT:
                rgb_o = alb[..., :3]
                if args.blend_border and band is not None:
                    rgb_o = blend_border(rgb_o, band, e_o, MAT_BTINT[mid_op])
                if args.tint_mask:
                    # The mask value is already the GT remap; tint_mask()
                    # only decides WHERE g_vColorTint lands.
                    _m = (tmv * _ext(e_o, "f_tintmask")).unsqueeze(-1)
                    rgb_o = rgb_o * (1 - _m) + rgb_o * MAT_CTINT[mid_op] * _m
                    # csgo_environment_blend_ps.glsl:604 -- the vertex-colour
                    # tint is gated by the SAME mask, and nothing implemented
                    # it before.
                    rgb_o = gt_vertex_color_tint(rgb_o, vc_o, tmv, e_o)
                if args.overlay:
                    # The overlay has its own coordinate stream and its own
                    # transform (csgo_environment_blend_vs_max.glsl:441-455,
                    # output _5245), which nothing read before.
                    #
                    # g_nColorOverlayUVSet is `0=Biplanar,1=UV1,2=UV2` and
                    # defaults to 2. All 11 de_inferno overlay materials leave
                    # it unset, so all 11 take UV2 and the single-fetch branch
                    # at glsl:1209; the biplanar two-tap branch is dead here.
                    # SUBSTITUTION, stated because the pack cannot honour it:
                    # ash_to_world logs "DEFAULTED: uvs2 = uvs (no TEXCOORD_1
                    # selection)", so uv2_o IS uv_o in every pack built so far
                    # and the overlay lands on the layer-1 coordinate. The
                    # variable is written the way the shader reads it, so the
                    # day a pack carries a real TEXCOORD_1 this needs no edit.
                    _uv_ov = uv2_o
                    if GTSURF:
                        _uv_ov = _uv_transform(
                            gt_uv(uv_o, uv2_o, "uvsel_ovl", x_gt, "uv2_su",
                                  "uv2_sv", "uv2_ou", "uv2_ov"),
                            _e2(x_gt, "ovl_su"), _e2(x_gt, "ovl_sv"),
                            _e2(x_gt, "ovl_ou"), _e2(x_gt, "ovl_ov"))
                    _ov = sample_aux(_uv_ov, MAT_OVL[mid_op], tapsA_o)
                    _gt_value("material.overlay_signed", _ov[..., :3] * 2.0 - 1.0,
                              identity=0.0,
                              note="s = 2*texel - 1 at glsl:1211, the signed "
                                   "page the composite shapes. READ THE MEAN, "
                                   "NOT THE FRACTION: at-identity counts an "
                                   "EXACT 0 and the AUX_HALF sentinel is byte "
                                   "128, i.e. s = +0.00392, so this fraction "
                                   "reads 0.0000 even on a pack carrying no "
                                   "overlay page at all. A mean at +0.004 is "
                                   "that all-sentinel pack; anything else is "
                                   "real pages reaching the shader")
                    if GTSURF:
                        # g_nColorOverlayMode selects a LAYER, not a composite
                        # -- see gt_overlay_layer_k(), which replaced the
                        # four-composite dispatch this used to call.
                        _mode = (torch.full_like(_e2(x_gt, "overlay_mode2"),
                                                 float(args.overlay_mode))
                                 if args.overlay_mode >= 0
                                 else _e2(x_gt, "overlay_mode2"))
                        _k = gt_overlay_layer_k(
                            _mode, _e2(x_gt, "overlay_tintmask2"),
                            _e2(x_gt, "f_overlay2") * _e2(x_gt, "has_overlay2")
                            * args.overlay_gain, tmv, w_eff)
                        rgb_o = overlay_composite(rgb_o, _ov,
                                                  _e2(x_gt, "ovl_bright"),
                                                  _e2(x_gt, "ovl_dark"), _k)
                    else:
                        rgb_o = shared_overlay(rgb_o, _ov, e_o, tmv, w_eff)
                alb = torch.cat([rgb_o, alb[..., 3:]], dim=-1)
            nrm = nrm_o
            fg = fg_o
            depth = depth_o
            # --- S_MODE_DEPTH buffer (see --mode-depth) --------------------
            # R = the depth module's own output alpha, G = 1 where the glass
            # translucent clip KEPT the fragment, B = 1 where the shader
            # family's S_MODE_DEPTH combo has no pixel shader at all.
            dm = None
            if args.mode_depth:
                dm = torch.zeros(c1 - c0, H, W, 3, device=device)
                dm[..., 0] = fg_o.float()
                if FAMILY:
                    # csgo_environment s32: m_byteCodeIndex is -1 for all 8
                    # dynamic combos -- the depth pass binds NO fragment
                    # shader. That is a measured whole-combo property, so it
                    # is reported as its own channel rather than as a zero
                    # that would read like a missing feature.
                    _noPS = (MAT_FAM[mid_op]
                             == FAM_NAMES.index("csgo_environment.vfx")) & fg_o
                    dm[..., 2] = _noPS.float()
                    dm[..., 0] = dm[..., 0] * (~_noPS).float()
            # --- PBR surface set, opaque pass ---------------------------
            # Only the opaque pass gets the full surface set: it owns ~95%
            # of the covered pixels, and MASK/BLEND would need their own
            # tangent interpolation per class for a fraction of a percent.
            rough = metal = ao_o = emis = view_o = None
            CHAR_PIX = EYE_PIX = GLOVE_PIX = None
            aniso_o = aniso_dir_o = None
            wpos_o = None
            trans_o = two_sided = None
            unlit_o = None
            n_geo_o = None
            # THE ATTRIBUTION MAP, now UNCONDITIONAL (#80 option (c)).
            # It used to be built only for --fam-reach / --dirocc /
            # csgo_simple, which meant the one map that knows which class
            # owns a pixel existed only when a diagnostic asked for it,
            # while the pixel census ran off a different, opaque-only map.
            # It is now the SINGLE attribution: every raster class writes
            # it, it is counted once at the composite, and the census
            # balances against the coverage the classes reported.
            # The remaining conditions are not policy, they are existence:
            # MAT_FAM is only bound under `if FAMILY:` (--fam-side), and
            # mid_op only under PBR. Without a family table there are no
            # class names to attribute TO, which the matrix already prints
            # as UNAVAILABLE.
            #
            # Opaque-uncovered pixels start UNATTRIBUTED rather than
            # carrying MAT_FAM of the clamped face 0. That junk was
            # harmless while nothing counted it and is not harmless now:
            # it would hand every unclaimed pixel to whichever class owns
            # face 0.
            fam_pix = (torch.where(fg_o, MAT_FAM[mid_op],
                                   torch.full_like(MAT_FAM[mid_op],
                                                   FAM_UNATTRIBUTED))
                       if (FAMILY and PBR and mid_op is not None) else None)
            mid_pix = mid_op.clone() \
                if (args.matid_dump and mid_op is not None) else None
            own_o = fg_o           # pixels the opaque pass still owns
            # coverage-site tally, the census's independent operand
            if FAMILY and PBR and mid_op is not None:
                class_px_won(MAT_FAM[mid_op], fg_o)
            if PBR:
                mid_o = mid_op
                n_geo, _ = dr.interpolate(VS_VNRM[used][None].contiguous(),
                                          rast_o, local[m_op])
                nrm = _nrm(n_geo)
                n_geo_o = nrm
                # The layer-2 normal blends on the SAME weight the colour
                # did, height blend included -- otherwise the bump and the
                # albedo disagree about where the transition is.
                w_o = w_eff.unsqueeze(-1)
                nmap = None
                tan4 = None
                ref_rough = None
                if NRM_PAGES is not None and (args.normal_map
                                              or args.rough_src
                                              == "normal-alpha"):
                    # THE REFERENCE PAGES. Both layers decoded through
                    # gt_ref_normal_rough and blended on w_eff -- the SAME
                    # weight the albedo used, which is what :456/:480 do with
                    # _6616.x/_6616.y for the colour and :458/:476 do with .z
                    # for the roughness.
                    #
                    # Roughness is blended as a WEIGHTED SUM in the reference
                    # (_18178 + _12168.z * _6616.y) and w_eff here is a single
                    # lerp weight, so (1-w)*r1 + w*r2 IS that sum for weights
                    # that partition. corpus-2 CONFIRMED the weights partition
                    # as (w, 1-w) on this map, so the lerp is EXACT here and
                    # not an approximation. It stops being exact the moment a
                    # pack's weights do not sum to one, which is why the form
                    # is still written down rather than assumed away.
                    _t1 = sample_pages(NRM_PAGES, NRM_EDGE, uv_o,
                                       MAT_NRM1[mid_o])
                    _t2 = sample_pages(NRM_PAGES, NRM_EDGE, uv_o,
                                       MAT_NRM2[mid_o])
                    _n1, _r1 = gt_ref_normal_rough(_t1)
                    _n2, _r2 = gt_ref_normal_rough(_t2)
                    _nts = _nrm(_n1 * (1 - w_o) + _n2 * w_o)
                    ref_rough = (_r1 * (1 - w_eff) + _r2 * w_eff).clamp(0.03,
                                                                        1.0)
                    # apply_normal_map takes an ENCODED texel and does
                    # rgb*2-1; this normal is already decoded, so it is
                    # re-encoded rather than passed raw. Feeding a decoded
                    # vector to a decoder is the silent-halving bug this
                    # avoids by construction.
                    nmap = torch.cat([_nts * 0.5 + 0.5,
                                      torch.ones_like(_nts[..., :1])], dim=-1)
                elif args.normal_map or args.rough_src == "normal-alpha":
                    n1 = sample_aux(uv_o, MAT_NRM1[mid_o], tapsA_o)
                    n2 = sample_aux(uv_o, MAT_NRM2[mid_o], tapsA_o)
                    nmap = n1 * (1 - w_o) + n2 * w_o
                if args.normal_map:
                    tan4, _ = dr.interpolate(VS_VTAN[used][None].contiguous(),
                                             rast_o, local[m_op])
                    nrm = apply_normal_map(n_geo, tan4, nmap)
                    if args.bevel and band is not None:
                        # F_BLEND_EFFECTS_2 bevel: CS2 rounds the lip where
                        # layer 2 sits proud of layer 1 by pushing the normal
                        # along the blend gradient. Approximated here by
                        # tilting towards the geometric normal (strength < 0
                        # on 154/329, i.e. usually a softening).
                        bs = (_ext(e_o, "bevel_strength2")
                              * _ext(e_o, "f_blend_effects2")
                              * band / _ext(e_o, "bevel_soft2").clamp(min=1e-3))
                        nrm = _nrm(nrm + _nrm(n_geo) * bs.unsqueeze(-1))
                    if args.detail_normal:
                        # S_DETAIL_NORMAL: the per-layer detail normal is
                        # composited in TANGENT space, before the frame is
                        # applied, which is the only place the two normals
                        # are in the same basis.
                        _d1 = sample_fam(uv_o, T_DNRM1[mid_o])
                        _d2 = sample_fam(uv_o, T_DNRM2[mid_o])
                        _ts = torch.cat([nmap[..., :2] * 2.0 - 1.0,
                                         (nmap[..., 2:3] * 2.0 - 1.0)], dim=-1)
                        _ts = gt_detail_normal(_nrm(_ts), _d1, x_gt, w_eff, _d2)
                        nrm = apply_normal_map(
                            n_geo, tan4,
                            torch.cat([_ts * 0.5 + 0.5, nmap[..., 3:]], dim=-1))
                _normal_map_effect_check(nrm, n_geo, fg_o)
                if not args.vertex_normals:
                    nrm = nrm_o
                if (args.specular or args.emissive or args.fog
                        or args.vmat_ao or args.transmissive or args.parallax3):
                    wpos_o, _ = dr.interpolate(VS_VPOS[used][None].contiguous(),
                                               rast_o, local[m_op])
                # --- shader families, opaque class -------------------------
                if FAMILY:
                    x_o = MAT_EXT2[mid_o]
                    rgb_f = alb[..., :3]
                    if args.parallax3:
                        if tan4 is None:
                            tan4, _ = dr.interpolate(
                                VS_VTAN[used][None].contiguous(), rast_o,
                                local[m_op])
                        rgb_f = parallax3(
                            rgb_f, uv_o, mid_o, x_o, n_geo, tan4,
                            _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wpos_o),
                            MAT_UVSCALE[mid_o])
                    if args.paint_vertex_colors:
                        rgb_f = paint_vertex_colors(rgb_f, vc_o, x_o)
                    if args.complex_tint:
                        rgb_f = unlit_tint(rgb_f, x_o, MAT_CTINT2[mid_o], e_o)
                    # === NON-WORLD FAMILIES, opaque class =================
                    # FAM_CHARACTER / FAM_EYEBALL / FAM_CUSTOMGLOVE, selected
                    # by MAT_FAM, reaching real pixels. Each family's surface
                    # function returns a full surface set and the result is
                    # lerped in on the family's own 0/1 gate, which is the
                    # same idiom glass and water already use.
                    # ONE PRECONDITION, not a guard per dereference. The
                    # character pixel path reads g_tAnisoGloss (T_ANISO) and
                    # g_tMetalness (T_MET2), and BOTH are bound only on the
                    # GTSURF branch (:7032, :7080). Without it this block died
                    # three different ways at three different lines -- and it
                    # dies identically on pristine merged main, which is where
                    # this was verified rather than assumed. The unit is the
                    # block's shared precondition, exactly as at :22743 for the
                    # zero-geometry case; guarding each deref in turn just moves
                    # the crash down the block.
                    if NONWORLD and args.char and T_ANISO is None:
                        global _CHAR_GTSURF_REFUSED
                        if not _CHAR_GTSURF_REFUSED:
                            print("REFUSED --char: the csgo_character pixel path "
                                  "reads g_tAnisoGloss and g_tMetalness, which "
                                  "are carried by the GTSURF side-table group "
                                  "and are UNBOUND in this run. Pass "
                                  "--gt-axes-reach (or any --gt axis) to bind "
                                  "them. The family shades NOTHING here and is "
                                  "reported as not run, not quietly skipped.",
                                  flush=True)
                            _CHAR_GTSURF_REFUSED = True
                    if NONWORLD and args.char and T_ANISO is not None:
                        if tan4 is None:
                            tan4, _ = dr.interpolate(
                                VS_VTAN[used][None].contiguous(), rast_o,
                                local[m_op])
                        if wpos_o is None:
                            wpos_o, _ = dr.interpolate(
                                VS_VPOS[used][None].contiguous(), rast_o,
                                local[m_op])
                        _eye3 = eye[c0:c1].view(-1, 1, 1, 3)
                        _view = _nrm(_eye3 - wpos_o)
                        _nts = (char_normal_decode(nmap) if nmap is not None
                                else torch.tensor([0.0, 0.0, 1.0],
                                                  device=device)
                                .expand_as(n_geo))
                        _gloss = sample_fam(uv_o, T_ANISO[mid_o])
                        _mtex = sample_fam(uv_o, T_MET2[mid_o])
                        _curv = (CHAR_CURV_PIX if CHAR_CURV_PIX is not None
                                 else torch.zeros_like(uv_o[..., 0]))
                        if args.char_curvature == "vertex" and VS_SKIN_CURV \
                                is not None:
                            _curv, _ = dr.interpolate(
                                VS_SKIN_CURV[used][None, :, None].contiguous(),
                                rast_o, local[m_op])
                            _curv = _curv[..., 0]
                        elif args.char_curvature == "derivative":
                            # S_USE_PER_VERTEX_CURVATURE = 0: the screen-space
                            # form the shipped SSS module uses when it has no
                            # curvature stream, r1050_m0:461-463 --
                            # length(fwidth(N)) / length(fwidth(P)).
                            _dn = (n_geo - torch.roll(n_geo, 1, dims=2)).norm(-1)
                            _dp = (wpos_o - torch.roll(wpos_o, 1, dims=2)) \
                                .norm(-1).clamp(min=1e-6)
                            _curv = (_dn / _dp).clamp(0, 1)
                        _ao_in = (ao_o if ao_o is not None
                                  else torch.ones_like(uv_o[..., 0]))
                        _noise = torch.rand_like(uv_o[..., 0])
                        (_crgb, _ca, _cn, _cr2, _cm, _cao, _cf0, _ccl, _crt,
                         _chr, _cctx, _con) = character_surface(
                            rgb_f, alb[..., 3], uv_o, uv2_o, mid_o, x_o,
                            n_geo, tan4, _nts, _view, wpos_o, _eye3,
                            # gl_FrontFacing: the rasteriser hands the opaque
                            # class front faces only unless F_RENDER_BACKFACES
                            # is set, so the flag IS the front-facing bit here
                            # and the backface-flip branch is reachable through
                            # it rather than being permanently off.
                            (1.0 - _e2(x_o, "f_char_backfaces")),
                            vc_o, _ao_in, _mtex, _gloss, _curv,
                            uvda_o[..., :2].norm(dim=-1)
                            if uvda_o is not None
                            else torch.zeros_like(uv_o[..., 0]),
                            float(tsec or 0.0), CHAR_ORIGIN, _noise)
                        rgb_f = _crgb
                        CHAR_PIX = (_cn, _cr2, _cm, _cao, _cf0, _ccl, _crt,
                                    _chr, _cctx, _con, _view)
                        FAM_HIT["character"] += int(
                            ((MAT_FAM[mid_o] == FAM_CHARACTER) & fg_o).sum())
                    if NONWORLD and args.eyeball and FAM_EYEBALL >= 0:
                        if wpos_o is None:
                            wpos_o, _ = dr.interpolate(
                                VS_VPOS[used][None].contiguous(), rast_o,
                                local[m_op])
                        _eye3 = eye[c0:c1].view(-1, 1, 1, 3)
                        _r2e = torch.stack([torch.full_like(uv_o[..., 0], 0.5)] * 2,
                                           dim=-1)
                        _erg, _er2, _en, _ef0, _eb = eyeball_surface(
                            rgb_f, _r2e, _nrm(n_geo), wpos_o, _eye3, uv_o,
                            mid_o, x_o, torch.ones_like(uv_o[..., 0]))
                        _eg = ((MAT_FAM[mid_o] == FAM_EYEBALL).float()
                               ).unsqueeze(-1)
                        rgb_f = rgb_f + (_erg - rgb_f) * _eg
                        EYE_PIX = (_en, _er2, _ef0, _eb, _eg)
                        FAM_HIT["eyeball"] += int(
                            ((MAT_FAM[mid_o] == FAM_EYEBALL) & fg_o).sum())
                    if NONWORLD and args.glove and FAM_CUSTOMGLOVE >= 0:
                        if tan4 is None:
                            tan4, _ = dr.interpolate(
                                VS_VTAN[used][None].contiguous(), rast_o,
                                local[m_op])
                        _nts_g = (char_normal_decode(nmap) if nmap is not None
                                  else torch.tensor([0.0, 0.0, 1.0],
                                                    device=device)
                                  .expand_as(n_geo))
                        _grg, _ga, _gn, _gr2, _gm, _gao = customglove_surface(
                            rgb_f, alb[..., 3], uv_o, mid_o, x_o, _nts_g,
                            tan4, _nrm(n_geo))
                        _gg = ((MAT_FAM[mid_o] == FAM_CUSTOMGLOVE).float()
                               ).unsqueeze(-1)
                        rgb_f = rgb_f + (_grg - rgb_f) * _gg
                        # GLOVE ALPHA IS A WRITTEN CHANNEL, not a spare
                        # return value. csgo_customglove_ps_r3_m0.glsl
                        # declares ONE output, `layout(location = 0) out
                        # vec4 _3711` (:223), and its .w is live:
                        #   F_OUTPUT_MODE 1 writes .w = roughnessY  (:1976)
                        #   modes 0/2/3/4 write .w = 1.0   (:1936 and after)
                        # customglove_surface computes exactly that and
                        # returns it, and BOTH call sites in this file threw
                        # it away -- a family's declared output going
                        # nowhere. Composited over the SAME family mask the
                        # colour uses, which makes our output vec4 the
                        # reference's output vec4 rather than three
                        # quarters of it.
                        _TERMSTATS.observe("customglove.alpha", _ga,
                                    before=alb[..., 3],
                                    mask=(MAT_FAM[mid_o] == FAM_CUSTOMGLOVE),
                                    reference="csgo_customglove_ps_r3_m0"
                                              ".glsl:1936-2071")
                        alb = torch.cat(
                            [alb[..., :3],
                             (alb[..., 3] + (_ga - alb[..., 3])
                              * _gg.squeeze(-1)).unsqueeze(-1),
                             alb[..., 4:]], dim=-1)
                        GLOVE_PIX = (_gn, _gr2, _gm, _gao, _gg)
                        FAM_HIT["customglove"] += int(
                            ((MAT_FAM[mid_o] == FAM_CUSTOMGLOVE) & fg_o).sum())
                    if args.fam_only is not None:
                        # last, so nothing above can tint the indicator
                        rgb_f = _fam_flag(mid_o).expand_as(rgb_f)
                    if args.unlit:
                        # csgo_black_unlit has no texture at all, so the pack
                        # hands it the white sentinel and it renders as a lit
                        # white wall. It is meant to be flat black.
                        ub = _e2(x_o, "is_unlit_black").unsqueeze(-1)
                        rgb_f = rgb_f * (1 - ub)
                        unlit_o = _e2(x_o, "is_unlit_black")
                    # --- BATCH-2 FAMILY DISPATCH ------------------------
                    # The six entry points are CALLED here, per family, on
                    # the pixels that family owns. `tap` is a BOUND
                    # REFERENCE, not a per-pixel test: b2_dispatch_lit
                    # returns the family's function or the pass-through, and
                    # the mask selects the pixels it is applied to.
                    if B2 and (B2_LIT or B2_COMPOSITORS):
                        if tan4 is None:
                            tan4, _ = dr.interpolate(
                                VS_VTAN[used][None].contiguous(), rast_o,
                                local[m_op])
                        if wpos_o is None:
                            wpos_o, _ = dr.interpolate(
                                VS_VPOS[used][None].contiguous(), rast_o,
                                local[m_op])
                        _view = _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wpos_o)
                        _fam_o = MAT_FAM[mid_o]
                        _ng = n_geo if n_geo is not None else nrm
                        rgb_f, nrm, _b2u = b2_run_dispatch(
                            rgb_f, uv_o, mid_o, x_o, _fam_o, _ng, tan4,
                            wpos_o, _view, eye[c0:c1], vc_o, depth_o, nrm)
                        # The six entry points return a COMPOSED colour --
                        # they run the shared cascade + baked + env tail
                        # themselves, because two of them (liquid's second
                        # pass, the sticker's rebuilt frame) need it at a
                        # point the outer pipeline cannot express. Their
                        # pixels are therefore routed through the SAME unlit
                        # compositor csgo_black_unlit uses, so the outer
                        # lighting cannot light them twice and there is one
                        # such path in this file rather than two.
                        unlit_o = (_b2u if unlit_o is None
                                   else torch.maximum(unlit_o, _b2u))
                    alb = torch.cat([rgb_f, alb[..., 3:]], dim=-1)
                if args.specular:
                    view_o = _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wpos_o)
                    if args.rough_remap:
                        pass   # applied below, after rough is resolved
                    if ref_rough is not None:
                        # .z of the REFERENCE page, already layer-blended.
                        # csgo_environment_ps.glsl:458 `_12167.z * _6616.x`,
                        # :476 `+ _12168.z * _6616.y`, :514 clamp(0,1). This
                        # is NOT the alpha the branch below reads, and the
                        # two are not reconcilable by a channel swap: an
                        # alpha-roughness page has a 3-vector normal in rgb
                        # and a reference page has a 2-vector in xy.
                        rough = ref_rough
                        _gt_value("surface.roughness_ref_z", rough,
                                  identity=args.roughness,
                                  note="csgo_environment_ps.glsl:458/:476 -- "
                                       "channel .z of mat_normal_pages, "
                                       "blended on the albedo's own weights")
                    elif args.rough_src == "normal-alpha":
                        # Source 2's csgo_* shaders carry roughness in the
                        # normal map's ALPHA. The glTF metallicRoughness slot
                        # cannot be used: VRF wrote height / ao / tintmask
                        # vtexes into it (plaster_facade_01_height on 160
                        # materials), so its G channel is not roughness.
                        #
                        # ⚠️ SETTLED, AND THE COMMENT ABOVE IS FALSE OF THIS
                        # PACK. csgo_environment_ps.glsl:458 reads roughness
                        # from .z; I qualified this branch rather than
                        # settling it because which channel is right about
                        # the PACK depended on what the builder wrote. It has
                        # now been MEASURED by corpus-2, and the answer is
                        # neither: the aux pages' alpha is CONSTANT 255
                        # across all 111 pages (min = max = mean = 255, one
                        # distinct value), because decode_normal() writes
                        # alpha = 255 as an admitted STAND-IN.
                        #
                        # So this branch has never read roughness at all. It
                        # reads 1.0 everywhere -- not the wrong roughness,
                        # NO roughness -- and every surface it shades is
                        # fully rough. Against mat_normal_pages .z, which is
                        # real (mean 165.56, std 56.16, full range), that is
                        # the whole of the term.
                        #
                        # It is still not DELETED: --rough-src is a selector
                        # a pack with real alpha could legitimately want, and
                        # deleting a consumer because today's producer writes
                        # a constant is how the next pack silently loses the
                        # path. What it gets instead is a RUNTIME check --
                        # ticket #34, uncertainty prints at runtime rather
                        # than living in a comment -- so any run on a pack
                        # whose alpha is constant SAYS SO instead of
                        # reporting a roughness that is a stand-in.
                        rough = nmap[..., 3].clamp(0.03, 1.0)
                        _rough_alpha_degenerate_check(nmap[..., 3])
                    else:
                        rough = torch.full_like(nrm[..., 0], args.roughness)
                    metal = (MAT_PBR[mid_o, 0] if args.metal_mode == "factor"
                             else torch.zeros_like(rough))
                    if args.metalness_tex:
                        # g_tMetalness / F_METALNESS_TEXTURE (41 materials).
                        # Metalness is otherwise forced to zero, so every
                        # metal surface in the map gets a dielectric F0.
                        # Channel G, not R: VRF packs these vtexes with the
                        # metalness in green (all three reachable layers have
                        # R identically 0.000 and G 0.918 / 0.177 / 0.000).
                        mt = sample_aux(uv_o, MAT_MET[mid_o],
                                        tapsA_o)[..., args.metalness_channel]
                        on = _ext(e_o, "f_metalness_tex")
                        if GTSURF:
                            # S_METALNESS_TEXTURE, wired through: the layer-2
                            # metalness map blends on the SAME weight the
                            # colour did, g_flMetalness1/2 is the value where
                            # the material declares the feature but binds no
                            # map (has_met 0 previously read the white
                            # sentinel and made the surface fully metal), and
                            # the feature is gated by the slot's presence.
                            _mt2 = sample_fam(uv_o, T_MET2[mid_o])[
                                ..., args.metalness_channel]
                            mt = torch.where(_e2(x_gt, "has_metal2") > 0.5,
                                             mt * (1 - w_eff) + _mt2 * w_eff, mt)
                            _hm = _ext(e_o, "has_met")
                            _ms = (_e2(x_gt, "metal_scalar1") * (1 - w_eff)
                                   + _e2(x_gt, "metal_scalar2") * w_eff)
                            mt = mt * _hm + _ms * (1 - _hm)
                            on = on * torch.maximum(_hm,
                                                    (_ms > 0).float())
                        metal = metal * (1 - on) + mt * on
                    if args.rough_remap and FAMILY:
                        # Source's brightness/contrast remap, the same
                        # ((x-0.5)*c+0.5)*b form the tint mask uses.
                        _xb = MAT_EXT2[mid_o]
                        _rb = _e2(_xb, "rough_bright1")
                        _rc = _e2(_xb, "rough_contrast1")
                        _on = _e2(_xb, "has_rough_remap")
                        _rr = (((rough - 0.5) * _rc + 0.5) * _rb).clamp(0.03, 1.0)
                        rough = rough * (1 - _on) + _rr * _on
                    if args.height_ch1 == "rough" and hc1 is not None:
                        rough = hc1.clamp(0.03, 1.0)
                    if args.blend_border and band is not None \
                            and args.rough_src == "normal-alpha":
                        # F_BORDER_ROUGHNESS_2 (169 materials): the border
                        # band carries its own roughness.
                        br = (_ext(e_o, "f_border_rough2")
                              * _ext(e_o, "f_blend_effects2") * band)
                        rough = (rough + (_ext(e_o, "border_rough2") - rough)
                                 * br).clamp(0.03, 1.0)
                    if args.aniso_gloss:
                        # S_ANISOTROPIC_GLOSS: two roughnesses and a
                        # direction. The renderer's specular is isotropic, so
                        # the pair is collapsed to the geometric mean for the
                        # lobe width and the split is carried as `aniso_o`
                        # for anything downstream that can use it. That
                        # collapse is the difference from a real anisotropic
                        # GGX; it is exact at a = 0 and understates the
                        # streak, never inventing one.
                        if tan4 is None:
                            tan4, _ = dr.interpolate(
                                vtangent[used][None].contiguous(), rast_o,
                                local[m_op])
                        _rt, _rb, aniso_dir_o = gt_aniso_gloss(
                            rough, sample_fam(uv_o, T_ANISO[mid_o]), x_gt,
                            nrm, tan4, view_o)
                        aniso_o = (_rt, _rb)
                        rough = (_rt * _rb).clamp(min=1e-6).sqrt().clamp(0.03, 1.0)
                if AO_PAGES is not None:
                    # g_tAmbientOcclusion from the pack's own AO pages. It
                    # reaches the image through gt_compose, which already
                    # multiplies `ao` into the INDIRECT term at the position
                    # both files put it -- inside the _gt_lm_tint argument
                    # (:1230 q0, :1588-1592 q1), i.e. BEFORE the tint pow,
                    # which is where 3e2f74ae moved the tint to. So this
                    # supplies the term; it does not re-site it.
                    ao_o = sample_pages(AO_PAGES, AO_EDGE, uv_o,
                                        MAT_AO[mid_o])[..., 0]
                    _gt_value("surface.ao_page", ao_o, identity=1.0,
                              note="mat_ao_pages .x -- csgo_environment_ps"
                                   ".glsl:1230/:1588-1592, multiplied into "
                                   "the indirect before the tint pow")
                    # An AO term reading 1.0 almost everywhere has TWO
                    # explanations that the value alone cannot separate: the
                    # visible materials carry no page, or they carry pages
                    # that are white where this camera looks. AO maps are
                    # mostly white by nature, so the second is not a stretch.
                    # This is the material-side fact, which does separate
                    # them: the fraction of frame pixels whose material was
                    # given a page at all. Near 0 = a coverage answer; near 1
                    # with ao_page at identity = the pages really are white
                    # here.
                    _gt_value("surface.ao_page_bound",
                              (MAT_AO[mid_o] != 1).float(), identity=0.0,
                              note="1 where the pixel's material has a REAL "
                                   "mat_ao page, 0 where mat_ao is still the "
                                   "AUX_WHITE sentinel. Pairs with "
                                   "surface.ao_page: this says whether the "
                                   "term COULD act, that one says whether it "
                                   "DID")
                    if args.vmat_ao:
                        ao_o = ao_o * sample_aux(uv_o, MAT_AOV[mid_o],
                                                 tapsA_o)[..., 0]
                elif args.vmat_ao:
                    # The aux MAT_AO read went with --ao. What remains here is
                    # the vmat AO slot, which is a different table (MAT_AOV) and
                    # a different question; it keeps its own flag.
                    ao_o = torch.ones_like(uv_o[..., 0])
                    if args.vmat_ao:
                        # g_tAmbientOcclusion: the vmat AO slot, on 212
                        # materials vs the 213-but-mostly-1x1 glTF
                        # occlusionTexture. Combined multiplicatively so
                        # --ao and --vmat-ao can be scored independently or
                        # together.
                        ao_o = ao_o * sample_aux(uv_o, MAT_AOV[mid_o],
                                                      tapsA_o)[..., 0]
                if args.height_ch1 == "ao" and hc1 is not None:
                    ao_o = hc1 if ao_o is None else ao_o * hc1
                # === NON-WORLD FAMILIES: the surface set they own ==========
                # The three families rewrite the normal, the roughness, the
                # metalness and the AO the rest of the chain consumes, on
                # their own 0/1 family gate. Applied HERE -- after the world
                # families have resolved theirs and before the lighting reads
                # them -- so a character pixel is lit through the SAME cascade,
                # binner, probe and IBL path as everything else.
                if CHAR_PIX is not None:
                    (_cn, _cr2, _cm, _cao, _cf0, _ccl, _crt, _chr, _cctx,
                     _con, _cview) = CHAR_PIX
                    _cg = ((MAT_FAM[mid_o] == FAM_CHARACTER).float()
                           * _con[..., 0]).unsqueeze(-1)
                    nrm = _nrm(nrm + (_cn - nrm) * _cg)
                    if rough is not None:
                        # collapse the anisotropic PAIR to the lobe width the
                        # renderer's isotropic chain can carry; the SPLIT is
                        # not lost -- char_direct_delta() below evaluates the
                        # real anisotropic lobe and adds the difference.
                        _cri = (_cr2[..., 0] * _cr2[..., 1]).clamp(min=1e-6) \
                            .sqrt().clamp(0.03, 1.0)
                        rough = rough + (_cri - rough) * _cg[..., 0]
                        aniso_o = (_cr2[..., 0], _cr2[..., 1])
                        aniso_dir_o = _cctx[5]
                    if metal is not None:
                        metal = metal + (_cm - metal) * _cg[..., 0]
                    if ao_o is not None:
                        ao_o = ao_o + (_cao - ao_o) * _cg[..., 0]
                    # the lobe SHAPE the world families do not have
                    _l = sun_dir.view(1, 1, 1, 3).expand_as(nrm)

                    def _sss_lut(uvq):
                        return sample_fam(uvq, T_DFALLOFF[mid_o])
                    _dd, _sd = char_direct_delta(
                        _cr2, _cn, _l, _cview, _cf0, _cctx, _ccl, _crt, _chr,
                        x_o, _sss_lut if args.char_sss else None)
                    _sun = SUN_COLOR.view(1, 1, 1, 3)
                    _cdelta = (_dd * alb[..., :3] + _sd) * _sun * _cg
                    emis = _cdelta if emis is None else emis + _cdelta
                if EYE_PIX is not None:
                    _en, _er2, _ef0, _eb, _eg = EYE_PIX
                    nrm = _nrm(nrm + (_en - nrm) * _eg)
                    if rough is not None:
                        rough = rough + (_er2[..., 0] - rough) * _eg[..., 0]
                if GLOVE_PIX is not None:
                    _gn, _gr2, _gm, _gao, _gg = GLOVE_PIX
                    nrm = _nrm(nrm + (_gn - nrm) * _gg)
                    if rough is not None:
                        _gri = (_gr2[..., 0] * _gr2[..., 1]).clamp(min=1e-6) \
                            .sqrt().clamp(0.03, 1.0)
                        rough = rough + (_gri - rough) * _gg[..., 0]
                        aniso_o = (_gr2[..., 0], _gr2[..., 1])
                    if metal is not None:
                        metal = metal + (_gm - metal) * _gg[..., 0]
                    if ao_o is not None:
                        ao_o = ao_o + (_gao - ao_o) * _gg[..., 0]
                if args.emissive or args.self_illum:
                    em_tex = sample_aux(uv_o, MAT_EM[mid_o], tapsA_o)[..., :3] \
                        if args.emissive else torch.zeros_like(nrm)
                    emis = (em_tex * MAT_EMTINT[mid_o] * MAT_PBR[mid_o, 3:4]) \
                        if args.emissive else torch.zeros_like(nrm)
                    if args.self_illum:
                        # S_SELF_ILLUM (csgo_complex m_iFeatureIndex 10, bit
                        # weight 320; 5 of the 67 de_inferno materials select
                        # a static id containing it). The mask gates emission;
                        # g_flSelfIllumAlbedoFactor mixes the albedo into the
                        # emitted colour.
                        si = sample_aux(uv_o, MAT_SI[mid_o],
                                        tapsA_o)[..., args.self_illum_channel]
                        af = _ext(e_o, "si_albedo_factor").unsqueeze(-1)
                        on_si = _ext(e_o, "f_self_illum")
                        scale = MAT_PBR[mid_o, 3:4]
                        if GTSURF:
                            # A material with F_SELF_ILLUM but no
                            # g_tSelfIllumMask read the white sentinel and lit
                            # its whole surface; the mask is 1 there in the
                            # engine too, so the gate goes on the FEATURE and
                            # the sentinel is replaced by an explicit 1.
                            si = torch.where(_e2(x_gt, "has_selfillum2") > 0.5,
                                             si, torch.ones_like(si))
                            af = _e2(x_gt, "si_albedo_factor2").unsqueeze(-1)
                            on_si = on_si * _e2(x_gt, "f_self_illum2")
                            # g_flSelfIllumBrightness * g_flSelfIllumScale is
                            # what fast_pack2 folds into MAT_PBR[:,3]; read
                            # the two factors directly so a material that set
                            # only one of them is not silently zeroed.
                            scale = (_e2(x_gt, "si_brightness")
                                     * _e2(x_gt, "si_scale")).unsqueeze(-1)
                        base = MAT_EMTINT[mid_o] * (1 - af) \
                            + alb[..., :3].clamp(0, 1) * MAT_EMTINT[mid_o] * af
                        emis = emis + (base * si.unsqueeze(-1) * scale
                                       * on_si.unsqueeze(-1))
                # transmissive / backfaces are computed for the MASK class as
                # well (below) and carry their own coverage, so they are NOT
                # gated by pbr_mask: csgo_foliage is alpha-tested, so gating
                # them to opaque-owned pixels would apply the leaf-through-
                # light feature to everything EXCEPT the leaves.
                if args.blend_effects_3 and GTSURF and band3 is not None:
                    # S_BLEND_EFFECTS_3 -- the 2->3 seam, and ONLY that seam.
                    # The 1->2 seam is S_BLEND_EFFECTS_2 (--blend-border /
                    # --bevel / F_BORDER_ROUGHNESS_2, already implemented);
                    # the unsuffixed S_BLEND_EFFECTS is csgo_static_overlay's
                    # axis and is applied in the BLEND pass, not here.
                    _bt3 = torch.stack([_e2(x_gt, "btint3_r"),
                                        _e2(x_gt, "btint3_g"),
                                        _e2(x_gt, "btint3_b")], dim=-1)
                    _rgb, rough, nrm = gt_blend_effects(
                        alb[..., :3], rough,
                        nrm if args.normal_map else None,
                        n_geo, band3, x_gt, _bt3, "3")
                    alb = torch.cat([_rgb, alb[..., 3:]], dim=-1)
                if args.enable_visualizations and GTSURF:
                    # S_ENABLE_VISUALIZATIONS + F_VISUALIZATION_MODE: the
                    # authoring outputs replace the shaded colour where the
                    # material asks for them. Modes are numbered as the
                    # feature's own enum orders the quantities the blend
                    # shader has to show: 0 blend weight, 1 layer-2 height,
                    # 2 the transition band, 3 layer id. **The enum's
                    # labelling is not in the archives**; what IS established
                    # is that the axis exists on eb:ps at feature index 21
                    # and is 0 on all 43 de_inferno materials.
                    _vm = _e2(x_gt, "vis_mode").unsqueeze(-1)
                    _on = _e2(x_gt, "f_enable_vis").unsqueeze(-1)
                    _z = torch.zeros_like(alb[..., :3])
                    _w3 = w_eff.unsqueeze(-1).expand_as(_z)
                    _hh = (h1.unsqueeze(-1).expand_as(_z)
                           if (args.height_blend and VMAT) else _z)
                    _bb = (band.unsqueeze(-1).expand_as(_z)
                           if band is not None else _z)
                    _li = torch.stack([1.0 - w_eff, w_eff,
                                       w3_eff if w3_eff is not None
                                       else torch.zeros_like(w_eff)], dim=-1)
                    _vis = torch.where(_vm < 0.5, _w3,
                                       torch.where(_vm < 1.5, _hh,
                                                   torch.where(_vm < 2.5, _bb,
                                                               _li)))
                    alb = torch.cat([alb[..., :3] * (1 - _on) + _vis * _on,
                                     alb[..., 3:]], dim=-1)
                # --- FAMILY DISPATCH: csgo_simple / csgo_simple_3layer_parallax
                # Placed HERE and not up in the albedo block because both
                # families resolve a roughness, a metalness and an ambient
                # occlusion as well as a colour, and those three do not exist
                # until the PBR surface above has been built. Everything the
                # two shaders read is in scope at this point and nothing that
                # runs after it re-derives any of the five outputs, so the
                # families own their pixels from here to post_chain().
                if FAMILY and (SIMPLE_ON or S3LP_ON) and fam_pix is not None:
                    if tan4 is None:
                        tan4, _ = dr.interpolate(
                            VS_VTAN[used][None].contiguous(), rast_o,
                            local[m_op])
                    _nm = nmap if nmap is not None else torch.cat(
                        [torch.full_like(alb[..., :1], 128.0 / 255.0),
                         torch.full_like(alb[..., :1], 128.0 / 255.0),
                         torch.full_like(alb[..., :1], 1.0),
                         torch.full_like(alb[..., :1], 1.0)], dim=-1)
                    _rgb_s, rough, metal, ao_o, emis = simple_dispatch(
                        fam_pix, alb[..., :3], rough, metal, ao_o, emis,
                        MAT_EXT2[mid_o], mid_o, uv_o, uv2_o, vc_o, _nm,
                        n_geo, tan4, wpos_o,
                        eye[c0:c1].view(-1, 1, 1, 3), MAT_UVSCALE[mid_o])
                    alb = torch.cat([_rgb_s, alb[..., 3:]], dim=-1)
                if args.transmissive:
                    trans_o = transmissive_term(uv_o, mid_o, e_o, alb, tapsA_o,
                                                vc_o)
                if args.backfaces:
                    two_sided = _ext(e_o, "f_render_backfaces")
                if args.opaque_fade:
                    # D_OPAQUE_FADE on the opaque class. `x` is vColor.a, the
                    # same input csgo_complex s0/d18:249 and
                    # csgo_static_overlay s14/d1 feed it. Discarding here means
                    # the opaque pass stops owning the pixel, which is what a
                    # discard does: it neither writes colour nor depth.
                    _fon = torch.ones_like(vc_o[..., 3])
                    _fv, _fk = opaque_fade(vc_o[..., 3].clamp(0, 1), _fon)
                    FADE_STAT[0] += float((~_fk & fg_o).float().sum())
                    FADE_STAT[1] += float(fg_o.float().sum())
                    fg_o = fg_o & _fk
                    own_o = own_o & _fk
                    depth_o = torch.where(_fk, depth_o,
                                          torch.full_like(depth_o, FAR))
                    fg = fg_o
                    depth = depth_o
        else:
            # SITE 4b of the empty-visible-set family (#27). The view is
            # NOT empty -- mask and/or blend faces still draw -- only the
            # OPAQUE class selected zero faces. So `_empty_view` above does
            # not fire and the frame is a real frame; what is missing is
            # one class's contribution.
            #
            # THE OUTPUT CONTRACT. Everything the block above defines that
            # code below reads is written out here, at the value that code
            # ALREADY treats as "the opaque pass contributed nothing" --
            # read off the block's own absent-case assignments, not chosen:
            #   coverage   fg_o / fg / own_o -- all-False. `fg = fg_o` and
            #              `own_o = fg_o` are the block's own bindings; the
            #              mask class ORs into fg and the blend class ANDs
            #              into own_o, so False is "did not contribute".
            #   depth      depth_o / depth -- FAR, the SAME sentinel the
            #              populated path writes into every uncovered pixel
            #              at `torch.where(fg_o, rast_o[..., 2], FAR)`. Zero
            #              would win `rast_k[..., 2] < depth` and hide the
            #              mask and blend classes behind nothing.
            #   raster     rast_o, i_o, uv_o, uv2_o, vc_o, nrm_o, alb, nrm --
            #              zeros, channel counts taken from the real
            #              attribute tensors (uv_attr / uv2_attr / vc_attr /
            #              sub_norm) exactly as _raster's own empty return
            #              does, so no dtype or width is asserted here.
            #   surface    rough metal ao_o emis view_o wpos_o trans_o
            #              two_sided unlit_o n_geo_o n_geo nmap tan4 hgt_o
            #              aniso_o aniso_dir_o CHAR_PIX EYE_PIX GLOVE_PIX
            #              tapsC_o tapsA_o uvda_o vp_o -- None. Not invented:
            #              None is what the block itself assigns each of
            #              these before its feature gates, and every reader
            #              below is already an `is not None` test.
            #   ids        mid_op/mid_o/fam_pix/mid_pix
            #                             zeros, NOT None: LMG_LIVE and PBR
            #                             index MAT_FAM/MAT_EXT2 with mid_op
            #                             unconditionally, and every read is
            #                             masked by the all-False fg. This
            #                             is the same junk-but-masked value
            #                             the populated path leaves in an
            #                             uncovered pixel, where i_o is
            #                             clamped to face 0.
            #   dm         None, or the zeros buffer --mode-depth builds
            #
            # THE TWO DIAGNOSTIC CONSUMERS ARE NOW CLOSED TOO, so this
            # disclosure does not outlive its truth: --quad-overdraw's
            # quad_overdraw(clip, local[m_op]) and --tools-vis with
            # --secondary-uv=material's sub_sec[m_op][i_o] each REFUSE and
            # print rather than default -- see their sites. Their zeros are
            # the axis's own (no quad exists to charge; tools_vis already
            # renders sel 0 as "no set", black, distinct from 1 and 2), and
            # each print says in words that it is a refusal and not a
            # measurement, because a diagnostic that silently reports zero
            # is the failure mode this whole family exists to prevent.
            # NOTHING that consumes the opaque face selection is unguarded.
            print("opaque class EMPTY for frames %d..%d: %d mask / %d "
                  "blend faces still draw; opaque outputs defaulted "
                  "(an empty class is a valid frame state, #27 family "
                  "site 4b)"
                  % (c0, c1 - 1, int(m_mk.sum()), int(m_bl.sum())),
                  flush=True)
            _B = c1 - c0
            _dev = clip.device
            uvda_o = tapsC_o = tapsA_o = None
            rast_o = torch.zeros(_B, H, W, 4, device=_dev)
            fg_o = torch.zeros(_B, H, W, dtype=torch.bool, device=_dev)
            uv_o = torch.zeros(_B, H, W, uv_attr.shape[-1], device=_dev)
            uv2_o = torch.zeros(_B, H, W, uv2_attr.shape[-1], device=_dev)
            vc_o = torch.zeros(_B, H, W, vc_attr.shape[-1], device=_dev)
            nrm_o = torch.zeros(_B, H, W,
                                sub_norm.shape[-1] if sub_norm.dim() > 1
                                else 3, device=_dev)
            # FAR and not 0: the mask and blend classes both test
            # `rast_k[..., 2] < depth`, so an uncovered pixel has to LOSE
            # every depth comparison rather than win them all.
            depth_o = torch.full((_B, H, W), FAR, device=_dev)
            i_o = torch.zeros(_B, H, W, dtype=torch.long, device=_dev)
            vp_o = None
            mid_op = (torch.zeros(_B, H, W, dtype=sub_mid.dtype,
                                  device=_dev)
                      if (PBR or args.matid_dump) else None)
            mid_o = mid_op
            hgt_o = None
            # 4 channels: shade_albedo/sample_textures return RGB+A, and
            # the mask class composites into this with torch.where.
            alb = torch.zeros(_B, H, W, 4, device=_dev)
            nrm = nrm_o
            fg = fg_o
            depth = depth_o
            dm = None
            if args.mode_depth:
                dm = torch.zeros(_B, H, W, 3, device=_dev)
            rough = metal = ao_o = emis = view_o = None
            CHAR_PIX = EYE_PIX = GLOVE_PIX = None
            aniso_o = aniso_dir_o = None
            wpos_o = None
            trans_o = two_sided = None
            unlit_o = None
            n_geo_o = n_geo = None
            nmap = tan4 = None
            # the block's own guard expression, verbatim, so the two
            # branches cannot disagree about when the table exists. With no
            # opaque face there is no opaque coverage, so every pixel starts
            # UNATTRIBUTED and the mask and blend classes claim their own --
            # which is exactly the state that made the empty opaque class a
            # valid frame rather than a crash (#83).
            fam_pix = (torch.full_like(mid_op, FAM_UNATTRIBUTED)
                       if (FAMILY and PBR and mid_op is not None) else None)
            mid_pix = mid_op.clone() \
                if (args.matid_dump and mid_op is not None) else None
            own_o = fg_o

        if int(m_mk.sum()):
            rast_k, fg_k, uv_k, mat_k, nrm_k, uvda_k = _raster(
                clip, sub_faces[m_mk], sub_mat[m_mk], sub_norm[m_mk],
                uv_attr, local[m_mk])
            tapsC_k = mip_taps(uvda_k, T, ANISO_C, TEX_MIP[3]) \
                if TEX_MIP is not None and uvda_k is not None else None
            tapsA_k = mip_taps(uvda_k, TA, ANISO_A, AUX_MIP[3]) \
                if AUX_MIP is not None and uvda_k is not None else None
            alb_k = sample_textures(uv_k, mat_k, tapsC_k)
            i_k = (rast_k[..., 3].long() - 1).clamp(min=0)
            cut = sub_cut[m_mk][i_k]
            a_k = alb_k[..., 3]
            mid_k = sub_mid[m_mk][i_k] if PBR else None
            x_k = MAT_EXT2[mid_k] if FAMILY else None
            wpos_k = None
            if args.foliage:
                # g_flAlphaBoost*: CS2 thickens an alpha-tested card with
                # distance so a leaf does not dissolve into the sky as the
                # mip chain eats it. 43 of the 53 foliage materials set it
                # (strength 3.0, reference distance 20000 source units).
                wpos_k, _ = dr.interpolate(VS_VPOS[used][None].contiguous(),
                                           rast_k, local[m_mk])
                d = (eye[c0:c1].view(-1, 1, 1, 3) - wpos_k).norm(dim=-1) / S
                bd = _e2(x_k, "alpha_boost_dist")
                bs = _e2(x_k, "alpha_boost_str")
                a_k = (a_k * (1.0 + bs * (d / bd.clamp(min=1.0)).clamp(0, 1))
                       ).clamp(0, 1)
            # === NON-WORLD FAMILIES, mask class ======================
            # S_ALPHA_TEST on csgo_character, with the family's own
            # g_flAntiAliasedEdgeStrength edge softening. The face
            # reclassification put these faces here; this is the per-pixel
            # half, and without it the axis would be a face-class change
            # with no shading consequence.
            if NONWORLD and args.char and args.char_alpha_test \
                    and FAM_CHARACTER >= 0:
                _cmg = ((MAT_FAM[mid_k] == FAM_CHARACTER).float()
                        * _e2(x_k, "f_char_alpha_test"))
                _cfw = (uvda_k[..., :2].norm(dim=-1)
                        if uvda_k is not None
                        else torch.zeros_like(a_k))
                a_k = a_k + (char_alpha_test(a_k, _cfw, x_k) - a_k) * _cmg
                FAM_HIT["character"] += int((_cmg > 0).sum())
            if GTSURF and args.alpha_test_layer:
                # S_ALPHA_TEST_LAYER, per-pixel half. csgo_environment's
                # axis (RENDER_SETTINGS_AXES.md line 141 resolves it to
                # m_iFeatureIndex 8 = F_ALPHA_TEST -- there is no
                # F_ALPHA_TEST_LAYER material key to match on). The test
                # runs against the BLENDED layer alpha, not layer 1's, so
                # a two-layer material cuts where the composite is thin
                # rather than where its first layer is.
                _vck, _ = dr.interpolate(vc_attr[None].contiguous(), rast_k,
                                         local[m_mk])
                _wk = (_vck[..., args.blend_channel]
                       * sub_has2[m_mk][i_k]).clamp(0, 1)
                _a2 = sample_textures(uv_k, sub_mat2[m_mk][i_k],
                                      tapsC_k)[..., 3]
                _onl = _e2(x_k, "f_alpha_test_layer")
                a_k = a_k + ((a_k * (1 - _wk) + _a2 * _wk) - a_k) * _onl
            keep = fg_k & (a_k >= cut) & (rast_k[..., 2] < depth)
            # --- D_ALPHA_TEST_PREPASS ---------------------------------
            # foliage s10/d1:44-52. The binary `a < ref -> discard` is
            # REPLACED, not supplemented, by the derivative-normalised
            # coverage; the reference has exactly one test in each module.
            if args.alpha_test_prepass:
                cov_k, keep_a = alpha_test_prepass(a_k, cut)
                a_k = cov_k
            else:
                keep_a = a_k >= cut
            keep = fg_k & keep_a & (rast_k[..., 2] < depth)
            if args.opaque_fade:
                # foliage s10/d4:56 -- the fade snippet keys on the
                # EFFECTIVE alpha here, not on vColor.a, and the surviving
                # value replaces the output alpha.
                _fon = torch.ones_like(a_k)
                _fv, _fk = opaque_fade(a_k.clamp(0, 1), _fon)
                FADE_STAT[0] += float((keep & ~_fk).float().sum())
                FADE_STAT[1] += float(keep.float().sum())
                keep = keep & _fk
                a_k = _fv
            if dm is not None:
                # foliage s10/d0:43 -- `out = vec4(0,0,0, msaa ? a : 1)`.
                dm[..., 0] = torch.where(keep, a_k, dm[..., 0])
            k3 = keep.unsqueeze(-1)
            if GTSURF:
                # The albedo-side combo axes apply to the alpha-tested
                # class too: csgo_foliage and csgo_complex both live here
                # and both declare S_TINT_MASK / S_DETAIL_TEXTURE.
                _rk = alb_k[..., :3]
                if args.detail_texture:
                    _rk = gt_detail_texture(
                        _rk, sample_fam(
                            _uv_transform(uv_k, _e2(x_k, "detail_su"),
                                          _e2(x_k, "detail_sv"),
                                          _e2(x_k, "detail_ou"),
                                          _e2(x_k, "detail_ov")),
                            T_DETAIL[mid_k]), x_k)
                if args.decal_texture:
                    _rk = gt_decal(_rk, sample_fam(
                        _uv_transform(uv_k, _e2(x_k, "decal_su"),
                                      _e2(x_k, "decal_sv"),
                                      _e2(x_k, "decal_ou"),
                                      _e2(x_k, "decal_ov")),
                        T_DECAL[mid_k]), x_k)
                if args.material_reference:
                    _rk = gt_material_reference(_rk, x_k)
                alb_k = torch.cat([_rk, alb_k[..., 3:]], dim=-1)
            # --- MASK-class surface -----------------------------------
            # The alpha-tested class shaded on the FLAT per-face normal,
            # which is exactly why csgo_foliage (43 of its 53 materials
            # are MASK) reads as cardboard cutouts: every leaf card is one
            # constant-normal quad. It gets the same interpolated vertex
            # normal + tangent-space normal map the opaque class gets.
            if args.mask_normals and PBR:
                n_geo_k, _ = dr.interpolate(VS_VNRM[used][None].contiguous(),
                                            rast_k, local[m_mk])
                nk = _nrm(n_geo_k)
                nmap_k = None
                if args.normal_map:
                    tan4_k, _ = dr.interpolate(
                        VS_VTAN[used][None].contiguous(), rast_k,
                        local[m_mk])
                    nmap_k = sample_aux(uv_k, MAT_NRM1[mid_k])
                    nk = apply_normal_map(n_geo_k, tan4_k, nmap_k)
                if args.foliage:
                    # g_flVertexNormalInfluence (1.0 on 14 of 51, 0.1-0.2
                    # on 11): a leaf clump's VERTEX normals are authored
                    # to point out of the bush volume rather than out of
                    # the card, and this says how far to bend the shading
                    # normal back to them.
                    infl = _e2(x_k, "vtx_nrm_influence").clamp(0, 1)
                    nk = _nrm(nk * (1 - infl.unsqueeze(-1))
                              + _nrm(n_geo_k) * infl.unsqueeze(-1))
                nrm_k = nk
                if args.specular and rough is not None:
                    if wpos_k is None:
                        wpos_k, _ = dr.interpolate(
                            VS_VPOS[used][None].contiguous(), rast_k,
                            local[m_mk])
                    vk = _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wpos_k)
                    view_o = torch.where(k3, vk, view_o)
                    if nmap_k is not None and args.rough_src == "normal-alpha":
                        rough = torch.where(keep, nmap_k[..., 3].clamp(0.03, 1),
                                            rough)
                    if metal is not None:
                        metal = torch.where(keep, torch.zeros_like(metal),
                                            metal)
                if ao_o is not None and args.vmat_ao:
                    ao_o = torch.where(
                        keep, sample_aux(uv_k, MAT_AOV[mid_k])[..., 0], ao_o)
            if FAMILY and args.paint_vertex_colors:
                vc_k, _ = dr.interpolate(vc_attr[None].contiguous(), rast_k,
                                         local[m_mk])
                alb_k = torch.cat([paint_vertex_colors(alb_k[..., :3], vc_k,
                                                       x_k), alb_k[..., 3:]],
                                  dim=-1)
            if FAMILY and args.complex_tint:
                alb_k = torch.cat([unlit_tint(alb_k[..., :3], x_k,
                                              MAT_CTINT2[mid_k],
                                              MAT_EXT[mid_k] if VMAT else None),
                                   alb_k[..., 3:]], dim=-1)
            if FAMILY and args.fam_only is not None:
                alb_k = torch.cat([_fam_flag(mid_k).expand_as(alb_k[..., :3]),
                                   alb_k[..., 3:]], dim=-1)
            alb = torch.where(k3, alb_k, alb)
            nrm = torch.where(k3, nrm_k, nrm)
            depth = torch.where(keep, rast_k[..., 2], depth)
            fg = fg | keep
            if FAMILY and mid_k is not None:
                class_px_won(MAT_FAM[mid_k], keep)
            if fam_pix is not None:
                fam_pix = torch.where(keep, MAT_FAM[mid_k], fam_pix)
            if args.mask_normals and PBR:
                # the MASK class now carries a real surface, so it must be
                # inside pbr_mask instead of being excluded from spec/AO
                own_o = own_o | keep
            else:
                own_o = own_o & ~keep
            if VMAT and (args.transmissive or args.backfaces):
                mid_k = sub_mid[m_mk][i_k]
                e_k = MAT_EXT[mid_k]
                kf = keep.float()
                if args.transmissive:
                    vc_k, _ = dr.interpolate(vc_attr[None].contiguous(),
                                             rast_k, local[m_mk])
                    trans_o = torch.where(
                        k3, transmissive_term(uv_k, mid_k, e_k, alb_k,
                                              tapsA_k, vc_k),
                        trans_o)
                if args.backfaces:
                    two_sided = torch.where(
                        keep, _ext(e_k, "f_render_backfaces"), two_sided)
                del kf

        def _shade_blend(rast_b, fg_b, uv_b, mat_b, nrm_b, uvda_b):
            """One BLEND-class layer -> its colour, alpha and depth.

            Factored out of the single-layer path unchanged so the MBOIT
            passes can run it per depth-peeled layer. Nothing here decides
            compositing; the caller does, because that is exactly what the
            MBOIT axis changes.
            """
            i_b = (rast_b[..., 3].long() - 1).clamp(min=0)
            # The BLEND class fetched the raw texture, skipping
            # baseColorFactor, the per-layer colour correction and the
            # second blend layer that the opaque class runs. That was
            # invisible while the class held 15 materials of glass; once
            # the 24 csgo_static_overlay decals are classed correctly it
            # owns 5.8% of the frame, so it gets the real albedo path.
            uv2_b, _ = dr.interpolate(uv2_attr[None].contiguous(), rast_b,
                                      local[m_bl])
            vcb, _ = dr.interpolate(vc_attr[None].contiguous(), rast_b,
                                    local[m_bl])
            # The BLEND class runs the same two-layer shade_albedo as the
            # opaque class (fam's rewrite), so it needs the same weight and
            # the same per-instance model-tint gate. Both were threaded into
            # the opaque path only, because model-tint/vpaint were written
            # against an ancestor whose BLEND path did a raw texture fetch.
            vp_b = None
            if VPAINT is not None:
                vp_b, _ = dr.interpolate(
                    VPAINT[used][:, None][None].contiguous(), rast_b,
                    local[m_bl])
                vp_b = vp_b[..., 0]
            w_src_b = (vp_b if args.blend_weight == "vpaint"
                       else vcb[..., args.blend_channel])
            if args.blend_weight == "vpaint" and args.vpaint_remap:
                w_src_b = (w_src_b * 1.10000002384185791 - 0.05).clamp(0, 1)
            w_b = (w_src_b * sub_has2[m_bl][i_b]).clamp(0, 1)
            tapsC_b = mip_taps(uvda_b, T, ANISO_C, TEX_MIP[3]) \
                if TEX_MIP is not None and uvda_b is not None else None
            alb_b = shade_albedo(uv_b, mat_b, sub_mat2[m_bl][i_b],
                                 sub_has2[m_bl][i_b],
                                 sub_params[m_bl][i_b], w_b, tapsC_b,
                                 MTINT[sub_mid[m_bl][i_b]]
                                 if (MTINT is not None
                                     and args.model_tint != "always" and PBR)
                                 else None)
            mid_b = sub_mid[m_bl][i_b] if PBR else None
            zb = rast_b[..., 2]
            a_raw = alb_b[..., 3]
            rgb_b = alb_b[..., :3]
            modulate = additive = None
            if FAMILY:
                x_b = MAT_EXT2[mid_b]
                if args.depth_bias > 0:
                    # F_DEPTH_BIAS (6 csgo_complex decals): CS2 pulls a
                    # coplanar decal toward the camera so it cannot lose
                    # the depth test to the very wall it is painted on.
                    # Every csgo_static_overlay is a projected overlay by
                    # construction, so it gets the same treatment once it
                    # is composited instead of drawn opaque.
                    bias_on = _e2(x_b, "f_depth_bias")
                    if args.overlay_blend:
                        bias_on = torch.maximum(
                            bias_on,
                            (MAT_FAM[mid_b] == FAM_OVERLAY).float())
                    zb = zb - args.depth_bias * bias_on
                if args.glass:
                    # THREADED, not defaulted: the geometric normal is a
                    # parameter of this closure, the vertex colour is
                    # interpolated at the top of it, and the world position
                    # comes from the same VS_VPOS interpolation the
                    # sf_effects branch below already does. `eye` is
                    # render_window's own argument. Four of the shader's
                    # inputs are therefore real frame state; what is left
                    # defaulted prints its own line at first draw.
                    wp_g, _ = dr.interpolate(
                        VS_VPOS[used][None].contiguous(), rast_b, local[m_bl])
                    g_rgb, g_a = glass_shade(
                        uv_b, mid_b, x_b,
                        nrm=_nrm(nrm_b), wpos=wp_g,
                        eye_pos=eye[c0:c1].view(-1, 1, 1, 3), vcolor=vcb)
                    on = _e2(x_b, "has_gdust")
                    rgb_b = rgb_b * (1 - on.unsqueeze(-1)) \
                        + g_rgb * on.unsqueeze(-1)
                    a_raw = a_raw * (1 - on) + g_a * on
                # --- FAM_EFFECTS: csgo_effects c10/r10m1 ---------------
                # The steam plume is an additive BLEND-class draw, so it
                # is shaded here and not in the opaque pass. `on` is the
                # family mask; nothing outside csgo_effects is touched.
                if args.sf_effects and FAM_EFFECTS >= 0:
                    wp_e, _ = dr.interpolate(
                        VS_VPOS[used][None].contiguous(), rast_b, local[m_bl])
                    # the 5th fetch site: the SCENE DEPTH target. The
                    # reference texelFetches the raw depth attachment, so
                    # this hands over gl_FragCoord.z and not metres -- see
                    # sf_effects_shade's note on why `lin` is 1/z.
                    _zl = _frag_depth(depth_o, fg_o)
                    _fwd = _nrm(fwd[c0:c1].view(-1, 1, 1, 3))
                    _ffe = ((_nrm(nrm_b)
                             * _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wp_e)
                             ).sum(-1, keepdim=True) >= 0.0)
                    e_rgb, e_a = sf_effects_shade(
                        uv_b, mid_b, x_b, wp_e,
                        eye[c0:c1].view(-1, 1, 1, 3), _nrm(nrm_b), vcb,
                        alb_b, _zl, _fwd,
                        (tsec[c0:c1].view(-1, 1, 1, 1)
                         if tsec is not None else None),
                        _ffe, SF_COMBO["effects"])
                    on = (MAT_FAM[mid_b] == FAM_EFFECTS).float()
                    rgb_b = rgb_b * (1 - on.unsqueeze(-1)) \
                        + e_rgb * on.unsqueeze(-1)
                    a_raw = a_raw * (1 - on) + e_a * on
                    if args.sf_tools_vis != 0:
                        # S_MODE_TOOLS_VIS = 1: csgo_effects c11/r11m2
                        # carries the same mode chain csgo_black_unlit
                        # c1/r1m0 does, against the same CB slot.
                        t_rgb = sf_tools_vis(
                            args.sf_tools_vis, rgb_b, _nrm(nrm_b), wp_e,
                            eye[c0:c1].view(-1, 1, 1, 3),
                            float(tsec[c0]) if tsec is not None else 0.0)
                        rgb_b = rgb_b * (1 - on.unsqueeze(-1)) \
                            + t_rgb * on.unsqueeze(-1)
                    SF_REACH["effects"][0] += float(((on > 0.5) & fg_b).sum())
                    SF_REACH["effects"][1] += float(fg_b.sum())
                # === NON-WORLD FAMILIES, blend class ==================
                # This is the MBOIT-on-player-geometry answer. `_shade_blend`
                # has exactly two callers -- the MBOIT pass-1/pass-2 loop
                # and the single-layer composite -- so a character hook
                # HERE is shaded by the moment path with no second entry
                # point, and D_MBOIT_PASS1/PASS2/4_MOMENTS compose with the
                # existing --mboit exactly as they do for glass. What was
                # missing was not the moment code, it was that no character
                # face ever reached the BLEND class; --char-translucent
                # reclassifies them, above.
                if NONWORLD and args.char and FAM_CHARACTER >= 0:
                    _cbg = (MAT_FAM[mid_b] == FAM_CHARACTER).float()
                    _crgb2 = char_decal(rgb_b, uv_b, uv_b, mid_b, x_b) \
                        if args.char_decals else rgb_b
                    if args.char_adjustments:
                        _crgb2, _ = char_adjustments(
                            _crgb2,
                            torch.stack([torch.full_like(_cbg, 0.5)] * 2,
                                        dim=-1),
                            torch.ones_like(_cbg), x_b)
                    if args.char_invulnerability:
                        _crgb2 = char_invulnerability(
                            _crgb2, _nrm(nrm_b), _nrm(nrm_b),
                            float(tsec or 0.0),
                            torch.rand_like(_cbg), x_b)
                    rgb_b = rgb_b + (_crgb2 - rgb_b) * _cbg.unsqueeze(-1)
                    # S_TRANSLUCENT / g_flOpacityScale on the alpha
                    a_raw = a_raw * (1.0 - _cbg) + (
                        a_raw * _e2(x_b, "ch_opacity_scale")) * _cbg
                    FAM_HIT["character"] += int((_cbg > 0).sum())
                if NONWORLD and args.glove and FAM_CUSTOMGLOVE >= 0:
                    _gbg = (MAT_FAM[mid_b] == FAM_CUSTOMGLOVE).float()
                    _gt4 = torch.cat(
                        [_nrm(torch.cross(
                            nrm_b,
                            torch.tensor([0.0, 1.0, 0.0], device=device)
                            .expand_as(nrm_b), dim=-1)),
                         torch.ones_like(_gbg).unsqueeze(-1)], dim=-1)
                    _grg2, _ga2, _gn2, _gr22, _gm2, _gao2 = \
                        customglove_surface(
                            rgb_b, a_raw, uv_b, mid_b, x_b,
                            torch.tensor([0.0, 0.0, 1.0], device=device)
                            .expand_as(nrm_b), _gt4, _nrm(nrm_b))
                    rgb_b = rgb_b + (_grg2 - rgb_b) * _gbg.unsqueeze(-1)
                    # THE SECOND CALL SITE, and the one where the drop cost
                    # something. Here `a_raw` is real blend alpha -- what
                    # the composite a few lines down multiplies by -- and
                    # the character branch immediately above consumes ITS
                    # alpha through ch_opacity_scale in exactly this shape.
                    # The glove's was computed and discarded, so a glove in
                    # the blend class composited at whatever alpha the
                    # material happened to carry rather than at the one its
                    # own shader wrote.
                    _TERMSTATS.observe("customglove.alpha", _ga2, before=a_raw,
                                mask=(MAT_FAM[mid_b] == FAM_CUSTOMGLOVE),
                                reference="csgo_customglove_ps_r3_m0"
                                          ".glsl:1936-2071")
                    a_raw = a_raw + (_ga2 - a_raw) * _gbg
                    FAM_HIT["customglove"] += int((_gbg > 0).sum())
                if args.water:
                    # F_REFRACTION + F_REFLECTION_TYPE 2: CS2 sees the
                    # street THROUGH the fountain plane and puts a Fresnel
                    # sky reflection on top of it, so at the angles this
                    # pose set uses the plane is nearly invisible -- which
                    # is exactly what the ground truth shows. Opacity is
                    # g_flReflectance ramped by (1 - N.V)^g_flFresnelExponent;
                    # the reflected colour is the sky the fog was fitted to.
                    wp_b, _ = dr.interpolate(
                        VS_VPOS[used][None].contiguous(), rast_b,
                        local[m_bl])
                    vv = _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wp_b)
                    ndv = (nrm_b * vv).sum(-1).abs().clamp(0, 1)
                    fres = (1.0 - ndv) ** 4.0
                    wa = (0.2 + 0.8 * fres).clamp(0, 1)
                    on = (MAT_FAM[mid_b] == FAM_WATER).float()
                    rgb_b = rgb_b * (1 - on.unsqueeze(-1)) \
                        + FOG_COLOR.view(1, 1, 1, 3).clamp(0, 1) ** (1 / 2.2) \
                        * on.unsqueeze(-1)
                    a_raw = a_raw * (1 - on) + wa * on
                if args.water_fancy:
                    # ===== csgo_water_fancy, the transcribed path =====
                    # This REPLACES the --water approximation above on
                    # the same pixels; running both is allowed only so an
                    # A/B can be taken, and the transcription wins because
                    # it is applied second.
                    wp_w, _ = dr.interpolate(
                        VS_VPOS[used][None].contiguous(), rast_b,
                        local[m_bl])
                    # g_tRefractionMap / g_tSceneDepth. WHAT DIFFERS: at
                    # this point in the frame `alb` holds the opaque
                    # class's ALBEDO, not its lit colour -- this renderer
                    # lights after compositing -- so the refraction and
                    # the SSR march read an unlit scene buffer. The
                    # coordinate, the filter, the offset and every
                    # consumer are the reference's; the buffer's CONTENT
                    # is one lighting stage early.
                    w_rgb, w_a = water_fancy_shade(
                        wp_w, nrm_b, vcb, mid_b, x_b,
                        alb[..., :3], _frag_depth(depth, fg),
                        eye[c0:c1], fwd[c0:c1],
                        tsec[c0:c1] if tsec is not None else None)
                    on = (MAT_FAM[mid_b] == FAM_WATER_FANCY).float()
                    rgb_b = rgb_b * (1 - on.unsqueeze(-1)) \
                        + w_rgb * on.unsqueeze(-1)
                    a_raw = a_raw * (1 - on) + w_a * on
                    WATER_REACH[0] += float(on.sum())
                    WATER_REACH[1] += float(on.numel())
                if args.overlay_blend:
                    a_raw = (a_raw * _e2(x_b, "opacity_scale")).clamp(0, 1)
                    modulate = (_e2(x_b, "f_blend_mode") > 2.5).float()
                    additive = _e2(x_b, "f_additive")
                if GTSURF and args.blend_effects:
                    # S_BLEND_EFFECTS -- csgo_static_overlay ONLY, and the
                    # shader REQUIRES S_MATERIAL_REFERENCE != 0
                    # (RENDER_SETTINGS_AXES.md line 372), so the gate is
                    # the product of the two flags, not F_BLEND_EFFECTS
                    # alone. A decal has no layer seam, so the band the
                    # border rides on is its own alpha ramp -- the same
                    # 4a(1-a) shaping the layer blend uses, so the two
                    # cannot disagree about what an edge is.
                    _bandb = (4.0 * a_raw * (1.0 - a_raw)).clamp(0, 1) \
                        * (_e2(x_b, "f_material_ref") > 0.5).float()
                    _btb = torch.stack([_e2(x_b, "btint_r"),
                                        _e2(x_b, "btint_g"),
                                        _e2(x_b, "btint_b")], dim=-1)
                    rgb_b, _, _ = gt_blend_effects(rgb_b, None, None, None,
                                                   _bandb, x_b, _btb, "")
                if GTSURF and args.material_reference:
                    # so:ps REQUIRES S_MATERIAL_REFERENCE -> S_LIT != 0
                    # and S_BLEND_MODE in {0,1,2}; both are per-material,
                    # so they gate the term instead of being assumed.
                    _okmr = ((_e2(x_b, "f_lit") > 0.5)
                             & (_e2(x_b, "f_blend_mode") < 2.5)).float()
                    _mr = gt_material_reference(rgb_b, x_b)
                    rgb_b = rgb_b + (_mr - rgb_b) * _okmr.unsqueeze(-1)
                if GTSURF and args.tint_mask:
                    # so:ps REQUIRES S_TINT_MASK -> S_MATERIAL_REFERENCE
                    # == 0: the two are mutually exclusive in this family.
                    _tmb = gt_tint_mask_value(
                        sample_aux(uv_b, MAT_TM[mid_b]),
                        sample_aux(uv_b, MAT_H1[mid_b]) if VMAT
                        else torch.zeros_like(alb_b),
                        x_b, MAT_EXT[mid_b] if VMAT else None)
                    _tmb = _tmb * _e2(x_b, "f_tint_mask2") \
                        * (_e2(x_b, "f_material_ref") < 0.5).float()
                    _m = _tmb.unsqueeze(-1)
                    rgb_b = rgb_b * (1 - _m) + rgb_b * MAT_CTINT2[mid_b] * _m
                # --- S_TRANSLUCENT ---------------------------------------
                # csgo_complex s80/d2:888 against s0/d2:936, which is the
                # whole axis in the pixel stage:
                #   S_TRANSLUCENT=0   out.a = vColor.a
                #   S_TRANSLUCENT=1   out.a = vColor.a * (colorTex.a
                #                                         * g_flOpacityScale)
                # shade_albedo returns layer 1's texture alpha, so the two
                # missing factors are the vertex-colour alpha and the
                # opacity scale.
                if args.translucent:
                    ton = _e2(x_b, "f_translucent")
                    a_tl = (vcb[..., 3] * (a_raw * _e2(x_b, "opacity_scale"))
                            ).clamp(0, 1)
                    TL_STAT[0] += float(((ton > 0.5) & fg_b).float().sum())
                    TL_STAT[1] += float(
                        ((ton > 0.5) & fg_b
                         & ((a_tl - a_raw).abs() > 1e-6)).float().sum())
                    TL_STAT[2] += float(fg_b.float().sum())
                    a_raw = a_raw * (1 - ton) + a_tl * ton
                # --- S_ADDITIVE_BLEND ------------------------------------
                # The blend equation dst+src*a is applied below; the fog
                # REDIRECTION is the pixel-stage half of the axis and is
                # applied where the fog is (see additive_fog).
                if args.additive_blend:
                    _ad = _e2(x_b, "f_additive")
                    AD_STAT[0] += float(((_ad > 0.5) & fg_b).float().sum())
                    AD_STAT[1] += float(fg_b.float().sum())
                    additive = _ad if additive is None else \
                        torch.maximum(additive, _ad)
                # --- S_OPAQUE_CUBEMAP_REFRACTION -------------------------
                if args.cubemap_refraction:
                    if CUBE is None:
                        raise SystemExit(
                            "--cubemap-refraction resolves the refracted ray "
                            "against the IBL cube ARRAY (glass s2/d8:1183), "
                            "so it needs --ibl-cube and --cube-probes")
                    wp_r, _ = dr.interpolate(
                        vertices[used][None].contiguous(), rast_b,
                        local[m_bl])
                    v_r = _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wp_r)
                    # roughness: dot(g_tRoughness.zz * vColor.a, vec2(0.5))
                    # (glass s2/d8:314 + 1114) reduces to that scalar; this
                    # renderer carries the same channel as `rough`, and the
                    # BLEND class has no per-pixel roughness, so the
                    # material constant is used.
                    r_r = (MAT_PBR[mid_b, 1] * vcb[..., 3]).clamp(0, 1)
                    ron = torch.maximum(
                        _e2(x_b, "f_opaque_cube_refract"),
                        (MAT_FAM[mid_b] == FAM_GLASS).float())
                    _amb = (AMBIENT_GROUND.view(1, 1, 1, 3)
                            + (AMBIENT_SKY - AMBIENT_GROUND).view(1, 1, 1, 3)
                            * (nrm_b[..., 1].clamp(-1, 1) * 0.5 + 0.5
                               ).unsqueeze(-1))
                    rgb_b, _ = cubemap_refraction(
                        rgb_b, wp_r, _nrm(nrm_b), v_r, r_r, _amb, ron, x_b)
                    REFRACT_STAT[0] += float(((ron > 0.5) & fg_b).float().sum())
                    REFRACT_STAT[1] += float(fg_b.float().sum())
                    # glass s2/d8:1191 -- `_3711 = vec4(..., 1.0)`. The
                    # refracting pane is OPAQUE; that is the axis's name.
                    a_raw = a_raw * (1 - ron) + ron
                # --- S_MODE_DEPTH translucent clip -----------------------
                # A pane that the depth pass discarded writes no depth, so
                # it cannot occlude; here that means it does not composite
                # over what is behind it either, because in this renderer
                # the two are the same decision.
                if args.disable_translucent_clip and dm is not None:
                    # glass s1/d1 is 288 bytes: `out = vec4(0)`, no
                    # discard. Every glass fragment writes depth, so the
                    # keep channel is 1 everywhere the pane covers.
                    dm[..., 1] = torch.where(
                        (MAT_FAM[mid_b] == FAM_GLASS) & fg_b,
                        torch.ones_like(dm[..., 1]), dm[..., 1])
                if args.translucent_clip and not args.disable_translucent_clip:
                    wp_c, _ = dr.interpolate(
                        vertices[used][None].contiguous(), rast_b,
                        local[m_bl])
                    v_c = _nrm(eye[c0:c1].view(-1, 1, 1, 3) - wp_c)
                    ndv_c = (_nrm(nrm_b) * v_c).sum(-1).clamp(0, 1)
                    _gon = (MAT_FAM[mid_b] == FAM_GLASS)
                    _kc = translucent_clip(rgb_b, ndv_c, a_raw) | (~_gon)
                    CLIP_STAT[0] += float((_gon & ~_kc & fg_b).float().sum())
                    CLIP_STAT[1] += float((_gon & fg_b).float().sum())
                    if dm is not None:
                        dm[..., 1] = torch.where(_gon & fg_b, _kc.float(),
                                                 dm[..., 1])
                    a_raw = a_raw * _kc.float()
                # --- D_OPAQUE_FADE on the blend class ---------------------
                if args.opaque_fade:
                    _fon = torch.ones_like(a_raw)
                    _fv, _fk = opaque_fade(vcb[..., 3].clamp(0, 1), _fon)
                    a_raw = _fv * _fk.float()
                if args.paint_vertex_colors:
                    rgb_b = paint_vertex_colors(rgb_b, vcb, x_b)
                if args.complex_tint:
                    rgb_b = unlit_tint(rgb_b, x_b, MAT_CTINT2[mid_b],
                                       MAT_EXT[mid_b] if VMAT else None)
            if FAMILY and args.fam_only is not None:
                rgb_b = _fam_flag(mid_b).expand_as(rgb_b)
                modulate = additive = None
            return (rgb_b, a_raw, zb, mid_b, nrm_b, modulate, additive,
                    vcb, i_b)

        _mboit_rt = None
        if int(m_bl.sum()) and MBOIT_N:
            # ===== D_MBOIT_PASS1 / D_MBOIT_PASS2 =========================
            # Two invocations over the SAME depth peel, in order, with the
            # moment buffers closed between them -- the structure the
            # engine gets from two render passes over two target sets.
            _cap = args.mboit_layers if args.mboit_layers > 0 else 64
            _peel = dr.DepthPeeler(ctx, clip, local[m_bl], (H, W))
            _layers = []
            with _peel as _pl:
                for _li in range(_cap):
                    _r, _rdb = _pl.rasterize_next_layer()
                    if not bool((_r[..., 3] > 0).any()):
                        break
                    _layers.append((_r, _rdb))
            MBOIT_STAT[0] = max(MBOIT_STAT[0], len(_layers))
            MBOIT_STAT[1] += 1
            if len(_layers) >= _cap:
                MBOIT_STAT[2] += 1

            def _unpack(_r, _rdb):
                _ib = (_r[..., 3].long() - 1).clamp(min=0)
                _fgb = _r[..., 3] > 0
                if MIP_ON:
                    _uvb, _uvda = dr.interpolate(
                        uv_attr[None].contiguous(), _r, local[m_bl],
                        rast_db=_rdb, diff_attrs="all")
                else:
                    _uvb, _ = dr.interpolate(uv_attr[None].contiguous(), _r,
                                             local[m_bl])
                    _uvda = None
                return (_fgb, _uvb, sub_mat[m_bl][_ib], sub_norm[m_bl][_ib],
                        _uvda)

            def _zlin(_r):
                _wp, _ = dr.interpolate(vertices[used][None].contiguous(),
                                        _r, local[m_bl])
                return ((_wp - eye[c0:c1].view(-1, 1, 1, 3))
                        * fwd[c0:c1].view(-1, 1, 1, 3)).sum(-1).clamp(min=1e-4)

            # ---- PASS 1: moments only. Shades nothing. ------------------
            _m0 = torch.zeros(c1 - c0, H, W, device=device)
            _m12 = torch.zeros(c1 - c0, H, W, 2, device=device)
            _m3456 = torch.zeros(c1 - c0, H, W, 4, device=device)
            _m4 = torch.zeros(c1 - c0, H, W, 4, device=device)
            for _r, _rdb in _layers:
                _fgb, _uvb, _matb, _nrmb, _uvda = _unpack(_r, _rdb)
                _o = _shade_blend(_r, _fgb, _uvb, _matb, _nrmb, _uvda)
                _ab = _o[1]
                _keepb = _fgb & (_o[2] < depth)
                _acc = mboit_pass1(_ab.clamp(0, 1), _zlin(_r), _keepb, MBOIT_N)
                _m0 = _m0 + _acc[0]
                if MBOIT_N == 4:
                    _m4 = _m4 + _acc[1]
                else:
                    _m12 = _m12 + _acc[1]
                    _m3456 = _m3456 + _acc[2]
            _mom = (_m0, _m4) if MBOIT_N == 4 else (_m0, _m12, _m3456)

            # ---- PASS 2: shade, resolve against pass 1, accumulate ------
            # complex s80/d66:1073 -- the pass-2 output is
            #   vec4(rgb*a, clamp(a,1e-5,0.9999)) * transmittance
            # into an ADDITIVE target, so the sum below is the engine's
            # blend state, and the final line is the resolve composite
            #   dst = accum.rgb + background * (1 - accum.a).
            _accum = torch.zeros(c1 - c0, H, W, 3, device=device)
            _acov = torch.zeros(c1 - c0, H, W, device=device)
            _near_n = torch.zeros(c1 - c0, H, W, 3, device=device)
            _near_z = torch.full((c1 - c0, H, W), 1e9, device=device)
            _near_id = torch.zeros(c1 - c0, H, W, dtype=torch.long,
                                   device=device)
            # THE DOMINANT CONTRIBUTOR, tracked at the write site.
            # `_near_id` is the NEAREST layer, which is the right answer for
            # a normal and a depth and the wrong one for attribution: under
            # MBOIT the pixel's colour is a weighted sum over every layer,
            # and the class that owns it is the one with the largest
            # effective weight `a * T`, not the one closest to the eye.
            # (Project law, five instances: prefer the DOMINANT draw call,
            # never draw call 0.) They agree on most pixels and the census
            # prints how many they do not.
            _dom_w = torch.zeros(c1 - c0, H, W, device=device)
            _dom_id = torch.zeros(c1 - c0, H, W, dtype=torch.long,
                                  device=device)
            _n_contrib = torch.zeros(c1 - c0, H, W, device=device)
            for _r, _rdb in _layers:
                _fgb, _uvb, _matb, _nrmb, _uvda = _unpack(_r, _rdb)
                (_rgbb, _ab, _zb, _midb, _nb, _mod, _add,
                 _vcb, _ib) = _shade_blend(_r, _fgb, _uvb, _matb, _nrmb,
                                           _uvda)
                _keepb = _fgb & (_zb < depth)
                _T = mboit_transmittance(_mom, _zlin(_r), MBOIT_N)
                _a = (_ab.clamp(0, 1) * _keepb.float())
                _accum = _accum + _rgbb * (_a * _T).unsqueeze(-1)
                _acov = _acov + _a.clamp(
                    9.9999997473787516355514526367188e-06,
                    0.99989998340606689453125) * _T * _keepb.float()
                _sel = _keepb & (_zb < _near_z)
                _near_z = torch.where(_sel, _zb, _near_z)
                _near_n = torch.where(_sel.unsqueeze(-1), _nb, _near_n)
                if PBR:
                    _near_id = torch.where(_sel, sub_mid[m_bl][_ib], _near_id)
                    # ---- THE WRITE SITE JOINS THE BALANCE LAW (#80 a) ----
                    # Until now this whole loop was invisible to the census:
                    # both operands were taken at the RESOLVE, off `_near_id`
                    # and `_acov > 0.5`. So a class that put real colour into
                    # `_accum` from a non-nearest layer was credited nothing
                    # and did not even appear as "won pixels", which is the
                    # exact shape -- a class drawing without claiming -- the
                    # law exists to catch. `class_px_won` is the operand
                    # recorded at the line that decides the IMAGE, and for
                    # this route that line is here, not at the resolve.
                    _lid = sub_mid[m_bl][_ib]
                    _w = _a * _T
                    _hit = _keepb & (_w > 1e-4)
                    if FAMILY:
                        class_px_won(MAT_FAM[_lid], _hit)
                    _n_contrib = _n_contrib + _hit.float()
                    _dsel = _hit & (_w > _dom_w)
                    _dom_w = torch.where(_dsel, _w, _dom_w)
                    _dom_id = torch.where(_dsel, _lid, _dom_id)
            _acov = _acov.clamp(0, 1)
            if args.mboit_resolve != "off":
                # ===== THE RESOLVE, mboit_mixed_combine + mboitfinal ====
                # These moment buffers were written and, until this
                # branch existed, NEVER RESOLVED: the `else` below is a
                # direct full-resolution over, which is what the pass-2
                # implementation had to do with no resolve to hand it to.
                # `mboit_mixed_combine` consumes THESE buffers -- `_accum`,
                # `_acov`, `_m0` and the moment tuple `_mom` -- at mixed
                # resolution and `mboitfinal` recombines them. There is no
                # second MBOIT and no second moment accumulation here.
                #
                # The composite is DEFERRED to the smoke block a few lines
                # below rather than done twice: D_UPSCALE_SMOKE_DEPTH and
                # D_DENOISE_SMOKE make the resolve and the smoke ONE
                # system, so the pair the combine mixes is (transparency,
                # smoke) and it cannot run before the smoke exists.
                _mboit_rt = dict(
                    rgb=_accum, a=_acov, b0=_m0,
                    z=torch.where(_near_z < 1e8,
                                  _lin_depth(_frag_depth(
                                      _near_z, _near_z < 1e8)),
                                  torch.full_like(_near_z, 1e9)),
                    mom=_m0, mom_tuple=_mom)
            else:
                _bg = torch.where(fg.unsqueeze(-1), alb[..., :3], _accum)
                over = _accum + _bg * (1.0 - _acov).unsqueeze(-1)
                alb = torch.cat([over, alb[..., 3:]], dim=-1)
            _cov = _acov > 0.5
            nrm = torch.where((_cov & ~fg).unsqueeze(-1), _near_n, nrm)
            fg = fg | (_acov > 0.05)
            # class_px_won is NOT called here any more -- it is called per
            # layer in the pass-2 loop above, at the line that writes the
            # colour. Calling it here as well would double-count the
            # dominant layer and leave every other layer still invisible.
            if fam_pix is not None and PBR:
                # ATTRIBUTED OVER THE MASK THAT ENTERS `fg`, not over
                # `_acov > 0.5`. `fg = fg | (_acov > 0.05)`, three lines
                # ABOVE, is what puts these pixels into `shaded`
                # operand, so attributing only above 0.5 left every pixel
                # between the two thresholds counted as shaded and credited
                # to nobody -- an UNATTRIBUTED residue that the census
                # reported as an unclaimed-pixel defect it could not
                # localise. Same mask, same pixels, no residue.
                # AND `_n_contrib > 0`, because `_dom_id` starts at
                # material 0 and material 0's family is a REAL family.
                # Attributing a pixel no layer actually contributed to
                # would write a real class id onto an unclaimed pixel --
                # exactly what FAM_UNATTRIBUTED = -1 exists to prevent, and
                # reachable here: `_acov` sums `a.clamp(1e-5, ...) * T`, so
                # many clamped near-zero layers can push it past 0.05 with
                # no single layer above the 1e-4 contribution floor.
                _fgm = _acov > 0.05
                _attr_m = _fgm & (_n_contrib > 0)
                MBOIT_ATTR[4] += int((_fgm & ~_attr_m).sum())
                _dom_fam = MAT_FAM[_dom_id]
                MBOIT_ATTR[0] += int(_attr_m.sum())
                MBOIT_ATTR[1] += int((_attr_m & (_n_contrib > 1.5)).sum())
                MBOIT_ATTR[2] += int((_attr_m
                                      & (_dom_fam != MAT_FAM[_near_id])).sum())
                MBOIT_ATTR[3] += int((_attr_m & ~_cov).sum())
                fam_pix = torch.where(_attr_m, _dom_fam, fam_pix)
            own_o = own_o & ~_cov
            if trans_o is not None:
                trans_o = trans_o * (~_cov).float().unsqueeze(-1)
            if two_sided is not None:
                two_sided = two_sided * (~_cov).float()
            # D_TRANSLUCENT_SCENE_DEPTH: the SECOND depth target
            # csgo_projected_decals min()s in (set 1, binding 47). It is
            # the nearest TRANSLUCENT fragment's depth; under MBOIT that
            # is `_near_z`, already accumulated above.
            NW_TRANS_Z[0] = _frag_depth(_near_z, _acov > 0.0)
        elif int(m_bl.sum()):
            rast_b, fg_b, uv_b, mat_b, nrm_b, uvda_b = _raster(
                clip, sub_faces[m_bl], sub_mat[m_bl], sub_norm[m_bl],
                uv_attr, local[m_bl])
            (rgb_b, a_raw, zb, mid_b, nrm_b, modulate, additive,
             vcb, i_b) = _shade_blend(rast_b, fg_b, uv_b, mat_b, nrm_b,
                                      uvda_b)
            if args.additive_blend and additive is not None and args.fog:
                # csgo_complex s240/d2:909 -- with S_ADDITIVE_BLEND the fog
                # multiplies the ALPHA, `a * (1 - fog)`, and never touches
                # the colour. That is the axis's structural content in the
                # pixel stage, and it is right: fading an additive surface
                # means contributing less, not turning grey.
                #
                # WHAT DIFFERS: this renderer composites the BLEND class in
                # ALBEDO space, before lighting and before _post_chain's
                # colour fog, so the surviving additive contribution is
                # also carried through that colour fog afterwards. The
                # engine lights and fogs the additive surface on its own
                # and adds the result. Structural to the class ordering,
                # not to this axis.
                wp_f, _ = dr.interpolate(vertices[used][None].contiguous(),
                                         rast_b, local[m_bl])
                _fin = torch.stack(
                    [(eye[c0:c1].view(-1, 1, 1, 3) - wp_f).norm(dim=-1) / S,
                     wp_f[..., 1] / S], dim=-1)
                a_raw = a_raw * (1.0 - fog_opacity(_fin) * additive)
            a = (a_raw * (fg_b & (zb < depth)).float()).unsqueeze(-1)
            # RETIRED (#80 option (c)): the blend class's own class_px_
            # accumulate (812a0319) stood here. It was correct for its
            # class and it is being removed anyway, because PER-CLASS CALL
            # SITES ARE THE DEFECT: the mask class was never given one, and
            # nothing in a per-class design can notice that a class is
            # missing. A fourth class would repeat it. The attribution map
            # fam_pix -- which this composite already writes, just below --
            # is now counted ONCE at the composite, and the census balances
            # what it counted against what the classes said they covered.
            # Where NOTHING was drawn behind the blend surface, `alb`
            # holds whatever the opaque pass left in an uncovered pixel
            # (triangle index clamped to 0), i.e. junk. Compositing a
            # decal against junk is what put black bands along the ground
            # overlays; with nothing behind it, the surface is its own
            # background.
            base = torch.where(fg.unsqueeze(-1), alb[..., :3], rgb_b)
            if GTSURF and args.overlay_blend:
                # S_BLEND_MODE, RANGE 0..6 -- all seven composites, not
                # the three the boolean pair below could express. The
                # opacity scale is already folded into a_raw, so it is
                # passed as 1 here rather than applied twice.
                #
                # BIND x_b FROM THIS ITERATION'S mid_b. It used to read a
                # name bound only inside the `if FAMILY:` block of the
                # OPAQUE class, ~350 lines up and in a different branch of
                # the same for-loop. Two distinct failures, and the second
                # is worse: on an iteration where that branch did not run,
                # this raised NameError; on one where an EARLIER iteration
                # ran it, Python's leaked loop binding made this silently
                # composite with the PREVIOUS material's f_blend_mode.
                # A crash is the good case.
                x_b = MAT_EXT2[mid_b] if FAMILY else None
                # OPACITY WAS HARDCODED TO ONES HERE. mode 1 is pinned
                # by the materials carrying g_flOpacityScale, and that
                # value IS packed -- ext2 column `opacity_scale`, fully
                # populated, no zeros, range 0.30..1.00. Passing ones
                # meant the evidence that PINS the mode could not reach
                # the composite, so mode 1 was plain src-over at full
                # opacity. Checked before wiring, because a column of
                # zeros would have multiplied every overlay away: on the
                # 20 csgo_static_overlay materials the values are
                # {0.30, 0.90, 1.00}, and on the 6 mode-2 materials they
                # are all exactly 1.00 -- so this reaches precisely the
                # mode-1 materials the pinning names and is a no-op
                # everywhere else, which is the pinning confirming itself.
                over = gt_composite(base, rgb_b, a,
                                    _e2(x_b, "f_blend_mode"),
                                    _e2(x_b, "opacity_scale"))
                if additive is not None:
                    # F_ADDITIVE_BLEND is a separate material flag from
                    # F_BLEND_MODE and still applies on top of it.
                    ad = additive.unsqueeze(-1)
                    over = over * (1 - ad) + (base + rgb_b * a) * ad
            else:
                over = base * (1 - a) + rgb_b * a
                if modulate is not None:
                    # F_BLEND_MODE 3 (`top_grime_2`): the overlay
                    # MULTIPLIES what is under it rather than replacing it.
                    m3 = modulate.unsqueeze(-1)
                    over = over * (1 - m3) + base * (1 - a + a * rgb_b) * m3
                if additive is not None:
                    ad = additive.unsqueeze(-1)
                    AD_STAT[2] += float(((additive > 0.5)
                                         & (a.squeeze(-1) > 0.01))
                                        .float().sum())
                    over = over * (1 - ad) + (base + rgb_b * a) * ad
            alb = torch.cat([over, alb[..., 3:]], dim=-1)
            # nrm_b is the FLAT face normal. Replacing a normal-mapped
            # opaque normal with it throws the wall's bump away wherever
            # a decal is painted on it; only take it where nothing else
            # covered the pixel.
            take_n = (a.squeeze(-1) > 0.5) & (~fg if args.overlay_blend
                                              else torch.ones_like(fg))
            nrm = torch.where(take_n.unsqueeze(-1), nrm_b, nrm)
            fg = fg | (a.squeeze(-1) > 0.05)
            if FAMILY and mid_b is not None:
                class_px_won(MAT_FAM[mid_b], a.squeeze(-1) > 0.5)
            if fam_pix is not None:
                fam_pix = torch.where(a.squeeze(-1) > 0.5, MAT_FAM[mid_b],
                                      fam_pix)
            own_o = own_o & (a.squeeze(-1) <= 0.5)
            # A BLEND decal on top owns the pixel; neither the opaque nor
            # the mask surface underneath should still transmit through it.
            _covered = (a.squeeze(-1) > 0.5)
            # D_TRANSLUCENT_SCENE_DEPTH: the SECOND depth target
            # csgo_projected_decals min()s in (set 1, binding 47) -- the
            # nearest TRANSLUCENT fragment's depth, which is this class's
            # zb wherever it survived the depth test against the opaque
            # pass. Uncovered pixels take the far plane, exactly as an
            # untouched depth attachment does after a clear.
            NW_TRANS_Z[0] = _frag_depth(zb, fg_b & (a.squeeze(-1) > 0.0))
            if trans_o is not None:
                trans_o = trans_o * (~_covered).float().unsqueeze(-1)
            if two_sided is not None:
                two_sided = two_sided * (~_covered).float()
            if not args.s_lit:
                # S_LIT=0 (csgo_static_overlay combo weight 14): the
                # overlay emits its albedo through the blend mode with
                # no lighting at all. Routed through the SAME unlit
                # compositor csgo_black_unlit already uses in
                # _post_chain (rgb = rgb*(1-u) + linear*u), so there is
                # one unlit path in this file, not two. 17/17 de_inferno
                # csgo_static_overlay materials set F_LIT=1, so this is
                # the non-default half of the axis.
                unlit_o = (unlit_o if unlit_o is not None
                           else torch.zeros_like(fg, dtype=torch.float32))
                unlit_o = torch.maximum(unlit_o, _covered.float())

        # =====================================================================
        # THE VOLUMETRIC SMOKE SUBSYSTEM AND THE MBOIT RESOLVE
        # =====================================================================
        # smoke_volume_depth -> smoke_volume_mask -> smoke_volume (the march,
        # writing MBOIT moments) -> mboit_mixed_combine -> mboitfinal ->
        # overlay_smoke -> write_smoke_depth_water_reflection.
        #
        # All seven are DYNAMIC-ONLY -- one static combo each, empty static
        # array, every axis per draw -- so they are PASSES, and they are
        # scheduled here by the render path rather than dispatched from a
        # material family. Nothing below reads MAT_FAM or a .vfx name.
        #
        # WHERE IN THE FRAME, and what differs. The engine marches smoke in
        # the translucent pass and composites it in HDR before the tonemap.
        # This renderer composites its whole BLEND class in ALBEDO space and
        # lights afterwards, so the smoke is composited here, in that same
        # stage, and its pixels are then marked through the file's EXISTING
        # unlit compositor (`unlit_o`, the one csgo_black_unlit already uses)
        # so that _post_chain passes the smoke's own already-lit radiance
        # through instead of lighting it a second time. Fog still applies
        # after, which is right: smoke at distance is fogged.
        if SMOKE_ON and (_mboit_rt is not None or args.mboit_resolve == "off"):
            _sz_lin = _lin_depth(_frag_depth(depth, fg))
            # ONE batched inverse for the whole chunk instead of a per-frame
            # linalg.inv launch inside the python loop -- HOSTLOOP_TOP10
            # row 2, x6.1. Same op, same double-precision route, batched;
            # the loop below INDEXES the precomputed result.
            _inv_all = torch.linalg.inv(
                mvp[c0:c1].double()).to(mvp.dtype)
            for _b in range(c1 - c0):
                _eye_b = eye[c0 + _b]
                _fwd_b = fwd[c0 + _b]
                _mvp_b = mvp[c0 + _b]
                _inv_b = _inv_all[_b]
                # The volumes. `--smoke synth` places them in WORLD space in
                # front of this frame's camera, because the world pack
                # contains no smoke at all and a subsystem with nothing to
                # march is a feature that reads as tested.
                _vols = SMOKE_VOLS.get(int(c0 + _b))
                if _vols is None and args.smoke == "demo":
                    # DEMO-DRIVEN: the volumes go where the demo's
                    # smokegrenade_detonate says, for as long as the
                    # detonate/expire tick pair says. Cached per frame
                    # index like the synth path, but keyed on the TICK, so
                    # two frames at the same tick share and two frames at
                    # different ticks do not.
                    _tk = int(TICK_OF_FRAME.get(int(c0 + _b), 0))
                    _vols = smoke_mboit.demo_volumes(
                        SMOKE_EVENTS, _tk, device,
                        tickrate=args.smoke_tickrate)
                    smoke_mboit.demo_smoke_banner(
                        SMOKE_EVENT_META,
                        0 if _vols is None else int(_vols.radius.shape[0]),
                        _tk)
                    if _vols is None:
                        # No smoke alive at this tick is a real answer, not
                        # a reason to synthesise one.
                        SMOKE_VOLS[int(c0 + _b)] = None
                if _vols is None and args.smoke != "demo":
                    _vols = smoke_mboit.synth_volumes(
                        args.smoke_instances, _eye_b, _fwd_b, device,
                        seed=20260807 + int(c0 + _b))
                    SMOKE_VOLS[int(c0 + _b)] = _vols
                if _vols is None:
                    continue
                args.smoke_wind_phase = float(
                    (tsec[c0 + _b] if tsec is not None else 0.0)) * 0.05
                _tr = None
                _mom_in = None
                if _mboit_rt is not None:
                    _tr = {k: (v[_b] if torch.is_tensor(v) else v)
                           for k, v in _mboit_rt.items()
                           if k != "mom_tuple"}
                    _mom_in = tuple(m[_b] for m in _mboit_rt["mom_tuple"])
                _res = smoke_mboit.smoke_pass(
                    _eye_b, _fwd_b, _mvp_b, _inv_b,
                    alb[_b, ..., :3], _sz_lin[_b], SMOKE_TEX, _vols,
                    sun_dir, SUN_COLOR,
                    -sun_dir, AMBIENT_SKY,
                    MBOIT_N if MBOIT_N else 6,
                    transparency=_tr, ss_shadow=None, mom_in=_mom_in,
                    viewmodel_rgb=SMOKE_VIEWMODEL.get("rgb"),
                    viewmodel_z=SMOKE_VIEWMODEL.get("z"),
                    proj_near=PROJ_NEAR, proj_far=PROJ_FAR)
                _out = (_res["viewmodel_rgb"] if _res["viewmodel_rgb"]
                        is not None else _res["rgb"])
                alb[_b, ..., :3] = _out
                _sa = _res["smoke"]["a"]
                if _sa.shape != alb.shape[1:3]:
                    _sa = torch.nn.functional.interpolate(
                        _sa[None, None], size=tuple(alb.shape[1:3]),
                        mode="bilinear", align_corners=False)[0, 0]
                _cov_s = _sa.clamp(0, 1)
                # smoke owns the pixel it covers: it is emissive-ish volume
                # radiance, not a surface to be lit again
                unlit_o = (unlit_o if unlit_o is not None
                           else torch.zeros_like(fg, dtype=torch.float32))
                unlit_o[_b] = torch.maximum(unlit_o[_b], _cov_s)
                fg[_b] = fg[_b] | (_cov_s > 0.05)
                if _res["moments"] is not None and _mboit_rt is not None:
                    # D_MBOIT_PASS1 on smoke_volume: the march's moments ADD
                    # into the same buffers the material pass 1 wrote. This
                    # is the line that makes smoke and MBOIT one system.
                    SMOKE_MOMENT_STAT[0] += float(
                        _res["moments"].abs().sum())
                    SMOKE_MOMENT_STAT[1] += 1
                if _res["reflection_depth"] is not None:
                    SMOKE_REFL_STAT[0] += float(
                        (_res["reflection_depth"] < 1.0).float().sum())
                    SMOKE_REFL_STAT[1] += float(
                        _res["reflection_depth"].numel())
        elif SMOKE_ON:
            SMOKE_NOTE.append("the MBOIT resolve was requested but the "
                              "blend class produced no moment buffer this "
                              "chunk")

        # ===============================================================
        # THE CLASS CENSUS -- ONE map, counted ONCE, and it must BALANCE
        # ===============================================================
        # Every raster class has now written fam_pix: the opaque class at
        # its construction, the mask class and both blend routes at their
        # composites. This is the only place CLASS_PX is touched.
        #
        # WHY IT IS COUNTED HERE AND NOT PER CLASS (#80). Per-class call
        # sites cannot notice a class that has none: the opaque/LMG site
        # and the blend site both existed and were both individually
        # correct, and the MASK class -- 145,692 of de_dust2's 199,888
        # foliage faces -- simply had no site, so no mask pixel could
        # reach its own class. The matrix printed 0 for a class that
        # `--fam-reach` measured at 0.0042% of shaded pixels IN THE SAME
        # PROCESS. Two maps, one frame, disagreeing.
        #
        # THE CONSERVATION LAW. `fg` is the coverage the opaque and mask
        # classes wrote for their own reasons -- an INDEPENDENT operand,
        # not derived from fam_pix. Every fg pixel must therefore be
        # attributed to exactly one class. A class that contributes
        # coverage and forgets the attribution map shows up HERE, as a
        # number, on the frame it happens. That is the property the old
        # design lacked: it could lose a whole class in silence, and did,
        # twice, for a day.
        if fam_pix is not None:
            _att = fam_pix >= 0
            _n_shaded = int(fg.sum())
            _n_att_fg = int((_att & fg).sum())
            _n_leak = _n_shaded - _n_att_fg
            _n_extra = int((_att & ~fg).sum())     # blend over background
            _n_counted = class_px_accumulate(fam_pix, _att)
            CLASS_PX_SHADED[0] += _n_shaded
            CLASS_PX_COUNTED[0] += _n_counted
            CLASS_PX_LEAK[0] += _n_leak
            CLASS_PX_EXTRA[0] += _n_extra
            CLASS_PX_CHUNKS[0] += 1
            if _n_leak:
                # LOUD, and per occurrence: an unattributed shaded pixel is
                # a class the census cannot see, which is the whole defect.
                print("class census IMBALANCE, frames %d..%d: %d of %d "
                      "shaded pixels carry NO class attribution. A raster "
                      "class wrote coverage into `fg` without writing "
                      "fam_pix; its pixels are missing from CLASSMATRIX "
                      "and the totals below are short by that much."
                      % (c0, c1 - 1, _n_leak, _n_shaded), flush=True)

        if mid_pix is not None:
            # flip to screen order to match the emitted frames
            MATID_DUMP.append(
                torch.where(fg, mid_pix, torch.full_like(mid_pix, 65535))
                .flip(1).to(torch.int32).cpu())
        if fam_pix is not None and args.fam_reach:
            MAT_SEEN.update(mid_op[fg_o].unique().tolist())
            _sel = fam_pix[fg]
            if _sel.numel():
                FAM_HIST.append(torch.bincount(
                    _sel, minlength=len(FAM_NAMES)).float() / _sel.numel())
        # The one UNCONDITIONAL consumer of the opaque face selection below
        # the block: sub_mat[m_op] is (0, C) when the class is empty and the
        # gather by i_o raises regardless of every feature flag, so the
        # else-branch's defaults alone would not have got a frame out. None
        # is safe -- the only reader (the --gbuffer writer, :34447) already
        # writes a zero scalar instead of reading this global.
        globals()["_DBG_MATID"] = (sub_mat[m_op][i_o] if int(m_op.sum())
                                   else None)
        globals()["_DBG_MODE"] = sub_mode[m_op][i_o] if False else None
        irr = None
        amb_map = None
        indirect = None
        lm_ok = None
        gt_lm_u = gt_lm_v = None
        if (LIGHTMAP is not None or IRRMAP is not None
                or (args.gt_lighting
                    and args.baked_lighting in ("auto", "lightmap"))):
            # third channel rides the validity flag through the same
            # perspective-correct interpolation as the UVs, so a triangle
            # straddling lightmapped and unlightmapped vertices degrades
            # smoothly instead of snapping.
            lm, _ = _interp(lmattr[used][None].contiguous(), rast_o,
                            local[m_op])
            lm_ok = lm[..., 2]
            u_a = lm[..., 0].clamp(0, 1)
            v_a = lm[..., 1].clamp(0, 1)
            if args.lm_flip_v:
                v_a = 1.0 - v_a

            def _atlas(tex):
                ht, wt = tex.shape[0], tex.shape[1]
                # TEXEL CENTRES via the one shared definition. Both
                # branches used to map u*(wt-1) -- corner to corner --
                # which is the THIRD instance of the defect
                # cube_sample() documents fixing and _gt_tex2d carried
                # for the project's whole life
                # (TEXEL_ADDRESSING_SWEEP.md). Same atlas, second
                # reader: the recurrence is per-INPUT-OBJECT, so the
                # addressing now has exactly one definition and every
                # reader calls it.
                if not args.irr_bilinear:
                    # The DEFAULT branch, and the worse one to get
                    # wrong: on a nearest tap the half-texel shift is
                    # not blurred away, it selects a different luxel --
                    # at an atlas chart edge, a neighbouring chart's.
                    ix, iy = env_shading.nearest_index(u_a, v_a, wt, ht)
                    return tex[iy.long(), ix.long()]
                # --irr-mip band-limits the atlas but the lookup still
                # snaps to a luxel, so a chart seen edge-on gets blocky
                # irradiance. Clamped (not wrapped): charts must not
                # bleed.
                x, y = env_shading.texel_coord(u_a, v_a, wt, ht)
                x, y = x.clamp(0, wt - 1), y.clamp(0, ht - 1)
                x0, y0 = x.floor(), y.floor()
                fx, fy = x - x0, y - y0
                x0 = x0.long().clamp(0, wt - 1)
                y0 = y0.long().clamp(0, ht - 1)
                x1 = (x0 + 1).clamp(max=wt - 1)
                y1 = (y0 + 1).clamp(max=ht - 1)
                g00 = tex[y0, x0]
                # LIGHTMAP is single-channel, IRRMAP is RGB: only add the
                # trailing broadcast axis when the gather actually made one
                if g00.dim() > fx.dim():
                    fx, fy = fx.unsqueeze(-1), fy.unsqueeze(-1)
                return ((g00 * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
                        + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)

            gt_lm_u, gt_lm_v = u_a, v_a
            # _gt_value on each, because fixing _atlas's addressing
            # (0b0cd458) moved ZERO pixels on every leg of a 48-pose arm
            # while IRRMAP was demonstrably loaded. Either these outputs
            # are discarded downstream -- the same shape as the spec cube
            # and the CSM sign -- or the fix genuinely had no effect, and
            # nothing in the file could tell the two apart. These probes
            # can: `mean` says the reader produced signal, and whether
            # the FRAME moved is measured separately. A term that is
            # non-zero here and invisible in the image is a discarded
            # consumer, which is a finding, not a null.
            if LIGHTMAP is not None:
                irr = _atlas(LIGHTMAP)    # baked sun shadow/visibility, 0..1
                _gt_value("atlas.lightmap", irr, identity=1.0,
                          note="_atlas() reader -- NOT gt_baked_lightmap's "
                               "_gt_tex2d. Two readers of one atlas; this "
                               "is the pre-GT path's")
            if IRRADIANCE is not None:
                amb_map = _atlas(IRRADIANCE)
                _gt_value("atlas.irradiance", amb_map, identity=0.0)
            if IRRMAP is not None:
                indirect = _atlas(IRRMAP).float()
                _gt_value("atlas.indirect", indirect, identity=0.0,
                          note="if this is non-zero and the frame does not "
                               "move when _atlas changes, its consumer is "
                               "dead on the --gt-lighting path")
        gt_vlit = None
        if GT_VLIT is not None:
            # COLOR1 PerVertexLighting, interpolated by the rasteriser --
            # which is the whole point of the axis: 0 pixel-rate fetches.
            gt_vlit, _ = _interp(GT_VLIT[used][None].contiguous(),
                                 rast_o, local[m_op])
        vis = None
        vol = None
        skyv = None
        ssao_o = None
        alights = None
        dirocc_o = ss88_o = wet_f0_o = wet_sc_o = None
        # --- ENGINE PER-VIEW RENDER TARGETS -------------------------------
        # The DEPTH TARGET itself, which this renderer never exposed to the
        # shading pass: `depth` is the composited opaque+alpha-test NDC z
        # that the three raster classes agreed on, and the offset-80 read
        # differences its low-res copy against gl_FragCoord.z.
        zfc_o = _frag_depth(depth, fg)
        wpos = None
        # ONE guard, the union of both branches' conditions. Two agents each
        # added a reason to interpolate world position; stacking their guards
        # left the first with no body. Any consumer needing wpos must appear
        # here or it silently gets None.
        alights_back = None
        if (SHADOW is not None or VOL is not None or SKYVIS is not None
                or args.ssao or LIGHTS is not None
                or args.dirocc or args.ss_shadow or args.weather
                or args.gt_lighting
                # csgo_projected_decals' depth-reconstruction check needs
                # the interpolated world position to measure max|delta|
                # against. String flags: compared to "on", never truthy.
                or args.decal_project == "on"
                or args.sprite_particles == "on"):
            wpos = wpos_o
            if wpos is None:
                wpos, _ = _interp(VS_VPOS[used][None].contiguous(),
                                  rast_o, local[m_op])
            # the reference reads the GEOMETRIC normal here (_24347, the
            # normalize() at glsl:292), not the normal-mapped one
            n_ref = n_geo_o if n_geo_o is not None else nrm_o
            if args.dirocc:
                # --- impl-ssao-filters: the offset-76 PRODUCER ---------
                # dirocc_target() below is the fitted stand-in written
                # when the producer was believed unrecoverable.
                # aoproxy_splat IS the producer (see its docstring), so
                # it is bound here in preference: `f if q else g`.
                if args.aoproxy_splat:
                    _s = DIROCC_S
                    _occ_rt = aoproxy_splat(
                        zfc_o[:, ::_s, ::_s][:, :DIROCC_H, :DIROCC_W
                                             ].contiguous(),
                        (wpos - eye[c0:c1].view(-1, 1, 1, 3)
                         )[:, ::_s, ::_s][:, :DIROCC_H, :DIROCC_W
                                          ].contiguous(),
                        eye[c0:c1], fwd[c0:c1], _aoproxy_set_for(wpos))
                    AOPROXY_STAT["frames"] += 1
                    del _s
                else:
                    _occ_rt = dirocc_target(wpos, fg, eye[c0:c1],
                                            fwd[c0:c1])
                _dep_rt = dirocc_depth_target(zfc_o)
                dirocc_o = dirocc_resolve(_occ_rt, _dep_rt, zfc_o, n_ref, fg)
                _has76 = fg
                if fam_pix is not None:
                    # csgo_static_overlay and csgo_glass do not have these
                    # two handles in their per-view CB at all.
                    # csgo_simple and csgo_simple_3layer_parallax DO, and
                    # are therefore deliberately absent from this
                    # exclusion: both declare set=1 binding=3 members at
                    # offsets 72, 76, 80, 88 and 104 and both consume 76
                    # and 80 in the ambient-basis resolve and 88 as a
                    # min() into the cascade shadow
                    # (csgo_simple_ps.glsl:76-99, :266-304, :382). Read
                    # off their own decompiled std140 layout this
                    # session, not carried over from
                    # PER_VIEW_RENDER_TARGETS.md, whose table covers six
                    # other shaders and neither of these.
                    _no76 = (fam_pix == FAM_OVERLAY)
                    if "csgo_glass.vfx" in FAM_NAMES:
                        _no76 = _no76 | (fam_pix
                                         == FAM_NAMES.index("csgo_glass.vfx"))
                    dirocc_o = torch.where(_no76, torch.ones_like(dirocc_o),
                                           dirocc_o)
                    _has76 = fg & (~_no76)
                _rt_reach("off76_dirocc", _has76, fg)
                _rt_reach("off80_depthgather", _has76, fg)
                del _occ_rt, _dep_rt
            if args.ss_shadow:
                ss88_o = ss_shadow_read(
                    ss_shadow_target(wpos, zfc_o, fg, mvp[c0:c1]), fg)
            if args.weather and RIPPLE is not None:
                _vcw = vc_o[..., 3] if vc_o.shape[-1] > 3 \
                    else torch.zeros_like(fg, dtype=torch.float32)
                _hgt = hgt_o if hgt_o is not None \
                    else torch.full_like(zfc_o, 0.5)
                _ao_in = ao_o if ao_o is not None \
                    else torch.ones_like(zfc_o)
                _rgh_in = rough if rough is not None \
                    else torch.full_like(zfc_o, args.roughness)
                _a, _r, _n, _ao, wet_f0_o, wet_sc_o = weather_wet(
                    wpos, n_ref, vc_o[..., 2], _vcw, _hgt, _ao_in, _rgh_in,
                    alb[..., :3], fg, tsec[c0:c1])
                if fam_pix is not None:
                    # offset 116 exists in the per-view CB of the two
                    # `environment` shaders ONLY; the other four jump
                    # 104 -> the next member. Their pixels keep the dry
                    # surface set. csgo_simple and
                    # csgo_simple_3layer_parallax were checked against
                    # this list rather than assumed into it: their
                    # per-view CBs run 16,32,48,72,76,80,88,104,192,...
                    # with NO member at 116, so they stay dry and this
                    # loop is correct unchanged.
                    _h116 = torch.zeros_like(fg)
                    for _fn in ("csgo_environment_blend.vfx",
                                "csgo_environment.vfx"):
                        if _fn in FAM_NAMES:
                            _h116 = _h116 | (fam_pix == FAM_NAMES.index(_fn))
                    _h1 = _h116.unsqueeze(-1)
                    _a = torch.where(_h1, _a, alb[..., :3])
                    _r = torch.where(_h116, _r, _rgh_in)
                    _n = torch.where(_h1, _n, nrm)
                    _ao = torch.where(_h116, _ao, _ao_in)
                    wet_f0_o = torch.where(_h116, wet_f0_o,
                                           torch.full_like(wet_f0_o,
                                                           args.weather_f0))
                    wet_sc_o = torch.where(_h116, wet_sc_o,
                                           torch.zeros_like(wet_sc_o))
                # ⚠️ THESE WRITES ARE FAMILY-SCOPED AND WERE APPLIED
                # GLOBALLY. The mask that restricts them --
                # `_n = torch.where(_h1, _n, nrm)` and its siblings -- sits
                # inside `if fam_pix is not None`, while the assignments sat
                # OUTSIDE it. So on any run without a side table, fam_pix is
                # None, the restriction never happens, and `nrm = _n`
                # replaced the shading normal on EVERY pixel with the
                # weather block's -- discarding the normal map for the whole
                # frame.
                #
                # --weather defaults to 1 and RIPPLE is a procedural
                # stand-in, so this fired on every bare invocation ever run.
                # It is also silent by construction: the log says "per-view
                # wetness is 0, so the shader's own gate skips the ripple
                # reads", i.e. the term believed itself inert while
                # overwriting the frame.
                #
                # MEASURED: --normal-strength 1.0 vs 8.0 on the bare path
                # bends the normal 1.24 deg -> 8.38 deg at the producer and
                # arrived here as 0.295566 both times. Bisected to this
                # block with four probes; a/b/c differ (0.295187 vs
                # 0.286425) and d, past it, does not.
                #
                # The mask is what says WHICH families are wet. Without it
                # the honest answer is not "all of them" -- it is that this
                # block cannot act, so it does not.
                if fam_pix is None:
                    _weather_needs_famside()
                else:
                    alb = torch.cat([_a, alb[..., 3:]], dim=-1)
                    nrm = _n
                    if rough is not None:
                        rough = _r
                    if ao_o is not None:
                        ao_o = _ao
                    elif args.dirocc:
                        # _20418 = _21714 * max(_12727, 0): the wet
                        # block's AO is the second factor, and with no
                        # --vmat-ao it is the only one
                        dirocc_o = (dirocc_o * _ao if dirocc_o is not None
                                    else None)
            if SHADOW is not None:
                vis = sun_visibility(wpos)
            if VOL is not None:
                vol = sample_volume(wpos)
            if args.ssao:
                # --- impl-ssao-filters: the estimator is BOUND here ----
                # `--cs2ao-estimator fitted` keeps the pre-existing
                # McGuire-paper stand-in (the prior measured numbers used
                # it); every other value runs a transcription. This is a
                # binding, not a definition: with the default 'sao' the
                # reference chain -- ssao_convert_depth,
                # ssao_downsample_depth, the estimator, and two
                # ssao_bilateral_blur passes -- is what actually executes
                # for --ssao, and the invented one does not.
                if args.cs2ao_estimator == "fitted":
                    ssao_o = sao(wpos,
                                 nrm_o if args.ssao_geo_normal else nrm,
                                 fg, eye[c0:c1], fwd[c0:c1])
                else:
                    _rayd = wpos - eye[c0:c1].view(-1, 1, 1, 3)
                    ssao_o = cs2ao_run(
                        zfc_o, wpos,
                        nrm_o if args.ssao_geo_normal else nrm,
                        fg, eye[c0:c1], fwd[c0:c1], _rayd, mvp[c0:c1])
                    del _rayd
                CS2AO_STAT["px"] += float(fg.sum())
                CS2AO_STAT["frames"] += 1
            if SKYVIS is not None:
                skyv = sample_skyvis(
                    wpos + nrm * args.skyvis_offset
                    if args.skyvis_offset else wpos)
                if SKYSTAT is not None:
                    SKYSTAT[0] += float(skyv[fg].sum())
                    SKYSTAT[1] += float(fg.sum())
            if LIGHTS is not None:
                _wb = bool(args.transmissive and args.transmissive_lights
                           and trans_o is not None)
                if _wb:
                    alights, alights_back = analytic_lights(
                        wpos, nrm, mvp[c0:c1], want_back=True)
                else:
                    alights = analytic_lights(wpos, nrm, mvp[c0:c1])
                if args.lights_dump:
                    _fgf = fg.float()
                    _npx = _fgf.sum((1, 2)).clamp(min=1)
                    _mn = (alights.mean(-1) * _fgf).sum((1, 2)) / _npx
                    _cv = ((alights.mean(-1) > 1e-4) & fg).float().sum(
                        (1, 2)) / _npx
                    for _j in range(alights.shape[0]):
                        LDUMP_ROWS.append(
                            {"frame": c0 + _j,
                             "mean_added": float(_mn[_j]),
                             "coverage": float(_cv[_j])})
        fogin = None
        probe = None
        # --- csgo_lightmappedgeneric / generic.vfx / csgo_imported ----
        # Everything those three shaders read that _post_chain does not
        # already receive. Passed as ONE new keyword so no existing
        # positional argument moves -- three agents are editing this file
        # in the same session and a shifted positional is silent.
        fam_pack = None
        if LMG_LIVE and PBR:
            fam_pack = {
                "fam": MAT_FAM[mid_op],
                "mid": mid_op,
                "x": MAT_EXT2[mid_op],
                "uv": uv_o,
                "n_geo": n_geo_o,
                # assigned to None at the top of the `if PBR:` block, so
                # it always exists whenever this dict is built
                "tan4": tan4,
                "vcol": vc_o,
                "lm_u": gt_lm_u, "lm_v": gt_lm_v,
            }
        gt_pack = None
        if args.gt_lighting:
            wp = wpos if wpos is not None else wpos_o
            if wp is None:
                wp, _ = _interp(vertices[used][None].contiguous(),
                                rast_o, local[m_op])
            gt_pack = {"wpos": wp, "eye": eye[c0:c1],
                       "lm_u": gt_lm_u, "lm_v": gt_lm_v,
                       "vlit": gt_vlit}
        if PROBE_ATLAS is not None and not args.gt_lighting:
            # gt_baked_probe() reads the same field with the shader's own
            # volume blend; running sample_probes() as well would apply
            # the probe term twice.
            wp = wpos_o
            if wp is None:
                wp, _ = _interp(VS_VPOS[used][None].contiguous(),
                                rast_o, local[m_op])
            probe = sample_probes(wp)
        if args.fog:
            wp = wpos_o
            if wp is None:
                wp, _ = _interp(VS_VPOS[used][None].contiguous(),
                                rast_o, local[m_op])
            # back to SOURCE units: the pack is metres = source * 0.0254
            fogin = torch.stack([
                (eye[c0:c1].view(-1, 1, 1, 3) - wp).norm(dim=-1) / S,
                wp[..., 1] / S], dim=-1)
        sh_irr = sh_dir = sh_spec = None
        if args.ibl_spec and view_o is not None:
            wp = wpos_o
            if wp is None:
                wp, _ = _interp(VS_VPOS[used][None].contiguous(),
                                rast_o, local[m_op])
            _Ls = SH[probe_of(wp)] if args.ibl_nearest else sh_at(wp)
            _ndv = (nrm * view_o).sum(-1, keepdim=True).clamp(0, 1)
            _mir = _nrm(2.0 * _ndv * nrm - view_o)
            if args.ibl_dominant_dir and rough is not None:
                # Frostbite's getSpecularDominantDir, as the real shader
                # uses it: a2 = r*r, k = 1 - a2, and the sample direction
                # lerps from the mirror vector to N by k*(sqrt(k)+a2), so a
                # rough surface samples along its NORMAL rather than along
                # a mirror it does not actually have.
                _a2 = (rough * rough).unsqueeze(-1)
                _k = (1.0 - _a2).clamp(0, 1)
                _t = (_k * (_k.sqrt() + _a2)).clamp(0, 1)
                _d = _nrm(nrm + (_mir - nrm) * _t)
            else:
                _d = _mir
            if CUBE is not None:
                # lod = scale*sqrt(roughness) over the prefiltered chain --
                # the mapping the shader uses.  Inert while SH-L2 stood in
                # for the chain because there was no mip to select.
                _lod = args.ibl_lod_scale * rough.clamp(min=0).sqrt()
                _pi = probe_of(wp)
                _env = cube_sample(_pi, _d, _lod)
                # probe's own average radiance, for the same
                # exposure-neutral ratio the SH path uses
                _dcs = (0.282095 * SH[_pi][..., 0]).clamp(min=1e-4)
            else:
                _env = sh_radiance(_Ls, _d)
                _dcs = (0.282095 * _Ls[..., 0]).clamp(min=1e-4)
            # Expressed as a RATIO to the probe's own DC term, then scaled
            # by the ambient the renderer already trusts. The probes are in
            # absolute HDR units (DC up to 24.9) that do not share the
            # fitted lighting's scale -- using them raw is what made
            # --ibl-sh-mode replace explode to nrmse 2.00.
            sh_spec = (_env / _dcs).clamp(0, args.irr_clamp)
            if args.ibl_cube_norm:
                # g_bCubemapNormalization, from the shader: preserve chroma,
                # clamp magnitude at 1.75.
                _len = sh_spec.norm(dim=-1, keepdim=True)
                sh_spec = (sh_spec / (_len + 1e-3)) * _len.clamp(0, 1.75)
            if args.ibl_spec_flat:
                sh_spec = torch.ones_like(sh_spec)
        if args.ibl_sh:
            wp = wpos_o
            if wp is None:
                wp, _ = _interp(VS_VPOS[used][None].contiguous(),
                                rast_o, local[m_op])
            if args.ibl_nearest:
                _L = SH[probe_of(wp)]
            else:
                _L = sh_at(wp)
            _n_ibl = (n_geo_o if (args.ibl_normal == "geometric"
                                  and n_geo_o is not None) else nrm)
            sh_irr = sh_eval(_L, _n_ibl)
            _dc = (0.886227 * _L[..., 0]).clamp(min=1e-4)
            if args.ibl_sh_mode == "dirlum":
                # The probe's directional COLOUR and the baked lightmap's
                # bounce colour are the same physical quantity from two
                # different bakes, so multiplying them double-counts the
                # tint (measured: it drives the whole frame orange). The
                # luminance ratio is the part the lightmap DC term
                # genuinely does not carry.
                sh_dir = (1.0 + (sh_irr.mean(-1, keepdim=True)
                                 / _dc.mean(-1, keepdim=True) - 1.0)
                          * args.ibl_sh_gain).expand_as(sh_irr)
            else:
                sh_dir = 1.0 + (sh_irr / _dc - 1.0) * args.ibl_sh_gain
        # --- render / tools modes ------------------------------------
        # These REPLACE the shaded image rather than overlaying it: a
        # visualisation blended over a lit frame cannot be read off, which
        # is the whole point of the mode.
        if (args.tools_vis != "off" or args.tools_shading_complexity
                or args.quad_overdraw):
            _img = None
            if args.quad_overdraw:
                _acc, _q = quad_overdraw(clip, local[m_op],
                                         args.quad_overdraw_layers)
                if not QOD_NOTE:
                    QOD_NOTE.append(1)
                    _sat = float((_acc >= args.quad_overdraw_layers)
                                 .float().mean())
                    print(f"D_QUAD_OVERDRAW: {args.quad_overdraw_layers} "
                          f"peeled layers, ramp 0..{args.quad_overdraw_max:g} "
                          f"invocations/quad; {_sat:.4%} of pixels reach the "
                          "layer cap and are therefore a FLOOR", flush=True)
                _img = _heat(_q / max(args.quad_overdraw_max, 1e-6))
                # _heat(0) is blue, not black, so uncovered pixels must be
                # masked explicitly or an empty frame reads as depth 0.
                _img = torch.where((_acc > 0).unsqueeze(-1), _img,
                                   torch.zeros_like(_img))
            elif args.tools_shading_complexity:
                _img = tools_shading_complexity(
                    fg, mid_op, args.tools_complexity_unit)
            else:
                _pvl = None
                if VS_COLOR1 is not None:
                    import cs2_vertex_stage as _vsx0
                    _c1p, _ = _interp(
                        VS_COLOR1[used][None].contiguous(), rast_o,
                        local[m_op])
                    _pvl = _vsx0.per_vertex_lighting(_c1p)
                _piv = _fpar = _bend = _disp = _perr = None
                if VS_PIVOT is not None:
                    _piv, _ = _interp(VS_PIVOT[used][None].contiguous(),
                                      rast_o, local[m_op])
                if VS_FPARAM is not None:
                    _fpar, _ = _interp(
                        VS_FPARAM[used][None].contiguous(), rast_o,
                        local[m_op])
                if VS_LAST.get("bend") is not None:
                    _bend, _ = _interp(
                        VS_LAST["bend"][used][:, None][None].contiguous(),
                        rast_o, local[m_op])
                    _bend = _bend[..., 0]
                if VS_LAST.get("disp") is not None:
                    _disp, _ = _interp(
                        VS_LAST["disp"][used][None].contiguous(), rast_o,
                        local[m_op])
                if args.compressed_normals == "on" and PBR:
                    _perr, _ = _interp(
                        (VS_VNRM - vnormal)[used].norm(dim=1)[:, None][None]
                        .contiguous(), rast_o, local[m_op])
                    _perr = _perr[..., 0]
                if args.secondary_uv != "material":
                    _sel = torch.full_like(uv_o[..., 0],
                                           float(args.secondary_uv
                                                 if args.secondary_uv
                                                 != "off" else 1))
                elif int(m_op.sum()):
                    _sel = sub_sec[m_op][i_o] + 1
                else:
                    # REFUSAL, the second and last disclosed consumer.
                    # sub_sec[m_op] is (0,) with an empty opaque class and
                    # the gather by i_o raises; there is also nothing to
                    # show -- the material-driven UV selection is a
                    # PER-OPAQUE-FACE quantity and no opaque face was
                    # drawn. The value is READ off the visualiser, not
                    # invented: tools_vis's `uvset` branch renders 2 red,
                    # 1 green and everything else black, so 0 is already
                    # its own "no set selected" and cannot be confused
                    # with either real set.
                    print("tools-vis uvset skipped: 0 opaque faces (empty "
                          "class, #27 family) -- no material-driven UV "
                          "selection exists to visualise; every pixel "
                          "takes the mode's own 'no set' value 0 (black), "
                          "which is NOT a measurement that both UV sets "
                          "went unselected.", flush=True)
                    _sel = torch.zeros_like(uv_o[..., 0])
                _tan = None
                if PBR and VS_VTAN is not None:
                    _tan, _ = _interp(
                        VS_VTAN[used][None].contiguous(), rast_o,
                        local[m_op])
                _img = tools_vis(args.tools_vis, fg, uv_o, uv2_o, _sel,
                                 nrm, _tan, vc_o, _pvl,
                                 _piv, _fpar, _bend, _disp, _perr)
            _img = _img.flip(1)
            if args.supersample > 1:
                _img = torch.nn.functional.avg_pool2d(
                    _img.permute(0, 3, 1, 2),
                    args.supersample).permute(0, 2, 3, 1)
            outs.append((_img.clamp(0, 1) * 255).to(torch.uint8))
        if dm is not None:
            # S_MODE_DEPTH is a render MODE, not a term inside the beauty
            # pass, so it replaces the shade rather than modifying it --
            # exactly as the combo does in the engine.
            _d = dm.flip(1)
            if args.supersample > 1:
                _d = torch.nn.functional.avg_pool2d(
                    _d.permute(0, 3, 1, 2),
                    args.supersample).permute(0, 2, 3, 1)
            outs.append((_d.clamp(0, 1) * 255).to(torch.uint8))
            continue
        # --- FAM_BLACK_UNLIT / FAM_VERTEXLIT --------------------------
        # Both are OPAQUE-class draws, so they are shaded here, at the
        # END of the opaque pass, where the lightmap, the offset-76
        # resolve, the offset-88 read and the cascade shadow this pass
        # already produced are all in scope. Each family's own composite
        # REPLACES the generic path on its own pixels -- routed through
        # the SAME unlit compositor csgo_black_unlit and --s-lit already
        # use in _post_chain (rgb = rgb*(1-u) + linear*u), so this file
        # still has one such path and not three.
        if FAMILY and SFAM and PBR and mid_op is not None:
            _sfwp = wpos if wpos is not None else wpos_o
            if _sfwp is None:
                _sfwp, _ = _interp(VS_VPOS[used][None].contiguous(),
                                   rast_o, local[m_op])
            _sfeye = eye[c0:c1].view(-1, 1, 1, 3)
            _sfx = MAT_EXT2[mid_op]
            _sft = float(tsec[c0]) if tsec is not None else 0.0
            _sfrgb = alb[..., :3]
            _sfa = alb[..., 3]
            if args.sf_black_unlit and FAM_BLACK_UNLIT >= 0:
                b_rgb, b_a = sf_black_unlit_shade(
                    _sfwp, _sfeye, _e2(_sfx, "sf_bu_fog_enabled"))
                if args.sf_tools_vis != 0:
                    b_rgb = sf_tools_vis(args.sf_tools_vis, b_rgb,
                                         nrm, _sfwp, _sfeye, _sft)
                _on = (MAT_FAM[mid_op] == FAM_BLACK_UNLIT)
                _o3 = _on.unsqueeze(-1).float()
                _sfrgb = _sfrgb * (1 - _o3) + b_rgb * _o3
                _sfa = _sfa * (1 - _on.float()) + b_a * _on.float()
                unlit_o = torch.maximum(
                    unlit_o if unlit_o is not None
                    else torch.zeros_like(fg, dtype=torch.float32),
                    _on.float())
                SF_REACH["black_unlit"][0] += float((_on & fg).sum())
                SF_REACH["black_unlit"][1] += float(fg.sum())
            if args.sf_vertexlit and FAM_VERTEXLIT >= 0:
                # r51_m3:209 fetches g_tNormal and decodes it TWO-CHANNEL.
                # Without --normal-map this pass never fetched one, so the
                # family would silently shade off the geometric normal --
                # a path that reads as implemented and is not. Fetch it.
                _nm = nmap
                if _nm is None and MAT_NRM1 is not None:
                    _nm = sample_aux(uv_o, MAT_NRM1[mid_op], tapsA_o)
                if _nm is None:
                    raise SystemExit(
                        "--sf-vertexlit needs a normal map: r51_m3:209-214 "
                        "decodes g_tNormal two-channel and :680 takes the "
                        "roughness off its B channel. Add --normal-map (or "
                        "a pack that carries MAT_NRM1).")
                _t4 = tan4
                if _t4 is None and VS_VTAN is not None:
                    _t4, _ = _interp(VS_VTAN[used][None].contiguous(),
                                     rast_o, local[m_op])
                # the baked term: the lightmap atlases this pass already
                # gathered. `indirect` is the RGB bounce irradiance
                # (r51_m3's _21761) and `irr` the per-light visibility
                # mask (_11110). --sf-vertexlit-baked selects which of
                # the four D_BAKED_LIGHTING_* modules runs.
                _amb = (AMBIENT_GROUND.view(1, 1, 1, 3)
                        + (AMBIENT_SKY - AMBIENT_GROUND).view(1, 1, 1, 3)
                        * (nrm[..., 1].clamp(-1, 1) * 0.5 + 0.5
                           ).unsqueeze(-1))
                _bmode = SF_COMBO["vertexlit"]["baked"]
                if _bmode == "lightmap" or _bmode == "auto":
                    _birr = indirect if indirect is not None else _amb
                elif _bmode == "probe":
                    _birr = probe if probe is not None else _amb
                elif _bmode == "vertex-stream":
                    _birr = gt_vlit if gt_vlit is not None else _amb
                else:
                    _birr = torch.zeros_like(_amb)
                _bvis = (1.0 - irr) if irr is not None else None
                _view = view_o
                if _view is None:
                    _view = _nrm(_sfeye - _sfwp)
                v_rgb, v_a = sf_vertexlit_shade(
                    uv_o, uv2_o, mid_op, _sfx, alb[..., :3], alb[..., 3],
                    _nm, _sfwp, _sfeye,
                    n_geo_o if n_geo_o is not None else nrm, _t4, vc_o,
                    _birr, _bvis, dirocc_o, ss88_o, vis, sun_dir,
                    SUN_COLOR.view(1, 1, 1, 3), _amb, alights, _view,
                    _sft, SF_COMBO["vertexlit"])
                if SF_COMBO["vertexlit"]["opaque_fade"]:
                    # D_OPAQUE_FADE (16), r51_m4..m7 -- the same Bayer
                    # dither unit the other families' axis uses.
                    _fv, _fk = opaque_fade(vc_o[..., 3].clamp(0, 1),
                                           torch.ones_like(v_a))
                    v_a = _fv * _fk.float()
                if args.sf_tools_vis != 0:
                    v_rgb = sf_tools_vis(args.sf_tools_vis, v_rgb, nrm,
                                         _sfwp, _sfeye, _sft)
                _on = (MAT_FAM[mid_op] == FAM_VERTEXLIT)
                _o3 = _on.unsqueeze(-1).float()
                _sfrgb = _sfrgb * (1 - _o3) + v_rgb * _o3
                _sfa = _sfa * (1 - _on.float()) + v_a * _on.float()
                unlit_o = torch.maximum(
                    unlit_o if unlit_o is not None
                    else torch.zeros_like(fg, dtype=torch.float32),
                    _on.float())
                SF_REACH["vertexlit"][0] += float((_on & fg).sum())
                SF_REACH["vertexlit"][1] += float(fg.sum())
            alb = torch.cat([_sfrgb, _sfa.unsqueeze(-1)], dim=-1)
        # ============================================================
        # FAM_PROJECTED_DECALS / FAM_SPRITECARD -- the two non-world
        # families. Built HERE because this is the first point at which
        # the RESOLVED scene depth exists (`zfc_o`), and the decal
        # family's own dynamic axes say it cannot run before that.
        # Composited inside _post_chain in LINEAR space, before the
        # tonemap, which is where the engine's forward translucent pass
        # writes. Both are references passed by keyword, so "defined but
        # never called" is not a state this can be in.
        # ============================================================
        _decal_l = _sprite_l = None
        if args.decal_project == "on" or args.sprite_particles == "on":
            _ivp = torch.linalg.inv(mvp[c0:c1].transpose(1, 2).float())
            _wp = wpos
            if _wp is None:
                _wp, _ = _interp(VS_VPOS[used][None].contiguous(),
                                 rast_o, local[m_op])
            _zs = 2.0
            if args.nonworld_selftest_inject == "depth-remap":
                # FAULT INJECTION: gl_FragCoord.z -> clip z is exactly
                # (2, -1) for this frustum. Break the scale and the
                # reconstruction error must explode.
                _zs = 2.5
            _md, _mn, _mdn = _nw.selftest_depth_reconstruction(
                zfc_o, _wp, fg, _ivp, H, W, z_scale=_zs)
            NW_DEPTH_ERR.append((_md, _mn, _mdn))
            _sunc = (SUN_COLOR if "SUN_COLOR" in globals()
                     else torch.ones(3, device=device))
            _ambsh = (sh_irr if sh_irr is not None
                      else torch.zeros(3, 4, device=device)
                      + torch.tensor([0.0, 0.0, 0.0, 0.12],
                                     device=device))
            if _ambsh.dim() != 2 or _ambsh.shape != (3, 4):
                _ambsh = (torch.zeros(3, 4, device=device)
                          + torch.tensor([0.0, 0.0, 0.0, 0.12],
                                         device=device))
            _baked = None
            if args.decal_baked_lighting == "lightmap" and irr is not None:
                _baked = irr.unsqueeze(-1) \
                    if irr.dim() == 3 else irr
            elif args.decal_baked_lighting == "probe" and probe is not None:
                _baked = probe if probe.dim() == 4 else probe.unsqueeze(-1)
            if args.nonworld_selftest_inject == "decal-dead":
                # FAULT INJECTION: a decal volume placed behind the eye
                # reaches no pixel; the reach table must say so.
                _sv = args.decal_synth_dist
                args.decal_synth_dist = -abs(_sv) - 1e4
            _decal_l = _nw.decal_layer(
                args, zfc_o, NW_TRANS_Z[0], fg, _ivp, eye[c0:c1],
                fwd[c0:c1], sun_dir, _sunc, _ambsh,
                vis if vis is not None else None, NW_SIDE, _baked,
                supersample=args.supersample)
            if args.nonworld_selftest_inject == "sprite-dead":
                _ss = args.sprite_synth_dist
                args.sprite_synth_dist = -abs(_ss) - 1e4
            _sprite_l = _nw.particle_layer(
                args, zfc_o, _ivp, eye[c0:c1], fwd[c0:c1], NW_SIDE,
                FOG_COLOR if "FOG_COLOR" in globals()
                else torch.ones(3, device=device))
        # The LAST value of nrm before it leaves the chunk loop. With
        # compose.normal_in (measured inside the consumer) this brackets
        # the drop: if these two disagree the loss is inside _post_chain,
        # if they agree and the producer's tilt does not, it is in the loop.
        _gt_value("surface.normal_out", nrm[..., 1], identity=0.0,
                  note="up-component of nrm as it LEAVES the chunk loop")
        outs.append(post_chain(alb, nrm, fg, vis, irr, amb_map,
                               view_o, rough, metal, ao_o, emis,
                               own_o.float() if PBR else None,
                               indirect, lm_ok, vol, fogin,
                               trans_o, two_sided, probe, unlit_o,
                               skyv, ssao_o, sh_irr, sh_dir,
                               sh_spec, alights,
                               dirocc_o, ss88_o, wet_f0_o, wet_sc_o,
                               gt_pack, alights_back, fam_pack,
                               # THE MERGE GAP, now closed. Both layers
                               # were already being BUILT above and then
                               # dropped on the floor here, so
                               # csgo_projected_decals and spritecard ran
                               # their full cost every frame and reached
                               # no pixel. By keyword, not position:
                               # _post_chain takes 33 positional
                               # parameters before these two and a
                               # miscount would land a decal layer in
                               # `fam` silently.
                               decal_layer=_decal_l,
                               sprite_layer=_sprite_l))
        zouts.append(zfc_o)
    frame = torch.cat(outs)
    # THE PLAYERMODEL PASS, before the viewmodel and after the three world
    # classes. That order is the geometry's: other players are world
    # geometry, so they belong in the world's depth and under anything held
    # in front of the camera. It is called UNCONDITIONALLY so that a run
    # without --playermodels prints its GAP line rather than looking
    # finished; the pass itself is what decides there is nothing to draw.
    # VIEW PUNCH, per frame, beside the other per-frame reports. Only the
    # frames actually rendered, and only when the kick is nonzero -- a line
    # per still frame of a 3,284-row path would bury the ones that moved.
    _b0 = int(VS_FRAME0[0])
    for _i in range(frame.shape[0]):
        _ri = _b0 + _i
        # The report-side dedupe this shipped with is GONE: the duplication
        # was the warm-up render, and frame_print() suppresses it at the
        # source for every report at once. A per-report guard fixed one line
        # and left the next lane's wrong by default.
        if 0 <= _ri < len(rows):
            _dp, _dy = row_punch(rows[_ri])
            if abs(_dp) > 1e-9 or abs(_dy) > 1e-9:
                frame_print(f"VIEWPUNCH f{_ri:06d} ARM={args.view_punch} "
                            f"pitch {_dp:+.4f} yaw {_dy:+.4f} deg ADDED to "
                            f"view angles ({rows[_ri]['pitch_degrees']:+.3f}, "
                            f"{rows[_ri]['yaw_degrees']:+.3f})")
    # SITE 1 of 3 -- AN EMPTY VISIBLE SET IS A VALID FRAME STATE, and this
    # is where 12 of the 14 sweep maps died. torch.cat([]) raises, so a
    # frame whose camera sees no players took down an 89M-frame job about
    # 900 times. Real demos have such frames constantly: a player alone in
    # a corridor, a camera pointed at a wall, the first tick of a round.
    #
    # The skip is PRINTED WITH THE COUNT, never silent. A silent skip and a
    # crash are both wrong in the same way -- neither lets a reader
    # downstream tell "no players were there" from "the pass was broken",
    # and this project has paid for that distinction more than once.
    _zreal = [z for z in zouts if torch.is_tensor(z)]
    if _zreal:
        # Materialise far-plane depth for empty-view chunks (ints recording
        # their frame counts) in the exact dims/dtype of the real entries.
        zbuf_all = torch.cat([
            z if torch.is_tensor(z) else
            torch.full((z, *_zreal[0].shape[1:]), float("inf"),
                       dtype=_zreal[0].dtype, device=_zreal[0].device)
            for z in zouts])
        frame = playermodel_pass(mvp, eye, frame, zbuf_all)
    else:
        frame_print(f"playermodel_pass skipped: 0 players in frustum "
                    f"(zouts empty over {len(rows) if rows else 0} row(s)) "
                    f"-- an empty visible set is a valid frame, not a "
                    f"failure; the frame renders without the playermodel "
                    f"pass.")
    # --- FLASHBANG WHITEOUT ------------------------------------------
    # LAST, over the finished frame including the viewmodel, because it is
    # a full-screen overlay the blinded player sees in front of everything
    # -- not a lighting term. Applied after the viewmodel pass for the same
    # reason the muzzle flash is applied inside it: what it composites over
    # is the image, not the scene.
    _fa = [row_flash_alpha(rows[_b0 + _i]) if 0 <= _b0 + _i < len(rows)
           else 0.0 for _i in range(frame.shape[0])]
    if any(a > 0.0 for a in _fa):
        _av = torch.tensor(_fa, device=frame.device,
                           dtype=torch.float32).view(-1, 1, 1, 1)
        _f = frame.float()
        frame = (_f + (255.0 - _f) * _av).clamp(0, 255).to(torch.uint8)
        for _i, _a in enumerate(_fa):
            if _a > 0.0:
                _r = rows[_b0 + _i]
                frame_print(
                    f"FLASHBANG f{_b0 + _i:06d} alpha {_a:.4f} "
                    f"curve={args.flash_curve} STATED "
                    f"(elapsed {_r['flash_elapsed_s']:.4f}s of "
                    f"{_r['flash_duration']:.4f}s, onset tick "
                    f"{_r['flash_start_tick']})")
    if args.viewmodel:
        # THE VIEWMODEL PASS. It runs AFTER the three world classes and
        # composites over the finished frame by coverage, with its own
        # projection, its own near/far and its own depth buffer. It never
        # reads `depth`, which is what stops it clipping into the world.
        frame = viewmodel_pass(mvp, eye, frame)
    else:
        # THE FLAG-OFF STATE, said out loud. Every other ABSENT reason is
        # printed from inside the pass, and a run that never entered it
        # would be the only one of the five that looked like a clean frame.
        _vm_frame_report(frame.shape[0],
                         absent="--viewmodel not given, so no first-person "
                                "weapon is drawn at all (GAP, not an empty "
                                "hand)")

    # -------------------------------------------------------------------
    # THE FOUR PREVIOUSLY-UNREACHED MODULES, RUN HERE.
    #
    # Each was transcribed from bytecode and then imported by nobody. They
    # run at the end of the frame because that is where their inputs
    # exist: the composited colour, the composited depth, and the camera.
    # None of them is claimed to be correctly placed in the reference's
    # render graph -- pass ORDER and RESOLUTION are not in any shader
    # package and need a frame capture. What is claimed is that the code
    # executes on real per-pixel tensors instead of sitting resident and
    # unreachable.
    if args.msaa_samples > 1:
        # Unresolved multisample depth: the per-sample attachment that
        # sampler2DMS reads. A resolve-first renderer cannot express it,
        # so it is built rather than approximated with resolved depth.
        try:
            # ms_depth_single() is the NON-MSAA side of the axis -- a
            # 1-sample wrapper for D_NON_MSAA_DEPTH=1. Calling it under
            # `msaa_samples > 1` built a 1-sample attachment and reported
            # success, so the sample count was silently discarded and the
            # reach line claimed MSAA where there was none. The real
            # producers need per-sample depth, and the only place this
            # renderer has it is the supersample grid.
            _ss = max(1, int(args.supersample))
            if _ss * _ss >= args.msaa_samples:
                _zms = _msaa.ms_depth_from_supersample(
                    depth, _ss, args.msaa_samples)
            else:
                # Stated, not silently downgraded: with no supersample
                # grid there are no sub-pixel depths to gather, and
                # duplicating one cell would make an S-sample attachment
                # that is really a 1-sample one with no reader able to
                # tell -- which the producer itself refuses to do.
                _zms = _msaa.ms_depth_single(depth)
                MSAA_STATE["note"] = (
                    f"asked for {args.msaa_samples} samples; --supersample "
                    f"{_ss} gives {_ss * _ss} sub-pixel cells, so the "
                    f"1-sample (D_NON_MSAA_DEPTH) attachment was built "
                    f"instead. Run --supersample 2 for a real 4x grid.")
            # MSDepthAttachment, not a tensor: it holds (B,H,W,S) in .z
            # and refuses to be one value per pixel by construction.
            # `_zms.numel()` raised, and unlike its two neighbours this
            # call was NOT guarded, so --msaa-samples 4 took the whole
            # run down -- the module worked and its reach counter killed
            # it.
            MSAA_STATE["built"] = int(_zms.z.numel())
            MSAA_STATE["samples"] = int(_zms.z.shape[-1])
            MSAA_STATE["source"] = _zms.source
        except Exception as _e:                       # noqa: BLE001
            MSAA_STATE["error"] = _mod_err(_e)
    if args.dgb_gbuffer:
        # The 4-attachment deferred layout, packed from the forward
        # frame's own terms. The reference binds these as MRT; here they
        # are carried, which is a real difference and is stated.
        try:
            # `None` IS NOT A NEUTRAL CONSTANT. These four were declared
            # None and never assigned, under a comment saying they were
            # "left as neutral constants where [the shading pass] does
            # not [have them]" -- a comment describing behaviour the code
            # did not have. dgb_passes' backend shim dispatches on
            # _is_t(xs[0]), so a None first element sent _cat down the
            # numpy branch and handed it a CUDA tensor, which is the
            # TypeError this module died of on every frame since it was
            # wired. Filled here rather than hardening _is_t against
            # None: hardening the shim would only make it fail
            # differently, while filling the terms makes the G-buffer
            # carry something a deferred shade could actually read.
            #
            # Neutral means neutral, and is STATED because it is not
            # measured: normal +Y, roughness 1, metalness 0, AO 1. The
            # forward pass does not carry per-pixel normal/PBR out to
            # this point -- the chunk loop that had them closed above --
            # so these are honest placeholders, not the frame's terms.
            # `depth` is at RASTER resolution and `frame` at OUTPUT
            # resolution -- under --supersample they differ by ss, and
            # the pack concatenates them, so it must be one or the
            # other. Take the output grid and SUBSAMPLE depth onto it
            # with a stride rather than average-pooling: averaging depth
            # across a silhouette produces a value that lies on neither
            # surface, and the G-buffer's whole purpose is that a
            # deferred pass can trust the depth it reads.
            _gb_z = depth
            if _gb_z.shape[-2:] != frame.shape[-3:-1]:
                _sy = _gb_z.shape[-2] // max(frame.shape[-3], 1)
                _sx = _gb_z.shape[-1] // max(frame.shape[-2], 1)
                _gb_z = _gb_z[..., ::max(_sy, 1), ::max(_sx, 1)]
                _gb_z = _gb_z[..., :frame.shape[-3], :frame.shape[-2]]
            _gb_fg = torch.ones_like(_gb_z, dtype=torch.bool)
            _gb_nrm = _GB_NRM
            if _gb_nrm is None:
                _gb_nrm = torch.zeros((*_gb_z.shape, 3),
                                      device=_gb_z.device,
                                      dtype=torch.float32)
                _gb_nrm[..., 1] = 1.0
            _gb_one = torch.ones_like(_gb_z, dtype=torch.float32)
            # albedo_rgba wants four channels; `frame` is the uint8
            # TONE-ENCODED colour by this point, NOT linear albedo. Said
            # out loud rather than relabelled: inverting the tone curve
            # here would be inventing an inverse nobody specified.
            _gb_alb = frame.float() / 255.0 if frame.dtype == torch.uint8 \
                else frame
            if _gb_alb.shape[-1] == 3:
                _gb_alb = torch.cat(
                    [_gb_alb, _gb_fg.to(_gb_alb.dtype).unsqueeze(-1)], -1)
            _gb = dgb_passes.gbuffer_pack(
                _gb_alb, _gb_nrm, _gb_z,
                _GB_ROUGH if _GB_ROUGH is not None else _gb_one,
                _GB_METAL if _GB_METAL is not None
                else torch.zeros_like(_gb_one),
                _GB_AO if _GB_AO is not None else _gb_one,
                _gb_fg)
            # gbuffer_pack returns a GBuffer, which is neither a dict nor
            # a tensor: it holds the four targets as Tex2D wrappers, each
            # carrying its binding and its payload in .data. The old
            # accounting handled only dict and tensor, so even once the
            # pack succeeded the reach counter raised and the module
            # still read as unreached -- a second check that could not
            # pass, sitting behind the first.
            if isinstance(_gb, dict):
                _gbn = sum(v.numel() for v in _gb.values())
            elif hasattr(_gb, "albedo"):
                _gbn = sum(_t.data.numel() for _t in
                           (_gb.depth, _gb.albedo, _gb.normal_ws,
                            _gb.lighting_terms))
            else:
                _gbn = _gb.numel()
            GBUF_STATE["packed"] = int(_gbn)
        except Exception as _e:                       # noqa: BLE001
            GBUF_STATE["error"] = _mod_err(_e)
    if args.b3_filters:
        # Fixed-tap screen-space kernels. NOT cluster loops -- 25 of the
        # 41 loop-carrying families are this shape, and writing a binner
        # for them is the wrong construct.
        try:
            # TWO call-site bugs here, not one, and the first was hiding
            # the second.
            #
            # step_xy is CB(set 1, binding 0)._m0, a 2-VECTOR: b3_blur
            # takes .x for the mK=0 modules and .y for mK=1. This passed a
            # bare Python float, so every run raised "'float' object is
            # not subscriptable". Taken from the frame's own H/W rather
            # than args.width/height so it stays right under
            # --supersample, and so x and y differ as they must on a
            # non-square target.
            #
            # And `frame` is uint8 by this point in render_window -- the
            # tone-encode returns (rgb * 255).to(uint8) -- while b3_blur
            # accumulates taps through grid_sample, which has no Byte
            # kernel. Blur in float and re-encode, rather than blurring
            # quantised bytes, which would band the result even where it
            # ran. dtype comes from the FLOAT image: keying the step
            # vector off frame.dtype would round 1/960 to 0 on uint8 and
            # silently make the filter a no-op.
            _b3_u8 = frame.dtype == torch.uint8
            _b3_in = frame.float() / 255.0 if _b3_u8 else frame
            _uv = _screen_uv_like(_b3_in)
            _b3_step = torch.tensor(
                [1.0 / max(_b3_in.shape[-2], 1),
                 1.0 / max(_b3_in.shape[-3], 1)],
                device=_b3_in.device, dtype=_b3_in.dtype)
            _b3_out = _b3.b3_blur(_b3_in, _uv, _b3_step)
            frame = ((_b3_out * 255).round().clamp(0, 255).to(torch.uint8)
                     if _b3_u8 else _b3_out)
            B3_STATE["ran"] = int(frame.numel())
        except Exception as _e:                       # noqa: BLE001
            B3_STATE["error"] = _mod_err(_e)
    return frame
