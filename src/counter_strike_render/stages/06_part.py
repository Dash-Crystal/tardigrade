if VS_ON or UVSTAGE_ON or args.tools_vis != "off":
    import cs2_vertex_stage as vsx

    # --- the vertex-rate sampler3D -----------------------------------
    # A WIND DISPLACEMENT FIELD.  Resolved by dataflow (the sampled value
    # reaches gl_Position and no lighting term); it is deliberately NOT
    # registered with the probe/irradiance atlases.
    _NEED_VOL = (args.foliage_animation or args.vertex_animation > 0
                 or args.vertex_stage_selftest)
    if not _NEED_VOL:
        pass
    elif args.vertex_noise_volume:
        VS_NOISE = torch.from_numpy(
            np.load(args.vertex_noise_volume)).float().to(device)
        print(f"vertex-rate volume: {tuple(VS_NOISE.shape)} from "
              f"{args.vertex_noise_volume}", flush=True)
    else:
        VS_NOISE = vsx.make_noise_volume(args.vertex_noise_size, 3,
                                         seed=args.vertex_noise_seed,
                                         device=device)
        VS_SUBST.append(
            f"sampler3D volume: CS2 selects it by a RUNTIME bindless "
            f"descriptor index, so the asset cannot be named from the "
            f"bytecode; a deterministic tiling value-noise volume "
            f"{args.vertex_noise_size}^3x3 (seed {args.vertex_noise_seed}) "
            f"stands in. The fetch path, coordinates, LOD-0 semantics, "
            f"repeat wrap and both decodes are the shader's")

    # --- per-vertex shader family -------------------------------------
    # An axis reaches exactly the vertices whose owning material selects
    # the family that declares it, which is what a static combo IS.
    _famv = torch.full((len(vertices),), -1, dtype=torch.long, device=device)
    if FAMILY:
        _ff = MAT_FAM[face_matid0]
        for _c in range(3):
            _famv[faces0[:, _c]] = _ff
    else:
        VS_SUBST.append(
            "no --fam-side: every vertex-stage axis is applied to ALL "
            "geometry because the per-material shader family is unknown")
    VS_FAMV = _famv
    _FAM_FOLIAGE = (FAM_NAMES.index("csgo_foliage.vfx")
                    if FAMILY and "csgo_foliage.vfx" in FAM_NAMES else -1)
    _FAM_GLASS = (FAM_NAMES.index("csgo_glass.vfx")
                  if FAMILY and "csgo_glass.vfx" in FAM_NAMES else -1)
    _FAM_COMPLEX = (FAM_NAMES.index("csgo_complex.vfx")
                    if FAMILY and "csgo_complex.vfx" in FAM_NAMES else -1)
    _FAM_CHARACTER = (FAM_NAMES.index("csgo_character.vfx")
                      if FAMILY and "csgo_character.vfx" in FAM_NAMES else -1)

    def _fam_mask(fid):
        if not FAMILY or fid < 0:
            return torch.ones(len(vertices), dtype=torch.bool, device=device)
        return VS_FAMV == fid

    # --- S_SECONDARY_UV: is our uvs2 actually CS2's LowPrecisionUv1? ---
    # Measured, not assumed.  UV repeats per world metre separates a tiling
    # texture UV from a unique unwrap by two orders of magnitude
    # (UV_CONSTRUCTION.md:90-97: uvs 0.485, uvs2 0.0028, lmuv 0.0031 --
    # glTF-ERA AND SUPERSEDED; the current ASH pack measures uvs2 EXACTLY
    # ZERO on 53.97% of the surface that selects it. See --secondary-uv.)
    if args.secondary_uv == "material":
        _p2 = vertices[faces0]
        _wa2 = torch.cross(_p2[:, 1] - _p2[:, 0], _p2[:, 2] - _p2[:, 0],
                           dim=1).norm(dim=1) * 0.5
        for _nm, _uv in (("uvs", uvs), ("uvs2", uvs2)):
            _t2 = _uv[faces0]
            _e1b, _e2b = _t2[:, 1] - _t2[:, 0], _t2[:, 2] - _t2[:, 0]
            _ua2 = (_e1b[:, 0] * _e2b[:, 1]
                    - _e1b[:, 1] * _e2b[:, 0]).abs() * 0.5
            _ok2 = (_wa2 > 1e-4) & (_ua2 > 1e-12)
            _d2 = (_ua2[_ok2] / _wa2[_ok2]).sqrt()
            print(f"S_SECONDARY_UV: {_nm} density median "
                  f"{float(_d2.median()):.4f} repeats/m", flush=True)
        print("  a set two orders of magnitude below the tiling stream is a "
              "unique unwrap, not CS2's LowPrecisionUv1; selecting it will "
              "sample colour maps with an atlas coordinate", flush=True)

    # --- vPivotPaint (TEXCOORD4) and vFoliageParams (TEXCOORD5) -------
    # Identified BY USE, not by slot: TEXCOORD4 carries
    # VertexPaintBlendParams on environment/blend/complex and PivotPaint on
    # foliage -- the same D3D slot, different data.  See STREAM_IDENTITY.
    _pv = data.get("vpivot")
    if _pv is not None:
        VS_PIVOT = _pv.to(device).float()
        print(f"vPivotPaint (TEXCOORD4, foliage): real stream on "
              f"{int((VS_PIVOT.norm(dim=1) > 0).sum()):,} verts", flush=True)
    else:
        # A zero pivot is the shader's OWN "no level 2" sentinel
        # (foliage_vs_max:273), so this substitute is the reference's
        # documented disabled state, not an invented value -- but it does
        # disable the branch level, so it is printed.
        VS_PIVOT = torch.zeros(len(vertices), 3, device=device)
        VS_SUBST.append(
            "vPivotPaint (TEXCOORD4) absent from the pack -> zeros, which "
            "is the shader's own |pivot|==0 sentinel (foliage_vs_max:273) "
            "and DISABLES the branch level of --foliage-animation. Repack "
            "with fast_pack2.py to supply it")
    _fpm = data.get("vfoliage")
    if _fpm is not None:
        VS_FPARAM = _fpm.to(device).float()
        print(f"vFoliageParams (TEXCOORD5): real stream, mean "
              f"{[round(float(x), 3) for x in VS_FPARAM.mean(0)]}",
              flush=True)
    else:
        VS_FPARAM = vsx.default_foliage_params_stream(len(vertices), device)
        VS_SUBST.append(
            "vFoliageParams (TEXCOORD5) absent from the pack -> ONES. "
            "Zeros would switch off the detail bend (foliage_vs_max:339) "
            "and the branch weight (:285), i.e. would silently disable two "
            "of the three levels of an axis that was explicitly asked for")
    _c1 = data.get("vcolor1")
    if _c1 is not None:
        VS_COLOR1 = _c1.to(device).float()
    else:
        VS_COLOR1 = None      # D_BAKED_LIGHTING_FROM_VERTEX_STREAM = 0

    # --- D_COMPRESSED_NORMALS_AND_TANGENTS ---------------------------
    if args.compressed_normals == "on":
        if not PBR:
            raise SystemExit("--compressed-normals on needs the normal/"
                             "tangent streams (--vertex-normals or a "
                             "fast_pack_normals.py pack)")
        VS_PACKED = vsx.encode_packed_frame(vnormal, vtangent)
        _n2, _t2 = vsx.decode_packed_frame(VS_PACKED)
        # The compressed frame can only represent a tangent IN THE
        # NORMAL'S TANGENT PLANE -- that is what an angle-about-the-normal
        # encoding IS.  Comparing against a raw glTF tangent therefore
        # charges the axis for the input's own non-orthogonality.  Both
        # numbers are printed so the two causes are separated rather than
        # summed into one misleading figure.
        _tin = vtangent[:, :3]
        _dotn = (_tin * vnormal).sum(1, keepdim=True)
        _torth = torch.nn.functional.normalize(_tin - _dotn * vnormal,
                                               dim=1)
        print("D_COMPRESSED_NORMALS_AND_TANGENTS=1: normal max|delta| "
              f"{float((_n2 - vnormal).abs().max()):.6g}; tangent max|delta| "
              f"vs the ORTHOGONALISED input "
              f"{float((_t2[:, :3] - _torth).abs().max()):.6g}, vs the raw "
              f"input {float((_t2[:, :3] - _tin).abs().max()):.6g} "
              f"(input |dot(t,n)| max {float(_dotn.abs().max()):.4g} -- the "
              "gap between the two columns is the input's own "
              "non-orthogonality, not the axis); sign disagreement "
              f"{float((_t2[:, 3] != vtangent[:, 3].sign()).float().mean()):.4%}"
              " -- this is the quantisation the engine ships, not an error",
              flush=True)
        del _tin, _dotn, _torth
        vnormal = _n2
        vtangent = _t2
        del _n2, _t2

    # --- S_PRE_BAKED_VERTEX_ANIMATION ---------------------------------
    if args.prebaked_vertex_animation:
        _tab = None
        if args.prebaked_anim_npz:
            _z = np.load(args.prebaked_anim_npz)
            _tab = (torch.from_numpy(_z["dpos"]).float().to(device),
                    (torch.from_numpy(_z["dnrm"]).float().to(device)
                     if "dnrm" in _z else None))
        elif data.get("morph_dpos") is not None:
            _tab = (data["morph_dpos"].to(device).float(),
                    (data["morph_dnrm"].to(device).float()
                     if data.get("morph_dnrm") is not None else None))
        if _tab is None:
            raise SystemExit(
                "--prebaked-vertex-animation has no table. It needs "
                "--prebaked-anim-npz, or a pack carrying glTF morph "
                "targets (fast_pack2.py writes `morph_dpos`). Refusing to "
                "run the axis as a no-op: a silently inert path reads as "
                "tested")
        if _tab[0].shape[1] != len(vertices):
            raise SystemExit(
                f"pre-baked animation table has {_tab[0].shape[1]:,} "
                f"vertices, the pack has {len(vertices):,}")
        VS_PREBAKE = _tab
        print(f"S_PRE_BAKED_VERTEX_ANIMATION: {_tab[0].shape[0]} keyframes, "
              f"max|dpos| {float(_tab[0].abs().max()):.4g} m", flush=True)

    # --- S_VERTEX_BREAK_ANIMATION -------------------------------------
    # Built lazily so `--vertex-stage-selftest` can enable the axis after
    # startup and still exercise it; without that the self-test would score
    # the axis 0 for a setup reason and report a real path as unreachable.
    def _ensure_break_data():
        global VS_BREAK_CENT, VS_BREAK_SEED, VS_BREAK_P
        if VS_BREAK_CENT is not None:
            return
        _gm = _fam_mask(_FAM_GLASS)
        if not bool(_gm.any()):
            raise SystemExit(
                "--vertex-break-animation found no csgo_glass geometry. "
                "Refusing to enable an axis that would reach nothing")
        # Shard identity.  The reference presumably carries a per-vertex or
        # per-instance shard id; it was not extracted (the glass VS was
        # never decompiled), so shards are the connected components of the
        # glass triangles, approximated here by the 0.5 m spatial cell each
        # glass vertex falls in.  Named as a substitution.
        _cell = (vertices / 0.5).floor().long()
        _key = ((_cell[:, 0] + 1 << 20) + (_cell[:, 1] + 1 << 10)
                + _cell[:, 2])
        _u, _inv = torch.unique(_key, return_inverse=True)
        _cnt = torch.zeros(len(_u), device=device).index_add_(
            0, _inv, torch.ones(len(vertices), device=device))
        _sum = torch.zeros(len(_u), 3, device=device).index_add_(
            0, _inv, vertices)
        VS_BREAK_CENT = (_sum / _cnt[:, None])[_inv]
        VS_BREAK_SEED = (torch.sin(_u.float() * 12.9898) * 43758.5469
                         ).frac()[_inv]
        VS_SUBST.append(
            "S_VERTEX_BREAK_ANIMATION shard identity: the glass vertex "
            "stage was never decompiled, so the per-shard grouping is "
            "0.5 m spatial cells of the csgo_glass geometry, not the "
            "engine's shard id")
        if args.break_shot_pos is None:
            args.break_shot_pos = tuple(
                float(x) for x in vertices[_gm].mean(0))
        if args.break_bounce_floor is None:
            args.break_bounce_floor = float(vertices[_gm][:, 1].min())
        VS_BREAK_P = vsx.BreakParams(
            break_time=args.break_time,
            shot_pos=tuple(args.break_shot_pos),
            shot_size=args.break_shot_size,
            bounce_floor=args.break_bounce_floor)
        print(f"S_VERTEX_BREAK_ANIMATION: {int(_gm.sum()):,} glass verts, "
              f"{len(_u):,} shards, shot at "
              f"{[round(x, 2) for x in args.break_shot_pos]}, floor "
              f"y={args.break_bounce_floor:.2f}", flush=True)

    if args.vertex_break_animation:
        _ensure_break_data()

    # --- csgo_character SKINNING data ---------------------------------
    # Built lazily for the same reason the break data is: --char-selftest
    # can enable the axis after startup and must still exercise it.
    def _ensure_skin_data():
        global VS_SKIN_BI, VS_SKIN_BW, VS_SKIN_BASE, VS_SKIN_NINF
        global VS_SKIN_CURV
        if VS_SKIN_BI is not None:
            return
        _cm = _fam_mask(_FAM_CHARACTER)
        if not bool(_cm.any()):
            raise SystemExit(
                "--char-skinning found no csgo_character geometry. Refusing "
                "to enable an axis that would reach nothing. These are MODEL "
                "materials and this pack is a MAP export -- pass "
                "--char-assign vertexlit (or complex, or all).")
        _n = int(args.char_bones)
        _jts = data.get("joints0")
        _wts = data.get("weights0")
        if _jts is not None and _wts is not None:
            VS_SKIN_BI = _jts.to(device).long()
            VS_SKIN_BW = _wts.to(device).float()
            VS_SKIN_NINF = (VS_SKIN_BW > 0).sum(-1).clamp(min=1).long()
            VS_SKIN_BASE = torch.zeros(len(vertices), dtype=torch.long,
                                       device=device)
            print(f"csgo_character skinning: JOINTS_0/WEIGHTS_0 from the "
                  f"pack, {int(VS_SKIN_BI.max()) + 1} distinct joints",
                  flush=True)
        else:
            VS_SKIN_BI, VS_SKIN_BW, VS_SKIN_BASE, VS_SKIN_NINF = \
                vsx.character_skin_binding(vertices, _n)
            VS_SUBST.append(
                "csgo_character BLENDINDICES/BLENDWEIGHT: the pack carries "
                "no glTF JOINTS_0/WEIGHTS_0 (it is a static map export), so "
                "the binding is DERIVED from the mesh -- bones laid out "
                "along +Y, two influences per vertex, weights the linear "
                "partition between the two bone centres. The attribute "
                "layout, the influence count and the blend are the "
                "reference's; only their VALUES are substituted")
        # S_USE_PER_VERTEX_CURVATURE reads a vertex stream the shipped
        # package names `Curvature` / `flSSSCurvature` and the vertex shader
        # passes straight through to the pixel shader as a lone float
        # varying. A map export has no such stream, so it is derived: the
        # mean angle between a vertex normal and its incident faces'
        # normals, which is a curvature by construction.
        _acc = torch.zeros(len(vertices), device=device)
        _cnt = torch.zeros(len(vertices), device=device)
        _fp = vertices[faces0]
        _fn = torch.cross(_fp[:, 1] - _fp[:, 0], _fp[:, 2] - _fp[:, 0],
                          dim=1)
        _fn = _fn / _fn.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        for _c in range(3):
            _vi = faces0[:, _c]
            _d = 1.0 - (vnormal[_vi] * _fn).sum(-1).clamp(-1, 1)
            _acc.index_add_(0, _vi, _d)
            _cnt.index_add_(0, _vi, torch.ones_like(_d))
        VS_SKIN_CURV = (_acc / _cnt.clamp(min=1)).clamp(0, 1)
        VS_SUBST.append(
            "S_USE_PER_VERTEX_CURVATURE: the `Curvature` vertex stream is "
            "not in a map export, so it is derived as the mean "
            "1 - dot(vertexNormal, faceNormal) over the incident faces. "
            "The SHADER-side use of it -- the g_tDiffuseFalloff lookup -- "
            "is the reference's; the stream is substituted")
        print(f"csgo_character skinning: {int(_cm.sum()):,} character "
              f"verts, {_n} bones, influences "
              f"{sorted(set(VS_SKIN_NINF.tolist()))[:4]}, curvature "
              f"range [{float(VS_SKIN_CURV.min()):.3f}, "
              f"{float(VS_SKIN_CURV.max()):.3f}]", flush=True)

    if args.char_skinning:
        _ensure_skin_data()

    for _s in VS_SUBST:
        print("vertex stage SUBSTITUTION: " + _s, flush=True)

VS_FOLIAGE_P = None
VS_ANIM_P = None
VS_BREAK_P = None
if VS_ON:
    VS_FOLIAGE_P = vsx.FoliageParams(wind_dir=tuple(args.wind_dir),
                                     wind_strength=args.wind_strength,
                                     gust=args.wind_gust)
    VS_ANIM_P = vsx.VertexAnimParams(wind_dir=tuple(args.wind_dir),
                                     wind_strength=args.wind_strength,
                                     gust=args.wind_gust)
    VS_BREAK_P = vsx.BreakParams(break_time=args.break_time,
                                 shot_pos=tuple(args.break_shot_pos or
                                                (0.0, 1.6, 0.0)),
                                 shot_size=args.break_shot_size,
                                 bounce_floor=(args.break_bounce_floor
                                               if args.break_bounce_floor
                                               is not None else 0.0))

# The last per-chunk vertex-stage result, so the tools modes can show what
# the animation actually did rather than re-deriving it.
VS_LAST = {}


def vertex_stage(t):
    """Run every enabled vertex axis at time `t` (seconds).

    Returns (positions, normals, tangents4).  Each axis writes only the
    vertices whose material selects the family that declares it; a vertex
    no axis reaches keeps its packed values, which is the correct result
    for a shader variant with that combo off, not a fallback.
    """
    pos = vertices
    nrm = vnormal if PBR else None
    tan = vtangent if PBR else None
    if not VS_ON:
        return pos, nrm, tan
    disp0 = pos
    bend = None

    if args.char_skinning:
        # csgo_character, FIRST SKINNED PATH IN THIS RENDERER. Runs FIRST so
        # every animation axis below composes on the posed mesh, which is
        # the order the engine has (skinning is upstream of everything).
        _ensure_skin_data()
        global VS_SKIN_PAL
        VS_SKIN_PAL = vsx.character_bone_palette(
            vertices, args.char_bones, t,
            amp=args.char_bone_amp, rate=args.char_bone_rate, device=device)
        p2, n2, t2 = vsx.character_skinning(
            pos,
            nrm if nrm is not None else None,
            tan if tan is not None else None,
            VS_SKIN_BI, VS_SKIN_BW, VS_SKIN_BASE, VS_SKIN_NINF, VS_SKIN_PAL)
        m = _fam_mask(_FAM_CHARACTER).unsqueeze(-1)
        pos = torch.where(m, p2, pos)
        if nrm is not None and n2 is not None:
            nrm = torch.where(m, n2, nrm)
        if tan is not None and t2 is not None:
            tan = torch.where(m, t2, tan)

    if args.prebaked_vertex_animation and VS_PREBAKE is not None:
        m = _fam_mask(_FAM_COMPLEX).unsqueeze(-1)
        p2, n2 = vsx.prebaked_vertex_animation(
            pos, nrm if nrm is not None else torch.zeros_like(pos),
            VS_PREBAKE[0], VS_PREBAKE[1], t, args.prebaked_anim_rate)
        pos = torch.where(m, p2, pos)
        if nrm is not None and VS_PREBAKE[1] is not None:
            nrm = torch.where(m, n2, nrm)

    if args.cs_vertex_animation:
        # The flags==1 transform re-index, transcribed; the buffer's
        # contents are time-varying here and that is the substitution.
        K = 8
        xf = torch.zeros(K, 3, 4, device=device)
        for k in range(K):
            a = args.cs_vertex_animation_rate * t + k * 0.7
            ca, sa = math.cos(a), math.sin(a)
            xf[k, 0, 0] = ca
            xf[k, 2, 0] = sa
            xf[k, 1, 1] = 1.0
            xf[k, 0, 2] = -sa
            xf[k, 2, 2] = ca
            xf[k, 0, 3] = args.cs_vertex_animation_amp * math.sin(a)
            xf[k, 1, 3] = args.cs_vertex_animation_amp * math.sin(a * 1.7)
            xf[k, 2, 3] = args.cs_vertex_animation_amp * math.cos(a)
        n = len(vertices)
        flags = torch.ones(n, dtype=torch.long, device=device)
        bi = (torch.arange(n, device=device) % (K - 2))
        base = torch.zeros(n, dtype=torch.long, device=device)
        p2 = vsx.cs_vertex_animation_instanced(
            pos, bi, flags, base, xf,
            uniform_scale_col=torch.ones(n, device=device))
        m = _fam_mask(_FAM_COMPLEX).unsqueeze(-1)
        pos = torch.where(m, p2, pos)

    if args.foliage_animation:
        m = _fam_mask(_FAM_FOLIAGE)
        i = m.nonzero().squeeze(1)
        if len(i):
            p2, n2, b2, fl = vsx.foliage_animation(
                pos[i],
                nrm[i] if nrm is not None else torch.zeros_like(pos[i]),
                VS_PIVOT[i], VS_FOLIAGE_P, VS_NOISE, t, VS_FPARAM[i])
            pos = pos.index_copy(0, i, p2)
            if nrm is not None:
                nrm = nrm.index_copy(0, i, n2)
            bend = torch.zeros(len(vertices), device=device).index_copy(
                0, i, b2)
            VS_LAST["flutter"] = (i, fl)

    if args.vertex_animation > 0:
        m = _fam_mask(_FAM_FOLIAGE)
        i = m.nonzero().squeeze(1)
        if len(i):
            c0 = vcolor[i]
            p2, n2, t2 = vsx.cs_vertex_animation(
                pos[i],
                nrm[i] if nrm is not None else torch.zeros_like(pos[i]),
                tan[i] if tan is not None else torch.cat(
                    [torch.zeros_like(pos[i]),
                     torch.ones(len(i), 1, device=device)], dim=1),
                c0, VS_ANIM_P, VS_NOISE, t, level=args.vertex_animation)
            pos = pos.index_copy(0, i, p2)
            if nrm is not None:
                nrm = nrm.index_copy(0, i, n2)
            if tan is not None:
                tan = tan.index_copy(0, i, t2)

    if args.vertex_break_animation:
        _ensure_break_data()
        m = _fam_mask(_FAM_GLASS)
        i = m.nonzero().squeeze(1)
        if len(i):
            p2, n2 = vsx.vertex_break_animation(
                pos[i],
                nrm[i] if nrm is not None else torch.zeros_like(pos[i]),
                VS_BREAK_CENT[i], VS_BREAK_SEED[i], VS_BREAK_P, t)
            pos = pos.index_copy(0, i, p2)
            if nrm is not None:
                nrm = nrm.index_copy(0, i, n2)

    VS_LAST["disp"] = (pos - disp0)
    VS_LAST["bend"] = bend
    return pos, nrm, tan


def vertex_stage_uv(uv0_sub, uv1_sub, sec_sub):
    """S_SECONDARY_UV, vertex-attribute side, per-vertex.

    `sec_sub` is the per-face g_nUVSet already gathered to pixels or a
    scalar; `--secondary-uv` overrides it globally.
    """
    if args.secondary_uv == "off":
        return uv0_sub
    if args.secondary_uv == "material":
        sel = (sec_sub * 1 + 1).long()      # pack flag 0/1 -> UV set 1/2
    else:
        sel = torch.full(uv0_sub.shape[:-1], int(args.secondary_uv),
                         dtype=torch.long, device=uv0_sub.device)
    # STREAM SELECTION ONLY. The affine belongs to the PIXEL stage (gt_uv),
    # which reads it per MATERIAL from ext2; supplying one here too applied
    # it on both sides of the interpolation.
    return vsx.secondary_uv(uv0_sub, uv1_sub, sel, None)


if args.vertex_stage_selftest:
    print("vertex-stage self-test on the LOADED geometry "
          f"({len(vertices):,} verts). Every axis must move something at "
          "its default parameters.", flush=True)
    _t = 1.0 / max(args.fps, 1) * 32
    _rows, _fail = [], []

    def _probe(name, fn):
        try:
            v = float(fn())
        except Exception as exc:                     # noqa: BLE001
            v = 0.0
            print(f"  {name}: RAISED {exc!r}", flush=True)
        _rows.append((name, v))
        if not (v > 0):
            _fail.append(name)

    _sv, _sn, _st = vertices, vnormal if PBR else None, vtangent if PBR else None
    for _ax, _set in (
            ("S_FOLIAGE_ANIMATION", dict(foliage_animation=True)),
            ("S_VERTEX_ANIMATION", dict(vertex_animation=1)),
            ("D_CS_VERTEX_ANIMATION", dict(cs_vertex_animation=True)),
            ("S_VERTEX_BREAK_ANIMATION", dict(vertex_break_animation=True)),
            ("S_PRE_BAKED_VERTEX_ANIMATION",
             dict(prebaked_vertex_animation=True))):
        _old = {k: getattr(args, k) for k in _set}
        if _set.get("prebaked_vertex_animation") and VS_PREBAKE is None:
            _rows.append((_ax, float("nan")))
            print(f"  {_ax}: no table supplied; the axis REFUSES to run "
                  "rather than no-op (see --prebaked-anim-npz)", flush=True)
            continue
        for k, v in _set.items():
            setattr(args, k, v)
        _probe(_ax, lambda: (vertex_stage(_t)[0] - _sv).abs().max())
        for k, v in _old.items():
            setattr(args, k, v)
    if PBR:
        _p, _t4 = vsx.decode_packed_frame(
            vsx.encode_packed_frame(vnormal, vtangent))
        _rows.append(("D_COMPRESSED_NORMALS_AND_TANGENTS",
                      float((_p - vnormal).abs().max())))
    # An axis is probed WITH ITSELF ENABLED: the question is "does this
    # path move anything at its default PARAMETERS", not "is the flag on".
    _sec_was = args.secondary_uv
    args.secondary_uv = "2"
    _rows.append(("S_SECONDARY_UV",
                  float((vertex_stage_uv(uvs, uvs2,
                                         torch.ones(len(uvs),
                                                    device=device))
                         - uvs).abs().max())))
    args.secondary_uv = "material"
    _rows.append(("S_SECONDARY_UV (material flag)",
                  float((vertex_stage_uv(uvs, uvs2, face_sec_0.new_ones(
                      len(uvs))) - uvs).abs().max())))
    args.secondary_uv = _sec_was
    # The "vertex UV transform" row is GONE with the axis it exercised
    # (#76). The affine is gt_uv's now and is measured on the PIXEL side;
    # a vertex-stage row here would assert an axis this stage no longer has.
    _rows.append(("S_FOLIAGE_UV_ANIMATION",
                  float((vsx.foliage_uv_animation(
                      uvs, torch.tensor(args.foliage_uv_scroll,
                                        device=device), _t)
                      - uvs).abs().max())))
    for _n1, _v1 in _rows:
        if _v1 == _v1 and not (_v1 > 0) and _n1 not in _fail:
            _fail.append(_n1)
    print(f"{'axis':38s} {'max|delta|':>14s}")
    for _n1, _v1 in _rows:
        print(f"{_n1:38s} {_v1:14.6g}")
    if _fail:
        print("UNREACHABLE AT DEFAULT: " + ", ".join(_fail), flush=True)
    raise SystemExit(1 if _fail else 0)

# --- mip chains --------------------------------------------------------
# nvdiffrast's own dr.texture(filter_mode='linear-mipmap-linear', uv_da=...)
# is the filter we want, but it CANNOT be used here: its `tex` minibatch
# axis is indexed by the uv minibatch (the frame), not per pixel, and this
# renderer selects a texture per PIXEL out of a 283-layer colour array and a
# 569-layer aux array via face_mat / mat_nrm1 / ... There is no per-pixel
# texture-array index in nvdiffrast 0.4.0, and gathering the 300 materials a
# frame touches into a dr.texture minibatch would copy ~1 GB per frame. So
# the FILTER is hand-rolled, but the hard part -- the screen-space UV
# derivatives -- is still nvdiffrast's: dr.rasterize(grad_db=True) plus
# dr.interpolate(rast_db=..., diff_attrs='all') yields exactly the
# (du/dx, du/dy, dv/dx, dv/dy) that dr.texture would consume.
#
# Layout: one flat (N,4) uint8 buffer per array holding every level of every
# layer, so a per-pixel (layer, level, u, v) is ONE gather:
#     idx = OFF[level] + layer*S**2 + v*S + u,   S = S0 >> level
MIP_ON = args.mip == "on"


def _build_mip_flat(tex, srgb, name):
    """(M,S,S,4) uint8 -> (flat (N,4) uint8, offsets, sizes, n_levels)."""
    M, S = tex.shape[0], tex.shape[1]
    n_lev = int(math.log2(S)) + 1
    if args.mip_max_levels:
        n_lev = min(n_lev, args.mip_max_levels)
    levels = [tex]
    for _ in range(1, n_lev):
        cur = levels[-1]
        s = cur.shape[1]
        out = torch.empty((M, s // 2, s // 2, 4), dtype=torch.uint8,
                          device=cur.device)
        # chunk over layers so the float temporary stays ~256 MB
        step = max(1, (1 << 24) // max(s * s, 1))
        for a in range(0, M, step):
            b = min(a + step, M)
            x = cur[a:b].float() / 255.0
            if srgb:
                # a hardware sRGB sampler filters in LINEAR space; averaging
                # the stored bytes darkens every downsample
                x = torch.cat([x[..., :3].clamp(min=0) ** 2.2, x[..., 3:]], -1)
            x = torch.nn.functional.avg_pool2d(
                x.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
            if srgb:
                x = torch.cat([x[..., :3].clamp(min=0) ** (1 / 2.2),
                               x[..., 3:]], -1)
            out[a:b] = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
        levels.append(out)
    offs, o = [], 0
    for t in levels:
        offs.append(o)
        o += t.shape[0] * t.shape[1] * t.shape[2]
    flat = torch.empty((o, 4), dtype=torch.uint8, device=tex.device)
    for t, off in zip(levels, offs):
        n = t.shape[0] * t.shape[1] * t.shape[2]
        flat[off:off + n] = t.reshape(n, 4)
    del levels
    print(f"mip chain [{name}] {M} layers @{S}^2 -> {n_lev} levels, "
          f"{o * 4 / 1e9:.3f} GB flat (+{(o / (M * S * S) - 1) * 100:.0f}%)",
          flush=True)
    return (flat, torch.tensor(offs, device=tex.device, dtype=torch.long),
            torch.tensor([S >> i for i in range(n_lev)],
                         device=tex.device, dtype=torch.long), n_lev)


TEX_MIP = AUX_MIP = FAM_MIP = None
if MIP_ON:
    if args.mip_color == "on":
        TEX_MIP = _build_mip_flat(textures, args.mip_gamma == "linear",
                                  "colour")
        # level 0 of the flat chain IS the original array, so keeping the
        # (M,S,S,4) copy resident would make the feature cost +133% VRAM
        # instead of the +33% a mip chain actually costs
        textures = None
    if args.mip_aux == "on" and AUX is not None:
        AUX_MIP = _build_mip_flat(AUX, False, "aux")
        AUX = None
    # The family array gets a chain on the same switch as aux. Without it
    # sample_fam is the one fetch in the frame with no mip, at 8x the
    # edge of the arrays beside it -- see sample_fam's docstring.
    if args.mip_aux == "on" and FAMTEX is not None and len(FAMTEX):
        FAM_MIP = _build_mip_flat(FAMTEX, False, "fam")
    torch.cuda.empty_cache()

MIP_DIAG_DONE = []
ANISO_C = max(1, args.max_aniso)
ANISO_A = max(1, args.max_aniso_aux or args.max_aniso)


def mip_taps(uvda, s0, n_taps, n_lev):
    """Footprint -> (per-pixel LOD, list of uv offsets along the major axis).

    uvda is nvdiffrast's (du/dx, du/dy, dv/dx, dv/dy) in PIXEL units, so
    px/py are the texel-space footprint vectors of one screen pixel. The
    OpenGL rule: N = clamp(Pmax/Pmin, 1, maxAniso) taps of width Pmax/N
    spread over the major axis; N = 1 degenerates to isotropic trilinear
    with lod = log2(Pmax), i.e. the standard `rho`.
    """
    px = torch.stack([uvda[..., 0], uvda[..., 2]], -1) * s0
    py = torch.stack([uvda[..., 1], uvda[..., 3]], -1) * s0
    lx = px.norm(dim=-1)
    ly = py.norm(dim=-1)
    pmax = torch.maximum(lx, ly).clamp(min=1e-8)
    pmin = torch.minimum(lx, ly).clamp(min=1e-8)
    n = (pmax / pmin).clamp(1.0, float(n_taps))
    lod = (torch.log2(pmax / n) + args.mip_bias).clamp(0.0, n_lev - 1.0)
    if n_taps == 1:
        return lod, None
    # full footprint extent along the major axis, back in UV units
    major = torch.where((ly > lx).unsqueeze(-1), py, px) / s0
    return lod, major


def _mip_report(uv, uvda, fg):
    """Machine-check nvdiffrast's derivative UNITS against a finite
    difference of the interpolated UV. If uv_da were in NDC rather than
    pixel units every LOD would be off by log2(W/2) ~ 8.9 levels, i.e. the
    whole frame would collapse to the 1x1 mip -- so this ratio is the one
    number the entire feature rests on."""
    with torch.no_grad():
        fdx = (uv[:, :, 1:, 0] - uv[:, :, :-1, 0]).abs()
        mx = fg[:, :, 1:] & fg[:, :, :-1]
        ax = uvda[:, :, :-1, 0].abs()
        sel = mx & (fdx > 1e-7) & (fdx < 1e-2)      # drop UV seams
        r = (ax[sel] / fdx[sel])
        lod, _ = mip_taps(uvda, T, 1, TEX_MIP[3] if TEX_MIP is not None else 11)
        lv = lod[fg]
        q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=device)
        print("MIP DIAG  analytic|du/dx| / finite-difference|du/dx|: "
              f"median {float(r.median()):.4f} "
              f"p05 {float(r.quantile(0.05)):.4f} "
              f"p95 {float(r.quantile(0.95)):.4f}  (1.0 = pixel units)",
              flush=True)
        print("MIP DIAG  isotropic LOD over covered pixels, quantiles "
              f"{[round(float(x), 2) for x in lv.quantile(q)]}  "
              f"max {float(lv.max()):.2f}; "
              f"{float((lv > 0.5).float().mean()):.1%} of pixels want a "
              f"mip coarser than level 0", flush=True)
        px = torch.stack([uvda[..., 0], uvda[..., 2]], -1) * T
        py = torch.stack([uvda[..., 1], uvda[..., 3]], -1) * T
        an = (torch.maximum(px.norm(dim=-1), py.norm(dim=-1))
              / torch.minimum(px.norm(dim=-1),
                              py.norm(dim=-1)).clamp(min=1e-8))[fg]
        print("MIP DIAG  anisotropy Pmax/Pmin quantiles "
              f"{[round(float(x), 2) for x in an.quantile(q)]}", flush=True)


def _mip_bilinear(mip, uv, layer, lvl):
    flat, offs, sizes, _ = mip
    s = sizes[lvl]                                   # (B,H,W) int64
    sf = s.float()
    fu_ = uv[..., 0] % 1.0
    fv_ = uv[..., 1] % 1.0
    if args.mip_uvconv == "legacy":
        u, v = fu_ * (sf - 1), fv_ * (sf - 1)
    else:
        u, v = fu_ * sf - 0.5, fv_ * sf - 0.5
    u0f, v0f = u.floor(), v.floor()
    fu = (u - u0f).unsqueeze(-1)
    fv = (v - v0f).unsqueeze(-1)
    u0 = u0f.long() % s
    v0 = v0f.long() % s
    u1 = (u0 + 1) % s
    v1 = (v0 + 1) % s
    base = offs[lvl] + layer * s * s
    r0 = base + v0 * s
    r1 = base + v1 * s
    c00 = flat[r0 + u0].float()
    c01 = flat[r0 + u1].float()
    c10 = flat[r1 + u0].float()
    c11 = flat[r1 + u1].float()
    return ((c00 * (1 - fu) + c01 * fu) * (1 - fv)
            + (c10 * (1 - fu) + c11 * fu) * fv) / 255.0


def _mip_trilinear(mip, uv, layer, lod):
    """Two mip levels blended -- OR one, selected nearest.

    r_texturefilteringquality tier 0 is 'Bilinear' and tier 1 is
    'Trilinear' (matforceaniso0 / matforceaniso1 in the engine's own UI
    resource), and the difference between them is exactly this blend.
    That is not a cosmetic detail: it is the reason tier 1 measures
    BLURRIER than tier 0 and correlates only 0.58-0.65 with the aniso
    tiers above it. An implementation that modelled the tier ladder as
    anisotropy alone could not produce a blur step at all, and would have
    had to explain tier 1 as a weak aniso setting -- which is the wrong
    mechanism on the wrong axis.

    VK_SAMPLER_MIPMAP_MODE_NEAREST picks level `ceil(lod + 0.5) - 1`,
    i.e. floor(lod + 0.5) for the non-degenerate case."""
    n_lev = mip[3]
    if MIP_FILTER == "point":
        i = (lod + 0.5).floor().long().clamp(0, n_lev - 1)
        return _mip_bilinear(mip, uv, layer, i)
    l0f = lod.floor()
    frac = (lod - l0f).unsqueeze(-1)
    i0 = l0f.long().clamp(0, n_lev - 1)
    i1 = (i0 + 1).clamp(max=n_lev - 1)
    c0 = _mip_bilinear(mip, uv, layer, i0)
    c1 = _mip_bilinear(mip, uv, layer, i1)
    return c0 * (1 - frac) + c1 * frac


def _mip_fetch(mip, uv, layer, taps, n_taps):
    lod, major = taps
    if n_taps == 1 or major is None:
        return _mip_trilinear(mip, uv, layer, lod)
    acc = None
    for i in range(n_taps):
        t = (i + 0.5) / n_taps - 0.5
        c = _mip_trilinear(mip, uv + major * t, layer, lod)
        acc = c if acc is None else acc + c
    return acc / n_taps


# --- spatial clusters (sort L0 faces once; LOD levels inherit the order) --
centroids = vertices[faces0].mean(dim=1)
cell = (centroids / CLUSTER_METERS).floor().long()
cell_key = ((cell[:, 0] + 2048) * 4096 + (cell[:, 1] + 2048)) * 4096 \
    + cell[:, 2] + 2048
order = torch.argsort(cell_key)
faces0 = faces0[order]
face_mat0 = face_mat0[order]
face_normals0 = face_normals0[order]
face_mode0 = face_mode0[order]
face_cut0 = face_cut0[order]
face_mat2_0 = face_mat2_0[order]
face_has2_0 = face_has2_0[order]
face_sec_0 = face_sec_0[order]
face_params_0 = face_params_0[order]
face_matid0 = face_matid0[order]
cell_key = cell_key[order]
unique_keys, counts0 = torch.unique_consecutive(cell_key, return_counts=True)
C = len(unique_keys)
cluster_of_face = torch.repeat_interleave(
    torch.arange(C, device=device), counts0)

fv = vertices[faces0]
fmin, fmax = fv.min(dim=1).values, fv.max(dim=1).values
cmin = torch.full((C, 3), float("inf"), device=device)
cmax = torch.full((C, 3), float("-inf"), device=device)
cmin.scatter_reduce_(0, cluster_of_face[:, None].expand(-1, 3), fmin,
                     reduce="amin", include_self=True)
cmax.scatter_reduce_(0, cluster_of_face[:, None].expand(-1, 3), fmax,
                     reduce="amax", include_self=True)
centers = (cmin + cmax) * 0.5
radii = ((cmax - cmin) * 0.5).norm(dim=1)
centers_h = torch.cat([centers, torch.ones_like(centers[:, :1])], dim=1)
corner_mask = torch.tensor(
    [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
    dtype=torch.float32, device=device)
corners = cmin[:, None, :] + (cmax - cmin)[:, None, :] * corner_mask[None]
corners_h = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=-1)
print(f"{C} clusters, mean {len(faces0)/C:.0f} tris/cluster", flush=True)


# --- LOD levels: index-buffer-only vertex clustering ---------------------
# THE PER-VERTEX MATERIAL SIGNATURE the LOD weld clusters on (#88).
# amin over the incident faces, not last-write-wins: the label has to be
# the same on every run and every machine or two levels of the same map
# stop being comparable.
_VMAT_SIG = None


def _vmat_signature():
    """Which material a vertex belongs to, for the LOD cluster key.

    A vertex on a material seam gets ONE of its materials, deterministically
    the lowest. That is conservative in the right direction: it can only
    keep two vertices apart that might have merged, never merge two that
    must not.
    """
    global _VMAT_SIG
    if _VMAT_SIG is None:
        # MATID_OF_FACE, not face_matid0. face_matid0 is the pack's array
        # ONLY under PBR or --matid-dump and is a ZERO ARRAY otherwise --
        # this file already carries the postmortem ("a plausible, non-zero,
        # wrong one, which is worse"). Keying the weld on it would make the
        # whole fix silently inert on a bare-flags run: every vertex would
        # read material 0, the key would collapse to position, and the
        # phantom textures would come back with nothing in the log to say
        # so. Dead-by-construction is the failure this lane keeps finding;
        # it does not get to happen inside its own fix.
        if MATID_OF_FACE is None:
            print("⚠️  LOD weld: this pack carries no `face_matid`, so the "
                  "cluster key CANNOT separate materials and vertices of "
                  "different surfaces will weld together, putting one "
                  "surface's UV on another's faces at distance (#88). "
                  "The level is still built -- stated, not silently "
                  "degraded. Rebuild the pack with fast_pack2.py.",
                  flush=True)
            _VMAT_SIG = torch.zeros(len(vertices), dtype=torch.long,
                                    device=device)
            return _VMAT_SIG
        v = torch.full((len(vertices),), 2 ** 40, dtype=torch.long,
                       device=device)
        v.scatter_reduce_(0, faces0.reshape(-1).long(),
                          MATID_OF_FACE.long().repeat_interleave(3),
                          reduce="amin", include_self=True)
        _VMAT_SIG = v
        print(f"LOD weld keys on (position cell, material): "
              f"{int(torch.unique(v).numel()):,} distinct per-vertex "
              f"material labels over {len(vertices):,} vertices. Vertices "
              f"of different materials can no longer weld to one "
              f"representative, which is what put another surface's UV on "
              f"a distant wall (#88).", flush=True)
    return _VMAT_SIG


_UV_DENS = None
_UV_FALLBACK_NOTE = []


def _uv_density():
    """Per-vertex texture-UV density, in uv units per metre.

    sqrt(uv_area / world_area) per face -- the same quantity and the same
    code shape the lightmap band machinery uses to identify streams
    (:5459-5467), computed here on the TEXTURE uvs because texture-UV
    error is what the weld introduces.

    MAX over the incident faces where a vertex touches faces of differing
    density. That is the CONSERVATIVE direction and it is a choice, so it
    is stated: the max gives the smallest tolerance, hence the least
    welding, hence correctness first. Vertices whose faces carry no usable
    UV area get 0 and are handled by the caller as a stated fallback.
    """
    global _UV_DENS
    if _UV_DENS is None:
        _p3 = vertices[faces0]
        _t3 = uvs[faces0]
        _wa = torch.cross(_p3[:, 1] - _p3[:, 0], _p3[:, 2] - _p3[:, 0],
                          dim=1).norm(dim=1) * 0.5
        _e1, _e2 = _t3[:, 1] - _t3[:, 0], _t3[:, 2] - _t3[:, 0]
        _ua = (_e1[:, 0] * _e2[:, 1] - _e1[:, 1] * _e2[:, 0]).abs() * 0.5
        _ok = (_wa > 1e-9) & (_ua > 1e-16)
        _d = torch.where(_ok, (_ua / _wa.clamp(min=1e-12)).sqrt(),
                         torch.zeros_like(_wa))
        v = torch.zeros(len(vertices), device=device)
        v.scatter_reduce_(0, faces0.reshape(-1).long(),
                          _d.repeat_interleave(3), reduce="amax",
                          include_self=True)
        _UV_DENS = v
    return _UV_DENS


def build_level(cell_size):
    # THE WELD CARRIES UVs, WHICH IS WHY IT MUST NOT CROSS MATERIALS (#88).
    # OWNER-OBSERVED on combat_stride1: arbitrary striped awning/rug
    # textures across distant stone walls, flickering, gone on approach.
    # This function welds vertices to one representative per cell and
    # re-points faces through it; the face keeps its OWN material but its
    # vertices become the representative's, so the representative's UV is
    # what gets sampled. Measured at the visibly wrong pixels: 99.0% carry
    # the SAME material id in both levels -- the material was never wrong,
    # the texture COORDINATE was, because it came from another surface.
    #
    # The material term is part of the KEY, not a filter afterwards, and it
    # is FREE: a face's own three vertices share that face's material, so
    # this can never split a face and can never change what degenerates.
    # MEASURED on de_mirage at the shipped cells -- clusters 325,539 ->
    # 352,250 (+8.2%) and 102,851 -> 128,467 (+24.9%), surviving triangles
    # 548,808 -> 548,808 and 193,639 -> 193,639, +0.0% at both. The
    # decimation is untouched; only cross-surface merges are removed.
    #
    # --lod-uv-cell already existed for this and was DEFAULTED OFF, which
    # is why the defect shipped: the guard was written and never armed. It
    # is kept and now composes with this, because clustering on uv cells
    # ALSO needs a float and this does not.
    grid = (vertices / cell_size).floor().long()
    _vm = _vmat_signature()
    key = ((grid[:, 0] + 8192) * 16384 + (grid[:, 1] + 8192)) * 16384 \
        + grid[:, 2] + 8192
    if args.lod_weld == "position":
        uniq, inverse = torch.unique(key, return_inverse=True)
    else:
        # THE WELD MAY NOT INTRODUCE MORE TEXTURE-SPACE ERROR THAN THE
        # POSITION-SPACE ERROR IT ALREADY ACCEPTS (#88).
        #
        # Two vertices in one position cell may weld only if
        #     |d(uv)| <= (local UV density) * (cell size)
        # Both operands are READ, neither is chosen: the density is the
        # per-face sqrt(uv_area/world_area) the lightmap band machinery
        # already computes, and the cell is the LOD parameter that already
        # exists. There is NO slack factor, because the bound is not a
        # threshold someone liked -- it is the equality of the two error
        # classes the weld introduces, which is a dimensional identity.
        # A weld already moves a vertex up to `cell` metres; density
        # converts that same displacement into uv units; allowing more
        # texture error than that would be accepting an error the position
        # weld itself does not accept.
        #
        # Quantising each vertex's uv on ITS OWN derived grid and joining
        # that to the cluster key is the implementation: two vertices in
        # the same position cell whose uvs differ by more than a tolerance
        # land in different uv bins and cannot merge.
        _dv = (_uv_density() if args.lod_weld == "derived"
               else torch.zeros(len(vertices), device=device))
        _tol = _dv * float(cell_size)
        _bad = ~(_tol > 0)
        # STATED FALLBACK, never silent: a vertex whose faces carry no
        # usable uv area has no density, so the derived bound does not
        # exist for it and it welds on position and material alone -- the
        # pre-#88 behaviour for those vertices only, reported by count.
        _uq = torch.zeros(len(vertices), 2, dtype=torch.long, device=device)
        _safe = _tol.clamp(min=1e-20).unsqueeze(-1)
        _uq = torch.where(_bad.unsqueeze(-1),
                          torch.zeros_like(_uq),
                          (uvs[:, :2] / _safe).floor().long())
        if (args.lod_weld == "derived" and bool(_bad.any())
                and not _UV_FALLBACK_NOTE):
            _UV_FALLBACK_NOTE.append(1)
            print(f"  LOD weld: {int(_bad.sum()):,} of {len(vertices):,} "
                  f"vertices carry no usable UV area, so the derived uv "
                  f"tolerance is undefined for them and they weld on "
                  f"position+material alone. Stated, not silent: those "
                  f"vertices keep the pre-#88 texture-bleed exposure.",
                  flush=True)
        uniq, inverse = torch.unique(
            torch.stack([key, _vm, _uq[:, 0], _uq[:, 1]], dim=1),
            dim=0, return_inverse=True)
    rep = torch.full((len(uniq),), len(vertices),
                     dtype=torch.long, device=device)
    rep.scatter_reduce_(0, inverse, torch.arange(len(vertices), device=device),
                        reduce="amin", include_self=True)
    vmap = rep[inverse]
    lf = vmap[faces0]
    keep = ((lf[:, 0] != lf[:, 1]) & (lf[:, 1] != lf[:, 2])
            & (lf[:, 0] != lf[:, 2]))
    # THE PRICE OF CORRECTNESS, MEASURED AND PRINTED EVERY RUN (#88).
    # Splitting clusters by material keeps faces that would otherwise have
    # collapsed, so the cost is real triangles. It is stated here rather
    # than in a commit message: a cost that lives in a ruling nobody
    # re-reads is a cost nobody can revisit, and this one is the whole
    # argument for keeping --lod-weld-position-only available.
    # The counterfactual is BUILT, not estimated -- one extra unique over
    # the scalar key at startup, which is what makes the number honest.
    if args.lod_weld != "position":
        _u0, _i0 = torch.unique(key, return_inverse=True)
        _r0 = torch.full((len(_u0),), len(vertices), dtype=torch.long,
                         device=device)
        _r0.scatter_reduce_(0, _i0,
                            torch.arange(len(vertices), device=device),
                            reduce="amin", include_self=True)
        _lf0 = _r0[_i0][faces0]
        _k0 = int(((_lf0[:, 0] != _lf0[:, 1]) & (_lf0[:, 1] != _lf0[:, 2])
                   & (_lf0[:, 0] != _lf0[:, 2])).sum())
        _k1 = int(keep.sum())
        print(f"  material-split welding at cell {cell_size}m: "
              f"{_k1 - _k0:+,} triangles "
              f"({100.0 * (_k1 - _k0) / max(_k0, 1):+.1f}%) against the "
              f"position-only weld ({_k0:,} -> {_k1:,}); clusters "
              f"{len(_u0):,} -> {len(uniq):,}. That is the price of NOT "
              f"putting one surface's UV on another's faces at distance; "
              f"--lod-weld-position-only buys it back and reinstates the "
              f"defect.", flush=True)
    counts_l = torch.zeros(C, dtype=torch.long, device=device)
    counts_l.scatter_add_(0, cluster_of_face[keep],
                          torch.ones(int(keep.sum()), dtype=torch.long,
                                     device=device))
    return (lf[keep], face_mat0[keep], face_normals0[keep], counts_l,
            face_mode0[keep], face_cut0[keep], face_mat2_0[keep],
            face_has2_0[keep], face_sec_0[keep], face_params_0[keep],
            face_matid0[keep])


levels = [(faces0, face_mat0, face_normals0, counts0,
           face_mode0, face_cut0, face_mat2_0, face_has2_0,
           face_sec_0, face_params_0, face_matid0)]
if not args.no_lod and args.lod_uv_cell > 0:
    # DEPRECATED WITH REASON, and IGNORED rather than obeyed (#88).
    # --lod-uv-cell was the fitted-float mitigation for the texture-bleed
    # weld. It is superseded by the DERIVED tolerance in build_level --
    # |d(uv)| <= density * cell -- which needs no chosen number and is
    # always on. Warned and ignored rather than refused, so an existing
    # config keeps running; ignored rather than composed, because two
    # rules for one decision is how a value silently wins.
    print(f"⚠️  --lod-uv-cell {args.lod_uv_cell} is DEPRECATED and IGNORED. "
          f"The uv-disagreement weld is now bounded by a DERIVED tolerance "
          f"(local uv density x cell size), which is read rather than "
          f"chosen and is active on every run. Passing a fitted cell can "
          f"only make the bound differ from the position error the weld "
          f"already accepts. Remove the flag.", flush=True)
if not args.no_lod:
    for cell_size in LOD_CELLS:
        with stage("startup.lod_build"):
            levels.append(build_level(cell_size))
        print(f"LOD cell {cell_size}m: {len(levels[-1][0]):,} tris "
              f"({len(levels[-1][0])/len(faces0):.0%})"
              , flush=True)
level_faces = [lv[0] for lv in levels]
level_mat = [lv[1] for lv in levels]
level_norm = [lv[2] for lv in levels]
level_mode = [lv[4] for lv in levels]
level_cut = [lv[5] for lv in levels]
level_mat2 = [lv[6] for lv in levels]
level_has2 = [lv[7] for lv in levels]
level_sec = [lv[8] for lv in levels]
level_params = [lv[9] for lv in levels]
level_matid = [lv[10] for lv in levels]
level_counts = torch.stack([lv[3] for lv in levels])          # (L, C)
level_begin = level_counts.cumsum(dim=1) - level_counts       # (L, C)
n_levels = len(levels)

# --- occluder set: big opaque tris rasterized for the Hi-Z depth pass ----
e1 = vertices[faces0[:, 1]] - vertices[faces0[:, 0]]
e2 = vertices[faces0[:, 2]] - vertices[faces0[:, 0]]
area = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1)
_occ = faces0[area > OCCLUDER_MIN_AREA]
occ_used, _occ_remap = torch.unique(_occ.reshape(-1), return_inverse=True)
occluder_faces = _occ_remap.reshape(-1, 3).int().contiguous()
print(f"occluders: {len(occluder_faces):,} tris, {len(occ_used):,} verts "
      f"(area > {OCCLUDER_MIN_AREA} m^2)", flush=True)

# --- cameras -------------------------------------------------------------
# A SESSION is the corpus's own shape: ten agents in one round share the map,
# the pack, the LOD levels and the baked lighting, and differ only in where
# they looked. Today that is ten processes, and the measured cost of a process
# is 8.2 s of pack load + LOD build + warm-up render that sits outside BOTH
# published fps numbers -- ~82 s per session thrown away. --session renders
# them all from one load.
#
# The agents' pose lists are CONCATENATED, and NO BATCH EVER SPANS TWO AGENTS.
# That is not tidiness, it is the difference between right and wrong pixels,
# and it was measured rather than assumed. CHUNK-aligning the agent boundaries
# is NOT enough: with --batch 512, agent a00's own 96 frames came out
# max|delta| 153, mean 4.27, 51.8% of pixels different from rendering that
# agent alone -- purely because eight other agents' poses shared the batch. The
# same agent at batch 96 against batch 32 is max|delta| 0, so the renderer is
# chunk-invariant; something in the batch is global in a way the chunk loop is
# not, and the term has not been identified. Until it is, the safe contract is
# batches that contain one agent, which measures max|delta| 0 against the
# single-agent render AND runs faster (0.94x pure-render against 0.76x).
# =======================================================================
# The scope trigger. Two sources; whichever drove a frame PRINTS.
#
#   demo-fov  the .dem camera json carries `fov` per tick. Measured over
#             test_demo.dem: p4 {0.0: 3128, 15.0: 97, 40.0: 223, 90.0: 551}
#             and p1 {0.0: 54806, 40.0: 948, 90.0: 2216}, and every non-90
#             tick holds an SSG 08.
#
#             ⚠️ fov == 0.0 IS MISSING DATA AND IS THE MAJORITY VALUE (78%
#             and 95%). Read as a FOV it gives tan(0/2)=0 and an infinite
#             magnification; read as "not 90, therefore scoped" it scopes
#             most of the corpus. Counted and reported absent.
#
#   flag      --scope-flags, for corpora whose state has no FOV at all;
#             cs1k does not. The magnification is then a DEFAULT.
# =======================================================================
SCOPE_FOV = {}
SCOPE_FLAG = {}
# frame index -> demo tick, the join --smoke demo needs. It lives HERE, above
# _install_scope_state's call site, because it used to be defined 600 lines
# below it: the module ran top to bottom and raised NameError on the first
# render. ast.parse cannot see that -- an undefined name is a runtime error
# -- which is why "it parses" was not evidence and a render was.
TICK_OF_FRAME = {}
_SCOPE_SOURCE_PRINTED = False


def scope_state_from_rows(rows, base_fov=90.0):
    """(fov_by_frame, stats) from camera rows carrying `fov`."""
    fov, n_missing, n_scoped = {}, 0, 0
    for i, r in enumerate(rows):
        v = r.get("fov")
        if v is None or float(v) <= 0.0:
            n_missing += 1
            continue
        v = float(v)
        fov[i] = v
        if v < base_fov:
            n_scoped += 1
    return fov, dict(n=len(rows), missing=n_missing, scoped=n_scoped)


def scope_for_frame(idx, base_fov=90.0):
    """(scoped, fov, source) for one frame. Never guesses."""
    global _SCOPE_SOURCE_PRINTED
    src, scoped, fov = "none", False, base_fov
    if idx in SCOPE_FLAG:
        src, scoped = "flag", bool(SCOPE_FLAG[idx])
        fov = float(SCOPE_FOV.get(idx, 40.0))
    elif idx in SCOPE_FOV:
        fov = float(SCOPE_FOV[idx])
        src, scoped = "demo-fov", 0.0 < fov < base_fov
    if not _SCOPE_SOURCE_PRINTED:
        _SCOPE_SOURCE_PRINTED = True
        print("[scope] frame %d trigger: %s (fov %.1f, scoped=%s)."
              % (idx, src, fov, scoped))
        if src == "none":
            print("[scope]   no FOV and no class flag reached this frame, "
                  "so the scope pass does NOT run.")
        elif src == "flag":
            print("[scope]   driven by the class label; the %.1f it "
                  "magnifies by is a DEFAULT." % fov)
    return scoped, fov, src


def _load_agent_rows(camera_json, tick_begin, tick_end):
    """Subsample a camera path to --fps, at the demo's OWN tick rate.

    The stride used to be `128 // args.fps` with 128 written in. That is
    right for the 517 camera paths that declare tick_rate 128 and silently
    2x too coarse for the 49 that declare 64 -- the demoparser2-test pair
    and the three probe paths among them. Rendering a 64-tick path at
    --fps 64 asked for every tick and got every OTHER one, so the clip
    played at double speed with half the game state never sampled. It cost
    16 of the 31 weapon_fire events in the combat clip that shipped.

    The rate is DECLARED in the camera json and was simply never read.
    Reading it leaves all 128-tick paths on the identical stride they had
    (128 // fps) and fixes only the paths that say otherwise. Where the
    key is absent (62 older captures) the historical 128 stands in, and
    says so in the banner rather than passing as read.
    """
    _doc = json.load(open(camera_json))
    _t = _doc["ticks"]
    _rate = _doc.get("tick_rate")
    _src = "declared by %s" % os.path.basename(camera_json)
    if not _rate:
        _rate, _src = 128, "ASSUMED (camera json declares no tick_rate)"
    _step = max(1, int(_rate) // max(args.fps, 1))
    _r = [r for r in _t if tick_begin <= r["tick"] < tick_end]
    _rows = [r for r in _r if (r["tick"] - tick_begin) % _step == 0]
    # Print it: a stride >1 drops game state, and the only thing worse
    # than dropping it is dropping it quietly. #34 -- uncertainty prints
    # at runtime, it does not live in a comment.
    print("[ticks] tick_rate %d (%s) / --fps %d -> stride %d: %d of %d "
          "ticks in [%d,%d)%s"
          % (_rate, _src, args.fps, _step, len(_rows), len(_r),
             tick_begin, tick_end,
             "" if _step == 1 else "  <-- %d ticks NOT sampled; pass "
             "--fps %d for every tick" % (len(_r) - len(_rows), _rate)))
    return _rows, _t


def _install_scope_state(rows):
    """Fill SCOPE_FOV / TICK_OF_FRAME from the camera rows, once.

    TICK_OF_FRAME is what --smoke demo needs to know which grenades are
    alive; it was defined and read and nothing filled it, so every frame
    resolved to tick 0. Both joins come off the same row list.
    """
    fov, st = scope_state_from_rows(rows)
    SCOPE_FOV.update(fov)
    for i, r in enumerate(rows):
        if r.get("tick") is not None:
            TICK_OF_FRAME[i] = int(r["tick"])
    if args.scope_flags:
        _f = json.load(open(args.scope_flags))
        _rows = _f.get("frames", _f) if isinstance(_f, dict) else _f
        for i, v in enumerate(_rows):
            SCOPE_FLAG[i] = bool(v.get("scoped") if isinstance(v, dict) else v)
        print("scope: %d per-frame class flags from %s"
              % (len(SCOPE_FLAG), args.scope_flags), flush=True)
    print("scope: %d/%d frames carry a usable fov (%d scoped, %d MISSING "
          "-- fov==0 is absent data, not a zero-degree fov); "
          "ticks joined for %d frames"
          % (len(fov), st["n"], st["scoped"], st["missing"],
             len(TICK_OF_FRAME)), flush=True)


SESSION = None
if args.session:
    SESSION = json.load(open(args.session))
    _ag = SESSION["agents"] if isinstance(SESSION, dict) else SESSION
    if args.png_dir or not args.benchmark:
        pass
    rows, ROW_AGENT, ROW_FIDX, ROW_PAD, AGENT_TICKS = [], [], [], [], []
    AGENT_SPAN = []
    # PER-AGENT ego exclusion: each agent's OWN camera json carries the
    # recording player's steamid (ego paths). The global read of
    # args.camera_json excluded ONE steamid for the whole session --
    # measured: p1-p9 ego videos each carried their own head at the near
    # plane, and p0's model was hidden from everyone else's view.
    AGENT_EGO = {}
    for _ai, _a in enumerate(_ag):
        try:
            _doc = json.load(open(_a["camera_json"]))
            if isinstance(_doc, dict) and _doc.get("steamid") is not None:
                AGENT_EGO[_ai] = int(_doc["steamid"])
        except Exception:                                    # noqa: BLE001
            pass
        _rr, _tt = _load_agent_rows(_a["camera_json"],
                                    int(_a.get("tick_begin", args.tick_begin)),
                                    int(_a.get("tick_end", args.tick_end)))
        if not _rr:
            raise SystemExit(f"session agent {_ai} ({_a.get('id', _ai)}): no "
                             f"ticks in [{_a.get('tick_begin')}, "
                             f"{_a.get('tick_end')}) of {_a['camera_json']}")
        AGENT_TICKS.append(_tt)
        _n = len(_rr)
        AGENT_SPAN.append((len(rows), _n))
        ROW_AGENT.extend([_ai] * _n)
        ROW_FIDX.extend(range(_n))
        ROW_PAD.extend([False] * _n)
        rows.extend(_rr)
        print(f"session agent {_ai} ({_a.get('id', _ai)}): {_n} frames",
              flush=True)
    print(f"session: {len(_ag)} agents, {len(rows)} poses, ONE pack load, "
          f"batches clamped to agent boundaries", flush=True)
    print(f"session ego exclusion: {len(AGENT_EGO)}/{len(_ag)} agents carry "
          f"a steamid in their camera json (each excludes only its OWN "
          f"subject); paths without one draw the subject by design "
          f"(third-person)", flush=True)
else:
    AGENT_EGO = {}
    rows, _ = _load_agent_rows(args.camera_json, args.tick_begin, args.tick_end)
    ROW_AGENT = [0] * len(rows)
    ROW_FIDX = list(range(len(rows)))
    ROW_PAD = [False] * len(rows)
    print(f"{len(rows)} frames", flush=True)

# --- A2: the firing signal IS on the wire ------------------------------
# `fire` is a per-tick boolean the exporter already writes beside the
# camera, so the muzzle flash does not need a synthesised trigger: it is
# gated on scene state, like every other thing that should be.
#
# The visibility WINDOW is read, not chosen. The AK's flash particle
# declares LIFE_DURATION 0.05 s (initializer [4] -> field 1), so a shot
# at tick T is visible for 0.05 s after it. That matters because
# decimation can drop the exact firing tick: at --fps 32 the step is 4,
# so a 1-tick flag has a 1-in-4 chance of surviving, while a 0.05 s
# window at 128 Hz spans 6.4 ticks and cannot be missed. Using the read
# lifetime is therefore both more correct AND what makes the gate
# robust to decimation -- one constant doing both jobs, and neither
# choice is a tuned number.
# THE TWO JOINS, once, on whichever path built `rows`. TICK_OF_FRAME was
# defined and read by --smoke demo and nothing filled it, so every frame
# resolved to tick 0 and the demo smoke could never be alive; SCOPE_FOV had
# the same shape of hole on the scope side.
_install_scope_state(rows)

if SESSION is not None:
    # per agent, because a fire tick belongs to the agent that fired
    _all_ticks = [r for _t in AGENT_TICKS for r in _t]
    _fire_by_agent = [sorted(r["tick"] for r in _t if r.get("fire"))
                      for _t in AGENT_TICKS]
    _fire_ticks = sorted(t for f in _fire_by_agent for t in f)
else:
    _all_ticks = [r for r in json.load(open(args.camera_json))["ticks"]]
    _fire_ticks = sorted(r["tick"] for r in _all_ticks if r.get("fire"))
_life_ticks = _mz.LIFE_DURATION_S * 128.0
if _fire_ticks:
    import bisect as _bisect
    def _firing(t):
        i = _bisect.bisect_right(_fire_ticks, t)
        return i > 0 and (t - _fire_ticks[i - 1]) <= _life_ticks
    if SESSION is not None:
        import bisect as _bisect2

        def _firing_a(ai, t):
            f = _fire_by_agent[ai]
            i = _bisect2.bisect_right(f, t)
            return i > 0 and (t - f[i - 1]) <= _life_ticks
        FIRE = [bool(_firing_a(ROW_AGENT[i], r["tick"]))
                for i, r in enumerate(rows)]
    else:
        FIRE = [bool(_firing(r["tick"])) for r in rows]
else:
    FIRE = [False] * len(rows)
print(f"firing signal: {len(_fire_ticks)} `fire` ticks on the wire "
      f"({len(_fire_ticks) / max(len(_all_ticks), 1):.2%} of {len(_all_ticks)}); "
      f"{sum(FIRE)} of {len(rows)} rendered frames fall inside a "
      f"{_mz.LIFE_DURATION_S}s flash window (READ from "
      f"uweapon_muzflsh_ak47_primaryflash.vpcf_c)", flush=True)


FLASH_K = math.log(255.0)   # DERIVED: exp(-K) == 1/255, one 8-bit level


def row_flash_alpha(row):
    """Whiteout alpha in [0,1] for this row, or 0.0 when not blind.

    Pure, like row_punch: the report and the composite cannot disagree
    about what was applied.
    """
    if args.flash_whiteout != "on":
        return 0.0
    e = row.get("flash_elapsed_s")
    d = row.get("flash_duration")
    if e is None or d is None or e != e or d != d or float(d) <= 0.0:
        return 0.0
    u = float(e) / float(d)
    if u < 0.0 or u >= 1.0:
        return 0.0
    if args.flash_curve == "linear":
        return max(0.0, 1.0 - u)
    return math.exp(-FLASH_K * u)


def row_punch(row):
    """(d_pitch, d_yaw) degrees this row's punch adds to the view angles.

    Pure: reads the row and nothing else, so view_matrix and the per-frame
    report cannot disagree about what was applied. A missing or null column
    is 0.0 -- absent data, not a zero measurement, and the GAP banner says
    which of the two the run is in.
    """
    if args.view_punch == "none":
        return 0.0, 0.0

    def _f(k):
        v = row.get(k)
        return float(v) if v is not None and v == v else 0.0
    if args.view_punch == "view":
        return _f("view_punch_pitch"), _f("view_punch_yaw")
    if args.view_punch == "aim":
        return _f("aim_punch_pitch"), _f("aim_punch_yaw")
    return (_f("view_punch_pitch") + _f("aim_punch_pitch"),
            _f("view_punch_yaw") + _f("aim_punch_yaw"))


def view_matrix(row):
    eye = np.array([row["y"] * S, (row["eye_z"] + args.eye_dz) * S,
                    row["x"] * S], dtype=np.float32)
    # PUNCH IS ADDED TO THE VIEW ANGLES, which is the position the engine
    # composes it at -- the client adds the punch to the eye angles before
    # building the view, rather than rotating the finished basis. Added
    # here, beside --yaw-offset/--pitch-offset, so all three angle
    # contributions meet in one expression instead of one of them being
    # applied to a matrix somewhere downstream.
    _dp, _dy = row_punch(row)
    yaw = math.radians(row["yaw_degrees"] + args.yaw_offset + _dy)
    pitch = math.radians(row["pitch_degrees"] + args.pitch_offset + _dp)
    f_src = np.array([math.cos(yaw) * math.cos(pitch),
                      math.sin(yaw) * math.cos(pitch),
                      -math.sin(pitch)])
    f = np.array([f_src[1], f_src[2], f_src[0]])
    up = np.array([0.0, 1.0, 0.0])
    zaxis = -f / np.linalg.norm(f)
    xaxis = np.cross(up, zaxis); xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = xaxis, yaxis, zaxis
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def projection(width, height, hfov_deg=None, near=0.05, far=800.0):
    hfov_deg = args.hfov if hfov_deg is None else hfov_deg
    hf = math.radians(hfov_deg) / 2
    right = near * math.tan(hf)
    top = right * height / width
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = near / right
    p[1, 1] = near / top
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)
    p[3, 2] = -1.0
    return p


W, H = args.width * args.supersample, args.height * args.supersample
proj = projection(args.width, args.height)
# The MBOIT depth warp's logNear / logFar. The engine reads them from a
# per-view CB; these are this renderer's OWN projection near/far, i.e. the
# same quantity taken from the same camera rather than a fitted constant.
MBOIT_NEAR, MBOIT_FAR = 0.05, 800.0
# --- VIEW PUNCH: does this path carry it, and did it act? -------------
# Printed BEFORE the matrices are built, because a run whose camera silently
# ignored recoil and one whose camera path never had it are the same picture
# and different facts.
_PUNCH_COLS = ("view_punch_pitch", "view_punch_yaw",
               "aim_punch_pitch", "aim_punch_yaw")
_pc_have = [c for c in _PUNCH_COLS
            if any(r.get(c) is not None for r in rows)]
if not _pc_have:
    print(f"GAP view punch: this camera path carries none of {list(_PUNCH_COLS)} "
          f"(schema is the cs1k ego path, not a demo-derived one), so the "
          f"camera does NOT kick and --view-punch {args.view_punch} changes "
          f"nothing. Recoil needs a path built by "
          f"harness/cs2_demo_camera.py.", flush=True)
elif args.view_punch == "none":
    print(f"view punch: columns {_pc_have} ARE present but --view-punch none, "
          f"so the camera does not kick. This is the pre-existing behaviour "
          f"and it is now a CHOICE rather than a gap.", flush=True)
else:
    _pp = [row_punch(r) for r in rows]
    _nz = [i for i, (a, b) in enumerate(_pp) if abs(a) > 1e-9 or abs(b) > 1e-9]
    _mp = max((abs(a) for a, _ in _pp), default=0.0)
    _my = max((abs(b) for _, b in _pp), default=0.0)
    print(f"view punch: ARM={args.view_punch} over {len(rows)} rows -- "
          f"{len(_nz)} ({100.0 * len(_nz) / max(1, len(rows)):.1f}%) carry a "
          f"nonzero kick, max |pitch| {_mp:.4f} deg, max |yaw| {_my:.4f} deg. "
          f"STATED: which punch the engine composes into the rendered view "
          f"is NOT settled (cs2_demo_camera:20-24); this applies what the "
          f"column name says and `sum` is available as the composition arm.",
          flush=True)
    if not _nz:
        print("view punch: columns present but EVERY row is zero -- the "
              "term is live and the data says nobody fired on this path.",
              flush=True)

# --- FLASHBANG: does this path carry a blind episode? -----------------
if args.flash_whiteout != "on":
    print("flashbang: --flash-whiteout off, so a blinded frame renders "
          "identically to an unblinded one. That is now a CHOICE.",
          flush=True)
elif not any("flash_elapsed_s" in r for r in rows):
    # THE COLUMN IS NOT THERE. loopfam's wording distinction, and it is a
    # real one: "the path carries no column" is a REGENERATION BUG, while
    # "the column carries no value" is a quiet frame. This line used to
    # test `is not None`, which is true of BOTH -- so a correctly
    # regenerated path on which nobody was blinded reported a missing
    # column, i.e. it named the wrong defect.
    print("GAP flashbang: this camera path has no `flash_elapsed_s` COLUMN "
          "at all, so it predates the derivation -- regenerate it with "
          "cs2_demo_camera.py. This is a path defect, NOT a statement "
          "about whether anyone was blinded.", flush=True)
elif not any(r.get("flash_elapsed_s") is not None for r in rows):
    print("flashbang: the `flash_elapsed_s` column IS present and every "
          "row is null -- nobody on this path was blinded. The path is "
          "fine; there is nothing to draw.", flush=True)
else:
    _fb = [row_flash_alpha(r) for r in rows]
    _fn = [a for a in _fb if a > 0.0]
    print(f"flashbang: {len(_fn)} of {len(rows)} rows are inside a blind "
          f"episode, peak alpha {max(_fn, default=0.0):.4f}, curve "
          f"{args.flash_curve} STATED -- the engine's own curve is not read; "
          f"exp decays to 1/255 at the blind duration (K = ln 255, DERIVED "
          f"from output precision, no free parameter).", flush=True)

# AN EMPTY TICK RANGE IS AN INPUT ERROR, NOT A FRAME STATE, and it is the
# one case in this file's zero-geometry family that must NOT be guarded into
# a silent skip. np.stack([]) raises "need at least one array to stack" from
# deep in numpy, which reads like a renderer bug; what it actually means is
# that --tick-begin/--tick-end selected nothing from this camera. Say that,
# with both ranges, so the next person does not go looking in the rasteriser.
if not rows:
    _av = [int(r["tick"]) for r in json.load(open(args.camera_json))["ticks"]]
    raise SystemExit(
        f"--tick-begin {args.tick_begin} --tick-end {args.tick_end} selected "
        f"0 of {len(_av)} rows from {args.camera_json}, whose ticks run "
        f"{min(_av) if _av else '-'}..{max(_av) if _av else '-'}. That is an "
        f"empty SELECTION, not an empty scene -- nothing downstream can "
        f"render a frame that was never asked for, so this refuses here "
        f"instead of raising out of numpy 30,000 lines later.")
mvps = torch.from_numpy(
    np.stack([proj @ view_matrix(r) for r in rows])).to(device)
eyes = torch.tensor([[r["y"] * S, r["eye_z"] * S, r["x"] * S] for r in rows],
                    device=device)
# camera forward per frame, in world space -- ssao_convert_depth needs the
# camera-space Z, which is the projection of (P - eye) onto this axis.
# Taken from the SAME view_matrix() rows the MVP is built from (row 2 of
# the view basis is -forward), so it cannot drift from the raster.
FWD = torch.from_numpy(np.stack(
    [-view_matrix(r)[2, :3] for r in rows])).to(device).float()

if args.frustum_ss:
    # Emitted from the SAME view_matrix()/projection() the raster uses, so
    # it cannot describe a camera the frames were not rendered with. The
    # basis rows ARE the view matrix's rotation; the tangents ARE the
    # projection's, recovered from p[0,0]/p[1,1] rather than from args.hfov,
    # so a future projection change cannot silently desync this dump.
    import json as _json
    _htan = 1.0 / float(proj[0, 0])
    _vtan = 1.0 / float(proj[1, 1])
    # near/far recovered from the matrix, not from a constant defined
    # later in the file: n = p23/(p22-1), f = p23/(p22+1).
    _p22, _p23 = float(proj[2, 2]), float(proj[2, 3])
    _near, _far = _p23 / (_p22 - 1.0), _p23 / (_p22 + 1.0)
    with open(args.frustum_ss, "w") as _fh:
        _fh.write(_json.dumps({
            "schema": "iji/frustum-state-space/v1", "side": "repro",
            "width": args.width, "height": args.height,
            "htan": _htan, "vtan": _vtan, "near": _near, "far": _far,
            "units": "metres (Source units * %r)" % S,
            "axes": "world = (src_y, src_z, src_x) * S",
        }) + "\n")
        for _i, _r in enumerate(rows):
            _V = view_matrix(_r)
            _x, _y, _z = _V[0, :3], _V[1, :3], _V[2, :3]
            _eye = np.array([_r["y"] * S, (_r["eye_z"] + args.eye_dz) * S,
                             _r["x"] * S])
            _fwd = -_z
            _c = [(_eye + _fwd * _d + _x * (_sx * _htan * _d)
                   + _y * (_sy * _vtan * _d)).tolist()
                  for _d in (_near, _far) for _sy in (1, -1) for _sx in (-1, 1)]
            _fh.write(_json.dumps({
                "i": _i, "tick": _r["tick"],
                "eye": _eye.tolist(),
                "basis": {"x": _x.tolist(), "y": _y.tolist(), "z": _z.tolist()},
                "corners": _c,
            }) + "\n")
    print("frustum-ss: wrote %d poses to %s" % (len(rows), args.frustum_ss),
          flush=True)
# Wall-clock seconds per frame, from the capture's own tick numbers.
# S_TEXTURE_ANIMATION's frame index and the g_vTexCoordScrollSpeed
# scroll are both functions of engine time (csgo_complex_vs_max.glsl:260
# and :279), so the renderer needs a real clock rather than a frame
# counter -- a pose set that skips ticks must not run the animation slow.
TIME_S = torch.tensor([float(r["tick"]) / max(args.fps, 1) for r in rows],
                      device=device, dtype=torch.float32)
focal_px = (args.width / 2) / math.tan(math.radians(45.0))
# level d thresholds: cell projects to < ~1.2 px
lod_dist = torch.tensor(
    [0.0] + [c * focal_px / 2.0 for c in LOD_CELLS], device=device)

# Fitted against 48 real CS2 frames by least squares over a sun-direction
# sweep (fit_lighting.py); the assumed el=52 az=215 was anti-correlated
# with the actual lighting and drove a NEGATIVE sun coefficient.
# PER-MAP SUN. Everything below defaulted to de_inferno's READ values,
# which is correct for one map and is the scene's light source ignored on
# the other six. The sidecar makes the sun a property of the pack.
LIGHT_ENV = None
if args.light_env == "auto":
    _le, _le_how = _sidecar(".light_environment.json", "light_environment")
    _le = (_le
           if args.world else None)
    if _le and os.path.exists(_le):
        LIGHT_ENV = json.load(open(_le))
        print(f"light_environment: auto -> {_le}", flush=True)
    else:
        print(f"GAP light_environment: no {_le}. The sun below is "
              f"de_inferno's, READ from that map's entity lump and its "
              f"capture -- on any other map it is a DEFAULT standing in "
              f"for a light source that exists and has not been read.",
              flush=True)
elif args.light_env and args.light_env != "none":
    LIGHT_ENV = json.load(open(args.light_env))
    print(f"light_environment: {args.light_env}", flush=True)

SKY_ENTITY = None
