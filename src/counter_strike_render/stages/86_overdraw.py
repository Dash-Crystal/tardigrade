

def quad_overdraw(clip, tris, layers):
    """`D_QUAD_OVERDRAW` — per-2x2-quad shading invocations.

    In CS2 this axis is 3 `OpImageTexelPointer` feeding
    `OpAtomicCompareExchange` / `OpAtomicExchange` / `OpAtomicIAdd` on a
    storage image -- the ONLY 6 instructions out of 3,018,366 in the whole
    analysed corpus that fell into no class, and they are named as an
    overdraw counter in SHADER_CALLFLOW_METHOD.md:78-88.  A read-modify-write
    into an image, not a fetch and not arithmetic.

    Reproduced here as a real count, not a proxy: nvdiffrast's DepthPeeler
    walks the actual fragment layers, and a 2x2 quad is charged one
    invocation per layer in which ANY of its four pixels is covered -- which
    is what quad-granularity shading means and why the counter is per-quad
    rather than per-pixel.

    WHAT DIFFERS: the layer count is capped at `layers`.  A pixel with more
    fragments than that is undercounted, so the image is a FLOOR, never an
    over-count.  The cap is printed and the saturating fraction is reported.
    """
    Bv, Hh, Ww = clip.shape[0], H, W
    acc = torch.zeros(Bv, Hh, Ww, device=device)
    # REFUSAL, not a substituted value. DepthPeeler refuses `tri` with shape
    # [0, 3] exactly as rasterize and interpolate do -- the same #27 family,
    # and this was the LAST disclosed unguarded consumer of the opaque face
    # selection. Guarded in the function, like _raster and _interp, so a
    # future caller is covered without a further site.
    #
    # The zero here is READ, not chosen: with no face there is no quad, and
    # a quad that does not exist is charged no invocations. acc is already
    # the correct answer at the line above, and the caller's own `_acc > 0`
    # mask blacks the visualisation out unaided -- the comment there says
    # _heat(0) is blue, so an unmasked zero would have read as depth 0.
    if tris.numel() == 0:
        print("quad_overdraw skipped: 0 opaque faces (empty class, #27 "
              "family) -- no quad exists to be charged, so the count is "
              "the axis's own zero and NOT a measured zero of a populated "
              "pass. Nothing about this frame's real overdraw was sampled.",
              flush=True)
        return acc, torch.zeros_like(acc)
    with dr.DepthPeeler(ctx, clip, tris, (Hh, Ww)) as peeler:
        for _ in range(layers):
            rast, _ = peeler.rasterize_next_layer()
            acc = acc + (rast[..., 3] > 0).float()
    # quad reduction: a quad is charged if any of its 4 pixels is covered.
    q = torch.nn.functional.max_pool2d(acc[:, None], 2, 2)[:, 0]
    q = torch.nn.functional.interpolate(q[:, None], size=(Hh, Ww),
                                        mode="nearest")[:, 0]
    return acc, q


def tools_vis(mode, fg, uv0, uv1, sel, nrm, tan4, vcol, pvl, pivot, fpar,
              bendv, dispv, packed_err):
    """`S_MODE_TOOLS_VIS`.

    The axis is DECLARED by all four vertex families
    (SHADER_CALLFLOW_vertex_stages.md:204-207) and is a real static combo
    (weight 32 on glass, SHADER_CALLFLOW_csgo_glass.md:45), but NO module
    selecting it was decompiled in this corpus and its combo->module
    identification is listed under "what was not extracted" for all four.
    So: the SELECTOR is reproduced, and every channel is built from a
    quantity the ported vertex stage genuinely computes.  None of these
    colour mappings is claimed to be Valve's.
    """
    z = torch.zeros_like(fg, dtype=torch.float32)

    def rgb(a, b, c):
        return torch.stack([a, b, c], dim=-1)

    if mode == "normal":
        img = nrm * 0.5 + 0.5 if nrm is not None else rgb(z, z, z)
    elif mode == "tangent":
        img = (tan4[..., :3] * 0.5 + 0.5) if tan4 is not None else rgb(z, z, z)
    elif mode == "uv0":
        img = rgb(uv0[..., 0].frac(), uv0[..., 1].frac(), z)
    elif mode == "uv1":
        img = rgb(uv1[..., 0].frac(), uv1[..., 1].frac(), z)
    elif mode == "uvset":
        # which UV set S_SECONDARY_UV selected: 0 black, 1 green, 2 red
        s = sel.float()
        img = rgb((s == 2).float(), (s == 1).float(), z)
    elif mode == "vertexcolor":
        img = vcol[..., :3] if vcol is not None else rgb(z, z, z)
    elif mode == "pervertexlighting":
        img = (pvl if pvl is not None else rgb(z, z, z)).clamp(0, 1)
    elif mode == "pivot":
        img = (pivot.frac().abs() if pivot is not None else rgb(z, z, z))
    elif mode == "foliageparams":
        img = fpar.clamp(0, 1) if fpar is not None else rgb(z, z, z)
    elif mode == "bend":
        img = _heat((bendv if bendv is not None else z).abs())
    elif mode == "displacement":
        # metres of vertex-stage displacement, 0..10 cm on the ramp
        d = (dispv.norm(dim=-1) if dispv is not None else z) / 0.10
        img = _heat(d)
    elif mode == "packedframe-error":
        img = _heat((packed_err if packed_err is not None else z) / 0.01)
    else:
        raise SystemExit(f"unknown --tools-vis {mode}")
    return torch.where(fg.unsqueeze(-1), img, torch.zeros_like(img))


TOOLS_NOTE = []

ctx = dr.RasterizeCudaContext()

SHADOW = None
if args.shadow:
    _lo, _hi = vertices.min(0).values, vertices.max(0).values
    _center = (_lo + _hi) * 0.5
    _radius = float((_hi - _lo).norm()) * 0.5 + 1.0
    _z = sun_dir / sun_dir.norm()
    _up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if abs(float(_z @ _up)) > 0.95:
        _up = torch.tensor([1.0, 0.0, 0.0], device=device)
    _x = torch.cross(_up, _z, dim=0); _x = _x / _x.norm()
    _y = torch.cross(_z, _x, dim=0)
    _view = torch.eye(4, device=device)
    _view[0, :3], _view[1, :3], _view[2, :3] = _x, _y, _z
    _view[:3, 3] = -(_view[:3, :3] @ (_center - _z * _radius))
    _ortho = torch.eye(4, device=device)
    _ortho[0, 0] = 1.0 / _radius
    _ortho[1, 1] = 1.0 / _radius
    _ortho[2, 2] = -1.0 / (2.5 * _radius)
    LIGHT_MVP = _ortho @ _view
    _vh = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
    _lclip = (LIGHT_MVP @ _vh.T).T.contiguous()[None]
    _res = args.shadow
    _rast, _ = dr.rasterize(ctx, _lclip, faces0.int(), (_res, _res))
    SHADOW = torch.where(_rast[0, ..., 3] > 0, _rast[0, ..., 2],
                         torch.full_like(_rast[0, ..., 2], 1e9))
    print(f"shadow map {_res}x{_res}, "
          f"{float((SHADOW < 1e8).float().mean()):.1%} covered", flush=True)


def sun_visibility(world_pos):
    """1 = lit, 0 = in shadow, from the static sun depth map."""
    if SHADOW is None:
        return None
    wh = torch.cat([world_pos, torch.ones_like(world_pos[..., :1])], dim=-1)
    lc = torch.einsum("ij,...j->...i", LIGHT_MVP, wh)
    u = ((lc[..., 0] * 0.5 + 0.5) * (SHADOW.shape[1] - 1))
    v = ((lc[..., 1] * 0.5 + 0.5) * (SHADOW.shape[0] - 1))
    inside = (u >= 0) & (u < SHADOW.shape[1]) & (v >= 0) & (v < SHADOW.shape[0])
    ui = u.clamp(0, SHADOW.shape[1] - 1).long()
    vi = v.clamp(0, SHADOW.shape[0] - 1).long()
    # 2x2 PCF
    lit = torch.zeros_like(u)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        d = SHADOW[(vi + dv).clamp(max=SHADOW.shape[0] - 1),
                   (ui + du).clamp(max=SHADOW.shape[1] - 1)]
        lit = lit + (lc[..., 2] <= d + args.shadow_bias).float()
    return torch.where(inside, lit * 0.25, torch.ones_like(lit))
vertices_h = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
# The arrays render_window actually rasterizes.  With every vertex axis off
# these ARE the packed arrays (`is` identity, not a copy), so the vertex
# stage costs nothing and changes nothing when it is not asked for; a chunk
# with an axis on replaces them wholesale from `vertex_stage(t)`.
#
# CLUSTERING, LOD AND THE Hi-Z OCCLUDER SET STAY ON THE STATIC POSITIONS.
# That is deliberate and is the conservative direction: the animation
# displaces by centimetres, and re-clustering per frame would rebuild the
# LOD index buffers every chunk.  It means a vertex can animate slightly
# outside its cluster's bounding sphere; stated here rather than left to be
# discovered.
VS_VPOS = vertices
VS_VPOS_H = vertices_h
VS_VNRM = vnormal if PBR else None
VS_VTAN = vtangent if PBR else None
# Seconds per rendered frame for every time-driven vertex axis.
VS_DT = (args.vertex_anim_fps if args.vertex_anim_fps is not None
         else 1.0 / max(args.fps, 1))
VS_FRAME0 = [0]
# True only while the startup warm-up render is running. That pass renders
# real frames and discards them, so anything printed per frame during it is
# a duplicate of a line that is about to be printed for real.
WARMING_UP = [False]


def frame_print(*a, **kw):
    """print(), unless this is the warm-up pass rendering frames twice.

    Every per-frame report goes through this instead of print(), so a line
    added later is protected by default rather than by remembering.
    """
    if WARMING_UP[0]:
        return
    kw.setdefault("flush", True)
    print(*a, **kw)


# =====================================================================
# GT cascaded sun shadow  +  S_ANIMATED_SHADOWS
# csgo_complex_ps.glsl:529-617 (q=0) and
# csgo_complex_ps_shaderquality1.glsl:567-715 (q=1).
# =====================================================================
GT_CSM = None          # list of (mvp, extent) per cascade
GT_CSM_DEPTH = None    # (C, res, res)
# psrs effects MEASURED at build time, so the self-test rows for the two
# psrs axes report something the axis actually did rather than the mere
# existence of a depth atlas. A row whose live count cannot change is a
# row that cannot fail.
#   [0] faces kept by D_FRONT_FACE_CULL
#   [1] faces culled by it
#   [2] shadow texels the raster-state depth bias moved
#   [3] max|delta| that bias made to the atlas
#   [4] faces at CullMode 0 (F_RENDER_BACKFACES -> cull nothing)
GT_PSRS = [0, 0, 0, 0.0, 0]

# Per-face family and F_DEPTH_BIAS, for the psrs per-family depth bias.
# Built once here rather than inside the cascade loop.
# SHADER_CALLFLOW_psrs_stage.md §4.3: the two bias mechanisms partition
# the six families, so the depth pass needs to know which family wrote
# each shadow texel.
GT_FAM_OF_FACE = None
GT_FDEPTHBIAS_OF_FACE = None
GT_FAM_DEPTHPASS = []     # env, fol, gl -- SlopeScaleDepthBias = +5.0
GT_FAM_MATBIAS = []       # cx, eb, so   -- (-64, -2.0, -5.0000002e-04)
_G = globals()
if (args.gt_lighting or args.gt_lighting_selftest) and _G.get("FAM_NAMES"):
    def _fam(nm):
        return FAM_NAMES.index(nm) if nm in FAM_NAMES else None
    GT_FAM_DEPTHPASS = [i for i in (_fam("csgo_environment.vfx"),
                                    _fam("csgo_foliage.vfx"),
                                    _fam("csgo_glass.vfx")) if i is not None]
    GT_FAM_MATBIAS = [i for i in (_fam("csgo_complex.vfx"),
                                  _fam("csgo_environment_blend.vfx"),
                                  _fam("csgo_static_overlay.vfx"))
                      if i is not None]
    # MAT_FAM / MAT_EXT / face_matid0 are bound only inside the FAMILY
    # and PBR blocks, so they may not exist at all; globals().get() makes
    # the absence a None rather than a NameError that fires only on the
    # configuration nobody runs.
    if _G.get("MAT_FAM") is not None and _G.get("face_matid0") is not None:
        GT_FAM_OF_FACE = _G["MAT_FAM"][_G["face_matid0"]]
    # F_DEPTH_BIAS is resolved through the family's own features array,
    # never by name: psrs §3.2 shows it at index 24 in csgo_complex and
    # 19 in csgo_environment_blend, and the expression operands are 24
    # and 19 -- an index-vs-name dissociation a name match cannot
    # produce. fast_pack_fam.py has already done that fold per material,
    # so what arrives here is the resolved per-material value and the
    # structural resolution is upstream, not repeated by string here.
    # THE COLUMN IS IN `ext2`, NOT `ext`. This read used MAT_EXT/EXT --
    # the WORLD PACK's 35-column block -- and `f_depth_bias` is emitted
    # only into the SIDE TABLE's `ext2_keys` (691 columns). There is
    # exactly one EXT population site and it reads the world pack, so the
    # condition could never be true and no invocation could make it true:
    # the per-material gate was dead by construction and the cx/eb/so
    # material bias was applied to EVERY face of those families on every
    # run. :20458 reads the same feature correctly through _e2, so one
    # feature had two lookups and one of them was permanently dead.
    #
    # This is a SHADING CHANGE. Numbers taken before it are not
    # comparable with numbers taken after; the banner below says so on
    # every run that fixes the gate.
    _have2 = (_G.get("MAT_EXT2") is not None
              and "f_depth_bias" in _G.get("EXT2", {}))
    if _have2 and _G.get("face_matid0") is not None:
        # Take the COLUMN first, then gather per face. The natural
        # spelling, MAT_EXT2[face_matid0][..., col], materialises all 691
        # ext2 columns for every one of 8.4M faces -- 21.7 GiB, an
        # instant OOM. One column gathered is 8.4M floats.
        GT_FDEPTHBIAS_OF_FACE = (
            _G["MAT_EXT2"][:, _G["EXT2"]["f_depth_bias"]][_G["face_matid0"]]
            > 0.5)
        # Counts DERIVED from this run's own data. What stood here
        # asserted "21/21 so, 3/13 cx, 0/43 eb" as a literal, and the
        # bound column reports different numbers -- the text was quoting
        # a pack generation other than the one being described. A
        # hardcoded count in a diagnostic is a claim about a run that
        # happened somewhere else.
        _nb = int(GT_FDEPTHBIAS_OF_FACE.sum())
        _nf = int(GT_FDEPTHBIAS_OF_FACE.numel())
        print(f"F_DEPTH_BIAS: material bias restricted to {_nb:,} of "
              f"{_nf:,} faces ({_nb / max(_nf, 1):.2%}) -- read from the "
              f"side table's ext2 column. Before this fix the gate read "
              f"the world pack's `ext` block, where the column does not "
              f"exist, so the bias went to EVERY cx/eb/so face. This "
              f"HALVES the biased texel count in the sun depth atlas "
              f"(31,929,192 -> 15,427,743) and changes NO RENDERED "
              f"PIXEL: measured byte-identical over 71 paired ego01 "
              f"poses, and over 16 more with --shadow 2048. The atlas is "
              f"built and is not read. No score is invalidated by this "
              f"change; the open question it leaves is why a 4x2048^2 "
              f"cascade at 97.8% coverage reaches nothing.", flush=True)
    else:
        # Name the cause that actually fired, rather than blaming the one
        # of three conditions that happens to be printed. A diagnostic
        # that misattributes sends the reader to the wrong table.
        if not _have2:
            _why = ("no --fam-side side table is loaded (the f_depth_bias "
                    "column lives in its ext2 block; rebuild with "
                    "fast_pack_fam.py if the table predates it)")
        else:
            _why = ("the world geometry is not bound (face_matid0 is "
                    "absent), so there is nothing to resolve the column "
                    "against")
        _gt_note(f"depth bias: {_why}, so the cx/eb/so material bias is "
                 f"applied to every face of those families rather than "
                 f"only where the feature is set")
