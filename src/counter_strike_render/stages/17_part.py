

def gt_compose(linear, normal, view, wpos, eye, rough, metal, quality,
               baked_irr, baked_occ, ao, sun_shadow, alights_diff=None,
               alights_spec=None, lm_tint=None):
    """Assemble the frame the way the pixel shader assembles it.

    q=0, csgo_complex_ps.glsl:618-945:
        atten  = shadow * (1 - dot(bakedOcc, g_vSunOcclusionMask));   :618
        (diff, spec) = sun BRDF at q=0;                            :623-636
        ... local lights accumulate into the same two ...          :672-921
        ao     = ssao * aoTex.x;                                      :933
        rgb    = (diff + bakedIrr*ao) * albedo*(1 - metal);           :934
        rgb   += spec * ao;                                           :940
        rgb   += specCube(q=0) * envBRDF(q=0) * ao;                   :945

    q=1, shaderquality1.glsl:721-1187, three extra structural terms the
    q=0 file has no counterpart for:
        ao     = ssao * aoTex.x * screenAO.x;                     :561,1175
        E      = envResponse + multiScatter;                      :1171-1174
        rgb    = (diff + bakedIrr*(1 - E)*ao) * albedo*(1 - metal);  :1176
        rgb   += spec * (1 + f0 * (0.125*(rx+ry)^4 * saturate(dot(n,v))))
                      * ao;                                          :1182
        rgb   += specCube(q=1) * E * ao;                             :1187
    -- an energy-conservation factor on the diffuse, a roughness-driven
    boost on the direct specular, and multiple scattering in E. The
    diffuse and specular ACCUMULATORS also swap slots between the two
    files (:635-636 vs :742-743, swapped back at :1051-1052); that is
    bookkeeping and gt_direct() returns them by name so it cannot leak.
    """
    mask = torch.tensor(args.gt_sun_occ_mask, device=linear.device)
    atten = sun_shadow * (1.0 - (baked_occ * mask).sum(-1))          # :618
    _gt_value("sun.shadow", sun_shadow, identity=1.0,
              note="the CSM product. the cascade depth-sign fix: this was 0.0000 "
                   "on 100.0% of 518,400 samples for months while the "
                   "term was labelled `live` from --gt-lighting default 1")
    _gt_value("sun.baked_occlusion",
              1.0 - (baked_occ * mask).sum(-1), identity=1.0,
              note="csgo_environment_ps.glsl:888 -- dot(bakedOcc, "
                   "g_vSunOcclusionMask), with the mask READ as "
                   "(1, 0, 0, 0) from the wallA2 capture at byte offset "
                   "31152, all four components. WITH --gt-lm-occlusion this "
                   "is the second lightmap page (direct_light_shadows) and "
                   "the term is live; WITHOUT it the page reads 0, the "
                   "shader's unbound value, and this pins at exactly 1 -- "
                   "which is what it did for as long as no map had the "
                   "page extracted")
    f0 = 0.04 * (1 - metal).unsqueeze(-1) + \
        linear.clamp(0, 1) * metal.unsqueeze(-1)
    l = sun_dir.view(1, 1, 1, 3).expand_as(normal)
    diff, spec = gt_direct(quality, normal, l, view, f0, rough,
                           SUN_COLOR.view(1, 1, 1, 3), atten, mat_ao=ao)
    if alights_diff is not None:
        diff = diff + alights_diff
    if alights_spec is not None:
        spec = spec + alights_spec
    ao3 = ao.unsqueeze(-1)
    ndv = (normal * view).sum(-1).clamp(0, 1)
    rmean = rough                                     # dot(rough2, vec2(.5))
    resp, ms = gt_env_brdf(quality, rmean, ndv, f0)
    dalb = linear * (1.0 - metal).unsqueeze(-1)
    tint = GT_LM_TINT_LIVE if lm_tint is None else bool(lm_tint)
    if quality == 0:
        ind = _gt_lm_tint(baked_irr * ao3, tint)                # :1230,:1234
        rgb = (diff + ind) * dalb                                    # :934
        rgb = rgb + spec * ao3                                       # :940
    else:
        E = resp + ms
        ind = _gt_lm_tint(baked_irr * (1.0 - E) * ao3, tint)   # :1588,:1592
        rgb = (diff + ind) * dalb                                   # :1176
        rb = 1.0 + f0 * (0.125 * ((rough + rough) ** 4)
                         * ndv.clamp(0, 1)).unsqueeze(-1)           # :1182
        rgb = rgb + spec * rb * ao3
    # The environment specular is UNCONDITIONAL: at q=0 it is in the
    # module outright (:945), and at q=1 it is present in both
    # D_SPECULAR_CUBE_MAP_STATIC states (the axis changes HOW the cube is
    # found, not WHETHER -- the cubearray fetch count is 1 either way,
    # SHADER_CALLFLOW_csgo_environment.md lines 667/675 and 668/676). So
    # it is not gated on --spec-cube-static, which would have made the
    # whole term vanish at the default and read as tested.
    env = gt_spec_cube(quality, wpos, normal, view, rough, baked_irr, diff,
                       static_cube=bool(args.spec_cube_static))
    rgb = rgb + env * (resp + ms) * ao3                         # :945/:1187
    _gt_value("env.brdf_response", resp, identity=0.0,
              note="split-sum B + f0*A. Multiplies the environment "
                   "specular, so if it is 0 the cube never reaches the "
                   "image no matter what the cube contains")
    _gt_value("env.multiscatter", ms, identity=0.0,
              note="q=1 only (shaderquality1.glsl:1172-1174); exactly 0 "
                   "at q=0 BY CONSTRUCTION, not by defect")
    _gt_value("compose.ao", ao3, identity=1.0)
    _gt_value("baked.irradiance", baked_irr, identity=0.0)
    _gt_value("compose.direct_diffuse", diff, identity=0.0)
    _gt_value("compose.direct_specular", spec, identity=0.0)
    return rgb


# =====================================================================
# csgo_lightmappedgeneric.vfx, generic.vfx (csgo_core) and the
# csgo_imported search path.
#
# PROVENANCE. Extracted here, not taken from a previous document:
#   csgo_core/shaders_vulkan_dir.vpk  shaders/vfx/csgo_lightmappedgeneric
#       _vulkan_50_ps.vcs  16,922,302 B, 1008 shipped static combos,
#       588 distinct bytecode records.
#   csgo_core/shaders_vulkan_dir.vpk  shaders/vfx/generic_vulkan_50_ps.vcs
#       1,942,120 B, 70 static combos, 70 records.
#   core/shaders_vulkan_dir.vpk       shaders/vfx/generic_vulkan_50_ps.vcs
#       88,650 B, 50 static combos, 43 records -- SHADOWED, see below.
#
# WHICH generic.vfx WINS. Decided from csgo/gameinfo.gi `SearchPaths`,
# which lists in this order:  Game csgo / Game csgo_imported /
# Game csgo_core / Game core.  Checked directly against the four
# directories: `csgo` ships a shaders_vulkan_dir.vpk but it contains NO
# `generic` and NO `csgo_lightmappedgeneric` (65 families, enumerated);
# `csgo_imported` ships no shaders vpk at all; `csgo_core` has both.
# So the FIRST search path that has generic.vfx is csgo_core and its
# copy is the one the engine loads. Not decided by file size -- the two
# copies also declare DIFFERENT axes (the csgo_core copy has
# S_MODE_TOOLS_VIS and a 7-axis dynamic space; the core copy has
# S_MODE_TOOLS_WIREFRAME, S_TOOLS_ENABLED, S_SPECULAR_CUBE_MAP,
# S_RENDER_BACKFACES and a single D_ALPHATINT dynamic axis), so picking
# the wrong one is not a size question, it is a different shader.
#
# RECORD SELECTION. Static combo 0 -> position 0 in m_staticComboIDs ->
# that entry's m_nByteCodeDataIdx, which is 0 for both families. Read
# with vcsx2.bcdi(); never the array position (vcs/README.md section 2).
#
# COMBO IDS ARE MIXED-RADIX. csgo_lightmappedgeneric's S_DETAILTEXTURE
# has range 0..2, so the next axis, S_TEXTURETRANSFORMS, sits at stride
# 96 and S_ALPHA_TEST at 192. lmg_static_combo() below builds the id
# with a running place value; anything using `&` is wrong here.
# =====================================================================

# (axis name, stride, min, max, ext2 column or None if not material-driven)
LMG_STATIC_AXES = (
    ("S_MODE_TOOLS_VIS", 1, 0, 1, None),
    ("S_SPECULAR_DIRECT", 2, 0, 1, "f_lmg_specular_direct"),
    ("S_SPECULAR_INDIRECT", 4, 0, 1, "f_lmg_specular_indirect"),
    ("S_LAYERS", 8, 0, 1, "f_lmg_layers"),
    ("S_FANCY_BLENDING", 16, 0, 1, "f_lmg_fancy_blending"),
    ("S_DETAILTEXTURE", 32, 0, 2, "f_lmg_detailtexture"),
    ("S_TEXTURETRANSFORMS", 96, 0, 1, "f_lmg_texturetransforms"),
    ("S_ALPHA_TEST", 192, 0, 1, "f_lmg_alpha_test"),
    ("S_TRANSLUCENT", 384, 0, 1, "f_lmg_translucent"),
    ("S_DETAILBLENDMODE", 768, 0, 1, "f_lmg_detailblendmode"),
    ("S_OVERLAY", 1536, 0, 1, "f_lmg_overlay"),
    ("S_METALNESS_TEXTURE", 3072, 0, 1, "f_lmg_metalness_texture"),
    ("S_SHADER_QUALITY", 6144, 0, 1, None),
)
LMG_DYN_AXES = (
    ("D_BAKED_LIGHTING_FROM_VERTEX_STREAM", 1, 0, 1),
    ("D_BAKED_LIGHTING_FROM_PROBE", 2, 0, 1),
    ("D_BAKED_LIGHTING_FROM_LIGHTMAP", 4, 0, 1),
    ("D_SPECULAR_CUBE_MAP_STATIC", 8, 0, 1),
    ("D_OPAQUE_FADE", 16, 0, 1),
    ("D_MBOIT_PASS1", 32, 0, 1),
    ("D_MBOIT_PASS2", 64, 0, 1),
    ("D_MBOIT_4_MOMENTS", 128, 0, 1),
)
GEN_STATIC_AXES = (
    ("S_SPECULAR", 1, 0, 1, "f_gen_specular"),
    ("S_UNLIT", 2, 0, 1, "f_gen_unlit"),
    ("S_MODE_TOOLS_VIS", 4, 0, 1, None),
    ("S_ALPHA_TEST", 8, 0, 1, "f_gen_alpha_test"),
    ("S_TRANSLUCENT", 16, 0, 1, "f_gen_translucent"),
    ("S_SELF_ILLUM", 32, 0, 1, "f_gen_self_illum"),
    ("S_TINT_MASK", 64, 0, 1, "f_gen_tint_mask"),
    ("S_OVERLAY", 128, 0, 1, "f_gen_overlay"),
)
GEN_DYN_AXES = (
    ("D_BAKED_LIGHTING_FROM_VERTEX_STREAM", 1, 0, 1),
    ("D_BAKED_LIGHTING_FROM_PROBE", 2, 0, 1),
    ("D_BAKED_LIGHTING_FROM_LIGHTMAP", 4, 0, 1),
    ("D_SPECULAR_CUBE_MAP_STATIC", 8, 0, 1),
    ("D_MBOIT_PASS1", 16, 0, 1),
    ("D_MBOIT_PASS2", 32, 0, 1),
    ("D_MBOIT_4_MOMENTS", 64, 0, 1),
)
# The SHADOWED core copy, recorded so the reader can see it was read and
# not skipped. Nothing dispatches to it; the search path picks csgo_core.
GEN_CORE_SHADOWED_AXES = (
    ("S_UNLIT", 1, 0, 1), ("S_MODE_TOOLS_WIREFRAME", 2, 0, 1),
    ("S_TOOLS_ENABLED", 4, 0, 0), ("S_ALPHA_TEST", 4, 0, 1),
    ("S_TRANSLUCENT", 8, 0, 1), ("S_SPECULAR", 16, 0, 1),
    ("S_SPECULAR_CUBE_MAP", 32, 0, 1), ("S_SELF_ILLUM", 64, 0, 1),
    ("S_RENDER_BACKFACES", 128, 0, 1), ("S_OVERLAY", 256, 0, 1),
)

LMG_NOTES = []
# Pixels each of the three families actually shaded, accumulated across
# every chunk of every frame. Printed at the end whether it is zero or
# not: a family that owns no pixels HERE must be visibly distinguishable
# from a family that was never routed to.
LMG_REACH = {}


def _lmg_note(msg):
    if msg not in LMG_NOTES:
        LMG_NOTES.append(msg)


def _lmg_inject(name):
    return args.lmg_selftest_inject == name


def lmg_static_combo(x, axes):
    """Per-material static combo id, MIXED-RADIX (vcs/README.md s1).

    id = sum_i clamp(value_i, min_i, max_i) * stride_i

    Built from the axis's own m_nComboIndexValue place value, never from
    a bit position. csgo_lightmappedgeneric is the family that proves the
    difference: S_DETAILTEXTURE has THREE values, so S_TEXTURETRANSFORMS
    is at 96 and S_ALPHA_TEST at 192, and a `1 << k` decode puts them at
    64 and 128 -- two different shaders.
    """
    out = torch.zeros(x.shape[:-1], device=x.device)
    for i, (_n, stride, lo, hi, col) in enumerate(axes):
        if col is None or col not in EXT2:
            continue
        v = _e2(x, col).clamp(float(lo), float(hi)).round()
        if _lmg_inject("combo-bitmask"):
            # The documented trap, on purpose: place value replaced by a
            # power of two. It must move csgo_lightmappedgeneric's ids.
            out = out + v * float(1 << i)
        else:
            out = out + v * float(stride)
    return out


# ---------------------------------------------------------------------
# csgo_lightmappedgeneric.vfx  (csgo_core), static combo 0, record 0
# module r0_m3 = D_BAKED_LIGHTING_FROM_LIGHTMAP (dyn 4), 714 GLSL lines
# ---------------------------------------------------------------------
def lmg_normal_hemioct(nr):
    """g_tLayer1NormalRoughness -> tangent-space normal.  r0_m3:193-197

        float a = (t.x + t.y) - 1.00392162799835205078125;
        float b =  t.x - t.y;
        n = normalize(vec3(a, b, (1.0 - abs(a)) - abs(b)));

    This is HemiOctAnisoRoughness, the encoding the vtex header names,
    and it is NOT the DXT5nm `(t.wy*2-1)` decode generic.vfx uses on its
    own normal map -- the two families disagree about their own normal
    texture and swapping the decoders silently tilts every normal.
    The constant is 1 + 1/255 exactly, which is why a flat texel of
    (127,127) lands on (-0.0078, 0, 0.99997) rather than exactly +Z.
    """
    a = (nr[..., 0] + nr[..., 1]) - 1.00392162799835205078125
    b = nr[..., 0] - nr[..., 1]
    z = (1.0 - a.abs()) - b.abs()
    return _nrm(torch.stack([a, b, z], dim=-1))
