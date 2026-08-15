"""A2 -- the first-person muzzle flash, at the attachment the asset names.

WHY THIS IS A SEPARATE FILE. `gpu_render.py`'s viewmodel path is owned
by another arm. This adds one function and one constants block; the
wiring into `viewmodel_pass` is ten lines there and nothing else.

============================================================
WHERE IT GOES -- read from the asset on BOTH sides, not eyeballed
============================================================

`extract_viewmodel.py` carries the AK-47's muzzle attachment in the
bundle:

    muzzle_flash_src  (22.6716, 0.2107, -0.6492)   source units,
                                                   rel. weapon_offset

⚠ CORRECTED 2026-08-08. Two claims in the previous wording were wrong,
and the second was actively misleading rather than merely loose.

  "reads the attachment table"  -- it does not read it at run time.
                                  `extract_viewmodel.py:192` carries
                                  `muzzle_flash_src` as a HARDCODED
                                  LITERAL, transcribed once. The value
                                  is still the asset's, but nothing in
                                  this repo re-derives it, so a change
                                  upstream would go unnoticed.

  "shell_eject ... in the bundle" -- IT IS NOT IN THE BUNDLE. The
                                  bundle's keys are muzzle_flash_src,
                                  muzzle_flash_view, pos/nrm/uv/tri,
                                  tex_*, and no shell_eject of any name.
                                  The triple (3.2959, -0.3806, -0.0741)
                                  appears in exactly one place:
                                  `muzzle_flash_selftest.py:83`, which
                                  INJECTS it as a planted fault. This
                                  docstring was citing, as a read value
                                  carried in the bundle, a literal whose
                                  only home is a decoy.

  Its provenance is UNESTABLISHED. It may well be the real shell_eject
  attachment transcribed the same way the muzzle one was, but no file
  here establishes that, and I could not re-read it -- Source2Viewer-CLI
  produced no output for weapon_rif_ak47.vmdl_c on this box.

  CONSEQUENCE for the self-test, which is why this is not cosmetic:
  fault 3 is labelled "shell_eject substituted for muzzle_flash" and
  scored DISTINGUISHABLE at 0.4926 m. The distinguishability result
  stands -- it only needs the decoy to be a different point, and it is.
  The LABEL does not: until the triple is established, fault 3 tests
  "a point 0.4926 m away", not "the shell ejection port".

And the particle system CS2 actually plays on that weapon binds to it
**by the same name** -- `particles/unified_weapon_fx/
uweapon_muzflsh_ak47_fps.vpcf_c`, control point 0:

    m_iAttachType   = PATTACH_POINT_FOLLOW
    m_attachmentName= "muzzle_flash"
    m_nViewModelEffect = INHERITABLE_BOOL_TRUE      <- the FPS variant

Two independent reads of the same attachment name. **So the coordinate
is a PREDICTION, not a placement**: project it and the flash must land
on the muzzle of the rendered barrel. If it does not, something in the
viewmodel transform is wrong, and that is a finding rather than a knob
to turn.

`m_nViewModelEffect` is also why this draws inside `viewmodel_pass`, in
the viewmodel's own projection and own depth range, and not as a
world-space sprite in the main pass.

THE TRANSFORM IS TAKEN FROM THE BUNDLE, NOT RE-DERIVED. The bundle
carries, for every vertex, both `loc` (model-local SOURCE units, the
frame `muzzle_flash_src` is in) and `pos` (view-space metres). That is
an exact affine map and it is recovered here by least squares from the
bundle's own data, with the residual printed. Two consequences:

  * no constant of that transform is duplicated here, so when the
    viewmodel arm replaces the provisional `FWD_PUSH` with the real
    bone out of ANIM/AGRP, the flash follows automatically;
  * the residual is a CHECK. A non-affine or mismatched bundle shows up
    as a large residual instead of as a flash in the wrong place.

============================================================
WHAT IT LOOKS LIKE -- every constant READ, and the one that is not
============================================================

From `uweapon_muzflsh_ak47_primaryflash.vpcf_c`, the child that draws
the flash sprite (`C_OP_RenderSprites`):

    m_nOutputBlendMode   = PARTICLE_OUTPUT_BLEND_MODE_ADD
    m_flOverbrightFactor = 4.0
    m_flRadiusScale      = 0.5
    m_flSelfIllumAmount  = 1.0
    m_flDiffuseAmount    = 0.0        -> unlit; no shading term applies
    m_vecTexturesInput   = materials/particle/fire_gas/
                           fire_gas_batch_b_top.vtex
    m_nAnimationType     = ANIMATION_TYPE_MANUAL_FRAMES

and from its initialisers:

    [9]  C_INIT_InitFloat  RANDOM_UNIFORM 1.0 .. 1.5    (radius)
    [6]  C_INIT_RandomColor m_ColorMin [198,131,80]
                            m_ColorMax [216,216,216]
    [4]  C_INIT_InitFloat  LITERAL 0.05  -> field 1 = LIFE_DURATION
         C_OP_FadeOut      m_flFadeOutTimeMin/Max = 0.015

A deterministic renderer cannot draw a random uniform, so the two
RANDOM_UNIFORM endpoints are collapsed to their midpoint. That is
DERIVED from two read numbers, not fitted -- and both endpoints are
recorded above so the choice is checkable.

**THE ONE UNRESOLVED TERM, stated rather than fitted.** The same
particle carries

    [16] C_INIT_GlobalScale  m_nScaleControlPointNumber = 5

and control point 5 is driven, in the parent's `game` configuration, by
`PATTACH_WORLDORIGIN` with `m_vecOffset [0.35, 1.0, 1.0]`. What that
control point's scale value is at fire time is **not established here**.
`--muzzle-flash-scale` therefore defaults to **1.0 -- the unloaded
term left at unity**, which is the NO_FITTED_FLOATS.md position: an
unread factor is reported as unloaded, never absorbed into a tuned
default. The consequence is a small flash, and that smallness is the
diagnostic magnitude of the unread term, not a reason to pick a number.

RADIUS, then, is entirely read except for that one named gap:

    (1.0 + 1.5)/2  *  0.5           = 0.625 source units
     ^ midpoint of [9]  ^ m_flRadiusScale
    * 0.0254                        = 0.015875 m half-extent
    * --muzzle-flash-scale (default 1.0, UNLOADED)
"""

import numpy as np
import torch

S_UNIT = 0.0254                       # source units -> metres

# --- READ from uweapon_muzflsh_ak47_primaryflash.vpcf_c ---------------
PD_BLEND_MODE = "PARTICLE_OUTPUT_BLEND_MODE_ADD"
OVERBRIGHT_FACTOR = 4.0               # m_flOverbrightFactor
RADIUS_SCALE = 0.5                    # m_flRadiusScale
SELF_ILLUM_AMOUNT = 1.0               # m_flSelfIllumAmount
DIFFUSE_AMOUNT = 0.0                  # m_flDiffuseAmount -> unlit
RADIUS_SRC_MIN = 1.0                  # initializer [9] RANDOM_UNIFORM
RADIUS_SRC_MAX = 1.5
COLOR_MIN = (198.0, 131.0, 80.0)      # C_INIT_RandomColor m_ColorMin
COLOR_MAX = (216.0, 216.0, 216.0)     # C_INIT_RandomColor m_ColorMax
LIFE_DURATION_S = 0.05                # initializer [4] -> field 1
FADE_OUT_S = 0.015                    # C_OP_FadeOut
SPRITE_TEXTURE = ("materials/particle/fire_gas/"
                  "fire_gas_batch_b_top.vtex")
ATTACHMENT_NAME = "muzzle_flash"
PARTICLE_SYSTEM = ("particles/unified_weapon_fx/"
                   "uweapon_muzflsh_ak47_fps.vpcf")

# DERIVED, each from two read endpoints above.
RADIUS_SRC_MID = 0.5 * (RADIUS_SRC_MIN + RADIUS_SRC_MAX)
HALF_EXTENT_M = RADIUS_SRC_MID * RADIUS_SCALE * S_UNIT
COLOR_MID = tuple(0.5 * (a + b) / 255.0
                  for a, b in zip(COLOR_MIN, COLOR_MAX))

# The one term this file does NOT resolve; see the module docstring.
UNLOADED_TERMS = (
    "C_INIT_GlobalScale m_nScaleControlPointNumber=5 -- control point 5's "
    "scale at fire time is UNREAD; --muzzle-flash-scale is left at 1.0",
)


def add_arguments(parser):
    """Namespaced --muzzle-flash*. on/off plus one stated-unloaded term."""
    g = parser.add_argument_group(
        "muzzle flash (uweapon_muzflsh_ak47_fps, viewmodel effect)")
    g.add_argument("--muzzle-flash", action="store_true",
                   help="draw the first-person muzzle flash at the "
                        "weapon's muzzle_flash attachment, in the "
                        "viewmodel's own projection. Needs --viewmodel "
                        "and --viewmodel-model; the attachment comes "
                        "from the bundle, not from this flag.")
    g.add_argument("--muzzle-flash-scale", type=float, default=1.0,
                   help="the UNREAD C_INIT_GlobalScale control-point-5 "
                        "factor, left at 1.0 = not loaded. This is a "
                        "diagnostic magnitude, NOT a tuning knob: a "
                        "value that makes the flash 'look right' is "
                        "measuring how big the unread term is.")
    g.add_argument("--muzzle-flash-gate", choices=("fire", "always"),
                   default="fire",
                   help="`fire` (default) draws the flash only on frames "
                        "inside a LIFE_DURATION window of a `fire` tick "
                        "in the camera json -- the firing signal IS on "
                        "the wire. `always` is the REACHABILITY arm: it "
                        "draws every frame so the pass can be shown to "
                        "cover pixels on a path with no shots in it. "
                        "`always` is a diagnostic, never a default.")
    g.add_argument("--muzzle-flash-predict-only", action="store_true",
                   help="print the predicted screen position of the "
                        "attachment and draw nothing. The check without "
                        "the pixels.")
    g.add_argument("--muzzle-flash-occlude", choices=("on", "off"),
                   default="on",
                   help="`on` (default) depth-tests the flash against the "
                        "viewmodel's own depth buffer, so gun geometry "
                        "between the eye and the muzzle occludes the "
                        "bloom. `off` is the DIAGNOSTIC arm that restores "
                        "the un-tested additive composite -- it exists to "
                        "reproduce the defect on demand and is never a "
                        "default. The sprite is still ADDITIVE where it "
                        "survives the test: not occluding and not "
                        "darkening are different properties, and only the "
                        "second is what m_nOutputBlendMode declares.")
    return g


def affine_from_bundle(loc_src, pos_view):
    """Recover the bundle's own (source units -> view metres) affine.

    `loc` and `pos` are the SAME vertices in the two frames, so the map
    is exact and this is a solve, not a fit. Returns (A, b, residual).
    The residual is the check: it must be ~0, and a bundle whose two
    arrays disagree shows up here rather than as a flash in the wrong
    place.
    """
    L = np.asarray(loc_src, np.float64)
    P = np.asarray(pos_view, np.float64)
    if L.shape[0] < 4:
        raise ValueError("need >= 4 vertices to solve the affine")
    Lh = np.concatenate([L, np.ones((L.shape[0], 1))], axis=1)
    M, *_ = np.linalg.lstsq(Lh, P, rcond=None)          # (4,3)
    A = M[:3].T                                          # (3,3)
    b = M[3]                                             # (3,)
    resid = np.abs(Lh @ M - P)
    return A, b, float(resid.max())


def attachment_view_point(bundle, key=ATTACHMENT_NAME + "_src"):
    """Map an attachment (source units) into the bundle's view frame.

    Returns (point_view_metres, max_affine_residual_m). Raises if the
    bundle does not carry the attachment -- silence here would be the
    failure that does not look like one.
    """
    if key not in bundle:
        raise KeyError(
            f"the viewmodel bundle carries no {key!r}. It is written by "
            "extract_viewmodel.py; rebuild the bundle rather than "
            "hardcoding the coordinate here.")
    # THE BAKED POINT IS THE ANSWER, and the affine below is the bug it
    # replaced -- not a fallback that happens to be worse.
    #
    # `muzzle_flash_src` is a bone of the ANIMATION SKELETON
    # (animation/skeletons/weapons/<w>.vnmskel, bone `muzzle`). `loc`/`pos`
    # are MESH vertices from the .vmdl_c. Fitting an affine from loc->pos
    # gives the MESH's source->view map, and applying it to a SKELETON-space
    # point is a frame confusion: the two spaces differ by the bind
    # translation the extractor subtracts explicitly, so the fit lands the
    # muzzle somewhere the gun is not.
    #
    # The tell was always there and read as reassurance: the residual is
    # 3e-08, because the fit reproduces the MESH perfectly. A perfect fit of
    # the wrong correspondence is still the wrong answer, and the small
    # number made it look verified.
    #
    # MEASURED over every shipped bundle: the affine and the extractor's own
    # baked view disagree on ALL of them -- 0.107 m on ak_47, 0.560 m on
    # m4a4, up to 0.479 m on g3sg1. The baked point sits inside the weapon
    # mesh at the barrel end for all of them; the affine point does not.
    # m4a4's affine value is [0.1130, -0.4075, -0.4406], which is exactly
    # what the combat clip logged and drew 271 px off the gun.
    #
    # So the baked view is used when present. extract_viewmodel computes it
    # by the same explicit chain it applies to the vertices -- bind
    # subtracted, hold pose applied, `wpn` bone added, cvar offset and DXY
    # -- which is the skeleton->view map this point needs.
    view = bundle.get(ATTACHMENT_NAME + "_view")
    loc = bundle["loc"]
    pos = bundle["pos"]
    loc = loc.detach().cpu().numpy() if torch.is_tensor(loc) else loc
    pos = pos.detach().cpu().numpy() if torch.is_tensor(pos) else pos
    A, b, resid = affine_from_bundle(loc, pos)
    src = np.asarray(bundle[key], np.float64)
    if view is not None:
        # `resid` is still the MESH fit's residual and is still worth
        # reporting -- it says the bundle's own geometry is self-consistent
        # -- but it is no longer evidence about the attachment.
        return np.asarray(view, np.float64), resid
    return A @ src + b, resid


def project_view_point(pt, rng, hw, device, vm_transform=None):
    """A view-space point -> pixel, through the SAME chain the flash uses.

    Exists so the barrel's two ends can be projected with the flash rather
    than beside it. A second implementation of this projection would be a
    second convention, and the question being asked -- which screen end is
    the muzzle -- is exactly the kind a convention mismatch answers wrongly
    while looking right.
    """
    Hh, Ww = hw
    if vm_transform is not None:
        t = torch.tensor(pt, dtype=torch.float32, device=device)[None]
        pt = vm_transform(
            t, torch.zeros(1, dtype=torch.bool, device=device)
        )[0].detach().cpu().numpy()
    v = torch.tensor([[float(pt[0]), float(pt[1]), float(pt[2])]],
                     dtype=torch.float32, device=device)
    clip = rng.clip_of_view(v)[None].contiguous()
    ndc = clip[0, :, :3] / clip[0, :, 3:4].clamp(min=1e-9)
    return (float((ndc[0, 0] * 0.5 + 0.5) * Ww),
            float((1.0 - (ndc[0, 1] * 0.5 + 0.5)) * Hh))


def _quad_view(centre, half, device):
    """Camera-facing quad at `centre`.

    View space is the camera's own frame, so 'camera-facing' is exactly
    'axis-aligned in XY at this depth' -- no billboard basis is needed
    and none is invented.
    """
    cx, cy, cz = [float(v) for v in centre]
    v = torch.tensor(
        [[cx - half, cy - half, cz],
         [cx + half, cy - half, cz],
         [cx + half, cy + half, cz],
         [cx - half, cy + half, cz]], dtype=torch.float32, device=device)
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32,
                       device=device)
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                      dtype=torch.float32, device=device)
    return v, tri, uv


def sprite_alpha(uv):
    """The sprite's own falloff when its texture is NOT loaded.

    STATED STAND-IN. `fire_gas_batch_b_top.vtex` is a 4-frame manual
    sequence and this is not it -- it is a radial falloff, so the quad
    reads as a flash rather than as a square. It is reported as
    NOT LOADED by `provenance()` and must not be read as the asset.
    """
    d = (uv - 0.5).mul(2.0).norm(dim=-1)
    return (1.0 - d).clamp(min=0.0) ** 2


def provenance(tex_loaded):
    """Per-slot provenance, printed rather than left in a comment."""
    rows = [
        f"attachment      {ATTACHMENT_NAME} READ from the bundle "
        f"(and named by {PARTICLE_SYSTEM})",
        f"blend           {PD_BLEND_MODE} READ",
        f"overbright      {OVERBRIGHT_FACTOR} READ (m_flOverbrightFactor)",
        f"radius          {RADIUS_SRC_MIN}..{RADIUS_SRC_MAX} src READ, "
        f"midpoint {RADIUS_SRC_MID} DERIVED, x{RADIUS_SCALE} READ "
        f"-> half-extent {HALF_EXTENT_M:.7f} m",
        f"colour          {COLOR_MIN} .. {COLOR_MAX} READ, "
        f"midpoint DERIVED -> {tuple(round(c, 4) for c in COLOR_MID)}",
        f"lifetime        {LIFE_DURATION_S} s READ, fade {FADE_OUT_S} s READ",
        f"texture         {SPRITE_TEXTURE} "
        + ("LOADED" if tex_loaded else "NOT LOADED -- radial stand-in"),
    ]
    return rows + ["UNLOADED: " + u for u in UNLOADED_TERMS]


def draw(dr, ctx, rng, args, bundle, hw, device, vm_transform=None,
         page=None):
    """Additive muzzle-flash layer in the viewmodel's own projection.

    Returns (rgb, alpha, info) with rgb premultiplied by the additive
    blend the reference declares, or (None, None, info) if the flash
    covers no pixel. `info` carries the predicted screen position, which
    is the thing to check against the rendered barrel.
    """
    Hh, Ww = hw
    pt, resid = attachment_view_point(bundle)
    if vm_transform is not None:
        # ride the SAME transform the geometry rides, so the flash
        # cannot drift away from the barrel it belongs to
        t = torch.tensor(pt, dtype=torch.float32, device=device)[None]
        pt = vm_transform(
            t, torch.zeros(1, dtype=torch.bool, device=device)
        )[0].detach().cpu().numpy()
    half = HALF_EXTENT_M * float(args.muzzle_flash_scale)
    v, tri, uv = _quad_view(pt, half, device)
    clip = rng.clip_of_view(v)[None].contiguous()
    ndc = clip[0, :, :3] / clip[0, :, 3:4].clamp(min=1e-9)
    cx = float((ndc[:, 0].mean() * 0.5 + 0.5) * Ww)
    cy = float((1.0 - (ndc[:, 1].mean() * 0.5 + 0.5)) * Hh)
    # --- IS THE ATTACHMENT ON THE WEAPON IT BELONGS TO? ----------------
    # in_frame=False is a true statement and a useless one: it cannot tell
    # "the player is aiming somewhere the barrel is off-screen" from "this
    # bundle's muzzle point is not on its own gun". Both print the same
    # line, and the second is a BUNDLE defect that no amount of looking at
    # the renderer will find.
    #
    # MEASURED across the 20 weapon bundles: the attachment normally sits
    # within the weapon's own bbox and near the front of it -- ak_47 muzzle
    # z -0.874 against geometry reaching -0.957, awp -1.264 against -1.374.
    # m4a4 does not: its muzzle is at y -0.408 while the whole weapon spans
    # y [-0.304, -0.011], i.e. BELOW every vertex of the gun, and at z
    # -0.441 against a barrel reaching -0.905, i.e. halfway along. That
    # bundle predicts 42.8 deg below the view axis where its own geometry
    # sits at 12.2, so the flash lands 517 px under a 720-high frame while
    # the weapon renders normally.
    #
    # So the check is against the bundle's OWN vertices, needs no reference
    # and no threshold to be tuned: a muzzle outside the mesh it is
    # attached to is wrong however the camera is pointed.
    _pos = bundle.get("pos")
    _bb = None
    if _pos is not None:
        _p = _pos.detach().cpu().numpy() if hasattr(_pos, "detach") else _pos
        _lo, _hi = _p.min(0), _p.max(0)
        # z is forward-negative, so a muzzle AHEAD of the mesh is expected
        # and only the lateral/vertical axes are bounded. A small pad
        # absorbs the half-extent of the quad rather than a fitted slack.
        _pad = half
        _out = [ax for ax, i in (("x", 0), ("y", 1))
                if not (_lo[i] - _pad <= pt[i] <= _hi[i] + _pad)]
        _bb = {"lo": [float(v) for v in _lo], "hi": [float(v) for v in _hi],
               "outside_axes": _out,
               "z_frac": (float(pt[2] / _lo[2]) if _lo[2] < -1e-6 else None)}
    # --- DOES THIS BUNDLE RE-DERIVE FROM ITS OWN PROVENANCE? -----------
    # The extractor now refuses to WRITE a bundle whose recorded chain does
    # not reproduce its baked view point. Bundles already on disk predate
    # that refusal, and the shipped set disagreed by up to 0.560 m -- so the
    # same invariant is checked at LOAD, where the alternative is trusting a
    # file whose two halves contradict each other.
    #
    # Reported, not corrected. Which half is stale is not decidable from
    # here: the baked point may be the newer read (it was, in the set I
    # measured) or the chain may be. Picking one would be the guess this
    # check exists to replace.
    _prov = None
    try:
        _src = bundle.get(ATTACHMENT_NAME + "_src")
        if _src is not None and "bind_applied_src" in bundle:
            # WHICH CHAIN -- READ, NEVER ASSUMED. This subtracted
            # bind_applied_src unconditionally, which is the MESH chain, and
            # applied it to the MUZZLE. The bundle declares two chains and
            # says so in two fields: mesh_bind_subtracted = 1,
            # muzzle_bind_subtracted = 0, because the .vnmskel's weapon bone
            # sits at the origin while the .vmdl_c's sits at the bind, and
            # the frames differ by exactly bind_applied_src.
            #
            # So the checker manufactured its own failure. On m4a4 it
            # reported the bundle 0.560182 m from its own provenance; through
            # the DECLARED chain the recomputed point matches the baked one
            # at 0.000000 m. The bundle was self-consistent the whole time
            # and I reported it as broken.
            #
            # This is the refusal-pattern tell in its exact form: a check
            # that fails on precisely the subset a second rule governs has
            # one rule hardcoded. The field naming the rule was sitting in
            # the bundle, unread, next to the one being used.
            _sub = bundle.get("muzzle_bind_subtracted")
            _sub = bool(np.asarray(_sub).ravel()[0]) if _sub is not None \
                else True
            _l = np.asarray(_src, np.float64)
            if _sub:
                _l = _l - np.asarray(bundle["bind_applied_src"], np.float64)
            _q = bundle.get("wpn_bone_quat")
            if _q is not None:
                _qx, _qy, _qz, _qw = np.asarray(_q, np.float64)
                _t = 2.0 * np.cross([_qx, _qy, _qz], _l)
                _l = _l + _qw * _t + np.cross([_qx, _qy, _qz], _t)
            _w = np.asarray(bundle["wpn_bone_src"], np.float64)
            _v = _l + _w
            _rec = (np.array([-_v[1], _v[2], -_v[0]]) * 0.0254
                    + np.asarray(bundle.get("offset_m", [0, 0, 0]), np.float64)
                    + np.asarray(bundle.get("dxy_landmark", [0, 0, 0]), np.float64)
                    + np.asarray(bundle.get("dxy_muzzle", [0, 0, 0]), np.float64))
            _d = float(np.linalg.norm(
                _rec - np.asarray(bundle[ATTACHMENT_NAME + "_view"], np.float64)))
            # BOTH SIDES OF THE COMPARISON are carried, because the
            # first draft of this print showed info["view_point_m"] as the
            # "baked" side -- that is the POST-vm_transform point, so on a
            # run with no offset it printed two identical numbers beside a
            # nonzero delta and read as a broken check. A message must show
            # the values it actually compared.
            _prov = {"baked": [float(x) for x in np.asarray(
                         bundle[ATTACHMENT_NAME + "_view"], np.float64)],
                     "recomputed": [float(x) for x in _rec], "delta_m": _d,
                     "chain": ("bind-subtracted (mesh chain)" if _sub
                               else "bind NOT subtracted (declared muzzle "
                                    "chain)"),
                     "consistent": _d <= 1e-5}
    except Exception:                                         # noqa: BLE001
        _prov = None
    info = {
        "provenance_recheck": _prov,
        "attachment_bbox": _bb,
        "attachment_src": [float(x) for x in bundle[ATTACHMENT_NAME + "_src"]],
        "view_point_m": [float(x) for x in pt],
        "affine_residual_m": resid,
        "predicted_px": (cx, cy),
        "half_extent_m": half,
        "in_frame": bool(0 <= cx < Ww and 0 <= cy < Hh),
        "texture_loaded": page is not None,
    }
    if args.muzzle_flash_predict_only:
        return None, None, info
    rast, _ = dr.rasterize(ctx, clip, tri, (Hh, Ww))
    covered = rast[..., 3] > 0
    # THE FLASH'S OWN DEPTH, for the caller's depth test (#87).
    #
    # Taken from THIS quad's rasterised z through the SAME `rng` the
    # viewmodel pass writes its depth buffer with, so the comparison has
    # one convention on both sides -- the viewport minDepth/maxDepth remap
    # -- and no scalar muzzle depth has to be re-derived anywhere. The
    # sprite is a billboard AT the attachment point, so this IS the
    # muzzle's depth; per-pixel rather than one number because that is
    # what the rasteriser produces and what the reference tests per
    # fragment. Returned in `info` rather than as a fourth return value so
    # the archived caller keeps working.
    info["window_depth"] = rng.window_depth(rast[..., 2])
    info["covered_mask"] = covered
    if not bool(covered.any()):
        info["covered_px"] = 0
        return None, None, info
    uvp, _ = dr.interpolate(uv[None].contiguous(), rast, tri)
    a = sprite_alpha(uvp) if page is None else _page_alpha(page, uvp)
    a = a * covered.float()
    col = torch.tensor(COLOR_MID, dtype=torch.float32, device=device)
    # PARTICLE_OUTPUT_BLEND_MODE_ADD with m_flOverbrightFactor: the
    # sprite adds colour*alpha*overbright and never occludes.
    rgb = col.view(1, 1, 1, 3) * a.unsqueeze(-1) * OVERBRIGHT_FACTOR
    info["covered_px"] = int(covered.sum())
    return rgb, a, info


def _page_alpha(page, uvp):
    """Bilinear alpha out of a loaded sprite page (H,W,4) in 0..1."""
    H, W = page.shape[0], page.shape[1]
    u = uvp[..., 0].clamp(0, 1) * (W - 1)
    v = uvp[..., 1].clamp(0, 1) * (H - 1)
    x0 = u.floor().long().clamp(0, W - 1)
    y0 = v.floor().long().clamp(0, H - 1)
    return page[y0, x0, 3]
