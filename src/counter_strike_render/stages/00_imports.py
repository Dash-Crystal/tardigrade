"""Batched H200 ego renderer for TARD 2.1 camera paths over a packed world.

v2 pipeline (all GPU-resident):
  1. spatial clusters (~8m) with AABBs + bounding spheres;
  2. per-frame frustum visibility (center+radius test, batched);
  3. Hi-Z occlusion: large-area opaque tris rasterize to a 128x72 depth
     buffer per frame, a max-mip pyramid conservatively culls clusters
     whose nearest point lies behind the occluder depth in their screen
     footprint (behind-camera footprints are never culled);
  4. LOD: index-buffer-only vertex-clustering levels (8cm / 32cm grid,
     original vertex attributes kept, cluster runs stay contiguous);
     level per cluster chosen by projected pixel error;
  5. 16-frame chunks rasterize instanced against their tight cluster
     union (nvdiffrast float32 tri ids cap any call at 2^24 tris);
  6. torch.compile-fused texture-array gather + sun/sky shade + tonemap;
  7. pinned double-buffered async D2H streamed into ffmpeg.

Coordinates: glTF = (src_y, src_z, src_x) * 0.0254 (ray-cast validated).

  venv/bin/python gpu_render.py --world assets/world_packed.pt \
      --camera-json assets/camera-p1.json --tick-begin 9600 \
      --tick-end 11100 --fps 32 --width 960 --height 540 --batch 512 \
      --out firefight_gpu.mp4
"""

import argparse
import collections
import json
import math
import os
import shlex
import subprocess

# --- STAGE TIMING -----------------------------------------------------
# The two-number summary (pure-render fps, end-to-end fps) says 70% of the
# wall clock is not rendering, and nothing said WHICH 70%. `elapsed` also
# starts AFTER the pack load, the LOD build and the warm-up render, so the
# per-agent process launch -- the thing a fleet pays ten times per session --
# was outside both numbers entirely. STAGE accumulates seconds by name and
# _T_PROC0 anchors them to process start, so the split is reported rather
# than inferred.
import time as _time
_T_PROC0 = _time.time()
STAGE = {}


class stage:
    """with stage('png.encode'): ..."""

    __slots__ = ("k", "t")

    def __init__(self, k):
        self.k = k

    def __enter__(self):
        self.t = _time.time()
        return self

    def __exit__(self, *a):
        STAGE[self.k] = STAGE.get(self.k, 0.0) + (_time.time() - self.t)
        return False
# ----------------------------------------------------------------------
import sys
import time
import traceback

import numpy as np
import torch

# --- the importable shading modules -----------------------------------
# This file parses argv at import, so nothing defined in it can be
# imported by a test or by the conformance registry -- and an expression
# no test can reach is an expression whose conformance is an assertion.
# Transcribed shading now lives in `src/counter_strike_render/shading/`, which
# imports nothing from here, and this file CALLS it. Same code object in
# the renderer and in the A/B, so a passing term is a statement about
# the renderer rather than about a parallel copy of it.
#
# Two load paths because the renderer is deployed as a lone file on
# ws-1 (~/region_attr/gpu_render.py) as well as living in the tree: the
# package import first, then a sibling `shading/` directory next to this
# file. A failure to find EITHER is fatal and says so -- silently
# falling back to an inline copy is how two implementations start.
# THE REPO ROOT GOES ON sys.path BEFORE THE FIRST PACKAGE IMPORT, not after.
# It used to be inserted 25 lines below this, so running the file without
# PYTHONPATH set took the fallback branch every time -- and the fallback
# loaded shading/environment.py under a BARE module name, which has no
# parent package, so its own `from ..passes._arraylib import np` died on a
# relative import beyond the top-level package. The failure looked like a
# missing shading module and was a path-ordering bug.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from counter_strike_render.shading import environment as env_shading
except ImportError as _e:
    import importlib.util as _ilu
    _sh = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "shading", "environment.py")
    if not os.path.isfile(_sh):
        raise SystemExit(
            f"REFUSED: cannot import the shading module. Tried the "
            f"package `counter_strike_render.shading.environment` and the "
            f"sibling path {_sh}. Deploy `shading/` alongside this file; "
            f"there is deliberately no inline fallback, because a "
            f"fallback copy is a second implementation nobody diffs.")
    # Load it UNDER ITS PACKAGE NAME so relative imports resolve. A bare
    # name cannot carry `from ..passes...` no matter where the file is.
    _spec = _ilu.spec_from_file_location(
        "counter_strike_render.shading.environment", _sh)
    env_shading = _ilu.module_from_spec(_spec)
    sys.modules["counter_strike_render.shading.environment"] = env_shading
    try:
        _spec.loader.exec_module(env_shading)
    except Exception as _e2:                                  # noqa: BLE001
        raise SystemExit(
            f"REFUSED: {_sh} exists but will not import.\n"
            f"  package import said: {_e!r}\n"
            f"  file import said:    {_e2!r}\n"
            f"This module uses package-relative imports, so it needs the "
            f"repo root importable. Run it as\n"
            f"    PYTHONPATH={_REPO_ROOT} python3 "
            f"src/counter_strike_render/gpu_render.py ...\n"
            f"or from {_REPO_ROOT} as\n"
            f"    python3 -m counter_strike_render.gpu_render ...")
# `--help` MUST NOT REQUIRE THE RASTERISER. This import is at module scope
# and the parser is built below it, so a box without nvdiffrast could not
# read the help text at all -- which is how the reported "--help crashes"
# stayed unverified end to end: fixing the format bug moved the failure
# four lines down to here, and I reported it fixed having only ever seen it
# reach this line.
#
# Under -h/--help the module binds a stub that raises on first USE. Help
# prints; any real run still dies loudly at the first rasteriser call, with
# the install instruction, and nothing silently renders without it.
if any(a in ("-h", "--help") for a in sys.argv[1:]):
    try:
        import nvdiffrast.torch as dr
    except ImportError as _nde:
        class _NvdiffrastMissing:
            def __getattr__(self, name):
                raise SystemExit(
                    "REFUSED: nvdiffrast is not installed, so `dr.%s` cannot "
                    "run. It was stubbed only because this process was "
                    "started with --help. Install nvdiffrast to render:\n"
                    "    pip install "
                    "git+https://github.com/NVlabs/nvdiffrast.git\n"
                    "(import said: %r)" % (name, _nde))
        dr = _NvdiffrastMissing()
else:
    import nvdiffrast.torch as dr

# The csgo_character / csgo_eyeball / csgo_customglove pixel kernels live in
# src/counter_strike_render/passes/ rather than here, because THIS file cannot be
# imported -- it builds its ArgumentParser at module scope -- and the
# conformance registry has to resolve `impl=` against the real callable to
# A/B it against the decompiled reference. The kernels are backend-agnostic
# (numpy for the registry, torch here), so there is ONE copy of each
# expression and the number the registry reports is about the code this file
# runs. See passes/character.py's header.
# (repo root already on sys.path -- see _REPO_ROOT above)
from counter_strike_render.passes import character as _ps_char      # noqa: E402
from counter_strike_render.passes import customglove as _ps_glove    # noqa: E402
from counter_strike_render.passes import eyeball as _ps_eye        # noqa: E402

# The csgo_weapon / csgo_legs_prepass term expressions live in an importable
# module so the conformance registry can A/B them against the decompiled
# reference. THE RENDERER CALLS THAT MODULE -- it does not keep a second
# copy. A registry that measures a parallel transcription nobody renders
# with is the exact shape of correctness this project has faked before.
# This file is run as a script from arbitrary working directories AND is
# deployed flat onto the render host, so the root that holds the `harness`
# package is LOCATED rather than assumed. Assuming three-directories-up is
# right in the repo and wrong in every deployment, and the failure mode is
# an ImportError at startup on the host only.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.dirname(os.path.dirname(_HERE)), _HERE,
              os.path.dirname(_HERE)):
    if os.path.isfile(os.path.join(_cand, "src", "counter_strike_render", "passes",
                                   "weapon.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
else:
    raise ImportError(
        "src/counter_strike_render/passes/weapon.py not found from %s -- the "
        "csgo_weapon term expressions live there and the renderer calls "
        "them; a deployment that drops the package renders a different "
        "shader, so this refuses rather than falling back to a copy."
        % _HERE)
from counter_strike_render.passes import weapon as wpn      # noqa: E402
from counter_strike_render.passes.loopfam import ao as _lf_ao  # noqa: E402
from counter_strike_render.passes.loopfam import depth as _lf_depth  # noqa: E402
from counter_strike_render.passes.loopfam import filters as _lf_filters  # noqa: E402
from counter_strike_render.passes.loopfam import tables as _lf_tables  # noqa: E402
from counter_strike_render.passes.loopfam import effects as _lf_effects  # noqa: E402
from counter_strike_render.passes.loopfam import shadows as _lf_shadows  # noqa: E402
from counter_strike_render.passes.loopfam import volumetrics as _lf_vol  # noqa: E402
from counter_strike_render.passes.loopfam import binner as _lf_binner  # noqa: E402
from counter_strike_render.post import chain as _po_chain  # noqa: E402
from counter_strike_render.post import cmaa2 as _po_cmaa  # noqa: E402
from counter_strike_render.post import fsr as _po_fsr  # noqa: E402
from counter_strike_render.post import msaa_resolve as _po_msaa  # noqa: E402
from counter_strike_render.post import scene_format as _po_fmt  # noqa: E402
from counter_strike_render.post import detail_tiers as _po_det  # noqa: E402
from counter_strike_render.post import texture_filtering as _po_tfq  # noqa: E402
from counter_strike_render.post import tonemap as _po_tone  # noqa: E402
from counter_strike_render.post import video_config as _po_vcfg  # noqa: E402
from counter_strike_render.shading import environment as _sh_env2  # noqa: E402

S = 0.0254
CLUSTER_METERS = 8.0
CHUNK = 32
LOD_CELLS = (0.08, 0.32)          # meters; L0 is the original mesh
OCCLUDER_MIN_AREA = 0.75          # m^2; tris this large form the depth pass
HIZ_W, HIZ_H = 128, 72

parser = argparse.ArgumentParser()
parser.add_argument("--world", default=None,
                    help="world pack .pt. With --content-root this is "
                         "RESOLVED (worlds/<map>.pt) and the flag is an "
                         "explicit override; without either the run "
                         "refuses post-parse.")
parser.add_argument("--content-root", default=None,
                    help="root of the FULL content tree (worlds/ with all "
                         "sidecars beside each pack by stem, pm_bundles/, "
                         "weapons/, vm/). Every input family resolves from "
                         "here by --map; each resolution PRINTS; a missing "
                         "family REFUSES BY NAME unless waived by name. "
                         "OWNER RULE 2026-08-13: render nodes carry the "
                         "whole tree -- no hand-selected asset subsets.")
parser.add_argument("--map", default=None,
                    help="map name for --content-root resolution. FITTED "
                         "by the caller from recording data "
                         "(iji_model.map_identity) -- never typed by hand.")
parser.add_argument("--camera-json", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tick-begin", type=int, required=True)
parser.add_argument("--tick-end", type=int, required=True)
parser.add_argument("--fps", type=int, default=32)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--supersample", type=int, default=1)
parser.add_argument("--occlusion", action="store_true", help="Hi-Z pass; currently net-negative, see GPU_RENDER_BACKEND.md")
parser.add_argument("--no-lod", action="store_true")
parser.add_argument("--lod-weld", choices=("derived", "material",
                                           "position"),
                    default="material",
                    help="How the LOD decimation may weld vertices (#88). "
                         "Three tiers, correctness-ordered, measured on "
                         "de_mirage against the --no-lod reference "
                         "(strongly-wrong pixels over 8 frames, and "
                         "triangles at cells 0.08m/0.32m):\n"
                         "  derived  position + material + the "
                         "derived texture bound |d(uv)| <= density x cell. "
                         "12,042 wrong px; 818,393 / 391,791 tris.\n"
                         "  material (DEFAULT) position + material only. "
                         "67,276 wrong px; 552,160 / 200,305 tris.\n"
                         "  position pre-#88. 76,762 wrong px; 548,772 / "
                         "193,641 tris. TEXTURE-BLEEDS AT DISTANCE: it "
                         "merges vertices of different materials and puts "
                         "an awning's texture on a stone wall.\n"
                         "`derived` costs +49%%/+102%% triangles over "
                         "`position` and measured WITHIN NOISE in render "
                         "time on both a close-range and a distant-geometry "
                         "fixture (raster_shade 1.832->1.862 s and "
                         "0.0609->0.0594 s; the fps column disagreed in "
                         "direction both times). The geometry cost is real "
                         "and the time cost is not measurable at n=1. The "
                         "owner's rule for that outcome is explicit: an "
                         "AMBIGUOUS throughput measurement leaves the "
                         "default at `material` and scale-banks `derived`, "
                         "because the swarm answers throughput at the scale "
                         "where the question actually lives. Renders that "
                         "serve a CORRECTNESS judgement pass --lod-weld "
                         "derived; that decoupling is the whole point of "
                         "the flag.")
parser.add_argument("--light-basis", type=int, default=None,
                    help="REFIT HARNESS. Render one basis of the lighting "
                         "solution: 0 AMBIENT_GROUND, 1 AMBIENT_SKY, "
                         "2 SUN_COLOR, 3 SKY_TOP, 4 SKY_HORIZON, 5 the "
                         "constant residue (emissive, fog). The post chain "
                         "is LINEAR in all five before the sRGB encode, so "
                         "these six HDR dumps span every lighting solution "
                         "reachable by this config -- and because they come "
                         "out of the real render path, they carry the "
                         "config's own AO, indirect, IBL and fog exactly "
                         "rather than through a proxy model.")
parser.add_argument("--lod-uv-cell", type=float, default=0.0,
                    help="UV-AWARE LOD. The index-buffer LOD clusters "
                         "vertices by WORLD POSITION only and rebuilds each "
                         "face from its cell representative, so a face "
                         "inherits whatever texture coordinate that "
                         "representative happened to carry. Measured on "
                         "this pack: at 0.08m cells 79.9%% of vertices are "
                         "remapped, world displacement stays at a 2.8cm "
                         "median -- geometry looks right -- while the UV "
                         "moves by a median 0.257 and by MORE THAN A FULL "
                         "TILE on 11.4%% of them. That scrambles surface "
                         "texture while preserving silhouettes, which is "
                         "exactly the structural signature we are chasing. "
                         "Setting this adds the texture coordinate to the "
                         "clustering key at this cell size, so vertices "
                         "merge only when they agree in position AND uv. "
                         "0 = off (position-only, the old behaviour).")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--gbuffer", default=None,
                    help="dump per-frame albedo/up/ndl/fg npz for lighting fit")
parser.add_argument("--lightmap", default=None,
                    help="directory holding directional_irradiance.png")
parser.add_argument("--lm-scale", type=float, default=1.0)
# --- CS2 baked indirect (bounce) irradiance -----------------------------
# maps/de_inferno/lightmaps/irradiance.vtex is 8192^2 BC6H HDR and is the
# term CS2 calls lighting.DiffuseIndirect: bounce + baked static lights,
# sampled by the lightmap UV set (TEXCOORD_1). The sun's DIRECT diffuse is
# NOT in it -- ValveResourceFormat's lighting.slang applies the sun at
# runtime, gated by direct_light_shadows -- so it is additive with ours.
parser.add_argument("--irr-npy", default="auto",
                    help="(8192,8192,3) fp16 .npy of the baked irradiance "
                         "lightmap, decoded from irradiance.vtex_c. "
                         "DEFAULT 'auto' = <world pack>.irradiance.npy "
                         "beside the pack, because the baked lighting "
                         "belongs to the map and should travel with it. "
                         "'none' disables. DEFAULTED ON 2026-08-08: it was "
                         "absent from every scored run, which sent all 37%% "
                         "of lightmapped world geometry down the probe "
                         "branch of gt_select_baked(); loading it moved "
                         "947 ego01 frames KL_rgb 5.5778 -> 2.4383, ncc "
                         "0.1908 -> 0.2366 and nrmse 1.4151 -> 1.3748 -- "
                         "all three together, which no other arm has done. "
                         "THIS IS A COMPARISON BOUNDARY: scores taken "
                         "before this default are not comparable with "
                         "scores taken after it.")
parser.add_argument("--irr-gain", type=float, default=1.0,
                    help="DEAD: read only inside _post_chain, which is "
                         "unreachable (its sole caller _single_shade is "
                         "never called). Bit-identical across 0.25..1.0. "
                         "Use --irr-scale, which acts on the live path.")
parser.add_argument("--require-pack-generation", default=None,
                    help="refuse unless the world pack carries this "
                         "pack_generation. For scoring across maps, where "
                         "a mixed set produces a well-formed table nobody "
                         "can reconcile afterwards.")
parser.add_argument("--irr-scale", type=float, default=1.0,
                    help="scale the baked irradiance at load, before the "
                         "volume is built from it, so both products stay "
                         "consistent. This is the live-path lever.")
parser.add_argument("--irr-mode", choices=("modulate", "replace"),
                    default="replace",
                    help="'replace': ambient := baked irradiance (CS2's own "
                         "structure, but its absolute scale must then be "
                         "right). 'modulate': ambient *= irr / mean(irr), "
                         "which injects only the SPATIAL variation and "
                         "leaves the already-fitted exposure alone")
parser.add_argument("--irr-mip", type=int, default=2,
                    help="box-downsample the 8192^2 atlas by 2^N before "
                         "sampling. The lightmap is ~16 luxel/m; at 960x540 "
                         "a wall 10m out gets <2 px/m, so point-sampling it "
                         "aliases into salt-and-pepper. Indirect light is "
                         "low-frequency, so the mip costs nothing")
parser.add_argument("--irr-clamp", type=float, default=4.0,
                    help="cap on the modulation factor; the atlas is HDR "
                         "and peaks at 38x its own median")
# --- indirect for everything the lightmap does not cover ----------------
# Only 6.4% of rendered pixels are lightmapped world geometry (measured
# with --debug-lighting lmvalid). The other 94% -- every prop, model and
# piece of foliage -- is what CS2 lights from env_light_probe_volume_*,
# and it is where most of the missing bounce lives. Rather than decode
# those volumes, rebuild the same quantity from the lightmap we have:
# 522k lightmapped vertices carry (world position, baked irradiance), a
# dense sampling of the map's indirect field. Splat them into a voxel
# grid and push-pull fill it, and every pixel can look the field up by
# position. Same source data as the probes, reconstructed rather than
# decoded -- so it is CS2's bounce, but resampled by us, not Valve's
# octree read back verbatim.
parser.add_argument("--irr-vol", type=float, default=0.5,
                    help="voxel size in metres for the reconstructed "
                         "irradiance volume (0 = off). Needs --irr-npy")
parser.add_argument("--irr-vol-gain", type=float, default=1.25,
                    help="FITTED COMPENSATION for a known-missing term, not a physical constant: probe volumes are blocked and only 6.4pct of pixels are lightmapped, so the reconstructed field under-represents real bounce. Still UNDER-shoots: p10 luminance 0.135 vs GT 0.270.")
# world.vwrld_c: m_worldLightingInfo.m_vLightmapUvScale = [1.14284, 1.14284].
# Source 2 stores TEXCOORD_1 pre-scaled and the vertex shader does
# vLightmapUVScaled = vLightmapUV * g_vLightmapUvScale. Confirmed against
# the glTF: p99 of every lightmapped primitive's max U is 0.8749 and
# 1/1.14284 = 0.875013. Without this the atlas lookup lands on an
# unrelated chart, which is why the baked shadow mask never helped.
parser.add_argument("--lm-uv-scale", type=float, default=1.14284)
parser.add_argument("--sun-vis", choices=("raw", "invert"), default="raw",
                    help="direct_light_shadows.r is a SHADOW amount in "
                         "Valve's shader (visibility = 1 - dot(dlsh, mask)); "
                         "'raw' keeps the pre-existing interpretation")
# The pre-existing lookup sampled the atlas at 1-v. A 3D-coherence sweep
# over {scale} x {flip} x {swap} (bounce_work/uvsweep.py) puts unflipped V
# at within-voxel irradiance variance 0.126 against 0.32-0.38 for every
# other convention and 0.425 for random UVs, so V is NOT flipped. The flip
# was corrupting the baked shadow mask as well.
parser.add_argument("--lm-flip-v", type=int, default=0)
parser.add_argument("--debug-lighting",
                    choices=("off", "irr", "lmvalid", "skyvis", "ssao",
                             "lights", "dirocc", "ss-shadow"),
                    default="off",
                    help="bypass albedo: show the sampled irradiance, or "
                         "green/red lightmap-UV validity")
# --- SKY / AMBIENT VISIBILITY (build_skyvis.py) --------------------------
# The hemispheric AMBIENT term is applied at full strength everywhere,
# including inside closed rooms, because an analytic ambient has no
# notion of enclosure. We already gate the SUN by CS2's baked
# direct_light_shadows; nothing gated the ambient. This is the gate:
# a purely GEOMETRIC world-space field of "what fraction of the upper
# hemisphere is open sky", ray-marched offline against a voxelisation of
# the packed mesh. Nothing in it is fitted to ground truth.
parser.add_argument("--skyvis", default=None,
                    help=".npz from build_skyvis.py: world-space sky "
                         "visibility field sampled per pixel")
parser.add_argument("--skyvis-frac", type=float, default=1.0,
                    help="FITTED SPLIT: fraction of the ambient slot that "
                         "is SKY light (occludable). The remainder is "
                         "bounce, which does NOT vanish indoors -- GT's "
                         "interior is 0.17, not 0. ambient *= "
                         "(1-f) + f*skyvis")
parser.add_argument("--skyvis-ref", type=float, default=0.0,
                    help="renormalisation reference for --skyvis-renorm: "
                         "the mean of this field over the RENDERED pixels "
                         "of this pose set, which the renderer prints on "
                         "every run. MANDATORY with --skyvis-renorm 1. It "
                         "is NOT the npz's `ref` -- that is the field's "
                         "global marched-cell mean over the whole map "
                         "(0.3620 for the 30m field) and is ~3x the "
                         "rendered-pixel mean (0.1179), because most of "
                         "the map is never on screen. A MEASURED constant, "
                         "never fitted to GT")
parser.add_argument("--skyvis-renorm", type=int, default=0,
                    help="divide the visibility by --skyvis-ref so the "
                         "modulation averages 1 over the field. Makes the "
                         "term exposure-neutral (pure spatial variance); "
                         "off, it is a net darkening as well")
parser.add_argument("--skyvis-clamp", type=float, default=4.0)
parser.add_argument("--skyvis-noocc", type=int, default=0,
                    help="CONTROL: force visibility to 1 everywhere, "
                         "keeping the sky/bounce chroma split. Isolates "
                         "how much of a chroma result needs the "
                         "OCCLUSION and how much is a global tint")
parser.add_argument("--bounce-chroma", "--skyvis-chroma", type=float,
                    default=0.0, dest="bounce_chroma",
                    help="tint the ambient by the chroma of the "
                         "RECONSTRUCTED indirect volume. --irr-vol already "
                         "samples RGB but the shader collapses it with "
                         ".mean(-1); this is that discarded colour, "
                         "luminance-normalised so it cannot act as a gain. "
                         "Needs --irr-npy ONLY -- it does not read the "
                         "sky-visibility field and does not need --skyvis. "
                         "It shipped under --skyvis-chroma, which is kept "
                         "as an alias; that prefix implied a dependency "
                         "that never existed and cost a round of "
                         "conflating the two terms. When --skyvis IS "
                         "given, the tint applies to the (1-frac) bounce "
                         "half exactly as before")
parser.add_argument("--skyvis-sky-chroma", type=float, default=0.0,
                    help="tint the SKY half toward (1.00,1.07,1.38), the "
                         "normalised up/sky slot DECODED from CS2 probe "
                         "volumes by another agent -- our fitted "
                         "AMBIENT_SKY is (1.00,0.86,0.93), i.e. not blue, "
                         "because the one slot absorbed sky AND bounce")
parser.add_argument("--skyvis-flip", default="",
                    help="FALSIFIER: mirror the field along x/y/z before "
                         "use. Identical marginal histogram, wrong "
                         "PLACES. If the metric gain survives this, the "
                         "term is not carrying spatial information and "
                         "the result is an artefact")
parser.add_argument("--skyvis-offset", type=float, default=0.35,
                    help="metres to push the lookup along the shading "
                         "normal before sampling the field. A surface "
                         "point sits ON the boundary, so a raw trilinear "
                         "tap averages the free cell in front with the "
                         "SOLID cell behind and halves the visibility of "
                         "every open wall and floor. 0 = no offset")
parser.add_argument("--skyvis-cos", type=int, default=0,
                    help="weight the visibility toward the surface normal "
                         "(N.up), on top of the ambient's own up-ramp")
parser.add_argument("--skyvis-sun", type=float, default=0.0,
                    help="DIAGNOSTIC: also gate the sun term by skyvis^this")
# --- SSAO: Scalable Ambient Obscurance (McGuire et al. 2012) -----------
# CS2 ships FIVE ssao_* shaders in game/core/shaders_vulkan_dir.vpk:
#   ssao_scalable_ambient_obscurance_vulkan_50_ps  <- the estimator
#   ssao_convert_depth_vulkan_50_ps                <- hardware depth -> camera-space Z
#   ssao_downsample_depth_vulkan_50_ps             <- the CSZ mip chain
#   ssao_bilateral_blur_vulkan_50_ps               <- edge-aware blur
#   ssao_vulkan_50_ps
# so the named algorithm is SAO, not a generic hemisphere AO. All four
# stages are implemented here. Deviation from the paper: samples are
# unprojected from a POINT-SAMPLED world-position mip chain rather than
# from a camera-space-Z chain plus an inverse projection. The estimator
# only ever consumes v = Q - C and v.n, both rotation-invariant, so the
# two agree by construction; it trades 3 channels of bandwidth for the
# unprojection. Point sampling (not averaging) is kept, because averaging
# two depths across a silhouette invents a surface that is at neither.
# ---------------------------------------------------------------------
# CLUSTERED ANALYTIC LIGHTS
#
# de_inferno ships 208 light entities (121 light_omni2, 61 light_rect,
# 25 light_barn, 1 light_environment = the sun, already implemented).
# They are the only unimplemented term supplying LOCAL, spatially
# varying light, and the standing dark-band defect (0.10-0.20 luminance,
# level ratio ~0.67) is a local-light-shaped hole.
#
# PROVENANCE, and it must be quoted with the numbers: the light
# PARAMETERS are DECODED from `default_ents.vents_c`. The falloff
# FUNCTION is RECONSTRUCTED -- the extracted SPIR-V is symbol-stripped
# and `grep -ci light` returns 0, so no engine code corroborates any of
# the forms below. `--lights-falloff` therefore exists to sweep the
# family rather than to configure one believed-correct choice.
parser.add_argument("--lights", default=None,
                    help="analytic light entity JSON (assets/inferno_lights"
                         ".json). Positions are WORLD/glTF convention "
                         "(Y-up), the same as `vertices`; camera poses are "
                         "SOURCE Z-up and are converted at ingest, so no "
                         "conversion happens here")
parser.add_argument("--lights-gain", type=float, default=0.10,
                    help="global scale on the analytic light term. This is "
                         "NOT a free fudge: the four brightness fields are "
                         "in absolute photometric units and this renderer's "
                         "lighting is a fit in arbitrary units, so SOME "
                         "constant is required. 0.10 is the prior work's "
                         "operating point and is kept as the default so a "
                         "rebuild is comparable")
parser.add_argument("--lights-unit",
                    choices=("candelas", "nits", "lumens", "legacy", "ev",
                             "selector"),
                    default="candelas",
                    help="which brightness field is authoritative. The four "
                         "are NOT unit conversions of each other (nits/"
                         "candelas spans 26.8..74.9 across the brightest "
                         "lights), so exactly one is right PER LIGHT and a "
                         "global choice is provably wrong for whichever "
                         "group disagrees. `selector` reads the per-light "
                         "`brightness_units` field: 1 -> nits, 0 -> 2^"
                         "brightness. That mapping is an INFERENCE from the "
                         "two groups' distributions, not read from the "
                         "engine -- it is here to be TESTED, not trusted. "
                         "`candelas` is what the lost implementation used")
parser.add_argument("--lights-falloff",
                    choices=("invsq-win", "invsq-clip", "invsq-soft",
                             "linear", "smooth2", "exp"),
                    default="invsq-win",
                    help="distance falloff family. RECONSTRUCTED -- see the "
                         "provenance note above. invsq-win is inverse-square "
                         "with the standard (1-(d/R)^4)^2 range window")
parser.add_argument("--lights-angular",
                    choices=("none", "cosine", "cone", "cone-barn"),
                    default="cone-barn",
                    help="angular falloff. `none` = every light omni "
                         "(ablation). `cosine` = one-sided cosine on `dir` "
                         "for rect/barn. `cone` = adds the omni2 spot cone "
                         "from inner_angle/outer_angle/skirt. `cone-barn` "
                         "adds the reconstructed barn-door gate from "
                         "size_params")
parser.add_argument("--lights-cone-angle", choices=("full", "half"),
                    default="full",
                    help="whether inner_angle/outer_angle are FULL cone "
                         "angles (half-angle = a/2) or already half-angles. "
                         "Undetermined from the entity dump; both are "
                         "renderable and this selects which is measured")
parser.add_argument("--lights-select",
                    choices=("direct1", "direct", "all"), default="direct1",
                    help="which entities light. The JSON's `enabled` is True "
                         "for all 207 analytic lights, so it is NOT the "
                         "gate; `directlight` is (184x '1', 6x '3', 17x "
                         "'0'). direct1 = the 184 that the prior measured "
                         "numbers used")
parser.add_argument("--lights-shadow", default=None,
                    help="bit-packed occupancy grid npz from build_skyvis.py "
                         "--occ-out. Enables per-light shadows by marching "
                         "pixel->light through the grid, instead of "
                         "rasterising 184 depth maps")
parser.add_argument("--lights-shadow-bias", type=float, default=1.0,
                    help="normal offset before marching, in VOXELS, before "
                         "the 1/(N.L) scaling. Conservative voxelisation "
                         "inflates every surface to a half-voxel shell, so "
                         "at grazing incidence a point marches inside its "
                         "OWN shell and a FIXED offset cannot fix that -- "
                         "the bias has to scale as 1/(N.L)")
parser.add_argument("--lights-shadow-step", type=float, default=0.75,
                    help="march step as a fraction of the voxel size")
parser.add_argument("--lights-shadow-maxsteps", type=int, default=96,
                    help="cap on march steps; beyond this the ray is treated "
                         "as UNOCCLUDED (fail-open, so the cap can only "
                         "under-shadow, never invent shadow)")
parser.add_argument("--lights-enclosure", type=float, nargs=2,
                    default=None, metavar=("MIN", "MAX"),
                    help="keep only lights whose OWN sky visibility lies in "
                         "[MIN,MAX], marched from the light position through "
                         "the occupancy grid (requires --lights-shadow). "
                         "SEPARATES TWO CAUSES the frame split cannot: an "
                         "exterior surface can be over-lit either because "
                         "an INTERIOR light leaks to it through geometry "
                         "the grid fails to close, or because an EXTERIOR "
                         "light is legitimately reaching it and is simply "
                         "too bright. `0 0.15` keeps only enclosed lights, "
                         "so any remaining exterior excess is leakage; "
                         "`0.15 1` keeps only open ones. Purely geometric, "
                         "nothing here touches ground truth")
parser.add_argument("--lights-enclosure-dirs", type=int, default=64)
parser.add_argument("--lights-neutral", action="store_true",
                    help="EXPOSURE-NEUTRAL: give the frame-mean added "
                         "irradiance back off the ambient, so only the "
                         "SPATIAL STRUCTURE of the term is measured. The "
                         "fitted AMBIENT_SKY was fitted on a render with no "
                         "analytic lights and has already absorbed their "
                         "average, so raw addition double-counts by "
                         "construction")
parser.add_argument("--lights-min-contrib", type=float, default=1e-4,
                    help="pixels whose unshadowed contribution from a light "
                         "is below this, AFTER --lights-gain and so on the "
                         "same scale as the ~0.5 fitted ambient, are "
                         "dropped and not marched. Pure cost control; "
                         "raising it can only remove light, never add it")
parser.add_argument("--lights-sabotage", choices=("off", "shuffle", "sign"),
                    default="off",
                    help="POSITIVE CONTROL. `shuffle` permutes light "
                         "positions among themselves, keeping the exact "
                         "same energy, count, colours and falloff but "
                         "destroying placement. `sign` negates the term. "
                         "If the metrics do not move for these, the harness "
                         "cannot see the term and every null it produces is "
                         "void")
parser.add_argument("--lights-dump", default=None,
                    help="write per-frame added-irradiance statistics to "
                         "this JSON, for the interior/exterior split")
parser.add_argument("--ssao", action="store_true",
                    help="Scalable Ambient Obscurance on the ambient term")
parser.add_argument("--ssao-radius", type=float, default=1.0,
                    help="world-space occlusion radius in metres")
parser.add_argument("--ssao-samples", type=int, default=11)
parser.add_argument("--ssao-turns", type=int, default=7,
                    help="spiral turns; 7 is the paper's value for 11 taps")
parser.add_argument("--ssao-intensity", type=float, default=1.0,
                    help="sigma in the SAO estimator")
parser.add_argument("--ssao-bias", type=float, default=0.01,
                    help="metres; rejects self-occlusion from depth noise")
parser.add_argument("--ssao-strength", type=float, default=1.0,
                    help="how much of the computed obscurance to apply to "
                         "the ambient term (1 = fully)")
parser.add_argument("--ssao-mips", type=int, default=3,
                    help="ssao_downsample_depth levels")
parser.add_argument("--ssao-blur", type=int, default=4,
                    help="ssao_bilateral_blur radius in taps (0 = off)")
parser.add_argument("--ssao-blur-scale", type=int, default=2,
                    help="tap stride for the blur")
parser.add_argument("--ssao-blur-edge", type=float, default=2.0,
                    help="bilateral edge sharpness; larger = respects "
                         "depth discontinuities more strictly")
parser.add_argument("--ssao-geo-normal", type=int, default=1,
                    help="feed SAO the GEOMETRIC (per-face) normal "
                         "instead of the tangent-space normal-mapped "
                         "one. A normal map perturbs n at texel "
                         "frequency; v.n then fires on flat ground and "
                         "SAO paints obscurance onto surfaces that have "
                         "no occluder near them at all")
parser.add_argument("--ssao-sun", type=float, default=0.0,
                    help="DIAGNOSTIC: also gate the sun by AO^this. CS2 "
                         "applies SSAO to indirect only")
# --- ENGINE PER-VIEW RENDER TARGETS -- set 1, binding 3 -----------------
# Every texture in the six extracted csgo_* pixel shaders is BINDLESS: one
# aliased descriptor array at set 4 / binding 46. The image descriptor
# therefore carries no information at all, and a render-target read is
# type-identical to a material-texture read -- which is exactly why
# memsig.py, classifying by sampler TYPE, cannot see this root class.
# The discriminating fact is WHICH constant buffer the 32-bit handle comes
# from and at WHICH byte offset (OpAccessChain -> OpVariable ->
# DescriptorSet/Binding -> MemberDecorate Offset; see
# src/counter_strike_render/vcs/fetch_provenance.py, no name matching anywhere).
#
# The per-view CB is set 1 / binding 3. Its identity is pinned by
# neighbours at IDENTICAL byte offsets across the shaders: offset 72 ->
# samplerCube (sky), offset 104 -> samplerCubeArray (IBL specular). Those
# two this renderer already supplies (--cube-probes / --ibl-spec).
# FOUR MORE were supplied by nothing here at all:
#
#  off  what it is                        reference line (env_blend s0/d0)
#  ---  -------------------------------   --------------------------------
#   76  low-res DIRECTIONAL-OCCLUSION      :1239  textureLod(...,uv,0.0)
#       target, 4 channels, one per
#       engine ambient-basis direction
#   80  the matching low-res DEPTH         :1212  textureGather(...) - z
#       target: gathered as 4 texels and
#       differenced against gl_FragCoord.z
#       to pick the nearest-depth texel
#   88  full-res SCREEN-SPACE SHADOW       :1322  min(cascade, tex(...).z)
#       target, channel .z, at
#       gl_FragCoord.xy * invViewport
#  116  per-view WEATHER RIPPLE map,       :1014, :1072, :1081
#       three reads
#
# The same four slots, by byte offset, in the other five shaders:
#   csgo_environment        76 :793   80 :766   88 :876   116 :566/624/633
#   csgo_complex            76 :523   80 :496   88 :606   116 absent from CB
#   csgo_foliage            76 :358   80 :331   88 :441   116 absent from CB
#   csgo_static_overlay     76 absent 80 absent 88 :323   116 absent
#   csgo_glass              76 absent 80 absent 88 :618   116 absent
# so 88 is read by ALL SIX, 76/80 by four of six (not overlay, not glass),
# and 116 by the two `environment` shaders.
#
# Defaults are ON. The reference gates 76/80/88 on per-view enable bits
# (_5037._m0.x / .y), which are set whenever the engine has produced the
# targets; a default that leaves them off would read as tested while never
# executing. --rt-reach measures the executed fraction of every one of the
# four reads.
parser.add_argument("--dirocc", type=int, default=1,
                    help="per-view offsets 76 + 80: build the low-res "
                         "4-channel directional-occlusion target and the "
                         "low-res depth target, gather the depth, pick the "
                         "nearest-depth texel and resolve the occlusion "
                         "against the engine ambient basis. glsl:1206-1244")
parser.add_argument("--dirocc-scale", type=int, default=2,
                    help="downsample factor of BOTH per-view targets "
                         "(_5037._m15 = 1/this). The nearest-depth gather "
                         "is only meaningful at >1; at 1 the four gathered "
                         "texels are the pixel's own depth and the "
                         "selector is a no-op, so 1 is a diagnostic value")
parser.add_argument("--dirocc-basis", default="45,135,225,315",
                    help="azimuths in degrees of the four engine ambient "
                         "basis directions (_5538._m1._m0[0..3].xyz). NOT "
                         "recoverable from the bytecode -- it is engine CB "
                         "data. The shader's own use of them "
                         "(normalize(vec3(b.xy, 0.25))) says they are four "
                         "mostly-horizontal directions with a fixed "
                         "up-tilt, which is what this default is.")
parser.add_argument("--dirocc-tilt", type=float, default=0.25,
                    help="up-component of the basis directions before "
                         "normalisation; the 0.25 is a LITERAL in the "
                         "shader (glsl:1237) so only the azimuths are free")
parser.add_argument("--dirocc-view-weights", default="1,1,1,1",
                    help="_5538._m2.xyzw, the per-basis scale on the "
                         "view-facing term at glsl:1237. Engine CB data.")
parser.add_argument("--dirocc-eps", type=float, default=0.05,
                    help="_5538._m3.x, the additive epsilon of the "
                         "weighted resolve at glsl:1239. Engine CB data.")
parser.add_argument("--dirocc-radius", type=float, default=1.0,
                    help="world-space obscurance radius of the target's "
                         "producer pass, metres")
parser.add_argument("--dirocc-samples", type=int, default=11)
parser.add_argument("--dirocc-turns", type=int, default=7)
parser.add_argument("--dirocc-bias", type=float, default=0.01)
parser.add_argument("--dirocc-intensity", type=float, default=1.0)
parser.add_argument("--dirocc-mips", type=int, default=3)
parser.add_argument("--ss-shadow", type=int, default=1,
                    help="per-view offset 88: build the screen-space shadow "
                         "target and min()-combine channel .z into the sun "
                         "visibility, as all six shaders do. glsl:1322")
parser.add_argument("--ss-shadow-scale", type=int, default=1,
                    help="downsample factor of the offset-88 target. The "
                         "reference reads it at gl_FragCoord.xy * "
                         "invViewport, i.e. full-res; >1 exercises the "
                         "bilinear read path")
parser.add_argument("--ss-shadow-steps", type=int, default=12,
                    help="ray-march steps of the target's PRODUCER pass "
                         "(a screen-space contact shadow traced through "
                         "the depth target toward the sun)")
parser.add_argument("--ss-shadow-len", type=float, default=0.6,
                    help="march length in metres")
parser.add_argument("--ss-shadow-thickness", type=float, default=0.5,
                    help="metres; an occluder further behind the ray than "
                         "this is treated as a separate surface, not a "
                         "blocker, so the trace cannot shadow through a "
                         "distant silhouette")
parser.add_argument("--ss-shadow-bias", type=float, default=0.02,
                    help="metres of view-depth bias against self-shadowing")
parser.add_argument("--weather", type=int, default=1,
                    help="per-view offset 116: the weather ripple map, "
                         "three reads. glsl:1014 (world-projected, .w is a "
                         "coverage noise) and glsl:1072/1081 (two "
                         "wind-scrolled ripple layers, .xy a tangent-space "
                         "normal, .z a ring height)")
parser.add_argument("--weather-ripple", default=None,
                    help="the offset-116 texture. When absent a "
                         "deterministic procedural RGBA stands in, because "
                         "the engine's own texture is not in any of the six "
                         "shaders -- the READS are exact, the CONTENT is "
                         "not the engine's")
parser.add_argument("--weather-wetness", type=float, default=0.0,
                    help="_5037._m11, per-view wetness. 0 = a dry map, "
                         "which is de_inferno's state. This is the "
                         "REFERENCE'S OWN data gate: at 0 the shader still "
                         "issues the glsl:1014 read (it is above the "
                         "wetness test) and skips :1072/:1081, exactly as "
                         "this port does.")
parser.add_argument("--weather-snow", type=float, default=0.0,
                    help="_5037._m12")
parser.add_argument("--weather-rain", type=float, default=0.0,
                    help="_5037._m8, rain-ripple amplitude; gates the "
                         "3-iteration procedural ring loop at glsl:1118")
parser.add_argument("--weather-wind", type=float, default=0.0,
                    help="_5037._m9, wind strength; with wetness it gates "
                         "the two ripple-map reads at glsl:1072/1081")
parser.add_argument("--weather-wind-angle", type=float, default=0.0,
                    help="_5037._m10, wind direction in turns (the shader "
                         "multiplies it by 2*pi at glsl:1059)")
parser.add_argument("--weather-tickrate", type=float, default=128.0,
                    help="ticks per second used to turn the camera-path "
                         "tick into _4459._m0, the time the two ripple "
                         "layers scroll by")
parser.add_argument("--weather-enable", type=int, default=1,
                    help="_5618._m88, the material's weather-enable int. "
                         "The world pack carries no such per-material "
                         "flag, so this is uniform over materials.")
parser.add_argument("--weather-wet-strength", type=float, default=1.0,
                    help="_5618._m70, the material's puddle strength")
parser.add_argument("--weather-cover-strength", type=float, default=1.0,
                    help="_5618._m78, the material's damp-coverage strength")
parser.add_argument("--weather-ripple-strength", type=float, default=1.0,
                    help="_5618._m79, the material's rain-ripple strength")
parser.add_argument("--weather-flow-strength", type=float, default=1.0,
                    help="_5618._m80, the material's wind-flow strength")
parser.add_argument("--weather-rough-wet", type=float, default=0.08,
                    help="_5618._m74, the roughness a fully wet surface "
                         "takes")
parser.add_argument("--weather-edge", type=float, default=0.05,
                    help="_5618._m75, puddle edge softness")
parser.add_argument("--weather-damp-strength", type=float, default=1.0,
                    help="_5618._m76")
parser.add_argument("--weather-height-offset", type=float, default=0.0,
                    help="_5618._m77, the height the puddle surface sits at")
parser.add_argument("--weather-drip-offset", type=float, default=0.0,
                    help="_5618._m72")
parser.add_argument("--weather-drip-strength", type=float, default=0.0,
                    help="_5618._m73")
parser.add_argument("--weather-tint", default="0.35,0.35,0.35",
                    help="_5618._m71, the colour a full puddle takes")
parser.add_argument("--weather-f0", type=float, default=0.04,
                    help="_5618._m86, the dry reflectance the wet path "
                         "raises to the literal 0.035 at glsl:1180")
parser.add_argument("--rt-reach", action="store_true",
                    help="DIAGNOSTIC: report, per engine per-view render "
                         "target, the fraction of SHADED pixels at which "
                         "the read actually executes, so a default that "
                         "silently disables one of them cannot be mistaken "
                         "for a tested path")
parser.add_argument("--shadow", type=int, default=0,
                    help="sun shadow-map resolution (0 = off)")
parser.add_argument("--shadow-bias", type=float, default=0.05)
parser.add_argument("--cc-mode", choices=("off", "tint", "sat", "full"),
                    default="tint",
                    help="Source per-layer colour correction to apply")
parser.add_argument("--no-cc", action="store_true", help="alias for off")
parser.add_argument("--blend-channel", type=int, default=0,
                    help="SUPERSEDED: which COLOR_0 channel to read when "
                         "--blend-weight is color0. COLOR_0 is NOT the "
                         "blend weight -- see the block below and use "
                         "--blend-weight vpaint. Kept only to reproduce "
                         "pre-fix arms.")
# --- where the csgo_environment_blend layer weight actually comes from -
# COLOR_0 is NOT it. Measured over the 271 M vertices of the 365
# csgo_environment_blend primitives (blend_diag5.py):
#   COLOR_0 is GREY (R==G==B to 3 dp), alpha identically 1.0, with modes
#   at exactly 1.0 / 0.8 / 0.6 / 0.4 and 66.9% of verts at 1.0 -- a
#   per-vertex TINT. 676 of the 2870 blend primitives have no COLOR_0 at
#   all. Reading it as the weight puts two thirds of every blended
#   surface on pure layer 2 and the rest on pure layer 1.
#   glTF `_TEXCOORD_4`.x is the vertex paint: byte-quantised (n/255),
#   continuous over 0..1, 92.9% exactly 0 and 0.35% exactly 1 over the
#   whole pack, non-zero on 12.9% of blend verts against 0.1% of every
#   other shader's -- i.e. the stream is used by exactly the shader that
#   needs it. corr(_TEXCOORD_4.x, COLOR_0.r) = 0.229.
parser.add_argument("--vpaint", default=None,
                    help="sidecar .pt from blend_sidecar.py holding the "
                         "per-vertex _TEXCOORD_4 vertex paint, index-"
                         "aligned to the pack")
parser.add_argument("--vpaint-remap", type=int, default=1,
                    help="apply the shader's clamp(w*1.1-0.05,0,1) dead "
                         "zone to the vertex paint (glsl:260)")
parser.add_argument("--blend-weight", choices=("color0", "vpaint"),
                    default="vpaint",
                    help="which vertex stream is the layer-2 weight. "
                         "DEFAULT IS vpaint, measured: on pose 0 of "
                         "de_inferno, against GT, color0 -> vpaint moves "
                         "ncc -0.1425 -> -0.0388 and the ground half "
                         "-0.3619 -> -0.0975, with MSE 0.04742 -> 0.04216. "
                         "The arms differ on 64.2%% of pixels, so this is "
                         "not a tweak. color0 is kept ONLY to reproduce "
                         "pre-fix arms and should be deleted once the "
                         "result is confirmed on more than one pose -- it "
                         "is a per-vertex TINT, not a weight, and the "
                         "block above records the measurement that "
                         "established that.")
# --- glTF baseColorFactor is a per-INSTANCE MODEL TINT ------------------
# 301 of the 365 csgo_environment_blend materials carry a strongly
# coloured pbrMetallicRoughness.baseColorFactor which matches NEITHER
# g_vTextureColorTint1 nor tint2 on any of the 308 that have one, and the
# same material NAME recurs with different factors -- so it is VRF baking
# a per-instance tint into a per-instance material variant, not a vmat
# parameter. This renderer multiplies it into BOTH layers unconditionally,
# which is where two long-standing signatures come from:
#   inferno_plaster_facade_01_basic_notint  factor [0.672 0.376 0.109]
#     -> the saturated ORANGE wall (its two layer textures are neutral,
#        mean sRGB [0.709 0.689 0.659] and [0.652 0.642 0.610]);
#   interno_tile_ground_blend_01[_notint]   factor [0.018 0.044 0.150]
#     -> the NAVY floor band (both layers are the same near-white tile,
#        mean sRGB [0.736 0.742 0.741]).
# Source 2 gates a model tint per layer on g_bModelTint<N>; both of those
# materials set it to 0 on both layers and are named `_notint`.
parser.add_argument("--model-tint", choices=("always", "flag", "blend-off"),
                    default="always",
                    help="'flag': apply glTF baseColorFactor to layer N "
                         "only where g_bModelTint<N> is not explicitly 0. "
                         "'blend-off': suppress it on every "
                         "csgo_environment_blend material, whose layer "
                         "colour is g_tColor<N> x g_vTextureColorTint<N> "
                         "and which has no parameter baseColorFactor "
                         "transcribes. Needs --model-tint-table")
parser.add_argument("--model-tint-table", default=None,
                    help="assets/modeltint.pt from blend_mtgate.py")
parser.add_argument("--light-dump", default=None,
                    help="DIAGNOSTIC .npz: per-pixel linear albedo, total "
                         "INDIRECT radiance and DIRECT sun radiance, so a "
                         "grey surface can be attributed to a grey albedo "
                         "or to grey light arriving on it")
parser.add_argument("--probe-dump", default=None,
                    help="DIAGNOSTIC .npz: per-frame material id and both "
                         "candidate blend weights, for attributing a "
                         "screen region to a material and a layer")
parser.add_argument("--albedo-only", action="store_true",
                    help="bypass lighting/tonemap: shows raw sampled texel")
# --- PBR surface set (fast_pack_normals.py pack required) --------------
parser.add_argument("--vertex-normals", action="store_true",
                    help="interpolated smooth NORMAL instead of flat face normals")
parser.add_argument("--normal-map", action="store_true",
                    help="tangent-space normal mapping (implies --vertex-normals)")
parser.add_argument("--normal-strength", type=float, default=1.0,
                    help="scale on the tangent-space XY of the normal map")
parser.add_argument("--aux-mip", type=int, default=0,
                    help="box-prefilter the aux array to this edge length. "
                         "sample_aux is a single bilinear tap with no mip "
                         "chain, so at 960x540 a 512^2 normal map aliases; "
                         "this is the cheap stand-in for a mip pyramid")
parser.add_argument("--specular", action="store_true",
                    help="GGX specular lobe on the sun")
parser.add_argument("--spec-gain", type=float, default=1.0)
parser.add_argument("--rough-src", choices=("normal-alpha", "const"),
                    default="normal-alpha",
                    help="glTF metallicRoughnessTexture is NOT a glTF MR map "
                         "here (VRF wrote height/ao/tintmask into that slot), "
                         "so roughness comes from the normal map's alpha")
parser.add_argument("--roughness", type=float, default=0.6,
                    help="constant roughness for --rough-src const")
parser.add_argument("--metal-mode", choices=("zero", "factor"), default="zero",
                    help="metalness source; 'factor' trusts glTF metallicFactor")
# --ao / --ao-strength are DELETED. They were the fitted aux-array stand-in
# for the material AO term, and that term now has a read replacement in both
# of its roles -- mat_ao_pages for the material factor (glsl:1229's _17119),
# --dirocc's offset-76 target for the screen-space one (_21714). The standing
# rule is that a validated replacement leaves ONE mechanism; keeping the
# stand-in beside it is how a run silently uses the one nobody meant.
#
# The deletion is paired with the arm that proves nothing depended on the
# branch, f700 / de_inferno_nrm.pt, md5 of the rendered PNG:
#     --dirocc 0        without --ao  b9b5017365ec1bd54e588557304b2868
#     --dirocc 0        with    --ao  b9b5017365ec1bd54e588557304b2868
#     --gt-lighting 0   without --ao  2d6abd43873829c6657b5b1f560f4ec7
#     --gt-lighting 0   with    --ao  2d6abd43873829c6657b5b1f560f4ec7
# Byte-identical on BOTH the GT else-branch and the legacy composition, so
# neither the shading path nor the --gt-lighting 0 measurement arm depended
# on it. The cause is the same one that made --rough-src normal-alpha inert:
# the aux AO layer these sampled is a WHITE SENTINEL, so the flag multiplied
# by 1.0 wherever it ran.
#
# --ao-strength went with it rather than being retargeted at the page: it was
# the fitted scalar ON the stand-in, and the page enters the reference
# expression at strength 1 by construction.
# DIAGNOSTIC, and it exists because this term was discarded for a whole
# session without anything saying so. `on` is the reference composition
# (glsl:1229, the material page multiplies the offset-76 resolve); `off`
# restores the pre-fix behaviour where --dirocc replaced gao outright, so
# the fix can be A/B'd on ONE tree instead of across two; `zero` forces the
# page to 0 everywhere, which is corpus-2's own falsifier -- if the image
# does not move under `zero`, the multiply is still not reaching pixels.
# Not a feature gate: the default is the reference and never changes.
parser.add_argument("--ao-page-product", default="on",
                    choices=("on", "off", "zero"),
                    help="DIAGNOSTIC for the material AO page's place in "
                         "the indirect scale. on = reference (page * "
                         "offset-76 resolve, glsl:1229). off = the "
                         "pre-fix discard, for A/B on one tree. zero = "
                         "force the page to 0, the falsifier that "
                         "separates 'dropped downstream' from 'reaching "
                         "an already-dark term'. Announces which arm ran.")
parser.add_argument("--emissive", action="store_true",
                    help="add emissiveTexture * g_flSelfIllumBrightness")
parser.add_argument("--emissive-gain", type=float, default=1.0)
# --- env_combined_light_probe_volume (the map's baked ambient cubes) ---
# Placement READ from maps/de_inferno/entities/default_ents.vents_c
# (Source2Viewer-CLI -b DATA); 116 volumes, class
# env_combined_light_probe_volume. Values are VRF's HDR EXR export of
# maps/de_inferno/lightmaps/env_light_probe_volume_atlas.vtex_c
# (160x208x840 BC6H). Both are packed into assets/probe_field.npz.
#
# VERIFIED, not assumed:
#   * handshake == 1806025118 + array_index for all 116; the octree
#     payload's own header u32s equal the entity's light_probe_size_x/y/z
#     on 116/116, so the entity<->file correspondence is checked from
#     inside the data, not from the filename.
#   * each volume owns [atlas, atlas+size) on all three axes with ZERO
#     overlapping pairs across all 116 (the 6x-depth alternative gives
#     239 overlaps); depth 840 = 6 bands of 140, one per cube direction,
#     and the band edge is visible as a std discontinuity at z == 136..139
#     (mod 140).
#   * slot order (+X,+Y,+Z,-X,-Y,-Z), established twice independently:
#     slot 2 is brightest and the only strongly blue direction (sky),
#     slot 5 the darkest and by far the warmest (ground bounce).
#   * atlas vs octree per-volume mean: corr 0.9949, median ratio 0.997.
# HEURISTIC, not read: when volumes overlap in world space (364 pairs),
# the highest indoor_outdoor_level wins, tie-broken by smallest world
# volume. The entity data does not state the engine's arbitration.
parser.add_argument("--probe-npz", default="off",
                    help="the map's baked ambient-cube field. DEFAULT OFF, "
                         "and not for want of the input: pass `auto` to "
                         "resolve <world stem>.probe_field.npz through the "
                         "same sidecar route as --irr-npy and "
                         "--gt-lm-occlusion, printing which route found it, "
                         "or pass a path to pin one. Under `auto`, a map "
                         "with no atlas prints a GAP and renders on -- the "
                         "probe path degrades to the volume path, which is "
                         "a DEFINED state and not a substitute model.")
parser.add_argument("--probe-mode", choices=("replace", "ratio"),
                    default="ratio",
                    help="'replace' uses the probe radiance as the ambient "
                         "outright; 'ratio' keeps the fitted ambient's level "
                         "and takes only the field's spatial ratio, which is "
                         "exposure-neutral (same convention as --irr-mode)")
parser.add_argument("--probe-gain", type=float, default=1.0)
parser.add_argument("--probe-clamp", type=float, default=4.0)
parser.add_argument("--probe-filter", choices=("nearest", "trilinear"),
                    default="trilinear")
parser.add_argument("--probe-stats", action="store_true")
parser.add_argument("--probe-trace", action="store_true",
                    help="DIAGNOSTIC: print a checksum of `ambient` at "
                         "each rewrite in _post_chain, so a term that is "
                         "computed but never reaches the frame can be "
                         "bracketed in two runs instead of read for.")
parser.add_argument("--session", default=None,
                    help="JSON {\"agents\":[{id, camera_json, tick_begin, "
                         "tick_end}, ...]} rendered from ONE pack load. The "
                         "corpus's own shape: ten agents of a round share the "
                         "map, the pack, the LOD levels and the baked "
                         "lighting and differ only in camera. Measured cost "
                         "of NOT doing this: 8.2 s of pack load + LOD build + "
                         "warm-up per agent, all of it outside both published "
                         "fps numbers.")
parser.add_argument("--unclamped-session-batches", action="store_true",
                    help="DIAGNOSTIC, and it renders WRONG PIXELS on purpose. "
                         "Restores the pre-fix schedule where a batch may span "
                         "several agents, which is the only way to exhibit #71: "
                         "an agent's own frames change by max|delta| 153 / 51.8%% "
                         "of pixels depending on which OTHER agents shared its "
                         "batch, while chunk-size invariance holds at "
                         "max|delta| 0. repro_batch_dependence.sh drives it. "
                         "Never use this to produce data.")
parser.add_argument("--session-out", default=None,
                    help="directory for one uint8 (frames,H,W,3) .pt per "
                         "session agent -- the DEVICE-RESIDENT path: frames "
                         "reach the consumer through a pinned copy with no "
                         "PNG round trip. This is what the training path "
                         "wants; --png-dir --png-every is what eval sampling "
                         "wants.")
parser.add_argument("--png-every", type=int, default=1,
                    help="write every Nth frame as PNG instead of every one. "
                         "PNG encode is 77.7%% of wall clock at N=1, so N=64 "
                         "turns eval sampling from the dominant cost into a "
                         "rounding error. Frame numbering is unchanged, so "
                         "f00128.png is still frame 128.")
parser.add_argument("--scope-flags", default=None,
                    help="JSON list (or {frames:[...]}) of per-frame "
                         "booleans, or {scoped: bool} rows, saying which "
                         "frames are scoped. For corpora whose state has "
                         "no FOV -- cs1k does not -- the pair's class "
                         "label is the only trigger there is. When a flag "
                         "drives a frame the magnification is a DEFAULT "
                         "and prints as one: a class label says scoped, "
                         "not by how much.")
parser.add_argument("--png-dir", default=None,
                    help="ALSO WRITE FRAMES AS PNGs. Measured 2026-08-08 at "
                         "960x540: this is 76.93 ms/frame of single-threaded "
                         "PIL deflate, 77.7%% of process wall clock, against "
                         "11.25 ms of raster+shade and 0.29 ms of "
                         "device->host copy. It takes end-to-end from 76 fps "
                         "to 11. Use --png-every to sample. Original help "
                         "follows: also write frames as lossless PNGs")
# --- Source 2 vmat feature set (fast_pack_shaders.py pack required) -----
# Each is a distinct qualitative draw-call feature CS2 composes and this
# renderer previously omitted entirely, NOT a tuning knob for one already
# implemented. They are flags only so each can be scored on its own.
parser.add_argument("--height-blend", action="store_true",
                    help="blend the two csgo_environment_blend layers by "
                         "g_tHeight1/2 instead of a linear vertex lerp")
parser.add_argument("--height-mode",
                    choices=("threshold", "hlerp", "hlerp-scaled"),
                    default="threshold",
                    help="which height-blend transition function; see "
                         "height_blend()")
parser.add_argument("--height-contrast", type=float, default=1.0,
                    help="scale on the height difference before the "
                         "softness ramp (1 = as authored)")
parser.add_argument("--blend-border", action="store_true",
                    help="F_BLEND_EFFECTS_2 border band: tint the blend "
                         "transition by g_vBorderTint2")
parser.add_argument("--bevel", action="store_true",
                    help="F_BLEND_EFFECTS_2 bevel: bend the normal along "
                         "the height-blend transition")
parser.add_argument("--overlay", action="store_true",
                    help="g_tSharedColorOverlay / F_SHARED_COLOR_OVERLAY")
parser.add_argument("--overlay-gain", type=float, default=1.0)
parser.add_argument("--vmat-ao", action="store_true",
                    help="g_tAmbientOcclusion (212 materials) instead of "
                         "the sparse glTF occlusionTexture")
parser.add_argument("--tint-mask", action="store_true",
                    help="g_tTintMask / F_TINT_MASK")
parser.add_argument("--transmissive", action="store_true",
                    help="S_TRANSMISSIVE_BACKFACE_NDOTL: g_tTransmissiveColor "
                         "lit by max(0,-N.L) per light "
                         "(csgo_foliage s32/d0:532, csgo_complex s10280/d2)")
parser.add_argument("--transmissive-gain", type=float, default=1.0)
parser.add_argument("--transmissive-lights", type=int, default=1,
                    help="accumulate the backface term over the analytic "
                         "point lights as well as the sun, which is what the "
                         "reference's light loop does (foliage s32/d0 "
                         "lines 529-792). 0 = sun only.")
parser.add_argument("--backfaces", action="store_true",
                    help="F_RENDER_BACKFACES: two-sided lighting where "
                         "CS2 draws both sides")
parser.add_argument("--metalness-tex", action="store_true",
                    help="g_tMetalness / F_METALNESS_TEXTURE")
parser.add_argument("--self-illum", action="store_true",
                    help="g_tSelfIllumMask / F_SELF_ILLUM")
parser.add_argument("--vertex-color-mode", action="store_true",
                    help="SUPERSEDED PREMISE. Written when COLOR_0 was "
                         "believed to be the blend weight, so it inverts "
                         "the weight on the three inferno_stonefloor07_* "
                         "materials. glsl:604 shows g_nVertexColorMode "
                         "gates the vertex-COLOUR tint, not the paint "
                         "stream, so the inversion has nothing to do with "
                         "the weight. Measured cost of leaving it ON, with "
                         "--blend-weight vpaint: global ncc 0.2824 -> "
                         "0.2766, matid 625 0.234 -> 0.162, everything "
                         "else +/-0.001. Turning it off is a structure "
                         "gain with an nrmse/KL cost and has not been "
                         "decided.")
parser.add_argument("--height-ch1", choices=("off", "ao", "rough"),
                    default="off",
                    help="DEAD END, kept for reproduction. ch1 of "
                         "g_tHeight is neither AO nor roughness -- both "
                         "were tested and measured nothing. glsl:593 "
                         "shows it drives _24803, the TINT MASK mix factor "
                         "between untinted albedo and tex*"
                         "g_vTextureColorTint, remapped by "
                         "g_fTintMaskContrast1/Brightness1, and it gates "
                         "the vertex-colour tint at glsl:604 too. Do not "
                         "re-run ao/rough; implement it as the tint mask.")
parser.add_argument("--probe-rect", default=None,
                    help="DIAGNOSTIC x,y,w,h: report which material ids "
                         "and blend weights actually shade a screen "
                         "region, so a colour mismatch can be attributed "
                         "to the blend or to the wrong asset")
parser.add_argument("--matid-reach", action="store_true",
                    help="DIAGNOSTIC: report what fraction of shaded "
                         "pixels each vmat feature flag actually reaches, "
                         "so a zero metric delta can be told apart from "
                         "dead code")
# --- Source 2 shader FAMILIES (fast_pack_fam.py side table) ------------
# The 923 materials are drawn by 14 different .vfx shaders and this
# renderer had ONE path for all of them. Everything below is a family
# whose behaviour differs from that path, each behind its own flag so it
# can be scored on its own. The side table is ~15 MB and rides on top of
# the UNMODIFIED world_fix2.pt, so nothing here is confounded by a repack.
parser.add_argument("--fam-side", default=None,
                    help="fast_pack_fam.py side table (fam_side.pt)")
parser.add_argument("--alpha-ref", action="store_true",
                    help="csgo_environment g_flAlphaTestReference1. VRF "
                         "wrote alphaCutoff = 0.5 for all 150 MASK "
                         "materials, but 43 of them actually specify "
                         "0.2-0.45, so we clip metal gates, grilles and "
                         "bird spikes harder than CS2 does.")
parser.add_argument("--rough-remap", action="store_true",
                    help="csgo_environment g_fTextureRoughnessBrightness1 / "
                         "Contrast1 (44 / 42 materials, typically 0.8 / "
                         "1.5). Source remaps the roughness channel before "
                         "shading; we read the normal-map alpha raw. Feeds "
                         "lod = scale*sqrt(roughness) and the split-sum "
                         "BRDF, so it now moves the environment specular.")
parser.add_argument("--fam-only", type=int, default=None,
                    help="INDEPENDENT reach measurement: paint every "
                         "material of this family index WHITE, everything "
                         "else and the sky BLACK, and emit albedo only. "
                         "Summing the frame then measures screen coverage "
                         "through the real rasteriser and the real alpha "
                         "compositing, sharing NO accounting with "
                         "--fam-reach. -1 = all families, the denominator.")
parser.add_argument("--fam-reach", action="store_true",
                    help="DIAGNOSTIC: fraction of frame pixels whose FINAL "
                         "owning surface belongs to each shader family, so "
                         "'no metric delta' can be told from 'no pixels'")
parser.add_argument("--overlay-blend", action="store_true",
                    help="csgo_static_overlay F_BLEND_MODE (24 materials). "
                         "VRF wrote alphaMode=OPAQUE for every one of them, "
                         "so a decal currently REPLACES the wall it sits "
                         "on. 1=translucent, 2=alpha-tested, 3=modulate, "
                         "plus g_flOpacityScale.")
parser.add_argument("--paint-vertex-colors", action="store_true",
                    help="F_PAINT_VERTEX_COLORS (18 materials): COLOR_0 "
                         "multiplies the albedo instead of being a blend "
                         "weight")
parser.add_argument("--depth-bias", type=float, default=0.0,
                    help="F_DEPTH_BIAS (6 csgo_complex decals): NDC depth "
                         "offset toward the camera so a coplanar decal "
                         "stops losing the depth test to its own wall")
parser.add_argument("--glass", action="store_true",
                    help="csgo_glass (4 materials). It has NO g_tColor at "
                         "all, so the pack falls back to the white sentinel "
                         "and every pane renders as opaque white. Shades "
                         "from g_tGlassDust / g_tGlassTintColor / "
                         "GlassMaskTransmission instead.")
# --- THE THREE SMALL FAMILIES ------------------------------------------
# `_f3` is defined ~700 lines below, after this block; using it here
# would be a NameError at import for EVERY invocation, which is the same
# failure mode as a duplicate flag name. Its own parser here.
_sf_f3 = lambda s: [float(x) for x in s.split(",")]   # noqa: E731
# csgo_effects (1 material, the steam plume, 0.089% of pixels),
# csgo_black_unlit (2, `black_simple`) and csgo_vertexlitgeneric (8,
# props, 0.0011%). Small SHARE is a usage fact about de_inferno and not a
# capability fact about the renderer, so every axis each of the three
# declares is implemented at every value it takes.
#
# FLAG NAMESPACE. Every flag in this block is prefixed `--sf-`. Four
# agents landed families into this file in the same release; two of them
# adding the same flag name in different files merges clean and then
# raises inside argparse at import, which takes the renderer down for
# EVERY invocation, not just the one that wanted the flag. `--sf-selftest`
# replays every add_argument in this file against a fresh ArgumentParser
# and reports the conflict count.
parser.add_argument("--sf-effects", action="store_true", default=None,
                    help="csgo_effects c10/r10m1 (S_ADDITIVE_BLEND + "
                         "S_DEPTH_FEATHER), the whole 145-line module: a "
                         "three-mask scrolling erosion, a facing ramp, a "
                         "scene-depth feather, a distance fade, and fog "
                         "consumed through the ALPHA because the surface "
                         "is additive. 194 FLOPs / 5 fetches, EXACT.")
parser.add_argument("--class-off", default=None,
                    help="comma list of shader classes to force OFF, by "
                         "family name (csgo_effects.vfx) or by flag name "
                         "(sf-effects). THE EXPLICIT OFF. These classes are "
                         "auto-enabled BY POPULATION -- the pack contains "
                         "faces of the class, so its transcribed path runs "
                         "-- and passing the flag could only force ON. "
                         "There was therefore no way to render a populated "
                         "map with a class's own path off, which is exactly "
                         "the arm that shows what its SUBSTITUTE looks like: "
                         "the class matrix's '** VISIBLE but its OWN PATH IS "
                         "OFF **' verdict was unreachable from the CLI on "
                         "any map that could produce it. DIAGNOSTIC: the "
                         "default stays auto-on-by-population and this "
                         "changes no default; it prints the population it "
                         "overrode so a run's log cannot hide it.")
parser.add_argument("--sf-black-unlit", action="store_true", default=None,
                    help="csgo_black_unlit c0/r0m0, the whole 115-line "
                         "module: the map's gradient fog and cube fog "
                         "composited over BLACK, alpha 1. Not the same as "
                         "--unlit, which forces the albedo black and leaves "
                         "the two faces UNFOGGED. 132 FLOPs / 1 fetch, "
                         "EXACT.")
parser.add_argument("--sf-vertexlit", action="store_true", default=None,
                    help="csgo_vertexlitgeneric c512/r51m3 (S_TINT_MASK, "
                         "D_BAKED_LIGHTING_FROM_LIGHTMAP): the 2-channel "
                         "hemi-octahedral normal decode, the tangent frame, "
                         "AO on TEXCOORD1, the tint mask, metalness, the "
                         "lightmap x offset-76 x AO term, the cascade + "
                         "offset-88 sun visibility, the clustered lights "
                         "and the fog.")
parser.add_argument("--sf-tools-vis", type=int, default=0,
                    help="S_MODE_TOOLS_VIS for csgo_black_unlit and "
                         "csgo_effects. 0 selects the in-game module "
                         "(combo 0); any other value selects combo 1 and "
                         "the mode it names. csgo_black_unlit c1/r1m0 is "
                         "the FIRST module in this corpus that actually "
                         "selects this axis -- --tools-vis above says none "
                         "was decompiled, which was true then. An "
                         "unrecognised value is an error, not a no-op.")
parser.add_argument("--sf-tools-vis-tint", type=_sf_f3, default=[1.0, 1.0, 1.0],
                    help="the tools tint constant at set 1/binding 0 offset "
                         "64, read by r1m0 modes 16 and 60")
parser.add_argument("--sf-tools-vis-range", type=float, default=1.0,
                    help="r1m0 mode 81's symmetric range constant")
parser.add_argument("--sf-tools-vis-flash", action="store_true",
                    help="r1m0:660-676, the module's trailing "
                         "mix(rgb, red, |frac(t*0.5)*1.6-0.8|) highlight, "
                         "gated in the reference by two per-view ints")
# --- env_cubemap_fog, the second fog entity ---------------------------
# All three families' modules end with the same TWO fog layers. The
# gradient one is env_gradient_fog and reuses --fog-* above; the cube one
# is env_cubemap_fog, whose shipped de_inferno keys the --fog block's
# header records as start 800 u / end 280000 u. Those are the DEFAULTS
# here because they are the map's data, and at them the cube term is
# small at de_inferno's distances -- `--sf-selftest` prints the max
# opacity it actually reaches over the world box rather than leaving that
# as an assumption, and the flags below move it so the path is
# exercisable at values where it is large.
parser.add_argument("--sf-cube-fog-source", default="probe",
                    choices=("probe", "fogcolor"),
                    help="what stands in for the env_cubemap_fog cubemap, "
                         "which is an engine render resource and is in no "
                         ".vcs: 'probe' samples the nearest baked "
                         "environment cube (needs --ibl-cube), 'fogcolor' "
                         "uses the fitted fog/sky colour. The fetch SITE, "
                         "direction, LOD and scale are transcribed either "
                         "way; only the texture CONTENT is substituted.")
parser.add_argument("--sf-cube-fog-start", type=float, default=800.0)
parser.add_argument("--sf-cube-fog-end", type=float, default=280000.0)
parser.add_argument("--sf-cube-fog-exponent", type=float, default=1.0)
parser.add_argument("--sf-cube-fog-strength", type=float, default=1.0)
parser.add_argument("--sf-cube-fog-gain", type=float, default=1.0,
                    help="_5037._m8.x, the cube sample's scale")
parser.add_argument("--sf-cube-fog-lod", type=float, default=6.0,
                    help="_5037._m5.w, the LOD at full occlusion")
parser.add_argument("--sf-cube-fog-lod-bias", type=float, default=1.0,
                    help="_5037._m4.z in lod = m5.w * clamp(1 - occ*m4.z)")
parser.add_argument("--sf-cube-fog-height-bias", type=float, default=1.0)
parser.add_argument("--sf-cube-fog-height-scale", type=float, default=0.0)
parser.add_argument("--sf-cube-fog-height-exponent", type=float, default=1.0)
parser.add_argument("--sf-cube-fog-height-cutoff", type=float, default=1e9,
                    help="_5037._m7.y -- the cube fog's height gate, "
                         "`m7.z * wp.z < m7.y` (r0m0:88). env_cubemap_fog "
                         "ships no height key on de_inferno, so the default "
                         "is above the world box and the gate passes; the "
                         "COMPARISON is transcribed either way and this "
                         "flag moves it.")
# --- csgo_effects combo axes, all four static and all seven dynamic ----
_SF_AXIS = ("material", "on", "off")
parser.add_argument("--sf-effects-additive", default="material",
                    choices=_SF_AXIS,
                    help="S_ADDITIVE_BLEND (place value 2). 'material' "
                         "reads F_ADDITIVE_BLEND. On: the fog attenuates "
                         "the output ALPHA (r10m1:120/130). Off: it blends "
                         "the COLOUR, the ordinary form.")
parser.add_argument("--sf-effects-tint-mask", default="material",
                    choices=_SF_AXIS,
                    help="S_TINT_MASK (4). Isolated by diffing c14/r14m1 "
                         "against c10/r10m1: off, the vertex colour tints "
                         "unconditionally; on, mix(col, col*vColor, "
                         "g_tTintMask.x).")
parser.add_argument("--sf-effects-depth-feather", default="material",
                    choices=_SF_AXIS,
                    help="S_DEPTH_FEATHER (8): reconstruct the scene world "
                         "position from the depth target and ramp the "
                         "particle out over g_flFeatherDistance")
parser.add_argument("--sf-effects-backfaces", default="material",
                    choices=_SF_AXIS,
                    help="F_RENDER_BACKFACES, r10m1:79-97 -- the normal is "
                         "flipped on back faces before the facing ramp")
parser.add_argument("--sf-effects-fog", default="material",
                    choices=_SF_AXIS, help="g_bFogEnabled, r10m1:112")
parser.add_argument("--sf-effects-baked", default="lightmap",
                    choices=("none", "vertex-stream", "probe", "lightmap"),
                    help="D_BAKED_LIGHTING_FROM_* (1/2/4). MEASURED INERT "
                         "in this family: r10_m0 (dyn 0) and r10_m1 (dyn 4) "
                         "differ in EXACTLY one line, the location= of the "
                         "vertex-colour varying. The flag exists so the "
                         "axis is selectable and the banner reports which "
                         "module the choice names.")
parser.add_argument("--sf-effects-mboit", default="off",
                    choices=("off", "pass1", "pass2"),
                    help="D_MBOIT_PASS1 (16) / D_MBOIT_PASS2 (32). Routed "
                         "into the existing MBOIT unit, so --mboit-moments "
                         "supplies D_MBOIT_4_MOMENTS (64).")
parser.add_argument("--sf-effects-msaa", action="store_true",
                    help="D_SCENE_DEPTH_MSAA (8): the depth texelFetch "
                         "takes a per-sample index. This renderer has one "
                         "sample per pixel, so the axis selects the same "
                         "texel; the SELECTOR is reproduced and the banner "
                         "says the sample count is 1.")
# --- csgo_vertexlitgeneric combo axes ---------------------------------
parser.add_argument("--sf-vertexlit-baked", default="lightmap",
                    choices=("auto", "none", "vertex-stream", "probe",
                             "lightmap"),
                    help="D_BAKED_LIGHTING_FROM_* (1/2/4). 'lightmap' is "
                         "r51_m3, the module de_inferno's props select and "
                         "the one that resolves the '100%% lightmapped' "
                         "finding: this family reads a lightmap because the "
                         "DYNAMIC combo says so, not because its name does.")
parser.add_argument("--sf-vertexlit-opaque-fade", action="store_true",
                    help="D_OPAQUE_FADE (16), r51_m4/m5/m6/m7. Routed into "
                         "the existing opaque_fade() dither unit.")
parser.add_argument("--sf-vertexlit-spec-cube-static", action="store_true",
                    help="D_SPECULAR_CUBE_MAP_STATIC (8). The family "
                         "declares it; static combo 512 ships no module "
                         "with the bit set (its dynamic ids are "
                         "0,1,2,4,16,17,18,20), so it pins the IBL cube to "
                         "the material's own probe instead of the nearest.")
parser.add_argument("--sf-vertexlit-shader-quality", type=int, default=0,
                    choices=(0, 1),
                    help="S_SHADER_QUALITY (2048), source type 2. Diffing "
                         "c2560/r306m3 against c512/r51m3 shows quality 1 "
                         "replaces the single cascade tap with a weighted "
                         "9-tap PCF; routed into gt_pcf()'s quality arm.")
for _sfa, _sfh in (
        ("spec-direct", "S_SPECULAR_DIRECT (1): GGX sun specular, "
                        "c515/r53m3"),
        ("spec-indirect", "S_SPECULAR_INDIRECT (2): roughness from the "
                          "normal map's B channel floored by the "
                          "normal-derivative AA term, the Lazarov DFG and "
                          "the offset-104 IBL cube array, c514/r52m3"),
        ("force-uv2", "S_FORCE_UV2 (4): PIXEL-IDENTICAL to the base "
                      "module -- c516 and c512 share RECORD 51 and their "
                      "decompiled modules differ in 0 bytes, so the axis "
                      "lives entirely in the vertex stage"),
        ("decal", "S_DECAL_TEXTURE (8), c520/r54m3"),
        ("detail", "S_DETAIL_TEXTURE (16), c528/r57m3"),
        ("alpha-test", "S_ALPHA_TEST (64), c576/r66m3: the BEAUTY module "
                       "contains no OpKill and no read of the colour "
                       "alpha; it writes alpha 1.0 and the test is in the "
                       "depth module"),
        ("translucent", "S_TRANSLUCENT (128), c640/r81m3"),
        ("additive", "S_ADDITIVE_BLEND (256): NOT isolated by diff -- no "
                     "shipped static combo carries bit 256 without also "
                     "carrying S_DECAL_TEXTURE or S_DETAIL_TEXTURE, so "
                     "512^256 does not exist; implemented as the "
                     "csgo_effects S_ADDITIVE_BLEND form"),
        ("tint-mask", "S_TINT_MASK (512), c512/r51m3, the axis de_inferno "
                      "selects"),
        ("self-illum", "S_SELF_ILLUM (1024), c1536/r153m3")):
    parser.add_argument(f"--sf-vertexlit-{_sfa}", default="material",
                        choices=_SF_AXIS, help=_sfh)
parser.add_argument("--sf-flop-audit", action="store_true",
                    help="price this file's csgo_black_unlit and "
                         "csgo_effects transcriptions with flopcount.py's "
                         "own rules and compare against the two EXACT "
                         "reference figures (132/1 and 194/5). Exits "
                         "non-zero on any mismatch.")
parser.add_argument("--sf-flop-audit-mutate", action="store_true",
                    help="run the audit with one term deleted, to show the "
                         "audit FAILS when the transcription changes. A "
                         "check that cannot fail is not a check "
                         "(CHECKS_THAT_CANNOT_FAIL.md).")
parser.add_argument("--sf-selftest", action="store_true",
                    help="prove every path this block adds is REACHABLE AT "
                         "ITS OWN DEFAULTS: runs each on a synthetic batch "
                         "spanning the world box, reports max|delta| "
                         "against the path suppressed, replays every "
                         "add_argument in this file against a fresh parser "
                         "and exits non-zero on a zero delta or a flag "
                         "conflict.")
# --- transparency / refraction combo axes ------------------------------
# Every axis below is transcribed from a module extracted out of the
# shipped .vcs by src/counter_strike_render/vcs/, not from a description of one.
# The (static, dynamic) combo each was read from is named in the help.
parser.add_argument("--translucent", action="store_true",
                    help="S_TRANSLUCENT (csgo_complex, 11 de_inferno "
                         "materials). complex s80/d2:888 -- the output alpha "
                         "becomes vColor.a * (colorTex.a * g_flOpacityScale) "
                         "instead of vColor.a alone.")
parser.add_argument("--additive-blend", action="store_true",
                    help="S_ADDITIVE_BLEND (csgo_complex, 1 material). "
                         "complex s240/d2:909 -- dst+src*a compositing, AND "
                         "the fog multiplies the ALPHA rather than tinting "
                         "the colour, which is the whole structural "
                         "difference in the pixel stage.")
parser.add_argument("--cubemap-refraction", action="store_true",
                    help="S_OPAQUE_CUBEMAP_REFRACTION (csgo_glass). glass "
                         "s2/d8:1116-1191 -- refract(V,N,0.66) resolved "
                         "against the IBL cube ARRAY with box-parallax "
                         "correction; the pane then writes alpha 1.0, i.e. "
                         "it is OPAQUE. 0 de_inferno materials set the F_ "
                         "flag (14/23 game-wide), so this flag selects the "
                         "combo for the csgo_glass family; a pack whose "
                         "f_opaque_cube_refract column is set drives it too.")
parser.add_argument("--refract-ior", type=float, default=1.0 / 0.66,
                    help="eta = 1/ior fed to refract(). The shipped constant "
                         "is eta = 0.660000026226043701171875 (glass "
                         "s2/d8:1116) => ior 1.51515; this is that value.")
parser.add_argument("--mboit", choices=("off", "4", "6"), default="off",
                    help="D_MBOIT_PASS1 + D_MBOIT_PASS2, with D_MBOIT_4_MOMENTS "
                         "selecting 4 power moments (complex s80/d162, "
                         "s80/d194) against 6 (s80/d34, s80/d66). Runs the "
                         "real two-pass structure over depth-peeled layers: "
                         "pass 1 accumulates moments only, pass 2 shades and "
                         "resolves against them.")
parser.add_argument("--mboit-layers", type=int, default=0,
                    help="cap on the depth peel. DEFAULT 0 = peel until a "
                         "layer is empty, because the engine has no such "
                         "bound (it blends into a moment target) and a "
                         "default that truncated would be a feature that "
                         "reads as tested. A hard ceiling of 64 exists only "
                         "to bound the loop and warns if it is reached; the "
                         "layer count actually used is always printed.")
parser.add_argument("--mboit-bias", type=float, default=5e-6,
                    help="the mix() weight toward the moment bias vector "
                         "(0,0.375,0,0.375) at 4 moments / "
                         "(0,0.48,0,0.451,0,0.45) at 6 -- complex "
                         "s80/d194:995 and s80/d66:92. "
                         "5e-6 is READ from libclient.so's r_csgo_mboit_bias "
                         "(movss/xmm0 site). IT REPLACES 6e-5, which was "
                         "Muenstermann et al.'s PUBLISHED power-moment value "
                         "standing in for an engine constant this pass could "
                         "not decode -- a paper number, never a read of CS2, "
                         "and its own help said so. A caveated read of the "
                         "shipped binary beats an acknowledged stand-in; the "
                         "two differ by 12x. CAVEAT CARRIED FROM THE READER: "
                         "the single lea site per name is not PROVEN to be "
                         "the cvar constructor rather than another use of "
                         "the name string, so this is a strong candidate, "
                         "not gospel. The evidence has since STRENGTHENED: "
                         "the call-target filter (ConVar ctor va 0xc84900) "
                         "and the instruction-shape filter partition "
                         "identically over the 23 resolvable cvars, so the "
                         "name->flag join is corroborated twice and only "
                         "the is-xmm0-the-default question remains. The "
                         "runtime banner carries it into every MBOIT "
                         "number rather than leaving it in this string.")
parser.add_argument("--mboit-overestimation", type=float, default=0.01,
                    help="the transmittance the resolve assumes at the first "
                         "reconstruction knot (complex s80/d194:1013 "
                         "`_5037._m20`). 0.01 is READ from libclient.so's "
                         "r_csgo_mboit_overestimation, replacing 0.25 -- the "
                         "published value, which this help previously "
                         "described as standing in for an engine CB that was "
                         "not decoded. 25x. Same reader caveat as "
                         "--mboit-bias: strong candidate, not gospel.")
parser.add_argument("--translucent-clip", action="store_true",
                    help="S_MODE_DEPTH on csgo_glass (glass s1/d0). The "
                         "depth pass shades the pane's TRANSMITTED colour "
                         "and discards -- i.e. writes no depth -- where its "
                         "luminance exceeds the threshold, so a clear pane "
                         "does not occlude what is behind it.")
parser.add_argument("--translucent-clip-threshold", type=float, default=0.7,
                    help="glass s1/d0:147, the shipped constant "
                         "0.699999988079071044921875.")
parser.add_argument("--disable-translucent-clip", action="store_true",
                    help="D_DISABLE_TRANSLUCENT_CLIP=1 (glass s1/d1): the "
                         "whole clip module collapses to `out=vec4(0)` with "
                         "no discard, so every glass fragment writes depth. "
                         "Overrides --translucent-clip.")
parser.add_argument("--opaque-fade", action="store_true",
                    help="D_OPAQUE_FADE (complex d16, static_overlay d1, "
                         "foliage d4). One shared engine snippet: "
                         "fade = mix(-F,1,x) + F*dither(fragCoord & mask).y; "
                         "discard if fade < 0.001; the surviving fade "
                         "REPLACES the output alpha.")
parser.add_argument("--opaque-fade-amount", type=float, default=1.0,
                    help="F above (set 1/binding 1 offset 460). No "
                         "de_inferno material carries a fade distance, so "
                         "the per-view amount is exposed directly. F=0 "
                         "degenerates the snippet to a plain alpha>0.001 "
                         "test, which is still a live path, not a disabled "
                         "one.")
parser.add_argument("--alpha-test-prepass", action="store_true",
                    help="D_ALPHA_TEST_PREPASS (foliage s10/d1 against "
                         "s10/d0). Replaces the binary alpha test with the "
                         "derivative-normalised coverage "
                         "clamp(0.5+(a-ref)/max(fwidth(a),1e-6),0,1), then "
                         "discards below 0.001.")
parser.add_argument("--mode-depth", action="store_true",
                    help="DIAGNOSTIC render mode: emit what the S_MODE_DEPTH "
                         "modules write instead of the beauty pass -- the "
                         "glass translucent-clip mask (glass s1/d0) and the "
                         "foliage alpha-test coverage (foliage s10/d0). "
                         "csgo_environment's S_MODE_DEPTH combo has "
                         "m_byteCodeIndex = -1 for every dynamic combo, i.e. "
                         "NO pixel shader at all, and renders as nothing.")
parser.add_argument("--parallax3", action="store_true",
                    help="csgo_simple_3layer_parallax (9 window materials): "
                         "g_tLayer1Color / g_tLayer2Color parallax-offset "
                         "by g_flLayer1Offset / g_flLayer2Offset behind "
                         "g_tLayer0Mask")
# =======================================================================
# csgo_simple.vfx  and  csgo_simple_3layer_parallax.vfx
# =======================================================================
# Transcribed from the modules de_inferno selects, extracted here:
#   csgo_simple                 record 0, dyn 128 (D_BAKED_LIGHTING_FROM_
#                               LIGHTMAP) and dyn 64 (..._FROM_PROBE)
#   csgo_simple_3layer_parallax record 4 (S_TINT_MASK), dyn 8 (PROBE),
#                               dyn 16 (LIGHTMAP)
# committed as docs/projects/counter-strike-sft/csgo_simple_ps.glsl and
# csgo_simple_3layer_parallax_ps.glsl; every line number cited below is a
# line of those two files.
#
# EVERY FLAG IN THIS BLOCK IS NAMESPACED --simple-* / --s3lp-*. Two agents
# added --secondary-uv in different files on the same day and argparse
# refused to build a parser for ANY invocation (commit efb652a); the
# namespace is the cheap half of not repeating that, and replaying the
# whole parser (see the parser self-check just below parse_args) is the
# half that actually catches it.
parser.add_argument("--simple-shading", type=int, default=1,
                    help="csgo_simple.vfx (18 materials, 0.235%% of "
                         "de_inferno's geometry pixels, all 18 OPAQUE): "
                         "albedo = g_tColor * vColor, roughness from the "
                         "normal map's BLUE channel with the derivative "
                         "floor, metalness from g_flMetalness or "
                         "g_tAmbientOcclusion.w, AO from "
                         "g_tAmbientOcclusion.x on the whole indirect "
                         "term. Default 1: the family is 100%% opaque so "
                         "this is reachable wherever --fam-side is given, "
                         "and a 0 default would read as tested while "
                         "executing nothing.")
parser.add_argument("--simple-metalness-texture", default="auto",
                    choices=("auto", "0", "1"),
                    help="S_METALNESS_TEXTURE (static place value 1). "
                         "'auto' takes F_METALNESS_TEXTURE per material "
                         "(2 of 18 on de_inferno); '0' forces the "
                         "g_flMetalness uniform, '1' forces "
                         "g_tAmbientOcclusion.w. STRING-VALUED: never "
                         "tested for truthiness, '0' is truthy.")
parser.add_argument("--simple-ao-texture", default="auto",
                    choices=("auto", "0", "1"),
                    help="S_AMBIENT_OCCLUSION_TEXTURE (place value 2). "
                         "The axis DEDUPLICATES: static combos 0 and 2 "
                         "share bytecode record 0 and 1/3 share record 1, "
                         "so the pixel shader is byte-for-byte the same "
                         "either way and g_tAmbientOcclusion is sampled "
                         "unconditionally. '0' forces AO to 1 so the "
                         "term's contribution is measurable; 'auto' and "
                         "'1' both sample. STRING-VALUED.")
parser.add_argument("--simple-reflectance", type=float, default=0.04,
                    help="the dielectric F0 csgo_simple mixes towards at "
                         "metalness 0 (r0_d128.glsl:254). No de_inferno "
                         "csgo_simple material sets g_flReflectance, so "
                         "this supplies the .vfx default the bytecode "
                         "does not carry.")
parser.add_argument("--s3lp-shading", type=int, default=1,
                    help="csgo_simple_3layer_parallax.vfx (9 window "
                         "materials, 0.306%% of pixels, 100%% under "
                         "models/ so the PROBE dynamic combo, all 9 "
                         "OPAQUE). Three layers behind one pane, each at "
                         "its own parallax depth. Default 1, same "
                         "reasoning as --simple-shading.")
parser.add_argument("--s3lp-tint-mask", default="auto",
                    choices=("auto", "0", "1"),
                    help="S_TINT_MASK (place value 4, the axis de_inferno "
                         "selects: static combo 4, record 4). At 1 the "
                         "vertex-colour tint is masked by g_tTintMask.x "
                         "(:289); at 0 the tint is applied whole. "
                         "STRING-VALUED.")
parser.add_argument("--s3lp-metalness-texture", default="auto",
                    choices=("auto", "0", "1"),
                    help="S_TINT_MASK's neighbour at place value 2: "
                         "metalness from g_tAmbientOcclusion.w instead of "
                         "g_flMetalness, then multiplied by the layer-0 "
                         "mask either way (:288). STRING-VALUED.")
parser.add_argument("--s3lp-secondary-uv", default="auto",
                    choices=("auto", "0", "1"),
                    help="S_SECONDARY_UV (place value 1). The module "
                         "carries THREE independent UV-set selectors -- "
                         "one for the normal map and the parallax layers, "
                         "one for the layer-0 mask, one for the tint mask "
                         "-- each choosing TEXCOORD.zw over .xy. Namespaced "
                         "away from the pre-existing --secondary-uv, which "
                         "is csgo_environment_blend's. STRING-VALUED.")
parser.add_argument("--s3lp-transmissive", default="auto",
                    choices=("auto", "0", "1"),
                    help="S_TRANSMISSIVE_BACKFACE_NDOTL (place value 8). "
                         "Accumulates max(0, -dot(N, L)) over the sun and "
                         "every local light, multiplies it by the "
                         "transmission map and adds it unlit; also swaps "
                         "the sun shadow bias for a backface one and drops "
                         "the dot(sun,N) > 0 early-out. STRING-VALUED.")
parser.add_argument("--s3lp-second-layer-cubemap", default="auto",
                    choices=("auto", "0", "1"),
                    help="S_SECOND_LAYER_CUBEMAP (place value 16). "
                         "Multiplies LAYER 2's emissive by a cube fetch "
                         "along vec3(nTS.xy,0)*refract2 + viewDir "
                         "(:746 of the r16 module). STRING-VALUED.")
parser.add_argument("--s3lp-depth-scale", type=float, default=0.0254,
                    help="SOURCE units -> the units g_flLayer1Offset / "
                         "g_flLayer2Offset are in when the pixel shader "
                         "multiplies them into a UV. The PS alone CANNOT "
                         "fix this: it offsets by "
                         "tangentViewDir * (depth * pow(NdotV,0.7) / "
                         "dot(V,N)), and whether `depth` is already in UV "
                         "units depends on the length convention of the "
                         "tangent the VERTEX stage writes -- and no vertex "
                         "stage was extracted for either family. 0.0254 "
                         "is source-to-metres, combined with the "
                         "per-face metres-per-UV in MAT_UVSCALE. Set 1.0 "
                         "to take g_flLayer*Offset as raw UV units.")
parser.add_argument("--s3lp-refract1", type=float, default=0.0,
                    help="g_flLayer1RefractScale fallback: the tangent "
                         "normal's xy is added to layer 1's UV scaled by "
                         "it (:278). NO de_inferno material sets it, so "
                         "the shipped value is the .vfx default and the "
                         ".vfx default is not in the PS bytecode. 0.0 is "
                         "declared here as the fallback, NOT measured; "
                         "the ext2 column has_s3lp_refract1 says per "
                         "material whether the asset overrode it.")
parser.add_argument("--s3lp-refract2", type=float, default=0.0,
                    help="g_flLayer2RefractScale fallback (:281). Same "
                         "provenance caveat as --s3lp-refract1.")
parser.add_argument("--s3lp-layer2-fresnel", type=float, default=1.0,
                    help="the exponent layer 2's emissive is raised to "
                         "against clamp(dot(-V,N),0,1) (:749). NO "
                         "de_inferno material sets it; 1.0 is a declared "
                         "fallback, not a measurement. At 0 the term "
                         "becomes 1 and the view dependence VANISHES, "
                         "which is why the default is not 0.")
parser.add_argument("--simple-reach", action="store_true",
                    help="print, per family, how many shaded pixels each "
                         "of the two paths and each of their combo axes "
                         "actually reached. This is the check that a "
                         "wired-in family is not a function nobody calls.")
parser.add_argument("--mask-normals", action="store_true",
                    help="give the alpha-tested (MASK) class the same "
                         "interpolated vertex normal + normal map the "
                         "opaque class gets; it currently shades on the "
                         "FLAT face normal, which is what makes foliage "
                         "read as cutouts")
parser.add_argument("--foliage", action="store_true",
                    help="csgo_foliage (53): g_flVertexNormalInfluence "
                         "bends the shading normal back to the authored "
                         "vertex normal (the leaf-clump sphere normal), "
                         "and g_flAlphaBoost* thickens distant cards")
parser.add_argument("--water", action="store_true",
                    help="csgo_water_fancy (1 material, the fountain, but "
                         "0.70%% of pixels). Its plane is drawn OPAQUE "
                         "today, so it sprawls across the street as an "
                         "olive slab; CS2 refracts through it and only "
                         "shows a Fresnel sky reflection.")
parser.add_argument("--water-fancy", action="store_true", default=None,
                    help="csgo_water_fancy transcribed from record 17 of "
                         "csgo_water_fancy_vulkan_50_ps.vcs (static combo "
                         "92, dynamic 2). REPLACES --water's generic "
                         "Fresnel-over-fog approximation, which predates "
                         "the family enumeration and is not this shader.")
# Axis overrides. STRING-VALUED, and resolved to None/float exactly once,
# below, by _water_axis_override -- no consumer ever tests these with
# bare truthiness, because "off" is truthy and that is the bug that was
# hiding behind today's argparse P0.
for _wa, _wh in (("refraction", "S_REFRACTION"),
                 ("caustics", "S_CAUSTICS"),
                 ("interaction", "S_INTERACTION_EFFECTS"),
                 ("blur-refraction", "S_BLUR_REFRACTION")):
    parser.add_argument("--water-" + _wa, default="material",
                        choices=("material", "on", "off"),
                        help=_wh + ": take the value from the material "
                             "(default), or force it on/off so the axis "
                             "can be exercised where no de_inferno "
                             "material selects it")
parser.add_argument("--water-reflection-type", default="material",
                    choices=("material", "0", "1", "2"),
                    help="S_REFLECTION_TYPE. A RANGE 0..2, not a flag: "
                         "0 = constant sky colour, 1 = environment cube, "
                         "2 = cube + the screen-space reflection march")
