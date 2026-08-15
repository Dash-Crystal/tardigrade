#!/usr/bin/env python3
"""Version-controllable render identity, and the refusal that uses it.

An ARGV line does NOT identify a render. Measured 2026-08-08: the SAME argv,
pack and GT against two copies of gpu_render.py gave kl_rgb 1.3223 and
4.0189 -- a 3x difference from renderer source alone. ws-1 carried ELEVEN
gpu_render.py with eleven distinct sha256, and the ARGV stamp records only
the basename.

So identity = flags + sha256(renderer) + sha256(pack) + sha256(fam-side) +
sha256(camera). This module captures that from a run that already happened,
and verifies it before one that has not.

    capture  --log RUN.log --renderer PATH [--out cfg.json]
    verify   cfg.json
    argv     cfg.json          # emit the flags, for `xargs python gpu_render.py`
    compare  a.json b.json     # refuse unless two runs are comparable
    presets                    # the GT preset schema, read from the manifest

capture is the retrofit path: it reconstructs identity from any log carrying
an `ARGV:` stamp, so arms that were never documented become reproducible
without rerunning them.

--------------------------------------------------------------------------
WHAT v2 ADDS, AND WHY (#55)
--------------------------------------------------------------------------
v1 pinned only flags that were PRESENT on the command line. That leaves the
larger hole open: a flag that is absent takes its default, and a DEFAULT CAN
CHANGE. "A superseded scalar defaults silently where a deleted flag fails
loudly" -- delete a flag and every old argv errors immediately; change its
default and every old argv keeps running, silently, against different
arithmetic. The failure is invisible on exactly the runs you most want to
trust: the ones whose command lines you never touched.

So v2 records the RESOLVED value of every PINNED flag, present or not, and
where that value came from (`argv` or `default`). The defaults are read out
of the renderer by parsing its source with `ast` -- gpu_render.py parses
argv at import, so it cannot be imported to ask it, but its
`parser.add_argument(...)` calls are perfectly readable statically. `verify`
re-derives them from the renderer as it stands TODAY and refuses if a
pinned flag's effective value has moved.

v2 also pins the GT SIDE. A render is only comparable against the ground
truth it was configured to match, and the match is not free: `preset2`'s
video.cfg carries `setting.shaderquality 1`, which is `S_SHADER_QUALITY = 1`
(RENDER_SETTINGS_AXES.md:489-492), while our `--shader-quality` defaults to
0. Scoring our default render against preset2 is a config mismatch that
produces a number, and the number has been read. With a `gt` block present,
`verify` REFUSES that pairing by name.

--------------------------------------------------------------------------
BLAST RADIUS, MEASURED BEFORE LANDING
--------------------------------------------------------------------------
`find . -name '*.render.json'` over the repo and over ws-1's home: ZERO
existing configs. So a blanket "every metric run must present a verified
config" would refuse EVERY metric run in the project on day one -- the same
shape as the sidecar refusal that would have broken 42 maps.

What lands instead:
  * `require()` is available to any caller and REFUSES on mismatch;
  * metric_compare.py accepts `--render-config` and, without one, stamps
    its output `identity: UNVERIFIED` and withholds the acceptance verdict.
    Diagnostics keep working; nothing unverified can be read as a gate
    result.
  * the gate runner requires a verified config unconditionally. It is a new
    surface, so its blast radius on existing work is zero by construction.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys

SCHEMA = "iji/render-identity/v3"
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
MANIFEST = os.path.join(REPO_ROOT, "docs", "projects", "counter-strike-sft",
                        "GT_QUALITY_LEVEL_MANIFEST.json")


def sha(p):
    if not p or not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# flags whose value is a FILE path we must pin by content
PATH_FLAGS = ('--world', '--fam-side', '--camera-json', '--irr-npy',
              '--lightmap', '--cube-probes', '--probe-npz', '--skyvis',
              '--lights', '--vpaint')

# flags whose value is a MANIFEST: a file that NAMES OTHER FILES.
#
# `--session` is `{"agents": [{id, camera_json, tick_begin, tick_end}, ...]}`
# -- it SELECTS WHICH POSES RENDER. Pinning its path pins nothing, so it
# started in PATH_FLAGS. But pinning its CONTENT is also not enough: the
# camera jsons it names are a second level, and swapping one changes which
# poses render while the session file's own sha still matches.
#
# That is the --ibl-cube hole exactly one layer down, and it is the third
# time this shape has appeared (directory of mips; auto-resolved atlas; now
# manifest-of-cameras). The pattern: A POINTER IS NOT ITS TARGET, and each
# level of indirection needs pinning separately or the deepest one is free.
#
# So a manifest is pinned by its own sha PLUS the sha of every file it
# references, rolled into one. The reference extraction is data-derived --
# it walks the JSON for any string that resolves to an existing file rather
# than knowing the key is called "camera_json" -- because a hard-coded key
# would miss the next manifest schema, which is how this hole got here.
MANIFEST_FLAGS = ('--session',)

# flags whose value is a DIRECTORY of assets. `--ibl-cube` is a directory of
# env_cubemap_mip{0..6}_f16.npy -- seven files, 365 MB, and the prefiltered
# radiance chain the specular term samples. Pinning the directory PATH pins
# nothing: swap one mip and every sha in the config still matches. So a
# directory is pinned by a manifest of its files' shas, rolled into one, with
# the per-file map kept so a drift can name the file rather than the folder.
DIR_FLAGS = ('--ibl-cube',)

# Flags whose RESOLVED VALUE is part of the identity, defaults included.
#
# This list is deliberately short and deliberately not "every flag". Pinning
# all ~600 of gpu_render.py's flags would make every config incomparable
# with every other for reasons that have nothing to do with the image, and a
# refusal that always fires is discarded. These are the axes that select
# WHICH REFERENCE EXPRESSION runs, plus the framing that decides what a
# pixel is:
PINNED_FLAGS = (
    '--shader-quality',      # S_SHADER_QUALITY: not a scale, a different shader
    '--s-lit',               # S_LIT
    '--animated-shadows',    # S_ANIMATED_SHADOWS
    '--spec-cube-static',    # D_SPECULAR_CUBE_MAP_STATIC (requires q1)
    '--msaa-samples',
    '--width', '--height', '--supersample',
    '--light-basis',
)

# GT video.cfg keys that are part of the preset schema. A capture's preset is
# these values; two captures with different values are different ground
# truths regardless of what their directories are called.
PRESET_KEYS = (
    'setting.shaderquality',
    'setting.r_texturefilteringquality',
    'setting.msaa_samples',
    'setting.r_csgo_cmaa_enable',
    'setting.videocfg_shadow_quality',
    'setting.videocfg_dynamic_shadows',
    'setting.videocfg_texture_detail',
    'setting.videocfg_particle_detail',
    'setting.videocfg_ao_detail',
    'setting.videocfg_hdr_detail',
    'setting.videocfg_fsr_detail',
)

# setting.shaderquality -> S_SHADER_QUALITY is the identity mapping, and the
# global preset maps onto it through cfg/video_defaults_N.txt.
# RENDER_SETTINGS_AXES.md:486-492. READ, not fitted.
PRESET_TO_SHADER_QUALITY = {0: 0, 1: 0, 2: 1, 3: 1}


# --------------------------------------------------------------------------
# THE PRESET -> OUR-AXIS MAP, and its GAP LIST (#91 step 1)
# --------------------------------------------------------------------------
# A preset is a VECTOR of 11 cvars, READ from the committed gtq manifest --
# not four names with a fidelity ordering. Rendering "at preset2" means
# resolving all 11, and a renderer that resolves two of them and renders
# anyway has not rendered at preset2; it has rendered at its own defaults
# with two values changed and a preset's name on the artifact.
#
# So every cvar gets a row, and a row is either MAPPED (there is an axis, and
# the cvar->value mapping was READ) or a GAP with its KIND. The gap list is
# the deliverable: it is the honest inventory of reference options this
# renderer does not implement, and it is the queue that #91's later steps
# work off. It is a floor, not a ceiling -- a MAPPED row asserts the axis is
# wired and the mapping is read, never that our pass matches the reference's.
#
# THE THREE GAP KINDS ARE DIFFERENT PROBLEMS AND ARE NOT MERGED:
MAPPED = 'MAPPED'
GAP_NO_AXIS = 'GAP:NO_AXIS'         # the cvar's meaning is known/read and
                                    # this renderer has no flag for it at all
GAP_UNREAD_TABLE = 'GAP:UNREAD_TABLE'   # we have a plausible flag, but the
                                    # cvar tier -> value table has NOT been
                                    # read out of the engine. Guessing it is
                                    # fitting a constant.
GAP_UNRESOLVED = 'GAP:UNRESOLVED'   # refused upstream, with a stated route
                                    # to resolve it
GAP_NOT_APPLIED = 'GAP:NOT_APPLIED'  # the REFERENCE did not apply this cvar
                                    # in the corpus we hold. Not our gap to
                                    # close, and not calibratable from here.

# --------------------------------------------------------------------------
# THE MEASURED LEVERAGE OF EACH AXIS (#91 step 2)
# --------------------------------------------------------------------------
# gtq_axis_response.py, over the 48-pose de_inferno GT corpus. Each arm is
# preset0 + ONE cvar, so its distance from preset0 IS that cvar's leverage.
# Quoted as a ratio to a SAME-CONFIG re-capture, because the captures are an
# x11grab of a live engine and are not bit-reproducible: the raw distance
# alone ranked every arm as a hit, including preset0 against itself.
#
# The floor is validated by FOUR independent same-config repeats
# (preset0_rep, w5_p0rep1/2/3, preset0_warmup), which land at 0.98-1.02.
# Pose 00000 is dropped -- it matches an unrelated index in every arm.
# Two arms are REFUSED as stuck-camera captures (#93).
#
# WHAT A FLOOR READING MEANS, and it is the narrower of the two claims:
# "this 48-pose de_inferno corpus does not exhibit the cvar", NOT "the cvar
# does nothing in CS2". A static pose set has no particles to budget and
# few dynamic shadows to cast, so an axis can be real and still measure
# flat here. That is a FIXTURE limit and it is recorded as one.
#
# ⚠️ AND THE FIXTURE LIMIT IS WORSE THAN THE POSE SET (#94). The floor is
# CONFIG-DEPENDENT and dominated by FSR, measured in this module's own
# metric over the same 47 poses:
#
#     preset0 vs preset0_rep   fsr_detail=3, FSR ON    box8 0.4105
#     preset2 vs preset2_rep   fsr_detail=0, FSR OFF   box8 0.0488
#
# -- 8.4x at box8, and 31x at full resolution (2.5962 vs 0.0826). EVERY
# archived single-axis arm is preset0 + one key, and preset0 sits at
# fsr_detail=3, so every ratio in this table is quoted against the NOISIER
# of the two available floors.
#
# The consequence is a DETECTION THRESHOLD, and it is the honest form of
# every "at the floor" row below: this corpus cannot resolve an effect
# smaller than ~0.41 box8. An axis under that is UNRESOLVED, not absent,
# and the ratios near 1.00 are upper bounds rather than measurements. The
# fix is not a better metric -- it is re-capturing the single-axis arms at
# fsr_detail=0, which drops the threshold to ~0.049 and is now one command
# per arm through the gt-capture service (#92). That is the single
# highest-value corpus action for #91, and it is a request, not a plan:
# ao_detail, particle_detail, texture_detail, cmaa and shadow_quality all
# currently sit in the band the FSR-ON floor cannot see into.
AXIS_RESPONSE = {
    # cvar: (box8 ratio to the floor, one-line reading)
    'setting.shaderquality': (
        8.27, "the largest real axis in the corpus, and it accounts for "
              "essentially ALL of preset2/preset3's 8.60-8.67 -- those two "
              "presets differ from preset0 by little more than q0->q1. This "
              "is #57's battle with a number on it."),
    'setting.videocfg_fsr_detail': (
        4.00, "tier 3 (preset0's value) vs 0/1/2 measures 3.93/3.64/3.46, "
              "but tier 4 measures 0.99 -- INDISTINGUISHABLE from tier 3. "
              "So the tier set collapses to {0}, {1}, {2}, {3,4}: four "
              "render scales, not five. The numeric scales are still "
              "unread; the equivalence structure is now read."),
    'setting.videocfg_hdr_detail': (
        2.53, "a REAL response -- the scene-colour format reaches the "
              "image. This is the measured argument for closing it first."),
    'setting.r_texturefilteringquality': (
        2.23, "tiers 3/4/5 measure 2.22/2.23/2.23 -- mutually identical. "
              "Tier 2 is 2.08 and tier 1 is 1.48, at the floor with tier 0. "
              "The six tiers collapse to THREE distinguishable states "
              "{0,1}, {2}, {3,4,5}, so the table to read has three rows, "
              "not six."),
    'setting.videocfg_texture_detail': (
        2.04, "tier 2 is marginally above the floor, tier 1 (1.18) is at "
              "it. One usable step in the corpus."),
    'setting.videocfg_shadow_quality': (
        1.39, "tiers 2 and 3 measure 1.33/1.39, at the floor. Tier 1's "
              "archived arm is a STUCK-CAMERA capture (#93) and its "
              "re-capture puts it at ~1.24x. The whole axis is flat HERE -- "
              "a 48-pose set with little dynamic shadow cannot rank a "
              "shadow axis, which is a fixture limit, not a finding about "
              "shadow quality."),
    'setting.r_csgo_cmaa_enable': (
        1.24, "measured on w5_cmaa1, because the primary arm is a "
              "stuck-camera capture (#93). At the floor -- and note that "
              "box-downsampling removes exactly the frequencies an AA pass "
              "lives in, so read the full-res column (1.04), which is also "
              "at the floor."),
    'setting.videocfg_ao_detail': (
        1.04, "tiers 2 and 3 both at the floor (1.04, 1.02)."),
    'setting.videocfg_particle_detail': (
        1.09, "tiers 1/2/3 at 0.96/0.98/1.09, all floor. EXPECTED: the "
              "corpus is 48 static teleport poses with no particle systems "
              "running, so this axis has nothing to act on. Ranking it "
              "needs a different fixture, not a better metric."),
    'setting.videocfg_dynamic_shadows': (
        1.00, "at the floor to two decimals -- and the engine agrees: "
              "'Unable to read video config convar videocfg_dynamic_shadows'"
              " appears in 58 of 58 capture console logs, the ONLY cvar in "
              "the set the engine reports it cannot read. Two independent "
              "routes, same answer: the reference never applied it."),
    'setting.msaa_samples': (
        None, "NOT SEPARABLE from this corpus: every archived msaa arm also "
              "moves videocfg_fsr_detail, and the _fsr0 variants are not in "
              "caps/. Our own 4-preset render is the other half of this "
              "gap -- msaa 1/2/4/8 produced BIT-IDENTICAL images, so the "
              "axis is dead on OUR side and unmeasured on theirs."),
}

# NO_AXIS gaps, PRICED rather than implemented (#91 step 2 mandate). The
# cost is what the axis implies in passes and wiring; the value is its
# MEASURED leverage above. Neither is a guess at the other.
NO_AXIS_COST = {
    'setting.videocfg_hdr_detail': (
        "SMALL", "Allocate the scene-colour buffer as B10G11R11_UFLOAT vs "
        "R16G16B16A16_SFLOAT and fix the MSAA resolve's clamp: 65504 is the "
        "largest finite HALF, i.e. the RGBA16F constant, and it is WRONG "
        "for R11G11B10. The format machinery is already written -- "
        "post/msaa_resolve.store_to_attachment() takes a vk_format, and "
        "post/video_config.HDR_FORMAT_BY_TIER already reads the tier map. "
        "The work is wiring post/ in at all (nothing imports it) plus one "
        "buffer allocation. HIGHEST VALUE ON THE LIST: 2.53x measured, it "
        "is the format 2 of 4 presets and nearly every single-axis arm "
        "run on, and the gate corpus is preset2 so the gate cannot catch a "
        "wrong R11G11B10 path."),
    'setting.videocfg_texture_detail': (
        "SMALL-MEDIUM", "A mip-bias / max-resident-mip parameter through "
        "the texture sampling path. Adjacent axes already exist "
        "(--mip-color, --mip-aux, --max-aniso), so this is a parameter and "
        "a plumbing change, not a new pass. Value: one usable step (2.04x) "
        "in this corpus."),
    'setting.videocfg_particle_detail': (
        "MEDIUM", "Drive smoke_mboit's step/sample counts from the tier. "
        "The pass exists and takes its own flags; nothing joins them to "
        "this cvar. WARNING BEFORE ANYONE STARTS: the GT corpus measures "
        "this axis at the floor because 48 static poses run no particles, "
        "so implementing it CANNOT be validated against gtq as it stands. "
        "It needs a fixture with live particles first -- otherwise the "
        "A/B is vacuous on a non-exhibiting fixture."),
    'setting.r_csgo_cmaa_enable': (
        "LARGE", "Three compute passes -- cs2_cmaa2_cs_edges, "
        "_process_candidates, _deferred_apply are all decompiled and "
        "committed -- plus the edge and candidate buffers between them and "
        "an indirect dispatch. Value: at the floor in this corpus at both "
        "scales, and its primary GT arm needs re-capturing (#93) before "
        "any per-tier claim is possible."),
    'setting.videocfg_ao_detail': (
        "LARGE", "A screen-space AO pass this renderer does not have. The "
        "AO flags here (--vmat-ao, --ao-page-product, --atrous-baked-ao) "
        "are BAKED-AO axes -- a different quantity -- so this is new "
        "surface, not a re-parameterisation. Value: both tiers at the "
        "floor."),
}


def _msaa(v):
    # post.video_config.msaa_sample_count: 0 in the config means
    # single-sample. Our flag spells single-sample as 1 (the attachment is
    # gated on >1), so the two spellings of the same state are joined here
    # rather than by whoever types the command.
    n = int(v)
    if n not in (0, 2, 4, 8):
        raise ValueError(f"msaa_samples={v} is not a legal count")
    return 1 if n == 0 else n


def _post_vcfg():
    """post.video_config, imported however this module was entered.

    render_config.py runs both as `python render_config.py` from
    src/counter_strike_render (so that directory is sys.path[0]) and as an
    import from the repo root. The two spellings of the package differ,
    and the tables below are READ from it rather than copied -- a second
    copy is a second thing to keep in step, which is how a table and its
    consumer drift apart silently."""
    try:
        from counter_strike_render.post import video_config
    except ImportError:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from counter_strike_render.post import video_config
    return video_config


def _hdr_format(v):
    # post.video_config.HDR_FORMAT_BY_TIER, consulted rather than copied.
    HDR_FORMAT_BY_TIER = _post_vcfg().HDR_FORMAT_BY_TIER
    tier = int(v)
    if tier not in HDR_FORMAT_BY_TIER:
        raise KeyError(
            f"videocfg_hdr_detail={v} was never exercised by any of the 30 "
            f"GT arms, so which HDR format the engine allocates for it has "
            f"not been read. Capture an arm at that tier and read the "
            f"RenderPipelineCsgo RT line rather than assuming")
    return HDR_FORMAT_BY_TIER[tier]


def _fsr(v):
    # post.video_config.FSR_TIER_SCALE, consulted rather than duplicated:
    # a second copy of the table here is a second thing to keep in step,
    # and the renderer reads the same dict.
    #
    # THE AXIS NOW EXISTS. --fsr-detail shrinks the SCENE RASTER by the
    # tier's render scale and presents through EASU (+RCAS), both
    # transcribed in post/fsr.py from the shipped modules. So the only
    # thing still missing for tiers 1-4 is the SCALE ITSELF, and that
    # stays a refusal: AMD's published quality-mode ratios (1.0 / 1.3 /
    # 1.5 / 1.7 / 2.0) are FSR's defaults, not necessarily Valve's
    # selection, and putting them here would be a fitted float in every
    # pixel of preset0 and preset1.
    _vc = _post_vcfg()
    FSR_TIER_SCALE = _vc.FSR_TIER_SCALE
    FSR_TIER_UNRESOLVED_NOTE = _vc.FSR_TIER_UNRESOLVED_NOTE
    tier = int(v)
    if tier not in FSR_TIER_SCALE:
        raise ValueError(f"videocfg_fsr_detail={v} is not a known tier")
    if FSR_TIER_SCALE[tier] is None:
        raise KeyError(
            f"videocfg_fsr_detail={v}: {FSR_TIER_UNRESOLVED_NOTE} The PASS "
            f"is implemented (post/fsr.py, --fsr-detail); what is missing "
            f"is this tier's render scale")
    return tier


PRESET_AXIS_MAP = {
    'setting.shaderquality': dict(
        status=MAPPED, flag='--shader-quality', resolve=lambda v: int(v),
        basis='identity map onto S_SHADER_QUALITY; global preset N maps '
              'through cfg/video_defaults_N.txt (RENDER_SETTINGS_AXES.md'
              ':486-492). READ. THE DRIVE IS NOW WIRED: --texture-detail takes the tier and post.detail_tiers.TEXTURE_DETAIL holds the (empty) value table, so a landed table is a table edit rather than an implementation. Until then the flag RECORDS the tier and refuses to apply values, and says which. THE DRIVE IS NOW WIRED: --particle-detail, against post.detail_tiers.PARTICLE_DETAIL. And the precondition is wired too rather than only written down -- the renderer prints a particle CENSUS for the fixture it is rendering, so a null measured on a render carrying zero smoke instances cannot be read later as evidence the tier is inert. THE DRIVE IS NOW WIRED: --ao-detail, against post.detail_tiers.AO_DETAIL, which carries tier 1 as an explicit None-with-a-reason rather than omitting it -- a missing key and an unexercised tier are different facts.'),
    'setting.msaa_samples': dict(
        status=MAPPED, flag='--msaa-samples', resolve=_msaa,
        basis='post.video_config.msaa_sample_count; 0 means single-sample, '
              'which our flag spells 1. READ.'),
    'setting.videocfg_fsr_detail': dict(
        status=MAPPED, flag='--fsr-detail', resolve=_fsr,
        basis='THE TABLE IS NOW READ, AND THE REFUSAL IS DISCHARGED. Two '
              'independent routes. (1) The engine\'s own UI resource: '
              'game/csgo/pak01_dir.vpk -> panorama/layout/settings/'
              'settings_video.vxml_c gives fsr0..fsr4 against Disabled / '
              'Ultra Quality / Quality / Balanced / Performance, so the '
              'axis is QUALITY-INVERTED -- higher tier, more aggressive '
              'upscale -- which video_defaults confirms (level0 -> 3, '
              'level1 -> 2, level2/3 -> 0). (2) The arms: the internal '
              'render grid imprints a sampling comb, and its beat against '
              'the 1280 display grid puts a second comb at 1280 - Wi, so '
              'the exact internal rasters are readable off the captures -- '
              '985x554, 857x482, 755x424. Those are floor(dim * pct) for '
              'pct 0.77/0.67/0.59 and NOT round(dim / 1.3|1.5|1.7): two of '
              'the three rows discriminate between the two forms (857 vs '
              '853, 755 vs 753), which is what makes it a read rather than '
              'a fit that landed. TIER 4 EXECUTES AS 0.59, NOT the 0.50 '
              'its "Performance" label implies -- its arm is '
              'indistinguishable from tier 3 on every instrument and '
              'inside the same-config repeat floor, and a 50% render is '
              'ruled out by the PRESENCE of the 755/525 comb. We '
              'reproduce what the engine did. STILL UNREAD: the RCAS '
              'sharpness constant (a runtime uniform; both '
              'upsample_*.vmat_c carry empty m_floatParams) and '
              'r_csgo_fsr_enable_mip_bias, so this renderer applies no FSR '
              'mip bias and says so. WHETHER RCAS runs IS read: tiers 2-4 '
              'gain 11-25% mid-band energy over native (sharpening) while '
              'tier 1 loses energy in every band and allocates no extra '
              'full-res LDR target -- Ultra Quality is EASU alone.'),

    'setting.r_csgo_cmaa_enable': dict(
        status=MAPPED, flag='--cmaa',
        resolve=lambda v: 'on' if int(v) else 'off',
        basis='CMAA2 is now IMPLEMENTED, all three shipped compute shaders '
              '(post/cmaa2.py from cs2_cmaa2_cs_edges.glsl:32-121, '
              '_process_candidates:52-397 and _deferred_apply:80-89). The '
              'edge rule -- an absolute 0.15 threshold AND 10% of the '
              'largest PERPENDICULAR gradient in the tap 2x2 -- was '
              'DERIVED from the shader\'s own shared-memory indexing, and '
              'the four direction bits were read off the CONSUMER\'s tap '
              'offsets rather than off the component order, which is the '
              'transposed assignment. NOTE THE SCOPE: this cvar is 0 in '
              'ALL FOUR presets, so it contributes nothing to the preset '
              'ladder; the arm that exercises it is '
              'gtq/caps/r_csgo_cmaa_enable-1.'),
    'setting.r_texturefilteringquality': dict(
        status=MAPPED, flag='--texture-filtering-quality',
        resolve=lambda v: int(v),
        basis='READ, and the superseded basis for this row was WRONG in '
              'three separate ways -- see post/texture_filtering.py, which '
              'records them. The tier -> sampler-state table comes out of '
              "the engine's own shipped UI resource: "
              'game/csgo/pak01_dir.vpk -> panorama/layout/settings/'
              'settings_video.vxml_c, whose decompressed KV3 string table '
              'lists six contiguous label/element-id pairs, '
              'matforceaniso0..5 against Bilinear / Trilinear / '
              'Anisotropic 2X / 4X / 8X / 16X. So aniso 1, 1, 2, 4, 8, 16 '
              'with the MIP FILTER changing point->linear at 0->1, and no '
              'per-tier LOD bias. The arms confirm the structure '
              'independently: tier 1 correlates only 0.58-0.65 with tiers '
              '2-5 and is BLURRIER than tier 0, which an anisotropy ladder '
              'cannot produce and a point->linear mip change is; tiers 2-5 '
              'correlate ~1.0 with each other at increasing amplitude. '
              'WHAT THE CORPUS DOES NOT ARBITRATE: tiers 3/4/5 sit at the '
              'same-config repeat floor against each other (pairwise box8 '
              '0.5020/0.5226/0.4723 vs a 0.4624-0.5090 floor), so 4/8/16 '
              'is the UI resource\'s read and the corpus is CONSISTENT '
              'with it rather than confirming it. The old basis called '
              'that mapping "contradicted by the reference"; below a '
              "fixture's threshold is a fact about the fixture."),
    'setting.videocfg_shadow_quality': dict(
        status=GAP_UNREAD_TABLE, flag='--gt-csm / --gt-csm-cascades',
        basis='MEASURED: THE AXIS IS BINARY ON THIS FIXTURE, {0} vs '
              '{1,2,3}. Tier-to-tier separation among 1/2/3 is at or '
              'BELOW the same-config repeat floor at every scale '
              'K=1..16, and their signed diff maps against preset0 '
              'correlate r=0.962-0.978 with each other while '
              'correlating 0.05-0.18 with the repeat controls. So '
              'preset1, preset2 and preset3 are IDENTICAL to each '
              'other on this axis and the whole ladder effect is the '
              '0->1 step (K64 2.96x floor on the CLEAN recap). The '
              'character of that step is measured too: a shadow '
              'COVERAGE change -- mid-luminance world surfaces darken, '
              '75-81% one-signed, near-ground weighted with the '
              'horizon band SUPPRESSED, which is evidence AGAINST a '
              'cascade-distance reading since that would band at a '
              'mid-screen depth instead. ⚠️ #93 UPDATED: the archived '
              'tier-1 arm on disk is STILL the stuck-camera capture '
              '(consecutive-pose diff 9x below every clean arm, and '
              'its log shows a different capture lane -- 55 gtvm_step '
              'execs, zero gtpose); the clean recap lives at '
              'gtq_ondemand/recap_shadowquality1, and its effect is '
              'the LARGEST of the three tiers, not the smallest. '
              'THE VALUES REMAIN UNREAD. '
              'THE SYSTEM IS READ, THE MECHANISM IS NOT, AND THREE ROUTES '
              'ARE NOW CLOSED. The engine\'s settings resource binds this '
              'to csmqualitylevel0..3 under "Global Shadow Quality", so it '
              'is cascaded shadow maps -- but the ids carry no resolution '
              'or cascade-count token and the labels are Low/Medium/High/'
              'Very High, which are words. CLOSED: (1) a csm/cascade/'
              'shadowmap search over BOTH depot archives returns only '
              'de_aztec waterfall props, no CSM material or .res; (2) all '
              '58 console logs have zero csm/cascade hits; (3) the '
              'RT-allocation dumps are IDENTICAL across all tiers -- and '
              'that closure is IMMUNE TO #93 because it reads the '
              'boot-time allocation dump rather than the frames, so the '
              'stuck-camera arm cannot corrupt it. The CSM atlas is simply '
              'not in the scratch RT pool, so unlike FSR this axis leaves '
              'no trace in the logs. perftest.cfg names '
              'lb_csm_cascade_size_override (1024/2048), which proves a '
              'per-cascade SIZE concept and is NOT the ladder -- those are '
              'the benchmark\'s own values for a Source1 comparison and '
              'there is no cascade COUNT in the file. HIGHEST-VALUE '
              'REMAINING ROUTE: a RenderDoc capture per tier on ws-1, '
              'where RenderDoc 1.6.0 is already attached -- it answers '
              'count AND resolution in one capture and does not need the '
              'fixture to exhibit shadows, which matters because #93 makes '
              'the arms the weakest route for exactly this axis. The '
              'earlier basis\'s "this corpus cannot calibrate it" stands '
              'and is now explained rather than merely observed.'),
    'setting.videocfg_dynamic_shadows': dict(
        status=GAP_NOT_APPLIED, flag='--animated-shadows',
        basis='THE REFERENCE NEVER APPLIED IT. The engine prints "Unable to '
              'read video config convar videocfg_dynamic_shadows" in 58 of '
              '58 capture console logs -- the only cvar in the set it says '
              'that about -- and the single-axis arm measures 1.00x the '
              'noise floor against four same-config controls. Two '
              'independent routes agree. Mapping our --animated-shadows '
              'onto it would pair our axis against a reference value that '
              'never reached a render, so this stops being an '
              'implementation gap and becomes a corpus fact. Re-opening it '
              'needs a build where the convar reads back, not more '
              'analysis of these captures. AND THE MECHANISM IS NOW READ, '
              'which does not reopen it: the settings resource gives '
              'dynamicshadowslevel0/1 with INLINE labels "Sun Only" '
              'and "All", tooltip "the light sources for which '
              'dynamic objects such as players and smoke cast '
              'shadows". So tier 0 is the sun alone and tier 1 is all '
              'lights. Worth having for the build where the convar '
              'reads back; it changes nothing about this corpus.'),
    'setting.videocfg_texture_detail': dict(
        status=GAP_UNREAD_TABLE, flag='--texture-detail',
        basis='⚠️ MEASURED, AND IT REFUTES THE MIP READING THIS ROW GAVE '
              'TWICE -- including the correction I made an hour '
              'earlier. It is NOT a mip bias, NOT a residency cap and '
              'NOT a blur: a blur fit minimises at sigma=0 exactly as '
              'it does for the same-config controls, HF power moves '
              '0.7% where one mip step HALVES it, and the signal GROWS '
              'under spatial averaging (1.14x floor at K1 -> 3.82x at '
              'K64) instead of shrinking. The RT table says what it is '
              'instead: tiers 1 and 2 both allocate a screen-space '
              'MULTI-RESOLUTION BUFFER FAMILY gaining a full-res '
              'level (+6 RTs, 1280x720 copies of R32F/RG1616F/'
              'RG3232F/2x RGBA16161616F/RGBA32323232F, the 320x180 and '
              '640x360 copies retained). Residency does not allocate '
              'screen-resolution targets. MEASURED STATES {0,1} vs '
              '{2}: tier 1 is an image NO-OP (K16 1.14x floor, and its '
              'diff map correlates r=0.672 with the preset0-re-run '
              'component) that allocates tier 2 buffers without using '
              'them. Implement as two-state, never as a mip knob. '
              'THE UI BINDING STILL STANDS AND IS CONSISTENT: '
              'gpumemlevel0/1/2 -- the engine\'s GPU-MEMORY level, the '
              'same axis as the setting.gpu_mem_level header key -- so '
              'it is a RESIDENCY-BUDGET ladder, not a sampler one. That '
              'makes --mip-max-levels the closer of our two knobs and '
              '--mip-bias the further, the opposite of the ordering '
              'this row gave before. What the budget does to the mip '
              'chain is still unread: gpumemlevel{N} carries no '
              'resolution or LOD token, exactly as matforceaniso{N} '
              'carried no bias token. PRIOR STATUS CORRECTION, kept: '
              'this row read GAP_NO_AXIS on the basis '
              'that "no mip-bias or residency axis exists in this '
              'renderer", and that is FALSE: --mip-bias adds to the '
              'computed LOD and --mip-max-levels caps the chain depth, '
              'both of which have been in the renderer since the mip '
              'chain landed. A row that says the axis does not exist puts '
              'a cvar on the "needs new machinery" list when it is '
              'actually on the "needs the table read" list, which is a '
              'different and much cheaper piece of work -- and it is the '
              'inventory misreporting itself, which is the one thing this '
              'table exists to prevent. What is genuinely missing is the '
              'tier -> (bias, resident-mip) table. Its route is the arms: '
              'videocfg_texture_detail-1 and -2 against preset0, where a '
              'mip-bias axis leaves a signature (confined to textured '
              'surfaces, growing with distance and grazing angle, a '
              'low-pass of the base arm rather than a colour shift) that '
              'a residency cap does not.'),
    'setting.videocfg_particle_detail': dict(
        status=GAP_UNREAD_TABLE, flag='--particle-detail',
        basis='⚠️ THE PRECONDITION FIRED: THE FIXTURE CANNOT EXHIBIT THIS '
              'AXIS, stated with numbers rather than as a quiet null. '
              'Tiers 1 and 2 clear the same-config floor on 1 and 0 of '
              '47 poses against a ~9/47 chance rate -- LESS OFTEN THAN '
              'CHANCE -- with RT tables byte-identical to tier 0. Tier '
              '3 looked like signal and was a 32-PIXEL FRAME-EDGE '
              'ARTIFACT (4.301 mean|d| in x>=1888 against 2.623 for the '
              'rest of the frame); excluded, it sits inside the repeat '
              'cloud. The map DOES carry particle entities -- every '
              'log including preset0 shows the same env_particle_glow '
              'and info_particle_system slots -- they just produce no '
              'visible pixels at these 47 teleport poses. WHAT IS '
              'READ: tier 3 alone moves the RT table (+3 x 1280x720 '
              'RGBA8888), so the engine boundary is 2->3, not 0->1 or '
              '1->2. Do NOT implement tier values from this corpus. '
              'PRIOR NOTE, kept: '
              'The old basis said "smoke_mboit takes its own step/sample '
              'flags, none of which is driven by this tier" -- the second '
              'clause is true and the FIRST clause makes it a gap in the '
              'DRIVE, not in the axis. smoke_mboit declares '
              'D_SMOKE_QUALITY itself (--smoke-quality, selecting '
              '--smoke-march-steps vs --smoke-march-steps-lowq), plus '
              'D_SMOKE_FULLRES, the light-step and shadow-tap counts and '
              'the DDA step budget. So the machinery is there and the '
              'tier -> value table is not. CAUTION ON THE ROUTE: a 48-pose '
              'teleport set on de_inferno may carry NO particles at all, '
              'in which case the arms cannot exhibit the axis and a null '
              'is vacuous rather than a finding -- that has to be '
              'established against the same-config repeat floor before '
              'any tier row is written.'),
    'setting.videocfg_ao_detail': dict(
        status=GAP_UNREAD_TABLE, flag='--ao-detail',
        basis='STATUS CORRECTED, and this one was the most misleading of '
              'the three. The old basis said "The AO flags here '
              '(--vmat-ao, --ao-page-product, --atrous-baked-ao) are '
              'BAKED-AO axes, a different quantity", which is true of the '
              'three flags it names and false about the renderer: it '
              'enumerated the baked-AO family and missed the '
              'SCREEN-SPACE one entirely. --ssao drives a transcription '
              'of the three shipped SSAO modules '
              '(glsl_loopfam/ssao_r0m0.glsl, '
              'ssao_scalable_ambient_obscurance_r0m3.glsl, '
              'ssao_bilateral_blur_r0m0.glsl) in '
              'passes/loopfam/ao.py, with its own conformance family, and '
              '--ssao-samples / --ssao-turns / --ssao-mips / --ssao-blur / '
              '--ssao-radius are exactly the tier-shaped knobs a quality '
              'level would key. A gap row that lists the wrong flags and '
              'concludes NO_AXIS is worse than one that says nothing: it '
              'closes the question against an implementation that is '
              'already there. What is missing is the tier -> (sample '
              'count, mip count, blur width) table. NOTE THE OBSERVED '
              'DOMAIN: the 53 GT configs only ever carry 0, 2 and 3 -- '
              'tier 1 is never exercised, so any table proposing a row '
              'for it is proposing a row no capture can check. STRENGTHENED: '
              'tier 1 does not merely go unexercised -- the engine\'s '
              'own settings resource enumerates aoproxy0, aoproxy2, '
              'aoproxy3 and SKIPS 1, so it is a row that does not '
              'exist rather than one no capture can check. The corpus '
              'gap and the resource gap agreeing exactly is also the '
              'strongest check that these element-id suffixes really '
              'are the cvar values here -- they are NOT in general: '
              'hdr_detail\'s ids are hdr0/hdr1 against values 3/-1.'),
    'setting.videocfg_hdr_detail': dict(
        status=MAPPED, flag='--hdr-scene-format',
        resolve=lambda v: _hdr_format(v),
        basis='the tier -> format map was already READ from the engine\'s '
              'own RT enumeration across all 30 GT arms with no '
              'exceptions (post.video_config.HDR_FORMAT_BY_TIER: 3 -> '
              'B10G11R11_UFLOAT, -1 -> R16G16B16A16_SFLOAT); what was '
              'missing was the axis, and --hdr-scene-format is it. '
              'AND THE AXIS IS NOT THE CLAMP. store_to_attachment() '
              'already applied the per-format representable maxima and '
              'that changed NOTHING, because nothing on this map reaches '
              '64512 -- which is why the row could honestly say "both '
              'tiers render the same buffer" while the range was already '
              'implemented. What differs is MANTISSA WIDTH: RGBA16F is '
              '10 bits, R11G11B10 is 6/6/5, a 16x coarser quantisation of '
              'every scene pixel, and that is what a 2.53x measured axis '
              'response is made of. post/scene_format.py implements the '
              '(1, 5, M) encode/decode including the denormal branch, and '
              'is calibrated against hardware: at M=10 the same code path '
              'reproduces numpy float16 BIT-EXACTLY over 20001 samples.'),
}


def resolve_preset(tag, manifest_path=MANIFEST):
    """(applied, gaps, vector) for a GT preset tag. READS the manifest.

    `applied` is [(cvar, gt_value, flag, our_value, basis)] -- the axes this
    renderer can actually put in the reference's position.
    `gaps` is [(cvar, gt_value, kind, flag_or_None, why)] -- the inventory.
    Neither list is truncated and neither is summarised away.
    """
    blocks = gt_presets(manifest_path)
    if tag not in blocks:
        raise SystemExit(
            f"REFUSED: no GT capture tagged {tag!r} in {manifest_path}. "
            f"The preset vector is READ from the manifest, never invented. "
            f"Known: {', '.join(sorted(blocks)) or '(none)'}")
    vector = blocks[tag]['preset']
    unmapped = [k for k in vector if k not in PRESET_AXIS_MAP]
    if unmapped:
        raise SystemExit(
            f"REFUSED: PRESET_KEYS carries {unmapped} with no row in "
            f"PRESET_AXIS_MAP. A cvar with no row is neither mapped nor "
            f"declared a gap -- it is invisible, which is the one state "
            f"this table exists to make impossible.")
    applied, gaps = [], []
    def _lev(cvar, basis):
        """Append the MEASURED leverage, so a gap row carries its priority."""
        r = AXIS_RESPONSE.get(cvar)
        if not r:
            return basis
        ratio, note = r
        head = ("leverage UNMEASURED" if ratio is None
                else f"leverage {ratio:.2f}x the same-config noise floor")
        cost = NO_AXIS_COST.get(cvar)
        tail = f" COST {cost[0]}: {cost[1]}" if cost else ""
        return f"{basis}  [{head} -- {note}]{tail}"

    for cvar in PRESET_KEYS:
        val = vector.get(cvar)
        row = PRESET_AXIS_MAP[cvar]
        if val is None:
            gaps.append((cvar, None, GAP_UNRESOLVED, row.get('flag'),
                         f"the manifest's {tag} block carries no value for "
                         f"this cvar, so there is nothing to resolve"))
            continue
        if row['status'] != MAPPED:
            gaps.append((cvar, val, row['status'], row.get('flag'),
                         _lev(cvar, row['basis'])))
            continue
        try:
            ours = row['resolve'](val)
        except (KeyError, ValueError) as e:
            gaps.append((cvar, val, row.get('gap_kind', GAP_UNRESOLVED),
                         row.get('flag'), _lev(cvar, str(e).strip("'"))))
            continue
        if ours is None or row.get('flag') is None:
            applied.append((cvar, val, None, None, row['basis']))
        else:
            applied.append((cvar, val, row['flag'], ours, row['basis']))
    return applied, gaps, vector


# AUTO-RESOLVED INPUTS. The renderer resolves some inputs itself and says so:
#     irradiance: auto -> /home/user/worlds/de_inferno.irradiance.npy
# No flag carries that path, so v2's flag-driven pinning missed it entirely,
# and NO stamped config anywhere carried the sha of the irradiance atlas the
# dominant lighting term samples. Swap that file and every config still
# verifies. It is #55's class exactly -- an input that is not on the command
# line cannot drift loudly, so it drifts silently -- and the fact that it is
# resolved BY THE RENDERER rather than by the operator makes it more
# dangerous, not less: nobody types it, so nobody thinks about it.
#
# So capture scans the log for its own auto-resolution announcements and pins
# whatever they name that is a real file or directory on disk.
_AUTO = re.compile(r'^\s*(?:--)?([A-Za-z0-9_-]+)\s*:?\s*auto\s*->\s*(\S+)')


def sha_dir(d):
    """(rolled sha, {name: sha}) for a directory's files, or (None, {}).

    Recursive, sorted, and the roll includes the NAMES, so adding a file or
    renaming one moves the roll. A directory pinned only by the shas of the
    files present when it was captured cannot detect an addition, and an
    extra mip in a prefiltered chain changes which one the LOD lands on.
    """
    if not d or not os.path.isdir(d):
        return None, {}
    per = {}
    for root, _dirs, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            per[os.path.relpath(p, d)] = sha(p)
    h = hashlib.sha256()
    for name in sorted(per):
        h.update(name.encode())
        h.update(b'\0')
        h.update((per[name] or 'ABSENT').encode())
        h.update(b'\n')
    return h.hexdigest(), per


def _json_file_refs(path, blob, out):
    """Every string in `blob` that resolves to an existing file, recursively."""
    base = os.path.dirname(os.path.abspath(path))
    if isinstance(blob, dict):
        for v in blob.values():
            _json_file_refs(path, v, out)
    elif isinstance(blob, list):
        for v in blob:
            _json_file_refs(path, v, out)
    elif isinstance(blob, str) and blob:
        p = blob if os.path.isabs(blob) else os.path.join(base, blob)
        if os.path.isfile(p):
            out.add(os.path.normpath(p))


def sha_manifest(path):
    """(rolled sha, {name: sha}) over a manifest AND every file it names."""
    if not path or not os.path.isfile(path):
        return None, {}
    per = {os.path.basename(path): sha(path)}
    try:
        with open(path) as f:
            blob = json.load(f)
    except (ValueError, OSError):
        # Unparseable: pin the manifest itself and SAY the references could
        # not be read, rather than silently pinning one level.
        per['__UNREADABLE__'] = 'references not enumerable'
        blob = None
    if blob is not None:
        refs = set()
        _json_file_refs(path, blob, refs)
        for p in sorted(refs):
            per[os.path.relpath(p, os.path.dirname(os.path.abspath(path)))] = sha(p)
    h = hashlib.sha256()
    for name in sorted(per):
        h.update(name.encode()); h.update(b'\0')
        h.update(str(per[name]).encode()); h.update(b'\n')
    return h.hexdigest(), per


def pin_inputs(pairs, resolve=lambda v: v):
    """{flag: {...sha...}} for every file and directory input on the argv."""
    inputs = {}
    for k, v in pairs:
        if not v:
            continue
        if k in PATH_FLAGS:
            p = resolve(v)
            inputs[k] = {'path': p, 'sha256': sha(p), 'kind': 'file'}
        elif k in DIR_FLAGS:
            p = resolve(v)
            rolled, per = sha_dir(p)
            inputs[k] = {'path': p, 'sha256': rolled, 'kind': 'dir',
                         'files': per}
        elif k in MANIFEST_FLAGS:
            p = resolve(v)
            rolled, per = sha_manifest(p)
            inputs[k] = {'path': p, 'sha256': rolled, 'kind': 'manifest',
                         'files': per}
    return inputs


def parse_argv(line):
    toks = line.split()
    if toks and toks[0].endswith('.py'):
        toks = toks[1:]
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith('--') and i + 1 < len(toks) and not toks[i + 1].startswith('--'):
            out.append((t, toks[i + 1])); i += 2
        else:
            out.append((t, None)); i += 1
    return out


# --------------------------------------------------------------------------
# Renderer defaults, read statically
# --------------------------------------------------------------------------
# The renderer's real argparse object. Calls on any OTHER receiver are the
# renderer's own selftest scratch parsers and are NOT part of its CLI.
PARSER_NAME = 'parser'


def renderer_defaults(renderer_path):
    """{flag: default} for every add_argument in the renderer, via `ast`.

    gpu_render.py parses argv at import, so importing it to ask for its
    defaults is not available. Its `parser.add_argument(...)` calls are
    still plain literals in the source, so they are read directly. A flag
    whose default is a computed expression rather than a literal is
    returned as the marker '<computed>' -- reported, not silently treated
    as absent, because "we could not read this default" and "this flag has
    no default" are different facts and only one of them is safe.

    THE RECEIVER IS PART OF THE READ (#85). This walked EVERY add_argument
    call in the file regardless of what object it was called on, while the
    sentence above has always said `parser.add_argument`. The renderer also
    builds throwaway parsers inside its own selftests, and one of them
    FAULT-INJECTS a duplicate on purpose:

        if _lmg_inject("dup-flag"):
            probe.add_argument("--lmg-family")

    -- a two-sided calibration of the renderer's duplicate DETECTOR, and
    correct code. But the guard is a runtime condition and this walk is
    static, so the injected line was read unconditionally, it carries no
    `default=`, and last-wins resolved `--lmg-family` to None instead of
    "on". Every consumer that pinned or compared that flag's default has
    used the wrong value since #55 landed.

    So the walk now takes calls on the module-level `parser` only. Measured
    on gpu_render.py: parser 659, probe 4, fresh 1, p 1 -- the filter drops
    exactly the six selftest-scratch calls and no real flag.

    AND A COLLISION IS NOW A REFUSAL. Filtering the receiver fixes today's
    instance; it does not stop a second `parser.add_argument("--x")` from
    silently overwriting the first, which is the same last-wins failure one
    level up. Two declarations of one flag on the real parser is a defect in
    the renderer, and a defaults table that quietly picks one of them is how
    it stays invisible -- so it raises instead.
    """
    if not renderer_path or not os.path.isfile(renderer_path):
        return {}
    with open(renderer_path, errors='replace') as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {'__parse_error__': repr(e)}
    out, seen = {}, {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        if getattr(node.func.value, 'id', None) != PARSER_NAME:
            continue
        names = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        default, action = None, None
        for kw in node.keywords:
            if kw.arg == 'default':
                try:
                    default = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    default = '<computed>'
            elif kw.arg == 'action' and isinstance(kw.value, ast.Constant):
                action = kw.value.value
        if default is None and action in ('store_true', 'count'):
            default = False if action == 'store_true' else 0
        for n in names:
            if not n.startswith('--'):
                continue
            if n in seen and seen[n] != (default, node.lineno):
                raise SystemExit(
                    f"{os.path.basename(renderer_path)} declares {n} TWICE "
                    f"on `{PARSER_NAME}` -- line {seen[n][1]} (default "
                    f"{seen[n][0]!r}) and line {node.lineno} (default "
                    f"{default!r}). argparse would raise on the second; this "
                    f"table would silently keep one of them, which is how "
                    f"--lmg-family resolved to None for the whole life of "
                    f"#55 (#85). Fix the duplicate declaration in the "
                    f"renderer; a defaults table cannot choose for you.")
            seen[n] = (default, node.lineno)
            out[n] = default
    return out


def delegated_flag_sources(renderer_path):
    """Modules the renderer hands its parser to -> flags they declare.

    THE HOLE THIS MEASURES, found while fixing #85 and NOT fixed here.
    `renderer_defaults()` parses ONE file. The renderer also delegates:

        smoke_mboit.add_arguments(parser)
        _nw.add_arguments(parser)          # fam_nonworld
        _mz.add_arguments(parser)          # muzzle_flash

    Those flags are real, they are on the real parser, `args.<name>` reads
    them all over the renderer -- and none of them has ever been in the
    defaults table. So #55's "default-deny over the RESOLVED VALUE OF EVERY
    FLAG THE RENDERER DECLARES" has always meant every flag it declares
    IN ITS OWN FILE, which is a smaller set than the sentence promises.

    HOW IT SURFACED, and why it stayed invisible: `--decal-project` is
    declared by fam_nonworld with default OFF, and the only literal
    add_argument for it inside gpu_render.py is a FAULT-INJECTION line in
    the nonworld selftest (`p.add_argument("--decal-project",
    default="off")`, line 5027). The old receiver-blind walk read that
    injection line and put "off" in the table -- the right value, by
    coincidence, from a line whose purpose is to be wrong. Exactly the
    --lmg-family defect with the sign flipped: there the injected line had
    no default and corrupted a real one; here it had a default and
    masked an absence.

    THIS FUNCTION DOES NOT WIDEN THE SCOPE. Adding ~123 flags to a
    default-deny comparison can start refusing run pairs that compare fine
    today, and changing what the gate compares is a ruling, not a cleanup
    (the blast-radius argument in this module's own header). So the hole is
    MEASURED and REPORTED here, and the widening is left as a decision with
    a number attached to it.
    """
    if not renderer_path or not os.path.isfile(renderer_path):
        return {}
    try:
        tree = ast.parse(open(renderer_path, errors='replace').read())
    except SyntaxError:
        return {}
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                alias[n.asname or n.name] = n.name
    mods = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_arguments'
                and any(getattr(a, 'id', None) == PARSER_NAME
                        for a in node.args)):
            who = getattr(node.func.value, 'id', None)
            if who:
                mods.append(alias.get(who, who))
    out = {}
    base = os.path.dirname(os.path.abspath(renderer_path))
    for m in sorted(set(mods)):
        p = os.path.join(base, m.replace('.', os.sep) + '.py')
        flags = set()
        if os.path.isfile(p):
            try:
                t = ast.parse(open(p, errors='replace').read())
            except SyntaxError:
                t = None
            if t is not None:
                for node in ast.walk(t):
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'add_argument'):
                        for a in node.args:
                            if (isinstance(a, ast.Constant)
                                    and isinstance(a.value, str)
                                    and a.value.startswith('--')):
                                flags.add(a.value)
        out[m] = sorted(flags)
    return out


# --------------------------------------------------------------------------
# #55: COMPARABILITY IS A PROPERTY OF A PAIR OF RUNS, NOT OF A FLAG
# --------------------------------------------------------------------------
# The asymmetry that produced this ticket:
#   --ibl-cube was DELETED       -> six stale scripts fail at argparse. Loud.
#   --pre-exposure was SUPERSEDED -> ten scripts run to completion and emit a
#                                    full scoreboard not comparable to the
#                                    landed numbers, with nothing to say so.
# A deleted flag fails loudly; a superseded scalar defaults silently.
#
# PINNED_FLAGS is the heuristic form of the fix and it is NOT sufficient: it
# is a hand-maintained list of what matters, so the next superseded scalar is
# omitted from it and passes silently. Pre-exposure is the instance that
# surfaced this; the next one will not be a recognisable float.
#
# So comparison is DEFAULT-DENY over the RESOLVED VALUE OF EVERY FLAG THE
# RENDERER DECLARES. The scope is derived from what the run stamps about
# itself -- its argv, and the renderer's own argparse read out of the source
# -- not from a list someone remembers to update.
#
# The only hand-maintained set left is NON_SEMANTIC below: things that do NOT
# affect the image. That direction is fail-safe, and the asymmetry is the
# whole point:
#   * forgetting to add a flag to NON_SEMANTIC  -> a spurious REFUSAL. Noisy,
#     visible, fixed in one line.
#   * forgetting to add a flag to a "matters" list -> a SILENT PASS on runs
#     that are not comparable. Invisible, and the defect this ticket exists
#     to end.
# A guard whose omissions refuse is a guard; a guard whose omissions pass is
# a decoration.
#
# PINNED_FLAGS survives for the CROSS-SIDE check (our quality vs the GT
# preset's), which is a different question -- "does this run match the ground
# truth it claims to match" rather than "are these two runs comparable".
# ⚠ --batch WAS IN THIS SET AND THAT WAS WRONG, MEASURED.
# I placed it here on the reasoning that chunking "affects wall-clock and
# VRAM, not pixels". W7 measured otherwise: at --batch 512 an agent's own
# frames differ from solo-rendering by max|delta| 153 and 51.8% of pixels,
# purely from OTHER agents' poses sharing the batch. Something batch-global
# leaks between frames (#71 chases the term); it is chunk-invariant at <=96,
# so the effect has a threshold rather than being absent.
#
# THE LESSON IS ABOUT THIS SET, NOT ABOUT BATCHING. NON_SEMANTIC is
# fail-safe against OMISSIONS -- forgetting to add a flag causes a spurious
# refusal -- but it is NOT fail-safe against WRONG INCLUSIONS: every entry
# here is an assertion that a flag cannot change the image, and a wrong one
# produces exactly the silent pass the whole design exists to prevent. I
# asserted this one from plausibility instead of measurement, which is the
# habit this project retracts results over.
#
# So each entry now carries WHY it cannot change a pixel, and the standard
# for adding one is that the reason be structural (it names an output path;
# it selects nothing) rather than a belief about the renderer's internals.
NON_SEMANTIC = frozenset({
    # Structural: these name WHERE OUTPUT GOES. They cannot change a pixel
    # because nothing reads them during shading.
    '--out', '--png-dir', '--gbuffer', '--report', '--frustum-ss',
    '--lights-dump',
    # Structural: names where the tensor sink is written.
    '--session-out',
    # Structural: selects whether timing is printed.
    '--benchmark',
    # Structural: selects WHICH FRAMES GET A PNG. It changes the output
    # SET, never a pixel.
    #
    # ⚠ BUT IT CAN SILENTLY BREAK POSITIONAL PAIRING. GT is pose%05d and
    # ours is f%05d, paired by INDEX; write every second frame and index k
    # no longer means pose k, so a scorer compares the wrong pairs and
    # reports a number rather than an error. gate.py's RENDER_COUNT catches
    # it because it demands one frame per pose -- that guard is what makes
    # this classification safe, and if RENDER_COUNT is ever relaxed this
    # entry must move.
    '--png-every',
    # The pose set is compared BY CONTENT as an input sha, so the tick window
    # that names a slice of it is not separately semantic -- a differing
    # window against an identical camera json is a differing FRAME COUNT,
    # which metric_compare refuses on its own.
    '--tick-begin', '--tick-end',
    # NOTE: --batch is deliberately ABSENT. See above.
})

# FLAGS WHOSE PURPOSE IS TO RENDER WRONG PIXELS.
# `--unclamped-session-batches` restores the pre-fix batch schedule so #71's
# cross-agent leakage can be reproduced. It is a reproducer, not a config,
# and a run carrying it must never yield an acceptance number -- the whole
# point of the flag is that its output is not comparable.
#
# resolve_all() already makes two runs differing in it refuse each other.
# This set is the stronger statement: a run using it is refused as a GATE
# regardless of what it is compared against, because the failure being
# reproduced is deliberate and a number measured through it describes the
# reproducer rather than the renderer.
DIAGNOSTIC_NONCOMPARABLE = frozenset({
    '--unclamped-session-batches',
})


def diagnostic_flags_in_use(cfg):
    """Diagnostic flags this run turned on that void an acceptance number."""
    res = cfg.get('resolved') or {}
    out = []
    for flag in sorted(DIAGNOSTIC_NONCOMPARABLE):
        rec = res.get(flag)
        if not rec:
            continue
        v = rec.get('value')
        if v not in (None, False, 0, '0', '<ABSENT>'):
            out.append(f"{flag} = {v!r} ({rec.get('from')})")
    return out


def resolve_all(pairs, renderer_path):
    """Resolved value + provenance for EVERY flag, argv or default.

    This is the data-derived scope. `renderer_defaults()` reads the
    renderer's own argparse statically, so the flag set comes from the
    binary-of-record rather than from anyone's memory of it.
    """
    given = {}
    for k, v in pairs:
        given[k] = v if v is not None else True
    defaults = renderer_defaults(renderer_path)
    out = {}
    for flag in sorted(set(defaults) | set(given)):
        if flag in given:
            out[flag] = {'value': given[flag], 'from': 'argv'}
        elif flag in defaults:
            out[flag] = {'value': defaults[flag], 'from': 'default'}
    return out


def resolve_pinned(pairs, renderer_path):
    """The effective value of every pinned flag, and where it came from."""
    given = {k: v for k, v in pairs}
    defaults = renderer_defaults(renderer_path)
    out = {}
    for flag in PINNED_FLAGS:
        if flag in given:
            out[flag] = {'value': given[flag], 'from': 'argv'}
        elif flag in defaults:
            out[flag] = {'value': defaults[flag], 'from': 'default'}
        else:
            # The flag does not exist in this renderer at all. That is a
            # fact about the identity, not a nuisance: a config naming a
            # flag the renderer no longer has is not reproducible on it.
            out[flag] = {'value': None, 'from': 'ABSENT_FROM_RENDERER'}
    return out


# --------------------------------------------------------------------------
# GT preset schema, read from the committed manifest
# --------------------------------------------------------------------------
_KV = re.compile(r'"([^"]+)"\s+"([^"]*)"')


def parse_video_cfg(txt):
    d = {}
    for k, v in _KV.findall(txt or ''):
        d[k] = v
    return d


def gt_presets(manifest_path=MANIFEST):
    """{tag: {...}} for every GT capture in the committed manifest."""
    if not os.path.isfile(manifest_path):
        return {}
    with open(manifest_path) as f:
        blob = json.load(f)
    out = {}
    for c in blob.get('captures', []):
        cfg = parse_video_cfg(c.get('cs2_video_txt', ''))
        out[c['tag']] = {
            'tag': c['tag'],
            'path_on_ws1': c.get('path_on_ws1'),
            'n_frames': c.get('n_frames'),
            'ungated_poses': c.get('ungated_poses'),
            'preset': {k: cfg.get(k) for k in PRESET_KEYS},
            'cs2_video_txt_sha256': c.get('cs2_video_txt_sha256'),
            'frameset_sha256': c.get('frameset_sha256'),
            'frame_sha256': c.get('frame_sha256'),
        }
    return out


def gt_block(tag, manifest_path=MANIFEST):
    p = gt_presets(manifest_path)
    if tag not in p:
        raise SystemExit(
            f"REFUSED: no GT capture tagged {tag!r} in {manifest_path}. "
            f"Known: {', '.join(sorted(p)) or '(none)'}")
    g = dict(p[tag])
    g.pop('frame_sha256', None)          # kept in the manifest, not inlined
    g['manifest'] = os.path.relpath(manifest_path, REPO_ROOT)
    g['manifest_sha256'] = sha(manifest_path)
    return g


# --------------------------------------------------------------------------
# The cross-side check: does our config match the GT it claims to match?
# --------------------------------------------------------------------------
def cross_side_failures(cfg):
    """Refusals arising from OUR flags vs the GT preset this config names.

    Empty list == the two sides are configured for the same computation.
    """
    g = cfg.get('gt')
    if not g:
        return []
    bad = []
    preset = g.get('preset') or {}
    sq = preset.get('setting.shaderquality')
    ours = (cfg.get('pinned', {}).get('--shader-quality') or {}).get('value')
    if sq is not None and ours is not None:
        try:
            want = int(sq)
        except (TypeError, ValueError):
            want = None
        if want is not None and int(ours) != want:
            bad.append(
                f"--shader-quality {ours} vs GT {g['tag']} "
                f"setting.shaderquality {want}. S_SHADER_QUALITY is not a "
                f"quality scale, it selects a DIFFERENT compiled module of "
                f"every family (RENDER_SETTINGS_AXES.md:481-492), so this "
                f"pairing scores one renderer against another's ground "
                f"truth. Re-run with --shader-quality {want}.")
    msaa = preset.get('setting.msaa_samples')
    ours_msaa = (cfg.get('pinned', {}).get('--msaa-samples') or {}).get('value')
    if msaa is not None and ours_msaa is not None:
        if str(ours_msaa) != str(msaa):
            bad.append(f"--msaa-samples {ours_msaa} vs GT {g['tag']} "
                       f"setting.msaa_samples {msaa}")
    return bad


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def _build_cfg(pairs, renderer, log=None, gt_tag=None):
    cwd = None
    inputs = {}
    for k, v in pairs:
        if k in PATH_FLAGS and v:
            p = v if os.path.isabs(v) else os.path.normpath(
                os.path.join(cwd or os.getcwd(), v))
            inputs[k] = {'path': p, 'sha256': sha(p)}
    cfg = {
        'schema': SCHEMA,
        'renderer': {'path': renderer, 'sha256': sha(renderer)},
        'flags': [[k, v] for k, v in pairs],
        'pinned': resolve_pinned(pairs, renderer),
        'inputs': inputs,
    }
    if log:
        cfg['captured_from'] = os.path.abspath(log)
    if gt_tag:
        cfg['gt'] = gt_block(gt_tag)
    return cfg


def auto_inputs(log_path, resolve=lambda v: v):
    """{'auto:<label>': {...sha...}} for every auto-resolved input the log names."""
    out = {}
    if not log_path or not os.path.isfile(log_path):
        return out
    for ln in open(log_path, errors='replace'):
        m = _AUTO.match(ln)
        if not m:
            continue
        label, val = m.group(1), m.group(2).rstrip(',;')
        p = resolve(val)
        if os.path.isdir(p):
            rolled, per = sha_dir(p)
            out['auto:' + label] = {'path': p, 'sha256': rolled,
                                    'kind': 'dir', 'files': per,
                                    'resolved_by': 'renderer'}
        elif os.path.isfile(p):
            out['auto:' + label] = {'path': p, 'sha256': sha(p),
                                    'kind': 'file', 'resolved_by': 'renderer'}
        # A value that is not a path ("auto -> off", "auto -> uv") is a
        # resolved SETTING, not an input. Recorded as such rather than
        # dropped, so a reader can see the renderer made a choice.
        else:
            out.setdefault('__auto_settings__', {})[label] = val
    return out


def cmd_capture(a):
    line = None
    for ln in open(a.log, errors='replace'):
        if ln.startswith('ARGV:'):
            line = ln[len('ARGV:'):].strip(); break
    if line is None:
        sys.exit('no ARGV: stamp in %s' % a.log)
    pairs = parse_argv(line)
    cwd = a.cwd or os.path.dirname(os.path.abspath(a.log))

    def resolve(v):
        return v if os.path.isabs(v) else os.path.normpath(os.path.join(cwd, v))

    inputs = pin_inputs(pairs, resolve)
    auto = auto_inputs(a.log, resolve)
    auto_settings = auto.pop('__auto_settings__', None)
    inputs.update(auto)
    cfg = {
        'schema': SCHEMA,
        'renderer': {'path': a.renderer, 'sha256': sha(a.renderer)},
        'flags': [[k, v] for k, v in pairs],
        'pinned': resolve_pinned(pairs, a.renderer),
        'resolved': resolve_all(pairs, a.renderer),
        'inputs': inputs,
        'captured_from': os.path.abspath(a.log),
    }
    if auto_settings:
        cfg['auto_settings'] = auto_settings
    if a.gt:
        cfg['gt'] = gt_block(a.gt)
    if a.supersedes:
        # Which identity this one REPLACES, and for what. Old arms stay
        # interpretable against their own recorded config; the diff between
        # the two configs is the documentation of what changed. Recorded in
        # the artefact rather than in a message, because the message is not
        # what a reader six weeks from now has.
        cfg['supersedes'] = {'role': a.supersedes_role,
                             'config': a.supersedes,
                             'reason': a.supersedes_reason}
    missing = [k for k, d in inputs.items() if d['sha256'] is None]
    if missing:
        cfg['UNRESOLVED'] = missing
    out = a.out or (os.path.splitext(a.log)[0] + '.render.json')
    json.dump(cfg, open(out, 'w'), indent=1)
    print('wrote', out)
    for f, d in cfg['pinned'].items():
        print(f"  pinned {f} = {d['value']!r}  ({d['from']})")
    if missing:
        print('  UNRESOLVED (path absent now):', ' '.join(missing))
    if cfg['renderer']['sha256'] is None:
        print('  UNRESOLVED renderer:', a.renderer)
    if a.gt:
        bad = cross_side_failures(cfg)
        for b in bad:
            print('  CROSS-SIDE MISMATCH: ' + b)


def failures(cfg):
    """Every reason this config does not describe the tree as it stands."""
    bad = []
    r = cfg['renderer']
    if sha(r['path']) != r['sha256']:
        bad.append('renderer %s: have %s want %s' % (
            r['path'], (sha(r['path']) or 'ABSENT')[:12],
            (r['sha256'] or '?')[:12]))
    for k, d in cfg.get('inputs', {}).items():
        if d.get('kind') in ('dir', 'manifest'):
            rolled, per = (sha_dir(d['path']) if d['kind'] == 'dir'
                           else sha_manifest(d['path']))
            if rolled != d['sha256']:
                # Name the FILE, not the folder. "the ibl-cube directory
                # changed" sends the reader to re-derive 365 MB; "mip3
                # changed" sends them to the one asset that moved.
                want = d.get('files', {})
                moved = sorted(
                    set(want) ^ set(per)
                    | {n for n in set(want) & set(per) if want[n] != per[n]})
                bad.append('%s %s: %d of %d files differ (%s)%s' % (
                    k, d['path'], len(moved), max(len(want), len(per)),
                    ', '.join(moved[:4]) or 'target absent',
                    '' if d['kind'] == 'dir' else
                    ' -- a manifest is pinned by its own content AND every '
                    'file it names; a pointer is not its target'))
            continue
        if sha(d['path']) != d['sha256']:
            bad.append('%s %s: have %s want %s' % (
                k, d['path'], (sha(d['path']) or 'ABSENT')[:12],
                (d['sha256'] or '?')[:12]))

    # #55: a pinned flag whose DEFAULT has moved under an unchanged argv.
    pinned = cfg.get('pinned')
    if pinned:
        now = renderer_defaults(r['path'])
        given = {k: v for k, v in cfg.get('flags', [])}
        for flag, rec in pinned.items():
            if rec.get('from') != 'default':
                continue
            if flag in given:
                continue
            cur = now.get(flag, '__MISSING__')
            if cur == '__MISSING__':
                bad.append(
                    f"pinned {flag}: the renderer no longer declares it, so "
                    f"the recorded default {rec['value']!r} describes no "
                    f"reachable configuration")
            elif cur != rec['value']:
                bad.append(
                    f"pinned {flag}: recorded default {rec['value']!r}, "
                    f"renderer default is now {cur!r}. The argv is unchanged "
                    f"and the arithmetic is not -- this is the silent case a "
                    f"deleted flag would have caught loudly (#55).")

    if cfg.get('gt'):
        g = cfg['gt']
        mp = os.path.join(REPO_ROOT, g.get('manifest', ''))
        if g.get('manifest_sha256') and sha(mp) != g['manifest_sha256']:
            bad.append(
                f"GT manifest {g.get('manifest')}: have "
                f"{(sha(mp) or 'ABSENT')[:12]} want "
                f"{g['manifest_sha256'][:12]} -- the acceptance corpus's own "
                f"identity moved")
        bad += cross_side_failures(cfg)
    return bad


def _refuse(bad, what):
    print(f'REFUSED -- {what}, this run is NOT comparable:', file=sys.stderr)
    for b in bad:
        print('  ' + b, file=sys.stderr)


def cmd_verify(a):
    cfg = json.load(open(a.cfg))
    bad = failures(cfg)
    if bad:
        _refuse(bad, 'identity mismatch')
        sys.exit(1)
    print('identity OK:', a.cfg)
    for f, d in (cfg.get('pinned') or {}).items():
        print(f"  pinned {f} = {d['value']!r}  ({d['from']})")
    for k, d in (cfg.get('inputs') or {}).items():
        if d.get('kind') == 'dir':
            print(f"  input {k} = {len(d.get('files', {}))} files, "
                  f"rolled {(d['sha256'] or '?')[:12]}")
    s = cfg.get('supersedes')
    if s:
        print(f"  supersedes for {s['role']}: {s['config']}"
              + (f" -- {s['reason']}" if s.get('reason') else ""))
    if cfg.get('gt'):
        g = cfg['gt']
        print(f"  gt {g['tag']}: {g.get('n_frames')} frames, "
              f"setting.shaderquality "
              f"{(g.get('preset') or {}).get('setting.shaderquality')}")


def cmd_compare(a):
    """Refuse unless two runs' numbers may be placed side by side.

    A scoreboard row against a landed number is a comparison, and a
    comparison across configs is not one. Printing a warning beside a null
    reads as commentary on a result; this exits non-zero instead.
    """
    A, B = json.load(open(a.a)), json.load(open(a.b))
    bad = []

    # DATA-DERIVED, DEFAULT-DENY. Every flag either side resolved, compared
    # by VALUE regardless of whether it came from argv or from a default --
    # that equivalence is the point. Two runs where one passes
    # `--pre-exposure 2.2` and the other inherits a 2.2 default ARE the same
    # computation; two runs that both omit it under renderers whose defaults
    # differ are NOT, and only the resolved value can tell them apart.
    ra, rb = A.get('resolved'), B.get('resolved')
    if ra and rb:
        for flag in sorted(set(ra) | set(rb)):
            if flag in NON_SEMANTIC:
                continue
            va = (ra.get(flag) or {}).get('value', '<ABSENT>')
            vb = (rb.get(flag) or {}).get('value', '<ABSENT>')
            if str(va) != str(vb):
                # Say which side got it from where. "both defaulted, and the
                # defaults differ" is a different bug report from "one was
                # passed explicitly".
                sa = (ra.get(flag) or {}).get('from', 'missing')
                sb = (rb.get(flag) or {}).get('from', 'missing')
                bad.append(f"{flag}: {va!r} ({sa}) vs {vb!r} ({sb})")
    else:
        # A v2-or-earlier config has no `resolved` block, so the full
        # comparison is UNAVAILABLE -- and saying so is mandatory. Falling
        # back to the pinned subset silently would report "comparable" from
        # a check that examined 9 flags out of the renderer's whole declared
        # set, which is the shape of the defect rather than a mitigation.
        #
        # The DENOMINATOR IS DERIVED, not written down. It used to read a
        # literal 636 and was already wrong before anyone noticed: flags are
        # added and removed by every lane, and the count was 650 when this
        # was corrected. A stale number in a message about exactness is
        # worse than no number, so it comes from whichever config actually
        # HAS a resolved block, and is omitted when neither does.
        _nres = len((A.get('resolved') or B.get('resolved') or {}))
        _of = f" of {_nres}" if _nres else ""
        bad.append(
            "one or both configs predate the resolved-flag block (schema "
            f"{A.get('schema')} vs {B.get('schema')}), so a full comparison "
            f"is NOT POSSIBLE. A pinned-subset check would compare "
            f"{len(PINNED_FLAGS)} flags{_of} and print 'comparable'.")
        bad.append(
            "If the run's LOG still exists, re-capture the config and the "
            "comparison becomes available. If it does not, THIS IS A "
            "PERMANENT BOUNDARY, not a task: the six prior-warp arms "
            "(cfg_J*, cfg_ovl*) were run before the resolved block and "
            "before pack-generation existed, so they are unjoinable to "
            "post-boundary numbers by construction. Record the boundary; do "
            "not reason across it.")
        for flag in PINNED_FLAGS:
            va = (A.get('pinned', {}).get(flag) or {}).get('value')
            vb = (B.get('pinned', {}).get(flag) or {}).get('value')
            if va != vb:
                bad.append(f"{flag}: {va!r} vs {vb!r}")
    ga = (A.get('gt') or {}).get('tag')
    gb = (B.get('gt') or {}).get('tag')
    if ga != gb:
        bad.append(f"ground truth: {ga!r} vs {gb!r} -- different corpora")
    # INPUTS ARE PART OF THE CONFIG, not scenery. Two runs whose flag sets
    # match but whose ASSETS differ -- one with the prefiltered radiance
    # chain connected, one without -- are not the same computation, and
    # placing their numbers side by side as though they were is the same
    # error as ignoring a pinned flag.
    ia, ib = A.get('inputs', {}), B.get('inputs', {})
    for k in sorted(set(ia) | set(ib)):
        if k not in ia or k not in ib:
            bad.append(f"input {k}: present on only one side "
                       f"({'A' if k in ia else 'B'})")
        elif ia[k].get('sha256') != ib[k].get('sha256'):
            bad.append(f"input {k}: {(ia[k].get('sha256') or '?')[:12]} vs "
                       f"{(ib[k].get('sha256') or '?')[:12]}")
    if not a.allow_renderer_drift:
        ra = A['renderer'].get('sha256')
        rb = B['renderer'].get('sha256')
        if ra != rb:
            bad.append(
                f"renderer sha {(ra or '?')[:12]} vs {(rb or '?')[:12]}. Two "
                f"copies of gpu_render.py on the same argv measured kl_rgb "
                f"1.3223 and 4.0189. Pass --allow-renderer-drift only when "
                f"the renderer change IS the thing being compared.")
    if bad:
        _refuse(bad, 'these two runs are not comparable')
        sys.exit(1)
    print(f'comparable: {a.a} vs {a.b}')


def cmd_argv(a):
    cfg = json.load(open(a.cfg))
    print(' '.join(x for k, v in cfg['flags'] for x in ((k, v) if v else (k,))))


def cmd_presets(a):
    p = gt_presets()
    if not p:
        sys.exit(f'no manifest at {MANIFEST}')
    keys = list(PRESET_KEYS)
    print(f"{'tag':28} {'frames':>6}  " + "  ".join(
        k.replace('setting.', '')[:9].rjust(9) for k in keys))
    for tag in sorted(p):
        g = p[tag]
        row = "  ".join(str(g['preset'].get(k)).rjust(9) for k in keys)
        print(f"{tag:28} {str(g['n_frames']):>6}  {row}")
    print("\nsetting.shaderquality -> S_SHADER_QUALITY is the identity map; "
          "global preset N maps onto it as "
          + ", ".join(f"{k}->{v}" for k, v in PRESET_TO_SHADER_QUALITY.items())
          + " (RENDER_SETTINGS_AXES.md:486-492, READ from "
            "cfg/video_defaults_N.txt).")


# --------------------------------------------------------------------------
# importable API -- the mandatory precondition
# --------------------------------------------------------------------------
def require(cfg_path, what='metric run'):
    """Verify identity or EXIT. Callers do not get an ignorable return value.

    This is the shape on purpose. A function returning False invites
    `if not require(...): print("warning")`, and a printed warning beside a
    number is read as commentary on a result rather than as its absence.
    """
    if not cfg_path or not os.path.isfile(cfg_path):
        print(f'REFUSED -- {what} requires a render-identity config; '
              f'{cfg_path!r} is not a file. Build one with '
              f'`render_config.py capture --log RUN.log --renderer PATH '
              f'[--gt TAG]`.', file=sys.stderr)
        sys.exit(1)
    cfg = json.load(open(cfg_path))
    bad = failures(cfg)
    if bad:
        _refuse(bad, f'{what} blocked by identity mismatch')
        sys.exit(1)
    return cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    c = s.add_parser('capture')
    c.add_argument('--log', required=True)
    c.add_argument('--renderer', required=True)
    c.add_argument('--cwd')
    c.add_argument('--out')
    c.add_argument('--gt', help='GT capture tag from GT_QUALITY_LEVEL_MANIFEST.json')
    c.add_argument('--supersedes',
                   help='the config this one replaces (for --supersedes-role)')
    c.add_argument('--supersedes-role', default='gate')
    c.add_argument('--supersedes-reason', default='')
    c.set_defaults(fn=cmd_capture)
    v = s.add_parser('verify')
    v.add_argument('cfg')
    v.set_defaults(fn=cmd_verify)
    g = s.add_parser('argv')
    g.add_argument('cfg')
    g.set_defaults(fn=cmd_argv)
    m = s.add_parser('compare')
    m.add_argument('a')
    m.add_argument('b')
    m.add_argument('--allow-renderer-drift', action='store_true')
    m.set_defaults(fn=cmd_compare)
    q = s.add_parser('presets')
    q.set_defaults(fn=cmd_presets)
    a = p.parse_args()
    a.fn(a)
