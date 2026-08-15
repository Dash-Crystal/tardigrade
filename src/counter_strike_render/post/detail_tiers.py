#!/usr/bin/env python3
"""The four remaining `videocfg_*` quality tiers, and their DRIVES.

    videocfg_shadow_quality    0..3   cascaded shadow maps (READ: the
                                      system; UNREAD: cascades/resolution)
    videocfg_texture_detail    0..2   GPU-MEMORY level (READ: the system;
                                      UNREAD: what the budget does)
    videocfg_particle_detail   0..3   particle + volumetric budget
    videocfg_ao_detail         0,2,3  AO proxy tier -- and tier 1 DOES NOT
                                      EXIST, read off the engine itself

WHY THIS FILE EXISTS BEFORE ITS TABLES ARE FILLED
--------------------------------------------------
All three rows spent this project's life as GAP_NO_AXIS -- "this renderer
has no such machinery" -- and all three were WRONG. The machinery was
already in the tree:

    texture_detail   --mip-bias adds to the computed LOD; --mip-max-levels
                     caps the chain depth
    particle_detail  smoke_mboit declares D_SMOKE_QUALITY itself
                     (--smoke-quality selecting --smoke-march-steps vs
                     --smoke-march-steps-lowq), plus D_SMOKE_FULLRES and
                     the light-step / shadow-tap / DDA budgets
    ao_detail        --ssao drives a transcription of all three shipped
                     SSAO modules in passes/loopfam/ao.py, with
                     --ssao-samples / --ssao-turns / --ssao-mips /
                     --ssao-blur / --ssao-radius beside it

Each row failed the same way: the cvar's NAME was searched for among the
flags, nothing that looked like it was found, and "no axis" was written.
`videocfg_ao_detail` is the clearest case -- its basis named three
BAKED-AO flags and concluded the screen-space family did not exist, when
it has its own registered conformance family.

So what is missing is the TIER -> VALUE table, not the axis. This file is
the drive: it expands one tier into the several flags it keys, in ONE
place, so that when a table is read the conversion is a table edit rather
than an implementation. Every row refuses until then, and the refusal
names its own route.

WHY THE DRIVE IS WORTH WIRING BEFORE THE TABLE IS READ
-------------------------------------------------------
Because a tier is not one flag. `particle_detail` plausibly keys the
march step count AND the resolution AND the light steps together; if
three consumers each read the tier themselves they can drift, and if the
table lands into an unwired axis the measurement that produced it has
nowhere to go. Expanding in one place also makes the CONFLICT rule
uniform: an explicitly typed flag beats the tier and SAYS SO, which is
the same rule --preset already applies one level up.

⚠️ AND ONE OF THESE THREE MAY NOT BE MEASURABLE ON THE STANDING FIXTURE
------------------------------------------------------------------------
`videocfg_particle_detail` bounds the smoke march. A 48-pose teleport set
on de_inferno may carry NO particles at all, and on a fixture that cannot
exhibit the effect a null result is VACUOUS -- a third state, distinct
from tested-and-clear. That is not a hypothetical here: the four
particle_detail arms are exactly such a set.

So :func:`particle_exhibition_note` exists and the drive CALLS it, so the
renderer states the fixture's particle population at the point where the
tier is applied. A measurement that comes back null next to a printed
"this render carried 0 smoke instances" cannot be mistaken for evidence
that the tier does nothing.
"""
from __future__ import annotations

from typing import Dict, Optional


class DetailTierUnread(Exception):
    """Raised for a tier whose value table has not been read."""


# --------------------------------------------------------------------------
# videocfg_texture_detail
# --------------------------------------------------------------------------
# Observed domain: 0, 1, 2 across the 53 GT configs; presets carry
# 0, 1, 2, 2. Arms: videocfg_texture_detail-{1,2} against preset0.
TEXTURE_DETAIL: Dict[int, Optional[dict]] = {0: None, 1: None, 2: None}

# THE ELEMENT IDS ARE READ, AND THEY CORRECT THIS AXIS'S MECHANISM.
# settings_video.vxml_c, string table :248-253 --
#     '#SFUI_Settings_Model_Texture_Detail'  'ModelTextureDetail'
#     'gpumemlevel0'  'gpumemlevel1'  'gpumemlevel2'
# The token is `gpumemlevel`, not a mip or LOD token: this dropdown drives
# the engine's GPU-MEMORY LEVEL, the same axis as the `setting.gpu_mem_level`
# header key every cs2_video.txt carries. So videocfg_texture_detail is a
# MEMORY-BUDGET ladder, not a sampler one -- which makes --mip-max-levels
# (a residency cap) the closer of our two knobs and --mip-bias the further,
# the opposite of the ordering this row's basis used to give.
#
# What the budget DOES to mips and model LODs is still unread:
# `gpumemlevel{N}` carries no resolution, mip-count or LOD token, exactly
# as `matforceaniso{N}` carried no bias token. The id names the SYSTEM and
# stops there.
TEXTURE_DETAIL_IDS = {0: "gpumemlevel0", 1: "gpumemlevel1", 2: "gpumemlevel2"}

# ⚠️ AND THE MIP HYPOTHESIS IS REFUTED -- INCLUDING THE VERSION OF IT THIS
# FILE ASSERTED. The note above concluded "--mip-max-levels is the closer
# of our two knobs and --mip-bias the further". Both are wrong: the
# measurement says this axis is not a texture-filtering or residency
# effect at all, and it was refuted on all three of the predictions a
# mip/residency axis makes.
#
#   (i)   IS IT A LOW-PASS OF THE BASE ARM?  No. Fitting
#         blur(preset0, sigma) against the tier-2 arm minimises the
#         residual at sigma = 0, exactly as it does for the same-config
#         controls, and box-downsample-by-2/3-then-up (the mip-shift
#         model) makes the match WORSE than identity in every case.
#   (ii)  DOES HIGH-FREQUENCY POWER DROP?  By 0.7%. One mip step HALVES
#         it. That is ~70x too small, and barely outside the repeat
#         spread (0.998-1.003).
#   (iii) DOES IT GROW WITH GRAZING ANGLE / DISTANCE?  No -- it is flat in
#         distance-to-edge (1.15/1.14/1.16/1.14/1.12/1.08) and flat in
#         row and column profile. And the signal GROWS under spatial
#         averaging (1.14x floor at K1 -> 3.82x at K64), which is the
#         opposite of a filtering change.
#
# WHAT IT IS INSTEAD, from the engine's own RT table: tiers 1 and 2 both
# allocate 62 RTs / 117,437,312 B against tier 0's 56 / 73,200,512, and
# the delta is a screen-space MULTI-RESOLUTION BUFFER FAMILY gaining a
# full-res level -- 1280x720 copies of R32F, RG1616F, RG3232F, two
# RGBA16161616F and RGBA32323232F, with the existing 320x180 and 640x360
# copies retained. Texture residency and mip bias do not allocate
# screen-resolution render targets. The pixels agree: a diffuse,
# achromatic (chroma/|luma| 0.192 vs a control's 0.239-0.324),
# low-frequency, region-darkening change confined to world geometry
# (sky-block ratio 0.52 where the noise floor's is 1.98).
#
# AND TIER 1 IS A NO-OP ON THE IMAGE. K16 1.14x / K64 1.12x the MAX
# floor, and its signed diff map correlates r = 0.672 with the
# preset0-re-run component -- i.e. most of its apparent difference is the
# shared run-to-run term every preset0 repeat also has. It allocates the
# tier-2 RT set and does not use it. Measured states: {0, 1} vs {2}.
TEXTURE_DETAIL_STATE = {0: "A", 1: "A", 2: "B"}
TEXTURE_DETAIL_REFUTED = (
    "NOT a mip bias, NOT a residency cap, NOT a blur. Refuted on all "
    "three predictions: argmin sigma = 0 in a blur fit, HF power moves "
    "0.7% where one mip step halves it, and the signal GROWS under "
    "spatial averaging instead of shrinking. The RT table says what it "
    "is instead: a screen-space multi-resolution buffer family gaining a "
    "1280x720 level. Tier 1 is an image no-op that allocates tier 2's "
    "buffers. Implement as a two-state axis {0,1} vs {2}, never as a mip "
    "knob -- and note this file ASSERTED the mip reading before the "
    "measurement arrived.")

TEXTURE_DETAIL_NOTE = (
    "videocfg_texture_detail tier -> (resident mip cap, mip bias) is NOT "
    "read, but its MECHANISM now is: the settings resource binds it to "
    "gpumemlevel0/1/2, the engine's GPU-MEMORY LEVEL, so it is a "
    "residency-budget ladder rather than a sampler one. --mip-max-levels "
    "is therefore the closer axis and --mip-bias the further. What the "
    "budget does to the mip chain is unread -- the id carries no "
    "resolution or LOD token. Route: the two arms "
    "(videocfg_texture_detail-{1,2}), where a residency cap and a bias "
    "leave different signatures, plus the sign of the move.")

# --------------------------------------------------------------------------
# videocfg_particle_detail
# --------------------------------------------------------------------------
# Observed domain: 0..3; presets carry 0, 1, 2, 3. Arms:
# videocfg_particle_detail-{1,2,3} against preset0.
PARTICLE_DETAIL: Dict[int, Optional[dict]] = {0: None, 1: None, 2: None,
                                              3: None}

# Element ids READ, and they are OPAQUE: settings_video.vxml_c :274-280
# gives 'ParticleDetail' -> 'particledetail0'..'particledetail3', which
# names nothing but itself. The localization tooltip says the setting
# "controls the complexity of particle effects and whether particles cast
# shadows", and `r_particle_shadows` is a real convar (perftest.cfg:76,94)
# -- but binding that convar to a tier from a tooltip is deriving the
# mechanism from the NAME, which is the error this whole table exists to
# stop. Recorded as a lead, not a row.
PARTICLE_DETAIL_IDS = {n: f"particledetail{n}" for n in range(4)}

# ⚠️ THE PRECONDITION FIRED. The fixture cannot exhibit this axis, and the
# measurement says so with numbers rather than by returning a quiet null:
#     tier 1  K16 0.2482 (0.98x floor)   1 of 47 poses above floor
#     tier 2  K16 0.2387 (0.94x floor)   0 of 47 poses above floor
# against a chance rate under the null of ~20%, i.e. ~9 of 47. Tiers 1
# and 2 exceed the floor LESS OFTEN THAN CHANCE, and their RT tables are
# byte-identical to tier 0.
#
# Tier 3's apparent signal was a FRAME-EDGE ARTIFACT: the whole of it sat
# in the outer 32 pixels (x >= 1888), where its mean|d| is 4.301 against
# 2.62 for every repeat control, while the rest of the frame reads 2.623.
# Excluding that strip it collapses into the repeat cloud (K16 0.2514,
# K64 0.0873, against controls at 0.2495-0.2870 and 0.0823-0.1273).
# Column luminance means are identical across arms there, so it is not a
# letterbox-width change.
#
# BUT TIER 3 DOES CHANGE THE PIPELINE, and that IS readable: it is the
# only particle tier that moves the RT table, adding 3 x 1280x720
# RGBA8888 (59 RTs / 84,259,712 B against 56 / 73,200,512). So the engine
# boundary is at 2 -> 3, not 0 -> 1 or 1 -> 2 -- which is the one thing
# this corpus licenses about the axis.
#
# The map DOES carry particle entities: every log including preset0 shows
# the same env_particle_glow and info_particle_system slots. They simply
# produce no visible pixels at these 47 teleport poses. That is the
# non-exhibiting fixture exactly, not an absent feature.
PARTICLE_DETAIL_STATE = {0: None, 1: None, 2: None, 3: None}
PARTICLE_DETAIL_FIXTURE_VERDICT = (
    "THE FIXTURE CANNOT EXHIBIT THIS AXIS. Tiers 1 and 2 clear the "
    "same-config floor on 1 and 0 of 47 poses against a ~9/47 chance "
    "rate, with RT tables byte-identical to tier 0. Tier 3's apparent "
    "signal was entirely a 32-pixel frame-edge artifact; excluded, it "
    "sits inside the repeat cloud. What IS read: tier 3 alone adds 3 x "
    "1280x720 RGBA8888, so the engine boundary is 2 -> 3. Do NOT "
    "implement tier values from this corpus -- it needs a fixture with "
    "live emitters in frame.")

PARTICLE_DETAIL_NOTE = (
    "videocfg_particle_detail tier -> (smoke quality, march steps, "
    "resolution, light steps) is NOT read. The AXIS EXISTS: smoke_mboit "
    "declares D_SMOKE_QUALITY (--smoke-quality, selecting "
    "--smoke-march-steps vs --smoke-march-steps-lowq) and D_SMOKE_FULLRES, "
    "plus --smoke-light-steps, --smoke-shadow-taps and --smoke-dda-steps. "
    "Route: the settings UI resource first, then the three arms -- BUT SEE "
    "THE PRECONDITION: on a fixture with no particles the arms cannot "
    "exhibit the axis and a null is vacuous, not a finding.")

# --------------------------------------------------------------------------
# videocfg_ao_detail
# --------------------------------------------------------------------------
# ⚠️ TIER 1 DOES NOT EXIST, and that is now a POSITIVE READ rather than an
# inference from the corpus. This table used to carry tier 1 as an
# explicit None on the grounds that "the 53 GT configs never exercise it,
# so no capture can check a row for it" -- correct as far as it went, and
# too weak. The engine's own settings resource settles it:
# settings_video.vxml_c :281-286 gives 'AOProxy' -> 'aoproxy0',
# 'aoproxy2', 'aoproxy3'. The id run LITERALLY SKIPS 1.
#
# So it is not a row no capture can check; it is a row that does not
# exist, and carrying one would have invented a tier the engine does not
# offer. The corpus gap and the resource gap agree exactly, which is also
# the strongest single check that these element-id suffixes really are
# the cvar values for this axis.
AO_DETAIL: Dict[int, Optional[dict]] = {0: None, 2: None, 3: None}
AO_DETAIL_IDS = {0: "aoproxy0", 2: "aoproxy2", 3: "aoproxy3"}
AO_DETAIL_NONEXISTENT = (1,)

AO_DETAIL_NOTE = (
    "videocfg_ao_detail tier -> (sample count, turns, mips, blur width) is "
    "NOT read. THE AXIS EXISTS AND ITS OLD GAP ROW NAMED THE WRONG FLAGS: "
    "--ssao drives a transcription of glsl_loopfam/ssao_r0m0.glsl, "
    "ssao_scalable_ambient_obscurance_r0m3.glsl and "
    "ssao_bilateral_blur_r0m0.glsl in passes/loopfam/ao.py, with its own "
    "registered conformance family; --ssao-samples / --ssao-turns / "
    "--ssao-mips / --ssao-blur / --ssao-radius are the tier-shaped knobs. "
    "The superseded basis listed --vmat-ao / --ao-page-product / "
    "--atrous-baked-ao, which are BAKED-AO axes, and concluded no axis "
    "exists. Route: the settings UI resource, then the two arms "
    "(videocfg_ao_detail-{2,3}). NOTE THE DOMAIN: tiers observed are "
    "{0, 2, 3}; tier 1 is never exercised, so any table proposing a row "
    "for it proposes a row no capture can check.")


# --------------------------------------------------------------------------
# videocfg_shadow_quality
# --------------------------------------------------------------------------
# Element ids AND labels both read, settings_video.vxml_c :230-239 --
#     '#SFUI_Settings_CSM'  ("Global Shadow Quality")  'CSMQualityLevel'
#     '#SFUI_CSM_Low'      'csmqualitylevel0'
#     '#SFUI_CSM_Med'      'csmqualitylevel1'
#     '#SFUI_CSM_High'     'csmqualitylevel2'
#     '#SFUI_CSM_VeryHigh' 'csmqualitylevel3'
#
# So the SYSTEM is read -- cascaded shadow maps -- and the MECHANISM is
# not. `csmqualitylevel{N}` carries no resolution token and no cascade
# count, and "High" is a word, not a number. Turning it into one would be
# deriving the mechanism from the label, which is the error this file is
# built around.
# ⚠️ MEASURED, AND IT IS BINARY ON THIS FIXTURE: {0} vs {1, 2, 3}.
# Poses 1-7 (the clean recap's coverage), K-block mean|d| against a
# same-config repeat floor taken as the MAX over all 10 repeat pairs:
#     preset0 -> tier 1 (clean recap)   K16 0.5804 (2.08x)  K64 0.4634 (2.96x)
#     preset0 -> tier 2                 K16 0.5109 (1.83x)  K64 0.3977 (2.54x)
#     preset0 -> tier 3                 K16 0.5506 (1.97x)  K64 0.4389 (2.81x)
#     tier 1 <-> tier 2                 K16 0.2903 (1.04x)  K64 0.1430 (0.91x)
#     tier 1 <-> tier 3                 K16 0.2503 (0.90x)  K64 0.0973 (0.62x)
#     tier 2 <-> tier 3                 K16 0.2665 (0.96x)  K64 0.1285 (0.82x)
# and the three tiers' SIGNED diff maps against preset0 correlate
# r = 0.962-0.978 with each other while correlating 0.05-0.18 with the
# repeat controls. Tier-to-tier separation is at or below the floor at
# EVERY scale K = 1, 2, 4, 8, 16 -- not merely after low-pass.
#
# So the whole effect is the 0 -> 1 step. THAT MATTERS TO THE LADDER: the
# four presets carry shadow_quality 0/1/2/3, and on this evidence
# preset1, preset2 and preset3 are IDENTICAL to each other on this axis.
#
# STATE_ONLY is what is read. The tier -> STATE map is measured; the
# state -> (cascade count, atlas resolution) map is NOT, and no amount of
# this corpus produces it -- see SHADOW_QUALITY_NOTE.
SHADOW_QUALITY_STATE = {0: "A", 1: "B", 2: "B", 3: "B"}

# The CHARACTER of the 0 -> B change, measured, because "shadows differ"
# is not a mechanism. Signed diff binned by the base frame's block
# luminance is a clean unimodal DARKENING peaked at mid-to-bright
# (130-160/255: -0.726 and -0.683 for tiers 2 and 3, against a repeat
# control's -0.009), zero on already-dark blocks and zero on sky. 45% of
# the excess energy sits in the top 1% of blocks (floor: 32%), 75-81% of
# changed blocks get DARKER, and the row profile rises monotonically
# toward the bottom of the frame -- the near ground plane -- with the
# horizon band SUPPRESSED. That last point is load-bearing: a
# cascade-DISTANCE change would put a ring of difference at a mid-screen
# depth band and leave the near ground alone. This is the opposite.
# Distance-to-edge shows a 1.20x peak in the 0-4px ring decaying to a
# 1.06x interior pedestal, so there is an edge-localised component (moved
# shadow boundaries) riding on a broad regional one.
# Read: a shadow COVERAGE change, not a filtering or cascade-range change.
SHADOW_QUALITY_CHARACTER = (
    "0 -> B is a shadow COVERAGE change: mid-luminance world surfaces "
    "darken (peak at 130-160/255), 75-81% one-signed, 45% of the energy "
    "in 1% of blocks, near-ground weighted with the horizon band "
    "suppressed, and a 4px edge component on a broad pedestal. The "
    "near-ground weighting is evidence AGAINST a cascade-distance "
    "reading, which would band at a mid-screen depth instead.")

SHADOW_QUALITY: Dict[int, Optional[dict]] = {0: None, 1: None, 2: None,
                                             3: None}
SHADOW_QUALITY_IDS = {n: f"csmqualitylevel{n}" for n in range(4)}
SHADOW_QUALITY_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

# THREE ROUTES CLOSED, and closed is worth more than untried.
#  1. The vpk index of BOTH archives: a csm/cascade/shadowmap/sunshadow
#     search over game/csgo returns 13 hits, all de_aztec waterfall props;
#     over game/core, zero. No CSM material, shader default, .res or .txt.
#  2. All 58 console logs: zero hits for r_csm*, csm_*, shadow*resolution
#     or cascade*.
#  3. The RT-allocation dumps -- and this one is the important closure,
#     because it is IMMUNE TO #93. It reads the BOOT-TIME allocation dump
#     rather than the frames, so the stuck-camera arm cannot corrupt it.
#     videocfg_shadow_quality-{1,2,3} and videocfg_dynamic_shadows-1 all
#     have RT sets IDENTICAL to preset0. The CSM atlas is not in the
#     scratch RT pool at all, so unlike FSR -- which announced itself as
#     one extra full-res RGBA8888 -- this axis leaves NO trace in the
#     logs. That route is exhausted, not merely unattempted.
#
# ONE ARTIFACT NAMES THE QUANTITY AND IS NOT THE TABLE.
# game/csgo/cfg/perftest.cfg (a generic Valve benchmark script whose
# comments still say "Dota") carries, under "Reduced Settings (similar to
# Source1)":
#     line  82:  lb_csm_cascade_size_override 1024
#     line 100:  lb_csm_cascade_size_override 2048
# That proves the engine has a per-cascade SIZE concept and hands us the
# convar name. It is NOT the videocfg_shadow_quality ladder: those are the
# benchmark's own two values for a Source1-lookalike comparison, and there
# is no cascade COUNT anywhere in the file. Recorded as a name, not a row.
SHADOW_QUALITY_NOTE = (
    "videocfg_shadow_quality tier -> (cascade count, cascade size) is NOT "
    "read. The SYSTEM is read -- the settings resource binds it to "
    "csmqualitylevel0..3, i.e. cascaded shadow maps -- but the ids carry "
    "no resolution or count token and the labels are Low/Medium/High/Very "
    "High. THREE ROUTES ARE CLOSED: the vpk index of both archives (no CSM "
    "material or .res), all 58 console logs (zero csm/cascade hits), and "
    "the RT-allocation dumps, which are IDENTICAL across all tiers -- the "
    "CSM atlas is not in the scratch RT pool, so this axis leaves no trace "
    "in the logs the way FSR did. perftest.cfg names "
    "`lb_csm_cascade_size_override` (1024/2048) which proves a per-cascade "
    "SIZE concept exists but is the benchmark's own values, not the "
    "ladder. HIGHEST-VALUE REMAINING ROUTE: a RenderDoc capture per tier "
    "on ws-1 -- RenderDoc 1.6.0 is already attached there (every console "
    "log line 3) -- which answers cascade count AND resolution in one "
    "capture and does not need the fixture to exhibit shadows. That "
    "matters because #93 makes the arms the weakest route for this axis.")

# ⚠️ THE ID SUFFIX IS A UI SLOT INDEX, NOT ALWAYS THE CVAR VALUE.
# It coincides with the cvar value only where the legal values happen to
# be 0..N-1. Checked against what the shipped presets actually carry:
#     shadow_quality   ids 0,1,2,3   preset values 0,1,2,3     agrees
#     texture_detail   ids 0,1,2     preset values 0,1,2,2     agrees
#     particle_detail  ids 0,1,2,3   preset values 0,1,2,3     agrees
#     ao_detail        ids 0,2,3     preset values 0,0,2,3     agrees, GAP AND ALL
#     hdr_detail       ids 0,1       preset values 3,3,-1,-1   ✘ DOES NOT
# `hdr0`/`hdr1` against values 3/-1 is the counterexample, and it is why
# this equivalence is recorded as a CHECKED coincidence per axis rather
# than assumed. The ao_detail gap matching the preset gap exactly is the
# strongest single piece of evidence that it holds for these four.
ID_SUFFIX_IS_NOT_ALWAYS_THE_VALUE = (
    "element-id suffixes are UI SLOT INDICES. They equal the cvar value "
    "for shadow_quality, texture_detail, particle_detail and ao_detail "
    "(each verified against the values the four shipped presets carry, "
    "including ao_detail's skipped 1), and they DO NOT for hdr_detail, "
    "whose ids are hdr0/hdr1 against values 3/-1.")

# BYCATCH, and it is a real mechanism read for a row that stays closed:
# videocfg_dynamic_shadows -> 'DynamicShadows', ids dynamicshadowslevel0/1,
# with INLINE labels '#SFUI_Shadows_SunOnly' = "Sun Only" and
# '#SFUI_Shadows_All' = "All", tooltip "Select the light sources for which
# dynamic objects such as players and smoke cast shadows."
#
# So tier 0 = the sun only, tier 1 = all lights. That does not reopen the
# row -- the engine prints "Unable to read video config convar
# videocfg_dynamic_shadows" in 58 of 58 capture logs and the single-axis
# arm measures 1.00x the noise floor, so THE REFERENCE NEVER APPLIED IT --
# but it does mean we now know what it WOULD have done, which is worth
# having when a build where the convar reads back finally appears.
DYNAMIC_SHADOWS_MECHANISM = {
    0: "sun light only", 1: "all lights",
}

_AXES = {
    "videocfg_shadow_quality": (SHADOW_QUALITY, SHADOW_QUALITY_NOTE),
    "videocfg_texture_detail": (TEXTURE_DETAIL, TEXTURE_DETAIL_NOTE),
    "videocfg_particle_detail": (PARTICLE_DETAIL, PARTICLE_DETAIL_NOTE),
    "videocfg_ao_detail": (AO_DETAIL, AO_DETAIL_NOTE),
}


def tier_values(axis: str, tier: int) -> dict:
    """{flag: value} for a tier, or a refusal naming what to read."""
    if axis not in _AXES:
        raise ValueError(f"{axis!r} is not one of {sorted(_AXES)}")
    table, note = _AXES[axis]
    if tier not in table:
        raise ValueError(
            f"{axis}={tier} is outside the domain the 53 GT configs "
            f"exercise ({sorted(table)}). A tier no capture used has no "
            f"values to read.")
    vals = table[tier]
    if vals is None:
        extra = ""
        if axis == "videocfg_ao_detail" and tier in AO_DETAIL_NONEXISTENT:
            extra = (" AND THIS TIER DOES NOT EXIST: the engine's own "
                     "settings resource enumerates aoproxy0, aoproxy2, "
                     "aoproxy3 and skips 1.")
        raise DetailTierUnread(f"{axis}={tier}: {note}{extra}")
    return vals


def describe(axis: str, tier: Optional[int]) -> str:
    """One banner line. An UNREAD tier prints as unread, never as a
    default: "this render used the reference's value" and "this render
    used ours" must not print the same thing."""
    if tier is None:
        return (f"{axis}: NOT SET -- this render is at this renderer's own "
                f"values on that axis, which is NO CS2 tier")
    table, _ = _AXES[axis]
    vals = table.get(tier)
    if vals is None:
        return (f"{axis}={tier}: UNREAD -- the tier is recorded and its "
                f"values are NOT applied; this render is at the renderer's "
                f"own values on that axis, NOT at the reference's")
    return (f"{axis}={tier}: "
            + ", ".join(f"{k} {v}" for k, v in sorted(vals.items()))
            + f"  (source: {vals.get('__source__', 'UNSTATED')})")


# --------------------------------------------------------------------------
# The non-exhibiting-fixture guard
# --------------------------------------------------------------------------
def particle_exhibition_note(smoke_mode: str, instances: int,
                             smoke_events: Optional[int] = None) -> str:
    """State whether THIS render can exhibit videocfg_particle_detail.

    `a map that CANNOT exhibit the effect` is a standing law here: a null
    measured on a fixture carrying no particles is a third state, not
    evidence the axis is inert. The census belongs BEFORE the A/B and it
    belongs in the log, so a later reader of a null result can see which
    state it was.

    Returns a line for the banner. It never raises: refusing to render
    because a fixture is weak would be the wrong remedy -- the point is
    that the weakness is VISIBLE, not that the render is blocked."""
    live = smoke_mode not in (None, "off")
    n = instances if live else 0
    if smoke_events is not None and live:
        n = min(n, smoke_events) if smoke_events == 0 else n
    if not live:
        return ("videocfg_particle_detail: THIS FIXTURE CANNOT EXHIBIT IT "
                "-- --smoke is off, so this render carries no particles at "
                "all. A null difference across particle_detail tiers "
                "measured here is VACUOUS: it is 'the fixture has nothing "
                "to act on', not 'the tier does nothing'. The axis needs a "
                "fixture with smoke events; the 48-pose de_inferno "
                "teleport set the gtq arms use may be exactly this case.")
    if n <= 0:
        return ("videocfg_particle_detail: --smoke is ON but this render "
                "resolved ZERO instances, so it still cannot exhibit the "
                "axis. Same vacuity as --smoke off; see the note there.")
    return (f"videocfg_particle_detail: this fixture CAN exhibit it -- "
            f"--smoke {smoke_mode} with {n} instance(s). A null across "
            f"tiers measured here would be a real null.")
