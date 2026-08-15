#!/usr/bin/env python3
"""`r_texturefilteringquality` — the sampler-state tier.

WHAT THIS MODULE IS FOR
------------------------
The renderer already has the MACHINERY this cvar selects between:
`mip_taps()` implements the OpenGL anisotropic rule
(N = clamp(Pmax/Pmin, 1, maxAniso) taps of width Pmax/N along the major
axis, degenerating to isotropic trilinear at N = 1), driven by
`--max-aniso` / `--max-aniso-aux`, with `--mip` and `--mip-bias` beside
it. What has never existed is the TIER -> SAMPLER-STATE table that says
which of those values each of CS2's six tiers selects.

This file is that table. It refused until it was READ, and it has now
been read out of the engine's own shipped UI resource -- not guessed from
the Source-lineage convention, even though the convention turns out to be
what the engine ships.

ROUTE 1 -- THE UI RESOURCE, READ.
game/csgo/pak01_dir.vpk ->
panorama/layout/settings/settings_video.vxml_c, the same decompressed KV3
string table that gave the FSR modes, lists six contiguous
label/element-id pairs:

    '#SFUI_Settings_Bilinear'         'matforceaniso0'
    '#SFUI_Settings_Trilinear'        'matforceaniso1'
    '#SFUI_Settings_Anisotropic_2X'   'matforceaniso2'
    '#SFUI_Settings_Anisotropic_4X'   'matforceaniso3'
    '#SFUI_Settings_Anisotropic_8X'   'matforceaniso4'
    '#SFUI_Settings_Anisotropic_16X'  'matforceaniso5'

Six options, indices 0-5, matching the six legal values of the cvar and
the four the shipped presets use (video_defaults_{0,1,2,3} -> 0, 1, 3, 5).

ROUTE 2 -- THE ARMS, AND THEY SAY SOMETHING THE LABELS ALONE DO NOT.
Subtracting a six-capture per-pixel mean of the tier-0 repeats from each
arm isolates the DETERMINISTIC effect of each tier; the noise model
validates exactly (predicted off-diagonal covariance -0.86710, measured
-0.86710). Correlating those deterministic effects:

    tier 1 correlates only 0.58-0.65 with tiers 2-5, and its spectrum vs
      tier 0 is BELOW 1 in every band -- tier 1 is BLURRIER than tier 0.
    tiers 2-5 correlate ~1.0 with each other, amplitude 2.78 -> 3.11 ->
      3.02 -> 3.10.

A ladder of anisotropy alone CANNOT produce a blur step, so tier 1 is a
different mechanism rather than a weaker one -- which is exactly
bilinear (point mip) -> trilinear (linear mip): blending two mips removes
the mip-seam aliasing that makes point-mip look artificially sharp. The
measurement and the labels agree, and neither alone would have given the
mip-filter column.

WHAT THE OLD GAP ROW GOT WRONG, RECORDED BECAUSE IT IS THE POINT
-----------------------------------------------------------------
The superseded basis in `render_config.py` read the five arms' effect
magnitudes against tier 0 (1.48 / 2.08 / 2.22 / 2.23 / 2.23) and
concluded: "tiers 3, 4 and 5 are mutually indistinguishable and tier 1
sits at the floor with tier 0. Six tiers, THREE distinguishable states
{0,1} {2} {3,4,5} ... a mapping that gives 3, 4 and 5 different tap
counts contradicts the reference."

Three things in that are wrong, and each is a different error:

  1. IT PAIRED EVERY ARM AGAINST TIER 0. Two arms both 2.23x from tier 0
     can still differ from each other; "equidistant from a third point"
     is not "identical". The pairwise measurement is a DIFFERENT
     measurement, and when it was finally run it put 3/4/5 at the
     same-config floor against each other -- so the conclusion happened
     to survive, but it did not follow from the evidence given for it.
  2. IT PUT TIER 1 WITH TIER 0. Tier 1 correlates only 0.58-0.65 with
     tiers 2-5 and is BLURRIER than tier 0. It is not a weak version of
     the aniso ladder, it is the mip filter changing. Four resolvable
     states in this corpus, not three: {0} {1} {2} {3,4,5}.
  3. IT SAID THE 4/8/16 MAPPING "CONTRADICTS THE REFERENCE". It does
     not. Those three tiers are BELOW THIS FIXTURE'S THRESHOLD, which is
     a statement about the fixture. The engine's own UI resource labels
     them 4X / 8X / 16X, and a corpus that cannot separate them is
     consistent with that, not evidence against it. Reading "we could
     not measure a difference" as "there is no difference" is the same
     move as reading a null on a non-exhibiting fixture as a finding.

THE OTHER HALF OF THE SAMPLER STATE
------------------------------------
A filtering tier in a Source-lineage engine is not only an aniso count,
so aniso, mip filter and LOD bias are separate columns here. That
mattered: a one-column aniso table would have had to explain tier 1 as
an aniso change, and tier 1 is a mip-filter change.
"""
from __future__ import annotations

from typing import Dict, Optional

# tier -> sampler state. READ from the engine's own UI resource, then
# CONFIRMED by measurement. See the module docstring for the two routes.
#
#   aniso : max anisotropic taps for the COLOUR array (--max-aniso)
#   mip   : the MIP FILTER -- "point" (nearest mip, no blend) or "linear"
#           (trilinear). This is the column a one-column aniso table
#           cannot express, and tier 0 -> 1 is exactly that step.
#   bias  : LOD bias. Zero on every tier; see NO_PER_TIER_BIAS.
TIER_SAMPLER_STATE = {
    0: dict(aniso=1, mip="point", bias=0.0,
            source="matforceaniso0 / '#SFUI_Settings_Bilinear'"),
    1: dict(aniso=1, mip="linear", bias=0.0,
            source="matforceaniso1 / '#SFUI_Settings_Trilinear'"),
    2: dict(aniso=2, mip="linear", bias=0.0,
            source="matforceaniso2 / '#SFUI_Settings_Anisotropic_2X'"),
    3: dict(aniso=4, mip="linear", bias=0.0,
            source="matforceaniso3 / '#SFUI_Settings_Anisotropic_4X'"),
    4: dict(aniso=8, mip="linear", bias=0.0,
            source="matforceaniso4 / '#SFUI_Settings_Anisotropic_8X'"),
    5: dict(aniso=16, mip="linear", bias=0.0,
            source="matforceaniso5 / '#SFUI_Settings_Anisotropic_16X'"),
}

# The element ids are pure `matforceaniso{N}` -- no bias token anywhere --
# and the measurement shows no global LOD shift beyond the single
# point->linear mip-filter change at 0->1. The mip/residency axis is
# videocfg_texture_detail, a DIFFERENT cvar.
NO_PER_TIER_BIAS = (
    "no per-tier LOD bias: the UI element ids carry none and the arms show "
    "no global LOD shift beyond the 0->1 mip-filter change. mip bias is "
    "videocfg_texture_detail's axis, not this one.")

# WHAT IS STILL UNREAD, and it is one level below this table: the literal
# VkSamplerCreateInfo each tier produces -- the `maxAnisotropy` float, the
# `mipmapMode`, the `mipLodBias`. `matforceaniso{N}` is the UI ELEMENT ID,
# not the sampler. That mapping lives in rendersystemvulkan.so / client.so,
# and the terul depot carries no executable. So the SEMANTICS of every
# tier are pinned and the exact float handed to the driver is not.
SAMPLER_STRUCT_UNREAD = (
    "the literal VkSamplerCreateInfo per tier (maxAnisotropy, mipmapMode, "
    "mipLodBias) is NOT read -- matforceaniso{N} is a UI element id, not a "
    "sampler. Route: rendersystemvulkan.so / client.so from the ws-1 "
    "install; the terul depot carries no executable.")

# AND THE CORPUS CANNOT ARBITRATE THE TOP THREE ROWS. Tiers 3, 4 and 5 sit
# at the same-config repeat floor against each other (pairwise box8
# 0.5020 / 0.5226 / 0.4723 against a floor of 0.4624-0.5090), so this
# fixture cannot distinguish "4 / 8 / 16 as the UI labels them" from "a
# table that saturates". The labels are the READ and the corpus is
# CONSISTENT with them; it does not independently confirm them.
#
# Why the fixture is weak here is itself readable, and it is fixable: all
# five tfq arms carry fsr_detail 3, so they sit on the noisy FSR-ON floor
# (#94). The base_fsr0 vs videocfg_fsr_detail-0 pair reads box8 0.2298 --
# less than half the FSR-ON floor. A re-capture of the tfq sweep at
# fsr_detail 0, on a fixture with grazing-angle floor coverage, would
# drop the threshold enough to separate 4x / 8x / 16x if they differ.
TOP_TIERS_UNARBITRATED = (
    "tiers 3/4/5 are mutually AT the same-config repeat floor in this "
    "corpus (pairwise box8 0.5020/0.5226/0.4723 vs a 0.4624-0.5090 "
    "floor), so the 4/8/16 aniso values are the UI's read and not this "
    "corpus's measurement. All five arms carry fsr_detail 3 and so sit on "
    "the noisy FSR-ON floor; a re-capture at fsr_detail 0 with "
    "grazing-angle coverage would halve the threshold.")

class FilteringTierUnread(Exception):
    """Raised for a tier whose sampler state has not been read."""


def sampler_state(tier: int) -> dict:
    """The sampler state for a tier, or a refusal naming what to read."""
    if tier not in TIER_SAMPLER_STATE:
        raise ValueError(
            f"r_texturefilteringquality={tier} is outside the domain the 53 "
            f"GT configs exercise (0..5). A tier no capture used has no "
            f"sampler state to read.")
    state = TIER_SAMPLER_STATE[tier]
    if state is None:
        raise FilteringTierUnread(
            f"r_texturefilteringquality={tier}: {TIER_UNRESOLVED_NOTE}")
    return state


def describe(tier: int) -> str:
    """One line for the runtime banner. An UNREAD tier prints as unread
    rather than as a default -- the distinction between 'this render used
    the reference's value' and 'this render used ours' is the whole
    point of the table."""
    state = TIER_SAMPLER_STATE.get(tier)
    if state is None:
        return (f"r_texturefilteringquality={tier}: UNREAD -- this render "
                f"is at the renderer's own --max-aniso/--mip values on "
                f"this axis, NOT at the reference's")
    return (f"r_texturefilteringquality={tier}: aniso {state['aniso']}, "
            f"mip {state['mip']}, bias {state['bias']:+g} "
            f"(source: {state.get('source', 'UNSTATED')})")
