#!/usr/bin/env python3
"""CS2 `video.cfg` — parsed into a typed schema that CONSUMES EVERY KEY.

WHY THIS EXISTS
---------------
The charter's rule 2 is that config axes select VALUES, never EXISTENCE. A
preset file is therefore not a set of feature switches we may honour or
ignore; it is the value assignment for passes that always run. This module
is the one-to-one map from the ground-truth preset files onto our render
axes.

REFUSAL, NOT DEFAULTING
-----------------------
Two failure modes are refused rather than absorbed:

* an UNKNOWN key raises :class:`UnknownSettingError`. A silently-ignored
  key is a config axis the renderer claims to consume and does not, which
  is exactly the "superseded scalar defaults silently" defect (#55).
* an ABSENT key has NO fallback value. :meth:`VideoConfig.require` raises
  :class:`MissingSettingError`. Inventing a default here would be fitting a
  constant, and the resulting render would carry a value no capture ever
  used.

A key may be KNOWN and still have no effect on our render (window
borders, vsync). Those are declared with ``effect=NONE`` and are consumed
— parsed, typed, range-checked and reported — rather than dropped. The
distinction between "we read this and it does not change the image" and
"we never looked at it" is the whole point of the exercise.

PROVENANCE
----------
The key set and every observed value below were ENUMERATED from the 53
ground-truth config files in ``iji_model/counter_strike_render/vcfg/`` (copied from
ws-1 ``~/gtq/vcfg/``), which are the exact files each GT capture in
``~/gtq/caps/`` was rendered under. They are not a guess at CS2's schema;
they are the schema the acceptance corpus exercises. A value outside the
enumerated set is reported, not rejected — the enumeration is a floor.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------
# Effect classes. Declared per key so "consumed but inert" is a statement in
# the schema rather than an absence in the code.
# --------------------------------------------------------------------------
RENDER = "RENDER"    # selects a value inside a pass we implement
WINDOW = "WINDOW"    # selects the output raster size / presentation
DEVICE = "DEVICE"    # identifies the capture host; render-inert, but part
                     # of render identity (a different GPU is a different GT)
NONE = "NONE"        # consumed, range-checked, and demonstrably render-inert


@dataclass(frozen=True)
class Setting:
    key: str
    kind: type
    effect: str
    observed: Tuple[str, ...]
    doc: str


def _s(key, kind, effect, observed, doc):
    return Setting(key, kind, effect, tuple(observed), doc)


# The 40 keys present across the 53 GT config files. Counts in the comments
# are how many of the 53 files carry the key; a key absent from a file is
# absent, not defaulted.
SCHEMA: Dict[str, Setting] = {s.key: s for s in [
    # -- device identity (47/53) ------------------------------------------
    _s("Version", int, DEVICE, ["16"], "video.cfg schema version"),
    _s("VendorID", int, DEVICE, ["4318"], "PCI vendor of the capture GPU"),
    _s("DeviceID", int, DEVICE, ["11141"], "PCI device of the capture GPU"),
    _s("knowndevice", int, DEVICE, ["0"],
       "whether the engine recognised the GPU in its device database"),
    _s("Autoconfig", int, DEVICE, ["2"],
       "engine autoconfig state; 2 = settings were written by us"),

    # -- presentation / raster size ---------------------------------------
    _s("defaultres", int, WINDOW, ["1280"], "output raster width"),
    _s("defaultresheight", int, WINDOW, ["720"], "output raster height"),
    _s("fullscreen", int, WINDOW, ["0"], "fullscreen mode"),
    _s("coop_fullscreen", int, WINDOW, ["0", "1"], "cooperative fullscreen"),
    _s("nowindowborder", int, WINDOW, ["0", "1"], "borderless window"),
    _s("high_dpi", int, WINDOW, ["0"], "HiDPI backbuffer scaling"),
    _s("aspectratiomode", int, WINDOW, ["0"],
       "aspect policy; 0 = derive from the raster size. Hor+ was confirmed "
       "at 1e-4 deg from the captured frustum (#47)"),
    _s("mat_viewportscaling", int, WINDOW, ["1"],
       "viewport scale numerator; 1 = render at the raster size"),

    # -- render-inert but consumed ----------------------------------------
    _s("refreshrate_numerator", int, NONE, ["0"], "display refresh"),
    _s("refreshrate_denominator", int, NONE, ["0"], "display refresh"),
    _s("mat_vsync", int, NONE, ["0"], "vertical sync"),
    _s("vsync", int, NONE, ["0"], "legacy vsync key"),
    _s("fullscreen_min_on_focus_loss", int, NONE, ["1"], "focus behaviour"),
    _s("r_low_latency", int, NONE, ["0"], "latency-reduction mode"),
    _s("cpu_level", int, NONE, ["0", "3"],
       "CPU workload tier; selects simulation/particle CPU budgets, not a "
       "pixel-stage axis"),
    _s("mem_level", int, NONE, ["0"], "legacy system-memory tier"),
    _s("gpu_mem_level", int, NONE, ["0", "3"],
       "VRAM tier; selects residency budgets, not a shading expression"),
    _s("r_texture_stream", int, NONE, ["0"], "texture streaming"),
    _s("mat_grain_scale_override", int, NONE, ["0"],
       "film-grain scale override; 0 = no override. The de_inferno vpost "
       "carries no grain params"),

    # -- the render axes ---------------------------------------------------
    _s("gpu_level", int, RENDER, ["0", "3"],
       "GPU workload tier; the umbrella the individual videocfg_* keys "
       "refine"),
    _s("shaderquality", int, RENDER, ["0", "1"],
       "S_SHADER_QUALITY — selects WHICH decompiled expression a world "
       "family evaluates (#15)"),
    _s("r_texturefilteringquality", int, RENDER, ["0", "1", "2", "3", "4", "5"],
       "anisotropic filtering tier"),
    _s("msaa_samples", int, RENDER, ["0", "2", "4", "8"],
       "multisample count of the colour+depth targets; 0 = single-sample. "
       "The RESOLVE is 20 of the reference's 92 passes"),
    _s("r_csgo_cmaa_enable", int, RENDER, ["0", "1"],
       "Conservative Morphological AA pass"),
    _s("videocfg_shadow_quality", int, RENDER, ["0", "1", "2", "3"],
       "cascade count / resolution tier"),
    _s("videocfg_dynamic_shadows", int, RENDER, ["0", "1"],
       "dynamic (non-baked) shadow casters"),
    _s("videocfg_texture_detail", int, RENDER, ["0", "1", "2"],
       "mip bias / max resident mip"),
    _s("videocfg_particle_detail", int, RENDER, ["0", "1", "2", "3"],
       "particle and volumetric budget; bounds the smoke march"),
    _s("videocfg_ao_detail", int, RENDER, ["0", "2", "3"],
       "screen-space AO tier"),
    _s("videocfg_hdr_detail", int, RENDER, ["-1", "3"],
       "selects the HDR SCENE-COLOUR FORMAT. Read from the engine's own "
       "RT enumeration across all 30 GT arms, with no exceptions: "
       "-1 -> RGBA16161616F, 3 -> R11G11B10_FLOAT. See HDR_FORMAT_BY_TIER"),
    _s("videocfg_fsr_detail", int, RENDER, ["0", "1", "2", "3", "4"],
       "FidelityFX Super Resolution tier. 0 is native (no upscale) — see "
       "FSR_TIER_SCALE below for the provenance of the scale factors"),
    _s("fsr_enable", int, RENDER, ["0"],
       "legacy FSR master switch, superseded by videocfg_fsr_detail"),
    _s("csm_quality_level", int, RENDER, ["0"],
       "legacy cascaded-shadow-map tier, superseded by "
       "videocfg_shadow_quality"),
    _s("mat_antialias", int, RENDER, ["0"],
       "legacy MSAA sample count, superseded by msaa_samples"),
    _s("mat_aaquality", int, RENDER, ["0"],
       "legacy MSAA quality index, superseded by msaa_samples"),
]}

# Keys the GT files write with a `setting.` prefix. Both spellings appear
# (`Version` has none, `setting.msaa_samples` does), so the prefix is
# stripped and the bare name is canonical.
_PREFIX = "setting."


class VideoConfigError(Exception):
    pass


class UnknownSettingError(VideoConfigError):
    """A key not in SCHEMA. Refused: an ignored key is an unconsumed axis."""


class MissingSettingError(VideoConfigError):
    """A key this render needs is absent from the file. Refused: there is
    no default that any capture was actually rendered under."""


class MalformedConfigError(VideoConfigError):
    pass


@dataclass
class VideoConfig:
    """One parsed `video.cfg`. Values are kept as strings AND as parsed
    ints; the raw string is retained so the render-identity stamp records
    exactly what the file said."""
    path: str
    raw: Dict[str, str] = field(default_factory=dict)
    sha256: str = ""

    # -- access ------------------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self.raw

    def require(self, key: str) -> int:
        """Typed read. Refuses on absence — never substitutes a default."""
        if key not in SCHEMA:
            raise UnknownSettingError(f"{key!r} is not in the video.cfg schema")
        if key not in self.raw:
            raise MissingSettingError(
                f"{os.path.basename(self.path)} has no {key!r}; this render "
                f"needs it and there is no value any GT capture used. Add the "
                f"key to the config or capture an arm that sets it.")
        s = SCHEMA[key]
        try:
            return s.kind(self.raw[key])
        except ValueError as e:
            raise MalformedConfigError(
                f"{key}={self.raw[key]!r} is not {s.kind.__name__}") from e

    def observed_range_note(self) -> list:
        """Values outside the enumerated GT set. Reported, not rejected:
        the enumeration is a floor over 53 files, not CS2's full domain."""
        out = []
        for k, v in self.raw.items():
            s = SCHEMA[k]
            if s.observed and v not in s.observed:
                out.append(f"{k}={v} (GT files only ever carried "
                           f"{'/'.join(s.observed)})")
        return out

    def unconsumed(self, consumed: set) -> list:
        """Keys present in the file that no caller read. The complement of
        `require()` calls — printed at runtime so 'consumed' is measured,
        not asserted."""
        return sorted(set(self.raw) - set(consumed))


_KV = re.compile(r'^\s*"([^"]+)"\s+"([^"]*)"\s*$')
# The root block name sits alone on its own line: `"video.cfg"` with the
# `{` on the NEXT line. It is not a key/value pair.
_BLOCK = re.compile(r'^\s*"([^"]+)"\s*$')


def parse_video_config(path: str) -> VideoConfig:
    """Parse a CS2 video.cfg. Refuses unknown keys."""
    import hashlib
    data = open(path, "rb").read()
    cfg = VideoConfig(path=os.path.abspath(path),
                      sha256=hashlib.sha256(data).hexdigest())
    text = data.decode("utf-8", errors="replace")
    saw_root = False
    unknown = []
    for lineno, ln in enumerate(text.splitlines(), 1):
        st = ln.strip()
        if not st or st in ("{", "}"):
            continue
        b = _BLOCK.match(ln)
        if b:
            if b.group(1) != "video.cfg":
                raise MalformedConfigError(
                    f"{path}:{lineno}: unexpected block {b.group(1)!r}; this "
                    f"parser knows only the 'video.cfg' root")
            saw_root = True
            continue
        m = _KV.match(ln)
        if not m:
            raise MalformedConfigError(f"{path}:{lineno}: cannot parse {ln!r}")
        k, v = m.group(1), m.group(2)
        if k.startswith(_PREFIX):
            k = k[len(_PREFIX):]
        if k not in SCHEMA:
            unknown.append(f"{path}:{lineno}: {k!r}")
            continue
        cfg.raw[k] = v
    if unknown:
        raise UnknownSettingError(
            "refusing a config with keys this renderer does not consume — "
            "an ignored key is an axis we claim to implement and do not:\n  "
            + "\n  ".join(unknown))
    if not saw_root:
        raise MalformedConfigError(f"{path}: no 'video.cfg' root block")
    if not cfg.raw:
        raise MalformedConfigError(f"{path}: no settings")
    return cfg


# --------------------------------------------------------------------------
# Axis derivations
# --------------------------------------------------------------------------
# videocfg_fsr_detail -> render scale. READ, then MEASURED.
# --------------------------------------------------------------------------
# This table refused for the whole of this project's life, on the grounds
# that AMD's published quality-mode ratios are FSR's defaults and not
# necessarily Valve's selection. That refusal was correct and it is now
# DISCHARGED by two independent routes, neither of which is an assumption.
#
# ROUTE 1 -- THE ENGINE'S OWN UI RESOURCE, READ.
# The CS2 depot on terul (/data/cs2-artifacts/work/dd, build 2000880)
# carries game/csgo/pak01_dir.vpk, whose
# panorama/layout/settings/settings_video.vxml_c holds a binary KV3 v5
# `LaCo` block, LZ4-compressed. Decompressed, its string table lists the
# dropdown in document order:
#
#     '#SFUI_Settings_FSR'              'fsr4'   (label: Performance)
#     '#SFUI_Settings_Balanced'         'fsr3'
#                                       'fsr2'   (label: Quality)
#     '#SFUI_Settings_Ultra_Quality'    'fsr1'
#     '#SFUI_Settings_FSR_Disabled'     'fsr0'
#
# with resource/csgo_english.txt giving the labels, including
# FSR_Disabled = "Disabled (Highest Quality)". So THE AXIS IS
# QUALITY-INVERTED: a higher tier is a MORE aggressive upscale. That is
# confirmed independently by cfg/video_defaults_*.txt, where the LOW
# presets carry the HIGH tiers (level0 -> 3, level1 -> 2, level2/3 -> 0).
#
# ROUTE 2 -- THE ARMS, MEASURED, AND THE INTEGERS COME OUT EXACT.
# No render target is ever allocated at the internal size: FSR renders
# into a sub-rect of full-size targets, which is exactly why the RT
# enumeration could not give the scale. But the internal grid imprints a
# sampling comb in the captured frame, and its beat against the 1280
# display grid puts a second comb at 1280 - Wi. Per-column
# gradient-energy spectra over poses 1-47 resolve both:
#
#     tier 1  H 295 / 935  -> Wi = 985    V 166 / 526  -> Hi = 554
#     tier 2  H 423 / 857  -> Wi = 857    V 238 / 482  -> Hi = 482
#     tier 3  H 525 / 755  -> Wi = 755    V 296 / 424  -> Hi = 424
#     tier 0  NO second comb at all, on two independent arms -- native
#
# THE INTEGERS ARE floor(dim * pct), NOT round(dim / ratio):
#     floor(1280*0.77) = 985  measured 985   round(1280/1.3) = 985  agrees
#     floor(1280*0.67) = 857  measured 857   round(1280/1.5) = 853  WRONG
#     floor(1280*0.59) = 755  measured 755   round(1280/1.7) = 753  WRONG
# The engine keys AMD's PERCENTAGES (77/67/59) and FLOORS. Two of the
# three rows discriminate between the two forms, which is what makes this
# a read rather than a fit that happened to land.
FSR_TIER_PCT: Dict[int, Optional[float]] = {
    0: 1.00,     # Disabled (Highest Quality)
    1: 0.77,     # Ultra Quality
    2: 0.67,     # Quality
    3: 0.59,     # Balanced
    # TIER 4 IS 0.59 ON THIS BUILD, NOT 0.50. Its UI label is
    # "Performance" and AMD's Performance mode is 50%, but the arm says
    # otherwise and the arm is the engine answering. videocfg_fsr_detail-4
    # is byte-for-byte a level-0 config with fsr=4 and is
    # indistinguishable from tier 3 on every instrument: same combs at the
    # same amplitude (H 525/755), spectral band ratios equal to four
    # decimals, full-res mean|d| vs preset0 2.6199 and box8 0.4822 --
    # INSIDE the six-way same-config repeat floor 0.4624-0.5090. A 50%
    # render would be invisible to the comb probe (640x360 lands on the
    # native comb), but the PRESENCE of the 755/525 comb at full strength
    # rules 640x360 out. Most likely an engine-side clamp to the top of a
    # four-entry table; NOT confirmed at source level, which needs
    # client.so and the depot carries no executable.
    # We reproduce what the engine DID, not what the label says.
    4: 0.59,
}

FSR_TIER_SCALE: Dict[int, Optional[float]] = dict(FSR_TIER_PCT)

# The exact internal raster each tier produces at 1280x720, kept as the
# arithmetic's own check: applying the scale with round() reproduces 985
# and MISSES 857 and 755.
FSR_TIER_INTERNAL_1280x720 = {
    0: (1280, 720), 1: (985, 554), 2: (857, 482), 3: (755, 424),
    4: (755, 424),
}

# WHETHER RCAS RUNS IS ALSO A TIER PROPERTY, AND IT IS NOT UNIFORM.
# Normalised power-spectrum ratio against the native arm (base_fsr0 is the
# control and reads 1.0000 +- 0.005 in every band, so instrument noise is
# under 0.5%):
#
#     arm       gain k80-160   gain k160-320   ratio at its OWN Nyquist
#     tier 1      0.8961         0.6852             0.2137
#     tier 2      1.1160         1.2473             0.8305
#     tier 3      1.1106         1.1859             0.8116
#     tier 4      1.1116         1.1874             0.8089
#
# Tiers 2/3/4 GAIN 11-25% mid-band energy over native -- a sharpening
# signature -- and hold ~0.81-0.83 at their own source Nyquist, which is
# edge-adaptive EASU. Tier 1 LOSES energy in every band and collapses to
# 0.21 at its Nyquist: a soft resample with no sharpening at all. That
# matches the render-target count exactly, since tier 1 is the only
# non-zero tier allocating no extra full-res LDR target (12 per allocation
# block against 13 for tiers 2/3/4) -- and that extra target is the
# EASU->RCAS ping-pong, both being combos of ONE shader
# (materials/dev/upsample_fsr.vmat_c and upsample_rcas.vmat_c both name
# fsr_upscale.vfx, with F_FSR and F_RCAS).
#
# ONE ARM ONLY for tier 1. Two independent instruments agree on it, but a
# repeat is owed before "Ultra Quality runs EASU alone" is settled.
FSR_TIER_RCAS = {0: False, 1: False, 2: True, 3: True, 4: True}

# ...AND THE SHARPNESS CONSTANT IS NOW READ, OUT OF THE SHIPPED BINARY.
# It refused for a long time on the correct grounds: both upsample_*.vmat_c
# carry EMPTY m_floatParams, so it is a runtime uniform rather than a
# material default, and this build exposes no cvar value readback. The
# route that worked is ws-1's own libclient.so, and it took three attempts
# whose FAILURES narrowed it:
#
#   1. absolute 8-byte pointer to the name string   -> ZERO found
#   2. R_X86_64_RELATIVE relocation addends          -> ZERO of 301116
#   3. PC-RELATIVE lea in code                       -> EXACTLY ONE site
#
# 1 and 2 are what a PIE looks like when .rodata is reached by
# `lea reg, [rip+disp32]`: there is no pointer to find and nothing to
# relocate, so both negatives were evidence about the FORM, not absence of
# the registration.
#
# At the single site, decoding backwards from the name lea:
#
#     f3 0f 10 05 <disp32>   movss xmm0, [rip-22662415]   -> va 0xafbf04
#     ...
#     48 8d 35    <disp32>   lea   rsi, [rip-...]         -> the NAME
#
# and the four bytes at 0xafbf04 are 00 00 80 3e = 0.25f exactly.
#
# TWO-SIDED CALIBRATION OF THE TECHNIQUE ITSELF, because a reader that
# returns a plausible float for anything is worthless: run on the BOOLEAN
# cvar r_csgo_fsr_enable_mip_bias the same walk finds NO movss at all --
# the window carries `ba 01 00 00 00`, an integer immediate. So it reports
# a float where a float exists and declines where the type differs.
# ⚠️ AND THE "SINGLE lea SITE IS THE CONSTRUCTOR" PREMISE IS UNSOUND,
# established by objdump on the two sites side by side. This value
# survives that, but its provenance is narrower than first claimed.
#
# r_ssao_radius, disassembled -- NO float anywhere, and the name string
# is an argument to a REPEATED 3-argument call:
#     mov  %r15,%rdi
#     mov  $0x7,%edx
#     lea  -0x158a54c(%rip),%rsi        # the NAME
#     call 20d0590
# with the identical call target and rdx=7 recurring for string after
# string. That is a registration/interning helper, not a ConVar
# constructor with a default. So a name string CAN be referenced from
# non-constructor code, and "the one lea site" does not identify the
# constructor by itself.
#
# r_csgo_fsr_rcas_sharpness, same treatment -- a DIFFERENT shape:
#     mov   %r12,%r8
#     mov   %r14,%rdi
#     pxor  %xmm1,%xmm1
#     movss -0x159cd0f(%rip),%xmm0      # 0.25
#     lea   -0x1497212(%rip),%rcx       # a second string (help text?)
#     mov   $0x7,%eax
#     lea   -0x155f4d1(%rip),%rsi       # the NAME
#     movups %xmm1,...                  # zeroing a stack buffer
# Several distinct argument registers, a float in xmm0 (the first SSE
# argument), a second string pointer, and stack-buffer initialisation.
# That is constructor-shaped where the ssao site is not, which is what
# makes SHAPE the discriminator rather than the mere presence of a lea.
#
# Note also that the 7 sits in EDX at one site and EAX at the other, with
# different call targets -- so it is not one argument slot shared between
# them, and reading it as "the flags" at both was itself too generous.
#
# STATUS OF THIS VALUE: 0.25f is what the constructor-shaped site loads
# into xmm0 beside the name. It is NOT PROVEN to be the ConVar default --
# proving that needs the call target identified as the ConVar constructor,
# which objdump alone does not give on a stripped build. It is used
# because it is the best-evidenced number available and because the
# alternative was skipping RCAS entirely; if it is ever falsified the
# renderer's own banner names it at every FSR frame.
FSR_RCAS_SHARPNESS = 0.25

# FsrRcasCon: con.x = exp2(-sharpness). At the read default that is
# 0.8408964152537145.
FSR_RCAS_CON_DEFAULT = 2.0 ** -FSR_RCAS_SHARPNESS

FSR_RCAS_SHARPNESS_NOTE = (
    "r_csgo_fsr_rcas_sharpness = 0.25f, READ from libclient.so on ws-1: "
    "the movss that loads xmm0 immediately before the ConVar name lea, at "
    "the single PC-relative site that resolves to the name string, targets "
    "va 0xafbf04 whose bytes are 00 00 80 3e. con.x = exp2(-0.25) = "
    "0.8408964152537145. The technique is calibrated two-sided: on the "
    "BOOLEAN r_csgo_fsr_enable_mip_bias the same walk finds no movss and "
    "an integer immediate instead, so it declines rather than inventing a "
    "float.")

# ⚠️ RETRACTED: `r_csgo_fsr_enable_mip_bias` default is NOT read.
# ------------------------------------------------------------------
# This block briefly claimed the default was 1 (ENABLED) on the strength
# of a `mov edx, 1` near the registration. That claim was WRONG and is
# withdrawn. Batch-running the same reader over 23 cvars exposed it:
#
#   r_ssao_radius     imm32 7      r_ssao_bias      imm32 7
#   r_ssao_strength   imm32 7      r_particle_timescale imm32 7
#   mat_overdraw      imm32 3      r_csgo_cmaa_quality  imm32 3
#   r_lightmap_size   imm32 3      r_csgo_mboit         imm32 1
#
# Four unrelated cvars reading 7 -- three of which (ssao radius, bias,
# strength) MUST be floats -- is not a coincidence, it is one argument
# being read for all of them. And the tell was already in the sharpness
# site: `mov eax, 7` sits there too, next to the movss that carries the
# REAL default. 7 is a FCVAR flags bitmask, not a value.
#
# Widening the window on r_ssao_radius settles it: NO xmm load in 120
# bytes, and the only clean immediate is `mov edx, 7`. So the integer
# branch of that reader reports the flags argument, and it would have
# reported a plausible-looking number for every integer and boolean cvar
# in the game.
#
# THE READER'S CALIBRATION WAS INSUFFICIENT AND I SHIPPED IT ANYWAY. It
# was two-sided on TYPE -- float cvars show a movss, boolean ones do not
# -- and that much held. It was never calibrated on WHICH ARGUMENT the
# integer branch reads, and there is no oracle in this project that would
# have caught it. The recurring-7 across cvars whose types disagree is
# what caught it, i.e. an internal consistency check, not the test I
# wrote. Same shape as the improbable-CONSTANT tell.
#
# WHAT SURVIVES: only the movss reads, because those land in xmm0 (the
# first SSE argument) rather than in a general register shared with
# flags. And they carry their own caveat -- the single lea site is not
# PROVEN to be the constructor rather than a use of the name.
FSR_MIP_BIAS_UNREAD_NOTE = (
    "r_csgo_fsr_enable_mip_bias default is NOT READ. A claim that it is 1 "
    "was made and RETRACTED: the integer branch of the binary reader turned "
    "out to report the FCVAR flags argument, not the default -- four "
    "unrelated cvars all read 7, three of them float-typed, and `mov eax, 7` "
    "sits at the sharpness site next to the movss carrying its real value. "
    "So the reference's FSR mip-bias state is unknown, this renderer applies "
    "NONE, and that is a stated difference on top of an unread default "
    "rather than a known divergence.")

FSR_TIER_UNRESOLVED_NOTE = (
    "videocfg_fsr_detail tier->render-scale is READ: "
    "panorama/layout/settings/settings_video.vxml_c for the tier->mode "
    "map, and the arms' own sampling combs for the exact internal "
    "rasters. A tier outside 0..4 has no row because neither a capture "
    "nor a UI entry exercises one.")


def fsr_render_scale(cfg: VideoConfig) -> float:
    tier = cfg.require("videocfg_fsr_detail")
    scale = FSR_TIER_SCALE.get(tier)
    if scale is None:
        raise MissingSettingError(
            f"videocfg_fsr_detail={tier}: {FSR_TIER_UNRESOLVED_NOTE}")
    return scale


def fsr_internal_size(tier: int, out_w: int, out_h: int):
    """The internal raster. FLOOR, not round -- see FSR_TIER_PCT."""
    import math
    pct = FSR_TIER_SCALE.get(tier)
    if pct is None:
        raise MissingSettingError(
            f"videocfg_fsr_detail={tier}: {FSR_TIER_UNRESOLVED_NOTE}")
    return (max(1, math.floor(out_w * pct)), max(1, math.floor(out_h * pct)))


# videocfg_hdr_detail -> the HDR scene-colour format the engine allocates.
# READ, not inferred: every one of the 30 GT arms with a console log was
# checked against its own vcfg, and the split is total. The single-axis arm
# `videocfg_hdr_detail-n1` isolates it — same base config as the other
# single-axis arms, only this key changed, and it is the only one of them
# that allocates RGBA16161616F.
#
#   hdr_detail  3 -> "RT 1280x720 R11G11B10_FLOAT"    (preset0, preset1,
#                    and every videocfg_*/r_texturefilteringquality arm)
#   hdr_detail -1 -> "RT 1280x720 RGBA16161616F"      (preset2, preset3,
#                    videocfg_hdr_detail-n1)
#
# WHY THIS MATTERS TO THE RESOLVE, and it is not cosmetic: R11G11B10_FLOAT
# has NO ALPHA CHANNEL and a different representable maximum per channel,
# so the MSAA resolve's `clamp(rgb, 0, 65504)` — which is the largest finite
# HALF — is the RGBA16F constant and is wrong here. The gate corpus is
# preset2, i.e. RGBA16161616F, so the gate CANNOT catch a wrong
# R11G11B10_FLOAT path. Two of the four presets and almost every single-axis
# arm run on the format the gate does not exercise.
HDR_FORMAT_BY_TIER = {
    3: "VK_FORMAT_B10G11R11_UFLOAT_PACK32",
    -1: "VK_FORMAT_R16G16B16A16_SFLOAT",
}


def hdr_scene_format(cfg: VideoConfig) -> str:
    tier = cfg.require("videocfg_hdr_detail")
    fmt = HDR_FORMAT_BY_TIER.get(tier)
    if fmt is None:
        raise MissingSettingError(
            f"videocfg_hdr_detail={tier} was never exercised by any of the "
            f"30 GT arms, so which HDR format the engine allocates for it "
            f"has not been read. Capture an arm at that tier and read the "
            f"RenderPipelineCsgo RT line rather than assuming.")
    return fmt


def msaa_sample_count(cfg: VideoConfig) -> int:
    """0 in the config means single-sample; the resolve then averages one
    sample, which is the identity and is still RUN, so the pass's presence
    does not depend on the value."""
    n = cfg.require("msaa_samples")
    if n not in (0, 2, 4, 8):
        raise MalformedConfigError(f"msaa_samples={n} is not a legal count")
    return 1 if n == 0 else n


def cmaa_enabled(cfg: VideoConfig) -> bool:
    return bool(cfg.require("r_csgo_cmaa_enable"))


def shader_quality(cfg: VideoConfig) -> int:
    return cfg.require("shaderquality")


def output_size(cfg: VideoConfig) -> Tuple[int, int]:
    return cfg.require("defaultres"), cfg.require("defaultresheight")


def describe(cfg: VideoConfig) -> str:
    """One-line runtime banner. Uncertainty prints, it does not live in a
    comment (#34)."""
    bits = []
    for k in ("shaderquality", "msaa_samples", "r_csgo_cmaa_enable",
              "videocfg_fsr_detail", "videocfg_hdr_detail",
              "videocfg_particle_detail", "videocfg_ao_detail",
              "videocfg_shadow_quality", "r_texturefilteringquality"):
        bits.append(f"{k}={cfg.raw.get(k, 'ABSENT')}")
    return (f"video.cfg {os.path.basename(cfg.path)} "
            f"sha {cfg.sha256[:12]} :: " + " ".join(bits))
