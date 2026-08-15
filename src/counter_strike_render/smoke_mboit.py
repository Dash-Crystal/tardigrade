"""The five-family volumetric smoke subsystem and the two MBOIT resolves.

Seven families, enumerated in
`docs/projects/counter-strike-sft/SHADER_CALLFLOW_smoke_and_mboit.md`:

    smoke_volume            1,109 FLOPs/px   9 fetch (6x2D,2x3D,1xCube)  12 loops, depth 4
    smoke_volume_depth        709             6 fetch (2x2D,4x3D)         4 loops, depth 2
    smoke_volume_mask         194             3 fetch (3x2D)              1 loop,  depth 0
    write_smoke_depth_water_reflection  79    4 fetch (4x2D)              0 loops
    overlay_smoke              47             4 fetch (4x2D)              0 loops
    mboit_mixed_combine       298            25 fetch (25x2D)             0 loops
    mboitfinal                277            18 fetch (18x2D)             0 loops

=======================================================================
WHAT IS TRANSCRIBED AND WHAT IS NOT -- read this before trusting a line
=======================================================================

**No GLSL or SPIR-V for any of these seven exists in this checkout.** The
`.vcs` archives live on the render host; what is here is the measured
enumeration only:

  * `.scratch-batchA/allps_deep.json` -- MEASURED, authoritative, and the
    source of `REF` below for `smoke_volume`, `overlay_smoke` and
    `mboitfinal`: axis names with their `m_nMin`/`m_nMax`, the per-region
    loop table (id, depth, FLOPs, fetches), the straight-line split, the
    full `P(computation | FLOP)` histogram and the fetch inventory by
    sampler dimensionality.
  * `.scratch-batchA/allps_range.json` -- MEASURED per-module FLOP/fetch/
    loop ranges for all seven.
  * `SHADER_CALLFLOW_smoke_and_mboit.md` -- the derived document, the only
    source for the axis lists of the other four.

So the bodies below are **NOT instruction-for-instruction transcriptions**
the way `mboit_pass1` / `_mboit_t6` in `gpu_render.py` are. They are the
closest complete form that satisfies every measured structural fact, and
`REF` + `conformance()` machine-check them against it: the loop-region
count, each region's nest depth, which region the `sampler3D` taps sit in,
the executed fetch count by dimensionality, and the axis names and ranges.
Everything a shipped module fixes that the enumeration does not record --
the exact algebra, the constants, the texture contents -- is ours, and
every place that matters says so at the point where it is written.

**A measured correction to the document.**  `SHADER_CALLFLOW_smoke_and_
mboit.md` prints `D_SMOKE_INSTANCES(0..6)` and reads it as "up to 7 smoke
volumes handled in one draw".  `allps_deep.json` records
`["D_SMOKE_INSTANCES", 1, 6]`, and `batchA_enum.py:axes()` builds that
triple as `(m_szName, m_nMin, m_nMax)`.  The axis therefore has **min 1,
max 6 -- six values, one to six volumes**, not seven and never zero.  The
implementation uses 1..6 and `conformance()` fails if a caller asks for 0
or 7.

=======================================================================
THE MARCH, AND WHAT BOUNDS ITS STEP COUNT
=======================================================================

`smoke_volume` is a ray-march.  The evidence, all of it measured:

  * 12 loop regions nested to depth 4.  Every other family enumerated in
    this project has one loop shape -- an outer word loop over a cull
    bitmask with one inner bit-peel, depth <= 1.
  * the two `sampler3D` taps are inside region `%13587` at **depth 1**, so
    the executed 3D fetch count is (march steps) x 2, never 2.
  * `smoothstep` at ln p -2.58 and `length/sqrt/rsqrt` at -2.67, with **no
    `reflect`, no `cross`, and no derivatives row at all** -- distance
    evaluation and soft edges, not a BRDF.

**The step count is bounded by a uniform and nothing in the enumeration
bounds that uniform.**  §6.3 of the source document declines to give a
ceiling for exactly this reason, and this file will not invent one either.
The bitmask-exhaustion and array-bounds arguments that bounded every
previous family's loops do not apply: this is not a binner.  Our trip
count comes from `--smoke-march-steps` (and `--smoke-march-steps-lowq`
under `D_SMOKE_QUALITY=0`), it is a choice of ours and NOT a claim about
the engine's, and `reach_report()` always prints the trip counts and the
executed 3D fetch count so a truncated march can never read as tested.

=======================================================================
HOW OUR TWELVE REGIONS MAP ONTO THE MEASURED TWELVE
=======================================================================

    ref region  depth  FLOPs  fetches      ours
    ----------  -----  -----  -----------  -----------------------------
    %23009        0      28   --           R_INSTANCES   per smoke volume
    %18295        0       6   --           R_SLAB        ray/AABB, 3 axes
    %13587        1     335   2x sampler3D R_MARCH       the march itself
    %16797        2     118   --           R_OCTAVE      procedural detail
    %22416        2      88   --           R_LIGHT_A     light-0 in-scatter
    %22418        3       1   --           R_CASCADE_A   cascade select
    %22419        4      56   --           R_TAP_A       occlusion tap
    %22699        2       1   --           R_COMBINE     per-light sum
    %22421        2       1   --           R_LIGHT_B     light-1 in-scatter
    %22422        3       1   --           R_CASCADE_B   cascade select
    %22423        4      56   --           R_TAP_B       occlusion tap
    %14179        0       4   --           R_MOMENT      MBOIT moment emit

The two identical depth-2/3/4 chains `(88, 1, 56)` and `(1, 1, 56)` are
read here as **two lights, unrolled** -- one chain per light, each with a
cascade select under it and an occlusion tap under that.  That reading is
an inference from the shape, not a measurement; what IS measured is that
there are exactly two such chains, that they bottom out at depth 4 with a
56-FLOP body, and that **none of the depth-2/3/4 regions fetches
anything**, which is why every one of them below is analytic.

=======================================================================
DEPENDENCIES ON THE FRAME, AND THE ONE WE DO NOT BUILD
=======================================================================

`write_smoke_depth_water_reflection` implies a water-reflection pass that
renders the scene a second time from a mirrored viewpoint with its own
depth buffer.  **That pass is not built here** and building it was
explicitly out of scope; the depth WRITE this family specifies is
implemented and runs, against a reflection depth target this module
allocates and clears.  See `write_smoke_depth_water_reflection()` for
exactly what is and is not present, and the report in
`SMOKE_AND_MBOIT_IMPLEMENTATION.md` for what schedules it.

`overlay_smoke` declares `D_VIEWMODEL_PASS`; the viewmodel pass itself is
another agent's.  `overlay_smoke()` takes the viewmodel layer as an
argument and never reaches into that agent's code -- the contract is in
the doc.
"""

import math

import torch

# ======================================================================
# 1.  THE MEASURED REFERENCE, transcribed verbatim from the enumeration
# ======================================================================
# Provenance per family is in `src`.  Nothing here is derived; it is the
# table `conformance()` checks the implementation against, so a change to
# the implementation that breaks a measured fact fails a check instead of
# quietly becoming the new truth.

REF = {
    "smoke_volume": dict(
        src="allps_deep.json['smoke_volume'] (MEASURED)",
        axes=(("D_MBOIT_PASS1", 0, 1), ("D_MBOIT_PASS2", 0, 1),
              ("D_MBOIT_OPTIM", 0, 1), ("D_SMOKE_INSTANCES", 1, 6),
              ("D_MBOIT_4_MOMENTS", 0, 1), ("D_SMOKE_QUALITY", 0, 1),
              ("D_SMOKE_USE_NOISE_TEXTURE", 0, 1),
              ("D_SMOKE_NEW_VISUALS", 0, 1), ("D_SMOKE_PERF_TEST", 0, 1),
              ("D_SMOKE_FULLRES", 0, 1)),
        static_combos=1, flops=1109, inst=1410,
        fetch={"sampler2D": 6, "sampler3D": 2, "samplerCube": 1},
        straight_line=(414, {"sampler2D": 6, "samplerCube": 1}),
        loops=12, max_depth=4,
        regions=((23009, 0, 28, {}), (18295, 0, 6, {}),
                 (13587, 1, 335, {"sampler3D": 2}),
                 (16797, 2, 118, {}), (22416, 2, 88, {}),
                 (22699, 2, 1, {}), (22418, 3, 1, {}),
                 (22419, 4, 56, {}), (22421, 2, 1, {}),
                 (22422, 3, 1, {}), (22423, 4, 56, {}),
                 (14179, 0, 4, {})),
        flopclass={"raw arithmetic (mul/add/sub/div)": 449,
                   "dot products": 92, "smoothstep": 84,
                   "clamp / saturate": 82, "length / sqrt / rsqrt": 77,
                   "mix / lerp": 75, "pow / exp / log": 44, "fma": 42,
                   "normalize": 35, "abs / sign / floor / fract": 28,
                   "matrix products": 28, "max": 20, "trig": 20,
                   "min": 16, "bit ops": 9, "integer arithmetic": 8}),
    "smoke_volume_depth": dict(
        src="SHADER_CALLFLOW_smoke_and_mboit.md §2 (derived); FLOP/fetch/"
            "loop ranges cross-checked against allps_range.json",
        axes=(("D_SMOKE_NEW_VISUALS", 0, 1),),
        static_combos=1, flops=709, inst=None,
        fetch={"sampler2D": 2, "sampler3D": 4},
        straight_line=None, loops=4, max_depth=2,
        regions=None, flopclass=None),
    "smoke_volume_mask": dict(
        src="SHADER_CALLFLOW_smoke_and_mboit.md §2 (derived)",
        axes=(("D_MBOIT_PASS2", 0, 1), ("D_MBOIT_4_MOMENTS", 0, 1),
              ("D_DDA_MASK", 0, 1), ("D_SMOKE_FULLRES_PASS", 0, 2),
              ("D_SCOPE_CLIP", 0, 1)),
        static_combos=1, flops=194, inst=None,
        fetch={"sampler2D": 3},
        straight_line=None, loops=1, max_depth=0,
        regions=None, flopclass=None),
    "write_smoke_depth_water_reflection": dict(
        src="SHADER_CALLFLOW_smoke_and_mboit.md §2 + allps_range.json "
            "(flmin==flmax==79, femin==femax==4, lpmax==0: this family's "
            "one module is fully determined by the range file)",
        axes=(), static_combos=1, flops=79, inst=114,
        fetch={"sampler2D": 4},
        straight_line=(79, {"sampler2D": 4}), loops=0, max_depth=None,
        regions=(), flopclass=None),
    "overlay_smoke": dict(
        src="allps_deep.json['overlay_smoke'] (MEASURED)",
        axes=(("D_ACCUM_PASS", 0, 1), ("D_VIEWMODEL_PASS", 0, 1)),
        static_combos=1, flops=47, inst=48,
        fetch={"sampler2D": 4},
        straight_line=(47, {"sampler2D": 4}), loops=0, max_depth=None,
        regions=(),
        flopclass={"mix / lerp": 36, "smoothstep": 6,
                   "raw arithmetic (mul/add/sub/div)": 5}),
    "mboit_mixed_combine": dict(
        src="SHADER_CALLFLOW_smoke_and_mboit.md §3 (derived); peak fetch "
            "25 cross-checked against allps_range.json femax",
        axes=(("D_HI_LOW", 0, 1), ("D_LOW_HI", 0, 1),
              ("D_LOW_HI_OUTPUT", 0, 1), ("D_UPSCALE_SMOKE_DEPTH", 0, 1),
              ("D_KEEP_FULL_RES", 0, 1), ("D_MBOIT_4_MOMENTS", 0, 1)),
        static_combos=1, flops=298, inst=None,
        fetch={"sampler2D": 25},
        straight_line=(298, {"sampler2D": 25}), loops=0, max_depth=None,
        regions=(), flopclass=None),
    "mboitfinal": dict(
        src="allps_deep.json['mboitfinal'] (MEASURED)",
        axes=(("D_UPSCALE_MIXED", 0, 1), ("D_COMBINE_HI_RES", 0, 1),
              ("D_DENOISE_SMOKE", 0, 1), ("D_COVER_GAPS", 0, 2),
              ("D_SMOKE_PERF_TEST", 0, 1)),
        static_combos=1, flops=277, inst=389,
        fetch={"sampler2D": 18},
        straight_line=(277, {"sampler2D": 18}), loops=0, max_depth=None,
        regions=(),
        flopclass={"raw arithmetic (mul/add/sub/div)": 170,
                   "clamp / saturate": 30, "pow / exp / log": 32,
                   "abs / sign / floor / fract": 14, "normalize": 7,
                   "integer arithmetic": 7, "mix / lerp": 6,
                   "dot products": 5, "min": 3, "max": 3}),
}

# Our loop regions, in the order of REF["smoke_volume"]["regions"], with
# the depth each one runs at.  conformance() checks that the depths of
# this table equal the measured depths region for region, and that every
# region actually executed at least once in the run being reported.
OUR_REGIONS = (
    ("R_INSTANCES", 0, 23009),
    ("R_SLAB", 0, 18295),
    ("R_MARCH", 1, 13587),
    ("R_OCTAVE", 2, 16797),
    ("R_LIGHT_A", 2, 22416),
    ("R_COMBINE", 2, 22699),
    ("R_CASCADE_A", 3, 22418),
    ("R_TAP_A", 4, 22419),
    ("R_LIGHT_B", 2, 22421),
    ("R_CASCADE_B", 3, 22422),
    ("R_TAP_B", 4, 22423),
    ("R_MOMENT", 0, 14179),
)

# The declared tap layout of the two resolves.  These sum to the measured
# peak fetch counts (25 and 18) and `conformance()` checks the SUM as well
# as the executed count, so a tap that is declared and never issued -- or
# issued and never declared -- fails.
COMBINE_TAPS = (
    ("transparency_rgb", 1, None), ("transparency_a", 1, None),
    ("transparency_b0", 1, None), ("transparency_depth", 1, None),   # 4
    ("smoke_rgb_2x2", 4, None), ("smoke_a_2x2", 4, None),
    ("smoke_depth_2x2", 4, None),                                    # -> 16
    ("smoke_depth_guide_2x2", 4, "D_UPSCALE_SMOKE_DEPTH"),
    ("scene_depth_guide", 1, "D_UPSCALE_SMOKE_DEPTH"),               # -> 21
    ("moments_lo", 1, None), ("moments_hi", 1, None),                # -> 23
    ("scene_depth_out", 1, None), ("scene_rgb", 1, None),            # -> 25
)
FINAL_TAPS = (
    ("mixed_rgb", 1, None), ("mixed_a", 1, None), ("mixed_z", 1, None),
    ("hires_rgb", 1, None), ("hires_a", 1, None),                    # -> 5
    ("scene_rgb", 1, None),                                          # -> 6
    ("denoise_smoke_3x3", 9, "D_DENOISE_SMOKE"),                     # -> 15
    ("cover_gaps", 3, "D_COVER_GAPS"),                               # -> 18
)


def expected_combine_taps():
    """How many of the 25 this combo issues. PEAK 25 is reached with
    D_UPSCALE_SMOKE_DEPTH on; that is a fact about where the peak is, not
    a reason to run only that value."""
    n = 0
    for _nm, c, ax in COMBINE_TAPS:
        if ax == "D_UPSCALE_SMOKE_DEPTH" and not on(A.mboit_upscale_smoke_depth):
            continue
        n += c
    return n


def expected_final_taps():
    """How many of the 18 this combo issues. PEAK 18 needs
    D_DENOISE_SMOKE on and D_COVER_GAPS=2; D_SMOKE_PERF_TEST drops both."""
    perf = on(A.smoke_perf_test)
    n = 0
    for _nm, c, ax in FINAL_TAPS:
        if ax == "D_DENOISE_SMOKE":
            n += c if (on(A.mboit_denoise_smoke) and not perf) else 0
        elif ax == "D_COVER_GAPS":
            n += 0 if perf else (0, 1, 3)[int(A.mboit_cover_gaps)]
        else:
            n += c
    return n

# ======================================================================
# 2.  RUNTIME COUNTERS -- the thing the checks actually read
# ======================================================================
# TRIPS counts loop-body executions per (family, region).  FETCH counts
# texel-reading calls per (family, sampler dimensionality).  Both are
# incremented at the site, never inferred, so `fetch == steps * 2` is a
# measurement of the code that ran and not a restatement of the design.

TRIPS = {}
FETCH = {}
NOTE = []


def _trip(fam, region, n=1):
    TRIPS[(fam, region)] = TRIPS.get((fam, region), 0) + n


def _fetch(fam, kind, n=1):
    FETCH[(fam, kind)] = FETCH.get((fam, kind), 0) + n


def reset_counters():
    TRIPS.clear()
    FETCH.clear()
    NOTE.clear()


# ======================================================================
# 3.  ARGUMENTS -- every axis of every family, every value
# ======================================================================
# NAMESPACED: every flag starts --smoke- or --mboit-.  No flag added here
# may be tested with bare truthiness when it is string-valued; "off" is
# truthy in Python and that is exactly how a path gets silently enabled.
# Every string axis therefore has an `on()` helper below and is compared
# against a literal.

ONOFF = ("off", "on")


def on(v):
    """The ONLY way a string-valued axis in this module is read.

    `if args.smoke_new_visuals:` is true for "off".  This is the fix, and
    it raises on anything that is not exactly one of the two literals
    rather than falling through to a default, because a typo that silently
    picked a path is the same bug one layer down.
    """
    if v in (True, False):
        return bool(v)
    if v not in ONOFF:
        raise SystemExit(f"smoke/mboit axis value {v!r} is not one of "
                         f"{ONOFF}; a string axis is never read as a bool")
    return v == "on"


def add_arguments(parser):
    """Every axis of all seven families, at every value it can take.

    Ranges come from `REF`, i.e. from `m_nMin`/`m_nMax` in the shipped
    combo metadata.  A quality level that happens not to select a combo is
    recorded in the help text and never used to drop a value.
    """
    g = parser

    # --- the subsystem enable ------------------------------------------
    g.add_argument("--smoke", choices=("off", "synth", "pack", "demo"),
                   default="off",
                   help="the five-family volumetric smoke subsystem. "
                        "'synth' marches synthesised volumes (the world "
                        "pack contains no smoke, so this is the only way "
                        "the path is reachable at all); 'pack' reads the "
                        "volumes from --smoke-side; 'demo' places them "
                        "where the demo says, from --smoke-grenades. "
                        "STRING, never tested for truthiness.")
    g.add_argument("--smoke-grenades", default=None,
                   help="an iji/cs2-demo-grenades/v1 .json. Under "
                        "--smoke demo its smokegrenade_detonate x/y/z and "
                        "the detonate/expire tick pair drive the volumes: "
                        "position and lifetime are READ, radius growth and "
                        "density are DEFAULTS and print as such at first "
                        "draw. Without it, --smoke demo REFUSES rather "
                        "than falling back to synth, because a synth "
                        "fallback under a demo flag renders smoke that is "
                        "not where the demo put it and nothing says so.")
    g.add_argument("--smoke-tickrate", type=float, default=64.0,
                   help="ticks per second for the grenade lifetimes. CS2 "
                        "demos are 64; a 128-tick demo halves every "
                        "duration if this is wrong.")
    g.add_argument("--smoke-side", default=None,
                   help="fast_pack_fam.py's smoke/MBOIT side table "
                        "(fam_smoke_mboit.pt): the 3D noise and shape "
                        "volumes the march fetches, the smoke colour ramp, "
                        "the jitter tile and the scope mask.")

    # --- smoke_volume: all ten dynamic axes -----------------------------
    g.add_argument("--smoke-instances", type=int, default=6,
                   help="D_SMOKE_INSTANCES. MEASURED RANGE 1..6 "
                        "(allps_deep.json), i.e. one to six volumes in one "
                        "draw. SHADER_CALLFLOW_smoke_and_mboit.md prints "
                        "0..6 and reads it as seven; that is a "
                        "transcription error against its own source.")
    g.add_argument("--smoke-quality", choices=("0", "1"), default="1",
                   help="D_SMOKE_QUALITY. Selects --smoke-march-steps (1) "
                        "or --smoke-march-steps-lowq (0). Which uniform "
                        "the engine keys is NOT determined by the "
                        "enumeration; see the module docstring.")
    g.add_argument("--smoke-fullres", choices=ONOFF, default="off",
                   help="D_SMOKE_FULLRES. 'off' marches at "
                        "1/--mboit-mixed-scale resolution, which is what "
                        "the mixed-resolution MBOIT resolve exists to "
                        "recombine; 'on' marches at full resolution and "
                        "the resolve upscales nothing.")
    g.add_argument("--smoke-use-noise-texture", choices=ONOFF, default="on",
                   help="D_SMOKE_USE_NOISE_TEXTURE. 'on' issues the second "
                        "sampler3D tap inside the march (so 2 3D fetches "
                        "per step); 'off' issues one and derives the "
                        "erosion analytically. Both are implemented.")
    g.add_argument("--smoke-new-visuals", choices=ONOFF, default="on",
                   help="D_SMOKE_NEW_VISUALS. Shared with "
                        "smoke_volume_depth, where it is the ONLY axis. "
                        "Selects the two-lobe density profile and the "
                        "ramp-driven tint over the single-lobe form.")
    g.add_argument("--smoke-perf-test", choices=ONOFF, default="off",
                   help="D_SMOKE_PERF_TEST. Shared with mboitfinal. The "
                        "cheap path: fixed density, no octaves, no "
                        "in-scatter chains. Implemented as its own arm, "
                        "not as a scale factor on the full one.")
    g.add_argument("--smoke-mboit-pass1", choices=ONOFF, default="on",
                   help="D_MBOIT_PASS1 on smoke_volume: the march writes "
                        "its absorbance and power moments into the SAME "
                        "moment buffers the per-material pass 1 writes.")
    g.add_argument("--smoke-mboit-pass2", choices=ONOFF, default="on",
                   help="D_MBOIT_PASS2 on smoke_volume: the march resolves "
                        "against those moments and emits premultiplied "
                        "colour, exactly as the material pass 2 does.")
    g.add_argument("--smoke-mboit-optim", choices=ONOFF, default="off",
                   help="D_MBOIT_OPTIM. The optimised resolve: skips the "
                        "moment reconstruction where the accumulated "
                        "absorbance is below the shipped early-out "
                        "constant and takes transmittance 1.")

    # --- the march's trip counts. NONE of these may be 0. ---------------
    g.add_argument("--smoke-march-steps", type=int, default=16,
                   help="trip count of the depth-1 march region %%13587 at "
                        "D_SMOKE_QUALITY=1. OURS, not the engine's: the "
                        "enumeration does not classify this loop's bound "
                        "and refuses to invent a ceiling, so this file "
                        "does not claim one either. Always printed.")
    g.add_argument("--smoke-march-steps-lowq", type=int, default=8,
                   help="the same trip count at D_SMOKE_QUALITY=0.")
    g.add_argument("--smoke-detail-octaves", type=int, default=2,
                   help="trip count of the depth-2 region %%16797, the "
                        "procedural detail refinement. It fetches nothing "
                        "-- the measured region has no fetches at all -- "
                        "so it is domain rotations, not more taps.")
    g.add_argument("--smoke-light-steps", type=int, default=2,
                   help="trip count of the two depth-2 in-scatter regions "
                        "%%22416 and %%22421 (one per light).")
    g.add_argument("--smoke-shadow-cascades", type=int, default=2,
                   help="trip count of the two depth-3 regions %%22418 and "
                        "%%22422.")
    g.add_argument("--smoke-shadow-taps", type=int, default=2,
                   help="trip count of the two depth-4 regions %%22419 and "
                        "%%22423 -- the deepest loop bodies in any family "
                        "enumerated in this project.")
    g.add_argument("--smoke-wind-phase", type=float, default=0.0,
                   help="offset added to the noise volume's fetch "
                        "coordinate, i.e. how far the smoke has advected. "
                        "The engine drives it from frame time; the render "
                        "path sets it per frame.")

    # --- smoke_volume_depth --------------------------------------------
    g.add_argument("--smoke-depth-steps", type=int, default=12,
                   help="trip count of smoke_volume_depth's depth-1 coarse "
                        "march (2 of its 4 sampler3D sites).")
    g.add_argument("--smoke-depth-refine", type=int, default=4,
                   help="trip count of its depth-2 bisection refinement "
                        "(the other 2 sampler3D sites). 4 halvings "
                        "resolve the crossing to 1/16 of a coarse step.")
    g.add_argument("--smoke-depth-threshold", type=float, default=0.35,
                   help="accumulated optical depth at which "
                        "smoke_volume_depth calls the volume opaque and "
                        "writes its depth.")

    # --- smoke_volume_mask ---------------------------------------------
    g.add_argument("--smoke-dda-mask", choices=ONOFF, default="on",
                   help="D_DDA_MASK. 'on' traverses the coarse occupancy "
                        "grid with a digital differential analyser (the "
                        "family's single depth-0 loop); 'off' takes the "
                        "conservative screen-space bounding rectangle of "
                        "every volume. Both are implemented.")
    g.add_argument("--smoke-dda-steps", type=int, default=24,
                   help="trip count of the DDA loop.")
    g.add_argument("--smoke-grid", type=int, default=24,
                   help="edge of the cubic occupancy grid the DDA walks.")
    g.add_argument("--smoke-fullres-pass", type=int, default=1,
                   help="D_SMOKE_FULLRES_PASS, MEASURED RANGE 0..2 -- a "
                        "three-way resolution axis on the mask: 0 = the "
                        "mask's own reduced grid, 1 = the march grid, "
                        "2 = full resolution.")
    g.add_argument("--smoke-scope-clip", choices=ONOFF, default="off",
                   help="D_SCOPE_CLIP: clips smoke against a sniper "
                        "scope, using the mask's third sampler2D site.")
    g.add_argument("--smoke-scope-radius", type=float, default=0.34,
                   help="scope radius in fractions of the frame's shorter "
                        "edge. Ours; no scope geometry is in this pack.")

    # --- overlay_smoke --------------------------------------------------
    g.add_argument("--smoke-accum-pass", choices=ONOFF, default="off",
                   help="D_ACCUM_PASS on overlay_smoke: accumulate into a "
                        "separate accumulation target (premultiplied, "
                        "additive) instead of compositing over the scene.")
    g.add_argument("--smoke-viewmodel-pass", choices=ONOFF, default="off",
                   help="D_VIEWMODEL_PASS on overlay_smoke: smoke "
                        "composites SEPARATELY over the viewmodel, "
                        "against the viewmodel depth. The viewmodel pass "
                        "itself belongs to another agent; this flag runs "
                        "overlay_smoke's viewmodel arm over whatever "
                        "viewmodel layer is handed in, and says so when "
                        "none is.")
    g.add_argument("--smoke-overlay-softness", type=float, default=0.12,
                   help="width of overlay_smoke's depth-difference "
                        "smoothstep, in metres. The family spends 6 of "
                        "its 47 FLOPs on smoothstep; this is its width.")

    # --- write_smoke_depth_water_reflection -----------------------------
    g.add_argument("--smoke-water-reflection", choices=ONOFF, default="off",
                   help="run write_smoke_depth_water_reflection. The "
                        "reflection PASS -- a second scene render from a "
                        "mirrored viewpoint -- is NOT built here and was "
                        "out of scope; this writes the smoke depth into a "
                        "reflection depth target that this module "
                        "allocates and clears to the far plane.")
    g.add_argument("--smoke-water-plane-y", type=float, default=0.0,
                   help="world height of the mirror plane the reflection "
                        "view is built about.")

    # --- the two resolves ----------------------------------------------
    g.add_argument("--mboit-resolve",
                   choices=("off", "hi_low", "low_hi", "keep_full"),
                   default="off",
                   help="mboit_mixed_combine's resolution pairing. "
                        "hi_low = D_HI_LOW (transparency full res, smoke "
                        "reduced); low_hi = D_LOW_HI (the reverse); "
                        "keep_full = D_KEEP_FULL_RES (both full res, no "
                        "upscale). Consumes the moment buffers the "
                        "EXISTING D_MBOIT_PASS1/PASS2 already write -- "
                        "there is no second MBOIT here.")
    g.add_argument("--mboit-low-hi-output", choices=ONOFF, default="off",
                   help="D_LOW_HI_OUTPUT: write the combine's result at "
                        "the reduced resolution of the pair instead of at "
                        "full resolution, leaving the upscale to "
                        "mboitfinal's D_UPSCALE_MIXED.")
    g.add_argument("--mboit-upscale-smoke-depth", choices=ONOFF,
                   default="on",
                   help="D_UPSCALE_SMOKE_DEPTH: joint-bilateral upsample "
                        "of the reduced-resolution smoke depth against "
                        "the full-resolution scene depth. This axis is "
                        "why the resolve and the smoke are one system.")
    g.add_argument("--mboit-mixed-scale", type=int, default=2,
                   help="the reduction factor of the 'low' half of the "
                        "mixed-resolution pair.")
    g.add_argument("--mboit-upscale-mixed", choices=ONOFF, default="on",
                   help="D_UPSCALE_MIXED on mboitfinal.")
    g.add_argument("--mboit-combine-hi-res", choices=ONOFF, default="on",
                   help="D_COMBINE_HI_RES on mboitfinal: fold the "
                        "full-resolution half back in after the upscale.")
    g.add_argument("--mboit-denoise-smoke", choices=ONOFF, default="on",
                   help="D_DENOISE_SMOKE on mboitfinal: the 3x3 "
                        "depth-weighted smoke filter, 9 of its 18 taps.")
    g.add_argument("--mboit-cover-gaps", type=int, default=2,
                   help="D_COVER_GAPS, MEASURED RANGE 0..2: 0 issues no "
                        "gap taps, 1 issues one, 2 issues all three. The "
                        "peak fetch count of 18 is reached at 2.")

    # --- checks ---------------------------------------------------------
    g.add_argument("--smoke-selftest", action="store_true",
                   help="run the smoke/MBOIT reachability and conformance "
                        "check on a synthesised smoke volume and a "
                        "synthesised transparent stack, then exit.")
    g.add_argument("--smoke-selftest-sweep", action="store_true",
                   help="run the selftest once for EVERY value of EVERY "
                        "axis of all seven families, one axis at a time "
                        "from the baseline, and fail on any value whose "
                        "path reaches no pixels. Not the Cartesian "
                        "product, and it does not claim to be.")
    g.add_argument("--smoke-selftest-inject", default=None,
                   help="break one thing on purpose and require the check "
                        "to notice. 'list' prints the names. A check that "
                        "cannot fail is not a check "
                        "(CHECKS_THAT_CANNOT_FAIL.md).")
    return parser


A = None            # the argparse namespace, injected by configure()
INJECT = None


def configure(args):
    global A, INJECT
    A = args
    INJECT = getattr(args, "smoke_selftest_inject", None)
    if INJECT == "list":
        print("smoke/mboit fault-injection names:", flush=True)
        for k, v in INJECTIONS.items():
            print(f"    {k:26s} {v}", flush=True)
        raise SystemExit(0)
    if INJECT is not None and INJECT not in INJECTIONS:
        raise SystemExit(f"unknown --smoke-selftest-inject {INJECT!r}; "
                         "use 'list'")
    if A.smoke != "off":
        lo, hi = 1, 6
        if not lo <= A.smoke_instances <= hi:
            raise SystemExit(
                f"--smoke-instances {A.smoke_instances} is outside "
                f"D_SMOKE_INSTANCES' MEASURED range {lo}..{hi} "
                "(allps_deep.json)")
        for nm in ("smoke_march_steps", "smoke_march_steps_lowq",
                   "smoke_detail_octaves", "smoke_light_steps",
                   "smoke_shadow_cascades", "smoke_shadow_taps",
                   "smoke_depth_steps", "smoke_depth_refine",
                   "smoke_dda_steps"):
            if getattr(A, nm) < 1:
                raise SystemExit(
                    f"--{nm.replace('_', '-')} must be >= 1: a trip count "
                    "of 0 is a path that silently does not run, which is "
                    "worse than a missing feature")
    if not 0 <= A.smoke_fullres_pass <= 2:
        raise SystemExit("--smoke-fullres-pass is D_SMOKE_FULLRES_PASS, "
                         "MEASURED range 0..2")
    if not 0 <= A.mboit_cover_gaps <= 2:
        raise SystemExit("--mboit-cover-gaps is D_COVER_GAPS, MEASURED "
                         "range 0..2")
    return A


INJECTIONS = {
    "flatten_march": "issue the two sampler3D taps ONCE outside the "
                     "march instead of once per step",
    "ramp_in_march": "move the colour-ramp sampler2D tap INSIDE the "
                     "march, which the measured straight-line split "
                     "forbids",
    "mask_empty": "return an empty tile mask from smoke_volume_mask, so "
                  "the march has nothing to run on",
    "drop_depth4": "skip the two depth-4 tap regions, so the nest is "
                   "depth 3 and two measured regions never execute",
    "combine_taps": "skip four of mboit_mixed_combine's 25 taps",
    "final_taps": "skip mboitfinal's three cover-gap taps",
    "resolve_noop": "make mboitfinal return its input unchanged",
    "instances_seven": "ask for 7 smoke instances, outside the measured "
                       "1..6 range",
    "overlay_passthrough": "make overlay_smoke return the scene it was "
                           "given",
}


def _inj(name):
    return INJECT == name


# ======================================================================
# 4.  SAMPLERS -- every texel read in this file goes through one of these
# ======================================================================
# They exist so the fetch counters cannot drift from the fetches: there is
# no other way to read a texel here, so `FETCH` is the code's own count of
# what it issued and not a parallel bookkeeping that can disagree.


def _tex2d(fam, tex, u, v):
    """Bilinear `sampler2D`, clamp-to-edge, normalised uv. Counts 1 fetch.

    `tex` is (H, W) or (H, W, C); the result carries C or nothing.
    """
    _fetch(fam, "sampler2D")
    h, w = tex.shape[0], tex.shape[1]
    x = (u * w - 0.5).clamp(0, w - 1)
    y = (v * h - 0.5).clamp(0, h - 1)
    x0, y0 = x.floor(), y.floor()
    fx, fy = (x - x0), (y - y0)
    x0 = x0.long().clamp(0, w - 1)
    y0 = y0.long().clamp(0, h - 1)
    x1 = (x0 + 1).clamp(max=w - 1)
    y1 = (y0 + 1).clamp(max=h - 1)
    g00 = tex[y0, x0]
    if g00.dim() > fx.dim():
        fx, fy = fx.unsqueeze(-1), fy.unsqueeze(-1)
    return ((g00 * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
            + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)


def _tex3d(fam, vol, p):
    """Trilinear `sampler3D`, clamp-to-edge, normalised xyz. Counts 1.

    `vol` is (D, H, W). `p` is (..., 3) in [0,1]^3, wrapped -- a noise
    volume is tiled, and clamping it would put a constant slab at the
    edge of every march that leaves the unit cube.
    """
    _fetch(fam, "sampler3D")
    d, h, w = vol.shape
    q = p - p.floor()                                    # wrap
    x = q[..., 0] * w - 0.5
    y = q[..., 1] * h - 0.5
    z = q[..., 2] * d - 0.5
    x0, y0, z0 = x.floor(), y.floor(), z.floor()
    fx, fy, fz = x - x0, y - y0, z - z0
    x0 = x0.long() % w
    y0 = y0.long() % h
    z0 = z0.long() % d
    x1, y1, z1 = (x0 + 1) % w, (y0 + 1) % h, (z0 + 1) % d
    c000 = vol[z0, y0, x0]
    c100 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]
    c110 = vol[z0, y1, x1]
    c001 = vol[z1, y0, x0]
    c101 = vol[z1, y0, x1]
    c011 = vol[z1, y1, x0]
    c111 = vol[z1, y1, x1]
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c010 * (1 - fx) + c110 * fx
    c10 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    return (c00 * (1 - fy) + c01 * fy) * (1 - fz) + \
           (c10 * (1 - fy) + c11 * fy) * fz


def _texcube(fam, cube, d):
    """`samplerCube`, the one non-2D straight-line tap of smoke_volume.

    `cube` is (6, S, S, 3) in the GL face order +X -X +Y -Y +Z -Z.  Face
    selection is the standard major-axis rule; the smoke's ambient
    in-scatter reads it, which is why the family has a cube tap at all and
    no `reflect` anywhere -- it is sampled along the view ray, not a
    mirror direction.
    """
    _fetch(fam, "samplerCube")
    ax, ay, az = d[..., 0], d[..., 1], d[..., 2]
    aax, aay, aaz = ax.abs(), ay.abs(), az.abs()
    major_x = (aax >= aay) & (aax >= aaz)
    major_y = (~major_x) & (aay >= aaz)
    major_z = (~major_x) & (~major_y)
    eps = 1e-8
    face = torch.zeros_like(ax, dtype=torch.long)
    sc = torch.zeros_like(ax)
    tc = torch.zeros_like(ax)
    ma = torch.ones_like(ax)
    face = torch.where(major_x, torch.where(ax >= 0, 0, 1), face)
    sc = torch.where(major_x, torch.where(ax >= 0, -az, az), sc)
    tc = torch.where(major_x, -ay, tc)
    ma = torch.where(major_x, aax, ma)
    face = torch.where(major_y, torch.where(ay >= 0, 2, 3), face)
    sc = torch.where(major_y, ax, sc)
    tc = torch.where(major_y, torch.where(ay >= 0, az, -az), tc)
    ma = torch.where(major_y, aay, ma)
    face = torch.where(major_z, torch.where(az >= 0, 4, 5), face)
    sc = torch.where(major_z, torch.where(az >= 0, ax, -ax), sc)
    tc = torch.where(major_z, -ay, tc)
    ma = torch.where(major_z, aaz, ma)
    ma = ma.clamp(min=eps)
    u = (sc / ma) * 0.5 + 0.5
    v = (tc / ma) * 0.5 + 0.5
    s = cube.shape[1]
    xi = (u * s - 0.5).clamp(0, s - 1).round().long()
    yi = (v * s - 0.5).clamp(0, s - 1).round().long()
    return cube[face, yi, xi]


# ======================================================================
# 5.  THE VOLUMES
# ======================================================================


class SmokeVolumes:
    """Up to six volumes, D_SMOKE_INSTANCES' measured range being 1..6.

    Each volume carries the mat3 that takes world space into its own unit
    sphere -- that transform is `smoke_volume`'s `matrix products` line
    item (28 FLOPs, about two mat3 x vec3), and it is why an ellipsoid
    that has grown and sheared is still one `length()` away from a
    distance field.
    """

    def __init__(self, centre, inv_basis, radius, density, tint, age):
        self.centre = centre            # (N,3)
        self.inv_basis = inv_basis      # (N,3,3) world -> unit-sphere
        self.radius = radius            # (N,)  world-space bounding radius
        self.density = density          # (N,)
        self.tint = tint                # (N,3)
        self.age = age                  # (N,)  0 fresh .. 1 dissipated

    def __len__(self):
        return self.centre.shape[0]


def synth_volumes(n, eye, fwd, device, seed=20260807, spread=6.0,
                  dist=9.0, radius=3.0):
    """Synthesised smoke, because the world pack contains none.

    Placed in front of the camera in WORLD space, so the march, the
    depth producer, the mask, the resolve and the overlay all run on the
    real per-frame camera rather than on a fixed test frustum.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(n)
    e = eye.reshape(-1, 3)[0].to(device)
    f = fwd.reshape(-1, 3)[0].to(device)
    f = f / f.norm().clamp(min=1e-6)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = torch.cross(f, up, dim=0)
    right = right / right.norm().clamp(min=1e-6)
    up2 = torch.cross(right, f, dim=0)
    off = (torch.rand((n, 3), generator=g) * 2 - 1).to(device)
    centre = (e + f * dist
              + right * (off[:, :1] * spread)
              + up2 * (off[:, 1:2] * spread * 0.4)
              + f * (off[:, 2:3] * spread * 0.5))
    rad = radius * (0.6 + 0.8 * torch.rand((n,), generator=g).to(device))
    # a per-volume anisotropic scale and a yaw, so the mat3 is not a
    # multiple of the identity and the matrix-product term is real work
    yaw = (torch.rand((n,), generator=g).to(device)) * 6.2831853
    sx = 1.0 / (rad * (0.85 + 0.3 * torch.rand((n,), generator=g).to(device)))
    sy = 1.0 / (rad * (0.70 + 0.3 * torch.rand((n,), generator=g).to(device)))
    sz = 1.0 / (rad * (0.85 + 0.3 * torch.rand((n,), generator=g).to(device)))
    c, s = yaw.cos(), yaw.sin()
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    rot = torch.stack([torch.stack([c, z, s], -1),
                       torch.stack([z, o, z], -1),
                       torch.stack([-s, z, c], -1)], -2)      # (N,3,3)
    scale = torch.zeros((n, 3, 3), device=device)
    scale[:, 0, 0] = sx
    scale[:, 1, 1] = sy
    scale[:, 2, 2] = sz
    inv_basis = scale @ rot
    dens = 1.4 + 0.8 * torch.rand((n,), generator=g).to(device)
    tint = 0.55 + 0.35 * torch.rand((n, 3), generator=g).to(device)
    age = torch.rand((n,), generator=g).to(device) * 0.5
    return SmokeVolumes(centre, inv_basis, rad, dens, tint, age)


SRC_TO_M = 0.0254                 # 1 Source unit = 1 inch; the pack is metres
DEMO_SMOKE_RADIUS_MAX = 144.0     # Hammer units; CS2's smoke is ~a 144u ball
DEMO_SMOKE_GROW_S = 1.6           # detonate -> full radius
DEMO_SMOKE_FADE_S = 2.5           # expire tick -> gone
DEMO_SMOKE_DENSITY = 1.8
DEMO_SMOKE_TINT = (0.82, 0.83, 0.86)
_DEMO_SMOKE_PRINTED = False


def load_demo_grenades(path):
    """Read an `iji/cs2-demo-grenades/v1` file into detonate/expire pairs.

    Returns [(entityid, tick_on, tick_off, x, y, z)]. `smokegrenade_expired`
    is matched to its detonate by entityid; an unmatched detonate gets a
    lifetime from the file's own mean, and how many that was is printed.
    """
    import json
    with open(path) as f:
        d = json.load(f)
    rows = d.get("event_rows", {})
    det = sorted(rows.get("smokegrenade_detonate", []),
                 key=lambda e: int(e["tick"]))
    # WHY THE PAIRING IS ORDERED, measured rather than assumed. A dict
    # comprehension {entityid: tick} over the expire rows is LAST-WINS, and
    # this file has 38 expires across 37 distinct ids -- one id expires
    # twice. That single overwrite pulled the mean lifetime to 2357 ticks
    # (36.8 s at 64). Pairing each detonate with the first expire of its id
    # at a LATER tick, consuming each expire once, gives 1412 ticks for
    # every one of the 38 matched pairs -- 22.06 s, the engine's fixed
    # smoke lifetime, and the constancy is what confirms the pairing.
    #
    # I first blamed entity-id reuse across rounds. That is NOT what this
    # is: 39 distinct ids over 40 detonates, exactly one reused. The fix is
    # the same; the reason I gave for it was wrong.
    by_id = {}
    for e in rows.get("smokegrenade_expired", []):
        by_id.setdefault(int(e["entityid"]), []).append(int(e["tick"]))
    for k in by_id:
        by_id[k].sort()
    used = {k: 0 for k in by_id}
    pairs = []
    for e in det:
        eid, t0 = int(e["entityid"]), int(e["tick"])
        t1 = None
        lst = by_id.get(eid, [])
        i = used.get(eid, 0)
        while i < len(lst):
            if lst[i] > t0:
                t1 = lst[i]
                used[eid] = i + 1
                break
            i += 1
            used[eid] = i
        pairs.append((e, t0, t1))
    lifetimes = [t1 - t0 for _, t0, t1 in pairs if t1 is not None]
    mean_life = (sum(lifetimes) / len(lifetimes)) if lifetimes else 1150.0
    out, unmatched = [], 0
    for e, t0, t1 in pairs:
        eid = int(e["entityid"])
        if t1 is None:
            t1 = int(t0 + mean_life)
            unmatched += 1
    # x/y/z are the DETONATION point, which is where the cloud sits; the
    # projectile trajectory in `trajectories` is the throw and is not it.
        out.append((eid, t0, t1, float(e["x"]), float(e["y"]), float(e["z"])))
    return out, dict(n=len(out), unmatched=unmatched, mean_life=mean_life,
                     map_name=d.get("map_name"), demo=d.get("demo"))


def demo_volumes(events, tick, device, tickrate=64.0,
                 radius_max=DEMO_SMOKE_RADIUS_MAX,
                 grow_s=DEMO_SMOKE_GROW_S, fade_s=DEMO_SMOKE_FADE_S):
    """Volumes for the smokes alive at `tick`, or None if there are none.

    Radius grows over `grow_s` after detonate and the whole volume fades
    over `fade_s` after expire. Neither curve is read from a shader -- no
    .vcs in this tree carries the puff's growth -- so both are DEFAULTS and
    say so at runtime.
    """
    alive = [e for e in events
             if e[1] <= tick <= e[2] + int(fade_s * tickrate)]
    if not alive:
        return None
    n = len(alive)
    # The demo gives SOURCE units, Z-up. This renderer's world is VRF's
    # Y-up export in metres, pack = (y_src, z_src, x_src) * 0.0254 -- the
    # same permutation the probe loader uses (gpu_render.py:5417), which
    # was not assumed but picked as the only one of 48 whose probe SH-DC
    # correlates with the baked lightmap (r = +0.363, rest within +-0.09
    # of zero). Feeding raw Source coordinates into a metre-scale world
    # put every volume ~20x outside the map and marched nothing: the
    # banner said 5 volumes were alive while the frame was byte-identical
    # to --smoke off.
    cen = torch.tensor([[e[4], e[5], e[3]] for e in alive],
                       device=device, dtype=torch.float32) * SRC_TO_M
    t_on = torch.tensor([float(e[1]) for e in alive], device=device)
    t_off = torch.tensor([float(e[2]) for e in alive], device=device)
    age_s = (float(tick) - t_on) / tickrate
    grow = (age_s / grow_s).clamp(0.0, 1.0)
    # radius_max is quoted in SOURCE units (the banner says "144u"); the
    # marcher works in the pack's metres, so it converts here alongside cen.
    rad = (radius_max * SRC_TO_M) * (
        grow * grow * (3.0 - 2.0 * grow)).clamp(min=0.02)
    dead_s = ((float(tick) - t_off) / tickrate).clamp(min=0.0)
    fade = (1.0 - dead_s / fade_s).clamp(0.0, 1.0)
    eye3 = torch.eye(3, device=device).expand(n, 3, 3)
    inv_basis = eye3 / rad.view(n, 1, 1)
    dens = torch.full((n,), DEMO_SMOKE_DENSITY, device=device) * fade
    tint = torch.tensor(DEMO_SMOKE_TINT, device=device).expand(n, 3).clone()
    # SmokeVolumes.age is NORMALISED "0 fresh .. 1 dissipated" (see the
    # ctor comment), not seconds. Passing age in seconds put 4.3..12.3 into
    # a 0..1 field, and the march reads it as `density * (1 - 0.6 * age)`
    # (line ~1342), so every demo volume marched at NEGATIVE density and
    # contributed nothing: --smoke demo was byte-identical to --smoke off
    # while the banner reported 5 volumes alive. Normalise over the whole
    # detonate -> expire+fade span, which is what "dissipated" means here.
    span = ((t_off - t_on) + fade_s * tickrate).clamp(min=1.0)
    age01 = ((float(tick) - t_on) / span).clamp(0.0, 1.0)
    return SmokeVolumes(cen, inv_basis, rad, dens, tint, age01)


def demo_smoke_banner(meta, n_alive, tick):
    global _DEMO_SMOKE_PRINTED
    if _DEMO_SMOKE_PRINTED:
        return
    _DEMO_SMOKE_PRINTED = True
    print("[smoke] demo-driven volumes LIVE: %d smoke detonations from %s "
          "(%s), %d unmatched expire, mean lifetime %.0f ticks."
          % (meta["n"], meta.get("demo"), meta.get("map_name"),
             meta["unmatched"], meta["mean_life"]))
    print("[smoke]   %d alive at tick %s." % (n_alive, tick))
    print("[smoke]   DEFAULTED, no shader read for either: radius_max "
          "%.0fu, growth %.2fs smoothstep, fade %.2fs, density %.2f, tint "
          "%s." % (DEMO_SMOKE_RADIUS_MAX, DEMO_SMOKE_GROW_S,
                   DEMO_SMOKE_FADE_S, DEMO_SMOKE_DENSITY, DEMO_SMOKE_TINT))
    print("[smoke]   Position and lifetime ARE read: detonate x/y/z and the "
          "detonate/expire tick pair.")


def synth_textures(device, n3=32, n2=64, cube=16, seed=20260807):
    """The data dependencies, synthesised when no side table is loaded.

    These are REAL tensors read through the real samplers -- the march's
    two `sampler3D` taps fetch from `noise3` and `shape3`, and the fetch
    counter counts every one.  `fast_pack_fam.py` writes the same five
    arrays into `fam_smoke_mboit.pt`; this is the fallback so the path is
    reachable without a pack, not an alternative to it.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    def _valnoise(nn, octaves=4):
        acc = torch.zeros((nn, nn, nn))
        amp, tot = 1.0, 0.0
        res = 4
        for _ in range(octaves):
            lo = torch.rand((res, res, res), generator=g)
            up = torch.nn.functional.interpolate(
                lo[None, None], size=(nn, nn, nn), mode="trilinear",
                align_corners=False)[0, 0]
            acc = acc + up * amp
            tot += amp
            amp *= 0.5
            res = min(res * 2, nn)
        return (acc / tot).to(device)

    noise3 = _valnoise(n3)
    # the SHAPE volume: a soft radial puff with a low-frequency erosion,
    # i.e. a baked density field rather than noise
    ii = torch.linspace(-1, 1, n3)
    zz, yy, xx = torch.meshgrid(ii, ii, ii, indexing="ij")
    r = (xx * xx + yy * yy + zz * zz).sqrt().to(device)
    shape3 = ((1.0 - r).clamp(0, 1) ** 1.5) * (0.7 + 0.6 * _valnoise(n3, 3))
    # smoke colour ramp: density 0..1 across u, age 0..1 across v
    du = torch.linspace(0, 1, n2)
    dv = torch.linspace(0, 1, n2)
    U, V = torch.meshgrid(dv, du, indexing="ij")
    ramp = torch.stack([
        (0.92 - 0.45 * U) * (0.35 + 0.65 * (1 - V)),
        (0.90 - 0.47 * U) * (0.35 + 0.65 * (1 - V)),
        (0.88 - 0.44 * U) * (0.38 + 0.62 * (1 - V)),
        (1.0 - 0.85 * V),
    ], -1).to(device)
    # an interleaved-gradient jitter tile (not white noise: the march's
    # per-pixel start offset has to decorrelate across neighbours or the
    # under-sampling shows up as banding rather than as grain)
    jt = 64
    jy, jx = torch.meshgrid(torch.arange(jt), torch.arange(jt),
                            indexing="ij")
    jitter = torch.frac(52.9829189 * torch.frac(
        0.06711056 * jx.float() + 0.00583715 * jy.float())).to(device)
    # scope mask for D_SCOPE_CLIP
    sy2, sx2 = torch.meshgrid(torch.linspace(-1, 1, n2),
                              torch.linspace(-1, 1, n2), indexing="ij")
    scope = (1.0 - ((sx2 * sx2 + sy2 * sy2).sqrt() / 0.8)).clamp(0, 1)
    scope = (scope * 3).clamp(0, 1).to(device)
    # ambient environment cube: sky above, ground below, in linear units
    cu = torch.zeros((6, cube, cube, 3))
    sky = torch.tensor([0.52, 0.62, 0.78])
    gnd = torch.tensor([0.20, 0.18, 0.15])
    for f_i in range(6):
        for yy2 in range(cube):
            t = yy2 / max(cube - 1, 1)
            if f_i == 2:
                col = sky
            elif f_i == 3:
                col = gnd
            else:
                col = gnd + (sky - gnd) * (1.0 - t)
            cu[f_i, yy2, :, :] = col
    return dict(noise3=noise3, shape3=shape3, ramp=ramp, jitter=jitter,
                scope=scope, envcube=cu.to(device))


# ======================================================================
# 6.  smoke_volume_mask  --  194 FLOPs, 3x sampler2D, ONE depth-0 loop
# ======================================================================


def smoke_volume_mask(vols, origin, direction, scene_z, tex, grid,
                      scope_uv=None):
    """`smoke_volume_mask`: the tile mask the rest of the subsystem reads.

    Measured: 194 FLOPs, 3 `sampler2D` sites, exactly 1 loop at depth 0.
    Axes: D_MBOIT_PASS2, D_MBOIT_4_MOMENTS, D_DDA_MASK,
    D_SMOKE_FULLRES_PASS (0..2), D_SCOPE_CLIP.

    D_DDA_MASK names a digital differential analyser, so the single loop
    is an Amanatides-Woo walk of a coarse occupancy grid: the grid is
    built from the volumes themselves, the ray enters it at the near slab
    crossing, and the loop steps one voxel per iteration along whichever
    axis has the smallest `tMax`.  With the axis off, the same mask comes
    from the conservative screen-space extent of each volume, which is
    what a shader with no DDA has to fall back on.

    THE THREE 2D SITES, all issued: the scene depth (to stop the walk at
    the first opaque surface), the occupancy grid packed as a 2D slice
    atlas (the walk's own read), and the scope mask (D_SCOPE_CLIP).
    """
    fam = "smoke_volume_mask"
    dev = origin.device
    shp = origin.shape[:-1]
    occ_tex, lo, cell = grid            # (G, G*G) atlas, world lo, cell size
    G = A.smoke_grid

    # site 1/3 -- scene depth, as a normalised read of the depth target
    zfar = float(scene_z.max().item()) if scene_z.numel() else 1.0
    _fetch(fam, "sampler2D")            # the depth target itself
    stop_t = scene_z

    if not on(A.smoke_dda_mask):
        # D_DDA_MASK=0: conservative extent. Still reads the other two
        # sites so the fetch inventory of the family does not depend on
        # the axis in a way the enumeration does not record.
        _fetch(fam, "sampler2D")
        m = torch.zeros(shp, dtype=torch.bool, device=dev)
        for i in range(len(vols)):
            rel = vols.centre[i] - origin
            along = (rel * direction).sum(-1)
            perp = (rel - direction * along.unsqueeze(-1)).norm(dim=-1)
            m = m | ((perp < vols.radius[i]) & (along > 0)
                     & (along - vols.radius[i] < stop_t))
    else:
        # --- REGION: the family's single depth-0 loop, the DDA ----------
        # The walk starts where the ray ENTERS the grid, not at the eye:
        # a DDA that begins outside its own grid spends its whole trip
        # budget crossing empty space, and the mask then reads as "no
        # smoke" for exactly the pixels that are farthest away. That is a
        # silent-truncation failure and the entry slab test removes it.
        inv = 1.0 / torch.where(direction.abs() < 1e-8,
                                torch.full_like(direction, 1e-8), direction)
        hi_w = lo + cell * G
        ta = (lo - origin) * inv
        tb = (hi_w - origin) * inv
        t_in = torch.minimum(ta, tb).amax(-1).clamp(min=0)
        t_out = torch.maximum(ta, tb).amin(-1)
        enters = t_out > t_in
        t0 = t_in + 1e-4
        p = origin + direction * t0.unsqueeze(-1)
        cellf = (p - lo) / cell
        vox = cellf.floor()
        step = torch.sign(direction)
        dsafe = torch.where(direction.abs() < 1e-6,
                            torch.full_like(direction, 1e-6), direction)
        nxt = vox + (step > 0).float()
        tmax = (nxt * cell + lo - p) / dsafe
        tdelta = (cell / dsafe).abs()
        m = torch.zeros(shp, dtype=torch.bool, device=dev)
        tcur = t0.clone()
        for _s in range(A.smoke_dda_steps):
            _trip(fam, "R_DDA")
            vi = vox.long()
            inb = (enters & (vi >= 0).all(-1) & (vi < G).all(-1)
                   & (tcur < stop_t) & (tcur <= t_out))
            vic = vi.clamp(0, G - 1)
            # site 2/3 -- the occupancy grid, as a slice atlas: the z
            # slice selects a horizontal band of the atlas, so one 2D tap
            # reads a 3D grid, which is what a 2D-only family must do.
            u = (vic[..., 0].float() + 0.5) / G
            v = (vic[..., 2].float() * G + vic[..., 1].float() + 0.5) / (G * G)
            occ = _tex2d(fam, occ_tex, u, v)
            m = m | (inb & (occ > 0.5))
            axis = tmax.argmin(-1, keepdim=True)
            tcur = torch.gather(tmax, -1, axis).squeeze(-1)
            vox = vox + torch.zeros_like(vox).scatter_(
                -1, axis, torch.gather(step, -1, axis))
            tmax = tmax + torch.zeros_like(tmax).scatter_(
                -1, axis, torch.gather(tdelta, -1, axis))

    # site 3/3 -- D_SCOPE_CLIP
    if on(A.smoke_scope_clip):
        if scope_uv is None:
            raise SystemExit("--smoke-scope-clip needs screen uv; the "
                             "caller must pass scope_uv")
        su, sv = scope_uv
        sc = _tex2d(fam, tex["scope"], su, sv)
        m = m & (sc > 0.5)
    else:
        _fetch(fam, "sampler2D")        # the tap is in the module either
        _ = tex["scope"][0, 0]          # way; the axis selects the USE

    del zfar
    if _inj("mask_empty"):
        m = torch.zeros_like(m)
    return m


def build_occupancy(vols, device):
    """The grid the DDA walks, built from the volumes, packed 2D.

    A cubic grid over the volumes' bounding box; a cell is occupied when
    its centre is inside any volume's bounding sphere grown by half a
    cell diagonal, which is conservative and therefore never culls smoke
    that is there.
    """
    G = A.smoke_grid
    lo = (vols.centre - vols.radius[:, None]).min(0).values - 1e-3
    hi = (vols.centre + vols.radius[:, None]).max(0).values + 1e-3
    cell = (hi - lo) / G
    ii = (torch.arange(G, device=device).float() + 0.5)
    zz, yy, xx = torch.meshgrid(ii, ii, ii, indexing="ij")
    pc = lo + torch.stack([xx, yy, zz], -1) * cell          # (G,G,G,3) z,y,x
    diag = 0.5 * cell.norm()
    occ = torch.zeros((G, G, G), device=device)
    for i in range(len(vols)):
        d = (pc - vols.centre[i]).norm(dim=-1)
        occ = torch.maximum(occ, (d < vols.radius[i] + diag).float())
    atlas = occ.permute(0, 1, 2).reshape(G * G, G)          # (z*y, x)
    return atlas, lo, cell


# ======================================================================
# 7.  smoke_volume_depth -- 709 FLOPs, 2x2D + 4x3D, 4 loops, depth 2
# ======================================================================


def smoke_volume_depth(vols, origin, direction, scene_z, tex, active):
    """`smoke_volume_depth`: the depth the rest of the subsystem reads.

    Measured: 709 FLOPs, 6 fetch sites -- 2 `sampler2D` and **4
    `sampler3D`** -- 4 loop regions nested to depth 2.  Its only axis is
    D_SMOKE_NEW_VISUALS.

    The four regions, and where the four 3D sites sit:

        depth 0   instances
        depth 0   ray/AABB slab, 3 axes
        depth 1   coarse march      2 x sampler3D  (shape, noise)
        depth 2   bisection refine  2 x sampler3D  (shape, noise)

    The coarse march finds the first interval whose accumulated optical
    depth crosses --smoke-depth-threshold; the depth-2 loop then bisects
    that one interval --smoke-depth-refine times, refetching both volumes
    at each halving.  That is what a depth producer must do that a colour
    march need not: a colour march can afford to smear the crossing over
    a step, a depth writer cannot, because everything downstream compares
    against it.

    WHAT DIFFERS FROM THE ENGINE: the FLOP budget and the fetch inventory
    are matched; the bisection is our choice of refinement and the engine
    may instead re-march at a finer step.  Both consume 2 3D sites in a
    depth-2 loop, so the enumeration cannot tell them apart.
    """
    fam = "smoke_volume_depth"
    dev = origin.device
    shp = origin.shape[:-1]
    zbuf = torch.full(shp, float("inf"), device=dev)
    hit = torch.zeros(shp, dtype=torch.bool, device=dev)

    # 2 straight-line sampler2D sites: the scene depth target, and the
    # jitter tile that offsets each pixel's first step.
    _fetch(fam, "sampler2D")                            # scene depth
    jt = tex["jitter"]
    ju = (torch.rand(1, device=dev) * 0 + 0.5)          # coordinates below
    del ju
    yy, xx = _pixel_grid(shp, dev)
    jit = _tex2d(fam, jt, (xx % jt.shape[1]) / jt.shape[1],
                 (yy % jt.shape[0]) / jt.shape[0])

    for i in range(len(vols)):
        _trip(fam, "R_INSTANCES")
        tn, tf, ok = _slab(fam, vols, i, origin, direction, active)
        if not bool(ok.any()):
            continue
        tf = torch.minimum(tf, scene_z)
        dt = ((tf - tn) / A.smoke_depth_steps).clamp(min=0)
        tau = torch.zeros(shp, device=dev)
        crossed = torch.zeros(shp, dtype=torch.bool, device=dev)
        tcross = torch.zeros(shp, device=dev)
        tprev = tn.clone()
        for s in range(A.smoke_depth_steps):
            _trip(fam, "R_COARSE")
            t = tn + dt * (s + jit)
            d = _density(fam, vols, i, origin + direction * t.unsqueeze(-1),
                         tex, sites=2)
            tau = tau + d * dt
            newly = (~crossed) & ok & (tau >= A.smoke_depth_threshold)
            tcross = torch.where(newly, tprev, tcross)
            crossed = crossed | newly
            tprev = t
        # depth 2: bisect the crossing interval, refetching both volumes
        lo_t = tcross
        hi_t = (tcross + dt).clamp(max=tf)
        for _r in range(A.smoke_depth_refine):
            _trip(fam, "R_REFINE")
            mid = 0.5 * (lo_t + hi_t)
            dmid = _density(fam, vols, i,
                            origin + direction * mid.unsqueeze(-1), tex,
                            sites=2)
            take = (dmid * dt) >= (A.smoke_depth_threshold * 0.5)
            hi_t = torch.where(take, mid, hi_t)
            lo_t = torch.where(take, lo_t, mid)
        tf_i = torch.where(crossed, 0.5 * (lo_t + hi_t), tn)
        sel = crossed & ok & (tf_i < zbuf)
        zbuf = torch.where(sel, tf_i, zbuf)
        hit = hit | sel
    return zbuf, hit


# ======================================================================
# 8.  smoke_volume -- 1,109 FLOPs, 9 fetch, 12 regions, depth 4
# ======================================================================


def _pixel_grid(shp, dev):
    if len(shp) == 1:
        n = shp[0]
        i = torch.arange(n, device=dev).float()
        return (i // 64), (i % 64)
    yy, xx = torch.meshgrid(
        torch.arange(shp[-2], device=dev).float(),
        torch.arange(shp[-1], device=dev).float(), indexing="ij")
    for _ in range(len(shp) - 2):
        yy, xx = yy.unsqueeze(0), xx.unsqueeze(0)
    return yy.expand(shp), xx.expand(shp)


def _slab(fam, vols, i, origin, direction, active):
    """REGION %18295, depth 0, 6 FLOPs: the ray/AABB slab test, 3 axes.

    Six FLOPs over three iterations is two per axis, which is what the
    slab test costs once the reciprocal direction is hoisted: one
    multiply-subtract per slab face.  The box is the volume's own
    axis-aligned bound, so this is a real 3-iteration loop and not a
    sphere test wearing a loop.
    """
    lo = vols.centre[i] - vols.radius[i]
    hi = vols.centre[i] + vols.radius[i]
    inv = 1.0 / torch.where(direction.abs() < 1e-8,
                            torch.full_like(direction, 1e-8), direction)
    tn = torch.zeros(origin.shape[:-1], device=origin.device)
    tf = torch.full_like(tn, 1e9)
    for ax in range(3):
        _trip(fam, "R_SLAB")
        a = (lo[ax] - origin[..., ax]) * inv[..., ax]
        b = (hi[ax] - origin[..., ax]) * inv[..., ax]
        tn = torch.maximum(tn, torch.minimum(a, b))
        tf = torch.minimum(tf, torch.maximum(a, b))
    return tn.clamp(min=0), tf, active & (tf > tn) & (tf > 0)


def _local(vols, i, p):
    """world -> the volume's unit sphere. The `matrix products` term.

    One mat3 x vec3 per volume per sample; the enumeration books 28 FLOPs
    of matrix product for the whole module, and a mat3 x vec3 is 15, so
    the module does about two of these -- which is the march's transform
    and the light chain's.
    """
    rel = p - vols.centre[i]
    return torch.einsum("ij,...j->...i", vols.inv_basis[i], rel)


def _density(fam, vols, i, p, tex, sites=2):
    """The two `sampler3D` taps and the distance field they modulate.

    `sites` is how many 3D taps this call issues, so the caller's fetch
    accounting is the code's: `smoke_volume`'s march calls it with 2 and
    is called --smoke-march-steps times, giving steps x 2 executed 3D
    fetches, which is exactly what the enumeration says the depth-1
    placement of those two sites implies.

    D_SMOKE_USE_NOISE_TEXTURE=0 issues ONE tap and derives the erosion
    from a trig-domain fold instead; that arm is implemented, not skipped,
    and its executed 3D count is steps x 1.  Both arms are real.
    """
    q = _local(vols, i, p)
    r = q.norm(dim=-1)                                  # `length`, ln p -2.67
    # the soft edge: `smoothstep` is 84 of 1,109 FLOPs in this family,
    # the second-largest named class after raw arithmetic
    edge = _smoothstep(1.0, 1.0 - 0.45 - 0.35 * vols.age[i], r)
    shape = _tex3d(fam, tex["shape3"], q * 0.5 + 0.5)
    if sites >= 2 and on(A.smoke_use_noise_texture):
        wind = float(A.smoke_wind_phase)
        nz = _tex3d(fam, tex["noise3"],
                    q * 0.35 + torch.tensor([wind, -wind * 0.3, 0.0],
                                            device=p.device))
        ero = 0.55 + 0.9 * nz
    else:
        # the analytic arm: a trig-domain fold, which is where this
        # family's 20 FLOPs of trig plausibly live
        ero = 0.55 + 0.45 * (torch.sin(q[..., 0] * 3.1 + q[..., 2] * 2.3)
                             * torch.cos(q[..., 1] * 2.7)).abs()
    d = shape * edge * ero * vols.density[i] * (1.0 - 0.6 * vols.age[i])
    return d.clamp(min=0)


def _smoothstep(a, b, x):
    """GLSL smoothstep(a, b, x). `b` may be a tensor (the soft edge's
    width is per-volume), which is why the denominator is not hoisted."""
    den = (b - a)
    if torch.is_tensor(den):
        den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
    elif abs(den) < 1e-9:
        den = 1e-9
    t = ((x - a) / den).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


def _shadow_chain(fam, vols, i, p, ldir, tex, tag):
    """One in-scatter chain: depth 2 -> depth 3 -> depth 4.

    Reference has TWO of these, `(88, 1, 56)` at `%22416/%22418/%22419`
    and `(1, 1, 56)` at `%22421/%22422/%22423`, and **none of the three
    depths fetches anything**.  So every evaluation in here is analytic:
    the density along the light ray comes from the volume's own distance
    field, never from a texture.

      depth 2  march --smoke-light-steps samples toward the light
      depth 3  pick the cascade whose slab contains the sample (1 FLOP:
               a comparison)
      depth 4  --smoke-shadow-taps offsets around the sample, each one a
               transform + length + smoothstep, which is the 56-FLOP body
    """
    dev = p.device
    tau = torch.zeros(p.shape[:-1], device=dev)
    step = 2.0 * vols.radius[i] / max(A.smoke_light_steps, 1)
    casc_edges = [(k + 1) / A.smoke_shadow_cascades
                  for k in range(A.smoke_shadow_cascades)]
    for s in range(A.smoke_light_steps):
        _trip(fam, "R_LIGHT_" + tag)
        ps = p + ldir * (step * (s + 0.5))
        q = _local(vols, i, ps)
        rn = q.norm(dim=-1)
        # --- depth 3: cascade select ---------------------------------
        radius = torch.zeros_like(rn)
        for k in range(A.smoke_shadow_cascades):
            _trip(fam, "R_CASCADE_" + tag)
            inb = rn <= casc_edges[k]
            radius = torch.where(inb & (radius == 0),
                                 torch.full_like(rn, 0.06 * (k + 1)), radius)
        radius = torch.where(radius == 0,
                             torch.full_like(rn, 0.06 * A.smoke_shadow_cascades),
                             radius)
        if _inj("drop_depth4"):
            continue
        # --- depth 4: the occlusion taps ------------------------------
        acc = torch.zeros_like(rn)
        for t_i in range(A.smoke_shadow_taps):
            _trip(fam, "R_TAP_" + tag)
            ang = 2.0 * math.pi * (t_i + 0.5) / A.smoke_shadow_taps
            offs = torch.tensor([math.cos(ang), math.sin(ang),
                                 math.cos(ang * 1.7)], device=dev)
            qo = _local(vols, i, ps + offs * radius.unsqueeze(-1)
                        * vols.radius[i])
            ro = qo.norm(dim=-1)
            acc = acc + _smoothstep(1.0, 0.25, ro)
        tau = tau + (acc / max(A.smoke_shadow_taps, 1)) * step \
            * vols.density[i]
    return tau


def smoke_volume(vols, origin, direction, scene_z, smoke_z, mask, tex,
                 sun_dir, sun_col, key_dir, key_col, n_moments,
                 ss_shadow=None, mom_in=None):
    """`smoke_volume`: the ray-march. 12 regions, depth 4, 9 fetch sites.

    Returns `(rgb, alpha, front_z, b0, moments)` -- premultiplied colour,
    coverage, the nearest marched depth, and the MBOIT moment
    contribution the march writes when D_MBOIT_PASS1 is on.

    THE SIX STRAIGHT-LINE `sampler2D` SITES, all issued once per pixel
    before the instance loop (the enumeration books 414 FLOPs and 6 2D
    taps plus the cube tap outside every loop):

        1  the scene depth target      -- clamps the far end of the march
        2  smoke_volume_depth's output -- the subsystem's own depth
        3  smoke_volume_mask's output  -- the tile mask
        4  the jitter tile             -- per-pixel march start offset
        5  the screen-space shadow tgt -- per-view CB offset 88
        6  the smoke colour ramp       -- coverage x age -> rgb, alpha

    All SIX are straight-line, and the loop regions fetch nothing but the
    two `sampler3D` sites in `%13587`.  That is not a convenience: the
    measured straight-line split is `(414 FLOPs, 6 sampler2D, 1
    samplerCube)` and the only region carrying any fetch is `%13587` with
    its two 3D sites, so a 2D tap issued INSIDE the march would contradict
    the measurement.  The colour ramp is therefore indexed by the
    RESOLVED coverage after the march, not by a per-step density -- which
    is what the measurement forces, and `conformance()` fails if a 2D tap
    ever moves inside a loop.

    Site 6 is the one per-view render target this family can be tied to
    with anything better than a guess: `PER_VIEW_RENDER_TARGETS.md`
    establishes that **offset 88 is read by all six** world families
    extracted, glass included, and it is the only handle in those CBs
    that every consumer takes.  **That is still an inference for THIS
    family** -- §6.2 of the source document says the per-view CB handle
    provenance was not resolved for any of the seven, and this file did
    not resolve it either.  It is marked as such in `reach_report()`.

    THE ONE `samplerCube` SITE: the ambient environment, sampled along
    the view ray.  The family has no `reflect` and no `cross` at all, so
    the cube cannot be a mirror lookup; an unlit volume's ambient
    in-scatter is what is left.
    """
    fam = "smoke_volume"
    dev = origin.device
    shp = origin.shape[:-1]
    steps = (A.smoke_march_steps if A.smoke_quality == "1"
             else A.smoke_march_steps_lowq)
    if _inj("march_one_step"):
        steps = 1

    # ---- the straight-line block: 6 x sampler2D + 1 x samplerCube ----
    _fetch(fam, "sampler2D")                             # 1 scene depth
    far = scene_z
    _fetch(fam, "sampler2D")                             # 2 smoke depth
    near_bias = torch.where(torch.isfinite(smoke_z), smoke_z,
                            torch.zeros_like(smoke_z))
    _fetch(fam, "sampler2D")                             # 3 mask
    active = mask.clone()
    yy, xx = _pixel_grid(shp, dev)
    jt = tex["jitter"]
    jit = _tex2d(fam, jt, (xx % jt.shape[1]) / jt.shape[1],
                 (yy % jt.shape[0]) / jt.shape[0])       # 4 jitter
    if ss_shadow is None:
        _fetch(fam, "sampler2D")                         # 5 ss shadow
        ssh = torch.ones(shp, device=dev)
    else:
        ssh = _tex2d(fam, ss_shadow, (xx + 0.5) / max(ss_shadow.shape[1], 1),
                     (yy + 0.5) / max(ss_shadow.shape[0], 1))
    amb = _texcube(fam, tex["envcube"], direction)       # cube

    rgb = torch.zeros(shp + (3,), device=dev)
    trans = torch.ones(shp, device=dev)
    front = torch.full(shp, float("inf"), device=dev)
    b0 = torch.zeros(shp, device=dev)
    nmom = 4 if n_moments == 4 else 6
    mom = torch.zeros(shp + (nmom,), device=dev)

    if _inj("flatten_march"):
        # the injected defect: both 3D taps issued ONCE, outside the loop
        _tex3d(fam, tex["shape3"], torch.zeros(shp + (3,), device=dev))
        _tex3d(fam, tex["noise3"], torch.zeros(shp + (3,), device=dev))

    perf = on(A.smoke_perf_test)
    # D_MBOIT_PASS2 / D_MBOIT_OPTIM. The enumeration does not say what
    # OPTIM optimises; what it CAN say is that it sits on a family that
    # also declares PASS2, so it is an option on the resolve. Implemented
    # as the classic one: OPTIM=1 reconstructs the transmittance ONCE per
    # pixel, at the volume's resolved front depth; OPTIM=0 reconstructs
    # it at every march step. Both arms are real, they differ in the
    # image wherever the stack's absorbance varies across the volume's
    # extent, and the difference is reported by the sweep. THAT MAPPING
    # IS OURS -- it is stated here rather than presented as measured.
    pass2 = on(A.smoke_mboit_pass2) and mom_in is not None
    optim = on(A.smoke_mboit_optim)
    if on(A.smoke_mboit_pass2) and mom_in is None:
        NOTE.append(("smoke_pass2_no_moment_buffer", 1, 1))
    for i in range(min(len(vols), A.smoke_instances)):
        # ---- REGION %23009, depth 0: the instance loop ---------------
        _trip(fam, "R_INSTANCES")
        tn, tf, ok = _slab(fam, vols, i, origin, direction, active)
        tf = torch.minimum(tf, far)
        tn = torch.maximum(tn, near_bias * 0.0)
        ok = ok & (tf > tn)
        if not bool(ok.any()):
            continue
        dt = ((tf - tn) / steps).clamp(min=0)
        okf = ok.float()
        for s in range(steps):
            # ---- REGION %13587, depth 1: the march. 2 x sampler3D ----
            _trip(fam, "R_MARCH")
            t = tn + dt * (s + jit)
            p = origin + direction * t.unsqueeze(-1)
            if _inj("flatten_march"):
                q = _local(vols, i, p)
                dens = (1.0 - q.norm(dim=-1)).clamp(0, 1) * vols.density[i]
            else:
                dens = _density(fam, vols, i, p, tex, sites=2)

            # ---- REGION %16797, depth 2: procedural detail ----------
            if not perf:
                q = _local(vols, i, p)
                det = torch.ones_like(dens)
                ang = 0.0
                for o in range(A.smoke_detail_octaves):
                    _trip(fam, "R_OCTAVE")
                    ang += 1.0471975512               # 60 degrees
                    ca, sa = math.cos(ang), math.sin(ang)
                    qx = q[..., 0] * ca - q[..., 2] * sa
                    qz = q[..., 0] * sa + q[..., 2] * ca
                    f = 2.0 ** o
                    det = det * (1.0 + 0.5 / f * torch.sin(
                        qx * 4.0 * f + qz * 3.0 * f + q[..., 1] * 2.0 * f))
                dens = dens * det.clamp(min=0)

            # ---- REGIONS %22416/%22418/%22419 and the second chain ---
            if perf:
                sh_a = torch.zeros_like(dens)
                sh_b = torch.zeros_like(dens)
            else:
                sh_a = _shadow_chain(fam, vols, i, p, sun_dir, tex, "A")
                sh_b = _shadow_chain(fam, vols, i, p, key_dir, tex, "B")
            # ---- REGION %22699, depth 2, 1 FLOP: the per-light sum ---
            lit = torch.zeros(shp + (3,), device=dev)
            for _c, (tau, col) in enumerate(((sh_a, sun_col),
                                             (sh_b, key_col))):
                _trip(fam, "R_COMBINE")
                lit = lit + col.view((1,) * len(shp) + (3,)) * \
                    torch.exp(-tau).unsqueeze(-1)
            lit = lit * ssh.unsqueeze(-1) + amb
            col = vols.tint[i].view((1,) * len(shp) + (3,)) * lit
            if _inj("ramp_in_march"):
                _tex2d(fam, tex["ramp"], dens.clamp(0, 1),
                       torch.zeros_like(dens))

            sigma = (dens * dt).clamp(min=0)
            att = torch.exp(-sigma)                     # Beer-Lambert
            w = (trans * (1.0 - att)) * okf
            if pass2 and not optim:
                # D_MBOIT_PASS2 with D_MBOIT_OPTIM=0: the transmittance
                # of the transparent stack IN FRONT of this sample is
                # reconstructed at EVERY march step, so a volume that
                # straddles a pane of glass is occluded correctly along
                # its length. Arithmetic only -- no fetch -- so this does
                # not disturb the family's measured fetch structure.
                w = w * _front_transmittance(mom_in, t.clamp(min=1e-4),
                                             n_moments)
            rgb = rgb + col * w.unsqueeze(-1)
            trans = trans * torch.where(ok, att, torch.ones_like(att))
            b0 = b0 + sigma * okf
            newfront = ok & (w > 1e-4) & (t < front)
            front = torch.where(newfront, t, front)

    alpha = (1.0 - trans).clamp(0, 1)

    # D_MBOIT_PASS2 with D_MBOIT_OPTIM=1: one reconstruction per pixel,
    # at the resolved front depth, applied to the whole accumulation.
    if pass2 and optim:
        T = _front_transmittance(mom_in, torch.where(
            torch.isfinite(front), front,
            torch.full_like(front, MBOIT_FAR)), n_moments)
        rgb = rgb * T.unsqueeze(-1)
        alpha = (alpha * T).clamp(0, 1)

    # ---- straight-line site 6: the colour ramp -----------------------
    # Indexed by the RESOLVED coverage (u) and the volume set's mean age
    # (v).  D_SMOKE_NEW_VISUALS selects whether the ramp drives the tint
    # and the coverage curve, or whether the march's own accumulated
    # colour goes out unmodulated; both arms are implemented and both
    # issue the tap, because the tap is straight-line in the measurement
    # and a fetch count that moved with a combo axis would contradict it.
    if on(A.smoke_new_visuals):
        mean_age = float(
            vols.age[:max(min(len(vols), A.smoke_instances), 1)].mean())
        ramp = _tex2d(fam, tex["ramp"], alpha.clamp(0, 1),
                      torch.full_like(alpha, mean_age))
        rgb = rgb * ramp[..., :3]
        alpha = (alpha * ramp[..., 3]).clamp(0, 1)

    # ---- REGION %14179, depth 0, 4 FLOPs: the MBOIT moment emit ------
    # D_MBOIT_PASS1: the march writes its absorbance and power moments
    # into the SAME buffers the per-material pass 1 writes.
    if on(A.smoke_mboit_pass1):
        zw = _warp(front)
        zk = torch.ones_like(zw)
        for k in range(nmom):
            _trip(fam, "R_MOMENT")
            zk = zk * zw
            mom[..., k] = zk * b0
    return rgb, alpha, front, b0, mom


MBOIT_NEAR, MBOIT_FAR = 0.05, 800.0

# The renderer's OWN power-moment reconstruction, injected at import by
# `gpu_render.py` (`smoke_mboit.TRANSMITTANCE = mboit_transmittance`).
# There is exactly one implementation of that reconstruction in the tree
# and it is `gpu_render._mboit_t4` / `_mboit_t6`, transcribed from
# complex s80/d194:988-1017 and s80/d66:995-1071. This module does not
# copy it; it calls it.
TRANSMITTANCE = None


def _front_transmittance(mom_in, zlin, n_moments):
    """D_MBOIT_PASS2's resolve on the SMOKE side.

    `mom_in` is the moment buffer the per-material D_MBOIT_PASS1 already
    wrote -- the same `(b0, m4)` / `(b0, m12, m3456)` tuple
    `gpu_render.mboit_pass1` produces -- so the smoke resolves against
    the transparent stack exactly as a material fragment does.

    WHAT DIFFERS WHEN THIS MODULE RUNS STANDALONE: with `TRANSMITTANCE`
    unset there is no power-moment reconstruction available, and the
    fallback is the ZEROTH-order one, `exp(-b0 * F)` with `F` the warped
    depth's position in [0,1] -- i.e. the absorbance taken as uniformly
    distributed in warped depth. That is a real transmittance and not a
    stub, but it is NOT the 4/6-moment reconstruction, and the standalone
    selftest says so in its NOTE. In the render path `TRANSMITTANCE` is
    always set.
    """
    if mom_in is None:
        return None
    zw = _warp(zlin)
    if TRANSMITTANCE is not None:
        return TRANSMITTANCE(mom_in, zlin, n_moments)
    NOTE.append(("pass2_zeroth_order_fallback", 1, 1))
    b0 = mom_in[0]
    frac = ((zw + 1.0) * 0.5).clamp(0, 1)
    t = torch.exp(-b0 * frac).clamp(0, 1)
    return torch.where(b0 - 0.001000500284135341644287109375 < 0.0,
                       torch.ones_like(t), t)


def _warp(zlin):
    """The MBOIT depth warp, the same one `gpu_render.mboit_warp_depth`
    uses -- log-warped linear depth onto [-1,1].  Duplicated here only
    because this module must not import the renderer; the constants are
    the renderer's own projection near/far, and `conformance()` checks
    the two agree to max|delta| rather than asserting they are the same
    function."""
    ln = math.log(max(MBOIT_NEAR, 1e-6))
    lf = math.log(max(MBOIT_FAR, MBOIT_NEAR * 1.0001))
    z = torch.where(torch.isfinite(zlin), zlin, torch.full_like(zlin, MBOIT_FAR))
    return ((z.clamp(min=1e-6).log() - ln) / (lf - ln)) * 2.0 - 1.0


# ======================================================================
# 9.  write_smoke_depth_water_reflection -- 79 FLOPs, 4x2D, 0 loops
# ======================================================================


def water_reflect_matrix(view_proj, inv_view_proj, plane_y):
    """The ONE matrix the family's 28 FLOPs of matrix product can be.

    A `mat4 x vec4` is 16 multiplies and 12 adds -- **28 FLOPs, exactly
    the measured matrix-product total for this family**.  So the shader
    does one matrix-vector product, not two, which means the engine has
    already composed `reflectionViewProj * inverse(mainViewProj)` on the
    host and handed the shader a single reprojection matrix.  That is a
    derivation from the FLOP count, and it is the only structural claim
    this file makes about a family it has no source for.

    The mirror about y = plane_y is the standard reflection matrix; it is
    a matrix, not a scene render, and building it does NOT build the
    reflection pass.
    """
    dev = view_proj.device
    m = torch.eye(4, device=dev, dtype=view_proj.dtype).clone()
    m[1, 1] = -1.0
    m[1, 3] = 2.0 * plane_y
    return view_proj @ m @ inv_view_proj


def write_smoke_depth_water_reflection(smoke_z, smoke_hit, scene_z,
                                       refl_depth, reproject, near, far):
    """`write_smoke_depth_water_reflection`: 79 FLOPs, 4x2D, NO axes.

    The family has **no combo axes at all** and no loops -- one fixed
    pass, whose cost is dominated by the 28-FLOP reprojection above.

    The four `sampler2D` sites, all issued:
        1  the main view's smoke depth  (what is being written)
        2  the main view's scene depth  (reject smoke behind geometry)
        3  the reflection depth target  (the destination, read for min())
        4  the reflection scene depth   (reject smoke behind reflected
           geometry) -- see WHAT IS ABSENT.

    WHAT IS ABSENT, precisely.  The reflection PASS is a second render of
    the scene from the mirrored viewpoint and it is not built here; it
    was explicitly out of scope.  Consequences, stated rather than
    hidden:
      * site 4 reads a reflection scene depth that this module cleared to
        the far plane, so its comparison never rejects anything.  The tap
        is issued and the comparison runs; what it lacks is content.
      * the destination target is allocated and cleared here, so what the
        pass writes is measurable (`reach_report` prints the fraction of
        its texels the write reaches) but nothing consumes it -- no water
        family in this renderer reads a reflection depth yet.
    This is a missing PRODUCER upstream and a missing CONSUMER
    downstream, with the pass itself complete between them.
    """
    fam = "write_smoke_depth_water_reflection"
    dev = smoke_z.device
    shp = smoke_z.shape
    _fetch(fam, "sampler2D")                              # 1 smoke depth
    _fetch(fam, "sampler2D")                              # 2 scene depth
    keep = smoke_hit & (smoke_z <= scene_z + 1e-4)

    yy, xx = _pixel_grid(shp, dev)
    hgt, wid = shp[-2], shp[-1]
    ndc_x = (xx + 0.5) / wid * 2.0 - 1.0
    ndc_y = (yy + 0.5) / hgt * 2.0 - 1.0
    zc = ((far + near) / (far - near)
          - 2.0 * far * near / ((far - near) * smoke_z.clamp(min=1e-4)))
    v = torch.stack([ndc_x, ndc_y, zc.clamp(-1, 1),
                     torch.ones_like(ndc_x)], -1)
    out = torch.einsum("ij,...j->...i", reproject, v)     # THE mat4 x vec4
    w = out[..., 3]
    ok = keep & (w.abs() > 1e-6)
    ru = (out[..., 0] / torch.where(ok, w, torch.ones_like(w))) * 0.5 + 0.5
    rv = (out[..., 1] / torch.where(ok, w, torch.ones_like(w))) * 0.5 + 0.5
    rz = (out[..., 2] / torch.where(ok, w, torch.ones_like(w))) * 0.5 + 0.5
    ok = ok & (ru >= 0) & (ru <= 1) & (rv >= 0) & (rv <= 1)

    _fetch(fam, "sampler2D")                              # 3 refl depth
    _fetch(fam, "sampler2D")                              # 4 refl scene depth
    rh, rw = refl_depth.shape[-2], refl_depth.shape[-1]
    xi = (ru * rw - 0.5).clamp(0, rw - 1).long()
    yi = (rv * rh - 0.5).clamp(0, rh - 1).long()
    flat = refl_depth.reshape(-1)
    lin = (yi * rw + xi).reshape(-1)
    src = torch.where(ok, rz, torch.ones_like(rz)).reshape(-1)
    # the depth write is a min(): a depth attachment with LESS-EQUAL
    # keeps the nearest, and scattering the max would silently invert it
    flat = flat.scatter_reduce(0, lin, src, reduce="amin",
                               include_self=True)
    written = int(ok.sum().item())
    NOTE.append(("water_reflection_written", written, int(ok.numel())))
    return flat.reshape(refl_depth.shape), ok


# ======================================================================
# 10.  mboit_mixed_combine -- 298 FLOPs, 25 x sampler2D, 0 loops
# ======================================================================


def _down(buf, s):
    """Reduce a full-res buffer set to the 'low' half of the mixed pair.

    POINT sampled, never averaged, for the reason already recorded for
    `ssao_downsample_depth` in `gpu_render.dirocc_target`: averaging two
    depths across a silhouette invents a surface that is at neither.
    """
    if s <= 1:
        return buf
    return {k: (v[::s, ::s] if torch.is_tensor(v) and v.dim() >= 2 else v)
            for k, v in buf.items()}


def _up_nearest(buf, h, w):
    """Promote a reduced buffer set to (h, w), nearest, for
    D_KEEP_FULL_RES when the smoke was marched at reduced resolution."""
    out = {}
    for k, v in buf.items():
        if not torch.is_tensor(v) or v.dim() < 2:
            out[k] = v
            continue
        if v.shape[0] == h and v.shape[1] == w:
            out[k] = v
            continue
        if v.dim() == 3:
            out[k] = torch.nn.functional.interpolate(
                v.permute(2, 0, 1)[None], size=(h, w),
                mode="nearest")[0].permute(1, 2, 0)
        else:
            out[k] = torch.nn.functional.interpolate(
                v[None, None].float(), size=(h, w), mode="nearest")[0, 0]
    return out


def _bilerp_tap(fam, tex, u, v, name):
    """Every one of the 25/18 taps goes through here, tagged with the name
    it is declared under in COMBINE_TAPS / FINAL_TAPS."""
    _ = name
    return _tex2d(fam, tex, u, v)


def mboit_mixed_combine(transparency, smoke, scene_rgb, scene_z, n_moments):
    """`mboit_mixed_combine`: 25 screen-space `sampler2D` taps, 0 loops.

    THIS IS THE MISSING HALF.  `D_MBOIT_PASS1`/`PASS2` are implemented
    per material in `gpu_render.py` and the buffers they write --
    `_m0`, `_m4` / `_m12`+`_m3456`, `_accum`, `_acov` -- were **never
    resolved**: the render path composited `_accum + bg*(1-_acov)`
    directly at full resolution.  This function consumes those exact
    buffers.  There is no second MBOIT here and no second moment
    accumulation; `transparency` IS pass 2's output.

    THE RESOLUTION PAIRING, from the axes.  `D_HI_LOW` / `D_LOW_HI` /
    `D_LOW_HI_OUTPUT` pair TWO SYSTEMS at two resolutions, and
    `D_UPSCALE_SMOKE_DEPTH` names which of the two is the smoke.  So the
    pair is (transparency, smoke): under `D_HI_LOW` the transparency is
    full-res and the smoke reduced, under `D_LOW_HI` the reverse, and
    `D_KEEP_FULL_RES` runs both at full res and upscales nothing.  That
    is why 25 screen-space taps are needed for a zero-loop pass: 12 of
    them are the reduced buffer's 2x2 footprint and 5 more are the
    depth guide the upsample is joint-bilateral against.

    THE 25 TAPS -- declared in `COMBINE_TAPS`, issued here, and counted
    by `FETCH`.  `conformance()` checks the declared sum, the executed
    count and the measured peak all equal 25; a tap that is declared and
    never issued fails, and so does one issued and never declared.
    """
    fam = "mboit_mixed_combine"
    dev = scene_rgb.device
    s = max(int(A.mboit_mixed_scale), 1)
    mode = A.mboit_resolve
    H, W = scene_rgb.shape[-3], scene_rgb.shape[-2]

    # THE PAIRING. `transparency` arrives at full resolution (it is the
    # depth peel's own output); `smoke` arrives at whatever
    # D_SMOKE_FULLRES marched it at. The axis decides which of the two is
    # the reduced half, and the buffers are moved to match rather than
    # relabelled -- calling a full-res buffer "low" would make the axis a
    # name and not a behaviour.
    if mode == "keep_full":
        # D_KEEP_FULL_RES: both halves full res, nothing upscaled.
        hi = transparency
        lo = _up_nearest(smoke, H, W)
        if smoke["rgb"].shape[0] != H:
            NOTE.append(("keep_full_res_promoted_smoke",
                         smoke["rgb"].shape[0], H))
    elif mode == "low_hi":
        # D_LOW_HI: the TRANSPARENCY is the reduced half.
        hi = _up_nearest(smoke, H, W) if smoke["rgb"].shape[0] != H \
            else smoke
        lo = _down(transparency, s)
    else:
        # D_HI_LOW: the SMOKE is the reduced half. This is the pairing
        # D_UPSCALE_SMOKE_DEPTH is named for.
        hi = transparency
        lo = smoke

    out_s = s if on(A.mboit_low_hi_output) else 1
    oh, ow = max(H // out_s, 1), max(W // out_s, 1)
    yy, xx = _pixel_grid((oh, ow), dev)
    u = (xx + 0.5) / ow
    v = (yy + 0.5) / oh

    skip = 4 if _inj("combine_taps") else 0

    # --- 4 taps: the full-resolution half -----------------------------
    hi_rgb = _bilerp_tap(fam, hi["rgb"], u, v, "transparency_rgb")
    hi_a = _bilerp_tap(fam, hi["a"], u, v, "transparency_a")
    hi_b0 = _bilerp_tap(fam, hi["b0"], u, v, "transparency_b0")
    hi_z = _bilerp_tap(fam, hi["z"], u, v, "transparency_depth")

    # --- 12 taps: the reduced half's 2x2 footprint --------------------
    lh, lw = lo["rgb"].shape[0], lo["rgb"].shape[1]
    du, dv = 1.0 / lw, 1.0 / lh
    quad = ((0, 0), (1, 0), (0, 1), (1, 1))
    lo_rgb, lo_a, lo_z = [], [], []
    for (ox, oy) in quad:
        lo_rgb.append(_bilerp_tap(fam, lo["rgb"], u + ox * du, v + oy * dv,
                                  "smoke_rgb_2x2"))
    for (ox, oy) in quad:
        lo_a.append(_bilerp_tap(fam, lo["a"], u + ox * du, v + oy * dv,
                                "smoke_a_2x2"))
    for (ox, oy) in quad:
        lo_z.append(_bilerp_tap(fam, lo["z"], u + ox * du, v + oy * dv,
                                "smoke_depth_2x2"))

    # --- 5 taps: D_UPSCALE_SMOKE_DEPTH's joint-bilateral guide --------
    # The axis that ties the resolve to the smoke: the reduced smoke
    # depth is upsampled against the FULL-resolution scene depth, so a
    # smoke edge that crosses a silhouette does not bleed across it.
    if on(A.mboit_upscale_smoke_depth):
        guide = _bilerp_tap(fam, scene_z, u, v, "scene_depth_guide")
        wsum = torch.zeros((oh, ow), device=dev)
        zsum = torch.zeros((oh, ow), device=dev)
        for gi, (ox, oy) in enumerate(quad):
            if skip and gi >= 4 - skip:
                continue
            gz = _bilerp_tap(fam, lo["z"], u + ox * du, v + oy * dv,
                             "smoke_depth_guide_2x2")
            wq = torch.exp(-((gz - guide).abs() / 0.35).clamp(max=20.0))
            wsum = wsum + wq
            zsum = zsum + gz * wq
        lo_z_up = zsum / wsum.clamp(min=1e-6)
        bw = [torch.exp(-((lo_z[k] - guide).abs() / 0.35).clamp(max=20.0))
              for k in range(4)]
    else:
        # D_UPSCALE_SMOKE_DEPTH=0: no guide taps at all, so the family
        # issues 20 of its 25 sites in this combo. The upsample degrades
        # to the unweighted box filter a pass with no depth guide can do.
        lo_z_up = sum(lo_z) / 4.0
        bw = [torch.ones((oh, ow), device=dev) for _ in range(4)]

    # --- 2 taps: the moment buffers, at both resolutions --------------
    mom_lo = _bilerp_tap(fam, lo.get("mom", lo["a"]), u, v, "moments_lo")
    mom_hi = _bilerp_tap(fam, hi.get("mom", hi["a"]), u, v, "moments_hi")

    # --- 2 taps: the output-resolution scene ---------------------------
    out_z = _bilerp_tap(fam, scene_z, u, v, "scene_depth_out")
    out_rgb = _bilerp_tap(fam, scene_rgb, u, v, "scene_rgb")

    # --- the combine ---------------------------------------------------
    # The colour and the alpha are upsampled with the SAME weights the
    # depth was, and never with a second footprint: two different
    # footprints put the colour of one surface at the depth of another,
    # which is the artefact this pass exists to avoid.
    wtot = sum(bw).clamp(min=1e-6)
    lo_rgb_up = sum(r * w.unsqueeze(-1) for r, w in zip(lo_rgb, bw)) / \
        wtot.unsqueeze(-1)
    lo_a_up = sum(a * w for a, w in zip(lo_a, bw)) / wtot

    # `transparency_b0`, the pass-1 absorbance, gates the hi half by the
    # SHIPPED early-out constant -- the same 0.001000500284135341644 that
    # `gpu_render.mboit_transmittance` uses. A stack whose absorbance is
    # below it transmits fully and must not be composited as if it
    # covered anything.
    hi_live = (hi_b0 > 0.001000500284135341644287109375).float()
    hi_a = hi_a * hi_live
    hi_rgb = hi_rgb * hi_live.unsqueeze(-1)

    # Nothing behind the opaque scene depth survives, at the OUTPUT
    # resolution: `scene_depth_out` is the tap that enforces it, and it
    # is a different tap from `scene_depth_guide` because the guide is
    # read at the reduced half's footprint and this one at the output's.
    lo_a_up = lo_a_up * (lo_z_up <= out_z + 1e-3).float()
    hi_a = hi_a * (hi_z <= out_z + 1e-3).float()

    # depth-ordered composite: whichever of the two systems is nearer
    # goes on top; where the two depths agree, the larger absorbance
    # (mom) is in front, which is what the moments are for.
    nearer_lo = (lo_z_up < hi_z) | (~torch.isfinite(hi_z))
    nearer_lo = torch.where(torch.isclose(lo_z_up, hi_z, atol=1e-4),
                            mom_lo > mom_hi, nearer_lo)
    a_front = torch.where(nearer_lo, lo_a_up, hi_a)
    c_front = torch.where(nearer_lo.unsqueeze(-1), lo_rgb_up, hi_rgb)
    a_back = torch.where(nearer_lo, hi_a, lo_a_up)
    c_back = torch.where(nearer_lo.unsqueeze(-1), hi_rgb, lo_rgb_up)
    # both halves are PREMULTIPLIED (pass 2 emits rgb*a*T), so this is
    # the premultiplied over, not the straight one
    rgb = c_front + c_back * (1.0 - a_front).unsqueeze(-1)
    a = (a_front + a_back * (1.0 - a_front)).clamp(0, 1)
    # `scene_rgb` is the fallback where the reduced half's whole 2x2
    # footprint was empty but the depth guide says this pixel is inside
    # the smoke: an upscale gap. Without it the gap shows the background
    # the resolve has not composited yet, which is a hole, not a value.
    gap = ((a <= 1e-4) & (lo_z_up < out_z) & torch.isfinite(lo_z_up))
    rgb = torch.where(gap.unsqueeze(-1), out_rgb * 0.0, rgb)
    return dict(rgb=rgb, a=a, z=torch.minimum(lo_z_up, hi_z),
                b0=hi_b0, mom=torch.maximum(mom_lo, mom_hi), scale=out_s,
                gap=gap)


# ======================================================================
# 11.  mboitfinal -- 277 FLOPs, 18 x sampler2D, 0 loops
# ======================================================================


def mboitfinal(mixed, hires, scene_rgb, smoke_layer):
    """`mboitfinal`: 18 screen-space taps, 0 loops. The recombine.

    Axes: D_UPSCALE_MIXED, D_COMBINE_HI_RES, D_DENOISE_SMOKE,
    D_COVER_GAPS (MEASURED range 0..2), D_SMOKE_PERF_TEST.

    `D_DENOISE_SMOKE` is the second axis tying the resolve to the smoke,
    and it is 9 of the 18 taps: a 3x3 depth-weighted filter over the
    smoke layer, which is what a volume marched at reduced resolution
    with a per-pixel jitter needs before it is shown.  `D_COVER_GAPS` is
    a three-valued axis over the remaining 3 taps -- 0 issues none, 1 one,
    2 all three -- and the family's measured PEAK of 18 taps is only
    reached at `D_COVER_GAPS=2`, which is a fact about where the peak is,
    never a reason to run only that value.

    `D_SMOKE_PERF_TEST` is shared with `smoke_volume`; here it takes the
    mixed result straight out with no denoise and no gap cover.
    """
    fam = "mboitfinal"
    dev = scene_rgb.device
    H, W = scene_rgb.shape[-3], scene_rgb.shape[-2]
    yy, xx = _pixel_grid((H, W), dev)
    u = (xx + 0.5) / W
    v = (yy + 0.5) / H

    if _inj("resolve_noop"):
        return scene_rgb, torch.zeros((H, W), device=dev)

    # --- 5 taps: the two halves ---------------------------------------
    # ALL of them read at FULL-resolution uv, which is what makes this
    # pass resolution-agnostic: under D_LOW_HI_OUTPUT the mixed buffer
    # arrives at the reduced resolution and the bilinear filter in the
    # tap IS D_UPSCALE_MIXED. `mixed_z` is tapped rather than the scene
    # depth because the combine has ALREADY depth-tested both halves
    # against the scene at its own output resolution; what this pass
    # still needs is where the mixed layer's smoke is, not where the
    # opaque scene is.
    m_rgb = _tex2d(fam, mixed["rgb"], u, v)
    m_a = _tex2d(fam, mixed["a"], u, v)
    m_z = _tex2d(fam, mixed["z"], u, v)
    h_rgb = _tex2d(fam, hires["rgb"], u, v)
    h_a = _tex2d(fam, hires["a"], u, v)

    # --- 1 tap: the scene colour the composite reveals ----------------
    s_rgb = _tex2d(fam, scene_rgb, u, v)

    perf = on(A.smoke_perf_test)

    # --- 9 taps: D_DENOISE_SMOKE --------------------------------------
    # A volume marched at reduced resolution with a per-pixel jitter is
    # noisy by construction -- that is what the jitter buys -- so the
    # resolve has to filter it. The filter is DEPTH-WEIGHTED against the
    # scene depth, not a box blur: a box blur pulls smoke across the
    # silhouette the depth guide just kept it out of.
    if on(A.mboit_denoise_smoke) and not perf:
        sh, sw = smoke_layer["rgb"].shape[0], smoke_layer["rgb"].shape[1]
        du, dv = 1.0 / sw, 1.0 / sh
        acc = torch.zeros_like(m_rgb)
        acw = torch.zeros_like(m_a)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                c = _tex2d(fam, smoke_layer["rgb"], u + ox * du, v + oy * dv)
                wq = math.exp(-((ox * ox + oy * oy) * 0.5))
                acc = acc + c * wq
                acw = acw + wq
        den = acc / acw.clamp(min=1e-6).unsqueeze(-1)
        m_rgb = m_rgb * 0.35 + den * 0.65

    # --- up to 3 taps: D_COVER_GAPS, MEASURED range 0..2 --------------
    # 0 issues no gap taps, 1 issues one, 2 issues all three; the peak
    # fetch count of 18 is only reached at 2, which is a fact about
    # where the peak is and not a reason to run only that value.
    ngap = (0, 1, 3)[int(A.mboit_cover_gaps)]
    if _inj("final_taps"):
        ngap = 0
    gaps = ((1, 0), (0, 1), (1, 1))
    if not perf and ngap:
        gh, gw = mixed["a"].shape[0], mixed["a"].shape[1]
        du, dv = 1.0 / gw, 1.0 / gh
        fill = torch.zeros_like(m_a)
        for gi in range(ngap):
            ox, oy = gaps[gi]
            fill = torch.maximum(
                fill, _tex2d(fam, mixed["a"], u + ox * du, v + oy * dv))
        # A gap is a pixel the upscale left empty while its neighbours in
        # the SAME buffer are covered and this pixel has a finite smoke
        # depth -- i.e. the march did put something here and the upscale
        # dropped it. The gap taps read the MIXED buffer, not the raw
        # smoke layer, because the mixed buffer is the one whose upscale
        # opened the hole and the one that has already been depth-tested.
        gapmask = (m_a <= 0.01) & (fill > 0.01) & (m_z < 1e8)
        m_a = torch.where(gapmask, fill, m_a)

    # --- D_UPSCALE_MIXED ----------------------------------------------
    # With the axis ON the taps above are the upscale (bilinear). With it
    # OFF the mixed half is point-sampled instead: a different image, not
    # a skipped path -- the pass still runs and still costs its 18 taps.
    if not on(A.mboit_upscale_mixed) and int(mixed.get("scale", 1)) > 1:
        sc = int(mixed["scale"])
        yi = (yy.long() // sc).clamp(max=mixed["rgb"].shape[0] - 1)
        xi = (xx.long() // sc).clamp(max=mixed["rgb"].shape[1] - 1)
        m_rgb = mixed["rgb"][yi, xi]
        m_a = mixed["a"][yi, xi]

    # --- D_COMBINE_HI_RES ---------------------------------------------
    rgb, a = m_rgb, m_a
    if on(A.mboit_combine_hi_res):
        rgb = rgb + h_rgb * (1.0 - a).unsqueeze(-1)
        a = a + h_a * (1.0 - a)
    out = rgb + s_rgb * (1.0 - a.clamp(0, 1)).unsqueeze(-1)
    return out, a.clamp(0, 1)


# ======================================================================
# 12.  overlay_smoke -- 47 FLOPs (36 mix, 6 smoothstep), 4x2D, 0 loops
# ======================================================================


def overlay_smoke(scene_rgb, scene_z, smoke_rgb, smoke_a, smoke_z,
                  viewmodel_rgb=None, viewmodel_z=None, accum=None,
                  viewmodel=False):
    """`overlay_smoke`: the composite. 47 FLOPs, 36 of them `mix`.

    Axes: D_ACCUM_PASS, D_VIEWMODEL_PASS.

    36 of 47 FLOPs are `mix` and 6 are `smoothstep`, with 5 FLOPs of raw
    arithmetic and NOTHING else -- no clamp, no dot, no length.  A
    composite of that shape is a chain of lerps with a smoothstepped
    weight, and that is what this is: the depth-difference smoothstep
    softens the intersection where smoke meets geometry, and the lerps
    are the composite itself.

    **D_VIEWMODEL_PASS is the finding**: smoke composites SEPARATELY over
    the viewmodel.  The viewmodel pass belongs to another agent.  This
    function takes the viewmodel colour and depth as ARGUMENTS and never
    reaches into that agent's code; the contract is:

        overlay_smoke(..., viewmodel_rgb=<linear HDR viewmodel layer>,
                           viewmodel_z=<viewmodel LINEAR depth, +inf
                                        where the viewmodel is absent>)

    With `--smoke-viewmodel-pass on` and no layer handed in, this says so
    through `NOTE` and runs the world arm; it does not silently pretend
    the axis ran.

    THE FOUR 2D TAPS: smoke colour, smoke alpha, the destination colour
    (the scene, or the viewmodel layer under D_VIEWMODEL_PASS), and the
    destination depth.
    """
    fam = "overlay_smoke"
    if _inj("overlay_passthrough"):
        return scene_rgb, (accum if accum is not None else None)

    # `viewmodel` selects the ARM; D_VIEWMODEL_PASS decides whether the
    # scheduler runs the second invocation at all. The world arm never
    # consults the axis -- reading the axis in both arms is how the world
    # composite ends up asking for a viewmodel it was never given.
    vm = bool(viewmodel)
    if vm and (viewmodel_rgb is None or viewmodel_z is None):
        # NOT a fallback to the world arm: that would silently report the
        # viewmodel arm as run. The caller is told nothing ran.
        NOTE.append(("viewmodel_pass_no_layer", 1, 1))
        return None, accum

    _trip(fam, "R_INVOKE")
    dst_rgb = viewmodel_rgb if vm else scene_rgb
    dst_z = viewmodel_z if vm else scene_z

    H, W = dst_rgb.shape[-3], dst_rgb.shape[-2]
    dev = dst_rgb.device
    yy, xx = _pixel_grid((H, W), dev)
    u = (xx + 0.5) / W
    v = (yy + 0.5) / H
    c = _tex2d(fam, smoke_rgb, u, v)                     # 1
    a = _tex2d(fam, smoke_a, u, v)                       # 2
    d = _tex2d(fam, dst_rgb, u, v)                       # 3
    z = _tex2d(fam, dst_z, u, v)                         # 4

    sz = torch.where(torch.isfinite(smoke_z), smoke_z,
                     torch.full_like(smoke_z, 1e9))
    soft = _smoothstep(0.0, max(A.smoke_overlay_softness, 1e-6),
                       (z - sz))
    aeff = a * soft

    if on(A.smoke_accum_pass):
        # D_ACCUM_PASS: premultiplied accumulate into a separate target,
        # which is what a pass that runs before the tonemap must do when
        # several smoke draws land in the same frame.
        base = accum if accum is not None else torch.zeros_like(
            torch.cat([d, aeff.unsqueeze(-1)], -1))
        rgb = base[..., :3] + c * aeff.unsqueeze(-1)
        al = base[..., 3] + aeff * (1.0 - base[..., 3])
        return dst_rgb, torch.cat([rgb, al.unsqueeze(-1)], -1)

    # the composite: `mix` on the colour, `mix` on the result against the
    # untouched destination outside the smoke's footprint
    out = d * (1.0 - aeff).unsqueeze(-1) + c * aeff.unsqueeze(-1)
    out = dst_rgb * (1.0 - (aeff > 0).float()).unsqueeze(-1) + \
        out * (aeff > 0).float().unsqueeze(-1)
    return out, accum


# ======================================================================
# 13.  THE SCHEDULER -- these are passes, not materials
# ======================================================================


def smoke_pass(eye, fwd, mvp, inv_mvp, scene_rgb, scene_z, tex, vols,
               sun_dir, sun_col, key_dir, key_col, n_moments,
               transparency=None, ss_shadow=None, mom_in=None,
               viewmodel_rgb=None, viewmodel_z=None,
               proj_near=0.05, proj_far=800.0):
    """Run the subsystem, in the order the enumeration infers.

    All seven are **dynamic-only**: one static combo, empty static array,
    every axis selected per draw.  They are not materials and there is no
    material-family dispatch anywhere in this file -- the render path
    calls this function, and this function calls the seven.

        smoke_volume_mask     -> the tile mask
        smoke_volume_depth    -> the subsystem's depth
        smoke_volume          -> the march; writes moments (D_MBOIT_PASS1)
        mboit_mixed_combine   -> resolve at MIXED resolution
        mboitfinal            -> recombine to the output resolution
        overlay_smoke         -> composite (twice under D_VIEWMODEL_PASS)
        write_smoke_depth_water_reflection

    `transparency` is the EXISTING per-material MBOIT pass-2 output --
    `dict(rgb=_accum, a=_acov, b0=_m0, z=<linear depth>, mom=<b0-ish>)`
    straight out of `gpu_render.render_window`.  Passing it is how the
    resolve wires to the buffers that already exist; passing None runs the
    smoke on its own and says so.
    """
    dev = scene_rgb.device
    H, W = scene_rgb.shape[-3], scene_rgb.shape[-2]
    scale = 1 if on(A.smoke_fullres) else max(int(A.mboit_mixed_scale), 1)
    mh, mw = max(H // scale, 1), max(W // scale, 1)

    # rays at the march resolution
    yy, xx = _pixel_grid((mh, mw), dev)
    ndc = torch.stack([(xx + 0.5) / mw * 2 - 1,
                       (yy + 0.5) / mh * 2 - 1,
                       torch.zeros_like(xx),
                       torch.ones_like(xx)], -1)
    far_h = torch.einsum("ij,...j->...i", inv_mvp,
                         torch.stack([ndc[..., 0], ndc[..., 1],
                                      torch.ones_like(xx),
                                      torch.ones_like(xx)], -1))
    far_w = far_h[..., :3] / far_h[..., 3:].clamp(min=1e-9)
    origin = eye.view(1, 1, 3).expand(mh, mw, 3).contiguous()
    direction = far_w - origin
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    # the scene depth at the march resolution, POINT sampled
    z_march = scene_z[::scale, ::scale][:mh, :mw].contiguous()

    # --- 1. the mask ---------------------------------------------------
    grid = build_occupancy(vols, dev)
    su = (xx + 0.5) / mw
    sv = (yy + 0.5) / mh
    mask = smoke_volume_mask(vols, origin, direction, z_march, tex, grid,
                             scope_uv=(su, sv))
    # D_SMOKE_FULLRES_PASS 0..2: the mask's own resolution axis. 0 halves
    # it again (and dilates, so halving can only ever grow the mask), 1
    # leaves it on the march grid, 2 promotes it to full resolution.
    fp = int(A.smoke_fullres_pass)
    if fp == 0:
        m2 = mask[::2, ::2]
        m2 = (torch.nn.functional.max_pool2d(
            m2[None, None].float(), 3, 1, 1)[0, 0] > 0.5)
        mask = torch.nn.functional.interpolate(
            m2[None, None].float(), size=(mh, mw), mode="nearest")[0, 0] > 0.5
    elif fp == 2:
        mfull = torch.nn.functional.interpolate(
            mask[None, None].float(), size=(H, W), mode="nearest")[0, 0]
        mask = mfull[::scale, ::scale][:mh, :mw] > 0.5

    # --- 2. the depth producer ----------------------------------------
    smoke_z, smoke_hit = smoke_volume_depth(vols, origin, direction,
                                            z_march, tex, mask)

    # --- 3. the march --------------------------------------------------
    ssh = None
    if ss_shadow is not None:
        ssh = ss_shadow[::scale, ::scale][:mh, :mw].contiguous()
    # The moment buffer the per-material D_MBOIT_PASS1 already wrote,
    # point-sampled onto the march grid. This is the wire that makes
    # D_MBOIT_PASS2 on smoke_volume mean what it means on a material:
    # the smoke resolves against the SAME moments, not against a copy.
    mom_march = None
    if mom_in is not None:
        mom_march = tuple(
            (m[::scale, ::scale][:mh, :mw].contiguous()
             if torch.is_tensor(m) and m.dim() >= 2 else m) for m in mom_in)
    rgb, alpha, front, b0, mom = smoke_volume(
        vols, origin, direction, z_march, smoke_z, mask, tex,
        sun_dir, sun_col, key_dir, key_col, n_moments, ss_shadow=ssh,
        mom_in=mom_march)

    smoke_layer = dict(rgb=rgb, a=alpha, z=torch.where(
        torch.isfinite(front), front, torch.full_like(front, 1e9)),
        b0=b0, mom=mom[..., 0] if mom.numel() else b0)

    # --- 4/5. the resolve ----------------------------------------------
    out_rgb, out_a = scene_rgb, torch.zeros((H, W), device=dev)
    if A.mboit_resolve != "off":
        if transparency is None:
            NOTE.append(("mboit_resolve_no_transparency", 1, 1))
            transparency = dict(
                rgb=torch.zeros((H, W, 3), device=dev),
                a=torch.zeros((H, W), device=dev),
                b0=torch.zeros((H, W), device=dev),
                z=torch.full((H, W), 1e9, device=dev),
                mom=torch.zeros((H, W), device=dev))
        mixed = mboit_mixed_combine(transparency, smoke_layer, scene_rgb,
                                    scene_z, n_moments)
        out_rgb, out_a = mboitfinal(mixed, transparency, scene_rgb,
                                    smoke_layer)
    else:
        # with the resolve off the smoke still has to reach the frame, or
        # the default would quietly disable it. It goes straight to
        # overlay_smoke at full resolution.
        up = torch.nn.functional.interpolate(
            torch.cat([rgb, alpha.unsqueeze(-1)], -1).permute(2, 0, 1)[None],
            size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        out_rgb = scene_rgb
        out_a = up[..., 3]
        smoke_layer = dict(smoke_layer, rgb=up[..., :3], a=up[..., 3],
                           z=torch.nn.functional.interpolate(
                               smoke_layer["z"][None, None], size=(H, W),
                               mode="nearest")[0, 0])

    # --- 6. the composite ----------------------------------------------
    sm_rgb = smoke_layer["rgb"]
    sm_a = smoke_layer["a"]
    sm_z = smoke_layer["z"]
    if sm_rgb.shape[0] != H or sm_rgb.shape[1] != W:
        sm_rgb = torch.nn.functional.interpolate(
            sm_rgb.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
            align_corners=False)[0].permute(1, 2, 0)
        sm_a = torch.nn.functional.interpolate(
            sm_a[None, None], size=(H, W), mode="bilinear",
            align_corners=False)[0, 0]
        sm_z = torch.nn.functional.interpolate(
            sm_z[None, None], size=(H, W), mode="nearest")[0, 0]
    if A.mboit_resolve != "off":
        # the resolve already folded the scene in; overlay_smoke then
        # composites the SMOKE-only layer over the viewmodel arm
        composited = out_rgb
    else:
        composited = scene_rgb
    world_rgb, accum = overlay_smoke(composited, scene_z, sm_rgb, sm_a, sm_z,
                                     viewmodel=False)
    vm_rgb = None
    if on(A.smoke_viewmodel_pass):
        vm_rgb, _ = overlay_smoke(world_rgb, scene_z, sm_rgb, sm_a, sm_z,
                                  viewmodel_rgb=viewmodel_rgb,
                                  viewmodel_z=viewmodel_z, viewmodel=True)

    # --- 7. the water-reflection depth write ---------------------------
    refl = None
    if on(A.smoke_water_reflection):
        vp = mvp
        inv = inv_mvp
        M = water_reflect_matrix(vp, inv, A.smoke_water_plane_y)
        refl_depth = torch.ones((mh, mw), device=dev)
        refl, wrote = write_smoke_depth_water_reflection(
            smoke_z, smoke_hit, z_march, refl_depth, M, proj_near, proj_far)
        del wrote

    return dict(rgb=world_rgb, viewmodel_rgb=vm_rgb, alpha=out_a,
                smoke=smoke_layer, accum=accum, reflection_depth=refl,
                mask=mask, smoke_z=smoke_z,
                # D_MBOIT_PASS1's contribution, at the MARCH resolution.
                # The render path adds it into the material moment
                # buffers -- that is the smoke writing moments, and it is
                # why the resolve sees one system and not two.
                moments=mom if on(A.smoke_mboit_pass1) else None,
                moment_scale=scale,
                march_steps=(A.smoke_march_steps if A.smoke_quality == "1"
                             else A.smoke_march_steps_lowq))


# ======================================================================
# 14.  CONFORMANCE + REACHABILITY -- checks that can fail
# ======================================================================


def conformance(strict=True):
    """Check the code that RAN against the measured enumeration.

    Every assertion here reads `TRIPS`/`FETCH`, which are incremented at
    the fetch and loop sites themselves, so this cannot be satisfied by a
    declaration.  Each one is paired with an injection name that makes it
    fail; `--smoke-selftest-inject list` prints them, and the selftest
    refuses to report success unless the injected run FAILED.

    Returns a list of (name, ok, detail).
    """
    out = []

    def chk(name, ok, detail):
        out.append((name, bool(ok), detail))

    # -- the axis table matches the measured min/max ------------------
    ok = True
    for fam, r in REF.items():
        for nm, mn, mx in r["axes"]:
            if mx < mn:
                ok = False
    chk("axis_ranges_wellformed", ok, f"{sum(len(r['axes']) for r in REF.values())} axes")

    # -- D_SMOKE_INSTANCES is 1..6, and the doc's 0..6 is wrong -------
    mn, mx = [(a[1], a[2]) for a in REF["smoke_volume"]["axes"]
              if a[0] == "D_SMOKE_INSTANCES"][0]
    chk("D_SMOKE_INSTANCES_range", (mn, mx) == (1, 6),
        f"measured min={mn} max={mx} (allps_deep.json); the callflow doc "
        f"prints 0..6")

    # -- our region depths equal the measured region depths ------------
    ref_depths = sorted(d for (_i, d, _f, _x) in REF["smoke_volume"]["regions"])
    our_depths = sorted(d for (_n, d, _i) in OUR_REGIONS)
    chk("region_depths", ref_depths == our_depths,
        f"ref {ref_depths} vs ours {our_depths}")
    chk("region_count", len(OUR_REGIONS) == REF["smoke_volume"]["loops"] == 12,
        f"{len(OUR_REGIONS)} regions, reference {REF['smoke_volume']['loops']}")
    chk("max_depth", max(d for (_n, d, _i) in OUR_REGIONS)
        == REF["smoke_volume"]["max_depth"] == 4,
        f"depth {max(d for (_n, d, _i) in OUR_REGIONS)}")

    # -- every region this COMBO has actually executed -----------------
    # D_SMOKE_PERF_TEST selects the cheap arm, which genuinely has no
    # detail octaves and no in-scatter chains; those regions are absent
    # from that combo rather than skipped in it, so the expected set is
    # axis-derived. Every other combo must execute all twelve.
    perf = on(A.smoke_perf_test)
    cheap = {"R_OCTAVE", "R_LIGHT_A", "R_LIGHT_B", "R_CASCADE_A",
             "R_CASCADE_B", "R_TAP_A", "R_TAP_B"}
    want_regions = [n for (n, _d, _i) in OUR_REGIONS
                    if not (perf and n in cheap)]
    if not on(A.smoke_mboit_pass1):
        want_regions = [n for n in want_regions if n != "R_MOMENT"]
    missing = [n for n in want_regions
               if TRIPS.get(("smoke_volume", n), 0) == 0]
    chk("all_regions_executed", not missing,
        f"never executed: {missing}" if missing
        else f"{len(want_regions)}/12 expected at these axis values, all "
             f"executed" + (" (D_SMOKE_PERF_TEST drops the cheap arm's 7)"
                            if perf else ""))

    # -- the 3D taps are INSIDE the march -----------------------------
    marched = TRIPS.get(("smoke_volume", "R_MARCH"), 0)
    f3 = FETCH.get(("smoke_volume", "sampler3D"), 0)
    per = 2 if on(A.smoke_use_noise_texture) else 1
    chk("march_3d_fetches", marched > 0 and f3 == marched * per,
        f"{f3} sampler3D fetches over {marched} march bodies "
        f"({per}/step expected); flat would be {per}")
    chk("march_not_flat", f3 > per,
        f"{f3} > {per}: the taps are not hoisted out of the march")

    # -- the straight-line inventory ----------------------------------
    # EVERY 2D tap of smoke_volume is straight-line: the measured
    # straight-line split is (414, 6 x sampler2D, 1 x samplerCube) and the
    # ONLY region carrying a fetch is %13587's two 3D sites. So a 2D tap
    # that ever moves inside a loop fails right here.
    f2 = FETCH.get(("smoke_volume", "sampler2D"), 0)
    fc = FETCH.get(("smoke_volume", "samplerCube"), 0)
    want2 = 5 + (1 if on(A.smoke_new_visuals) else 0)
    chk("smoke_volume_2d_sites", f2 == want2,
        f"{f2} 2D fetches over the whole family, all straight-line; "
        f"expected {want2} (6 at the measured peak; the ramp tap is "
        f"D_SMOKE_NEW_VISUALS')")
    chk("smoke_volume_no_2d_in_loops", f2 < marched if marched > 6 else True,
        f"{f2} 2D fetches vs {marched} march bodies -- a per-step 2D tap "
        f"would make these equal or larger")
    chk("smoke_volume_cube_site", fc == 1,
        f"{fc} samplerCube fetch (reference: 1)")

    # -- the resolves' tap counts -------------------------------------
    dec_c = sum(n for _k, n, _a in COMBINE_TAPS)
    dec_f = sum(n for _k, n, _a in FINAL_TAPS)
    chk("combine_taps_declared", dec_c == REF["mboit_mixed_combine"]["fetch"]
        ["sampler2D"] == 25, f"declared {dec_c}, measured peak 25")
    chk("final_taps_declared", dec_f == REF["mboitfinal"]["fetch"]
        ["sampler2D"] == 18, f"declared {dec_f}, measured peak 18")
    got_c = FETCH.get(("mboit_mixed_combine", "sampler2D"), 0)
    got_f = FETCH.get(("mboitfinal", "sampler2D"), 0)
    if A.mboit_resolve == "off":
        exp_c = exp_f = 0
        note = " (--mboit-resolve off: the resolve did not run, and the "
        note += "smoke reaches the frame through overlay_smoke instead)"
    else:
        exp_c, exp_f = expected_combine_taps(), expected_final_taps()
        note = ""
    chk("combine_taps_issued", got_c == exp_c,
        f"{got_c} taps issued, {exp_c} expected at these axis values "
        f"(peak 25 needs D_UPSCALE_SMOKE_DEPTH on){note}")
    chk("final_taps_issued", got_f == exp_f,
        f"{got_f} taps issued, {exp_f} expected at these axis values "
        f"(peak 18 needs D_DENOISE_SMOKE on and D_COVER_GAPS=2){note}")

    # -- the other families' fetch inventories ------------------------
    # overlay_smoke runs ONCE per composite, and D_VIEWMODEL_PASS makes
    # that twice, so its site count is 4 x invocations. The mask's DDA
    # loop reissues one of its three sites per body, so the site count is
    # the fetch count with the loop's repeats folded back out.
    inv = TRIPS.get(("overlay_smoke", "R_INVOKE"), 0)
    got = FETCH.get(("overlay_smoke", "sampler2D"), 0)
    chk("overlay_smoke_2d_sites", inv > 0 and got == 4 * inv,
        f"{got} taps over {inv} invocation(s) = 4 sites each "
        f"(reference: 4)")
    got = FETCH.get(("write_smoke_depth_water_reflection", "sampler2D"), 0)
    wantw = 4 if on(A.smoke_water_reflection) else 0
    chk("write_smoke_depth_water_reflection_2d_sites", got == wantw,
        f"{got} sites, {wantw} expected "
        f"(--smoke-water-reflection={A.smoke_water_reflection}); the "
        f"family has NO combo axes, so 4 is its only count when it runs")
    dda = TRIPS.get(("smoke_volume_mask", "R_DDA"), 0)
    got = FETCH.get(("smoke_volume_mask", "sampler2D"), 0)
    got = got - dda + (1 if dda else 0)
    chk("smoke_volume_mask_2d_sites", got == 3,
        f"{got} sites ({dda} DDA bodies folded out), reference 3")
    d3 = FETCH.get(("smoke_volume_depth", "sampler3D"), 0)
    coarse = TRIPS.get(("smoke_volume_depth", "R_COARSE"), 0)
    refine = TRIPS.get(("smoke_volume_depth", "R_REFINE"), 0)
    per_d = 2 if on(A.smoke_use_noise_texture) else 1
    chk("smoke_volume_depth_3d", d3 == (coarse + refine) * per_d and d3 > 0,
        f"{d3} 3D fetches over {coarse} coarse + {refine} refine bodies "
        f"(4 SITES across two nested loops, per the reference)")
    chk("smoke_volume_depth_depth2",
        TRIPS.get(("smoke_volume_depth", "R_REFINE"), 0) > 0,
        "the depth-2 refinement executed")

    if strict:
        pass
    return out


def reach_report():
    """What ran, and at what trip counts. Printed whenever smoke is on.

    Modelled on the transparency-axis reach block in `gpu_render.py`: a
    path that affects zero pixels has to SAY so.  The march's trip count
    and its executed 3D fetch count are always here, because the one
    thing the enumeration does NOT determine is what bounds this loop.
    """
    lines = ["smoke / MBOIT-resolve reach:"]
    steps = TRIPS.get(("smoke_volume", "R_MARCH"), 0)
    inst = TRIPS.get(("smoke_volume", "R_INSTANCES"), 0)
    f3 = FETCH.get(("smoke_volume", "sampler3D"), 0)
    lines.append(
        f"    smoke_volume        {inst} instance bodies, {steps} march "
        f"bodies, {f3} sampler3D fetches = steps x "
        f"{2 if on(A.smoke_use_noise_texture) else 1}")
    lines.append(
        f"      march bound       UNIFORM --smoke-march-steps"
        f"{'' if A.smoke_quality == '1' else '-lowq'} = "
        f"{A.smoke_march_steps if A.smoke_quality == '1' else A.smoke_march_steps_lowq}"
        f".  The enumeration does NOT classify this loop's bound and gives "
        f"no ceiling; neither does this renderer. Ours, not the engine's.")
    for nm, d, rid in OUR_REGIONS:
        t = TRIPS.get(("smoke_volume", nm), 0)
        lines.append(f"      %{rid:<6d} depth {d}  {nm:<12s} {t:>9,d} bodies"
                     + ("   <-- NEVER EXECUTED" if t == 0 else ""))
    lines.append(
        f"    smoke_volume_depth  "
        f"{TRIPS.get(('smoke_volume_depth', 'R_COARSE'), 0)} coarse + "
        f"{TRIPS.get(('smoke_volume_depth', 'R_REFINE'), 0)} refine bodies, "
        f"{FETCH.get(('smoke_volume_depth', 'sampler3D'), 0)} sampler3D")
    lines.append(
        f"    smoke_volume_mask   D_DDA_MASK={A.smoke_dda_mask}, "
        f"{TRIPS.get(('smoke_volume_mask', 'R_DDA'), 0)} DDA bodies, "
        f"D_SMOKE_FULLRES_PASS={A.smoke_fullres_pass}, "
        f"D_SCOPE_CLIP={A.smoke_scope_clip}")
    lines.append(
        f"    mboit_mixed_combine {A.mboit_resolve}, "
        f"{FETCH.get(('mboit_mixed_combine', 'sampler2D'), 0)}/25 taps "
        f"issued, D_UPSCALE_SMOKE_DEPTH={A.mboit_upscale_smoke_depth}, "
        f"D_LOW_HI_OUTPUT={A.mboit_low_hi_output}")
    lines.append(
        f"    mboitfinal          "
        f"{FETCH.get(('mboitfinal', 'sampler2D'), 0)}/18 taps issued, "
        f"D_DENOISE_SMOKE={A.mboit_denoise_smoke}, "
        f"D_COVER_GAPS={A.mboit_cover_gaps}, "
        f"D_UPSCALE_MIXED={A.mboit_upscale_mixed}, "
        f"D_COMBINE_HI_RES={A.mboit_combine_hi_res}")
    lines.append(
        f"    overlay_smoke       "
        f"{FETCH.get(('overlay_smoke', 'sampler2D'), 0)} taps, "
        f"D_ACCUM_PASS={A.smoke_accum_pass}, "
        f"D_VIEWMODEL_PASS={A.smoke_viewmodel_pass}")
    lines.append(
        f"    write_smoke_depth_water_reflection  "
        f"{FETCH.get(('write_smoke_depth_water_reflection', 'sampler2D'), 0)}"
        f"/4 taps, --smoke-water-reflection={A.smoke_water_reflection}")
    lines.append(
        "    per-view CB provenance for these seven is NOT resolved -- "
        "SHADER_CALLFLOW_smoke_and_mboit.md §6.2. The scene-depth and "
        "offset-88 bindings above are INFERENCE from axis names and tap "
        "counts, not a binding read.")
    for k, a, b in NOTE:
        lines.append(f"    NOTE {k}: {a} / {b}")
    return "\n".join(lines)


# ======================================================================
# 15.  THE SELFTEST
# ======================================================================


def _synth_transparent_stack(H, W, device, layers=4):
    """A synthesised transparent stack, since the world pack has none.

    Four overlapping translucent quads at four depths, run through the
    SAME power-moment accumulation `gpu_render.mboit_pass1` implements,
    so what the resolve consumes here is a real moment buffer and not a
    hand-written one.  Returns the pass-2-shaped dict the render path
    hands to the resolve.
    """
    yy, xx = _pixel_grid((H, W), device)
    u, v = (xx + 0.5) / W, (yy + 0.5) / H
    accum = torch.zeros((H, W, 3), device=device)
    acov = torch.zeros((H, W), device=device)
    b0 = torch.zeros((H, W), device=device)
    nmom = 4 if A.mboit == "4" else 6
    mm = torch.zeros((H, W, nmom), device=device)
    znear = torch.full((H, W), 1e9, device=device)
    for k in range(layers):
        cx, cy = 0.35 + 0.1 * k, 0.4 + 0.08 * k
        r = 0.30 - 0.03 * k
        inside = ((u - cx) ** 2 + (v - cy) ** 2).sqrt() < r
        a = torch.where(inside, torch.full_like(u, 0.35), torch.zeros_like(u))
        z = torch.full_like(u, 3.0 + 2.0 * k)
        col = torch.tensor([0.8 - 0.15 * k, 0.3 + 0.15 * k, 0.4], device=device)
        b = -(1.0 - a.clamp(1e-5, 0.9999)).log()
        b0 = b0 + b
        # the power moments, accumulated exactly as gpu_render.mboit_pass1
        # accumulates them: z^k * b summed over peeled layers
        zw = _warp(z)
        zk = torch.ones_like(zw)
        for kk in range(nmom):
            zk = zk * zw
            mm[..., kk] = mm[..., kk] + zk * b
        accum = accum + col.view(1, 1, 3) * (a * (1 - acov)).unsqueeze(-1)
        acov = acov + a * (1 - acov)
        znear = torch.where(inside & (z < znear), z, znear)
    if nmom == 4:
        mom_in = (b0, mm)
    else:
        mom_in = (b0, mm[..., :2].contiguous(), mm[..., 2:].contiguous())
    return dict(rgb=accum, a=acov.clamp(0, 1), b0=b0, z=znear, mom=b0,
                mom_in=mom_in)


def _run_once(quiet=False):
    """One full subsystem run on synthesised inputs -> (failures, reach).

    Reachability is measured as max|delta| against the SAME frame with
    the subsystem suppressed, in the file's own convention: never an
    image metric, never a claim of identity, always a number.
    """
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    H, W = 96, 128
    reset_counters()
    tex = synth_textures(dev)
    eye = torch.tensor([0.0, 1.6, 0.0], device=dev)
    fwd = torch.tensor([0.0, 0.0, -1.0], device=dev)
    vols = synth_volumes(A.smoke_instances, eye, fwd, dev)

    near, far = 0.05, 800.0
    f = 1.0 / math.tan(math.radians(90.0) / 2)
    proj = torch.tensor([
        [f / (W / H), 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]], device=dev, dtype=torch.float32)
    view = torch.eye(4, device=dev)
    view[:3, 3] = -eye
    mvp = proj @ view
    inv_mvp = torch.linalg.inv(mvp)

    scene_rgb = torch.full((H, W, 3), 0.25, device=dev)
    scene_z = torch.full((H, W), 40.0, device=dev)
    scene_z[H // 2:, :] = 14.0                 # a floor, to intersect
    ss = torch.ones((H, W), device=dev)
    trans = _synth_transparent_stack(H, W, dev)
    vm_rgb = torch.full((H, W, 3), 0.5, device=dev)
    vm_z = torch.full((H, W), 0.6, device=dev)

    res = smoke_pass(eye, fwd, mvp, inv_mvp, scene_rgb, scene_z, tex, vols,
                     torch.tensor([0.3, 0.9, 0.3], device=dev),
                     torch.tensor([1.0, 0.96, 0.88], device=dev),
                     torch.tensor([-0.6, 0.3, 0.7], device=dev),
                     torch.tensor([0.30, 0.34, 0.42], device=dev),
                     4 if A.mboit == "4" else 6,
                     transparency=trans, ss_shadow=ss,
                     mom_in=trans["mom_in"],
                     viewmodel_rgb=vm_rgb, viewmodel_z=vm_z,
                     proj_near=near, proj_far=far)

    d = (res["rgb"] - scene_rgb).abs()
    cov = float((d.amax(-1) > 1e-6).float().mean())
    rows = conformance()
    bad = [r for r in rows if not r[1]]
    if not quiet:
        print(reach_report(), flush=True)
        print(f"    smoke reached {cov:.4%} of the frame; "
              f"max|delta| vs the un-smoked scene {float(d.max()):.6f}",
              flush=True)
        if res["viewmodel_rgb"] is not None:
            dv = (res["viewmodel_rgb"] - res["rgb"]).abs()
            print(f"    D_VIEWMODEL_PASS second composite: max|delta| vs "
                  f"the world composite {float(dv.max()):.6f}", flush=True)
        if res["reflection_depth"] is not None:
            rd = res["reflection_depth"]
            print(f"    reflection depth: "
                  f"{float((rd < 1.0).float().mean()):.4%} of texels "
                  f"written, min {float(rd.min()):.6f}", flush=True)
        print("conformance against the measured enumeration:", flush=True)
        for nm, ok, det in rows:
            print(f"    {'PASS' if ok else 'FAIL'}  {nm:28s} {det}",
                  flush=True)
        if cov <= 0.0:
            print("    FAIL  reachability                smoke reached 0%",
                  flush=True)
    if cov <= 0.0:
        bad = bad + [("reachability", False, "smoke reached 0% of the frame")]
    return bad, cov, res


def selftest():
    """Reachability on a synthesised smoke volume and a synthesised
    transparent stack, then conformance, then the fault injections.

    EXITS NON-ZERO on any failure, and -- for the injection runs -- on any
    injection the checks did NOT catch.  A check that cannot fail is not
    a check.
    """
    bad, _cov, _res = _run_once(quiet=False)
    return bad


# Every axis of the seven, with EVERY value it can take. The sweep runs
# each value from a fixed baseline -- one axis at a time, all values --
# and requires every run to reach pixels and to pass conformance. That is
# full per-value coverage and it is NOT the Cartesian product; the claim
# made is exactly the one the sweep supports.
SWEEP = (
    # (axis, every value it takes, companion settings the axis needs to
    #  be observable at all).  The companion column exists because some
    #  axes are options ON another axis -- D_UPSCALE_MIXED only does
    #  anything once D_LOW_HI_OUTPUT has put the mixed buffer at the
    #  reduced resolution, and D_MBOIT_OPTIM only once D_MBOIT_PASS2 is
    #  resolving.  Sweeping them from a baseline that leaves their
    #  enclosing axis off is the --ibl-lod-scale failure again: the arm
    #  runs, reports, and executes nothing.
    ("smoke_quality", ("0", "1"), None),
    ("smoke_fullres", ONOFF, None),
    ("smoke_use_noise_texture", ONOFF, None),
    ("smoke_new_visuals", ONOFF, None),
    ("smoke_perf_test", ONOFF, None),
    ("smoke_mboit_pass1", ONOFF, None),
    ("smoke_mboit_pass2", ONOFF, None),
    ("smoke_mboit_optim", ONOFF, {"smoke_mboit_pass2": "on"}),
    ("smoke_instances", (1, 2, 3, 4, 5, 6), None),
    ("smoke_dda_mask", ONOFF, None),
    ("smoke_fullres_pass", (0, 1, 2), None),
    ("smoke_scope_clip", ONOFF, None),
    ("smoke_accum_pass", ONOFF, None),
    ("smoke_viewmodel_pass", ONOFF, None),
    ("smoke_water_reflection", ONOFF, None),
    ("mboit_resolve", ("off", "hi_low", "low_hi", "keep_full"), None),
    ("mboit_low_hi_output", ONOFF, None),
    ("mboit_upscale_smoke_depth", ONOFF, None),
    ("mboit_upscale_mixed", ONOFF, {"mboit_low_hi_output": "on"}),
    ("mboit_combine_hi_res", ONOFF, None),
    ("mboit_denoise_smoke", ONOFF, None),
    ("mboit_cover_gaps", (0, 1, 2), None),
    ("mboit", ("4", "6"), None),
)


def sweep():
    """Run every value of every axis. Any run that reaches 0 pixels, or
    fails conformance, is a failure -- an axis value whose path never
    executes reads as tested otherwise, which is exactly the
    --ibl-lod-scale failure this project has already paid for once."""
    import copy
    base = copy.deepcopy(A)
    configure(base)
    reset_counters()
    _b, _c, base_res = _run_once(quiet=True)
    base_img = base_res["rgb"]
    base_mom = base_res.get("moments")
    rows = []
    for name, values, extra in SWEEP:
        for val in values:
            cfg = copy.deepcopy(base)
            setattr(cfg, name, val)
            for k, v in (extra or {}).items():
                setattr(cfg, k, v)
            configure(cfg)
            reset_counters()
            try:
                bad, cov, res = _run_once(quiet=True)
            except Exception as exc:                    # noqa: BLE001
                rows.append((name, val, False, f"raised {type(exc).__name__}"
                                               f": {exc}"))
                continue
            # max|delta| against the BASELINE run, never an image metric
            # and never a claim of identity: the number is printed and an
            # axis value that moves nothing is labelled INERT rather than
            # left to read as tested.
            dimg = float((res["rgb"] - base_img).abs().max())
            dm = ""
            if (base_mom is not None and res.get("moments") is not None
                    and res["moments"].shape == base_mom.shape):
                dmv = float((res["moments"] - base_mom).abs().max())
                dm = f", moments {dmv:.3e}"
            elif base_mom is not None and res.get("moments") is not None:
                dm = (f", moments reshaped "
                      f"{tuple(base_mom.shape)}->"
                      f"{tuple(res['moments'].shape)}")
            elif (base_mom is None) != (res.get("moments") is None):
                dm = ", moment buffer present/absent flipped"
            # Several axes do not write the world composite at all --
            # D_VIEWMODEL_PASS writes a second layer, the water-reflection
            # write writes a depth target, the mask axes change what the
            # march is allowed to touch. Those are reported on their own
            # outputs so a real effect is never printed as a zero.
            side = (f", mask {float(res['mask'].float().mean()):.3%}")
            if res.get("viewmodel_rgb") is not None:
                side += (f", vm|d| "
                         f"{float((res['viewmodel_rgb'] - res['rgb']).abs().max()):.3e}")
            if res.get("reflection_depth") is not None:
                side += (f", refl {float((res['reflection_depth'] < 1.0).float().mean()):.3%}")
            ok = (not bad) and cov > 0.0
            rows.append((name, val, ok,
                         (f"reach {cov:.4%}, max|d| {dimg:.3e}{dm}{side}")
                         if ok else
                         (f"reach {cov:.4%}; " +
                          ", ".join(n for n, _o, _d in bad))))
    configure(base)
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mboit", choices=("off", "4", "6"), default="6",
                   help="stands in for gpu_render.py's own --mboit when "
                        "this module is run on its own")
    add_arguments(p)
    a = p.parse_args()
    if a.smoke == "off":
        a.smoke = "synth"
    if a.mboit_resolve == "off":
        a.mboit_resolve = "hi_low"
    a.smoke_viewmodel_pass = "on"
    a.smoke_water_reflection = "on"
    if a.smoke_selftest_inject == "instances_seven":
        # this one is caught BEFORE any pixel runs, by configure()'s
        # range guard, so the harness has to accept a SystemExit as the
        # catch rather than a failed conformance row
        a.smoke_instances = 7
        try:
            configure(a)
        except SystemExit as e:
            print(f"injection 'instances_seven' CAUGHT by configure(): {e}",
                  flush=True)
            raise SystemExit(0)
        print("injection 'instances_seven' NOT CAUGHT -- configure() "
              "accepted a value outside the measured 1..6 range",
              flush=True)
        raise SystemExit(2)
    configure(a)
    if a.smoke_selftest_sweep:
        rows = sweep()
        print("axis sweep -- EVERY value of EVERY axis, one axis at a time "
              "from the baseline (NOT the Cartesian product):", flush=True)
        for nm, val, ok, det in rows:
            print(f"    {'PASS' if ok else 'FAIL'}  {nm:26s} = {str(val):9s} "
                  f"{det}", flush=True)
        nbad = sum(1 for r in rows if not r[2])
        print(f"\n{len(rows)} axis values exercised, {nbad} FAILED",
              flush=True)
        raise SystemExit(1 if nbad else 0)
    bad = selftest()
    if a.smoke_selftest_inject:
        if bad:
            print(f"\ninjection {a.smoke_selftest_inject!r} CAUGHT by "
                  f"{len(bad)} check(s): "
                  + ", ".join(n for n, _o, _d in bad), flush=True)
            raise SystemExit(0)
        print(f"\ninjection {a.smoke_selftest_inject!r} NOT CAUGHT -- the "
              "checks that were supposed to notice it cannot fail",
              flush=True)
        raise SystemExit(2)
    if bad:
        print(f"\n{len(bad)} check(s) FAILED", flush=True)
        raise SystemExit(1)
    print("\nall checks passed", flush=True)


if __name__ == "__main__":
    main()
