# What the three slot families are worth, on five frames instead of one

> ⚠️ **EVERY ROW ABOVE THE "environment" SECTION WAS RENDERED WITHOUT A SKY
> CUBE.** `--sky-cube auto` resolves by pack stem and every pack in this
> lane is a variant stem, so the map's own cube was never loaded — see
> that section. Those rows are internally consistent and uniformly
> understated; **do not join them against any row measured after the
> resolver fix**. The rows to trust for cross-arm comparison are the
> de-aliased table at the end.

Executed 2026-08-09 by `harness/gpu_render/arms/five_frames.sh` over every
frame of the only de_inferno pair in the CS-1K corpus
(`match_bfec1f06ef8b__r010__p03`, 5 frames). One snapshot of the tree, one
variable between arms.

    A  de_inferno_slots.pt, no flags       the ladder's best-arm control
    B  de_inferno_ovl.pt,   --overlay      + g_tSharedColorOverlay page
    D  de_inferno_ao.pt,    --overlay      + normal/roughness + AO pages

## nrmse

| frame | A | B | Δ(A→B) | D | Δ(B→D) |
|---|---|---|---|---|---|
| f300 | 1.2027 | 1.2746 | **+0.0719** | 1.2801 | +0.0055 |
| f400 | 1.6579 | 1.4987 | −0.1592 | 1.5632 | +0.0645 |
| f500 | 1.7716 | 1.5610 | −0.2106 | 1.5735 | +0.0125 |
| f600 | 1.5078 | 1.2574 | −0.2504 | 1.2919 | +0.0345 |
| f700 | 1.6857 | 1.3724 | −0.3133 | 1.3870 | +0.0146 |

`B→D` is worse on **5 of 5**. ncc moves the other way on 4 of 5
(+0.0018 to +0.0025) and KL splits 3 worse / 2 better. The normal and AO
pages are real, the decode is the reference's, and on this pair they cost
nrmse everywhere.

## The A→B column is NOT the overlay

`material.overlay_signed` is the signed page the composite shapes, and
the AUX_HALF sentinel reads **+0.00392** exactly. On **f600 and f700 it is
that constant over every sample** — the runtime flags it, "CONSTANT across
every sample this run". No overlay page is visible in those two frames at
all; the composite is the identity by construction. They are also the two
largest A→B gains in the table, −0.2504 and −0.3133.

So the overlay cannot be what moved them. `--overlay` is a `GT_AXES`
member and flips **GTSURF** for the whole frame; that switch is what the
A→B column mostly measures. Isolated at f400 by holding GTSURF on and
setting the composite to identity (`--overlay-gain 0`):

| f400 arm | nrmse | KL | ncc |
|---|---|---|---|
| A | 1.6579 | 3.0948 | 0.1405 |
| C — GTSURF on, composite at identity | 1.5172 | 3.1031 | 0.1335 |
| B — + overlay page | 1.4987 | 3.0375 | 0.1315 |

**88% of the f400 move is GTSURF; 100% of the f600/f700 moves are.** The
overlay's own price, where it can act, is nrmse −0.0185, KL −0.0656,
ncc −0.0020 — small, two of three favourable, one frame.

And the sign is not established: on f300 the overlay is nearly inert
(mean −0.0011) and A→B is **+0.0719 worse**, so GTSURF is not uniformly
good either — it is large and mostly favourable, with one frame against.

**GTSURF is the unpriced axis this table keeps pointing at**, worth
roughly 6× the overlay and owned by no one.

## AO coverage, per frame

`surface.ao_page_bound` is the fraction of frame pixels whose material has
a real `mat_ao` page — the material-side fact, which separates "no page
here" from "the page is white here" (AO maps are mostly white, so the
value alone cannot).

| frame | f300 | f400 | f500 | f600 | f700 |
|---|---|---|---|---|---|
| bound | 17.3% | **0.59%** | 2.6% | 9.9% | 21.6% |

f400 — the tick all three families were originally priced against — is
the *worst* frame in the pair for AO by a factor of 29. Pricing AO there
gave a difference of exactly zero to every digit, and that zero was a
property of the camera, not of the family. 104 of 327 de_inferno
materials carry AO; declaration count ranked it third and pixel
visibility ranks it last, which is `pack_drop_census`'s own caveat about
itself ("material count is not screen area") landing on the first family
where it bites.

## Standing caution

One pair is not a corpus. Every number here is de_inferno, one camera
path, five frames 100 ticks apart, and the frames are not independent.

---

# The environment: one missing input, one refuted hypothesis

Executed 2026-08-09, tick 77159 / f000400, same snapshot.

`--sky-cube` defaults to `auto` and resolves `<world stem>.sky_cube.npz`.
Every pack this lane built is named `de_inferno_ovl/_nrm/_ao`, and the
cube on disk is `de_inferno.sky_cube.npz` — so **every arm above ran with
no sky cube**, including the ladder's A control on `de_inferno_slots.pt`.
The comparisons were internally consistent and all of them understated.
The fix was four symlinks; the same trap applies to
`.irradiance.npy` and `.light_environment.json`.

| arm | pack | sky | ibl | nrmse | KL | ncc |
|---|---|---|---|---|---|---|
| A | slots | – | – | 1.6579 | 3.0948 | 0.1405 |
| B | ovl | – | – | 1.4987 | 3.0375 | 0.1315 |
| D | ao | – | – | 1.5632 | 3.1253 | 0.1333 |
| **B+sky** | ovl | ✓ | – | **1.4813** | **2.2715** | 0.1354 |
| D+sky | ao | ✓ | – | 1.5463 | 2.3320 | 0.1374 |
| B+sky+ibl | ovl | ✓ | ✓ | 1.5683 | 2.7997 | 0.1320 |
| D+sky+ibl | ao | ✓ | ✓ | 1.6795 | 3.0378 | 0.1335 |

**The sky cube moves all three metrics together, on both arms.**
B: −0.0174 / −0.7660 / +0.0039. D: −0.0170 / −0.7933 / +0.0041. KL falls
25%. It is a missing input restored, not a feature added.

**The specular IBL makes all three worse, on both arms.** B: +0.0870 /
+0.5282 / −0.0034. D: +0.1332 / +0.7058 / −0.0039. Built by
`sky_cube_to_ibl.py` from the same cube; the standing suspect is its
stated `P = 1` approximation — one sky environment fetched everywhere,
including under roofs — which is not the reference's per-probe
parallax-corrected radiance. Not yet separated from "the term itself is
wrong here".

## The normals hypothesis is refuted by its own test

The claim was that real per-pixel roughness decides how a *missing*
environment is spread, so the normal/roughness arm should recover once
radiance exists. The D−B gap, in nrmse:

    no environment      +0.0645
    sky cube present    +0.0650
    sky + specular IBL  +0.1112

Radiance did not flip the sign and did not shrink the gap. With the sky
cube — the input whose absence the hypothesis blamed — the penalty
changes by **+0.0005**, which is nothing. The hypothesis is dead and the
normal/roughness loss is unexplained again.

## Two defects found on the way

* `--ibl-cube` without `--cube-probes` died with `NameError: name 'PLO'
  is not defined`, thirty thousand lines in — a real precondition,
  undeclared, surfacing as a code fault rather than a missing input.
  `probe_of()` now answers index 0 for a single-environment cube, where
  0 is not a fallback but the only index there is, and refuses by name
  otherwise.
* `sky_cube_to_ibl.py`'s first run put the texel solid angle into the
  VALUE instead of the WEIGHT, computing `Σ w·L·dω / Σ w` — mip4 came out
  mean 0.0053 against the source's 2.6993, a factor of 509. The images
  looked like plausible blurred cubes. The per-mip mean is printed for
  exactly this reason.

---

# GTSURF de-aliased, and priced under its own name

`GTSURF` was `any(GT_AXES.values())` — reachable only by asking for an
unrelated surface feature. `--overlay` therefore turned the GT-surface
path on for the whole frame as a side effect, and the arm that was meant
to price the overlay priced both. Same class as `--vmat-ao` riding
`--normal-map`. `--gt-surface {auto,on,off}` makes it addressable;
`auto` is the historical coupling, so no existing command line changes
meaning.

**What it is, read off the dispatch, not assumed.** It is not a shading
model. The whole mechanism is `x_gt = MAT_EXT2[mid_op] if GTSURF else
None` — "is the EXT2 side table bound". Every `gt_*` transcription takes
`x_gt`, so with it off those functions have no parameters and are
**absent, not substituted**. Nearly every `if GTSURF:` interior is
further gated by its own feature flag, so the switch alone is not
obviously a large change — which is why its size needed measuring rather
than asserting.

## Three namings tested and refuted before the right arm existed

1. *"GTSURF selects gt_\* over legacy shading."* Refuted by reading: the
   interiors are individually flag-gated, and the only unconditional swap
   on the overlay path is `gt_tint_mask_value` for `tint_mask_value`.
2. *"It is `--gt-lighting`, the ported CS2 combo path."* Refuted:
   `gt_on = bool(args.gt_lighting) and gt is not None`, a separate flag,
   default 1.
3. *"It is the auto-enabled `--specular/--normal-map/--vertex-normals`
   trio that a family axis turns on."* Refuted by arm: those three on the
   slots pack with no overlay and no GTSURF give 1.6340 / 2.3680 /
   0.1408, nowhere near the move being explained.

## The de-aliased answer, tick 77159 f000400, sky cube present on all arms

| arm | overlay | gt-surface | nrmse | KL | ncc |
|---|---|---|---|---|---|
| spec trio only | – | off | 1.6340 | 2.3680 | 0.1408 |
| **G0** | ✓ | **off** | 1.6341 | 2.2651 | 0.1372 |
| **G1** | – | **on** | **1.4977** | 2.3262 | 0.1363 |
| B+sky | ✓ | on (auto) | 1.4813 | 2.2715 | 0.1354 |

* **GTSURF alone: nrmse −0.1363**, with the overlay provably not in the
  arm. That is essentially the whole −0.1407 that the aliased arm C
  attributed to "overlay". The axis is real and it is the large one.
* **The overlay alone, GTSURF off: nrmse +0.0001.** Its own composite,
  through the world pack's `mat_ext`, moves nrmse by nothing on this
  frame (KL −0.1029, ncc −0.0036).
* **The overlay on top of GTSURF: −0.0164.** So the overlay's
  contribution is CONDITIONAL on GTSURF — the two paths read the same
  read expression from different parameter sources (`mat_ext` vs `EXT2`,
  and only the EXT2 path carries the overlay's own UV transform).

Every "overlay gain" this lane reported before this table was the
GT-surface axis wearing the overlay's flag.

## Still owed

The five-frame sweep under the new flag, which is now one command:
`five_frames.sh` with arms `--gt-surface on` / `off` instead of
`--overlay`. And f300, the one frame where the aliased arm went the
wrong way, should be investigated as "what does the transcription get
wrong there", not averaged away.

---

# AO re-priced at 21.6% coverage: the term is DEAD, not small

Ordered re-price on f700, the highest-AO-coverage frame in the pair
(21.6% against f400's 0.59%), with the sky cube resolved.

| pack | nrmse | KL | ncc |
|---|---|---|---|
| `de_inferno_nrm.pt` (normals, no AO) | 1.3869768592631135 | 2.1333824012989298 | 0.0778 |
| `de_inferno_ao.pt` (normals + AO) | 1.3869768592631135 | 2.1333824012989298 | 0.0778 |

The rendered PNGs are **byte-identical** — md5
`869182ae0cad973418b8989778cbfd1b` both. And the term is demonstrably
being sampled with real values:

    surface.ao_page        mean 0.996609  [0.1176, 1]  at-identity(1) 0.9656
    surface.ao_page_bound  mean 0.216363               at-identity(0) 0.7836

21.6% of pixels have a page bound, **3.44% of samples carry occlusion
that is not 1.0**, values reach 0.1176 — and the output does not change
by one bit. At f400 the zero was a coverage answer. At f700 coverage is
29× higher, real occlusion reaches the shader, and the frame is
identical. **That is not a small term, it is a term that does not reach
the image.**

`GAP CLOSED ambient occlusion` prints, so the page is loaded and
`sample_pages` runs; the multiply lands in the INDIRECT term
(glsl:1230 q0 / :1588-1592 q1). The mechanism by which it is discarded
downstream is **NOT established** and I am not guessing at one. The
falsifier is cheap and specific: force `ao_o = 0` everywhere and re-render
— if the frame is still byte-identical, the multiply is dropped after the
fact; if it goes black, the indirect term is simply near zero in this
configuration and AO is real but has nothing to occlude.

This supersedes "AO is worth exactly zero because f400 has no coverage".
The coverage explanation was correct about f400 and wrong as a general
account.

---

# Five frames under `--gt-surface`, and `--gt-lighting` is NOT inert

## The GT-surface axis, de-aliased, all five frames

Same pack, same feature flags, one variable (`harness/gpu_render/arms/five_frames.sh axis`).

| frame | nrmse on | off | Δ | KL on | off | ncc on | off |
|---|---|---|---|---|---|---|---|
| f300 | 1.2551 | 1.2765 | **−0.0215** | 1.9777 | 1.9895 | 0.1581 | 0.1654 |
| f400 | 1.4813 | 1.6341 | **−0.1528** | 2.2720 | 2.2670 | 0.1355 | 0.1372 |
| f500 | 1.5636 | 1.7664 | **−0.2028** | 1.6949 | 1.6920 | 0.0298 | 0.0240 |
| f600 | 1.2574 | 1.5078 | **−0.2504** | 2.3821 | 2.4239 | 0.3862 | 0.3996 |
| f700 | 1.3724 | 1.6857 | **−0.3133** | 2.1503 | 2.1626 | 0.0753 | 0.0784 |

**nrmse: ON is better on 5 of 5**, −0.0215 to −0.3133, mean −0.1882.
**KL: better on 3 of 5 and flat** (−0.042 to +0.005).
**ncc: WORSE on 4 of 5** (−0.0017 to −0.0134); better only at f500.

A mixed triplet, stated as one. ncc is contrast- and brightness-invariant
by construction, so nrmse improving while ncc degrades says the LEVEL got
better and the structure got slightly worse — not a clean win, and not
the shape of a change that is merely cosmetic either.

**f300 is no longer the wrong-way frame.** Under the alias, A→B at f300
was +0.0719 and was the one result pointing against the axis. De-aliased,
f300 goes the RIGHT way (−0.0215). The +0.0719 belonged to the confound —
the overlay, the pack change, and the missing sky cube all rode that
comparison. There is no f300 anomaly to investigate; the thing to
investigate was the arm.

## `--gt-lighting` is not inert, and now says so every run

The suspicion was the `#33` shape: `gt_on = bool(args.gt_lighting) and gt
is not None`, defaulting to 1, where the second conjunct is a per-frame
input the caller may never build. It ACTS. The ack print now states
which, every run, and the arms differ by md5 on all five frames.

| frame | nrmse 1 | 0 | KL 1 | 0 | ncc 1 | 0 |
|---|---|---|---|---|---|---|
| f300 | 1.1817 | 1.2125 | 1.9345 | 1.9421 | 0.1765 | 0.1655 |
| f400 | **1.6406** | **1.5731** | 2.3636 | 2.3576 | 0.1431 | 0.1396 |
| f500 | 1.7733 | 1.9512 | 1.7467 | 1.9646 | 0.0290 | 0.0210 |
| f600 | 1.5078 | 1.5290 | 2.4238 | 2.5225 | 0.3996 | 0.3247 |
| f700 | 1.6857 | 1.7837 | 2.1630 | 2.3294 | 0.0784 | 0.0317 |

**KL better 5 of 5, ncc better 5 of 5, nrmse better 4 of 5** — the
outlier is f400 (+0.0675), a different frame from the GT-surface axis's.
The ported CS2 combo path beats the fitted composition on this pair.

## Default-change proposal, with its own counter-evidence

`--gt-surface auto` already resolves ON whenever any GT axis is selected,
so moving the default to *on-when-the-side-table-is-present* changes only
runs that load a side table and select no GT axis. The nrmse case is 5 of
5 and large. **The ncc case is 4 of 5 AGAINST**, and that is the reason
this is a proposal and not a landing: a default that improves the
level-sensitive metric everywhere and degrades the structure metric
almost everywhere is a trade, and which side is honest is not a metric
vote — it is the same question the `--weapon` branch decision faced.

---

# Census by PIXEL VISIBILITY, not declaration count

`pack_drop_census.py --pixels <matid-dump-dir> --world <pack>` ranks every
declared slot by the frame pixels its materials own. Run over all five
frames of the pair via `gpu_render --matid-dump`. Shares do not sum to
100%: each is the union of pixels owned by materials declaring that
param, and a material declares several.

| param | declared | decl-rank | px-share | px-rank | move |
|---|---|---|---|---|---|
| `g_tColor1` | 217 | 1 | 85.26% | 1 | — |
| `g_tHeight1` | 217 | 2 | 85.26% | 2 | — |
| `g_tNormal1` | 217 | 3 | 85.26% | 3 | — |
| `g_tColor2` | 69 | 7 | 74.82% | 4 | +3 |
| `g_tHeight2` | 69 | 8 | 74.82% | 5 | +3 |
| `g_tNormal2` | 69 | 9 | 74.82% | 6 | +3 |
| **`g_tSharedColorOverlay`** | **11** | **14** | **22.47%** | **7** | **+7** |
| `g_tNormal` | 105 | 5 | 12.48% | 8 | −3 |
| `g_tAmbientOcclusion` | 103 | 6 | 12.21% | 9 | −3 |
| `g_tColor` | 108 | 4 | 12.21% | 10 | −6 |
| `g_tMetalness` | 28 | 11 | 5.26% | 11 | — |
| `g_tTintMask` | 24 | 12 | 4.78% | 12 | — |
| `g_tSelfIllumMask` | 41 | 10 | 2.12% | 13 | −3 |
| `g_tTransmissiveColor` | 13 | 13 | 0.01% | 20 | −7 |

**The overlay moves up seven ranks.** This file's own header called it
"1.2% of the hole" and "explains ONE WALL", and ordered the work on that
basis. By pixels it is **22.47%** — seventh of thirty-three, ahead of
every normal, AO and metalness family. The declaration count was not
wrong, it was answering a different question, and the sentence warning
that "material count is not screen area" sat in the same docstring as
the ordering it invalidates.

**Height and normal layer 1 tie albedo at 85.26%** — the same 217
materials own those pixels, which is also a consistency check on the
join. `g_tHeight1` is behind `--height-pages`, a DIAGNOSTIC gate held off
because it measured worse, and it covers as much of the frame as the
albedo does. That is the largest untouched surface in the table.

**Layer 2 is 74.82%**, not the 69-material afterthought its declaration
rank suggests.

`g_tAmbientOcclusion` at 12.21% averaged is consistent with the 0.59% to
21.6% per-frame spread already recorded — the average hides exactly the
variation that made pricing it on one tick meaningless.

---

# `--height-pages` re-measured under corrected conditions

The gate's original verdict — "OFF BY DEFAULT BECAUSE IT WAS MEASURED AND
IT LOST", every metric worse and KL 14× — was rendered on tick 77159
alone, **without the sky cube, before the GT-surface de-alias, and before
the sidecar resolver**. Three conditions now known wrong. `g_tHeight1`
covers **85.26% of pixels**, tying albedo, so this is the largest surface
in the table and the verdict was worth re-taking.

Pack `de_inferno_hgt.pt`: 217/327 materials with a real height page, 286
pages, `has_h2` on 69.

**The C-arm settles the coupling first.** `--height-blend` on the
overlay pack — flag on, no pages — is **numerically equal to base on all
five frames, to every digit**, and the ack print says why: `--height-blend
is INERT this run: the pack carries NO decoded height page (has_h1 0,
has_h2 0)`. So the entire base→real delta below is the PAGES, and none of
it is the flag.

| frame | base | +flag (C-arm) | +pages | Δ nrmse | Δ KL | Δ ncc |
|---|---|---|---|---|---|---|
| f300 | 1.2551 | 1.2551 | 1.2596 | **+0.0045** | −0.0086 | +0.0009 |
| f400 | 1.4813 | 1.4813 | 1.4921 | **+0.0108** | **−0.0805** | −0.0115 |
| f500 | 1.5636 | 1.5636 | 1.5625 | −0.0010 | −0.0173 | −0.0006 |
| f600 | 1.2574 | 1.2574 | 1.2652 | **+0.0078** | +0.0069 | −0.0142 |
| f700 | 1.3724 | 1.3724 | 1.3842 | **+0.0118** | +0.0055 | −0.0015 |

**nrmse worse on 4 of 5. KL better on 3 of 5, including −0.0805 at f400.
ncc worse on 4 of 5.**

## The gate does NOT flip, and the original number was still wrong

Heights do not measure well under corrected conditions, so the
diagnostic gate stays off — the validated-replacement door does not open
on this evidence.

But the ORIGINAL verdict does not survive either. It recorded ratio
1.155 → 1.299, MSE 0.0466 → 0.1030, KL 0.2587 → **3.8353 (14×)**, ncc
−0.0449 → −0.2067 — a catastrophe. Measured now: nrmse moves by
**+0.0045 to +0.0118**, under 1%, and KL *improves* on three frames. The
family is marginal-and-mixed, not catastrophic. Whatever produced the 14×
belonged to the conditions, not to the pages, and the two suspects the
gate's own comment named — `decode_albedo` being an albedo path applied
to single-channel linear data, and `csgo_environment_blend` carrying its
TINT MASK in g_tHeight's green channel — are still unchecked and are now
the whole remaining question.

So the gate's text needs amending from "it lost" to "it is marginal and
its inputs are unverified", which is a different instruction to whoever
picks it up: the work is a decode check, not a re-measure.

---

# The indirect environment: it is the DOMINANT light, not a refinement

Opening measurement for the `GAP probe` / `GAP spec-cube` work, from the
diagnostics the current best arm already prints. No new code.

| frame | `baked.irradiance` mean | at-0 | `compose.direct_diffuse` mean | at-0 | indirect/direct | `direct_specular` mean |
|---|---|---|---|---|---|---|
| f300 | 0.4300 | 23.97% | 0.0943 | 94.81% | **4.56×** | 0.00175 |
| f400 | 0.6012 | 7.64% | 0.2514 | 89.19% | **2.39×** | 0.00360 |
| f500 | 0.5604 | 15.23% | 0.1179 | 94.35% | **4.75×** | 0.00093 |
| f700 | 0.5361 | 2.42% | 0.2555 | 89.61% | **2.10×** | 0.00293 |

**The indirect is 2.1× to 4.8× the direct diffuse.** And the at-identity
columns bound how much of the image it alone lights: `direct_diffuse` is
*exactly* zero on 89–95% of pixels, while `baked.irradiance` is zero on
only 2–24%. Background cannot explain the gap, because background would
zero both — so at least **71% (f300) to 87% (f700) of the frame's
geometry receives no direct sun at all** and is lit entirely by the
indirect term.

That reframes the assignment. The environment is not a refinement layered
on a sun-lit image; for the large majority of pixels it *is* the light,
and everything still missing from it — probe SH, spec-cube, lightmap
occlusion — is missing from the term that carries the image.

## It also explains the refuted normals hypothesis, properly this time

`compose.direct_specular` is **0.0009–0.0036**, two to three orders below
the indirect diffuse. Roughness governs the specular lobe. So the
normal/roughness family was always adjusting a term worth ~0.3% of the
frame's light, which is why adding radiance could not rescue it and why
the arm moved nrmse by ~0.06 in the wrong direction on noise-scale
grounds. The earlier account — "roughness spreads a missing environment"
— was refuted by its own test; **this** is the account, and unlike that
one it is measured rather than proposed.

## Two components of the indirect are flagged INERT by the run itself

    ambient.seed_after_hemisphere_deletion  mean 0  at-identity(0) 1.0000  <== INERT
    baked.lm_tint                           mean 1  at-identity(1) 1.0000  <== INERT

Both run on every pixel and change nothing. They are inside the term that
carries 2–5× the direct light, so they are the first two things to read
rather than the last.

---

# Real TEXCOORD_1 lands in the pack — and changes nothing

Ordered as step (1) of the ncc investigation: `DEFAULTED: uvs2 = uvs` was
the named spatial-error candidate, and level-improves/structure-degrades
is the signature of a spatial error.

**The substitution was real and larger than estimated.** TEXCOORD_1 is in
the extraction on 286 of 1428 vertex buffers (20.0%) — but those are the
big shared buffers, so it covers **5450 of 6610 draw calls (82.5%)**.
Every one of them was being handed UV1 in its place.

`de_inferno_uv2.pt` now carries a real `uvs2`. Against the identical
baseline pack, five frames, one variable:

| frame | base nrmse / KL / ncc | real UV2 |
|---|---|---|
| f300 | 1.255053065522461 / 1.977674823913971 / 0.1581 | **identical** |
| f400 | 1.4813110482362115 / 2.272013483852071 / 0.1355 | **identical** |
| f500 | 1.563550124940152 / 1.6949238805572024 / 0.0298 | **identical** |
| f600 | 1.2574025287279051 / 2.382059027527658 / 0.3862 | **identical** |
| f700 | 1.3723601229824918 / 2.1502597608230336 / 0.0753 | **identical** |

To every digit, on every frame. And with the consumer flag turned on —
`--secondary-uv material` — **both packs still produce the identical
number**, so it is not simply that the axis is off by default.

**UV2 does not reach the image.** The input is now correct and something
downstream does not read it. That is the third "carried, plausible, and
dead" finding in this lane — AO before its two fixes, `--rough-src`
reading a constant alpha, and now this.

**THE NCC EXPERIMENT HAS NOT RUN.** The order was: land real UV2, re-run
the standing arms, read whether ncc's disagreement shrinks. Step 1 landed
and steps 2–3 are VACUOUS — every arm reproduces its baseline exactly, so
no ncc column could have moved and nothing has been learned about the
hypothesis. The UV substitution is **not yet excluded** as the spatial
error; it is un-tested, which is a different state from tested-and-clear
and must not be recorded as the latter.

The next step is the same disambiguation that turned AO from "worth zero"
into "does not reach the image": find where `uv2_o` dies between
`dr.interpolate` at :27446 and the samplers. `shade_albedo(uv_o, uv2_o,
…)` and the `gt_uv(…, "uvsel_c1"/"uvsel_c2")` selectors at :27531-27534
are the three consumers to instrument.

---

# The two INERT indirect components: both CORRECT-inert, and not alike

Read-only. Both sit inside the term carrying 2.1–4.8× the direct light,
so "flagged INERT" had to be resolved before it could be dismissed.
**Neither is a defect. Neither is a lead.** But they are correct for
different reasons, and the difference is the useful part.

## `ambient.seed_after_hemisphere_deletion` — correct-inert, STRUCTURALLY

`gpu_render.py:14150` constructs the seed as a literal zero tensor, and
the reasoning above it is a read, not a choice: `g_vSunAmbient` — the
only constant ambient anywhere in the reference's sun path
(`csgo_environment_ps.glsl:923`, `csgo_complex_ps.glsl:641`) — **reads
(0,0,0) in 717 of 718 block copies across two captures**. The fitted
hemisphere was deleted deliberately, and its own `_gt_value` note already
says "a zero seed is the reference's value, not an ablation".

Zero here is the reference's value **always**, on every map and every
material. There is no configuration in which this term should be nonzero.

## `baked.lm_tint` — correct-inert, CONDITIONALLY

`:16522` takes a `if not live:` early return that yields the untinted
indirect and reports a ratio of exactly 1.0, noting that **`g_bLightmapTint`
(`_5618._m6`, BYTE OFFSET 60) is off for this baked source, so the ratio
is 1 by construction and the pow never ran**. Identity is precisely what
a zero gate should produce — the term is inert *because its input says
so*, which is the correct behaviour.

But this is a fact about **this baked source**, not about the renderer.
The `pow(indirect, [1.05, 0.95, 0.80])` path exists and would run if the
gate read on. So its inertness is contingent, and would become a defect
signal the moment the gate reads ON and the ratio stays 1.

## What that implies for the INERT detector

The detector flags "runs on every pixel, changes nothing", which is right
in general and produces a **false positive on both of these**. The
sharper predicate is *inert while its own gate says it should act*:

* `ambient.seed` should never be flagged — its declared identity IS the
  reference's read value, permanently.
* `baked.lm_tint` should be flagged only when `g_bLightmapTint` reads
  nonzero, which is exactly the case where identity would be wrong.

## And the negative result that matters

Neither dead component explains the indirect's dominance. The indirect is
2.1–4.8× the direct because it genuinely carries the image, not because
something inside it is broken. So the missing pieces are the honestly
absent ones — **probe SH, spec-cube, lightmap occlusion** — and there is
no cheap fix hiding in the terms already present.

---

# ANSWERED: `uv2_o` dies at the SELECTOR, and the selector column was
# filled from a key name that does not exist

Executed, not specified. Build: instrument hunks landed inside
`865ffd10` (swept — see the note at the end); `gt_uv` co-factor +
packer fix in the sha carrying this section. Pack census 2026-08-10 over
all 43 `.scratch-matside/*.mats.json` and all 44 built world packs.
Precondition state: `de_inferno_uv2.pt`, `--fam-side
de_inferno.fam_side.pt`, `--secondary-uv material`, GT-surface on.

## The runtime answer, from the frame that was actually rendered

`cs1k-inferno-720/match_bfec1f06ef8b__r001__p00` frame 32, tick 10395,
1280x720, nrmse 1.0002 / kl 0.7031 / ncc 0.206:

| probe | at-identity | reading |
|---|---|---|
| `uv.uv2_differs` | **0.1842** | mean 0.6670, range [0, 508] — UV2 IS arriving, on 81.6% of shaded pixels |
| `uv.gt_uv_c1_moved` | **1.0000** | range [0, 0] — layer 1 kept TEXCOORD0 on every pixel |
| `uv.gt_uv_c2_moved` | **1.0000** | range [0, 0] — layer 2 kept TEXCOORD0 on every pixel |

That is branch 2 of this document's own decision tree, confirmed: the
attribute is present and **the selectors dropped it**. Not a pack fault.

## Why the selectors said no — the name was derived, not read

`fast_pack_fam.py` filled `uvsel_c1/c2/c3/dn/ovl` from
`g_nTexCoordSource1/2/3`, `g_nDetailTexCoordSource` and
`g_nSharedColorOverlayTexCoordSource`. **Those five spellings occur in 0
of 8254 material rows across all 43 built maps.** The field CS2's
`.vmat_c` actually carries is `g_nUVSet1/2/3`, `g_nDetailUVSet1/2/3`,
`g_nColorOverlayUVSet`.

Positive control, same dicts, same probe — the search could have found a
positive and did: 32 IntParams keys match `/texcoord|uv/i` with real
values, and **78 slots across 11 maps select UV set 2**. The `-1`
inherit sentinel is there too (22 materials on de_cache), matching the
`if (_5618._m9 == -1)` construct at `..._vs_max.glsl:325`.

Rebuild of `de_cache.fam_side.pt`, one variable, the key name:

| column | old (`g_nTexCoordSource*`) | new (`g_nUVSet*`) |
|---|---|---|
| `uvsel_c1` | `{0.0: 221}` | `{0.0: 90, 1.0: 124, 2.0: 7}` |
| `uvsel_c2` | `{0.0: 221}` | `{0.0: 138, 1.0: 71, 2.0: 12}` |
| `uvsel_c3` | `{0.0: 221}` | `{0.0: 172, 1.0: 34, 2.0: 15}` |
| `uvsel_dn` | `{0.0: 221}` | `{-1.0: 22, 0.0: 178, 1.0: 13, 2.0: 8}` |
| `uvsel_ovl` | `{0.0: 221}` | `{0.0: 177, 1.0: 42, 2.0: 2}` |
| **slots selecting UV1** | **0** | **44** |

## FOUR independent zeros, and fixing one would not have shown

1. **The packer key name** (above) — every `uvsel_*` column 0 on every
   material of every pack. FIXED.
2. **`gt_uv`'s `* f_secondary_uv` co-factor**, which no read supports.
   All three compiled variants of the vertex stage guard the construct on
   the per-material uniform alone — `if (_5618._m0 > 0)` at
   `csgo_environment_blend_vs_max.glsl:344`, `_vs_base.glsl:162`,
   `_vs_gameplay_max.glsl:335` — and select with `_m0 == 2` on the next
   line. And it was load-bearing: **0 of the 78 UV1-selecting slots sit
   on a material that also sets F_SECONDARY_UV**, so fixing only (1)
   still renders identical frames and reads as "the fix did not work".
   REMOVED, on the read.
3. **`face_secondary` is omitted from all 44 world packs**, so the
   renderer defaults it to zeros, `_selv` is 1 everywhere, and the
   VERTEX-stage route — the one `--secondary-uv material` actually
   drives — is inert independently of (1) and (2). OPEN; owner is
   `ash_to_world`.
4. **`shade_albedo`'s `uv2` parameter is never read in its body.** Layer
   2 reaches UV2 only through `uv_l2`, i.e. only through `gt_uv`. The
   call site reads as a consumer and is not one. OPEN (cosmetic while
   (3) stands).

## ⚠️ de_inferno CANNOT EXHIBIT THIS, and the five-frame result stands

**Not one de_inferno material selects UV set 2 on any slot.**
`g_nUVSet1` is 1 on 8 materials, `g_nUVSet2` is 1 on 3, and
`g_nUVSet3`/`g_nDetailUVSet*`/`g_nColorOverlayUVSet` are absent from the
map entirely. So the five identical frames were **the correct output for
that map**, and the experiment could not have produced a positive
whatever the renderer did. That is a precondition failure, not a
renderer defect — a different state again from both "tested and clear"
and "carried but dead".

Maps that CAN exhibit it: de_cache 45, de_cache_vanity 12, de_ancient 5,
de_ancient_night 5, ar_shoots 3, ar_shoots_night 3, de_train 1,
de_train_vanity 1, cs_shelter 1, de_debris 1, de_eldorado 1.
**de_cache is the only one with enough to price a term on**, and it has
no `_uv2` pack yet. No term is priced from one camera and none is priced
here at all: (3) still gates the flag's own route, so the honest state is
*two of four zeros removed, not yet re-measured*.

## A fifth thing the census found, unowned

Six `g_bUseSecondaryUvFor*` booleans — AmbientOcclusion, Decal,
DetailMask, DetailTexture, SelfIllum, TintMask — are a SEPARATE per-slot
secondary-UV mechanism, read by the packer for none of them. de_inferno
itself sets four of them (`...ForAmbientOcclusion` 1, `...ForDetailMask`
1, `...ForDetailTexture` 1, and 0s elsewhere), and de_dust2 sets
`...ForDecal` on 10 materials. So a live UV2 route does exist on
de_inferno — just not one anything implements.

## The stand-in now says so at runtime

`uvs2 = data.get("uvs2", data["uvs"])` was silent, and **11 of the 44
built world packs take that branch** — including plain `de_inferno`,
`de_mirage`, `de_nuke`, `de_overpass`. A pack with no TEXCOORD_1 renders
UV2==UV1 and, from the frames alone, is indistinguishable from a pack
whose real UV2 is discarded downstream. It now prints which case it is,
with the vertex count. Same omission as `--rough-src` reading a constant
alpha and `mat_ao` reading the AUX_WHITE sentinel.

## Also fixed while tracing

`uv_l2 = gt_uv(uv_o, ...)` read the `uv_o` that line :27642 had just
rebound, so layer 2's "inherit the previous slot" branch would have
inherited layer 1's *selection* rather than TEXCOORD0. Both selectors now
read the same pre-selection `_uv_c0`. Inert today because both picks are
0 — which is why it had to be fixed before the gate opens, not after.

## Note on authorship

The four instrument hunks in `gpu_render.py` were swept into `865ffd10`
("Flashbang whiteout"), whose message does not cover them. Not reverted —
the standing rule — so this section is the record that they are this
lane's, verified present and intact in that sha at :5375, :5383, :7667,
:7669, :27605, :27705, :27714.

---

# Next action, specified: where does `uv2_o` die?

Written as a spec rather than done, because `.gpu_render.lock` is held.
This is the AO disambiguation applied to UV2 — the house route from
"worth zero" to "which link drops it".

**The fact to explain.** `de_inferno_uv2.pt` carries a real `uvs2` on
5450/6610 draw calls (82.5%). Against the identical baseline pack it
produces numerically identical frames on all five, and identical again
with `--secondary-uv material`. Something between the vertex attribute
and the samplers discards it.

**Three consumers, and the whole chain is short:**

    :27446  uv2_o, _ = dr.interpolate(uv2_attr[None]…)      <- enters here
    :27531  uv_o   = gt_uv(uv_o, uv2_o, "uvsel_c1", x_gt, …)
    :27533  uv_l2  = gt_uv(uv_o, uv2_o, "uvsel_c2", x_gt, …)
    :27624  alb    = shade_albedo(uv_o, uv2_o, …)

**The instrument, one line each, following `surface.ao_page_bound`'s
form** — the material-side fact that separates "no input" from "input
ignored":

    _gt_value("uv.uv2_differs", (uv2_o - uv_o).abs().amax(-1), identity=0.0,
              note="fraction at identity = pixels where UV2 IS UV1. A pack "
                   "with real TEXCOORD_1 on 82.5% of draw calls should read "
                   "WELL below 1.0 here; 1.0000 means the attribute never "
                   "reached the interpolator.")

then the same probe on `gt_uv`'s two outputs. That splits it in one run:

* `uv.uv2_differs` at identity 1.0000 → the attribute is not arriving;
  the packer or the `uv2_attr` gather is the fault, not the shading.
* differs < 1.0 but `uvsel_c1`/`uvsel_c2` outputs identical to `uv_o` →
  the **selectors** are choosing UV1, i.e. the EXT2 `uvsel_*` columns are
  absent or zero, which is a side-table question and NOT a pack question.
* both differ and the frame is still identical → the divergence is
  downstream of `shade_albedo`, same shape as the AO overwrite.

**Do not skip the first branch.** UV2's own `_gt_value` would have caught
this on the day the substitution was introduced, and the reason it went
unnoticed for the whole slot-family arc is that nothing ever printed
whether UV2 was UV1. That is the same omission as `--rough-src` reading a
constant alpha and `mat_ao` reading the AUX_WHITE sentinel: **an input
that stands in for another input must say so at runtime.**

---

# de_ancient: UV2 REACHES THE IMAGE, and the ncc hypothesis is REFUTED there

Executed 2026-08-10. Build: `gpu_render.py` md5 `f2049c57e82f8e9edc1aa94689dcdcce`,
the committed blob at `ad51aca8`, unchanged through `b67c3c1f`. Pack census
2026-08-10 over `.scratch-matside/de_ancient.mats.json` (204 materials) and
the packs built this session. GT: the 7 real CS2 pairs of
`pairs/class_sweep/de_ancient`, two matches, five distinct camera paths.

## FIRST, A CORRECTION TO THE BRIEF I WAS GIVEN

The task named de_ancient's 5 UV2 slots as the `g_nUVSet1 >= 2` family.
**They are not.** On de_ancient `g_nUVSet1` is `1` on 161 materials and is
**never** 2. Every one of the 5 selections is `g_nColorOverlayUVSet = 2`:

| key | materials | values |
|---|---|---|
| `g_nUVSet1` | 161 | `{1: 161}` |
| `g_nUVSet2` | 93 | `{1: 82, 0: 11}` |
| `g_nUVSet3` | 17 | `{1: 7, 0: 10}` |
| **`g_nColorOverlayUVSet`** | **5** | **`{2: 5}`** |
| `g_nDetailUVSet1/2` | 2 / 1 | `{-1: …}` the inherit sentinel |

That is not a detail. `ash_to_world` builds `face_secondary` from
`g_nUVSet1 >= 2` **alone** (`:1164`), so on de_ancient the VERTEX-stage
route is correctly inert and `--secondary-uv` cannot act. The only live
route is the side table's `uvsel_ovl` into `gt_uv` at `gpu_render.py:28069`.
**de_ancient tests the OVERLAY's UV2 and nothing else** — it cannot test
`uvsel_c1`/`uvsel_c2`, which is what de_cache's 45 slots are for.

## The visibility gate: NOT sub-percent, and the AO spread again

`--matid-dump` over all 7 GT frames, matids 29/33/35/36/37 of 192.

| frame | class | px-share of the 5 |
|---|---|---|
| f0 | bomb_planted | **19.458%** |
| f1 | bright | 0.000% |
| f2 | dark | 0.000% |
| **f3** | **flash** | **52.071%** |
| f4 | grenade_in_hand | 0.000% |
| f5 | haze | 0.000% |
| f6 | scoped | 0.003% |
| **all 7** | | **10.219%** |

Positive control, same dump: the top owner is matid 93 at 8.51%, and matid
37 is third of all 192 at 7.44% — the join to `mat_paths` produces a sane
ranking, so a zero here would have meant something.

**The 10.2% average is the AO mistake waiting to happen again**: two frames
carry the whole thing and four are exactly zero. And the largest, f3, is the
**flash** class at nrmse ~9 — a flashbang whiteout. **f0 at 19.5% is the
only honest frame in this pair for this family.**

## Two structural preconditions, both failing, both fixed

1. `de_ancient.pt` carries **no `uvs2`** — the runtime says so
   ("uvs2 SUBSTITUTED"), `uv.uv2_differs` at-identity 1.0000.
2. `mat_ovl` is the aux sentinel `3` on **192 of 192** materials — **zero
   real overlay pages**. A UV-set selection on a constant texture is the
   identity by construction.

Rebuilt with `--tex-dir --overlay-pages`: `uvs2` real (differs from `uvs` on
76.4% of verts pack-wide, **96.7% of the 6750 verts the 5 materials own**,
range [-14.2, 4.6], all finite — the 8.8e6 pack-wide outlier is elsewhere,
99.9th pct 12.2), overlay real on 5/5, `mat_paths` ordering identical so the
matids still name the same materials.

**One variable, honestly**: `--tex-dir` also gates the NORMAL block, so
base→uv2ovl moves three slots. The arm is therefore `de_ancient_uv2ovl.pt`
against a twin made by **deleting the `uvs2` key** from it — identical bytes
otherwise, and the renderer prints which branch it took.

## Arm 1: everything live, and STILL byte-identical on all 7

`--overlay --gt-surface on`. `uv2_differs` at-identity **0.0668** (UV2 on
93.3% of shaded pixels). Overlay pages real. And the sampled value **differs
between the arms** — `material.overlay_signed` mean **0.00987962** vs
**0.00971245**, ranges [-0.3328, 0.3395] vs [-0.3319, 0.3411].

**The PNGs are md5-identical on all 7 frames**, including f3 at 52.07%.

So the input reaches the sampler, the sampled texel changes, and the image
does not. That is the AO f700 shape proven one level deeper.

## WHERE IT DIES, read rather than guessed

    _k  = gt_overlay_layer_k(mode, overlay_tintmask2,
                             f_overlay2 * has_overlay2 * gain, tmv, w_eff)
    rgb_o = overlay_composite(rgb_o, _ov, ovl_bright, ovl_dark, _k)

Every EXT2 gate column READS CORRECTLY on the 5: `uvsel_ovl` 2.0,
`uv2_su/sv` 1.0, `uv2_ou/ov` 0.0, `ovl_su/sv` 1.0, `f_overlay2` 1.0,
`has_overlay2` 1.0, `ovl_bright/ovl_dark` five distinct real values,
`overlay_tintmask2` 0.0. With tintmask 0 the layer-k form collapses to

    k = on * tm * (l1*(1-w) + l2*w)

and **`tm` multiplies the whole composite**. The run says what `tm` was:

    material.tint_mask_raw    mean 1  [1, 1]  at-identity(1) 0.9998
    material.tint_mask_used   mean 0  [0, 0]  at-identity(0) 1.0000  <== INERT

`k == 0` on every pixel, so `overlay_composite` is the exact identity.

And `gt_tint_mask_value` says why, on a read of
`csgo_environment_blend_ps.glsl:593` — the mask comes from the HEIGHT
texture's GREEN channel:

    return src_tm * have_tm + src_h * (1 - have_tm) * has_h1

`has_tintmask2` is 0 (the pack reports "tintmask 0") and **`has_h1` is 0**,
because the pack was built without `--height-pages`. **The overlay composite
on this family is gated on the HEIGHT PAGES.** That is a sixth independent
zero, and it is the one nobody predicted: the height family and the UV2
family are the same question.

## Arm 2: with `--height-pages`, UV2 REACHES THE IMAGE

Same one-variable construction on `de_ancient_uv2hgt.pt` vs its
`uvs2`-deleted twin. `tint_mask_used` goes from INERT to **mean 0.831663,
[0, 1], at-identity 0.0684** — live on 93.2% of pixels, exactly as the read
predicted.

| frame | cover | md5 same | Δ nrmse | Δ KL | Δ ncc |
|---|---|---|---|---|---|
| f0 bomb_planted | 19.458% | **no** | +0.0003 | −0.0001 | **−0.0005** |
| f1 bright | 0.000% | YES | 0 | 0 | 0 |
| f2 dark | 0.000% | YES | 0 | 0 | 0 |
| f3 flash | 52.071% | **no** | −0.0130 | −0.3027 | **−0.0001** |
| f4 grenade_in_hand | 0.000% | YES | 0 | 0 | 0 |
| f5 haze | 0.000% | YES | 0 | 0 | 0 |
| f6 scoped | 0.003% | no | −0.0000 | +0.0000 | +0.0000 |

**The frames differ on exactly the three frames with nonzero coverage and
are identical on the four with zero.** Nothing enforced that; it is the
internal consistency check that says the effect is landing where those five
materials are and nowhere else.

## THE PRE-REGISTERED ANSWER: NO

The registered test was whether real UV2 shrinks the ncc column's
disagreement. **It does not.** ncc moves −0.0005 and −0.0001, in the WRONG
direction on both frames that can move, and by two to three orders less than
the ~0.1–0.3 disagreement being explained. KL improves −0.3027 at f3 and
nrmse −0.0130 there, but f3 is the flashbang frame at nrmse ~9 and no term
should be priced on it.

**On de_ancient the UV2 substitution is now TESTED and CLEAR** — a different
state from both "untested" and "carried but dead", and the third such
transition this lane has made.

⚠️ **SCOPE.** This tests the OVERLAY's UV2 on 5 materials. de_ancient has no
`uvsel_c1`/`uvsel_c2` >= 2 at all, so the albedo-layer UV2 — the large
population, 45 slots on de_cache — is **untouched by this result** and the
spatial-error hypothesis survives there. `de_cache_uv2.pt` is built and
still has no camera pair.

## Also read on the way: the height page's BYTES

The `--height-pages` gate's two named suspects, from the ash extraction's
own `.vtex_c` (161 de_ancient materials declaring `g_tHeight1`, all BC7):

* **Suspect (a) — "decode_albedo is an albedo path" — REFUTED by reading.**
  `decode_albedo()` is container + BC decode + resize with **no colour
  transform**, and `sample_aux()` documents that it does no sRGB decode.
  `ash_to_world:1646` says it outright: "THE PAGE GOES IN AS LINEAR."
  The function's NAME is the only albedo thing about the path.
* **Suspect (b) — the tint mask in GREEN — CONFIRMED as structure.**
  R and G are independent on **159 of 161** pages, max|R−G| up to 255.
  R is the height (means 78.8–147.7, full range), G is a near-saturated
  mask (means ~254), **B is exactly 0.0** and A is 0.0 on most — a
  two-channel map in a four-channel BC7 container.

So the height page is not corrupted by its decode and its green channel is
real. The gate's text should now read "marginal, and its inputs are now
VERIFIED", and its re-price is a live piece of work rather than a blocked
one — with the new fact that turning it on is also what makes the overlay
composite act at all.

---

# de_cache PRESENCE VERIFIED — and the `_uv2` pack's variable is NOT uv2

Executed 2026-08-10, same build (`gpu_render.py` md5
`f2049c57e82f8e9edc1aa94689dcdcce`). No GT: this is a presence check, four
ticks of `worlds-ws1/de_cache.camera.json` (the synthesised orbit), one
variable, `--secondary-uv material --gt-surface on --overlay`.

## The name is wrong, and the packs say so when loaded

| | `de_cache.pt` | `de_cache_uv2.pt` |
|---|---|---|
| generation | `a969dab6397d9972` | `25bd0d5d95b7ad1a` |
| verts / faces / mats | 3,944,983 / 3,273,211 / 221 | identical |
| `uvs2` | **REAL, 2,750,368 verts (69.7%)** | **REAL, 2,750,368 verts (69.7%)** |
| `face_secondary` | **ABSENT** → defaulted to zeros | **PRESENT, 422,158 (12.90%)** |

**`de_cache.pt` already carried real TEXCOORD_1.** The two packs are
identical in `uvs2` to the vertex, and the ONLY thing `_uv2` changes is
`face_secondary` — zero #3. A pack named for one input whose actual variable
is a different input is the same class of trap as `--vmat-ao` riding
`--normal-map`, and it was caught by loading the packs rather than by
reading the arm's name or its banner (both arms print the identical
"uvs2 REAL … 69.7%" line, so the log could not have distinguished them).

## Both albedo selectors are OFF at-identity 1.0000 — the last link is VERIFIED

| probe | `de_cache.pt` | `de_cache_uv2.pt` |
|---|---|---|
| `uv.uv2_differs` at-identity | 0.7671 | 0.8576 |
| `uv.gt_uv_c1_moved` at-identity | **0.9071** | **0.9208** |
| `uv.gt_uv_c2_moved` at-identity | **0.9392** | **0.9532** |
| `c1_moved` range | [0, **4401**] | [0, **573.6**] |
| `c2_moved` range | [0, **4401**] | [0, **117.1**] |
| frame md5 | `ef99d1a6ca3e1959e2f1f5829e460c6c` | `f3b0f998c23ba981fc3824a207775310` |

This is the state `SLOT_FAMILY_PRICES.md` recorded as branch 2 of its own
decision tree — "the attribute is present and the selectors dropped it",
`gt_uv_c1_moved`/`c2_moved` pinned at 1.0000 on every pixel. **They are no
longer pinned.** Layer 1 moves the coordinate on 7.9% of shaded pixels and
layer 2 on 4.7%. de_cache is the only map of 43 that can move these rows at
all (7 `uvsel_c1` and 12 `uvsel_c2` selections of the corpus's 44), so a
zero here would have meant something and it did not read zero.

## What `face_secondary` actually does: it RESTRICTS

Both arms move the selectors, so the fix is not "a dead path switched on".
Adding real `face_secondary` moves MORE pixels to identity (0.9071 → 0.9208,
0.9392 → 0.9532) and collapses the range from 4401 to 573.6 and 117.1.

That is the correct direction and it is the reason zero #3 was a defect:
with `face_secondary` absent the renderer defaults it to zeros, `_selv` is 1
everywhere, and the vertex route hands TEXCOORD_1 to **every** face instead
of the 12.90% that declare it. The old behaviour was an OVER-selection, not
an absence — and an over-selection renders plausibly, which is why it
survived.

**And the frames differ.** Unlike de_ancient's overlay UV2, this route
reaches the image with no further precondition.

## What this does NOT establish

No term is priced. This is presence, on a synthesised orbit camera, with no
GT pair — de_cache was never a cs1k map. The albedo-layer spatial-error
experiment still needs a real camera, which still needs a public de_cache
`.dem`. What has changed is that the experiment is now known to be
*runnable*: the consumers move, so a pair would measure something.

---

# de_cache albedo-UV2 is HIGHLY VISIBLE — and the demo pull is BLOCKED

Executed 2026-08-10, same build. The experiment that survives only on
de_cache is worth running: this measures how much of the frame its 18
albedo-UV2 materials own, which is the priceability question, and it needs
no demo.

## Build-era precondition, checked FIRST because it would have voided everything

Cache was **redesigned and re-released for CS2 on 2026-04-28**, and entered
Active Duty / Premier S5 on 2026-06-22 / 2026-07-09. A pre-April demo, or a
CS:GO-era one, is a DIFFERENT MAP and would have paired a real camera
against geometry that does not exist — the "build-era boundaries
structurally split a corpus" lesson, arriving as a live trap.

`ash/de_cache.ash/manifest.json` says `created_unix` 1786342787 =
**2026-08-10T06:19:47Z**, from `work/dd/game/csgo/maps/de_cache.vpk` pulled
**2026-08-10 01:59**. The depot content is TODAY's. **Our geometry is
current Cache**, so a recent demo pairs correctly. Precondition CLEARED.

## The 18 albedo-UV2 materials, and what they own

`uvsel_c1 >= 2` on 7 materials, `uvsel_c2 >= 2` on 12 — **18 distinct**
(the corpus's whole albedo-selection population; `c3` 15, `dn` 8, `ovl` 2
bring de_cache's total to 44).

    ALL FRAMES: 130,385 / 921,600 = 14.1477%

and the positive control reads the frame's composition, which is what makes
that number interpretable:

| matid | px-share | material |
|---|---|---|
| 65535 | **68.8235%** | *background / no owner* |
| **22** | **6.3322%** | `ch2_blend_concrete_panel_gravel_grass_001` |
| 0 | 3.8435% | `ch2_blend_asphalt_gravel_grass_001` |
| 47 | 2.9669% | `ch2_blend_gravel_grass_001` |
| **17** | **2.7605%** | `ch2_painted_plaster_blend_01b` |
| 11 | 2.3230% | `plant_panel_001` |

**68.8% of the frame is background.** So of the 31.18% that is geometry at
all, the albedo-UV2 materials own **45.4%** — and matid 22 is the
single largest real owner in the frame, ahead of every non-selecting
material. 15 of the 18 are visible at all; the tail runs down to 0.0035%.

This is the opposite of de_ancient's answer. There the term was 5 materials
on the overlay slot; here it is the dominant surface family of the map.

⚠️ **CAMERA CAVEAT, and it is a large one.** ONE frame, from the
SYNTHESISED ELEVATED ORBIT `ash_to_world` writes
(`worlds-ws1/de_cache.camera.json`) — not an eye-level play pose, and the
68.8% background is that camera looking at sky. This BOUNDS the term; it
does not price it, and an eye-level demo camera would change both the
background fraction and the mix. No term is priced from one camera.

## The demo pull: BLOCKED, and exactly where

Pre-authorised and attempted. Both hosts have working internet
(`api.github.com` 200 from the workstation and from terul-cluster). Routes
tried and what each returned:

| route | result |
|---|---|
| `hltv.org` — the pro-demo host | **403** by curl from BOTH hosts, and 403 via WebFetch |
| `faceit.com` | **403**, and demos are behind a per-account login |
| `csgostats.gg` | **403** |
| `demos.esea.net` | **000** — no route |
| `bo3.gg` | 200, but a JS SPA; renders empty to a fetcher, no demo links |
| `cs-demo-manager.com` | 200, but it is a TOOL — it pulls via Steam auth / share codes |
| GitHub REST API repo+code search | no CS2-era de_cache `.dem` fixture; the demo-parser repos ship CS:GO-era fixtures, which are the WRONG MAP and the wrong container |
| HuggingFace CS2 datasets | parquet/derived features, no raw `.dem` |

Every surviving route needs either a **Steam/FACEIT account** or defeating
HLTV's bot protection. The 403 is the site saying no, and I did not work
around it.

**What would unblock it, cheapest first:**
1. **An owner-supplied Steam match share code or a `.dem` from their own
   Watch tab** — Cache has been in Casual/Competitive since April and in
   Premier since July, so any owner-played Cache match is a valid camera.
2. An authenticated pull via `cs-demo-manager` / `cs-demo-downloader` with
   owner Steam credentials.
3. Explicit owner authorisation for a browser-driven HLTV fetch, which is a
   ToS question and therefore the owner's call, not this lane's.

**What is NOT blocked:** the visibility number above, the presence
verification, and the pack. The experiment is fully prepared and waiting on
one file.

---

# The probe field for all 43 maps, and the height re-price under its C-arm

Executed 2026-08-10, `gpu_render.py` md5 `f2049c57e82f8e9edc1aa94689dcdcce`
(committed blob at `ad51aca8`) for the priced arms; builder
`probe_atlas_build.py` at `2754d527`.

## The sweep: 43 built, 0 GAP, verified by LOADING

`--probe-npz` was one map's asset behind a comment that said it did not
exist. It is now a sidecar for every map, on the `<map>.probe_field.npz`
convention that `irradiance`/`direct_light_shadows` already use.

    sweep done: 43 built, 0 GAP        1,745 volumes total
    all 44 files load, every atlas finite, every depth == 6*band

(44 files, 43 maps: `de_inferno_hgtovl.probe_field.npz` is a deliberate
sidecar SYMLINK for the variant stem — the resolver trap this file has been
bitten by twice.)

Sample, read off the loaded artifacts:

| map | volumes | band | atlas | +Z/−Z |
|---|---|---|---|---|
| de_inferno | 116 | 140 | (840, 208, 160, 3) | 4.09 |
| de_train | 94 | 168 | (1008, 176, 236, 3) | 4.89 |
| de_overpass | 112 | 120 | (720, 200, 196, 3) | 2.26 |
| de_nuke | 92 | 108 | (648, 188, 200, 3) | **0.86** |
| ar_baggage | 57 | 112 | (672, 180, 180, 3) | 3.75 |

**Four maps read +Z/−Z below 1.05** (de_nuke 0.86, and three `_vanity`
backdrops). That is FLAGGED, NOT called a defect: the ratio is an
outdoor-sky heuristic, and de_nuke is largely interior while the vanity
scenes are lit menu dioramas. Someone should look; nothing here establishes
they are wrong.

### Two silent defects the de_inferno validation caught

The builder was validated against the hand-built ws-1 artifact before the
sweep ran — **atlas max|Δ| 0 over 27,955,200 texels**, 13 of 16 table
columns max|Δ| 0, the rest fp32 rounding (c13 is 1 in ~1.8e11). Two faults
surfaced only because a reference existed:

1. **`light_probe_atlas_Z` is capital Z.** The container spells the axes
   `_x`, `_Y`, `_Z` — lower, upper, upper. Hardcoding `("x","Y","z")` with
   `.get(...) or 0` behind it turned the missed axis into a silent zero, so
   every volume addressed atlas plane 0: a WRONG probe, not a missing one,
   and it renders plausibly. max|Δ| 125 on that column, and nothing said
   "missing".
2. **The extra-data walk was 12 bytes off**, re-derived rather than reused.
   At the wrong base it reads a PHANTOM type-0 entry where type 4 lives,
   concludes there is no COMPRESSED_MIP_SIZE table, and takes the
   uncompressed branch on a file that is LZ4.

And the first sweep built **16 of 43** and reported 27 as "missing
origin/box_mins/box_maxs" — every field was present. This lump mixes typed
and string values in the SAME entity (de_inferno's `origin` is a list, its
`voxel_size` is `'24.0'`), and on 27 maps the vectors are strings. The
refusal was honest and the diagnosis was wrong; naming the field rather
than the map is what made it findable. 16 → 43.

## The height re-price, C-armed, on ONE pack

`--height-pages` does TWO things, which is why the standing verdict could
not be re-taken by turning it on: it feeds the height blend, AND it sets
`has_h1`, which unlocks the overlay composite through the tint mask's green
channel. Three arms on `de_inferno_hgtovl.pt`, varying only render flags so
no pack change rides the comparison:

    A  heights off at the consumer, overlay at identity (--overlay-gain 0)
    C  heights ON,                  overlay at identity   -> HEIGHTS ALONE
    B  heights ON,                  overlay live          -> + the overlay

| frame | A nrmse/KL/ncc | C (heights alone) | B (+ overlay) |
|---|---|---|---|
| f300 | 0.9317 / 1.9317 / 0.2116 | 0.9386 / 1.9058 / 0.2116 | 0.9348 / 1.9501 / 0.2070 |
| f400 | 1.1197 / 2.3249 / 0.1700 | 1.1393 / 2.3162 / 0.1665 | 1.1024 / 2.2098 / 0.1544 |
| f500 | 1.1426 / 1.6865 / 0.0443 | 1.1522 / 1.6627 / 0.0445 | 1.1402 / 1.6217 / 0.0401 |
| f600 | 0.8466 / 2.6390 / 0.4099 | 0.8581 / 2.6472 / 0.3963 | 0.8581 / 2.6472 / 0.3963 |
| f700 | 0.9203 / 2.2871 / 0.1130 | 0.9368 / 2.2957 / 0.1132 | 0.9368 / 2.2957 / 0.1132 |

**A→C, the height family's own price:** nrmse WORSE on 5 of 5 (+0.0069 to
+0.0196), KL better on 3 of 5 (−0.0259 to +0.0086), ncc flat (−0.0136 to
+0.0002). Small and mixed-negative — consistent with the corrected
re-measure, and nothing like the original 14× KL catastrophe.

**C→B, the overlay term the heights unlock:** it is EXACTLY ZERO on f600
and f700 — B and C agree to every digit — which is this document's own
earlier finding arriving independently: on those two frames no overlay
material is visible, so the composite is the identity by construction. Where
it can act it improves nrmse (−0.0038, −0.0368, −0.0120) and KL at f400
(−0.1064) and f500 (−0.0410), and costs ncc every time (−0.0046, −0.0121,
−0.0044).

## ⚠️ EVERY OVERLAY PRICE IN THIS DOCUMENT WAS MEASURED ON ~0.6% OF THE FRAME

`material.tint_mask_used` on the arm all those tables used
(`de_inferno_ovl.pt`, no height pages) reads **at-identity 0.9944** — the
composite is suppressed on 99.44% of pixels. On `de_inferno_hgtovl.pt` it
reads at-identity **0.0007 to 0.0275**, i.e. live on 97–99.9%.

So the overlay was never inert on de_inferno the way it is on de_ancient,
but it was acting on **half a percent of the frame** while being reported as
a slot family. Its numbers above, from the arm where it reaches 99% of
pixels, are the first honest ones — and they are a TRADE, not a win: nrmse
and KL better, ncc worse on 3 of 3 frames where it acts.

## Still owed

The A-arm here is `--overlay-gain 0` on a pack that already carries the
height pages, so `has_h1` is set in every arm and this measures the height
CONSUMER, not the height PAGES. The pages-vs-no-pages question needs the
pack change and therefore carries the normal-block confound that
`--tex-dir` gates. That is a fourth arm, not a re-reading of these three.
