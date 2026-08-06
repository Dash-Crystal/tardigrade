# TARD 2.1 data-quality and reader-defect report

Independent re-implementation and cross-validation of the TARD 2.1 wire
format (`read_v2.py`, `data/*.tard.zst`, 155 packs, commit `030f7ca`),
performed 2026-08-05/06 while building a second decoder from the wire alone.
Every claim below is backed by a measured number over the published corpus;
none is a style opinion.

Cross-validation summary: an independent decoder agrees with `read_v2.py` at
max|Δ| = 0 for every integer stream (mouse, view deltas, moves, fire, all 23
event streams, sample-aligned positions) over 81,219 ticks / 1,868,037 event
cells of `0.tard.zst`, and to ≤ 4.8e-11 degrees on reconstructed view angles
over 376,782 ticks across 3 packs (residual attributable to `read_v2.py`'s
float cumsum, see D4). The format itself decodes cleanly: 35,650 event
streams + 1,550 fire streams across all 155 packs with zero framing errors.
The defects below are therefore *producer/reader* defects, not wire
corruption.

## D1 — `_interpolate` fabricates positions across teleports

`read_v2.py:122-139` (used at `:242-244`) linearly interpolates the 16 Hz
position keyframes to 128 Hz unconditionally. Physical CS movement caps near
16 units per 8-tick interval; pack `0.tard.zst` alone contains 23
inter-sample displacements above 136 units (8.7× that ceiling), the largest
3,023.7 units — respawns and round resets. Interpolation invents a
straight-line path across each discontinuity, placing players at coordinates
they never occupied (typically inside world geometry) for 7 of every 8
ticks in the affected intervals. Off-sample divergence between interpolation
and last-sample-hold on pack 0, x-axis: mean 3.99 units, p50 0.000,
p95 14.7, max 2,256.98.

**Fix:** suppress interpolation across any inter-sample displacement above a
threshold (e.g. 64 units per interval); hold the prior sample instead, or
expose both projections and let the consumer choose.

## D2 — decoded values are computed and then discarded

- Weapon change events are decoded and immediately thrown away
  (`read_v2.py:252`); the weapon dictionary parsed from the header is never
  applied.
- Location events are decoded inside a bare `try/except: pass`
  (`read_v2.py:284-287`); the location dictionary parsed at `:209-213` is
  never used. The bare `except:` also swallows `KeyboardInterrupt`.
- Empirically stream 31 needs no guard: it is plain integer zigzag events —
  222,207 values across 155 packs, every one a valid dictionary index.

**Fix:** surface `weapon` and `location` columns (index + resolved label);
remove the bare except.

## D3 — dead `_decode_string_events` signals unresolved format doubt

`read_v2.py:107-119` is never called. The corpus resolves the doubt it
encodes: stream 31 carries integer dictionary indices, not strings (530,739
weapon + 222,207 location indices across 155 packs, 0 out of range against
their headers' dictionaries under the sentinel-at-0 convention).

**Fix:** delete the dead path; document the sentinel convention.

## D4 — README misstates the view-angle output; float cumsum adds avoidable error

`README.md:57-66` claims "0.000° max reconstruction error"; that statement
is true of *delta fidelity* but the published reconstruction is an unbounded
accumulator, not a valid angle: measured absolute yaw spans
[-1,418.96°, +2,425.37°] across pack 0's players (one player reads 432.51°
for a physically-72.51° facing). Any consumer treating the column as a
heading without wrapping to [-180°, 180°) silently learns a many-to-one
representation. Separately, `read_v2.py` cumsums `value/100.0` in float64;
cumulating integer centidegrees and dividing once removes the measured
5.2e-12°–4.8e-11° drift entirely.

**Fix:** document the accumulator semantics explicitly, publish a wrapped
column (or wrap in the reader), and cumsum in integer centidegrees.

## D5 — `forward_move`/`left_move` documented as analog, actually ternary

`README.md:33-34` describes analog movement axes. Across the corpus exactly
3 distinct values occur: {-1, 0, +1}. No magnitude is transmitted.

**Fix:** correct the README (documentation defect only).

## D6 — event streams outrun the full-rate horizon

Event streams may carry ticks beyond the player's full-rate sample count
(one player: last event tick 11,080 vs 8,723 full-rate samples; 1,727
stream-instances affected across 12 packs). Both the reference reader and
any length-driven consumer silently drop these events, and nothing in the
format states which horizon is authoritative.

**Fix:** producer should either truncate event streams at the full-rate
horizon or document that events past it are valid (and what they mean for a
player with no corresponding control rows).

## D7 — undocumented 8-byte trailer

All 155 packs end with exactly 8 bytes after the last player record, varying
per file, read by neither `read.py` nor `read_v2.py`. Likely a digest or
counter; currently unverifiable.

**Fix:** document it (and if it is a checksum, verify it in the reader) or
remove it.

## D8 — no t = 0 state event

State streams (health, weapon, armor, …) almost always emit their first
event at tick 1 (32,798 of ~33k streams), but the gap reaches tick 65
(~0.5 s of undefined state). `read_v2.py` masks this with hardcoded
defaults (e.g. `health = 100`), which is a guess, not data.

**Fix:** producer should emit a t = 0 event for every state stream; readers
should distinguish "no event yet" from a real value rather than defaulting.

## D10 — the `map_name` header does not name the actual map

Across all 155 packs the header declares exactly two values: `de_mirage`
(128 packs) and `unknown` (27 packs) — yet the packs carry **31 distinct
location-callout dictionaries**, i.e. many different maps.

Direct falsification on `0.tard.zst` (declared `de_mirage`): its callout
dictionary is Inferno's (Banana, Library, Ruins, Arch, TRamp, SecondMid, …;
no Mirage-exclusive callout present), and projecting all 10,156 sample-
aligned player positions through the official overview calibrations gives:

- `de_inferno` (pos_x −2087, pos_y 3870, scale 4.9): **99.59%** of points on
  drawn map area, 100% in bounds, zero fitted offset — trajectories trace
  corridors exactly;
- `de_mirage` (pos_x −3230, pos_y 1713, scale 5.0): 36% at zero offset, at
  best 61% after fitting a translation, with points off the mapped world
  entirely.

Pack 0 is a de_inferno match labeled `de_mirage`. Any consumer keying
geometry, navigation, or rendering off the header will use the wrong world.

**Fix:** producer should write the real map name. Until re-published,
consumers must infer the map from the location dictionary (the callout sets
are map-unique) or by the projection test above; treat `map_name` as
untrusted.

## D9 — v1 and v2.1 packs share an extension with no dispatch

`.tard.zst` files with magic `TARD` (v1) and `TR21` (v2.1) coexist under the
same glob patterns used by `load_all()`/`iter_sessions()`; a mixed directory
hard-fails on an assert partway through iteration.

**Fix:** dispatch on magic bytes in `load()`; skip-or-route rather than
assert.

---

Methodology: second decoder written from the wire bytes alone, then
cross-checked numerically (magnitudes reported above; no "identical" claims
— every agreement is stated as a measured max|Δ|). Physical validation:
circular mean of (velocity heading − reconstructed absolute yaw) during
pure-forward movement = +0.25° (R = 0.984, n = 1,456 8-tick windows), which
jointly validates position decode, yaw anchors, and delta accumulation.
