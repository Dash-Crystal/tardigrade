# Top-10 slowest host-side python loops in the renderer — and the scatter/gather rewrites

Date: 2026-08-11, deadline report (17:30 PT). Branch `refactor-decimate`.
Instrument: `tools/hostloop_bench.py` — each loop's exact op pattern at the
tard21 sweep's REAL sizes (window B=48, 10 entities/frame, ctm_sas bundle
9,899 verts / 13,504 tris, 2,796 frames/video, 640×360), loop vs rewrite,
numerically equivalence-checked, CPU torch 2.13.

**Why CPU is the right instrument for HOST loops, and what it cannot see:**
host-side loop cost is python dispatch — launching many small tensor ops from
an interpreted loop. The kernels launched are identical either way, so the
CPU ratio measures the dispatch overhead itself and is a FLOOR for the GPU
win: on GPU the same python loops additionally SERIALIZE kernel launches and
several force a host↔device sync per iteration (`.item()`, f-string reads of
tensor scalars, per-level `bool(x.any())`), which stalls the whole pipeline.
Rows marked `GPU-sync` have their main cost only on a GPU node; end-to-end
fps re-measurement there is the follow-up (H200 released today).

Baseline (measured, p4_ego full-match render): `render.raster_shade` 155.5 s
for 2,527 frames = **16 fps / 3.7 Mpx/s** at 640×360, window 48, chunk 32.

| # | site (fn @line) | what the loop does | iterations | measured loop→vect (CPU) | technique |
|---|---|---|---|---|---|
| 1 | `playermodel_pass` @32080 | per-frame `pm_place` + 2× `pm_rotate_dirs` calls, then `torch.stack` — 3·B small launches per window per team | 3·48·teams /window | **78.0 → 20.4 ms, ×3.8** | ONE batched quaternion-rotate over `(B,NE,NV,3)` — broadcast, no per-frame calls |
| 2 | `render_window` @34598 (smoke) | `torch.linalg.inv(mvp[b])` per frame in python | 48 /window | **0.32 → 0.05 ms, ×6.1** | batched `linalg.inv` over the whole window |
| 3 | `playermodel_pass` @32298 | per-frame per-entity report f-strings reading tensor SCALARS (`e['x']:.1f`, `ent_px[i,j]`) — each read is a device sync on GPU | 48·10 /window | **2.97 → 0.35 ms, ×8.6** (GPU-sync: floor) | bulk `.tolist()` once, format from python floats — one sync per window instead of 480 |
| 4 | `pm_entities_for_frame` @31950 | python dict filter over rows per frame | 10·2,796 /video | 0.02 → 0.01 ms, ×1.7 (CPU-cheap) | per-video gather tables (pos/yaw/alive tensors), index per frame; wins on GPU where the dicts feed device uploads per frame |
| 5 | `playermodel_pass` @32097 | `torch.cat([tri1 + i*nv for i in range(nslots)])` | nslots /window | **0.39 → 0.04 ms, ×9.2** | broadcast add `tri1[None] + (arange·nv)[:,None,None]` |
| 6 | `render_window` @32398 (LOD) | per-level boolean mask + gather + cat (11 fancy-index gathers, 10 cats per chunk) | 6 levels /chunk | 1.01 → 0.74 ms, ×1.4 | single `torch.isin` gather over the level table; bigger win is not re-materializing per chunk |
| 7 | `water_fancy_shade` @22899 | sequential SSR march steps in python | 8 /frame | **0.1× — vectorized is SLOWER on CPU** (8× working set, memory-bound) | parallel `(n_iter,H,W)` steps + `argmax` is a GPU-bandwidth technique; on CPU keep the loop. Config-driven backend per project law — no silent auto-gate |
| 8 | `_muzzle_flash_over` @30308 | per-blocked-flash composite | ≤14 /frame | 1.37 → 1.31 ms, ×1.0 (GPU-sync) | `index_add_` scatter; CPU-neutral, on GPU removes k launches + syncs |
| 9 | `render_window` @35482 | per-frame d2h + shape check + accounting | 48 /window | 11.9 → 11.6 ms, ×1.0 (bandwidth-bound) | one batched d2h; win is one pinned copy per window on GPU (already the .pt sink's design: 0.29 vs 76.9 ms/frame measured earlier) |
| 10 | `occlusion_refine` @11576 | pyramid levels each doing `float(x.mean())` + `bool(alive.any())` — 2 host syncs per level | 6·2 syncs /chunk | 0.11 → 0.09 ms, ×1.2 (GPU-sync) | defer all reductions, one sync at the end; levels stay sequential, the SYNCS don't |

## The three findings that matter

1. **The #1 host cost is per-frame *calls*, not per-frame *math*.** The
   playermodel placement loop issues ~144 small launches per window per team;
   batched it is one. Projected per full-match video (58 windows): ~4.5 s of
   pure host dispatch removed on CPU alone — more on GPU where each launch
   serializes.
2. **String-formatting from tensor scalars is a hidden sync storm** (row 3):
   every `f"{e['x']:.1f}"` on a CUDA tensor is a device→host read. 480 of
   them per window. `.tolist()` first = one transfer. This preserves the
   per-entity frame lines (the project's loud-print law) at 1/8.6 the cost.
3. **Vectorization is not free-standing doctrine** (row 7): the SSR parallel
   march LOSES on CPU (memory-bound). The rewrite is correct for the GPU
   backend only — exactly the scale-banked, config-driven backend split the
   project already mandates (`config-driven-backends-not-latency-chasing`).

## Honest scope

- Ratios are CPU-measured floors for dispatch cost; rows tagged GPU-sync are
  argued from sync counts, not measured on GPU (H200 returned today). The
  re-measurement recipe: rerun `tools/cs_match_three_view_videos.py render`
  on any torch+nvdiffrast node, read `STAGE JSON render.raster_shade`.
- The 3,379-line `render_window` body (1,018 tensor-op sites at module level
  of one function) is itself the structural cause: the decimation pass
  (this branch) moves each stage behind a batched interface so these loops
  stop being writable in the first place.
- Rewrites 1/2/3/5 are drop-in (same outputs, equivalence-checked here);
  6/8/9/10 are neutral-on-CPU and should land with the GPU re-measurement;
  7 lands as the GPU arm of a config-driven backend split.

## Final dispositions (static-source refactor complete, 2026-08-11 evening)

| # | disposition |
|---|---|
| 1 | **LANDED** `8ac80d0c` — pm_place_batch/pm_rotate_dirs_batch, bitwise-equal (max|Δ|=0.0), ~3,840 ops→~18 launches/window/team |
| 2 | **LANDED** `345201f5` — batched chunk inverse, max|Δ|=0.0 |
| 3 | **LANDED** — the real storm was `int(tensor.sum())` per (frame,entity) IN THE RENDER PATH (480 syncs/window/team), not just the f-strings: one bincount + one `.tolist()`, exact-equal verified |
| 4 | **SUBSUMED by row 1** — remaining per-frame work is trivial host bookkeeping (clip-phase is inherently per-frame); the heavy downstream was the placement, now batched |
| 5 | **LANDED** `345201f5` — tri broadcast, exact integer equality |
| 6 | **DEFERRED to stage promotion** — the per-level rebuild is a table-layout restructure, not a loop swap; belongs to 80_render_window's promotion with its own byte-count budget |
| 7 | **GPU-arm at stage promotion** — parallel march LOSES on CPU (measured ×0.1); lands as the config-driven GPU backend of 55_water, per the no-silent-auto-gate law |
| 8 | **LANDED (corrected)** — the real loop is a report-print sync storm (4 tensor-scalar reads per flash): bulk `.tolist()` hoist. One residual one-line site (`_g`) outside the guarded block, k≤14 once/window, noted |
| 9 | **ALREADY-BY-DESIGN** — the .pt session sink IS the batched d2h (0.29 vs 76.9 ms/frame, measured earlier); the mp4 path's per-frame loop is the encoder's own interface |
| 10 | **WITHDRAWN** — the real occlusion_refine is already fully vectorized (batched einsum + max-pool pyramid); the benchmark modeled a strawman. Corrected per the retraction discipline |

Static totals landed: per 48-frame window per team, host-side tensor-op
launches in the playermodel path fall ~4,300 → ~25 and device→host syncs
~480 → 1; the smoke path drops B inverse launches to 1. All landed rewrites
are exact-equal or ≤1e-6 verified on CPU at real sizes; the loader runs
(--help RC=0, 4,408 lines) after every change.

## Addendum: the "outside the map" investigation (serialization EXONERATED, D10 CONVICTED)

Two-chains decode audit vs upstream read_v2 on the same .tard: viewangles
agree to millidegrees (anchors applied, integer cumsum -- D4 n/a); positions
differ <=27 units = one 16 Hz source interval (upstream v22 "frame-perfect"
interpolates with teleport-holds, ours causal-holds; neither fabricates).
Bounds census: 0 of 81,219 rows outside generous bounds. The defect is
upstream D10, live: the map_name header says de_mirage; the trajectory bbox
x[-1676,2562] y[-684,3461] FITS de_inferno (+369 slack all sides) and
violates de_mirage's overview by -1748. Correct data, wrong world. Fix:
map identity is now FIT FROM THE DATA with refusal on mismatch; header
demoted to a comment, as the format's own D10 report always said.
