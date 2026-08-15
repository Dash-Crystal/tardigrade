# WIRING LEDGER — registered modules vs what gpu_render.py actually calls

AUTHORITATIVE REGENERATOR: `python -m harness.gpu_render.conformance.wiring`
(26 CALLED / 6 NAME-ONLY / 189 DEAD of 221 registered, ws-1 merged main).
The hand table below predates the tool and UNDERCOUNTED; the tool is the number.
Originally measured by import/call census on merged main (regenerate: grep imports of
passes/, shading/, post/ in gpu_render.py; a term green in the registry says
nothing about dispatch — this table says whether the renderer CALLS it).

| module | terms green | dispatched by gpu_render? | evidence |
|---|---|---|---|
| shading/environment.py | 7 | **YES** (2 import sites, same-code-object A/B) | W2's importable-first construction |
| passes/lighting.py | 4 | **PARTIAL** — cascade_fade + shadow_compare_ref called; cascade_atlas_uv + sun_diffuse conforming-but-unwired (W3, stated) | gt_csm_shadow() call sites |
| passes/weapon.py | 28 | **YES** (1 import site; "the renderer CALLS the registered module — not a parallel copy") | impl-viewmodel 42d9a415 |
| passes/glass.py | 13 | **NO — INLINE DUPLICATE RUNS INSTEAD** (FAM_GLASS branches at :6492/:7210/:7597 are a separate implementation; passes/glass.py never imported) | this census |
| post/chain.py (tonemap, msaa_resolve transitively) | 5+ | YES via chain import; per-call-site verification NOT done | chain.py:88-96 imports |
| post/video_config.py | — (config, not terms) | consumed by capture/config tooling, not the render loop | by design |
| loopfam 27 families (conformance/terms/*) | 119+16 | **NO — none dispatched** ("importable, A/B'd pass modules; connecting them to draw submission is separate work" — impl-loopfam, stated) | their ledger |
| character/eyeball/customglove (branch) | 35 | pending the scripted reapply onto main | impl-character db77732c |

RULE (from three prior incidents — spec-cube-at-zero, atlas.indirect,
the vm depth remap): a term that is green in the registry and unreached in
dispatch is the project's most common defect class. This ledger exists so
"green" and "wired" are never conflated again. Rows move to YES only with a
named call site.

## PER-WIRE DOCTRINE (added after the second substitution)

CALLED says nothing about whether the called code computes the same thing.
Two of the first two inline paths replaced (glass, sao) were SUBSTITUTIONS,
not duplicates — different hash groupings, different falloffs, missing
exponents — sharing one concept in thirteen and three expressions in shape.
Therefore EVERY wire-in reports, as part of its landing:
  1. inline-vs-module divergence (duplicate / substitution / absent),
     with the tie-break always the decompiled GLSL, never either copy;
  2. the torch-callability of every function newly reached (3/3 modules
     integrated so far raised on first tensor call — harness-only green
     is not runnable green);
  3. any smoke row that CANNOT move (a saturating check proves finiteness
     and nothing else — add a row in the band where the expression bends).
