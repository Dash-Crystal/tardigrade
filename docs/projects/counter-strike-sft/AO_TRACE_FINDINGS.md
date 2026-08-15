# The AO trace: two defects, both confirmed, neither endorsed by score

Record for a change whose CODE landed in `04b419d4` — a sweep of the shared
working tree by another lane — and whose evidence therefore never reached a
commit message. The code in HEAD is correct and complete; this is the
reasoning and the measurements that belong with it.

Lanes: corpus-2 found it, w3-lighting-2 supplied the reference semantics and
the `simple_dispatch` lead, feature-writer (me) held the falsifier dispatch.

## Defect 1 — `--dirocc` discarded the material AO page (`b6d28df3`)

**Branch**: `_post_chain`, `if dirocc is not None: gao = dirocc.clamp(min=0)`.
It *replaces* `gao`, so the AO page sampled and multiplied into the indirect
was thrown away one assignment later. The page loaded, `surface.ao_page`
reported mean 0.996609 over [0.1176, 1] at 21.6% coverage, and none of it
reached a pixel — corpus-2's byte-identical renders.

Located without a render: the run's own log said
`AO: offset-76 directional-occlusion target in use (--dirocc); --ssao/--ao
are the fitted stand-ins for this same slot and are NOT additionally applied`.
That note is right about `--ssao`/`--ao` and wrong that the material page is
one of them.

**The reference** — `csgo_environment_ps.glsl:1229`:

    vec3 _20418 = vec3(_21714 * _17119).xyz;
    vec3 _20015 = _8745 * _20418;            // ambient diffuse

The indirect scale is a **product**, not a slot with alternatives:

| factor | site | what it is |
|---|---|---|
| `_21714` | :793 | screen-space weighted fetch — the offset-76 resolve; :797 sets it to 1.0 when absent, which is exactly the slot `--dirocc`/`--ssao` fill |
| `_17119` | :732/:742 | `mix(_14713, 1.0, …)` / `_14713`, and `_14713` (:514) is the MATERIAL occlusion |

Screen-space and material occlusion multiply. Gated on `AO_PAGES` rather than
`ao is not None`, because `ao` carries two different things — the material
page when the pack ships one, the fitted `--ao` aux stand-in otherwise. The
stand-in *is* a `_21714` substitute and must keep not double-applying.

**Stated difference in source, not role**: at :514 the reference's material
occlusion is the albedo texel's ALPHA (`_22057.w`); ours is `mat_ao_pages`
(`g_tAmbientOcclusion`). Same position in the product, same role, different
channel — that is what our pack carries.

## Defect 2 — `csgo_simple` overwrote the page with an unbound slot's identity

w3-lighting-2's `simple_dispatch` lead, confirmed with a number. **It is not
the shape the lead guessed**: `simple_dispatch` *does* forward `ao_o` — takes
it and returns it. What it does is blend over it unconditionally:

    out_ao = out_ao * (1 - mf) + o * mf

and `o` comes from `simple_surface`, which reads csgo_simple's own
`g_tAmbientOcclusion` (:205, :710) and substitutes 1.0 where the material
binds none: `ao = aotex.x * have_ao + (1 - have_ao)`.

Right in isolation, wrong as a fold. **Measured on de_inferno f700**:
`simple.ao_texture_bound` is **0** across all **147,264** of that family's
pixels — ~16% of a 1280×720 frame — so every one overwrote the incoming AO
with a constant 1.0, including the page defect 1 had just rescued.

1.0 is not a value here; it is the **identity standing in for an absent
texture**, and an identity must not displace a value the caller already
holds. `simple_surface` now returns `have_ao` — "this family's slot said 1.0"
and "this family has no slot" are the same float and different facts — and
the fold applies only where the slot is bound. Where it *is* bound the family
still wins: it is the more specific read of the same physical slot.

`simple.ao_slot_displaced_page` counts it: **147,264 of 147,264** on f700. A
run where the slot is bound reads 0, making the fix a visible no-op rather
than an invisible one.

## w3's single-frame check, settled

w3 flagged that their cross-frame comparison could not establish this. On ONE
frame, one run, after both fixes:

    surface.ao_page             mean 0.996609  [0.1176, 1]  at-identity 0.9656
    compose.ao_page_in_product  mean 0.996609  [0.1176, 1]  at-identity 0.9656
    compose.ao                  mean 0.996264  [0.1175, 1]  at-identity 0.7408

The page's distribution now arrives in the product **exactly**. Before defect
2 was fixed the same two terms read 0.996609 vs 0.998222 (at-identity 0.9656
vs 0.9748) — the family's 1.0 flattening the page's spread on 16% of the
frame. Their hypothesis was right and their caveat was the right caveat.

Also from w3, and it closed a branch of the falsifier before it was run: the
shader multiplies AO into indirect diffuse (:1230), direct specular (:940)
and env specular (:945) — **never direct diffuse**, whose accumulator
`_16324` is untouched at :1240. With `baked.irradiance` 0.410 against
`compose.direct_diffuse` 0.431, the indirect is ~49% of the composite, so
"the indirect is near zero and AO has nothing to occlude" was already dead.

## The falsifier

`--ao-page-product {on,off,zero}`, DIAGNOSTIC, announces its arm. f700,
`de_inferno_ao.pt`, one tree:

| arm | md5 | nrmse | kl_rgb | ncc |
|---|---|---|---|---|
| off (pre-fix discard) | `869182ae0cad973418b8989778cbfd1b` | 1.386977 | 2.133382 | 0.0778 |
| on (defect 1 fixed) | `5dbafa3132c14136c9f7c0748487b0ae` | 1.383579 | 2.132974 | 0.0759 |
| both fixed | `c64feefc12f764b777f430928eae9eb5` | 1.384088 | 2.130476 | 0.0716 |
| zero (falsifier) | `3b6493d4653a16726495ce837a629c41` | 2.183899 | 7.777188 | 0.0114 |

`off` reproduces corpus-2's byte-identical hash **exactly** — `869182ae` is
the md5 they reported for both packs — so the arm is a faithful control and
the named branch is provably the discard. `zero` collapses `compose.ao` to a
constant 0 and moves nrmse 1.384 → 2.184, so the term genuinely reaches
pixels. `compose.ao`'s range says the same: [0.984, 1] under `off`, where
dirocc alone barely occludes, against [0.1175, 1] with the page arriving.

## ⚠️ The metrics do not endorse either fix

Against `off`, both together move nrmse −0.21% and kl −0.14% while **ncc goes
the wrong way, 0.0778 → 0.0716**. Against `on` alone, nrmse is slightly
worse. By the triplet rule neither correction is validated by score.

Both stand on **reference reads** — :1229 is a product; an identity is not a
value — and on counters proving they reach pixels. Not on the number.

One caveat on reading that ncc at all: it is 0.07 in absolute terms, and this
project has already measured paired ncc landing at unrelated-frame ncc.
Differences of 0.006 at that level should not be read as signal in either
direction without a falsifier of their own.
