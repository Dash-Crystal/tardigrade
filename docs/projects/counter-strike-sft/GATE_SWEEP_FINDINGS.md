# Structural sweep: computed-under-one-gate, consumed-under-another

Ordered after `df4632a4`, on the reasoning that a defect which fired on every
bare invocation the project had ever run, silently, is unlikely to be the only
one of its shape. Tool: `tools/analysis/gate_sweep.py`.

## Result

**171 files swept. 15 candidate sites. Zero surviving instances of the
family.** Every one of the 15 reads as intended dataflow when opened.

That is a negative result, and it is only worth anything because the scan is
calibrated to find the thing it is looking for. `--calibrate` runs both halves
against git and exits nonzero on either failure:

```
$ python3 tools/analysis/gate_sweep.py --calibrate
PASS  finds the known defect at df4632a4^ (2 of its terms at lines 29996/29997)
PASS  the fixed weather block is clear at HEAD
```

Both halves are required. Finding the defect alone is satisfied by a scan that
always fires; clearing it alone is satisfied by one that never does.

## What the scan looks for

```
if OUTER:                                   # the feature's own gate
    _n = <compute for every pixel>          # unrestricted result
    if SCOPE:                               # "which pixels may I touch"
        _n = torch.where(mask, _n, nrm)     # RESTRICT to that subset
    nrm = _n                                # PUBLISH -- outside SCOPE
```

With `SCOPE` false there is no mask, so the unrestricted result is published to
every pixel. No exception, no shape change.

**The discriminator is the else-arm**, and it is what takes the report from 168
rows to 14. A restriction whose else-arm preserves *the publish target's own
prior value* is the only thing standing between that write and a global
clobber. A restriction whose else-arm is the temp itself is additive — the
pre-gate value is already the correct default, and skipping the gate is a
defined no-op. Identical syntax, opposite meaning.

## Two wrong versions came first

Recorded because the failure mode is more reusable than the result.

| version | what it compared | outcome |
|---|---|---|
| v1 | gate a term is WRITTEN under vs. gate it is READ under | 1,358 sites, and the known defect was **not among them** — the read was never the problem |
| v2 | right shape, no else-branch reasoning | reported the **fixed** weather block as still broken |

v1 is the important one. It was a plausible, well-formed scan producing a large
confident report about a defect family it structurally could not detect. The
only thing that caught it was running it against the known instance first.

v2's failure matters differently: a sweep that re-reports its own lane's fixes
trains everyone to ignore it.

## The 15 sites, and why each is clear

| site | why it is not the defect |
|---|---|
| `gpu_render.py:7812,7813` (`_am`, `_ac`, 6 rows) | the publish is *itself* a masked select preserving `face_mode0`/`face_cut0` — already restricted |
| `gpu_render.py:14429` (`_smoothstep_ab`) | the `torch.is_tensor(d)` branch has an `elif abs(d) < 1e-9` companion; **both** paths guard the divide. The scan cannot see an `elif` as the other half of a guard |
| `gpu_render.py:15090` (`_sh = ambient`) ×2 | the probe and indirect blocks updating `ambient` *is* the intended dataflow; they preserve non-hit pixels, which is correct, not accidental |
| `gpu_render.py:28197` (`w_eff`) | `w_raw` is the blend weight, legitimately updated by vcmode then perturbed by height-blend where `use` selects |
| `gpu_render.py:30132` (`_n`) ×2 | this line *is* the post-fix weather restriction; the scan paired it with unrelated restrictions 1,100 lines earlier |
| `gpu_render.py:30457,30458` (`_sfrgb`, `_sfa`) | reads of `alb`, which the mask block legitimately updates |
| `ash_to_world.py:1214` (`q = uvs[fl]`) | `uvs` sanitised under `n_uvbad` and read after — intended |

## Limits — read a clean run against these, not as a proof

- **Restriction by indexed assignment** (`x[m] = v`) is a different shape and is
  not caught. Only select-style masking is.
- **Aliases deeper than one hop.** At `df4632a4` the scan finds `alb` and `nrm`
  but not `rough`/`ao_o`, whose restriction folds against `_rgh_in` — bound by a
  *conditional expression*. The site is still reported via its other two terms,
  so what the calibration demonstrates is **site** recall; per-term recall
  inside a site is partial.
- **Python only.** The `.metal` and `.glsl` sources are not parsed, and this
  defect shape is equally possible there. That is the largest uncovered
  surface and it is not small.
