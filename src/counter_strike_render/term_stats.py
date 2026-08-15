"""Runtime output statistics per term: does it REACH PIXELS, and with what.

THE RULE THIS EXISTS FOR
------------------------
The charter's acceptance is two-part. A kernel-level A/B says the expression
matches the reference; that is `conformance/`. This file is the other half --
the form-1 precondition that the term "demonstrably REACHES PIXELS with real
inputs (runtime print of its output statistics -- mean, fraction at identity)".

The failure it is aimed at is specific and this project has hit it repeatedly:
a term that is present, correct, dispatched, and multiplying by 1.0 on every
pixel of every frame. A pixel COUNT cannot see that -- the counters already in
`B2_STAT` report how many pixels a family owned, which is the same number
whether the family changed them or not. `fraction at identity` is the number
that separates "ran" from "did anything", and it is the one that was missing.

WHAT `identity` MEANS HERE
--------------------------
Identity is per-term and must be passed in, because there is no universal one:
a multiplicative term's identity is 1, an additive term's is 0, and a
compositor's is "output equals the input it was handed". Guessing would make
the statistic agree with itself. `before=` gives the compositor form and is
the honest default for a shading entry point, which is handed a colour and
returns one.

NOT GATED. These print on every run that reaches the term, because a
diagnostic you have to remember to enable is a diagnostic that is off on the
run whose result you are about to publish. They are one line per term per run,
not per frame: the accumulator folds frames together and `report()` prints
once at the end.
"""
from __future__ import annotations

import math

_ACC: dict = {}
_ORDER: list = []

# Two values that differ only in the last bits of float32 are the SAME value
# for this purpose -- the question is "did this term move the pixel", not
# "did the arithmetic round identically". 1e-6 is the conformance epsilon.
IDENTITY_TOL = 1e-6


def observe(name, out, *, before=None, identity=None, mask=None,
            reference=""):
    """Fold one batch of a term's output into its accumulator.

    name       term id, e.g. "b2.simple.shade"
    out        the term's output tensor
    before     what it was handed, for a compositor-form identity test
    identity   a scalar identity instead (1.0 multiplicative, 0.0 additive)
    mask       bool tensor selecting the pixels the term actually owned;
               statistics over pixels a term never touched are meaningless
               and would drive `fraction at identity` to 1 for every term
    reference  file:line of the decompiled expression, carried into the print
    """
    import torch

    if out is None:
        return
    if mask is not None:
        while mask.dim() < out.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(out)
        n_px = int(mask.sum())
        if n_px == 0:
            # A term that owned no pixel this batch is recorded as such
            # rather than skipped: "never ran" and "ran and did nothing"
            # are different findings and both must survive to the report.
            a = _acc(name, reference)
            a["batches_empty"] += 1
            return
        sel = out[mask]
    else:
        sel = out.reshape(-1)
        n_px = sel.numel()

    a = _acc(name, reference)
    a["n"] += n_px
    a["sum"] += float(sel.sum())
    a["sumsq"] += float((sel.double() ** 2).sum())
    a["min"] = min(a["min"], float(sel.min()))
    a["max"] = max(a["max"], float(sel.max()))

    if before is not None:
        d = (out - before).abs()
        if mask is not None:
            d = d[mask]
        else:
            d = d.reshape(-1)
        a["n_ident"] += int((d <= IDENTITY_TOL).sum())
        a["max_move"] = max(a["max_move"], float(d.max()))
        a["ident_kind"] = "output == input"
    elif identity is not None:
        d = (out - float(identity)).abs()
        if mask is not None:
            d = d[mask]
        else:
            d = d.reshape(-1)
        a["n_ident"] += int((d <= IDENTITY_TOL).sum())
        a["max_move"] = max(a["max_move"], float(d.max()))
        a["ident_kind"] = f"output == {float(identity):g}"
    else:
        a["ident_kind"] = None
    a["batches"] += 1


def _acc(name, reference):
    if name not in _ACC:
        _ACC[name] = {"n": 0, "sum": 0.0, "sumsq": 0.0,
                      "min": math.inf, "max": -math.inf,
                      "n_ident": 0, "max_move": 0.0, "batches": 0,
                      "batches_empty": 0, "ident_kind": None,
                      "reference": reference}
        _ORDER.append(name)
    elif reference and not _ACC[name]["reference"]:
        _ACC[name]["reference"] = reference
    return _ACC[name]


def report(title="TERM OUTPUT STATISTICS"):
    """One line per term. Prints even when nothing registered, and says so."""
    print(f"\n{title} -- reaches pixels, and with what "
          f"(charter form-1 precondition)", flush=True)
    if not _ORDER:
        print("  no term registered an observation this run. That is not "
              "'all terms fine' -- it is no evidence either way.", flush=True)
        return
    print(f"  {'term':<34s} {'pixels':>12s} {'mean':>10s} {'sd':>9s} "
          f"{'min':>8s} {'max':>8s} {'@identity':>10s}  identity",
          flush=True)
    for name in _ORDER:
        a = _ACC[name]
        if a["n"] == 0:
            print(f"  {name:<34s} {'0':>12s}  NEVER OWNED A PIXEL this run "
                  f"({a['batches_empty']} empty batches) -- unmeasured, "
                  f"NOT measured-as-zero", flush=True)
            continue
        mean = a["sum"] / a["n"]
        var = max(a["sumsq"] / a["n"] - mean * mean, 0.0)
        frac = (a["n_ident"] / a["n"]) if a["ident_kind"] else float("nan")
        fs = "     n/a" if a["ident_kind"] is None else f"{frac:9.4%}"
        print(f"  {name:<34s} {a['n']:>12,d} {mean:>10.5f} "
              f"{math.sqrt(var):>9.5f} {a['min']:>8.4f} {a['max']:>8.4f} "
              f"{fs:>10s}  {a['ident_kind'] or '(none declared)'}",
              flush=True)
        if a["ident_kind"] and frac >= 0.999999:
            print(f"      ^ {name} is at identity on EVERY pixel it owned "
                  f"(max move {a['max_move']:.3e}). It ran; it changed "
                  f"nothing. {a['reference']}", flush=True)
        if a["batches_empty"]:
            print(f"      ^ plus {a['batches_empty']} batches where it owned "
                  f"no pixel at all", flush=True)


def reset():
    _ACC.clear()
    _ORDER.clear()
