"""In-process batch extraction: amortize what the per-process sweep repaid
13,000 times.

The GT engine cold-loads a map in seconds; a pack build that takes hours is
a deserialization-strategy defect, not a format cost. The per-process sweep
paid (python start + torch import + full 137k-entry vpk index parse) PER
MODEL -- 10-30x the actual decode work. This runner pays each cost ONCE per
worker: multiprocessing Pool whose initializer imports the extractor and
MEMOIZES the VPK index (class-level cache keyed by path), then calls the
extractor's own main(argv) in-process per model. Same converter, same
output, same logs -- the overhead is amortized, not the work changed.

Usage: python extract_batch_inprocess.py --game G --list all_vmdl.txt \
           --out DIR [--workers 12]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_G = {}


def _init(game, out):
    import vpk
    # memoize the index: ONE parse per worker for the whole run
    if not hasattr(vpk.VPK, "_memo"):
        real = vpk.VPK

        class MemoVPK:
            _cache = {}

            def __new__(cls, path):
                if path not in cls._cache:
                    cls._cache[path] = real(path)
                return cls._cache[path]
        MemoVPK._memo = True
        vpk.VPK = MemoVPK
    import extract_playermodel as ep
    _G["ep"] = ep
    _G["game"] = game
    _G["out"] = out


def _one(ref):
    ep, game, out = _G["ep"], _G["game"], _G["out"]
    name = ref.replace("/", "__").replace(".vmdl_c", "")
    dst = os.path.join(out, name + ".pt")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return ("skip", ref)
    log = os.path.join(out, name + ".log")
    # PER-WORKER SCRATCH: the extractor uses a FIXED intermediate filename
    # (.pm_slot.vtex_c) inside its out-dir; parallel workers sharing one
    # out-dir race on it (measured: 148 FileNotFoundError fails). Each
    # worker extracts into its own subdir and promotes the pack up.
    wout = os.path.join(out, f".w{os.getpid()}")
    os.makedirs(wout, exist_ok=True)
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ep.main(["--game", game, "--vmdl", ref, "--allow-stub",
                     "--out-dir", wout, "--name", name])
        wdst = os.path.join(wout, name + ".pt")
        if os.path.exists(wdst) and os.path.getsize(wdst) > 0:
            os.replace(wdst, dst)
        ok = os.path.exists(dst) and os.path.getsize(dst) > 0
    except SystemExit as e:
        wdst = os.path.join(wout, name + ".pt")
        if os.path.exists(wdst) and os.path.getsize(wdst) > 0:
            os.replace(wdst, dst)
        ok = (not e.code) and os.path.exists(dst)
    except Exception as e:                                  # noqa: BLE001
        buf.write(f"\n{type(e).__name__}: {e}\n")
        ok = False
    with open(log, "w") as fh:
        fh.write(buf.getvalue())
    return ("ok" if ok else "fail", ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    refs = [ln.strip() for ln in open(a.list) if ln.strip()]
    os.makedirs(a.out, exist_ok=True)
    counts = {"ok": 0, "fail": 0, "skip": 0}
    with mp.Pool(a.workers, initializer=_init,
                 initargs=(a.game, a.out)) as pool:
        for i, (st, ref) in enumerate(pool.imap_unordered(_one, refs, 8)):
            counts[st] += 1
            if i % 100 == 0 or st == "fail":
                print(f"[{i+1}/{len(refs)}] {counts} last={st}:{ref}",
                      flush=True)
    print(f"DONE {counts}", flush=True)


if __name__ == "__main__":
    main()
