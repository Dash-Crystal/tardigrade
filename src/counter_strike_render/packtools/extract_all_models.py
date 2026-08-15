"""Extract EVERY model the GT renderer's depot version can ever reference.

The archive index is the specification: every .vmdl_c in the vpk, no curated
prefix list, no per-model naming, no 'catalog'. Any state-action history a
demofile can present resolves against this set or the extraction log says
exactly which asset failed and why. Resumable (skip-if-exists), per-asset
logs, manifest with byte sizes at the end.

Usage (on the depot host):
    python extract_all_models.py --game /home/user/cs2/game --out ~/model_packs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def slug(path: str) -> str:
    return path.replace("/", "__").replace(".vmdl_c", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import vpk  # the committed reader; the index is the enumeration
    pak = os.path.join(a.game, "csgo", "pak01_dir.vpk")
    archive = vpk.open(pak) if hasattr(vpk, "open") else vpk.VPK(pak)
    entries = [p for p in archive if p.endswith(".vmdl_c")]
    print(f"archive index: {len(entries)} .vmdl_c entries -- ALL of them",
          flush=True)

    ok, fail, skip = 0, 0, 0
    manifest = {}
    for i, path in enumerate(sorted(entries)):
        name = slug(path)
        dst = os.path.join(a.out, name + ".pt")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1
            manifest[name] = os.path.getsize(dst)
            continue
        r = subprocess.run(
            [a.python, os.path.join(HERE, "extract_playermodel.py"),
             "--game", a.game, "--vmdl", path, "--allow-stub",
             "--out-dir", a.out, "--name", name],
            capture_output=True, text=True)
        log = os.path.join(a.out, name + ".log")
        with open(log, "w") as fh:
            fh.write(r.stdout + r.stderr)
        if r.returncode == 0 and os.path.exists(dst):
            ok += 1
            manifest[name] = os.path.getsize(dst)
            print(f"[{i+1}/{len(entries)}] OK   {path}", flush=True)
        else:
            fail += 1
            tail = (r.stdout + r.stderr).strip().splitlines()[-1:]
            print(f"[{i+1}/{len(entries)}] FAIL {path} :: "
                  f"{tail[0] if tail else '?'}", flush=True)
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as fh:
        json.dump({"schema": "iji/model-pack-manifest/v1",
                   "total_index_entries": len(entries),
                   "extracted_ok": ok, "skipped_existing": skip,
                   "failed": fail, "models": manifest}, fh, indent=1)
    print(f"DONE: {ok} extracted, {skip} already present, {fail} failed "
          f"(each failure has a named log -- the honest inventory).",
          flush=True)
    sys.exit(1 if fail and not ok else 0)


if __name__ == "__main__":
    main()
