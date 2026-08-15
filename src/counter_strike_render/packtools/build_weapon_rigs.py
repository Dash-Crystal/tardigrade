"""Give every weapon SKELETON in the depot a bundle, not every cs1k weapon id.

The cs1k enum has one `knife` id, so an id-driven build covers the two default
knives and leaves each knife SKIN -- which ships its own rig and its own clips
-- with no bundle at all. Enumerating rigs instead closed 44 -> 65 in one pass:
20 knife variants plus healthshot.

The model is located by the file-anchored rule (match the FILE stem at any
depth), never by assuming the directory is named after the rig.
"""
import argparse
import glob
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcs"))

import weapon_ids as W

RIG_DIR = "animation/skeletons/weapons/"
VMDL_EXT = ".vmdl_c"          # SEVEN characters -- stripping 9 truncated every
                              # stem by two and matched nothing, which reads as
                              # "no model ships for this rig".


def covered_rigs(vm_dir):
    have = set()
    for path in glob.glob(os.path.join(vm_dir, "*.pt")):
        if ".clips" in path or os.path.basename(path).startswith("arms_"):
            continue
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if bundle.get("nm_skeleton"):
            have.add(bundle["nm_skeleton"])
    return have


def model_for(index, short):
    hits = [k for k in index
            if k.startswith("weapons/models/") and k.endswith(VMDL_EXT)
            and os.path.basename(k).startswith("weapon_")
            and os.path.basename(k)[:-len(VMDL_EXT)].endswith(short)]
    if not hits:
        return None
    return sorted(hits, key=lambda k: (k.count("/"), len(k)))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--vm-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    index = W.vpk_index(args.game)
    rigs = sorted(k[:-2] for k in index
                  if k.startswith(RIG_DIR) and k.endswith(".vnmskel_c"))
    have = covered_rigs(args.vm_dir)
    missing = [r for r in rigs if r not in have]
    print("%d rigs, %d uncovered" % (len(rigs), len(missing)))

    here = os.path.dirname(os.path.abspath(__file__))
    built, failed = [], []
    for rig in missing:
        short = os.path.basename(rig)[:-len(".vnmskel")]
        vmdl = model_for(index, short)
        if vmdl is None:
            failed.append((short, "no weapon_*%s%s under weapons/models/"
                           % (short, VMDL_EXT)))
            continue
        proc = subprocess.run(
            [args.python, os.path.join(here, "extract_viewmodel.py"),
             "--vmdl", vmdl, "--game", args.game,
             "--name", short, "--out-dir", args.vm_dir],
            capture_output=True, text=True)
        if proc.returncode == 0:
            built.append(short)
        else:
            tail = (proc.stderr or proc.stdout).strip().split("\n")[-1]
            failed.append((short, tail[:90]))

    print("BUILT %d: %s" % (len(built), built))
    for name, why in failed:
        print("   NOT BUILT %-18s %s" % (name, why))


if __name__ == "__main__":
    main()
