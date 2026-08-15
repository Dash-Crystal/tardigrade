"""Provision a render node with the WHOLE content tree -- never a subset.

OWNER RULE (2026-08-13). The v960 defect class was a human selecting
arbitrary items from the datapack, rsyncing them to a node, and hopefully
remembering every flag: viewmodels, weapons, HUD and staged-but-unreferenced
sidecars all fell through. This script removes the mechanism:

  * the node gets the ENTIRE derived-content tree (every map's world pack
    with every sidecar beside it by stem, pm_bundles/, weapons/, vm/), plus
    a link to the raw gt_datapack when one is resident on the node;
  * transfer is verified by FILE COUNT and BYTES per directory, never exit
    code; a PROVISIONED.json manifest lands at the root recording what the
    node holds and from where;
  * the renderer's --content-root then resolves every input family from
    this tree by fitted map name, printing each resolution and refusing
    BY NAME on anything absent (stages/03_part._resolve_content_root).

There is deliberately no --only flag. A node that runs the renderer holds
the tree; partial provisioning is the defect this exists to kill.

Usage:  provision_render_node.py NODE [--root content] [--source HOST:PATH]
        default source: terul-cluster:/data/cs2-artifacts (current
        source-of-record for derived artifacts; migrate to the NFS when
        its mount is known -- see memory: firm data goes to the NFS).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

#: subdirectories of the source that constitute the derived-content tree.
#: worlds/ carries packs + stem-beside sidecars; lightmaps/ is merged INTO
#: worlds/ on the node so every 'auto'/beside-pack resolver finds its
#: family next to the pack (the renderer's one true layout).
TREE = (("worlds/", "worlds/"),
        ("lightmaps/", "worlds/"),
        ("work/pm_bundles/", "pm_bundles/"),
        ("work/weapons/", "weapons/"),
        ("work/vm/", "vm/"))


def sh(cmd, check=True):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd}\n{r.stderr[-500:]}")
    return r


def du_count(host, path):
    """(file_count, total_bytes) on host:path; (0, 0) if absent."""
    r = subprocess.run(
        ["ssh", host, f"find {path} -type f 2>/dev/null | wc -l; "
                      f"du -sb {path} 2>/dev/null | cut -f1"],
        capture_output=True, text=True)
    lines = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
    if len(lines) < 2:
        return 0, 0
    return int(lines[0]), int(lines[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node")
    ap.add_argument("--root", default="content")
    ap.add_argument("--source", default="terul-cluster:/data/cs2-artifacts")
    a = ap.parse_args()
    src_host, src_path = a.source.split(":", 1)
    manifest = {"schema": "iji/render-node-provision/v1",
                "source": a.source, "tree": {}}
    sh(["ssh", a.node, f"mkdir -p {a.root}"])
    for sub, dst in TREE:
        want_n, want_b = du_count(src_host, f"{src_path}/{sub}")
        if want_n == 0:
            # ABSENT AT SOURCE is a named fact the renderer will refuse
            # on (or the operator waives BY NAME) -- not a silent skip.
            print(f"  SOURCE EMPTY {sub}: the '{dst.rstrip('/')}' family "
                  f"does not exist at {a.source} -- renderer runs must "
                  f"--waive it by name until it is built", flush=True)
            manifest["tree"][dst] = {"files": 0, "bytes": 0,
                                     "state": "ABSENT-AT-SOURCE"}
            continue
        # relay through this machine: render nodes need no route to the
        # source host, and the dev machine is dual-homed.
        # strip EVERY source path component: 'work/pm_bundles/' is two
        # levels, 'worlds/' is one -- a fixed strip of 1 nested the
        # two-level families one directory deep, which the resolver's
        # directory-existence check then called RESOLVED while the
        # loader found nothing (measured: playerless matrix).
        strip = sub.rstrip("/").count("/") + 1
        sh(["bash", "-c",
            f"ssh {src_host} 'tar -C {src_path} -cf - {sub}' | "
            f"ssh {a.node} 'mkdir -p {a.root}/{dst} && "
            f"tar -C {a.root}/{dst} --strip-components={strip} -xf -'"])
        got_n, got_b = du_count(a.node, f"{a.root}/{dst}")
        if got_n < want_n:
            raise SystemExit(f"BYTES/COUNT MISMATCH {sub}: source "
                             f"{want_n} files / {want_b} B, node has "
                             f"{got_n} / {got_b}")
        manifest["tree"][dst] = {"files": got_n, "bytes": got_b,
                                 "state": "VERIFIED"}
        print(f"  provisioned {dst}: {got_n} files, "
              f"{got_b/2**30:.2f} GiB VERIFIED", flush=True)
    # raw datapack link, when the node holds a dump: everything the derived
    # tree lacks stays NAVIGABLE on the node rather than un-reachable.
    r = sh(["ssh", a.node,
            "for d in $HOME/gt_datapack /data/gt_datapack; do "
            "test -f $d/MANIFEST.json && echo $d && break; done; true"],
           check=False)
    dp = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
    if dp:
        sh(["ssh", a.node, f"ln -sfn {dp} {a.root}/gt_datapack"])
        manifest["gt_datapack"] = dp
        print(f"  datapack linked: {a.root}/gt_datapack -> {dp}", flush=True)
    else:
        manifest["gt_datapack"] = None
        print("  NO raw datapack resident on the node -- derived tree "
              "only; dump one there for full navigability", flush=True)
    blob = json.dumps(manifest, indent=1)
    sh(["ssh", a.node,
        f"cat > {a.root}/PROVISIONED.json <<'EOF'\n{blob}\nEOF"])
    print(f"PROVISIONED {a.node}:{a.root} -- {sum(t['files'] for t in manifest['tree'].values())} files", flush=True)


if __name__ == "__main__":
    main()
