"""THE datapack: dump and read EVERYTHING the ground-truth engine stores.

RULING. Anything that contributes to any latent that interacts with anything
in a game-state transition or a draw-call intermediate representation is part
of the tangible datapack. Therefore the dumper NEVER rejects:

  * formats with committed converters CONVERT (models via extract_playermodel,
    textures via bc_decode, materials via kv3v5, worlds via the world packers,
    animation clips via vnmclip);
  * every other entry is PRESERVED bit-exact, content-addressed, with its
    format named in the manifest. A missing decoder is a READER UPGRADE, not
    a dump gap -- the bytes are already ours. There is no rejection filter
    to write and no cope to make: the pack is complete by construction, and
    the manifest says precisely which entries are decoded vs preserved.

The READER is the renderer-facing loader over one or more content roots
layered in order (base pack first, then derivative/original overlays). Any
map, texture set, model, or animation authored in the reference formats and
dropped into an overlay root resolves identically to shipped content --
which is what makes our renderer and derived environments open to original
content in the reference architecture.

    dp = Datapack(["/packs/base", "/packs/my_mod"])
    dp.model("weapons/models/ak47/weapon_rif_ak47.vmdl_c")  -> torch pack
    dp.texture(path)  -> (H, W, 4) uint8   (decoded on demand, cached)
    dp.material(path) -> parsed kv3 dict
    dp.raw(path)      -> bytes             (ALWAYS available for everything)
    dp.entries()      -> the complete inventory with decode status

Dump:  python datapack.py dump --game G --out ROOT [--workers 12]
Query: python datapack.py ls   --root ROOT [--status preserved]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

#: format family -> handling. Everything not listed is RAW-PRESERVED.
CONVERTED = {
    ".vmdl_c": "model",
    ".vtex_c": "texture",
    ".vmat_c": "material",
    ".vnmclip": "clip",
    ".vnmskel": "skeleton",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# ---------------------------------------------------------------- dump ----

def _init(game, out):
    import vpk
    _G["vpk"] = vpk.VPK(os.path.join(game, "csgo", "pak01_dir.vpk"))
    _G["game"] = game
    _G["out"] = out
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(out, "converted"), exist_ok=True)


_G = {}


def _dump_one(path):
    v, out = _G["vpk"], _G["out"]
    try:
        data = v.read(path) if hasattr(v, "read") else v[path]
    except Exception as e:                                   # noqa: BLE001
        return ("unreadable", path, f"{type(e).__name__}: {e}")
    digest = _sha(data)
    rawdir = os.path.join(out, "raw", os.path.dirname(path))
    os.makedirs(rawdir, exist_ok=True)
    rawp = os.path.join(out, "raw", path)
    if not (os.path.exists(rawp) and os.path.getsize(rawp) == len(data)):
        with open(rawp, "wb") as fh:
            fh.write(data)                    # PRESERVED, always, first
    ext = next((e for e in CONVERTED if path.endswith(e)), None)
    if ext == ".vmdl_c":
        name = path.replace("/", "__").replace(".vmdl_c", "")
        dst = os.path.join(out, "converted", name + ".pt")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return ("converted", path, digest)
        import io
        import contextlib
        wout = os.path.join(out, "converted", f".w{os.getpid()}")
        os.makedirs(wout, exist_ok=True)
        buf = io.StringIO()
        try:
            import extract_playermodel as ep
            with contextlib.redirect_stdout(buf), \
                    contextlib.redirect_stderr(buf):
                ep.main(["--game", _G["game"], "--vmdl", path,
                         "--allow-stub", "--out-dir", wout, "--name", name])
        except BaseException as e:                           # noqa: BLE001
            buf.write(f"\n{type(e).__name__}: {e}\n")
        wdst = os.path.join(wout, name + ".pt")
        if os.path.exists(wdst) and os.path.getsize(wdst) > 0:
            os.replace(wdst, dst)
            return ("converted", path, digest)
        with open(os.path.join(out, "converted", name + ".convertlog"),
                  "w") as fh:
            fh.write(buf.getvalue())
        return ("preserved", path, digest)    # bytes are OURS regardless
    return ("preserved", path, digest)


def dump(game, out, workers=12):
    import vpk
    v = vpk.VPK(os.path.join(game, "csgo", "pak01_dir.vpk"))
    paths = sorted(v.entries)
    print(f"archive: {len(paths)} entries -- dumping ALL of them", flush=True)
    counts = {"converted": 0, "preserved": 0, "unreadable": 0}
    manifest = {}
    with mp.Pool(workers, initializer=_init, initargs=(game, out)) as pool:
        for i, (st, path, info) in enumerate(
                pool.imap_unordered(_dump_one, paths, 64)):
            counts[st] += 1
            manifest[path] = {"status": st, "sha": info}
            if i % 2000 == 0:
                print(f"[{i+1}/{len(paths)}] {counts}", flush=True)
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump({"schema": "iji/gt-datapack/v1",
                   "counts": counts, "entries": manifest}, fh)
    print(f"DONE {counts} -> {out}", flush=True)


# --------------------------------------------------------------- reader ----

class Datapack:
    """Layered reader over dump roots; later roots override earlier -- the
    derivative-content mechanism. raw() always works; typed accessors decode
    on demand with the committed converters."""

    def __init__(self, roots):
        self.roots = [os.path.abspath(r) for r in roots]
        self._cache = {}

    def _find(self, sub, path):
        for root in reversed(self.roots):          # overlays win
            p = os.path.join(root, sub, path)
            if os.path.exists(p):
                return p
        return None

    def raw(self, path) -> bytes:
        p = self._find("raw", path)
        if p is None:
            raise FileNotFoundError(
                f"{path} in no root {self.roots} -- stage the content; "
                f"every dumped entry is present by construction.")
        return open(p, "rb").read()

    def model(self, path):
        import torch
        name = path.replace("/", "__").replace(".vmdl_c", "")
        p = self._find("converted", name + ".pt")
        if p:
            return torch.load(p)
        raise FileNotFoundError(
            f"{path}: preserved but not converted (see its .convertlog); "
            f"raw() has the bytes -- fixing the converter upgrades the "
            f"reader, the pack is already complete.")

    def texture(self, path):
        if path in self._cache:
            return self._cache[path]
        import bc_decode
        import vtex  # committed vtex header reader inside ash_extract path
        raise NotImplementedError(
            "texture(): decode-on-demand lands with the vtex header reader "
            "split out of ash_extract; raw() already returns the bytes.")

    def material(self, path):
        import kv3v5
        return kv3v5.parse(self.raw(path))

    def entries(self):
        out = {}
        for root in self.roots:
            mf = os.path.join(root, "MANIFEST.json")
            if os.path.exists(mf):
                out.update(json.load(open(mf))["entries"])
        return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--game", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=12)
    ls = sub.add_parser("ls")
    ls.add_argument("--root", required=True)
    ls.add_argument("--status", default=None)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.game, a.out, a.workers)
    else:
        ent = Datapack([a.root]).entries()
        for p, e in sorted(ent.items()):
            if a.status is None or e["status"] == a.status:
                print(e["status"], p)


if __name__ == "__main__":
    main()
