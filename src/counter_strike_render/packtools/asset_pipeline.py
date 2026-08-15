"""Format-closed on-demand asset resolution.

THE CLOSURE PROPERTY. The renderable universe is every state the underlying
engine's SERIALIZATION FORMATS can express -- all models, all maps, all
animations, including modded and future-arriving content in the same
formats. No enumeration (a depot sweep, a catalog, a manifest) can be that
universe; only a converter pipeline keyed on the FORMAT SIGNATURES can.
This module is that pipeline:

    resolve(reference, content_roots) -> pack path

  * A reference is whatever a demofile / world / rig names: a model path
    (.vmdl_c), a map (.vpk world data), an animation (.vnmclip/.vnmskel),
    a material/texture (.vmat_c/.vtex_c).
  * On cache hit (content-addressed by source digest): return the pack.
  * On miss: dispatch to the committed converter FOR THAT FORMAT
    (extract_playermodel for models, pack_world/ash for maps, vnmclip for
    animation clips), build the pack, cache it, return it.
  * On a source that does not parse as its format: REFUSE with the format
    error, by name. Not-yet-arrived content is not a failure mode of this
    module -- the moment a conformant file exists under a content root, it
    resolves. That is the whole point.

The depot sweep (extract_all_models) remains useful as a WARM of this
cache over today's archive; it is not, and cannot be, the specification.
Coverage claims are therefore structural (the format dispatch below) plus
measured (the cache's own inventory), never a list.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: format signature -> (converter script, output kind). THE DISPATCH IS BY
#: SERIALIZATION FORMAT. Adding a format = one row; naming a file = never.
CONVERTERS = {
    ".vmdl_c": ("extract_playermodel.py", "model"),
    ".vnmclip": ("vnmclip.py", "clip"),
    ".vnmskel": ("vnmclip.py", "skeleton"),
    ".vpk": ("pack_world.py", "world"),
    ".ash": ("ash_to_world.py", "world"),
}


class UnresolvableAsset(SystemExit):
    pass


def _digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _find_source(reference: str, content_roots) -> str | None:
    for root in content_roots:
        cand = os.path.join(root, reference)
        if os.path.isfile(cand):
            return cand
    return None


def resolve(reference: str, content_roots, cache_dir: str,
            python: str = sys.executable) -> str:
    """The one entry point. Returns a pack path or refuses by name."""
    ext = next((e for e in CONVERTERS if reference.endswith(e)), None)
    if ext is None:
        raise UnresolvableAsset(
            f"REFUSING {reference!r}: no converter for its format. Known "
            f"formats: {sorted(CONVERTERS)}. If this is a NEW engine "
            f"format, add its converter row -- never a filename.")
    src = _find_source(reference, content_roots)
    if src is None:
        raise UnresolvableAsset(
            f"REFUSING {reference!r}: not present under any content root "
            f"{list(content_roots)}. The moment a conformant file exists "
            f"there, it resolves -- stage the content, not a workaround.")
    os.makedirs(cache_dir, exist_ok=True)
    tool, kind = CONVERTERS[ext]
    key = f"{kind}__{os.path.basename(reference).replace(ext, '')}__{_digest(src)}"
    out = os.path.join(cache_dir, key + ".pt")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    if kind == "model":
        r = subprocess.run(
            [python, os.path.join(HERE, tool),
             "--game", _game_root_of(src, reference),
             "--vmdl", reference, "--allow-stub",
             "--out-dir", cache_dir, "--name", key],
            capture_output=True, text=True)
    else:
        r = subprocess.run(
            [python, os.path.join(HERE, tool), src, "--out", out],
            capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        raise UnresolvableAsset(
            f"REFUSING {reference!r}: its converter ({tool}) failed on a "
            f"file that exists -- a FORMAT nonconformance or a converter "
            f"defect, both of which are named, not swallowed:\n  "
            + "\n  ".join(tail))
    return out


def _game_root_of(src: str, reference: str) -> str:
    """The depot game/ root above a source file (the extractor's --game)."""
    base = src[: len(src) - len(reference)].rstrip("/")
    # loose file trees end with .../game/csgo -> root is .../game
    for up in (base, os.path.dirname(base)):
        if os.path.basename(up) == "game":
            return up
    return base
