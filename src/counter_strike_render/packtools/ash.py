"""ASH -- Action State History. Serialise, deserialise, egocentric video.

WHY THIS EXISTS. The chain into our renderer ran through glTF, and glTF
is lossy for this purpose in a way that is measured rather than
suspected: exporting de_inferno emits 6,107 material-channel exceptions,
one per channel the spec cannot express. glTF carries one base colour,
one normal encoding and no notion of a combo axis; CS2 shades from two
full layer stacks, at least three mutually-indistinguishable normal
encodings, 51 combo axes on the world families alone, and 179 shader
families selected by an `m_shaderName` field glTF discards outright.

We already own the decoders (vcs/kv3.py, vcs/vcsx2.py), so the glTF hop
was never load-bearing. It was a detour that dropped 6,107 channels.

THE RULE. Store what the reference reads, keyed by what the reference
calls it. No renaming, no canonicalisation into another material model,
no "closest equivalent". A parameter whose meaning is unresolved is
still stored, verbatim and typed, under its own name -- an unresolved
parameter that is stored can be resolved later; one that is dropped
cannot.

Format spec: docs/projects/counter-strike-sft/ASH_FORMAT.md
"""

from __future__ import annotations

import hashlib
import json
import os
import time

__all__ = [
    "ASH_VERSION", "AshWriter", "AshReader", "AshError",
    "ManifestMismatch", "sha256_file",
]

ASH_VERSION = 1

# The parts a history may carry. A reader asks for one by name; a writer
# declares which it produced. Nothing is implicit -- a consumer that
# wants geometry and gets a state-only history should be told, not
# handed an empty tensor.
PARTS = ("geometry", "materials", "textures", "state", "video")


class AshError(Exception):
    """Base for every refusal this module makes."""


class ManifestMismatch(AshError):
    """The manifest does not describe the payload on disk.

    Raised rather than warned. A manifest that cannot say no is the
    check-that-cannot-fail shape this project has been bitten by twice:
    a stale or partial history that loads cleanly is indistinguishable
    from a correct one right up until a number is wrong.
    """


def sha256_file(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(_buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _torch():
    """Imported lazily so a manifest can be read on a host without torch.

    Provenance inspection must not require a GPU stack -- the extraction
    hosts in this project have no torch at all.
    """
    import torch
    return torch


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
class AshWriter:
    """Build a `<name>.ash/` directory.

    Every `add_*` records its own sha256 as it writes, so the manifest is
    a statement about bytes that exist rather than bytes that were
    intended. `close()` is what makes the history readable; a directory
    without a manifest is refused by AshReader by design.
    """

    def __init__(self, root, source=None, tool=None, command=None):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "textures"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "state"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "video"), exist_ok=True)
        self._files = {}
        self._parts = set()
        self._notes = []
        self.meta = {
            "ash_version": ASH_VERSION,
            "created_unix": int(time.time()),
            # Provenance is not decoration. A render whose settings are
            # not recorded is not ground truth, and the same is true of
            # a history whose source is not recorded.
            "source": source,
            "tool": tool,
            "command": command,
        }

    # -- internals ------------------------------------------------------
    def _record(self, relpath):
        full = os.path.join(self.root, relpath)
        self._files[relpath] = {
            "sha256": sha256_file(full),
            "bytes": os.path.getsize(full),
        }

    def _save(self, obj, relpath):
        torch = _torch()
        full = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        torch.save(obj, full)
        self._record(relpath)

    # -- geometry -------------------------------------------------------
    def add_geometry(self, **arrays):
        """Positions, indices, every UV set, colours, blend data.

        Named arrays are stored under the names given. COLOR_1 and the
        blend index/weight pair are ordinary entries here rather than
        special cases -- the bone palette needs them and glTF drops them.
        """
        self._save(dict(arrays), "geometry.pt")
        self._parts.add("geometry")

    # -- materials ------------------------------------------------------
    def add_materials(self, rows, encodings=None):
        """One row per material, verbatim.

        `rows` maps material path -> dict with at minimum `shader_name`.
        `encodings` maps texture-slot name -> declared encoding.

        ENCODING TRAVELS WITH THE SLOT, NOT THE FILE. `g_tNormal` is
        DXT5nm in one family and hemi-octahedral in another; the same
        bytes decode to different normals depending on the consumer.
        Storing an image without its consumer's encoding is how a tilted
        normal ends up looking plausible.
        """
        missing = [k for k, v in rows.items()
                   if not isinstance(v, dict) or "shader_name" not in v]
        if missing:
            # shader_name is the field that selects among 179 families.
            # A row without it is not a material, it is a guess.
            raise AshError(
                f"{len(missing)} material row(s) without shader_name, "
                f"first: {missing[0]!r}. m_shaderName is the field glTF "
                f"drops and the one that decides which family shades a "
                f"surface; a row without it cannot be dispatched.")
        self._save({"rows": rows, "slot_encodings": dict(encodings or {})},
                   "materials.pt")
        self._parts.add("materials")

    def add_texture(self, name, data_bytes, encoding=None):
        """Store an image in its SOURCE encoding, with that encoding named."""
        rel = os.path.join("textures", name)
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(data_bytes)
        self._record(rel)
        if encoding:
            self._files[rel]["encoding"] = encoding
        self._parts.add("textures")

    # -- the actual product --------------------------------------------
    def add_state(self, **columns):
        """Tick-indexed action-state history, columnar.

        Camera, action, entities, and the per-tick engine selections that
        change the call flow (quality level, baked-lighting path, active
        passes). Every column must share a leading tick dimension; a
        history whose columns disagree on length is refused here rather
        than producing a silent off-by-one at join time.
        """
        lens = {k: (len(v) if hasattr(v, "__len__") else None)
                for k, v in columns.items()}
        n = {v for v in lens.values() if v is not None}
        if len(n) > 1:
            raise AshError(
                f"state columns disagree on tick count: {lens}. A history "
                f"whose columns are different lengths joins wrongly "
                f"against video and reads as a finding.")
        self._save(dict(columns), os.path.join("state", "history.pt"))
        self._parts.add("state")

    def add_video_frame(self, tick, data_bytes, ext="png", settings=None):
        """A frame addressed BY TICK, so state[t] and video[t] join both ways.

        `settings` is the render configuration that produced it. A frame
        whose settings are not recorded is not ground truth, so this is
        stored per frame rather than once per history.
        """
        rel = os.path.join("video", f"{int(tick):08d}.{ext}")
        full = os.path.join(self.root, rel)
        with open(full, "wb") as fh:
            fh.write(data_bytes)
        self._record(rel)
        self._files[rel]["tick"] = int(tick)
        if settings is not None:
            self._files[rel]["settings"] = settings
        self._parts.add("video")

    def note(self, text):
        """Record something the format cannot express.

        Known incompleteness belongs in the artifact, not in somebody's
        memory of it. Pass ordering and resolution are not in any shader
        package; a demo-sourced history has no animation on the wire.
        """
        self._notes.append(text)

    def close(self):
        self.meta["parts"] = sorted(self._parts)
        self.meta["files"] = self._files
        self.meta["notes"] = self._notes
        path = os.path.join(self.root, "manifest.json")
        with open(path, "w") as fh:
            json.dump(self.meta, fh, indent=2, sort_keys=True)
        return self.root


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
class AshReader:
    """Open a `<name>.ash/`, refusing anything that does not check out.

    `verify=True` (the default) hashes every file the manifest names
    BEFORE any part is handed out. That is the point of the manifest: it
    is only worth having if it can say no.
    """

    def __init__(self, root, verify=True):
        self.root = str(root)
        mpath = os.path.join(self.root, "manifest.json")
        if not os.path.exists(mpath):
            raise ManifestMismatch(
                f"{self.root}: no manifest.json. A directory without one "
                f"is not an ASH history -- it may be a partial write, and "
                f"loading it anyway is how a stale payload passes for a "
                f"good one.")
        with open(mpath) as fh:
            self.meta = json.load(fh)
        if self.meta.get("ash_version") != ASH_VERSION:
            raise ManifestMismatch(
                f"{self.root}: ash_version "
                f"{self.meta.get('ash_version')!r} != {ASH_VERSION}")
        self.parts = set(self.meta.get("parts", []))
        if verify:
            self.verify()

    def verify(self):
        """Hash every listed file. Raise on the FIRST discrepancy."""
        bad = []
        for rel, rec in sorted(self.meta.get("files", {}).items()):
            full = os.path.join(self.root, rel)
            if not os.path.exists(full):
                bad.append((rel, "missing"))
                continue
            if os.path.getsize(full) != rec.get("bytes"):
                bad.append((rel, f"size {os.path.getsize(full)} != "
                                 f"{rec.get('bytes')}"))
                continue
            if sha256_file(full) != rec.get("sha256"):
                bad.append((rel, "sha256 mismatch"))
        if bad:
            head = "; ".join(f"{r}: {w}" for r, w in bad[:4])
            raise ManifestMismatch(
                f"{self.root}: {len(bad)} file(s) do not match the "
                f"manifest -- {head}")
        return True

    def _need(self, part):
        if part not in self.parts:
            raise AshError(
                f"{self.root}: no {part!r} in this history; it has "
                f"{sorted(self.parts)}. Asked-for-and-absent is reported "
                f"rather than returned empty, because an empty tensor "
                f"reads as 'nothing there' instead of 'never written'.")

    def geometry(self):
        self._need("geometry")
        return _torch().load(os.path.join(self.root, "geometry.pt"),
                             map_location="cpu", weights_only=True)

    def materials(self):
        self._need("materials")
        return _torch().load(os.path.join(self.root, "materials.pt"),
                             map_location="cpu", weights_only=False)

    def state(self):
        self._need("state")
        return _torch().load(os.path.join(self.root, "state", "history.pt"),
                             map_location="cpu", weights_only=True)

    def ticks(self):
        """Ticks that have a video frame, ascending."""
        return sorted(rec["tick"] for rec in self.meta.get("files", {}).values()
                      if "tick" in rec)

    def frame(self, tick):
        """Frame bytes plus the settings that produced it."""
        self._need("video")
        for rel, rec in self.meta.get("files", {}).items():
            if rec.get("tick") == int(tick):
                with open(os.path.join(self.root, rel), "rb") as fh:
                    return fh.read(), rec.get("settings")
        raise AshError(f"{self.root}: no video frame at tick {tick}; "
                       f"have {self.ticks()[:8]}...")

    def notes(self):
        return list(self.meta.get("notes", []))
