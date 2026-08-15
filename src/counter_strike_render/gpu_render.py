"""CS2 draw call -- thin orchestrator over per-stage source files.

The renderer's stages live in stages/ (see stages/ORDER) and are executed
IN ORDER IN THIS MODULE'S SCOPE: the same statements in the same order in
the same namespace as the original single file, so behavior is identical by
construction (verified: reassembling stages/ in ORDER reproduces the
original file byte-for-byte at the decimation commit). Stage files are
promoted to real import modules one at a time as their interfaces harden;
each promotion shrinks this scope and grows the tree.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_STAGES = _os.path.join(_HERE, "stages")
_SRC = _os.path.dirname(_HERE)          # the src/ package root
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

with open(_os.path.join(_STAGES, "ORDER")) as _fh:
    _ORDER = [_l.strip() for _l in _fh if _l.strip()]

for _stage in _ORDER:
    _path = _os.path.join(_STAGES, _stage)
    with open(_path) as _fh:
        _code = compile(_fh.read(), _path, "exec")
    exec(_code, globals())
