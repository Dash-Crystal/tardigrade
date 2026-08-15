"""ctypes binding for the meshoptimizer decoders ash_extract.py needs.

WHY THIS FILE IS IN THE REPO
----------------------------
Every CS2 world vertex and index buffer is meshopt-compressed
(ash_extract.py:1152). Without these three entry points an extraction still
"succeeds" -- see the silent-failure note below -- and produces a map with no
geometry. The module that provided them lived only in `~/ashwork/meshopt` on
ws-1, was never committed, and was deleted on 2026-08-07 with the rest of that
directory on the claim that the intermediates were "regenerable from ~/cs2".
They were not: ~/cs2 regenerates materials and textures and ZERO triangles.
Measured on de_train, 2026-08-08, before this file existed:

    flat_triangles 32 of 7,578,208 declared; model_problems 4910/4910

THE SILENT FAILURE THIS EXISTS TO CLOSE
---------------------------------------
`load_meshopt()` returns None rather than raising when the decoder is missing
(ash_extract.py:287), by design, so the material half still runs. But the
manifest's `summary.triangles` is the DECLARED element count read from the
buffer descriptors -- it reads 7,578,208 on a run that decoded 32. A guard of
`summary.triangles > 0` therefore passes on a geometry-free extraction. The
fields that actually discriminate are `summary.flat_triangles`,
`summary.model_problems`, and the "GEOMETRY IS ABSENT OR PARTIAL" note.

THE LIBRARY
-----------
libmeshoptimizer is not a separate install here: `imagecodecs` vendors it, and
that copy exports the full C decode API. This module finds it rather than
requiring a hand-placed libmeshopt.so, and PRINTS which file it bound, because
a decoder resolved from an unnamed path is how the last one disappeared.
"""
import ctypes
import glob
import os
import sys

__all__ = ["decode_vertex_buffer", "decode_index_buffer",
           "decode_index_sequence", "library_path", "version"]

_SEARCH = [
    # vendored by imagecodecs, which the extraction env already requires
    os.path.join(sys.prefix, "lib", "python*", "site-packages",
                 "imagecodecs.libs", "libmeshoptimizer*.so*"),
    os.path.join(sys.prefix, "lib*", "python*", "site-packages",
                 "imagecodecs.libs", "libmeshoptimizer*.so*"),
    # a hand-placed copy beside this file still wins if someone provides one
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "libmeshopt*.so*"),
    "/usr/lib/x86_64-linux-gnu/libmeshoptimizer*.so*",
    "/usr/local/lib/libmeshoptimizer*.so*",
]

_ENTRY = ("meshopt_decodeVertexBuffer", "meshopt_decodeIndexBuffer",
          "meshopt_decodeIndexSequence")


def _load():
    tried = []
    override = os.environ.get("MESHOPT_LIB")
    if override and not os.path.isfile(override):
        # An explicit override that silently falls back to a search is how a
        # run ends up bound to a library nobody named.
        raise ImportError(f"MESHOPT_LIB={override!r} is not a file")
    cands = ([override] if override else [])
    for pat in _SEARCH:
        cands.extend(sorted(glob.glob(pat)))
    for path in cands:
        if not path or not os.path.isfile(path):
            tried.append(path)
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError as e:
            tried.append(f"{path}: {e}")
            continue
        missing = [s for s in _ENTRY if not hasattr(lib, s)]
        if missing:
            tried.append(f"{path}: missing {missing}")
            continue
        return lib, path
    raise ImportError(
        "no libmeshoptimizer exporting " + ", ".join(_ENTRY) +
        " was found. Every CS2 world buffer is meshopt compressed, so an "
        "extraction without it decodes no geometry. Set MESHOPT_LIB to the "
        "shared library, or `pip install imagecodecs` which vendors it. "
        "Tried: " + "; ".join(str(t) for t in tried))


_LIB, _PATH = _load()

for _s in _ENTRY:
    _f = getattr(_LIB, _s)
    _f.restype = ctypes.c_int
    _f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                   ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]


def library_path():
    return _PATH


def version():
    if hasattr(_LIB, "meshopt_decodeVertexVersion"):
        f = _LIB.meshopt_decodeVertexVersion
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
        return None      # needs a buffer; the encoder version is per-buffer
    return None


def _decode(entry, data, count, size):
    """Run one decoder into a fresh count*size buffer.

    The C API returns 0 on success and a negative code otherwise; it does NOT
    signal partial decodes, so a non-zero return is raised rather than returning
    a short buffer that the caller would silently accept as geometry.
    """
    if count <= 0 or size <= 0:
        return b""
    total = count * size
    dst = ctypes.create_string_buffer(total)
    src = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    rc = entry(ctypes.cast(dst, ctypes.c_void_p), count, size, src, len(data))
    if rc != 0:
        raise ValueError(
            f"{entry.__name__} returned {rc} for count={count} size={size} "
            f"src={len(data)}B (lib {_PATH})")
    return dst.raw[:total]


def decode_vertex_buffer(data, count, size):
    return _decode(_LIB.meshopt_decodeVertexBuffer, data, count, size)


def decode_index_buffer(data, count, size):
    return _decode(_LIB.meshopt_decodeIndexBuffer, data, count, size)


def decode_index_sequence(data, count, size):
    return _decode(_LIB.meshopt_decodeIndexSequence, data, count, size)
