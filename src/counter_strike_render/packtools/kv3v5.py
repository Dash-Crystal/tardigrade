"""Binary KV3 reader covering versions 1-5, for shipping CS2 resources.

WHY THIS EXISTS, ON TOP OF `kv3.py`. `kv3.py` reads KV3 *version 4*
(magic `\\x043VK`) and is the decoder this project has used for
`.vmat_c` DATA blocks. Every resource in the CS2 build on the render
host is now **version 5** (`\\x053VK`), and version 5 is not a tweak of
version 4 -- it is a different container:

  * the payload is **two independently LZ4-compressed buffers**, not one,
    and the header grew from 64 to **120 bytes** to describe them;
  * buffer 1 holds the **strings** (plus the 8-byte values) and is the
    AUXILIARY buffer; buffer 2 holds the 1/2/4/8-byte value buffers, the
    object member counts, and the type stream;
  * **object lengths moved out of the ints buffer** into their own array
    at the head of buffer 2 (`countObjects_buffer2` u32s);
  * a new node type, `ARRAY_TYPE_AUXILIARY_BUFFER` (25), reads its
    elements out of the *other* buffer, so a reader that does not model
    both buffers desynchronises the moment one appears.

Pointing `kv3.py` at a v5 file does not fail loudly: it finds the LZ4
payload at the wrong offset and raises somewhere in the middle of a
match copy, which reads like a corrupt file rather than a wrong format.
Reading v5 with the v4 layout is not a near miss, and that is why this
is a separate module rather than a patch with a version flag.

WHAT IT VALIDATES. `parse()` asserts on the way out that **every** buffer
was consumed exactly -- types, object lengths, binary blobs, and all
four value buffers of both the primary and auxiliary sides -- and that
the `0xFFEEDD00` trailer is where the container says it is. A walk that
took a wrong turn and happened to land on readable data leaves at least
one cursor short; `strict=True` (the default) turns that into a refusal
instead of a plausible-looking tree.

The one hole `kv3.py` documents is still here and is worth restating: a
type substitution between two node types of the SAME WIDTH (STRING /
INT32 / UINT32 / FLOAT all take 4 bytes from the same buffer) leaves
every cursor where it was and cannot be caught by consumption checks.

Reference for the v5 container layout: ValveResourceFormat's
`BinaryKV3.cs` (`ReadBuffer`, `ReadValue`), read directly rather than
inferred from the bytes -- the two-buffer split and the auxiliary-buffer
array type are not guessable from a hex dump.

Usage:
    from kv3v5 import load, parse, blocks
    mat = load("some.vmat_c")              # DATA block of a resource
    ctrl = parse(blocks(path)["CTRL"])     # any block, any version
"""
from __future__ import annotations

import struct

__all__ = [
    "load", "parse", "blocks", "block_bytes", "resource_refs",
    "Kv3Error", "lz4_decompress",
]

# `KV3\x0N` little-endian. v0's magic is the odd one out (`VKV\x03`).
MAGIC = {
    b"VKV\x03": 0,
    b"\x013VK": 1,
    b"\x023VK": 2,
    b"\x033VK": 3,
    b"\x043VK": 4,
    b"\x053VK": 5,
}

TRAILER = 0xFFEEDD00

(NULL, BOOLEAN, INT64, UINT64, DOUBLE, STRING, BINARY_BLOB, ARRAY, OBJECT,
 ARRAY_TYPED, INT32, UINT32, BOOLEAN_TRUE, BOOLEAN_FALSE, INT64_ZERO,
 INT64_ONE, DOUBLE_ZERO, DOUBLE_ONE, FLOAT, INT16, UINT16, UNKNOWN_22,
 INT32_AS_BYTE, ARRAY_TYPE_BYTE_LENGTH,
 ARRAY_TYPE_AUXILIARY_BUFFER) = range(1, 26)


class Kv3Error(Exception):
    """Any refusal this module makes. Raised, never warned past."""


def lz4_decompress(src, expected=None):
    """LZ4 *block* format (no frame header).

    `prefix` semantics matter for the binary-blob path: CS2 chains frames
    so a match may reach back into the previous frame's output, which is
    why the destination is grown rather than restarted per frame.
    """
    dst = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        dst += src[i:i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        if off == 0:
            raise Kv3Error("LZ4: zero match offset")
        ml = token & 0xF
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = len(dst) - off
        if start < 0:
            raise Kv3Error("LZ4: match reaches before start of output")
        for k in range(ml):
            dst.append(dst[start + k])
    out = bytes(dst)
    if expected is not None and len(out) != expected:
        raise Kv3Error(f"LZ4: produced {len(out)} bytes, header says "
                       f"{expected}")
    return out


def _zstd(src, expected):
    """Decompress a zstd payload whose frame header may omit the size.

    `ZstdDecompressor.decompress` requires the content size in the frame
    header and raises "error determining content size" when it is
    absent -- which is how `cs_italy` failed while forty-two other maps
    passed. A streaming read across frames does not need it, and CS2
    does emit multi-frame blob payloads, so both gaps close here.
    """
    import io
    import zstandard
    rdr = zstandard.ZstdDecompressor().stream_reader(
        io.BytesIO(bytes(src)), read_across_frames=True)
    return rdr.read(expected)


def _align(off, a):
    return (off + a - 1) & ~(a - 1)


class _Buf:
    """The four width-segregated value buffers of one KV3 side."""

    __slots__ = ("b1", "p1", "e1", "b2", "p2", "e2",
                 "b4", "p4", "e4", "b8", "p8", "e8")

    def __init__(self, buf):
        self.b1 = self.b2 = self.b4 = self.b8 = buf
        self.p1 = self.e1 = self.p2 = self.e2 = 0
        self.p4 = self.e4 = self.p8 = self.e8 = 0

    def u8(self):
        if self.p1 >= self.e1:
            raise Kv3Error("bytes1 buffer exhausted")
        v = self.b1[self.p1]
        self.p1 += 1
        return v

    def take2(self, fmt):
        if self.p2 + 2 > self.e2:
            raise Kv3Error("bytes2 buffer exhausted")
        v = struct.unpack_from(fmt, self.b2, self.p2)[0]
        self.p2 += 2
        return v

    def take4(self, fmt):
        if self.p4 + 4 > self.e4:
            raise Kv3Error("bytes4 buffer exhausted")
        v = struct.unpack_from(fmt, self.b4, self.p4)[0]
        self.p4 += 4
        return v

    def take8(self, fmt):
        if self.p8 + 8 > self.e8:
            raise Kv3Error("bytes8 buffer exhausted")
        v = struct.unpack_from(fmt, self.b8, self.p8)[0]
        self.p8 += 8
        return v

    def blob1(self, n):
        if self.p1 + n > self.e1:
            raise Kv3Error("bytes1 buffer exhausted for blob")
        v = self.b1[self.p1:self.p1 + n]
        self.p1 += n
        return v

    def remaining(self):
        return (self.e1 - self.p1, self.e2 - self.p2,
                self.e4 - self.p4, self.e8 - self.p8)


class _Ctx:
    def __init__(self, version):
        self.version = version
        self.types = b""
        self.pt = 0
        self.objlen = b""
        self.po = 0
        self.blobs = b""
        self.pblob = 0
        self.bloblen = b""
        self.pbloblen = 0
        self.strings = []
        self.buf = None
        self.aux = None

    # -- type stream ----------------------------------------------------
    def read_type(self):
        if self.pt >= len(self.types):
            raise Kv3Error("type stream exhausted")
        d = self.types[self.pt]
        self.pt += 1
        flag = 0
        if self.version >= 3:
            if d & 0x80:
                d &= 0x3F
                flag = self.types[self.pt]
                self.pt += 1
        elif d & 0x80:
            d &= 0x7F
            flag = self.types[self.pt]
            self.pt += 1
        return d, flag

    def object_length(self):
        if self.version >= 5:
            if self.po + 4 > len(self.objlen):
                raise Kv3Error("object-length array exhausted")
            v = struct.unpack_from("<i", self.objlen, self.po)[0]
            self.po += 4
            return v
        return self.buf.take4("<i")

    def string(self, sid):
        if sid == -1:
            return ""
        if not 0 <= sid < len(self.strings):
            raise Kv3Error(f"string id {sid} outside {len(self.strings)}")
        return self.strings[sid]


def _read_value(ctx, t):
    b = ctx.buf
    if t == NULL:
        return None
    if t == BOOLEAN_TRUE:
        return True
    if t == BOOLEAN_FALSE:
        return False
    if t == INT64_ZERO:
        return 0
    if t == INT64_ONE:
        return 1
    if t == DOUBLE_ZERO:
        return 0.0
    if t == DOUBLE_ONE:
        return 1.0
    if t == BOOLEAN:
        return b.u8() == 1
    if t == INT32_AS_BYTE:
        return b.u8()
    if t == INT16:
        return b.take2("<h")
    if t == UINT16:
        return b.take2("<H")
    if t == INT32:
        return b.take4("<i")
    if t == UINT32:
        return b.take4("<I")
    if t == FLOAT:
        return b.take4("<f")
    if t == INT64:
        return b.take8("<q")
    if t == UINT64:
        return b.take8("<Q")
    if t == DOUBLE:
        return b.take8("<d")
    if t == STRING:
        return ctx.string(b.take4("<i"))
    if t == BINARY_BLOB:
        if ctx.version < 2:
            n = b.take4("<i")
            return b.blob1(n) if n > 0 else b""
        if ctx.pbloblen + 4 > len(ctx.bloblen):
            raise Kv3Error("binary-blob length array exhausted")
        n = struct.unpack_from("<i", ctx.bloblen, ctx.pbloblen)[0]
        ctx.pbloblen += 4
        out = ctx.blobs[ctx.pblob:ctx.pblob + n] if n > 0 else b""
        ctx.pblob += n
        return out
    if t == ARRAY:
        n = b.take4("<i")
        return [_parse(ctx, None) for _ in range(n)]
    if t in (ARRAY_TYPED, ARRAY_TYPE_BYTE_LENGTH):
        n = b.u8() if t == ARRAY_TYPE_BYTE_LENGTH else b.take4("<i")
        st, _ = ctx.read_type()
        return [_read_value(ctx, st) for _ in range(n)]
    if t == ARRAY_TYPE_AUXILIARY_BUFFER:
        # The elements live in the OTHER buffer. Swapping and calling the
        # same reader is what the reference does; reimplementing the
        # switch against the aux buffer is where a divergence would hide.
        n = b.u8()
        st, _ = ctx.read_type()
        ctx.buf, ctx.aux = ctx.aux, ctx.buf
        try:
            out = [_read_value(ctx, st) for _ in range(n)]
        finally:
            ctx.buf, ctx.aux = ctx.aux, ctx.buf
        return out
    if t == OBJECT:
        n = ctx.object_length()
        o = {}
        for _ in range(n):
            _parse(ctx, o)
        return o
    raise Kv3Error(f"unknown KV3 node type {t}")


def _parse(ctx, parent):
    t, _flag = ctx.read_type()
    if parent is None:            # array element: no key
        return _read_value(ctx, t)
    sid = ctx.buf.take4("<i")
    parent[ctx.string(sid)] = _read_value(ctx, t)
    return None


def parse(data, strict=True):
    """Decode one KV3 payload (a resource block's bytes) to Python."""
    magic = bytes(data[:4])
    if magic not in MAGIC:
        raise Kv3Error(f"not a binary KV3 payload: magic {magic!r}")
    version = MAGIC[magic]
    if version == 0:
        raise Kv3Error("KV3 v0 (uncompressed VKV3) is not handled here")

    p = 4 + 16                                    # magic + format guid
    comp, = struct.unpack_from("<I", data, p)
    p += 4
    if version == 1:
        dict_id = frame_size = 0
        n1, n4, n8, unc_total = struct.unpack_from("<iiii", data, p)
        p += 16
        n_types = n_obj = n_arr = 0
        cmp_total = len(data) - p
        n_blocks = blob_bytes = 0
    else:
        dict_id, frame_size = struct.unpack_from("<HH", data, p)
        p += 4
        (n1, n4, n8, n_types) = struct.unpack_from("<iiii", data, p)
        p += 16
        n_obj, n_arr = struct.unpack_from("<HH", data, p)
        p += 4
        (unc_total, cmp_total, n_blocks,
         blob_bytes) = struct.unpack_from("<iiii", data, p)
        p += 16

    n2 = 0
    block_sizes_bytes = 0
    if version >= 4:
        n2, block_sizes_bytes = struct.unpack_from("<ii", data, p)
        p += 8

    if version >= 5:
        (unc1, cmp1, unc2, cmp2) = struct.unpack_from("<iiii", data, p)
        p += 16
        (n1b, n2b, n4b, n8b, _unk13,
         n_obj_b, n_arr_b, _unk16) = struct.unpack_from("<8i", data, p)
        p += 32
        if unc_total != unc1 + unc2:
            raise Kv3Error(f"v5 sizes disagree: total {unc_total} != "
                           f"{unc1} + {unc2}")
    else:
        unc1, cmp1, unc2, cmp2 = unc_total, cmp_total, 0, 0
        n1b = n2b = n4b = n8b = n_obj_b = n_arr_b = 0

    ctx = _Ctx(version)
    _full1 = None

    # -- buffer 1 -------------------------------------------------------
    if comp == 0:
        buf1 = bytes(data[p:p + unc1])
        p += unc1
    elif comp == 1:
        buf1 = lz4_decompress(data[p:p + cmp1], unc1)
        p += cmp1
    elif comp == 2:
        # Before v5, zstd puts the binary blobs in the SAME stream as
        # buffer 1, so the blobs are already decompressed here and the
        # tail must be kept rather than sliced away.
        _full1 = _zstd(data[p:p + cmp1], unc1 + blob_bytes)
        buf1 = _full1[:unc1]
        p += cmp1
    else:
        raise Kv3Error(f"unknown compressionMethod {comp}")

    b1 = _Buf(buf1)
    off = 0
    if n1 > 0:
        b1.p1, b1.e1 = off, off + n1
        off += n1
    if n2 > 0:
        off = _align(off, 2)
        b1.p2, b1.e2 = off, off + n2 * 2
        off += n2 * 2
    if n4 > 0:
        off = _align(off, 4)
        b1.p4, b1.e4 = off, off + n4 * 4
        off += n4 * 4
    if n8 > 0:
        off = _align(off, 8)
        b1.p8, b1.e8 = off, off + n8 * 8
        off += n8 * 8
    elif version < 5:
        off = _align(off, 8)

    if n4 <= 0:
        raise Kv3Error("countBytes4 is 0; the string count lives there")
    n_strings = struct.unpack_from("<i", buf1, b1.p4)[0]
    b1.p4 += 4

    if version >= 5:
        # v5: strings are NUL-separated inside the 1-byte region of the
        # auxiliary buffer, and the container's own accounting expects
        # them consumed from it.
        ctx.aux = b1
        for _ in range(n_strings):
            e = buf1.index(b"\0", b1.p1)
            ctx.strings.append(buf1[b1.p1:e].decode("utf-8", "replace"))
            b1.p1 = e + 1
        if strict and b1.p1 != b1.e1:
            raise Kv3Error(f"v5 strings left {b1.e1 - b1.p1} bytes of the "
                           f"auxiliary 1-byte buffer unread")
    else:
        ctx.buf = b1
        start = off
        for _ in range(n_strings):
            e = buf1.index(b"\0", off)
            ctx.strings.append(buf1[off:e].decode("utf-8", "replace"))
            off = e + 1
        if version == 1:
            tlen = unc_total - off - 4
        else:
            tlen = n_types - off + start
        ctx.types = buf1[off:off + tlen]
        off += tlen
        if n_blocks == 0:
            tr, = struct.unpack_from("<I", buf1, off)
            off += 4
            if tr != TRAILER:
                raise Kv3Error(f"trailer {tr:#x} != {TRAILER:#x}")
            tail = b""
        else:
            tail = buf1[off:]

    # -- buffer 2 (v5 only) ---------------------------------------------
    if version >= 5:
        if comp == 0:
            buf2 = bytes(data[p:p + unc2])
            p += unc2
        elif comp == 1:
            buf2 = lz4_decompress(data[p:p + cmp2], unc2)
            p += cmp2
        else:
            buf2 = _zstd(data[p:p + cmp2], unc2)
            p += cmp2

        b2 = _Buf(buf2)
        ctx.buf = b2
        off = n_obj_b * 4
        ctx.objlen = buf2[:off]
        if n1b > 0:
            b2.p1, b2.e1 = off, off + n1b
            off += n1b
        if n2b > 0:
            off = _align(off, 2)
            b2.p2, b2.e2 = off, off + n2b * 2
            off += n2b * 2
        if n4b > 0:
            off = _align(off, 4)
            b2.p4, b2.e4 = off, off + n4b * 4
            off += n4b * 4
        if n8b > 0:
            off = _align(off, 8)
            b2.p8, b2.e8 = off, off + n8b * 8
            off += n8b * 8
        ctx.types = buf2[off:off + n_types]
        off += n_types
        if n_blocks == 0:
            tr, = struct.unpack_from("<I", buf2, off)
            off += 4
            if tr != TRAILER:
                raise Kv3Error(f"trailer {tr:#x} != {TRAILER:#x}")
            tail = b""
        else:
            tail = buf2[off:]

    # -- binary blobs ---------------------------------------------------
    if n_blocks > 0:
        ctx.bloblen = tail[:n_blocks * 4]
        tail = tail[n_blocks * 4:]
        tr, = struct.unpack_from("<I", tail, 0)
        if tr != TRAILER:
            raise Kv3Error(f"blob-size trailer {tr:#x} != {TRAILER:#x}")
        tail = tail[4:]
        if comp == 0:
            ctx.blobs = bytes(data[p:p + blob_bytes])
            p += blob_bytes
        elif comp == 1:
            # Chained LZ4 frames: a match may reach into the PREVIOUS
            # frame's output, so the frames decode into one growing
            # buffer rather than being decoded independently.
            out = bytearray()
            q = 0
            while q + 2 <= len(tail):
                clen, = struct.unpack_from("<H", tail, q)
                q += 2
                want = min(frame_size, blob_bytes - len(out))
                if want <= 0:
                    break
                out += _lz4_with_prefix(data[p:p + clen], out, want)
                p += clen
            ctx.blobs = bytes(out)
        elif version >= 5:
            n_cmp = cmp_total - cmp1 - cmp2
            ctx.blobs = _zstd(data[p:p + n_cmp], blob_bytes)
            p += n_cmp
        else:
            ctx.blobs = _full1[unc1:unc1 + blob_bytes]
        if len(ctx.blobs) != blob_bytes:
            raise Kv3Error(f"binary blobs {len(ctx.blobs)} != {blob_bytes}")

    # -- walk -----------------------------------------------------------
    t, _ = ctx.read_type()
    root = _read_value(ctx, t)

    if strict:
        bad = []
        if ctx.pt != len(ctx.types):
            bad.append(f"types {ctx.pt}/{len(ctx.types)}")
        if ctx.po != len(ctx.objlen):
            bad.append(f"objlens {ctx.po}/{len(ctx.objlen)}")
        if ctx.pblob != len(ctx.blobs):
            bad.append(f"blobs {ctx.pblob}/{len(ctx.blobs)}")
        if ctx.pbloblen != len(ctx.bloblen):
            bad.append(f"bloblens {ctx.pbloblen}/{len(ctx.bloblen)}")
        for label, bb in (("buf", ctx.buf), ("aux", ctx.aux)):
            if bb is None:
                continue
            r = bb.remaining()
            if any(r):
                bad.append(f"{label} left {r} (b1,b2,b4,b8)")
        if bad:
            raise Kv3Error("KV3 v%d walk did not consume: %s"
                           % (version, ", ".join(bad)))
    return root


def _lz4_with_prefix(src, prefix, want):
    """Decode one chained frame whose matches may reach into `prefix`."""
    dst = bytearray(prefix)
    base = len(dst)
    i, n = 0, len(src)
    while i < n and len(dst) - base < want:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        dst += src[i:i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = token & 0xF
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        st = len(dst) - off
        if st < 0:
            raise Kv3Error("LZ4 chain: match before start of window")
        for k in range(ml):
            dst.append(dst[st + k])
    return bytes(dst[base:])


# ----------------------------------------------------------------------
# resource container
# ----------------------------------------------------------------------
def block_bytes(data):
    """{tag: bytes} for a compiled Source 2 resource held in memory."""
    _fs, _hv, _ver, bo, bc = struct.unpack_from("<IHHII", data, 0)
    p, out = 8 + bo, {}
    for _ in range(bc):
        tag = data[p:p + 4].decode("ascii", "replace")
        off, size = struct.unpack_from("<II", data, p + 4)
        out[tag] = bytes(data[p + 4 + off:p + 4 + off + size])
        p += 12
    return out


def blocks(path):
    with open(path, "rb") as fh:
        return block_bytes(fh.read())


def resource_refs(rerl):
    """External resource paths from a RERL block, in declaration order.

    This is how a `.vmdl_c` names its materials and a `.vmat_c` names its
    textures: the DATA block often carries only an index or a hash, and
    the human-readable path lives here.
    """
    if not rerl:
        return []
    off, cnt = struct.unpack_from("<II", rerl, 0)
    out, p = [], off
    for _ in range(cnt):
        so, = struct.unpack_from("<i", rerl, p + 8)
        s = p + 8 + so
        e = rerl.index(b"\0", s)
        out.append(rerl[s:e].decode("utf-8", "replace"))
        p += 16
    return out


def load(path, strict=True, block="DATA"):
    return parse(blocks(path)[block], strict=strict)


if __name__ == "__main__":
    import json
    import sys
    tag = sys.argv[2] if len(sys.argv) > 2 else "DATA"
    print(json.dumps(load(sys.argv[1], block=tag), indent=1, default=str))
