"""Minimal Valve VPK v1/v2 directory reader (read-only)."""
import struct
import os


class VPK:
    def __init__(self, path):
        self.path = path
        self.entries = {}
        with open(path, "rb") as fh:
            data = fh.read()
        sig, ver, tree_size = struct.unpack_from("<III", data, 0)
        assert sig == 0x55AA1234, hex(sig)
        if ver == 1:
            off = 12
        elif ver == 2:
            off = 28
        else:
            raise ValueError("vpk version %d" % ver)
        self.data_offset = off + tree_size
        self.data = data
        p = off

        def cstr():
            nonlocal p
            e = data.index(b"\0", p)
            s = data[p:e].decode("utf-8", "replace")
            p = e + 1
            return s

        while True:
            ext = cstr()
            if not ext:
                break
            while True:
                d = cstr()
                if not d:
                    break
                while True:
                    name = cstr()
                    if not name:
                        break
                    (crc, pre_len, arch_idx, entry_off,
                     entry_len, term) = struct.unpack_from("<IHHIIH", data, p)
                    p += 18
                    pre = data[p:p + pre_len]
                    p += pre_len
                    full = ("%s/%s.%s" % (d, name, ext)) if d != " " \
                        else "%s.%s" % (name, ext)
                    self.entries[full] = (arch_idx, entry_off, entry_len, pre)

    def read(self, name):
        arch_idx, off, ln, pre = self.entries[name]
        if ln == 0:
            return pre
        if arch_idx == 0x7FFF:
            return pre + self.data[self.data_offset + off:
                                   self.data_offset + off + ln]
        base = self.path[:-8] if self.path.endswith("_dir.vpk") \
            else self.path[:-4]
        with open("%s_%03d.vpk" % (base, arch_idx), "rb") as fh:
            fh.seek(off)
            return pre + fh.read(ln)


if __name__ == "__main__":
    import sys
    v = VPK(sys.argv[1])
    pat = sys.argv[2] if len(sys.argv) > 2 else ""
    hits = [k for k in v.entries if pat in k]
    print("%d entries, %d matching %r" % (len(v.entries), len(hits), pat))
    for k in sorted(hits)[:200]:
        print("  %10d  %s" % (v.entries[k][2] or len(v.entries[k][3]), k))
