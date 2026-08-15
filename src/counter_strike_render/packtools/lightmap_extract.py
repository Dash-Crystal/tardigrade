#!/usr/bin/env python3
"""maps/<map>/lightmaps/irradiance.vtex_c -> <map>.irradiance.npy

THE BRIDGE THAT WAS NEVER BUILT
-------------------------------
Exactly one map in this project has baked lighting, and the reason is not that
the data was missing: every CS2 map VPK carries `maps/<map>/lightmaps/`
(~123 entries, ~248 MB). The reason is that nothing in the tree ever read it.
`ash_extract.py` does not touch the lightmaps directory; `ash_to_world.py`
never writes a `.npy`. The de_inferno sidecar in ~/worlds was produced on
2026-08-07 by a process its own adopting commit (931b5eb) describes as one
"nobody has documented", and that process did not survive. So the sidecar-
refusal gate (d2a16ee) correctly refuses 42 of the 43 packs and there has been
no way to answer it except by turning baked lighting off.

This is that missing producer. It reads the map VPK directly -- the same
archive `ash_extract.py` opens -- so a sidecar can be built for any map from
`~/cs2` alone, with no intermediate.

VALIDATION IS AGAINST THE SURVIVING ARTIFACT, NOT AGAINST ITSELF.
`--validate-against ~/worlds/de_inferno.irradiance.npy` re-derives de_inferno's
sidecar from the VPK and reports max|delta| against the undocumented one. That
is the only check that can tell "I decoded the lightmap" apart from "I decoded
something 8192x8192 and plausible".

FORMAT. The container is the standard Source 2 resource: RED2 + DATA blocks,
DATA holding (width, height, depth, format, mip count) and an EXTRA_DATA table
whose type 4 is COMPRESSED_MIP_SIZE. Mip payloads follow the resource header,
SMALLEST FIRST, each LZ4-block compressed when its stored size differs from its
raw size. Pixels are BC6H (format 19), one byte per texel, decoded through the
same `bc6/libbc6.so` the cubemap path uses. `bc6/bc6.py:VTex` parses the same
container but assumes a CUBE ARRAY -- it requires the type-5
CUBEMAP_RADIANCE_SH block, which a 2D lightmap does not have, and computes
`nfaces = 6 * depth`. This reader is the 2D case, not a rewrite of that one.

CARRIED RISK, stated because the identical one just cost this project its
geometry: the BC6H block decode goes through `bc6/libbc6.so`, and that .so
exists in exactly ONE place on ws-1 (~/cubework/bc6/libbc6.so). It cannot be
rebuilt from the tree -- `bc6/bc6lib.c` includes `bcdec.h`, which is not in the
repo. Delete that directory and this bridge stops working, exactly as
~/ashwork/meshopt did. The run prints the library it bound so a log says which
one answered.
"""
import argparse
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

VTEX_FORMAT_BC6H = 19
EXTRA_COMPRESSED_MIP_SIZE = 4

# Bytes per texel, per vtex format number. This is NOT decoration: it sizes the
# LZ4 decompression buffer, and a wrong value either truncates the payload or
# fails the length check. BC6H/BC7/BC5 carry 16 bytes per 4x4 block (1.0 B/texel)
# and BC1/BC4 carry 8 (0.5). Every value here is CHECKED against the stored mip
# size at load, so a format whose rate is wrong is a refusal, not a bad image.
BYTES_PER_TEXEL = {
    1: 0.5,    # DXT1/BC1
    2: 1.0,    # DXT5/BC3
    19: 1.0,   # BC6H
    20: 1.0,   # BC7
    21: 1.0,   # ATI2N/BC5
    23: 0.5,   # ATI1N/BC4
    29: 0.5,   # BC4
    # 27: RESOLVED, and NOT from the enum name -- the enum cannot answer
    # this. THREE tables in this repo transcribe VRF's VTexFormat.cs and
    # they DISAGREE at 27: ash_extract.py:753 says ATI1N, ash_to_world.py:125
    # and bc_decode.py:95 say RG11_EAC. That disagreement is the reason the
    # name is not evidence: the enum shifted between VRF versions and the
    # transcriptions were taken at different times.
    #
    # What settles it was already MEASURED in this tree and never joined to
    # this refusal: bc_decode.py:107-127 establishes that format 27 in the
    # SHIPPED build is BC4, three ways, none of them the name --
    #   * BLOCK SIZE: payloads are exactly 8 bytes per 4x4 block (2048x2048
    #     stores 2,097,152 for 262,144 blocks). A two-channel EAC RG11
    #     block is 16.
    #   * MIP-CHAIN SELF-CONSISTENCY: the encoder's own mip 1 against a
    #     box-downsample of our mip 0. As BC4 the residual is mean
    #     1.18/255; as EAC 41.37/255, with BC7 (1.89) and DXT1 (2.33) as
    #     in-family controls.
    #   * PLATFORM: EAC is an ETC2 mobile format and this is a desktop
    #     Vulkan build.
    # bc_decode.py:128 carries it as FORMAT_OVERRIDE = {27: "BC4"}, so the
    # decoder has been reading these as BC4 all along; only this table did
    # not know.
    #
    # BC4 is 8 bytes per 4x4 block = 0.5 bytes per texel, which picks
    # 33,554,432 of the two candidate decoded sizes for de_mirage's
    # 8192x8192 page and rejects 67,108,864. The refusal was right to hold
    # -- the rates differ by exactly the 2x that separates a one-channel
    # block from a two-channel one, and a wrong pick truncates silently.
    27: 0.5,   # BC4, per bc_decode.py:128 FORMAT_OVERRIDE (measured)
    # ONE asset ships in three formats across maps -- de_dust2 is BC7 (20),
    # de_inferno is ATI2N (21), de_mirage is 27/BC4 -- so per-map format
    # variation is normal for this page and a table lookup by asset name
    # would be wrong.
}


class VTex2D:
    """A 2D Source 2 .vtex_c, parsed from bytes."""

    def __init__(self, data: bytes, name="<bytes>"):
        self.d = data
        self.name = name
        self.res_size = struct.unpack_from("<I", data, 0)[0]
        p = 8 + struct.unpack_from("<I", data, 8)[0]
        nblocks = struct.unpack_from("<I", data, 12)[0]
        self.blocks = {}
        for _ in range(nblocks):
            nm = data[p:p + 4].decode(errors="replace")
            o, s = struct.unpack_from("<II", data, p + 4)
            self.blocks[nm] = (p + 4 + o, s)
            p += 12
        if "DATA" not in self.blocks:
            raise ValueError(f"{name}: no DATA block, blocks={list(self.blocks)}")
        o = self.blocks["DATA"][0]
        self.width, self.height, self.depth = struct.unpack_from("<HHH", data, o + 20)
        self.fmt, self.nmip = struct.unpack_from("<BB", data, o + 26)
        edo, edc = struct.unpack_from("<II", data, o + 32)
        self.extra = {}
        p = o + 32 + edo
        for _ in range(edc):
            t, eo, es = struct.unpack_from("<III", data, p)
            self.extra[t] = (p + 4 + eo, es)
            p += 12
        # ITS ABSENCE IS A STATE, NOT AN ERROR -- the same fact bc6.py's
        # VTexCube was taught at 0b4f4534, arriving here from the other side.
        # This used to refuse cleanly ("no COMPRESSED_MIP_SIZE table; extra
        # types present: [3]"), which cost lobby_mapveto its shadow atlas in
        # the 43-map sweep. 4 is an EXTRA-DATA BLOCK TYPE, not a texture
        # format code, and the table is absent because those mips are NOT
        # compressed.
        #
        # mip_bytes() below already discriminates the two cases: it reaches
        # for LZ4 only when the stored size differs from raw_mip_size(). So
        # absent means "every mip is its raw size", and filling the table
        # from raw_mip_size() makes the two branches agree BY CONSTRUCTION --
        # stored == raw, the lz4 call is not taken, and the length check that
        # follows is the same check either way.
        self.mip_sizes_source = (
            "COMPRESSED_MIP_SIZE" if EXTRA_COMPRESSED_MIP_SIZE in self.extra
            else "ABSENT -- mips are uncompressed, sizes are the raw sizes")
        self.data_off = self.res_size
        if EXTRA_COMPRESSED_MIP_SIZE in self.extra:
            eo, _ = self.extra[EXTRA_COMPRESSED_MIP_SIZE]
            _ver, off, cnt = struct.unpack_from("<III", data, eo)
            self.mip_sizes = list(
                struct.unpack_from(f"<{cnt}I", data, eo + 4 + off))
        else:
            # raw_mip_size() raises by name on an unknown format rather than
            # guessing a byte rate, so an unsupported format still refuses
            # here instead of silently sizing every mip wrong.
            self.mip_sizes = [self.raw_mip_size(m) for m in range(self.nmip)]

    def raw_mip_size(self, m):
        bpt = BYTES_PER_TEXEL.get(self.fmt)
        if bpt is None:
            raise ValueError(
                f"{self.name}: vtex format {self.fmt} has no known byte rate. "
                f"Guessing one sizes the LZ4 buffer wrong and either truncates "
                f"the page or fails the length check; add it to "
                f"BYTES_PER_TEXEL with the arithmetic that establishes it.")
        return int(max(self.width >> m, 4) * max(self.height >> m, 4) * bpt)

    def mip_offset(self, m):
        """Mips are stored SMALLEST FIRST, so mip m sits after all finer... no:
        after all COARSER ones (indices nmip-1 down to m+1)."""
        return self.data_off + sum(self.mip_sizes[i]
                                   for i in range(self.nmip - 1, m, -1))

    def mip_bytes(self, m):
        off = self.mip_offset(m)
        stored = self.mip_sizes[m]
        raw = self.raw_mip_size(m)
        blob = self.d[off:off + stored]
        if len(blob) != stored:
            raise ValueError(f"{self.name}: mip {m} runs past end of file "
                             f"({len(blob)}B of {stored}B at {off})")
        if stored != raw:
            import lz4.block as _L
            blob = _L.decompress(blob, uncompressed_size=raw)
        if len(blob) != raw:
            raise ValueError(f"{self.name}: mip {m} decompressed to "
                             f"{len(blob)}B, declared {raw}B")
        return blob


def decode_page(blob, fmt, w, h, strip_rows=512, dtype=None, progress=None):
    """Block bytes -> (h, w, C). HDR formats go float, LDR formats go uint8.

    THE LIGHTMAP IS NOT ONE FORMAT AND THE FORMAT IS NOT PER-MAP EITHER.
    Read from the VPKs: de_train carries irradiance as BC6H (19),
    directional_irradiance as BC1 (1) and direct_light_shadows as BC7 (20),
    while de_inferno's direct_light_shadows is ATI2N/BC5 (21). Dispatch on the
    number the file declares; never on the entry's name and never on what the
    last map did.

    Decoded in strips because an 8192^2 page is 805 MB as float32.
    """
    if h % 4 or w % 4:
        raise ValueError(f"block formats need 4-aligned dimensions, got {w}x{h}")
    strip_rows = max(4, strip_rows - strip_rows % 4)

    if fmt == VTEX_FORMAT_BC6H:
        import bc6.bc6 as bc6
        print(f"BC6H backend: {bc6._lib._name}", flush=True)
        dtype = dtype or np.float16
        chans, bpb = 3, 16

        def _one(chunk, rows):
            return bc6.decode(chunk, w, rows, 0)
    else:
        import bc_decode
        # THE EFFECTIVE FORMAT, not the enum name. bc_decode applies
        # FORMAT_OVERRIDE before decoding -- 27 decodes as BC4 -- so
        # printing VTEX_FORMAT's name alone announced "RG11_EAC" for a
        # page being correctly decoded as BC4. A log line that names a
        # format the decoder does not use is worse than no line: it is
        # the evidence someone reaches for when the image looks wrong.
        _eff = bc_decode.FORMAT_OVERRIDE.get(fmt)
        _nm = bc_decode.VTEX_FORMAT.get(fmt, '?')
        print(f"LDR backend: bc_decode {bc_decode.backend_name()} "
              f"format {fmt} -> {_eff or _nm}"
              + (f" (enum name {_nm} OVERRIDDEN, bc_decode.py:128)"
                 if _eff else "")
              + f", {BYTES_PER_TEXEL[fmt]} B/texel", flush=True)
        dtype = dtype or np.uint8
        chans = 4
        bpb = int(BYTES_PER_TEXEL[fmt] * 16)

        def _one(chunk, rows):
            return bc_decode.decode(chunk, fmt, w, rows)

    out = np.empty((h, w, chans), dtype=dtype)
    bytes_per_block_row = (w // 4) * bpb
    for y0 in range(0, h, strip_rows):
        y1 = min(y0 + strip_rows, h)
        off = (y0 // 4) * bytes_per_block_row
        n = ((y1 - y0) // 4) * bytes_per_block_row
        out[y0:y1] = _one(blob[off:off + n], y1 - y0).astype(dtype)
        if progress:
            progress(y1, h)
    return out


def read_from_vpk(vpk_path, map_name, entry="lightmaps/irradiance.vtex_c"):
    from vcs.vpk import VPK
    pak = VPK(vpk_path)
    want = f"maps/{map_name}/{entry}"
    names = list(pak.entries) if hasattr(pak, "entries") else list(pak)
    hit = [n for n in names if n.replace("\\", "/").lower() == want.lower()]
    if not hit:
        near = [n for n in names if "lightmap" in n.lower()]
        raise SystemExit(
            f"{vpk_path}: no {want}. lightmaps entries present: "
            f"{near[:12]}{' ...' if len(near) > 12 else ''} "
            f"({len(near)} total)")
    return pak.read(hit[0]), hit[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True,
                    help="path to <name>.vpk under ~/cs2/game/csgo/maps")
    ap.add_argument("--out", required=True, help="<map>.irradiance.npy")
    ap.add_argument("--entry", default="lightmaps/irradiance.vtex_c")
    ap.add_argument("--mip", type=int, default=0)
    ap.add_argument("--dtype", default=None,
                    choices=("float16", "float32", "uint8"),
                    help="default: float16 for the HDR BC6H page, uint8 for the "
                         "LDR direction/shadow pages")
    ap.add_argument("--validate-against", default=None,
                    help="an existing sidecar to report max|delta| against; "
                         "the run REFUSES to write if they disagree")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="max|delta| accepted by --validate-against")
    ap.add_argument("--from-file", default=None,
                    help="read the .vtex_c from here instead of the VPK "
                         "(for diffing against a rescued copy)")
    args = ap.parse_args()

    map_name = os.path.basename(args.map)
    for suf in ("_dir.vpk", ".vpk"):
        if map_name.endswith(suf):
            map_name = map_name[:-len(suf)]
            break

    if args.from_file:
        blob = open(args.from_file, "rb").read()
        src = args.from_file
    else:
        blob, src = read_from_vpk(args.map, map_name, args.entry)
    print(f"source: {src} ({len(blob):,} B)", flush=True)

    t = VTex2D(blob, name=src)
    # WHICH of the two the sizes came from, every run. A table filled from
    # raw_mip_size() and a table read off the container are indistinguishable
    # downstream, and only one of them means "this texture is uncompressed".
    print(f"vtex: {t.width}x{t.height}x{t.depth} format {t.fmt} "
          f"mips {t.nmip} mip_sizes {t.mip_sizes} "
          f"[sizes from {t.mip_sizes_source}]", flush=True)
    if t.fmt not in BYTES_PER_TEXEL:
        raise SystemExit(f"vtex format {t.fmt} has no known byte rate here; "
                         "add it to BYTES_PER_TEXEL with the arithmetic that "
                         "establishes it rather than letting it guess")
    if t.depth != 1:
        raise SystemExit(f"depth {t.depth} != 1: this is not a 2D lightmap. "
                         "bc6/bc6.py:VTex is the cube-array reader")

    raw = t.mip_bytes(args.mip)
    w = max(t.width >> args.mip, 4)
    h = max(t.height >> args.mip, 4)

    def prog(done, total):
        if done == total or (done // 512) % 4 == 0:
            print(f"  decoded {done}/{total} rows", flush=True)

    img = decode_page(raw, t.fmt, w, h,
                      dtype=np.dtype(args.dtype) if args.dtype else None,
                      progress=prog)
    finite = np.isfinite(img.astype(np.float32)).all()
    print(f"decoded {img.shape} {img.dtype}: min {float(img.min()):.6g} "
          f"max {float(img.max()):.6g} mean {float(img.astype(np.float32).mean()):.6g} "
          f"finite {bool(finite)} zero-fraction "
          f"{float((img == 0).all(axis=2).mean()):.4f}", flush=True)
    if not finite:
        raise SystemExit("non-finite texels in the decoded lightmap")

    if args.validate_against:
        ref = np.load(args.validate_against, mmap_mode="r")
        if ref.shape != img.shape:
            raise SystemExit(f"shape {img.shape} vs reference {ref.shape}")
        worst = 0.0
        for y0 in range(0, img.shape[0], 1024):
            a = img[y0:y0 + 1024].astype(np.float32)
            b = np.asarray(ref[y0:y0 + 1024]).astype(np.float32)
            worst = max(worst, float(np.abs(a - b).max()))
        print(f"VALIDATE vs {args.validate_against}: max|delta| = {worst:g}",
              flush=True)
        if worst > args.tolerance:
            raise SystemExit(
                f"REFUSING to write: max|delta| {worst:g} exceeds tolerance "
                f"{args.tolerance:g}. A sidecar that does not reproduce the "
                f"one already in service is a different physical model, and "
                f"the render would be scored against the same ground truth.")

    np.save(args.out, img)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} B)", flush=True)


if __name__ == "__main__":
    main()
