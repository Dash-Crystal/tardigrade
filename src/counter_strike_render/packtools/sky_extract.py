#!/usr/bin/env python3
"""Depot .vtex_c -> <pack>.sky_cube.npz, the sky the renderer's #48 pass samples.

    sky_extract.py --pak <pak01_dir.vpk> --tex materials/skybox/<name>.vtex_c
                   --out <pack>.sky_cube.npz

WHY FROM THE DEPOT AND NOT THE CAPTURE. The capture route worked and its
page is stranded on a machine that went away; the depot is on the cluster
beside every pack and covers all seven maps, only one of which was ever
captured. Same asset, durable source.

THE MAP -> SKY JOIN IS NOT DONE HERE, AND MUST NOT BE DONE BY NAME. The
shared pak carries 30 skybox textures whose filenames LOOK decisive --
sky_de_dust2, sky_de_mirage, sky_de_nuke, sky_de_overpass,
sky_de_annubis (sic), s2_de_inferno_sky01 -- and that is six of our seven
maps. de_ancient has NO name-matching entry at all. So name-matching
resolves six and silently fails the seventh, which is exactly the shape of
a method that is wrong even where it agrees: `sky_hr_aztec_02` is the
thematically plausible guess for Ancient and a guess is what it would be.
This tool takes the texture path as an ARGUMENT so the join stays a read
somebody performs, not a string match this file performs.

⚠ THIS TOOL CANNOT YET FIND A CUBE IN THE SHIPPED DEPOT, and the reason is
measured rather than guessed. Two facts:

  * the name-matched inferno candidate,
    `materials/skybox/test/s2_de_inferno_sky01_exr_94895b8b.vtex_c`, parses
    as **512x512, depth 1** -- a 2D texture under a `test/` directory. The
    runtime cube the capture showed is **1024x1024 x6**. Different size AND
    different shape, so it is not the same asset.
  * across **3,677 sampled .vtex_c in pak01_dir.vpk, EVERY ONE reports
    depth 1.** A field that is 1 on every texture in the game is not
    carrying a layer count.

So either `VTex2D` reads depth from the wrong header position (it was
written for 2D lightmaps and never needed it), or Source 2 does not ship
sky cubes as cubes and the engine builds the 1024^2 x6 BC6H image at load
from the 2D source. BOTH ARE LIVE. The second would mean "re-derivable
from the depot" is false as stated -- reproducing an engine-side
conversion is a reconstruction, not a read, and rule 4 bans it.

The guard below is what produced this: pointed at the plausible candidate
it refused rather than reshaping a 512^2 2D page into a "cube", which
would have rendered a wrong sky that looked like a sky.

LAYER COUNT IS READ FROM THE FILE, NOT ASSUMED. A cube is six 2D chains,
and `lightmap_extract.VTex.raw_mip_size()` prices ONE -- correct for the
lightmap it was written for. The same omission made `extract_pages.py`
refuse the captured sky page, and it was found because declared/computed
came out at exactly 6.000000. Here `depth` is read from the DATA block and
a value other than 6 is a stop, not a reshape.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from bc6 import bc6                                            # noqa: E402
from bc6.bc6 import VTex as VTexCube                          # noqa: E402
VTEX_FORMAT_BC6H = 19
from vcs import vpk                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pak", required=True)
    ap.add_argument("--tex", required=True,
                    help="full vpk path of the sky .vtex_c, READ from the "
                         "map rather than matched by name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-faces", type=int, default=6)
    a = ap.parse_args()

    d = vpk.VPK(a.pak)
    if a.tex not in d.entries:
        raise SystemExit("not in pak: %s" % a.tex)
    raw = d.read(a.tex)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".vtex_c", delete=False) as tf:
        tf.write(raw)
        _tmp = tf.name
    tex = VTexCube(_tmp)
    # THE CUBE BIT, read where it lives: flags is a u16 at DATA+2 and the
    # cube bit is 0x10 (corpus-fallback, from three de_inferno vtex whose
    # ONLY differing bit is that one). `depth` is the ARRAY-SLICE count, so
    # a single cubemap has depth 1 and is STILL a cube -- which is why 3,677
    # sampled textures all read depth 1 and why testing depth == 6 refused
    # every cube in the game, including this one.
    _do = tex.blocks["DATA"][0]
    flags = struct.unpack_from("<H", tex.d, _do + 2)[0]
    print("  flags 0x%02x  cube-bit(0x10) %s  extra blocks %s"
          % (flags, bool(flags & 0x10), sorted(tex.extra)))
    if not (flags & 0x10):
        raise SystemExit("flags 0x%02x has no cube bit; this is a 2D page "
                         "and decoding it as six faces would produce a "
                         "plausible wrong sky." % flags)

    print("vtex %s" % a.tex)
    print("  %dx%d depth %d  fmt %d  mips %d"
          % (tex.width, tex.height, tex.depth, tex.fmt, tex.nmip))
    if tex.fmt != VTEX_FORMAT_BC6H:
        raise SystemExit("format %d is not BC6H(%d); this decoder would "
                         "produce plausible garbage rather than fail"
                         % (tex.fmt, VTEX_FORMAT_BC6H))
    if tex.nfaces != a.expect_faces:
        raise SystemExit("nfaces %d (= 6 x depth %d), expected %d."
                         % (tex.nfaces, tex.depth, a.expect_faces))

    # mip 0, all faces. raw_mip_size() prices ONE 2D chain, so the cube is
    # six of them -- the same arithmetic that made extract_pages.py refuse
    # the captured page until it was taught about layers.
    # raw_mip_size() on the CUBE reader already multiplies by nfaces.
    all_faces = tex.raw_mip_size(0)
    per_face = all_faces // tex.nfaces
    blob = tex.mip_bytes(0)
    want = all_faces
    if len(blob) != want:
        raise SystemExit("mip 0 is %d bytes, expected %d = %d per face x %d "
                         "faces (ratio %.6f). Reported, not trimmed."
                         % (len(blob), want, per_face, tex.nfaces,
                            len(blob) / max(per_face, 1)))
    print("  mip 0: %d bytes = %d per face x %d  (exact)"
          % (len(blob), per_face, tex.nfaces))

    faces = []
    for f in range(tex.nfaces):
        img = bc6.decode(blob[f * per_face:(f + 1) * per_face],
                         tex.width, tex.height, 0)
        faces.append(img)
        print("  face %d  min %.5f  max %.5f  mean %.5f  nonfinite %d"
              % (f, float(img.min()), float(img.max()), float(img.mean()),
                 int((~np.isfinite(img)).sum())))

    cube = np.stack(faces, 0).astype(np.float32)
    neg = int((cube < 0).sum())
    print("cube %s  min %.5f  max %.5f  mean %.5f  negative %d%s"
          % (cube.shape, float(cube.min()), float(cube.max()),
             float(cube.mean()), neg,
             "" if neg == 0 else "   <== is_signed WRONG for a UFLOAT page"))
    np.savez_compressed(a.out, cube=cube.astype(np.float16))
    meta = {"source_pak": a.pak, "source_tex": a.tex,
            "width": tex.width, "height": tex.height, "faces": tex.depth,
            "fmt": tex.fmt, "mips": tex.nmip,
            "min": float(cube.min()), "max": float(cube.max()),
            "mean": float(cube.mean()),
            "face_means": [float(f.mean()) for f in faces]}
    open(os.path.splitext(a.out)[0] + ".json", "w").write(
        json.dumps(meta, indent=1))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
