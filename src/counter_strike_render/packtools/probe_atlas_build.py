"""Build <map>.probe_field.npz -- the ambient-cube probe field -- from a map .vpk.

WHY THIS EXISTS AS A TOOL. de_inferno's probe field was built once, by hand,
on ws-1: `dec_probe_atlas.py` for the atlas and Source2Viewer-CLI plus
`probe_map.py` for the volume table, with the table parsed out of a text dump
that no longer exists anywhere on the render host. The result
(.scratch-tone/probe_field.npz) then sat on disk for three days while
gpu_render.py's own comment said the asset "we do not have", and the six-face
directional path it feeds was never selected on any map.

NOTHING HERE NEEDS Source2Viewer-CLI. `default_ents.vents_c` is KV3 version 5
and the repo owns `vcs/kv3v5.py`; the atlas is BC6H and the repo owns
`bc6.decode`. Both were already imported by other tools in this directory.
That is the same shape as the four other "missing" pieces that turned out to
be written already.

    python3 probe_atlas_build.py --map .../de_inferno.vpk --out de_inferno.probe_field.npz
    python3 probe_atlas_build.py --sweep-dir .../maps --out-dir /data/cs2-artifacts/worlds

THE ATLAS IS A 3D TEXTURE, not a cube: (w, h, depth) with depth = 6 * band,
six axial faces stacked along Z in slot order (+X, +Y, +Z, -X, -Y, -Z). A
texel is (ax+ix, ay+iy, d*band + az+iz), which is the packing
`sample_probes()` in gpu_render.py already indexes.

TABLE COLUMNS, in the order gpu_render.py's PROBE_T reads them:
    0:3   world_mins      3:6   world_maxs      6:9   size
    9:12  atlas slot      12    indoor_outdoor_level
    13    world_volume    14    array_index      15   handshake_off
    16:19 edge_fade_dists 19:22 box_mins(local) 22:25 box_maxs(local)
    25:28 origin          28    yaw (angles[1]) 29:32 scales

COLUMNS 16: ARE THE ARBITRATION, AND THEY CORRECT THIS DOCSTRING. It used to
say "columns 12 and 13 are the engine's overlap arbitration key and are READ,
not invented". Half of that was true: the two FIELDS are read from the
container. What was invented was the claim that they arbitrate. Reading
csgo_complex_ps:425-454 shows the shader never looks at indoor_outdoor_level
here at all -- it walks the volume list in ASCENDING INDEX order and
ACCUMULATES, weighting each volume by a smoothstepped per-axis boundary fade
times the remaining transmittance, stopping at 0.99. There is no winner to
pick. "Read, not invented" was said of the fields and quietly inherited by the
rule they were fed to, which is the more dangerous half to get wrong because
it reads as provenance.

edge_fade_dists is the fade width per axis, and zero means a HARD edge (the
shader's reciprocal saturates the clamp immediately inside the box). It is
mostly NON-zero in the corpus -- 38/58 volumes on de_mirage and 42/82 on
de_ancient are [8,8,8] -- so this was never a term that happened not to fire.

angles is nonzero on 14 of de_inferno's 116 volumes (yaw only, in every map
censused), and world_mins/world_maxs above are computed as origin+box_mins,
which silently assumes yaw 0. Those 14 volumes have had wrong bounds since
this file was written; the local bounds and yaw in cols 19:29 are what let the
renderer do the transform the shader does.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ENTRY_ATLAS = "lightmaps/env_light_probe_volume_atlas.vtex_c"
ENTRY_ENTS = "entities/default_ents.vents_c"
SLOT_ORDER = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
PROBE_CLASS = "env_combined_light_probe_volume"


def _blocks(d):
    """The resource block table. Same walk as lightmap_extract's VTex2D."""
    _fs, _hv, _ver, bo, bc = struct.unpack_from("<IHHII", d, 0)
    p, out = 8 + bo, {}
    for _ in range(bc):
        t = d[p:p + 4].decode("ascii", "replace")
        off, size = struct.unpack_from("<II", d, p + 4)
        out[t] = (p + 4 + off, size)
        p += 12
    return out


def decode_atlas(blob, name="atlas"):
    """(depth, h, w, 3) float32 from a BC6H 3D vtex. Raises by name, never guesses."""
    import bc6
    import lz4.block as _L
    bl = _blocks(blob)
    if "DATA" not in bl:
        raise ValueError(f"{name}: no DATA block, blocks={list(bl)}")
    do, ds = bl["DATA"]
    # OFFSETS ARE lightmap_extract.VTex2D's, not re-derived. The first
    # version of this function walked the extra-data table from the wrong
    # base and was 12 bytes off, which does not crash -- it reads a
    # PHANTOM entry (type 0 instead of type 4), concludes there is no
    # COMPRESSED_MIP_SIZE table, and takes the uncompressed branch on a
    # file that is in fact LZ4. The symptom was a size mismatch; the cause
    # was re-deriving a walk this directory already owns.
    w, h, dep = struct.unpack_from("<3H", blob, do + 20)
    fmt, nmip = struct.unpack_from("<2B", blob, do + 26)
    edo, edc = struct.unpack_from("<II", blob, do + 32)
    mipsizes = None
    p = do + 32 + edo
    for _ in range(edc):
        t, eo, _es = struct.unpack_from("<III", blob, p)
        if t == 4:
            _i1, _i2, mc = struct.unpack_from("<3I", blob, p + 4 + eo)
            mipsizes = struct.unpack_from(f"<{mc}I", blob, p + 4 + eo + 12)
        p += 12
    # BC6H is 16 bytes per 4x4 block; a 3D texture is `dep` such planes.
    unc = (w // 4) * (h // 4) * dep * 16
    start = do + ds
    if mipsizes is None:
        # The same state lightmap_extract was taught at 75d48a7e: an absent
        # table means the mips are NOT compressed, i.e. every mip is raw.
        raw, src = blob[start:start + unc], "ABSENT -- mips are uncompressed"
    else:
        raw, src = blob[start:start + mipsizes[0]], "COMPRESSED_MIP_SIZE"
        if len(raw) != unc:
            raw = _L.decompress(raw, uncompressed_size=unc)
    if len(raw) != unc:
        raise ValueError(f"{name}: atlas mip0 is {len(raw)}B, declared {unc}B")
    plane = (w // 4) * (h // 4) * 16
    out = np.empty((dep, h, w, 3), dtype=np.float32)
    for z in range(dep):
        px = bc6.decode(raw[z * plane:(z + 1) * plane], w, h, 0)
        out[z] = np.asarray(px, dtype=np.float32).reshape(h, w, 4)[..., :3] \
            if np.asarray(px).size == h * w * 4 else \
            np.asarray(px, dtype=np.float32).reshape(h, w, 3)
    return out, (w, h, dep, fmt, nmip, src)


def _walk(node, out):
    """Every dict in a parsed KV3 tree, depth-first. The entity lump nests
    differently across builds, so the class is searched for rather than
    reached by a hardcoded path -- a derived path fails silently."""
    if isinstance(node, dict):
        out.append(node)
        for v in node.values():
            _walk(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk(v, out)


def read_volumes(blob, name="ents"):
    """The env_combined_light_probe_volume rows, READ from KV3, never derived."""
    import kv3v5
    bl = _blocks(blob)
    if "DATA" not in bl:
        raise ValueError(f"{name}: no DATA block, blocks={list(bl)}")
    do, ds = bl["DATA"]
    doc = kv3v5.parse(blob[do:do + ds])
    nodes = []
    _walk(doc, nodes)
    rows = []
    for n in nodes:
        vals = {k: v for k, v in n.items() if isinstance(k, str)}
        if PROBE_CLASS not in str(vals.get("classname", "")):
            continue
        rows.append(vals)
    return rows, len(nodes)


def _f3(d, k):
    """A 3-vector, whether KV3 typed it as a list or as a STRING.

    This lump mixes the two in the same entity: de_inferno's `origin` parses
    as [676.0, 1740.0, 208.0] while its `voxel_size` and
    `indoor_outdoor_level` are the strings '24.0' and '0'. On 27 of 43 maps
    `origin`/`box_mins`/`box_maxs` are strings too, and the first version of
    this function returned None for them -- so the sweep built 16 maps and
    reported the other 27 as "missing origin/box_mins/box_maxs" when every
    field was present and readable. The refusal was honest and the diagnosis
    was wrong, which is why the GAP text names the field rather than the
    map: it is what made the mis-parse findable.
    """
    v = _ci(d, k)
    if v is None:
        return None
    if isinstance(v, str):
        parts = v.replace(",", " ").split()
        if len(parts) != 3:
            return None
        try:
            return [float(x) for x in parts]
        except ValueError:
            return None
    try:
        if len(v) != 3:
            return None
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def _ci(d, name, default=None):
    """Fetch by name, CASE-INSENSITIVELY, because the container mixes cases.

    de_inferno's volumes spell the atlas slot `light_probe_atlas_x`,
    `light_probe_atlas_Y`, `light_probe_atlas_Z` -- lower, upper, upper. The
    first version of this file hardcoded ("x", "Y", "z"), and `.get(...) or 0`
    turned the missed Z into a SILENT ZERO: every volume addressed atlas plane
    0, which is a wrong probe rather than a missing one and renders perfectly
    plausibly. It cost max|d| 125 against the ws-1 reference on that column and
    nothing anywhere said "missing". READ the field, never derive the name.
    """
    if name in d:
        return d[name]
    low = name.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == low:
            return v
    return default


def build_table(rows, band, atlas_shape):
    """(N, 16) float32. Missing required fields REFUSE the map by name."""
    tab, why = [], []
    hs = [float(_ci(r, "handshake", 0) or 0) for r in rows]
    hs0 = min(hs) if hs else 0.0
    for r in rows:
        org, lo, hi = _f3(r, "origin"), _f3(r, "box_mins"), _f3(r, "box_maxs")
        if org is None or lo is None or hi is None:
            why.append("missing origin/box_mins/box_maxs")
            continue
        wmin = [org[i] + lo[i] for i in range(3)]
        wmax = [org[i] + hi[i] for i in range(3)]
        size = [float(_ci(r, f"light_probe_size_{a}", 0) or 0) for a in "xyz"]
        slot = [_ci(r, f"light_probe_atlas_{a}") for a in "xyz"]
        if any(s is None for s in slot):
            # REFUSE rather than default: a missing slot axis addresses atlas
            # plane 0, which is a wrong probe, not an absent one.
            why.append("missing light_probe_atlas_[xyz] (any case)")
            continue
        slot = [float(s) for s in slot]
        if not any(size):
            why.append("no light_probe_size_*")
            continue
        vol = max(1e-6, (wmax[0] - wmin[0]) * (wmax[1] - wmin[1])
                  * (wmax[2] - wmin[2]))
        # COLUMNS 16: THE ARBITRATION INPUTS, added when csgo_complex_ps was
        # finally read. The shader does not arbitrate overlap by picking a
        # winner -- it accumulates volumes front-to-back with a per-axis
        # boundary fade, and the two fields that drive it (edge_fade_dists,
        # angles) were sitting unread in every row this builder has ever
        # parsed. Cols 0:16 keep their meaning so an old reader still works.
        fade = _f3(r, "edge_fade_dists") or [0.0, 0.0, 0.0]
        ang = _f3(r, "angles") or [0.0, 0.0, 0.0]
        sc = _f3(r, "scales") or [1.0, 1.0, 1.0]
        # scales is [1,1,1] on all 256 volumes of the three maps censused, so
        # the transform's scale part is identity BY MEASUREMENT, not by
        # assumption. Recorded per row anyway: the day it is not, the
        # renderer's precondition check should fire rather than silently
        # render a mis-sized volume.
        tab.append(wmin + wmax + size + slot
                   + [float(_ci(r, "indoor_outdoor_level", 0) or 0), vol,
                      float(_ci(r, "array_index", len(tab))),
                      float(_ci(r, "handshake", 0) or 0) - hs0]
                   + fade + lo + hi + org + [ang[1]] + sc)
    # WIDTH IS ASSERTED, NOT RESHAPED INTO. The first version of this line
    # said reshape(-1, 29) while each row carried 32 values, and 116*32 is
    # divisible by 29 -- so numpy silently produced 128 rows of shuffled
    # columns instead of 116 correct ones, and every field past col 16 read
    # as garbage that still had plausible magnitudes. reshape is not a
    # schema check; a row count that changes is.
    arr = np.array(tab, dtype=np.float32)
    if arr.size and arr.shape[1] != 32:
        raise ValueError(f"probe table row width {arr.shape[1]}, expected 32")
    return arr.reshape(-1, 32), why


def build_one(vpk, map_name, out_path, verbose=True):
    sys.path.insert(0, os.path.join(HERE, "vcs"))
    sys.path.insert(0, os.path.join(HERE, "bc6"))
    from lightmap_extract import read_from_vpk
    ab, aname = read_from_vpk(vpk, map_name, ENTRY_ATLAS)
    atlas, meta = decode_atlas(ab, aname)
    w, h, dep, fmt, nmip, src = meta
    if dep % 6:
        raise ValueError(f"{map_name}: atlas depth {dep} is not 6*band")
    band = dep // 6
    eb, ename = read_from_vpk(vpk, map_name, ENTRY_ENTS)
    rows, nnodes = read_volumes(eb, ename)
    table, why = build_table(rows, band, atlas.shape)
    if table.shape[0] == 0:
        raise ValueError(f"{map_name}: 0 usable {PROBE_CLASS} rows from "
                         f"{len(rows)} matched / {nnodes} KV3 nodes; {why[:3]}")
    note = (f"atlas {atlas.shape} fp16 = repo decode of maps/{map_name}/"
            f"{ENTRY_ATLAS} (BC6H, HDR, mip sizes from {src}). table cols: "
            "world_mins3, world_maxs3, size3, atlas3, indoor_outdoor_level, "
            "world_volume, array_index, handshake_off, edge_fade_dists3, "
            "box_mins3, box_maxs3, origin3, yaw, scales3. slots "
            "(+X,+Y,+Z,-X,-Y,-Z). texel = (ax+ix, ay+iy, d*band+az+iz).")
    # .npz suffix REQUIRED on the tmp name: np.savez appends ".npz" to any
    # filename lacking it, so "<out>.tmp" is written as "<out>.tmp.npz" and
    # the rename then fails on a file that was never there.
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, atlas=atlas.astype(np.float16), table=table,
             band=np.int64(band), slot_order=SLOT_ORDER,
             note=np.array([note]))
    os.replace(tmp, out_path)
    if verbose:
        f = atlas.reshape(6, band, h, w, 3)
        print(f"  {map_name}: atlas {atlas.shape} band {band} fmt {fmt} "
              f"mips {nmip} sizes-from {src}")
        print(f"    volumes {table.shape[0]} (of {len(rows)} matched, "
              f"{nnodes} KV3 nodes){'; dropped: ' + str(why[:2]) if why else ''}")
        print("    face means " +
              " ".join(f"{f[i].mean():.4f}" for i in range(6)))
        print(f"    -> {out_path}")
    return table.shape[0], band


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", help="path to <name>.vpk")
    ap.add_argument("--out", help="<map>.probe_field.npz")
    ap.add_argument("--sweep-dir", help="dir of <name>.vpk to build all of")
    ap.add_argument("--out-dir", help="where <name>.probe_field.npz go")
    a = ap.parse_args()
    if a.map:
        n = os.path.basename(a.map)[:-4]
        build_one(a.map, n, a.out or f"{n}.probe_field.npz")
        return 0
    if not (a.sweep_dir and a.out_dir):
        ap.error("--map/--out, or --sweep-dir/--out-dir")
    vpks = sorted(f for f in os.listdir(a.sweep_dir) if f.endswith(".vpk"))
    print(f"sweep: {len(vpks)} vpks in {a.sweep_dir}", flush=True)
    ok, gap = 0, []
    for i, v in enumerate(vpks):
        n = v[:-4]
        out = os.path.join(a.out_dir, f"{n}.probe_field.npz")
        print(f"[{i + 1}/{len(vpks)}] {n}", flush=True)
        try:
            build_one(os.path.join(a.sweep_dir, v), n, out)
            ok += 1
        except BaseException as e:                       # noqa: BLE001
            # Named, per map, and counted. A sweep that hides which maps
            # failed reports coverage it does not have.
            gap.append((n, f"{type(e).__name__}: {e}"))
            print(f"    GAP {type(e).__name__}: {str(e)[:160]}", flush=True)
    print(f"\nsweep done: {ok} built, {len(gap)} GAP")
    for n, w in gap:
        print(f"  GAP {n}: {w[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
