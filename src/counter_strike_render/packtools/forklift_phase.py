"""fl_ maps: the counterfactual FORKLIFT PHASE -- original content through
every API at once.

Three artifacts, all in OUR pack formats, all consumable by the layered
Datapack reader as overlay content:

1. GLB -> MODEL PACK converter (a new CONVERTERS row: the gratis forklift,
   CC BY 3.0, KolosStudios via Poly Pizza, attribution carried in the pack).
   Minimal pure-python GLB parse: JSON chunk + BIN chunk, POSITION/
   TEXCOORD_0/indices accessors, all primitives, flat gray material pages.

2. MAP ANNEX AUTHOR: fl_<base>_freight = the base world pack plus a
   grotesque bolted-on freight yard in the plr_hightower lineage -- a
   payload-RACE phase where the payload is a FORKLIFT on a spline track:
   two parallel elevated plank runs (ugly by construction: raw box
   geometry, repeating crate stacks, a spiral ramp to a pointlessly tall
   delivery platform, exactly the hightower joke), pallet objectives, and
   the track spline + phase spec emitted as JSON. The annex is APPENDED to
   the base pack's vertex/face arrays -- same tensors, same renderer, no
   new code path -- and an overview calibration is emitted so map_identity
   FITS the new map from trajectories (the counterfactual map is a
   first-class citizen of the wrong-map interlock).

3. PHASE SESSION: a synthetic state-action history ON the annex -- the
   forklift advancing along the spline with escorting players -- emitted as
   renderer camera paths + playermodels rows + the forklift as an entity
   row. NO canonical recording of this can exist: the reference client
   never produced such a state, which is exactly what distinguishes a
   RENDERER over a state space from a reimplementation of one client.

Usage:
  python forklift_phase.py glb2pack  IN.glb OUT.pt
  python forklift_phase.py annex     BASE.pt OUT_DIR BASENAME
  python forklift_phase.py session   OUT_DIR/fl_spec.json OUT_DIR
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys

import torch

S = 0.0254

# ------------------------------------------------------------ glb2pack ----

def _glb_chunks(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF", "not a GLB"
    off, js, bin_ = 12, None, None
    while off < len(b):
        ln, typ = struct.unpack_from("<I4s", b, off)
        chunk = b[off + 8: off + 8 + ln]
        if typ == b"JSON":
            js = json.loads(chunk)
        elif typ == b"BIN\x00":
            bin_ = chunk
        off += 8 + ln + (-ln % 4)
    return js, bin_


_CTYPE = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
          5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(js, bin_, idx):
    a = js["accessors"][idx]
    bv = js["bufferViews"][a["bufferView"]]
    fmt, sz = _CTYPE[a["componentType"]]
    n = _NCOMP[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    count = a["count"]
    stride = bv.get("byteStride", sz * n)
    out = []
    for i in range(count):
        vals = struct.unpack_from("<" + fmt * n, bin_, off + i * stride)
        out.append(vals if n > 1 else vals[0])
    return out


def glb2pack(src, dst):
    js, bin_ = _glb_chunks(src)
    verts, uvs, tris, base = [], [], [], 0
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos = _accessor(js, bin_, prim["attributes"]["POSITION"])
            uv = (_accessor(js, bin_, prim["attributes"]["TEXCOORD_0"])
                  if "TEXCOORD_0" in prim["attributes"]
                  else [(0.0, 0.0)] * len(pos))
            idx = (_accessor(js, bin_, prim["indices"])
                   if "indices" in prim else list(range(len(pos))))
            verts += pos
            uvs += uv
            tris += [(idx[i] + base, idx[i + 1] + base, idx[i + 2] + base)
                     for i in range(0, len(idx) - 2, 3)]
            base += len(pos)
    # glTF is meters Y-up; the bundle space is source-units (y, z, x) like
    # extract_playermodel documents. meters -> inches = /0.0254.
    v = torch.tensor(verts, dtype=torch.float32) / S
    v = torch.stack((v[:, 2], v[:, 1], v[:, 0]), dim=1)   # (src_y, src_z, src_x)
    pack = {
        "schema": "iji/model-pack/glb/v1",
        "pos": v, "uv": torch.tensor(uvs, dtype=torch.float32),
        "nrm": torch.zeros_like(v), "tan": torch.zeros(v.shape[0], 4),
        "tri": torch.tensor(tris, dtype=torch.int32),
        "materials": [{"name": "flat", "page": None}],
        "attribution": "Forklift by KolosStudios [CC BY 3.0] via Poly Pizza "
                       "(https://poly.pizza/m/DTQBuenKJY)",
    }
    torch.save(pack, dst)
    print(f"glb2pack: {v.shape[0]} verts, {len(tris)} tris -> {dst}")
    return pack


# --------------------------------------------------------------- annex ----

def _box(cx, cy, z0, z1, hx, hy, verts, tris, uvs):
    b = len(verts)
    for dz in (z0, z1):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append((cx + sx * hx, cy + sy * hy, dz))
            uvs.append(((sx + 1) / 2, (sy + 1) / 2))
    F = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
         (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
         (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    tris += [(b + a, b + c, b + d) for a, c, d in F]


def annex(base_pack, out_dir, base_name):
    os.makedirs(out_dir, exist_ok=True)
    d = torch.load(base_pack)
    # WORLD-PACK SPACE, READ from stages/00_imports.py:18 (ray-cast
    # validated): pack = (src_y, src_z, src_x) * 0.0254 -- glTF meters
    # Y-up. The annex is DESIGNED in src units (cameras/track/spawns are
    # src) and CONVERTED to pack space on append; appending src verts raw
    # rendered the annex 39x oversized and axis-swapped (the measured
    # giant slab).
    S = 0.0254
    vx = d["vertices"]
    src = torch.stack((vx[:, 2], vx[:, 0], vx[:, 1]), dim=1) / S
    # bolt the freight yard beyond the base bbox on +y: grotesque on purpose
    ymax = float(src[:, 1].max()) + 200.0
    x0 = 0.0
    verts, tris, uvs = [], [], []
    # two parallel elevated plank runs (the payload-race pair)
    for lane, off in (("A", -220.0), ("B", 220.0)):
        for i in range(24):
            _box(x0 + off, ymax + i * 180.0, 120.0 + i * 14.0,
                 140.0 + i * 14.0, 90.0, 92.0, verts, tris, uvs)
    # crate spam (ugly by construction)
    for i in range(60):
        gx = x0 - 500 + (i * 137) % 1000
        gy = ymax + (i * 263) % 4000
        _box(gx, gy, 0.0, 48.0 + (i % 5) * 24.0, 24.0, 24.0,
             verts, tris, uvs)
    # the pointlessly tall delivery platform + spiral ramp (the hightower joke)
    top = 900.0
    _box(x0, ymax + 4600.0, top, top + 24.0, 300.0, 300.0, verts, tris, uvs)
    for i in range(40):
        a = i * 0.35
        _box(x0 + 420 * math.cos(a), ymax + 4600.0 + 420 * math.sin(a),
             i * (top / 40), i * (top / 40) + 16.0, 70.0, 70.0,
             verts, tris, uvs)
    av = torch.tensor(verts, dtype=torch.float32)
    ann_v = (torch.stack((av[:, 1], av[:, 2], av[:, 0]), dim=1) * S) \
        .to(vx.dtype)                       # src -> pack, the READ transform
    ann_t = torch.tensor(tris, dtype=torch.int64) + vx.shape[0]
    ann_uv = torch.tensor(uvs, dtype=torch.float32)
    # append to EVERY aligned per-vertex array; unknown arrays keep base-only
    out = dict(d)
    vkey = "vertices"
    out[vkey] = torch.cat([vx, ann_v])
    for k in ("uvs", "uv"):
        if k in d and d[k].shape[0] == vx.shape[0]:
            out[k] = torch.cat([d[k], ann_uv])
    # lmuv=(0,0) samples a BLACK lightmap texel and the lightmapped path
    # MULTIPLIES by it (measured: annex stayed black after the vcolor fix,
    # so the killer is the baked-light term, not albedo). Aim every annex
    # lmuv at the brightest texel of the map's own irradiance page.
    lm_uv = None
    irr_p = base_pack.replace(".pt", ".irradiance.npy")
    if os.path.exists(irr_p):
        import numpy as np
        irr = np.load(irr_p)
        page = irr[0] if irr.ndim == 4 else irr
        lum = page[..., :3].astype("float32").sum(axis=-1)
        iy, ix = divmod(int(lum.argmax()), lum.shape[1])
        lm_uv = ((ix + 0.5) / lum.shape[1], (iy + 0.5) / lum.shape[0])
        print(f"annex lmuv -> irradiance texel ({ix},{iy}) of "
              f"{lum.shape[1]}x{lum.shape[0]}, rgb "
              f"{page[iy, ix][:3].tolist()}")
    for k, t in d.items():
        if torch.is_tensor(t) and t.dim() >= 1 and \
                t.shape[0] == vx.shape[0] and k not in (vkey, "uvs", "uv"):
            pad = torch.zeros(ann_v.shape[0], *t.shape[1:], dtype=t.dtype)
            # VISIBLE defaults, not zeros: zero vcolor multiplies the annex
            # to black and zero normals kill its shading (measured: the
            # first render was black-on-black). White vertex color, +z
            # normals; lit lmuv per above.
            if k == "vcolor":
                pad = torch.full_like(pad, 255 if t.dtype == torch.uint8
                                      else 1)
            elif k == "vnormal":
                pad[:, 2] = 1.0
            elif k in ("lmuv", "uvs2") and lm_uv is not None \
                    and pad.dim() == 2 and pad.shape[1] == 2:
                pad[:, 0] = lm_uv[0]
                pad[:, 1] = lm_uv[1]
            out[k] = torch.cat([t, pad])
    fkey = "faces"
    nf = d[fkey].shape[0]
    out[fkey] = torch.cat([d[fkey], ann_t.to(d[fkey].dtype)])
    for k, t in d.items():
        if torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] == nf \
                and k != fkey:
            pad = torch.zeros(ann_t.shape[0], *t.shape[1:], dtype=t.dtype)
            # annex faces take the base map's MOST COMMON value for the
            # material joins (a real texture, really ugly on a crate) and
            # up-normals for face_normals.
            if k in ("face_mat", "face_matid", "face_mat2",
                     "face_alpha_mode"):
                pad = torch.full_like(pad, int(t.mode().values))
            elif k == "face_normals":
                pad[:, 2] = 1.0
            out[k] = torch.cat([t, pad])
    name = f"fl_{base_name}_freight"
    torch.save(out, os.path.join(out_dir, name + ".pt"))
    # the forklift track spline + phase spec
    track = [{"x": x0, "y": float(ymax + 200 + i * 180), "z": 140.0 + i * 14.0}
             for i in range(24)] + \
            [{"x": x0, "y": float(ymax + 4600.0), "z": 924.0}]
    spec = {"schema": "iji/fl-phase-spec/v1", "map": name,
            "phase": "forklift", "lineage": "plr_hightower (payload race)",
            "track": track,
            "objective": "escort the forklift to the pointlessly tall "
                         "delivery platform; last 900 units are vertical, "
                         "which is the joke",
            "forklift_model": "overlay: forklift_kolosstudios_ccby30",
            }
    json.dump(spec, open(os.path.join(out_dir, "fl_spec.json"), "w"),
              indent=1)
    # overview calibration so map_identity FITS this map from data
    allv = out[vkey]
    px = float(allv[:, 0].min()) - 200
    py = float(allv[:, 1].max()) + 200
    scale = max(float(allv[:, 0].max() - allv[:, 0].min()),
                float(allv[:, 1].max() - allv[:, 1].min())) / 1024.0 * 1.1
    with open(os.path.join(out_dir, name + ".txt"), "w") as fh:
        fh.write(f'"{name}"\n{{\n\t"pos_x"\t"{px}"\n\t"pos_y"\t"{py}"\n'
                 f'\t"scale"\t"{scale:.3f}"\n}}\n')
    print(f"annex: {name}: +{ann_v.shape[0]} verts, +{ann_t.shape[0]} tris, "
          f"track {len(track)} nodes, overview emitted")
    return name


# -------------------------------------------------------------- session ----

def session(spec_path, out_dir):
    spec = json.load(open(spec_path))
    track = spec["track"]
    T = 1400
    ticks, rows_by_tick = [], {}
    for t in range(T):
        f = t / (T - 1)
        seg = min(int(f * (len(track) - 1)), len(track) - 2)
        u = f * (len(track) - 1) - seg
        a, b = track[seg], track[seg + 1]
        fx = a["x"] + (b["x"] - a["x"]) * u
        fy = a["y"] + (b["y"] - a["y"]) * u
        fz = a["z"] + (b["z"] - a["z"]) * u
        yaw = math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"] + 1e-9))
        ents = [{"tick": t * 4, "steamid": 99, "team": "t", "x": fx,
                 "y": fy, "z": fz, "yaw": yaw, "is_alive": True,
                 "agent": "forklift_kolosstudios_ccby30"}]
        for pi in range(4):                     # escorting players
            ang = t * 0.02 + pi * 1.57
            ents.append({"tick": t * 4, "steamid": pi, "team": "ct" if pi % 2
                         else "t", "x": fx + 150 * math.cos(ang),
                         "y": fy + 150 * math.sin(ang), "z": fz,
                         "yaw": math.degrees(ang) + 180, "is_alive": True})
        rows_by_tick[str(t * 4)] = ents
        ticks.append({"tick": t * 4, "x": fx - 320 * math.cos(math.radians(yaw)),
                      "y": fy - 320 * math.sin(math.radians(yaw)),
                      "z": fz, "eye_z": fz + 160.0, "yaw_degrees": yaw,
                      "pitch_degrees": 12.0, "fire": False,
                      "is_alive": True, "uses_future_sample": False})
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"schema": "iji/cs2-demo-ego-camera-path/v1",
               "variant": "forklift_chase", "tick_rate": 32,
               "map_name": spec["map"], "n_ticks": T,
               "n_future_steered": 0, "ticks": ticks},
              open(os.path.join(out_dir, "forklift_chase.camera.json"), "w"))
    json.dump({"schema": "iji/cs2-demo-playermodels/v1",
               "ticks": rows_by_tick},
              open(os.path.join(out_dir, "forklift_playermodels.json"), "w"))
    print(f"session: {T} ticks of counterfactual forklift phase -- a state "
          f"history NO canonical recording can contain")





def flythrough(out_dir, spec_path, tard_path):
    """The gating milestone: static fl_ map flythroughs, both teams at
    spawn, idle. Spawn positions are READ from the recording's own tick-0
    state (real spawns, not invented); the pm pass plays idle clips for
    stationary entities by its own inference. Two sweeps: a perimeter
    orbit of the whole map and a track fly-along through the annex."""
    import sys as _s
    _s.path.insert(0, HERE_REPO)   # repo root itself, not its parent
    from tardigrade_v21 import TardigradeV21Recording
    spec = json.load(open(spec_path))
    rec = TardigradeV21Recording(open(tard_path, "rb").read())
    ents = []
    for pi in range(len(rec.players)):
        s = rec.player(pi).sample(0)
        if s is None or not s.position.present:
            continue
        team = "ct" if int(getattr(rec.players[pi], "team", 0)) == 3 else "t"
        ents.append({"steamid": pi, "team": team, "x": s.position.x,
                     "y": s.position.y, "z": s.position.z,
                     "yaw": s.absolute_yaw_degrees or 0.0,
                     "is_alive": True})
    track = spec["track"]
    cx = sum(e["x"] for e in ents) / len(ents)
    cy = (sum(e["y"] for e in ents) / len(ents) + track[-1]["y"]) / 2.0
    T = 900
    rows_by_tick, orbit, along = {}, [], []
    R = max(2600.0, abs(track[-1]["y"] - cy) + 800.0)
    for t in range(T):
        tick = t * 4
        rows_by_tick[str(tick)] = [dict(e, tick=tick) for e in ents]
        a = 2 * math.pi * t / T
        ox, oy = cx + R * math.cos(a), cy + R * math.sin(a)
        yaw = math.degrees(math.atan2(cy - oy, cx - ox))
        orbit.append({"tick": tick, "x": ox, "y": oy, "z": 500.0,
                      "eye_z": 560.0, "yaw_degrees": yaw,
                      "pitch_degrees": 14.0, "fire": False,
                      "is_alive": True, "uses_future_sample": False})
        f = t / (T - 1)
        seg = min(int(f * (len(track) - 1)), len(track) - 2)
        u = f * (len(track) - 1) - seg
        A, B = track[seg], track[seg + 1]
        seg_yaw = math.degrees(math.atan2(B["y"] - A["y"],
                                          B["x"] - A["x"] + 1e-9))
        along.append({"tick": tick,
                      "x": A["x"] + (B["x"] - A["x"]) * u
                           - 260.0 * math.cos(math.radians(seg_yaw)),
                      "y": A["y"] + (B["y"] - A["y"]) * u
                           - 260.0 * math.sin(math.radians(seg_yaw)),
                      "z": A["z"], "eye_z": A["z"] + 120.0,
                      "yaw_degrees": seg_yaw, "pitch_degrees": 8.0,
                      "fire": False, "is_alive": True,
                      "uses_future_sample": False})
    for name, ticks in (("fly_orbit", orbit), ("fly_track", along)):
        json.dump({"schema": "iji/cs2-demo-ego-camera-path/v1",
                   "variant": name, "tick_rate": 32,
                   "map_name": spec["map"], "n_ticks": T,
                   "n_future_steered": 0, "ticks": ticks},
                  open(os.path.join(out_dir, name + ".camera.json"), "w"))
    json.dump({"schema": "iji/cs2-demo-playermodels/v1",
               "ticks": rows_by_tick},
              open(os.path.join(out_dir, "spawn_idle_playermodels.json"),
                   "w"))
    print(f"flythrough: 2 sweeps x {T} ticks, {len(ents)} players idle at "
          f"their RECORDED spawns")


HERE_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "glb2pack":
        glb2pack(sys.argv[2], sys.argv[3])
    elif cmd == "annex":
        annex(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "session":
        session(sys.argv[2], sys.argv[3])
    elif cmd == "flythrough":
        flythrough(sys.argv[2], sys.argv[3], sys.argv[4])
