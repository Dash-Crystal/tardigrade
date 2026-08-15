"""maps/<map>.vpk -> <map>.light_environment.json, the SCENE's sun.

gpu_render.py already reads this sidecar (`--light-env auto` resolves
`<world pack>.light_environment.json`) and prints a GAP when it is absent.
Nothing wrote it. This does: it reads the map's `light_environment` entity
out of the compiled entity lump, verbatim.

WHY THE ENTITY LUMP AND NOT A FIT. The renderer's sun direction, colour and
angular floor were de_inferno's values written as module constants -- correct
for one map and "the scene's light source ignored" on the other six. The map
ships the answer: `maps/<map>/entities/default_ents.vents_c`, a Source 2
resource whose DATA block is KV3 v5, holding every entity's keyvalues. The
sun is the one with `classname == light_environment`.

WHAT IS COPIED AND WHAT IS DERIVED. Every key of the entity is copied into
`raw` verbatim -- an unresolved parameter that is stored can be resolved
later and one that is dropped cannot. The promoted top-level keys are the
ones the renderer reads by name (`angles`, `color`, `brightness`,
`angulardiameter`, `map`). NOTHING is converted here: sRGB-decode and the
brightness multiply happen in the renderer, which already does it and prints
what it did, and doing it twice in two places is how two answers start.

de_inferno reads: angles [63.0, 218.0, 0.0], color [255, 247, 235],
brightness 3.0, angulardiameter 0.3, skycolor [136, 199, 255], skyintensity
0.875, and skyambientbounce [0, 0, 0] -- which is the entity agreeing, from
a different direction, with the capture finding that g_vSunAmbient reads as
zero.

Example:
    python3 harness/gpu_render/light_environment_extract.py \
        --game-dir /data/cs2-artifacts/work/dd/game \
        --out-dir /data/cs2-artifacts/worlds \
        de_inferno de_mirage de_dust2 de_nuke de_overpass de_ancient de_anubis
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from vcs.vpk import VPK          # noqa: E402
import vcs.kv3v5 as kv3          # noqa: E402

PROMOTE = ("angles", "color", "brightness", "brightnessscale",
           "angulardiameter", "skycolor", "skyintensity", "skybouncescale",
           "skyambientbounce", "bouncescale", "castshadows", "origin",
           "nearclipplane", "enabled", "minroughness", "skytexture")


def _blocks(data: bytes):
    p = 8 + struct.unpack_from("<I", data, 8)[0]
    n = struct.unpack_from("<I", data, 12)[0]
    out, q = {}, p
    for _ in range(n):
        name = data[q:q + 4].decode(errors="replace")
        off, size = struct.unpack_from("<II", data, q + 4)
        out[name] = (q + 4 + off, size)
        q += 12
    return out


def _find_class(node, classname):
    """Depth-first for the entity dict carrying this classname."""
    if isinstance(node, dict):
        if node.get("classname") == classname:
            return node
        for value in node.values():
            hit = _find_class(value, classname)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for item in node:
            hit = _find_class(item, classname)
            if hit is not None:
                return hit
    return None


def read_entities(vpk_path: str, map_name: str):
    """The map's compiled entity lump, parsed. LZ4 KV3 v5 in the DATA block."""
    pak = VPK(vpk_path)
    entry = f"maps/{map_name}/entities/default_ents.vents_c"
    if entry not in pak.entries:
        cands = [n for n in pak.entries if n.endswith(".vents_c")]
        if not cands:
            raise SystemExit(f"{map_name}: no .vents_c in {vpk_path}")
        entry = cands[0]
    data = pak.read(entry)
    off, size = _blocks(data)["DATA"]
    return entry, kv3.parse(bytes(data[off:off + size]))


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def read_sky(doc):
    """The map's own word for its sky: env_sky's `skyname`, a .vmat path.

    NOT worldspawn's `skyname`. Every one of the seven maps carries
    `sky_day01_01` on worldspawn -- the same string on all seven, a legacy
    stub that names no asset any of them use. The entity that names the real
    material is `env_sky`, and taking the first `skyname` found would have
    returned the stub on every map and looked like an answer.
    """
    out, fog = [], []
    for ent in _walk(doc):
        cn = ent.get("classname")
        flat = {k: v for k, v in ent.items()
                if isinstance(v, (str, int, float, bool, list))}
        if cn == "env_sky" and "skyname" in ent:
            out.append(flat)
        elif cn == "env_cubemap_fog" and ent.get("cubemapfogskymaterial"):
            fog.append(flat)
    return out, fog


def choose_sky(skies, fog):
    """The DRAWN sky, which is not simply the first env_sky.

    StartDisabled decides it, and ignoring that got two of seven maps wrong
    before anyone noticed:

      de_inferno ships the pattern openly -- two env_sky, targetnames
        `..._MAT_sky` (StartDisabled False, s2_de_inferno_sky01) and
        `..._LIGHT_sky` (StartDisabled True, a skymodel_hosekwilkie_ref
        procedural). MAT is drawn; LIGHT is the lighting reference, and
        `light_environment.skytexture` names it by targetname.
      de_ancient has ONE env_sky and it is StartDisabled=True, targetname
        `sky_lighting`, which `light_environment.skytexture` points at. Its
        material is the `_lighting` variant, and the map's enabled sky
        material is elsewhere: env_cubemap_fog carries
        sky_hr_aztec_02_v1.vmat. So ancient's drawn sky is the _v1 variant
        and the `_lighting` one is an irradiance reference -- which is
        exactly what a cube topping out at 0.898 with no value above 1.0
        looks like.
      de_anubis is the same shape: its only env_sky is StartDisabled=True.

    Taking skies[0] returns the right answer on inferno by ORDERING LUCK and
    the wrong one on ancient and anubis. Reported here as (material, source,
    disabled_count) so a consumer can see which route answered.
    """
    enabled = [s for s in skies if not s.get("StartDisabled")]
    if enabled:
        return enabled[0].get("skyname"), "env_sky(StartDisabled=False)"
    if fog:
        return (fog[0].get("cubemapfogskymaterial"),
                "env_cubemap_fog(cubemapfogskymaterial) -- NO ENABLED env_sky")
    if skies:
        return (skies[0].get("skyname"),
                "env_sky(StartDisabled=True) -- ONLY a disabled sky exists")
    return None, "none"


def read_light_environment(vpk_path: str, map_name: str):
    entry, doc = read_entities(vpk_path, map_name)
    ent = _find_class(doc.get("m_entityKeyValues", doc), "light_environment")
    if ent is None:
        raise SystemExit(f"{map_name}: no light_environment entity in {entry}")
    return entry, ent, doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--game-dir", default=os.path.expanduser("~/cs2/game"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for map_name in args.maps:
        vpk = os.path.join(args.game_dir, "csgo", "maps", f"{map_name}.vpk")
        if not os.path.isfile(vpk):
            print(f"{map_name}: MISSING {vpk}")
            continue
        entry, ent, doc = read_light_environment(vpk, map_name)
        # Values arrive as whatever KV3 stored; anything not JSON-serialisable
        # is kept as its repr rather than dropped, so `raw` stays complete.
        raw = {}
        for key, value in sorted(ent.items()):
            try:
                json.dumps(value)
                raw[key] = value
            except (TypeError, ValueError):
                raw[key] = repr(value)
        doc_out = {"schema": "iji/light-environment/v1",
               "map": map_name, "source_vpk": os.path.basename(vpk),
               "source_entry": entry,
               "conversion": "NONE -- sRGB decode and the brightness multiply "
                             "are the renderer's, which prints what it did",
               "raw": raw}
        for key in PROMOTE:
            if key in raw:
                doc_out[key] = raw[key]
        skies, fog = read_sky(doc)
        chosen, how = choose_sky(skies, fog)
        doc_out["env_sky"] = skies
        doc_out["env_cubemap_fog"] = fog
        doc_out["skyname"] = chosen
        doc_out["skyname_source"] = how
        out = os.path.join(args.out_dir, f"{map_name}.light_environment.json")
        with open(out, "w") as fh:
            json.dump(doc_out, fh, indent=1)
        print(f"{map_name}: angles {doc_out.get('angles')} color "
              f"{doc_out.get('color')} brightness {doc_out.get('brightness')} "
              f"angdiam {doc_out.get('angulardiameter')} "
              f"skyambientbounce {doc_out.get('skyambientbounce')}")
        print(f"    skyname {doc_out.get('skyname')}")
        print(f"      via {doc_out.get('skyname_source')}  "
              f"({len(doc_out['env_sky'])} env_sky, "
              f"{len(doc_out['env_cubemap_fog'])} env_cubemap_fog) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
