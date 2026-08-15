#!/usr/bin/env python3
"""cs1k `active_weapon_id` -> the shipped v_model.

THE ID IS NOT AN ITEM DEFINITION INDEX. It is CounterStrike-1K's own ordinal
enum (`schema/weapons.json`, array index == id), which is why ids 21 and 22 --
the 3rd and 4th most common in the corpus -- have no items_game entry at
those indices: nothing was ever meant to look them up there. 21 is ak_47 and
22 is m4a1_s. corpus-fallback verified it empirically against state.bin's
second field `active_weapon` (the coarse category ordinal) over 147,470
frames of match bfec1f06ef8b: every id maps to exactly one category, and the
only id that does not is 0, which the schema itself defines as "unknown or
missing".

So the join is NAME FIRST: id -> cs1k name (WEAPON_NAMES, the enum) -> the
shipped model directory -> the .vmdl_c out of the VPK index.

Two thirds of the names bridge by dropping separators (ak_47 -> ak47,
five_seven -> fiveseven, sawed_off -> sawedoff). The rest do not and cannot
be munged into place -- desert_eagle is `deagle`, m4a1_s is `m4a1_silencer`,
p2000 is `hkp2000`, dual_berettas is `elite`, zeus is `taser`. Those are in
ALIAS below, and every entry was READ off the 41 real directory names under
`weapons/models/` in the shipped paks, not guessed from what the weapon is
called in English.

id 0 is "unknown or missing" per the schema and is treated as ABSENT -- no
viewmodel drawn -- rather than as a weapon.
"""
from __future__ import annotations

import os
import re
import sys


def _items_text(content):
    b = content.find("scripts/items/items_game.txt")
    if not b:
        raise SystemExit("items_game.txt not found in the content mount")
    return b.decode("utf-8", "replace")


def id_to_name(items_text):
    """{def_index: weapon_name} for every entry that has a name.

    The `items` block is `"<index>" { "name" "weapon_x" ... }`. Indices are
    quoted decimal strings at one nesting level; the regex takes the FIRST
    name inside each block, which is the entry's own.
    """
    i = items_text.find('"items"')
    if i < 0:
        raise SystemExit("no items block")
    # BOUND the scan to the items block. Unbounded, `"<digits>" {` also
    # matches numeric keys inside attribute and prefab blocks, which is how
    # id 21 came back as "secondary clip size" -- a parse artefact wearing
    # the shape of an answer.
    j = items_text.index("{", i)
    depth, k = 0, j
    while k < len(items_text):
        if items_text[k] == "{":
            depth += 1
        elif items_text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    seg = items_text[j:k]
    out = {}
    for m in re.finditer(r'"(\d+)"\s*\{', seg):
        idx = int(m.group(1))
        nm = re.search(r'"name"\s*"([^"]+)"', seg[m.end():m.end() + 2000])
        if nm and idx not in out:
            out[idx] = nm.group(1)
    return out


def vpk_index(game):
    """Every entry across the shipped paks, one set.

    Needed because the v_model filename carries a CLASS PREFIX that is not
    in items_game.txt: the AK is `weapons/models/ak47/weapon_rif_ak47.vmdl_c`,
    not `weapon_ak47`. Guessing the prefix per weapon is a table that rots;
    the index knows, so the index is asked.
    """
    import glob
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "vcs"))
    from vpk import VPK
    ent = set()
    for p in sorted(glob.glob(os.path.join(game, "csgo*", "pak01_dir.vpk"))) \
            + sorted(glob.glob(os.path.join(game, "core", "pak01_dir.vpk"))):
        try:
            ent |= set(VPK(p).entries)
        except Exception:                                       # noqa: BLE001
            continue
    return ent


# cs1k enum -> shipped model directory, for the names that do not bridge by
# dropping separators. Read off the directory listing, not from English.
ALIAS = {
    # the grenades' model dirs drop the separator and `incendiary_grenade`
    # and `decoy_grenade` drop the word entirely -- READ off the 10 utility
    # .vmdl_c under weapons/models/, not guessed from English.
    "smoke_grenade": "smokegrenade", "he_grenade": "hegrenade",
    "incendiary_grenade": "incendiarygrenade", "decoy_grenade": "decoy",
    "desert_eagle": "deagle", "m4a1_s": "m4a1_silencer",
    "usp_s": "usp_silencer", "p2000": "hkp2000", "cz75_auto": "cz75a",
    "dual_berettas": "elite", "r8_revolver": "revolver", "zeus": "taser",
    "pp_bizon": "bizon", "sg_553": "sg556", "galil_ar": "galilar",
}
# Not weapons with a v_model: the schema's unknown slot and the grenades /
# bomb, which are separate model families. Reported as absent, never
# defaulted to a rifle.
# ONLY the schema's unknown slot. c4 and the six grenades were listed here as
# "not weapons with a v_model" and the archive says otherwise: every one of
# them ships a viewmodel, two levels deep like the knife --
# weapons/models/grenade/<x>/weapon_<x>.vmdl_c and weapons/models/c4/
# weapon_c4.vmdl_c. The table was asserting an absence the index disproves,
# and because they were excluded before resolution nothing ever looked.
NO_VMODEL = {"unknown"}

# THE KNIFE IS NOT ONE MODEL AND NOT ONE DIRECTORY. Every other weapon is
# `weapons/models/<short>/weapon_*_<short>.vmdl_c`; the knife is
# `weapons/models/knife/<variant>/weapon_knife_<variant>.vmdl_c` -- one level
# deeper, and the basename ends with the VARIANT, not with "knife". That is
# why `resolve_cs1k` returned None for id 1 and reported "no shipped v_model
# located" when the depot ships 22 of them: the shape assumption was wrong,
# not the archive.
#
# WHICH knife is a SIDE question, and items_game.txt answers it by name:
# `weapon_knife` and `weapon_knife_t` each carry their own used_by_classes,
# CT and T respectively. So the two defaults are read, not chosen.
KNIFE_BY_SIDE = {
    "CT": "weapons/models/knife/knife_default_ct/weapon_knife_default_ct.vmdl_c",
    "T": "weapons/models/knife/knife_default_t/weapon_knife_default_t.vmdl_c",
}

# cs1k enum -> the name items_game.txt uses, where it differs from the MODEL
# directory ALIAS above. Read off the 66 `weapon_*` entries in the items
# block: there is no `weapon_glock18` and no `weapon_m4a4` -- those two cost
# the side derivation two weapons until they were looked up rather than
# assumed to match the model path.
ITEMS_ALIAS = {"glock_18": "glock", "m4a4": "m4a1", "decoy_grenade": "decoy"}


def cs1k_dir(name):
    """cs1k enum name -> the `weapons/models/<dir>` component."""
    if name in NO_VMODEL:
        return None
    return ALIAS.get(name, name.replace("_", ""))


def resolve_cs1k(index, name):
    """The shipped v_model for a cs1k name, AT ANY DEPTH under weapons/models.

    The old rule assumed `weapons/models/<short>/weapon_*<short>.vmdl_c`, one
    level. That shape is not universal and the exceptions are a whole class,
    not a special case: the knives live at
    `weapons/models/knife/<variant>/weapon_knife_<variant>.vmdl_c` and every
    grenade at `weapons/models/grenade/<x>/weapon_<x>.vmdl_c`. Anchoring on
    the DIRECTORY is what made 8 ids report "no shipped v_model located" for
    models that ship.

    So the anchor is the FILE: any `weapon_*.vmdl_c` anywhere under
    weapons/models/ whose basename stem ends with the short name. That
    resolves ak47 (weapon_rif_ak47), smokegrenade (weapon_smokegrenade), c4
    (weapon_c4) and decoy (weapon_decoy) by one rule. Part models -- _mag,
    _clip and the grenade pin/spoon -- do not end with the short name and are
    excluded by the same test rather than by a list.
    """
    d = cs1k_dir(name)
    if not d:
        return None
    hits = [k for k in index
            if k.startswith("weapons/models/") and k.endswith(".vmdl_c")
            and os.path.basename(k).startswith("weapon_")
            and os.path.basename(k)[:-7].endswith(d)]
    # shallowest, then shortest: the parent model, never a variant beside it
    return sorted(hits, key=lambda k: (k.count("/"), len(k)))[0] if hits else None


def resolve(index, weapon_name):
    """The shipped v_model for this weapon, matched out of the index.

    `weapons/models/<short>/weapon_*<short>.vmdl_c`, excluding the part
    models (`_mag`, `_clip`, ...) which live beside the parent and are not
    the thing to draw. Returns None if nothing matches -- an id whose model
    is not located is REPORTED, never defaulted to another weapon, because a
    wrong rifle in frame scores as a near miss.
    """
    short = weapon_name.replace("weapon_", "")
    pre = f"weapons/models/{short}/"
    hits = [k for k in index
            if k.startswith(pre) and k.endswith(".vmdl_c")
            and os.path.basename(k).startswith("weapon_")
            and os.path.basename(k)[:-7].endswith(short)]
    return sorted(hits, key=len)[0] if hits else None


def _block_at(s, open_idx):
    """The braced block starting at `open_idx`, matched rather than windowed.

    A fixed-size window round the opening brace was the first attempt and it
    read a NEIGHBOURING item's fields for any entry longer than the window.
    """
    depth, k = 0, open_idx
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx:k + 1]
        k += 1
    return s[open_idx:]


def _top_block(text, name):
    i = text.find('"%s"' % name)
    return _block_at(text, text.index("{", i)) if i >= 0 else ""


def side_table(content):
    r"""{cs1k id: 'T' | 'CT' | 'BOTH'} for every id items_game.txt decides.

    READ from `used_by_classes`, following the `prefab` chain: 41 of the 44
    enum entries carry no used_by_classes of their own and inherit it from a
    prefab, so reading only the item block resolves almost nothing.

    THE CLASS NAME HAS A HYPHEN. Matching `"(\w+)"\s*"1"` drops
    "counter-terrorists" silently and every CT weapon comes back
    T-only -- a parse artefact that reads exactly like an answer, which is
    how the first version of this looked right and was not.
    """
    text = _items_text(content)
    prefabs, items = _top_block(text, "prefabs"), _top_block(text, "items")
    pre = {}
    for m in re.finditer(r'"([\w\-\+\.]+)"\s*\{', prefabs):
        pre.setdefault(m.group(1), _block_at(prefabs, m.end() - 1))

    def classes(blk, depth=0):
        if depth > 6 or not blk:
            return None
        u = re.search(r'"used_by_classes"\s*\{', blk)
        if u:
            got = sorted(set(re.findall(r'"([A-Za-z_-]+)"\s*"1"',
                                        _block_at(blk, u.end() - 1))))
            if got:
                return got
        p = re.search(r'"prefab"\s*"([^"]+)"', blk)
        if p:
            for one in p.group(1).split():
                got = classes(pre.get(one, ""), depth + 1)
                if got:
                    return got
        return None

    by_name = {}
    for m in re.finditer(r'"(\d+)"\s*\{', items):
        blk = _block_at(items, m.end() - 1)
        nm = re.search(r'"name"\s*"([^"]+)"', blk)
        if nm:
            by_name.setdefault(nm.group(1), blk)

    out = {}
    for i, n in enumerate(cs1k_names()):
        blk = (by_name.get("weapon_" + ITEMS_ALIAS[n]) if n in ITEMS_ALIAS
               else by_name.get("weapon_" + ALIAS.get(n, n.replace("_", "")))
               or by_name.get("weapon_" + n))
        c = classes(blk) if blk else None
        if c is None:
            continue
        if c == ["terrorists"]:
            out[i] = "T"
        elif c == ["counter-terrorists"]:
            out[i] = "CT"
        else:
            out[i] = "BOTH"
    return out


def side_of_ids(ids, table):
    """(side, evidence) for a clip, from the SIDE-EXCLUSIVE weapons it holds.

    The cs1k corpus carries no team field -- not in camera.json, not in the
    tick rows (checked across every pair on disk). What it does carry is
    active_weapon_id per frame, and a weapon items_game.txt marks as
    used_by_classes {terrorists} cannot be held by a CT. So a clip that ever
    holds one is that side, and one that holds both is REFUSED rather than
    resolved by majority -- a conflict means the premise is wrong, not that
    the count should decide.

    id 1 (knife) is deliberately NOT evidence even though `weapon_knife` is
    CT-only in items_game: that entry is the CT DEFAULT knife item, and the
    cs1k enum's `knife` is the category both sides hold.
    """
    t = sorted(i for i in ids if table.get(i) == "T" and i != 1)
    ct = sorted(i for i in ids if table.get(i) == "CT" and i != 1)
    if t and ct:
        return None, ("CONFLICT", t, ct)
    if t:
        return "T", ("T-only ids held", t)
    if ct:
        return "CT", ("CT-only ids held", ct)
    return None, ("no side-exclusive id in this clip", sorted(ids))


def resolve_knife(index, side):
    """The default knife .vmdl_c for a side, checked against the index."""
    p = KNIFE_BY_SIDE.get(side)
    return p if p and p in index else None


def cs1k_names():
    """The enum, from the repo's single copy."""
    # Parsed out of the literal rather than imported: cs1k_systems.py pulls
    # in cs1k_corpus at import time, which is not on this path, and a
    # weapon-name table should not require the corpus reader to be
    # installed to be read.
    import ast as _ast
    p = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "cs1k_systems.py")
    tree = _ast.parse(open(p, encoding="utf-8").read(), p)
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Assign) and any(
                getattr(t, "id", None) == "WEAPON_NAMES" for t in n.targets):
            return [e.value for e in n.value.elts]
    raise SystemExit(f"WEAPON_NAMES not found in {p}")


def build(content, index, wanted=None):
    names = cs1k_names()
    ids = sorted(wanted) if wanted else list(range(len(names)))
    sides = side_table(content)
    out = {}
    for i in ids:
        nm = names[i] if 0 <= i < len(names) else None
        row = {"name": nm,
               "vmdl": resolve_cs1k(index, nm) if nm else None,
               "no_vmodel_by_design": bool(nm and nm in NO_VMODEL),
               # 'T' / 'CT' / 'BOTH' from items_game.txt, or absent when that
               # file decides nothing for this id. Carried in the JSON so a
               # consumer can derive a clip's side without re-reading a
               # 2 MB text file and re-walking the prefab chain.
               "side": sides.get(i)}
        if nm == "knife":
            # NOT ONE MODEL. The default knife is per side, and both are
            # carried so the consumer picks with the side it derived rather
            # than this file guessing one.
            row["vmdl_by_side"] = {s: resolve_knife(index, s)
                                   for s in ("T", "CT")}
        out[i] = row
    return out


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", required=True, help="depot game/ root")
    ap.add_argument("--vpk", default=None,
                    help="defaults to <game>/csgo/pak01_dir.vpk")
    ap.add_argument("--ids", default=None,
                    help="comma-separated ids; default = every id in the file")
    ap.add_argument("--from-pairs", default=None,
                    help="scan a pairs tree and use the ids it actually "
                         "carries -- the only set that has to work")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ash_extract as A
    vpk = a.vpk or os.path.join(a.game, "csgo", "pak01_dir.vpk")
    content = A.Content(vpk, a.game, "pak01")

    wanted = None
    if a.ids:
        wanted = {int(x) for x in a.ids.split(",") if x.strip()}
    elif a.from_pairs:
        import glob
        import json as _j
        wanted = set()
        for f in glob.glob(os.path.join(a.from_pairs, "*", "*", "*",
                                        "frames", "*.json")):
            try:
                wanted.add(_j.load(open(f)).get("active_weapon_id"))
            except Exception:                                   # noqa: BLE001
                pass
        wanted.discard(None)

    index = vpk_index(a.game)
    print(f"  vpk index: {len(index):,} entries")
    table = build(content, index, wanted)
    found = {i: v for i, v in table.items()
             if v["vmdl"] or any((v.get("vmdl_by_side") or {}).values())}
    missing = {i: v for i, v in table.items() if i not in found}
    for i in sorted(table):
        v = table[i]
        shown = v["vmdl"] or (
            " | ".join(f"{s}:{p}" for s, p in v["vmdl_by_side"].items())
            if v.get("vmdl_by_side") else "NOT LOCATED")
        print(f"  {i:>3}  {str(v['name']):<24} [{v.get('side') or '?':<4}] "
              f"{shown}")
    print(f"\n  {len(found)} of {len(table)} ids resolve to a shipped v_model")
    if missing:
        print(f"  NOT LOCATED ({len(missing)}): "
              + ", ".join(f"{i}:{table[i]['name']}" for i in sorted(missing)))
    if a.json:
        json.dump({str(k): v for k, v in table.items()}, open(a.json, "w"),
                  indent=1)
        print(f"  -> {a.json}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
