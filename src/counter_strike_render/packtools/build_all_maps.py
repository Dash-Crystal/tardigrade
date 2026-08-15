#!/usr/bin/env python3
"""Every shipped map through the whole pack pipeline, as a SET.

    python build_all_maps.py --game <depot game/> --out-dir /data/.../worlds \
        --ash-dir /data/.../ash --mats-dir <dir of <map>.mats.json>

WHY A DRIVER AND NOT A SHELL LOOP. The corpus is every map this engine
ships, not the six or seven that happened to get built by hand. 43 maps x 8
stages is 344 invocations, and the failure that matters is not a bad file --
per-stage atomic writes make a truncated one essentially impossible -- it is
a PARTIAL SET: 30 maps at one generation beside 13 at another, which a sweep
reads as one thing. So this pre-flights every input before the first build,
writes every artifact through a .tmp and an atomic replace, keeps going past
a stage that fails, and ends with a coverage table that is computed from the
artifacts on disk rather than from these logs. A run that reports success
and a set that IS coherent are different claims and only the second matters.

STAGES, in dependency order. Each one is skipped when its output exists, so
a re-run resumes rather than rebuilds, and a stage that fails is recorded as
a GAP with the tool's own last line -- some maps genuinely ship no lightmap
and no sky, and "record, do not stall" is the instruction and the right
behaviour.

  ash     ash_extract.py            <map>.vpk        -> <ash>/<map>.ash/
  pack    ash_to_world.py           the ash dir      -> <map>.pt
                                    with --overlay-pages and --ao-pages, so a
                                    pack never ships albedo-only again
  irr     lightmap_extract.py       lightmaps/irradiance.vtex_c
                                                     -> <map>.irradiance.npy
  shadow  lightmap_extract.py       lightmaps/direct_light_shadows.vtex_c
                                            -> <map>.direct_light_shadows.npy
  sun     light_environment_extract -> <map>.light_environment.json, which
                                    also carries `skyname`
  sky     the skyname JOIN, then sky_extract.py -> <map>.sky_cube.npz
  fam     fast_pack_fam.py          --vmats + --align <map>.pt
                                                     -> <map>.fam_side.pt
  repack  ash_to_world.py           the ash dir + <map>.fam_side.pt
                                                     -> <map>.pt, REWRITTEN

THE PACK RUNS TWICE, and the second run is not a retry. ash_to_world.py's
normal-map block is gated on a --mat-paths side table (its :1428 / :944-954),
and the only side table that covers a map's materials is <map>.fam_side.pt --
which the `fam` stage produces FROM the pack. Pack -> fam -> repack resolves
that circle by running the pack again with the input it could not have had
the first time. Measured before this stage existed: 0 of 43 base packs in
worlds/ carried mat_nrm_basis while all 43 fam_side tables carried the
columns it needs, and ar_pool_day's aux was (41,64,64,4) against the ws1
lineage's (204,64,64,4).

`repack` is the ONE stage whose skip condition is not "its output exists" --
it has no output of its own, it rewrites the pack. It skips when the pack
already CARRIES mat_nrm_basis. That difference is deliberate: every other
stage's file-exists skip is why seven flagship maps (de_ancient, de_anubis,
de_dust2, de_inferno, de_mirage, de_nuke, de_overpass) are still pure 4-layer
albedo-only scaffolds with no AO either -- their .pt was written by an
earlier pass and no later run would ever touch it.

THE SKY JOIN IS A READ AND IS PERFORMED HERE, not a name match.
sky_extract.py refuses to do it on purpose -- its docstring shows why, with
de_ancient as the case: no skybox texture in the shared pak name-matches it,
and `sky_hr_aztec_02` is the thematically plausible guess. The entity route
answers it: the light_environment sidecar's `skyname` is a .vmat path, that
material is `sky.vfx`, and its ONE texture parameter `g_tSkyTexture` names
the .vtex_c. Read that way de_ancient resolves to
`materials/skybox/sky_hr_aztec_02_v1_exr_34a0bfbd.vtex` -- the same asset the
guess would have reached, which is exactly why the guess was not safe: a
method that is right where you can check it and wrong where you cannot is
indistinguishable from a good one until it matters.

--height-pages is NOT passed. It is diagnostic and it MEASURED WORSE (ncc
-0.0449 -> -0.2067 on de_inferno pose 0); see its own help.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("ash", "pack", "irr", "shadow", "sun", "sky", "fam", "repack")


def pack_has_key(path, key):
    """Does this .pt carry `key`, without paying to load 1-3 GB of tensors?

    Returns (bool, route) -- the ROUTE is returned and printed because two
    methods answer this question and a caller that does not know which one
    spoke cannot tell "absent" from "could not look".

    torch.save writes a zip whose `data.pkl` holds the key STRINGS; the
    tensor storages are separate members. Reading that one small member and
    searching it for the key is milliseconds against a multi-second full
    load. If anything about that shape fails -- a legacy non-zip .pt, a
    renamed member -- it falls back to the full torch.load rather than
    guessing, because a false "absent" here triggers a rebuild of a pack
    that did not need one and a false "present" leaves the defect in place.
    """
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            pkls = [n for n in z.namelist() if n.endswith("/data.pkl")
                    or n == "data.pkl"]
            if len(pkls) != 1:
                raise ValueError("%d data.pkl members" % len(pkls))
            return (key.encode() in z.read(pkls[0])), "data.pkl"
    except Exception as ex:                                    # noqa: BLE001
        try:
            import torch
            d = torch.load(path, map_location="cpu", weights_only=False)
            return (key in d), "torch.load (data.pkl route: %s)" % (
                type(ex).__name__)
        except Exception as ex2:                               # noqa: BLE001
            return None, "UNREADABLE (%s / %s)" % (type(ex).__name__,
                                                   type(ex2).__name__)


def _run(cmd, log):
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    tail = ""
    try:
        lines = [l for l in open(log).read().strip().split("\n") if l.strip()]
        tail = lines[-1] if lines else ""
    except OSError:
        pass
    return p.returncode, tail


def _tmp_for(final):
    """A temp name that KEEPS THE EXTENSION.

    numpy's save/savez APPEND `.npy`/`.npz` when the filename does not
    already end in one, so `<map>.irradiance.npy.tmp` made lightmap_extract
    write `<map>.irradiance.npy.tmp.npy` and this driver then reported the
    stage as failed because the file it looked for was not there. The tool
    had worked; the temp name broke it. Insert `.tmp` before the suffix.
    """
    root, ext = os.path.splitext(final)
    return root + ".tmp" + ext


def _replace(tmp, final):
    """Atomic per artifact. A reader can never see a half-written file."""
    if os.path.isdir(tmp):
        if os.path.isdir(final):
            shutil.rmtree(final)
        os.replace(tmp, final)
    else:
        os.replace(tmp, final)


def paths(a, m):
    return {
        "vpk": os.path.join(a.game, "csgo", "maps", m + ".vpk"),
        "ash": os.path.join(a.ash_dir, m + ".ash"),
        "mats": os.path.join(a.mats_dir, m + ".mats.json"),
        "pack": os.path.join(a.out_dir, m + ".pt"),
        "irr": os.path.join(a.out_dir, m + ".irradiance.npy"),
        "shadow": os.path.join(a.out_dir, m + ".direct_light_shadows.npy"),
        "sun": os.path.join(a.out_dir, m + ".light_environment.json"),
        "sky": os.path.join(a.out_dir, m + ".sky_cube.npz"),
        "fam": os.path.join(a.out_dir, m + ".fam_side.pt"),
    }


def sky_texture(a, m, p):
    """(vtex path, note) from the sidecar's skyname. THE JOIN, read."""
    if not os.path.isfile(p["sun"]):
        return None, "no light_environment sidecar, so no skyname to read"
    sn = (json.load(open(p["sun"])) or {}).get("skyname")
    if not sn:
        return None, "the sidecar carries no skyname (map has no env_sky)"
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "vcs"))
    import ash_extract as A
    # THE MAP'S OWN VPK. Passing the shared pak01 searched only the shared
    # archives, and a map can ship its sky in its OWN vpk: ar_pool_day's
    # `materials/skybox/poolday_skybox.vmat` and the .vtex_c it names are
    # both inside ar_pool_day.vpk and in no shared archive. (Not community
    # addons -- this depot has no csgo_community_addons directory at all,
    # which I checked rather than assuming after the first fix appeared to
    # work.) The map vpk goes first in Content's search order, so this also
    # lets a map override a shared material, which is the engine's own
    # precedence.
    content = A.Content(p["vpk"], a.game, m)
    row, err = A.read_material(content, sn)
    if err:
        return None, f"skyname {sn}: {err}"
    tp = row["texture_params"]
    tex = tp.get("g_tSkyTexture")
    if not tex:
        return None, (f"skyname {sn} is {row['shader_name']} with no "
                      f"g_tSkyTexture; params {sorted(tp)}")
    # WHICH ARCHIVE HOLDS IT, because sky_extract takes a single --pak and
    # the texture is not always in the shared one.
    entry = tex + "_c"
    pak = None
    for label, v in content.vpks:
        if entry in v.entries:
            pak = p["vpk"] if label == "<map>" else os.path.join(a.game, label)
            break
    if pak is None:
        return None, (f"{sn} :: g_tSkyTexture -> {tex} :: that .vtex_c is in "
                      f"none of {[l for l, _v in content.vpks]}")
    return (entry, pak), f"{sn} :: g_tSkyTexture -> {tex} (in {pak})"


def build_map(a, m, log_dir):
    p = paths(a, m)
    got, gaps = {}, {}
    L = lambda s: os.path.join(log_dir, f"{m}.{s}.log")     # noqa: E731

    def done(stage):
        got[stage] = True

    def gap(stage, why):
        gaps[stage] = why

    # --- ash -----------------------------------------------------------
    if os.path.isdir(p["ash"]):
        done("ash")
    elif not os.path.isfile(p["vpk"]):
        gap("ash", "no .vpk in the depot")
    else:
        tmp = p["ash"] + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        rc, tail = _run([a.python, os.path.join(HERE, "ash_extract.py"),
                         "--map", p["vpk"], "--out", tmp,
                         "--game-dir", a.game], L("ash"))
        if rc == 0 and os.path.isdir(tmp):
            _replace(tmp, p["ash"])
            done("ash")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            gap("ash", tail[:160])

    # --- pack ----------------------------------------------------------
    if os.path.isfile(p["pack"]):
        done("pack")
    elif "ash" not in got:
        gap("pack", "no ash extraction to convert")
    else:
        tmp = _tmp_for(p["pack"])
        cmd = [a.python, os.path.join(HERE, "ash_to_world.py"),
               "--ash", p["ash"], "--out", tmp,
               "--tex-dir", os.path.join(p["ash"], "textures"),
               "--overlay-pages", "--ao-pages"]
        rc, tail = _run(cmd, L("pack"))
        if rc == 0 and os.path.isfile(tmp):
            _replace(tmp, p["pack"])
            done("pack")
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            gap("pack", tail[:160])

    # --- irradiance and direct_light_shadows ----------------------------
    for stage, entry in (("irr", "lightmaps/irradiance.vtex_c"),
                         ("shadow",
                          "lightmaps/direct_light_shadows.vtex_c")):
        if os.path.isfile(p[stage]):
            done(stage)
            continue
        if not os.path.isfile(p["vpk"]):
            gap(stage, "no .vpk")
            continue
        tmp = _tmp_for(p[stage])
        rc, tail = _run([a.python, os.path.join(HERE, "lightmap_extract.py"),
                         "--map", p["vpk"], "--out", tmp,
                         "--entry", entry], L(stage))
        if rc == 0 and os.path.isfile(tmp):
            _replace(tmp, p[stage])
            done(stage)
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            gap(stage, tail[:160])

    # --- sun ------------------------------------------------------------
    if os.path.isfile(p["sun"]):
        done("sun")
    else:
        stage_dir = os.path.join(log_dir, m + ".sun.d")
        os.makedirs(stage_dir, exist_ok=True)
        rc, tail = _run([a.python,
                         os.path.join(HERE, "light_environment_extract.py"),
                         m, "--game-dir", a.game, "--out-dir", stage_dir],
                        L("sun"))
        made = glob.glob(os.path.join(stage_dir, "*.light_environment.json"))
        if rc == 0 and made:
            _replace(made[0], p["sun"])
            done("sun")
        else:
            gap("sun", tail[:160])
        shutil.rmtree(stage_dir, ignore_errors=True)

    # --- sky ------------------------------------------------------------
    if os.path.isfile(p["sky"]):
        done("sky")
    else:
        found, note = sky_texture(a, m, p)
        if not found:
            gap("sky", note)
        else:
            tex, pak = found
            tmp = _tmp_for(p["sky"])
            rc, tail = _run([a.python, os.path.join(HERE, "sky_extract.py"),
                             "--pak", pak,
                             "--tex", tex, "--out", tmp], L("sky"))
            if rc == 0 and os.path.isfile(tmp):
                _replace(tmp, p["sky"])
                # the tool writes a sidecar json beside its --out
                sj = os.path.splitext(tmp)[0] + ".json"
                if os.path.isfile(sj):
                    _replace(sj, os.path.splitext(p["sky"])[0] + ".json")
                done("sky")
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
                gap("sky", f"{note} :: {tail[:120]}")

    # --- fam_side -------------------------------------------------------
    if os.path.isfile(p["fam"]):
        done("fam")
    elif "pack" not in got:
        gap("fam", "no world pack to --align against")
    elif not os.path.isfile(p["mats"]):
        gap("fam", "no <map>.mats.json from vmat_extract")
    else:
        tmp = _tmp_for(p["fam"])
        cmd = [a.python,
               os.path.join(HERE, "fam_analysis", "fast_pack_fam.py"),
               "--vmats", p["mats"], "--align", p["pack"], "--out", tmp]
        vtex = os.path.join(p["ash"], "textures")
        if os.path.isdir(vtex):
            cmd += ["--vtex-root", vtex]
        rc, tail = _run(cmd, L("fam"))
        if rc == 0 and os.path.isfile(tmp):
            _replace(tmp, p["fam"])
            done("fam")
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            gap("fam", tail[:160])

    # --- repack: the pack stage's SECOND pass, with --mat-paths ----------
    # THE CIRCULAR DEPENDENCY, resolved by running the pack twice rather
    # than by dropping an input.
    #
    # ash_to_world.py gates the whole normal-map block on
    # `if tex_dir and nrm_slot is not None:` (:1428), and nrm_slot is only
    # populated when a --mat-paths / --mat-paths-dir side table is adopted
    # (:944-954). Without it there is no mat_nrm_basis, no
    # mat_normal_pages, and no normal aux layers -- and the pack stage
    # above has NEVER passed that flag, in any revision of this driver.
    # The file it wants is <map>.fam_side.pt, which the `fam` stage two
    # blocks up produces from THIS pack and leaves sitting beside it.
    #
    # MEASURED before this stage existed: 0 of 43 base packs in worlds/
    # carried mat_nrm_basis, while all 43 fam_side tables carried the
    # nrm_slot1/nrm_enc1/nrm_slot2/nrm_enc2/nrm_enc_basis columns it needs.
    # ar_pool_day: aux (41,64,64,4) here against (204,64,64,4) in the ws1
    # lineage. The positive control is de_inferno_nrm.pt, a manual
    # single-map invocation that DID pass --mat-paths and logs "111
    # distinct pages, 115 aux layers".
    #
    # THE CONDITION IS THE KEY, NOT THE FILE. Every other stage skips when
    # its output EXISTS, and that is exactly why seven flagship maps
    # (de_ancient, de_anubis, de_dust2, de_inferno, de_mirage, de_nuke,
    # de_overpass) are still pure 4-layer albedo-only scaffolds with no AO
    # either: their .pt was written by an earlier pass and no later run
    # would ever touch it. This stage asks whether the pack CARRIES
    # mat_nrm_basis, so a pack built without the input is upgraded and a
    # pack built with it is left alone.
    if "pack" not in got:
        gap("repack", "no world pack to repack")
    elif "fam" not in got:
        gap("repack", "no <map>.fam_side.pt, which is the --mat-paths input "
                      "the normal-map block needs")
    else:
        _has, _route = pack_has_key(p["pack"], "mat_nrm_basis")
        if _has is None:
            gap("repack", "could not read the pack to ask whether it already "
                          "carries mat_nrm_basis: " + _route)
        elif _has:
            done("repack")
        else:
            tmp = _tmp_for(p["pack"])
            cmd = [a.python, os.path.join(HERE, "ash_to_world.py"),
                   "--ash", p["ash"], "--out", tmp,
                   "--tex-dir", os.path.join(p["ash"], "textures"),
                   "--mat-paths", p["fam"],
                   "--overlay-pages", "--ao-pages"]
            rc, tail = _run(cmd, L("repack"))
            _ok = False
            if rc == 0 and os.path.isfile(tmp):
                # DID IT ACTUALLY GAIN THE KEY. rc == 0 is not the answer:
                # ash_to_world REFUSES a side table whose rows do not cover
                # this pack's materials and carries on WITHOUT it, exiting
                # 0 with a log line and no normals. Replacing the good pack
                # with that one would be a silent downgrade.
                _got_key, _r2 = pack_has_key(tmp, "mat_nrm_basis")
                if _got_key:
                    _replace(tmp, p["pack"])
                    done("repack")
                    _ok = True
                else:
                    gap("repack", "ran clean but the rebuilt pack STILL has "
                                  "no mat_nrm_basis (%s) -- the side table "
                                  "was refused or no material decoded a "
                                  "normal map; the existing pack is KEPT. "
                                  "Last line: %s" % (_r2, tail[:100]))
            else:
                gap("repack", tail[:160])
            if not _ok and os.path.exists(tmp):
                os.remove(tmp)
    return got, gaps


def coverage(a, maps):
    """Computed from the ARTIFACTS, never from the run's own bookkeeping."""
    rows = []
    for m in maps:
        p = paths(a, m)
        r = {}
        for s in STAGES:
            if s == "ash":
                r[s] = os.path.isdir(p[s])
            elif s == "repack":
                # `repack` has NO artifact of its own -- it rewrites the
                # pack in place. Its coverage is the CONTENT question the
                # stage itself asks, so the table cannot report a stage as
                # covered on the strength of a file the stage never wrote.
                r[s] = bool(os.path.isfile(p["pack"])
                            and pack_has_key(p["pack"], "mat_nrm_basis")[0])
            else:
                r[s] = os.path.isfile(p[s])
        rows.append((m, r))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ash-dir", required=True)
    ap.add_argument("--mats-dir", required=True)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--maps", default=None,
                    help="comma-separated; default every .vpk in the depot")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-only", action="store_true",
                    help="print the coverage table and exit, building nothing")
    a = ap.parse_args(argv)

    maps = ([x for x in a.maps.split(",") if x.strip()] if a.maps else
            sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(a.game, "csgo", "maps",
                                                   "*.vpk"))))
    log_dir = a.log_dir or os.path.join(a.out_dir, "_build_logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.ash_dir, exist_ok=True)

    if not a.report_only:
        # PRE-FLIGHT, before the first build: every input for every map.
        missing = [(m, k) for m in maps
                   for k in ("vpk", "mats")
                   if not os.path.isfile(paths(a, m)[k])]
        print(f"pre-flight: {len(maps)} maps, "
              f"{len(missing)} missing inputs"
              + ("" if not missing else
                 " -- those stages will GAP, the rest still build"))
        for m, k in missing[:20]:
            print(f"  {m}: no {k}")
        if a.limit:
            maps = maps[:a.limit]
        t0 = time.time()
        for i, m in enumerate(maps, 1):
            ts = time.time()
            got, gaps = build_map(a, m, log_dir)
            print(f"[{i}/{len(maps)}] {m:<26} "
                  f"{'+'.join(s for s in STAGES if s in got) or 'none'}"
                  + (f"   GAP {', '.join(f'{k}({v})' for k, v in gaps.items())}"
                     if gaps else "")
                  + f"   {time.time() - ts:.0f}s", flush=True)
        print(f"\nsweep wall clock {time.time() - t0:.0f}s")

    rows = coverage(a, maps)
    print(f"\nCOVERAGE, read off the artifacts ({len(rows)} maps)")
    print("%-26s %s" % ("map", "  ".join(f"{s:<6}" for s in STAGES)))
    for m, d in rows:
        print("%-26s %s" % (m, "  ".join("%-6s" % ("yes" if d[s] else "-")
                                         for s in STAGES)))
    print("\nper stage:")
    for s in STAGES:
        n = sum(1 for _m, d in rows if d[s])
        print(f"  {s:<7} {n:>3} of {len(rows)}")
    out = os.path.join(a.out_dir, "coverage.json")
    json.dump({"maps": {m: d for m, d in rows}}, open(out, "w"), indent=1)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
