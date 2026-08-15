"""End-to-end: one cs-tard21 match -> three view-matrix VIDEO sets + GT eval.

Runs on the dual-homed laptop. Phases (each resumable, each fail-loud):

  stage   terul assets -> laptop cache -> render node, verified by BYTES.
          The node ends up with everything the full draw call needs; there
          is no reduced-asset path (owner ruling).
  gen     from the .tard logstream: per-(player,view) camera paths
          (ego = the player's eye verbatim, steamid carried so the core
          excludes the subject's own model; gow/sm64 = retrocausal
          third-person, NO steamid so the subject IS drawn), the
          playermodels rows JSON (all 10 players, team ct/t), the render
          profile, and the eval camera from the GT capture's provenance.
  render  on the node: per (view, player) single-camera render, full
          profile, --out mp4  ->  3 video sets x 10 players.
  eval    render the GT capture's exact poses with the same profile and
          score frame-by-frame (metric_compare on the node).
  pull    mp4s + eval numbers -> docs/projects/counter-strike-sft/figures/.

Usage:  python3 tools/cs_match_three_view_videos.py <phase>
        phases: stage gen render eval pull all
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "gpu_render_libs"))

TERUL = "terul-cluster"
NODE = "ws-1"                                   # 2x RTX 5090, mortenv has
NODE_ROOT = "matchrender"                       # nvdiffrast (fl renders RC=0)
NODE_PY = "~/mortenv/bin/python"
CACHE = os.path.expanduser("~/dox/runs/cs-mirage-stage")
TARD = os.path.expanduser("~/dox/runs/tardigrade-v21-0.tard")
FIGS = os.path.join(REPO, "docs/projects/counter-strike-sft/figures",
                    "tard21_match_views")
FPS = 32                                         # output video rate


def derive_tick_rate(rec):
    """DERIVE the recording's tick rate from movement physics -- the
    dataset name says 128 Hz and the data says otherwise (measured: the
    moving-segment displacement plateau is ~498 units/128 ticks; CS2
    full-run is 250 u/s, so 128 ticks ~= 2 s => 64 Hz; assuming 128 made
    every video play at 2x). The header/name is a COMMENT (D10); there
    is no rate field to READ, so the constant is derived from the
    run-speed plateau and printed. Refuses if the plateau matches no
    plausible rate within 25%."""
    import math as _m
    ds = []
    for pi in range(len(rec.players)):
        p = rec.player(pi)
        for t0 in range(0, 30000, 128):
            a, b = p.sample(t0), p.sample(t0 + 128)
            if a and b and a.position.present and b.position.present:
                d = _m.dist((a.position.x, a.position.y),
                            (b.position.x, b.position.y))
                if d > 5:
                    ds.append(d)
    ds.sort()
    plateau = ds[int(len(ds) * 0.90) - 1]        # below the teleport tail
    for rate in (128, 64, 32, 16):
        expect = 250.0 * 128.0 / rate            # run speed over 128 ticks
        if abs(plateau - expect) / expect < 0.25:
            print(f"tick rate DERIVED: {rate} Hz (p90 moving displacement "
                  f"{plateau:.0f} units/128t vs full-run {expect:.0f}; "
                  f"n={len(ds)} segments)")
            return rate
    raise SystemExit(
        f"REFUSING: displacement plateau {plateau:.0f} units/128t matches "
        f"no rate in (128,64,32,16) at 250 u/s run speed -- the recording's "
        f"physics are unexplained and any stride would be a guess")
VIEWS = ("ego", "gow", "sm64", "ac")
# ws-1 5090s are 32 GiB: --lod-per-frame drops the MAT_EXT2 gather
# 19.6 GiB -> ~0.6 GiB (measured on the fl renders, peak 6.9 GiB at
# 640x360, RC=0), which unlocks the 960x540 rerun the nkcut2 note asked
# for once a free GPU was back.
W, H = 960, 540
BATCH = 24                                       # proven 5090 config

def fitted_map_name():
    from tardigrade_v21 import TardigradeV21Recording
    from map_identity import identify_map
    rec = TardigradeV21Recording(open(TARD, "rb").read())
    return identify_map(rec)


# per-map GT source for the eval phase (READ captures, never cross-map).
# de_inferno has NO registered source: nkcut2:gt48's own provenance says
# map=de_mirage (caught by the default-deny guard 2026-08-12) -- a fresh
# same-map capture via the on-demand GT service is the only way in.
GT_BY_MAP = {"de_mirage": ("terul", "/data/cs2-artifacts/gtq_ondemand/calib_b_mirage_p0")}
GT_DIR = "/data/cs2-artifacts/gtq_ondemand/calib_b_mirage_p0"


def sh(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), **kw)
    if r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd}")
    return r


def remote_size(host, path):
    r = subprocess.run(["ssh", host, f"stat -c %s '{path}' 2>/dev/null || stat -f %z '{path}'"],
                       capture_output=True, text=True)
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else -1


# --------------------------------------------------------------- stage ----
def stage():
    m = fitted_map_name()
    # NODE INPUTS ARE NOT SELECTED HERE. The node carries the WHOLE
    # content tree (owner rule 2026-08-13); provision_render_node.py
    # materializes and verifies it, and the renderer resolves every input
    # family from it by fitted map name, refusing BY NAME on gaps. The
    # only per-map pull below is the LAPTOP's copy of the world pack --
    # the frustum solver's local working input, not node provisioning.
    sh([sys.executable,
        os.path.join(REPO, "tools/provision_render_node.py"), NODE,
        "--root", f"{NODE_ROOT}/content"])
    dst = os.path.join(CACHE, m + ".pt")
    want = remote_size(TERUL, f"/data/cs2-artifacts/worlds/{m}.pt")
    if want <= 0:
        raise SystemExit(f"REFUSING: no world pack for fitted map {m}")
    if not (os.path.exists(dst) and os.path.getsize(dst) == want):
        sh(["rsync", "-a",
            f"{TERUL}:/data/cs2-artifacts/worlds/{m}.pt", dst])
    print(f"  solver pack {m}.pt: {want:,} B OK")
    # GT staging is PER-MAP (READ captures, never cross-map). A missing
    # same-map GT refuses the EVAL phase only -- named here, not fatal to
    # the 40 match videos.
    gt_src = GT_BY_MAP.get(m)
    if gt_src is None:
        print(f"  REFUSED GT stage: no GT source registered for {m}")
    else:
        host = TERUL if gt_src[0] == "terul" else "nkcut2"
        r = subprocess.run(["rsync", "-a", f"{host}:{gt_src[1]}/",
                            os.path.join(CACHE, "gt_calib_p0/")])
        n_gt = len([f for f in os.listdir(os.path.join(CACHE, "gt_calib_p0"))
                    if f.endswith(".png")]) if r.returncode == 0 else 0
        if n_gt == 0:
            print(f"  REFUSED GT stage: {host}:{gt_src[1]} unreachable or "
                  f"empty -- eval phase will refuse; match videos proceed")
        else:
            print(f"  staged GT: {n_gt} pose PNGs + provenance from {host}")

    # -> node: code, recording, GT -- never asset selections
    sh(["ssh", NODE, f"mkdir -p {NODE_ROOT}/gt {NODE_ROOT}/inputs "
                     f"{NODE_ROOT}/out"])
    sh(["rsync", "-a", os.path.join(CACHE, "gt_calib_p0"),
        f"{NODE}:{NODE_ROOT}/assets/"]) if os.path.isdir(
        os.path.join(CACHE, "gt_calib_p0")) else None
    sh(["rsync", "-a", os.path.join(REPO, "src"), f"{NODE}:{NODE_ROOT}/code/"])
    # render_config resolves the preset manifest at
    # code/docs/projects/counter-strike-sft/ relative to its own file --
    # without it every --preset run REFUSES at startup (measured: all 4
    # views, 2026-08-12).
    sh(["ssh", NODE, f"mkdir -p {NODE_ROOT}/code/docs/projects/counter-strike-sft"])
    sh(["rsync", "-a",
        os.path.join(REPO, "docs/projects/counter-strike-sft/GT_QUALITY_LEVEL_MANIFEST.json"),
        f"{NODE}:{NODE_ROOT}/code/docs/projects/counter-strike-sft/"])
    sh(["rsync", "-a", TARD, f"{NODE}:{NODE_ROOT}/match.tard"])
    print("stage: node provisioned (content tree verified by the "
          "provision manifest) + code/recording shipped")


# ----------------------------------------------------------------- gen ----
def gen():
    from tardigrade_v21 import TardigradeV21Recording
    import demo_camera as dc
    from demo_match_export import build_timeline_from_tardigrade

    rec = TardigradeV21Recording(open(TARD, "rb").read())
    # MAP IDENTITY IS FIT FROM DATA WITH REFUSAL (iji_model.map_identity).
    # There is no argument to pass a map name; nothing can pass it wrongly.
    from map_identity import identify_map
    fitted_map = identify_map(rec)
    teams = {}
    for i, p in enumerate(rec.players):
        teams[i] = "ct" if int(getattr(p, "team", 0)) == 3 else "t"
    print("teams:", teams)

    out = os.path.join(CACHE, "inputs")
    os.makedirs(out, exist_ok=True)

    # playermodels rows: every tick we render, every player, world truth.
    rows_by_tick = {}
    rate = derive_tick_rate(rec)
    stride = max(1, rate // FPS)                 # realtime at FPS out
    tls = {}
    #: recording display name -> weapon bundle short, via the READ table
    #: in packtools/weapon_ids.py (dual_berettas is `elite` etc.); the
    #: normalized direct hit covers the rest. An unmapped weapon is NOT
    #: drawn and the pass names it per entity -- never substituted.
    #: READ against the actual bundle shorts in content/weapons (ls'd,
    #: not derived): deagle/usp_silencer/c4/hegrenade/... The first map
    #: emitted deserteagle/usps/c4explosive -- name-derived joins that
    #: would have silently drawn nothing, the exact defect class.
    _wspecial = {"dualberettas": "elite", "r8revolver": "revolver",
                 "zeus": "taser", "zeusx27": "taser", "p2000": "hkp2000",
                 "deserteagle": "deagle", "usps": "usp_silencer",
                 "c4explosive": "c4", "highexplosivegrenade": "hegrenade",
                 "m4a1s": "m4a1_silencer", "cz75auto": "cz75a",
                 "galilar": "galilar", "incendiarygrenade": None}
    seen_weap = {}

    def _weapon_short(disp):
        if not disp:
            return None
        k = str(disp).lower().replace(" ", "").replace("-", "")
        return _wspecial.get(k, k)

    for i in range(len(rec.players)):
        tl = build_timeline_from_tardigrade(rec, i, 0, None, stride)
        tl.tick_rate = rate
        tls[i] = tl
        p = rec.player(i)
        for r in tl.focus_rows:
            s = p.sample(int(r["tick"]))
            wv = (s.events or {}).get("weapon_index") if s else None
            wname = wv.value if wv is not None and wv.present else None
            if wname:
                seen_weap[str(wname)] = _weapon_short(wname)
            rows_by_tick.setdefault(int(r["tick"]), []).append({
                "tick": int(r["tick"]), "steamid": i, "team": teams[i],
                "x": r["x"], "y": r["y"], "z": r["z"],
                "yaw": r["yaw_degrees"],
                # KEY READ from pm_weapon_for (71_muzzle_flash:1617): the
                # row field is active_weapon_id (carrying the display
                # name); emitting active_weapon_name left every hand
                # empty across a full 40-video render -- the flag's help
                # text describes the MAP's keys, not the row field.
                "active_weapon_id": str(wname) if wname else None,
                "is_alive": bool(r.get("is_alive", True)),
            })
    rows_doc = {"schema": "iji/cs2-demo-playermodels/v1",
                "ticks": {str(k): v for k, v in sorted(rows_by_tick.items())}}
    json.dump(rows_doc, open(os.path.join(out, "playermodels.json"), "w"))
    json.dump(seen_weap, open(os.path.join(out, "weapon_map.json"), "w"))
    print(f"rows: {len(rows_by_tick)} ticks x 10 players; weapons seen: "
          f"{seen_weap}")

    def write_path(doc, name):
        json.dump(doc, open(os.path.join(out, name), "w"))
        print(f"  {name}: {doc['n_ticks']} ticks, "
              f"future_steered={doc.get('n_future_steered', 0)}")

    for i in range(len(rec.players)):
        tl = tls[i]
        # ego: the eye verbatim; steamid CARRIED so the core excludes the
        # subject's own head from its own camera.
        ego_ticks = [{
            "tick": int(r["tick"]), "x": r["x"], "y": r["y"], "z": r["z"],
            "eye_z": r["eye_z"], "yaw_degrees": r["yaw_degrees"],
            "pitch_degrees": r["pitch_degrees"], "fire": bool(r.get("fire")),
            "is_alive": bool(r.get("is_alive", True)),
            "uses_future_sample": False,
        } for r in tl.focus_rows]
        write_path({"schema": "iji/cs2-demo-ego-camera-path/v1",
                    "variant": "ego", "tick_rate": 32, "steamid": i,
                    "map_name": fitted_map, "n_ticks": len(ego_ticks),
                    "n_future_steered": 0, "ticks": ego_ticks},
                   f"p{i}_ego.camera.json")
        # third person: retrocausal controllers; NO steamid -> subject drawn.
        for v in ("gow", "sm64"):
            doc = dc.compute_camera_path(tl, dc.VARIANTS[v])
            doc["tick_rate"] = 32
            doc.pop("steamid", None)
            doc["map_name"] = fitted_map
            write_path(doc, f"p{i}_{v}.camera.json")

    # ac view: the 4th viewmatrix, from the verified lock-on adapter.
    sys.path.insert(0, os.path.join(REPO, "src", "gpu_render_libs"))
    from ac_lockon_adapter import adapt_match
    ac = adapt_match(rec, stride=stride, tick_rate=rate)
    for pdoc in ac['players']:
        i = pdoc['player']
        json.dump(pdoc['camera_path'],
                  open(os.path.join(out, f'p{i}_ac.camera.json'), 'w'))
    json.dump(ac, open(os.path.join(out, 'ac_adapter.json'), 'w'))
    print(f"ac view: {len(ac['players'])} camera paths + tapes emitted")

    # per-agent HUD state, drawn by the RENDERER into each agent's screen
    # space (stages/89_hud). Columns READ from the recording: weapon,
    # fire, scope. health/ammo are NOT on the tard21 wire (probed) and
    # are not invented -- the renderer prints that absence per agent.
    for i in range(len(rec.players)):
        p = rec.player(i)
        rows_h = []
        for r in tls[i].focus_rows:
            s = p.sample(int(r["tick"]))
            ev = s.events if s else {}
            wv = ev.get("weapon_index") if ev else None
            sv = ev.get("scope") if ev else None
            rows_h.append({
                "fire": bool(r.get("fire")),
                "weapon": (wv.value if wv is not None and wv.present
                           else None),
                "scope": bool(sv.value) if sv is not None and sv.present
                         else False})
        json.dump({"scheme": "cs2", "rows": rows_h},
                  open(os.path.join(out, f"p{i}_hud.json"), "w"))
        # ac view gets the FCS wireframe driven by the adapter's own
        # lock ladder + command tape + camera yaws
        pdoc = next(pp for pp in ac["players"] if pp["player"] == i)
        json.dump({"scheme": "ac",
                   "lock": pdoc["overlay_track"]["lock"],
                   "turn": pdoc["command_tape"]["turn"],
                   "yaw_degrees": [t["yaw_degrees"] for t in
                                   pdoc["camera_path"]["ticks"]]},
                  open(os.path.join(out, f"p{i}_achud.json"), "w"))
    print("hud: 10 cs2 + 10 ac state files emitted (weapon/fire/scope "
          "READ; health/ammo absent on this wire, stated)")

    # FRUSTUM-SOLVE every third-person path against the fitted map's own
    # pack BEFORE shipping (eye-in-solid + LOS, occupancy in SRC space --
    # the pre-fix solver was VACUOUS over the city, which retracts the
    # no-clip claim on every earlier third-person render). Solved in
    # place: the render phase consumes p{i}_{v}.camera.json and there is
    # no unsolved path left to ship by accident. Ego paths are recorded
    # eyes, in-world by construction.
    from frustum_camera_solver import solve_camera_json
    pack = os.path.join(CACHE, fitted_map + ".pt")
    for i in range(len(rec.players)):
        for v in ("gow", "sm64", "ac"):
            p = os.path.join(out, f"p{i}_{v}.camera.json")
            r = solve_camera_json(pack, p, p)
            if r["after_eye_in_solid"]:
                raise SystemExit(
                    f"REFUSING p{i}_{v}: {r['after_eye_in_solid']} frames "
                    f"still eye-in-solid after solve")
    print("frustum solve: all 30 third-person paths verified clean")

    # eval camera from the GT capture's own provenance poses -- ONLY if
    # the staged GT is the fitted map's own (READ captures, never
    # cross-map; the staged calib_b_mirage_p0 is de_mirage's and MUST NOT
    # score a de_inferno render).
    provp = os.path.join(CACHE, "gt_calib_p0/provenance.json")
    if not os.path.exists(provp) or json.load(open(provp)).get(
            "requested", {}).get("map") != fitted_map:   # default-DENY:
        # a provenance without a map field cannot prove it is same-map
        print(f"REFUSED eval camera: no same-map GT staged for {fitted_map} "
              f"(GT_BY_MAP names the source; stage it, then rerun gen+eval). "
              f"The 40 match videos do not depend on it.")
        prov = None
    else:
        prov = json.load(open(provp))
    if prov is not None:
        ev = []
        for p in prov["requested"]["poses"]:
            ev.append({"tick": int(p["tick"]), "x": p["x"], "y": p["y"],
                       "z": p["z_foot"], "eye_z": p["z_foot"] + 64.0,
                       "yaw_degrees": p["yaw"], "pitch_degrees": p["pitch"],
                       "fire": False, "is_alive": True})
        json.dump({"schema": "iji/cs2-demo-ego-camera-path/v1",
                   "variant": "eval", "tick_rate": 128,
                   "map_name": fitted_map,
                   "n_ticks": len(ev), "ticks": ev},
                  open(os.path.join(out, "eval.camera.json"), "w"))
        print(f"eval camera: {len(ev)} GT poses")

    # the profile: full asset surface, FITTED map (a hardcoded map name
    # here is the wrong-map root cause all over again). weld=derived is
    # the one explicit non-default, per the combat_stride3 systems line.
    profile = {"asset_root": "assets", "flags": [
        ["--world", f"{fitted_map}.pt"],
        ["--irr-npy", f"{fitted_map}.irradiance.npy"],
        ["--fam-side", f"{fitted_map}.fam_side.pt"],
        ["--probe-npz", f"{fitted_map}.probe_field.npz"],
        ["--lod-weld", "derived"],
        ["--playermodels", "../inputs/playermodels.json"],
        ["--playermodel-bundles", "pm_bundles"],
    ]}
    json.dump(profile, open(os.path.join(out, "profile.json"), "w"), indent=1)
    sh(["rsync", "-a", out + "/", f"{NODE}:{NODE_ROOT}/inputs/"])
    print("gen: inputs shipped to the node")


# -------------------------------------------------------------- render ----
def _profile_argv():
    """OWNER RULE: no hand-assembled input lists. The renderer resolves
    EVERY input family from the provisioned content tree by fitted map
    name and refuses BY NAME on anything missing; only behavioral flags
    and recording-derived inputs are passed here. Waivers are named,
    stamped into the log, and shrink as families get built."""
    m = fitted_map_name()
    return ["--content-root", "content", "--map", m,
            "--lod-weld", "derived", "--lod-per-frame",
            "--playermodels", "inputs/playermodels.json",
            "--playermodel-weapon-map", "inputs/weapon_map.json"]


def render():
    # SESSION-BATCHED: one renderer process PER VIEW renders all 10 players
    # from ONE pack load (10x fewer loads than per-player jobs; the GPU is
    # fed continuously instead of relaunching). mp4s are encoded on-node
    # from the device-resident .pt tensors. --preset 0 stamps the bracket
    # (with the honest gap inventory acknowledged in the log).
    # optional argv view filter + RENDER_GPU env pin -> the two 5090s run
    # disjoint view subsets concurrently (one swarm per CARD, not per SoC).
    views = [v for v in sys.argv[2:] if v in VIEWS] or list(VIEWS)
    gpu = os.environ.get("RENDER_GPU", "0")
    for v in views:
        agents = [{"id": f"p{p}_{v}", "camera_json": f"inputs/p{p}_{v}.camera.json",
                   "hud_json": f"inputs/p{p}_" +
                               ("achud" if v == "ac" else "hud") + ".json",
                   "tick_begin": 0, "tick_end": 10 ** 6} for p in range(10)]
        sj = f"inputs/session_{v}.json"
        import tempfile, json as _j
        blob = _j.dumps({"agents": agents})
        sh(["ssh", NODE, f"cat > {NODE_ROOT}/{sj} <<'EOF'\n{blob}\nEOF"])
        script = (
            f"cd {NODE_ROOT} && "
            f"(CUDA_VISIBLE_DEVICES={gpu} "
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"{NODE_PY} code/src/counter_strike_render/gpu_render.py "
            + " ".join(_profile_argv()) +
            f" --batch {BATCH} --preset {os.environ.get('RENDER_PRESET', '3')}"
            f" --preset-accept-gaps"
            f" --session {sj} --session-out out/{v}"
            f" --camera-json inputs/p0_{v}.camera.json"
            f" --tick-begin 0 --tick-end 1 --out out/{v}/u.mp4"
            f" --fps 32 --width {W} --height {H} > out/{v}.log 2>&1; "
            f"rc=$?; echo RC=$rc view={v}; tail -2 out/{v}.log | head -1; "
            f"exit $rc)")   # PROPAGATE: a refused view must fail the phase
        sh(["ssh", NODE, script])
        # mp4s are STREAMED by the session sink itself (owner rule: the
        # currency is demovideos -- no dense tensor intermediates, so
        # there is nothing to encode after the fact).
    return
    for v, p in []:
        cam = f"inputs/p{p}_{v}.camera.json"
        out = f"out/p{p}_{v}.mp4"
        log = f"out/p{p}_{v}.log"
        script = (
            f"cd {NODE_ROOT} && test -s {out} && echo SKIP {out} || "
            f"(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"{NODE_PY} code/src/counter_strike_render/gpu_render.py "
            + " ".join(_profile_argv()) +
            f" --batch {BATCH}"
            f" --camera-json {cam} --tick-begin 0 --tick-end 999999"
            f" --fps 32 --width {W} --height {H}"
            f" --out {out} > {log} 2>&1; "
            f"echo RC=$? {out}; tail -2 {log} | head -1)")
        sh(["ssh", NODE, script])
    r = subprocess.run(["ssh", NODE, f"ls -la {NODE_ROOT}/out/*.mp4 | wc -l; "
                                     f"du -sh {NODE_ROOT}/out"],
                       capture_output=True, text=True)
    print(r.stdout)


# ---------------------------------------------------------------- eval ----
def eval_():
    script = (
        f"cd {NODE_ROOT} && rm -rf out/evalpng && mkdir -p out/evalpng && "
        f"{NODE_PY} code/src/counter_strike_render/gpu_render.py "
        + " ".join(_profile_argv()) +
        f" --camera-json inputs/eval.camera.json --tick-begin 0"
        f" --tick-end 999999 --fps 128 --width {W} --height {H}"
        f" --png-dir out/evalpng --out out/eval.mp4 > out/eval.log 2>&1; "
        f"echo RC=$?; "
        f"{NODE_PY} /home/bigboi/cs-render/metric_compare.py "
        f"--gt assets/gt_calib_p0 --render out/evalpng --resize "
        f"| tee out/eval_metrics.txt")
    sh(["ssh", NODE, script])


# ---------------------------------------------------------------- pull ----
def pull():
    os.makedirs(FIGS, exist_ok=True)
    sh(["rsync", "-a", f"{NODE}:{NODE_ROOT}/out/*.mp4", FIGS + "/"])
    sh(["rsync", "-a", f"{NODE}:{NODE_ROOT}/out/eval_metrics.txt", FIGS + "/"])
    sh(["bash", "-c", f"ls -la {FIGS}/ | head -40"])


PHASES = {"stage": stage, "gen": gen, "render": render, "eval": eval_,
          "pull": pull}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (PHASES if which == "all" else {which: PHASES[which]}):
        print(f"===== PHASE {name} =====", flush=True)
        PHASES[name]()
