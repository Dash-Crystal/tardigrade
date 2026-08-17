# macOS `csgo_legacy` native capture boundary

This directory contains an offline-only, fail-closed capture probe for the
installed Intel macOS CS:GO build `12426195`. It combines Valve's
`IServerPluginCallbacks004` interface with exact-build, client-interface vtable
instrumentation for `VClient018::CreateMove` and `RenderView`. It does not
bypass or evade VAC and rejects use outside a consented local `-insecure`
bot/listen-server run.

The result is a **native boundary diagnostic plus the engine's `.dem`**, not a
complete total capture. The diagnostic schema is intentionally different from
`tardigrade/total-capture-record/v1`. `contract_gate.py` refuses to promote the
current output into the integrator's canonical total-capture protocol. This is
the required behavior while bot/server command application, final poses,
physics, effects, all-POV camera/UI, visibility, and complete render histories
remain unavailable.

## Exact target and references

The persistent monitor and module both validate the same allowlist in
`expected_build_12426195.json`. The installed Steam manifest says app `730`,
branch `csgo_legacy`, build/target build `12426195`, fully installed. The target
is x86_64 and runs under Rosetta on the current arm64 host.

Public layout references are pinned to:

- AlliedModders `hl2sdk` CS:GO commit
  `9cf2f325ea273559c7cae27b4f98518c18b8e322`;
- Valve Source SDK 2013 commit
  `22288b919617be6c8ca3cefd7cca979cbb39a88c`.

The exact CS:GO header adds `ClientFullyConnect` before `ClientDisconnect` and
the two network-crypt methods at the end of v004. The module implements that
exact vtable order and exports v004 only. The crypt callbacks never emit or
accept key material: if unexpectedly reached, they record a safety violation
and force rejection.

## What is actually captured

After the safety, module SHA, interface-slot, target-address, and instruction
prefix checks pass inside the game process, the module can record:

- plugin load/unload, level init/shutdown, and `GameFrame` boundaries;
- `ServerActivate`, edict allocation/free, and public client lifecycle events;
- command-client selection and the occurrence of `ClientCommand` (not the
  opaque `CCommand` payload in this minimal ABI);
- public engine/server interface availability;
- authoritative ego `CUserCmd` command/tick, target view angles, movement,
  buttons, weapon selection, random seed, and interval-integrated mouse counts;
- ego `CViewSetup` camera/FOV/viewport plus world/view/projection/pixel matrices;
- in-process native-demo record/stop requests owned by the transaction;
- per-faculty availability **and completeness**, writer failures, safety
  violations, and a clean/refused/incomplete terminal.

Demo segments rotate on local-client ready/disconnect and map boundaries, with
unique process/map/segment epochs. Pointer values identify callback arguments only within one diagnostic process;
they are not stable entity IDs or reconstruction state. A native `.dem` is
recorded in parallel and copied only after the process exits and its size and
mtime are stable.

## Exact-build faculties and remaining blockers

Static inventory confirms that the exact modules export `CreateInterface` and
contain the expected version strings (`VClient018`, `VClientEntityList003`,
`VEngineClient014`, `VEngineModel016`, `VEngineRenderView014`,
`ServerGameClients004`, `BotManager001`, `VPhysics031`, `VStudioRender026`,
and others). An interface string is not a passive observation hook.

| Required faculty | Exact blocker on this build | Diagnostic disposition |
|---|---|---|
| Human `CreateMove` / `CUserCmd` | Exact `VClient018` slot 24 and `CInput::GetUserCmd` slot 8 are guarded by full module hashes, target offsets, and code prefixes. No consented live invocation has yet been demonstrated. | implemented, structurally verified; runtime-unproven |
| Bot command generation | `BotManager001` does not expose generated `CUserCmd` data/application order. | hard-refused |
| Server command application | `IServerGameClients004::ProcessUsercmds` is a callable service, not a passive callback; observation requires a detour. | hard-refused |
| Ego rendered camera/ViewSetup/FOV/matrices | Exact `VClient018` slot 28 and `VEngineRenderView014` matrix slot 56 are guarded by target offsets and code prefixes. | implemented, structurally verified; runtime-unproven |
| Every scored POV camera | One local rendered client does not evaluate every possible POV. | hard-refused |
| Final bones/attachments | `IClientRenderable::SetupBones` is a per-object virtual call, not an exported passive callback. | hard-refused |
| VPhysics body state | `VPhysics031` exposes services but no safe callback enumerating the live environments at every boundary. | hard-refused |
| Effects/UI/render histories | Exposed interfaces are mutable services, not complete passive event/state histories. | hard-refused |
| Entity reconstruction | Edict lifecycle callbacks omit stable generation, class, baselines, properties, dormancy/PVS, parents, and model/pose state. | lifecycle-only, incomplete |

The two client hooks are installed only after the exact allowlist passes and
are restored before plugin terminal emission. The vtable clone preserves the
Itanium ABI RTTI prefix. Static validation is not a runtime-success claim: a
real consented local run must still prove invocation, ordering, loss accounting,
native demo closure, and clean teardown.

The future client-side join must retain raw HID interval counts and
focus/discontinuity boundaries alongside authoritative `CUserCmd`
target/viewangles and rendered camera/view matrices, all with monotonic clocks
and source sequence. Downstream may fit one continuous angular-target curve
whose interval integrals match the counts and whose knots align to those engine
anchors. It must not label `count / dt` as instantaneous velocity.

## Commands

All build, probe, fixture, and recording artifacts stay under
`/Users/mdot/dox/runs`, outside the repository.

```sh
python3 tools/csgo_total_capture/inventory.py \
  --json-out /Users/mdot/dox/runs/csgo-total-capture-build/12426195/inventory.json
python3 tools/csgo_total_capture/probe.py
python3 tools/csgo_total_capture/test_csgo_total_capture.py
```

The probe compiles x86_64 code and loads it in an x86_64 non-game host. The
module must refuse activation and create no capture file. This proves the
off-target gate; it does **not** prove the game loaded the plugin.

Create a schema fixture, also explicitly refused as native evidence:

```sh
python3 tools/csgo_total_capture/fixture_producer.py \
  --output /Users/mdot/dox/runs/csgo-total-capture-fixtures/fixture.jsonl
python3 tools/csgo_total_capture/contract_gate.py \
  /Users/mdot/dox/runs/csgo-total-capture-fixtures/fixture.jsonl
```

Inspect an outer-session plan without modifying the game installation:

```sh
python3 tools/csgo_total_capture/monitor.py \
  --consent --offline-only --insecure \
  --target-active-gameplay-seconds 3600 \
  --device 0x046d:0xb041 --dry-run
```

Remove `--dry-run` to arm one explicitly consented outer session. The monitor
does not launch the game or choose a map: ordinary Steam/game launches and map
choices happen independently. It installs a temporary `addons/*.vdf` plugin
descriptor and an external same-UID consent/nonce/lease control, starts one HID
stream, and watches successive exact-build processes. The module remains inert
unless the process is `-insecure`, locally connected, exactly hashed, and the
monitor PID and fresh lease remain valid. A stale activation is quarantined
only when its exact ownership/content and dead monitor PID are proven.

The monitor does not infer liveness from HID or demo byte growth: a stationary
trackball can legitimately emit nothing and the engine can buffer demo writes.
It observes ordinary process lifecycles and native `GameFrame` health. The
one-hour target counts active, simulating gameplay—not downloads, menus,
background time, gaps, or crashes. Coverage is the monotonic-time intersection
of an accepted PID, HID-eligible focus, native simulating frames, and a fully
parsed closed demo segment. Ten-minute native checkpoints prevent one long map
from leaving the entire target provisional.
The close handshake stops the current segment without killing the game. Even a
closed bundle remains `closed-incomplete-total-capture` until the promotion
gate's remaining faculties exist.

The module additionally stays inert unless all of these are true inside the
target process:

- exact consent/build/root environment values or a same-UID live monitor
  control carrying a unique nonce and fresh lease;
- exact canonical `csgo_osx64` executable path;
- `-insecure` (the deprecated environment-driven compatibility launcher also
  requires its offline-only argv marker);
- no connect/playcast/lobby-connect argument;
- canonical output parent below `/Users/mdot/dox/runs` and outside this repo;
- output created exclusively, without following a final symlink;
- exact SHA-256 for the executable, engine, client, server, vphysics, and
  studiorender modules.

## Evidence scope

`launch.py` is a refused-by-default one-process compatibility tool, not a
recording entrypoint. `scripts/cs2_capture_session.py` likewise refuses
`csgo_legacy`.

Until a consented monitored process is observed, the evidence consists of real installed
binary/module inventory, real compiler output, and the real off-target refusal
probe. Fixture output is synthetic and labelled `fixture-only`. There is no
claim yet that the game loaded this plugin or invoked any engine callback.
