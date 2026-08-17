# Authoritative replay state: capture and access contract

Status: design review and implementation gate. This document defines what a
TARD recording must preserve, what a renderer must have access to, and what
must be tested before a render may be described as reconstructing the state of
the recorded game.

## Decision

A recorded replay is not a bag of independent rows. It is an ordered state
machine history:

```text
S[t] = apply(S[t-1], delta[t], action[t], event[t], build, content, cvars)
I[t] = render(S[t], presentation[t], render_history[t], content)
```

`S[t]` includes every recurrent game subsystem that can affect the meaning or
appearance of an action: entity lifetimes, weapon state, objectives, dynamic
physics, animation graphs or evaluated poses, effects, and the camera target.
A delta without its predecessor is not a state. An event is not a pose. An
action label is not the pre-action world in which the action occurred.

The claim in `src/gpu_render_libs/scan_camera.py` that "Rendering a RECORDED
replay has no recurrence at all: every tick's state is data" is only true if
the recording actually contains a complete state snapshot at every tick. TARD
2.1 does not. Its rows contain a mixture of sparse events, decimated samples,
reconstructed accumulators, and defaults. The renderer therefore has to
reconstruct recurrent state, or clearly declare an approximation.

This contract requires:

1. an immutable authoritative source recording;
2. a complete initial checkpoint and ordered, lossless deltas;
3. periodic checkpoints for seek, recovery, and verification;
4. the exact build and content needed to interpret state identifiers; and
5. evaluated-state fallbacks where exact engine reevaluation is unavailable.

The raw `.dem` is the source of record for everything it contains. A derived
TARD file is an index and transport format, not a replacement for the source.

## What “reconstruct” is allowed to mean

The project must label each output with one of these levels:

| Level | Required result | Required sources |
|---|---|---|
| **Authoritative semantic state** | The same players, entities, inventory, objectives, actions, outcomes, and server-observable transforms at a tick. | Complete server/demo state, lifecycle journal, rules, timebase, and checkpoints. |
| **Authoritative world pose** | Semantic state plus the same dynamic rigid-body transforms, breakage, attachments, player/viewmodel poses, and effect state that the reference showed. | Exact engine reevaluation from sufficient state, or evaluated rigid-body/bone/effect checkpoints. |
| **Authoritative client presentation** | The same POV camera, HUD, viewmodel, interpolation/prediction result, exposure, temporal effects, and pixels. | World pose plus the original client-side capture/configuration and renderer histories. |
| **Canonical reconstruction** | A deterministic, coherent visualization of authoritative semantic state, with documented replacement rules for missing presentation state. | Semantic state and pinned canonical renderer/content. |
| **Approximation** | A useful visualization containing inferred or fabricated state. | Whatever inputs are available, with every unsupported subsystem identified. |

SourceTV is principally a record of networked/server-observable state. Its
existence does not prove that client-only graph internals, evaluated bones,
physics solver intermediates, HUD timers, prediction, camera impulse, exposure,
or temporal anti-aliasing history were recorded. Those are separate capture
requirements. Conversely, failure of one parser or serializer to expose a
field does not prove that the source demo omitted it.

Ordinary-demo coverage is not the ceiling of this project. For a capture whose
purpose is authoritative state replay, an omitted client/server state family is
a requirement on the custom total-capture recorder, not a reason to accept an
impossible replay. The native demo remains the base authoritative stream, while
the recorder must add synchronized engine/server and client-presentation
sidecars at the state-transition and render-evaluation boundaries.

The passive HID recorder is only an ego-intent sensor. It is not the total
recorder. The total recorder is complete only when it additionally captures:

- decoded user commands and the exact simulation application order for the
  human and bots, together with authoritative outcomes;
- complete initial and periodic checkpoints plus ordered lifecycle/delta state
  for every server, physics, animation, effect, and presentation state machine;
- final evaluated bones, attachments, viewmodel pose, and ragdoll bodies at
  every required render sample whenever complete deterministic reevaluation is
  not proven;
- rigid-body transforms/velocities, constraints, wake state, breakage, and
  solver discontinuities at the same state horizon;
- camera/view target/FOV/interpolation/prediction, visibility, UI state, and
  renderer histories for every scored client vantage—not merely the recording
  client's `democmdinfo_t`; and
- exact clocks, sequence order, dropped-sample markers, build/content identity,
  and hashes joining every sidecar to the native demo and canonical journal.

If all bot/player POVs are a required output, the recorder must either capture
each actual client view or execute and record an explicitly defined per-view
evaluation inside the pinned engine. Entity transforms observed by one client
do not constitute every client's camera, prediction, visibility, viewmodel, or
diegetic UI state.

## Source material and infrastructure we must retain

### 1. Original recording and provenance

For every match, retain:

- the original `.dem` or `.dem.zst` bytes, never regenerated from TARD;
- original filename, byte length, SHA-256 of compressed and uncompressed data;
- provider match ID and source URL/object key, not an ordinal pack number;
- acquisition time, uploader/capture job, and retention location;
- demo protocol, server tick interval, map name, server name, and match time;
- an explicit completeness result: clean end, truncated, corrupt, or gapped.

The manifest must be sufficient to fetch the same bytes on another machine.
An undocumented path on one workstation or cluster is not preservation.

### 2. Exact executable and data build

Retain or identify immutably:

- CS2 build/patch and demo protocol;
- Steam app, depot, and manifest IDs for the client and dedicated server;
- executable/module hashes where lawful and operationally possible;
- schema/sendtable/serializer definitions and string tables;
- server and match convars that affect simulation or presentation;
- parser source revision, command line, requested field list, and output schema.

“Current CS2” is not a build identifier. A later engine may parse or simulate
an older demo differently, and integer animation/model identifiers only have
meaning in the content version that assigned them.

### 3. Complete content, not just renderable triangles

The renderer or an exact-engine replay worker needs the matching:

- world geometry, entity lump, visibility data, collision and `.vphys` data;
- materials, textures, decals, sky, lights, and color-correction resources;
- player, weapon, projectile, prop, hostage, and objective models;
- skeletons, animation graphs, clips, sequences, pose metadata, attachments,
  material groups, body groups, mesh groups, skins, and hitbox sets;
- particle systems, sounds, overview/radar, navigation, and localization data;
- physics materials, collision shapes, constraints, and breakable definitions.

Every consumed artifact needs a content hash or depot-manifest identity. A
renderer must refuse authoritative mode when a required content family is
missing or comes from a different build.

### 4. Client sidecar for client-presentation claims

Exact ego imagery additionally requires the recorded client's:

- resolution, aspect ratio, render scale, video preset, and graphics convars;
- HUD, crosshair, radar, color, accessibility, and safe-area configuration;
- viewmodel FOV/offsets and left/right handedness;
- spectator/POV target, camera mode, FOV, punch and shake state;
- interpolation, prediction, input sampling, and latency configuration;
- exposure, bloom, motion blur, depth of field, TAA jitter/history, and other
  temporal renderer state;
- local sound mix if audio equivalence is claimed.

If no client sidecar exists, SourceTV can still support semantic or canonical
reconstruction. It cannot by itself support an exact original-client claim.

## What the serializer must log

The protocol must be a generic ordered entity/state journal. A manually
selected player table is not an adequate substrate.

“Must log” below means “must be present in the reconstruction package,” not
“must already exist in SourceTV.” Every required value must have an explicit
capture route:

1. preserve it from the raw `.dem` when it was transmitted;
2. extend the parser when the demo contains it but the current API does not
   expose it;
3. capture an exact-engine or client sidecar when it was never transmitted; or
4. mark it unavailable and refuse the affected fidelity level.

Manufacturing a value from an action label is not a fifth route.

### 1. Time and ordering

Log for every record:

- server tick and tick interval;
- subtick fraction or exact command/event timestamp;
- packet/demo sequence and record order within the packet;
- simulation time, and render/presentation time when captured separately;
- pause, time-scale, round reset, seek, gap, dropped-packet, and discontinuity
  markers;
- the state horizon and event horizon, which must agree or explicitly differ.

Events sharing a tick must retain wire/application order. Sorting only by tick
can change create/update/destroy and weapon/action semantics.

### 2. Entity identity and lifecycle

For every networked or captured entity class, not just players, log:

- entity index plus serial/generation, stable capture ID, class and archetype;
- create, baseline, update, dormancy/transmit change, wake, and destroy;
- complete creation baseline before any delta is applied;
- owner, parent, hierarchy, attachment, and carried/ground relationships;
- model, skin/material/mesh/body group, hitbox set, collision group, and flags;
- origin, rotation, scale, linear/angular velocity, and teleport/reset marker;
- raw qualified property path, raw value, type, quantization, and presence bit.

Entity indices are reused. Index alone is not identity. “Not observed” must be
different from zero, false, empty, or a hard-coded default.

The capture API must enumerate generic classes. An export path that only
exposes player, weapon, grenade, team, and game-rules tables cannot carry
doors, breakables, movable props, hostages, scene controllers, dynamic lights,
effects, and new classes introduced by a patch, even if a lower parser layer
maintains some of those entities internally.

### 3. Player, controller, and pawn state

At minimum preserve:

- controller/pawn association, identity, team, life state, spawn, and observer
  target/mode;
- position, eye position, body/eye angles, velocity, base velocity, ground
  entity, movement mode, water/ladder state, collision flags, duck and fall;
- health, armor, helmet, flash, immunity, scoped state, FOV, recoil, aim punch,
  view punch, and relevant timers;
- inventory entities, active weapon entity, ammo/reserve, money, purchases,
  equipment, and item attributes/skins;
- spotted/radar state and any visibility facts used by a diegetic UI.

Controller, pawn, weapon, and viewmodel are separate lifetimes and must be
joined by captured handles, not by nearest timestamp or player slot.

### 4. Intent, actions, and resolved outcomes

Preserve both sides of an action:

- original user command number, tick/subtick, view angles, mouse deltas,
  movement axes, buttons, weapon selection, impulse, and prediction metadata
  when present;
- the authoritative server-applied result: fire, reload, use, jump, pickup,
  damage, death, blind, plant, defuse, throw, detonate, and other game events;
- actor, target, assister, weapon/projectile entity, position, hit group,
  damage, penetration, and event-specific fields;
- rejected, suppressed, or rate-limited intents where the source exposes them.

An action is contextualized by snapshotting or hashing both `S[t-before]` and
`S[t-after]`. A `fire=1` column is not enough: the active weapon state,
magazine, recoil state, shooter pose, muzzle attachment, target transforms,
occluders, smoke/effects, and resolved bullet event are part of its meaning.

### 5. Weapon and projectile state machines

Log:

- weapon lifecycle and owner, definition/item identity, active state, clip and
  reserve ammo;
- attack/reload/zoom/silencer/burst state, next-action timers, accuracy penalty,
  recoil index, shot counter, and sequence/parity fields;
- bullet origin, direction/angles, seed, inaccuracy, spread, penetration and
  impacts when transmitted;
- projectile create/throw/release, transform and velocity, bounce/collision,
  owner, fuse, detonation, and destroy;
- smoke, inferno, flash, HE, decoy, molotov, and dropped-weapon lifetimes.

Do not infer a reload solely from an ammo increase or choose an animation
solely from an event label in authoritative mode. Those are canonical
fallbacks, not recovered state.

### 6. Rules, objectives, economy, and diegetic UI inputs

Log:

- game phase, warmup/freeze/live/post-round state and all phase timers;
- round number, score, win reason, time remaining, overtime and side switches;
- team/player economy, loss bonus, purchases, inventory values, and awards;
- bomb carrier, dropped position, site, plant/defuse state, timers and outcome;
- hostage and other mode-specific objective entity state;
- kills, assists, damage, blind, MVP and other feed/scoreboard inputs;
- chat, radio, voice activity, names, clan tags and avatars if the target UI
  renders them and policy permits retaining them;
- server cvars and string/localization tokens that change displayed values.

The renderer should derive HUD widgets from this reconstructed state. It
should not render UI from a disconnected event list with independent defaults.

### 7. Dynamic physics

For every rigid or articulated dynamic object preserve:

- entity/body identity and transform;
- linear and angular velocity, sleep/wake, kinematic/dynamic state;
- mass, inertia, center of mass, physics material, collision layer/group;
- collision-shape/content identity and body-to-entity transform;
- constraints, joints, parents, attachments, breakable health and piece IDs;
- applied impulses/forces, teleports, activation, breakage, spawn and destroy;
- random seeds and deterministic solver configuration where available.

Sparse entity positions plus a modern physics engine are not sufficient for
exact replay. Contact ordering, solver implementation, floating point, and
build changes can diverge. Authoritative world-pose mode therefore requires
one of:

1. the exact engine/build and enough initial state and ordered inputs to prove
   deterministic reevaluation; or
2. authoritative body transforms and velocities at every required sample,
   with periodic full rigid-body checkpoints and explicit discontinuities.

If the GT capture contains rigid-body transforms, breakage, doors, debris, or
ragdoll body state, those values must be preserved. They must not be replaced
by a new simulation driven only by an action label.

### 8. Animation, pose, and ragdoll state

For each animated entity preserve:

- model/content identity, skeleton, skin, body/mesh/material group and hitbox
  set;
- animation graph asset/version and graph initialization/reset epoch;
- active graph nodes, states, slots/layers, transitions and blend weights;
- sequence/clip handles, cycle, playback rate, start time, parity/reset parity,
  graph parameters, predicted variables, pose parameters and random seeds;
- locomotion state, root motion, gestures/overlays, weapon/viewmodel sequences,
  attachment transforms, and look/aim targets;
- animation-to-ragdoll transition and ragdoll body transforms/velocities.

`m_bAnimGraphUpdateEnabled`, `m_bRagdollEnabled`, an action label, or a
sequence ID alone is not a pose. The current event-to-clip scripter chooses
clips, phases, and thresholds procedurally; it is explicitly an approximation.

Authoritative world-pose mode requires one of:

1. exact graph assets/build plus a complete graph checkpoint, ordered graph
   inputs, clocks, transition state, and random seeds sufficient to reproduce
   the evaluated pose; or
2. evaluated local-to-parent or model/world bone transforms for every required
   render sample, with skeleton identity and periodic pose checkpoints.

Bone matrices are the lossless visual fallback when graph state cannot be
proven complete. Ragdolls additionally require body transforms/velocities or
deterministic reevaluation. A seek must resume the same clip phase, blend,
attachments, and ragdoll, not restart idle or bind pose.

The absence or non-emission of `DEM_AnimationData` in an inspected demo does
not establish that all animation-relevant network properties are absent.
Qualified animgraph, sequence, parity, timing, predicted-variable, viewmodel,
and ragdoll properties must be probed in the source schema and parser before
declaring them missing. Even if all are decoded, they still must be shown
sufficient to evaluate the same bones; otherwise use evaluated-pose capture or
label the result canonical.

### 9. Effects and presentation entities

Log entity or event state for:

- particle system create/destroy, control points, owner/attachment, seed,
  elapsed time and lifetime;
- smoke/inferno field state or periodic authoritative checkpoints;
- impacts, decals, tracers, muzzle flashes, shell ejection and blood;
- dynamic lights, fog/tonemap controllers, post-process volumes and shakes;
- sound start/stop/parameters if audiovisual reconstruction is in scope.

An effect with a lifetime is recurrent state. Recording only its spawn tick is
usable only if the exact effect definition, clock, seed, controls, and update
rules are also available.

### 10. Camera, viewmodel, HUD, and renderer history

For each scored POV log or derive unambiguously:

- camera entity/target, transform, eye offset, FOV, mode and cut/teleport;
- viewmodel entity, model, weapon association, sequence/graph state and FOV;
- punch, shake, scope, flash and observer transitions;
- HUD state-machine timers and all source values displayed by the HUD;
- viewport/aspect/render scale and local configuration;
- temporal presentation state such as exposure and TAA history if exact pixels
  are claimed.

For canonical rendering, checkpointing may deliberately reset temporal render
history, but that reset must be part of the declared renderer specification.

## Checkpoint and delta protocol

The next TARD protocol should have four layers.

### Capture manifest

One immutable manifest identifies the source bytes, build/content, parser,
field/schema coverage, tick domains, output chunks, and hashes.

### Initial checkpoint

At the first renderable tick, serialize complete state for every recurrent
subsystem after baselines and string tables have been applied. This includes
all live entities, rules, weapons, physics bodies, animations/poses, effects,
and camera/UI state. No stream may begin with an implicit default.

### Ordered delta transactions

Each tick/subtick transaction preserves original order and contains creates,
property updates, actions/events, and destroys. Each chunk declares:

- base checkpoint ID and sequence range;
- previous canonical-state hash and resulting canonical-state hash;
- source packet range, field-coverage manifest and gap flags;
- payload checksum and schema version.

Raw decoded values should be retained before lossy quantization. Derived
columns may be shipped as caches, but must name their inputs and algorithm and
must never overwrite the source-valued column.

### Periodic checkpoints

Emit full checkpoints at a bounded interval and at round boundaries, respawns,
map/phase resets, major teleports, and parser gaps. They make arbitrary seek
possible, bound corruption, and allow independently reconstructed intervals to
converge on the same state hash.

Server-authoritative state, client-presentation state, and renderer-derived
caches must be separate namespaces. Mixing them makes it impossible to tell a
recorded value from an inference.

## Parser and serializer behavior

The pipeline must fail closed:

- enumerate the source schema and store its hash;
- request qualified property paths and report each requested path as populated,
  present-but-default, absent, unsupported, or parse-failed;
- reject “success” when a field request returns only identity columns;
- enumerate and serialize unknown entity classes and properties generically;
- preserve nullable/missing separately from zero/default;
- never use bare `except` or silently drop a field, entity, tick, or event;
- count creates, updates, destroys, values, nulls, and distinct values by field;
- require event and state horizons to be reconciled;
- retain raw source recordings after every derived serializer revision.

“All requested fields” is not the same as “all available fields,” and neither
is the same as “all state needed for rendering.” A coverage manifest must make
those three sets auditable.

## Acceptance gates

No implementation may claim authoritative reconstruction until it passes:

1. **Linear/seek equivalence:** checkpoint + deltas at arbitrary seeks produces
   the same canonical state hash as linear replay.
2. **Checkpoint convergence:** independently replayed adjacent checkpoint
   intervals agree at their overlap.
3. **Lifecycle integrity:** no update targets a nonexistent generation; every
   live entity has a baseline and every reuse has a new generation.
4. **Pre/post action context:** every action can return complete before/after
   state and link intent, actor, weapon/projectile, target, and resolved result.
5. **Physics equivalence:** sampled body transforms, sleep state, constraints,
   breakage and ragdolls match GT within declared tolerances.
6. **Animation equivalence:** evaluated bones, clip phase/blends, attachments,
   weapon/viewmodel pose and ragdoll transition match GT; seeks cause no bind
   pose or idle reset.
7. **World identity:** map, build and every accessed content family match their
   recorded hashes. A header label alone is insufficient.
8. **No invented continuity:** teleports, respawns and resets are
   discontinuities; interpolation never creates travel through world geometry.
9. **Horizon integrity:** state, action, event, physics and presentation ranges
   are complete or explicitly gapped; no tail is silently discarded.
10. **Coverage integrity:** required missing/default-only fields and entity
    families fail the requested reconstruction level.
11. **Visual validation:** golden GT comparisons cover players, weapons,
    projectiles, dynamic props, doors/breakage, smoke/inferno, HUD and seeks.
    Images supplement state tests; they do not replace them.

## Current TARD assessment

### Evidence anchors

This assessment is based on the published compact corpus, the current renderer,
one recovered raw FACEIT demo and its total-capture derivative, and these
upstream implementation points:

- The upstream
  [TARD doctrine at `6b6fef3`](https://github.com/Dash-Crystal/tardigrade/blob/6b6fef30fc9108db1ef13f6d776680d185b3fe72/docs/TARDIGRADE_DOCTRINE.md#L11-L55)
  describes `.dem -> Parquet`, followed by selecting 20 of 222 fields and
  discarding 91% of the decoded state. Its own account of entity baselines and
  deltas therefore establishes that many compact-format omissions occurred
  after the authoritative source was decoded.
- [`tardigrade_total.py` at `6b6fef3`](https://github.com/Dash-Crystal/tardigrade/blob/6b6fef30fc9108db1ef13f6d776680d185b3fe72/tardigrade_total.py)
  builds a manual candidate-field list and probes it through parser calls. It
  is broader than TARD 2.1, but is not a schema-complete generic journal. Its
  probe accepts a field whenever `parse_ticks([field])` does not raise, without
  verifying that the requested column was returned. The parser can instead
  return only `tick`, `steamid`, and `name`; this false success was reproduced
  for `m_SerializePoseRecipeAG2Dynamic` and `m_flViewmodelFOV`.
- [`total_pipeline.py` at `6b6fef3`](https://github.com/Dash-Crystal/tardigrade/blob/6b6fef30fc9108db1ef13f6d776680d185b3fe72/total_pipeline.py)
  watches a local `demo_downloads/*.dem.zst` directory. It contains no source
  host, bucket, provider URL, or corpus acquisition manifest from which the
  original demos can be recovered.
- [demoparser's current updated-field fixture](https://github.com/LaihoE/demoparser/blob/266a831f08b0264dd722b017a5c05d765206a7ed/src/parser/src/e2e_test.rs#L208-L222)
  and a direct `list_updated_fields()` probe demonstrate a substantially
  larger qualified property surface. The probe found player animgraph
  predicted-variable arrays, random-seed offset, sequence/reset parities, a
  player viewmodel handle, ragdoll flags and force/origin/damage metadata, and
  weapon/grenade sequence properties. It did not demonstrate complete
  viewmodel state, per-body ragdoll transforms, or a player-pawn sequence. This
  is evidence that parser coverage must be re-probed, not evidence that these
  fields close animation state.
- [demoparser's generic-entity selection](https://github.com/LaihoE/demoparser/blob/266a831f08b0264dd722b017a5c05d765206a7ed/src/parser/src/second_pass/entities.rs#L396-L407)
  restricts this updated-field/listening API surface to players/controllers,
  rules, teams, weapons, and grenades. It does not prove that no lower parser
  layer maintains generic entities, but it does prove that the ordinary export
  path cannot enumerate them. Dynamic world classes therefore require explicit
  coverage work below the player-tick table API.
- The current renderer's `anim_script.py` and third-person pass explicitly
  select clips, phases, and thresholds from events and movement. That path is
  useful canonical behavior, but it does not read an evaluated recorded pose.

### Source access found during this review

The following files are available outside the repository and must be entered
in a real source manifest rather than left as path folklore:

| Source | Identity | What it is good for |
|---|---|---|
| `test.dem.zst` | compressed SHA-256 `b4016095ac3ec95e38d4b5409abed7081e60fe04b2d4f8570c9a4bd84e199cc2`; decompressed SHA-256 `96bc5d333a0884de9ac6d8ca7943db412fa2e0e22285a6b07154976b3f3ccecc`; FACEIT SourceTV, `de_mirage` | Re-running and extending the serializer against one real corpus-style source demo. Its basename explains the total sample's `match_id = "test"`; that ID is not provider provenance. |
| `/data/cs2-artifacts/demos/combatgt.dem` | 1,021,840 bytes; SHA-256 `5a742cbd27915edb5012848dd18ab116c78bb1670ec052b0de55126bd9fa0108`; loopback/local CS2 capture | Ground-truth fixture for combat renderer and capture instrumentation, not evidence of the FACEIT corpus. |
| `/data/cs2-artifacts/demos/smokegt.dem` | 743,479 bytes; SHA-256 `d769b7ae4bad51c04812863a02248c6f66448fe5d94db10ee570cfb174579947`; loopback/local CS2 capture | Ground-truth fixture for smoke/effect state and capture instrumentation, not evidence of the FACEIT corpus. |
| `/data/cs2-artifacts/demos/demoparser2-test/test_demo.dem` | 60,601,900 bytes; SHA-256 `84a1a4191302bdd2a3bbb5a727842093744b1fb1a228aeec630369e44b622cb2`; Valve/SourceTV `de_mirage` parser fixture | Parser regression and schema probing, not evidence of the FACEIT corpus. |

The checked total derivative reports `total_v1`, patch 14173, 1,182,470
player-tick rows, 10 players, 190 table columns including identity columns, 48
event families, grenade records, voice packets, player info, and skins. It is
materially richer than the compact pack and is the quickest path to identifying
serializer-only omissions. Its grenade export is not yet a usable trajectory
guarantee: approximately 805,354 of 984,980 rows (81.8%) have `x/y/z = NaN`,
while about 179,633 have numeric `x`. Some null rows may represent held or
non-flight grenade entities, but the export does not classify or validate that
distinction and emits bare `NaN`, which is not strict JSON.

A targeted search of the accessible Nebius and Terul storage found no bulk
source set for the 155 audited published compact packs. The README separately
claims 200 sessions; that is not a verified raw-demo inventory. Therefore the
published compact corpus is not presently reparsable in bulk from known source
locations. This is an availability finding, not proof that no other copy
exists.

| Requirement | TARD 2.1 compact packs | Audited total capture/sample | Consequence |
|---|---|---|---|
| Raw `.dem` and fetchable provenance | Not in protocol; `match_id` is an ordinal in published compact packs | One FACEIT `test.dem.zst` has been recovered for reparse; the bulk source corpus has not | One sample can drive extractor work; bulk recovery remains blocked without a source manifest or demos. |
| Build/content identity | Map header is not trustworthy; no immutable depot manifest | Patch/server/map metadata is richer, but does not itself preserve the depot | Renderer cannot safely select exact world/assets from pack metadata alone. |
| Initial/full checkpoints | Absent; state streams can begin thousands of ticks late | Not a complete recurrent-subsystem checkpoint | Arbitrary seek and authoritative initialization are impossible. |
| Ordered generic entity lifecycle | Absent | The exporter is a field allowlist over parser APIs, not a generic entity journal | Doors, props, breakables, hostages, effects and new classes can disappear silently. |
| Player/control state | Partial; the audited `030f7ca` packs decimate positions to 16 Hz, while `ea7bcf2` re-encodes full-rate positions without changing the format version; several other fields are lossy or defective | Much richer player property set | Useful substrate, not sufficient state closure; the wire version alone does not identify position cadence. |
| Actions and events | Sparse/defective (`fire` disagrees with shots) | Rich events including bullet records; grenade rows exist but 81.8% have null positions and flight-state usability is unvalidated | Total data can recover much more action context but still needs lifecycle/checkpoint semantics and projectile validation. |
| Weapon/projectile machines | Partial weapon/ammo state; weapon labels were discarded by the reader | Richer weapon/event/grenade fields | Requires serializer and reconstructor work, not new rendering guesses. |
| Dynamic physics entities | Absent | No demonstrated complete generic rigid-body journal | Authoritative physics reconstruction is not supported. |
| Animation graph state/evaluated pose | Absent | Only weak animation/ragdoll flags were found in the checked sample; no demonstrated complete graph checkpoint or evaluated bones | Current event-to-clip animation is canonical approximation, not recovered pose. |
| Camera/client presentation | Control/view basics only | No complete original-client sidecar | Exact ego pixels are unsupported. |
| Integrity/coverage hashes | Undocumented trailer; no state hash or field coverage result | No proof that “requested” equals “available” or “render-complete” | Silent omission cannot be distinguished from source absence. |

The measured compact-format failures are detailed in
[`DATA_QUALITY_REPORT.md`](../../../DATA_QUALITY_REPORT.md): fabricated
interpolation across teleports (D1), discarded weapon/location values (D2),
state/event horizon mismatch (D6), missing initial state (D8), false map
identity (D10), default-only aim punch (D11), and incoherent fire RLE (D12).

The audited “total” encoder materially improves field and event coverage, but
its manually assembled field list, exception-based probing, and dependence on
parser-exposed entity families do not prove total state capture. The original
demo may contain properties that this serialization omitted. It may also lack
client-only or evaluated state that was never transmitted. Only source-schema
coverage plus exact-engine or GT-side instrumentation can distinguish those
cases.

## Implementation order

1. **Preserve and index sources.** Build a content-addressed manifest for every
   raw demo already available; record external locations and hashes; do not
   discard source bytes after serialization.
2. **Build a coverage probe.** Against the recovered FACEIT sample and GT demo
   fixtures, enumerate qualified fields and entity classes, detect false-success
   empty requests, and compare current versus latest parser coverage.
3. **Define TARD state-v3.** Implement generic lifecycle records, typed values,
   a complete initial state, ordered transactions, checkpoints, state hashes,
   provenance, and coverage manifests.
4. **Add recurrent subsystem capture.** Extend below the player-table API for
   physics entities, animation/viewmodel fields, effects, objectives and UI
   inputs. Where the demo is insufficient, add exact-engine capture sidecars
   for bones, rigid bodies, particles, camera and client presentation.
5. **Implement the state reconstructor first.** It owns entity generations,
   baselines, checkpoints, deltas and state hashes. Renderers consume `S[t]`;
   they do not independently interpolate tables or infer unrelated state.
6. **Gate every fidelity label.** Missing required sources cause an explicit
   refusal of authoritative mode. Canonical fallbacks remain available and are
   reported per subsystem.

This is the minimum boundary for contextualizing any action against the game
that produced it. Better shading cannot repair a missing predecessor state,
and procedural animation cannot recover a pose that was neither serialized nor
recomputed from a complete authoritative graph state.
