# Source 1 implementation audit

Status date: 2026-08-15. This is an adversarial implementation audit, not a
completion announcement. The protocol reference used for comparison is Valve's
BSD-licensed `csgo-demoinfo` source in
`/Users/mdot/dox/runs/valve-csgo-demoinfo-reference`.

## What is implemented

The Source 1 lane is no longer merely a variant-name switch. It contains:

- bounded `HL2DEMO` header and command framing;
- packet protobuf framing and selected message envelope decoding;
- Valve-order send-table flattening, including excludes, collapsible tables,
  non-collapsible recursion, priorities, and `SPROP_CHANGES_OFTEN`;
- bounded ordinary string-table updates and command-level
  `dem_stringtables` snapshots, including transactional instance baselines;
- bit-packed sendprop decoding for the supported Source 1 encodings;
- PacketEntities reconstruction from full states and recorded `delta_from`
  histories, with old-to-target reconciliation rather than blind mutation of
  the latest packet;
- create/update/leave/delete reduction with monotonically advancing canonical
  generations;
- recorded `democmdinfo_t` camera origins and angles;
- game-event descriptors and values, including signed protobuf `int32`;
- shared SQLite integration, checkpoints, state hashes, seeking, batching, and
  action-context joins;
- hashed canonical string-table resource entities, exact five-field entry
  documents, and active-entity model rebinding when `modelprecache` changes;
- lossless derived model bindings whose path, table revision, entry bytes, and
  entry hash are cross-checked against the same already-hash-verified frame;
- strict Source 1 BSP world geometry, VPK CRC checks, MDL 48/49, VVD v4, and
  DX90 VTX v7 LOD0 readers;
- deterministic loose-file/VPK search paths for BSP and MDL companion assets;
- deterministic CPU triangles, PVS filtering, final-bone-matrix skinning, and
  independently hashable PPM and uint64 depth artifacts; and
- explicit variant dispatch which does not substitute Source 2 assets.

The adversarial regression suite is `tests/test_source1_qa.py`. It specifically
covers Valve flatten order, signed event values, event type/value agreement,
baseline and string-table failure atomicity, full-snapshot replacement,
duplicate send-table and string-table identity refusal, reducer rollback,
forged state-hash refusal, and leave-PVS pixel exclusion. At the status date,
`python3 -m unittest discover -s tests -v` passes all 114 tests, including
eleven adversarial QA tests.

## What the evidence proves

The current tests prove behavior on constructed, bit-exact fixtures. They prove
that those fixtures are bounded, deterministic, transactionally handled, and
joined to emitted reference pixels. The fixture MDL/VVD/VTX files exercise the
documented public layouts and final-matrix skinning path.

They do **not** prove that a legacy CS:GO demo from a particular shipped build
can be ingested end to end. No immutable real legacy `.dem` is presently in the
local CS install or this checkout, and no test runs the decoder against one.
Consequently there is no measured real-demo message, field, class, model-binding,
animation, physics, or UI coverage, and no pixel comparison against legacy
CS:GO.

The asset evidence did materially improve during this audit. Steam's current
App 730 manifest is mounted on `BetaKey=csgo_legacy`, build 12426195, and lists
legacy depots 731/733/735. The installed tree now has `csgo_osx64`, loose legacy
BSPs, and a legacy `pak01_dir.vpk`; it still has no `.dem`.

Read-only real-asset probes established the following:

- installed `pak01_dir.vpk` is VPK v2 with 133,676 indexed entries;
- installed `de_dust2.bsp` is BSP v21, 325,735,952 bytes, and the current world
  parser produces 20,912 vertices and 20,539 triangles;
- the real `ctm_sas` MDL/VVD/DX90.VTX triplet parses to 98 bones and renders as
  five meshes / 16,471 triangles when supplied explicit synthetic bone-to-world
  matrices; and
- the real `tm_phoenix` triplet parses to 86 bones and assembles 11,703 vertices
  / 17,427 triangles after correcting fixed-width 64-byte model-name handling.

The reproducible `ctm_sas` probe, canonical snapshot, scene, PPM, depth buffer,
source hashes, and output hashes are under
`/Users/mdot/dox/runs/source1-real-asset-smoke`. Its evidence is deliberately
narrow: the assets are real Valve legacy bytes, while camera, player state, and
final matrices are synthetic. It is not a demo-ingest or engine-parity result.

## Total-capture recorder work still required

The current passive HID process plus native-demo normalizer is not the custom
total-capture recorder required by this project. The points below are recorder
implementation requirements. They are not claims that native-demo omissions
make state replay inherently impossible:

1. `modelprecache` is now a hashed canonical resource and its exact recorded
   entry content is losslessly joined to `m_nModelIndex`. The binding is
   honestly marked derived, active bindings are updated immediately when the
   table changes, and the renderer validates the same-frame entry document.
   Render LOD is not demo entity state: canonical reference mode records an
   explicit renderer-policy choice of LOD0; authoritative mode remains refused.
   Skin/bodygroup coverage is still incomplete, and selectable bodygroups need
   an unambiguous selection.
2. The total recorder must capture complete animation-machine checkpoints and
   ordered inputs sufficient for proven engine reevaluation, or final client
   `SetupBones` matrices at every required render sample. The native demo
   normalizer currently exposes neither, so the implemented skinning path
   correctly refuses current captures; that refusal identifies unfinished
   recorder work rather than an acceptable final limitation.
3. The default reducer labels physics unavailable even when some motion and
   collision netprops are present. Rigid bodies, constraints, asleep/awake state,
   interpolation history, and client physics objects are not normalized.
4. Player/observer identity is not joined through `userinfo`, handles, view
   entity, and player slot into a complete egocentric camera/player record.
5. `democmdinfo_t` does not contain projection FOV. Canonical reference output
   can use an explicit profile FOV with an omission record; authoritative output
   correctly refuses it. A normal replay needs recorded/derived FOV and scoped,
   observer, and zoom transitions from authoritative game state.
6. `net_Tick`, `svc_ServerInfo`, `svc_SetView`, `svc_FixAngle`, temp entities,
   sounds, decals, user messages, and many presentation events are not reduced.
   These omissions affect exact tick identity, observer camera, transient
   effects, audio, and diegetic HUD/UI.
7. The custom recorder must preserve decoded buttons, view changes, weapon
   selection, impulse, movement, mouse deltas, command number, prediction
   metadata, and engine/server application order for the human and bots.
   `dem_usercmd` is presently only framed and hashed. The passive HID sidecar is
   useful pre-engine intent evidence, but it neither replaces usercmd state nor
   establishes engine-consumption timing.
8. String-table state is now hashed, seekable, checkpoint-persistent canonical
   state. Aggregate provenance distinguishes recorded entry bytes and svc-create
   metadata from reducer revisions, accumulated map presence, namespace
   separation, and metadata inferred from command snapshots. This closes the
   previous late-modelprecache/checkpoint hole; it does not yet perform the
   higher-level `userinfo`/handle/player/camera joins.
9. Sparse instance baselines remain sparse. The reducer no longer calls them
   complete: it labels them
   `partial-instancebaseline-missing-constructor-defaults`. Protocol defaults
   and game-class constructor defaults for omitted flattened props are not
   materialized, and send-table schemas are not carried as canonical snapshot
   resources. The claim is fixed; the missing state is not.

Until those capture streams and joins exist, “entity netprops decoded” is not
equivalent to “renderer-ready state reconstructed,” and the armed HID process
must not be described as a full action-state recorder.

## Missing image faculties

The backend currently identifies itself as a deterministic reference renderer,
which is accurate. It is not a normal-looking CS:GO renderer. It lacks:

- VMT/VTF material evaluation and material proxies;
- BSP lightmaps, light styles, displacement surfaces, static props, sky, water,
  fog, and visibility acceleration;
- Source animation graph/sequence evaluation, IK, procedural bones, attachments,
  flexes, and ragdolls when final matrices are absent;
- weapon/viewmodel assembly, gloves, muzzle flashes, tracers, decals, particles,
  smoke, and post-processing;
- HUD, menus, killfeed, scope overlay, spectator UI, and other user messages; and
- triangle clipping at the near plane and the other raster behavior needed for
  engine parity.

Flat BSP faces and flat-shaded meshes demonstrate the state-to-pixel service
contract. They do not demonstrate CS:GO visual fidelity.

## Remaining correctness and lifecycle work

- Ingest now stages the SQLite journal, publishes it only after `dem_stop`, and
  writes a complete marker. Keep this invariant when adding streaming ingest or
  custom reducers; a half-demo must never become an ordinary renderable stream.
- The renderer now refuses mixed-stream batches before assigning tick/subtick
  output names. Keep the one-stream invariant at every service entry point.
- PacketEntities `baseline`/`update_baseline` semantics need validation against
  the actual CS:GO client implementation and a real demo; Valve's small dump
  tool reads those fields but is not itself a full client recurrence oracle.
- Duplicate send-table/string-table names are now refused instead of aliasing a
  last-wins schema or canonical resource. Noncanonical protobufs, external VPK
  archive length bounds, and metadata limits still deserve corpus-driven
  malformed-input tests beyond the current unit fixtures.
- `Source1ReplayAdapter` is explicitly one-shot. A second ingest is refused
  rather than leaking entity generations, table revisions, or model bindings
  across streams.
- The ingest manifest phrase “entities reconstructed when datatables and
  baselines are present” must always be read as supported PacketEntities/netprop
  reconstruction, not complete game/client/render state.

## Real acceptance gate

Do not advance the evidence level until all of the following exist:

1. an immutable legacy `.dem` plus SHA-256 and demo/network protocol, joined to
   the now-present build-12426195 depot manifests and exact Source 1 assets;
2. a side-by-side parse against Valve's reference for commands, tables, classes,
   string tables, events, and representative entity properties;
3. linear replay versus checkpoint/seek hash convergence across the full demo;
4. coverage reports for every observed message type, sendprop encoding, class,
   model binding, animation input, physics object, and UI event;
5. an end-to-end ingest-produced snapshot rendered without hand-authored camera,
   geometry, model, or animation components; and
6. image comparison against frames from the matching legacy client, with every
   expected gap named rather than hidden by a placeholder.

Without that corpus, the honest result is a rigorously tested Source 1 protocol
and reference-rendering scaffold—not a finished CS:GO state renderer.
