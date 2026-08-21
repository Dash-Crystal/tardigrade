# CS2 pre-Tardigrade capture audit and reconstruction wishlist

## Conclusion

The data currently arriving at `gpu-02:/data/dashcrystal/cs2-data/` is not a
total-state capture.

The completed `demo_downloads/*.dem.zst` files are genuine Source 2
`PBDEMS2` SourceTV demos and are the most authoritative artifacts found. They
are materially richer than both `parsed_lean/*.jsonl` and the adjacent
`*_events.json` files. Reparse the demos; do not treat either JSON derivative
as the source of truth. A current parser exposes extensive server-replicated
player, weapon, grenade, rules and team state from the audited demo, including
applied user-command fields and serialized animation-recipe carriers.

That is still not enough for 100% state recurrence or 100% visual recurrence.
The native demo does not demonstrate final evaluated bones, the complete
server and client rigid-body worlds, all client-local prediction and
interpolation state, every rendered point of view, Panorama state, final
particle state, or the temporal buffers consumed by the renderer. The lean
JSON projection discards almost all of even the state that the demo does
carry.

An inverse solver can find a render-equivalent latent path when given enough
observations. It cannot establish that the path is the original hidden state
when multiple hidden states produce the same observations. Exact-original
claims therefore require direct state capture or a closed deterministic
recurrence contract. Solver-produced state must remain labelled `derived`,
with its source buffers, constraints and residuals retained.

## Evidence and scope

This is a point-in-time, read-only audit performed on 2026-08-17 while rsync
was still receiving the corpus. Dot-prefixed rsync temporary files were
excluded. Counts will increase as transfer completes.

At the audited instant the remote tree held:

- `demo_downloads/`: seven completed `.dem.zst` files and two completed
  `_events.json` derivatives, 1,467,892,050 bytes total;
- `parsed_lean/`: 87 completed `.jsonl` files, 11,527,240,791 bytes total;
- no manifest connecting a numeric `parsed_lean/N.jsonl` name to a UUID demo;
- no producer or parser source beside the data; and
- only receiving rsync processes on `gpu-02`, not a live capture or parsing
  process.

The representative raw source used for the semantic probe was:

| Property | Value |
|---|---|
| Remote basename | `1-0e10f025-793a-46a3-9d70-177c230f557f.dem.zst` |
| Compressed bytes | 175,985,655 |
| Compressed SHA-256 | `0975bd59537e34530933918ec661ef93bc72c9a7b1049e78adfcc5442793f7b1` |
| Decompressed bytes | 230,961,881 |
| Demo SHA-256 | `2b2b86e2f0efca7f8ca1f173e6dbe85f3946e53d44309c772262c9770e4125cb` |
| Magic / format | `PBDEMS2\0`, `valve_demo_2` |
| Header | SourceTV, `de_dust2`, patch `14174`, full-packets version `2` |
| Parsed horizon | ticks 1 through 106,138; 10 players; 1,061,380 player-tick rows |
| Probe parser | `demoparser2 0.42.0` |

The parser version is part of the evidence. `list_updated_fields()` and the
high-level table APIs are not a generic registry of every entity and message
on the wire. Counts below establish what this parser actually recovered; they
are lower bounds on the native demo, not proof that unexposed wire state is
absent.

The decompressed probe copy and installed audit environment live outside the
source tree under `/Users/mdot/dox/runs/cs2-raw-log-audit/` and
`/Users/mdot/dox/runs/cs-tard21-venv/`.

## 2026-08-20 upstream `966-field` archive follow-up

Upstream commit `6d3d851` added
`data/cs2_966field_10demos.tar.zst` through Git LFS. This is a 133-byte LFS
pointer in Git to an object declaring:

```text
oid sha256:9b103ce1f72a3ec79e7ca07f00db1e331b22b364228ceca953ed6ef8cee92eec
size 598122496
```

The downloaded object exactly matches that size and SHA-256. It is not,
however, a valid complete tar archive. Decompression and tar traversal end
inside
`parsed_max/1-1eb55c60-c801-48b8-ba77-38d92ef47bca.dem/ticks.parquet`.
That file has no Parquet footer. The archive contains eight complete demo
directories, the beginning of a ninth, and no tenth directory. Thus all
three published descriptions disagree with the bytes:

- the filename says ten demos;
- the commit title says nine demos; and
- only eight demos have a readable `ticks.parquet` plus manifest and sibling
  tables.

The archive was audited outside the checkout under
`/Users/mdot/dox/runs/cs2-966field-audit`. The eight readable tick tables
contain 11,069,790 player-tick rows. They have 751 columns in common and 753
columns in union; `Weapon.m_fFireTime` and `Weapon.m_bMagazineRemoved` account
for the per-demo schema variation. The manifests themselves report 751 or
752 columns, not 966.

For the raw demo already audited above, a current direct parser probe reports
967 updated qualified properties. Exact-name comparison with its archived
Parquet finds only 673 of those properties. The 294 absent qualified names
break down as 225 grenade, 33 player-pawn, 27 player-controller, six rules,
two weapon and one team property. Some non-grenade omissions have a normalized
alias in the 78 unqualified convenience columns, but this does not explain
the missing grenade surface. The archive is a wide high-level player table,
not “every entity property” and not a schema-generic entity journal.

The eight manifests contain 204,602 events. Individual demos have 46 through
50 nonempty event families; their union is 51. This materially repairs the
earlier 12/13-family event derivatives. The event rows still have no ordered
join to entity transactions or state hashes.

The eight grenade tables contain 13,235,137 rows, of which 1,775,312 (13.4%)
have numeric positions. This is not random missing trajectory data. Across all
eight tables, coordinate presence agrees perfectly with the recorded entity
class: every row whose class name ends in `Projectile` has all three
coordinates, every non-projectile grenade-weapon row has none, and no row has
a partial coordinate tuple. The null rows are coherent carried/inventory
grenade state, not failed projectile samples. It remains inaccurate to call
all 13.2 million rows “trajectory samples”; most are repeated inventory-state
snapshots. Rendering a carried grenade also requires its attachment/final-bone
transform, which class and owner identity alone do not supply.

### Serialization fidelity defects in the wide tables

The wide tables preserve many useful values exactly. Positions, view angles,
qualified FOV, the scalar pose-recipe carrier and the server serialization
context matched a fresh parse of the same raw demo row for row. Other surfaces
do not:

- `usercmd_viewangle_x`, `usercmd_viewangle_y`, action bits such as `FIRE`,
  and most other command fields are absent;
- archived `usercmd_mouse_dx` has 30 distinct values in `[-26,116]`, while a
  fresh parse has 428 in `[-879,937]`;
- archived `usercmd_mouse_dy` has eight distinct values in `[-3,5]`, while a
  fresh parse has 200 in `[-167,122]`;
- among the 975,558 rows where both parses have a mouse value, 454,378 `dx`
  values and 291,439 `dy` values disagree; the archive additionally fills
  85,812 rows that the fresh parser leaves null; and
- the manifest records neither parser package/version nor requested and
  returned field lists, so parser-version behavior cannot be distinguished
  from all-fields-request corruption.

Several vector-valued properties are serialized as `large_string` rather
than typed arrays. The pose-recipe topology contains Unicode replacement
characters, demonstrating that opaque/binary bytes passed through a lossy
text conversion. `m_SerializePoseRecipeAG2Dynamic` is stored as a scalar
`uint32` with values 0 through 255, not a documented recipe byte buffer.

The 751-column schema has useful animation inputs, 32 fields mentioning basic
physics/collision/velocity, and 34 camera/view properties. It has:

- zero final bone-matrix or skeleton-transform columns;
- zero constraint, contact, manifold, inertia, mass or solver-state columns;
- zero view/projection matrix columns;
- zero particle or smoke-volume/grid columns;
- only `m_iHideHUD` resembling UI state and no Panorama/layout/draw-list
  state; and
- no temporal color/depth/motion/exposure/TAA render-history buffers.

The manifests contain map names and row/event counts, but no source-demo hash,
build/content identity, parser identity, schema hash, table hashes, lifecycle
coverage, state hash or closure proof. `players.json` and skins are useful
side tables but do not close those gaps. The many tiny files named `.opus`
are packet payloads, not demonstrated standalone Ogg/Opus streams.

### Repository and publication consequence

This 598 MB LFS payload should not be the canonical corpus distribution. A
normal LFS-enabled pull can materialize it inside the source checkout, contrary
to the repository's artifact-root policy, and the published object is
truncated. Publish a new content-addressed archive as a release/object-store
artifact with an external checksum and immutable manifest. Keep only the
manifest, schema, verifier and retrieval instructions in Git. Do not replace
the existing LFS object under the same semantic name: retain its hash as a
known-bad artifact and give the repaired archive a new identity.

The eight valid tables are still worth preserving as a broad derivative and
parser-regression corpus. They do not change the reconstruction conclusion:
retain the raw demos, reparse them with returned-field validation, and add
engine-side capture for state never present in a SourceTV demo.

## 2026-08-21 upstream replacement follow-up

Upstream subsequently removed the corrupt batch in commit `862c1c4` and then
replaced its interim solo parse in commit `e11226b` with
`data/cs2_complete_capture_1demo.tar.zst`. This is a new LFS object, not a
relabelling of the known-bad bytes:

```text
oid sha256:3c9c32d2f6f79175e78a82bb83f2cf66068d309b7893beb6e36e1ad13ad50c15
size 65997063
```

The downloaded object matches that identity. Zstandard decompression and a
complete tar traversal succeed. Its sole demo is the representative
`1-0e10f025-793a-46a3-9d70-177c230f557f` already audited above. Excluding the
voice packet directory, all four Parquet files open cleanly:

| Table | Rows | Columns |
|---|---:|---:|
| `ticks.parquet` | 1,061,380 | 961 |
| `events.parquet` | 22,219 | 86 |
| `grenades.parquet` | 1,280,564 | 8 |
| `skins.parquet` | 19 | 7 |

This replacement does fix the earlier cross-field parse contamination. A
fresh `demoparser2 0.42.0` parse of each field in isolation matched archived
row keys, null placement and values exactly for mouse X/Y, `FIRE`, the scalar
pose-recipe carrier, the server serialization-context iteration and recipe
topology. In particular, the mouse domains are now the direct-parser domains:
428 X values in `[-879,937]` and 200 Y values in `[-167,122]`, with 975,558
non-null rows. The earlier corrupted ranges and hundreds of thousands of
value disagreements are repaired.

The new title and manifest still overstate coverage. The table has two key
columns and 959 data columns. Those data columns comprise only 896 exact names
from the current parser's 967 updated-property list plus 63 unqualified
convenience fields. Seventy-one updated qualified properties remain absent:
59 grenade, ten player-pawn, one weapon and one player-controller property.
Notable omissions include smoke voxel/frame data, inferno positions and
parent positions, smoke and explosion origins, initial grenade position and
velocity, player weapons/ammunition arrays, body-group choices and secondary
skeleton slot IDs. Calling all 959 data columns “entity fields” hides this
alias substitution and the missing schema surface.

Applied command coverage also remains incomplete even though the repaired
mouse columns and five decoded action flags are useful. The current parser can
recover, from these same demo bytes, command view-angle X/Y/Z, forward and
left movement, and three command button-state words. Those fields have
975,558 or 958,940 non-null player-tick rows, respectively, but none is in the
replacement table. Therefore these are still serializer omissions, not
original-log omissions.

“Byte-level capture” is not an accurate description of the Parquet. Its tick
schema contains no binary column. Recipe topology is `large_string`; every
one of the first 100,000 sampled values contains Unicode replacement
characters, so arbitrary source bytes cannot be recovered. The
`m_SerializePoseRecipeAG2Dynamic` column remains a scalar `uint32` whose domain
is 0 through 255, not a documented byte buffer. Solo parsing makes this table
a faithful serialization of the current parser's returned values; it does not
make the parser's lossy string conversion byte-preserving.

The grenade table makes a useful literal state-machine distinction: exactly
212,328 of 1,280,564 rows (16.6%) are projectile-class rows, and exactly those
rows have all three coordinates. Every one of the 1,068,236 base
grenade-weapon rows has null coordinates, consistent with a grenade carried in
a player's inventory rather than an independently positioned projectile; no
row has a partial tuple. Those rows are meaningful inventory-state snapshots,
not corrupt trajectories. They still cannot all be described as “trajectory
samples,” and owner/class state does not reconstruct the held model's exact
attachment pose for rendering.

The replacement manifest records method, map and aggregate counts, but still
does not record the raw demo hash, parser package/version, requested and
returned field lists, schema/table hashes, exact executable or content
identity, lifecycle coverage, state hashes or closure evidence. The commit's
archive checksum is useful transport integrity but is not inside a sealed
capture manifest. The advertised event artifact is Parquet, not the earlier
quoted `events.jsonl` form.

Most importantly, the replacement does not add the state surfaces needed for
visual recurrence. Keyword and type inspection still finds no final bone
matrix palette, physics contacts/constraints/manifolds/solver state, final
view/projection matrices, particle or smoke-volume state, Panorama/layout/draw
state, or temporal color/depth/motion/exposure/TAA/HZB buffers. Three fields
mention interpolation-history initialization, but no history buffers are
present. This is a repaired and substantially wider one-demo derivative, not
a total-state or total-visual-state log.

The current replacement was independently audited outside the checkout under
`/Users/mdot/dox/runs/cs2-complete-capture-audit`. The repository-distribution
recommendation remains: publish large capture bytes through a content-addressed
artifact store or release, and keep their sealed manifest, schema and verifier
in Git.

## What is actually in the lean JSON

Twelve stable files spread across the received numeric range were streamed in
full: 1,703,323,765 bytes, 1,407,177 records and 10,058,391 player rows. Every
sampled record had exactly two top-level fields:

```text
t  integer tick
p  list of player rows
```

Every sampled player row used exactly this 15-field vocabulary:

```text
i hp tm pos yaw pitch dy dp mdx mdy spd w f sc rk
```

Observed types and behavior establish positions, view angles, integral mouse
deltas, angle deltas, health, team, weapon labels and several scalar flags.
The compact names are not self-describing and no schema document accompanied
the files. In one checked file `f` and `sc` were binary and `rk` took ten
integer values, but attributing their meanings solely from those values would
be an inference. `dy` and `dp` matched consecutive `yaw` and `pitch`
differences to the JSON's approximately four-decimal quantization. `spd`
reached 193,307.1 without a discontinuity marker, so it cannot safely drive
continuous physics reconstruction across every row.

Player lists ranged from one to ten rows per record. The sampled streams were
monotonic and had no duplicate ticks, but every sample had skipped tick steps.
There are no explicit entity enter/leave/destroy records, generations,
baselines, full checkpoints, clocks other than `t`, delta-base identities,
state hashes, discontinuities, map/build identity, asset identity, Steam IDs,
player names, source-demo hash, or terminal closure record.

The lean form is useful for player-motion or action modeling. It is not a
state journal and cannot be made recurrent by adding a starting seed: most
state-machine inputs and outputs never enter the projection.

## What is actually in the event derivatives

The two completed `_events.json` files held 12 and 13 event families. The
representative file adjacent to the probed demo held these 12:

```text
bomb_exploded bomb_planted flashbang_detonate inferno_expire
inferno_startburn player_blind player_death player_hurt round_freeze_end
round_officially_ended smokegrenade_detonate weapon_fire
```

The raw-demo parser found 46 event families in the same demo. The derivative
therefore omitted 34 event families that were demonstrably recoverable,
including `fire_bullets`, `hegrenade_detonate`, `player_spawn`,
`player_footstep`, `weapon_reload`, `weapon_zoom`, item transitions, bomb
begin/drop/pickup transitions, smoke expiry, server cvars and multiple round
transitions. This is a serializer omission, not an original-log omission.

The event JSON is an event index. It does not replace ordered entity deltas or
state checkpoints, and it has no hash binding it to the demo bytes.

## What the raw demo demonstrably carries

For the representative demo, `list_updated_fields()` returned 967 unique
qualified properties on the parser's high-level surface:

| Parser family | Updated qualified properties |
|---|---:|
| `CCSPlayerPawn` | 318 |
| grenade projection | 239 |
| weapon projection | 175 |
| `CCSPlayerController` | 119 |
| `CCSGameRulesProxy` | 101 |
| `CCSTeam` | 15 |

Examples that were returned with values include:

- position, velocity, eye angles, duck amount, health, team, weapon and round
  state;
- `usercmd_mouse_dx`, `usercmd_mouse_dy`, forward/left movement,
  user-command view angles, three button-state words and decoded action bits;
- qualified FOV, view punch, FOV-transition and view-entity properties;
- graph definition, serialized pose recipe byte-like values, recipe topology,
  active slot, server graph/context iterations, model and sequence handles;
- collision groups, physics-enable values, base and initial velocities;
- smoke/fire/explosion seeds, tick origins, colors and positions; and
- voice packets, player identity, skins and event payloads.

Applied user-command coverage was substantial but not total: mouse and
view-angle fields were non-null on 975,558 of 1,061,380 player-tick rows;
decoded button fields were non-null on 958,940 rows. These are server-demo
observations, not a record of each client's unconsumed device reports or local
prediction timeline.

The qualified camera FOV property was present on every row with values
`0/10/15/40/90`. The convenience alias `fov` returned only zero, and a request
for the convenience alias `aim_punch_angle` was silently omitted. Requests
for unrelated grenade properties through the player-tick API were also
silently omitted. This is why a field-request list is not a coverage proof:
the returned columns and value domains must be checked.

Animation state is richer than the lean export. The parser returned a
serialized pose-recipe carrier, topology data, graph identifiers and
iterations. It did not return a final matrix palette for each player and
render frame. Re-evaluating the recipe might be possible with the exact game
build, graph resources, animation assets, evaluation clock and all local
inputs, but those dependencies are not sealed by this corpus. A serialized
recipe is not itself evidence of the final evaluated pose.

`parse_grenades()` produced 1,280,564 rows. Its `grenade_type` classification
fully explains coordinate presence: all 212,328 projectile rows have numeric
`x/y/z`, while all 1,068,236 base grenade-weapon rows have null coordinates.
The latter are carried/inventory state, not missing projectile positions. The
table preserves owner and class but not the final bone/attachment transform
needed to place a visibly held grenade exactly.

## Recovery sufficiency

| Surface | Raw `.dem.zst` | Lean/event derivatives | Sufficient for authoritative recurrence? |
|---|---|---|---|
| Source identity | PBDEMS2 header, map, patch string | Absent | No: no exact executable/module/depot/shader/content hashes |
| Player/server actions | Applied user-command fields on most player rows | Mouse/angle deltas and a few flags | Partial: no complete client input/prediction/apply-order record |
| Player and rules netstate | Extensive parser-visible properties | Tiny projection | Partial: raw is reusable; lean is not a state journal |
| Generic entity lifecycle | Native demo packets exist; current high-level probe is restricted to six families | Absent | Not demonstrated; requires a schema-generic decoder and explicit lifecycle audit |
| Initialization and seek | Native full-packet mechanism advertised in header | Absent | Potentially for replicated demo state; no checkpoint contract for client/evaluated state |
| Animation state | Recipe/graph/model/sequence carriers | Absent | No final bones, attachments, IK, cloth or render-frame evaluation closure |
| Physics | Selected flags, collision groups and velocities | Position/speed only | No complete bodies, shapes, constraints, contacts or client physics world |
| Effects | Events and selected smoke/fire/grenade properties | 12/13 event families | No final particle/voxel/decal/light state or recurrent effect checkpoints |
| POV/camera | Server eye/FOV/punch/view properties | Per-player angles only | No final per-client view/projection matrices, interpolation or visibility |
| UI | A few gameplay values that could feed UI | Absent | No Panorama graph, layout, animation or draw state |
| Render history | Not demonstrated | Absent | No prior-frame transforms, exposure, TAA, HZB, motion or other temporal buffers |
| Audio/voice | 135,201 voice packets in the representative demo | Absent | Recoverable for voice, unrelated to visual recurrence closure |
| Integrity | Source bytes can be hashed | No source binding/state hashes | No end-to-end state or frame hash chain |

The raw demos should be retained and reparsed with a schema-generic Source 2
decoder before declaring any replicated property missing. The renderer still
needs custom capture for state that was never server-replicated or never
stored in the demo.

## Logging wishlist for state and visual closure

The producer should emit an append-only, hash-chained transaction stream plus
periodic full checkpoints. Every record needs source sequence, simulation
tick/subtick, render-frame ordinal, monotonic interval, process/map/capture
epoch and predecessor hash. A reset, map change, reconnect, device-focus
change or dropped record is a first-class discontinuity, never an inferred
gap.

### 1. Immutable identity and recurrence inputs

- Exact executable and loaded-module hashes, build and protocol versions.
- Map, BSP/world, model, animation graph, particle, material, texture, shader,
  localization and Panorama content hashes.
- All replication schemas, serializer tables, string tables and class IDs.
- Initial and changing cvars/convars, tick rate, interpolation/prediction
  settings, graphics settings, viewport, color space and driver/GPU identity.
- RNG algorithm/state or seed plus draw ordinal for every gameplay,
  animation, particle, audio and rendering subsystem.
- A full checkpoint at capture start and after every discontinuity, followed
  by independently applicable deltas and regular checkpoint barriers.

### 2. Generic world and action journal

- Every entity's stable ID, handle serial/generation, class, create, enter,
  leave/dormancy, parent/reparent and destroy transitions.
- Complete baseline values and every ordered property delta, including
  dynamic props, doors, breakables, hostages, dropped equipment, projectiles,
  ragdolls, decals, lights and controller/global resources.
- PVS/visibility state per POV and the exact delta base used by each packet.
- For every human or bot: generated command, prediction input, command after
  engine mutation, server-applied command and post-action state, with exact
  subtick histories and before/after hashes.
- HID reports as interval-integrated counts with report sequence and
  `[t_begin,t_end]`; do not label `count/dt` an instantaneous velocity.
  Separately log engine-applied angular targets and final rendered camera
  anchors. The continuous trajectory between anchors is derived, not an
  inversion of a unique physical trackball motion.

### 3. Animation and deformable state

- Graph definition and instance IDs, parameters, state-machine node, all
  transitions, layers, sequence/cycle/rate, pose parameters, recipe bytes,
  evaluation iteration, reset/no-interpolation serials and evaluation time.
- Model, skin, material group, body groups, hitbox set and selected LOD.
- Final local and world bone matrices for every rendered skeleton at every
  render sample; attachment matrices and ownership; viewmodel bones in their
  own projection.
- IK goals/results, look/aim constraints, procedural bones, morph weights,
  cloth and secondary skeleton state.
- Ragdoll handoff and the final render palette, not only the damage bone and
  initial force.

Final matrices are the strongest practical checkpoint. If graph recurrence
later proves exact, they become validation witnesses; until then they prevent
animation guesses from contaminating rendering.

### 4. Physics worlds

- Separate authoritative/server, predicted-client and presentation physics
  world IDs and step ordinals.
- Body lifecycle, collision shape/resource hash, transform, linear/angular
  velocity, accumulated force/impulse, mass, inertia, center of mass,
  material, collision filters, kinematic/sleep flags and island identity.
- Constraint lifecycle and parameters, contact manifolds, trigger overlaps,
  solver warm-start/cache state and post-step body checkpoints.
- Character-controller sweeps, ground/contact body, ladders, moving parents,
  break events and debris lifecycle.

Logging only transforms can reproduce visible rigid placement. Logging the
solver inputs and recurrent cache is required to continue the physics state
machine beyond a checkpoint.

### 5. Every rendered POV and presentation state

- One stream for every scored human, bot, observer, replay/spectator and
  viewmodel POV; a world-state demo is not equivalent to these client views.
- Simulation eye, predicted eye and rendered eye transforms; roll, FOV,
  aspect, viewport, near/far planes, jittered and unjittered view/projection
  matrices and previous matrices.
- Interpolation fraction, render time, prediction correction, recoil/view
  punch composition, observer target/mode and camera transitions.
- Per-POV visibility/PVS, occlusion decisions, renderable list and entity to
  draw-instance mapping.
- Exposure, tonemap, color correction, flash/scope/damage overlays,
  post-process volumes and viewmodel projection/depth policy.

### 6. Effects, smoke, lighting and UI

- Particle-system lifecycle, definition hash, control points, parent and
  attachment, RNG state, per-particle attributes or lossless simulation
  checkpoints, and spawn/update/death order.
- CS2 smoke volume/grid state and advection inputs, inferno cells, tracers,
  shell casings, muzzle flashes, decals, blood, debris, fog and transient
  lights/shadows.
- Panorama document/tree identity, data bindings, state transitions, style
  and layout results, animation clocks, visibility, focus/input state,
  localization strings, glyph/font atlas identity and final UI draw list per
  POV.
- Scoreboard, kill feed, radar, chat, buy menus, spectator panels and every
  screen overlay as recurrent state, not merely the gameplay values that
  might feed them.

### 7. Render-history and inverse-solver witness buffers

For exact temporal rendering, retain the renderer's actual previous-state
inputs: previous transforms and bones, exposure/adaptation state, TAA/history
color and validity, motion vectors, previous depth, HZB/occlusion state,
shadow caches, particle simulation buffers, viewmodel history and every reset
mask.

For a tractable render-equivalence solver, the most useful per-frame/per-POV
witness set is:

- pre-tonemap HDR and final display color;
- linear and device depth;
- world/view normals, albedo, roughness/metalness and material ID;
- entity, draw, primitive, skeleton and particle IDs;
- motion vectors/optical flow and disocclusion/history-valid masks;
- direct/indirect lighting, shadow masks, fog/transmittance and transparency
  layers or weighted-accumulation buffers;
- UI color/alpha and UI element IDs as separate layers; and
- the exact visibility/renderable list for the frame.

For pixel-exact reproduction independent of semantic inversion, also capture
the render graph and API-facing command stream: pipeline/shader hashes,
descriptor/resource bindings, push/root constants, dynamic uniform/storage
buffers, vertex/index/instance buffers, skin palettes, indirect-draw buffers,
texture subresource updates, render-pass order and synchronization. Static
resources should be content-addressed once; dynamic uploads should be
hash-chained by frame.

### 8. `d_buffer/d_t` should be a delta stream, not a fictional derivative

For each dynamic buffer record:

```text
buffer_id, schema/format, shape, byte_length
t_begin, t_end, source_sequence, frame/tick/subtick
predecessor_buffer_sha256, reconstructed_buffer_sha256
encoding = full | xor | changed-spans | typed-delta
payload, reset/discontinuity mask, lossless/quantization declaration
```

`d_buffer/d_t` should mean a lossless change over a stated interval. Preserve
the predecessor and reconstructed hashes and insert full keyframes often
enough to bound corruption and seek cost. If a lossy residual or learned
codec is used for an auxiliary solver stream, retain its scale, saturation
mask and reconstruction error; it cannot be the authoritative state record.

### 9. Terminal and coverage proof

- Seal every stream with start/end horizon, record count, dropped-record
  count, schema hash and terminal hash.
- Publish a coverage manifest enumerating every required subsystem and POV,
  with `recorded`, `derived`, `unavailable` or `discontinuous` status.
- Reconstruct every checkpoint from empty state, replay deltas to the next
  checkpoint and compare state hashes before promoting a capture.
- Join rendered outputs to the exact state hash, POV state hash, content
  manifest and all witness-buffer hashes.

## Practical priority

1. Preserve and hash every raw `.dem.zst`; finish a source manifest that joins
   UUID demos, event derivatives and numeric lean files.
2. Replace the lean serializer with a schema-generic lifecycle/checkpoint/
   delta journal over every demo-replicated entity and global table.
3. Add sidecars for final cameras, bones/attachments, physics and effects at
   simulation and render clocks.
4. Add per-POV UI and render-history capture.
5. Capture witness buffers and dynamic render uploads for frames used to fit
   or verify inverse reconstruction.

This ordering recovers everything already present before asking a solver to
invent anything. Any remaining inverse result is then constrained against
explicit buffers and can be reported honestly as one render-equivalent path,
not as proof of an unobserved original state.
