# Legacy CS:GO Source 1 ingestion

The selectable adapter name is `source1-hl2demo-v1`. The command-line entry
point is:

```sh
python3 scripts/csgo_state_ingest.py match.dem \
  --output /Users/mdot/dox/runs/csgo-state-journal \
  --stream-id csgo/my-match --camera-split-slot 0
```

The output directory must be new. Ingestion uses a sibling staging directory
and atomically publishes it only after `dem_stop`, final checkpoint, and a
`source1-ingest.json` completion/coverage record; malformed late input cannot
leave a valid-looking partial journal at the requested path. `--tick-rate` can override the integer rate
derived from the demo header. The adapter does not launch, attach to, inject
into, or modify a game process.

## Protocol authority and evidence labels

Container layouts, command framing, message IDs, send-table flattening,
string-table coding, PacketEntities headers/field indices, and sendprop scalar
decoders are based on Valve's BSD-2-Clause `ValveSoftware/csgo-demoinfo`
reference, principally `demofile.h`, `demofile.cpp`, `demofiledump.cpp`,
`demofilebitbuf.cpp`, `demofilepropdecode.cpp`, and
`netmessages_public.proto`.

The adapter distinguishes these conditions:

- `parsed`: bytes were structurally decoded (demo framing, protobuf fields,
  send tables, event descriptors and values).
- `derived`: canonical state was deterministically reconstructed by applying
  decoded baselines and deltas. Renderer-facing netprops state says whether an
  `instancebaseline` made the entered entity complete.
- `unavailable`: the source does not provide the field or this adapter has not
  reconstructed it. In particular, final bone matrices and client animation
  state are never invented.

`Source1DemoReader` accepts demo protocol 4, including initial tick `-1`
signon/preamble commands. The adapter retains that recorded tick in action
provenance and maps it to canonical tick zero with ordered subticks. Other
negative or decreasing source ticks fail.

Packet network messages use bounded varint type/length framing. Data-table
commands decode `svc_SendTable` messages and the server-class directory.
Flattening preserves Source field-index order, exclusions, collapsible tables,
non-collapsible child-before-parent ordering, and changes-often priority.

`svc_CreateStringTable`, `svc_UpdateStringTable`, and command-level
`dem_stringtables` snapshots decode normal substring
history and fixed/variable user data. `instancebaseline` entries are decoded as
entity property streams and applied before enter-PVS deltas. Dictionary-coded
tables fail explicitly because the Valve reference does not implement that
codec.
Command snapshots replace the complete server-entry map; stale indices are
removed. Client-side snapshot entries are bounded and structurally consumed but
kept separate from (and do not overwrite) the authoritative server index map.
Every table is also a lifecycle-managed `Source1StringTable` canonical entity.
Its derived materialized-resource component contains table metadata, a monotonic content revision,
all server strings and base64 user data, plus a separate client-entry array.
Nested `field_origins` distinguish recorded entry content and
`svc_CreateStringTable` metadata from accumulated entry-map presence, derived
table ordinals/revisions, client/server namespace separation, and the
capacity/userdata-layout values inferred when only a command snapshot exists.
Creates, changes, full-snapshot removals, and client/server contents therefore
participate in state hashes, checkpoints, seeks, and reopened journals rather
than living only in decoder memory.

PacketEntities reconstruction includes ReadUBitVar edict indices, new/old
field-index coding, enter/leave/delete/delta PVS transitions, 10-bit recorded
serials, authoritative full-update replacement, and `delta_from` snapshot
selection. It reconciles the complete target map against the prior canonical
map, including implicit reversions when a delta is based on an older tick.
Canonical generations are independent monotonic per-edict lifetime counters;
the recorded serial remains in `source1_identity` and may wrap 1023 to 0.

Sendprop decoding covers int, float, vector, vector-XY, string, array, and int64
types, including varints, coordinates, multiplayer/cell coordinates, normals,
and no-scale floats. A malformed property fails the packet transactionally.
The `SPROP_XYZE` vector codec is currently refused with the property name.

Game-event descriptors and values become canonical action operations. Raw
demo commands also become actions, retaining source tick, player slot, file
offset, network sequence numbers where present, and payload digest.

## Exact remaining gaps

These are gaps in the current native-demo adapter and total-capture
implementation, not accepted limits on the intended replay package. State
families absent from the ordinary demo must be emitted by the synchronized
custom server/client recorder described in
`docs/projects/counter-strike-sft/STATE_RECONSTRUCTION_CAPTURE_CONTRACT.md`.
The currently armed HID process supplies only local pre-engine input intent.

- No real legacy CS:GO `HL2DEMO` fixture has been found locally. The only `.dem`
  discovered during this work starts with `PBDEMS2` and is a CS2 Source 2 demo,
  so it is not validation evidence. Current byte-level coverage is synthetic.
- A `dem_stringtables`-only table lacks the max-entry/fixed-userdata metadata
  carried by `svc_CreateStringTable`. Its authoritative snapshot is recovered,
  but a later network delta beyond the inferred snapshot capacity is refused.
- `dem_customdata` has no framing reader in Valve's reference and is rejected
  rather than guessed. Command compression is also rejected.
- User-command, console-command, user-message, sound, temp-entity, decal, and
  particle semantics remain recorded as framing/digests rather than renderer
  state.
- Model, material, BSP, pose-parameter, sequence, animation-state, ragdoll,
  bone, particle, and rigid-body render reconstruction is outside this adapter
  and must be supplied by the total-capture/renderer lanes rather than silently
  abandoned.
  Canonical `source1_reconstruction` records those animation/physics gaps
  without adding `animation` or `physics` routing components that would falsely
  classify every edict as visual/dynamic. A decoded `m_nModelIndex` is joined
  to `modelprecache` as a derived `source1_model.source1_mdl`, with the exact
  derivation `lossless join of decoded m_nModelIndex to recorded modelprecache
  entry`, source resource ID/revision, and canonical entry hash. Active bindings
  are updated or removed whenever the table changes. Render LOD remains
  explicitly unavailable rather than defaulting to LOD zero.
- Sparse `instancebaseline` records are overlaid before enter-PVS fields, but
  the adapter does not reconstruct game-class constructor defaults for omitted
  leaves. Such entities are marked
  `partial-instancebaseline-missing-constructor-defaults`, never complete.
- Packet/signon `democmdinfo_t` poses produce exactly one configured split-view
  canonical camera and preserve both recorded split records in action
  provenance. Resampled origin/angles flags are honored. The structure does not
  record FOV, so `camera.value.fov` is explicitly unavailable and the reference
  renderer must refuse projection until another authoritative source supplies it.
- Real-demo validation may reveal a later CS:GO network-protocol variant. Any
  unsupported field or codec fails with its byte/bit position or property name;
  the adapter does not substitute guessed state.

The reducer extension points are `on_data_tables`, `on_string_table`,
`on_entities`, and `on_game_event`. A different domain mapper can be supplied
without modifying the state integrator or Source container/parser modules.
