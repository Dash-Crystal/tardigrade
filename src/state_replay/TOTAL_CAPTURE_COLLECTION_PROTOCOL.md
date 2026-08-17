# Multi-epoch total-capture collections

A collection is an outer host-clock transcript around the existing single-demo
total-capture protocol. A recurrence domain is one `capture_epoch_id`: exactly
one authoritative initial checkpoint, one closed `HL2DEMO` plus sidecars, and
one independent journal stream. Process and map epoch IDs are additional axes;
reconnects and recording rotations can create several capture epochs under the
same process and map.

The request specifies a target quantity of complete target-active gameplay, not
a fixed wall duration. `required_target_active_ns` can therefore take an hour to
collect across a longer wall span containing downloads, menus, restarts, or
inactive periods. `max_wall_ns` is only an optional safety bound. The observed
host-clock end comes from the sealed collection terminal.

Coverage intervals exactly partition that observed wall span as
`target-active`, `target-inactive`, or `gap`. Total active time and active time
belonging specifically to complete epochs are separately recomputed. Only the
latter counts toward `corpus_status=target-met`. `continuity_status` independently
reports `gap-free` or `gapped`; a declared crash does not invalidate later good
epochs or get silently bridged. A target-not-met collection requires an explicit
salvage-publication option.
The sealed manifest also hashes exactly two outer witnesses: the raw monitor
event stream and a canonical coverage ledger. Ingestion verifies their bytes and
requires the ledger clock and intervals to equal the manifest exactly; coverage
is not accepted as an unaudited summary field.

Every complete epoch repeats all single-epoch checks: parsed `dem_stop`, sidecar
closures, ordered records, aligned full checkpoints, and atomic journal publish.
Its initial checkpoint also contains a persistent, losslessly derived entity
`total-capture:epoch-context`, identifying collection, process, map, capture
epoch, host horizon, prior epoch, and reset kind. Reset kinds distinguish session
start, process restart, crash recovery, map change, reconnect, and recording
rotation. Each epoch receives a distinct stream ID and action namespace.

Incomplete/crashed epochs have a sealed salvage document and verified artifacts,
but no journal, frames, or canonical state. Host-clock lookup into inactive,
gap, or incomplete coverage is a refusal. Complete epoch clock indexes map
recorded host timestamps to tick/subtick/stream sequence without interpolation.

The collection is assembled in a private sibling staging directory. A failure
in a later epoch removes already staged journals and leaves the requested output
absent. On success it publishes the sealed manifest, terminal, epoch journals,
salvage documents, clock indexes, and collection index in one rename.

The current `tools/csgo_total_capture/monitor.py` emits a diagnostic
`tardigrade/csgo-outer-monitor-plan/v1` outer manifest whose own status and claim
say it is not total capture. It is intentionally rejected by the collection
validator. `scripts/total_capture_monitor_handoff.py` seals that output as
`tardigrade/total-capture-monitor-handoff/v1` salvage evidence with stable
process/map/capture segment identities; it never promotes the diagnostic.
`promote_verified_total_epoch()` is the separate promotion boundary: it accepts
only a complete single-epoch manifest, records, and artifacts that pass all
hash, sidecar, and parsed-demo closure checks, and returns an exact collection
descriptor plus ingestion input. Thus future total sidecars have an explicit
path into the collection without reinterpreting today's diagnostic logs.
