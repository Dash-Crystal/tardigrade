# Total-capture replay protocol

This protocol is the engine-neutral boundary between a recorder/state integrator
and a renderer. It does not claim that rendering is stateless. Every scored
frame is a recurrence over an authoritative initial state, ordered state/action
deltas, recorded final simulation/render products, and renderer history.

The shared machine-readable contract is `total_capture_contract.py`. Documents
use canonical JSON and are SHA-256 sealed. A manifest immutably identifies the
engine build, content manifest, native demo, every sidecar, tick horizon,
checkpoint cadence, scored POVs, streams, and a field-level completeness matrix.
Ingestion verifies the bytes of the demo and every declared sidecar; hash claims
alone are insufficient.

Completeness additionally requires `producer_closure` owned by the capture
monitor. For Source 1, ingestion parses the actual `HL2DEMO` v4 bytes through
the terminal `dem_stop` and compares its stop tick, playback ticks, map, and
server to the manifest horizon. Each `capture-stream` sidecar must end in a
self-hashed `tardigrade/capture-sidecar-terminal/v1` JSON record with the same
capture-session ID and horizon. That terminal embeds the exact native-demo SHA,
and the manifest names the monitor-log sidecar, binding ownership and terminal
evidence to those demo bytes. `content-package` sidecars remain byte-hashed but
do not pretend to have a timeline terminal. A crash-truncated demo or sidecar is
representable only as `tardigrade/incomplete-capture/v1` salvage evidence; that
schema is deliberately not accepted by the replay ingestor or authoritative
renderer.

Required global streams are `usercmd`, `entity_lifecycle`, `final_pose`,
`rigid_body`, and `effects`. Every scored POV separately requires `camera`, `ui`,
`visibility`, and `render_history`. Camera values include recorded origin,
orientation, FOV, exact view/projection matrices, clip planes, vantage,
prediction state, and the engine/monotonic/source clock that produced them.
The camera also records its viewport plus matrix memory layout, row/column-vector
usage, handedness, view-axis order, clip-depth range, and NDC Y direction; no
renderer may infer one API's matrix convention from another engine's name.
Viewmodels carry their own viewmodel projection/FOV and a `final_pose` in view
space. World actors and ragdolls use world-space final poses. Final bones and
attachments are post-animation matrices; the reducer never synthesizes them.
Rigid bodies include final transforms and velocities; constraint and effect
records are explicit components rather than renderer guesses.
Entity lifecycle checkpoints and deltas also carry complete component state and
recorded `world_transform.matrix_3x4`; identity/lifetime events alone are not a
state reconstruction.

`model_binding` is recorded and contains `engine`, opaque `model_id`,
`model_content_sha256`, `material_set`, `skin`, `bodygroups`, and `lod`.
Resource bytes resolve only by the recorded SHA through an explicitly configured
capture content-package sidecar whose own SHA and content-manifest SHA are in the
manifest. Filename scans and fallback models are non-conforming.

Each stream begins at its declared horizon with sequence-zero `checkpoint` and
`payload={"full":true,"ops":[...]}`. Further full checkpoints must occur within
`checkpoint_interval_ticks`. Deltas and checkpoints have strictly consecutive
stream/source sequences, monotonic clocks, nondecreasing engine positions, and
sealed record hashes. Every stream ends exactly at the capture horizon with an
`end` record whose payload is exactly `{"status":"complete"}`. A gap, duplicate,
missing scored POV, unavailable required field, bad hash, missing terminal, or
post-terminal record rejects the whole capture.

User-command deltas distinguish human and bot actors and contain the recorded
intent plus engine-confirmed applied outcome operations. The reducer emits the
action before outcome operations, so `StateIntegrator.action_context` returns the
state immediately before the command and after its effects.

HID `relative_counts` are signed counts integrated over the explicitly recorded
`start_monotonic_ns..end_monotonic_ns` report interval. They are never described
as instantaneous velocity or a unique sub-report physical trajectory. Each
applied command also carries the authoritative target view-angle anchor and a
focus-segment ID plus an explicit boundary bit. The first human segment and
every segment-ID change must declare a boundary; a continued segment must not.
Bots carry explicit nulls because they have no HID/focus provenance. The optional
canonical `angular_target_history` is explicitly
derived, cites its source artifact hashes, retains every HID integral and
usercmd/render-camera anchor with exact clocks, and supplies at least one
strictly ordered continuous sampled trajectory between anchors. Missing anchors,
intervals outside the anchor horizon, or undisclosed focus discontinuities are a
refusal, not an invitation to invent motion.

Canonical state contains `total-capture:metadata`. Its
`total_capture_manifest` component contains the sealed manifest. Its derived
`capture_completeness` component uses the exact shared derivation string and is
initially open; only validated terminal hashes close every stream and transition
it to complete. No discontinuity is permitted inside the horizon. The ingest
journal is built in a private sibling staging directory, receives a final
checkpoint, and is atomically renamed into place only after closure. Failure
removes the self-created staging directory and never publishes the requested
output path.

Run `scripts/total_capture_ingest.py MANIFEST RECORDS --demo DEMO` with one
`--sidecar ID=PATH` for every declared sidecar and a new `--output` directory.
