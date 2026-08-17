# State/action replay dyad: headless CS2 demonstration plan

Status: executable foundation, with hard gaps named below. This design is for
one consented subjective client playing local bot matches, captured once for
roughly an hour, then replayed repeatedly through a headless state service and
batched renderer.

## The claim we are trying to earn

Given:

- an immutable native CS2 `.dem`;
- a passive high-rate ego input sidecar;
- exact build/content provenance;
- a complete initial state plus ordered entity deltas; and
- explicit presentation-state availability;

the system should reconstruct any requested tick, return the state immediately
before and after each ego action, and feed coherent batches to a renderer much
faster than real time. A separate GPU result must then show that the rendered
sequence is visually normal and temporally coherent.

That is two claims, not one:

1. **state/action recovery:** deterministic checkpoints, deltas, lifecycle,
   action context, seek equivalence and state hashes; and
2. **render consumption:** every state family asserted by the fidelity label
   actually reaches a renderer implementation and affects the resulting image.

Fast state recovery does not establish rendering quality. Attractive video
does not establish state recovery. The demonstration must publish both result
sets and their join.

## Architecture

```text
             supported CS2 interfaces only
  trackball/keyboard ── passive IOHID sidecar ─┐
                                               ├─ capture manifest
  local bot match ──── native CS2 .dem ───────┘
                             │
                             ▼
             demo + ego-sidecar normalizer       NOT YET IMPLEMENTED
          (generic entities, qualified fields,
           lifecycle, actions, provenance)
                             │ transactions/checkpoints
                             ▼
  state-integrator service :8791
  - hash-chained journal
  - arbitrary seek / batched states
  - pre/post action context
                             │ tardigrade/state-snapshot/v1
                             ▼
  strict renderer bridge
  - authoritative/canonical policy
  - camera/player/physics/animation routing
  - current-renderer staging + gap manifest
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
  repaired GPU render service :8815   explicit refusal
  - canonical profile once            - physics unconsumed
  - session batching                   - recorded bones unconsumed
  - output validation                  - missing/derived fields
                 │
                 ▼
      MP4s + state hashes + timing + gap manifest
```

The normalizer is deliberately outside the integrator. It decides what the
source actually says. The integrator applies validated state transitions but
does not know CS2 field meanings. The bridge decides which normalized state the
renderer can consume but does not integrate or interpolate it. This prevents
the old renderer pattern in which a draw stage independently guessed history
from sparse rows.

## Implemented now

### Persistent state-integrator engine

`src/state_replay/integrator.py` and `scripts/cs_state_integrator_server.py`
implement:

- complete initial and periodic checkpoints;
- ordered create/update/destroy/action transactions;
- stable entity IDs plus generation reuse checks;
- strict sequence, time, base-hash and optional post-hash validation;
- omission-as-unchanged, JSON-null-as-data and explicit component removal;
- deterministic canonical state hashes;
- atomic batch ingestion;
- arbitrary seek, one-pass batch retrieval and checkpoint convergence;
- action context with exact pre-action and post-transaction snapshots;
- persistent SQLite WAL storage in a caller-owned artifact directory;
- bounded localhost HTTP endpoints for manifest, ingest, checkpoint, state,
  batch, action context, health and statistics.

The renderer-facing snapshot protocol is documented in
[`src/state_replay/PROTOCOL.md`](../../../src/state_replay/PROTOCOL.md).
`subtick` is a non-negative integer ordering ordinal. Capture-specific timing
units and original timestamps belong in the manifest/action payload rather
than being rounded into this ordinal.

### Strict state-to-render bridge

`src/counter_strike_render/replay_bridge.py` accepts complete
`tardigrade/state-snapshot/v1` states. It:

- requires explicit `recorded`, `derived` or `unavailable` envelopes;
- validates stream ordering, hash chaining when supplied, generations and
  explicit discontinuities;
- produces subsystem-major camera, player, dynamic-physics and animation
  batches;
- refuses authoritative world-pose mode unless dynamic bodies have recorded
  transforms and animated entities have recorded bones or declared complete
  graph state;
- stages current camera/playermodel JSON only from explicit values;
- writes `tardigrade/legacy-render-gap-manifest/v1`, aggregating every field or
  subsystem that the current renderer cannot consume;
- refuses authoritative legacy staging if physics, evaluated bones, graph
  state, effects, subticks, derived values or another unsupported input would
  be dropped.

This is the decoupling instrument: it can prove that state recovery succeeded
while separately proving that the old renderer cannot yet consume that state.

### Repaired legacy render service boundary

The existing service previously passed a profile/world path where an argument
vector was expected, expanding the path character by character. It also could
return a nominally successful manifest after renderer failure or absent output.

`scripts/cs_render_service.py`, `cs_render_server.py` and
`cs_render_client.py` now:

- load one canonical render profile once per job;
- build a real argument vector shared by all views;
- validate views, players, ranges, resolution and request size;
- use stable SHA-256 job identities and confined output directories;
- reject the obsolete single-world input with migration guidance;
- refuse nonzero renderer exits and missing, empty or stale MP4 outputs;
- distinguish invalid request, render refusal and internal failure over HTTP.

The service still consumes the current TARD/camera-file renderer interface. It
does not magically add physics or evaluated-bone consumers.

### Polite subjective-client capture

The implementation and operating instructions are in
[`tools/cs2_ego_capture/README.md`](../../../tools/cs2_ego_capture/README.md),
with the public-API and observability analysis in
[`tools/cs2_ego_capture/BACKEND_AUDIT.md`](../../../tools/cs2_ego_capture/BACKEND_AUDIT.md).

The Swift recorder's production default uses a non-seizing, per-device
`IOHIDDeviceRegisterInputReportWithTimeStampCallback` to retain the raw
interrupt report, public arrival timestamp, report descriptor and exact
`VID:PID[:LOCATION]` identity. Optional decoded IOHID values and a listen-only,
physically unattributed CGEvent stream provide independent calibration views;
they are never summed as additional actions. Permission is diagnosed with the
API belonging to each backend, and a public permission request occurs only
through a separate, explicit consented command. The old cross-backend design
error—using a failed CoreGraphics preflight to refuse IOHID before attempting
it—is removed.

Capture requires explicit consent and foreground application gating. Production
sessions bind the first matching foreground PID and run until it terminates;
finite duration remains only for empirical probes. Visible UI reminders are
delivered at a configured interval while focused, or on refocus if overdue. It
records raw reports or numeric values and multiple clocks, never
text. It does not inject, modify or seize input; attach to, debug or read CS2;
load code into CS2; install persistence; modify Steam/game configuration;
issue console commands; inspect network traffic; or upload data.

For `csgo_legacy`, `tools/csgo_total_capture/monitor.py` is the only supported
session coordinator. It starts exact-device HID before ordinary game lifecycles, owns native
demo start/stop through the allowlisted in-process engine interface, joins the
native diagnostic and demo, hashes the outputs, and refuses completeness when
any required producer or state faculty is absent. The older passive supervisor
explicitly refuses this variant; no manual console workflow is accepted as a
legacy total-capture session.

Passive HID time is the finest polite host observation available here. It is
not the exact moment CS2 consumes an input. Device delivery, OS processing,
client sampling, user-command construction, server application and demo
serialization are different clocks. The manifest can retain independently
established affine anchors; the final report must show residual/error bounds
and must not relabel the estimate as engine-poll time.

## Current machine facts

At implementation time:

- host: macOS 26.3.2, ARM64;
- trackball: Logitech ERGO M575S over BLE, VID `0x046d`, PID `0xb041`,
  97-byte report descriptor, 20-byte maximum report and reported 8,000 us
  interval (125 Hz device cadence);
- Steam app 730 selected branch: `csgo_legacy`, build `12426195`;
- installed App 730 content: 96,926,789,098 bytes, including legacy depots
  `731`, `733`, and `735` alongside retained Source 2 depots;
- runnable legacy `csgo_osx64` binary and Source 1 BSP/VPK assets: present;
- runnable native macOS CS2 client binary: absent;
- the desktop “Counter-Strike 2.app” is an unsigned shell shortcut containing
  only `open steam://run/730`;
- IOHID listen/Input Monitoring from the cmux-launched stable recorder: granted;
- CoreGraphics listen preflight from that context: true.

The original public-branch download was Source 2-only. Steam subsequently
switched to `csgo_legacy` and installed the matching Source 1 macOS client and
assets. No legacy `.dem` exists yet, and no client process has successfully
become foreground during the armed capture session. Retained Source 2 assets
remain invalid inputs to the Source 1 backend.

Therefore no match has yet produced a demo or baseline input capture. The code
can inventory the actual trackball, compile and open both capture backends on
this machine. The remaining host blocker is a supported runnable client
environment; that condition may not be bypassed with process instrumentation.

## Capture package for the one-time bot session

Each session must contain:

- original native `.dem`, byte length and SHA-256;
- HID JSONL and its SHA-256;
- CS2 app/build/depot manifests and content identity;
- host/OS, capture-source and capture-binary hashes;
- exact selected HID device identities and transports;
- wall, Mach continuous/absolute, monotonic-raw and HID clock anchors;
- focus transitions and capture termination reason;
- any independently obtained demo-tick alignment anchors and fit residuals;
- parser/normalizer revision, source field/entity coverage and all gaps;
- state-integrator manifest, database and checkpoint policy;
- render profile/content identities and legacy gap manifest;
- state, bridge, staging and GPU timing reports.

Do not retain hardware serial numbers or interpret keyboard usages as text.

## Demo plus input normalization

This is the largest missing implementation. It must:

1. parse the native demo generically below the player-table-only API;
2. preserve entity create/baseline/update/destroy order and generations;
3. emit a complete initial checkpoint and bounded periodic checkpoints;
4. retain qualified source paths, source types and presence separately from
   default zero;
5. normalize player/controller/pawn, weapon/projectile, objective, physics,
   animation, effects and camera state into provenance envelopes;
6. join passive ego HID actions by bounded clock alignment without replacing
   native demo outcomes;
7. keep both intent and authoritative result as separate action records;
8. emit a coverage manifest distinguishing source absence, parser absence,
   serializer absence and client-only absence;
9. refuse fields that “parse successfully” without returning the requested
   column; and
10. retain the raw demo and input sidecar after every derived revision.

The complete capture inventory is in
[`STATE_RECONSTRUCTION_CAPTURE_CONTRACT.md`](STATE_RECONSTRUCTION_CAPTURE_CONTRACT.md).

## Demonstration gates

### Capture safety and provenance

- Recorder remains a separate visible process and produces no input events.
- Capture pauses outside the selected foreground process.
- Source demo/input/build/content bytes and code are hashed.
- Capture permission denial produces no event stream.
- A complete bound-target process session ends with a stable native demo and a
  clean `target_terminated` capture terminal record.

### State correctness

- Linear replay and checkpoint seek produce identical state hashes.
- Adjacent checkpoint windows converge at overlap.
- Entity generation/lifecycle violations and base-hash corruption are refused.
- Null, absent and explicit removal remain distinguishable.
- Every ego action returns complete pre/post snapshots and its native outcome.
- Teleports, respawns, gaps and resets remain discontinuities.

### Input alignment

- At least two independent anchors span the session; more are preferred.
- The affine or piecewise fit publishes maximum and distributional residuals.
- Fire/button anchors are checked against native weapon-fire outcomes, but
  rejected/suppressed clicks remain distinct from successful shots.
- The result is described as HID-to-demo alignment, never exact engine polling.

### Renderer consumption and quality

- Every recovered subsystem appears as consumed, canonical downgrade or
  authoritative refusal in the gap manifest.
- Authoritative mode refuses until recorded physics and animation pose actually
  reach real draw consumers.
- Canonical renders label procedural animation and omitted physics.
- Arbitrary seeks do not restart idle/bind pose or temporal effects silently.
- GT comparisons cover camera, player/world placement, viewmodel, animation,
  dynamic props, smoke/effects and diegetic UI.

### Performance

Publish separately:

- transaction generation, ingest and checkpoint transactions/s;
- seek latency by checkpoint distance;
- batched snapshot states/s and bridge frames/s;
- CPU, memory and storage footprint;
- GPU cold-start, content-load, frames/s, batch occupancy and encoder rate;
- end-to-end wall time for an hour at the declared output FPS;
- hashes tying every timing result to code, data, content and profile.

`scripts/benchmark_state_replay.py` generates a machine-readable synthetic
state-engine report outside the checkout. Its result is explicitly CPU/storage
evidence, not a visual or GPU benchmark.

## Existing measured baseline

On this machine, a synthetic single-entity run measured:

- 25,000 ordered transactions ingested atomically in 0.956 s, approximately
  26,200 transactions/s;
- 251 requested positions reconstructed in one pass in 0.471 s;
- an uncached midpoint seek across 12,500 deltas in 0.233 s;
- 16.2 MB of SQLite storage.

These numbers show that the integrator architecture is comfortably faster than
a 64/128 Hz one-transaction-per-tick stream. They do not establish ten-player
generic-entity scale, one-hour query throughput, GPU speed or image quality.
The reproducible hour-shaped benchmark and actual capture must replace this
development measurement in any client-facing claim.

## Next executable milestones

1. Obtain a supported runnable CS2 client environment without anti-cheat
   interference or process instrumentation, then prove exact-device event
   delivery under the now-granted IOHID permission with the finite probe.
2. Record a short five-minute bot pilot before the one-hour capture; validate
   demo stability, foreground gating, clocks, report cadence and disk growth.
3. Implement and coverage-test the native-demo normalizer against the pilot.
4. Run the state correctness gates and the hour-shaped benchmark.
5. Add real GPU consumers for evaluated bones and dynamic physics entities;
   keep authoritative mode refusing until both pass GT comparisons.
6. Run the one-hour trackball session once, freeze its evidence package and
   publish state, alignment, render-quality and performance reports together.

Until milestones 3 and 5 are complete, the honest demonstration is: fast,
deterministic state integration; safe subjective input capture; coherent
canonical camera/player staging; and precise evidence of what the current
renderer still cannot render.
