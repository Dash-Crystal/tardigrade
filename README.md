# HCI-128 v2.1

128Hz human-computer interaction telemetry with full input state, viewangle anchors, and game context.

## Quick Start

```python
from read_v2 import load

df = load("data/0.tard.zst")
print(df.head())
print(f"Map: {df.attrs['map_name']}")
```

## What's In It

200 sessions, 10 agents each, ~30 minutes per session, 128 ticks/second.

| Column | Type | Description |
|--------|------|-------------|
| `tick` | int | Timestep (128/sec) |
| `agent` | int | Anonymized agent ID |
| `team` | int | Team (0/1) |
| `skill` | int | Skill tier |
| `hp` | int | Health (0-100) |
| `x, y, z` | float | 3D position |
| `heading` | float | Horizontal angular velocity (°/tick) |
| `elevation` | float | Vertical angular velocity (°/tick) |
| `abs_yaw` | float | Absolute horizontal angle (reconstructed from anchor + deltas) |
| `abs_pitch` | float | Absolute vertical angle |
| `input_dx` | int | Raw pointing device delta X |
| `input_dy` | int | Raw pointing device delta Y |
| `fwd_move` | int | Forward/backward analog input |
| `left_move` | int | Left/right analog input |
| `action` | int | Primary action (0/1) |
| `zoom` | int | Zoom state (0/1) |
| `forward` | int | Forward key (0/1) |
| `back` | int | Backward key (0/1) |
| `left` | int | Left key (0/1) |
| `right` | int | Right key (0/1) |
| `reload` | int | Reload key (0/1) |
| `use_key` | int | Use/interact key (0/1) |
| `ducking` | int | Crouch state (0/1) |
| `walking` | int | Walk state (0/1) |
| `airborne` | int | In air (0/1) |
| `ammo` | int | Current clip ammo |
| `shots_fired` | int | Consecutive shots in current spray |
| `zoom_lvl` | int | Zoom level (0/1/2) |
| `armor` | int | Armor value |
| `flash` | int | Flash blindness (0-100) |
| `spotted` | int | Detected by opponents (0/1) |
| `bomb_planted` | int | Objective planted (0/1) |
| `defusing` | int | Agent is defusing (0/1) |
| `freeze` | int | Freeze period active (0/1) |
| `aim_punch` | int | Recoil offset (quantized to 0.01°) |

## Viewangle Reconstruction

Absolute viewangles are reconstructed from initial anchor + cumulative deltas:

```python
abs_yaw(t) = yaw₀ + Σ heading(0..t)
abs_pitch(t) = pitch₀ + Σ elevation(0..t)
```

Verified 0.000° max reconstruction error.

The true crosshair position including recoil is:

```python
crosshair_yaw(t) = abs_yaw(t) + aim_punch_yaw(t)
crosshair_pitch(t) = abs_pitch(t) + aim_punch_pitch(t)
```

## Stats

- **Sessions**: 155+
- **Agents per session**: 10
- **Total agent-hours**: 775+
- **Sample rate**: 128 Hz
- **Columns**: 37
- **Compressed size**: ~196 MB
- **Format**: .tard.zst (v2.1)
- **Bytes/row**: 6.7 (compressed binary)

## Requirements

```
pip install pandas numpy zstandard
```

## Loading

```python
from read_v2 import load, info

df = load("data/001.tard.zst")        # → DataFrame, 37 columns
info("data/001.tard.zst")             # → summary without loading

# Access metadata
print(df.attrs["map_name"])           # e.g. "de_mirage"
print(df.attrs["match_id"])
```

## Changelog

- **v2.1**: 37 columns, viewangle anchors, WASD inputs, aim punch, game state, location callouts, map name. 6.7 B/row.
- **v1**: 14 columns, no anchors, no keyboard, no game state. 4.6 B/row.

## CS2 renderer (src/ + scripts/)

Upstreamed 2026-08-14 from the iji-arm-port working repo (source commit
recorded in the upstream commit message). Layout:

- `src/counter_strike_render/` — the renderer: 23-line `gpu_render.py`
  loader executing `stages/` in ORDER in one shared scope; `render_config.py`
  (preset vectors READ from `docs/projects/counter-strike-sft/
  GT_QUALITY_LEVEL_MANIFEST.json`); `packtools/` — data preparation and
  extraction CLIs (datapack dump/reader, model/viewmodel/world extraction,
  forklift-phase authoring), each directly runnable.
- `src/gpu_render_libs/` — camera/adapter libraries: associative-scan
  camera controllers, frustum camera solver (9-ray near-plane, src-space
  occupancy), AC-PSX lock-on adapter (tick_rate is a REQUIRED derived
  parameter), vector HUD, demo camera/timeline export.
- `src/tardigrade_v21.py`, `src/map_identity.py`,
  `src/compatibility_identity.py` — the packaged v22 decoder (frame-perfect
  positions, D1 teleport-hold) and trajectory-fit map identity with refusal
  (the recording's map_name header is a comment, not an identifier — D10;
  the tick rate is DERIVED from the movement plateau, the dataset name's
  128 Hz is wrong, it is 64).
- `scripts/` — `cs_match_three_view_videos.py` (stage/gen/render/eval/pull
  pipeline: 10 players × ego/gow/sm64/ac from one recording),
  `provision_render_node.py` (a render node gets the WHOLE content tree,
  verified by count+bytes; the renderer resolves every input family from
  `--content-root` by fitted map name and REFUSES BY NAME on gaps — no
  waiver exists), REST render server/client/service.

Run: `python src/counter_strike_render/gpu_render.py --content-root
CONTENT --map <fitted> ...` (see --help; 1,700+ documented options).
Deps: torch, nvdiffrast (GPU), imageio_ffmpeg, numpy. Content trees are
built by `provision_render_node.py` from the derived-artifact corpus and
the raw gt_datapack; no game assets are included in this repository.

### Replay-state capture contract

The renderer's fidelity is bounded by the state and source material the
recording preserves. See
[`STATE_RECONSTRUCTION_CAPTURE_CONTRACT.md`](docs/projects/counter-strike-sft/STATE_RECONSTRUCTION_CAPTURE_CONTRACT.md)
for the required raw-demo/build/content provenance, generic entity and physics
journal, animation or evaluated-pose capture, checkpoint/delta protocol, and
acceptance gates. TARD 2.1 does not currently satisfy the authoritative
world-pose or original-client-presentation levels defined there.

### State/action replay dyad

[`STATE_ACTION_REPLAY_DYAD.md`](docs/projects/counter-strike-sft/STATE_ACTION_REPLAY_DYAD.md)
defines and inventories the forward demonstrator: a polite passive ego-input
sidecar plus native demo, persistent hash-chained state-integrator service,
strict state-to-render bridge, repaired headless render service, explicit
legacy-render gap manifests, and reproducible performance/quality gates. The
state engine, bridge, capture tool and service repairs are implemented; the
native-demo normalizer, runnable client capture, evaluated-bone/physics draw
consumers and real GPU benchmark remain explicit gates rather than implied
features.
