# HCI-128

128Hz human-computer interaction telemetry. 1,000 hours of motor control data from adversarial 3D navigation tasks.

## Quick Start

```python
from read import load

df = load("data/0.tard.zst")
print(df.head())
```

## What's In It

200 sessions, 10 agents each, ~30 minutes per session. Every row is one tick (1/128th of a second).

| Column | Type | Description |
|--------|------|-------------|
| `tick` | int | Timestep index (128 per second) |
| `agent` | int | Anonymized agent ID |
| `team` | int | Team (0 or 1) |
| `skill` | int | Skill tier |
| `hp` | int | Health (0-100) |
| `x, y, z` | float | 3D position |
| `heading` | float | Horizontal angular velocity (°/tick) |
| `elevation` | float | Vertical angular velocity (°/tick) |
| `input_dx` | int | Raw pointing device delta X |
| `input_dy` | int | Raw pointing device delta Y |
| `action` | int | Primary action (0/1) |
| `zoom` | int | Zoom/scope state (0/1) |

## Loading

```python
from read import load, load_all, iter_sessions, info

# Single session → DataFrame
df = load("data/001.tard.zst")

# All sessions → single DataFrame with session column
df = load_all("data/")

# Memory-efficient iteration
for session in iter_sessions("data/"):
    train(session)

# File info without loading
info("data/001.tard.zst")
```

## Stats

- **Sessions**: 200
- **Agents per session**: 10
- **Total agent-hours**: 1,000
- **Sample rate**: 128 Hz (7.8ms resolution)
- **Rows**: ~200M
- **Compressed size**: ~220 MB
- **Format**: .tard.zst (zstandard-compressed binary)

## Requirements

```
pip install pandas numpy zstandard
```

## Format

`.tard` is a compact binary format using delta encoding, zigzag varint, and sparse event streams. `read.py` handles decoding — no external dependencies beyond pandas/numpy.

Compression ratio vs JSON: **44x**. The same data as JSONL would be ~50 GB.
