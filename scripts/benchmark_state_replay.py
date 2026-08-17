#!/usr/bin/env python3
"""Produce a reproducible CPU state-integrator/bridge throughput report.

This benchmark deliberately does not claim GPU rendering performance. It
measures the state half of the dyad over an hour-shaped synthetic history:
ordered hash-chained ingest, checkpoints, arbitrary seek, batched snapshot
recovery, and conversion through the strict renderer bridge. Artifacts are
always written outside the source checkout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counter_strike_render.replay_bridge import StateToRenderBridge  # noqa: E402
from state_replay import StateIntegrator, state_hash  # noqa: E402


def envelope(value):
    return {"provenance": "recorded", "value": value}


def entity_at(tick: int) -> dict:
    x = float(tick) * 0.25
    yaw = float(tick % 360)
    fired = tick > 0 and tick % 64 == 0
    return {
        "id": "ego:1",
        "generation": 0,
        "class": "CCSPlayerPawn",
        "components": {
            "camera": envelope({
                "x": x,
                "y": 32.0,
                "z": 8.0,
                "eye_z": 64.0,
                "yaw_degrees": yaw,
                "pitch_degrees": 0.0,
                "fov": 90.0,
                "fire": fired,
                "is_alive": True,
                "active_weapon_id": "weapon_ak47",
            }),
            "player": envelope({
                "x": x,
                "y": 32.0,
                "z": 8.0,
                "yaw": yaw,
                "is_alive": True,
                "team": 2,
                "fire": fired,
                "active_weapon_id": "weapon_ak47",
            }),
            "animation": {
                "provenance": "unavailable",
                "reason": "synthetic benchmark does not manufacture pose state",
            },
        },
    }


def transaction(stream_id: str, sequence: int, base_hash: str,
                post_hash: str, tick: int) -> dict:
    state = entity_at(tick)
    ops = [{
        "op": "update",
        "id": state["id"],
        "generation": state["generation"],
        "components": state["components"],
    }]
    if tick % 64 == 0:
        ops.append({
            "op": "action",
            "action_id": f"synthetic/fire/{tick}",
            "kind": "attack",
            "actor": {"id": state["id"], "generation": 0},
            "payload": {"source": "synthetic-benchmark"},
        })
    return {
        "schema": "tardigrade/state-transaction/v1",
        "stream_id": stream_id,
        "sequence": sequence,
        "tick": tick,
        "subtick": 0,
        "base_hash": base_hash,
        "post_hash": post_hash,
        "ops": ops,
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/Users/mdot/dox/runs/state-replay-benchmarks") / stamp,
    )
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--ticks", type=int,
                        help="override hours * tick-rate for a short smoke run")
    parser.add_argument("--tick-rate", type=int, default=128)
    parser.add_argument("--render-fps", type=int, default=32)
    parser.add_argument("--ingest-batch", type=int, default=4096)
    parser.add_argument("--query-batch", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("tick_rate", "render_fps", "ingest_batch", "query_batch"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    total_ticks = args.ticks if args.ticks is not None else int(
        args.hours * 3600 * args.tick_rate)
    if total_ticks < 1:
        raise SystemExit("history must contain at least one tick")
    artifact_root = args.artifact_root.expanduser().resolve()
    if artifact_root == ROOT or ROOT in artifact_root.parents:
        raise SystemExit("--artifact-root must be outside the source checkout")
    artifact_root.mkdir(parents=True, exist_ok=False)

    stream_id = "benchmark/subjective-client"
    initial_entity = entity_at(0)
    initial = StateIntegrator.make_checkpoint(
        stream_id, 0, 0, 0, [initial_entity])
    manifest = {
        "schema": "tardigrade/state-manifest/v1",
        "stream_id": stream_id,
        "tick_rate_hz": args.tick_rate,
        "subtick_unit": "ordered-integer-ordinal",
        "source": {"kind": "synthetic-throughput-fixture"},
    }
    integrator = StateIntegrator.create(artifact_root, manifest, initial)

    generated_seconds = 0.0
    ingested_seconds = 0.0
    base_hash = initial["state_hash"]
    last_post_hash = base_hash
    for begin in range(1, total_ticks + 1, args.ingest_batch):
        end = min(total_ticks + 1, begin + args.ingest_batch)
        started = time.perf_counter()
        batch = []
        for tick in range(begin, end):
            post_hash = state_hash([entity_at(tick)])
            batch.append(transaction(stream_id, tick, base_hash, post_hash, tick))
            base_hash = post_hash
            last_post_hash = post_hash
        generated_seconds += time.perf_counter() - started

        started = time.perf_counter()
        integrator.ingest_many(batch)
        integrator.checkpoint()
        ingested_seconds += time.perf_counter() - started

    stride = max(1, args.tick_rate // args.render_fps)
    positions = [{"tick": tick, "subtick": 0}
                 for tick in range(0, total_ticks + 1, stride)]
    query_seconds = 0.0
    bridge_seconds = 0.0
    bridged_frames = 0
    bridge = StateToRenderBridge("canonical")
    for position_batch in chunks(positions, args.query_batch):
        started = time.perf_counter()
        snapshots = integrator.batch_states(position_batch)
        query_seconds += time.perf_counter() - started
        started = time.perf_counter()
        rendered = bridge.batch(snapshots)
        bridge_seconds += time.perf_counter() - started
        bridged_frames += len(rendered.frames)

    started = time.perf_counter()
    midpoint = integrator.state_at(total_ticks // 2)
    final_seek = integrator.state_at(total_ticks)
    seek_seconds = time.perf_counter() - started
    current = integrator.current_snapshot()
    action_context_ok = True
    last_action_tick = total_ticks - (total_ticks % 64)
    if last_action_tick:
        context = integrator.action_context(f"synthetic/fire/{last_action_tick}")
        action_context_ok = (
            context["pre"]["tick"] == last_action_tick
            and context["post"]["tick"] == last_action_tick
        )

    integrator.close()
    storage_files = sorted(artifact_root.glob(StateIntegrator.DB_NAME + "*"))
    storage_bytes = sum(path.stat().st_size for path in storage_files)
    report = {
        "schema": "tardigrade/state-replay-benchmark/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
        },
        "scope": {
            "state_integrator": True,
            "renderer_bridge": True,
            "gpu_renderer": False,
            "real_cs2_capture": False,
            "claim": (
                "CPU/storage state replay evidence only; this report cannot "
                "establish visual fidelity or GPU rendering throughput."
            ),
        },
        "workload": {
            "ticks": total_ticks,
            "equivalent_hours": total_ticks / args.tick_rate / 3600.0,
            "tick_rate_hz": args.tick_rate,
            "render_fps": args.render_fps,
            "requested_render_positions": len(positions),
            "entities": 1,
            "ingest_batch": args.ingest_batch,
            "query_batch": args.query_batch,
        },
        "measurements": {
            "transaction_generation_seconds": generated_seconds,
            "ingest_checkpoint_seconds": ingested_seconds,
            "ingest_transactions_per_second": total_ticks / ingested_seconds,
            "ingest_realtime_factor": (
                total_ticks / ingested_seconds / args.tick_rate
            ),
            "batch_query_seconds": query_seconds,
            "batch_snapshots_per_second": len(positions) / query_seconds,
            "batch_query_realtime_factor": (
                len(positions) / query_seconds / args.render_fps
            ),
            "bridge_seconds": bridge_seconds,
            "bridge_frames_per_second": bridged_frames / bridge_seconds,
            "bridge_realtime_factor": (
                bridged_frames / bridge_seconds / args.render_fps
            ),
            "two_seek_seconds": seek_seconds,
            "state_store_bytes": storage_bytes,
            "state_store_files": {
                path.name: path.stat().st_size for path in storage_files
            },
        },
        "correctness": {
            "generated_final_hash": last_post_hash,
            "current_hash": current["state_hash"],
            "seek_final_hash": final_seek["state_hash"],
            "final_hashes_equal": (
                last_post_hash == current["state_hash"]
                == final_seek["state_hash"]
            ),
            "midpoint_tick": midpoint["tick"],
            "action_context_ok": action_context_ok,
            "bridged_frames": bridged_frames,
        },
    }
    atomic_json(artifact_root / "benchmark-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all((report["correctness"]["final_hashes_equal"],
                     report["correctness"]["action_context_ok"],
                     bridged_frames == len(positions))) else 2


if __name__ == "__main__":
    raise SystemExit(main())
