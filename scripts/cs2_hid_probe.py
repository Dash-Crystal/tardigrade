#!/usr/bin/env python3
"""Run a short, visible empirical probe of passive macOS input backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = REPO / "scripts/cs2_capture_session.py"
SPEC = importlib.util.spec_from_file_location("cs2_capture_session_probe_support", SUPERVISOR_PATH)
assert SPEC and SPEC.loader
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)

DEFAULT_PROBE_ROOT = Path("/Users/mdot/dox/runs/cs2-hid-probes")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def interval_summary(timestamps_ns: Sequence[float]) -> dict[str, Any]:
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps_ns, timestamps_ns[1:])
        if later > earlier
    ]
    if not intervals:
        return {"sample_count": 0}
    median_ns = statistics.median(intervals)
    return {
        "sample_count": len(intervals),
        "minimum_ns": min(intervals),
        "p50_ns": median_ns,
        "p95_ns": percentile(intervals, 0.95),
        "maximum_ns": max(intervals),
        "median_rate_hz": 1_000_000_000 / median_ns if median_ns else None,
    }


def analyze_jsonl(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    parse_errors = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
    counts = Counter(record.get("type", "missing_type") for record in records)
    anchor = next((record for record in records if record.get("type") == "clock_anchor"), {})
    numer = int(anchor.get("mach_timebase_numer", 1))
    denom = int(anchor.get("mach_timebase_denom", 1)) or 1
    timestamp_fields = {
        "hid_value": "hid_timestamp_ticks",
        "hid_report": "hid_report_timestamp_ticks",
        "cg_event": "cg_event_timestamp_ns_since_startup",
    }
    intervals: dict[str, Any] = {}
    callback_lag: dict[str, Any] = {}
    first_event_timestamps: dict[str, Any] = {}
    for backend, field in timestamp_fields.items():
        selected = [record for record in records if record.get("type") == backend]
        if backend == "cg_event":
            timestamps = [float(record[field]) for record in selected if field in record]
        else:
            timestamps = [float(record[field]) * numer / denom for record in selected if field in record]
            lags = [
                (float(record["received_absolute_ticks"]) - float(record[field])) * numer / denom
                for record in selected
                if field in record and "received_absolute_ticks" in record
            ]
            nonnegative = [lag for lag in lags if lag >= 0]
            callback_lag[backend] = {
                "samples": len(lags),
                "negative_samples": len(lags) - len(nonnegative),
                "minimum_ns": min(nonnegative) if nonnegative else None,
                "p50_ns": percentile(nonnegative, 0.5),
                "p95_ns": percentile(nonnegative, 0.95),
                "maximum_ns": max(nonnegative) if nonnegative else None,
                "meaning": "callback observation minus public HID arrival/value timestamp; not engine latency",
            }
        intervals[backend] = interval_summary(timestamps)
        first_event_timestamps[backend] = timestamps[0] if timestamps else None

    usages = Counter(
        (record.get("usage_page"), record.get("usage"))
        for record in records
        if record.get("type") == "hid_value"
    )
    end = next((record for record in reversed(records) if record.get("type") == "capture_end"), {})
    report_shapes = Counter(
        (record.get("report_id"), record.get("report_length"))
        for record in records
        if record.get("type") == "hid_report"
    )
    observed_sources = [source for source in timestamp_fields if counts[source] > 0]
    terminal_status = end.get("observation_status")
    empirical_status = (
        terminal_status
        if terminal_status in {"events_observed", "inconclusive_zero_events"}
        else ("events_observed" if observed_sources else "inconclusive_zero_events")
    )
    return {
        "parse_errors": parse_errors,
        "record_counts": dict(sorted(counts.items())),
        "source_intervals": intervals,
        "callback_observation_lag": callback_lag,
        "first_event_source_timestamps": first_event_timestamps,
        "hid_report_shapes": [
            {"report_id": shape[0], "report_length": shape[1], "count": count}
            for shape, count in sorted(report_shapes.items(), key=lambda item: str(item[0]))
        ],
        "hid_value_usages": [
            {"usage_page": pair[0], "usage": pair[1], "count": count}
            for pair, count in sorted(usages.items(), key=lambda item: str(item[0]))
        ],
        "capture_end": end,
        "empirical_status": empirical_status,
        "cross_backend_policy": (
            "Streams are parallel observations at different layers. Never sum them as actions; "
            "retain backend identity and calibrate correspondence empirically."
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--consent", action="store_true", required=True)
    result.add_argument("--device", action="append", required=True, metavar="VID:PID[:LOCATION]")
    result.add_argument("--foreground-name", action="append", required=True)
    result.add_argument("--duration-seconds", type=int, default=10)
    result.add_argument(
        "--backend",
        choices=["hid-value", "hid-report", "hid-dual", "cgevent", "calibration", "auto"],
        default="calibration",
    )
    result.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE_ROOT)
    result.add_argument("--capture-binary", type=Path, default=support.DEFAULT_CAPTURE_BINARY)
    return result


def run(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.duration_seconds <= 30:
        raise SystemExit("error: probe duration must be in 1...30 seconds")
    args.probe_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    probe = args.probe_root / time.strftime("probe-%Y%m%dT%H%M%S")
    probe.mkdir(mode=0o700)
    binary, build = support.compile_capture(REPO, args.capture_binary.resolve())
    diagnostics = support.command_output([str(binary), "--diagnose-permissions"])
    inventory = support.command_output([str(binary), "--list-devices"])
    events = probe / "events.jsonl"
    command = [
        str(binary), "--consent", "--backend", args.backend,
        "--duration-seconds", str(args.duration_seconds), "--output", str(events),
    ]
    for device in args.device:
        command.extend(["--device", device])
    for foreground in args.foreground_name:
        command.extend(["--foreground-name", foreground])
    print(
        "VISIBLE INPUT PROBE: move the selected pointing device, click each button, "
        "and scroll while the named application remains foreground. No events are modified.",
        file=sys.stderr,
    )
    completed = subprocess.run(command, check=False)
    analysis = analyze_jsonl(events) if events.is_file() else {
        "record_counts": {},
        "empirical_status": "inconclusive_backend_start_failed",
        "capture_file_present": False,
    }
    report = {
        "schema": "cs2-hid-probe-v1",
        "capture_returncode": completed.returncode,
        "command": command,
        "capture_build": build,
        "permission_diagnostics": diagnostics,
        "device_inventory": inventory,
        "analysis": analysis,
        "events_path": str(events),
        "engine_timing_claim": (
            "None. Public host timestamps bound device/report and WindowServer observation; "
            "they do not reveal CS2 input-thread sampling or user-command construction."
        ),
    }
    support.atomic_json(probe / "probe-report.json", report)
    print(json.dumps(report["analysis"], indent=2, sort_keys=True))
    print(f"Probe report: {probe / 'probe-report.json'}", file=sys.stderr)
    if completed.returncode != 0:
        return 1
    return 0 if analysis.get("empirical_status") == "events_observed" else 3


if __name__ == "__main__":
    raise SystemExit(run())
