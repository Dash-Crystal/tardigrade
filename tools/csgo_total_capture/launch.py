#!/usr/bin/env python3
"""Deprecated one-process compatibility launcher; use persistent monitor.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from build import DEFAULT_BUILD_ROOT, build  # noqa: E402
from inventory import (  # noqa: E402
    DEFAULT_GAME_ROOT,
    DEFAULT_MANIFEST,
    inventory,
    require_allowlisted,
    sha256_file,
)
from hid import DEFAULT_BINARY as DEFAULT_HID_BINARY, build_hid, last_json_record  # noqa: E402
from wire import WireError, read_records, terminal_summary  # noqa: E402
from contract_gate import refusal as promotion_refusal  # noqa: E402
from state_replay.source1_demo import Source1DemoError, Source1DemoReader  # noqa: E402

DEFAULT_SESSION_ROOT = Path("/Users/mdot/dox/runs/csgo-total-capture-sessions")
MAP_RE = re.compile(r"^[a-zA-Z0-9_]+$")


class LaunchRefusal(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ensure_external_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    try:
        root.relative_to(REPO)
    except ValueError:
        pass
    else:
        raise LaunchRefusal("capture output must be outside the source repository")
    if root == Path("/") or root == Path.home().resolve():
        raise LaunchRefusal("capture output root is dangerously broad")
    return root


def stable_copy(source: Path, destination: Path) -> dict[str, Any]:
    before = source.stat()
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        temporary.unlink(missing_ok=True)
        raise LaunchRefusal("native demo changed while being copied")
    os.replace(temporary, destination)
    return {
        "source": str(source),
        "stored": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def has_hl2demo_header(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        return handle.read(8) == b"HL2DEMO\x00"


def validate_demo(path: Path) -> dict[str, Any]:
    with Source1DemoReader.open(path) as reader:
        commands = 0
        last_tick = -1
        for command in reader.commands():
            commands += 1
            last_tick = command.tick
        return {
            "parser": "state_replay.source1_demo.Source1DemoReader",
            "demo_protocol": reader.header.demo_protocol,
            "network_protocol": reader.header.network_protocol,
            "map_name": reader.header.map_name,
            "playback_time": reader.header.playback_time,
            "playback_ticks": reader.header.playback_ticks,
            "commands": commands,
            "terminal_command": "stop",
            "terminal_tick": last_tick,
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--consent", action="store_true", required=True)
    result.add_argument("--offline-only", action="store_true", required=True)
    result.add_argument("--insecure", action="store_true", required=True)
    result.add_argument("--map", required=True)
    result.add_argument("--bot-quota", type=int, default=9)
    result.add_argument("--device", action="append", required=True, metavar="VID:PID[:LOCATION]")
    result.add_argument(
        "--hid-backend",
        choices=["hid-value", "hid-report", "hid-dual", "cgevent", "calibration", "auto"],
        default="hid-report",
    )
    result.add_argument("--reminder-minutes", type=int, default=15)
    result.add_argument("--hid-binary", type=Path, default=DEFAULT_HID_BINARY)
    result.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    result.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--deprecated-one-shot-compatibility", action="store_true",
                        help="explicitly acknowledge this cannot create the required outer session")
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not args.consent or not args.offline_only or not args.insecure:
        raise LaunchRefusal("explicit consent, offline-only, and insecure gates are mandatory")
    if not MAP_RE.fullmatch(args.map):
        raise LaunchRefusal("map must be a simple local map name")
    if isinstance(args.bot_quota, bool) or not 0 <= args.bot_quota <= 31:
        raise LaunchRefusal("bot quota must be in 0..31")
    if not 1 <= args.reminder_minutes <= 1440:
        raise LaunchRefusal("reminder minutes must be in 1..1440")
    session_root = ensure_external_root(args.session_root)
    report = inventory(args.game_root, args.manifest)
    require_allowlisted(report)
    plugin, build_record = build(args.build_root)
    hid_binary, hid_build = build_hid(args.hid_binary)
    return {
        "session_root": session_root,
        "inventory": report,
        "plugin": plugin,
        "build": build_record,
        "hid_binary": hid_binary,
        "hid_build": hid_build,
        "game_root": Path(report["game_root"]),
    }


def run(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.deprecated_one_shot_compatibility:
        print("one-shot launch refused: arm tools/csgo_total_capture/monitor.py instead", file=sys.stderr)
        return 2
    try:
        prepared = prepare(args)
    except (LaunchRefusal, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"total capture refused: {exc}", file=sys.stderr)
        return 2
    session_id = time.strftime("source1-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:12]
    session_root: Path = prepared["session_root"]
    session_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = session_root / session_id
    session.mkdir(mode=0o700)
    native_capture = session / "native-capture-diagnostic.jsonl"
    hid_capture = session / "hid-events.jsonl"
    monitor_capture = session / "monitor-events.jsonl"
    launch_record = session / "launch.json"
    game_root: Path = prepared["game_root"]
    demo_name = f"tardigrade_{session_id}"
    game_demo = game_root / "csgo" / f"{demo_name}.dem"
    if game_demo.exists():
        print(f"total capture refused: demo target already exists: {game_demo}", file=sys.stderr)
        return 2
    executable = game_root / "csgo_osx64"
    command = [
        "/usr/bin/arch", "-x86_64", str(executable),
        "-steam", "-insecure", "-novid", "-windowed",
        "-tardigrade-total-capture-offline",
        "+sv_lan", "1", "+bot_quota", str(args.bot_quota),
        "+plugin_load", str(prepared["plugin"]),
        "+map", args.map,
    ]
    hid_command = [
        str(prepared["hid_binary"]), "--consent",
        "--game-variant", "csgo_legacy", "--backend", args.hid_backend,
        "--session", "--reminder-minutes", str(args.reminder_minutes),
        "--foreground-name", "csgo_osx64",
        "--foreground-name", "Counter-Strike: Global Offensive",
        "--output", str(hid_capture),
    ]
    for device in args.device:
        hid_command.extend(["--device", device])
    environment = os.environ.copy()
    environment.update({
        "TARDIGRADE_TOTAL_CAPTURE_CONSENT": "offline-only-v1",
        "TARDIGRADE_TOTAL_CAPTURE_OUTPUT": str(native_capture),
        "TARDIGRADE_CSGO_ROOT": str(game_root),
        "TARDIGRADE_CSGO_BUILD_ID": "12426195",
        "TARDIGRADE_TOTAL_CAPTURE_SESSION": session_id,
        "TARDIGRADE_DEMO_NAME": demo_name,
    })
    record: dict[str, Any] = {
        "schema": "tardigrade/csgo-total-capture-launch/v1",
        "status": "preflight-complete" if args.dry_run else "running",
        "session_id": session_id,
        "offline_only": True,
        "insecure": True,
        "command": command,
        "environment_contract": sorted(
            key for key in environment if key.startswith("TARDIGRADE_")
        ),
        "inventory": prepared["inventory"],
        "plugin_build": prepared["build"],
        "hid_build": prepared["hid_build"],
        "hid_command": hid_command,
        "native_capture": str(native_capture),
        "native_demo_expected": str(game_demo),
        "safety": {
            "remote_connect_arguments": "not accepted by this launcher",
            "server_remote_clients": "capture plugin rejects non-loopback ClientConnect",
            "vac": "-insecure is mandatory and module revalidates argv",
            "completion": "requires PID-bound HID closure, native advance records, clean terminal, and a fully parsed native demo",
            "transaction": "HID starts before game; plugin owns demo record/stop; monitor joins all outputs",
        },
    }
    atomic_json(launch_record, record)
    if args.dry_run:
        print(launch_record)
        return 0
    monitor_sequence = 0
    with monitor_capture.open("x", encoding="utf-8") as monitor:
        def monitor_event(kind: str, **fields: Any) -> None:
            nonlocal monitor_sequence
            value = {
                "schema": "tardigrade/csgo-capture-monitor/v1",
                "sequence": monitor_sequence,
                "monotonic_ns": time.monotonic_ns(),
                "kind": kind,
                **fields,
            }
            monitor_sequence += 1
            monitor.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            monitor.flush()
            os.fsync(monitor.fileno())

        monitor_event("transaction_start", hid_started_before_game=True)
        hid_process = subprocess.Popen(hid_command, cwd=game_root)
        time.sleep(0.75)
        if hid_process.poll() is not None:
            monitor_event("refused", reason="HID producer exited before game launch",
                          hid_returncode=hid_process.returncode)
            record["hid_returncode"] = hid_process.returncode
            record["status"] = "incomplete"
            atomic_json(session / "salvage.json", {
                "schema": "tardigrade/csgo-capture-salvage/v1",
                "session_id": session_id,
                "status": "not-total-capture",
                "reason": "HID producer refused before game launch",
            })
            atomic_json(launch_record, record)
            return 3
        try:
            game_process = subprocess.Popen(command, cwd=game_root, env=environment)
        except OSError as exc:
            hid_process.terminate()
            hid_process.wait(timeout=10)
            monitor_event("refused", reason="game launch failed", error=str(exc))
            record["status"] = "incomplete"
            record["game_launch_error"] = str(exc)
            atomic_json(session / "salvage.json", {
                "schema": "tardigrade/csgo-capture-salvage/v1",
                "session_id": session_id,
                "status": "not-total-capture",
                "reason": "game launch failed after HID producer start",
            })
            atomic_json(launch_record, record)
            return 3
        monitor_event("game_started", pid=game_process.pid, hid_pid=hid_process.pid)
        game_finished = False
        game_returncode = -1
        hid_returncode = -1
        forced_stop = False
        try:
            prior_sizes: dict[str, int] = {}
            last_health = 0.0
            while game_process.poll() is None:
                if hid_process.poll() is not None:
                    monitor_event("producer_failure", producer="hid",
                                  returncode=hid_process.returncode)
                    game_process.terminate()
                    forced_stop = True
                    break
                sizes = {
                    "hid": hid_capture.stat().st_size if hid_capture.exists() else 0,
                    "native": native_capture.stat().st_size if native_capture.exists() else 0,
                    "demo": game_demo.stat().st_size if game_demo.exists() else 0,
                }
                now = time.monotonic()
                if sizes != prior_sizes or now - last_health >= 10.0:
                    monitor_event("health", **{
                        f"{key}_bytes": value for key, value in sizes.items()
                    })
                    prior_sizes = sizes
                    last_health = now
                time.sleep(0.25)
            try:
                game_returncode = game_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                game_process.kill()
                game_returncode = game_process.wait()
                forced_stop = True
                monitor_event("forced_kill", reason="game ignored transaction-stop termination")
            if hid_process.poll() is None:
                try:
                    hid_returncode = hid_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    hid_process.terminate()
                    hid_returncode = hid_process.wait(timeout=10)
            else:
                hid_returncode = int(hid_process.returncode)
            game_finished = True
            monitor_event("transaction_terminal", game_returncode=game_returncode,
                          hid_returncode=hid_returncode, forced_stop=forced_stop)
        except KeyboardInterrupt:
            forced_stop = True
            monitor_event("transaction_interrupted", reason="operator_interrupt")
        finally:
            if game_process.poll() is None:
                game_process.terminate()
                try:
                    game_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    game_process.kill()
                    game_process.wait()
            if hid_process.poll() is None:
                hid_process.terminate()
                try:
                    hid_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    hid_process.kill()
                    hid_process.wait()
            if not game_finished:
                game_returncode = int(game_process.returncode or -1)
                hid_returncode = int(hid_process.returncode or -1)
                monitor_event("transaction_terminal", game_returncode=game_returncode,
                              hid_returncode=hid_returncode, forced_stop=True,
                              cleanup="child-processes-terminated")
    record["process_returncode"] = game_returncode
    record["game_pid"] = game_process.pid
    record["hid_returncode"] = hid_returncode
    try:
        capture_summary = terminal_summary(native_capture)
        record["capture_summary"] = capture_summary
        native_records = read_records(native_capture)
        native_kind_counts: dict[str, int] = {}
        demo_operations: set[str] = set()
        for item in native_records:
            kind = str(item.get("kind", ""))
            native_kind_counts[kind] = native_kind_counts.get(kind, 0) + 1
            if kind == "demo_lifecycle" and isinstance(item.get("operation"), str):
                demo_operations.add(str(item["operation"]))
        native_required_advance = (
            native_kind_counts.get("client_usercmd", 0) > 0
            and native_kind_counts.get("client_render_view", 0) > 0
            and {"record-request", "stop-request"} <= demo_operations
        )
        record["native_advance"] = {
            "kind_counts": native_kind_counts,
            "demo_operations": sorted(demo_operations),
            "required_observations_present": native_required_advance,
        }
    except (OSError, WireError) as exc:
        record["capture_error"] = str(exc)
        capture_summary = None
        native_required_advance = False
    hid_terminal = last_json_record(hid_capture)
    record["hid_terminal"] = hid_terminal
    if native_capture.is_file():
        record["native_capture_artifact"] = {
            "path": str(native_capture), "bytes": native_capture.stat().st_size,
            "sha256": sha256_file(native_capture),
        }
    if hid_capture.is_file():
        record["hid_artifact"] = {
            "path": str(hid_capture), "bytes": hid_capture.stat().st_size,
            "sha256": sha256_file(hid_capture),
        }
    if monitor_capture.is_file():
        record["monitor_artifact"] = {
            "path": str(monitor_capture), "bytes": monitor_capture.stat().st_size,
            "sha256": sha256_file(monitor_capture),
        }
    if has_hl2demo_header(game_demo):
        record["demo"] = stable_copy(game_demo, session / game_demo.name)
        record["demo"]["header"] = "HL2DEMO\\0"
        try:
            record["demo"]["validation"] = validate_demo(Path(record["demo"]["stored"]))
        except (OSError, Source1DemoError) as exc:
            record["demo_error"] = f"shared Source1 parser refused demo: {exc}"
            record["demo_rejected_artifact"] = record["demo"]
            record.pop("demo")
    else:
        record["demo_error"] = "in-process coordinator produced no valid HL2DEMO file"
    hid_bound_pid = hid_terminal.get("bound_target_pid") if isinstance(hid_terminal, dict) else None
    hid_complete = (
        hid_returncode == 0 and isinstance(hid_terminal, dict)
        and hid_terminal.get("type") == "capture_end"
        and isinstance(hid_bound_pid, int) and not isinstance(hid_bound_pid, bool)
        and hid_bound_pid == game_process.pid
        and hid_terminal.get("reason") == "target_terminated"
    )
    coordinated_closed = (
        game_returncode == 0
        and capture_summary is not None
        and capture_summary["terminal"].get("status") == "clean"
        and native_required_advance
        and hid_complete
        and "demo" in record
    )
    promotion = promotion_refusal(native_capture) if capture_summary is not None else None
    record["promotion_gate"] = promotion
    complete = coordinated_closed and promotion is not None and bool(promotion["promotable"])
    record["coordinated_artifacts_closed"] = coordinated_closed
    record["status"] = (
        "complete" if complete else
        "closed-incomplete-total-capture" if coordinated_closed else
        "incomplete"
    )
    if not complete:
        salvage = {
            "schema": "tardigrade/csgo-capture-salvage/v1",
            "session_id": session_id,
            "status": "not-total-capture",
            "reason": "one or more coordinated producers failed closure",
            "available_artifacts": {
                key: record[key] for key in (
                    "native_capture_artifact", "hid_artifact", "monitor_artifact", "demo"
                ) if key in record
            },
        }
        atomic_json(session / "salvage.json", salvage)
    atomic_json(launch_record, record)
    print(launch_record)
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(run())
