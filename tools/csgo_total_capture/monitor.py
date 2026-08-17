#!/usr/bin/env python3
"""Monitor an outer offline CS:GO capture session across process/map epochs."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from build import DEFAULT_BUILD_ROOT, build  # noqa: E402
from hid import DEFAULT_BINARY as DEFAULT_HID_BINARY, build_hid, last_json_record  # noqa: E402
from inventory import DEFAULT_GAME_ROOT, DEFAULT_MANIFEST, inventory, require_allowlisted, sha256_file  # noqa: E402
from launch import atomic_json, ensure_external_root, stable_copy, validate_demo  # noqa: E402
from wire import WireError, read_records, terminal_summary, validate_record  # noqa: E402
from state_replay.source1_demo import Source1DemoError  # noqa: E402

ACTIVE_ROOT = Path("/Users/mdot/dox/runs/csgo-total-capture-active")
DEFAULT_SESSION_ROOT = Path("/Users/mdot/dox/runs/csgo-total-capture-sessions")


class MonitorRefusal(RuntimeError):
    pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--consent", action="store_true", required=True)
    result.add_argument("--offline-only", action="store_true", required=True)
    result.add_argument("--insecure", action="store_true", required=True)
    result.add_argument("--device", action="append", required=True, metavar="VID:PID[:LOCATION]")
    result.add_argument("--target-active-gameplay-seconds", type=int, default=3600)
    result.add_argument("--close-file", type=Path)
    result.add_argument("--reminder-minutes", type=int, default=15)
    result.add_argument("--hid-backend", default="hid-report",
                        choices=["hid-value", "hid-report", "hid-dual", "cgevent", "calibration", "auto"])
    result.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    result.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    result.add_argument("--hid-binary", type=Path, default=DEFAULT_HID_BINARY)
    result.add_argument("--dry-run", action="store_true")
    return result


def process_snapshot(executable: Path) -> dict[int, dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,command="], text=True, capture_output=True, check=True
    )
    expected = str(executable)
    result: dict[int, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        pieces = line.strip().split(maxsplit=1)
        if len(pieces) != 2 or expected not in pieces[1]:
            continue
        try:
            pid = int(pieces[0])
        except ValueError:
            continue
        command = pieces[1]
        tokens = command.split()
        remote_prefixes = ("+connect=", "-connect=", "+playcast=", "+connect_lobby=")
        result[pid] = {
            "pid": pid,
            "command": command,
            "insecure": "-insecure" in tokens,
            "remote_argument_present": any(
                token in ("+connect", "-connect", "+playcast", "+connect_lobby")
                or token.startswith(remote_prefixes) for token in tokens),
        }
    return result


def write_new(path: Path, content: str, mode: int = 0o600) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def atomic_text(path: Path, content: str, mode: int = 0o600) -> str:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    write_new(temporary, content, mode)
    os.replace(temporary, path)
    return sha256_file(path)


class Activation:
    def __init__(self, game_root: Path, plugin: Path, session: Path,
                 session_id: str, demo_base: str):
        self.addons = game_root / "csgo" / "addons"
        self.vdf = self.addons / "tardigrade_total_capture.vdf"
        self.control = ACTIVE_ROOT / "control.v1"
        self.close_request = ACTIVE_ROOT / "close.request"
        self.lease = ACTIVE_ROOT / "lease.v1"
        self.plugin = plugin
        self.session = session
        self.session_id = session_id
        self.demo_base = demo_base
        self.nonce = uuid.uuid4().hex
        self.created_addons = False
        self.vdf_hash: str | None = None
        self.control_hash: str | None = None
        self.close_hash: str | None = None
        self.lease_hash: str | None = None
        self.recovery: list[dict[str, Any]] = []

    @property
    def vdf_content(self) -> str:
        return '"Plugin"\n{\n\t"file"\t"' + str(self.plugin) + '"\n}\n'

    def recover_stale(self) -> None:
        if not ACTIVE_ROOT.exists():
            return
        if ACTIVE_ROOT.is_symlink() or not ACTIVE_ROOT.is_dir() or ACTIVE_ROOT.stat().st_uid != os.getuid():
            raise MonitorRefusal(f"untrusted active monitor root: {ACTIVE_ROOT}")
        control = ACTIVE_ROOT / "control.v1"
        if not control.is_file() or control.is_symlink() or control.stat().st_uid != os.getuid():
            raise MonitorRefusal(f"untrusted active monitor control exists: {ACTIVE_ROOT}")
        fields: dict[str, str] = {}
        for line in control.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in fields:
                raise MonitorRefusal("malformed stale monitor control")
            fields[key] = value
        if fields.get("schema") != "tardigrade-csgo-outer-monitor-v1":
            raise MonitorRefusal("unrecognized stale monitor control")
        try:
            monitor_pid = int(fields["monitor_pid"])
        except (KeyError, ValueError) as exc:
            raise MonitorRefusal("invalid stale monitor PID") from exc
        try:
            os.kill(monitor_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise MonitorRefusal(f"outer monitor PID {monitor_pid} is still live")
        if self.vdf.is_file() and self.vdf.read_text(encoding="utf-8") != self.vdf_content:
            raise MonitorRefusal("stale VDF does not exactly match this capture plugin")
        quarantine = self.session / "recovered-stale-activation"
        quarantine.mkdir(mode=0o700)
        for path in (self.vdf, control, ACTIVE_ROOT / "lease.v1", ACTIVE_ROOT / "close.request"):
            if path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid():
                destination = quarantine / path.name
                os.replace(path, destination)
                self.recovery.append({"source": str(path), "quarantined": str(destination),
                                      "sha256": sha256_file(destination)})
        try: ACTIVE_ROOT.rmdir()
        except OSError: pass

    def install(self) -> dict[str, Any]:
        self.recover_stale()
        if ACTIVE_ROOT.exists():
            raise MonitorRefusal(f"active monitor control exists: {ACTIVE_ROOT}")
        ACTIVE_ROOT.mkdir(mode=0o700)
        if self.addons.is_symlink():
            ACTIVE_ROOT.rmdir()
            raise MonitorRefusal(f"addons path is a symlink: {self.addons}")
        if not self.addons.exists():
            self.addons.mkdir(mode=0o755)
            self.created_addons = True
        if self.vdf.exists():
            ACTIVE_ROOT.rmdir()
            raise MonitorRefusal(f"auto-load descriptor already exists: {self.vdf}")
        control = "\n".join([
            "schema=tardigrade-csgo-outer-monitor-v1",
            "consent=offline-only-v1",
            "build=12426195",
            f"game_root={self.addons.parents[1]}",
            f"session={self.session_id}",
            f"output_dir={self.session}",
            f"demo_base={self.demo_base}",
            f"monitor_pid={os.getpid()}",
            f"nonce={self.nonce}",
        ]) + "\n"
        try:
            self.control_hash = write_new(self.control, control)
            self.heartbeat()
            self.vdf_hash = write_new(self.vdf, self.vdf_content)
        except BaseException:
            self.remove()
            raise
        return {
            "mechanism": "temporary Source server-plugin VDF auto-load descriptor",
            "vdf": str(self.vdf), "vdf_sha256": self.vdf_hash,
            "control": str(self.control), "control_sha256": self.control_hash,
            "lease": str(self.lease), "lease_sha256": self.lease_hash,
            "monitor_pid": os.getpid(), "nonce": self.nonce,
            "stale_activation_recovery": self.recovery,
            "scope": "new process/server-plugin load only; no live-process injection",
        }

    def remove(self) -> None:
        for path, expected in ((self.vdf, self.vdf_hash), (self.control, self.control_hash),
                               (self.close_request, self.close_hash), (self.lease, self.lease_hash)):
            if expected and path.is_file() and sha256_file(path) == expected:
                path.unlink()
        if ACTIVE_ROOT.is_dir():
            try: ACTIVE_ROOT.rmdir()
            except OSError: pass
        if self.created_addons and self.addons.is_dir():
            try: self.addons.rmdir()
            except OSError: pass

    def request_close(self) -> None:
        self.close_hash = write_new(self.close_request, self.session_id + "\n")

    def heartbeat(self) -> None:
        self.lease_hash = atomic_text(
            self.lease, self.session_id + "\n" + self.nonce + "\n"
        )


def monitor_event(handle: Any, sequence: int, kind: str, **fields: Any) -> int:
    row = {
        "schema": "tardigrade/csgo-outer-monitor-event/v1",
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "wall_unix_ns": time.time_ns(),
        "kind": kind,
        **fields,
    }
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return sequence + 1


def collect_native(session: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    captures: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for path in sorted(session.glob("native-*.jsonl")):
        item: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        try:
            summary = terminal_summary(path)
            item["terminal"] = summary["terminal"]
            for record in read_records(path):
                if record.get("kind") == "demo_lifecycle":
                    segments.append({"native_capture": str(path), **record})
        except (OSError, WireError) as exc:
            item["error"] = str(exc)
        captures.append(item)
    return captures, segments


def collect_demos(game_root: Path, session: Path, demo_base: str,
                  segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = sorted({str(item["name"]) for item in segments
                       if item.get("operation") == "record-request" and item.get("name")})
    result: list[dict[str, Any]] = []
    destination = session / "demos"
    destination.mkdir(mode=0o700, exist_ok=True)
    for name in expected:
        if not name.startswith(demo_base + "_i"):
            result.append({"name": name, "status": "refused-name-outside-session-prefix"})
            continue
        source = game_root / "csgo" / f"{name}.dem"
        if not source.is_file():
            result.append({"name": name, "status": "missing"})
            continue
        try:
            copied = stable_copy(source, destination / source.name)
            copied["validation"] = validate_demo(Path(copied["stored"]))
            copied["name"] = name
            copied["status"] = "closed"
            result.append(copied)
        except (OSError, Source1DemoError, RuntimeError) as exc:
            result.append({"name": name, "status": "rejected", "error": str(exc)})
    return result


def interval_intersection_seconds(
    simulation: list[tuple[int, int]], focus: list[tuple[int, int]],
    lower: int, upper: int,
) -> float:
    total = 0
    focus_index = 0
    for sim_start, sim_end in simulation:
        sim_start, sim_end = max(sim_start, lower), min(sim_end, upper)
        if sim_end <= sim_start: continue
        while focus_index < len(focus) and focus[focus_index][1] <= sim_start:
            focus_index += 1
        index = focus_index
        while index < len(focus) and focus[index][0] < sim_end:
            start = max(sim_start, focus[index][0])
            end = min(sim_end, focus[index][1])
            if end > start: total += end - start
            index += 1
    return total / 1_000_000_000


class IncrementalCoverage:
    """Incrementally joins native simulation, demo closure, and HID focus."""
    def __init__(self, session: Path, game_root: Path, hid_path: Path):
        self.session, self.game_root, self.hid_path = session, game_root, hid_path
        self.offsets: dict[Path, int] = {}
        self.partial: dict[Path, bytes] = {}
        self.pid_by_path: dict[Path, int] = {}
        self.instance_by_path: dict[Path, str] = {}
        self.previous_simulating: dict[Path, int | None] = {}
        self.simulation: dict[int, list[tuple[int, int]]] = {}
        self.segments: dict[str, dict[str, Any]] = {}
        self.focus_open: tuple[int, int] | None = None
        self.focus: dict[int, list[tuple[int, int]]] = {}
        self.demo_signatures: dict[str, tuple[int, int]] = {}
        self.acked_pids: set[int] = set()
        self.hid_timebase: tuple[int, int] | None = None
        self.corruption_errors: list[dict[str, Any]] = []
        self.corrupt_paths: set[Path] = set()
        self.native_ordinals: dict[Path, int | None] = {}
        self.hid_sequence: int | None = None
        self.hid_header_seen = False

    def _new_rows(self, path: Path) -> list[dict[str, Any]]:
        if path in self.corrupt_paths: return []
        if not path.is_file(): return []
        with path.open("rb") as handle:
            handle.seek(self.offsets.get(path, 0))
            data = self.partial.get(path, b"") + handle.read()
            self.offsets[path] = handle.tell()
        lines = data.split(b"\n")
        self.partial[path] = lines.pop()
        rows: list[dict[str, Any]] = []
        for line in lines:
            try: value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.corrupt_paths.add(path)
                self.corruption_errors.append({"path": str(path), "offset": self.offsets[path],
                                               "error": str(exc)})
                break
            if not isinstance(value, dict):
                self.corrupt_paths.add(path)
                self.corruption_errors.append({"path": str(path), "offset": self.offsets[path],
                                               "error": "newline-terminated record is not an object"})
                break
            if path.name.startswith("native-"):
                try:
                    validate_record(value, previous_ordinal=self.native_ordinals.get(path))
                except WireError as exc:
                    self.corrupt_paths.add(path)
                    self.corruption_errors.append({"path": str(path), "offset": self.offsets[path],
                                                   "error": str(exc)})
                    break
                self.native_ordinals[path] = int(value["ordinal"])
            elif path == self.hid_path:
                if not self.hid_header_seen:
                    if (value.get("type") != "capture_header" or
                        value.get("schema") != "counter-strike-ego-hid-v3" or
                        value.get("game_variant") != "csgo_legacy"):
                        self.corrupt_paths.add(path)
                        self.corruption_errors.append({"path": str(path), "offset": self.offsets[path],
                                                       "error": "HID stream does not begin with capture_header"})
                        break
                    self.hid_header_seen = True
                else:
                    sequence = value.get("sequence")
                    expected = 1 if self.hid_sequence is None else self.hid_sequence + 1
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != expected:
                        self.corrupt_paths.add(path)
                        self.corruption_errors.append({"path": str(path), "offset": self.offsets[path],
                                                       "error": f"HID sequence expected {expected}, got {sequence}"})
                        break
                    self.hid_sequence = sequence
            rows.append(value)
        return rows

    def poll(self) -> None:
        for path in sorted(self.session.glob("native-*.jsonl")):
            for row in self._new_rows(path):
                kind = row.get("kind")
                if kind == "session_start":
                    process_id = row.get("process_id")
                    instance = row.get("process_instance_id")
                    if (isinstance(process_id, bool) or not isinstance(process_id, int) or
                        process_id <= 0 or not isinstance(instance, str) or
                        not re.fullmatch(r"[0-9]+_[0-9]+", instance)):
                        self.corrupt_paths.add(path)
                        self.corruption_errors.append({"path": str(path),
                                                       "error": "invalid native process instance identity"})
                        continue
                    self.pid_by_path[path] = process_id
                    self.instance_by_path[path] = instance
                pid = self.pid_by_path.get(path, -1)
                if kind == "game_frame":
                    timestamp = int(row.get("monotonic_ns", 0))
                    previous = self.previous_simulating.get(path)
                    if row.get("simulating") is True:
                        if previous is not None and 0 <= timestamp - previous <= 1_000_000_000:
                            self.simulation.setdefault(pid, []).append((previous, timestamp))
                        self.previous_simulating[path] = timestamp
                    else:
                        self.previous_simulating[path] = None
                elif kind == "demo_lifecycle":
                    name = str(row.get("name", ""))
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                        self.corrupt_paths.add(path)
                        self.corruption_errors.append({"path": str(path),
                                                       "error": "invalid demo segment name"})
                        continue
                    if row.get("operation") == "record-request":
                        self.segments[name] = {
                            "name": name, "pid": pid,
                            "process_instance_id": row.get("process_instance_id"),
                            "native_path": path,
                            "start": int(row.get("monotonic_ns", 0)), "stop": None,
                            "demo_valid": False, "playback_time": None,
                        }
                    elif row.get("operation") == "stop-request" and name in self.segments:
                        self.segments[name]["stop"] = int(row.get("monotonic_ns", 0))
                elif kind == "outer_session_close_ack":
                    self.acked_pids.add(pid)
        for row in self._new_rows(self.hid_path):
            if row.get("type") == "clock_anchor":
                numer, denom = row.get("mach_timebase_numer"), row.get("mach_timebase_denom")
                if isinstance(numer, int) and isinstance(denom, int) and numer > 0 and denom > 0:
                    self.hid_timebase = (numer, denom)
                else:
                    self.corrupt_paths.add(self.hid_path)
                    self.corruption_errors.append({"path": str(self.hid_path),
                                                   "error": "invalid HID mach timebase anchor"})
                continue
            if row.get("type") != "focus_transition": continue
            if self.hid_timebase is None: continue
            numer, denom = self.hid_timebase
            ticks = row.get("received_continuous_ticks")
            pid_value = row.get("pid")
            if (isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0 or
                isinstance(pid_value, bool) or not isinstance(pid_value, int)):
                self.corrupt_paths.add(self.hid_path)
                self.corruption_errors.append({"path": str(self.hid_path),
                                               "error": "invalid HID focus transition"})
                continue
            timestamp = ticks * numer // denom
            if self.focus_open is not None:
                pid, start = self.focus_open
                if timestamp > start: self.focus.setdefault(pid, []).append((start, timestamp))
                self.focus_open = None
            if row.get("eligible") is True and pid_value > 0:
                self.focus_open = (pid_value, timestamp)
        for name, segment in self.segments.items():
            if segment["stop"] is None or segment["demo_valid"]: continue
            demo = self.game_root / "csgo" / f"{name}.dem"
            if not demo.is_file(): continue
            stat = demo.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if self.demo_signatures.get(name) == signature: continue
            self.demo_signatures[name] = signature
            try:
                validation = validate_demo(demo)
            except (OSError, Source1DemoError):
                continue
            segment["demo_valid"] = True
            segment["playback_time"] = validation["playback_time"]

    def measure(self) -> dict[str, Any]:
        if self.hid_path in self.corrupt_paths:
            return {
                "finalized_focused_simulating_seconds": 0.0,
                "provisional_focused_simulating_seconds": 0.0,
                "active_gameplay_seconds": 0.0,
                "finalized_demo_playback_seconds": 0.0,
                "finalized_segments": [],
                "semantics": "refused: HID stream contains newline-terminated corruption",
            }
        now = time.monotonic_ns()
        focus = {pid: list(intervals) for pid, intervals in self.focus.items()}
        if self.focus_open is not None:
            pid, start = self.focus_open
            focus.setdefault(pid, []).append((start, now))
        finalized = provisional = 0.0
        finalized_names: list[str] = []
        for name, segment in self.segments.items():
            pid = int(segment["pid"])
            if segment["native_path"] in self.corrupt_paths: continue
            stop = int(segment["stop"] or now)
            coverage = interval_intersection_seconds(
                self.simulation.get(pid, []), focus.get(pid, []),
                int(segment["start"]), stop,
            )
            if segment["demo_valid"]:
                finalized += coverage
                finalized_names.append(name)
            elif segment["stop"] is None:
                provisional += coverage
        return {
            "finalized_focused_simulating_seconds": finalized,
            "provisional_focused_simulating_seconds": provisional,
            "active_gameplay_seconds": finalized + provisional,
            "finalized_demo_playback_seconds": sum(
                float(segment["playback_time"] or 0) for segment in self.segments.values()
                if segment["demo_valid"]),
            "finalized_segments": sorted(finalized_names),
            "semantics": "intersection of accepted PID, HID-focus, native simulating frames, and valid closed demo horizon",
        }


def run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 60 <= args.target_active_gameplay_seconds <= 14_400:
        raise SystemExit("--target-active-gameplay-seconds must be in 60..14400")
    if not 1 <= args.reminder_minutes <= 1440:
        raise SystemExit("--reminder-minutes must be in 1..1440")
    session_root = ensure_external_root(args.session_root)
    report = inventory(args.game_root, args.manifest)
    require_allowlisted(report)
    plugin, plugin_build = build(args.build_root)
    hid_binary, hid_build = build_hid(args.hid_binary)
    session_id = time.strftime("source1-outer-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:10]
    demo_base = "tardigrade_" + session_id.replace("-", "_")
    session = session_root / session_id
    plan = {
        "schema": "tardigrade/csgo-outer-monitor-plan/v1",
        "session_id": session_id, "session": str(session),
        "target_active_gameplay_seconds": args.target_active_gameplay_seconds,
        "ordinary_process_watch": True, "launches_game": False,
        "plugin_build": plugin_build, "hid_build": hid_build,
        "inventory": report,
        "activation": {
            "vdf": str(Path(report["game_root"]) / "csgo/addons/tardigrade_total_capture.vdf"),
            "control": str(ACTIVE_ROOT / "control.v1"),
            "claim": "structural-only until a consented local process proves auto-load",
        },
    }
    if args.dry_run:
        session.mkdir(parents=True, mode=0o700)
        atomic_json(session / "monitor-plan.json", plan)
        print(session / "monitor-plan.json")
        return 0
    session.mkdir(parents=True, mode=0o700)
    activation = Activation(Path(report["game_root"]), plugin, session, session_id, demo_base)
    hid_path = session / "hid-events.jsonl"
    eligibility_path = session / "hid-eligible-pids.txt"
    write_new(eligibility_path, "")
    events_path = session / "monitor-events.jsonl"
    manifest_path = session / "outer-manifest.json"
    close_file = args.close_file.expanduser().resolve() if args.close_file else session / "CLOSE"
    hid_command = [str(hid_binary), "--consent", "--game-variant", "csgo_legacy",
                   "--backend", args.hid_backend, "--outer-session-owner-pid", str(os.getpid()),
                   "--eligibility-control", str(eligibility_path),
                   "--reminder-minutes", str(args.reminder_minutes),
                   "--foreground-name", "csgo_osx64",
                   "--foreground-name", "Counter-Strike: Global Offensive",
                   "--output", str(hid_path)]
    for device in args.device: hid_command.extend(["--device", device])
    process_epochs: list[dict[str, Any]] = []
    current: dict[int, int] = {}
    hid_process: subprocess.Popen[Any] | None = None
    activation_record: dict[str, Any] | None = None
    terminal_reason = "target_active_gameplay"
    sequence = 0
    coverage = IncrementalCoverage(session, Path(report["game_root"]), hid_path)
    try:
        activation_record = activation.install()
        atomic_json(session / "monitor-plan.json", {**plan, "activation": activation_record})
        hid_process = subprocess.Popen(hid_command)
        with events_path.open("x", encoding="utf-8") as events:
            sequence = monitor_event(events, sequence, "outer_session_start",
                                     activation=activation_record, hid_pid=hid_process.pid)
            while True:
                activation.heartbeat()
                if close_file.exists():
                    terminal_reason = "explicit_close_file"
                    break
                if hid_process.poll() is not None:
                    terminal_reason = "hid_producer_failed"
                    sequence = monitor_event(events, sequence, "producer_failure",
                                             producer="hid", returncode=hid_process.returncode)
                    break
                snapshot = process_snapshot(Path(report["game_root"]) / "csgo_osx64")
                for pid, process in snapshot.items():
                    if pid not in current:
                        epoch = len(process_epochs) + 1
                        current[pid] = epoch
                        accepted = process["insecure"] and not process["remote_argument_present"]
                        row = {"process_epoch": epoch, **process, "accepted_offline": accepted,
                               "start_monotonic_ns": time.monotonic_ns()}
                        process_epochs.append(row)
                        sequence = monitor_event(events, sequence, "process_start", **row)
                for pid in sorted(set(current) - set(snapshot)):
                    epoch = current.pop(pid)
                    process_epochs[epoch - 1]["end_monotonic_ns"] = time.monotonic_ns()
                    sequence = monitor_event(events, sequence, "process_end", pid=pid,
                                             process_epoch=epoch)
                eligible_pids = sorted(pid for pid, process in snapshot.items()
                                       if process["insecure"] and not process["remote_argument_present"])
                atomic_text(eligibility_path, "".join(
                    f"{pid}:{current[pid]}\n" for pid in eligible_pids
                ))
                coverage.poll()
                gameplay = coverage.measure()
                if coverage.corruption_errors:
                    terminal_reason = "producer_stream_corruption"
                    break
                if gameplay["finalized_focused_simulating_seconds"] >= args.target_active_gameplay_seconds:
                    terminal_reason = "target_active_gameplay_reached"
                    break
                if int(time.monotonic()) % 10 == 0:
                    sequence = monitor_event(events, sequence, "health",
                                             active_process_epochs=sorted(current.values()),
                                             active_gameplay=gameplay)
                time.sleep(1)
            sequence = monitor_event(events, sequence, "outer_session_close_requested",
                                     reason=terminal_reason)
            activation.request_close()
            ack_pids = {pid for pid, epoch in current.items()
                        if process_epochs[epoch - 1]["accepted_offline"]}
            deadline = time.monotonic() + 15
            while ack_pids and time.monotonic() < deadline:
                activation.heartbeat()
                coverage.poll()
                if ack_pids <= coverage.acked_pids:
                    sequence = monitor_event(events, sequence, "outer_session_close_ack",
                                             active_pids=sorted(ack_pids))
                    break
                time.sleep(0.25)
            else:
                if ack_pids:
                    terminal_reason += "-native-close-unacknowledged"
    except KeyboardInterrupt:
        terminal_reason = "operator_interrupt"
        try:
            activation.request_close()
            ack_pids = {pid for pid, epoch in current.items()
                        if process_epochs[epoch - 1]["accepted_offline"]}
            deadline = time.monotonic() + 15
            while ack_pids and time.monotonic() < deadline:
                activation.heartbeat()
                coverage.poll()
                if ack_pids <= coverage.acked_pids: break
                time.sleep(0.25)
            coverage.poll()
            if ack_pids and not ack_pids <= coverage.acked_pids:
                terminal_reason += "-native-close-unacknowledged"
        except OSError as exc:
            terminal_reason += f"-close-request-failed-{type(exc).__name__}"
    except (MonitorRefusal, OSError, subprocess.SubprocessError) as exc:
        terminal_reason = f"activation-or-monitor-refused:{exc}"
    finally:
        activation.remove()
        if hid_process is not None and hid_process.poll() is None:
            hid_process.send_signal(signal.SIGTERM)
            try: hid_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                hid_process.kill(); hid_process.wait()
    captures, segments = collect_native(session)
    demos = collect_demos(Path(report["game_root"]), session, demo_base, segments)
    coverage.poll()
    gameplay = coverage.measure()
    finalized_target_met = (
        terminal_reason == "target_active_gameplay_reached"
        and gameplay["finalized_focused_simulating_seconds"] >= args.target_active_gameplay_seconds
    )
    manifest = {
        **plan, "status": "closed-incomplete-total-capture",
        "terminal_reason": terminal_reason,
        "active_gameplay": gameplay,
        "producer_corruption_errors": coverage.corruption_errors,
        "target_met": finalized_target_met,
        "activation": activation_record,
        "process_epochs": process_epochs,
        "native_captures": captures,
        "demo_segments": demos,
        "segment_boundaries": segments,
        "hid": {
            "path": str(hid_path),
            "sha256": sha256_file(hid_path) if hid_path.is_file() else None,
            "terminal": last_json_record(hid_path),
            "join": "HID process_epoch+monotonic clocks join monitor process epochs and native map/segment epochs",
        },
        "monitor_events": {"path": str(events_path),
                           "sha256": sha256_file(events_path) if events_path.is_file() else None},
        "claim": "not total capture; auto-activation and hook invocation remain runtime-unproven unless records exist",
    }
    atomic_json(manifest_path, manifest)
    print(manifest_path)
    return 3


if __name__ == "__main__":
    raise SystemExit(run())
