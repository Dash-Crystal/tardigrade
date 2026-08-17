#!/usr/bin/env python3
"""Supervise a visible, passive Counter-Strike input + manual demo capture.

This script never starts or controls a game. It compiles the non-seizing Swift HID
observer outside the repository, watches a directory for demos created by the
operator through supported ``record``/``stop`` console commands, and
copies immutable evidence into a new external session directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from variant_config import VARIANT_IDS, get_variant  # noqa: E402


DEFAULT_SESSION_ROOT = Path("/Users/mdot/dox/runs/cs2-capture-sessions")
DEFAULT_CAPTURE_BINARY = Path("/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture")
DEFAULT_APP_MANIFEST = Path.home() / (
    "Library/Application Support/Steam/steamapps/appmanifest_730.acf"
)
SCHEMA = "counter-strike-ego-session-v3"
SAFETY_BOUNDARY = (
    "Passive HID delivery times are host observations, not exact game-engine "
    "consumption times. Demo-tick alignment is estimated from explicit anchors."
)


@dataclasses.dataclass(frozen=True)
class FileObservation:
    size: int
    mtime_ns: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write one JSON document durably, then publish it with an atomic rename."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def observe_demos(directory: Path) -> dict[Path, FileObservation]:
    observations: dict[Path, FileObservation] = {}
    if not directory.is_dir():
        return observations
    for path in sorted(directory.rglob("*.dem")):
        if not path.is_file():
            continue
        stat = path.stat()
        observations[path.resolve()] = FileObservation(stat.st_size, stat.st_mtime_ns)
    return observations


def changed_demos(
    before: dict[Path, FileObservation], after: dict[Path, FileObservation]
) -> list[Path]:
    return sorted(path for path, observation in after.items() if before.get(path) != observation)


def copy_stable_demo(source: Path, destination_dir: Path, attempts: int = 4) -> dict[str, Any]:
    """Copy a demo after checking that its size/mtime did not change during copy."""
    destination_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = destination_dir / source.name
    if destination.exists():
        destination = destination_dir / f"{source.stem}-{sha256_file(source)[:12]}{source.suffix}"
    last_error = ""
    for _ in range(attempts):
        pre = source.stat()
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            post = source.stat()
            if (pre.st_size, pre.st_mtime_ns) != (post.st_size, post.st_mtime_ns):
                last_error = "source changed while it was being copied"
                temporary.unlink(missing_ok=True)
                time.sleep(0.5)
                continue
            digest = sha256_file(temporary)
            os.replace(temporary, destination)
            return {
                "source_path": str(source),
                "stored_path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": digest,
                "source_mtime_ns": post.st_mtime_ns,
                "stable_copy": True,
            }
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"could not copy stable demo {source}: {last_error}")


def parse_appmanifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    text = path.read_text(encoding="utf-8", errors="replace")

    def field(name: str) -> str | None:
        match = re.search(rf'^\s*"{re.escape(name)}"\s+"([^"]*)"', text, re.MULTILINE)
        return match.group(1) if match else None

    installed_depots: dict[str, str] = {}
    # Retain depot/manifest identity without pretending this is a full VDF parser.
    for depot, manifest in re.findall(
        r'"(\d+)"\s*\{\s*"manifest"\s*"(\d+)"', text, re.MULTILINE
    ):
        installed_depots[depot] = manifest
    return {
        "path": str(path),
        "present": True,
        "sha256": sha256_file(path),
        "appid": field("appid"),
        "name": field("name"),
        "build_id": field("buildid"),
        "target_build_id": field("TargetBuildID"),
        "install_dir_name": field("installdir"),
        "installed_depots": installed_depots,
    }


def parse_alignment_anchor(text: str) -> tuple[int, int]:
    pieces = text.split(":", 1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("anchor must be MONOTONIC_NS:DEMO_TICK")
    try:
        monotonic_ns, demo_tick = int(pieces[0]), int(pieces[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("anchor values must be integers") from error
    if monotonic_ns < 0 or demo_tick < 0:
        raise argparse.ArgumentTypeError("anchor values must be non-negative")
    return monotonic_ns, demo_tick


def affine_alignment(anchors: Sequence[tuple[int, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "insufficient_anchors",
        "clock": "received_monotonic_raw_ns",
        "target": "demo_tick",
        "anchors": [
            {"monotonic_raw_ns": x, "demo_tick": y} for x, y in anchors
        ],
        "interpretation": SAFETY_BOUNDARY,
    }
    if len(anchors) < 2:
        return result
    x_center = sum(x for x, _ in anchors) / len(anchors)
    y_center = sum(y for _, y in anchors) / len(anchors)
    variance = sum((x - x_center) ** 2 for x, _ in anchors)
    if variance == 0:
        result["status"] = "degenerate_anchors"
        return result
    slope = sum((x - x_center) * (y - y_center) for x, y in anchors) / variance
    intercept = y_center - slope * x_center
    residuals = [y - (slope * x + intercept) for x, y in anchors]
    result.update(
        {
            "status": "estimated_affine",
            "formula": "demo_tick = slope_ticks_per_ns * monotonic_raw_ns + intercept_ticks",
            "slope_ticks_per_ns": slope,
            "intercept_ticks": intercept,
            "max_abs_residual_ticks": max(abs(value) for value in residuals),
        }
    )
    return result


def command_output(command: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as error:
        return {"command": list(command), "error": str(error)}
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def jsonl_terminal_record(path: Path) -> dict[str, Any] | None:
    """Return the last valid JSON object without loading an hour-long trace."""
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                last = value
    return last


def compile_capture(repo: Path, binary: Path) -> tuple[Path, dict[str, Any]]:
    source = repo / "tools/cs2_ego_capture/Capture.swift"
    source_digest = sha256_file(source)
    binary.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    build_record_path = binary.with_suffix(".build.json")
    if binary.is_file() and build_record_path.is_file():
        try:
            old_record = json.loads(build_record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_record = {}
        if old_record.get("source_sha256") == source_digest:
            old_record.update(
                {
                    "binary_path": str(binary),
                    "binary_sha256": sha256_file(binary),
                    "reused_stable_external_binary": True,
                }
            )
            return binary, old_record
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{binary.name}.build-", dir=binary.parent)
    )
    # Keep the executable basename stable so the linker/code-signing identifier
    # is not a UUID-suffixed staging name that changes on every rebuild.
    temporary = temporary_directory / binary.name
    command = [
        "swiftc",
        "-swift-version",
        "6",
        "-O",
        "-framework",
        "AppKit",
        "-framework",
        "CoreGraphics",
        "-framework",
        "IOKit",
        str(source),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, binary)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_directory.rmdir()
    record = {
        "source_path": str(source),
        "source_sha256": source_digest,
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary),
        "compiler": command_output(["swiftc", "--version"]),
        "compile_command": command,
        "reused_stable_external_binary": False,
    }
    atomic_json(build_record_path, record)
    return binary, record


def detect_client(app_manifest: dict[str, Any], game_variant: str) -> dict[str, Any]:
    variant = get_variant(game_variant)
    steamapps = DEFAULT_APP_MANIFEST.parent
    install_name = app_manifest.get("install_dir_name") or "Counter-Strike Global Offensive"
    root = steamapps / "common" / install_name
    candidates = [root / relative for relative in variant.client_executables_macos]
    found = [str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)]
    return {
        "game_variant": variant.id,
        "content_root": str(root),
        "content_root_present": root.is_dir(),
        "runnable_client_candidates": [str(path) for path in candidates],
        "runnable_client_found": found,
        "game_was_not_launched_by_supervisor": True,
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-variant", choices=VARIANT_IDS, default="cs2",
        help="exact game/client lifecycle to observe (default: cs2)",
    )
    parser.add_argument(
        "--external-client-context",
        help=("required provenance description for CS2 capture initiated from "
              "macOS, where no native CS2 client exists"),
    )
    parser.add_argument("--consent", action="store_true", help="required capture acknowledgement")
    parser.add_argument(
        "--reminder-minutes",
        type=int,
        default=15,
        help="visible logging reminder interval while the bound target is focused",
    )
    parser.add_argument("--device", action="append", required=True, metavar="VID:PID")
    parser.add_argument(
        "--input-backend",
        choices=["hid-value", "hid-report", "hid-dual", "cgevent", "calibration", "auto"],
        default="hid-report",
    )
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--capture-binary", type=Path, default=DEFAULT_CAPTURE_BINARY)
    parser.add_argument("--session-name")
    parser.add_argument("--foreground-name", action="append", default=[])
    parser.add_argument("--foreground-bundle-id", action="append", default=[])
    parser.add_argument("--app-manifest", type=Path, default=DEFAULT_APP_MANIFEST)
    parser.add_argument(
        "--game-build", "--cs2-build", dest="game_build",
        help="manual game-build provenance override when no manifest exists",
    )
    parser.add_argument(
        "--alignment-anchor",
        action="append",
        type=parse_alignment_anchor,
        default=[],
        metavar="MONOTONIC_NS:DEMO_TICK",
    )
    return parser


def safe_session_name(raw: str | None, game_variant: str = "cs2") -> str:
    if raw is None:
        prefix = get_variant(game_variant).id.replace("_", "-")
        return dt.datetime.now(dt.timezone.utc).strftime(
            f"{prefix}-ego-%Y%m%dT%H%M%SZ"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", raw):
        raise ValueError("--session-name must be 1-80 safe filename characters")
    return raw


def run(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    variant = get_variant(args.game_variant)
    if not args.consent:
        raise SystemExit("error: --consent is required; capture must be deliberate and visible")
    if args.game_variant == "csgo_legacy":
        raise SystemExit(
            "error: this supervisor is a passive HID diagnostic and cannot own a "
            "legacy total-capture transaction. Use "
            "tools/csgo_total_capture/launch.py --consent --offline-only "
            "--insecure --map MAP --device VID:PID[:LOCATION]; that coordinator "
            "launches the local match, HID producer, native demo, and state sidecars "
            "without manual console intervention."
        )
    if (args.game_variant == "cs2" and platform.system() == "Darwin"
            and not (args.external_client_context or "").strip()):
        raise SystemExit(
            "error: CS2 has no native macOS client; select --game-variant "
            "csgo_legacy, or provide --external-client-context with the "
            "supported external execution context being observed"
        )
    if not 1 <= args.reminder_minutes <= 1_440:
        raise SystemExit("error: --reminder-minutes must be in 1...1440")
    if not args.demo_dir.is_dir():
        raise SystemExit(f"error: demo directory does not exist: {args.demo_dir}")
    selectors_supplied = bool(args.foreground_name or args.foreground_bundle_id)
    foreground_names = (
        list(args.foreground_name) if selectors_supplied
        else list(variant.foreground_names_macos)
    )
    foreground_bundle_ids = (
        list(args.foreground_bundle_id) if selectors_supplied
        else list(variant.foreground_bundle_ids_macos)
    )
    if not foreground_names and not foreground_bundle_ids:
        raise SystemExit("error: foreground gating is required")

    repo = Path(__file__).resolve().parents[1]
    args.session_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = args.session_root / safe_session_name(args.session_name, variant.id)
    session.mkdir(mode=0o700)
    started_wall_ns = time.time_ns()
    before = observe_demos(args.demo_dir)
    app_manifest = parse_appmanifest(args.app_manifest)
    if args.game_build:
        app_manifest["operator_build_override"] = args.game_build
    binary, build = compile_capture(repo, args.capture_binary.resolve())

    inventory_result = command_output([str(binary), "--list-devices"])
    permission_result = command_output([str(binary), "--diagnose-permissions"])
    atomic_json(session / "device-inventory.json", inventory_result)
    open_record = {
        "schema": SCHEMA,
        "game_variant": variant.id,
        "status": "in_progress",
        "session": session.name,
        "started_wall_unix_ns": started_wall_ns,
        "safety_boundary": SAFETY_BOUNDARY,
        "capture_lifecycle": "first_matching_foreground_pid_until_termination",
        "reminder_interval_minutes": args.reminder_minutes,
        "demo_directory_observed": str(args.demo_dir.resolve()),
        "selected_device_ids": args.device,
    }
    atomic_json(session / "session-open.json", open_record)

    print("\n" + "=" * 72, file=sys.stderr)
    print(f"{variant.display_name.upper()} EGO CAPTURE IS VISIBLE AND WILL RECORD RAW HID USAGE TRANSITIONS", file=sys.stderr)
    print(f"It pauses whenever {variant.display_name} is not the foreground application.", file=sys.stderr)
    print(f"It binds the first matching {variant.display_name} PID and ends when that process exits.", file=sys.stderr)
    print(f"This program will not start, inspect, configure, or send commands to {variant.display_name}.", file=sys.stderr)
    print("This is a passive HID diagnostic, not a total-state recorder.", file=sys.stderr)
    print("It cannot promote a session to total capture or require manual game commands.", file=sys.stderr)
    print("Press Ctrl-C here for a clean early close.", file=sys.stderr)
    print("=" * 72 + "\n", file=sys.stderr)

    events = session / "hid-events.jsonl"
    capture_command = [
        str(binary),
        "--consent",
        "--game-variant",
        variant.id,
        "--output",
        str(events),
        "--session",
        "--reminder-minutes",
        str(args.reminder_minutes),
        "--backend",
        args.input_backend,
    ]
    for selector in args.device:
        capture_command.extend(["--device", selector])
    for name in foreground_names:
        capture_command.extend(["--foreground-name", name])
    for bundle_id in foreground_bundle_ids:
        capture_command.extend(["--foreground-bundle-id", bundle_id])

    interrupted = False
    process = subprocess.Popen(capture_command)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        process.send_signal(signal.SIGINT)
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            returncode = process.wait(timeout=5)

    # Give the explicitly issued `stop` a short window to finish its own write.
    time.sleep(1)
    after = observe_demos(args.demo_dir)
    demo_records: list[dict[str, Any]] = []
    demo_errors: list[str] = []
    for path in changed_demos(before, after):
        try:
            demo_records.append(copy_stable_demo(path, session / "demos"))
        except (OSError, RuntimeError) as error:
            demo_errors.append(str(error))

    status = "complete"
    terminal_record = jsonl_terminal_record(events)
    input_inconclusive = (
        terminal_record is None
        or terminal_record.get("type") != "capture_end"
        or terminal_record.get("observation_status") != "events_observed"
    )
    terminal_reason = terminal_record.get("reason") if terminal_record else None
    operator_stopped = terminal_reason in {"sigint", "sigterm", "operator_stop"}
    if returncode != 0 or demo_errors or not demo_records or input_inconclusive:
        status = "failed"
    elif interrupted or operator_stopped:
        status = "interrupted"
    elif terminal_reason != "target_terminated":
        status = "failed"
    manifest = {
        "schema": SCHEMA,
        "game_variant": variant.id,
        "game": variant.public_descriptor(),
        "status": status,
        "session": session.name,
        "started_wall_unix_ns": started_wall_ns,
        "finished_wall_unix_ns": time.time_ns(),
        "consent": True,
        "capture_policy": {
            "input_observation": args.input_backend,
            "physical_device_attribution_scope": (
                "runtime_dependent"
                if args.input_backend == "auto"
                else "iohid_records_only"
                if args.input_backend == "calibration"
                else "none"
                if args.input_backend == "cgevent"
                else "all_input_records"
            ),
            "foreground_gated": True,
            "capture_lifecycle": "first_matching_foreground_pid_until_termination",
            "reminder_interval_minutes": args.reminder_minutes,
            "records_characters_or_text": False,
            "injects_or_modifies_events": False,
            "reads_or_debugs_game_process": False,
            "loads_code_into_game_process": False,
            "changes_game_configuration": False,
            "sends_game_or_network_commands": False,
            "network_upload": False,
            "duration_limit_seconds": None,
        },
        "limitations": [
            SAFETY_BOUNDARY,
            (
                "IOHID records are limited to explicitly selected devices; CGEvent records are "
                "aggregate and physically unattributed."
            ),
            "The SourceTV/client demo remains authoritative for game state and outcomes.",
            "Exact replay requires proving alignment and handling client prediction separately.",
        ],
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "sw_vers": command_output(["sw_vers"]),
        },
        "client_provenance": {
            "app_manifest": app_manifest,
            "client": detect_client(app_manifest, variant.id),
            "external_client_context": args.external_client_context,
        },
        "capture_build": build,
        "capture_command": capture_command,
        "capture_returncode": returncode,
        "interrupted": interrupted,
        "hid_events": {
            "path": str(events),
            "present": events.is_file(),
            "bytes": events.stat().st_size if events.is_file() else 0,
            "sha256": sha256_file(events) if events.is_file() else None,
            "terminal_record": terminal_record,
            "empirically_inconclusive": input_inconclusive,
        },
        "device_inventory": inventory_result,
        "permission_diagnostics": permission_result,
        "demo_directory_observed": str(args.demo_dir.resolve()),
        "demo_files": demo_records,
        "demo_copy_errors": demo_errors,
        "alignment": affine_alignment(args.alignment_anchor),
    }
    atomic_json(session / "manifest.json", manifest)
    print(f"Finalized {session / 'manifest.json'}", file=sys.stderr)
    if not demo_records:
        print("NOTICE: no externally produced .dem was observed; this remains HID-only.", file=sys.stderr)
    return 0 if status in {"complete", "interrupted"} else 1


if __name__ == "__main__":
    raise SystemExit(run())
