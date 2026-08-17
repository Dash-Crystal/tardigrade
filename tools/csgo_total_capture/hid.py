"""Build and validate the stable passive HID producer used by the coordinator."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from inventory import sha256_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "tools/cs2_ego_capture/Capture.swift"
DEFAULT_BINARY = Path("/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture")


def _command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_hid(binary: Path = DEFAULT_BINARY) -> tuple[Path, dict[str, Any]]:
    output = binary.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_hash = sha256_file(SOURCE)
    record_path = output.with_suffix(".build.json")
    if output.is_file() and record_path.is_file():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("source_sha256") == source_hash:
            old["binary_sha256"] = sha256_file(output)
            old["reused"] = True
            return output, old
    temporary_root = Path(tempfile.mkdtemp(prefix=".hid-build-", dir=output.parent))
    temporary = temporary_root / output.name
    command = [
        "swiftc", "-swift-version", "6", "-O",
        "-framework", "AppKit", "-framework", "CoreGraphics",
        "-framework", "IOKit", str(SOURCE), "-o", str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_root.rmdir()
    record = {
        "schema": "tardigrade/passive-hid-build/v1",
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "binary": str(output),
        "binary_sha256": sha256_file(output),
        "compiler": _command(["swiftc", "--version"]),
        "compile_command": command,
        "reused": False,
    }
    temporary_record = record_path.with_suffix(".json.tmp")
    temporary_record.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_record, record_path)
    return output, record


def last_json_record(path: Path) -> dict[str, Any] | None:
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


__all__ = ["DEFAULT_BINARY", "build_hid", "last_json_record"]
