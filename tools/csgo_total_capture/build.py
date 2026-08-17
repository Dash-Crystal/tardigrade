#!/usr/bin/env python3
"""Build the x86_64 server-plugin capture dylib outside the repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "capture_plugin.cpp"
DEFAULT_BUILD_ROOT = Path("/Users/mdot/dox/runs/csgo-total-capture-build/12426195")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build(build_root: Path = DEFAULT_BUILD_ROOT) -> tuple[Path, dict[str, object]]:
    root = build_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if str(root).startswith(str(HERE.parents[1])):
        raise RuntimeError("build artifacts must remain outside the source repository")
    output = root / "libtardigrade_csgo_total_capture.dylib"
    record_path = root / "build.json"
    source_hash = sha256_file(SOURCE)
    compiler = command_output(["xcrun", "clang++", "--version"])
    if output.is_file() and record_path.is_file():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("source_sha256") == source_hash:
            old["binary_sha256"] = sha256_file(output)
            old["reused"] = True
            return output, old
    temporary_root = Path(tempfile.mkdtemp(prefix=".total-capture-build-", dir=root))
    temporary = temporary_root / output.name
    command = [
        "xcrun", "clang++", "-std=c++17", "-arch", "x86_64",
        "-dynamiclib", "-O2", "-fvisibility=hidden", "-Wall", "-Wextra",
        "-Wpedantic", "-Werror", str(SOURCE), "-o", str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_root.rmdir()
    record: dict[str, object] = {
        "schema": "tardigrade/csgo-total-capture-build/v1",
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "binary": str(output),
        "binary_sha256": sha256_file(output),
        "compile_command": command,
        "compiler": compiler,
        "architecture": command_output(["file", str(output)]),
        "reused": False,
    }
    temporary_record = record_path.with_suffix(".json.tmp")
    temporary_record.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_record, record_path)
    return output, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    args = parser.parse_args()
    output, record = build(args.build_root)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
