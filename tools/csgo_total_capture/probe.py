#!/usr/bin/env python3
"""Build/run an x86_64 host probe proving the module stays inert off-target."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import DEFAULT_BUILD_ROOT, build, command_output, sha256_file  # noqa: E402


def main() -> int:
    plugin, plugin_record = build(DEFAULT_BUILD_ROOT)
    host = DEFAULT_BUILD_ROOT / "total_capture_probe_host"
    source = HERE / "probe_host.cpp"
    command = [
        "xcrun", "clang++", "-std=c++17", "-arch", "x86_64",
        "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        str(source), "-o", str(host),
    ]
    subprocess.run(command, check=True)
    refused_output = DEFAULT_BUILD_ROOT / "probe-must-not-exist.jsonl"
    if refused_output.exists():
        raise RuntimeError(
            f"stale refusal probe output must be removed manually: {refused_output}"
        )
    environment = os.environ.copy()
    environment.update({
        "TARDIGRADE_PROBE_PLUGIN": str(plugin),
        "TARDIGRADE_TOTAL_CAPTURE_CONSENT": "offline-only-v1",
        "TARDIGRADE_TOTAL_CAPTURE_OUTPUT": str(refused_output),
        "TARDIGRADE_CSGO_ROOT": (
            "/Users/mdot/Library/Application Support/Steam/steamapps/common/"
            "Counter-Strike Global Offensive"
        ),
        "TARDIGRADE_CSGO_BUILD_ID": "12426195",
    })
    run_command = [
        "/usr/bin/arch", "-x86_64", str(host),
        "-insecure", "-tardigrade-total-capture-offline",
    ]
    completed = subprocess.run(
        run_command, env=environment, text=True, capture_output=True, check=False
    )
    report = {
        "schema": "tardigrade/csgo-total-capture-capability-probe/v1",
        "claim": (
            "The x86_64 dylib exports only IServerPluginCallbacks004 and remains "
            "inert when hosted by a non-allowlisted executable. This does not "
            "claim the game loaded the plugin."
        ),
        "plugin_build": plugin_record,
        "probe_source_sha256": sha256_file(source),
        "probe_binary_sha256": sha256_file(host),
        "probe_compile_command": command,
        "probe_run_command": run_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "refused_output_absent": not refused_output.exists(),
        "host_architecture": command_output(["file", str(host)]),
    }
    report_path = DEFAULT_BUILD_ROOT / "capability-probe.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if completed.returncode == 0 and not refused_output.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
