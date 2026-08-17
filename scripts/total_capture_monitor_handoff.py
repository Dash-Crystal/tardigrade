#!/usr/bin/env python3
"""Seal a current outer-monitor diagnostic as salvage-only handoff evidence."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay.total_capture_contract import (  # noqa: E402
    TotalCaptureContractError,
    canonical_json,
)
from state_replay.total_capture_monitor_handoff import (  # noqa: E402
    diagnostic_monitor_handoff,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    try:
        diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
        handoff = diagnostic_monitor_handoff(diagnostic)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(handoff) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, json.JSONDecodeError, TotalCaptureContractError) as exc:
        print(f"total_capture_monitor_handoff: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
