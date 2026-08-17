#!/usr/bin/env python3
"""Validate and atomically publish an engine-neutral total capture."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay.total_capture import TotalCaptureIngestor  # noqa: E402
from state_replay.total_capture_contract import TotalCaptureContractError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("records", type=Path, help="one sealed record JSON object per line")
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--sidecar", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-id", default="total-capture")
    args = parser.parse_args()
    artifacts = {"demo": args.demo}
    try:
        for binding in args.sidecar:
            artifact_id, separator, path = binding.partition("=")
            if not separator or not artifact_id or artifact_id in artifacts:
                raise TotalCaptureContractError(f"invalid/duplicate --sidecar {binding!r}")
            artifacts[artifact_id] = Path(path)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        with args.records.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        summary = TotalCaptureIngestor().ingest(
            manifest, records, artifacts, args.output, args.stream_id)
    except (OSError, json.JSONDecodeError, TotalCaptureContractError, ValueError) as exc:
        print(f"total_capture_ingest: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dataclasses.asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
