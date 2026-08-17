#!/usr/bin/env python3
"""Atomically ingest an ordered process/map/capture-epoch collection."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay.total_capture_collection import (  # noqa: E402
    CollectionEpochInput,
    TotalCaptureCollectionIngestor,
)
from state_replay.total_capture_contract import TotalCaptureContractError  # noqa: E402


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("terminal", type=Path)
    parser.add_argument("inputs", type=Path,
                        help="JSON object mapping three-axis epoch keys to paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-artifact", action="append", default=[],
                        metavar="ID=PATH", required=True)
    parser.add_argument("--allow-target-not-met", action="store_true")
    args = parser.parse_args()
    try:
        raw_inputs = _json(args.inputs)
        if not isinstance(raw_inputs, dict):
            raise TotalCaptureContractError("inputs file must be an object")
        inputs = {}
        for key, value in raw_inputs.items():
            if not isinstance(value, dict) or set(value) != {
                    "manifest", "records", "salvage", "artifacts"}:
                raise TotalCaptureContractError(
                    f"epoch input {key!r} has wrong fields")
            artifacts = {str(name): Path(path)
                         for name, path in value["artifacts"].items()}
            if value["manifest"] is not None:
                manifest = _json(Path(value["manifest"]))
                with Path(value["records"]).open("r", encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle if line.strip()]
                supplied = CollectionEpochInput(
                    manifest=manifest, records=records, artifact_paths=artifacts)
            else:
                if value["records"] is not None or value["salvage"] is None:
                    raise TotalCaptureContractError(
                        f"salvage epoch input {key!r} is incoherent")
                supplied = CollectionEpochInput(
                    salvage=_json(Path(value["salvage"])),
                    artifact_paths=artifacts)
            inputs[str(key)] = supplied
        outer_artifacts = {}
        for binding in args.outer_artifact:
            artifact_id, separator, path = binding.partition("=")
            if not separator or not artifact_id or artifact_id in outer_artifacts:
                raise TotalCaptureContractError(
                    f"invalid/duplicate --outer-artifact {binding!r}")
            outer_artifacts[artifact_id] = Path(path)
        summary = TotalCaptureCollectionIngestor().ingest(
            _json(args.manifest), _json(args.terminal), inputs, args.output,
            outer_artifact_paths=outer_artifacts,
            allow_target_not_met=args.allow_target_not_met)
    except (OSError, json.JSONDecodeError, TotalCaptureContractError,
            TypeError, ValueError) as exc:
        print(f"total_capture_collection_ingest: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dataclasses.asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
