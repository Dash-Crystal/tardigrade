#!/usr/bin/env python3
"""Ingest a legacy CS:GO HL2DEMO file into the state replay journal."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from state_replay.source1_adapter import Source1ReplayAdapter  # noqa: E402
from state_replay.source1_demo import Source1DemoError, Source1DemoReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="new state-journal directory (normally under /Users/mdot/dox/runs)")
    parser.add_argument("--stream-id", default="csgo/source1-demo")
    parser.add_argument("--adapter", choices=[Source1ReplayAdapter.ADAPTER],
                        default=Source1ReplayAdapter.ADAPTER)
    parser.add_argument("--tick-rate", type=int,
                        help="override header-derived integer tick rate")
    parser.add_argument("--camera-split-slot", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    try:
        with Source1DemoReader.open(args.demo) as reader:
            summary = Source1ReplayAdapter(
                camera_split_slot=args.camera_split_slot).ingest(
                reader, args.output, args.stream_id, tick_rate_hz=args.tick_rate)
    except (OSError, Source1DemoError, ValueError) as exc:
        print(f"csgo_state_ingest: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dataclasses.asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
