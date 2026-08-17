#!/usr/bin/env python3
"""Write a clearly labelled fixture capture without claiming native hooks."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wire import terminal_summary, write_fixture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session-id", default=f"fixture-{uuid.uuid4().hex}")
    args = parser.parse_args()
    write_fixture(args.output, args.session_id)
    print(json.dumps(terminal_summary(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
