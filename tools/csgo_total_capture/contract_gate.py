#!/usr/bin/env python3
"""Refuse promotion of incomplete native diagnostics to total-capture records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wire import terminal_summary  # noqa: E402


REQUIRED_FOR_CANONICAL_STREAM = {
    "usercmd": {
        "create_move_human_usercmd", "bot_usercmd_generation",
        "server_usercmd_apply_order",
    },
    "entity_lifecycle": {"entity_lifecycle"},
    "final_pose": {"setup_bones_final_matrices_attachments"},
    "rigid_body": {"vphysics_active_body_state"},
    "effects": {"effects_history"},
    "camera": {"democmdinfo_all_pov_view_fov"},
    "ui": {"panorama_ui_history"},
    "visibility": {"render_history"},
    "render_history": {"render_history"},
}


def refusal(diagnostic: Path) -> dict[str, object]:
    summary = terminal_summary(diagnostic)
    capabilities = summary["capabilities"]
    missing: dict[str, list[str]] = {}
    incomplete: dict[str, list[str]] = {}
    for stream, requirements in REQUIRED_FOR_CANONICAL_STREAM.items():
        missing_names = sorted(requirements - set(capabilities))
        if missing_names:
            missing[stream] = missing_names
        bad = sorted(
            name for name in requirements & set(capabilities)
            if capabilities[name]["status"] != "available"
            or capabilities[name]["completeness"] not in {
                "complete-for-callback-surface",
            }
        )
        # entity_lifecycle's public callback surface explicitly lacks complete
        # baselines/properties, so even its available status cannot satisfy the
        # canonical stream contract.
        if stream == "entity_lifecycle" and requirements <= set(capabilities):
            bad = sorted(set(bad) | requirements)
        if bad:
            incomplete[stream] = bad
    promotable = not missing and not incomplete
    return {
        "schema": "tardigrade/source1-total-capture-promotion-gate/v1",
        "diagnostic": str(diagnostic.resolve()),
        "diagnostic_terminal": summary["terminal"].get("status"),
        "promotable": promotable,
        "missing_capabilities": missing,
        "incomplete_capabilities": incomplete,
        "claim": (
            "No canonical total-capture manifest or records were emitted."
            if not promotable else
            "All static capability gates passed; payload-level canonical "
            "validation is still mandatory before publication."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = refusal(args.diagnostic)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        if args.report.exists():
            raise SystemExit(f"refusing to overwrite {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["promotable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
