#!/usr/bin/env python3
"""Inventory and validate the installed x86_64 csgo_legacy build."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected_build_12426195.json"
DEFAULT_GAME_ROOT = Path(
    "/Users/mdot/Library/Application Support/Steam/steamapps/common/"
    "Counter-Strike Global Offensive"
)
DEFAULT_MANIFEST = Path(
    "/Users/mdot/Library/Application Support/Steam/steamapps/appmanifest_730.acf"
)


class InventoryError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def manifest_fields(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="strict")

    def field(name: str) -> str | None:
        matches = re.findall(rf'^\s*"{re.escape(name)}"\s+"([^"]*)"', text, re.MULTILINE)
        return matches[-1] if matches else None

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "app_id": field("appid"),
        "build_id": field("buildid"),
        "target_build_id": field("TargetBuildID"),
        "beta_key": field("BetaKey"),
        "state_flags": field("StateFlags"),
    }


def module_inventory(path: Path, expected: dict[str, Any], interfaces: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False, "matches": False}
    file_result = command(["file", str(path)])
    uuid_result = command(["dwarfdump", "--uuid", str(path)])
    nm_result = command(["nm", "-gU", str(path)])
    strings_result = command(["strings", "-a", str(path)])
    strings = set(strings_result["stdout"].splitlines()) if strings_result["returncode"] == 0 else set()
    digest = sha256_file(path)
    size = path.stat().st_size
    uuid_match = re.search(r"UUID: ([0-9A-F-]+) \(x86_64\)", uuid_result["stdout"])
    uuid = uuid_match.group(1) if uuid_match else None
    result = {
        "path": str(path),
        "present": True,
        "size": size,
        "sha256": digest,
        "uuid": uuid,
        "architecture": "x86_64" if "x86_64" in file_result["stdout"] else "unknown",
        "exports_create_interface": "_CreateInterface" in nm_result["stdout"],
        "required_interface_strings": {
            name: name in strings for name in interfaces
        },
    }
    result["matches"] = (
        digest == expected["sha256"]
        and ("size" not in expected or size == expected["size"])
        and ("uuid" not in expected or uuid == expected["uuid"])
        and result["architecture"] == "x86_64"
        and (path.name == "csgo_osx64" or result["exports_create_interface"])
        and all(result["required_interface_strings"].values())
    )
    return result


def inventory(game_root: Path = DEFAULT_GAME_ROOT, manifest: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    root = game_root.expanduser().resolve()
    manifest_data = manifest_fields(manifest.expanduser().resolve())
    modules: dict[str, Any] = {}
    for name, module in expected["modules"].items():
        modules[name] = module_inventory(
            root / module["relative_path"],
            module,
            list(expected["required_interface_strings"].get(name, [])),
        )
    manifest_matches = (
        manifest_data["app_id"] == expected["app_id"]
        and manifest_data["build_id"] == expected["build_id"]
        and manifest_data["target_build_id"] == expected["build_id"]
        and manifest_data["beta_key"] == expected["beta_key"]
        and manifest_data["state_flags"] == "4"
    )
    abi_checks: dict[str, Any] = {}
    for name, anchor in expected.get("abi_code_anchors", {}).items():
        module = expected["modules"][anchor["module"]]
        path = root / module["relative_path"]
        with path.open("rb") as handle:
            handle.seek(anchor["file_offset"])
            data = handle.read(anchor["length"])
        digest = hashlib.sha256(data).hexdigest()
        abi_checks[name] = {
            "module": anchor["module"],
            "file_offset": anchor["file_offset"],
            "length": len(data),
            "sha256": digest,
            "matches": len(data) == anchor["length"] and digest == anchor["sha256"],
        }
    return {
        "schema": "tardigrade/csgo-legacy-binary-inventory/v1",
        "game_root": str(root),
        "expected_build": expected,
        "manifest": manifest_data,
        "manifest_matches": manifest_matches,
        "modules": modules,
        "abi_code_anchors": abi_checks,
        "matches_allowlist": (
            manifest_matches and all(item["matches"] for item in modules.values())
            and all(item["matches"] for item in abi_checks.values())
        ),
    }


def require_allowlisted(report: dict[str, Any]) -> None:
    if not report["matches_allowlist"]:
        mismatches = [name for name, value in report["modules"].items() if not value["matches"]]
        abi_mismatches = [
            name for name, value in report.get("abi_code_anchors", {}).items()
            if not value["matches"]
        ]
        raise InventoryError(
            "installed build is not the exact offline capture allowlist: "
            f"manifest_matches={report['manifest_matches']} modules={mismatches} "
            f"abi_code_anchors={abi_mismatches}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = inventory(args.game_root, args.manifest)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    require_allowlisted(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
