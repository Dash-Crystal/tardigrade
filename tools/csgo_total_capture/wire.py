"""Validation and fixture support for native capture diagnostic records.

This deliberately is not the canonical total-capture record schema.  A build
whose required capture faculties are unavailable must never manufacture an
integrator-ingestable total-capture manifest.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "tardigrade/source1-native-capture-diagnostic/v1"
CAPABILITIES = (
    "server_plugin_callbacks",
    "server_frame_boundary",
    "entity_lifecycle",
    "client_console_command_boundary",
    "native_demo_coordination",
    "ego_create_move_usercmd_viewangles",
    "observed_render_view_matrices",
    "create_move_human_usercmd",
    "bot_usercmd_generation",
    "server_usercmd_apply_order",
    "democmdinfo_all_pov_view_fov",
    "setup_bones_final_matrices_attachments",
    "vphysics_active_body_state",
    "effects_history",
    "panorama_ui_history",
    "render_history",
)
CAPABILITY_STATUSES = frozenset(
    {"available", "unavailable-hard-refused", "fixture-only", "probe-only"}
)


class WireError(ValueError):
    pass


def validate_record(record: Mapping[str, Any], *, previous_ordinal: int | None = None) -> None:
    if record.get("schema") != SCHEMA:
        raise WireError("wrong total-capture schema")
    ordinal = record.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise WireError("ordinal must be an integer >= 0")
    if previous_ordinal is not None and ordinal != previous_ordinal + 1:
        raise WireError("record ordinals must be contiguous")
    monotonic_ns = record.get("monotonic_ns")
    if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
        raise WireError("monotonic_ns must be an integer >= 0")
    kind = record.get("kind")
    if not isinstance(kind, str) or not kind:
        raise WireError("kind must be nonempty text")
    if kind == "capability":
        if record.get("capability") not in CAPABILITIES:
            raise WireError("unknown capability")
        if record.get("status") not in CAPABILITY_STATUSES:
            raise WireError("unknown capability status")
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise WireError("capability records require a reason")
        completeness = record.get("completeness")
        if completeness not in {
            "complete-for-callback-surface", "boundary-only",
            "lifecycle-only-no-state", "metadata-only", "none", "fixture-only",
            "authoritative-ego-command", "observed-client-render-calls",
        }:
            raise WireError("capability records require explicit completeness")
    if kind == "terminal":
        if record.get("status") not in {"clean", "refused", "incomplete"}:
            raise WireError("invalid terminal status")
        drops = record.get("drop_counters")
        if not isinstance(drops, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in drops.values()
        ):
            raise WireError("terminal drop_counters must be nonnegative integers")


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WireError(f"line {line_number} is not JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise WireError(f"line {line_number} is not an object")
            validate_record(value, previous_ordinal=previous)
            previous = value["ordinal"]
            records.append(value)
    if not records:
        raise WireError("capture contains no records")
    return records


def terminal_summary(path: Path) -> dict[str, Any]:
    records = read_records(path)
    if records[0].get("kind") != "session_start":
        raise WireError("capture must begin with session_start")
    terminal = records[-1]
    if terminal.get("kind") != "terminal":
        raise WireError("capture lacks a terminal record")
    terminal_count = sum(record.get("kind") == "terminal" for record in records)
    if terminal_count != 1:
        raise WireError("capture must contain exactly one terminal record")
    capabilities: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("kind") != "capability":
            continue
        name = record["capability"]
        if name in capabilities:
            raise WireError(f"duplicate capability record: {name}")
        capabilities[name] = record
    missing = sorted(set(CAPABILITIES) - set(capabilities))
    if missing:
        raise WireError(f"capture omits capability records: {missing}")
    return {
        "records": len(records),
        "terminal": terminal,
        "capabilities": capabilities,
    }


def fixture_records(session_id: str) -> Iterable[dict[str, Any]]:
    ordinal = 0

    def record(kind: str, **fields: Any) -> dict[str, Any]:
        nonlocal ordinal
        result = {
            "schema": SCHEMA,
            "ordinal": ordinal,
            "monotonic_ns": time.monotonic_ns(),
            "kind": kind,
            **fields,
        }
        ordinal += 1
        return result

    yield record(
        "session_start",
        session_id=session_id,
        evidence="fixture",
        offline_only=True,
        insecure=True,
        build_id="12426195",
    )
    for capability in CAPABILITIES:
        yield record(
            "capability",
            capability=capability,
            status="fixture-only",
            reason="schema fixture; no game process or engine boundary was observed",
            completeness="fixture-only",
        )
    yield record(
        "terminal",
        status="refused",
        evidence="fixture",
        reason="fixture is not native total-capture evidence",
        drop_counters={"writer_failures": 0},
    )


def write_fixture(path: Path, session_id: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise WireError(f"refusing to overwrite {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for record in fixture_records(session_id):
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
