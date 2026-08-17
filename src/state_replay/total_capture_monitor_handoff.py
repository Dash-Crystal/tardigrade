"""Honest handoff from the current diagnostic monitor to collection ingestion."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Mapping

from .source1_demo import Source1DemoReader
from .total_capture import verify_artifacts
from .total_capture_collection import CollectionEpochInput
from .total_capture_collection_contract import epoch_key
from .total_capture_contract import (
    TotalCaptureContractError,
    document_sha256,
    seal_document,
    validate_manifest,
    validate_record,
)


MONITOR_DIAGNOSTIC_SCHEMA = "tardigrade/csgo-outer-monitor-plan/v1"
MONITOR_HANDOFF_SCHEMA = "tardigrade/total-capture-monitor-handoff/v1"


@dataclass(frozen=True)
class PromotedEpochInput:
    key: str
    descriptor: dict[str, object]
    supplied: CollectionEpochInput


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TotalCaptureContractError(f"{label} must be nonempty text")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TotalCaptureContractError(f"{label} must be an integer >= 0")
    return value


def _hash(value: object, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise TotalCaptureContractError(f"{label} must be lowercase sha256")
    return digest


def diagnostic_monitor_handoff(document: Mapping[str, object]) -> dict[str, object]:
    """Convert monitor diagnostics only to sealed salvage/handoff evidence."""
    if not isinstance(document, Mapping) or document.get("schema") != MONITOR_DIAGNOSTIC_SCHEMA:
        raise TotalCaptureContractError("not a current outer-monitor diagnostic")
    session_id = _text(document.get("session_id"), "diagnostic session_id")
    if document.get("status") != "closed-incomplete-total-capture":
        raise TotalCaptureContractError("diagnostic did not close as incomplete total capture")
    claim = _text(document.get("claim"), "diagnostic claim")
    if "not total capture" not in claim:
        raise TotalCaptureContractError("diagnostic does not explicitly disclaim completeness")
    boundaries = document.get("segment_boundaries")
    demos = document.get("demo_segments")
    if not isinstance(boundaries, list) or not isinstance(demos, list):
        raise TotalCaptureContractError("diagnostic segment evidence is malformed")
    demos_by_name = {str(item.get("name")): item for item in demos
                     if isinstance(item, Mapping) and item.get("name")}
    grouped: dict[str, dict[str, object]] = {}
    for raw in boundaries:
        if not isinstance(raw, Mapping) or raw.get("kind") != "demo_lifecycle":
            continue
        name = _text(raw.get("name"), "diagnostic demo name")
        item = grouped.setdefault(name, {
            "name": name, "process_instance_id": raw.get("process_instance_id"),
            "map_epoch": raw.get("map_epoch"), "segment_epoch": raw.get("segment_epoch"),
            "map_name": raw.get("map"), "start_monotonic_ns": None,
            "end_monotonic_ns": None,
        })
        if any(item[field] != raw.get(source) for field, source in (
                ("process_instance_id", "process_instance_id"),
                ("map_epoch", "map_epoch"), ("segment_epoch", "segment_epoch"),
                ("map_name", "map"))):
            raise TotalCaptureContractError("diagnostic segment identity changed mid-segment")
        operation = raw.get("operation")
        if operation == "record-request":
            if item["start_monotonic_ns"] is not None:
                raise TotalCaptureContractError("duplicate diagnostic segment start")
            item["start_monotonic_ns"] = _integer(raw.get("monotonic_ns"), "segment start")
        elif operation == "stop-request":
            if item["end_monotonic_ns"] is not None:
                raise TotalCaptureContractError("duplicate diagnostic segment stop")
            item["end_monotonic_ns"] = _integer(raw.get("monotonic_ns"), "segment stop")
    segments: list[dict[str, object]] = []
    for name in sorted(grouped):
        item = grouped[name]
        process_instance = _text(item["process_instance_id"], "process instance")
        map_epoch = _integer(item["map_epoch"], "map epoch")
        capture_epoch = _integer(item["segment_epoch"], "segment epoch")
        map_name = _text(item["map_name"], "segment map")
        demo = demos_by_name.get(name, {})
        demo_status = str(demo.get("status", "missing"))
        demo_hash = demo.get("sha256") if demo_status == "closed" else None
        if demo_hash is not None:
            _hash(demo_hash, "diagnostic demo hash")
        segments.append({
            "process_epoch_id": f"process:{process_instance}",
            "map_epoch_id": f"map:{process_instance}:{map_epoch}",
            "capture_epoch_id": f"capture:{process_instance}:{capture_epoch}",
            "map_name": map_name, "demo_name": name,
            "start_monotonic_ns": item["start_monotonic_ns"],
            "end_monotonic_ns": item["end_monotonic_ns"],
            "diagnostic_demo_status": demo_status,
            "native_demo_sha256": demo_hash,
            "promotion_status": "blocked-missing-total-faculties",
        })
    artifact_hashes: dict[str, str] = {}
    for index, item in enumerate(document.get("native_captures", [])):
        if isinstance(item, Mapping) and isinstance(item.get("sha256"), str):
            artifact_hashes[f"native:{index}"] = str(item["sha256"])
    for item in demos:
        if isinstance(item, Mapping) and isinstance(item.get("sha256"), str):
            artifact_hashes[f"demo:{item.get('name', len(artifact_hashes))}"] = str(
                item["sha256"])
    for label in ("hid", "monitor_events"):
        item = document.get(label)
        if isinstance(item, Mapping) and isinstance(item.get("sha256"), str):
            artifact_hashes[label] = str(item["sha256"])
    return seal_document({
        "schema": MONITOR_HANDOFF_SCHEMA, "session_id": session_id,
        "status": "salvage-only", "reason": claim,
        "terminal_reason": _text(document.get("terminal_reason"),
                                 "diagnostic terminal_reason"),
        "segments": segments, "artifact_sha256s": artifact_hashes,
    }, "handoff_sha256")


def validate_monitor_handoff(document: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(document, Mapping) or set(document) != {
            "schema", "session_id", "status", "reason", "terminal_reason",
            "segments", "artifact_sha256s", "handoff_sha256"}:
        raise TotalCaptureContractError("monitor handoff fields are not exact")
    if document.get("schema") != MONITOR_HANDOFF_SCHEMA or document.get(
            "status") != "salvage-only":
        raise TotalCaptureContractError("monitor handoff is not salvage-only")
    _text(document.get("session_id"), "handoff session_id")
    _text(document.get("reason"), "handoff reason")
    _text(document.get("terminal_reason"), "handoff terminal_reason")
    if not isinstance(document.get("segments"), list) or not isinstance(
            document.get("artifact_sha256s"), Mapping):
        raise TotalCaptureContractError("monitor handoff arrays are malformed")
    seen: set[str] = set()
    for segment in document["segments"]:
        if not isinstance(segment, Mapping) or set(segment) != {
                "process_epoch_id", "map_epoch_id", "capture_epoch_id", "map_name",
                "demo_name", "start_monotonic_ns", "end_monotonic_ns",
                "diagnostic_demo_status", "native_demo_sha256",
                "promotion_status"}:
            raise TotalCaptureContractError("monitor handoff segment fields are not exact")
        key = epoch_key(_text(segment.get("process_epoch_id"), "process epoch"),
                        _text(segment.get("map_epoch_id"), "map epoch"),
                        _text(segment.get("capture_epoch_id"), "capture epoch"))
        if key in seen:
            raise TotalCaptureContractError("duplicate monitor handoff segment")
        seen.add(key)
        _text(segment.get("map_name"), "segment map_name")
        _text(segment.get("demo_name"), "segment demo_name")
        start = segment.get("start_monotonic_ns")
        end = segment.get("end_monotonic_ns")
        if start is not None:
            start = _integer(start, "segment start")
        if end is not None:
            end = _integer(end, "segment end")
        if start is not None and end is not None and end <= start:
            raise TotalCaptureContractError("monitor segment does not advance")
        if segment.get("promotion_status") != "blocked-missing-total-faculties":
            raise TotalCaptureContractError("diagnostic segment cannot claim promotion")
        if segment.get("native_demo_sha256") is not None:
            _hash(segment.get("native_demo_sha256"), "segment demo sha256")
    for artifact_id, digest in document["artifact_sha256s"].items():
        _text(artifact_id, "handoff artifact id")
        _hash(digest, "handoff artifact sha256")
    if document.get("handoff_sha256") != document_sha256(document, "handoff_sha256"):
        raise TotalCaptureContractError("monitor handoff sha256 mismatch")
    return copy.deepcopy(dict(document))


def promote_verified_total_epoch(
        *, ordinal: int, process_epoch_id: str, map_epoch_id: str,
        capture_epoch_id: str, host_horizon: Mapping[str, object],
        reset: Mapping[str, object], manifest: Mapping[str, object],
        records: Iterable[Mapping[str, object]],
        artifact_paths: Mapping[str, object]) -> PromotedEpochInput:
    """Promote only an independently valid, closed total-capture epoch."""
    checked_manifest = validate_manifest(manifest)
    verified_records = [validate_record(item, checked_manifest) for item in records]
    verify_artifacts(checked_manifest, artifact_paths)
    with Source1DemoReader.open(artifact_paths["demo"]) as reader:
        playback_ns = round(reader.header.playback_time * 1_000_000_000)
    descriptor = {
        "ordinal": _integer(ordinal, "ordinal"),
        "process_epoch_id": _text(process_epoch_id, "process_epoch_id"),
        "map_epoch_id": _text(map_epoch_id, "map_epoch_id"),
        "capture_epoch_id": _text(capture_epoch_id, "capture_epoch_id"),
        "map_name": checked_manifest["producer_closure"]["native_demo"]["map_name"],
        "status": "complete", "host_horizon": copy.deepcopy(dict(host_horizon)),
        "reset": copy.deepcopy(dict(reset)),
        "capture_manifest_sha256": checked_manifest["manifest_sha256"],
        "salvage_sha256": None,
        "native_demo_sha256": checked_manifest["artifacts"]["demo_sha256"],
        "demo_playback_ns": playback_ns,
    }
    key = epoch_key(process_epoch_id, map_epoch_id, capture_epoch_id)
    return PromotedEpochInput(
        key, descriptor,
        CollectionEpochInput(checked_manifest, verified_records, artifact_paths))


__all__ = [
    "MONITOR_DIAGNOSTIC_SCHEMA", "MONITOR_HANDOFF_SCHEMA", "PromotedEpochInput",
    "diagnostic_monitor_handoff", "promote_verified_total_epoch",
    "validate_monitor_handoff",
]
