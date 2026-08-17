"""Strict, transactional ingestion for engine-neutral total captures."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .integrator import MANIFEST_SCHEMA, TRANSACTION_SCHEMA, StateIntegrator
from .source1_demo import Source1DemoError, Source1DemoReader
from .total_capture_contract import (
    CAPTURE_COMPLETENESS_COMPONENT,
    COMPLETENESS_DERIVATION,
    TOTAL_CAPTURE_MANIFEST_COMPONENT,
    TOTAL_CAPTURE_METADATA_ENTITY_ID,
    TotalCaptureContractError,
    capture_completeness_document,
    canonical_json,
    validate_capture_completeness,
    validate_hid_interval,
    validate_manifest,
    pov_entity_id,
    validate_record,
    validate_recorded_component,
    validate_sidecar_terminal,
    validate_view_angle_anchor,
)


@dataclass(frozen=True)
class TotalCaptureSummary:
    capture_id: str
    stream_id: str
    records: int
    transactions: int
    final_state_hash: str
    output: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(manifest: Mapping[str, object],
                     paths: Mapping[str, os.PathLike[str] | str]) -> None:
    """Verify demo and every declared sidecar; extra/unreadable paths fail."""
    expected = {"demo": str(manifest["artifacts"]["demo_sha256"])}
    expected.update({str(item["id"]): str(item["sha256"])
                     for item in manifest["artifacts"]["sidecars"]})
    if set(paths) != set(expected):
        raise TotalCaptureContractError(
            f"artifact paths must exactly be {sorted(expected)}")
    for artifact_id, digest in expected.items():
        path = Path(paths[artifact_id]).expanduser().resolve()
        if not path.is_file():
            raise TotalCaptureContractError(
                f"artifact {artifact_id!r} is not a regular file")
        actual = _sha256_file(path)
        if actual != digest:
            raise TotalCaptureContractError(
                f"artifact {artifact_id!r} sha256 mismatch: expected {digest}, got {actual}")
    evidence = manifest["producer_closure"]
    native = evidence["native_demo"]
    demo_path = Path(paths["demo"]).expanduser().resolve()
    try:
        with Source1DemoReader.open(demo_path) as reader:
            commands = list(reader.commands())
            header = reader.header
    except Source1DemoError as exc:
        raise TotalCaptureContractError(
            f"native demo lacks valid parsed dem_stop closure: {exc}") from exc
    stop = commands[-1]
    observed = {
        "format": "source1-hl2demo-v4", "terminal_command": "dem_stop",
        "terminal_tick": stop.tick, "playback_ticks": header.playback_ticks,
        "map_name": header.map_name, "server_name": header.server_name,
    }
    if observed != native:
        raise TotalCaptureContractError(
            f"native demo closure evidence mismatch: observed {observed!r}")
    session_id = str(evidence["capture_session_id"])
    horizon = manifest["horizon"]
    sidecars = {str(item["id"]): item for item in manifest["artifacts"]["sidecars"]}
    for sidecar_id, closure in evidence["sidecars"].items():
        if sidecars[sidecar_id]["role"] != "capture-stream":
            raise TotalCaptureContractError("closure attached to non-stream sidecar")
        path = Path(paths[sidecar_id]).expanduser().resolve()
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 1024 * 1024))
                tail = handle.read().decode("utf-8")
            lines = [line for line in tail.splitlines() if line.strip()]
            terminal = json.loads(lines[-1])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
            raise TotalCaptureContractError(
                f"capture sidecar {sidecar_id!r} has no bounded JSON terminal") from exc
        terminal = validate_sidecar_terminal(
            terminal, sidecar_id=sidecar_id, capture_session_id=session_id,
            start_tick=int(horizon["start_tick"]), end_tick=int(horizon["end_tick"]))
        if terminal["native_demo_sha256"] != manifest["artifacts"]["demo_sha256"]:
            raise TotalCaptureContractError(
                "capture sidecar does not bind the exact native demo bytes")
        if terminal["terminal_sha256"] != closure["terminal_sha256"]:
            raise TotalCaptureContractError("sidecar terminal evidence hash mismatch")


def _ops(payload: Mapping[str, object], *, checkpoint: bool) -> list[dict[str, object]]:
    allowed = {"full", "ops"} if checkpoint else {"ops"}
    if set(payload) != allowed:
        raise TotalCaptureContractError(
            f"{'checkpoint' if checkpoint else 'delta'} payload fields must be {sorted(allowed)}")
    if checkpoint and payload.get("full") is not True:
        raise TotalCaptureContractError("checkpoint payload must declare full=true")
    raw = payload.get("ops")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TotalCaptureContractError("payload.ops must be an array")
    result: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TotalCaptureContractError("each operation must be an object")
        op = dict(item)
        if op.get("op") in {"create", "update"}:
            components = (op.get("entity", {}).get("components", {})
                          if op.get("op") == "create" else op.get("components", {}))
            if not isinstance(components, Mapping):
                raise TotalCaptureContractError("operation components must be an object")
            for name, envelope in components.items():
                if name in {
                    "final_pose", "viewmodel", "ragdoll", "rigid_body", "constraints",
                    "effects", "camera", "diegetic_ui", "visibility",
                    "render_history", "model_binding", "world_transform",
                }:
                    validate_recorded_component(str(name), envelope)
        target = (op.get("entity", {}).get("id") if op.get("op") == "create"
                  and isinstance(op.get("entity"), Mapping) else op.get("id"))
        if target == TOTAL_CAPTURE_METADATA_ENTITY_ID:
            raise TotalCaptureContractError(
                "capture records cannot mutate protocol metadata")
        result.append(op)
    return result


def _record_ops(record: Mapping[str, object], kind: str) -> list[dict[str, object]]:
    payload = record["payload"]
    record_type = str(record["type"])
    if record_type == "end":
        if payload != {"status": "complete"}:
            raise TotalCaptureContractError("terminal payload must be exactly complete")
        return []
    if kind == "usercmd" and record_type == "delta":
        allowed = {"actor_kind", "actor_id", "intent", "applied", "outcome_ops",
                   "hid_interval", "view_angle_anchor", "focus_segment_id",
                   "focus_discontinuity_before"}
        if set(payload) != allowed:
            raise TotalCaptureContractError("usercmd payload has wrong fields")
        if payload.get("actor_kind") not in {"human", "bot"}:
            raise TotalCaptureContractError("usercmd actor_kind must be human|bot")
        if not isinstance(payload.get("actor_id"), str) or not payload["actor_id"]:
            raise TotalCaptureContractError("usercmd actor_id must be nonempty")
        intent = payload.get("intent")
        if not isinstance(intent, Mapping) or set(intent) != {"action_id", "kind", "payload"}:
            raise TotalCaptureContractError("usercmd intent has wrong fields")
        if not all(isinstance(intent.get(field), str) and intent[field]
                   for field in ("action_id", "kind")):
            raise TotalCaptureContractError("usercmd action identity is missing")
        if not isinstance(intent.get("payload"), Mapping):
            raise TotalCaptureContractError("usercmd intent payload must be an object")
        if payload.get("applied") is not True:
            raise TotalCaptureContractError("only engine-confirmed applied usercmds are accepted")
        if payload["actor_kind"] == "human":
            validate_hid_interval(payload.get("hid_interval"))
            if not isinstance(payload.get("focus_segment_id"), str) or not payload[
                    "focus_segment_id"]:
                raise TotalCaptureContractError("human usercmd focus_segment_id must be nonempty")
            if not isinstance(payload.get("focus_discontinuity_before"), bool):
                raise TotalCaptureContractError(
                    "human usercmd focus_discontinuity_before must be boolean")
        elif payload.get("hid_interval") is not None or payload.get(
                "focus_segment_id") is not None or payload.get(
                    "focus_discontinuity_before") is not None:
            raise TotalCaptureContractError(
                "bot usercmd must use null HID interval/focus segment")
        validate_view_angle_anchor(payload.get("view_angle_anchor"))
        outcome = _ops({"ops": payload.get("outcome_ops")}, checkpoint=False)
        action = {
            "op": "action", "action_id": intent["action_id"],
            "kind": intent["kind"], "actor": payload["actor_id"],
            "actor_kind": payload["actor_kind"], "payload": dict(intent["payload"]),
        }
        return [action, *outcome]
    return _ops(payload, checkpoint=record_type == "checkpoint")


def _merge(target: object, patch: object) -> object:
    if not isinstance(target, Mapping) or not isinstance(patch, Mapping):
        return copy.deepcopy(patch)
    result = copy.deepcopy(dict(target))
    for key, value in patch.items():
        result[str(key)] = _merge(result.get(str(key)), value)
    return result


def _checkpoint_state(records: Sequence[tuple[Mapping[str, object],
                                                Mapping[str, object]]]
                      ) -> dict[str, dict[str, object]]:
    """Execute an aligned full-checkpoint program from empty state."""
    state: dict[str, dict[str, object]] = {}
    for record, descriptor in records:
        for operation in _record_ops(record, str(descriptor["kind"])):
            kind = operation.get("op")
            if kind == "create":
                entity = operation.get("entity")
                if not isinstance(entity, Mapping) or not isinstance(entity.get("id"), str):
                    raise TotalCaptureContractError("checkpoint create has invalid entity")
                entity_id = str(entity["id"])
                if entity_id in state:
                    raise TotalCaptureContractError("checkpoint creates duplicate entity")
                state[entity_id] = copy.deepcopy(dict(entity))
            elif kind == "update":
                entity_id = operation.get("id")
                if entity_id not in state:
                    raise TotalCaptureContractError(
                        f"checkpoint update precedes create for {entity_id!r}")
                entity = state[str(entity_id)]
                if operation.get("generation") != entity.get("generation"):
                    raise TotalCaptureContractError("checkpoint update has stale generation")
                components = operation.get("components")
                if components is not None:
                    entity["components"] = _merge(entity.get("components", {}), components)
                removals = operation.get("remove_components", [])
                if not isinstance(removals, list):
                    raise TotalCaptureContractError("checkpoint removals must be an array")
                for name in removals:
                    if name not in entity["components"]:
                        raise TotalCaptureContractError("checkpoint removes absent component")
                    del entity["components"][name]
            else:
                raise TotalCaptureContractError(
                    "full checkpoint programs may contain only create/update operations")
    # Reuse the integrator's exact entity/state validator without publishing it.
    StateIntegrator.make_checkpoint("total-capture-validation", 0, 0, 0,
                                    list(state.values()))
    return state


def _reconcile(current: Mapping[str, Mapping[str, object]],
               target: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    operations: list[dict[str, object]] = []
    for entity_id in sorted(set(current) - set(target)):
        operations.append({"op": "destroy", "id": entity_id,
                           "generation": current[entity_id]["generation"]})
    for entity_id in sorted(target):
        wanted = target[entity_id]
        existing = current.get(entity_id)
        if existing is None:
            operations.append({"op": "create", "entity": copy.deepcopy(dict(wanted))})
            continue
        if (existing.get("generation"), existing.get("class")) != (
                wanted.get("generation"), wanted.get("class")):
            operations.append({"op": "destroy", "id": entity_id,
                               "generation": existing["generation"]})
            operations.append({"op": "create", "entity": copy.deepcopy(dict(wanted))})
            continue
        old_components = existing["components"]
        new_components = wanted["components"]
        changed = {name for name in new_components
                   if old_components.get(name) != new_components[name]}
        removed = set(old_components) - set(new_components)
        to_remove = sorted(changed | removed)
        if to_remove:
            operations.append({"op": "update", "id": entity_id,
                               "generation": wanted["generation"],
                               "remove_components": to_remove})
        if changed or set(new_components) - set(old_components):
            names = changed | (set(new_components) - set(old_components))
            operations.append({"op": "update", "id": entity_id,
                               "generation": wanted["generation"],
                               "components": {name: copy.deepcopy(new_components[name])
                                              for name in sorted(names)}})
    return operations


def _validate_records(manifest: Mapping[str, object],
                      records: Iterable[Mapping[str, object]]) -> tuple[
                          list[tuple[dict[str, object], Mapping[str, object]]],
                          dict[str, tuple[int, str]]]:
    descriptors = {str(item["id"]): item for item in manifest["streams"]}
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in descriptors}
    count = 0
    for raw in records:
        count += 1
        if count > 50_000_000:
            raise TotalCaptureContractError("record limit exceeded")
        record = validate_record(raw, manifest)
        grouped[str(record["stream_id"])].append(record)
    ordered: list[tuple[dict[str, object], Mapping[str, object]]] = []
    terminals: dict[str, tuple[int, str]] = {}
    interval = int(manifest["checkpoint_interval_ticks"])
    start, end = (int(manifest["horizon"][key])
                  for key in ("start_tick", "end_tick"))
    for stream_id, descriptor in descriptors.items():
        items = grouped[stream_id]
        if not items:
            raise TotalCaptureContractError(f"stream {stream_id!r} has no records")
        last_position = (-1, -1)
        last_clock = -1
        last_checkpoint = start
        terminal = False
        last_focus_segment: str | None = None
        for expected_sequence, record in enumerate(items):
            if int(record["sequence"]) != expected_sequence:
                raise TotalCaptureContractError(
                    f"stream {stream_id!r} sequence gap/duplicate at {expected_sequence}")
            clock = record["clock"]
            if int(clock["source_sequence"]) != expected_sequence:
                raise TotalCaptureContractError("source_sequence must equal stream sequence")
            now_clock = int(clock["monotonic_ns"])
            if now_clock <= last_clock:
                raise TotalCaptureContractError("stream monotonic clock did not advance")
            last_clock = now_clock
            position = (int(record["tick"]), int(record["subtick"]))
            if position < last_position:
                raise TotalCaptureContractError("stream position moved backwards")
            last_position = position
            if terminal:
                raise TotalCaptureContractError("record follows terminal record")
            if expected_sequence == 0 and (
                    record["type"] != "checkpoint" or position[0] != start):
                raise TotalCaptureContractError("stream must begin with full checkpoint")
            if int(record["tick"]) - last_checkpoint > interval:
                raise TotalCaptureContractError("periodic checkpoint horizon exceeded")
            checked_ops = _record_ops(record, str(descriptor["kind"]))
            if (descriptor["kind"] == "usercmd" and record["type"] == "delta"
                    and record["payload"].get("actor_kind") == "human"):
                focus = str(record["payload"]["focus_segment_id"])
                discontinuity = bool(record["payload"]["focus_discontinuity_before"])
                if last_focus_segment is None:
                    if not discontinuity:
                        raise TotalCaptureContractError(
                            "first human focus segment must declare its boundary")
                elif (focus == last_focus_segment) == discontinuity:
                    raise TotalCaptureContractError(
                        "focus segment transition/discontinuity is incoherent")
                last_focus_segment = focus
            if expected_sequence == 0 and descriptor.get("pov_id") is not None:
                component_for_kind = {
                    "camera": "camera", "ui": "diegetic_ui",
                    "visibility": "visibility", "render_history": "render_history",
                }[str(descriptor["kind"])]
                expected_id = pov_entity_id(str(descriptor["pov_id"]))
                present = False
                for operation in checked_ops:
                    if operation.get("op") == "create":
                        entity = operation.get("entity", {})
                        present |= (isinstance(entity, Mapping) and
                                    entity.get("id") == expected_id and
                                    component_for_kind in entity.get("components", {}))
                    elif operation.get("op") == "update":
                        present |= (operation.get("id") == expected_id and
                                    component_for_kind in operation.get("components", {}))
                if not present:
                    raise TotalCaptureContractError(
                        f"initial {descriptor['kind']} checkpoint does not materialize "
                        f"scored POV {descriptor['pov_id']!r}")
            if record["type"] == "checkpoint":
                last_checkpoint = int(record["tick"])
            elif record["type"] == "end":
                if int(record["tick"]) != end:
                    raise TotalCaptureContractError("terminal record is not at horizon end")
                terminal = True
                terminals[stream_id] = (
                    expected_sequence, str(record["record_sha256"]))
            ordered.append((record, descriptor))
        if not terminal:
            raise TotalCaptureContractError(f"stream {stream_id!r} has no terminal record")
        if end - last_checkpoint > interval:
            raise TotalCaptureContractError("terminal checkpoint coverage gap")
    checkpoint_positions = {
        stream_id: {(int(record["tick"]), int(record["subtick"]))
                    for record in grouped[stream_id]
                    if record["type"] == "checkpoint"}
        for stream_id in descriptors
    }
    expected_positions = next(iter(checkpoint_positions.values()))
    if any(positions != expected_positions for positions in checkpoint_positions.values()):
        raise TotalCaptureContractError(
            "full checkpoint barriers must align across every stream")
    ordered.sort(key=lambda item: (
        int(item[0]["tick"]), int(item[0]["subtick"]),
        int(item[0]["clock"]["monotonic_ns"]), str(item[0]["stream_id"]),
        int(item[0]["sequence"])))
    return ordered, terminals


class TotalCaptureIngestor:
    """One-shot validator/reducer that publishes no partial journal."""

    def __init__(self) -> None:
        self._used = False

    def ingest(self, manifest_document: Mapping[str, object],
               records: Iterable[Mapping[str, object]],
               artifact_paths: Mapping[str, os.PathLike[str] | str],
               output: os.PathLike[str] | str,
               stream_id: str = "total-capture", *,
               initial_entities: Sequence[Mapping[str, object]] = (),
               ) -> TotalCaptureSummary:
        if self._used:
            raise TotalCaptureContractError("TotalCaptureIngestor is one-shot")
        self._used = True
        manifest = validate_manifest(manifest_document)
        verify_artifacts(manifest, artifact_paths)
        ordered, terminals = _validate_records(manifest, records)
        complete = capture_completeness_document(
            manifest, closed=True, terminals=terminals)
        validate_capture_completeness(manifest, complete)
        final = Path(output).expanduser().resolve()
        if final.exists():
            raise TotalCaptureContractError("output already exists")
        final.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.partial-", dir=final.parent))
        journal: StateIntegrator | None = None
        try:
            open_value = capture_completeness_document(manifest, closed=False)
            metadata = {
                "id": TOTAL_CAPTURE_METADATA_ENTITY_ID, "generation": 0,
                "class": "TotalCaptureMetadata", "components": {
                    TOTAL_CAPTURE_MANIFEST_COMPONENT: {
                        "provenance": "recorded", "value": manifest,
                    },
                    CAPTURE_COMPLETENESS_COMPONENT: {
                        "provenance": "derived", "derivation": COMPLETENESS_DERIVATION,
                        "value": open_value,
                    },
                },
            }
            extra_entities = [copy.deepcopy(dict(item)) for item in initial_entities]
            if any(item.get("id") == TOTAL_CAPTURE_METADATA_ENTITY_ID
                   for item in extra_entities):
                raise TotalCaptureContractError(
                    "initial context cannot replace total-capture metadata")
            protected_initial_ids = {str(item.get("id")) for item in extra_entities}
            initial = StateIntegrator.make_checkpoint(
                stream_id, int(manifest["horizon"]["start_tick"]), 0, 0,
                [metadata, *extra_entities])
            journal = StateIntegrator.create(stage, {
                "schema": MANIFEST_SCHEMA, "stream_id": stream_id,
                "tick_rate_hz": int(manifest["tick_rate_hz"]),
                "capture_manifest_sha256": manifest["manifest_sha256"],
            }, initial)
            sequence = 0
            checkpoint_groups: dict[tuple[int, int], list[tuple[dict[str, object],
                                                                  Mapping[str, object]]]] = {}
            events: list[tuple[tuple[int, int, int], str, object]] = []
            for record, descriptor in ordered:
                if record["type"] == "checkpoint":
                    position = (int(record["tick"]), int(record["subtick"]))
                    checkpoint_groups.setdefault(position, []).append((record, descriptor))
                elif record["type"] != "end":
                    events.append(((int(record["tick"]), int(record["subtick"]),
                                    int(record["clock"]["monotonic_ns"])),
                                   "delta", (record, descriptor)))
            descriptor_order = {str(item["id"]): index
                                for index, item in enumerate(manifest["streams"])}
            for position, group in checkpoint_groups.items():
                group.sort(key=lambda item: descriptor_order[str(item[0]["stream_id"])])
                events.append(((position[0], position[1],
                                max(int(item[0]["clock"]["monotonic_ns"])
                                    for item in group)), "checkpoint", group))
            events.sort(key=lambda item: item[0])
            for _, event_kind, event in events:
                if event_kind == "checkpoint":
                    group = event
                    target = _checkpoint_state(group)
                    current = {entity["id"]: entity
                               for entity in journal.current_snapshot()["entities"]
                               if entity["id"] != TOTAL_CAPTURE_METADATA_ENTITY_ID and
                               entity["id"] not in protected_initial_ids}
                    record_ops = _reconcile(current, target)
                    audit_records = [item[0] for item in group]
                    record = audit_records[-1]
                else:
                    record, descriptor = event
                    record_ops = _record_ops(record, str(descriptor["kind"]))
                    audit_records = [record]
                for operation in record_ops:
                    target_id = (operation.get("entity", {}).get("id")
                                 if operation.get("op") == "create" and
                                 isinstance(operation.get("entity"), Mapping)
                                 else operation.get("id"))
                    if target_id in protected_initial_ids:
                        raise TotalCaptureContractError(
                            "capture record cannot mutate collection epoch context")
                audits = [{
                    "op": "action",
                    "action_id": f"total-capture:record:{item['stream_id']}:{item['sequence']}",
                    "kind": "total_capture.record",
                    "payload": {
                        "record_sha256": item["record_sha256"],
                        "stream_id": item["stream_id"],
                        "stream_sequence": item["sequence"],
                        "clock": item["clock"],
                    },
                } for item in audit_records]
                sequence += 1
                tx = {
                    "schema": TRANSACTION_SCHEMA, "stream_id": stream_id,
                    "sequence": sequence, "tick": record["tick"],
                    "subtick": record["subtick"],
                    "base_hash": journal.current_snapshot()["state_hash"],
                    "ops": [*record_ops, *audits],
                }
                journal.ingest(tx)
                if event_kind == "checkpoint":
                    journal.checkpoint()
            sequence += 1
            snapshot = journal.current_snapshot()
            terminal_tick = int(manifest["horizon"]["end_tick"])
            terminal_subtick = (snapshot["subtick"] + 1
                                if snapshot["tick"] == terminal_tick else 0)
            journal.ingest({
                "schema": TRANSACTION_SCHEMA, "stream_id": stream_id,
                "sequence": sequence, "tick": terminal_tick,
                "subtick": terminal_subtick, "base_hash": snapshot["state_hash"],
                "ops": [{
                    "op": "update", "id": TOTAL_CAPTURE_METADATA_ENTITY_ID,
                    "generation": 0, "components": {
                        CAPTURE_COMPLETENESS_COMPONENT: {
                            "provenance": "derived", "derivation": COMPLETENESS_DERIVATION,
                            "value": complete,
                        },
                    },
                }],
            })
            published = journal.current_snapshot()
            journal.checkpoint()
            journal.close()
            journal = None
            (stage / "total-capture-ingest.json").write_text(canonical_json({
                "capture_id": manifest["capture_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "final_state_hash": published["state_hash"],
                "records": len(ordered), "status": "complete",
            }) + "\n", encoding="utf-8")
            os.replace(stage, final)
            return TotalCaptureSummary(
                str(manifest["capture_id"]), stream_id, len(ordered), sequence,
                str(published["state_hash"]), str(final))
        except Exception:
            if journal is not None:
                journal.close()
            shutil.rmtree(stage, ignore_errors=True)
            raise


__all__ = ["TotalCaptureIngestor", "TotalCaptureSummary", "verify_artifacts"]
