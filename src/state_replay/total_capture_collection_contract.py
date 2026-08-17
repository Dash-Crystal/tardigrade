"""Outer process/map-epoch transcript contract for total captures."""
from __future__ import annotations

import copy
from typing import Mapping, Sequence

from .total_capture_contract import (
    TotalCaptureContractError,
    canonical_json,
    document_sha256,
    seal_document,
)


TOTAL_CAPTURE_COLLECTION_SCHEMA = "tardigrade/total-capture-collection/v1"
TOTAL_CAPTURE_COLLECTION_TERMINAL_SCHEMA = (
    "tardigrade/total-capture-collection-terminal/v1"
)
TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA = (
    "tardigrade/total-capture-collection-render/v1"
)
TOTAL_CAPTURE_COVERAGE_LEDGER_SCHEMA = (
    "tardigrade/total-capture-coverage-ledger/v1"
)
TOTAL_CAPTURE_EPOCH_CONTEXT_ENTITY_ID = "total-capture:epoch-context"
TOTAL_CAPTURE_EPOCH_CONTEXT_COMPONENT = "total_capture_epoch_context"
COLLECTION_CLOCK_DOMAIN = "host-monotonic-ns"
EPOCH_CONTEXT_DERIVATION = (
    "lossless projection of sealed collection epoch descriptor into canonical state"
)
COLLECTION_INTERVAL_STATES = frozenset({"target-active", "target-inactive", "gap"})
EPOCH_STATUSES = frozenset({"complete", "crashed", "incomplete"})
RESET_KINDS = frozenset({
    "session-start", "map-change", "process-restart", "crash-recovery",
    "reconnect", "record-rotation",
})


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TotalCaptureContractError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> Sequence[object]:
    if (not isinstance(value, Sequence) or
            isinstance(value, (str, bytes, bytearray))):
        raise TotalCaptureContractError(f"{label} must be an array")
    return value


def _exact(value: Mapping[str, object], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise TotalCaptureContractError(
            f"{label} fields must exactly be {sorted(fields)}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TotalCaptureContractError(f"{label} must be nonempty text")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TotalCaptureContractError(
            f"{label} must be an integer >= {minimum}")
    return value


def _hash(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise TotalCaptureContractError(f"{label} must be lowercase sha256")
    return digest


def epoch_key(process_epoch_id: str, map_epoch_id: str,
              capture_epoch_id: str) -> str:
    return (f"{_string(process_epoch_id, 'process_epoch_id')}::"
            f"{_string(map_epoch_id, 'map_epoch_id')}::"
            f"{_string(capture_epoch_id, 'capture_epoch_id')}")


def epoch_stream_id(collection_id: str, ordinal: int,
                    capture_epoch_id: str) -> str:
    return (f"{_string(collection_id, 'collection_id')}/epoch/"
            f"{_integer(ordinal, 'ordinal')}/"
            f"{_string(capture_epoch_id, 'capture_epoch_id')}")


def collection_action_locator(process_epoch_id: str, map_epoch_id: str,
                              capture_epoch_id: str,
                              action_id: str) -> dict[str, str]:
    return {
        "process_epoch_id": _string(process_epoch_id, "process_epoch_id"),
        "map_epoch_id": _string(map_epoch_id, "map_epoch_id"),
        "capture_epoch_id": _string(capture_epoch_id, "capture_epoch_id"),
        "action_id": _string(action_id, "action_id"),
    }


def validate_collection_manifest(document: Mapping[str, object]) -> dict[str, object]:
    doc = _object(document, "collection manifest")
    _exact(doc, {"schema", "collection_id", "request", "clock", "coverage",
                 "epochs", "outer_artifacts", "collection_manifest_sha256"},
           "collection manifest")
    if doc.get("schema") != TOTAL_CAPTURE_COLLECTION_SCHEMA:
        raise TotalCaptureContractError("wrong total-capture collection schema")
    _string(doc.get("collection_id"), "collection_id")
    request = _object(doc.get("request"), "collection request")
    _exact(request, {"requested_start_monotonic_ns", "required_target_active_ns",
                     "close_policy", "max_wall_ns", "target_executable_sha256"},
           "collection request")
    requested_start = _integer(request.get("requested_start_monotonic_ns"),
                               "requested_start_monotonic_ns")
    required_active = _integer(request.get("required_target_active_ns"),
                               "required_target_active_ns", 1)
    if request.get("close_policy") != "target-active-threshold":
        raise TotalCaptureContractError(
            "close_policy must be target-active-threshold")
    max_wall = request.get("max_wall_ns")
    if max_wall is not None:
        max_wall = _integer(max_wall, "max_wall_ns", 1)
    _hash(request.get("target_executable_sha256"), "target executable sha256")
    clock = _object(doc.get("clock"), "collection clock")
    _exact(clock, {"domain", "start_monotonic_ns", "end_monotonic_ns"},
           "collection clock")
    if clock.get("domain") != COLLECTION_CLOCK_DOMAIN:
        raise TotalCaptureContractError("collection clock must be host-monotonic-ns")
    start = _integer(clock.get("start_monotonic_ns"), "clock start")
    end = _integer(clock.get("end_monotonic_ns"), "clock end")
    if start != requested_start or end <= start:
        raise TotalCaptureContractError(
            "collection clock must start at request and advance to observed terminal")
    if max_wall is not None and end - start > max_wall:
        raise TotalCaptureContractError("observed wall span exceeds max_wall_ns safety limit")
    outer_artifacts = _array(doc.get("outer_artifacts"), "outer_artifacts")
    roles: set[str] = set()
    artifact_ids: set[str] = set()
    for raw in outer_artifacts:
        artifact = _object(raw, "outer artifact")
        _exact(artifact, {"id", "role", "sha256"}, "outer artifact")
        artifact_id = _string(artifact.get("id"), "outer artifact id")
        role = _string(artifact.get("role"), "outer artifact role")
        if artifact_id in artifact_ids or role in roles:
            raise TotalCaptureContractError("duplicate outer artifact id/role")
        artifact_ids.add(artifact_id)
        roles.add(role)
        _hash(artifact.get("sha256"), "outer artifact sha256")
    if roles != {"monitor-events", "coverage-ledger"}:
        raise TotalCaptureContractError(
            "outer artifacts must contain monitor-events and coverage-ledger")

    raw_epochs = _array(doc.get("epochs"), "epochs")
    if not raw_epochs:
        raise TotalCaptureContractError("collection must declare at least one epoch")
    epochs: list[Mapping[str, object]] = []
    epoch_keys: set[str] = set()
    previous_end = start
    previous: Mapping[str, object] | None = None
    for ordinal, raw in enumerate(raw_epochs):
        epoch = _object(raw, "epoch")
        _exact(epoch, {"ordinal", "process_epoch_id", "map_epoch_id",
                       "capture_epoch_id", "map_name", "status",
                       "host_horizon", "reset", "capture_manifest_sha256",
                       "salvage_sha256", "native_demo_sha256",
                       "demo_playback_ns"}, "epoch")
        if _integer(epoch.get("ordinal"), "epoch.ordinal") != ordinal:
            raise TotalCaptureContractError("epoch ordinals must be contiguous from zero")
        process_id = _string(epoch.get("process_epoch_id"), "process_epoch_id")
        map_id = _string(epoch.get("map_epoch_id"), "map_epoch_id")
        capture_id = _string(epoch.get("capture_epoch_id"), "capture_epoch_id")
        _string(epoch.get("map_name"), "epoch map_name")
        key = epoch_key(process_id, map_id, capture_id)
        if key in epoch_keys:
            raise TotalCaptureContractError("duplicate process/map epoch identity")
        epoch_keys.add(key)
        status = epoch.get("status")
        if status not in EPOCH_STATUSES:
            raise TotalCaptureContractError("invalid epoch status")
        host = _object(epoch.get("host_horizon"), "epoch.host_horizon")
        _exact(host, {"start_monotonic_ns", "end_monotonic_ns"},
               "epoch.host_horizon")
        epoch_start = _integer(host.get("start_monotonic_ns"), "epoch start")
        epoch_end = _integer(host.get("end_monotonic_ns"), "epoch end")
        if not start <= epoch_start < epoch_end <= end or epoch_start < previous_end:
            raise TotalCaptureContractError("epoch host horizons overlap or escape request")
        previous_end = epoch_end
        reset = _object(epoch.get("reset"), "epoch.reset")
        _exact(reset, {"kind", "previous_process_epoch_id", "previous_map_epoch_id",
                       "previous_capture_epoch_id", "reason"}, "epoch.reset")
        reset_kind = reset.get("kind")
        if reset_kind not in RESET_KINDS:
            raise TotalCaptureContractError("invalid epoch reset kind")
        _string(reset.get("reason"), "epoch reset reason")
        if ordinal == 0:
            if (reset_kind != "session-start" or
                    reset.get("previous_process_epoch_id") is not None or
                    reset.get("previous_map_epoch_id") is not None or
                    reset.get("previous_capture_epoch_id") is not None):
                raise TotalCaptureContractError("first epoch must be a session-start reset")
        else:
            assert previous is not None
            if (reset.get("previous_process_epoch_id") !=
                    previous.get("process_epoch_id") or
                    reset.get("previous_map_epoch_id") != previous.get("map_epoch_id")):
                raise TotalCaptureContractError("epoch reset does not name prior epoch")
            if reset.get("previous_capture_epoch_id") != previous.get("capture_epoch_id"):
                raise TotalCaptureContractError("epoch reset does not name prior capture epoch")
            same_process = process_id == previous.get("process_epoch_id")
            same_map = map_id == previous.get("map_epoch_id")
            expected = ({"reconnect", "record-rotation"} if same_process and same_map
                        else {"map-change"} if same_process else
                        {"crash-recovery"} if previous.get("status") != "complete"
                        else {"process-restart"})
            if reset_kind not in expected:
                raise TotalCaptureContractError(
                    f"epoch reset kind must be one of {sorted(expected)!r}")
        capture_hash = epoch.get("capture_manifest_sha256")
        salvage_hash = epoch.get("salvage_sha256")
        if status == "complete":
            _hash(capture_hash, "epoch capture manifest sha256")
            if salvage_hash is not None:
                raise TotalCaptureContractError("complete epoch cannot carry salvage")
        else:
            if capture_hash is not None:
                raise TotalCaptureContractError("incomplete epoch cannot claim manifest")
            _hash(salvage_hash, "epoch salvage sha256")
        demo_hash = epoch.get("native_demo_sha256")
        if status == "complete":
            _hash(demo_hash, "epoch native demo sha256")
            _integer(epoch.get("demo_playback_ns"), "epoch demo_playback_ns", 1)
        elif demo_hash is not None:
            _hash(demo_hash, "incomplete epoch native demo sha256")
        if status != "complete" and epoch.get("demo_playback_ns") is not None:
            raise TotalCaptureContractError(
                "incomplete epoch cannot claim closed demo playback duration")
        epochs.append(epoch)
        previous = epoch

    coverage = _object(doc.get("coverage"), "coverage")
    _exact(coverage, {"intervals", "target_active_ns", "complete_target_active_ns",
                      "target_inactive_ns", "gap_ns", "corpus_status",
                      "continuity_status"}, "coverage")
    intervals = _array(coverage.get("intervals"), "coverage.intervals")
    if not intervals:
        raise TotalCaptureContractError("coverage intervals cannot be empty")
    cursor = start
    totals = {"target-active": 0, "target-inactive": 0, "gap": 0}
    complete_target_active = sum(
        int(item["demo_playback_ns"]) for item in epochs
        if item["status"] == "complete")
    referenced_epoch_keys: set[str] = set()
    epoch_by_key = {epoch_key(str(item["process_epoch_id"]), str(item["map_epoch_id"]),
                              str(item["capture_epoch_id"])):
                    item for item in epochs}
    for raw in intervals:
        item = _object(raw, "coverage interval")
        _exact(item, {"start_monotonic_ns", "end_monotonic_ns", "state",
                      "process_epoch_id", "map_epoch_id", "capture_epoch_id",
                      "reason"},
               "coverage interval")
        interval_start = _integer(item.get("start_monotonic_ns"), "interval start")
        interval_end = _integer(item.get("end_monotonic_ns"), "interval end")
        if interval_start != cursor or interval_end <= interval_start or interval_end > end:
            raise TotalCaptureContractError(
                "coverage intervals must exactly partition the request")
        cursor = interval_end
        state = item.get("state")
        if state not in COLLECTION_INTERVAL_STATES:
            raise TotalCaptureContractError("invalid coverage interval state")
        totals[str(state)] += interval_end - interval_start
        if state == "target-active":
            process_id = _string(item.get("process_epoch_id"),
                                 "active interval process_epoch_id")
            map_id = _string(item.get("map_epoch_id"),
                             "active interval map_epoch_id")
            capture_id = _string(item.get("capture_epoch_id"),
                                 "active interval capture_epoch_id")
            key = epoch_key(process_id, map_id, capture_id)
            epoch = epoch_by_key.get(key)
            if epoch is None:
                raise TotalCaptureContractError("active interval references unknown epoch")
            host = epoch["host_horizon"]
            if not (int(host["start_monotonic_ns"]) <= interval_start and
                    interval_end <= int(host["end_monotonic_ns"])):
                raise TotalCaptureContractError("active interval escapes epoch horizon")
            if item.get("reason") is not None:
                raise TotalCaptureContractError("active interval cannot carry gap reason")
            referenced_epoch_keys.add(key)
        else:
            identity = (item.get("process_epoch_id"), item.get("map_epoch_id"),
                        item.get("capture_epoch_id"))
            if any(value is not None for value in identity):
                if not all(isinstance(value, str) and value for value in identity):
                    raise TotalCaptureContractError(
                        "inactive/gap epoch identity must be complete or wholly null")
                key = epoch_key(str(identity[0]), str(identity[1]), str(identity[2]))
                epoch = epoch_by_key.get(key)
                if epoch is None:
                    raise TotalCaptureContractError(
                        "inactive/gap interval references unknown epoch")
                host = epoch["host_horizon"]
                if not (int(host["start_monotonic_ns"]) <= interval_start and
                        interval_end <= int(host["end_monotonic_ns"])):
                    raise TotalCaptureContractError(
                        "inactive/gap interval escapes referenced epoch horizon")
                referenced_epoch_keys.add(key)
            _string(item.get("reason"), "inactive/gap reason")
    if cursor != end:
        raise TotalCaptureContractError("coverage intervals do not reach request end")
    expected_totals = {
        "target-active": _integer(coverage.get("target_active_ns"), "target_active_ns"),
        "target-inactive": _integer(coverage.get("target_inactive_ns"),
                                    "target_inactive_ns"),
        "gap": _integer(coverage.get("gap_ns"), "gap_ns"),
    }
    if totals != expected_totals:
        raise TotalCaptureContractError("coverage duration totals are false")
    if _integer(coverage.get("complete_target_active_ns"),
                "complete_target_active_ns") != complete_target_active:
        raise TotalCaptureContractError(
            "complete_target_active_ns includes incomplete/crashed coverage")
    if complete_target_active > totals["target-active"]:
        raise TotalCaptureContractError(
            "closed demo playback exceeds observed target-active host coverage")
    if set(epoch_by_key) != referenced_epoch_keys:
        raise TotalCaptureContractError(
            "every declared epoch must appear in coverage and vice versa")
    expected_corpus = "target-met" if complete_target_active >= required_active else "target-not-met"
    if coverage.get("corpus_status") != expected_corpus:
        raise TotalCaptureContractError(
            f"coverage corpus_status must be {expected_corpus!r}")
    expected_continuity = (
        "gap-free" if totals["gap"] == 0 and
        all(item["status"] == "complete" for item in epochs) else "gapped")
    if coverage.get("continuity_status") != expected_continuity:
        raise TotalCaptureContractError(
            f"coverage continuity_status must be {expected_continuity!r}")
    expected_hash = document_sha256(doc, "collection_manifest_sha256")
    if doc.get("collection_manifest_sha256") != expected_hash:
        raise TotalCaptureContractError(
            f"collection_manifest_sha256 mismatch: expected {expected_hash}")
    canonical_json(doc)
    return copy.deepcopy(dict(doc))


def validate_coverage_ledger(document: Mapping[str, object],
                             manifest: Mapping[str, object]) -> dict[str, object]:
    ledger = _object(document, "coverage ledger")
    _exact(ledger, {"schema", "collection_id", "clock", "intervals",
                    "ledger_sha256"}, "coverage ledger")
    if ledger.get("schema") != TOTAL_CAPTURE_COVERAGE_LEDGER_SCHEMA:
        raise TotalCaptureContractError("wrong coverage ledger schema")
    if ledger.get("collection_id") != manifest.get("collection_id"):
        raise TotalCaptureContractError("coverage ledger collection mismatch")
    if (ledger.get("clock") != manifest.get("clock") or
            ledger.get("intervals") != manifest.get("coverage", {}).get("intervals")):
        raise TotalCaptureContractError(
            "coverage ledger does not exactly evidence manifest intervals")
    expected = document_sha256(ledger, "ledger_sha256")
    if ledger.get("ledger_sha256") != expected:
        raise TotalCaptureContractError("coverage ledger sha256 mismatch")
    return copy.deepcopy(dict(ledger))


def validate_collection_terminal(document: Mapping[str, object],
                                 manifest: Mapping[str, object]) -> dict[str, object]:
    terminal = _object(document, "collection terminal")
    _exact(terminal, {"schema", "collection_id", "collection_manifest_sha256",
                      "end_monotonic_ns", "corpus_status", "continuity_status",
                      "close_reason", "terminal_sha256"},
           "collection terminal")
    if terminal.get("schema") != TOTAL_CAPTURE_COLLECTION_TERMINAL_SCHEMA:
        raise TotalCaptureContractError("wrong collection terminal schema")
    if (terminal.get("collection_id") != manifest.get("collection_id") or
            terminal.get("collection_manifest_sha256") !=
            manifest.get("collection_manifest_sha256")):
        raise TotalCaptureContractError("collection terminal identity mismatch")
    if _integer(terminal.get("end_monotonic_ns"), "collection terminal end") != int(
            manifest["clock"]["end_monotonic_ns"]):
        raise TotalCaptureContractError("collection terminal end mismatch")
    if (terminal.get("corpus_status") != manifest["coverage"]["corpus_status"] or
            terminal.get("continuity_status") !=
            manifest["coverage"]["continuity_status"]):
        raise TotalCaptureContractError("collection terminal coverage status mismatch")
    close_reason = terminal.get("close_reason")
    if close_reason not in {"target-met", "user-stop", "max-wall",
                            "target-terminated", "monitor-failure"}:
        raise TotalCaptureContractError("invalid collection close_reason")
    if (manifest["coverage"]["corpus_status"] == "target-met" and
            close_reason in {"max-wall", "monitor-failure"}):
        raise TotalCaptureContractError("target-met corpus has incoherent close_reason")
    if (manifest["coverage"]["corpus_status"] == "target-not-met" and
            close_reason == "target-met"):
        raise TotalCaptureContractError("target-not-met corpus cannot close as target-met")
    if close_reason == "max-wall":
        max_wall = manifest["request"]["max_wall_ns"]
        if max_wall is None or int(terminal["end_monotonic_ns"]) - int(
                manifest["clock"]["start_monotonic_ns"]) != int(max_wall):
            raise TotalCaptureContractError("max-wall close lacks exact safety-bound hit")
    expected = document_sha256(terminal, "terminal_sha256")
    if terminal.get("terminal_sha256") != expected:
        raise TotalCaptureContractError("collection terminal hash mismatch")
    return copy.deepcopy(dict(terminal))


def locate_complete_epoch(manifest: Mapping[str, object],
                          host_monotonic_ns: int) -> dict[str, object]:
    """Resolve an outer clock only when it lies in reconstructible active state."""
    checked = validate_collection_manifest(manifest)
    target = _integer(host_monotonic_ns, "host_monotonic_ns")
    for interval in checked["coverage"]["intervals"]:
        if (int(interval["start_monotonic_ns"]) <= target <
                int(interval["end_monotonic_ns"])):
            if interval["state"] != "target-active":
                raise TotalCaptureContractError(
                    f"host time lies in {interval['state']} coverage; no state bridge")
            key = epoch_key(str(interval["process_epoch_id"]),
                            str(interval["map_epoch_id"]),
                            str(interval["capture_epoch_id"]))
            for descriptor in checked["epochs"]:
                if epoch_key(str(descriptor["process_epoch_id"]),
                             str(descriptor["map_epoch_id"]),
                             str(descriptor["capture_epoch_id"])) == key:
                    if descriptor["status"] != "complete":
                        raise TotalCaptureContractError(
                            "host time lies in crashed/incomplete epoch; salvage has no state")
                    return copy.deepcopy(dict(descriptor))
    raise TotalCaptureContractError("host time lies outside collection horizon")


def validate_salvage_document(document: Mapping[str, object]) -> dict[str, object]:
    salvage = _object(document, "incomplete capture salvage")
    _exact(salvage, {"schema", "capture_session_id", "status", "reason",
                     "artifact_sha256s", "salvage_sha256"}, "salvage")
    if salvage.get("schema") != "tardigrade/incomplete-capture/v1" or salvage.get(
            "status") != "incomplete":
        raise TotalCaptureContractError("not an incomplete capture salvage document")
    _string(salvage.get("capture_session_id"), "salvage capture_session_id")
    _string(salvage.get("reason"), "salvage reason")
    artifacts = _object(salvage.get("artifact_sha256s"), "salvage artifacts")
    if not artifacts:
        raise TotalCaptureContractError("salvage has no artifacts")
    for key, digest in artifacts.items():
        _string(key, "salvage artifact id")
        _hash(digest, "salvage artifact sha256")
    expected = document_sha256(salvage, "salvage_sha256")
    if salvage.get("salvage_sha256") != expected:
        raise TotalCaptureContractError("salvage sha256 mismatch")
    return copy.deepcopy(dict(salvage))


def validate_collection_render_envelope(
        document: Mapping[str, object]) -> dict[str, object]:
    envelope = _object(document, "collection render envelope")
    _exact(envelope, {"schema", "manifest", "terminal", "epochs"},
           "collection render envelope")
    if envelope.get("schema") != TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA:
        raise TotalCaptureContractError("wrong collection render schema")
    manifest = validate_collection_manifest(
        _object(envelope.get("manifest"), "render manifest"))
    validate_collection_terminal(
        _object(envelope.get("terminal"), "render terminal"), manifest)
    entries = _array(envelope.get("epochs"), "render epochs")
    if len(entries) != len(manifest["epochs"]):
        raise TotalCaptureContractError("render epoch count mismatch")
    for ordinal, raw in enumerate(entries):
        entry = _object(raw, "render epoch")
        _exact(entry, {"ordinal", "process_epoch_id", "map_epoch_id",
                       "capture_epoch_id", "status",
                       "stream_id", "frames", "salvage"}, "render epoch")
        descriptor = manifest["epochs"][ordinal]
        if (entry.get("ordinal"), entry.get("process_epoch_id"),
                entry.get("map_epoch_id"), entry.get("capture_epoch_id"),
                entry.get("status")) != (
                    ordinal, descriptor["process_epoch_id"],
                    descriptor["map_epoch_id"], descriptor["capture_epoch_id"],
                    descriptor["status"]):
            raise TotalCaptureContractError("render epoch identity mismatch")
        frames = _array(entry.get("frames"), "render epoch frames")
        if descriptor["status"] == "complete":
            expected_stream = epoch_stream_id(
                str(manifest["collection_id"]), ordinal,
                str(descriptor["capture_epoch_id"]))
            if entry.get("stream_id") != expected_stream or entry.get("salvage") is not None:
                raise TotalCaptureContractError("complete render epoch routing mismatch")
            if not frames:
                raise TotalCaptureContractError("complete render epoch has no snapshots")
            for frame in frames:
                frame = _object(frame, "render frame")
                if frame.get("stream_id") != expected_stream:
                    raise TotalCaptureContractError("frame crosses epoch stream")
        else:
            if entry.get("stream_id") is not None or frames:
                raise TotalCaptureContractError(
                    "incomplete epoch cannot supply canonical frames")
            salvage = validate_salvage_document(
                _object(entry.get("salvage"), "render salvage"))
            if salvage["salvage_sha256"] != descriptor["salvage_sha256"]:
                raise TotalCaptureContractError("render salvage identity mismatch")
    return copy.deepcopy(dict(envelope))


def epoch_context_entity(manifest: Mapping[str, object], ordinal: int) -> dict[str, object]:
    checked = validate_collection_manifest(manifest)
    epochs = checked["epochs"]
    ordinal = _integer(ordinal, "epoch ordinal")
    if ordinal >= len(epochs):
        raise TotalCaptureContractError("epoch ordinal out of range")
    epoch = epochs[ordinal]
    if epoch["status"] != "complete":
        raise TotalCaptureContractError("incomplete epoch has no canonical state")
    value = {
        "collection_id": checked["collection_id"],
        "collection_manifest_sha256": checked["collection_manifest_sha256"],
        "ordinal": ordinal,
        "process_epoch_id": epoch["process_epoch_id"],
        "map_epoch_id": epoch["map_epoch_id"],
        "capture_epoch_id": epoch["capture_epoch_id"],
        "map_name": epoch["map_name"],
        "status": "complete",
        "host_horizon": copy.deepcopy(epoch["host_horizon"]),
        "reset": copy.deepcopy(epoch["reset"]),
        "capture_manifest_sha256": epoch["capture_manifest_sha256"],
        "demo_playback_ns": epoch["demo_playback_ns"],
    }
    return {
        "id": TOTAL_CAPTURE_EPOCH_CONTEXT_ENTITY_ID,
        "generation": ordinal,
        "class": "TotalCaptureEpochContext",
        "components": {
            TOTAL_CAPTURE_EPOCH_CONTEXT_COMPONENT: {
                "provenance": "derived", "derivation": EPOCH_CONTEXT_DERIVATION,
                "value": value,
            },
        },
    }


def validate_epoch_context_component(
        envelope: Mapping[str, object],
        manifest: Mapping[str, object] | None = None) -> dict[str, object]:
    component = _object(envelope, "epoch context component")
    _exact(component, {"provenance", "derivation", "value"},
           "epoch context component")
    if (component.get("provenance") != "derived" or
            component.get("derivation") != EPOCH_CONTEXT_DERIVATION):
        raise TotalCaptureContractError("epoch context derivation is invalid")
    value = _object(component.get("value"), "epoch context value")
    _exact(value, {"collection_id", "collection_manifest_sha256", "ordinal",
                   "process_epoch_id", "map_epoch_id", "capture_epoch_id",
                   "map_name", "status", "host_horizon",
                   "reset", "capture_manifest_sha256", "demo_playback_ns"},
           "epoch context value")
    _string(value.get("collection_id"), "epoch context collection_id")
    _hash(value.get("collection_manifest_sha256"),
          "epoch context collection manifest sha256")
    ordinal = _integer(value.get("ordinal"), "epoch context ordinal")
    _string(value.get("process_epoch_id"), "epoch context process_epoch_id")
    _string(value.get("map_epoch_id"), "epoch context map_epoch_id")
    _string(value.get("capture_epoch_id"), "epoch context capture_epoch_id")
    _string(value.get("map_name"), "epoch context map_name")
    if value.get("status") != "complete":
        raise TotalCaptureContractError("canonical epoch context must be complete")
    host = _object(value.get("host_horizon"), "epoch context host horizon")
    _exact(host, {"start_monotonic_ns", "end_monotonic_ns"},
           "epoch context host horizon")
    if _integer(host.get("end_monotonic_ns"), "epoch context end") <= _integer(
            host.get("start_monotonic_ns"), "epoch context start"):
        raise TotalCaptureContractError("epoch context horizon does not advance")
    reset = _object(value.get("reset"), "epoch context reset")
    _exact(reset, {"kind", "previous_process_epoch_id", "previous_map_epoch_id",
                   "previous_capture_epoch_id", "reason"}, "epoch context reset")
    if reset.get("kind") not in RESET_KINDS:
        raise TotalCaptureContractError("epoch context reset kind is invalid")
    _string(reset.get("reason"), "epoch context reset reason")
    _hash(value.get("capture_manifest_sha256"),
          "epoch context capture manifest sha256")
    _integer(value.get("demo_playback_ns"), "epoch context demo_playback_ns", 1)
    if manifest is not None:
        expected = epoch_context_entity(manifest, ordinal)["components"][
            TOTAL_CAPTURE_EPOCH_CONTEXT_COMPONENT]
        if component != expected:
            raise TotalCaptureContractError(
                "epoch context does not exactly join collection manifest")
    canonical_json(component)
    return copy.deepcopy(dict(component))


__all__ = [name for name in globals() if name.isupper()] + [
    "collection_action_locator", "epoch_context_entity", "epoch_key",
    "epoch_stream_id", "locate_complete_epoch",
    "validate_epoch_context_component",
    "validate_collection_manifest", "validate_collection_render_envelope",
    "validate_collection_terminal", "validate_coverage_ledger",
    "validate_salvage_document",
]
