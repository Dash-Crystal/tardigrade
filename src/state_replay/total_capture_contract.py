"""Engine-neutral total-capture schemas and strict shared validators."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Mapping, Sequence


TOTAL_CAPTURE_MANIFEST_SCHEMA = "tardigrade/total-capture-manifest/v1"
TOTAL_CAPTURE_RECORD_SCHEMA = "tardigrade/total-capture-record/v1"
CAPTURE_SIDECAR_TERMINAL_SCHEMA = "tardigrade/capture-sidecar-terminal/v1"
INCOMPLETE_CAPTURE_SCHEMA = "tardigrade/incomplete-capture/v1"
TOTAL_CAPTURE_METADATA_ENTITY_ID = "total-capture:metadata"
TOTAL_CAPTURE_MANIFEST_COMPONENT = "total_capture_manifest"
CAPTURE_COMPLETENESS_COMPONENT = "capture_completeness"
COMPLETENESS_DERIVATION = "validated closure of declared total-capture stream horizons"
ANGULAR_TARGET_DERIVATION = (
    "continuous angular target trajectory constrained by interval-integrated HID "
    "relative counts and authoritative usercmd/render-camera anchors"
)

STREAM_KINDS = frozenset({
    "usercmd", "entity_lifecycle", "final_pose", "rigid_body", "effects",
    "camera", "ui", "visibility", "render_history",
})
POV_STREAM_KINDS = frozenset({"camera", "ui", "visibility", "render_history"})
GLOBAL_REQUIRED_KINDS = frozenset({
    "usercmd", "entity_lifecycle", "final_pose", "rigid_body", "effects",
})
STREAM_REQUIRED_FIELDS = {
    "usercmd": frozenset({"applied_intent", "applied_outcome", "actor_kind",
                          "hid_interval_integrals", "view_angle_anchors",
                          "focus_segments"}),
    "entity_lifecycle": frozenset({"entity_id", "generation", "class", "operation",
                                   "world_transform", "component_state"}),
    "final_pose": frozenset({"final_bones", "attachments", "viewmodels",
                             "ragdolls", "model_binding"}),
    "rigid_body": frozenset({"bodies", "constraints"}),
    "effects": frozenset({"instances"}),
    "camera": frozenset({"pose", "fov", "projection", "prediction"}),
    "ui": frozenset({"diegetic_ui"}),
    "visibility": frozenset({"visible_entities"}),
    "render_history": frozenset({"temporal_state"}),
}
RECORD_TYPES = frozenset({"checkpoint", "delta", "end"})

CANONICAL_COMPONENTS = frozenset({
    "final_pose", "viewmodel", "ragdoll", "rigid_body", "constraints",
    "effects", "camera", "diegetic_ui", "visibility", "render_history",
    "model_binding", "world_transform", "angular_target_history",
})


class TotalCaptureContractError(ValueError):
    """A total-capture document violates the shared protocol."""


def canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TotalCaptureContractError(f"not canonical JSON: {exc}") from exc


def document_sha256(document: Mapping[str, object], hash_field: str) -> str:
    body = {key: copy.deepcopy(value) for key, value in document.items()
            if key != hash_field}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def seal_document(document: Mapping[str, object],
                  hash_field: str) -> dict[str, object]:
    result = copy.deepcopy(dict(document))
    result[hash_field] = document_sha256(result, hash_field)
    return result


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TotalCaptureContractError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> Sequence[object]:
    if (not isinstance(value, Sequence) or
            isinstance(value, (str, bytes, bytearray))):
        raise TotalCaptureContractError(f"{label} must be an array")
    return value


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
    value = _string(value, label)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise TotalCaptureContractError(f"{label} must be lowercase sha256")
    return value


def _exact(value: Mapping[str, object], allowed: set[str], label: str) -> None:
    extras = set(value) - allowed
    if extras:
        raise TotalCaptureContractError(f"unknown {label} fields: {sorted(extras)}")


def validate_manifest(document: Mapping[str, object]) -> dict[str, object]:
    doc = _object(document, "manifest")
    _exact(doc, {"schema", "capture_id", "engine", "artifacts", "producer_closure",
                 "tick_rate_hz",
                 "checkpoint_interval_ticks", "horizon", "scored_povs", "streams",
                 "completeness", "manifest_sha256"}, "manifest")
    if doc.get("schema") != TOTAL_CAPTURE_MANIFEST_SCHEMA:
        raise TotalCaptureContractError("wrong total-capture manifest schema")
    capture_id = _string(doc.get("capture_id"), "capture_id")
    engine = _object(doc.get("engine"), "engine")
    _exact(engine, {"name", "build_id", "build_sha256",
                    "content_manifest_sha256"}, "engine")
    for field in ("name", "build_id"):
        _string(engine.get(field), f"engine.{field}")
    for field in ("build_sha256", "content_manifest_sha256"):
        _hash(engine.get(field), f"engine.{field}")
    artifacts = _object(doc.get("artifacts"), "artifacts")
    _exact(artifacts, {"demo_sha256", "sidecars"}, "artifacts")
    _hash(artifacts.get("demo_sha256"), "artifacts.demo_sha256")
    sidecars = _array(artifacts.get("sidecars"), "artifacts.sidecars")
    sidecar_ids: set[str] = set()
    for item in sidecars:
        item = _object(item, "sidecar")
        _exact(item, {"id", "sha256", "role"}, "sidecar")
        sidecar_id = _string(item.get("id"), "sidecar.id")
        if sidecar_id in sidecar_ids:
            raise TotalCaptureContractError(f"duplicate sidecar {sidecar_id!r}")
        sidecar_ids.add(sidecar_id)
        _hash(item.get("sha256"), "sidecar.sha256")
        if item.get("role") not in {"capture-stream", "content-package"}:
            raise TotalCaptureContractError("sidecar.role is invalid")
    tick_rate = _integer(doc.get("tick_rate_hz"), "tick_rate_hz", 1)
    interval = _integer(doc.get("checkpoint_interval_ticks"),
                        "checkpoint_interval_ticks", 1)
    horizon = _object(doc.get("horizon"), "horizon")
    _exact(horizon, {"start_tick", "end_tick"}, "horizon")
    start = _integer(horizon.get("start_tick"), "horizon.start_tick")
    end = _integer(horizon.get("end_tick"), "horizon.end_tick")
    if end <= start:
        raise TotalCaptureContractError("horizon.end_tick must exceed start_tick")
    closure = _object(doc.get("producer_closure"), "producer_closure")
    _exact(closure, {"ownership", "capture_session_id", "monitor_sidecar_id",
                     "native_demo", "sidecars"},
           "producer_closure")
    if closure.get("ownership") != "monitor-owned-native-demo":
        raise TotalCaptureContractError(
            "native demo must be owned by the capture monitor")
    session_id = _string(closure.get("capture_session_id"),
                         "producer_closure.capture_session_id")
    native = _object(closure.get("native_demo"), "producer_closure.native_demo")
    _exact(native, {"format", "terminal_command", "terminal_tick", "playback_ticks",
                    "map_name", "server_name"}, "producer_closure.native_demo")
    if native.get("format") != "source1-hl2demo-v4" or native.get(
            "terminal_command") != "dem_stop":
        raise TotalCaptureContractError(
            "complete capture requires parsed Source1 HL2DEMO dem_stop evidence")
    if (_integer(native.get("terminal_tick"), "native_demo.terminal_tick") != end or
            _integer(native.get("playback_ticks"), "native_demo.playback_ticks") != end):
        raise TotalCaptureContractError("native demo terminal does not bind horizon end")
    _string(native.get("map_name"), "native_demo.map_name")
    _string(native.get("server_name"), "native_demo.server_name")
    closure_sidecars = _object(closure.get("sidecars"), "producer_closure.sidecars")
    timeline_ids = {str(item["id"]) for item in sidecars
                    if item["role"] == "capture-stream"}
    monitor_sidecar_id = _string(closure.get("monitor_sidecar_id"),
                                 "producer_closure.monitor_sidecar_id")
    if monitor_sidecar_id not in timeline_ids:
        raise TotalCaptureContractError(
            "monitor_sidecar_id must name a capture-stream sidecar")
    if set(closure_sidecars) != timeline_ids:
        raise TotalCaptureContractError(
            "producer closure must exactly cover capture-stream sidecars")
    for sidecar_id, raw in closure_sidecars.items():
        evidence = _object(raw, f"producer_closure.sidecars.{sidecar_id}")
        _exact(evidence, {"terminal_schema", "terminal_sha256", "start_tick",
                          "end_tick", "capture_session_id"}, "sidecar closure")
        if evidence.get("terminal_schema") != CAPTURE_SIDECAR_TERMINAL_SCHEMA:
            raise TotalCaptureContractError("wrong sidecar terminal schema")
        _hash(evidence.get("terminal_sha256"), "sidecar terminal sha256")
        if evidence.get("capture_session_id") != session_id:
            raise TotalCaptureContractError("sidecar capture session mismatch")
        evidence_start = _integer(evidence.get("start_tick"),
                                  "sidecar closure start_tick")
        evidence_end = _integer(evidence.get("end_tick"),
                                "sidecar closure end_tick")
        if (evidence_start, evidence_end) != (start, end):
            raise TotalCaptureContractError("sidecar terminal horizon mismatch")
    povs = [_string(value, "scored_pov") for value in
            _array(doc.get("scored_povs"), "scored_povs")]
    if not povs or len(set(povs)) != len(povs):
        raise TotalCaptureContractError("scored_povs must be nonempty and unique")
    streams = _array(doc.get("streams"), "streams")
    stream_ids: set[str] = set()
    descriptors: dict[str, Mapping[str, object]] = {}
    for raw in streams:
        stream = _object(raw, "stream")
        _exact(stream, {"id", "kind", "pov_id", "producer", "required",
                        "horizon", "fields"}, "stream")
        stream_id = _string(stream.get("id"), "stream.id")
        if stream_id in stream_ids:
            raise TotalCaptureContractError(f"duplicate stream {stream_id!r}")
        stream_ids.add(stream_id)
        kind = _string(stream.get("kind"), "stream.kind")
        if kind not in STREAM_KINDS:
            raise TotalCaptureContractError(f"unsupported stream kind {kind!r}")
        if stream.get("required") is not True:
            raise TotalCaptureContractError("every declared total-capture stream is required")
        _string(stream.get("producer"), "stream.producer")
        stream_horizon = _object(stream.get("horizon"), "stream.horizon")
        _exact(stream_horizon, {"start_tick", "end_tick"}, "stream.horizon")
        stream_start = _integer(stream_horizon.get("start_tick"),
                                "stream.horizon.start_tick")
        stream_end = _integer(stream_horizon.get("end_tick"),
                              "stream.horizon.end_tick")
        if (stream_start, stream_end) != (start, end):
            raise TotalCaptureContractError(
                f"stream {stream_id!r} horizon differs from capture horizon")
        fields = [_string(field, "stream.fields[]") for field in
                  _array(stream.get("fields"), "stream.fields")]
        if not fields or len(set(fields)) != len(fields):
            raise TotalCaptureContractError(
                f"stream {stream_id!r} fields must be nonempty and unique")
        if set(fields) != STREAM_REQUIRED_FIELDS[kind]:
            raise TotalCaptureContractError(
                f"stream {stream_id!r} fields must exactly cover "
                f"{sorted(STREAM_REQUIRED_FIELDS[kind])}")
        pov_id = stream.get("pov_id")
        if kind in POV_STREAM_KINDS:
            if _string(pov_id, "stream.pov_id") not in povs:
                raise TotalCaptureContractError("POV stream references unscored POV")
        elif pov_id is not None:
            raise TotalCaptureContractError("non-POV stream cannot have pov_id")
        descriptors[stream_id] = stream
    present_global = {str(item["kind"]) for item in descriptors.values()
                      if item.get("pov_id") is None}
    if not GLOBAL_REQUIRED_KINDS <= present_global:
        raise TotalCaptureContractError(
            f"missing global streams: {sorted(GLOBAL_REQUIRED_KINDS - present_global)}")
    for pov_id in povs:
        kinds = {str(item["kind"]) for item in descriptors.values()
                 if item.get("pov_id") == pov_id}
        if kinds != POV_STREAM_KINDS:
            raise TotalCaptureContractError(
                f"scored POV {pov_id!r} requires exactly {sorted(POV_STREAM_KINDS)}")
    matrix = _object(doc.get("completeness"), "completeness")
    if set(matrix) != stream_ids:
        raise TotalCaptureContractError(
            "completeness keys must exactly match declared streams")
    for stream_id, descriptor in descriptors.items():
        entry = _object(matrix[stream_id], f"completeness.{stream_id}")
        _exact(entry, {"status", "fields"}, "completeness entry")
        if entry.get("status") != "recorded":
            raise TotalCaptureContractError(
                f"required stream {stream_id!r} is not recorded")
        field_matrix = _object(entry.get("fields"), "completeness fields")
        if set(field_matrix) != set(descriptor["fields"]):
            raise TotalCaptureContractError(
                f"completeness fields mismatch for stream {stream_id!r}")
        if any(value != "recorded" for value in field_matrix.values()):
            raise TotalCaptureContractError(
                f"stream {stream_id!r} contains unavailable required fields")
    expected = document_sha256(doc, "manifest_sha256")
    if doc.get("manifest_sha256") != expected:
        raise TotalCaptureContractError(
            f"manifest_sha256 mismatch: expected {expected}")
    canonical_json(doc)
    return copy.deepcopy(dict(doc))


def validate_record(document: Mapping[str, object],
                    manifest: Mapping[str, object]) -> dict[str, object]:
    doc = _object(document, "record")
    _exact(doc, {"schema", "capture_id", "manifest_sha256", "stream_id", "type", "sequence",
                 "tick", "subtick", "clock", "payload", "record_sha256"},
           "record")
    if doc.get("schema") != TOTAL_CAPTURE_RECORD_SCHEMA:
        raise TotalCaptureContractError("wrong total-capture record schema")
    if doc.get("capture_id") != manifest.get("capture_id"):
        raise TotalCaptureContractError("record capture_id mismatch")
    if doc.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise TotalCaptureContractError("record manifest_sha256 mismatch")
    stream_id = _string(doc.get("stream_id"), "record.stream_id")
    stream_ids = {str(item["id"]) for item in manifest["streams"]}
    if stream_id not in stream_ids:
        raise TotalCaptureContractError(f"record references unknown stream {stream_id!r}")
    kind = _string(doc.get("type"), "record.type")
    if kind not in RECORD_TYPES:
        raise TotalCaptureContractError(f"invalid record type {kind!r}")
    _integer(doc.get("sequence"), "record.sequence")
    tick = _integer(doc.get("tick"), "record.tick")
    _integer(doc.get("subtick"), "record.subtick")
    horizon = manifest["horizon"]
    if not int(horizon["start_tick"]) <= tick <= int(horizon["end_tick"]):
        raise TotalCaptureContractError("record tick outside declared horizon")
    clock = _object(doc.get("clock"), "record.clock")
    _exact(clock, {"monotonic_ns", "engine_tick", "source_sequence"}, "clock")
    _integer(clock.get("monotonic_ns"), "clock.monotonic_ns")
    if _integer(clock.get("engine_tick"), "clock.engine_tick") != tick:
        raise TotalCaptureContractError("clock.engine_tick must equal record.tick")
    _integer(clock.get("source_sequence"), "clock.source_sequence")
    _object(doc.get("payload"), "record.payload")
    expected = document_sha256(doc, "record_sha256")
    if doc.get("record_sha256") != expected:
        raise TotalCaptureContractError(
            f"record_sha256 mismatch: expected {expected}")
    canonical_json(doc)
    return copy.deepcopy(dict(doc))


def validate_sidecar_terminal(document: Mapping[str, object], *, sidecar_id: str,
                              capture_session_id: str, start_tick: int,
                              end_tick: int) -> dict[str, object]:
    terminal = _object(document, "sidecar terminal")
    _exact(terminal, {"schema", "sidecar_id", "capture_session_id", "start_tick",
                      "end_tick", "status", "native_demo_sha256",
                      "terminal_sha256"}, "sidecar terminal")
    if terminal.get("schema") != CAPTURE_SIDECAR_TERMINAL_SCHEMA:
        raise TotalCaptureContractError("wrong sidecar terminal schema")
    if terminal.get("sidecar_id") != sidecar_id or terminal.get(
            "capture_session_id") != capture_session_id:
        raise TotalCaptureContractError("sidecar terminal identity mismatch")
    terminal_start = _integer(terminal.get("start_tick"),
                              "sidecar terminal start_tick")
    terminal_end = _integer(terminal.get("end_tick"),
                            "sidecar terminal end_tick")
    if (terminal_start, terminal_end) != (start_tick, end_tick):
        raise TotalCaptureContractError("sidecar terminal horizon mismatch")
    if terminal.get("status") != "complete":
        raise TotalCaptureContractError("sidecar terminal is not complete")
    _hash(terminal.get("native_demo_sha256"), "sidecar native_demo_sha256")
    expected = document_sha256(terminal, "terminal_sha256")
    if terminal.get("terminal_sha256") != expected:
        raise TotalCaptureContractError("sidecar terminal hash mismatch")
    return copy.deepcopy(dict(terminal))


def incomplete_capture_document(capture_session_id: str, reason: str,
                                artifact_sha256s: Mapping[str, str]) -> dict[str, object]:
    """Make explicit salvage evidence; this schema is never ingestible as complete."""
    artifacts = {str(key): _hash(value, f"incomplete artifact {key}")
                 for key, value in artifact_sha256s.items()}
    if not artifacts:
        raise TotalCaptureContractError("incomplete capture has no salvage artifacts")
    return seal_document({
        "schema": INCOMPLETE_CAPTURE_SCHEMA,
        "capture_session_id": _string(capture_session_id, "capture_session_id"),
        "status": "incomplete", "reason": _string(reason, "reason"),
        "artifact_sha256s": artifacts,
    }, "salvage_sha256")


def pov_entity_id(pov_id: str) -> str:
    return f"total-capture:pov:{_string(pov_id, 'pov_id')}"


def capture_completeness_document(
        manifest: Mapping[str, object], *, closed: bool,
        terminals: Mapping[str, tuple[int, str]] | None = None) -> dict[str, object]:
    """Build the only valid open or terminal completeness value."""
    checked = validate_manifest(manifest)
    terminal_map = dict(terminals or {})
    streams: dict[str, object] = {}
    for descriptor in checked["streams"]:
        stream_id = str(descriptor["id"])
        if closed:
            if stream_id not in terminal_map:
                raise TotalCaptureContractError(
                    f"missing terminal evidence for stream {stream_id!r}")
            sequence, record_hash = terminal_map[stream_id]
            streams[stream_id] = {
                "status": "closed",
                "last_sequence": _integer(sequence, "last_sequence"),
                "terminal_record_sha256": _hash(
                    record_hash, "terminal_record_sha256"),
            }
        else:
            streams[stream_id] = {
                "status": "open", "last_sequence": -1,
                "terminal_record_sha256": None,
            }
    return {
        "status": "complete" if closed else "open",
        "manifest_sha256": checked["manifest_sha256"],
        "horizon": copy.deepcopy(checked["horizon"]),
        "streams": streams,
    }


def validate_capture_completeness(
        manifest: Mapping[str, object], value: Mapping[str, object]) -> dict[str, object]:
    candidate = _object(value, "capture_completeness.value")
    status = candidate.get("status")
    if status not in {"open", "complete"}:
        raise TotalCaptureContractError("capture completeness must be open|complete")
    terminals: dict[str, tuple[int, str]] = {}
    if status == "complete":
        streams = _object(candidate.get("streams"), "capture completeness streams")
        for stream_id, raw in streams.items():
            entry = _object(raw, f"capture completeness {stream_id}")
            if entry.get("status") != "closed":
                raise TotalCaptureContractError("complete capture has open stream")
            terminals[str(stream_id)] = (
                _integer(entry.get("last_sequence"), "last_sequence"),
                _hash(entry.get("terminal_record_sha256"),
                      "terminal_record_sha256"),
            )
    expected = capture_completeness_document(
        manifest, closed=status == "complete", terminals=terminals)
    if candidate != expected:
        raise TotalCaptureContractError(
            "capture_completeness.value is not the canonical closure document")
    return copy.deepcopy(dict(candidate))


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TotalCaptureContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TotalCaptureContractError(f"{label} must be finite")
    return result


def _vector(value: object, count: int, label: str) -> None:
    values = _array(value, label)
    if len(values) != count:
        raise TotalCaptureContractError(f"{label} must have {count} values")
    for item in values:
        _finite(item, label)


def validate_hid_interval(value: Mapping[str, object]) -> dict[str, object]:
    interval = _object(value, "hid_interval")
    _exact(interval, {"start_monotonic_ns", "end_monotonic_ns", "relative_counts",
                      "report_sequence_start", "report_sequence_end"},
           "hid_interval")
    start = _integer(interval.get("start_monotonic_ns"),
                     "hid_interval.start_monotonic_ns")
    end = _integer(interval.get("end_monotonic_ns"),
                   "hid_interval.end_monotonic_ns")
    if end <= start:
        raise TotalCaptureContractError("HID integration interval must advance")
    counts = _array(interval.get("relative_counts"), "hid_interval.relative_counts")
    if len(counts) != 2 or any(isinstance(item, bool) or not isinstance(item, int)
                               for item in counts):
        raise TotalCaptureContractError(
            "relative_counts must be two signed interval-integrated integers")
    first = _integer(interval.get("report_sequence_start"),
                     "hid_interval.report_sequence_start")
    last = _integer(interval.get("report_sequence_end"),
                    "hid_interval.report_sequence_end")
    if last < first:
        raise TotalCaptureContractError("HID report sequence moves backwards")
    return copy.deepcopy(dict(interval))


def validate_view_angle_anchor(value: Mapping[str, object]) -> dict[str, object]:
    anchor = _object(value, "view_angle_anchor")
    _exact(anchor, {"kind", "yaw_degrees", "pitch_degrees", "roll_degrees"},
           "view_angle_anchor")
    if anchor.get("kind") not in {"usercmd_target", "render_camera"}:
        raise TotalCaptureContractError("invalid view-angle anchor kind")
    for field in ("yaw_degrees", "pitch_degrees", "roll_degrees"):
        _finite(anchor.get(field), f"view_angle_anchor.{field}")
    return copy.deepcopy(dict(anchor))


def validate_angular_target_history(
        envelope: Mapping[str, object]) -> dict[str, object]:
    component = _object(envelope, "angular_target_history")
    _exact(component, {"provenance", "derivation", "value"},
           "angular_target_history")
    if component.get("provenance") != "derived" or component.get(
            "derivation") != ANGULAR_TARGET_DERIVATION:
        raise TotalCaptureContractError("invalid angular-target derivation")
    value = _object(component.get("value"), "angular_target_history.value")
    _exact(value, {"focus_segment_id", "hid_intervals", "anchors", "samples",
                   "source_artifact_sha256s"}, "angular_target_history.value")
    _string(value.get("focus_segment_id"), "angular_target_history.focus_segment_id")
    intervals = [validate_hid_interval(item) for item in
                 _array(value.get("hid_intervals"), "angular_target_history.hid_intervals")]
    for previous, following in zip(intervals, intervals[1:]):
        if following["start_monotonic_ns"] < previous["end_monotonic_ns"]:
            raise TotalCaptureContractError("HID integration intervals overlap/reorder")
    anchors = _array(value.get("anchors"), "angular_target_history.anchors")
    samples = _array(value.get("samples"), "angular_target_history.samples")
    if len(anchors) < 2 or len(samples) < 2:
        raise TotalCaptureContractError("angular trajectory needs at least two anchors/samples")
    last_ns = -1
    anchor_values: dict[int, tuple[float, float, float]] = {}
    for raw in anchors:
        anchor = _object(raw, "angular target anchor")
        _exact(anchor, {"clock", "angle"}, "angular target anchor")
        clock = _object(anchor.get("clock"), "angular target anchor clock")
        _exact(clock, {"monotonic_ns", "engine_tick", "source_sequence"},
               "angular target anchor clock")
        now = _integer(clock.get("monotonic_ns"), "anchor.monotonic_ns")
        if now <= last_ns:
            raise TotalCaptureContractError("angular anchors are not strictly ordered")
        last_ns = now
        _integer(clock.get("engine_tick"), "anchor.engine_tick")
        _integer(clock.get("source_sequence"), "anchor.source_sequence")
        validate_view_angle_anchor(anchor.get("angle"))
        anchor_values[now] = tuple(float(anchor["angle"][field]) for field in
                                   ("yaw_degrees", "pitch_degrees", "roll_degrees"))
    last_ns = -1
    sample_values: dict[int, tuple[float, float, float]] = {}
    for raw in samples:
        sample = _object(raw, "angular target sample")
        _exact(sample, {"monotonic_ns", "yaw_degrees", "pitch_degrees",
                        "roll_degrees"}, "angular target sample")
        now = _integer(sample.get("monotonic_ns"), "sample.monotonic_ns")
        if now <= last_ns:
            raise TotalCaptureContractError("trajectory samples are not strictly ordered")
        last_ns = now
        for field in ("yaw_degrees", "pitch_degrees", "roll_degrees"):
            _finite(sample.get(field), f"sample.{field}")
        sample_values[now] = tuple(float(sample[field]) for field in
                                   ("yaw_degrees", "pitch_degrees", "roll_degrees"))
    if any(sample_values.get(timestamp) != angle
           for timestamp, angle in anchor_values.items()):
        raise TotalCaptureContractError(
            "trajectory samples do not exactly pass through every anchor")
    if intervals and (intervals[0]["start_monotonic_ns"] <
                      anchors[0]["clock"]["monotonic_ns"] or
                      intervals[-1]["end_monotonic_ns"] >
                      anchors[-1]["clock"]["monotonic_ns"]):
        raise TotalCaptureContractError("HID intervals escape anchor horizon")
    hashes = _array(value.get("source_artifact_sha256s"),
                    "angular_target_history.source_artifact_sha256s")
    if not hashes:
        raise TotalCaptureContractError("angular trajectory has no source evidence")
    for digest in hashes:
        _hash(digest, "angular trajectory source hash")
    if len(set(hashes)) != len(hashes):
        raise TotalCaptureContractError("duplicate angular trajectory source hash")
    canonical_json(component)
    return copy.deepcopy(dict(component))


def validate_recorded_component(name: str,
                                envelope: Mapping[str, object]) -> dict[str, object]:
    """Validate the invariant portion of renderer-facing captured components."""
    if name not in CANONICAL_COMPONENTS:
        raise TotalCaptureContractError(f"unknown total-capture component {name!r}")
    component = _object(envelope, f"component {name}")
    if name == "angular_target_history":
        return validate_angular_target_history(component)
    _exact(component, {"provenance", "value"}, f"component {name}")
    if component.get("provenance") != "recorded":
        raise TotalCaptureContractError(
            f"declared captured component {name!r} must be recorded")
    value = _object(component.get("value"), f"component {name}.value")
    if name == "camera":
        _exact(value, {"pov_id", "vantage", "origin", "yaw_degrees",
                       "pitch_degrees", "roll_degrees", "fov", "projection",
                       "near", "far", "prediction", "clock", "view_matrix_4x4",
                       "projection_matrix_4x4", "viewport", "matrix_convention"},
               "camera.value")
        for field in ("pov_id", "vantage", "projection"):
            _string(value.get(field), f"camera.{field}")
        _vector(value.get("origin"), 3, "camera.origin")
        for field in ("yaw_degrees", "pitch_degrees", "roll_degrees", "fov",
                      "near", "far"):
            _finite(value.get(field), f"camera.{field}")
        _vector(value.get("view_matrix_4x4"), 16, "camera.view_matrix_4x4")
        _vector(value.get("projection_matrix_4x4"), 16,
                "camera.projection_matrix_4x4")
        viewport = _array(value.get("viewport"), "camera.viewport")
        if len(viewport) != 2:
            raise TotalCaptureContractError("camera.viewport must be [width,height]")
        for dimension in viewport:
            _integer(dimension, "camera.viewport[]", 1)
        convention = _object(value.get("matrix_convention"),
                             "camera.matrix_convention")
        _exact(convention, {"layout", "vectors", "view_handedness",
                            "view_axis_order", "clip_depth_range",
                            "ndc_y_direction", "view_transform_direction"},
               "camera.matrix_convention")
        enums = {
            "layout": {"row-major", "column-major"},
            "vectors": {"row-vectors", "column-vectors"},
            "view_handedness": {"left-handed", "right-handed"},
            "clip_depth_range": {"zero-to-one", "negative-one-to-one"},
            "ndc_y_direction": {"up", "down"},
            "view_transform_direction": {"world-to-view", "view-to-world"},
        }
        for field, choices in enums.items():
            if convention.get(field) not in choices:
                raise TotalCaptureContractError(
                    f"camera.matrix_convention.{field} is invalid")
        axes = _array(convention.get("view_axis_order"),
                      "camera.matrix_convention.view_axis_order")
        if len(axes) != 3 or set(axes) != {"right", "up", "forward"}:
            raise TotalCaptureContractError(
                "view_axis_order must be a permutation of right/up/forward")
        clock = _object(value.get("clock"), "camera.clock")
        _exact(clock, {"monotonic_ns", "engine_tick", "source_sequence"},
               "camera.clock")
        for field in ("monotonic_ns", "engine_tick", "source_sequence"):
            _integer(clock.get(field), f"camera.clock.{field}")
        prediction = _object(value.get("prediction"), "camera.prediction")
        _exact(prediction, {"mode", "command_number", "predicted_tick",
                            "source_sequence", "prediction_error", "corrected"},
               "camera.prediction")
        _string(prediction.get("mode"), "camera.prediction.mode")
        for field in ("command_number", "predicted_tick", "source_sequence"):
            _integer(prediction.get(field), f"camera.prediction.{field}")
        _vector(prediction.get("prediction_error"), 3,
                "camera.prediction.prediction_error")
        if not isinstance(prediction.get("corrected"), bool):
            raise TotalCaptureContractError("camera.prediction.corrected must be boolean")
    elif name == "final_pose":
        _exact(value, {"space", "model_content_sha256", "skeleton_id",
                       "pose_sequence", "bones", "attachments"},
               "final_pose.value")
        if value.get("space") not in {"world", "view"}:
            raise TotalCaptureContractError("final_pose.space must be world|view")
        _hash(value.get("model_content_sha256"), "final_pose.model_content_sha256")
        _string(value.get("skeleton_id"), "final_pose.skeleton_id")
        _integer(value.get("pose_sequence"), "final_pose.pose_sequence")
        bones = _array(value.get("bones"), "final_pose.bones")
        bone_names: set[str] = set()
        for bone_index, bone in enumerate(bones):
            bone = _object(bone, "bone")
            _exact(bone, {"name", "parent", "matrix_3x4"}, "bone")
            bone_name = _string(bone.get("name"), "bone.name")
            if bone_name in bone_names:
                raise TotalCaptureContractError("duplicate final-pose bone name")
            bone_names.add(bone_name)
            parent = _integer(bone.get("parent"), "bone.parent", -1)
            if parent >= bone_index:
                raise TotalCaptureContractError("bone parent must precede child")
            _vector(bone.get("matrix_3x4"), 12, "bone.matrix_3x4")
        attachment_names: set[str] = set()
        for attachment in _array(value.get("attachments"), "final_pose.attachments"):
            attachment = _object(attachment, "attachment")
            _exact(attachment, {"name", "bone", "matrix_3x4"}, "attachment")
            attachment_name = _string(attachment.get("name"), "attachment.name")
            if attachment_name in attachment_names:
                raise TotalCaptureContractError("duplicate attachment name")
            attachment_names.add(attachment_name)
            bone_name = _string(attachment.get("bone"), "attachment.bone")
            if bone_name not in bone_names:
                raise TotalCaptureContractError("attachment references unknown bone")
            _vector(attachment.get("matrix_3x4"), 12,
                    "attachment.matrix_3x4")
    elif name == "viewmodel":
        _exact(value, {"pov_id", "handedness", "viewmodel_fov", "projection",
                       "near", "far", "projection_matrix_4x4"},
               "viewmodel.value")
        _string(value.get("pov_id"), "viewmodel.pov_id")
        if value.get("handedness") not in {"left", "right", "ambidextrous"}:
            raise TotalCaptureContractError("invalid viewmodel.handedness")
        _finite(value.get("viewmodel_fov"), "viewmodel.viewmodel_fov")
        _string(value.get("projection"), "viewmodel.projection")
        _finite(value.get("near"), "viewmodel.near")
        _finite(value.get("far"), "viewmodel.far")
        _vector(value.get("projection_matrix_4x4"), 16,
                "viewmodel.projection_matrix_4x4")
    elif name == "model_binding":
        _exact(value, {"engine", "model_id", "model_content_sha256",
                       "material_set", "skin", "bodygroups", "lod"},
               "model_binding.value")
        _string(value.get("engine"), "model_binding.engine")
        _string(value.get("model_id"), "model_binding.model_id")
        _hash(value.get("model_content_sha256"),
              "model_binding.model_content_sha256")
        _string(value.get("material_set"), "model_binding.material_set")
        _integer(value.get("skin"), "model_binding.skin")
        _integer(value.get("lod"), "model_binding.lod")
        for bodygroup in _array(value.get("bodygroups"),
                                "model_binding.bodygroups"):
            _integer(bodygroup, "model_binding.bodygroups[]")
    elif name == "rigid_body":
        _exact(value, {"body_id", "transform_3x4", "linear_velocity",
                       "angular_velocity", "sleeping"}, "rigid_body.value")
        _string(value.get("body_id"), "rigid_body.body_id")
        _vector(value.get("transform_3x4"), 12, "rigid_body.transform_3x4")
        _vector(value.get("linear_velocity"), 3, "rigid_body.linear_velocity")
        _vector(value.get("angular_velocity"), 3, "rigid_body.angular_velocity")
        if not isinstance(value.get("sleeping"), bool):
            raise TotalCaptureContractError("rigid_body.sleeping must be boolean")
    elif name == "world_transform":
        _exact(value, {"matrix_3x4"}, "world_transform.value")
        _vector(value.get("matrix_3x4"), 12, "world_transform.matrix_3x4")
    elif name == "ragdoll":
        _exact(value, {"source_entity_id", "active"}, "ragdoll.value")
        _string(value.get("source_entity_id"), "ragdoll.source_entity_id")
        if not isinstance(value.get("active"), bool):
            raise TotalCaptureContractError("ragdoll.active must be boolean")
    elif name == "constraints":
        _exact(value, {"items"}, "constraints.value")
        constraint_ids: set[str] = set()
        for item in _array(value.get("items"), "constraints.items"):
            item = _object(item, "constraint")
            _exact(item, {"constraint_id", "kind", "body_a", "body_b",
                          "frame_a_3x4", "frame_b_3x4", "enabled", "parameters"},
                   "constraint")
            constraint_id = _string(item.get("constraint_id"),
                                    "constraint.constraint_id")
            if constraint_id in constraint_ids:
                raise TotalCaptureContractError("duplicate constraint id")
            constraint_ids.add(constraint_id)
            for field in ("kind", "body_a", "body_b"):
                _string(item.get(field), f"constraint.{field}")
            _vector(item.get("frame_a_3x4"), 12, "constraint.frame_a_3x4")
            _vector(item.get("frame_b_3x4"), 12, "constraint.frame_b_3x4")
            if not isinstance(item.get("enabled"), bool):
                raise TotalCaptureContractError("constraint.enabled must be boolean")
            _object(item.get("parameters"), "constraint.parameters")
    elif name == "effects":
        _exact(value, {"instances"}, "effects.value")
        effect_ids: set[str] = set()
        for item in _array(value.get("instances"), "effects.instances"):
            item = _object(item, "effect")
            _exact(item, {"effect_id", "kind", "active", "transform_3x4",
                          "resource_content_sha256", "parameters"}, "effect")
            effect_id = _string(item.get("effect_id"), "effect.effect_id")
            if effect_id in effect_ids:
                raise TotalCaptureContractError("duplicate effect id")
            effect_ids.add(effect_id)
            _string(item.get("kind"), "effect.kind")
            if not isinstance(item.get("active"), bool):
                raise TotalCaptureContractError("effect.active must be boolean")
            _vector(item.get("transform_3x4"), 12, "effect.transform_3x4")
            if item.get("resource_content_sha256") is not None:
                _hash(item.get("resource_content_sha256"),
                      "effect.resource_content_sha256")
            _object(item.get("parameters"), "effect.parameters")
    elif name == "diegetic_ui":
        _exact(value, {"pov_id", "viewport", "elements"}, "diegetic_ui.value")
        _string(value.get("pov_id"), "diegetic_ui.pov_id")
        _vector(value.get("viewport"), 2, "diegetic_ui.viewport")
        element_ids: set[str] = set()
        for item in _array(value.get("elements"), "diegetic_ui.elements"):
            item = _object(item, "UI element")
            _exact(item, {"element_id", "kind", "bounds", "z_order", "visible",
                          "content"}, "UI element")
            element_id = _string(item.get("element_id"), "UI element.element_id")
            if element_id in element_ids:
                raise TotalCaptureContractError("duplicate UI element id")
            element_ids.add(element_id)
            _string(item.get("kind"), "UI element.kind")
            _vector(item.get("bounds"), 4, "UI element.bounds")
            _integer(item.get("z_order"), "UI element.z_order", -(2 ** 31))
            if not isinstance(item.get("visible"), bool):
                raise TotalCaptureContractError("UI element.visible must be boolean")
            _object(item.get("content"), "UI element.content")
    elif name == "visibility":
        _exact(value, {"pov_id", "visible_entity_ids", "method"},
               "visibility.value")
        _string(value.get("pov_id"), "visibility.pov_id")
        _string(value.get("method"), "visibility.method")
        visible = _array(value.get("visible_entity_ids"),
                         "visibility.visible_entity_ids")
        for entity_id in visible:
            _string(entity_id, "visible entity id")
        if len(set(visible)) != len(visible):
            raise TotalCaptureContractError("visibility contains duplicate entity ids")
    elif name == "render_history":
        _exact(value, {"pov_id", "samples"}, "render_history.value")
        _string(value.get("pov_id"), "render_history.pov_id")
        last_clock = -1
        for item in _array(value.get("samples"), "render_history.samples"):
            item = _object(item, "render-history sample")
            _exact(item, {"clock", "previous_view_projection_matrix_4x4",
                          "jitter", "exposure", "motion_blur_scale"},
                   "render-history sample")
            clock = _object(item.get("clock"), "render-history clock")
            _exact(clock, {"monotonic_ns", "engine_tick", "source_sequence"},
                   "render-history clock")
            now = _integer(clock.get("monotonic_ns"), "render-history monotonic_ns")
            if now <= last_clock:
                raise TotalCaptureContractError("render history is not ordered")
            last_clock = now
            _integer(clock.get("engine_tick"), "render-history engine_tick")
            _integer(clock.get("source_sequence"), "render-history source_sequence")
            _vector(item.get("previous_view_projection_matrix_4x4"), 16,
                    "previous_view_projection_matrix_4x4")
            _vector(item.get("jitter"), 2, "render-history jitter")
            _finite(item.get("exposure"), "render-history exposure")
            _finite(item.get("motion_blur_scale"), "render-history motion_blur_scale")
    canonical_json(component)
    return copy.deepcopy(dict(component))


__all__ = [name for name in globals() if name.isupper()] + [
    "TotalCaptureContractError", "canonical_json", "document_sha256",
    "capture_completeness_document", "pov_entity_id", "seal_document",
    "incomplete_capture_document",
    "validate_angular_target_history", "validate_hid_interval",
    "validate_capture_completeness", "validate_manifest", "validate_record",
    "validate_recorded_component", "validate_sidecar_terminal",
    "validate_view_angle_anchor",
]
