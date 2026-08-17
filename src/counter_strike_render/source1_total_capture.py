"""Strict renderer-side join for canonical total-capture snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from state_replay.total_capture_contract import (
    CANONICAL_COMPONENTS,
    CAPTURE_COMPLETENESS_COMPONENT,
    COMPLETENESS_DERIVATION,
    TOTAL_CAPTURE_MANIFEST_COMPONENT,
    TOTAL_CAPTURE_METADATA_ENTITY_ID,
    TotalCaptureContractError,
    pov_entity_id,
    validate_capture_completeness,
    validate_manifest,
    validate_recorded_component,
)
from state_replay.total_capture_collection_contract import (
    TOTAL_CAPTURE_EPOCH_CONTEXT_COMPONENT,
    TOTAL_CAPTURE_EPOCH_CONTEXT_ENTITY_ID,
    epoch_stream_id,
    validate_epoch_context_component,
)

from .replay_bridge import RenderEntityRecord, RenderFrame


@dataclass(frozen=True)
class TotalCapturePov:
    pov_id: str
    record: RenderEntityRecord
    camera: Mapping[str, Any]
    visibility: Mapping[str, Any]
    diegetic_ui: Mapping[str, Any]
    render_history: Mapping[str, Any]
    angular_target_history: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TotalCaptureFrame:
    manifest: Mapping[str, Any]
    completeness: Mapping[str, Any]
    povs: tuple[TotalCapturePov, ...]


@dataclass(frozen=True)
class TotalCaptureEpochContext:
    collection_id: str
    collection_manifest_sha256: str
    ordinal: int
    process_epoch_id: str
    map_epoch_id: str
    capture_epoch_id: str
    map_name: str
    capture_manifest_sha256: str
    demo_playback_ns: int
    status: str
    host_horizon: Mapping[str, Any]
    reset: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "collection_manifest_sha256": self.collection_manifest_sha256,
            "ordinal": self.ordinal,
            "process_epoch_id": self.process_epoch_id,
            "map_epoch_id": self.map_epoch_id,
            "capture_epoch_id": self.capture_epoch_id,
            "map_name": self.map_name,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "demo_playback_ns": self.demo_playback_ns,
            "status": self.status,
            "host_horizon": dict(self.host_horizon),
            "reset": dict(self.reset),
        }


def inspect_epoch_context(
    frame: RenderFrame,
    collection_manifest: Mapping[str, Any] | None = None,
) -> TotalCaptureEpochContext | None:
    matches = [
        item for item in frame.entities
        if item.entity_id == TOTAL_CAPTURE_EPOCH_CONTEXT_ENTITY_ID
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise TotalCaptureContractError(
            "frame contains duplicate total-capture epoch contexts"
        )
    record = matches[0]
    if record.entity_class != "TotalCaptureEpochContext":
        raise TotalCaptureContractError(
            "total-capture:epoch-context has the wrong entity class"
        )
    component = record.components.get(TOTAL_CAPTURE_EPOCH_CONTEXT_COMPONENT)
    if component is None:
        raise TotalCaptureContractError(
            "epoch context entity is missing total_capture_epoch_context"
        )
    checked = validate_epoch_context_component(
        component.to_dict(), collection_manifest
    )
    value = checked["value"]
    ordinal = int(value["ordinal"])
    if record.generation != ordinal:
        raise TotalCaptureContractError(
            "epoch context entity generation must equal epoch ordinal"
        )
    expected_stream = epoch_stream_id(
        str(value["collection_id"]), ordinal, str(value["capture_epoch_id"])
    )
    if frame.stream_id != expected_stream:
        raise TotalCaptureContractError(
            "frame stream_id does not equal its capture epoch stream"
        )
    return TotalCaptureEpochContext(
        str(value["collection_id"]),
        str(value["collection_manifest_sha256"]),
        ordinal,
        str(value["process_epoch_id"]),
        str(value["map_epoch_id"]),
        str(value["capture_epoch_id"]),
        str(value["map_name"]),
        str(value["capture_manifest_sha256"]),
        int(value["demo_playback_ns"]),
        str(value["status"]),
        value["host_horizon"],
        value["reset"],
    )


def inspect_total_capture(frame: RenderFrame) -> TotalCaptureFrame | None:
    metadata = next(
        (item for item in frame.entities
         if item.entity_id == TOTAL_CAPTURE_METADATA_ENTITY_ID),
        None,
    )
    if metadata is None:
        return None
    if metadata.entity_class != "TotalCaptureMetadata":
        raise TotalCaptureContractError(
            "total-capture:metadata must have class TotalCaptureMetadata"
        )
    manifest_component = metadata.components.get(TOTAL_CAPTURE_MANIFEST_COMPONENT)
    completeness_component = metadata.components.get(CAPTURE_COMPLETENESS_COMPONENT)
    if manifest_component is None or manifest_component.provenance != "recorded":
        raise TotalCaptureContractError(
            "total_capture_manifest must be a recorded component"
        )
    if not isinstance(manifest_component.value, Mapping):
        raise TotalCaptureContractError("total_capture_manifest.value must be an object")
    manifest = validate_manifest(manifest_component.value)
    if (
        completeness_component is None
        or completeness_component.provenance != "derived"
        or completeness_component.derivation != COMPLETENESS_DERIVATION
    ):
        raise TotalCaptureContractError(
            "capture_completeness must use the exact validated closure derivation"
        )
    if not isinstance(completeness_component.value, Mapping):
        raise TotalCaptureContractError("capture_completeness.value must be an object")
    completeness = validate_capture_completeness(
        manifest, completeness_component.value
    )

    records = {record.entity_id: record for record in frame.entities}
    povs: list[TotalCapturePov] = []
    for pov_id in manifest["scored_povs"]:
        entity_id = pov_entity_id(str(pov_id))
        record = records.get(entity_id)
        if record is None:
            raise TotalCaptureContractError(
                f"scored POV {pov_id!r} has no canonical entity {entity_id!r}"
            )
        values: dict[str, Mapping[str, Any]] = {}
        for name in ("camera", "visibility", "diegetic_ui", "render_history"):
            component = record.components.get(name)
            if component is None:
                raise TotalCaptureContractError(
                    f"scored POV {pov_id!r} is missing component {name!r}"
                )
            checked = validate_recorded_component(name, component.to_dict())
            value = checked["value"]
            if value.get("pov_id") != pov_id:
                raise TotalCaptureContractError(
                    f"component {name!r} does not join to POV {pov_id!r}"
                )
            values[name] = value
        angular = record.components.get("angular_target_history")
        angular_value = None
        if angular is not None:
            angular_value = validate_recorded_component(
                "angular_target_history", angular.to_dict()
            )["value"]
        povs.append(TotalCapturePov(
            str(pov_id), record, values["camera"], values["visibility"],
            values["diegetic_ui"], values["render_history"], angular_value,
        ))

    expected_pov_ids = {pov_entity_id(str(item)) for item in manifest["scored_povs"]}
    for record in frame.entities:
        if str(record.entity_id).startswith("total-capture:pov:"):
            if record.entity_id not in expected_pov_ids:
                raise TotalCaptureContractError(
                    f"undeclared scored POV entity {record.entity_id!r}"
                )
        for name, component in record.components.items():
            if name in CANONICAL_COMPONENTS:
                validate_recorded_component(name, component.to_dict())
    return TotalCaptureFrame(manifest, completeness, tuple(povs))


def require_closed_horizon(frame: RenderFrame, capture: TotalCaptureFrame) -> None:
    if capture.completeness.get("status") != "complete":
        raise TotalCaptureContractError(
            "total-capture authoritative rendering requires complete stream closure"
        )
    horizon = capture.manifest["horizon"]
    if not int(horizon["start_tick"]) <= frame.tick <= int(horizon["end_tick"]):
        raise TotalCaptureContractError(
            "render frame lies outside the declared total-capture horizon"
        )
    if frame.discontinuity is not None:
        raise TotalCaptureContractError(
            "total-capture authoritative horizon contains a discontinuity"
        )


__all__ = [
    "TotalCaptureEpochContext",
    "TotalCaptureFrame",
    "TotalCapturePov",
    "inspect_total_capture",
    "inspect_epoch_context",
    "require_closed_horizon",
]
