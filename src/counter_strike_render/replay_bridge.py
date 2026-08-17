"""Strict state-snapshot boundary for the Counter-Strike renderer.

The bridge deliberately does not integrate, interpolate, or repair state.  It
accepts full ``tardigrade/state-snapshot/v1`` snapshots and turns them into
typed, renderer-oriented batches while retaining the provenance of every
component.  State integration belongs on the other side of this boundary.

Snapshot entities use this shape::

    {
        "id": "entity-stable-id",
        "generation": 3,
        "class": "CCSPlayerPawn",
        "components": {
            "player": {"provenance": "recorded", "value": {...}},
            "animation": {
                "provenance": "recorded",
                "value": {"evaluated_bones": [...]},
            },
        },
    }

``derived`` components must name their derivation. ``unavailable`` components
must name the reason and carry no value.  A component field may itself use the
same envelope, which permits (for example) a recorded physics component with a
derived ``world_transform`` to be represented honestly.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "tardigrade/state-snapshot/v1"
_PROVENANCE = frozenset(("recorded", "derived", "unavailable"))
_DISCONTINUITIES = frozenset(("gap", "seek", "reset", "source-switch"))


class BridgeMode(str, Enum):
    """Fidelity policy applied at the state-to-render boundary."""

    AUTHORITATIVE = "authoritative"
    CANONICAL = "canonical"


class SnapshotValidationError(ValueError):
    """A snapshot violates the normalized input contract."""


class StateContinuityError(SnapshotValidationError):
    """Snapshot ordering, hash chaining, or entity identity is incoherent."""


class AuthoritativeStateUnavailable(SnapshotValidationError):
    """Authoritative rendering was requested without authoritative pose data."""


class LegacyRendererUnsupported(AuthoritativeStateUnavailable):
    """The legacy renderer cannot consume every authoritative input.

    ``manifest`` is the same machine-readable document written to
    ``manifest_path``.  Camera and playermodel inputs are not written when
    this exception is raised.
    """

    def __init__(
        self, message: str, manifest: Mapping[str, Any], manifest_path: Path
    ) -> None:
        super().__init__(message)
        self.manifest = manifest
        self.manifest_path = manifest_path


@dataclass(frozen=True)
class ProvenancedValue:
    """A value together with its epistemic status at capture time."""

    provenance: str
    value: Any = None
    reason: str | None = None
    derivation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"provenance": self.provenance}
        if self.provenance != "unavailable":
            result["value"] = copy.deepcopy(self.value)
        if self.reason is not None:
            result["reason"] = self.reason
        if self.derivation is not None:
            result["derivation"] = self.derivation
        result.update(copy.deepcopy(dict(self.metadata)))
        return result


@dataclass(frozen=True)
class Discontinuity:
    kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "reason": self.reason}


@dataclass(frozen=True)
class RenderEntityRecord:
    """One entity routed to a renderer subsystem for a particular snapshot."""

    stream_id: str
    tick: int
    subtick: int
    entity_id: str | int
    generation: int
    entity_class: str
    primary_component: str
    components: Mapping[str, ProvenancedValue]

    @property
    def primary(self) -> ProvenancedValue:
        return self.components[self.primary_component]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "tick": self.tick,
            "subtick": self.subtick,
            "entity_id": self.entity_id,
            "generation": self.generation,
            "class": self.entity_class,
            "primary_component": self.primary_component,
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            },
        }


@dataclass(frozen=True)
class RenderFrame:
    """Validated renderer inputs for a single authoritative state snapshot."""

    stream_id: str
    tick: int
    subtick: int
    state_hash: str
    previous_state_hash: str | None
    mode: BridgeMode
    discontinuity: Discontinuity | None
    entities: tuple[RenderEntityRecord, ...]
    cameras: tuple[RenderEntityRecord, ...]
    players: tuple[RenderEntityRecord, ...]
    dynamic_physics: tuple[RenderEntityRecord, ...]
    animations: tuple[RenderEntityRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "stream_id": self.stream_id,
            "tick": self.tick,
            "subtick": self.subtick,
            "state_hash": self.state_hash,
            "previous_state_hash": self.previous_state_hash,
            "mode": self.mode.value,
            "discontinuity": (
                self.discontinuity.to_dict() if self.discontinuity else None
            ),
            "entities": [entity.to_dict() for entity in self.entities],
            "cameras": [entity.to_dict() for entity in self.cameras],
            "players": [entity.to_dict() for entity in self.players],
            "dynamic_physics": [
                entity.to_dict() for entity in self.dynamic_physics
            ],
            "animations": [entity.to_dict() for entity in self.animations],
        }


@dataclass(frozen=True)
class RenderBatch:
    """An ordered set of frames, with convenient subsystem-major views."""

    frames: tuple[RenderFrame, ...]

    @property
    def cameras(self) -> tuple[RenderEntityRecord, ...]:
        return tuple(record for frame in self.frames for record in frame.cameras)

    @property
    def players(self) -> tuple[RenderEntityRecord, ...]:
        return tuple(record for frame in self.frames for record in frame.players)

    @property
    def dynamic_physics(self) -> tuple[RenderEntityRecord, ...]:
        return tuple(
            record for frame in self.frames for record in frame.dynamic_physics
        )

    @property
    def animations(self) -> tuple[RenderEntityRecord, ...]:
        return tuple(record for frame in self.frames for record in frame.animations)

    def to_dict(self) -> dict[str, Any]:
        return {"frames": [frame.to_dict() for frame in self.frames]}


@dataclass(frozen=True)
class LegacyStageResult:
    """Paths and gap accounting produced by :func:`stage_legacy_inputs`."""

    camera_json: Path
    playermodel_json: Path
    gap_manifest_json: Path
    gap_manifest: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_json": str(self.camera_json),
            "playermodel_json": str(self.playermodel_json),
            "gap_manifest_json": str(self.gap_manifest_json),
            "gap_manifest": copy.deepcopy(dict(self.gap_manifest)),
        }


@dataclass
class _StreamCursor:
    position: tuple[int, int] | None = None
    state_hash: str | None = None
    generations: dict[str | int, int] = field(default_factory=dict)
    active: set[str | int] = field(default_factory=set)


class StateToRenderBridge:
    """Validate snapshots and route their components to renderer subsystems.

    ``authoritative`` with ``world_pose=True`` requires each dynamic physics
    object to have a recorded ``world_transform`` and every player/animated
    entity to have either recorded evaluated bones or recorded complete graph
    state.  ``canonical`` preserves derived and unavailable values verbatim.

    A bridge instance retains a cursor per stream so calls to :meth:`convert`
    and :meth:`batch` validate continuity across call boundaries.  Use
    :meth:`reset` only when the caller deliberately starts a new session.
    """

    def __init__(
        self,
        mode: BridgeMode | str = BridgeMode.CANONICAL,
        *,
        world_pose: bool = True,
    ) -> None:
        try:
            self.mode = BridgeMode(mode)
        except ValueError as exc:
            raise ValueError("mode must be 'authoritative' or 'canonical'") from exc
        self.world_pose = bool(world_pose)
        self._streams: dict[str, _StreamCursor] = {}

    def reset(self, stream_id: str | None = None) -> None:
        """Forget validation cursors for all streams or one named stream."""

        if stream_id is None:
            self._streams.clear()
        else:
            self._streams.pop(stream_id, None)

    def batch(self, snapshots: Iterable[Mapping[str, Any]]) -> RenderBatch:
        """Convert snapshots atomically; validation failure advances no cursor."""

        old_streams = copy.deepcopy(self._streams)
        try:
            frames = tuple(self._convert(snapshot) for snapshot in snapshots)
        except Exception:
            self._streams = old_streams
            raise
        return RenderBatch(frames)

    def convert(self, snapshot: Mapping[str, Any]) -> RenderFrame:
        """Validate and convert one normalized snapshot atomically."""

        return self._convert(snapshot)

    def _convert(self, snapshot: Mapping[str, Any]) -> RenderFrame:
        """Implementation for :meth:`convert`; callers provide rollback."""

        root = _mapping(snapshot, "snapshot")
        if root.get("schema") != SCHEMA:
            raise SnapshotValidationError(f"snapshot.schema must equal {SCHEMA!r}")

        stream_id = _nonempty_string(root.get("stream_id"), "snapshot.stream_id")
        tick = _integer(root.get("tick"), "snapshot.tick", minimum=0)
        subtick = _subtick(root.get("subtick"), "snapshot.subtick")
        state_hash = _nonempty_string(root.get("state_hash"), "snapshot.state_hash")
        previous_state_hash = root.get("previous_state_hash")
        if previous_state_hash is not None:
            previous_state_hash = _nonempty_string(
                previous_state_hash, "snapshot.previous_state_hash"
            )
        discontinuity = _discontinuity(root.get("discontinuity"))

        raw_entities = root.get("entities")
        if not isinstance(raw_entities, Sequence) or isinstance(
            raw_entities, (str, bytes, bytearray)
        ):
            raise SnapshotValidationError("snapshot.entities must be an array")

        # A discontinuity begins a fresh identity domain.  Keep the candidate
        # cursor detached until all validation (including fidelity policy) has
        # succeeded, so a rejected frame cannot advance stream state.
        cursor = (
            _StreamCursor()
            if discontinuity is not None
            else self._streams.get(stream_id, _StreamCursor())
        )

        position = (tick, subtick)
        if cursor.position is not None and position <= cursor.position:
            raise StateContinuityError(
                f"stream {stream_id!r} did not advance: {position!r} is not after "
                f"{cursor.position!r}; an explicit discontinuity is required"
            )
        if (
            previous_state_hash is not None
            and cursor.state_hash is not None
            and previous_state_hash != cursor.state_hash
        ):
            raise StateContinuityError(
                "snapshot.previous_state_hash does not match the preceding state_hash"
            )

        records: list[RenderEntityRecord] = []
        present: set[str | int] = set()
        generation_updates: dict[str | int, int] = {}
        for index, raw_entity in enumerate(raw_entities):
            path = f"snapshot.entities[{index}]"
            entity = _mapping(raw_entity, path)
            entity_id = _entity_id(entity.get("id"), f"{path}.id")
            if entity_id in present:
                raise StateContinuityError(
                    f"{path}.id duplicates entity id {entity_id!r} in this snapshot"
                )
            present.add(entity_id)
            generation = _integer(
                entity.get("generation"), f"{path}.generation", minimum=0
            )
            entity_class = _nonempty_string(entity.get("class"), f"{path}.class")
            components = _components(entity.get("components"), f"{path}.components")

            previous_generation = cursor.generations.get(entity_id)
            if previous_generation is not None:
                if generation < previous_generation:
                    raise StateContinuityError(
                        f"{path}.generation regressed from {previous_generation} "
                        f"to {generation}"
                    )
                if generation == previous_generation and entity_id not in cursor.active:
                    raise StateContinuityError(
                        f"{path} reuses inactive generation {generation}; increment the "
                        "generation or declare a discontinuity"
                    )

            base = dict(
                stream_id=stream_id,
                tick=tick,
                subtick=subtick,
                entity_id=entity_id,
                generation=generation,
                entity_class=entity_class,
                components=components,
            )
            primary = _primary_component(components)
            records.append(RenderEntityRecord(primary_component=primary, **base))
            generation_updates[entity_id] = generation

        cameras = _reroute(records, "camera")
        players = _reroute(records, "player")
        physics = _route_dynamic_physics(records)
        animations = _reroute(records, "animation")

        if self.mode is BridgeMode.AUTHORITATIVE and self.world_pose:
            _require_authoritative_world_pose(players, physics, animations)

        cursor.generations.update(generation_updates)
        cursor.active = present
        cursor.position = position
        cursor.state_hash = state_hash
        self._streams[stream_id] = cursor

        return RenderFrame(
            stream_id=stream_id,
            tick=tick,
            subtick=subtick,
            state_hash=state_hash,
            previous_state_hash=previous_state_hash,
            mode=self.mode,
            discontinuity=discontinuity,
            entities=tuple(records),
            cameras=cameras,
            players=players,
            dynamic_physics=physics,
            animations=animations,
        )


def snapshots_to_render_batch(
    snapshots: Iterable[Mapping[str, Any]],
    *,
    mode: BridgeMode | str = BridgeMode.CANONICAL,
    world_pose: bool = True,
) -> RenderBatch:
    """Stateless convenience interface for converting one independent batch."""

    return StateToRenderBridge(mode, world_pose=world_pose).batch(snapshots)


class _GapCollector:
    """Aggregate repeated legacy downgrades without producing hour-sized logs."""

    def __init__(self) -> None:
        self._items: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(
        self,
        frame: RenderFrame,
        code: str,
        detail: str,
        *,
        record: RenderEntityRecord | None = None,
        component: str | None = None,
        field_name: str | None = None,
        provenance: str | None = None,
        legacy_consumer: str | None = None,
    ) -> None:
        entity_id = record.entity_id if record is not None else None
        generation = record.generation if record is not None else None
        key = (
            code,
            frame.stream_id,
            entity_id,
            generation,
            component,
            field_name,
            provenance,
            legacy_consumer,
            detail,
        )
        occurrence = {
            "tick": frame.tick,
            "subtick": frame.subtick,
            "state_hash": frame.state_hash,
        }
        item = self._items.get(key)
        if item is None:
            item = {
                "code": code,
                "stream_id": frame.stream_id,
                "entity_id": entity_id,
                "generation": generation,
                "component": component,
                "field": field_name,
                "provenance": provenance,
                "legacy_consumer": legacy_consumer,
                "detail": detail,
                "occurrences": 1,
                "first": occurrence,
                "last": occurrence,
            }
            self._items[key] = item
        else:
            item["occurrences"] += 1
            item["last"] = occurrence

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._items.values())


def stage_legacy_inputs(
    batch: RenderBatch,
    out_dir: str | os.PathLike[str],
    tick_rate: int,
    *,
    ego_entity_id: str | int | None = None,
) -> LegacyStageResult:
    """Stage normalized states for the current file-based legacy renderer.

    Three files are written with replace-atomic JSON writes:

    ``camera.json``
        ``iji/cs2-demo-ego-camera-path/v1`` rows consumed by
        ``gpu_render``'s camera/viewmodel path.
    ``playermodels.json``
        ``iji/cs2-demo-playermodels/v1`` rows consumed by the current
        third-person player pass.
    ``legacy_gap_manifest.json``
        Every absent, derived, ambiguous, or unsupported normalized field,
        aggregated by entity generation with exact first/last state hashes.

    The adapter only renames explicitly present fields.  It never supplies
    legacy defaults.  Canonical mode writes the supported subset and records
    every downgrade.  If any frame is authoritative, every downgrade is a
    refusal: only the gap manifest is emitted, allowing the caller to inspect
    why the legacy renderer cannot claim authoritative output.
    """

    if not isinstance(batch, RenderBatch):
        raise TypeError("batch must be a RenderBatch")
    if isinstance(tick_rate, bool) or not isinstance(tick_rate, int) \
            or tick_rate <= 0:
        raise ValueError("tick_rate must be an integer > 0")
    if ego_entity_id is not None:
        _entity_id(ego_entity_id, "ego_entity_id")

    frames = batch.frames
    streams = {frame.stream_id for frame in frames}
    if len(streams) > 1:
        raise SnapshotValidationError(
            "legacy staging accepts one stream per batch; split the batch by "
            "stream_id so overlapping integer ticks cannot be conflated"
        )
    modes = {frame.mode for frame in frames}
    if len(modes) > 1:
        raise SnapshotValidationError(
            "legacy staging refuses a batch mixing canonical and authoritative frames"
        )
    authoritative = modes == {BridgeMode.AUTHORITATIVE}

    gaps = _GapCollector()
    camera_rows: list[dict[str, Any]] = []
    player_rows: dict[str, list[dict[str, Any]]] = {}
    staged_ticks: set[int] = set()

    for frame in frames:
        consumed: dict[tuple[str | int, int, str], set[str]] = {}
        if frame.tick in staged_ticks:
            gaps.add(
                frame,
                "integer-tick-collision",
                "the legacy schemas have no subtick key; this later snapshot "
                "was not staged because it would overwrite or mix an earlier state",
            )
            _report_all_unconsumed(frame, consumed, gaps)
            continue
        staged_ticks.add(frame.tick)
        if frame.subtick != 0:
            gaps.add(
                frame,
                "subtick-not-consumed",
                "legacy camera and playermodel schemas address integer ticks only; "
                f"capture subtick ordinal {frame.subtick} is not represented",
            )

        camera_record = _select_ego_camera(frame, ego_entity_id, gaps)
        if camera_record is not None:
            row = _legacy_camera_row(frame, camera_record, consumed, gaps)
            if row is not None:
                camera_rows.append(row)

        rows: list[dict[str, Any]] = []
        for player in frame.players:
            row = _legacy_player_row(frame, player, consumed, gaps)
            if row is not None:
                rows.append(row)
        if rows:
            player_rows[str(frame.tick)] = rows

        _report_all_unconsumed(frame, consumed, gaps)

    if frames and not camera_rows:
        gaps.add(
            frames[0],
            "no-stageable-camera-rows",
            "no frame had the complete explicit camera pose required by gpu_render",
            legacy_consumer="camera",
        )

    output_root = Path(out_dir)
    camera_path = output_root / "camera.json"
    playermodel_path = output_root / "playermodels.json"
    manifest_path = output_root / "legacy_gap_manifest.json"
    mode_name = next(iter(modes)).value if modes else BridgeMode.CANONICAL.value
    stream_id = next(iter(streams)) if streams else None
    camera_doc: dict[str, Any] = {
        "schema": "iji/cs2-demo-ego-camera-path/v1",
        "variant": "state-replay-ego",
        "tick_rate": tick_rate,
        "n_ticks": len(camera_rows),
        "n_future_steered": 0,
        "ticks": camera_rows,
    }
    legacy_ego_id = _legacy_identity(ego_entity_id)
    if legacy_ego_id is not None:
        camera_doc["steamid"] = legacy_ego_id
    elif ego_entity_id is not None and frames:
        gaps.add(
            frames[0],
            "ego-identity-not-representable",
            f"ego id {ego_entity_id!r} is not an integer steamid accepted by the "
            "legacy playermodel exclusion path",
            legacy_consumer="camera.steamid",
        )

    playermodel_doc = {
        "schema": "iji/cs2-demo-playermodels/v1",
        "ticks": player_rows,
    }
    manifest: dict[str, Any] = {
        "schema": "tardigrade/legacy-render-gap-manifest/v1",
        "source_schema": SCHEMA,
        "stream_id": stream_id,
        "mode": mode_name,
        "tick_rate": tick_rate,
        "ego_entity_id": ego_entity_id,
        "outputs": {
            "camera": camera_path.name,
            "playermodels": playermodel_path.name,
        },
        "counts": {
            "input_frames": len(frames),
            "camera_rows": len(camera_rows),
            "playermodel_ticks": len(player_rows),
            "playermodel_rows": sum(len(rows) for rows in player_rows.values()),
            "gap_kinds": len(gaps.entries),
            "gap_occurrences": sum(
                int(item["occurrences"]) for item in gaps.entries
            ),
        },
        "authoritative_refusal": bool(authoritative and gaps.entries),
        "outputs_written": {
            "gap_manifest": True,
            "camera": not (authoritative and gaps.entries),
            "playermodels": not (authoritative and gaps.entries),
        },
        "gaps": gaps.entries,
    }

    # A refusal emits its evidence but does not write new camera/playermodel
    # inputs. All validation and document construction happens before this
    # first filesystem mutation.
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, manifest)
    if authoritative and gaps.entries:
        raise LegacyRendererUnsupported(
            "authoritative state cannot be staged for the legacy renderer; "
            f"{len(gaps.entries)} distinct gap(s) are recorded in {manifest_path}",
            manifest,
            manifest_path,
        )

    _atomic_json(camera_path, camera_doc)
    _atomic_json(playermodel_path, playermodel_doc)
    return LegacyStageResult(
        camera_json=camera_path,
        playermodel_json=playermodel_path,
        gap_manifest_json=manifest_path,
        gap_manifest=MappingProxyType(manifest),
    )


def _select_ego_camera(
    frame: RenderFrame,
    ego_entity_id: str | int | None,
    gaps: _GapCollector,
) -> RenderEntityRecord | None:
    candidates = (
        [record for record in frame.cameras if record.entity_id == ego_entity_id]
        if ego_entity_id is not None
        else list(frame.cameras)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        gaps.add(
            frame,
            "missing-ego-camera",
            (
                f"no camera component belongs to ego entity {ego_entity_id!r}"
                if ego_entity_id is not None
                else "the snapshot contains no camera component"
            ),
            legacy_consumer="camera",
        )
    else:
        gaps.add(
            frame,
            "ambiguous-ego-camera",
            f"{len(candidates)} camera components are present and ego_entity_id "
            "was not supplied",
            legacy_consumer="camera",
        )
    return None


_CAMERA_FIELDS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "x": (("camera", "x"), ("transform", "x")),
    "y": (("camera", "y"), ("transform", "y")),
    "z": (("camera", "z"), ("transform", "z")),
    "eye_z": (("camera", "eye_z"), ("transform", "eye_z")),
    "yaw_degrees": (
        ("camera", "yaw_degrees"),
        ("camera", "yaw"),
        ("transform", "yaw_degrees"),
        ("transform", "yaw"),
    ),
    "pitch_degrees": (
        ("camera", "pitch_degrees"),
        ("camera", "pitch"),
        ("transform", "pitch_degrees"),
        ("transform", "pitch"),
    ),
    "fov": (("camera", "fov"),),
    "fire": (("camera", "fire"), ("player", "fire")),
    "is_alive": (
        ("camera", "is_alive"),
        ("camera", "alive"),
        ("player", "is_alive"),
        ("player", "alive"),
    ),
    "active_weapon_name": (
        ("camera", "active_weapon_name"),
        ("camera", "active_weapon_id"),
        ("camera", "weapon"),
        ("inventory", "active_weapon_name"),
        ("inventory", "active_weapon_id"),
        ("inventory", "weapon"),
    ),
}

_PLAYER_FIELDS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "x": (("player", "x"), ("transform", "x")),
    "y": (("player", "y"), ("transform", "y")),
    "z": (("player", "z"), ("transform", "z")),
    "yaw": (
        ("player", "yaw"),
        ("player", "yaw_degrees"),
        ("transform", "yaw"),
        ("transform", "yaw_degrees"),
    ),
    "is_alive": (("player", "is_alive"), ("player", "alive")),
    "team": (("player", "team"),),
    "fire": (("player", "fire"),),
    "active_weapon_id": (
        ("player", "active_weapon_id"),
        ("player", "active_weapon_name"),
        ("player", "weapon"),
        ("inventory", "active_weapon_id"),
        ("inventory", "active_weapon_name"),
        ("inventory", "weapon"),
    ),
}


def _legacy_camera_row(
    frame: RenderFrame,
    record: RenderEntityRecord,
    consumed: dict[tuple[str | int, int, str], set[str]],
    gaps: _GapCollector,
) -> dict[str, Any] | None:
    row: dict[str, Any] = {"tick": frame.tick}
    required = frozenset(("x", "y", "z", "eye_z", "yaw_degrees", "pitch_degrees"))
    invalid_required = False
    for target, sources in _CAMERA_FIELDS.items():
        got = _explicit_field(frame, record, sources, target, consumed, gaps, "camera")
        if got is None:
            gaps.add(
                frame,
                "missing-legacy-field",
                f"no explicit value can drive camera.{target}",
                record=record,
                field_name=target,
                legacy_consumer="camera",
            )
            invalid_required = invalid_required or target in required
            continue
        value = _coerce_legacy_value(
            frame, record, target, got.value, gaps, consumer="camera"
        )
        if value is _INVALID:
            invalid_required = invalid_required or target in required
            continue
        row[target] = value
        if target == "active_weapon_name":
            # Both names are live legacy consumers.  They carry the identical
            # explicit weapon identity; no second value is inferred.
            row["active_weapon_id"] = value
    if invalid_required:
        gaps.add(
            frame,
            "unstageable-camera-row",
            "required camera pose fields are absent or invalid; the row was omitted",
            record=record,
            legacy_consumer="camera",
        )
        return None
    return row


def _legacy_player_row(
    frame: RenderFrame,
    record: RenderEntityRecord,
    consumed: dict[tuple[str | int, int, str], set[str]],
    gaps: _GapCollector,
) -> dict[str, Any] | None:
    legacy_id = _legacy_identity(record.entity_id)
    if legacy_id is None:
        gaps.add(
            frame,
            "entity-identity-not-representable",
            f"stable id {record.entity_id!r} is not an integer steamid accepted "
            "by the legacy playermodel consumer",
            record=record,
            legacy_consumer="playermodels.steamid",
        )
        return None
    row: dict[str, Any] = {"tick": frame.tick, "steamid": legacy_id}
    required = frozenset(("x", "y", "z", "yaw", "is_alive", "team"))
    invalid_required = False
    for target, sources in _PLAYER_FIELDS.items():
        got = _explicit_field(
            frame, record, sources, target, consumed, gaps, "playermodels"
        )
        if got is None:
            gaps.add(
                frame,
                "missing-legacy-field",
                f"no explicit value can drive playermodels.{target}",
                record=record,
                field_name=target,
                legacy_consumer="playermodels",
            )
            invalid_required = invalid_required or target in required
            continue
        value = _coerce_legacy_value(
            frame, record, target, got.value, gaps, consumer="playermodels"
        )
        if value is _INVALID:
            invalid_required = invalid_required or target in required
            continue
        row[target] = value
    if invalid_required:
        gaps.add(
            frame,
            "unstageable-playermodel-row",
            "required placement/alive/team fields are absent or invalid; the row "
            "was omitted rather than filled from a renderer default",
            record=record,
            legacy_consumer="playermodels",
        )
        return None
    return row


def _explicit_field(
    frame: RenderFrame,
    record: RenderEntityRecord,
    sources: Sequence[tuple[str, str]],
    legacy_field: str,
    consumed: dict[tuple[str | int, int, str], set[str]],
    gaps: _GapCollector,
    consumer: str,
) -> ProvenancedValue | None:
    found: list[tuple[str, str, ProvenancedValue]] = []
    for component_name, field_name in sources:
        component = record.components.get(component_name)
        if component is None or component.provenance == "unavailable" \
                or not isinstance(component.value, Mapping) \
                or field_name not in component.value:
            continue
        value = _component_field_value(
            component,
            field_name,
            f"entity {record.entity_id!r}.{component_name}.{field_name}",
        )
        consumed.setdefault(
            (record.entity_id, record.generation, component_name), set()
        ).add(field_name)
        if value.provenance == "unavailable":
            gaps.add(
                frame,
                "unavailable-field",
                value.reason or "field is explicitly unavailable",
                record=record,
                component=component_name,
                field_name=field_name,
                provenance=value.provenance,
                legacy_consumer=f"{consumer}.{legacy_field}",
            )
            continue
        if value.provenance == "derived":
            gaps.add(
                frame,
                "derived-field",
                f"legacy input retains derived value from {value.derivation}",
                record=record,
                component=component_name,
                field_name=field_name,
                provenance=value.provenance,
                legacy_consumer=f"{consumer}.{legacy_field}",
            )
        found.append((component_name, field_name, value))
    if not found:
        return None
    reference = _json_comparable(found[0][2].value)
    if any(_json_comparable(value.value) != reference for _, _, value in found[1:]):
        paths = [f"{component}.{name}" for component, name, _ in found]
        gaps.add(
            frame,
            "ambiguous-explicit-fields",
            f"conflicting explicit sources {paths} target {consumer}.{legacy_field}; "
            "none was selected",
            record=record,
            field_name=legacy_field,
            legacy_consumer=consumer,
        )
        return None
    return found[0][2]


def _component_field_value(
    component: ProvenancedValue, field_name: str, path: str
) -> ProvenancedValue:
    value = component.value[field_name]
    if isinstance(value, Mapping) and "provenance" in value:
        return _provenanced(value, path)
    return ProvenancedValue(
        component.provenance,
        copy.deepcopy(value),
        component.reason,
        component.derivation,
    )


_INVALID = object()
_NUMERIC_LEGACY_FIELDS = frozenset(
    ("x", "y", "z", "eye_z", "yaw", "yaw_degrees", "pitch_degrees", "fov")
)


def _coerce_legacy_value(
    frame: RenderFrame,
    record: RenderEntityRecord,
    field_name: str,
    value: Any,
    gaps: _GapCollector,
    *,
    consumer: str,
) -> Any:
    valid = True
    converted = value
    if field_name in _NUMERIC_LEGACY_FIELDS:
        valid = not isinstance(value, bool) and isinstance(value, (int, float))
        valid = valid and math.isfinite(float(value))
        if valid:
            converted = float(value)
    elif field_name in ("fire", "is_alive"):
        valid = isinstance(value, bool)
    elif field_name == "team":
        valid = isinstance(value, str) and value in ("ct", "t")
    elif field_name in ("active_weapon_name", "active_weapon_id"):
        valid = not isinstance(value, (Mapping, Sequence)) or isinstance(value, str)
        valid = valid and not isinstance(value, bool) and value is not None
    if not valid:
        gaps.add(
            frame,
            "legacy-field-type-mismatch",
            f"explicit value {value!r} cannot be represented by {consumer}.{field_name}",
            record=record,
            field_name=field_name,
            legacy_consumer=consumer,
        )
        return _INVALID
    return converted


def _report_all_unconsumed(
    frame: RenderFrame,
    consumed: Mapping[tuple[str | int, int, str], set[str]],
    gaps: _GapCollector,
) -> None:
    for record in frame.entities:
        for component_name, component in record.components.items():
            fields = consumed.get(
                (record.entity_id, record.generation, component_name), set()
            )
            if component.provenance == "unavailable":
                gaps.add(
                    frame,
                    "unavailable-component",
                    component.reason or "component is explicitly unavailable",
                    record=record,
                    component=component_name,
                    provenance=component.provenance,
                )
                continue
            if not isinstance(component.value, Mapping):
                gaps.add(
                    frame,
                    "unsupported-component",
                    "the legacy renderer has no consumer for this scalar/array component",
                    record=record,
                    component=component_name,
                    provenance=component.provenance,
                )
                continue
            if not component.value:
                gaps.add(
                    frame,
                    "unsupported-empty-component",
                    "the component is present but contains no field a legacy "
                    "renderer consumer can observe",
                    record=record,
                    component=component_name,
                    provenance=component.provenance,
                )
                continue
            for field_name in component.value:
                if field_name in fields:
                    continue
                category = (
                    "unsupported-world-pose"
                    if component_name in ("animation", "physics", "dynamic_physics")
                    else "unsupported-field"
                )
                gaps.add(
                    frame,
                    category,
                    _unsupported_detail(component_name, field_name),
                    record=record,
                    component=component_name,
                    field_name=field_name,
                    provenance=component.provenance,
                )


def _unsupported_detail(component: str, field_name: str) -> str:
    if component == "animation" and field_name == "evaluated_bones":
        return (
            "evaluated bones reached the bridge, but the current playermodel pass "
            "only selects/invents legacy animation clips"
        )
    if component == "animation" and field_name == "graph_state":
        return (
            "complete animation graph state reached the bridge, but gpu_render has "
            "no graph-state or evaluated-pose consumer"
        )
    if component in ("physics", "dynamic_physics"):
        return (
            "dynamic physics state reached the bridge, but gpu_render has no "
            "dynamic-entity/rigid-body consumer"
        )
    if component in ("effects", "effect", "particles"):
        return (
            "recorded effect state reached the bridge, but the legacy file inputs "
            "do not expose it to an effect consumer"
        )
    return (
        f"normalized field {component}.{field_name} has no mapping into the "
        "current camera or playermodel JSON consumers"
    )


def _legacy_identity(value: str | int | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, str) and str(converted) != value.strip():
        return None
    return converted


def _json_comparable(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SnapshotValidationError(f"{path} must be an object")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotValidationError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SnapshotValidationError(f"{path} must be an integer >= {minimum}")
    return value


def _subtick(value: Any, path: str) -> int:
    return _integer(value, path, minimum=0)


def _entity_id(value: Any, path: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SnapshotValidationError(f"{path} must be a string or integer")
    if isinstance(value, str) and not value.strip():
        raise SnapshotValidationError(f"{path} must not be empty")
    return value


def _discontinuity(value: Any) -> Discontinuity | None:
    if value is None:
        return None
    item = _mapping(value, "snapshot.discontinuity")
    kind = _nonempty_string(item.get("kind"), "snapshot.discontinuity.kind")
    if kind not in _DISCONTINUITIES:
        choices = ", ".join(sorted(_DISCONTINUITIES))
        raise SnapshotValidationError(
            f"snapshot.discontinuity.kind must be one of: {choices}"
        )
    reason = _nonempty_string(item.get("reason"), "snapshot.discontinuity.reason")
    return Discontinuity(kind=kind, reason=reason)


def _components(value: Any, path: str) -> Mapping[str, ProvenancedValue]:
    raw = _mapping(value, path)
    if not raw:
        raise SnapshotValidationError(f"{path} must contain at least one component")
    result: dict[str, ProvenancedValue] = {}
    for name, envelope in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise SnapshotValidationError(f"{path} keys must be non-empty strings")
        result[name] = _provenanced(envelope, f"{path}.{name}")
    return MappingProxyType(result)


def _provenanced(value: Any, path: str) -> ProvenancedValue:
    envelope = _mapping(value, path)
    provenance = envelope.get("provenance")
    if provenance not in _PROVENANCE:
        raise SnapshotValidationError(
            f"{path}.provenance must be recorded, derived, or unavailable"
        )
    reason = envelope.get("reason")
    derivation = envelope.get("derivation")
    if reason is not None:
        reason = _nonempty_string(reason, f"{path}.reason")
    if derivation is not None:
        derivation = _nonempty_string(derivation, f"{path}.derivation")

    if provenance == "unavailable":
        if "value" in envelope and envelope["value"] is not None:
            raise SnapshotValidationError(
                f"{path} is unavailable and therefore cannot carry a value"
            )
        if reason is None:
            raise SnapshotValidationError(
                f"{path}.reason is required when provenance is unavailable"
            )
        payload = None
    else:
        if "value" not in envelope or envelope["value"] is None:
            raise SnapshotValidationError(
                f"{path}.value is required when provenance is {provenance}"
            )
        if provenance == "derived" and derivation is None:
            raise SnapshotValidationError(
                f"{path}.derivation is required when provenance is derived"
            )
        payload = copy.deepcopy(envelope["value"])

    known = {"provenance", "value", "reason", "derivation"}
    metadata = MappingProxyType(
        {key: copy.deepcopy(item) for key, item in envelope.items() if key not in known}
    )
    return ProvenancedValue(provenance, payload, reason, derivation, metadata)


def _primary_component(components: Mapping[str, ProvenancedValue]) -> str:
    for preferred in ("camera", "player", "dynamic_physics", "physics", "animation"):
        if preferred in components:
            return preferred
    return next(iter(components))


def _with_primary(record: RenderEntityRecord, name: str) -> RenderEntityRecord:
    return RenderEntityRecord(
        stream_id=record.stream_id,
        tick=record.tick,
        subtick=record.subtick,
        entity_id=record.entity_id,
        generation=record.generation,
        entity_class=record.entity_class,
        primary_component=name,
        components=record.components,
    )


def _reroute(
    records: Iterable[RenderEntityRecord], component: str
) -> tuple[RenderEntityRecord, ...]:
    return tuple(
        _with_primary(record, component)
        for record in records
        if component in record.components
    )


def _route_dynamic_physics(
    records: Iterable[RenderEntityRecord],
) -> tuple[RenderEntityRecord, ...]:
    result: list[RenderEntityRecord] = []
    for record in records:
        name = (
            "dynamic_physics"
            if "dynamic_physics" in record.components
            else "physics" if "physics" in record.components else None
        )
        if name is None:
            continue
        component = record.components[name]
        if (
            component.provenance != "unavailable"
            and isinstance(component.value, Mapping)
            and component.value.get("dynamic") is False
        ):
            continue
        result.append(_with_primary(record, name))
    return tuple(result)


def _field(
    component: ProvenancedValue, name: str, path: str
) -> tuple[str, Any] | None:
    if component.provenance == "unavailable" or not isinstance(
        component.value, Mapping
    ):
        return None
    if name not in component.value:
        return None
    value = component.value[name]
    if isinstance(value, Mapping) and "provenance" in value:
        nested = _provenanced(value, path)
        return nested.provenance, nested.value
    return component.provenance, value


def _require_authoritative_world_pose(
    players: Sequence[RenderEntityRecord],
    physics: Sequence[RenderEntityRecord],
    animations: Sequence[RenderEntityRecord],
) -> None:
    for record in physics:
        component = record.primary
        transform = _field(
            component,
            "world_transform",
            f"entity {record.entity_id!r}.{record.primary_component}.world_transform",
        )
        if transform is None or transform[0] != "recorded" or transform[1] is None:
            raise AuthoritativeStateUnavailable(
                f"dynamic physics entity {record.entity_id!r} requires a recorded "
                "world_transform in authoritative world-pose mode"
            )

    animated_by_identity = {
        (record.entity_id, record.generation): record for record in animations
    }
    required = dict(animated_by_identity)
    for player in players:
        identity = (player.entity_id, player.generation)
        if identity not in required:
            raise AuthoritativeStateUnavailable(
                f"player entity {player.entity_id!r} requires an animation component "
                "in authoritative world-pose mode"
            )

    for record in required.values():
        component = record.primary
        bones = _field(
            component,
            "evaluated_bones",
            f"entity {record.entity_id!r}.animation.evaluated_bones",
        )
        if bones is not None and bones[0] == "recorded" and _nonempty_pose(bones[1]):
            continue
        graph = _field(
            component,
            "graph_state",
            f"entity {record.entity_id!r}.animation.graph_state",
        )
        if (
            graph is not None
            and graph[0] == "recorded"
            and isinstance(graph[1], Mapping)
            and graph[1].get("complete") is True
        ):
            continue
        raise AuthoritativeStateUnavailable(
            f"animated entity {record.entity_id!r} requires recorded evaluated_bones "
            "or recorded graph_state with complete=true in authoritative "
            "world-pose mode"
        )


def _nonempty_pose(value: Any) -> bool:
    return isinstance(value, (Mapping, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ) and len(value) > 0


__all__ = (
    "SCHEMA",
    "AuthoritativeStateUnavailable",
    "BridgeMode",
    "Discontinuity",
    "LegacyRendererUnsupported",
    "LegacyStageResult",
    "ProvenancedValue",
    "RenderBatch",
    "RenderEntityRecord",
    "RenderFrame",
    "SnapshotValidationError",
    "StateContinuityError",
    "StateToRenderBridge",
    "snapshots_to_render_batch",
    "stage_legacy_inputs",
)
