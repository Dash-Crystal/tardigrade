"""Adapter from a legacy CS:GO Source 1 demo into canonical state replay."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .integrator import StateIntegrator
from .source1_demo import DemoCommand, DemoHeader, Source1DemoError, Source1DemoReader
from .source1_entities import (
    EntityChange, PacketEntityDecoder, StringTableDecoder, StringTableState,
)
from .source1_net import (
    DataTables,
    decode_create_string_table,
    decode_game_event,
    decode_game_event_list,
    decode_packet_entities,
    decode_update_string_table,
    parse_data_tables,
    parse_net_messages,
)
from .source1_resource_contract import (
    MODEL_BINDING_DERIVATION,
    MODEL_BINDING_PROVENANCE,
    MODEL_PRECACHE_ENTITY_ID,
    STRING_TABLE_RESOURCE_DERIVATION,
    STRING_TABLE_RESOURCE_PROVENANCE,
    make_string_table_entry,
    string_table_entry_sha256,
)


@dataclasses.dataclass(frozen=True)
class ReductionContext:
    tick: int
    player_slot: int
    command_offset: int
    message_offset: int | None = None


class Source1Reducer(Protocol):
    """Extension boundary for domain-specific canonical component mapping."""

    def on_data_tables(self, tables: DataTables, context: ReductionContext) -> Sequence[dict]: ...
    def on_string_table(self, kind: str, table: StringTableState,
                        changed_indices: Sequence[int],
                        context: ReductionContext,
                        decoder: PacketEntityDecoder) -> Sequence[dict]: ...
    def on_string_tables_removed(self, tables: Sequence[StringTableState],
                                 context: ReductionContext,
                                 decoder: PacketEntityDecoder) -> Sequence[dict]: ...
    def on_entities(self, changes: Sequence[EntityChange],
                    decoder: PacketEntityDecoder,
                    context: ReductionContext) -> Sequence[dict]: ...
    def on_game_event(self, event: Mapping[str, object],
                      context: ReductionContext) -> Sequence[dict]: ...


class CanonicalSource1Reducer:
    """Map decoded edicts/netprops to renderer-facing canonical entities."""

    def __init__(self) -> None:
        # Canonical generations are monotonic lifetime counters. Source's
        # recorded serial is only 10 bits and legitimately wraps.
        self.active: dict[int, tuple[int, int]] = {}
        self.watermarks: dict[int, int] = {}
        self.modelprecache: dict[int, tuple[str, bytes]] = {}
        self.model_bindings: dict[int, tuple[str, int, str]] = {}
        self.table_active: dict[str, int] = {}
        self.table_watermarks: dict[str, int] = {}
        self.table_revisions: dict[str, int] = {}

    def on_data_tables(self, tables: DataTables,
                       context: ReductionContext) -> Sequence[dict]:
        return ()

    def on_string_table(self, kind: str, table: StringTableState,
                        changed_indices: Sequence[int],
                        context: ReductionContext,
                        decoder: PacketEntityDecoder) -> Sequence[dict]:
        active = dict(self.table_active)
        watermarks = dict(self.table_watermarks)
        revisions = dict(self.table_revisions)
        revision = revisions.get(table.name, -1) + 1
        revisions[table.name] = revision
        entity_id = f"source1:stringtable:{table.name}"
        component = self._string_table_component(table, revision, kind)
        ops: list[dict] = []
        if table.name not in active:
            generation = watermarks.get(table.name, -1) + 1
            watermarks[table.name] = generation
            active[table.name] = generation
            ops.append({"op": "create", "entity": {
                "id": entity_id, "generation": generation,
                "class": "Source1StringTable",
                "components": {"source1_string_table": component}}})
        else:
            ops.append({"op": "update", "id": entity_id,
                "generation": active[table.name],
                "components": {"source1_string_table": component}})
        if table.name == "modelprecache":
            modelprecache = {
                index: (name, data) for index, (name, data) in table.entries.items()
                if name
            }
            self.modelprecache = modelprecache
            self.table_revisions = revisions
            ops.extend(self._reconcile_model_bindings(decoder, revision))
        self.table_active, self.table_watermarks = active, watermarks
        self.table_revisions = revisions
        return ops

    def on_string_tables_removed(self, tables: Sequence[StringTableState],
                                 context: ReductionContext,
                                 decoder: PacketEntityDecoder) -> Sequence[dict]:
        active = dict(self.table_active)
        ops: list[dict] = []
        model_removed = False
        for table in tables:
            generation = active.pop(table.name, None)
            if generation is not None:
                ops.append({"op": "destroy",
                    "id": f"source1:stringtable:{table.name}",
                    "generation": generation})
            model_removed |= table.name == "modelprecache"
        self.table_active = active
        if model_removed:
            self.modelprecache = {}
            self.table_revisions.pop("modelprecache", None)
            ops.extend(self._reconcile_model_bindings(decoder, -1))
        return ops

    @staticmethod
    def _entry_document(index: int, item: tuple[str, bytes]) -> dict[str, object]:
        name, data = item
        return make_string_table_entry(index, name, data)

    def _string_table_component(self, table: StringTableState,
                                revision: int, kind: str) -> dict[str, object]:
        inferred = table.metadata_origin == "dem_stringtables-inferred"
        return {"provenance": STRING_TABLE_RESOURCE_PROVENANCE,
            "derivation": STRING_TABLE_RESOURCE_DERIVATION,
            "value": {
            "table_id": table.table_id, "name": table.name,
            "revision": revision, "max_entries": table.max_entries,
            "user_data_fixed_size": table.user_data_fixed_size,
            "user_data_size": table.user_data_size,
            "user_data_size_bits": table.user_data_size_bits,
            "server_entries": [self._entry_document(index, table.entries[index])
                               for index in sorted(table.entries)],
            "client_entries": [self._entry_document(index,
                               table.client_entries[index])
                               for index in sorted(table.client_entries)],
            "client_separation": "client entries never overwrite server indices",
            "field_origins": {
                "revision": {"provenance": "derived",
                    "derivation": "monotonic reducer revision for hashed content"},
                "table_id": {"provenance": "derived",
                    "derivation": ("ordinal assigned from recorded dem_stringtables order"
                                   if kind == "snapshot" else
                                   "ordinal assigned from recorded svc_CreateStringTable order")},
                "name": {"provenance": "recorded",
                    "source": "Source 1 table name retained across ordered history"},
                "server_entries": {"provenance": "derived",
                    "derivation": "accumulated server map from recorded create/update/snapshot operations",
                    "entry_content_provenance": "recorded"},
                "client_entries": {"provenance": "derived",
                    "derivation": "separate client map from latest authoritative command snapshot, or empty",
                    "entry_content_provenance": "recorded when present"},
                "capacity_and_userdata_layout": ({
                    "provenance": "derived",
                    "derivation": "snapshot count is only a lower bound; fixed-userdata metadata absent"
                } if inferred else {
                    "provenance": "recorded", "source": table.metadata_origin
                }),
                "client_separation": {"provenance": "derived",
                    "derivation": "protocol namespaces retained as separate arrays"},
            },
        }}

    @staticmethod
    def _entry_hash(index: int, name: str, data: bytes) -> str:
        return string_table_entry_sha256(index, name, data)

    def _model_component(self, entity) -> tuple[dict[str, object] | None,
                                                tuple[str, int, str] | None]:
        model_index = next((value for key, value in entity.properties.items()
                            if key.endswith("m_nModelIndex") and
                            isinstance(value, int) and not isinstance(value, bool)), None)
        model_entry = self.modelprecache.get(model_index) if model_index is not None else None
        if model_entry is None:
            return None, None
        model_path, user_data = model_entry
        if not model_path.casefold().endswith(".mdl"):
            return None, None
        revision = self.table_revisions.get("modelprecache")
        if revision is None:
            return None, None
        entry_hash = self._entry_hash(model_index, model_path, user_data)
        binding = (model_path, revision, entry_hash)
        return ({
            "provenance": MODEL_BINDING_PROVENANCE,
            "derivation": MODEL_BINDING_DERIVATION,
            "value": {"source1_mdl": model_path, "model_index": model_index,
                      "source_table_entity_id": MODEL_PRECACHE_ENTITY_ID,
                      "source_table_revision": revision,
                      "source_entry_sha256": entry_hash,
                      "lod": {"provenance": "unavailable",
                              "reason": "render LOD is not entity state"}},
        }, binding)

    def _reconcile_model_bindings(self, decoder: PacketEntityDecoder,
                                  revision: int) -> list[dict]:
        bindings = dict(self.model_bindings)
        ops: list[dict] = []
        for index, identity in sorted(self.active.items()):
            entity = decoder.entities.get(index)
            if entity is None:
                continue
            component, binding = self._model_component(entity)
            old = bindings.get(index)
            if binding == old:
                continue
            op = {"op": "update", "id": f"source1:edict:{index}",
                  "generation": identity[0]}
            if component is None:
                if old is None:
                    continue
                op["remove_components"] = ["source1_model"]
                bindings.pop(index, None)
            else:
                op["components"] = {"source1_model": component}
                assert binding is not None
                bindings[index] = binding
            ops.append(op)
        self.model_bindings = bindings
        return ops

    def _components(self, entity) -> dict[str, object]:
        props = dict(entity.properties)
        complete = ("partial-instancebaseline-missing-constructor-defaults"
                    if entity.baseline_present else "partial-no-instancebaseline")
        components: dict[str, object] = {
            "source1_identity": {
                "provenance": "recorded",
                "value": {"edict": entity.index, "class_id": entity.class_id,
                          "serial": entity.serial},
            },
            "visibility": {
                "provenance": "derived",
                "derivation": "Source 1 PacketEntities PVS header transition",
                "value": {"in_pvs": entity.in_pvs},
            },
            "source1_netprops": {
                "provenance": "derived",
                "derivation": "Source 1 sendtable baseline plus ordered property deltas",
                "completeness": complete,
                "value": props,
            },
            "source1_reconstruction": {
                "provenance": "derived",
                "derivation": "coverage classification of decoded Source 1 fields",
                "value": {
                    "animation": {"status": "unavailable",
                        "reason": "final bones/client animation state are not demo sendprops"},
                    "physics": {"status": "unavailable",
                        "reason": "rigid-body state has not been mapped from Source 1 netprops"},
                },
            },
        }
        origin = next((value for key, value in props.items()
                       if key.endswith("m_vecOrigin") and isinstance(value, list)), None)
        angles = next((value for key, value in props.items()
                       if (key.endswith("m_angRotation") or
                           key.endswith("m_angEyeAngles")) and
                       isinstance(value, list)), None)
        if origin is not None:
            components["transform"] = {
                "provenance": "derived",
                "derivation": "canonical transform selected from decoded Source 1 netprops",
                "value": {"origin": origin, "angles": angles},
            }
        else:
            components["transform"] = {
                "provenance": "unavailable",
                "reason": "no decoded vector origin sendprop in current entity state",
            }
        player_names = ("m_iHealth", "m_ArmorValue", "m_iTeamNum", "m_lifeState",
                        "m_bIsScoped", "m_hActiveWeapon", "m_fFlags", "m_vecVelocity")
        player = {key: value for key, value in props.items()
                  if any(key.endswith(name) for name in player_names)}
        if player:
            components["player_state"] = {
                "provenance": "derived",
                "derivation": "canonical player fields selected from decoded Source 1 netprops",
                "value": player}
        model, _binding = self._model_component(entity)
        if model is not None:
            components["source1_model"] = model
        return components

    def on_entities(self, changes: Sequence[EntityChange],
                    decoder: PacketEntityDecoder,
                    context: ReductionContext) -> Sequence[dict]:
        active = dict(self.active)
        watermarks = dict(self.watermarks)
        bindings = dict(self.model_bindings)
        ops: list[dict] = []
        for change in changes:
            entity_id = f"source1:edict:{change.index}"
            if change.kind == "delete":
                identity = active.pop(change.index, None)
                bindings.pop(change.index, None)
                if identity is None:
                    raise Source1DemoError(f"delete of untracked edict {change.index}")
                ops.append({"op": "destroy", "id": entity_id,
                            "generation": identity[0]})
            elif change.kind == "leave":
                identity = active.get(change.index)
                if identity is None:
                    raise Source1DemoError(f"leave of untracked edict {change.index}")
                ops.append({"op": "update", "id": entity_id,
                    "generation": identity[0], "components": {"visibility": {
                        "provenance": "derived",
                        "derivation": "Source 1 leave-PVS header transition",
                        "value": {"in_pvs": False,
                        "transition": "leave-without-delete"}}}})
            else:
                entity = decoder.entities[change.index]
                components = self._components(entity)
                _model, binding = self._model_component(entity)
                previous = active.get(change.index)
                if change.kind == "enter" and previous is None:
                    generation = watermarks.get(change.index, -1) + 1
                    watermarks[change.index] = generation
                    active[change.index] = (generation, entity.serial)
                    if binding is not None:
                        bindings[change.index] = binding
                    ops.append({"op": "create", "entity": {
                        "id": entity_id, "generation": generation,
                        "class": entity.class_name, "components": components}})
                elif change.kind == "enter" and previous[1] != entity.serial:
                    ops.append({"op": "destroy", "id": entity_id,
                                "generation": previous[0]})
                    generation = watermarks.get(change.index, previous[0]) + 1
                    watermarks[change.index] = generation
                    active[change.index] = (generation, entity.serial)
                    if binding is not None:
                        bindings[change.index] = binding
                    else:
                        bindings.pop(change.index, None)
                    ops.append({"op": "create", "entity": {
                        "id": entity_id, "generation": generation,
                        "class": entity.class_name, "components": components}})
                else:
                    identity = active.get(change.index)
                    if identity is None or identity[1] != entity.serial:
                        raise Source1DemoError(
                            f"edict {change.index} serial changed outside enter-PVS")
                    update = {"op": "update", "id": entity_id,
                        "generation": identity[0], "components": components}
                    if bindings.get(change.index) is not None and binding is None:
                        update["remove_components"] = ["source1_model"]
                        bindings.pop(change.index, None)
                    elif binding is not None:
                        bindings[change.index] = binding
                    ops.append(update)
        self.active, self.watermarks = active, watermarks
        self.model_bindings = bindings
        return ops

    def on_game_event(self, event: Mapping[str, object],
                      context: ReductionContext) -> Sequence[dict]:
        digest = hashlib.sha256(repr(sorted(event.items())).encode()).hexdigest()[:16]
        return ({
            "op": "action",
            "action_id": f"source1:event:{context.tick}:{context.command_offset}:{context.message_offset}:{digest}",
            "kind": f"source1.game_event.{event.get('name') or event.get('event_id')}",
            "payload": dict(event),
        },)


@dataclasses.dataclass(frozen=True)
class IngestSummary:
    stream_id: str
    commands: int
    net_messages: int
    entity_updates: int
    game_events: int
    final_tick: int
    final_state_hash: str


class Source1ReplayAdapter:
    """Configuration-selectable ``source1-hl2demo-v1`` replay adapter."""

    ADAPTER = "source1-hl2demo-v1"

    def __init__(self, reducer: Source1Reducer | None = None, *,
                 camera_split_slot: int = 0) -> None:
        self.reducer = reducer or CanonicalSource1Reducer()
        if camera_split_slot not in (0, 1):
            raise Source1DemoError("camera_split_slot must be 0 or 1")
        self.camera_split_slot = camera_split_slot
        self._used = False

    @staticmethod
    def manifest(header: DemoHeader, stream_id: str,
                 tick_rate_hz: int | None = None) -> dict[str, object]:
        if tick_rate_hz is None:
            if header.playback_time <= 0 or header.playback_ticks <= 0:
                raise Source1DemoError("cannot derive tick rate from empty demo header")
            tick_rate_hz = round(header.playback_ticks / header.playback_time)
        if tick_rate_hz <= 0:
            raise Source1DemoError("tick rate must be positive")
        return {
            "schema": "tardigrade/state-manifest/v1",
            "stream_id": stream_id,
            "tick_rate_hz": tick_rate_hz,
            "source": {
                "kind": "csgo-source1-demo", "adapter": Source1ReplayAdapter.ADAPTER,
                "demo_protocol": header.demo_protocol,
                "network_protocol": header.network_protocol,
                "map": header.map_name,
                "game_directory": header.game_directory,
                "evidence": {
                    "container": "parsed", "net_message_framing": "parsed",
                    "entities": "reconstructed when datatables and baselines are present",
                    "animation_bones": "unknown-not-recorded-as-sendprops",
                },
            },
        }

    def ingest(self, reader: Source1DemoReader, storage_root: Path | str,
               stream_id: str, *, tick_rate_hz: int | None = None) -> IngestSummary:
        if self._used:
            raise Source1DemoError(
                "Source1ReplayAdapter instances are one-shot; create a fresh adapter")
        self._used = True
        manifest = self.manifest(reader.header, stream_id, tick_rate_hz)
        checkpoint = StateIntegrator.make_checkpoint(stream_id, 0, 0, 0, [])
        final_root = Path(storage_root).expanduser().resolve()
        if final_root.exists():
            raise Source1DemoError(f"output already exists: {final_root}")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".{final_root.name}.partial-", dir=final_root.parent))
        reducer_rollback = None
        if isinstance(self.reducer, CanonicalSource1Reducer):
            reducer_rollback = (dict(self.reducer.active),
                                dict(self.reducer.watermarks),
                                dict(self.reducer.modelprecache),
                                dict(self.reducer.model_bindings),
                                dict(self.reducer.table_active),
                                dict(self.reducer.table_watermarks),
                                dict(self.reducer.table_revisions))
        integrator = StateIntegrator.create(staging, manifest, checkpoint)
        command_count = message_count = entity_count = event_count = 0
        final_tick = 0
        subtick_tick = -1
        subtick = -1
        entity_decoder: PacketEntityDecoder | None = None
        string_decoder: StringTableDecoder | None = None
        pending_string_tables: list[tuple[str, dict[str, object], ReductionContext]] = []
        pending_snapshots: list[tuple[bytes, ReductionContext]] = []
        descriptors: dict[int, dict[str, object]] = {}
        camera_created = False
        try:
            for command in reader.commands():
                command_count += 1
                canonical_tick = max(0, command.tick)
                final_tick = canonical_tick
                if canonical_tick != subtick_tick:
                    subtick_tick, subtick = canonical_tick, 0
                else:
                    subtick += 1
                context = ReductionContext(canonical_tick, command.player_slot,
                                           command.offset)
                ops: list[dict] = [{
                    "op": "action",
                    "action_id": f"source1:command:{command_count}",
                    "kind": f"source1.demo.{command.name}",
                    "payload": self._command_payload(command),
                }]
                if command.name == "datatables":
                    tables = parse_data_tables(command.payload)
                    entity_decoder = PacketEntityDecoder(tables)
                    string_decoder = StringTableDecoder(entity_decoder)
                    ops.extend(self.reducer.on_data_tables(tables, context))
                    for kind, envelope, pending_context in pending_string_tables:
                        ops.extend(self._string(kind, envelope, string_decoder,
                                                pending_context))
                    pending_string_tables.clear()
                    for raw, pending_context in pending_snapshots:
                        for table, changed in string_decoder.snapshot(raw):
                            ops.extend(self.reducer.on_string_table(
                                "snapshot", table, changed, pending_context,
                                entity_decoder))
                        ops.extend(self.reducer.on_string_tables_removed(
                            string_decoder.last_snapshot_removed,
                            pending_context, entity_decoder))
                    pending_snapshots.clear()
                elif command.name in ("packet", "signon"):
                    camera_ops, camera_created = self._camera_ops(
                        command, camera_created)
                    ops.extend(camera_ops)
                    for message in parse_net_messages(command.payload):
                        message_count += 1
                        message_context = dataclasses.replace(
                            context, message_offset=message.offset)
                        if message.message_type == 12:
                            envelope = decode_create_string_table(message.payload)
                            if string_decoder is None:
                                pending_string_tables.append(
                                    ("create", envelope, message_context))
                            else:
                                ops.extend(self._string("create", envelope,
                                    string_decoder, message_context))
                        elif message.message_type == 13:
                            envelope = decode_update_string_table(message.payload)
                            if string_decoder is None:
                                pending_string_tables.append(
                                    ("update", envelope, message_context))
                            else:
                                ops.extend(self._string("update", envelope,
                                    string_decoder, message_context))
                        elif message.message_type == 26:
                            if entity_decoder is None:
                                raise Source1DemoError(
                                    "PacketEntities arrived before dem_datatables")
                            changes = entity_decoder.apply(
                                decode_packet_entities(message.payload),
                                tick=canonical_tick)
                            entity_count += len(changes)
                            ops.extend(self.reducer.on_entities(
                                changes, entity_decoder, message_context))
                        elif message.message_type == 30:
                            descriptors.update(decode_game_event_list(message.payload))
                        elif message.message_type == 25:
                            event = decode_game_event(message.payload, descriptors)
                            event_count += 1
                            ops.extend(self.reducer.on_game_event(event, message_context))
                elif command.name == "stringtables":
                    if string_decoder is None:
                        pending_snapshots.append((command.payload, context))
                    else:
                        for table, changed in string_decoder.snapshot(command.payload):
                            ops.extend(self.reducer.on_string_table(
                                "snapshot", table, changed, context,
                                entity_decoder))
                        ops.extend(self.reducer.on_string_tables_removed(
                            string_decoder.last_snapshot_removed,
                            context, entity_decoder))
                transaction = {
                    "schema": "tardigrade/state-transaction/v1",
                    "stream_id": stream_id,
                    "sequence": command_count,
                    "tick": canonical_tick,
                    "subtick": subtick,
                    "base_hash": integrator.current_snapshot()["state_hash"],
                    "ops": ops,
                }
                integrator.ingest(transaction)
            if pending_string_tables or pending_snapshots:
                raise Source1DemoError(
                    "demo ended before datatables required by string-table baselines")
            snapshot = integrator.current_snapshot()
            integrator.checkpoint()
            summary = IngestSummary(stream_id, command_count, message_count,
                entity_count, event_count, final_tick, snapshot["state_hash"])
            integrator.close()
            integrator = None
            (staging / "source1-ingest.json").write_text(json.dumps({
                "status": "complete", "adapter": self.ADAPTER,
                "summary": dataclasses.asdict(summary),
                "coverage": manifest["source"]["evidence"],
            }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(staging, final_root)
            return summary
        finally:
            if integrator is not None:
                integrator.close()
            if staging.exists():
                shutil.rmtree(staging)
                if reducer_rollback is not None:
                    (self.reducer.active, self.reducer.watermarks,
                     self.reducer.modelprecache, self.reducer.model_bindings,
                     self.reducer.table_active, self.reducer.table_watermarks,
                     self.reducer.table_revisions) = reducer_rollback

    def _string(self, kind: str, envelope: Mapping[str, object],
                decoder: StringTableDecoder,
                context: ReductionContext) -> list[dict]:
        if kind == "create":
            table, changed = decoder.create(envelope)
        else:
            table, changed = decoder.update(envelope)
        return list(self.reducer.on_string_table(
            kind, table, changed, context, decoder.entities))

    @staticmethod
    def _command_payload(command: DemoCommand) -> dict[str, object]:
        result: dict[str, object] = {
            "tick": command.tick,
            "player_slot": command.player_slot,
            "file_offset": command.offset,
            "framing": {"provenance": "parsed"},
            "payload": {"byte_length": len(command.payload),
                        "sha256": hashlib.sha256(command.payload).hexdigest()},
        }
        if command.sequence_in is not None:
            result["network_sequences"] = {
                "provenance": "recorded", "incoming": command.sequence_in,
                "outgoing": command.sequence_out}
        if command.outgoing_sequence is not None:
            result["usercmd_sequence"] = {
                "provenance": "recorded", "value": command.outgoing_sequence}
        if command.views:
            result["views"] = [{
                "flags": view.flags,
                "view_origin": list(view.view_origin),
                "view_angles": list(view.view_angles),
                "local_view_angles": list(view.local_view_angles),
                "view_origin_resampled": list(view.view_origin_resampled),
                "view_angles_resampled": list(view.view_angles_resampled),
                "local_view_angles_resampled": list(
                    view.local_view_angles_resampled),
                "provenance": "recorded",
            } for view in command.views]
        return result

    def _camera_ops(self, command: DemoCommand,
                    created: bool) -> tuple[list[dict], bool]:
        if len(command.views) != 2 or self.camera_split_slot not in (0, 1):
            raise Source1DemoError("packet command lacks selected split-screen view")
        view = command.views[self.camera_split_slot]
        origin = (view.view_origin_resampled if view.flags & 1
                  else view.view_origin)
        angles = (view.view_angles_resampled if view.flags & 2
                  else view.view_angles)
        component = {
            "provenance": "recorded",
            "value": {
                "origin": list(origin),
                "pitch_degrees": angles[0],
                "yaw_degrees": angles[1],
                "roll_degrees": angles[2],
                "fov": {
                    "provenance": "unavailable",
                    "reason": "democmdinfo_t does not record projection FOV",
                },
                "player_slot": command.player_slot,
                "split_screen_slot": self.camera_split_slot,
                "flags": view.flags,
            },
        }
        if not created:
            return ([{"op": "create", "entity": {
                "id": "source1:camera", "generation": 0, "class": "camera",
                "components": {"camera": component}}}], True)
        return ([{"op": "update", "id": "source1:camera", "generation": 0,
                  "components": {"camera": component}}], True)
