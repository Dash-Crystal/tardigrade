"""Bounded Source 1 protobuf framing and selected envelope decoders.

These routines intentionally stop at the boundary where Source's bit-packed
sendprop/string-table codecs begin.  A successfully decoded protobuf envelope
is not evidence that its embedded ``entity_data`` or ``string_data`` has been
reconstructed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import struct
from typing import Iterator, Mapping, Sequence

from .source1_demo import DemoLimits, Source1DemoError


NET_MESSAGE_NAMES = {
    0: "net_NOP", 1: "net_Disconnect", 2: "net_File", 4: "net_Tick",
    5: "net_StringCmd", 6: "net_SetConVar", 7: "net_SignonState",
    8: "svc_ServerInfo", 9: "svc_SendTable", 10: "svc_ClassInfo",
    11: "svc_SetPause", 12: "svc_CreateStringTable",
    13: "svc_UpdateStringTable", 14: "svc_VoiceInit",
    15: "svc_VoiceData", 16: "svc_Print", 17: "svc_Sounds",
    18: "svc_SetView", 19: "svc_FixAngle", 20: "svc_CrosshairAngle",
    21: "svc_BSPDecal", 23: "svc_UserMessage", 25: "svc_GameEvent",
    26: "svc_PacketEntities", 27: "svc_TempEntities",
    28: "svc_Prefetch", 29: "svc_Menu", 30: "svc_GameEventList",
    31: "svc_GetCvarValue",
}


@dataclasses.dataclass(frozen=True)
class NetMessage:
    message_type: int
    name: str
    payload: bytes
    offset: int


@dataclasses.dataclass(frozen=True)
class DataTableClass:
    class_id: int
    class_name: str
    data_table_name: str


@dataclasses.dataclass(frozen=True)
class DataTables:
    send_tables: tuple[Mapping[str, object], ...]
    server_classes: tuple[DataTableClass, ...]


def _varint(data: bytes, offset: int, label: str) -> tuple[int, int]:
    value = 0
    for index in range(10):
        if offset >= len(data):
            raise Source1DemoError(f"truncated {label} varint at byte {offset}")
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise Source1DemoError(f"overflowing {label} varint")
        value |= (byte & 0x7f) << (index * 7)
        if not byte & 0x80:
            return value, offset
    raise Source1DemoError(f"overlong {label} varint")


def parse_net_messages(payload: bytes, *,
                       limits: DemoLimits | None = None) -> Iterator[NetMessage]:
    """Parse ``type varint, length varint, bytes`` packet framing."""
    limits = limits or DemoLimits()
    offset = 0
    for _ in range(limits.max_net_messages):
        if offset == len(payload):
            return
        start = offset
        message_type, offset = _varint(payload, offset, "message type")
        size, offset = _varint(payload, offset, "message length")
        if size > limits.max_net_message_bytes:
            raise Source1DemoError(
                f"net message at byte {start} is {size} bytes; limit is "
                f"{limits.max_net_message_bytes}")
        end = offset + size
        if end > len(payload):
            raise Source1DemoError(
                f"truncated net message at byte {start}: declares {size} bytes")
        yield NetMessage(message_type, NET_MESSAGE_NAMES.get(
            message_type, f"unknown_{message_type}"), payload[offset:end], start)
        offset = end
    raise Source1DemoError("net-message count limit reached")


# Values are tagged so a bytes field can never silently masquerade as a varint.
WireFields = dict[int, list[tuple[int, int | bytes]]]


def parse_wire_fields(payload: bytes, *, max_fields: int = 1_000_000) -> WireFields:
    fields: WireFields = {}
    offset = 0
    for _ in range(max_fields):
        if offset == len(payload):
            return fields
        tag, offset = _varint(payload, offset, "protobuf tag")
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise Source1DemoError("protobuf field number zero")
        if wire == 0:
            value, offset = _varint(payload, offset, "protobuf value")
        elif wire == 1:
            if offset + 8 > len(payload):
                raise Source1DemoError("truncated protobuf fixed64")
            value = payload[offset:offset + 8]
            offset += 8
        elif wire == 2:
            size, offset = _varint(payload, offset, "protobuf bytes length")
            end = offset + size
            if end > len(payload):
                raise Source1DemoError("truncated protobuf bytes field")
            value = payload[offset:end]
            offset = end
        elif wire == 5:
            if offset + 4 > len(payload):
                raise Source1DemoError("truncated protobuf fixed32")
            value = payload[offset:offset + 4]
            offset += 4
        else:
            raise Source1DemoError(f"unsupported protobuf wire type {wire}")
        fields.setdefault(number, []).append((wire, value))
    raise Source1DemoError("protobuf field-count limit reached")


def _one(fields: WireFields, number: int, wire: int,
         default: int | bytes | None = None) -> int | bytes | None:
    values = fields.get(number, [])
    if not values:
        return default
    if len(values) != 1 or values[0][0] != wire:
        raise Source1DemoError(f"invalid protobuf field {number}")
    return values[0][1]


def _many_bytes(fields: WireFields, number: int) -> list[bytes]:
    values = fields.get(number, [])
    if any(wire != 2 for wire, _ in values):
        raise Source1DemoError(f"invalid protobuf field {number}")
    return [value for _, value in values if isinstance(value, bytes)]


def _text(raw: bytes | None, label: str) -> str | None:
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Source1DemoError(f"{label} is not UTF-8") from exc


def _blob(raw: bytes | None) -> Mapping[str, object]:
    raw = raw or b""
    return {
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": "unknown",
        "reason": "embedded Source 1 bitstream not decoded",
    }


def _int32(value: int | bytes | None) -> int | None:
    if value is None or not isinstance(value, int):
        return None
    value &= 0xffffffff
    return value - (1 << 32) if value & (1 << 31) else value


def decode_create_string_table(payload: bytes) -> dict[str, object]:
    f = parse_wire_fields(payload)
    data = _one(f, 8, 2, b"")
    assert isinstance(data, bytes)
    return {
        "name": _text(_one(f, 1, 2), "string-table name"),
        "max_entries": _one(f, 2, 0, 0),
        "num_entries": _one(f, 3, 0, 0),
        "user_data_fixed_size": bool(_one(f, 4, 0, 0)),
        "user_data_size": _one(f, 5, 0, 0),
        "user_data_size_bits": _one(f, 6, 0, 0),
        "flags": _one(f, 7, 0, 0),
        "string_data": _blob(data),
        "_string_data_bytes": data,
    }


def decode_update_string_table(payload: bytes) -> dict[str, object]:
    f = parse_wire_fields(payload)
    data = _one(f, 3, 2, b"")
    assert isinstance(data, bytes)
    return {
        "table_id": _one(f, 1, 0, 0),
        "num_changed_entries": _one(f, 2, 0, 0),
        "string_data": _blob(data),
        "_string_data_bytes": data,
    }


def decode_packet_entities(payload: bytes) -> dict[str, object]:
    f = parse_wire_fields(payload)
    data = _one(f, 7, 2, b"")
    assert isinstance(data, bytes)
    return {
        "max_entries": _int32(_one(f, 1, 0, 0)),
        "updated_entries": _int32(_one(f, 2, 0, 0)),
        "is_delta": bool(_one(f, 3, 0, 0)),
        "update_baseline": bool(_one(f, 4, 0, 0)),
        "baseline": _int32(_one(f, 5, 0, 0)),
        "delta_from": _int32(_one(f, 6, 0, 0)),
        "entity_data": _blob(data),
        "_entity_data_bytes": data,
    }


_GAME_VALUE_FIELDS = {
    2: (2, "string", 1), 3: (5, "float", 2), 4: (0, "long", 3),
    5: (0, "short", 4), 6: (0, "byte", 5), 7: (0, "bool", 6),
    8: (0, "uint64", 7), 9: (2, "wstring", 8),
}


def decode_game_event_list(payload: bytes) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for descriptor_raw in _many_bytes(parse_wire_fields(payload), 1):
        descriptor = parse_wire_fields(descriptor_raw)
        event_id = _one(descriptor, 1, 0)
        name = _text(_one(descriptor, 2, 2), "game-event name")
        if not isinstance(event_id, int) or name is None or event_id in result:
            raise Source1DemoError("invalid/duplicate game-event descriptor")
        keys = []
        for key_raw in _many_bytes(descriptor, 3):
            key = parse_wire_fields(key_raw)
            key_type = _one(key, 1, 0)
            key_name = _text(_one(key, 2, 2), "game-event key name")
            if not isinstance(key_type, int) or key_name is None:
                raise Source1DemoError("invalid game-event key descriptor")
            keys.append({"type": key_type, "name": key_name})
        result[event_id] = {"event_id": event_id, "name": name, "keys": keys}
    return result


def decode_game_event(payload: bytes,
                      descriptors: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    event = parse_wire_fields(payload)
    event_id = _int32(_one(event, 2, 0))
    if not isinstance(event_id, int):
        raise Source1DemoError("game event has no eventid")
    descriptor = descriptors.get(event_id)
    wire_name = _text(_one(event, 1, 2), "game-event name")
    name = wire_name or (descriptor.get("name") if descriptor else None)
    raw_keys = _many_bytes(event, 3)
    declared_keys = descriptor.get("keys", []) if descriptor else []
    if descriptor and len(raw_keys) != len(declared_keys):
        raise Source1DemoError(
            "game-event value count does not match descriptor")
    values: list[dict[str, object]] = []
    for index, key_raw in enumerate(raw_keys):
        key = parse_wire_fields(key_raw)
        key_type = _one(key, 1, 0)
        if not isinstance(key_type, int):
            raise Source1DemoError("game-event value has no type")
        candidates = [(number, wire, kind, expected) for number, (wire, kind, expected) in
                      _GAME_VALUE_FIELDS.items() if number in key]
        if len(candidates) != 1:
            raise Source1DemoError("game-event value is absent or ambiguous")
        number, wire, kind, expected_type = candidates[0]
        if key_type != expected_type:
            raise Source1DemoError(
                f"game-event key type {key_type} disagrees with {kind} value field")
        value = _one(key, number, wire)
        if kind == "float":
            assert isinstance(value, bytes)
            value = struct.unpack("<f", value)[0]
        elif kind == "bool":
            value = bool(value)
        elif kind in {"long", "short", "byte"}:
            value = _int32(value)
        elif kind == "string":
            assert isinstance(value, bytes)
            value = _text(value, "game-event string")
        elif kind == "wstring":
            assert isinstance(value, bytes)
            value = {"byte_length": len(value), "sha256": hashlib.sha256(
                value).hexdigest(), "provenance": "parsed-bytes"}
        declared = declared_keys[index] if index < len(declared_keys) else {}
        declared_type = declared.get("type")
        if declared_type is not None and declared_type != key_type:
            raise Source1DemoError(
                f"game-event key type {key_type} disagrees with descriptor "
                f"type {declared_type}")
        values.append({
            "name": declared.get("name"), "declared_type": declared_type,
            "wire_type": key_type, "kind": kind, "value": value,
            "provenance": "parsed",
        })
    return {
        "event_id": event_id,
        "name": name,
        "values": values,
        "descriptor_status": "parsed" if descriptor else "unknown",
    }


def _c_string(data: bytes, offset: int, label: str,
              max_bytes: int = 4096) -> tuple[str, int]:
    end_limit = min(len(data), offset + max_bytes)
    end = data.find(b"\x00", offset, end_limit)
    if end < 0:
        raise Source1DemoError(f"unterminated/oversize {label}")
    return _text(data[offset:end], label) or "", end + 1


def parse_data_tables(payload: bytes, *,
                      limits: DemoLimits | None = None) -> DataTables:
    """Parse send-table protobufs and the trailing server-class directory."""
    limits = limits or DemoLimits()
    offset = 0
    tables: list[Mapping[str, object]] = []
    table_names: set[str] = set()
    for _ in range(limits.max_net_messages):
        message_type, offset = _varint(payload, offset, "send-table type")
        size, offset = _varint(payload, offset, "send-table length")
        if message_type != 9:
            raise Source1DemoError(
                f"data-tables payload contains message type {message_type}, not 9")
        if size > limits.max_net_message_bytes or offset + size > len(payload):
            raise Source1DemoError("invalid send-table protobuf length")
        fields = parse_wire_fields(payload[offset:offset + size])
        offset += size
        is_end = bool(_one(fields, 1, 0, 0))
        if is_end:
            break
        name = _text(_one(fields, 2, 2), "send-table name")
        if name is None:
            raise Source1DemoError("send table lacks net_table_name")
        if name in table_names:
            raise Source1DemoError(f"duplicate send-table name {name!r}")
        table_names.add(name)
        props = []
        for prop_raw in _many_bytes(fields, 4):
            prop = parse_wire_fields(prop_raw)
            low_raw = _one(prop, 7, 5, struct.pack("<f", 0.0))
            high_raw = _one(prop, 8, 5, struct.pack("<f", 0.0))
            assert isinstance(low_raw, bytes) and isinstance(high_raw, bytes)
            props.append({
                "type": _one(prop, 1, 0),
                "var_name": _text(_one(prop, 2, 2), "sendprop name"),
                "flags": _one(prop, 3, 0, 0),
                "priority": _one(prop, 4, 0, 0),
                "data_table_name": _text(_one(prop, 5, 2), "sendprop table"),
                "num_elements": _one(prop, 6, 0, 0),
                "low_value": struct.unpack("<f", low_raw)[0],
                "high_value": struct.unpack("<f", high_raw)[0],
                "num_bits": _one(prop, 9, 0, 0),
                "provenance": "parsed",
            })
        tables.append({"name": name, "needs_decoder": bool(
            _one(fields, 3, 0, 0)), "properties": props,
            "provenance": "parsed"})
    else:
        raise Source1DemoError("send-table count limit reached")
    if offset + 2 > len(payload):
        raise Source1DemoError("missing server-class count")
    (count,) = struct.unpack_from("<h", payload, offset)
    offset += 2
    if count <= 0 or count > 65535:
        raise Source1DemoError(f"invalid server-class count {count}")
    classes = []
    seen: set[int] = set()
    table_names = {str(table["name"]) for table in tables}
    for _ in range(count):
        if offset + 2 > len(payload):
            raise Source1DemoError("truncated server-class id")
        (class_id,) = struct.unpack_from("<h", payload, offset)
        offset += 2
        if class_id < 0 or class_id >= count or class_id in seen:
            raise Source1DemoError(f"invalid/duplicate server-class id {class_id}")
        seen.add(class_id)
        class_name, offset = _c_string(payload, offset, "server-class name")
        table_name, offset = _c_string(payload, offset, "server-class table")
        if table_name not in table_names:
            raise Source1DemoError(
                f"server class {class_name!r} refers to unknown table {table_name!r}")
        classes.append(DataTableClass(class_id, class_name, table_name))
    if offset != len(payload):
        raise Source1DemoError(f"{len(payload) - offset} trailing data-table bytes")
    return DataTables(tuple(tables), tuple(classes))
