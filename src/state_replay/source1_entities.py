"""Source 1 send-table flattening and PacketEntities reconstruction.

The algorithms are a clean-room Python transcription of the documented logic
in Valve's BSD-2-Clause ``csgo-demoinfo`` reference (notably
``demofiledump.cpp`` and ``demofilepropdecode.cpp``).  All reads are bounded;
there are no assert-and-continue recovery paths.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import struct
from typing import Mapping, Sequence

from .source1_demo import Source1DemoError
from .source1_net import DataTables


DPT_INT, DPT_FLOAT, DPT_VECTOR, DPT_VECTORXY = 0, 1, 2, 3
DPT_STRING, DPT_ARRAY, DPT_DATATABLE, DPT_INT64 = 4, 5, 6, 7

SPROP_UNSIGNED = 1 << 0
SPROP_COORD = 1 << 1
SPROP_NOSCALE = 1 << 2
SPROP_NORMAL = 1 << 5
SPROP_EXCLUDE = 1 << 6
SPROP_XYZE = 1 << 7
SPROP_INSIDEARRAY = 1 << 8
SPROP_COLLAPSIBLE = 1 << 11
SPROP_COORD_MP = 1 << 12
SPROP_COORD_MP_LOWPRECISION = 1 << 13
SPROP_COORD_MP_INTEGRAL = 1 << 14
SPROP_CELL_COORD = 1 << 15
SPROP_CELL_COORD_LOWPRECISION = 1 << 16
SPROP_CELL_COORD_INTEGRAL = 1 << 17
SPROP_CHANGES_OFTEN = 1 << 18
SPROP_VARINT = 1 << 19

MAX_EDICTS = 1 << 11
SERIAL_BITS = 10


class BitReader:
    """Least-significant-bit-first Source bit reader."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bit = 0

    @property
    def remaining(self) -> int:
        return len(self.data) * 8 - self.bit

    def read(self, count: int) -> int:
        if count < 0 or count > 64:
            raise Source1DemoError(f"invalid bit count {count}")
        if self.remaining < count:
            raise Source1DemoError(
                f"truncated entity bitstream at bit {self.bit}: need {count}")
        value = 0
        for shift in range(count):
            value |= ((self.data[self.bit >> 3] >> (self.bit & 7)) & 1) << shift
            self.bit += 1
        return value

    def signed(self, count: int) -> int:
        if count <= 0:
            raise Source1DemoError(f"invalid signed bit count {count}")
        value = self.read(count)
        sign = 1 << (count - 1)
        return value - (1 << count) if value & sign else value

    def read_ubitvar(self) -> int:
        value = self.read(6)
        selector = value & 48
        if selector == 16:
            return (value & 15) | (self.read(4) << 4)
        if selector == 32:
            return (value & 15) | (self.read(8) << 4)
        if selector == 48:
            return (value & 15) | (self.read(28) << 4)
        return value

    def varint(self, max_bytes: int) -> int:
        value = 0
        for index in range(max_bytes):
            byte = self.read(8)
            value |= (byte & 0x7f) << (7 * index)
            if not byte & 0x80:
                return value
        raise Source1DemoError("overlong bitstream varint")

    def byte_string(self, length: int) -> bytes:
        if length < 0 or length > 511:
            raise Source1DemoError(f"invalid sendprop string length {length}")
        return bytes(self.read(8) for _ in range(length))

    def c_string(self, max_bytes: int = 1024) -> str:
        raw = bytearray()
        for _ in range(max_bytes):
            byte = self.read(8)
            if byte == 0:
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise Source1DemoError("bitstream string is not UTF-8") from exc
            raw.append(byte)
        raise Source1DemoError("unterminated/oversize bitstream string")

    def bit_bytes(self, count: int) -> bytes:
        if count < 0 or count > 16 * 1024 * 8:
            raise Source1DemoError(f"invalid userdata bit count {count}")
        output = bytearray((count + 7) // 8)
        for index in range(count):
            output[index >> 3] |= self.read(1) << (index & 7)
        return bytes(output)

    def require_zero_padding(self) -> None:
        if self.remaining >= 8:
            raise Source1DemoError(f"unconsumed entity data: {self.remaining} bits")
        if self.remaining and self.read(self.remaining):
            raise Source1DemoError("nonzero entity-data padding")


@dataclasses.dataclass(frozen=True)
class FlatProp:
    path: str
    prop: Mapping[str, object]
    array_element: Mapping[str, object] | None = None


@dataclasses.dataclass
class EntityState:
    index: int
    class_id: int
    class_name: str
    serial: int
    properties: dict[str, object]
    in_pvs: bool = True
    complete_from_baseline: bool = False
    baseline_present: bool = False


@dataclasses.dataclass(frozen=True)
class EntityChange:
    kind: str  # enter, delta, leave, delete
    index: int
    class_id: int | None
    class_name: str | None
    serial: int | None
    properties: Mapping[str, object]
    complete_from_baseline: bool


def flatten_send_tables(tables: DataTables) -> dict[int, tuple[FlatProp, ...]]:
    by_name = {str(table["name"]): table for table in tables.send_tables}
    if len(by_name) != len(tables.send_tables):
        raise Source1DemoError("duplicate send-table name")
    result: dict[int, tuple[FlatProp, ...]] = {}

    def gather_excludes(table: Mapping[str, object], seen: set[str]) -> set[tuple[str, str]]:
        name = str(table["name"])
        if name in seen:
            raise Source1DemoError(f"send-table recursion at {name!r}")
        seen = seen | {name}
        excludes: set[tuple[str, str]] = set()
        for prop in table["properties"]:  # type: ignore[index]
            flags = int(prop["flags"] or 0)
            if flags & SPROP_EXCLUDE:
                target = prop.get("data_table_name")
                var_name = prop.get("var_name")
                if not target or not var_name:
                    raise Source1DemoError("malformed exclusion sendprop")
                excludes.add((str(target), str(var_name)))
            if int(prop["type"] if prop["type"] is not None else -1) == DPT_DATATABLE:
                target = prop.get("data_table_name")
                if target not in by_name:
                    raise Source1DemoError(f"missing nested send table {target!r}")
                excludes |= gather_excludes(by_name[str(target)], seen)
        return excludes

    def gather_into(table: Mapping[str, object], excludes: set[tuple[str, str]],
                    prefix: str, stack: set[str], output: list[FlatProp],
                    local: list[FlatProp]) -> None:
        table_name = str(table["name"])
        if table_name in stack:
            raise Source1DemoError(f"send-table recursion at {table_name!r}")
        stack = stack | {table_name}
        props = table["properties"]  # type: ignore[index]
        for index, prop in enumerate(props):
            flags = int(prop["flags"] or 0)
            var_name = str(prop.get("var_name") or "")
            prop_type = int(prop["type"] if prop["type"] is not None else -1)
            if (flags & (SPROP_INSIDEARRAY | SPROP_EXCLUDE) or
                    (table_name, var_name) in excludes):
                continue
            path = f"{prefix}.{var_name}" if prefix else var_name
            if prop_type == DPT_DATATABLE:
                target = str(prop.get("data_table_name") or "")
                if target not in by_name:
                    raise Source1DemoError(f"missing nested send table {target!r}")
                if flags & SPROP_COLLAPSIBLE:
                    gather_into(by_name[target], excludes, prefix, stack,
                                output, local)
                else:
                    gather_table(by_name[target], excludes, path, stack, output)
            elif prop_type == DPT_ARRAY:
                if index == 0 or not int(props[index - 1]["flags"] or 0) & SPROP_INSIDEARRAY:
                    raise Source1DemoError(f"array {path!r} lacks preceding element prop")
                local.append(FlatProp(path, prop, props[index - 1]))
            elif prop_type in {DPT_INT, DPT_FLOAT, DPT_VECTOR, DPT_VECTORXY,
                              DPT_STRING, DPT_INT64}:
                local.append(FlatProp(path, prop))
            else:
                raise Source1DemoError(f"unsupported sendprop type {prop_type} at {path}")

    def gather_table(table: Mapping[str, object], excludes: set[tuple[str, str]],
                     prefix: str, stack: set[str], output: list[FlatProp]) -> None:
        local: list[FlatProp] = []
        gather_into(table, excludes, prefix, stack, output, local)
        output.extend(local)

    for server_class in tables.server_classes:
        root = by_name[server_class.data_table_name]
        excludes = gather_excludes(root, set())
        props: list[FlatProp] = []
        gather_table(root, excludes, "", set(), props)
        # This stable effective-priority ordering is the swap loop in Valve's
        # FlattenDataTable expressed without pointer mutation.
        props.sort(key=lambda item: 64
                   if int(item.prop["flags"] or 0) & SPROP_CHANGES_OFTEN
                   else int(item.prop["priority"] or 0))
        result[server_class.class_id] = tuple(props)
    return result


def _coord(reader: BitReader) -> float:
    has_int, has_fraction = reader.read(1), reader.read(1)
    if not (has_int or has_fraction):
        return 0.0
    negative = reader.read(1)
    integer = reader.read(14) + 1 if has_int else 0
    fraction = reader.read(5) if has_fraction else 0
    value = integer + fraction / 32.0
    return -value if negative else value


def _coord_mp(reader: BitReader, *, low: bool, integral: bool) -> float:
    in_bounds = bool(reader.read(1))
    if integral:
        has_int = reader.read(1)
        if not has_int:
            return 0.0
        negative = reader.read(1)
        value = float(reader.read(11 if in_bounds else 14) + 1)
    else:
        has_int = reader.read(1)
        negative = reader.read(1)
        integer = reader.read(11 if in_bounds else 14) + 1 if has_int else 0
        fraction = reader.read(3 if low else 5)
        value = integer + fraction / (8.0 if low else 32.0)
    return -value if negative else value


def _float(reader: BitReader, prop: Mapping[str, object]) -> float:
    flags = int(prop["flags"] or 0)
    bits = int(prop["num_bits"] or 0)
    if flags & SPROP_COORD:
        return _coord(reader)
    if flags & SPROP_COORD_MP:
        return _coord_mp(reader, low=False, integral=False)
    if flags & SPROP_COORD_MP_LOWPRECISION:
        return _coord_mp(reader, low=True, integral=False)
    if flags & SPROP_COORD_MP_INTEGRAL:
        return _coord_mp(reader, low=False, integral=True)
    if flags & SPROP_NOSCALE:
        return struct.unpack("<f", struct.pack("<I", reader.read(32)))[0]
    if flags & SPROP_NORMAL:
        negative = reader.read(1)
        value = reader.read(11) / 2047.0
        return -value if negative else value
    if flags & (SPROP_CELL_COORD | SPROP_CELL_COORD_LOWPRECISION |
                SPROP_CELL_COORD_INTEGRAL):
        if bits <= 0 or bits > 32:
            raise Source1DemoError(f"invalid cell-coordinate width {bits}")
        integer = reader.read(bits)
        if flags & SPROP_CELL_COORD_INTEGRAL:
            return float(integer)
        low = bool(flags & SPROP_CELL_COORD_LOWPRECISION)
        return integer + reader.read(3 if low else 5) / (8.0 if low else 32.0)
    if bits <= 0 or bits > 32:
        raise Source1DemoError(f"invalid scaled-float width {bits}")
    raw = reader.read(bits)
    low, high = float(prop["low_value"]), float(prop["high_value"])
    return low + (high - low) * raw / ((1 << bits) - 1)


def _property(reader: BitReader, prop: Mapping[str, object],
              element: Mapping[str, object] | None = None) -> object:
    prop_type = int(prop["type"])
    flags = int(prop["flags"] or 0)
    if flags & SPROP_XYZE:
        raise Source1DemoError(
            f"SPROP_XYZE decoding is unsupported for {prop.get('var_name')!r}")
    bits = int(prop["num_bits"] or 0)
    if prop_type == DPT_INT:
        if flags & SPROP_VARINT:
            raw = reader.varint(5)
            return raw if flags & SPROP_UNSIGNED else (raw >> 1) ^ -(raw & 1)
        if bits <= 0 or bits > 32:
            raise Source1DemoError(f"invalid integer width {bits}")
        return reader.read(bits) if flags & SPROP_UNSIGNED else reader.signed(bits)
    if prop_type == DPT_FLOAT:
        return _float(reader, prop)
    if prop_type in (DPT_VECTOR, DPT_VECTORXY):
        x, y = _float(reader, prop), _float(reader, prop)
        if prop_type == DPT_VECTORXY:
            return [x, y]
        if flags & SPROP_NORMAL:
            negative = reader.read(1)
            z = math.sqrt(max(0.0, 1.0 - x * x - y * y))
            return [x, y, -z if negative else z]
        return [x, y, _float(reader, prop)]
    if prop_type == DPT_STRING:
        raw = reader.byte_string(reader.read(9))
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Source1DemoError("sendprop string is not UTF-8") from exc
    if prop_type == DPT_ARRAY:
        if element is None:
            raise Source1DemoError("array lacks element descriptor")
        count = int(prop["num_elements"] or 0)
        if count <= 0 or count > 4096:
            raise Source1DemoError(f"invalid array capacity {count}")
        count_bits = count.bit_length()
        actual = reader.read(count_bits)
        if actual > count:
            raise Source1DemoError(f"array length {actual} exceeds capacity {count}")
        return [_property(reader, element) for _ in range(actual)]
    if prop_type == DPT_INT64:
        if flags & SPROP_VARINT:
            raw = reader.varint(10)
            return raw if flags & SPROP_UNSIGNED else (raw >> 1) ^ -(raw & 1)
        signed_bits = bits - (0 if flags & SPROP_UNSIGNED else 1)
        if signed_bits < 1 or signed_bits > 64:
            raise Source1DemoError(f"invalid int64 width {bits}")
        negative = reader.read(1) if not flags & SPROP_UNSIGNED else 0
        value = reader.read(signed_bits)
        if negative:
            value = -value
        return value
    raise Source1DemoError(f"unsupported property type {prop_type}")


def _field_index(reader: BitReader, last: int, new_way: bool) -> int:
    if new_way and reader.read(1):
        return last + 1
    if new_way and reader.read(1):
        delta = reader.read(3)
    else:
        delta = reader.read(7)
        selector = delta & 96
        if selector == 32:
            delta = (delta & ~96) | (reader.read(2) << 5)
        elif selector == 64:
            delta = (delta & ~96) | (reader.read(4) << 5)
        elif selector == 96:
            delta = (delta & ~96) | (reader.read(7) << 5)
    return -1 if delta == 0xfff else last + 1 + delta


def _read_properties(reader: BitReader, flattened: Sequence[FlatProp]) -> dict[str, object]:
    new_way = bool(reader.read(1))
    indices: list[int] = []
    last = -1
    while True:
        index = _field_index(reader, last, new_way)
        if index < 0:
            break
        if index >= len(flattened):
            raise Source1DemoError(
                f"field index {index} exceeds flattened table size {len(flattened)}")
        if len(indices) >= 4096:
            raise Source1DemoError("too many changed entity fields")
        indices.append(index)
        last = index
    return {flattened[index].path: _property(
        reader, flattened[index].prop, flattened[index].array_element)
            for index in indices}


class PacketEntityDecoder:
    """Stateful decoder for one demo stream's PacketEntities messages."""

    def __init__(self, tables: DataTables) -> None:
        self.tables = tables
        self.flattened = flatten_send_tables(tables)
        self.classes = {item.class_id: item for item in tables.server_classes}
        self.class_bits = len(tables.server_classes).bit_length()
        self.entities: dict[int, EntityState] = {}
        self.baselines: dict[int, dict[str, object]] = {}
        self.history: dict[int, dict[int, EntityState]] = {}

    def install_baseline(self, class_id: int, entity_data: bytes) -> None:
        """Decode an ``instancebaseline`` entry for a server class."""
        self.baselines[class_id] = self.decode_baseline(class_id, entity_data)

    def decode_baseline(self, class_id: int, entity_data: bytes) -> dict[str, object]:
        if class_id not in self.flattened:
            raise Source1DemoError(f"baseline has unknown class {class_id}")
        reader = BitReader(entity_data)
        decoded = _read_properties(reader, self.flattened[class_id])
        reader.require_zero_padding()
        return decoded

    def apply(self, envelope: Mapping[str, object], *,
              tick: int | None = None) -> tuple[EntityChange, ...]:
        raw = envelope.get("_entity_data_bytes")
        if not isinstance(raw, bytes):
            raise Source1DemoError("PacketEntities envelope lacks entity bytes")
        count = int(envelope.get("updated_entries", 0))
        max_entries = int(envelope.get("max_entries", 0))
        is_delta = bool(envelope.get("is_delta", False))
        if count < 0 or count > max_entries or max_entries > MAX_EDICTS:
            raise Source1DemoError(
                f"invalid PacketEntities counts {count}/{max_entries}")
        reader = BitReader(raw)
        old_entities = copy.deepcopy(self.entities)
        # A corrupt tail must not commit a valid-looking prefix. Delta packets
        # are based on their recorded server-tick snapshot, not blindly on the
        # immediately preceding demo packet.
        delta_from = int(envelope.get("delta_from", 0))
        if is_delta and tick is not None:
            if delta_from not in self.history:
                raise Source1DemoError(
                    f"delta packet at tick {tick} references unavailable "
                    f"baseline tick {delta_from}")
            entities = copy.deepcopy(self.history[delta_from])
        elif is_delta:
            entities = copy.deepcopy(self.entities)
        else:
            entities = {}
        header_base = -1
        for _ in range(count):
            index = header_base + 1 + reader.read_ubitvar()
            header_base = index
            if index < 0 or index >= MAX_EDICTS:
                raise Source1DemoError(f"entity index {index} outside edict range")
            leave = bool(reader.read(1))
            if leave:
                delete = bool(reader.read(1))
                if not is_delta:
                    raise Source1DemoError("leave-PVS record in full update")
                entity = entities.get(index)
                if entity is None:
                    raise Source1DemoError(f"leave for unknown entity {index}")
                kind = "delete" if delete else "leave"
                if delete:
                    del entities[index]
                else:
                    entity.in_pvs = False
                continue
            enter = bool(reader.read(1))
            if enter:
                class_id = reader.read(self.class_bits)
                serial = reader.read(SERIAL_BITS)
                if class_id not in self.classes:
                    raise Source1DemoError(f"entity {index} has unknown class {class_id}")
                values = copy.deepcopy(self.baselines.get(class_id, {}))
                changed = _read_properties(reader, self.flattened[class_id])
                values.update(changed)
                server_class = self.classes[class_id]
                entity = EntityState(index, class_id, server_class.class_name,
                    serial, values, True, False, class_id in self.baselines)
                entities[index] = entity
            else:
                entity = entities.get(index)
                if entity is None or not entity.in_pvs:
                    raise Source1DemoError(f"delta for inactive entity {index}")
                changed = _read_properties(reader, self.flattened[entity.class_id])
                entity.properties.update(changed)
        reader.require_zero_padding()
        # Wire headers construct an authoritative target. Derive transitions
        # only from old -> target; replaying the wire operations themselves can
        # double-delete when delta_from names an older snapshot.
        changes: list[EntityChange] = []
        for index in sorted(set(old_entities) | set(entities)):
            if old_entities.get(index) == entities.get(index):
                continue
            old, new = old_entities.get(index), entities.get(index)
            if new is None:
                assert old is not None
                changes.append(EntityChange("delete", index, old.class_id,
                    old.class_name, old.serial, {}, old.complete_from_baseline))
            elif old is None or (old.class_id, old.serial) != (new.class_id, new.serial):
                changes.append(EntityChange("enter", index, new.class_id,
                    new.class_name, new.serial, dict(new.properties),
                    new.complete_from_baseline))
            elif old.in_pvs and not new.in_pvs:
                changes.append(EntityChange("leave", index, new.class_id,
                    new.class_name, new.serial, {}, new.complete_from_baseline))
            else:
                changes.append(EntityChange("delta", index, new.class_id,
                    new.class_name, new.serial, dict(new.properties),
                    new.complete_from_baseline))
        self.entities = entities
        if tick is not None:
            self.history[tick] = copy.deepcopy(entities)
            if len(self.history) > 8192:
                del self.history[min(self.history)]
        return tuple(changes)


@dataclasses.dataclass
class StringTableState:
    table_id: int
    name: str
    max_entries: int
    user_data_fixed_size: bool
    user_data_size: int
    user_data_size_bits: int
    entries: dict[int, tuple[str, bytes]] = dataclasses.field(default_factory=dict)
    client_entries: dict[int, tuple[str, bytes]] = dataclasses.field(default_factory=dict)
    metadata_origin: str = "svc_CreateStringTable"


class StringTableDecoder:
    """Decode svc_Create/UpdateStringTable payload bitstreams.

    The optional dictionary codec is rejected because Valve's own reference
    cannot decode it.  Ordinary substring-history coding and user data are
    fully bounded.  ``instancebaseline`` entries are installed directly into
    the associated :class:`PacketEntityDecoder`.
    """

    def __init__(self, entities: PacketEntityDecoder) -> None:
        self.entities = entities
        self.tables: list[StringTableState] = []
        self.last_snapshot_removed: tuple[StringTableState, ...] = ()

    def create(self, envelope: Mapping[str, object]) -> tuple[StringTableState,
                                                               tuple[int, ...]]:
        name = envelope.get("name")
        if not isinstance(name, str) or not name:
            raise Source1DemoError("create string table lacks name")
        if any(table.name == name for table in self.tables):
            raise Source1DemoError(f"duplicate string-table name {name!r}")
        max_entries = int(envelope.get("max_entries", 0))
        count = int(envelope.get("num_entries", 0))
        if max_entries <= 0 or max_entries > 65536 or count < 0 or count > max_entries:
            raise Source1DemoError(f"invalid string-table counts {count}/{max_entries}")
        raw = envelope.get("_string_data_bytes")
        if not isinstance(raw, bytes):
            raise Source1DemoError("create string table lacks data bytes")
        table = StringTableState(len(self.tables), name, max_entries,
            bool(envelope.get("user_data_fixed_size", False)),
            int(envelope.get("user_data_size", 0)),
            int(envelope.get("user_data_size_bits", 0)))
        changed = self._decode(table, raw, count)
        baselines = self._decode_baselines(table, changed)
        self.tables.append(table)
        self.entities.baselines.update(baselines)
        return table, changed

    def update(self, envelope: Mapping[str, object]) -> tuple[StringTableState,
                                                               tuple[int, ...]]:
        table_id = int(envelope.get("table_id", -1))
        count = int(envelope.get("num_changed_entries", 0))
        if table_id < 0 or table_id >= len(self.tables):
            raise Source1DemoError(f"update references unknown string table {table_id}")
        table = self.tables[table_id]
        if count < 0 or count > table.max_entries:
            raise Source1DemoError(f"invalid string-table update count {count}")
        raw = envelope.get("_string_data_bytes")
        if not isinstance(raw, bytes):
            raise Source1DemoError("update string table lacks data bytes")
        # Decode transactionally so malformed updates do not partially mutate.
        clone = copy.deepcopy(table)
        changed = self._decode(clone, raw, count)
        baselines = self._decode_baselines(clone, changed)
        self.tables[table_id] = clone
        self.entities.baselines.update(baselines)
        return clone, changed

    def snapshot(self, raw: bytes) -> tuple[tuple[StringTableState,
                                                   tuple[int, ...]], ...]:
        """Apply a command-level ``dem_stringtables`` full snapshot."""
        reader = BitReader(raw)
        old_by_name = {table.name: table for table in self.tables}
        staged: list[StringTableState] = []
        results: list[tuple[StringTableState, tuple[int, ...]]] = []
        table_count = reader.read(8)
        snapshot_names: set[str] = set()
        for _ in range(table_count):
            name = reader.c_string(256)
            if name in snapshot_names:
                raise Source1DemoError(
                    f"duplicate string-table snapshot name {name!r}")
            snapshot_names.add(name)
            count = reader.read(16)
            prior = old_by_name.get(name)
            if prior is None:
                # Snapshot framing does not carry max_entries or fixed-userdata
                # metadata. This is enough to recover the snapshot itself;
                # a later delta for this table is refused by its inferred bound.
                table = StringTableState(len(staged), name, max(count, 1),
                                         False, 0, 0,
                                         metadata_origin="dem_stringtables-inferred")
            else:
                table = copy.deepcopy(prior)
                table.table_id = len(staged)
            staged.append(table)
            if count > table.max_entries:
                raise Source1DemoError(
                    f"snapshot count {count} exceeds table {name!r} capacity")
            table.entries = {}
            table.client_entries = {}
            changed: list[int] = []
            for index in range(count):
                entry_name = reader.c_string(4096)
                data = b""
                if reader.read(1):
                    size = reader.read(16)
                    if size > 16 * 1024:
                        raise Source1DemoError("snapshot string-table userdata too large")
                    data = reader.byte_string(size)
                table.entries[index] = (entry_name, data)
                changed.append(index)
            # Client-side entries are structurally parsed but intentionally do
            # not overwrite the authoritative server entry index space.
            if reader.read(1):
                client_count = reader.read(16)
                for client_index in range(client_count):
                    client_name = reader.c_string(4096)
                    client_data = b""
                    if reader.read(1):
                        size = reader.read(16)
                        if size > 16 * 1024:
                            raise Source1DemoError(
                                "snapshot client userdata too large")
                        client_data = reader.byte_string(size)
                    table.client_entries[client_index] = (
                        client_name, client_data)
            results.append((table, tuple(changed)))
        reader.require_zero_padding()
        baselines: dict[int, dict[str, object]] = {}
        for table, changed in results:
            baselines.update(self._decode_baselines(table, changed))
        present = {table.name for table in staged}
        self.last_snapshot_removed = tuple(
            copy.deepcopy(table) for table in self.tables
            if table.name not in present)
        self.tables = staged
        self.entities.baselines = baselines
        return tuple(results)

    @staticmethod
    def _decode(table: StringTableState, raw: bytes, count: int) -> tuple[int, ...]:
        reader = BitReader(raw)
        if reader.read(1):
            raise Source1DemoError(
                f"string table {table.name!r} uses unsupported dictionary coding")
        entry_bits = (table.max_entries - 1).bit_length()
        last = -1
        history: list[str] = []
        changed: list[int] = []
        for _ in range(count):
            index = last + 1 if reader.read(1) else reader.read(entry_bits)
            last = index
            if index < 0 or index >= table.max_entries:
                raise Source1DemoError(
                    f"string table {table.name!r} index {index} out of range")
            prior_name, prior_data = table.entries.get(index, ("", b""))
            name = prior_name
            if reader.read(1):
                if reader.read(1):
                    history_index = reader.read(5)
                    prefix = reader.read(5)
                    if history_index >= len(history):
                        raise Source1DemoError("invalid string-table history index")
                    if prefix > len(history[history_index]):
                        raise Source1DemoError("invalid string-table history prefix")
                    name = history[history_index][:prefix] + reader.c_string()
                else:
                    name = reader.c_string()
            data = prior_data
            if reader.read(1):
                if table.user_data_fixed_size:
                    if table.user_data_size_bits <= 0:
                        raise Source1DemoError("fixed string-table userdata has no size")
                    data = reader.bit_bytes(table.user_data_size_bits)
                else:
                    size = reader.read(14)
                    data = reader.byte_string(size)
            table.entries[index] = (name, data)
            changed.append(index)
            history.append(name)
            if len(history) > 32:
                history.pop(0)
        reader.require_zero_padding()
        return tuple(changed)

    def _decode_baselines(self, table: StringTableState,
                          changed: Sequence[int]) -> dict[int, dict[str, object]]:
        if table.name != "instancebaseline":
            return {}
        decoded: dict[int, dict[str, object]] = {}
        for index in changed:
            name, data = table.entries[index]
            try:
                class_id = int(name, 10)
            except ValueError as exc:
                raise Source1DemoError(
                    f"instancebaseline key {name!r} is not a class id") from exc
            decoded[class_id] = self.entities.decode_baseline(class_id, data)
        return decoded
