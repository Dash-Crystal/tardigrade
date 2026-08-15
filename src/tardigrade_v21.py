"""Lossless, lazy reader for the Tardigrade HCI-128 v2.1 wire.

The upstream tabular reader is useful for exploration but expands every
player/tick into pandas rows, drops weapon and location values, and linearly
interpolates positions.  This module keeps the source representation intact,
decodes one player only when selected, and makes causal hold versus visual
interpolation an explicit caller choice.
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

from compatibility_identity import content_id_hex


TARDIGRADE_V21_MAGIC = b"TR21"
TARDIGRADE_V21_VERSION_FIELD = 2
TARDIGRADE_V21_NATIVE_TICK_RATE = 128
TARDIGRADE_V21_POSITION_SAMPLE_TICKS = 8

FULL_RATE_STREAM_NAMES = (
    "mouse_dx",
    "mouse_dy",
    "view_yaw_centidegrees",
    "view_pitch_centidegrees",
    "forward_move",
    "left_move",
)
POSITION_STREAM_NAMES = ("position_x", "position_y", "position_z")
EVENT_STREAM_NAMES = (
    "scope",
    "weapon_index",
    "health",
    "key_forward",
    "key_back",
    "key_left",
    "key_right",
    "key_reload",
    "key_use",
    "ducking",
    "walking",
    "airborne",
    "ammo",
    "shots_fired",
    "zoom_level",
    "armor",
    "flash",
    "spotted",
    "bomb_planted",
    "defusing",
    "freeze_period",
    "location_index",
    "aim_punch_centidegrees",
)
TARDIGRADE_V21_STREAM_NAMES = (
    *FULL_RATE_STREAM_NAMES,
    *POSITION_STREAM_NAMES,
    "fire",
    *EVENT_STREAM_NAMES,
)

_PLAYER = struct.Struct("<HBHff")
_POSITION_BASES = struct.Struct("<iii")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")


@dataclass(frozen=True, slots=True)
class TardigradeV21PlayerDescriptor:
    player_id: int
    team: int
    rank: int
    initial_yaw_degrees: float
    initial_pitch_degrees: float


@dataclass(frozen=True, slots=True)
class TardigradeObservedValue:
    present: bool
    value: int | str | None
    source_tick: int | None
    encoded_value: int | None = None


@dataclass(frozen=True, slots=True)
class TardigradePositionValue:
    present: bool
    x: float | None
    y: float | None
    z: float | None
    source_tick_begin: int | None
    source_tick_end: int | None
    interpolation_fraction: float | None
    uses_future_sample: bool


@dataclass(frozen=True, slots=True)
class TardigradeV21PlayerSample:
    tick: int
    descriptor: TardigradeV21PlayerDescriptor
    absolute_yaw_degrees: float | None
    absolute_pitch_degrees: float | None
    position: TardigradePositionValue
    full_rate: dict[str, int | None]
    events: dict[str, TardigradeObservedValue]
    fire: TardigradeObservedValue


@dataclass(frozen=True, slots=True)
class _StreamExtent:
    offset: int
    length: int


class _Cursor:
    __slots__ = ("data", "position")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def take(self, count: int) -> bytes:
        end = self.position + count
        if end > len(self.data):
            raise ValueError("TARD 2.1 payload is truncated")
        value = self.data[self.position:end]
        self.position = end
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return _U16.unpack(self.take(_U16.size))[0]

    def u32(self) -> int:
        return _U32.unpack(self.take(_U32.size))[0]

    def fixed_text(self) -> str:
        return self.take(32).rstrip(b"\0").decode(
            "utf-8", errors="replace"
        )


def _read_varint(data: bytes, position: int, end: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while position < end:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7
    raise ValueError("TARD 2.1 varint is truncated")


def _zigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _decode_signed_stream(data: bytes, extent: _StreamExtent) -> array:
    output = array("q")
    position = extent.offset
    end = position + extent.length
    while position < end:
        value, position = _read_varint(data, position, end)
        output.append(_zigzag(value))
    return output


def _signed_stream_count(data: bytes, extent: _StreamExtent) -> int:
    count = 0
    position = extent.offset
    end = position + extent.length
    while position < end:
        _, position = _read_varint(data, position, end)
        count += 1
    return count


def _decode_events(
    data: bytes,
    extent: _StreamExtent,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if extent.length == 0:
        return (), ()
    position = extent.offset
    end = position + extent.length
    count, position = _read_varint(data, position, end)
    ticks = []
    values = []
    tick = 0
    for _ in range(count):
        delta, position = _read_varint(data, position, end)
        encoded, position = _read_varint(data, position, end)
        tick += delta
        ticks.append(tick)
        values.append(_zigzag(encoded))
    if position != end:
        raise ValueError("TARD 2.1 event stream has trailing bytes")
    return tuple(ticks), tuple(values)


def _decode_fire(
    data: bytes,
    extent: _StreamExtent,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if extent.length == 0:
        return (), ()
    position = extent.offset
    end = position + extent.length
    count, position = _read_varint(data, position, end)
    starts = []
    ends = []
    start = 0
    for _ in range(count):
        delta, position = _read_varint(data, position, end)
        duration, position = _read_varint(data, position, end)
        start += delta
        starts.append(start)
        ends.append(start + duration)
    if position != end:
        raise ValueError("TARD 2.1 fire stream has trailing bytes")
    return tuple(starts), tuple(ends)


def _decode_position(
    data: bytes,
    extent: _StreamExtent,
    base: int,
) -> array:
    coded = _decode_signed_stream(data, extent)
    if not coded:
        return array("q")
    output = array("q", [base])
    delta = coded[0]
    value = base
    for delta_of_delta in coded[1:]:
        delta += delta_of_delta
        value += delta
        output.append(value)
    return output


def _zstd_decompress(data: bytes) -> bytes:
    try:
        from compression import zstd  # type: ignore[attr-defined]
    except ImportError:
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "reading compressed TARD 2.1 data requires Python's "
                "compression.zstd or the zstandard package"
            ) from error
        return zstandard.ZstdDecompressor().decompress(data)
    return zstd.decompress(data)


class TardigradeV21Player:
    """One lazily decoded player stream."""

    __slots__ = (
        "descriptor",
        "_catalog_locations",
        "_catalog_weapons",
        "_events",
        "_fire_ends",
        "_fire_starts",
        "_full",
        "_pitch_cumulative",
        "_positions",
        "_yaw_cumulative",
    )

    def __init__(
        self,
        data: bytes,
        descriptor: TardigradeV21PlayerDescriptor,
        streams: tuple[_StreamExtent, ...],
        position_bases: tuple[int, int, int],
        weapons: tuple[str, ...],
        locations: tuple[str, ...],
    ) -> None:
        self.descriptor = descriptor
        self._catalog_weapons = weapons
        self._catalog_locations = locations
        self._full = {
            name: _decode_signed_stream(data, streams[index])
            for index, name in enumerate(FULL_RATE_STREAM_NAMES)
        }
        self._positions = tuple(
            _decode_position(data, streams[6 + index], base)
            for index, base in enumerate(position_bases)
        )
        self._fire_starts, self._fire_ends = _decode_fire(data, streams[9])
        self._events = {
            name: _decode_events(data, streams[10 + index])
            for index, name in enumerate(EVENT_STREAM_NAMES)
        }
        self._yaw_cumulative = self._cumulative(
            self._full["view_yaw_centidegrees"]
        )
        self._pitch_cumulative = self._cumulative(
            self._full["view_pitch_centidegrees"]
        )

    @staticmethod
    def _cumulative(values: array) -> array:
        output = array("q")
        total = 0
        for value in values:
            total += value
            output.append(total)
        return output

    @property
    def sample_count(self) -> int:
        return len(self._full["mouse_dx"])

    def _event(self, name: str, tick: int) -> TardigradeObservedValue:
        ticks, values = self._events[name]
        index = bisect_right(ticks, tick) - 1
        if index < 0:
            return TardigradeObservedValue(False, None, None, None)
        encoded_value = values[index]
        value: int | str | None = encoded_value
        if name == "weapon_index":
            value = (
                self._catalog_weapons[value]
                if 0 <= value < len(self._catalog_weapons)
                else None
            )
        elif name == "location_index":
            value = (
                self._catalog_locations[value]
                if 0 <= value < len(self._catalog_locations)
                else None
            )
        return TardigradeObservedValue(
            True, value, ticks[index], encoded_value
        )

    def _fire(self, tick: int) -> TardigradeObservedValue:
        index = bisect_right(self._fire_starts, tick) - 1
        active = index >= 0 and tick <= self._fire_ends[index]
        return TardigradeObservedValue(
            True, int(active), tick, int(active)
        )

    @property
    def position_sample_ticks(self) -> int:
        """Native ticks per position sample, detected from the wire.

        Packs published before 2026-08-06 carry one position sample per
        8 native ticks; upstream ea7bcf2 re-encoded every pack at full
        rate (one sample per tick) without changing the version field.
        The rate is therefore NOT a constant and must be derived per
        player, or positions are silently read from the wrong index.
        """
        available = min(map(len, self._positions), default=0)
        if available <= 1:
            return TARDIGRADE_V21_POSITION_SAMPLE_TICKS
        return max(1, round(self.sample_count / available))

    def _position(
        self,
        tick: int,
        projection: str,
    ) -> TardigradePositionValue:
        available = min(map(len, self._positions))
        if available == 0:
            return TardigradePositionValue(
                False, None, None, None, None, None, None, False
            )
        rate = self.position_sample_ticks
        sample = min(tick // rate, available - 1)
        begin_tick = sample * rate
        if projection == "causal_hold" or sample + 1 >= available:
            values = tuple(axis[sample] / 10.0 for axis in self._positions)
            return TardigradePositionValue(
                True,
                *values,
                begin_tick,
                begin_tick,
                0.0,
                False,
            )
        if projection != "visual_linear":
            raise ValueError(f"unknown TARD position projection {projection!r}")
        # D1 (upstream, fixed d521763; v22 default): interpolating across a
        # teleport fabricates positions the player never occupied. Judged
        # JOINTLY across axes (per-axis holds fabricate L-paths). 64 world
        # units per 16Hz interval = 640 wire tenths.
        if any(abs(axis[sample + 1] - axis[sample]) > 640
               for axis in self._positions):
            values = tuple(axis[sample] / 10.0 for axis in self._positions)
            return TardigradePositionValue(
                True, *values, begin_tick, begin_tick, 0.0, False)
        fraction = (tick - begin_tick) / rate
        values = tuple(
            (axis[sample] + (axis[sample + 1] - axis[sample]) * fraction)
            / 10.0
            for axis in self._positions
        )
        return TardigradePositionValue(
            True,
            *values,
            begin_tick,
            begin_tick + rate,
            fraction,
            fraction != 0.0,
        )

    def sample(
        self,
        tick: int,
        *,
        # v22 OVERRIDE (owner ruling, no-obsolete-gates): frame-perfect
        # positions are THE behavior; "causal_hold" remains only for
        # callers with an explicit causality contract (the recorded env
        # must never read a future sample), which is a semantics choice,
        # not a format version.
        position_projection: str = "visual_linear",
    ) -> TardigradeV21PlayerSample | None:
        if tick < 0 or tick >= self.sample_count:
            return None
        full = {
            name: values[tick] if tick < len(values) else None
            for name, values in self._full.items()
        }
        yaw = (
            self.descriptor.initial_yaw_degrees
            + self._yaw_cumulative[tick] / 100.0
            if tick < len(self._yaw_cumulative)
            else None
        )
        pitch = (
            self.descriptor.initial_pitch_degrees
            + self._pitch_cumulative[tick] / 100.0
            if tick < len(self._pitch_cumulative)
            else None
        )
        return TardigradeV21PlayerSample(
            tick=tick,
            descriptor=self.descriptor,
            absolute_yaw_degrees=yaw,
            absolute_pitch_degrees=pitch,
            position=self._position(tick, position_projection),
            full_rate=full,
            events={
                name: self._event(name, tick)
                for name in EVENT_STREAM_NAMES
            },
            fire=self._fire(tick),
        )


class TardigradeV21Recording:
    """Parsed catalogs and lazy player streams for one immutable pack."""

    __slots__ = (
        "content_id",
        "locations",
        "map_name",
        "map_name_header",
        "match_id",
        "payload_id",
        "players",
        "tick_count",
        "trailer",
        "version_field",
        "weapons",
        "_data",
        "_decoded_players",
        "_position_bases",
        "_sample_counts",
        "_streams",
    )

    def __init__(self, source: bytes, *, content_id: str | None = None):
        self.content_id = content_id or content_id_hex(source)
        data = (
            _zstd_decompress(source)
            if source[:4] == b"\x28\xb5\x2f\xfd"
            else source
        )
        self._data = data
        self.payload_id = content_id_hex(data)
        cursor = _Cursor(data)
        magic = cursor.take(4)
        self.version_field = cursor.u16()
        if (
            magic != TARDIGRADE_V21_MAGIC
            or self.version_field != TARDIGRADE_V21_VERSION_FIELD
        ):
            raise ValueError(
                f"not TARD 2.1: magic={magic!r}, version={self.version_field}"
            )
        self.match_id = cursor.fixed_text()
        # D10: this header is a COMMENT, not an identifier -- measured
        # live (match 0: header de_mirage, trajectories fit de_inferno
        # +369/all sides). Identify the map from DATA (fit_map).
        self.map_name_header = cursor.fixed_text()
        self.map_name = self.map_name_header  # compat alias; UNVERIFIED
        self.tick_count = cursor.u32()
        self.players = tuple(
            TardigradeV21PlayerDescriptor(*_PLAYER.unpack(cursor.take(_PLAYER.size)))
            for _ in range(cursor.u8())
        )
        self.weapons = self._catalog(cursor)
        self.locations = self._catalog(cursor)
        streams = []
        bases = []
        for _ in self.players:
            player_streams = []
            for _ in TARDIGRADE_V21_STREAM_NAMES:
                length = cursor.u32()
                offset = cursor.position
                cursor.take(length)
                player_streams.append(_StreamExtent(offset, length))
            streams.append(tuple(player_streams))
            bases.append(_POSITION_BASES.unpack(cursor.take(_POSITION_BASES.size)))
        self._streams = tuple(streams)
        self._position_bases = tuple(bases)
        self.trailer = data[cursor.position:]
        self._decoded_players: dict[int, TardigradeV21Player] = {}
        self._sample_counts: dict[int, int] = {}

    @staticmethod
    def _catalog(cursor: _Cursor) -> tuple[str, ...]:
        return ("", *(cursor.fixed_text() for _ in range(cursor.u8())))

    @classmethod
    def load(cls, path: str | Path) -> "TardigradeV21Recording":
        source = Path(path).expanduser().resolve(strict=True).read_bytes()
        return cls(source, content_id=content_id_hex(source))

    def player(self, index: int) -> TardigradeV21Player:
        value = self._decoded_players.get(index)
        if value is None:
            descriptor = self.players[index]
            value = TardigradeV21Player(
                self._data,
                descriptor,
                self._streams[index],
                self._position_bases[index],
                self.weapons,
                self.locations,
            )
            self._decoded_players[index] = value
        return value

    def player_sample_count(self, index: int) -> int:
        value = self._sample_counts.get(index)
        if value is None:
            value = _signed_stream_count(
                self._data, self._streams[index][0]
            )
            self._sample_counts[index] = value
        return value

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "tardigrade/hci-128/v2.1",
            "version_field": self.version_field,
            "match_id": self.match_id,
            "map_name": self.map_name,
            "tick_count": self.tick_count,
            "native_tick_rate": TARDIGRADE_V21_NATIVE_TICK_RATE,
            "player_count": len(self.players),
            "weapon_catalog": self.weapons,
            "location_catalog": self.locations,
            "content_id": self.content_id,
            "payload_id": self.payload_id,
            "opaque_trailer_bytes": len(self.trailer),
        }


__all__ = [
    "EVENT_STREAM_NAMES",
    "FULL_RATE_STREAM_NAMES",
    "TARDIGRADE_V21_NATIVE_TICK_RATE",
    "TARDIGRADE_V21_POSITION_SAMPLE_TICKS",
    "TARDIGRADE_V21_STREAM_NAMES",
    "TardigradeObservedValue",
    "TardigradePositionValue",
    "TardigradeV21Player",
    "TardigradeV21PlayerDescriptor",
    "TardigradeV21PlayerSample",
    "TardigradeV21Recording",
]
