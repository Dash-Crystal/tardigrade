"""Strict, bounded reader for legacy Source 1 ``HL2DEMO`` containers.

The byte layout follows Valve's BSD-licensed ``csgo-demoinfo`` reference.  This
module parses the container and packet-message framing only.  It does not call
packet-entity bit streams "state" before a send-table/entity decoder has
actually reconstructed them.
"""
from __future__ import annotations

import dataclasses
import io
import math
import os
import struct
from pathlib import Path
from typing import BinaryIO, Iterator


DEMO_STAMP = b"HL2DEMO\x00"
DEMO_PROTOCOL = 4
HEADER_SIZE = 1072
CMD_INFO_SIZE = 152  # two democmdinfo_t::Split_t records

COMMAND_NAMES = {
    1: "signon",
    2: "packet",
    3: "synctick",
    4: "consolecmd",
    5: "usercmd",
    6: "datatables",
    7: "stop",
    8: "customdata",
    9: "stringtables",
}


class Source1DemoError(ValueError):
    """A demo is unsupported, truncated, corrupt, or exceeds a safety bound."""


@dataclasses.dataclass(frozen=True)
class DemoLimits:
    max_demo_bytes: int = 8 * 1024 * 1024 * 1024
    max_command_bytes: int = 64 * 1024 * 1024
    max_commands: int = 20_000_000
    max_net_messages: int = 1_000_000
    max_net_message_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")


@dataclasses.dataclass(frozen=True)
class DemoHeader:
    demo_protocol: int
    network_protocol: int
    server_name: str
    client_name: str
    map_name: str
    game_directory: str
    playback_time: float
    playback_ticks: int
    playback_frames: int
    signon_length: int


@dataclasses.dataclass(frozen=True)
class DemoView:
    flags: int
    view_origin: tuple[float, float, float]
    view_angles: tuple[float, float, float]
    local_view_angles: tuple[float, float, float]
    view_origin_resampled: tuple[float, float, float]
    view_angles_resampled: tuple[float, float, float]
    local_view_angles_resampled: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class DemoCommand:
    command: int
    name: str
    tick: int
    player_slot: int
    offset: int
    payload: bytes = b""
    views: tuple[DemoView, ...] = ()
    sequence_in: int | None = None
    sequence_out: int | None = None
    outgoing_sequence: int | None = None


class _Reader:
    def __init__(self, stream: BinaryIO, size: int, limits: DemoLimits) -> None:
        self.stream = stream
        self.size = size
        self.limits = limits
        self.offset = 0

    def read(self, size: int, label: str) -> bytes:
        if size < 0:
            raise Source1DemoError(f"negative {label} length at byte {self.offset}")
        if self.offset + size > self.size:
            raise Source1DemoError(
                f"truncated {label} at byte {self.offset}: need {size}, "
                f"only {self.size - self.offset} remain")
        data = self.stream.read(size)
        if len(data) != size:
            raise Source1DemoError(f"short read for {label} at byte {self.offset}")
        self.offset += size
        return data

    def unpack(self, fmt: str, label: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt), label))

    def block(self, label: str) -> bytes:
        (size,) = self.unpack("<i", f"{label} length")
        if size < 0 or size > self.limits.max_command_bytes:
            raise Source1DemoError(
                f"{label} length {size} exceeds [0, "
                f"{self.limits.max_command_bytes}] at byte {self.offset - 4}")
        return self.read(size, label)


def _cstring(raw: bytes, label: str) -> str:
    value = raw.split(b"\x00", 1)[0]
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Source1DemoError(f"{label} is not UTF-8") from exc


def _view(raw: bytes) -> DemoView:
    values = struct.unpack("<i18f", raw)
    triples = tuple(tuple(values[i:i + 3]) for i in range(1, 19, 3))
    if not all(math.isfinite(v) for triple in triples for v in triple):
        raise Source1DemoError("packet command contains non-finite camera data")
    return DemoView(values[0], *triples)


class Source1DemoReader:
    """Single-pass reader that requires protocol 4 and an explicit stop tag."""

    def __init__(self, stream: BinaryIO, size: int, *,
                 limits: DemoLimits | None = None) -> None:
        self.limits = limits or DemoLimits()
        if size < HEADER_SIZE:
            raise Source1DemoError(f"demo is smaller than {HEADER_SIZE}-byte header")
        if size > self.limits.max_demo_bytes:
            raise Source1DemoError(
                f"demo is {size} bytes; limit is {self.limits.max_demo_bytes}")
        self._reader = _Reader(stream, size, self.limits)
        self.header = self._read_header()
        self._iterated = False

    @classmethod
    def from_bytes(cls, data: bytes, *,
                   limits: DemoLimits | None = None) -> "Source1DemoReader":
        return cls(io.BytesIO(data), len(data), limits=limits)

    @classmethod
    def open(cls, path: os.PathLike[str] | str, *,
             limits: DemoLimits | None = None) -> "_DemoFileContext":
        return _DemoFileContext(Path(path), limits)

    def _read_header(self) -> DemoHeader:
        raw = self._reader.read(HEADER_SIZE, "demo header")
        stamp, demo_protocol, network_protocol, server, client, map_name, game = \
            struct.unpack("<8sii260s260s260s260s", raw[:1056])
        playback_time, ticks, frames, signon = struct.unpack("<fiii", raw[1056:])
        if stamp != DEMO_STAMP:
            raise Source1DemoError(
                f"not a Source 1 HL2DEMO stream (stamp {stamp!r})")
        if demo_protocol != DEMO_PROTOCOL:
            raise Source1DemoError(
                f"unsupported demo protocol {demo_protocol}; expected 4")
        if not math.isfinite(playback_time) or playback_time < 0:
            raise Source1DemoError("invalid playback_time")
        if ticks < 0 or frames < 0 or signon < 0:
            raise Source1DemoError("negative header count")
        return DemoHeader(
            demo_protocol, network_protocol, _cstring(server, "server name"),
            _cstring(client, "client name"), _cstring(map_name, "map name"),
            _cstring(game, "game directory"), playback_time, ticks, frames,
            signon)

    def commands(self) -> Iterator[DemoCommand]:
        if self._iterated:
            raise Source1DemoError("demo command stream is single-pass")
        self._iterated = True
        previous_tick = -1
        for number in range(self.limits.max_commands):
            offset = self._reader.offset
            raw_command, tick, slot = self._reader.unpack("<BiB", "command header")
            if raw_command & 0x80:
                raise Source1DemoError(
                    f"compressed demo command unsupported at byte {offset}")
            if raw_command not in COMMAND_NAMES:
                raise Source1DemoError(
                    f"unknown demo command {raw_command} at byte {offset}")
            if tick < -1 or tick < previous_tick:
                raise Source1DemoError(
                    f"invalid/non-monotonic tick {tick} at byte {offset}")
            previous_tick = tick
            name = COMMAND_NAMES[raw_command]
            kwargs: dict[str, object] = {}
            if raw_command in (1, 2):
                info = self._reader.read(CMD_INFO_SIZE, "packet command info")
                kwargs["views"] = (_view(info[:76]), _view(info[76:]))
                seq_in, seq_out = self._reader.unpack("<ii", "packet sequences")
                kwargs["sequence_in"] = seq_in
                kwargs["sequence_out"] = seq_out
                kwargs["payload"] = self._reader.block("packet payload")
            elif raw_command in (4, 6, 9):
                kwargs["payload"] = self._reader.block(f"{name} payload")
            elif raw_command == 5:
                (kwargs["outgoing_sequence"],) = self._reader.unpack(
                    "<i", "usercmd sequence")
                kwargs["payload"] = self._reader.block("usercmd payload")
            elif raw_command == 8:
                # Valve's reference declares this tag but provides no framing
                # reader. Guessing its payload length would desynchronise the
                # remainder, so it is a hard error.
                raise Source1DemoError(
                    f"customdata command framing unsupported at byte {offset}")
            command = DemoCommand(raw_command, name, tick, slot, offset, **kwargs)
            yield command
            if raw_command == 7:
                if self._reader.offset != self._reader.size:
                    raise Source1DemoError(
                        f"{self._reader.size - self._reader.offset} trailing "
                        "bytes after stop command")
                return
        raise Source1DemoError("command-count limit reached before stop command")


class _DemoFileContext:
    def __init__(self, path: Path, limits: DemoLimits | None) -> None:
        self.path = path
        self.limits = limits
        self._stream: BinaryIO | None = None

    def __enter__(self) -> Source1DemoReader:
        try:
            size = self.path.stat().st_size
            self._stream = self.path.open("rb")
        except OSError as exc:
            raise Source1DemoError(f"cannot open demo {self.path}: {exc}") from exc
        try:
            return Source1DemoReader(self._stream, size, limits=self.limits)
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def __exit__(self, *_args: object) -> None:
        if self._stream is not None:
            self._stream.close()
