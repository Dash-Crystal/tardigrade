"""Cheap fixed-width identities for cold compatibility records.

These IDs are never authentication, runtime routing, or synchronization. They
only preserve a compact deterministic name where an older persisted format
still needs to compare two cold byte strings.
"""

from __future__ import annotations

import zlib


_U64_MASK = (1 << 64) - 1


def content_id_hex(payload: bytes | bytearray | memoryview) -> str:
    """Return one deterministic 32-byte-shaped non-cryptographic ID."""

    view = memoryview(payload).cast("B")
    length = len(view) & _U64_MASK
    crc = zlib.crc32(view) & 0xFFFF_FFFF
    adler = zlib.adler32(view) & 0xFFFF_FFFF
    checksums = (crc << 32) | adler
    words = (
        length,
        checksums,
        ((length << 23) | (length >> 41))
        ^ checksums
        ^ 0x494A_4943_4845_434B,
        (((checksums << 37) | (checksums >> 27)) & _U64_MASK)
        ^ length
        ^ 0x5355_4D2D_5631_0000,
    )
    return "".join(f"{word & _U64_MASK:016x}" for word in words)


__all__ = ("content_id_hex",)
