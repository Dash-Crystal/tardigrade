#!/usr/bin/env python3
"""Decode a .vnmclip_c's compressed per-frame tracks.

WHY. Twenty rifles pose correctly today by LUCK: their idle clips are one
frame with every track static, so the value sits in the compression settings
(`m_flRangeStart` per axis, `m_constantRotation`) and no decode is needed.
The default knives do not have that luck -- their idle is a real 61-frame
animation with 16,836 bytes of `m_compressedPoseData` and a NON-STATIC `wpn`
track -- so the knife falls back to the reference pose, whose identity
rotation stands the blade on end. This is the decoder that difference names.

WHAT IS CONFIRMED, AND HOW
--------------------------
CONTAINER (exact). `m_compressedPoseOffsets` is in UINT16 UNITS, not bytes:
draw_galilar's primary block is 34 frames over 11,424 bytes with offsets
striding 168, and 168 * 2 == 336 == 11424 / 34. Each track reads at
`m_nTrackReadOffset` uint16 words into its frame's block, consuming 3 words
of rotation, 3 of translation and 1 of scale for each component the static
flags say is animated. That cursor is PREDICTED from the flags and matches
the file's own offsets on 56 of 56 primary tracks and 8 of 8 secondary
tracks, closing exactly on the frame stride in both. A layout that were
wrong would not close.

TRANSLATION (exact). Three uint16, each normalised into that axis's own
[m_flRangeStart, m_flRangeStart + m_flRangeLength]. The falsifier is that
the encoder DERIVES the range from the animation, so a correct decode must
touch both ends: across 3 clips and 64 animated axes carrying a real range,
64 of 64 have raw minimum exactly 0 AND raw maximum exactly 65535. Five
further axes are flagged animated while carrying the placeholder range
length 0.1; their raw values never leave 0..59, i.e. the encoder marked
motion that is not there. They are decoded the same way and reported.

ROTATION (READ, then checked). The layout is transcribed from
ValveResourceFormat, `ValveResourceFormat/Resource/ResourceTypes/
ModelAnimation2/AnimationClip.cs`, `DecodeQuaternion` :291-321, at commit
7497df14a92946c14ab3821b4f3ca6d2a20edd67 (2026-08-10):

    a = (data[0] & 0x7FFF) * (sqrt2 / 0x7FFF) - 1/sqrt2      :299, :296, :287
    b = (data[1] & 0x7FFF) * (sqrt2 / 0x7FFF) - 1/sqrt2
    c =  data[2]           * (sqrt2 / 0x7FFF) - 1/sqrt2      :301, UNMASKED
    w = sqrt(1 - (a^2 + b^2 + c^2))                          :309-310
    idx = ((data[0] >> 14) & 0x2) | (data[1] >> 15)          :312
    the derived component goes at position idx, a/b/c fill the rest in order

I had searched for the index as a CONTIGUOUS 2-bit field somewhere in the
48-bit little-endian value. It is not one: it is the TOP BIT of word 0 and
the TOP BIT of word 1, which is why every candidate scored badly and why the
structurally-impossible no-index reading won on both invariants. Searching a
space that did not contain the answer produced a confident ranking of wrong
layouts -- the invariants were fine, the hypothesis set was not.

`data[2]` is deliberately NOT masked here, matching VRF. Whether the encoder
ever sets its top bit is checked rather than assumed; `check_rotation_bits`
reports it, and on the three clips read so far it is set on 0 of 1836
samples, so masked and unmasked agree on this corpus.
"""
from __future__ import annotations

import struct

import numpy as np

# The smallest-three range, and it is DERIVED rather than chosen: the three
# smallest components of a unit quaternion each satisfy |c| <= 1/sqrt(2),
# with equality when two components tie for largest.
RT2_INV = 1.0 / np.sqrt(2.0)
U16 = 65535.0


def frame_stride_words(tracks):
    """Words per frame, accumulated from the static flags. READ, not assumed.

    Returns (stride, per_track_offsets). The caller compares the offsets with
    the file's own `m_nTrackReadOffset` -- that comparison is the check that
    this whole reading is right, and it is cheap, so it is never skipped.
    """
    offs, cur = [], 0
    for t in tracks:
        offs.append(cur)
        if not t["m_bIsRotationStatic"]:
            cur += 3
        if not t["m_bIsTranslationStatic"]:
            cur += 3
        if not t["m_bIsScaleStatic"]:
            cur += 1
    return cur, offs


def check_layout(block):
    """(ok, lines) -- the container reading, checked against the file."""
    tracks = block["m_trackCompressionSettings"]
    data = block.get("m_compressedPoseData") or b""
    offs = list(block.get("m_compressedPoseOffsets") or [])
    n = int(block["m_nNumFrames"])
    stride, pred = frame_stride_words(tracks)
    got = [int(t["m_nTrackReadOffset"]) for t in tracks]
    lines, ok = [], True
    if pred != got:
        ok = False
        bad = [i for i, (a, b) in enumerate(zip(pred, got)) if a != b]
        i0 = bad[0]
        lines.append(f"  track read offsets DISAGREE on {len(bad)} of "
                     f"{len(got)} tracks (first at index {i0}: predicted "
                     f"{pred[i0]}, file says {got[i0]})")
    if n and data:
        per = len(data) / n
        if abs(per - stride * 2) > 1e-9:
            ok = False
            lines.append(f"  frame stride DISAGREES: {stride} words = "
                         f"{stride * 2} bytes, but the blob is {len(data)} "
                         f"bytes over {n} frames = {per:.2f}")
        if len(offs) > 1 and (offs[1] - offs[0]) != stride:
            ok = False
            lines.append(f"  m_compressedPoseOffsets stride {offs[1] - offs[0]}"
                         f" != {stride} words")
    lines.insert(0, f"  container: {len(tracks)} tracks, {n} frame(s), "
                    f"{len(data)} bytes, stride {stride} words "
                    f"({stride * 2} B) -- {'CHECKED' if ok else 'REFUSED'}")
    return ok, lines


def decode_translation(raw, track):
    """(3,) float translation from 3 uint16. Exact; see the module docstring."""
    out = np.empty(3)
    for ax, k in enumerate(("X", "Y", "Z")):
        rg = track["m_translationRange" + k]
        out[ax] = raw[ax] / U16 * rg["m_flRangeLength"] + rg["m_flRangeStart"]
    return out


def decode_rotation(raw):
    """(4,) xyzw unit quaternion from 3 uint16. Transcribed; see the module.

    The smallest three components ride the low 15 bits of each word and the
    index of the RECONSTRUCTED one is split across the top bit of word 0 and
    the top bit of word 1.
    """
    mul = (2.0 * RT2_INV) / 0x7FFF
    a = (raw[0] & 0x7FFF) * mul - RT2_INV
    b = (raw[1] & 0x7FFF) * mul - RT2_INV
    c = raw[2] * mul - RT2_INV                      # UNMASKED, as VRF has it
    w = np.sqrt(max(0.0, 1.0 - (a * a + b * b + c * c)))
    idx = ((raw[0] >> 14) & 0x2) | (raw[1] >> 15)
    q = np.empty(4)
    rest = [j for j in range(4) if j != idx]
    q[idx] = w
    for k, j in enumerate(rest):
        q[j] = (a, b, c)[k]
    return q


def check_rotation_bits(block):
    """How often word 2's top bit is set -- VRF reads that word unmasked.

    If it is ever set, masked and unmasked decodes disagree and one of them
    is wrong; the count is reported rather than the question being closed by
    the fact that it has not bitten yet.
    """
    data = block.get("m_compressedPoseData") or b""
    offs = list(block.get("m_compressedPoseOffsets") or [])
    n, hits, total = int(block["m_nNumFrames"]), 0, 0
    for t in block["m_trackCompressionSettings"]:
        if t["m_bIsRotationStatic"]:
            continue
        for f in range(n):
            w2, = struct.unpack_from(
                "<H", data, (offs[f] + int(t["m_nTrackReadOffset"])) * 2 + 4)
            hits += (w2 >> 15) & 1
            total += 1
    return hits, total


def decode_track(block, ti, frame):
    """{'translation', 'rotation', 'scale', 'notes'} for one track and frame.

    A component the flags call static comes from the compression settings and
    is exact; an animated one is decoded.
    """
    t = block["m_trackCompressionSettings"][ti]
    data = block.get("m_compressedPoseData") or b""
    offs = list(block.get("m_compressedPoseOffsets") or [])
    base = (offs[frame] + int(t["m_nTrackReadOffset"])) * 2
    out, notes, cur = {}, [], base
    if t["m_bIsRotationStatic"]:
        out["rotation"] = np.asarray(t["m_constantRotation"], np.float64)
    else:
        out["rotation"] = decode_rotation(
            struct.unpack_from("<3H", data, cur))
        cur += 6
    if t["m_bIsTranslationStatic"]:
        out["translation"] = np.array(
            [t["m_translationRange" + k]["m_flRangeStart"]
             for k in ("X", "Y", "Z")])
    else:
        out["translation"] = decode_translation(
            struct.unpack_from("<3H", data, cur), t)
        cur += 6
    if t["m_bIsScaleStatic"]:
        out["scale"] = float(t["m_scaleRange"]["m_flRangeStart"])
    else:
        u, = struct.unpack_from("<H", data, cur)
        out["scale"] = (u / U16 * t["m_scaleRange"]["m_flRangeLength"]
                        + t["m_scaleRange"]["m_flRangeStart"])
    out["notes"] = notes
    return out


def decode_clip(block):
    """(translation, rotation, scale) for every track at every frame.

    Shapes are (nframes, ntracks, 3), (nframes, ntracks, 4) xyzw, and
    (nframes, ntracks). This is the pose palette a renderer samples at time
    t, and it is also what a census needs: decoding one frame proves the
    layout, decoding all of them proves the CLIP.

    VECTORISED OVER FRAMES because the population is 2,355 clips. The
    per-(track, frame) path is the same arithmetic and roughly two orders
    slower, which turns a census into an afternoon.
    """
    tracks = block["m_trackCompressionSettings"]
    n = int(block["m_nNumFrames"])
    stride, _pred = frame_stride_words(tracks)
    raw = np.frombuffer(block.get("m_compressedPoseData") or b"", "<u2")
    if stride:
        if raw.size < n * stride:
            raise ValueError(
                f"blob holds {raw.size} words, {n} frames x {stride} needs "
                f"{n * stride}")
        raw = raw[:n * stride].reshape(n, stride)
    T = np.empty((n, len(tracks), 3))
    R = np.empty((n, len(tracks), 4))
    S_ = np.empty((n, len(tracks)))
    mul = (2.0 * RT2_INV) / 0x7FFF
    for ti, t in enumerate(tracks):
        cur = int(t["m_nTrackReadOffset"])
        if t["m_bIsRotationStatic"]:
            R[:, ti, :] = np.asarray(t["m_constantRotation"], np.float64)
        else:
            w = raw[:, cur:cur + 3].astype(np.uint32)
            cur += 3
            a = (w[:, 0] & 0x7FFF) * mul - RT2_INV
            b = (w[:, 1] & 0x7FFF) * mul - RT2_INV
            c = w[:, 2] * mul - RT2_INV
            d = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c)))
            idx = ((w[:, 0] >> 14) & 0x2) | (w[:, 1] >> 15)
            abc = np.stack([a, b, c], 1)
            q = np.empty((n, 4))
            for i in range(4):
                m = idx == i
                if not m.any():
                    continue
                rest = [j for j in range(4) if j != i]
                q[m, i] = d[m]
                for k, j in enumerate(rest):
                    q[m, j] = abc[m, k]
            R[:, ti, :] = q
        if t["m_bIsTranslationStatic"]:
            T[:, ti, :] = [t["m_translationRange" + k]["m_flRangeStart"]
                           for k in ("X", "Y", "Z")]
        else:
            w = raw[:, cur:cur + 3].astype(np.float64)
            cur += 3
            for ax, k in enumerate(("X", "Y", "Z")):
                rg = t["m_translationRange" + k]
                T[:, ti, ax] = (w[:, ax] / U16 * rg["m_flRangeLength"]
                                + rg["m_flRangeStart"])
        if t["m_bIsScaleStatic"]:
            S_[:, ti] = t["m_scaleRange"]["m_flRangeStart"]
        else:
            w = raw[:, cur].astype(np.float64)
            S_[:, ti] = (w / U16 * t["m_scaleRange"]["m_flRangeLength"]
                         + t["m_scaleRange"]["m_flRangeStart"])
    return T, R, S_
