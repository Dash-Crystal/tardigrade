"""
HCI-128 Dataset Reader

128Hz human-computer interaction telemetry.
Motor control + spatial context + skill labels.

Usage:
    from read import load, load_all

    # Single session
    df = load("data/001.tard.zst")

    # All sessions
    df = load_all("data/")

    # Iterate without loading all into memory
    for session_df in iter_sessions("data/"):
        train(session_df)
"""
import struct
import numpy as np
import pandas as pd
import io
from pathlib import Path

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# ═══════════════════════════════════════════════════════
# BINARY DECODER
# ═══════════════════════════════════════════════════════

def _read_varint(data, pos):
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _zigzag_decode(n):
    return (n >> 1) ^ -(n & 1)


def _decode_zz_stream(data, offset, length):
    end = offset + length
    values = []
    pos = offset
    while pos < end:
        v, pos = _read_varint(data, pos)
        values.append(_zigzag_decode(v))
    return values


def _decode_dod_stream(data, offset, length, base, scale=10):
    coded = _decode_zz_stream(data, offset, length)
    if not coded:
        return []
    dod = np.array(coded, dtype=np.int64)
    deltas = np.zeros(len(dod), dtype=np.int64)
    deltas[0] = dod[0]
    for i in range(1, len(dod)):
        deltas[i] = deltas[i-1] + dod[i]
    values = np.zeros(len(dod), dtype=np.int64)
    values[0] = base
    for i in range(1, len(dod)):
        values[i] = values[i-1] + deltas[i]
    return (values / scale).tolist()


def _decode_events(data, offset, length):
    pos = offset
    n, pos = _read_varint(data, pos)
    events = []
    pt = 0
    for _ in range(n):
        dt, pos = _read_varint(data, pos)
        tick = pt + dt
        v, pos = _read_varint(data, pos)
        events.append((tick, _zigzag_decode(v)))
        pt = tick
    return events


def _decode_fire_rle(data, offset, length):
    pos = offset
    n, pos = _read_varint(data, pos)
    runs = []
    pt = 0
    for _ in range(n):
        dt, pos = _read_varint(data, pos)
        start = pt + dt
        dur, pos = _read_varint(data, pos)
        runs.append((start, dur))
        pt = start
    return runs


def _interpolate(reduced, n_full, rate):
    if not reduced:
        return np.zeros(n_full)
    full = np.zeros(n_full)
    for i, v in enumerate(reduced):
        idx = i * rate
        if idx < n_full:
            full[idx] = v
    for i in range(len(reduced) - 1):
        s = i * rate
        e = min((i+1) * rate, n_full)
        for j in range(s+1, e):
            t = (j - s) / rate
            full[j] = reduced[i] + (reduced[i+1] - reduced[i]) * t
    last = (len(reduced)-1) * rate
    if last < n_full:
        full[last:] = reduced[-1]
    return full


def _expand_events(events, n, default=0):
    result = np.full(n, default, dtype=np.int32)
    current = default
    for tick, val in events:
        current = val
        if tick < n:
            result[tick:] = current
    return result


def _expand_fire(runs, n):
    result = np.zeros(n, dtype=np.int32)
    for start, dur in runs:
        for t in range(start, min(start + dur + 1, n)):
            result[t] = 1
    return result


# ═══════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════

def load(filepath):
    """Load a .tard or .tard.zst file → pandas DataFrame.

    Returns DataFrame with columns:
        tick, agent, team, skill, hp,
        x, y, z, heading, elevation,
        input_dx, input_dy, action, zoom

    Each row = one agent at one tick (1/128th second).
    """
    filepath = Path(filepath)
    data = filepath.read_bytes()

    if filepath.suffix == ".zst" or data[:4] == b'\x28\xb5\x2f\xfd':
        if not HAS_ZSTD:
            raise ImportError("pip install zstandard")
        data = zstandard.ZstdDecompressor().decompress(data)

    pos = 0

    # Header
    magic = data[pos:pos+4]; pos += 4
    assert magic == b"TARD", f"Not a .tard file (magic: {magic})"
    version = struct.unpack_from("<H", data, pos)[0]; pos += 2
    match_id = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
    tick_count = struct.unpack_from("<I", data, pos)[0]; pos += 4
    player_count = struct.unpack_from("<B", data, pos)[0]; pos += 1

    players = []
    for _ in range(player_count):
        pid, team, rank = struct.unpack_from("<HBH", data, pos); pos += 5
        players.append({"id": pid, "team": team, "rank": rank})

    n_weapons = struct.unpack_from("<B", data, pos)[0]; pos += 1
    weapons = [""]
    for _ in range(n_weapons):
        w = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
        weapons.append(w)

    # Decode per-player streams → build rows
    all_rows = []

    for pi, player in enumerate(players):
        streams = []
        for _ in range(11):
            slen = struct.unpack_from("<I", data, pos)[0]; pos += 4
            streams.append((pos, slen))
            pos += slen
        x_base, y_base, z_base = struct.unpack_from("<iii", data, pos); pos += 12

        mdx = _decode_zz_stream(data, *streams[0])
        mdy = _decode_zz_stream(data, *streams[1])
        dyaw = [v / 100.0 for v in _decode_zz_stream(data, *streams[2])]
        dpitch = [v / 100.0 for v in _decode_zz_stream(data, *streams[3])]
        n = len(mdx)

        x_red = _decode_dod_stream(data, *streams[4], x_base)
        y_red = _decode_dod_stream(data, *streams[5], y_base)
        z_red = _decode_dod_stream(data, *streams[6], z_base)
        x = _interpolate(x_red, n, 8)
        y = _interpolate(y_red, n, 8)
        z = _interpolate(z_red, n, 8)

        fire = _expand_fire(_decode_fire_rle(data, *streams[7]), n)
        scope = _expand_events(_decode_events(data, *streams[8]), n)
        # stream 9 = weapon (skipped for now), stream 10 = health
        health = _expand_events(_decode_events(data, *streams[10]), n, 100)

        for t in range(n):
            all_rows.append((
                t, player["id"], player["team"], player["rank"],
                int(health[t]),
                round(x[t], 1), round(y[t], 1), round(z[t], 1),
                round(dyaw[t], 2), round(dpitch[t], 2),
                mdx[t], mdy[t],
                int(fire[t]), int(scope[t])
            ))

    return pd.DataFrame(all_rows, columns=[
        "tick", "agent", "team", "skill", "hp",
        "x", "y", "z", "heading", "elevation",
        "input_dx", "input_dy", "action", "zoom"
    ])


def load_numpy(filepath):
    """Load a .tard file → dict of numpy arrays. Faster than load() for training."""
    df = load(filepath)
    return {col: df[col].values for col in df.columns}


def load_all(directory, pattern="*.tard*"):
    """Load all .tard files in a directory → single DataFrame with session_id column."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    dfs = []
    for i, f in enumerate(files):
        df = load(f)
        df["session"] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def iter_sessions(directory, pattern="*.tard*"):
    """Iterate over sessions without loading all into memory."""
    directory = Path(directory)
    for f in sorted(directory.glob(pattern)):
        yield load(f)


def info(filepath):
    """Print summary of a .tard file without fully decoding."""
    filepath = Path(filepath)
    data = filepath.read_bytes()
    if data[:4] == b'\x28\xb5\x2f\xfd':
        data = zstandard.ZstdDecompressor().decompress(data)

    pos = 4  # skip magic
    version = struct.unpack_from("<H", data, pos)[0]; pos += 2
    match_id = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
    tick_count = struct.unpack_from("<I", data, pos)[0]; pos += 4
    player_count = struct.unpack_from("<B", data, pos)[0]; pos += 1

    raw_size = len(data)
    compressed_size = filepath.stat().st_size

    print(f"File:       {filepath.name}")
    print(f"Players:    {player_count}")
    print(f"Ticks:      {tick_count:,}")
    print(f"Duration:   {tick_count/128:.0f}s ({tick_count/128/60:.1f}min)")
    print(f"Rows:       ~{tick_count * player_count:,}")
    print(f"Size:       {compressed_size/1e6:.1f}MB" +
          (f" ({raw_size/1e6:.1f}MB decompressed)" if compressed_size != raw_size else ""))
    print(f"Hz:         128")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python read.py <file.tard[.zst]>")
        print("       python read.py info <file.tard[.zst]>")
        sys.exit(1)
    if sys.argv[1] == "info":
        info(sys.argv[2])
    else:
        df = load(sys.argv[1])
        print(df)
        print(f"\n{len(df):,} rows, {df['agent'].nunique()} agents, {df['tick'].max()} ticks")
