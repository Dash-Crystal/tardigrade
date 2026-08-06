"""
HCI-128 v2.1 Dataset Reader

128Hz human-computer interaction telemetry with full input state.

v2.1 additions over v1:
  - Initial viewangle anchor (yaw₀, pitch₀) per agent
  - Map name
  - WASD keyboard inputs (forward_move, left_move)
  - Key states (FORWARD, BACK, LEFT, RIGHT, RELOAD, USE)
  - Ducking, walking, airborne state
  - Aim punch (recoil offset)
  - Weapon ammo, shots fired, zoom level
  - Armor, flash duration, spotted state
  - Bomb planted, defusing, freeze period
  - Location callout

Usage:
    from read_v2 import load, info

    df = load("data/001.tard.zst")
    info("data/001.tard.zst")
"""
import struct
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


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


# Maximum plausible per-axis displacement (world units) between two 16Hz
# position samples. Physical CS movement stays near 16 units per interval;
# larger jumps are respawns/round resets, and interpolating across them
# fabricates positions the player never occupied (inside geometry).
TELEPORT_UNITS_PER_INTERVAL = 64.0


def _teleport_intervals(axes, max_step=TELEPORT_UNITS_PER_INTERVAL):
    """Sample intervals where ANY axis jumps implausibly far.

    Judged jointly across axes so a teleport holds the full 3D position;
    per-axis decisions could interpolate one axis while holding another,
    fabricating an L-shaped path through geometry.
    """
    count = max((len(a) for a in axes), default=0)
    hold = set()
    for i in range(count - 1):
        for axis in axes:
            if i + 1 < len(axis) and abs(axis[i+1] - axis[i]) > max_step:
                hold.add(i)
                break
    return hold


def _interpolate(reduced, n_full, rate, hold_intervals=frozenset()):
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
        if i in hold_intervals:
            # Discontinuity (respawn/teleport): hold, don't fabricate a path.
            for j in range(s+1, e):
                full[j] = reduced[i]
            continue
        for j in range(s+1, e):
            t = (j - s) / rate
            full[j] = reduced[i] + (reduced[i+1] - reduced[i]) * t
    last = (len(reduced)-1) * rate
    if last < n_full:
        full[last:] = reduced[-1]
    return full


def _expand_events(events, n, default=0):
    result = np.full(n, default, dtype=np.int32)
    for tick, val in events:
        if tick < n:
            result[tick:] = val
    return result


def _expand_fire(runs, n):
    result = np.zeros(n, dtype=np.int32)
    for start, dur in runs:
        for t in range(start, min(start + dur + 1, n)):
            result[t] = 1
    return result


def load(filepath):
    """Load a v2.1 .tard file → pandas DataFrame.

    Returns DataFrame with columns:
        tick, agent, team, skill, hp,
        x, y, z,
        heading, elevation (angular velocity per tick),
        abs_yaw, abs_pitch (reconstructed absolute viewangle; unbounded
            accumulator — abs_yaw_wrapped is the physical heading in
            [-180, 180)),
        input_dx, input_dy (raw mouse),
        fwd_move, left_move (WASD; ternary -1/0/+1, no analog magnitude),
        action (fire), zoom,
        forward, back, left, right, reload, use_key (key states),
        ducking, walking, airborne,
        ammo, shots_fired, zoom_lvl,
        armor, flash, spotted,
        bomb_planted, defusing, freeze,
        aim_punch,
        weapon_idx, weapon, location_idx, location
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
    if magic == b"TARD":
        raise ValueError(
            f"{filepath.name} is a v1 pack (magic TARD); use read.load(), "
            "not read_v2.load(). v1 and v2.1 share the .tard.zst extension; "
            "dispatch on the magic bytes.")
    if magic != b"TR21":
        raise ValueError(f"Not a v2.1 .tard file (magic: {magic!r})")
    version = struct.unpack_from("<H", data, pos)[0]; pos += 2
    match_id = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
    map_name = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
    tick_count = struct.unpack_from("<I", data, pos)[0]; pos += 4
    player_count = struct.unpack_from("<B", data, pos)[0]; pos += 1

    # Player table with viewangle anchors
    players = []
    for _ in range(player_count):
        pid, team, rank, yaw0, pitch0 = struct.unpack_from("<HBHff", data, pos); pos += 13
        players.append({"id": pid, "team": team, "rank": rank, "yaw0": yaw0, "pitch0": pitch0})

    # Weapon dictionary
    n_weapons = struct.unpack_from("<B", data, pos)[0]; pos += 1
    weapons = [""]
    for _ in range(n_weapons):
        w = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
        weapons.append(w)

    # Location dictionary
    n_locations = struct.unpack_from("<B", data, pos)[0]; pos += 1
    locations = [""]
    for _ in range(n_locations):
        loc = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
        locations.append(loc)

    # Decode streams per player
    N_STREAMS = 33  # v2.1 has 33 streams per player
    all_rows = []

    for pi, player in enumerate(players):
        streams = []
        for _ in range(N_STREAMS):
            slen = struct.unpack_from("<I", data, pos)[0]; pos += 4
            streams.append((pos, slen))
            pos += slen
        x_base, y_base, z_base = struct.unpack_from("<iii", data, pos); pos += 12

        # Stream 0-3: mouse dx/dy, dyaw, dpitch (full rate)
        mdx = _decode_zz_stream(data, *streams[0])
        mdy = _decode_zz_stream(data, *streams[1])
        dyaw_cd = _decode_zz_stream(data, *streams[2])
        dpitch_cd = _decode_zz_stream(data, *streams[3])
        dyaw = [v / 100.0 for v in dyaw_cd]
        dpitch = [v / 100.0 for v in dpitch_cd]
        n = len(mdx)

        # Stream 4-5: forward_move, left_move (full rate)
        fwd_move = _decode_zz_stream(data, *streams[4])
        left_move = _decode_zz_stream(data, *streams[5])

        # Stream 6-8: position (16Hz). Teleports (respawn/round reset) are
        # judged jointly across axes and held, not interpolated across.
        x_red = _decode_dod_stream(data, *streams[6], x_base)
        y_red = _decode_dod_stream(data, *streams[7], y_base)
        z_red = _decode_dod_stream(data, *streams[8], z_base)
        hold = _teleport_intervals((x_red, y_red, z_red))
        x = _interpolate(x_red, n, 8, hold)
        y = _interpolate(y_red, n, 8, hold)
        z = _interpolate(z_red, n, 8, hold)

        # Stream 9: fire (RLE)
        fire = _expand_fire(_decode_fire_rle(data, *streams[9]), n)

        # Stream 10-12: scope, weapon, health (events)
        scope = _expand_events(_decode_events(data, *streams[10]), n)
        weapon_idx = _expand_events(_decode_events(data, *streams[11]), n)
        health = _expand_events(_decode_events(data, *streams[12]), n, 100)

        # Stream 13-18: FORWARD, BACK, LEFT, RIGHT, RELOAD, USE
        key_fwd = _expand_events(_decode_events(data, *streams[13]), n)
        key_back = _expand_events(_decode_events(data, *streams[14]), n)
        key_left = _expand_events(_decode_events(data, *streams[15]), n)
        key_right = _expand_events(_decode_events(data, *streams[16]), n)
        key_reload = _expand_events(_decode_events(data, *streams[17]), n)
        key_use = _expand_events(_decode_events(data, *streams[18]), n)

        # Stream 19-21: ducking, walking, airborne
        ducking = _expand_events(_decode_events(data, *streams[19]), n)
        walking = _expand_events(_decode_events(data, *streams[20]), n)
        airborne = _expand_events(_decode_events(data, *streams[21]), n)

        # Stream 22-24: ammo, shots_fired, zoom_lvl
        ammo = _expand_events(_decode_events(data, *streams[22]), n)
        shots_fired = _expand_events(_decode_events(data, *streams[23]), n)
        zoom_lvl = _expand_events(_decode_events(data, *streams[24]), n)

        # Stream 25-27: armor, flash, spotted
        armor = _expand_events(_decode_events(data, *streams[25]), n)
        flash = _expand_events(_decode_events(data, *streams[26]), n)
        spotted = _expand_events(_decode_events(data, *streams[27]), n)

        # Stream 28-30: bomb, defusing, freeze
        bomb = _expand_events(_decode_events(data, *streams[28]), n)
        defusing = _expand_events(_decode_events(data, *streams[29]), n)
        freeze = _expand_events(_decode_events(data, *streams[30]), n)

        # Stream 31: location (integer events indexing the location dictionary)
        location_idx = _expand_events(_decode_events(data, *streams[31]), n)

        # Stream 32: aim_punch (event-encoded)
        aim_punch = _expand_events(_decode_events(data, *streams[32]), n)

        # Reconstruct absolute viewangles from anchor + cumsum. Cumulate in
        # integer centidegrees and divide once: float cumsum drifts. The
        # accumulator is unbounded by construction (a player spinning in one
        # direction passes 360); *_wrapped are the physical headings.
        abs_yaw = player["yaw0"] + np.cumsum(
            np.asarray(dyaw_cd, dtype=np.int64)) / 100.0
        abs_pitch = player["pitch0"] + np.cumsum(
            np.asarray(dpitch_cd, dtype=np.int64)) / 100.0
        abs_yaw_wrapped = ((abs_yaw + 180.0) % 360.0) - 180.0

        for t in range(n):
            all_rows.append((
                t, player["id"], player["team"], player["rank"],
                int(health[t]),
                round(x[t], 1), round(y[t], 1), round(z[t], 1),
                round(dyaw[t], 2), round(dpitch[t], 2),
                round(float(abs_yaw[t]), 2), round(float(abs_pitch[t]), 2),
                round(float(abs_yaw_wrapped[t]), 2),
                mdx[t], mdy[t],
                fwd_move[t] if t < len(fwd_move) else 0,
                left_move[t] if t < len(left_move) else 0,
                int(fire[t]), int(scope[t]),
                int(key_fwd[t]), int(key_back[t]), int(key_left[t]), int(key_right[t]),
                int(key_reload[t]), int(key_use[t]),
                int(ducking[t]), int(walking[t]), int(airborne[t]),
                int(ammo[t]), int(shots_fired[t]), int(zoom_lvl[t]),
                int(armor[t]), int(flash[t]), int(spotted[t]),
                int(bomb[t]), int(defusing[t]), int(freeze[t]),
                int(aim_punch[t]),
                int(weapon_idx[t]),
                weapons[weapon_idx[t]] if 0 <= weapon_idx[t] < len(weapons) else "",
                int(location_idx[t]),
                locations[location_idx[t]] if 0 <= location_idx[t] < len(locations) else "",
            ))

    df = pd.DataFrame(all_rows, columns=[
        "tick", "agent", "team", "skill", "hp",
        "x", "y", "z",
        "heading", "elevation",
        "abs_yaw", "abs_pitch", "abs_yaw_wrapped",
        "input_dx", "input_dy",
        "fwd_move", "left_move",
        "action", "zoom",
        "forward", "back", "left", "right", "reload", "use_key",
        "ducking", "walking", "airborne",
        "ammo", "shots_fired", "zoom_lvl",
        "armor", "flash", "spotted",
        "bomb_planted", "defusing", "freeze",
        "aim_punch",
        "weapon_idx", "weapon", "location_idx", "location",
    ])
    df.attrs["map_name"] = map_name
    df.attrs["match_id"] = match_id
    df.attrs["version"] = version
    return df


def info(filepath):
    filepath = Path(filepath)
    data = filepath.read_bytes()
    if data[:4] == b'\x28\xb5\x2f\xfd':
        data = zstandard.ZstdDecompressor().decompress(data)

    pos = 4
    version = struct.unpack_from("<H", data, pos)[0]; pos += 2
    match_id = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
    map_name = data[pos:pos+32].rstrip(b"\0").decode(); pos += 32
    tick_count = struct.unpack_from("<I", data, pos)[0]; pos += 4
    player_count = struct.unpack_from("<B", data, pos)[0]; pos += 1

    compressed = filepath.stat().st_size
    raw = len(data)

    print(f"File:       {filepath.name}")
    print(f"Version:    {version} (v2.1)")
    print(f"Map:        {map_name}")
    print(f"Match:      {match_id}")
    print(f"Players:    {player_count}")
    print(f"Ticks:      {tick_count:,}")
    print(f"Duration:   {tick_count/128:.0f}s ({tick_count/128/60:.1f}min)")
    print(f"Size:       {compressed/1e6:.1f}MB" +
          (f" ({raw/1e6:.1f}MB decompressed)" if compressed != raw else ""))
    print(f"Fields:     42 columns")
    print(f"Hz:         128")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python read_v2.py <file.tard[.zst]>")
        sys.exit(1)
    if sys.argv[1] == "info":
        info(sys.argv[2])
    else:
        df = load(sys.argv[1])
        print(df)
        print(f"\n{len(df):,} rows, {df['agent'].nunique()} agents")
        print(f"Map: {df.attrs.get('map_name', 'unknown')}")
        print(f"Columns: {list(df.columns)}")
