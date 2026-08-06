# DashCrystals Dataset Export -- Technical Research Document

> **Generated**: 2026-08-04
> **Project**: DashCrystals -- Human-Computer Interaction Data from Game Demos
> **Pipeline**: TARDIGRADE (632 matches / 3,160 player-hours parsed, .tard binary format)
> **Purpose**: Comprehensive technical research for building multi-game, multi-tier data export

---

## Table of Contents

1. [CS2 Screen-Space Projection](#1-cs2-screen-space-projection)
2. [Non-Culling Games with Demo/Replay Systems](#2-non-culling-games-with-demoreplay-systems)
3. [Standard Intelligence and the Computer Use Data Market](#3-standard-intelligence-and-the-computer-use-data-market)
4. [Demo File Rendering at Scale](#4-demo-file-rendering-at-scale)
5. [2D Map Visualization](#5-2d-map-visualization)
6. [BSP Map Geometry for Occlusion](#6-bsp-map-geometry-for-occlusion)

---

## 1. CS2 Screen-Space Projection

### 1.1 CS2 Coordinate System

The Source/Source 2 engine inherits its coordinate system from Quake. It is a **right-handed, Z-up** system:

| Axis | Direction | World Cardinal |
|------|-----------|----------------|
| **+X** | Forward | East |
| **+Y** | Left | North |
| **+Z** | Up | Up |

- `(0, 0, 0)` is the map origin.
- `(0, 0, 1)` is one unit directly above the origin.
- The coordinate system is **right-handed**: if you curl the fingers of your right hand from +X toward +Y, your thumb points in the +Z direction.

**Sources:**
- Valve Developer Community -- Coordinates: https://developer.valvesoftware.com/wiki/Coordinates
- Source Engine Coordinate System (shystudios): http://shystudios.us/blog/source_xyz/source_engine_coordinate.html
- advancedfx Half-Life coordinate system: https://github.com/advancedfx/advancedfx/wiki/Half-Life-coordinate-system

### 1.2 Units (Hammer Units)

Source engine uses **Hammer Units (HU)**:

- **Maps/architecture**: 1 foot = 16 HU, so **1 HU ~ 1.905 cm ~ 0.75 inches**
- **Character models**: 1 foot = 12 HU, so **1 HU ~ 2.54 cm ~ 1 inch**
- Standard mapping convention: **1 HU ~ ~1 inch** for most purposes

**Sources:**
- TF2 Wiki -- Hammer Unit: https://wiki.teamfortress.com/wiki/Hammer_unit
- Valve Developer Community -- Dimensions: https://developer.valvesoftware.com/wiki/Dimensions_(Half-Life_2_and_Counter-Strike:_Source)

### 1.3 Angle Conventions (QAngle)

The Source engine uses `QAngle` for Euler angle representation, stored as `(pitch, yaw, roll)`:

| Component | Axis | Convention | Range |
|-----------|------|------------|-------|
| **Pitch** | Rotation about Y-axis | +down / -up | -90 to +90 |
| **Yaw** | Rotation about Z-axis | +left / -right (CCW from +X viewed from above) | -180 to +180 |
| **Roll** | Rotation about X-axis | +right / -left | -180 to +180 |

Example: `QAngle(-45, 10, 0)` means 45 degrees up, 10 degrees left, 0 roll.

#### AngleVectors Function

The core function that converts QAngle to forward/right/up basis vectors (from `hl2sdk-csgo/mathlib/mathlib_base.cpp`):

```cpp
void AngleVectors(const QAngle &angles, Vector *forward, Vector *right, Vector *up)
{
    float sr, sp, sy, cr, cp, cy;

    SinCos(DEG2RAD(angles[YAW]),   &sy, &cy);
    SinCos(DEG2RAD(angles[PITCH]), &sp, &cp);
    SinCos(DEG2RAD(angles[ROLL]),  &sr, &cr);

    if (forward) {
        forward->x = cp * cy;
        forward->y = cp * sy;
        forward->z = -sp;
    }
    if (right) {
        right->x = (-1*sr*sp*cy + -1*cr*-sy);
        right->y = (-1*sr*sp*sy + -1*cr*cy);
        right->z = -1*sr*cp;
    }
    if (up) {
        up->x = (cr*sp*cy + -sr*-sy);
        up->y = (cr*sp*sy + -sr*cy);
        up->z = cr*cp;
    }
}
```

The inverse function `VectorAngles`:
```
yaw   = atan2(forward.y, forward.x)
pitch = atan2(-forward.z, sqrt(forward.x^2 + forward.y^2))
roll  = atan2(left.z, up.z)
```

**Sources:**
- Valve Developer Community -- QAngle: https://developer.valvesoftware.com/wiki/QAngle
- hl2sdk-csgo mathlib_base.cpp: https://github.com/pmrowla/hl2sdk-csgo/blob/master/mathlib/mathlib_base.cpp
- GuidedHacking -- CSGO Aimbot Angles: https://guidedhacking.com/threads/csgo-aimbot-angles-calculation.10297/

### 1.4 CS2 FOV Values

CS2 uses **Hor+ (Horizontal Plus) FOV scaling**: vertical FOV is fixed, horizontal FOV scales with aspect ratio. Base FOV is **90 degrees horizontal at 4:3 aspect ratio**.

| Aspect Ratio | Horizontal FOV | Vertical FOV | Diagonal FOV |
|-------------|---------------|-------------|-------------|
| **4:3** | **90.00** (base) | ~73.74 | ~100.39 |
| **16:10** | ~100.39 | ~73.74 | ~110.38 |
| **16:9** | **~106.26** | **~73.74** | ~113.66 |
| **21:9** | ~121.28 | ~73.74 | ~126.51 |

**Key insight**: The **vertical FOV is fixed** at approximately 73.74 degrees across all aspect ratios. Only horizontal FOV expands.

#### FOV Conversion Formulas

```
vFOV = 2 * atan(tan(hFOV / 2) * (height / width))
hFOV = 2 * atan(tan(vFOV / 2) * (width / height))
```

Converting from the 4:3 base (90 degrees) to any aspect ratio:
```
hFOV_target = 2 * atan(tan(90/2) * (target_width / target_height) / (4/3))
```

For 16:9: `hFOV = 2 * atan(tan(45) * (16/9) / (4/3)) = 2 * atan(1.0 * 1.333) ~ 106.26`

#### FOV Types in CS2

- **World/Camera FOV**: Fixed at 90 (4:3 base). Cannot be changed in competitive play. `fov_cs_debug` requires `sv_cheats 1`.
- **Viewmodel FOV** (`viewmodel_fov`): Controls ONLY weapon/hand model rendering. Default 60, range 54-68. Uses a **separate projection matrix**. Irrelevant for world-space projection.

#### 4:3 Stretched vs Black Bars

- **Black Bars**: Renders at 90 hFOV with black bars on sides. ~16 less horizontal FOV than 16:9.
- **Stretched**: Same 90 hFOV rendering, but stretched to fill screen. Models appear ~1.33x wider. No actual FOV change.
- Both modes render the exact same scene with the same FOV; only display method differs.

**Sources:**
- Steam Community -- Aimlabs FOV Discussion: https://steamcommunity.com/app/714010/discussions/0/4289187621809424581/
- FOV Calculator Pro -- CS2: https://fovcalculatorpro.com/games/cs2-fov/
- cs.money -- 4:3 vs 16:9: https://cs.money/blog/esports/43-or-169-at-what-resolution-is-it-better-to-play-csgo/
- setup.gg -- CS2 FOV: https://www.setup.gg/game/cs2/fov-viewmodel/
- blog.cs2.ad -- CS2 4:3 Resolution: https://blog.cs2.ad/cs2-4-3-resolution/

### 1.5 View Matrix Reconstruction from Yaw/Pitch/Position

#### Step 1: Compute Basis Vectors from Angles

Given player eye angles `(pitch, yaw)` (roll is typically 0 for player cameras):

```python
import math

def angle_vectors(pitch_deg, yaw_deg):
    """Convert Source engine pitch/yaw to forward/right/up vectors."""
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    forward = [cp * cy, cp * sy, -sp]
    right = [-sy, cy, 0.0]
    up = [sp * cy, sp * sy, cp]

    return forward, right, up
```

Full formula with roll=0 (from Source SDK):
```
forward = (cos(pitch)*cos(yaw),  cos(pitch)*sin(yaw),  -sin(pitch))
right   = (sin(yaw),             -cos(yaw),             0          )
up      = (sin(pitch)*cos(yaw),  sin(pitch)*sin(yaw),   cos(pitch) )
```

#### Step 2: Construct the 4x4 View Matrix

The view matrix transforms world coordinates into camera/eye space:

```
     | right.x    right.y    right.z    -dot(right, eye_pos) |
V =  | up.x       up.y       up.z       -dot(up, eye_pos)    |
     | -fwd.x     -fwd.y     -fwd.z     dot(fwd, eye_pos)    |
     | 0          0          0          1                      |
```

#### Step 3: Construct the Perspective Projection Matrix

Given hFOV, aspect ratio, near/far clipping planes:

```
xScale = 1 / tan(hFOV / 2)
yScale = xScale * aspect

        | xScale   0        0                          0 |
P =     | 0        yScale   0                          0 |
        | 0        0        far/(far-near)             1 |
        | 0        0        -near*far/(far-near)       0 |
```

#### Step 4: World-to-Screen Projection (Combined)

The `FrustumTransform` from Source SDK:

```python
def world_to_screen(world_pos, view_proj_matrix, screen_width, screen_height):
    """Project a 3D world position to 2D screen coordinates."""
    M = view_proj_matrix  # 4x4 matrix

    x = M[0][0]*world_pos[0] + M[0][1]*world_pos[1] + M[0][2]*world_pos[2] + M[0][3]
    y = M[1][0]*world_pos[0] + M[1][1]*world_pos[1] + M[1][2]*world_pos[2] + M[1][3]
    w = M[3][0]*world_pos[0] + M[3][1]*world_pos[1] + M[3][2]*world_pos[2] + M[3][3]

    if w < 0.001:
        return None  # Behind camera

    inv_w = 1.0 / w
    ndc_x = x * inv_w  # range [-1, 1]
    ndc_y = y * inv_w  # range [-1, 1]

    screen_x = (1.0 + ndc_x) * 0.5 * screen_width
    screen_y = (1.0 - ndc_y) * 0.5 * screen_height

    return (screen_x, screen_y)
```

**Sources:**
- RCE Endeavors -- Creating an ESP World To Screen: https://www.codereversing.com/archives/530
- GuidedHacking -- World2Screen Functions: https://guidedhacking.com/threads/world2screen-direct3d-and-opengl-worldtoscreen-functions.8044/
- Reversing The ViewProjection Matrix -- Part 2.1: https://zero-irp.github.io/ViewProj-Blog/part-2.1-view-matrix/
- Reversing The ViewProjection Matrix -- Part 2.2: https://zero-irp.github.io/ViewProj-Blog/part-2.2-projection-matrix/

### 1.6 Handling Different FOV Settings Across Players

In competitive CS2, **all players share the same world FOV**. The base FOV is locked at 90 (4:3 horizontal). The only variable is the player's aspect ratio.

For demo analysis / data export:
1. **Assume uniform FOV**: All players have 90 base FOV. Vertical FOV is always ~73.74.
2. **If aspect ratio is unknown**: Default to 16:9 (most common), giving hFOV ~ 106.26.
3. **Viewmodel FOV** (54-68) affects ONLY weapon model rendering. Uses a separate projection matrix. Ignore for world-space calculations.

### 1.7 demoparser2 Field Reference

| Field Name | Network Property | Description |
|------------|-----------------|-------------|
| `X` | m_vec + m_cell | Player X position (world coords) |
| `Y` | m_vec + m_cell | Player Y position (world coords) |
| `Z` | m_vec + m_cell | Player Z position (world coords) |
| `pitch` | `m_angEyeAngles[0]` | Eye pitch angle (+down/-up) |
| `yaw` | `m_angEyeAngles[1]` | Eye yaw angle (rotation around Z) |
| `velocity_X` | - | X velocity component |
| `velocity_Y` | - | Y velocity component |
| `velocity_Z` | - | Z velocity component |
| `is_airborne` | m_hGroundEntity | Whether player is in the air |
| `aim_punch_angle` | CCSPlayerPawn.m_aimPunchAngle | Recoil punch offset |
| `aim_punch_angle_vel` | CCSPlayerPawn.m_aimPunchAngleVel | Recoil punch velocity |
| `usercmd_viewangle_x` | - | Client-side view angle X (pitch, higher precision) |
| `usercmd_viewangle_y` | - | Client-side view angle Y (yaw, higher precision) |
| `usercmd_viewangle_z` | - | Client-side view angle Z (roll) |
| `usercmd_forward_move` | - | Forward movement input |
| `usercmd_left_move` | - | Lateral movement input |

**Sources:**
- demoparser2 PyPI: https://pypi.org/project/demoparser2/0.0.7/
- demoparser README: https://github.com/LaihoE/demoparser/blob/main/README.md
- Awpy Documentation: https://awpy.readthedocs.io/en/latest/examples/parse_demo.html
- DeepWiki -- cs2-demo-to-dataset: https://deepwiki.com/vast7777/cs2-demo-to-dataset/2.3-demo-parsing-scripts

### 1.8 Complete World-to-Screen Pipeline Summary

Given from demo data:
- Player eye position: `(eye_x, eye_y, eye_z)` -- the X, Y, Z fields
- Player eye angles: `(pitch, yaw)` -- the pitch and yaw fields
- Target position: `(target_x, target_y, target_z)` -- another player's X, Y, Z

Steps:
1. Compute basis vectors from `(pitch, yaw)` using `AngleVectors()`
2. Build view matrix from basis vectors and eye position
3. Build projection matrix from FOV (90 at 4:3 = 106.26 at 16:9) and assumed aspect ratio
4. Multiply `ViewProjection = Projection * View`
5. Transform target position: `clip = VP * [target_x, target_y, target_z, 1]`
6. Perspective divide: `ndc = clip.xy / clip.w`
7. Viewport transform: `screen = (ndc + 1) * 0.5 * [width, height]` (Y inverted)
8. Bounds check: If `clip.w < 0` or NDC outside [-1,1], target is off-screen

---

## 2. Non-Culling Games with Demo/Replay Systems

### 2.1 Architectural Overview: Why Culling Matters

Games fall into two networking architectures that determine whether full world state is available:

**Deterministic Lockstep** (no culling -- all clients have everything): Factorio, OpenTTD, Age of Empires, StarCraft, Warcraft III. Every client runs the full simulation; only player inputs are transmitted. Replays store inputs and require the game engine to reconstruct state.

**Server-Authoritative with PVS/AOI filtering** (culled -- clients get partial state): Source engine games (CS2, GMOD, TF2), Quake, Fortnite, Minecraft. The server only sends entities relevant to the client's position/visibility. Client-recorded demos are therefore incomplete. However, **server-side recordings** (GOTV, SourceTV) bypass culling and capture everything.

**Sources:**
- Factorio FFF-302: https://www.factorio.com/blog/post/fff-302
- Gafferon Games -- Deterministic Lockstep: https://gafferongames.com/post/deterministic_lockstep/
- Snapnet Netcode Architectures: https://www.snapnet.dev/blog/netcode-architectures-part-1-lockstep/

### 2.2 Garry's Mod (GMOD)

**Demo System:** Yes. Standard Source Engine `.dem` format. Record with `record <name>` / `stop`. Files saved to `garrysmod/demos/`.

**File Format:** Standard Source Engine DEM. 1,040-byte header (`HL2DEMO` stamp, protocol versions, server/client/map names, playback duration, tick/frame counts). Body contains command frames: `dem_packet` (entity updates), `dem_consolecmd`, `dem_usercmd` (player input), `dem_datatables` (entity schemas), `dem_stringtables`.

**Data Captured:** All movement data, voice/text chat, player nicknames, SteamIDs, spray data, physics interactions, entity state changes via DataTable networking.

**Entity Scope:** PVS-filtered. Client demos contain **only** entities within the recording player's Potentially Visible Set. GMOD has `TRANSMIT_ALWAYS` per-entity override, but default is PVS-filtered. SourceTV exists for GMOD but is rarely used.

**Version Fragility:** Demos break across GMOD updates ("about every year, developers make changes that affect .dem files"). Must use matching GMOD version for playback.

**Parsers:** No GMOD-specific Python parser. Generic Source engine parsers can read the outer DEM structure. Inner entity data requires game-specific DataTable knowledge. Best generic options: `demoparser2` (Rust+Python, CS2-focused), UntitledParser (C#, supports Portal/HL2/TF2/L4D), Soniclev's parser (Python, very basic).

**Verdict:** Client demos are PVS-culled -- not suitable for full world state. SourceTV demos would work but are uncommon. Parser ecosystem is weak for GMOD specifically.

**Sources:**
- GMod FAQ: https://wiki.facepunch.com/gmod/Help/FAQ
- Controlling Entity Transmission: https://wiki.facepunch.com/gmod/Controlling_Entity_Transmission
- DEM Format: https://developer.valvesoftware.com/wiki/DEM_(file_format)
- Play Old Demos guide: https://steamcommunity.com/sharedfiles/filedetails/?id=1873297334

### 2.3 Wizard101

**Network Protocol:** KingsIsle Networking Protocol (KINP). Packets start with `F00D` (2-byte LE header), followed by 2-byte payload length. Two message categories: Control Messages and DML Messages. DML messages have serviceId (1 byte), messageType (1 byte), length (2 bytes), payload. Service 5 = GAME service.

**Encryption:** AES-GCM encryption was added November 2020 for GAME service packets. Passive packet capture is no longer trivial.

**Entity Position Messages:**
- `MSG_SERVERMOVE` -- server-to-client move update with LocationX, LocationY, LocationZ, direction, mobile ID
- `MSG_SERVERTELEPORT` -- forced teleport with position coordinates and GID
- `MSG_MARK_LOCATION_RESPONSE` -- location message (type 0x6F, service 5)

**Entity Scope:** Unknown. Available protocol documentation does not specify whether the client receives position data for ALL entities in a zone or only nearby ones. This is a critical gap requiring testing.

**Available Tools:**

| Tool | Language | Method | Notes |
|------|----------|--------|-------|
| **wizproxy** | Python 3.11 | MITM proxy, compromises session handshake to get AES keys | `pip install wizproxy`. Dumps to pcapng. |
| **wiz-packet-map** | C++ | Hooks `AuthenticatedSymmetricCipherBase::ProcessData` via vtable overwrite | Intercepts packets before/after AES-GCM |
| **wizwalker** | Python | Direct memory reading | `pip install wizwalker`. Archived June 2022, forks exist. |
| **On-Wiz** | Documentation | Protocol reference | Covers KINP framing, lacks entity specifics |

**Verdict:** No replay system. Packet capture is feasible with wizproxy (Python) but AES-GCM encryption adds complexity. Whether all zone entities are sent to the client is unconfirmed. Medium viability -- needs experimental verification.

**Sources:**
- wizproxy on PyPI: https://pypi.org/project/wizproxy/
- wiz-packet-map: https://github.com/xgladius/wiz-packet-map
- wizwalker: https://github.com/StarrFox/wizwalker
- On-Wiz: https://github.com/latelylk/On-Wiz
- Wizard101 packet docs XML: https://github.com/AmJayden/wizard101-spoofer/blob/main/Wizard101%20packet%20docs.xml

### 2.4 Minecraft (ReplayMod)

**Replay System:** ReplayMod is a mature, widely-used client-side mod. It intercepts all network packets via Netty pipeline hooks at two injection points: `replay_recorder_raw` (before decoding) and `replay_recorder_decoded` (after decoding). Also synthesizes packets the server does not send (e.g., the recording player's own movement).

**File Format (.mcpr):** A ZIP archive containing:

| File | Description |
|------|-------------|
| `recording.tmcpr` | Core recording: binary stream of MC protocol packets with timestamps |
| `metaData.json` | Replay metadata (duration, date, mcVersion, protocol, selfId, players) |
| `entity_positions.json` | Cache of entity positions, generated on first replay load |
| `markers.json` | User-placed markers |
| `mods.json` | (optional) Mod list |

**`recording.tmcpr` binary format** (per packet frame):
```
[timestamp: big-endian int32 (4 bytes)] -- ms from replay start
[length: big-endian int32 (4 bytes)]    -- payload byte count
[varint packetId]                       -- MC protocol packet ID
[packet bytes]                          -- raw payload
```

Current file format version: 14. Max duration: ~24 days 20 hours (int32 ms limit).

**Entity Scope:** Client-side ReplayMod captures ONLY what the recording player's client receives from the server. This includes entities within the server's entity tracking range (typically 11 chunks / ~176 blocks on multiplayer). Entities behind the player ARE included if within tracking range. Entities outside tracking distance are NOT captured.

**Server-Side Alternative:** ServerReplay (Fabric mod) runs server-side and records ALL players and entities, including container contents and player inventories. Produces .mcpr files compatible with ReplayMod for viewing.

**Parsers:**

| Parser | Language | Notes |
|--------|----------|-------|
| **ReplayStudio** | Java (Gradle) | Official. Load/save .mcpr, cut/concat replays, filter packets. |
| **mc-replay-go** | Go | Streaming writer, full format spec in code. |
| **PCRC** | Python 3.6+ | Creates .mcpr files (recording bot), does NOT parse existing ones. Uses pyCraft + pynbt. |
| **No dedicated Python parser on PyPI** | -- | Custom scripts can open ZIP + decode binary with `struct.unpack`, but requires MC protocol packet decoding. |

**Verdict:** Client-side ReplayMod is chunk-range-limited, not full world state. ServerReplay (server-side mod) IS full world state. No mature Python parser exists -- would need custom implementation. Format is straightforward to decode (ZIP of timestamped raw packets).

**Sources:**
- ReplayMod .mcpr format: https://www.replaymod.com/forum/thread/609
- Go mcpr package: https://pkg.go.dev/github.com/reallyoldfogie/mc-replay-go/mcpr
- DeepWiki Recording System: https://deepwiki.com/ReplayMod/ReplayMod/2.1-recording-system
- ServerReplay: https://github.com/senseiwells/ServerReplay
- ReplayStudio: https://github.com/ReplayMod/ReplayStudio

### 2.5 Fortnite

**Replay System:** Built on Unreal Engine 4's DemoNetDriver. Uses Local File Streamer writing `.replay` files to `Saved/Demos/`.

**File Format (.replay):**
1. **Header** -- magic number, file version, duration (ms), network version, changelist, friendly name, timestamp, compression flag
2. **Metadata** -- match info (may include encryption keys)
3. **Chunks** -- three types:
   - Baseline data (starting world state)
   - Checkpoints (periodic snapshots, default every 30s)
   - ReplayData (incremental object changes between checkpoints)

ReplayData contains network packets with bunches (small actor channel updates). Compressed with Oodle SDK (lossless). A Kaitai Struct `.ksy` formal spec exists by Kuinox.

**Entity Scope:** Client-side replays capture ONLY entities within the "Replay Region" -- approximately **250m radius** around the recording player. NOT all 100 players. When watching replays, players visibly "enter and exit the Replay Region." Epic's internal server replays (used for FNCS broadcasts) capture all 100 players but are not publicly available.

**Parsers:**

| Parser | Language | Notes |
|--------|----------|-------|
| **FortniteReplayDecompressor** (SL-x-TnT) | C# (.NET) | Most complete. Parse modes: EventsOnly/Minimal/Normal/Full/Debug. Extracts positions. |
| **FortniteReplayDecompressor** (Shiqan) | C# (.NET) | Original. NuGet: "FortniteReplayReader". |
| **replay-reader** (xNocken) | JavaScript | 99-100% parse rate. Configurable parseLevel. |
| **fortnite-replay-reader** | **Python** | Archived (Sept 2022). Alpha. Extracts stats/eliminations. Does NOT extract positions. |
| **fortnite-replay-parser** | **Python** | Minimal. Header + elimination events only. Does NOT extract positions. |

**Data Available (via C# parser, Full parse):** Per-player: display name, Epic ID, platform, bot status, team, eliminations, placement, positions (X/Y/Z + rotation + timestamps at ~1s intervals), landing position, inventory, combat data (shots fired with weapon/damage/critical/fatal/target), health/shield timeline, cosmetics. Also: supply drops, storm zone data, game state.

**Verdict:** Client replays are proximity-culled (~250m radius). Python parsers exist but cannot extract positions -- only the C# parser can. Server replays have everything but are Epic-internal only. Medium-low viability for full world state.

**Sources:**
- FortniteReplayDecompressor ReadTheDocs: https://fortnitereplaydecompressor.readthedocs.io/en/latest/overview/
- UE4 DemoNetDriver: https://dev.epicgames.com/documentation/unreal-engine/demonetdriver-and-streamers-in-unreal-engine
- FortniteReplayDecompressor (SL-x-TnT): https://github.com/SL-x-TnT/FortniteReplayDecompressor
- FortniteReplayDecompressor (Shiqan): https://github.com/Shiqan/FortniteReplayDecompressor
- fortnite-replay-reader PyPI: https://pypi.org/project/fortnite-replay-reader/

### 2.6 Roblox

**Replay System:** No official built-in replay API. "Roblox Moments" captures 30s video clips (screen recording, not data). "Roblox Replay" (2025) is a badge-based gaming history feature, not gameplay replay.

All replay systems are community-built Lua modules:

| Module | Approach | Entities | Storage |
|--------|----------|----------|---------|
| **ReplayModule** | State-based, delta recording | All developer-registered active models | In-memory Lua tables only |
| **ReplayService** | Keyframe at 60 FPS | Single player's character only | In-memory dictionary |
| **VPF Replay Module** | Frame-delta with interpolation | Only explicitly `Register()`-ed objects | In-memory only |
| **Advanced Replay & Match Demo** | 10Hz delta-encoded positions | Players + CollectionService-tagged entities | Roblox DataStores with custom compression |

**Python Parsers:** None. All modules are Lua-only running inside Roblox. No external file format exists. Data lives in Lua memory or Roblox DataStores, neither externally accessible.

**Verdict: NOT VIABLE for DashCrystals.** No standardized replay format, no external access to replay data, no parsers.

**Sources:**
- ReplayModule DevForum: https://devforum.roblox.com/t/replaymodule-non-deterministic-state-based-replay-system/4507014
- ReplayService DevForum: https://devforum.roblox.com/t/replayservice-replay-your-best-moments-any-place-any-time-pre-release-version-available/1241539
- Recording Gameplay Discussion: https://devforum.roblox.com/t/recording-gameplay-replay-storage/2084882

### 2.7 Counter-Strike 2 / Source Engine

**Demo System:** Very mature. `.dem` files based on Source 2 engine with protobuf encoding.

**Two Demo Types:**
- **GOTV (server-side):** ALL player positions for both teams, every tick. 64 ticks/sec. 50-150 MB per match. Server records complete authoritative state.
- **POV (client-side):** Single player's perspective only. PVS-filtered. 10-30 MB. CS2 also has server-side fog-of-war (`CS2FOW`) that restricts what position data clients receive during live play.

**DEM Header (1,040 bytes):** File stamp `HL2DEMO`, demo protocol version, network protocol version, server/client/map names (260 chars each), game directory, playback duration (float32), tick count, frame count, signon length.

**GOTV demos are confirmed to contain ALL entity positions.** They are "server-side recordings that capture the entire match from all players' perspectives" with "tick-accurate positions, shots, and events."

**Parsers (excellent ecosystem):**

| Parser | Language | Speed | Notes |
|--------|----------|-------|-------|
| **demoparser2** | Rust core, Python+JS bindings | 749 MB/s | `pip install demoparser2`. X/Y/Z, velocity, health, armor, weapons, buttons, pitch/yaw. |
| **Awpy** | Rust core, Python bindings | Fast | `pip install awpy`. Returns Polars DataFrames. 20+ event types, 53+ columns. |
| **Clarity** | Java | "Comically fast" | Supports CS2, CS:GO, Dota 2, Deadlock. |
| **DemoFile.Net** | C# | Sub-second | Forward/backward seeking. CS2 + Deadlock. |
| **demoinfocs-golang** | Go | Fast | CS2 + CS:GO. Full POV + GOTV. Most mature Go parser. |

**Data Per Tick:** X/Y/Z positions, velocity (3-axis), health, armor, stamina, duck amount/speed, jump/fall velocity, movement state, button states, pitch, yaw, FOV, active weapon (name, ammo, skin, recoil index, accuracy penalty), team, bomb zone presence, defuse kit. Per-event: kills, damages, weapon fires, grenade throws, bomb plant/defuse, round starts/ends.

**Cross-Game Source Engine Support:**

| Parser | Games Supported |
|--------|----------------|
| UntitledParser (C#) | Portal 1/2, HL2, TF2, L4D 1/2 |
| sdp (TypeScript) | Portal 1/2, HL2, custom engines |
| deadem (JavaScript) | Deadlock, CS2, Dota 2 |
| csgo_demo_parser (Python) | CS:GO (Kaitai + protobuf) |

**Sources:**
- demoparser2: https://github.com/LaihoE/demoparser
- Awpy docs: https://awpy.readthedocs.io/en/latest/examples/parse_demo.html
- DEM Format Wiki: https://developer.valvesoftware.com/wiki/DEM_(file_format)
- demboyz DemFormat.md: https://github.com/SizzlingStats/demboyz/blob/master/docs/DemFormat.md
- CS2FOW HN: https://news.ycombinator.com/item?id=48805965
- DemoFile.Net: https://github.com/saul/demofile-net

### 2.8 Dota 2

**Replay System:** Fully mature. `.dem` files using Source 2 engine with protobuf + Snappy compression.

**File Format:** 8-byte header `PBUFDEM\0`, 4-byte offset to `CDemoFileInfo`. Repeating `(kind, tick, size, message)` sequences; 14 DEM-level message types. Raw uncompressed ~120 MB; parsed JSON output 150 MB-1.2 GB.

**Entity Scope: ALL entity positions are included regardless of fog of war.** Forum discussions confirm that even fogged/invisible units are present in the data -- they are only hidden by the spectator client rendering, not omitted from the file.

**Parsers:**

| Parser | Language | Notes |
|--------|----------|-------|
| **Clarity** | Java | "Comically fast". Entities, combat log, modifiers, voice, game events. |
| **Manta** | Go | Low-level, by Dotabuff. Source 2 only. |
| **gem-dota** | Python | Full Source 2 pipeline: stream decoding, send tables, field paths, entity delta. |
| **Smoke** | Python/Cython | Selective entity parsing (heroes, players, creeps). |
| **Alice** | C++ | ~500ms for a 45-min replay. |

**Data:** K/D/A, gold/XP time-series, net worth, damage by type, stuns, lane metrics, purchase logs, buyback events, ward placements with exact coordinates, Roshan kills, tower/barracks destruction, teamfight breakdowns, draft, courier state, chat.

**Sources:**
- Compendium -- Demo File: https://github.com/skadistats/compendium/blob/master/chapters/01-the-demo-file.md
- Smoke Wiki: https://github.com/skadistats/smoke/wiki/Anatomy-of-a-Dota-2-Replay-File
- Dotabuff Forum -- Fog of War: https://www.dotabuff.com/topics/2014-05-02-fog-of-warinvisible-units-on-replay
- gem-dota: https://github.com/whanyu1212/gem-dota
- Clarity: https://github.com/skadistats/clarity
- Manta: https://github.com/dotabuff/manta

### 2.9 Rocket League

**Replay System:** Fully mature. Binary `.replay` files.

**File Format:** Header (match metadata, goals, scores, player info) + network stream (frame-by-frame). Actor-based model: each game object (car, ball, boost pad) is an "actor" with ActorId. Network stream uses delta encoding with KeyFrames as baselines. Positions as `Vector3f` with `RigidBody` state (location + quaternion rotation).

**Entity Scope: ALL actor positions included every frame.** No fog of war -- every car, ball, and game object is always visible. Delta encoding means only changed values per frame, but full state is reconstructable.

**Parsers (excellent ecosystem):**

| Parser | Language | Notes |
|--------|----------|-------|
| **carball** | Python | `pip install carball`. JSON, protobuf, gzip numpy arrays. |
| **Pyrope** | Python | `from pyrope import Replay`. Simple API. |
| **RocketLeague-ReplayParser** | Python | Outputs pandas DataFrames. Frame-by-frame car/ball data. |
| **Boxcars** | Rust | 100 replays/sec/core. JSON output ~50% smaller. |
| **Rattletrap** | Haskell | Binary replay-to-JSON. Supports all RL versions through 2.46. |

**Data Per Frame:** Car positions (X,Y,Z), rotations (quaternion), velocities, throttle, boost amount, ball position/rotation/velocity. Per match: goals, scores, player names, ping, demolitions, saves, assists.

**Sources:**
- carball on PyPI: https://pypi.org/project/carball/
- Pyrope: https://github.com/rocket-league-replays/pyrope
- RocketLeague-ReplayParser: https://github.com/DivvyCr/RocketLeague-ReplayParser
- Boxcars: https://github.com/nickbabcock/boxcars
- Rattletrap: https://github.com/tfausak/rattletrap

### 2.10 Overwatch 2

**Replay System:** Entirely server-side with NO exportable files. Players view their 10 most recent matches through the in-game viewer only. No mechanism to export, download, or share replay files. No file format, no parsers, no external access.

**Verdict: NOT VIABLE for DashCrystals.**

**Sources:**
- Blizzard Forums -- Replay Files: https://us.forums.blizzard.com/en/overwatch/t/where-are-the-replay-files/367647
- Blizzard Forums -- Save Replays: https://us.forums.blizzard.com/en/overwatch/t/is-it-possible-to-save-replays-somehow/359264

### 2.11 Valorant

**Replay System:** Launched with Patch 11.06 (September 2025). Relatively new.

**File Format:** Proprietary, undocumented, changes with each patch (~2 weeks). Files saved locally. No mature external parsers. Community reverse engineering hampered by format instability. Only available for Competitive and Premier modes. Replays wiped each patch.

**Data:** In-game viewer supports all 10 players' perspectives, free camera, player outlines toggle. Known limitations: ability minimaps (Brimstone smokes etc.) are NOT viewable.

**Verdict:** Files exist locally but no stable parser exists. Format undocumented and changes per patch. High-effort, low-stability target.

**Sources:**
- Valorant Replays Guide: https://playvalorant.com/en-us/news/dev/replays-everything-you-need-to-know/
- Riot Developer Relations Issue #312: https://github.com/RiotGames/developer-relations/issues/312

### 2.12 Other Notable Games

**League of Legends:** `.rofl` replay files exist. Positions included but encryption changes each patch. Parsers: roflxd (https://github.com/fraxiinus/roflxd), ROFL (https://github.com/Mowokuma/ROFL). Medium viability -- fragile per-patch.

**StarCraft 2:** `.SC2Replay` in MPQ format. Official Blizzard `s2protocol` parser (Go). However, unit positions are NOT fully captured -- only units that have inflicted/taken damage appear in `SUnitPositionsEvent`, with 256-unit limit, at periodic intervals. Full reconstruction requires running through the game engine. Source: https://github.com/Blizzard/s2protocol

**Factorio:** All clients have complete map state (deterministic lockstep). Replays record Input Actions. Full world state reconstruction requires the Factorio engine. No external file parser. Source: https://www.factorio.com/blog/post/fff-302

**Terraria:** Server broadcasts NPC data (position, velocity, life, AI state) to all clients with no documented area-of-interest filtering. No built-in replay system. Protocol fully documented. Source: https://seancode.com/terrafirma/net.html

**OpenTTD:** Full deterministic simulation transferred to clients as savegame. Open source (C++). Source: https://github.com/OpenTTD/OpenTTD/blob/master/docs/desync.md

### 2.13 Master Viability Table for DashCrystals

| Game | Replay Files? | Format | ALL Entity Positions? | Python Parser? | Viability |
|------|:---:|---|:---:|---|:---:|
| **CS2 (GOTV)** | Yes | `.dem` (protobuf) | YES | `demoparser2`, `awpy` (pip) | **HIGH** |
| **Dota 2** | Yes | `.dem` (protobuf+Snappy) | YES (incl. fogged) | `gem-dota`, `Smoke` | **HIGH** |
| **Rocket League** | Yes | `.replay` (binary) | YES (no fog of war) | `carball`, `Pyrope` (pip) | **HIGH** |
| **Terraria** | No replay | N/A (live packets) | YES (no AOI filtering) | Protocol documented, custom needed | **MEDIUM-HIGH** |
| **Minecraft** | Yes (.mcpr) | ZIP of raw MC packets | NO (chunk-range limited) | None on PyPI; custom struct decode | **MEDIUM** |
| **MC ServerReplay** | Yes (.mcpr) | Same as above | YES (server-side) | Same | **MEDIUM** |
| **Fortnite** | Yes | `.replay` (Oodle-compressed) | NO (~250m radius) | Python: stats only; C#: positions | **MEDIUM-LOW** |
| **Wizard101** | No replay | Live packets (AES-GCM) | Unknown | `wizproxy` (pip) | **MEDIUM-LOW** |
| **League of Legends** | Yes | `.rofl` (encrypted) | Yes but encrypted | Partial (breaks per patch) | **MEDIUM-LOW** |
| **StarCraft 2** | Yes | `.SC2Replay` (MPQ) | PARTIAL (damaged units only) | `s2protocol` (official) | **MEDIUM-LOW** |
| **Valorant** | Yes | Proprietary | Likely yes | None stable | **LOW** |
| **GMOD** | Yes | `.dem` (Source) | NO (PVS-filtered) | Generic Source parsers only | **LOW** |
| **Factorio** | Yes | Input actions | YES (via engine) | None (requires engine) | **LOW** |
| **Roblox** | No | N/A | Developer-selected only | None | **NOT VIABLE** |
| **Overwatch 2** | No files | Server-side only | N/A | None | **NOT VIABLE** |

### 2.14 Top Recommendations for Expansion

**Tier 1 -- Ready to build on today:**
1. **CS2** -- Already built (TARDIGRADE). `pip install demoparser2`. GOTV demos have all 10 players at 64 ticks/sec.
2. **Dota 2** -- `gem-dota` (Python) or Clarity (Java). Full entity state including fogged units. Large competitive replay archive.
3. **Rocket League** -- `pip install carball`. All actors every frame, no culling by design. Excellent Python tooling.

**Tier 2 -- Feasible with extra work:**
4. **Terraria** -- No replay system but protocol is fully documented and appears to broadcast all NPC data globally. Would need live packet capture + custom parser.
5. **Minecraft (ServerReplay)** -- Requires server-side mod installation but then produces full-world .mcpr files. Would need custom Python parser for MC protocol packets.

---

## 3. Standard Intelligence and the Computer Use Data Market

### 3.1 What is Standard Intelligence?

**Standard Intelligence** (si.inc) is a San Francisco-based AI startup founded in 2024 by Waseem AlShikh. They build foundation models trained on raw screen recording video for computer use. Key facts:

- **Valuation**: ~$500M (as of Sequoia Capital investment)
- **Funding**: Backed by Sequoia Capital; Andrej Karpathy is an angel investor
- **Team size**: ~6 people (as of early reports)
- **Data**: 11 million hours of screen recording data
- **Product**: AI models that can autonomously operate a computer (click, type, navigate) based on natural language instructions
- **Approach**: Train on continuous 30 FPS video + tokenized action annotations (mouse coords in 49 discrete bins, keystrokes as tokens)
- **Contractor data**: Paid contractors for 40,000 hours of labeled ground-truth (actions annotated frame-by-frame)

**Why this matters for DashCrystals**: Standard Intelligence's entire pipeline depends on an Inverse Dynamics Model (IDM) trained on contractor-labeled data. Pre-labeled game telemetry (synchronized input + state data) is exactly the ground-truth data that bootstraps or replaces that IDM step. Game data provides what they spend contractor money to get -- synchronized action-state pairs -- at massively lower cost.

**Sources:**
- Standard Intelligence company: https://si.inc
- Sequoia investment coverage: search results reference Sequoia Capital backing
- Andrej Karpathy angel investment referenced in funding announcements

### 3.2 Computer Use Training Data Format Requirements

Across the industry, the canonical format for computer use training data is **trajectories of (screenshot, action) pairs**:

```
Trajectory = [
    {
        "screenshot": <image or path>,
        "action_type": "click" | "type" | "scroll" | "key" | "drag",
        "coordinates": [x, y],           # normalized [0,1] becoming standard
        "value": "text to type",          # for type actions
        "bounding_box": [x1,y1,x2,y2],   # optional, for element targeting
        "accessibility_tree": {...},       # optional, for DOM/UI state
        "timestamp": float,
        "task_description": "natural language goal"
    },
    ...
]
```

Key format observations:
- **Normalized coordinates** ([0,1] range) are becoming the industry standard over pixel coordinates, for resolution independence
- **PC Agent-E** demonstrated that just 312 high-quality trajectories can bootstrap strong performance
- The richest formats include accessibility tree / DOM state alongside screenshots
- Action granularity varies: some capture every mouse movement, others only discrete actions (click, type)

### 3.3 Companies Buying Human-Computer Interaction Data

**Frontier AI Labs (direct consumers):**
- **Anthropic** -- Computer Use feature (Claude), actively training on interaction data
- **OpenAI** -- Operator product for autonomous computer use
- **Google DeepMind** -- Project Mariner for browser automation
- **Meta** -- Research on UI interaction models
- **Microsoft** -- Copilot Vision, Windows Agent Arena
- **Amazon** -- Nova Act (browser agent), Bedrock Agent capabilities

**Data Intermediaries / Labeling Companies:**
- **Scale AI** -- $29B valuation, major data labeling for all frontier labs
- **Surge AI** -- $25B+ valuation, human annotation platform
- **Mercor** -- $20B valuation, AI-matched human data collection
- **Appen** -- Public company, crowd-sourced data annotation
- **Toloka** -- Crowdsourcing platform for data labeling

**Specialized Computer Use Data Companies:**
- **Claru AI** -- Most direct competitor to DashCrystals. Charges $0.004/sec ($14.40/hr) for synchronized game telemetry capture. Covers Valorant, CS2, Minecraft, GTA V, Roblox, and others with 10K+ hours. Captures screen recording + mouse/keyboard + game state simultaneously.

**Sources:**
- Scale AI valuation: public reporting
- Claru AI: referenced in search results as game telemetry data provider
- Anthropic Computer Use: https://docs.anthropic.com/en/docs/agents-and-tools/computer-use

### 3.4 Pricing for Computer Use Datasets

| Data Type | Price Range | Notes |
|-----------|-------------|-------|
| Commodity annotation (clicks, labels) | $3-10/hr | Basic screen interaction recording |
| Skilled annotation (workflows) | $15-30/hr | Task-oriented computer use sequences |
| Domain expert annotation | $60-500/hr | Specialized software workflows |
| Synchronized game telemetry (Claru AI) | $14.40/hr ($0.004/sec) | Screen + input + game state |
| High-quality labeled trajectories | $50-150/trajectory | Complete task demonstrations |
| Pre-trained action recognition models | $50K-500K/model | Transfer learning starting points |

The market is capacity-constrained: demand for labeled computer use data far exceeds supply. All frontier labs are competing for the same contractor pools.

### 3.5 Existing Datasets for Computer Use Training

| Dataset | Format | Size | Collection Method | Source |
|---------|--------|------|-------------------|--------|
| **Mind2Web** | DOM + action sequences | 2,350 tasks across 137 websites | Human annotators | https://osu-nlp-group.github.io/Mind2Web/ |
| **WebArena** | Browser environment + tasks | 812 tasks across 4 web apps | Programmatic task generation | https://webarena.dev/ |
| **ScreenSpot** | Screenshots + click targets | 1,200+ annotated screens | Manual annotation | Research paper |
| **OSWorld** | Full desktop environment + tasks | 369 tasks across Ubuntu/Win/Mac | Synthetic + human | https://os-world.github.io/ |
| **Windows Agent Arena** | Windows VM + tasks | 154 tasks | Microsoft research | https://microsoft.github.io/WindowsAgentArena/ |
| **VisualWebArena** | Visual web tasks | 910 tasks | Extension of WebArena | Research paper |
| **AndroidWorld** | Mobile interaction | 116 tasks | Google DeepMind | Research paper |
| **SWE-bench** | Code editing tasks | 2,294 GitHub issues | Real GitHub PRs | https://www.swebench.com/ |

### 3.6 Game Telemetry vs Screen Recording for Computer Use Training

**Why game telemetry is potentially superior:**

1. **Perfect action-state synchronization**: Game demos provide frame-exact input-state pairs. Screen recordings require an IDM (Inverse Dynamics Model) to infer actions from frame differences -- this is "the primary error source" in video-only pipelines.

2. **Structured state representation**: Game data provides XYZ positions, velocities, health, inventory as structured data. Screen recordings require OCR/vision models to extract this information.

3. **Massive scale at low cost**: Game demos are generated naturally by players. Standard Intelligence paid contractors for 40K hours; TARDIGRADE already has 3,160 player-hours from organic gameplay.

4. **High-frequency data**: 64-128 Hz tick rate vs. 30 FPS screen recordings. 2-4x more temporal resolution.

5. **Multi-agent data**: Game demos capture all players simultaneously. Screen recordings capture one user.

6. **No privacy concerns**: Game characters, not real people/screens. No PII, no passwords, no sensitive documents.

**Limitations of game data:**

1. **Domain gap**: Game UIs are not desktop/web UIs. Transfer to general computer use is unproven.
2. **Limited action space**: Move, aim, shoot, jump vs. click, drag, type, scroll.
3. **No text input**: Games have minimal typing compared to productivity software.
4. **Synthetic environment**: Game worlds are simulated, not real applications.

**The bridge**: Game data is strongest for training **spatial reasoning**, **reaction-based targeting** (mouse precision/speed), and **multi-step planning under uncertainty** -- all transferable skills. It is weakest for **text-heavy workflows** and **menu navigation**.

### 3.7 Market Size and Growth

- **AI agents market**: $10-12B in 2026, projected to $183-295B by 2033-2035 at 44-50% CAGR
- **Training data market**: $3.3B in 2024, projected to $16B by 2030
- **Computer use specifically**: Fastest-growing segment within AI agents, driven by Anthropic Computer Use, OpenAI Operator, Google Mariner

**Sources:**
- Market size projections from industry analyst reports referenced in search results
- Standard Intelligence data volumes from company announcements
- Claru AI pricing from their documentation

---

## 4. Demo File Rendering at Scale

### 4.1 Core Constraint

CS2 demo files are **event logs, not video**. Only the Source 2 engine can re-render them in 3D. Every platform that produces video from demos runs a real CS2 game client with GPU access. There is no headless mode, no alternative renderer, no shortcut.

### 4.2 The `startmovie` Problem

The native Valve frame-capture command is problematic in CS2:

- **Windows**: `startmovie` is hidden by default. Can be unlocked via `cvar-unhide-s2` plugin + `-insecure` launch flag. Once unlocked, captures TGA frames to disk.
- **Linux**: `startmovie` is completely inaccessible. Valve closed the feature request as "not planned."
- **Syntax** (when available): `startmovie <filename> [tga|avi] [fps]` captures frames during demo playback. `endmovie` stops. The command pauses the game engine to render each frame, so output is always at the specified framerate regardless of real-time performance.

**Sources:**
- Valve Developer Community -- startmovie: https://developer.valvesoftware.com/wiki/Startmovie
- CS2 feature request (Linux startmovie): closed as "not planned" by Valve

### 4.3 Two Practical Recording Paths

#### Path A: HLAE (Windows -- Industry Standard)

**Half-Life Advanced Effects (HLAE)** is the dominant tool for CS2 demo recording:

- **`mirv_streams`**: Pipes rendered frames directly to FFmpeg, bypassing TGA disk bottleneck
- **`mirv_cmd`**: Tick-based automation for automated frame capture (e.g., "at tick 12000, switch to player X, start recording")
- **Multi-stream support**: Can simultaneously capture color, depth, alpha, entity masks
- **Camera control**: Free camera, smooth interpolation, dolly/orbit/track
- **Usage**: Professional esports content creation, highlight reels, analysis clips

**Limitations**: Windows-only. Requires game client with GPU. Not trivially scriptable for batch processing thousands of demos.

**Sources:**
- HLAE: https://github.com/advancedfx/advancedfx
- HLAE Wiki: https://github.com/advancedfx/advancedfx/wiki

#### Path B: cs2rec (Linux -- Emerging)

**FASTCUP cs2rec** is a MetaMod plugin for Linux:

- Hooks `startmovie` TGA output and pipes to FFmpeg
- Runs on dedicated servers
- Less mature but functional for automated pipelines
- Requires a running CS2 server instance with the MetaMod plugin loaded

### 4.4 Headless Rendering

**CS2 cannot render headlessly natively.** Xvfb (virtual framebuffer) fails because CS2 uses Vulkan (not OpenGL), and Vulkan requires real GPU access.

**Viable approach: docker-steam-headless**

- Docker container with a real Xorg server using NVIDIA dummy config + GPU passthrough
- Runs a full desktop environment inside the container
- CS2 launches inside this environment with real Vulkan rendering
- GPU must be physically present (NVIDIA recommended)
- Can be orchestrated with Docker Compose for multiple render workers

**Requirements per render worker:**
- NVIDIA GPU with Vulkan support
- Docker with nvidia-container-toolkit
- ~8 GB RAM for CS2 client
- ~50 GB disk for CS2 installation + frame output

**Sources:**
- docker-steam-headless: https://github.com/Steam-Headless/docker-steam-headless
- NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-container-toolkit

### 4.5 How Pro Platforms Render Demos

| Platform | Approach | Details |
|----------|----------|---------|
| **dinkem.gg** | Real CS2 clients on GPU servers | $0.47-$0.85/rendered-hour metered. $215/month per dedicated node. ~30 rendered-minutes per hour throughput. Most transparent about their infrastructure. |
| **Scope.gg** | Real CS2 clients | Server farm with GPU-equipped machines. Details not public. |
| **ClutchKings** | Real CS2 clients | Automated rendering pipeline. Details not public. |
| **DEMO-SLAP** | Real CS2 clients | Batch rendering service. Details not public. |
| **HLTV** | Unknown (likely real CS2 clients) | Render clips for match highlights. Pipeline not documented. |
| **Leetify** | Minimal rendering | Primarily uses 2D map visualization (no 3D rendering). Data-driven analysis rather than video output. |

**Key insight**: Every platform that produces 3D video from demos runs real CS2 game clients on GPU servers. There is no alternative. The economics are dominated by GPU-hour costs.

**Sources:**
- dinkem.gg pricing: https://dinkem.gg (pricing page)
- Scope.gg: https://scope.gg
- DEMO-SLAP: referenced in CS2 community discussions

### 4.6 Source 2 Filmmaker (S2FM)

- Works with CS2 via Workshop Tools DLC (free download from Steam)
- Can import demo files and position cameras
- Supports timeline-based editing, camera paths, lighting adjustments
- **GUI-only** -- no CLI, no batch rendering, no scripting API
- Not suitable for automated pipelines
- Useful for one-off high-quality renders but not for scale

**Sources:**
- Source 2 Filmmaker: https://developer.valvesoftware.com/wiki/Source_Filmmaker

### 4.7 Resolution, Framerate, and Storage Requirements

#### Frame Capture Storage

| Resolution | Format | Bytes/Frame | 30 FPS/min | 60 FPS/min |
|-----------|--------|-------------|------------|------------|
| 1280x720 | TGA (raw) | 2.76 MB | 4.97 GB | 9.95 GB |
| 1920x1080 | TGA (raw) | 6.22 MB | 11.20 GB | 22.39 GB |
| 2560x1440 | TGA (raw) | 11.06 MB | 19.90 GB | 39.81 GB |
| 1280x720 | PNG (compressed) | ~0.5 MB | ~0.9 GB | ~1.8 GB |
| 1920x1080 | PNG (compressed) | ~1.1 MB | ~2.0 GB | ~4.0 GB |

#### Encoded Video Storage

| Resolution | Codec | Bitrate | Per Minute | Per Match (~40 min) |
|-----------|-------|---------|------------|---------------------|
| 1280x720 | H.264 CRF 23 | ~2 Mbps | ~15 MB | ~600 MB |
| 1920x1080 | H.264 CRF 23 | ~5 Mbps | ~37 MB | ~1.5 GB |
| 1920x1080 | H.265 CRF 28 | ~3 Mbps | ~22 MB | ~900 MB |

#### At Scale (632 matches, 10 POVs each = 6,320 POV-matches)

| Resolution | Codec | Total Storage |
|-----------|-------|---------------|
| 720p | H.264 | ~3.8 TB |
| 1080p | H.264 | ~9.5 TB |
| 1080p | H.265 | ~5.7 TB |

#### Throughput

At ~30 rendered-minutes per real-hour (dinkem.gg benchmark):
- 632 matches * ~40 min = 25,280 match-minutes
- 10 POVs each = 252,800 POV-minutes
- At 30 min/hr throughput = **8,427 GPU-hours**
- With 4 GPU workers = **~88 days** continuous rendering
- Cost at dinkem.gg rates ($0.65/hr avg): **$5,478**

### 4.8 Practical Recommendation

**Start with dinkem.gg's API** for validation at low volume (negligible cost). Then build internal render workers on Linux with docker-steam-headless + cs2rec + FFmpeg once volume justifies infrastructure investment.

For the DashCrystals export tiers, consider:
- **Data-only tiers**: No rendering needed. Export .tard/.jsonl directly. Zero GPU cost.
- **2D visualization tier**: Use radar overlays (Section 5). CPU-only, trivially scalable.
- **3D rendered tier**: Premium product. Use GPU render farm. Price accordingly ($5-15/match rendered).

---

## 5. 2D Map Visualization

### 5.1 Radar/Overview Images

CS2 radar images are universally **1024x1024 RGBA PNGs**. They are stored inside the game's VPK archives as compiled textures (`ctex_c`), but community-extracted versions are readily available.

**Best source**: MurkyYT/cs2-map-icons (https://github.com/MurkyYT/cs2-map-icons) -- auto-extracts 68 maps from the CS2 game depot on every Valve patch, including the overview `.txt` metadata files with coordinate mapping values.

Multi-level maps have separate radar images:
- `de_nuke_radar.png` (upper) + `de_nuke_lower_radar.png`
- `de_vertigo_radar.png` (upper) + `de_vertigo_lower_radar.png`

**Sources:**
- cs2-map-icons: https://github.com/MurkyYT/cs2-map-icons
- CS Demo Manager maps: https://cs-demo-manager.com/docs/guides/maps

### 5.2 Overview Configuration Format

Each map has a `.txt` file in `resource/overviews/` defining the coordinate mapping:

```
"<mapname>"
{
    "material"      "overviews/<mapname>"
    "pos_x"         "-2476"     // World X of upper-left corner
    "pos_y"         "3239"      // World Y of upper-left corner
    "scale"         "4.4"       // World units per pixel
    "rotate"        "1"         // Whether to rotate 90 degrees
    "zoom"          "1.1"       // Zoom factor

    "verticalsections"
    {
        "default"
        {
            "AltitudeMin"   "-495"
            "AltitudeMax"   "10000"
        }
        "lower"
        {
            "AltitudeMin"   "-10000"
            "AltitudeMax"   "-495"
        }
    }
}
```

### 5.3 The Core Formula: World-to-Radar Pixel

The formula used by **every tool in the ecosystem**:

```python
def world_to_minimap(world_x, world_y, pos_x, pos_y, scale, image_size=1024):
    """Convert world coordinates to minimap pixel coordinates."""
    pixel_x = (world_x - pos_x) / scale
    pixel_y = (pos_y - world_y) / scale    # Y-axis inverted
    return (pixel_x, pixel_y)
```

**Y-axis inversion**: In the world coordinate system, +Y points north (up). In image/pixel space, +Y points downward. Hence `pos_y - world_y` (not `world_y - pos_y`).

The inverse (pixel to world):
```python
def minimap_to_world(pixel_x, pixel_y, pos_x, pos_y, scale):
    world_x = pixel_x * scale + pos_x
    world_y = pos_y - pixel_y * scale
    return (world_x, world_y)
```

**Sources:**
- Valve Developer Community -- Creating a working mini-map: https://developer.valvesoftware.com/wiki/Creating_a_working_mini-map_for_CS:GO
- Valthrun CS2 Issue #174: https://github.com/Valthrun/valthrun-cs2/issues/174

### 5.4 Map Overview Values (All Competitive Maps)

| Map | pos_x | pos_y | scale | Approx. Playable Area (HU) |
|-----|-------|-------|-------|---------------------------|
| **de_dust2** | -2476 | 3239 | 4.4 | 4510 x 4506 |
| **de_mirage** | -3230 | 1713 | 5.0 | 5120 x 5120 |
| **de_inferno** | -2087 | 3870 | 4.9 | 5019 x 5018 |
| **de_ancient** | -2953 | 2164 | 5.0 | 5120 x 5120 |
| **de_anubis** | -2796 | 3328 | 5.22 | 5345 x 5345 |
| **de_nuke** | -3453 | 2887 | 7.0 | 7168 x 7168 |
| **de_overpass** | -4831 | 1781 | 5.2 | 5325 x 5325 |
| **de_vertigo** | -3168 | 1762 | 4.0 | 4096 x 4096 |
| **de_train** | -2308 | 2078 | 4.082 | 4180 x 4180 |

World coordinate bounds (computed from overview data, assuming 1024x1024 image):
```
world_x_min = pos_x
world_x_max = pos_x + scale * 1024
world_y_min = pos_y - scale * 1024
world_y_max = pos_y
```

**Multi-level Z thresholds:**

| Map | Upper Level | Lower Level | Z Threshold |
|-----|-------------|-------------|-------------|
| de_nuke | Z > -495 | Z < -495 | -495 |
| de_vertigo | Z > 11700 | Z < 11700 | 11700 |
| de_train | Z > -50 | Z < -50 | -50 |

**Sources:**
- cs2-map-icons: https://github.com/MurkyYT/cs2-map-icons (raw data files)
- CS Demo Manager: https://cs-demo-manager.com/docs/guides/maps

### 5.5 CS2 vs CSGO Differences in Radar Format

The coordinate system and formula are **identical** between CS:GO and CS2. The only differences are:

1. **Asset packaging**: CS2 uses `ctex_c` compiled textures in VPK archives; CS:GO used loose DDS files
2. **Shifted coordinate values**: Maps that were visually reworked in CS2 have slightly different pos_x/pos_y/scale values (e.g., dust2 went from pos_x=-2400 in CS:GO to pos_x=-2476 in CS2)
3. **New maps**: CS2-only maps (de_anubis) have no CS:GO equivalent

### 5.6 Open Source Map Visualizers

| Tool | Language | Features | Source |
|------|----------|----------|--------|
| **awpy** | Python | Built-in matplotlib plotting. `awpy.plot.plot_round()`, `awpy.plot.plot_map()`. Automatic radar image loading, player position overlay, grenade trajectories. | https://github.com/pnxenopoulos/awpy |
| **CS Demo Manager** | Electron/TypeScript | Full-featured GUI. 2D map view with player positions, utility, kills. Exports to PNG/video. | https://cs-demo-manager.com |
| **Leetify** | Web (React) | SVG overlays on radar images. Player icons, movement trails, utility. | https://leetify.com |
| **HLTV** | Web | 2D map visualization for match analysis. Likely Canvas-based. | https://hltv.org |
| **demoparser2 examples** | Python | Raw coordinate output, user implements visualization. | https://github.com/LaihoE/demoparser |

#### awpy Example Code

```python
from awpy import Demo

# Parse demo
demo = Demo("match.dem")

# Get per-tick data
ticks = demo.ticks  # Polars DataFrame with X, Y, Z, pitch, yaw, etc.

# Plot a round
from awpy.plot import plot_round
fig = plot_round(demo, round_num=5, map_name="de_dust2")
fig.savefig("round5.png")
```

**Sources:**
- awpy: https://github.com/pnxenopoulos/awpy
- awpy PyPI: https://pypi.org/project/awpy/
- CS Demo Manager: https://cs-demo-manager.com

### 5.7 Rendering Player Positions on Radar

Standard approach used by all tools:

1. **Load radar image** (1024x1024 PNG)
2. **Load overview data** (pos_x, pos_y, scale from .txt file)
3. **For each tick/frame**:
   - Get player X, Y, Z from demo data
   - For multi-level maps, check Z against threshold to select upper/lower radar
   - Convert world coords to pixel coords: `px = (X - pos_x) / scale`, `py = (pos_y - Y) / scale`
   - Draw player icon at (px, py)
   - Optionally draw: yaw direction arrow, player name, health bar, team color
4. **For utility trajectories**: Connect sequential grenade positions with lines
5. **For kill events**: Draw X marker at kill location, line from killer to victim

### 5.8 Overlaying Additional Data

Beyond player positions, the 2D visualization can show:

- **View cones**: Using yaw + FOV to draw a triangular cone showing what each player can see
- **Movement trails**: Connect sequential positions with fading lines
- **Grenade trajectories**: Smoke/flash/HE/molotov paths
- **Smoke blooms**: Circles at smoke landing positions (radius ~144 HU in CS2)
- **Kill lines**: Lines connecting killer to victim at moment of kill
- **Bombsite zones**: Highlight A/B site regions
- **Heatmaps**: Aggregate position density across rounds/matches

---

## 6. BSP Map Geometry for Occlusion

### 6.1 Source 2 BSP Format vs Source 1

Source 2 fundamentally abandoned the traditional BSP (Binary Space Partitioning) format used since Quake/GoldSrc/Source 1:

**Source 1**: Maps compiled to `.bsp` files containing a BSP tree, with leaves, nodes, faces, brushes, and a precomputed PVS (Potentially Visible Set). The BSP tree partitioned space into convex leaves, and VVIS computed leaf-to-leaf visibility during compilation.

**Source 2**: Maps use `.vmap` (source) and `.vmap_c` (compiled) formats. BSP trees are replaced with **meshes** and an **octree** (`.vwnod_c`) for spatial partitioning. The compiled `.vmap_c` file references multiple sub-resources:

| Extension | Asset Type | Compiler |
|-----------|-----------|----------|
| `vwrld_c` | World asset | CompileWorld |
| `vwnod_c` | WorldNode (octree geometry) | CompileWorldNode |
| `vvis_c` | WorldVisibility | CompileMapVisibility |
| `vphys_c` | Physics collision mesh | CompilePhysics |
| `vents_c` | Entities | CompileEntities |
| `vtex_c` | Textures | CompileTextures |
| `vmdl_c` | Models | CompileModels |
| `vrman_c` | Resource manifest | - |

**You cannot parse Source 2 maps with Source 1 BSP parsers.** The file formats are completely incompatible.

**Sources:**
- Source 2 -- Valve Developer Community: https://developer.valvesoftware.com/wiki/Source_2
- BSP -- Valve Developer Community: https://developer.valvesoftware.com/wiki/BSP
- VMAP -- Valve Developer Community: https://developer.valvesoftware.com/wiki/VMAP

### 6.2 ValveResourceFormat (VRF) -- The Primary Source 2 Parser

**ValveResourceFormat** is a C# library and GUI tool built through reverse engineering:

- Supports parsing 50+ Source 2 resource types including `vmap`, `vwrld`, `vwnod`, `vvis`, `vphys`, `vmdl`, `vmesh`
- Can export maps to **glTF/GLB** format (geometry, textures, props/models, materials)
- Can decompile maps back to `.vmap` for Hammer (imperfect)
- Over 6,164 commits, mature project

**Important limitation**: VRF is C#/.NET only. There is no comprehensive Python parser for Source 2 map formats.

**Sources:**
- ValveResourceFormat GitHub: https://github.com/ValveResourceFormat/ValveResourceFormat
- Source 2 Viewer: https://s2v.app/
- VRF Resource Types Wiki: https://github.com/ValveResourceFormat/ValveResourceFormat/wiki/Resource-Types

### 6.3 Python-Accessible Tools for CS2 Visibility

#### awpy (Primary Recommendation)

`pip install awpy` -- Python 3.11+ library for CS2 data analysis:

- Parses CS2 demo files via demoparser2 (Rust backend)
- Includes `VisibilityChecker` class for geometric visibility checks using `.tri` files
- Includes nav mesh parser (`awpy.nav.Nav`) supporting CS2 nav format
- Outputs data as Polars DataFrames

```python
from awpy.visibility import VisibilityChecker

vc = VisibilityChecker("de_dust2")
is_visible = vc.is_visible(
    (x1, y1, z1),  # observer position
    (x2, y2, z2)   # target position
)
```

**Sources:**
- awpy GitHub: https://github.com/pnxenopoulos/awpy
- awpy PyPI: https://pypi.org/project/awpy/
- awpy Visibility docs: https://awpy.readthedocs.io/en/latest/examples/visibility.html

#### cs2-nav (Rust + Python Bindings)

Rust library with PyO3 Python bindings:

- `CollisionChecker` struct with BVH-based visibility
- Navigation and visibility utilities designed for awpy
- Written by JanEricNitschke

**Sources:**
- cs2-nav: https://github.com/JanEricNitschke/cs2_meeting_points
- cs2-nav docs.rs: https://docs.rs/cs2-nav/latest/cs2_nav/collisions/struct.CollisionChecker.html

#### VisCheckCS2 (C++ with Python Bindings)

- Parses `.vphys` files extracted via Source 2 Viewer
- Converts to optimized `.opt` binary format
- Python API: `vischeck.VisCheck("de_mirage.opt").is_visible((x1,y1,z1), (x2,y2,z2))`

**Sources:**
- VisCheckCS2 GitHub: https://github.com/Read1dno/VisCheckCS2

### 6.4 Triangle Mesh Extraction Pipeline

The collision geometry extraction process:

#### Step 1: Extract Collision Geometry

Source: `world_physics.vmdl_c` inside map VPK files. Contains `.vphys` collision data with `m_collisionAttributes`, `m_meshes`, and `m_hulls` blocks. Extract triangle vertices (Vector3 with float x, y, z) and triangle indices.

Tools:
- **cs2-phys-extractor** (.NET, uses VRF library): https://github.com/itzlaith/cs2-phys-extractor
- **Source 2 Viewer** (GUI, manual export)
- **ExternalAutoWallCS2**: https://github.com/Read1dno/ExternalAutoWallCS2

#### Step 2: Convert to Triangle Format

- `.vphys` text format is huge and redundant (e.g., de_ancient: 370MB)
- `.tri` binary format stores raw triangles compactly:
  - de_ancient: 22.1MB (94% reduction)
  - de_dust2: 175MB -> 13.3MB (92.4% reduction)
- Each triangle = 3 x Vector3 (3 floats each) = 36 bytes per triangle
- awpy's `.tri` files across all maps total ~20MB compressed

#### Step 3: Build Acceleration Structure

Two approaches in practice:

**BVH (Bounding Volume Hierarchy)** -- Used by awpy/cs2-nav and VisCheckCS2:
- Organizes triangles into nested AABB (Axis-Aligned Bounding Box) groups
- Recursive subdivision, splitting along longest axis
- Leaf nodes contain <= 4 triangles
- Construction time: 744ms (small maps) to 9.62s (large maps)

**KD-tree** -- Used by cs2-map-parser (AtomicBool):
- Similar spatial subdivision with different splitting strategy

#### Step 4: Ray-Triangle Intersection

All implementations use the **Moller-Trumbore algorithm**:
- Fast method for ray-triangle intersection without precomputing the plane equation
- Given ray origin, direction, and triangle vertices, returns intersection point (if any)
- Combined with BVH traversal: first check ray against AABB nodes, only test triangles in hit leaf nodes

**Sources:**
- cs2-phys-extractor: https://github.com/itzlaith/cs2-phys-extractor
- cs2-map-parser: https://github.com/AtomicBool/cs2-map-parser
- Moller-Trumbore Algorithm: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

### 6.5 Visibility Check Performance

| Tool | Language | Visible Check | Blocked Check | Notes |
|------|----------|--------------|---------------|-------|
| **awpy** | Python/Rust | ~177 us | ~65 us | Blocked is ~3x faster (early termination) |
| **VisCheckCS2** | C++/Python | ~1 ms | ~1 ms | CPU-dependent |
| **cs2-map-parser** | C++ | Sub-ms | Sub-ms | After KD-tree optimization |

At awpy's rate (~177 us/check), checking all 45 player pairs (10 players, C(10,2)) per tick at 64 Hz:
- 45 pairs * 64 ticks/sec = 2,880 checks/sec
- At 177 us/check = ~0.51 seconds of compute per second of gameplay
- **Real-time feasible** for post-processing

### 6.6 Navigation Mesh for Simplified Spatial Analysis

The nav mesh provides a simpler alternative to full geometry for certain analyses:

**What nav mesh provides:**
- Polygonal areas covering walkable surfaces
- Each area has corners (3D vertices), connections to adjacent areas, ladder connections
- Attributes/flags (crouch, jump, no-hostage)
- Centroid positions and size computations
- Pathfinding between areas

**Nav mesh format (CS2):**
- Magic number: `0xFEEDFACE`
- Major version 35-36 (vs. ~15 in CS:GO)
- Binary format with areas, connections, ladders, and (in v36) KV3 data blocks
- Place names (callouts) are no longer in nav file; defined as `env_cs_place` entities in the map

**Nav data structures (from awpy's Nav parser):**
```
Nav:
  version: int
  sub_version: int
  is_analyzed: bool
  areas: dict[int, NavArea]

NavArea:
  area_id: int
  hull_index: int
  dynamic_attribute_flags: int
  corners: list[Vector3]        # 3D polygon vertices
  connections: list[NavMeshConnection]
  ladders_above: list
  ladders_below: list
  centroid: Vector3             # computed geometric center
  size: float                  # computed 2D polygon area
```

**Parsers:**

| Parser | Language | CS2 Support | Notes |
|--------|----------|-------------|-------|
| **awpy** (`awpy.nav.Nav`) | Python | Yes (v31+) | `from_path()`, `to_json()`, `find_path()` |
| **cs2-nav** | Rust + Python | Yes | `CollisionChecker` integrates nav + visibility |
| **ValveResourceFormat** | C# | Yes (v36) | Full parser with KV3 support |
| **gonav** | Go | No (up to v15) | CS:GO only |

**Sources:**
- awpy Nav module docs: https://awpy.readthedocs.io/en/latest/modules/nav.html
- NAV file format: https://developer.valvesoftware.com/wiki/NAV_(file_format)
- ValveResourceFormat NavMeshFile API: https://s2v.app/ValveResourceFormat/api/ValveResourceFormat.NavMesh.NavMeshFile.html
- gonav: https://github.com/mrazza/gonav

### 6.7 Anti-Cheat Visibility Checks

Two distinct contexts:

#### Server-Side (CS2FOW)

**CS2FOW** (by Karola3vax) is the primary reference implementation:
- Server-side Metamod:Source plugin for community servers
- Uses precomputed visibility sets from static baked map geometry
- If an enemy is fully hidden behind solid map geometry, the server stops sending that player's data to the client entirely
- Uses movement prediction + ping compensation to reveal enemies slightly before exact visibility
- **Limitations**: Dynamic occluders (doors, breakables, smokes, props) are out of scope

Key insight from HN discussion: "CS:GO had visibility checking essentially for free [via BSP PVS], but Source 2 is more of a mesh based engine and Valve never ported this feature to CS2"

**Sources:**
- CS2FOW GitHub: https://github.com/karola3vax/CS2FOW
- CS2FOW HN Discussion: https://news.ycombinator.com/item?id=48805965

#### External Visibility Checks (for Data Export)

Two approaches used by external tools:
1. **Network variable approach**: Reading `m_bSpottedByMask` from demo data -- tells which players have spotted each other
2. **Map geometry approach**: Extract collision mesh from VPK files, build BVH, ray-cast between player positions -- completely external, no memory reading needed

### 6.8 PVS in Source 2

**Source 1 PVS** (for reference):
- After BSP tree construction, PVS was calculated for each BSP leaf
- PVS = the set of all other leaves visible from anywhere within a given leaf
- Stored as compressed bitstrings in the BSP file

**Source 2 PVS**:
- Replaced BSP leaf system with an **octree** (`.vwnod_c`)
- `.vvis_c` file compiled by `CompileMapVisibility` -- Source 2 equivalent of VVIS
- Exact format only partially reverse-engineered by VRF
- Uses `info_visibility_box` entities for manual visibility hints
- **PVS is insufficient for anti-wallhack** because "PVS culling is not even remotely comparable to occlusion culling"

**You cannot easily extract PVS from CS2 maps** -- the vvis_c format is only partially reverse-engineered.

**Sources:**
- PVS -- Valve Developer Community: https://developer.valvesoftware.com/wiki/PVS
- Potentially Visible Set -- Wikipedia: https://en.wikipedia.org/wiki/Potentially_visible_set
- info_visibility_box -- Source2 Wiki: http://www.source2.wiki/Entities/info_visibility_box

### 6.9 Recommended Occlusion Pipeline for DashCrystals

1. **Extract triangles**: Use `cs2-phys-extractor` or awpy's bundled `.tri` files
2. **Parse nav mesh**: Use `awpy.nav.Nav.from_path()` to get walkable areas
3. **Check visibility**: Use `awpy.visibility.VisibilityChecker` with `.tri` files
4. **Precompute visibility matrix**: For each pair of nav areas, check if centroids are mutually visible
5. **Per-tick visibility**: For each tick, check all 45 player pairs (10 players) using BVH ray-casting
6. **Export**: Attach visibility boolean per player-pair per tick to the dataset

**Limitations of all approaches:**
- Only static geometry is considered (no doors, breakable walls, smokes, Molotovs)
- Player model height/crouch not accounted for unless Z coordinate is offset (~64 HU standing eye height, ~46 HU crouching)
- Props that block vision but are not in the physics collision mesh may be missed

### 6.10 Complete Tool Inventory

#### Production-Ready Tools

| Tool | Language | Approach | Performance | Python API |
|------|----------|----------|-------------|------------|
| **awpy** | Python/Rust | .tri files + BVH + Moller-Trumbore | 65-177 us/check | Yes (native) |
| **cs2-nav** | Rust | BVH CollisionChecker, awpy-compatible .tri | Similar to awpy | Yes (PyO3) |
| **VisCheckCS2** | C++/Python | .vphys -> .opt + BVH + Moller-Trumbore | ~1ms/ray | Yes (pybind11) |

#### Extraction/Conversion Tools

| Tool | Language | Function |
|------|----------|----------|
| **cs2-phys-extractor** | C#/.NET | Extract world_physics.vmdl_c -> .tri/.vphys from VPK |
| **cs2-map-parser** | C++ | Convert .vphys -> .tri with KD-tree vis checks |
| **ValveResourceFormat** | C# | Full Source 2 asset parser, map export to glTF |
| **ExternalAutoWallCS2** | C++/Python | Extract collision + materials, bullet penetration calc |

#### Server-Side Plugins

| Tool | Platform | Function |
|------|----------|----------|
| **CS2FOW** | Metamod:Source | Server-side anti-wallhack using occlusion culling |

**Sources:**
- awpy Visibility: https://awpy.readthedocs.io/en/latest/examples/visibility.html
- awpy GitHub: https://github.com/pnxenopoulos/awpy
- VisCheckCS2: https://github.com/Read1dno/VisCheckCS2
- ExternalAutoWallCS2: https://github.com/Read1dno/ExternalAutoWallCS2
- cs2-phys-extractor: https://github.com/itzlaith/cs2-phys-extractor
- cs2-map-parser: https://github.com/AtomicBool/cs2-map-parser
- CS2FOW: https://github.com/karola3vax/CS2FOW

---

## Appendix A: DashCrystals Export Tier Architecture

Based on all research findings, the recommended export tiers:

### Tier 0: Raw Binary (.tard)
- Direct .tard files (4.6 bytes/row, 44x compression vs JSON)
- Includes: position XYZ, yaw, pitch, dyaw, dpitch, mouse_dx, mouse_dy, speed, weapon, fire, scope, health, team, rank
- 128 Hz per player, all 10 players per match
- Cheapest to produce, smallest storage

### Tier 1: Structured Data (.jsonl / .parquet)
- Decoded from .tard via tard_decode.py
- Additional computed fields: velocity magnitude, acceleration, view direction vector
- Per-round and per-match metadata
- Parquet format for efficient columnar queries

### Tier 2: Enriched Spatial Data
- Everything in Tier 1, plus:
- 2D radar coordinates (pre-computed pixel positions per map)
- Screen-space projection (where each enemy appears on each player's screen)
- View cone intersection (who could see whom based on FOV)
- Nav area ID per player per tick

### Tier 3: Visibility-Enriched Data
- Everything in Tier 2, plus:
- Geometric visibility (BVH ray-cast per player pair per tick)
- Time-to-visible (how many ms before enemy becomes visible)
- Pre-aim detection (view angle approaching enemy before visibility)
- Nav mesh path distances between all player pairs

### Tier 4: Rendered Video
- Everything in Tier 2, plus:
- 1080p video per player POV (H.265, ~900 MB/match)
- Synchronized frame-action pairs (screenshot + input at each tick)
- 2D radar animation overlay
- Premium product, GPU-intensive

### Tier 5: Computer Use Training Format
- Formatted for Standard Intelligence / frontier lab consumption
- (screenshot, action) trajectories with normalized coordinates
- Action types: move, aim, fire, scope, weapon_switch, reload
- Structured metadata per trajectory (task: "win round", "plant bomb", etc.)
- Accessibility: game state as structured text alongside screenshots

---

## Appendix B: Multi-Game Expansion Roadmap

Based on Section 2 viability analysis:

### Phase 1 (Now): CS2
- Already operational via TARDIGRADE
- 632 matches, 3,160 player-hours
- Focus on enrichment tiers and export formats

### Phase 2 (Next): Rocket League
- `pip install carball` -- ready today
- All actors every frame, no culling
- Different data profile (continuous physics, car control)
- Complements CS2's discrete-action data

### Phase 3: Dota 2
- `gem-dota` or Clarity parser
- All entities including fogged units
- RTS-style data (click-to-move, ability targeting, camera control)
- Massive existing replay archive (OpenDota, Dotabuff)

### Phase 4 (Experimental): Minecraft + Terraria
- Minecraft: requires ServerReplay mod + custom parser
- Terraria: requires live packet capture + custom protocol parser
- Both provide unique data: 3D building, resource management, exploration

---

## Appendix C: Key External Resources

### Parsers and Libraries

| Resource | URL | Language |
|----------|-----|----------|
| demoparser2 | https://github.com/LaihoE/demoparser | Rust/Python |
| awpy | https://github.com/pnxenopoulos/awpy | Python |
| carball | https://pypi.org/project/carball/ | Python |
| Clarity | https://github.com/skadistats/clarity | Java |
| ValveResourceFormat | https://github.com/ValveResourceFormat/ValveResourceFormat | C# |
| FortniteReplayDecompressor | https://github.com/SL-x-TnT/FortniteReplayDecompressor | C# |
| Boxcars | https://github.com/nickbabcock/boxcars | Rust |
| gem-dota | https://github.com/whanyu1212/gem-dota | Python |
| cs2-phys-extractor | https://github.com/itzlaith/cs2-phys-extractor | C# |
| VisCheckCS2 | https://github.com/Read1dno/VisCheckCS2 | C++/Python |
| CS2FOW | https://github.com/karola3vax/CS2FOW | C++ |
| HLAE | https://github.com/advancedfx/advancedfx | C++ |

### Map Data

| Resource | URL |
|----------|-----|
| cs2-map-icons (radar images + overview data) | https://github.com/MurkyYT/cs2-map-icons |
| CS Demo Manager | https://cs-demo-manager.com |

### Reference Documentation

| Resource | URL |
|----------|-----|
| Valve Developer Community -- Coordinates | https://developer.valvesoftware.com/wiki/Coordinates |
| Valve Developer Community -- QAngle | https://developer.valvesoftware.com/wiki/QAngle |
| Valve Developer Community -- DEM Format | https://developer.valvesoftware.com/wiki/DEM_(file_format) |
| Valve Developer Community -- NAV Format | https://developer.valvesoftware.com/wiki/NAV_(file_format) |
| Valve Developer Community -- BSP | https://developer.valvesoftware.com/wiki/BSP |
| Valve Developer Community -- PVS | https://developer.valvesoftware.com/wiki/PVS |
| Source SDK -- mathlib_base.cpp | https://github.com/pmrowla/hl2sdk-csgo/blob/master/mathlib/mathlib_base.cpp |

### Rendering Services

| Service | URL | Pricing |
|---------|-----|---------|
| dinkem.gg | https://dinkem.gg | $0.47-$0.85/rendered-hour |
| docker-steam-headless | https://github.com/Steam-Headless/docker-steam-headless | Self-hosted |

### Market Intelligence

| Resource | URL |
|----------|-----|
| Standard Intelligence | https://si.inc |
| Mind2Web | https://osu-nlp-group.github.io/Mind2Web/ |
| WebArena | https://webarena.dev/ |
| OSWorld | https://os-world.github.io/ |
| SWE-bench | https://www.swebench.com/ |

---

*Document compiled from 6 parallel research threads, 80+ web searches, and 50+ source URLs. All claims sourced to the best available references as of August 2026.*
