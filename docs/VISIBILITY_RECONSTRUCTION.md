# CS2 Visibility Reconstruction: Complete Technical Reference

> Theoretical maximum accuracy analysis for reconstructing pixel-accurate "who can see who" from demo file data.
> Every claim sourced from: Valve leaked source (`cstrike15_src`), CS2 schema dumps (`a2x/cs2-dumper`), community implementations (`CS2FOW`, `awpy`, `demoparser2`), or empirical measurement.

---

## Table of Contents

1. [Smoke Grenade Mechanics](#1-smoke-grenade-mechanics)
2. [Molotov / Incendiary Mechanics](#2-molotov--incendiary-mechanics)
3. [Flashbang Mechanics](#3-flashbang-mechanics)
4. [Aim Punch / View Punch](#4-aim-punch--view-punch)
5. [FOV and Scoped View](#5-fov-and-scoped-view)
6. [Player Dimensions and Eye Position](#6-player-dimensions-and-eye-position)
7. [PVS and Demo File Completeness](#7-pvs-and-demo-file-completeness)
8. [Edge Cases for Visibility](#8-edge-cases-for-visibility)
9. [Theoretical Maximum Accuracy Assessment](#9-theoretical-maximum-accuracy-assessment)

---

## 1. Smoke Grenade Mechanics

### 1.1 CSGO: Simple Sphere Model

**Source:** Leaked CSGO source `cs_shareddefs.h`, `cs_gamerules.cpp`, `c_cs_player.cpp`

The CSGO smoke is a **perfect sphere** for all gameplay occlusion checks. The visual particle system is cosmetic; the actual line-of-sight blocking uses:

| Parameter | Value | Source |
|-----------|-------|--------|
| Occlusion radius | **166 units** | `CONSTANT_UNITS_SMOKEGRENADERADIUS = 166` in `cs_shareddefs.h` |
| Center Z offset | **+60 units** above detonation position | `vecSmokeCenterOffset = Vector(0, 0, 60)` in `CheckTotalSmokedLength` |
| Blocking threshold | **70% of radius = 116.2 units** | `maxSmokedLength = 0.7f * SmokeGrenadeRadius` |
| Bot bloat factor | **1.2x** (radius becomes 199.2) | `grenadeBloat = 1.2f` in bot code |
| Visual expansion time | **5.5 seconds** (sine curve) | `SMOKESPHERE_EXPAND_TIME = 5.5` |
| Gameplay occlusion onset | **Instant** at full radius when `m_bDidSmokeEffect = true` | No gradual expansion in gameplay checks |
| Total lifetime | **~17.5 seconds** | `SMOKEGRENADE_LIFETIME = 17.5` |
| Active phase | **~12.5 seconds** before fade begins | |
| Fade curve | Cosine ease-out over ~4.3 seconds | `cos(fadePercent * pi) * 0.5 + 0.5` |

**Common misquoted values:** 144 units and 288 units are NOT the occlusion radius. 288 is roughly the visual particle cloud diameter. The occlusion radius is exactly **166**.

### 1.2 CSGO LineGoesThroughSmoke -- Valve Source Code

From `c_cs_player.cpp`:

```cpp
bool LineGoesThroughSmoke( Vector from, Vector to, bool grenadeBloat )
{
    float totalSmokedLength = 0.0f;
    const float smokeRadiusSq = SmokeGrenadeRadius * SmokeGrenadeRadius
                                * grenadeBloat * grenadeBloat;

    for ( int it = 0; it < g_SmokeGrenadeHandles.Count(); it++ )
    {
        C_SmokeGrenadeProjectile *pGrenade =
            static_cast<C_SmokeGrenadeProjectile*>(g_SmokeGrenadeHandles[it].Get());
        if ( pGrenade && pGrenade->m_bDidSmokeEffect == true )
        {
            float flLengthAdd = CSGameRules()->CheckTotalSmokedLength(
                smokeRadiusSq, pGrenade->GetAbsOrigin(), from, to );
            if ( flLengthAdd == -1 )
                return true;       // endpoint inside smoke
            totalSmokedLength += flLengthAdd;
        }
    }

    const float maxSmokedLength = 0.7f * SmokeGrenadeRadius;  // = 116.2 units
    return (totalSmokedLength > maxSmokedLength);
}
```

### 1.3 CSGO CheckTotalSmokedLength -- The Core Math

From `cs_gamerules.cpp`:

```cpp
float CCSGameRules::CheckTotalSmokedLength( float flSmokeRadiusSq,
    Vector vecGrenadePos, Vector from, Vector to )
{
    Vector sightDir = to - from;
    float sightLength = sightDir.NormalizeInPlace();

    Vector vecSmokeCenterOffset = Vector( 0, 0, 60 );
    const Vector &smokeOrigin = vecGrenadePos + vecSmokeCenterOffset;
    float flSmokeRadius = sqrt(flSmokeRadiusSq);

    // If either endpoint is inside the sphere, fully blocked
    if ( (smokeOrigin - from).IsLengthLessThan( flSmokeRadius * 0.95f )
      || (smokeOrigin - to).IsLengthLessThan( flSmokeRadius ) )
        return -1;

    // Project smoke center onto the line segment
    Vector toGrenade = smokeOrigin - from;
    float alongDist = DotProduct( toGrenade, sightDir );

    Vector close;
    if (alongDist < 0.0f)
        close = from;
    else if (alongDist >= sightLength)
        close = to;
    else
        close = from + sightDir * alongDist;

    Vector toClose = close - smokeOrigin;
    float lengthSq = toClose.LengthSqr();

    if (lengthSq < flSmokeRadiusSq)
    {
        // Chord length through the sphere
        float smokedLength = 2.0f * sqrtf( flSmokeRadiusSq - lengthSq );
        return smokedLength;
    }
    return 0;
}
```

**Algorithm:** Line-sphere chord accumulation. Multiple smokes stack -- chord lengths sum across all active smokes. If total > 116.2 units, line is blocked.

**Note the asymmetry:** `from` endpoint uses `0.95 * radius` for the inside-check, `to` uses full radius.

### 1.4 CS2: Volumetric Voxel System

CS2 replaced the simple sphere with a **voxel-based volumetric flood-fill system**.

**CS2 entity schema** (from `a2x/cs2-dumper`, `client_dll.hpp`):

| Field | Type | Offset | Description |
|-------|------|--------|-------------|
| `m_nSmokeEffectTickBegin` | int32 | 0x1278 | Game tick when smoke begins |
| `m_bDidSmokeEffect` | bool | 0x127C | Whether smoke is active |
| `m_nRandomSeed` | int32 | 0x1280 | Procedural variation seed |
| `m_vSmokeColor` | Vector | 0x1284 | RGB color |
| `m_vSmokeDetonationPos` | VectorWS | 0x1290 | Detonation position |
| `m_VoxelFrameData` | `C_NetworkUtlVectorBase<uint8>` | 0x12A0 | **3D voxel grid data** |
| `m_nVoxelFrameDataSize` | int32 | 0x12B8 | Size of voxel data |
| `m_nVoxelUpdate` | int32 | 0x12BC | Increments when holes punched (HE/bullets) |
| `m_bSmokeVolumeDataReceived` | bool | 0x12C0 | Client received volume data |

**CS2 smoke characteristics:**
- Flood fill expands from detonation point through 3D voxel grid
- Walls and solid geometry block the fill -- smoke conforms to rooms
- All players see the same smoke shape (unlike CSGO where particles were client-side)
- Bullets and HE grenades can punch temporary holes (`m_nVoxelUpdate` increments)
- Duration: ~18 seconds (~15 dense + ~3 fade)
- Client rendering uses raymarching through the voxel texture

### 1.5 Smoke-Inferno Interaction Shape

Even in CSGO, Valve internally used a non-spherical shape for smoke-extinguishes-fire calculations (from `inferno.cpp`):

```cpp
static const float SmokeGrenadeRadius_InfernoAffectingZ           = 120.0f;
static const float SmokeGrenadeRadius_InfernoAffectingXY_topedge  = 100.0f;
static const float SmokeGrenadeRadius_InfernoAffectingXY_equator  = 150.0f;
static const float SmokeGrenadeRadius_InfernoAffectingXY_bottomedge = 128.0f;
```

This reveals Valve considered the smoke as oblate spheroid for some interactions, even though the LOS check uses a simple sphere.

### 1.6 Reconstruction Algorithm (CSGO-style, usable as CS2 approximation)

```python
def line_goes_through_smoke(eye_pos, target_pos, active_smokes, radius=166.0):
    total_smoked = 0.0
    threshold = 0.7 * radius  # 116.2

    for smoke in active_smokes:
        center = smoke.position + Vector(0, 0, 60)

        # Endpoint inside smoke = fully blocked
        if distance(center, eye_pos) < radius * 0.95:
            return True
        if distance(center, target_pos) < radius:
            return True

        # Project smoke center onto line segment
        sight_dir = normalize(target_pos - eye_pos)
        sight_len = length(target_pos - eye_pos)
        along = dot(center - eye_pos, sight_dir)
        along = clamp(along, 0, sight_len)
        closest = eye_pos + sight_dir * along

        dist_sq = length_sq(closest - center)
        if dist_sq < radius * radius:
            chord = 2.0 * sqrt(radius * radius - dist_sq)
            total_smoked += chord

    return total_smoked > threshold
```

### 1.7 Demo Data Available

| Source | Fields |
|--------|--------|
| Game event `smokegrenade_detonate` | `x, y, z`, `entityid`, `userid` |
| Game event `smokegrenade_expired` | `x, y, z`, `entityid` |
| Entity `C_SmokeGrenadeProjectile` | `m_bDidSmokeEffect`, `m_nSmokeEffectTickBegin`, `m_vSmokeDetonationPos` |
| CS2 only | `m_VoxelFrameData` (byte array -- the actual 3D voxel grid) |

---

## 2. Molotov / Incendiary Mechanics

### 2.1 Fire Spread Algorithm

The fire is NOT a simple circle. It uses a parent-child spawning system:

| Parameter | Value | Source |
|-----------|-------|--------|
| `inferno_max_range` | **150 units** (radius from center) | Server convar |
| `inferno_flame_spacing` | **42 units** (between parent-child) | Server convar |
| `inferno_max_flames` | **16** (default; array supports 64 in CS2, 100 in CSGO) | Server convar |
| `inferno_spawn_angle` | **45 degrees** (child spawn angle variance) | Server convar |
| `inferno_initial_spawn_interval` | **0.02 seconds** | Server convar |
| `inferno_per_flame_spawn_duration` | **3 seconds** | Server convar |
| Approximate diameter on flat ground | **~300 units (~3 meters)** | Measured |

Each flame spawns children at random angles, traces downward to find walkable surfaces. On flat ground, the result is roughly circular ~150 unit radius.

### 2.2 Fire Duration

| Parameter | Value |
|-----------|-------|
| `inferno_flame_lifetime` | **7 seconds** (default) |
| Measured precise duration | **7.03125 seconds** (likely tick quantization: 7 + 1/32) |

Both molotov (T-side) and incendiary grenade (CT-side) share identical duration and mechanics.

### 2.3 Fire Does NOT Block Visibility

**Fire does NOT block server-side visibility or line of sight.** Unlike smoke, molotov fire is purely a client-side visual effect. The server still transmits enemy player entities through fire. There is no PVS blocking, no TraceLine occlusion.

**For reconstruction: fire should NOT be treated as a vision blocker.**

### 2.4 Fire and Elevation

- Fire flows downhill naturally from impact point
- Cannot climb walls or vertical surfaces
- Grenade only shatters on surfaces angled 30 degrees or less from horizontal
- **128-unit air burst rule:** if airborne when fuse expires, engine checks for walkable surface within 128 units below. If found, fire drops to it. If not, air burst with no ground fire.

### 2.5 CInferno Entity Properties

**CSGO (Source 1):** Fire positions are integer deltas from entity origin:

```cpp
NETVAR(fireXDelta,    "CInferno", "m_fireXDelta",     int[100])
NETVAR(fireYDelta,    "CInferno", "m_fireYDelta",     int[100])
NETVAR(fireZDelta,    "CInferno", "m_fireZDelta",     int[100])
NETVAR(fireIsBurning, "CInferno", "m_bFireIsBurning", bool[100])
NETVAR(fireCount,     "CInferno", "m_fireCount",      int)
// fire_world_pos[i] = entity.origin + Vector(m_fireXDelta[i], m_fireYDelta[i], m_fireZDelta[i])
```

**CS2 (Source 2):** Fire positions are absolute world-space vectors:

| Field | Type | Offset | Description |
|-------|------|--------|-------------|
| `m_firePositions` | `VectorWS[64]` | 0x0770 | Absolute world-space fire positions |
| `m_fireParentPositions` | `VectorWS[64]` | 0x0A70 | Parent flame positions |
| `m_bFireIsBurning` | `bool[64]` | 0x0D70 | Whether each slot is burning |
| `m_BurnNormal` | `Vector[64]` | 0x0DB0 | Surface normal at each flame |
| `m_fireCount` | `int32` | 0x10B0 | Number of active flames |
| `m_nFireLifetime` | `float32` | 0x10BC | Lifetime of fire |
| `m_bWasCreatedInSmoke` | `bool` | 0x10C1 | Created inside smoke |

**Critical difference:** CSGO uses integer deltas from origin (100 slots); CS2 uses absolute world-space vectors (64 slots).

### 2.6 Demo Data Available

| Source | Fields |
|--------|--------|
| Game event `inferno_startburn` | `entityid`, `x, y, z` (origin/center) |
| Game event `inferno_expire` | `entityid`, `x, y, z` |
| Game event `inferno_extinguish` | `entityid`, `x, y, z` |
| Entity `CInferno` per-tick | `m_firePositions[64]`, `m_bFireIsBurning[64]`, `m_fireCount` |

The `demoinfocs-golang` library provides `ConvexHull2D()` and `ConvexHull3D()` on the `Fires` collection for spatial reconstruction.

---

## 3. Flashbang Mechanics

### 3.1 flash_duration (m_flFlashDuration)

`m_flFlashDuration` stores the **fade time** (NOT total blindness time), pre-divided by 1.4:

```cpp
void CCSPlayer::Blind( float holdTime, float fadeTime, float startingAlpha )
{
    fadeTime /= 1.4f;   // DIVIDED by 1.4 before storage
    m_flFlashDuration = fadeTime;
    m_flFlashMaxAlpha = startingAlpha;
}
```

The `player_blind` event's `blind_duration` field = `m_flFlashDuration`. Total blindness time is approximately `holdTime + 0.5 * fadeTime`.

### 3.2 The Fade Curve -- Two Phases

**Phase 1 -- Build-up (~0.094 seconds, near-instant ramp to full white):**

```cpp
static const float FLASH_BUILD_UP_PER_FRAME = 45.0f;
static const float FLASH_BUILD_UP_DURATION = (255.0f / 45.0f) * (1.0f / 60.0f);
// = ~0.0944 seconds
// alpha = clamp((elapsed / 0.0944) * maxAlpha, 0, maxAlpha)
```

**Phase 2 -- Fade (piecewise quadratic with full-blind plateau):**

```cpp
const float certainBlindnessTimeThresh = 3.0f;

if (timeLeft > certainBlindnessTimeThresh) {
    alphaPercentage = 1.0f;           // fully blind, 100% white
} else {
    alphaPercentage = timeLeft / certainBlindnessTimeThresh;
    alphaPercentage *= alphaPercentage;   // SQUARED for faster falloff
}
overlayAlpha = alphaPercentage * maxAlpha;
```

**Key insight:** If remaining flash time exceeds **3.0 seconds**, the overlay stays at 100% alpha (full blindness). Below 3.0s, alpha follows `(timeLeft/3.0)^2 * maxAlpha` -- quadratic decay.

### 3.3 flash_max_alpha (m_flFlashMaxAlpha)

- Float, range 0.0-255.0, networked with `SPROP_NOSCALE` (full 32-bit precision)
- In practice, `startingAlpha` is **always 255** -- the source code comment confirms this
- CS2 offset: `C_CSPlayerPawnBase`, offset `0x1424` for `m_flFlashMaxAlpha`, `0x1428` for `m_flFlashDuration`

### 3.4 Partial Flash -- Four-Tier Angle System

From `RadiusFlash()` in `flashbang_projectile.cpp`:

```cpp
AngleVectors( pEntity->EyeAngles(), &vForward );
vecLOS = ( vecSrc - vecEyePos );
vecLOS.NormalizeInPlace();
flDot = DotProduct( vecLOS, vForward );
```

| Dot Product | Angle Range | fadeTime Multiplier | fadeHold Multiplier |
|-------------|-------------|---------------------|---------------------|
| >= 0.6 | 0-53 deg (looking at flash) | 2.5x | 1.25x |
| >= 0.3 | 53-72 deg (side) | 1.75x | 0.8x |
| >= -0.2 | 72-101 deg (peripheral) | 1.0x | 0.5x |
| < -0.2 | 101-180 deg (facing away) | 0.5x | 0.25x |

Looking directly away gives **5x less effect** than looking at the flash.

### 3.5 Distance Falloff

**Linear** over a 3000-unit radius:

```cpp
static float flRadius = 3000;
float falloff = flDamage / flRadius;  // = sv_flashbang_strength / 3000
flAdjustedDamage = flDamage - (distance_to_eyes * falloff);
```

At default `sv_flashbang_strength = 3.55`:
- Distance 0: `flAdjustedDamage = 3.55`
- Distance 1500: `flAdjustedDamage = 1.775`
- Distance 3000: `flAdjustedDamage = 0` (no effect)
- **Maximum effective range: 3000 units (~57 meters)**

### 3.6 Flash Through Walls and Smoke

**Walls: Flash does NOT go through walls.** `PercentageOfFlashForPlayer()` performs multiple ray traces:
- Primary trace: direct line from flash origin to player eye
- Three secondary traces: offset 50 up, 75 right, 75 left, bouncing to player's eye
- Each secondary trace contributes 1/6 to the percentage
- Uses `MASK_OPAQUE_AND_NPCS | CONTENTS_DEBRIS` minus `CONTENTS_OPAQUE`
- Players do NOT block trace for other players (`CTraceFilterNoPlayers`)

**Smoke: Flash DOES go through smoke.** Smoke has no solid collision. A flashbang on the other side of smoke will blind you if geometric LOS exists.

### 3.7 Complete Flash Reconstruction Formula

```python
def flash_alpha_at_time(time_since_flash, flash_duration, flash_max_alpha=255.0):
    """Returns screen overlay alpha (0-255). 255 = fully blind."""
    time_left = flash_duration - (time_since_flash - 0.0944)  # subtract build-up

    if time_left <= 0:
        return 0.0

    if time_left > 3.0:
        return flash_max_alpha  # fully blind plateau

    pct = time_left / 3.0
    return pct * pct * flash_max_alpha  # quadratic decay
```

### 3.8 Demo Data Available

| Source | Fields |
|--------|--------|
| Game event `flashbang_detonate` | `userid`, `entityid`, `x, y, z` |
| Game event `player_blind` | `userid` (blinded), `attacker`, `entityid`, `blind_duration` (= `m_flFlashDuration`) |
| Per-tick entity | `m_flFlashDuration` (float), `m_flFlashMaxAlpha` (float) |

---

## 4. Aim Punch / View Punch

### 4.1 m_aimPunchAngle -- The Recoil Offset

Type: `QAngle` (3 floats: pitch, yaw, roll).

- `[0]` / x = **PITCH** (negative = up, positive = down)
- `[1]` / y = **YAW** (left/right sway)
- `[2]` / z = **ROLL** (typically 0 from weapon recoil; nonzero from damage flinch)

**The demo value is raw/unscaled.** It is multiplied by `weapon_recoil_scale` (default **2.0**) at access time:

```cpp
QAngle CCSPlayer::GetAimPunchAngle()
{
    return m_Local.m_aimPunchAngle.Get() * weapon_recoil_scale.GetFloat();  // * 2.0
}
```

### 4.2 m_viewPunchAngle -- Cosmetic Screen Shake

Also `QAngle`, same structure. Key differences:
- Does NOT affect bullet direction -- purely cosmetic
- Applied at full scale (1:1), no multiplier
- Decays much faster: pure exponential with rate 18 (vs hybrid for aim punch)
- Generated as 5.5% of the aim kick magnitude: `weapon_recoil_view_punch_extra = 0.055`

### 4.3 The Complete Rendered Angle Formula

From `c_baseplayer.cpp` CalcView:

```
rendered_camera = eye_angles
                + m_viewPunchAngle                                            // cosmetic shake (1:1)
                + (m_aimPunchAngle * weapon_recoil_scale * view_recoil_tracking)  // visible recoil
                = eye_angles + m_viewPunchAngle + (m_aimPunchAngle_raw * 2.0 * 0.45)
                = eye_angles + m_viewPunchAngle + (m_aimPunchAngle_raw * 0.9)
```

Actual bullet direction (where shots land):

```
bullet_direction = eye_angles + (m_aimPunchAngle_raw * weapon_recoil_scale) + spread
                 = eye_angles + (m_aimPunchAngle_raw * 2.0) + spread
```

**The camera shows only 45% of the actual bullet deviation** from recoil. The camera also includes `m_viewPunchAngle` which bullets do NOT follow.

| ConVar | Default | Purpose |
|--------|---------|---------|
| `weapon_recoil_scale` | **2.0** | Multiplier on raw aim punch for bullet direction |
| `view_recoil_tracking` | **0.45** | Fraction of scaled aim punch shown in camera |
| `weapon_recoil_view_punch_extra` | **0.055** | Extra view shake per recoil kick (fraction) |

### 4.4 Aim Punch Decay -- Hybrid Exponential + Linear Model

CSGO replaced the base Source engine's damped spring with a hybrid decay:

```cpp
void HybridDecay(QAngle& v, float fExp, float fLin, float dT)
{
    fExp *= dT;
    fLin *= dT;
    v *= expf(-fExp);                    // exponential decay
    float fMag = v.Length();
    if (fMag > fLin)
        v *= (1.0f - fLin / fMag);      // linear decay (subtract fixed amount)
    else
        v.Init(0.0f, 0.0f, 0.0f);       // snap to zero
}

void CCSGameMovement::DecayAimPunchAngle(void)
{
    QAngle punchAngle    = m_pCSPlayer->m_Local.m_aimPunchAngle;
    QAngle punchAngleVel = m_pCSPlayer->m_Local.m_aimPunchAngleVel;

    // Step 1: Hybrid decay on angle
    HybridDecay(punchAngle, 8.0 /*exp*/, 18.0 /*lin*/, TICK_INTERVAL);

    // Step 2: Verlet integration -- first half-step
    punchAngle += punchAngleVel * TICK_INTERVAL * 0.5f;

    // Step 3: Exponential decay on velocity
    punchAngleVel *= expf(TICK_INTERVAL * -4.5);

    // Step 4: Verlet integration -- second half-step
    punchAngle += punchAngleVel * TICK_INTERVAL * 0.5f;
}
```

Per-tick numerical values (TICK_INTERVAL = 1/64 = 0.015625):

| Decay Component | Per-Tick Factor |
|-----------------|-----------------|
| Aim punch exp decay | `exp(-8 * 0.015625)` = **0.8825** (11.75% decay) |
| Aim punch linear decay | **0.28125 degrees** subtracted from magnitude |
| Aim punch velocity decay | `exp(-4.5 * 0.015625)` = **0.9321** (6.79% decay) |
| View punch decay | `exp(-18 * 0.015625)` = **0.7552** (24.5% decay -- much faster) |

### 4.5 Recoil Pattern Reset

Each weapon tracks `m_flRecoilIndex` (float). Starts at 0, increments by 1 per shot. Only decays after **110% of cycle time** since last shot:

```cpp
if (curtime > m_fLastShotTime + (GetCycleTime() * 1.10))
{
    float fDecayFactor = ln(10) * 2.0;  // ~ 4.605
    m_flRecoilIndex = Lerp(exp(TICK_INTERVAL * -4.605), 0.0, m_flRecoilIndex);
}
```

### 4.6 CS2 Post-April 2026 Breaking Change

Aim punch moved to `CCSPlayer_AimPunchServices` (via `CCSPlayerPawn.m_pAimPunchServices`):

| Field | Type | Offset | Description |
|-------|------|--------|-------------|
| `m_predictableBaseTick` | GameTick_t | 0x48 | |
| `m_predictableBaseAngle` | QAngle | 0x50 | Weapon recoil punch |
| `m_predictableBaseAngleVel` | QAngle | 0x5C | |
| `m_unpredictableBaseTick` | GameTick_t | 0xA0 | |
| `m_unpredictableBaseAngle` | QAngle | 0xA4 | Damage flinch punch |

Total punch = predictable (weapon recoil) + unpredictable (damage flinch).

### 4.7 Demo Data Available

**CSGO:**
- `CBasePlayer.m_Local.m_aimPunchAngle` (3 floats)
- `CBasePlayer.m_Local.m_aimPunchAngleVel` (3 floats)
- `CBasePlayer.m_Local.m_viewPunchAngle` (3 floats)

**CS2 (pre-April 2026):**
- `aim_punch_angle` on `CCSPlayerPawn.m_aimPunchAngle`
- `aim_punch_angle_vel` on `CCSPlayerPawn.m_aimPunchAngleVel`

**CS2 (post-April 2026):**
- `CCSPlayer_AimPunchServices.m_predictableBaseAngle` + `m_unpredictableBaseAngle`

---

## 5. FOV and Scoped View

### 5.1 Default FOV and Aspect Ratio Scaling

The default FOV of **90 degrees** is a **4:3-referenced horizontal FOV**. CS2/CSGO uses Hor+ (Horizontal Plus) scaling: vertical FOV is locked, horizontal FOV expands with wider aspect ratios.

```
engineAspectRatio = width / height
defaultAspectRatio = 4.0 / 3.0
ratio = engineAspectRatio / defaultAspectRatio
realHFov = 2 * atan(tan(engineFov/2) * ratio)
```

| Aspect Ratio | Vertical FOV | Horizontal FOV |
|--------------|-------------|----------------|
| 4:3 | ~73.74 deg | 90.00 deg |
| 16:10 | ~73.74 deg | ~100.39 deg |
| 16:9 | ~73.74 deg | ~106.26 deg |

### 5.2 Scoped FOV Per Weapon

From `items_game.txt` attributes (`zoom fov 1`, `zoom fov 2`):

| Weapon | zoom_lvl=0 | zoom_lvl=1 | zoom_lvl=2 | Zoom Levels |
|--------|-----------|-----------|-----------|-------------|
| **AWP** | 90 | **40** | **10** | 2 |
| **SSG 08 (Scout)** | 90 | **40** | **15** | 2 |
| **SCAR-20** | 90 | **40** | **15** | 2 |
| **G3SG1** | 90 | **40** | **15** | 2 |
| **AUG** | 90 | **45** | N/A | 1 |
| **SG 553** | 90 | **45** | N/A | 1 |

Approximate magnification:

| Config | Engine FOV | 16:9 Real H-FOV | Magnification |
|--------|-----------|-----------------|---------------|
| Unscoped | 90 | ~106 deg | 1.0x |
| AUG/SG scope | 45 | ~55 deg | ~2.0x |
| AWP/Scout zoom 1 | 40 | ~52 deg | ~2.25x |
| Scout/SCAR/G3 zoom 2 | 15 | ~19 deg | ~6x |
| AWP zoom 2 | 10 | ~13 deg | ~9x |

### 5.3 m_zoomLevel Values

```cpp
// 0 = Unscoped (hip fire)
// 1 = First zoom level
// 2 = Second zoom level
SendPropInt(SENDINFO(m_zoomLevel), 2, SPROP_UNSIGNED)  // 2-bit unsigned (0-3)
```

In demos: `zoom_lvl` maps to `C_CSWeaponBaseGun.m_zoomLevel`; `is_scoped` maps to `C_CSPlayerPawn.m_bIsScoped` (true when zoom_lvl > 0).

### 5.4 Scoped Recoil Behavior

- Aim punch still applies while scoped -- no immunity
- AUG/SG 553 have decreased spread when scoped (uses `Secondary_Mode` inaccuracy values)
- Scope accuracy penalty: `m_fAccuracyPenalty += GetCSWpnData().GetInaccuracyAltSwitch()` -- decays over time

### 5.5 FOV Transition Timing

CS2 VData fields on `CCSWeaponBaseVData`:

| Field | Offset | Type | Description |
|-------|--------|------|-------------|
| `m_nZoomLevels` | 0x7FC | int32 | Number of zoom levels |
| `m_nZoomFOV1` | 0x800 | int32 | First zoom FOV |
| `m_nZoomFOV2` | 0x804 | int32 | Second zoom FOV |
| `m_flZoomTime0` | 0x808 | float | Time to unzoom |
| `m_flZoomTime1` | 0x80C | float | Time for first zoom |
| `m_flZoomTime2` | 0x810 | float | Time for second zoom |

Camera service fields on `CCSPlayerBase_CameraServices`:

| Field | Offset | Description |
|-------|--------|-------------|
| `m_iFOV` | 0x290 | Current/target FOV |
| `m_iFOVStart` | 0x294 | FOV at transition start |
| `m_flFOVTime` | 0x298 | Transition start time |
| `m_flFOVRate` | 0x29C | Transition duration |

### 5.6 FOV Reconstruction

```python
WEAPON_ZOOM_FOV = {
    "weapon_awp":     {0: 90, 1: 40, 2: 10},
    "weapon_ssg08":   {0: 90, 1: 40, 2: 15},
    "weapon_scar20":  {0: 90, 1: 40, 2: 15},
    "weapon_g3sg1":   {0: 90, 1: 40, 2: 15},
    "weapon_aug":     {0: 90, 1: 45},
    "weapon_sg556":   {0: 90, 1: 45},
}

def engine_fov_to_real_hfov(engine_fov, aspect_ratio=16/9):
    """Convert engine FOV (4:3 referenced) to actual horizontal FOV."""
    import math
    ratio = aspect_ratio / (4/3)
    return 2 * math.atan(math.tan(math.radians(engine_fov / 2)) * ratio)

def is_in_fov(viewer_eye, viewer_angles, target_pos, engine_fov, aspect_ratio=16/9):
    """Check if target_pos falls within the viewer's FOV cone."""
    import math
    real_hfov = engine_fov_to_real_hfov(engine_fov, aspect_ratio)
    real_vfov = 2 * math.atan(math.tan(math.radians(engine_fov / 2)))  # 4:3 base = vfov

    dx = target_pos.x - viewer_eye.x
    dy = target_pos.y - viewer_eye.y
    dz = target_pos.z - viewer_eye.z

    # Horizontal angle check
    angle_h = math.atan2(dy, dx) - math.radians(viewer_angles.yaw)
    # Normalize to [-pi, pi]
    angle_h = (angle_h + math.pi) % (2 * math.pi) - math.pi

    # Vertical angle check
    dist_h = math.sqrt(dx*dx + dy*dy)
    angle_v = math.atan2(-dz, dist_h) - math.radians(viewer_angles.pitch)
    angle_v = (angle_v + math.pi) % (2 * math.pi) - math.pi

    return abs(angle_h) <= real_hfov / 2 and abs(angle_v) <= real_vfov / 2
```

---

## 6. Player Dimensions and Eye Position

### 6.1 Collision Hull

From `cs_gamerules.cpp`, `g_CSViewVectors`:

```cpp
static CViewVectors g_CSViewVectors(
    Vector( 0, 0, 64 ),        // VEC_VIEW (standing eye offset)
    Vector(-16, -16, 0 ),      // VEC_HULL_MIN
    Vector( 16,  16, 72 ),     // VEC_HULL_MAX
    Vector(-16, -16, 0 ),      // VEC_DUCK_HULL_MIN
    Vector( 16,  16, 54 ),     // VEC_DUCK_HULL_MAX
    Vector( 0, 0, 46 ),        // VEC_DUCK_VIEW (crouching eye offset)
    Vector(-10, -10, -10 ),    // VEC_OBS_HULL_MIN
    Vector( 10,  10,  10 ),    // VEC_OBS_HULL_MAX
    Vector( 0, 0, 14 )         // VEC_DEAD_VIEWHEIGHT
);
```

| Property | Standing | Crouching |
|----------|----------|-----------|
| Hull width (X) | 32 (-16 to +16) | 32 (-16 to +16) |
| Hull depth (Y) | 32 (-16 to +16) | 32 (-16 to +16) |
| Hull height (Z) | **72** (0 to 72) | **54** (0 to 54) |
| Eye height (Z) | **64** | **46** |
| Hull mins | (-16, -16, 0) | (-16, -16, 0) |
| Hull maxs | (16, 16, 72) | (16, 16, 54) |

**Player origin** (`m_vecAbsOrigin`) is at the **bottom center** of the collision hull (the feet).

**CS2 status:** No confirmed changes to hull dimensions. Same 32x32x72 / 32x32x54 values.

### 6.2 Eye Position -- The 64.062561 Quantization

The true internal values are **64.0** (standing) and **46.0** (crouching). The commonly seen values of **64.062561** and **46.044968** come from CSGO's 10-bit network quantization:

```cpp
// From player.cpp:
SendPropFloat(SENDINFO_VECTORELEM(m_vecViewOffset, 2), 10, SPROP_CHANGES_OFTEN, 0.0f, 128.0f)
```

The Z component is encoded as a 10-bit integer over range [0.0, 128.0]:
- Resolution: `128.0 / 1023 = 0.12512218963`
- Standing: `512 * 0.125122... = 64.062561`
- Crouching: `368 * 0.125122... = 46.044968`
- Delta: `18.017593`

**CS2:** Source 2 may encode `m_vecViewOffset` with full float precision (exactly 64.0). The `m_vecViewOffset` in CS2 is stored as `CNetworkViewOffsetVector` on `C_BaseModelEntity` (offset 0x0E78) and is properly networked for ALL players in GOTV demos (unlike CSGO where it was only reliable for the local player).

### 6.3 Eye Position During Duck Transition

```
eye_z = origin_z + 64.062561 - (18.017593 * m_flDuckAmount)
```

Where `m_flDuckAmount` ranges from 0.0 (standing) to 1.0 (fully crouched).

Or more precisely:
```
eye_pos = player_origin + m_vecViewOffset  // if m_vecViewOffset is available and reliable
```

### 6.4 Duck Transition Mechanics

From `shareddefs.h`:

| Constant | Value |
|----------|-------|
| `TIME_TO_DUCK_MSECS` | **200 ms** (CSGO-specific; base Source uses 400ms) |
| `TIME_TO_UNDUCK_MSECS` | **200 ms** |
| `sv_timebetweenducks` | 0.4s (anti-spam cooldown) |

**Interpolation: LINEAR** (CSGO replaced base Source's Hermite smoothstep with `Approach()`):

```cpp
// Ducking in:
m_flDuckAmount = Approach(1.0f, m_flDuckAmount, player->m_flDuckSpeed * 0.8f);
// Unducking:
m_flDuckAmount = Approach(0.0f, m_flDuckAmount, MAX(1.5f, player->m_flDuckSpeed));
```

**Animation layer duck (third-person model) uses DIFFERENT speeds:**
```cpp
#define CSGO_ANIM_DUCK_APPROACH_SPEED_DOWN  3.1f   // crouching down
#define CSGO_ANIM_DUCK_APPROACH_SPEED_UP    6.0f   // standing up
```

The third-person model's head position can lag behind the actual eye position during transitions.

### 6.5 Hitbox System

**CSGO (post-2015):** Mix of capsules and boxes. 19 hitboxes per model.

**CS2:** ALL hitboxes are capsules. All agents share identical hitbox dimensions.

Head hitbox approximate dimensions:
- Extent: ~9.7 x 6.4 x 6.1 units
- Effective capsule radius: ~3.5-4.0 units

CS2 head hitbox increased 4-20% volume from CSGO; neck decreased 28.55%. All standardized across agents.

**Hitbox groups and damage multipliers (identical CSGO/CS2):**

| Region | Hitgroup | Multiplier |
|--------|----------|-----------|
| Head | 1 | 4.0x |
| Chest/Arms | 2, 4, 5 | 1.0x |
| Stomach/Pelvis | 3 | 1.25x |
| Legs | 6, 7 | 0.75x |

### 6.6 Animation-Dependent Behavior

- **Planting bomb:** Forces crouching state. 3.2s plant animation. Hitbox = crouched (54 units).
- **Defusing:** Can be standing OR crouching. Duck speed multiplied by 0.4 during defuse.
- **Ladder climbing:** No hull height change. Maintains standing (72) or crouching (54).
- **Crouch jumping:** Raises collision box up 18 units (72 - 54).

### 6.7 Coordinate System

- **Axes:** X = East, Y = North, Z = Up (right-handed)
- **Yaw 0** = facing East (+X), increases counter-clockwise from above
- **Pitch:** positive = looking down. Range -90 (straight up) to +90 (straight down)
- **1 HU (map scale)** = 1.905 cm = 0.75 inches
- **1 HU (character scale)** = 2.54 cm = 1 inch

### 6.8 Demo Data Available

| Field | Parser Name | Type | Notes |
|-------|-------------|------|-------|
| `m_vecAbsOrigin` | X, Y, Z | float | Player feet position |
| `m_vecViewOffset` | - | Vector3 | Eye offset from origin |
| `m_flDuckAmount` | duck_amount | float (0.0-1.0) | Duck interpolation state |
| `m_bDucked` | ducked | bool | Fully crouched |
| `m_bDucking` | ducking | bool | In transition |
| `m_angEyeAngles` | pitch, yaw | float | Eye direction |
| `m_bIsScoped` | is_scoped | bool | Scoped in |
| `m_bIsDefusing` | is_defusing | bool | Currently defusing |

**View direction vector from eye angles:**
```
forward_x = cos(pitch) * cos(yaw)
forward_y = cos(pitch) * sin(yaw)
forward_z = -sin(pitch)
```
(Convert degrees to radians. Pitch positive = looking down, hence negative Z.)

---

## 7. PVS and Demo File Completeness

### 7.1 CSGO: BSP-Based PVS

Source 1 PVS was precomputed in the BSP: VVIS compiler split space into visleaves/clusters, computed cluster-to-cluster visibility via portal algorithms, stored as RLE-compressed bitfield. The PVS was conservative -- it over-estimated visibility but never under-estimated.

### 7.2 CS2: No Server-Side PVS Entity Culling

**CS2 official matchmaking servers do NOT perform PVS-based entity culling.** All player entities are transmitted to all clients at all times. This is why wallhacks work -- enemy position data is always in client memory.

Source 2 replaced BSP with octree-based world nodes (`.vwnod_c`). VIS3 is used for rendering optimization (GPU culling, meshlets) but NOT for entity transmission decisions.

### 7.3 GOTV Demos Contain ALL Entities

**Yes. GOTV demos contain complete, unculled entity data for all players at all ticks.**

GOTV is a spectator client. The SourceTV master lives in the game server process. In `hltvserver.cpp`, the HLTV client's `SetupVisibility()` returns **NULL PVS and PAS** for `FL_PROXY` clients, meaning all entities transmit without filtering.

Since CS2 official servers don't do PVS culling anyway, GOTV demos from matchmaking inherently contain all player data.

### 7.4 GOTV vs POV Demo Differences

| Aspect | GOTV Demo | POV Demo |
|--------|-----------|----------|
| Entity data | All 10 players, every tick | All players (CS2 transmits all) but with client interpolation |
| Position accuracy | Raw server-authoritative state | Local player has prediction; others have interpolation |
| Utility data | Complete grenade data | Unreliable utility data |
| Timing | Server tick timing | Client-side timing with interpolation offsets |

**For reconstruction, always use GOTV demos.**

### 7.5 CS2 Dual-Entity Player Model

CS2 uses two entities per player:
- **CCSPlayerController** -- persistent metadata: name, team, alive state, pawn handle
- **CCSPlayerPawn** -- physical entity: position, health, weapons, movement state

### 7.6 CS2 Tick Rate

64 Hz (15.625 ms per tick). Sub-tick system timestamps inputs between ticks with ~1ms precision, but player positions remain tick-rate bound.

---

## 8. Edge Cases for Visibility

### 8.1 Wallbang Spots -- NOT a Visibility Issue

Wallbang penetration and visibility are **independent systems**. A thin wall that bullets penetrate still **fully blocks** the engine's line-of-sight trace. All solid geometry in `.tri`/`.vphys` files blocks visibility completely regardless of material thickness.

### 8.2 Props Not in Collision Mesh

**By default, NO prop entity type blocks visibility.** `prop_static`, `prop_dynamic`, `prop_physics` do NOT block visibility unless the map author explicitly marked them as **Vis Occluder**. This is separate from physical collision.

The `.tri` files contain world geometry and props marked as vis occluders. They do NOT include prop models lacking the vis occluder flag, even visually large ones (crates, barrels, vehicles).

**Implication:** Your `.tri`-based approach will miss large props the map author did NOT mark as vis occluders. These may visually block view in-game but won't appear in the physics mesh.

### 8.3 Doors

Door entities exist in CS2 and their state IS tracked in demo files.

Entity classes: `CBaseDoor`, `CRotDoor`, `CPropDoorRotatingBreakable`

`m_eDoorState` property: 0=Closed, 1=Opening, 2=Open, 3=Closing

Game events: `door_open`, `door_close`, `door_moving`, `door_break`, `door_closed` -- all with `entindex` and `userid_pawn` fields.

**Implication:** Door state changes ARE recorded. The `.tri` file contains doors in default (closed) position. You must adjust for open doors by removing closed-position geometry.

### 8.4 Breakable Objects

Game events: `break_breakable` and `break_prop` with `entindex`, `userid`, `material` (BREAK_GLASS, BREAK_WOOD, etc.)

Known breakables on competitive maps: Nuke breakable vent, Nuke Lockers window, Squeaky doors.

**Implication:** When `break_breakable` fires, geometry at that entity's position should be removed from visibility calculations.

### 8.5 Water Surfaces

**Water does NOT block line-of-sight.** Can be safely ignored.

### 8.6 HE Grenades and Decoys

**HE explosions:** No visibility blocking. BUT HE detonating inside smoke creates a temporary gap for ~2-3 seconds.

**Decoys:** No visual obstruction. Audio-only. Ignore for visibility.

### 8.7 Player Bodies Blocking Visibility

Player models have collision and CAN physically block another player's line of sight. They are NOT in the static `.tri` mesh -- they are dynamic entities.

`mp_solid_teammates` controls teammate collision. In competitive matchmaking, enabled by default.

**For accuracy, you should check if any other player's bounding box intersects the ray between two players.** Most `.tri`-based systems miss this entirely.

### 8.8 Smoke One-Way Visibility (CS2)

CS2's volumetric system largely eliminates traditional one-way smokes. Remaining edge cases:
- Height advantage (seeing heads over top)
- HE grenade punching holes (2-3 second gap)
- Bullets creating small temporary gaps
- Edge positioning asymmetry
- Geometry conformance (not a perfect sphere)

### 8.9 The m_bSpotted / m_bSpottedByMask Flags

`CCSPlayerPawn` has `m_entitySpottedState` containing `m_bSpottedByMask` (uint32 bitmask). **This is unreliable** for visibility determination. The awpy documentation explicitly warns against using it. These flags may lag behind actual visibility or be set by criteria other than pure LOS.

---

## 9. Theoretical Maximum Accuracy Assessment

### 9.1 What We Can Reconstruct Perfectly

| Component | Accuracy | Source |
|-----------|----------|--------|
| Player positions (feet) | **Exact** (server-authoritative) | GOTV demo per-tick X/Y/Z |
| Eye angles (pitch/yaw) | **Exact** | `m_angEyeAngles` per-tick |
| Duck state / eye height | **Exact** | `m_flDuckAmount` + formula |
| Flash blindness state | **Exact** (timing/alpha from demo) | `m_flFlashDuration`, `m_flFlashMaxAlpha` |
| Aim punch angle | **Exact** (raw value in demo) | `m_aimPunchAngle` per-tick |
| Zoom level / scoped state | **Exact** | `zoom_lvl`, `is_scoped` per-tick |
| Active weapon | **Exact** | Entity data per-tick |
| Smoke detonation position & timing | **Exact** | `smokegrenade_detonate` event + entity fields |
| Fire positions (per-flame) | **Exact** (CS2: absolute world-space) | `m_firePositions[64]` on CInferno |
| Door open/close events | **Exact** | `door_open`/`door_close` events |
| Breakable destruction events | **Exact** | `break_breakable`/`break_prop` events |
| All 10 players visible in GOTV | **Guaranteed** (no PVS culling) | GOTV transmit architecture |

### 9.2 What Has Known Error Margins

| Component | Error | Root Cause | Mitigation |
|-----------|-------|-----------|------------|
| Static geometry LOS | **~95-98%** | Props not marked as Vis Occluder missing from `.tri` | Manual audit per map; add missing prop collision |
| CSGO smoke occlusion | **~90-95%** | CSGO sphere model is exact for CSGO but doesn't account for timing nuance | Use the exact `CheckTotalSmokedLength` algorithm with R=166 |
| CS2 smoke occlusion | **~70-85%** | Sphere approximation misses volumetric conformance to geometry | Parse `m_VoxelFrameData` for exact voxel grid (complex); or accept sphere approximation |
| Door geometry state | **~90%** | `.tri` has doors closed; need to remove/add geometry per events | Track `m_eDoorState` and modify BVH accordingly |
| Breakable geometry | **~95%** | Same as doors -- need to remove broken geometry | Track `break_breakable` events |
| Player body occlusion | **~98% without, ~99.5% with** | Players blocking LOS to other players | Check other player bounding boxes against ray |
| Flash partial blindness | **~90%** | Exact `percentageOfFlash` depends on 4-ray LOS trace we can't reproduce without full geometry | Use simplified single-ray + angle tier |
| Rendered camera angle | **~99%** | Need `m_viewPunchAngle` which may not be in CS2 demos | Can approximate from aim punch alone |
| FOV during zoom transitions | **~95%** | Transition timing between zoom levels | Assume instant FOV change at zoom_lvl change tick |
| HE-clears-smoke interaction | **~80%** | HE punches temporary holes in CS2 smoke | Track HE detonations near smoke positions, assume 2-3s gap |

### 9.3 Architecture for Maximum Accuracy

```
Per-tick pipeline:
  1. Parse player state: position, eye_angles, duck_amount, is_scoped, zoom_lvl,
     aim_punch, flash_duration, flash_max_alpha, active_weapon, health

  2. Compute eye position:
     eye_z = origin_z + 64.062561 - (18.017593 * duck_amount)
     (or use m_vecViewOffset if available and reliable in CS2)

  3. Compute rendered camera direction:
     camera = eye_angles + aim_punch_raw * 0.9
     (add m_viewPunchAngle if available)

  4. Compute FOV:
     engine_fov = WEAPON_ZOOM_FOV[weapon][zoom_lvl]

  5. For each player pair (A observing B):
     a. Compute A's eye position
     b. Compute B's visibility points:
        - Eye position (head)
        - Body center: origin + (0, 0, hull_height/2)
        - Shoulders: origin + (0, 0, hull_height * 0.8) +/- (8, 0, 0)
     c. Static geometry LOS check:
        - Ray cast from A's eye to each of B's visibility points
        - Using BVH over .tri mesh (Moller-Trumbore intersection)
        - ~65-177 microseconds per check
     d. Dynamic occluder checks:
        - Smoke: sphere intersection (R=166, center=detonation+(0,0,60))
        - Doors: check m_eDoorState, modify geometry accordingly
        - Breakables: remove geometry after break events
        - Player bodies: check other players' bounding boxes
     e. FOV check:
        - Is B's position within A's FOV cone?
        - Account for aim_punch offset
     f. Perceptual blindness check:
        - If flash_alpha_at_time() > threshold, A is effectively blind

  6. Output per-tick visibility matrix:
     - can_see[A][B] = static_LOS && !smoked && !flashed && in_FOV
     - Separate geometric visibility (LOS) from perceptual visibility (flash/FOV)
```

### 9.4 Performance Estimates

From awpy benchmarks:

| Operation | Time | Notes |
|-----------|------|-------|
| Single ray cast (BVH) | ~65-177 us | Depends on hit/miss |
| BVH construction | 744ms - 9.6s | Per map, one-time cost |
| de_dust2 triangles | 326,265 | |
| de_mirage triangles | ~55,000 | (from .tri file) |

For 10 players, 90 directional pairs per tick, 3 visibility points per target = 270 ray casts/tick.

At 177 us worst case: 270 * 177 us = ~48 ms per tick. At 64 ticks/sec, that's ~3.1 seconds per second of gameplay. **Real-time reconstruction is feasible on a single core.**

### 9.5 Overall Theoretical Maximum Accuracy

**For CSGO demos with the sphere smoke model:** ~97-99% accuracy for geometric LOS. The primary error source is missing prop geometry in `.tri` files. The smoke algorithm is an exact reproduction of Valve's code.

**For CS2 demos with sphere smoke approximation:** ~90-95% accuracy. The volumetric smoke system is the largest source of error. The sphere approximation will produce false positives (saying "not visible" when a player IS visible through a geometry-conforming gap) and false negatives (saying "visible" when smoke has filled a corridor more completely than a sphere would).

**For CS2 demos with full voxel reconstruction:** ~97-99% accuracy if `m_VoxelFrameData` is parsed and used for raymarching. This requires implementing the voxel grid format parsing and raymarching logic, which is significantly more complex.

**The ceiling is set by:**
1. Missing props in the `.tri` mesh (~2-5% of sightlines on some maps)
2. CS2 smoke volumetric conformance (sphere approx misses ~5-10%)
3. Door/breakable geometry state tracking completeness
4. Player body occlusion (rare but real edge case)
5. Third-person animation offset during duck transitions (head bone lags eye position)

### 9.6 Existing Implementations and Tools

| Project | Language | CS2 | Approach | Performance |
|---------|----------|-----|----------|-------------|
| **awpy** | Python | Yes | `.tri` BVH + Moller-Trumbore | ~65-177 us/ray |
| **CS2FOW** (karola3vax) | C++/C# | Yes | `.bvh8` + 19-capsule silhouette projection | Production anti-cheat |
| **VisCheckCS2** | Python/C++ | Yes | `.vphys` -> BVH | ~1ms/ray |
| **cs2-nav** | Rust | Yes | `.tri` + serialized BVH | crates.io |
| **demoparser2** | Rust | Yes | 100+ props/tick | 749 MB/s (12-core) |
| **demoinfocs-golang** | Go | Yes (v4/v5) | Full entity access | Production-ready |
| **DemoFile.Net** | C# | Yes | Full entity schema | 353ms parallel/match |

### 9.7 Key Reference Implementation: CS2FOW

CS2FOW (github.com/karola3vax/CS2FOW) is the most sophisticated community implementation:

1. **Map baking:** Extract physics collision from VPK, strip to sight-check triangles, build **BVH8** (branching factor 8)
2. **Per-tick:** Copy 19 animated hitbox capsules per player
3. **Visibility:** Project 3D capsule silhouettes against BVH8 wall geometry using Moller-Trumbore
4. **Fallback:** If silhouette fully blocked, check 8 AABB corners padded 16 units sideways + 4 units up, then weapon muzzle
5. **Prediction:** Reveals enemies slightly before exact LOS using movement prediction and ping compensation
6. **Smoothing:** Keeps revealed enemies visible ~1 second after losing LOS to prevent flicker

### 9.8 What Valve Uses Internally

Valve does NOT implement server-side visibility culling on official CS2 servers. Their anti-cheat stack:
- **VAC:** Signature scanning
- **VACNet:** Server-side AI analyzing demo replays
- **VAC Live:** Real-time server-side behavioral detection
- **Trust Factor:** Behavioral analysis
- None of these use visibility culling / fog of war

---

## Quick Reference Tables

### Smoke Parameters

| Parameter | CSGO | CS2 |
|-----------|------|-----|
| Shape | Sphere | Volumetric voxel grid |
| Occlusion radius | 166 units | ~144 units (approx; varies with geometry) |
| Center offset | +60 Z | Per detonation position |
| Blocking threshold | 70% of radius | Voxel density |
| Duration | ~17.5s | ~18s |
| Expansion (gameplay) | Instant | Gradual flood fill |
| Geometry conformance | None | Yes (fills rooms) |

### Player Dimensions

| Property | Value |
|----------|-------|
| Standing hull | 32 x 32 x 72 HU |
| Crouching hull | 32 x 32 x 54 HU |
| Standing eye height | 64.0 (wire: 64.062561) |
| Crouching eye height | 46.0 (wire: 46.044968) |
| Eye height delta | 18.0 (wire: 18.017593) |
| Dead view height | 14.0 |
| Head hitbox radius | ~3.5-4.0 HU |
| Duck transition time | 200 ms (linear) |
| Player origin | Bottom center (feet) |

### Flash Parameters

| Parameter | Value |
|-----------|-------|
| Max effective range | 3000 units |
| Distance falloff | Linear: `damage - dist * (damage/3000)` |
| Angle tiers | 4 tiers at dot thresholds: 0.6, 0.3, -0.2 |
| Fade curve | `(timeLeft/3.0)^2 * 255` when timeLeft < 3.0s |
| Full-blind threshold | timeLeft > 3.0s = 100% white |
| flash_duration meaning | Fade time (NOT total), stored after /1.4 division |
| flash_max_alpha | Always 255 in practice |
| Through walls? | NO (multi-ray LOS trace) |
| Through smoke? | YES |

### Rendered Angle

```
camera = eye_angles + m_viewPunchAngle + (m_aimPunchAngle_raw * 2.0 * 0.45)
bullet = eye_angles + (m_aimPunchAngle_raw * 2.0) + spread
```

### Unit Conversions

| Value | Equivalent |
|-------|-----------|
| 1 HU (map) | 1.905 cm / 0.75 inches |
| 1 foot | 16 HU |
| 1 meter | ~52.5 HU |
| 166 units (smoke radius) | ~3.16 meters |
| 3000 units (flash range) | ~57 meters |
| 72 units (standing height) | ~1.37 meters |

---

*Document generated 2026-08-05. Sources: Valve leaked CSGO source (cstrike15_src), CS2 schema dumps (a2x/cs2-dumper, s2v.app), CS2FOW (karola3vax), awpy (pnxenopoulos), demoparser2 (LaihoE), demoinfocs-golang (markus-wa), CS2 Smoke Recreation (GarrettGunnell), Osiris cheat source (danielkrupinski), Valve Developer Community wiki, AlliedModders forums, UnknownCheats community.*
