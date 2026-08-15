"""Event -> clip -> t: choose an animation from what the demo actually says.

    from anim_script import ClipSet, Scripter

THE WIRE CARRIES NO POSE. Valve deleted DEM_AnimationData at proto 14150,
so nothing in a .dem says which frame of which clip a player was on. What
the demo DOES carry is state, and state plus the authored clip names is
enough to choose: a tick where `fire` is set is a tick where the shoot clip
is playing, whatever its phase. That inference is the whole of this file,
and every choice it makes is printed with the column that drove it.

COLUMNS ARE PROBED, NOT ASSUMED. The camera rows on the cs1k corpus carry
exactly these, read off a real file rather than a schema doc:

    fire              1 on the tick a shot was fired        -> shoot
    active_weapon_id  changes when the player switches      -> draw
    ammo_clip         RISES when a magazine is replaced     -> reload
    x, y, z           per-tick deltas give speed and phase  -> walk / run
    duck_button       held while crouched                   -> crouch
    health            reaches 0                             -> death

There is no weapon_reload column and none is invented: a reload is inferred
from ammo_clip RISING, which is the observable the event would have named.
That is stated on the line it drives, so nobody reads it as a wire event.

WHERE NO EVENT DRIVES A STATE, the choice is STATED procedural -- authorized
by the owner's fake-data ruling and printed as STATED, never as READ. The
distinction is the point: `phase` within a walk cycle is not on the wire and
cannot be, so it is derived from distance travelled and labelled so.
"""
from __future__ import annotations

import os

import numpy as np
import torch

SCHEMA = "iji/model-clips/v2"

# Source units per second below which a player is standing still. CS2's walk
# is ~250 u/s and its run ~
# 250; the threshold only has to separate "not moving" from "moving", and it
# is STATED rather than fitted -- no metric was optimised against it.
STILL_UPS = 8.0
RUN_UPS = 140.0


class ClipSet:
    """One `<asset>.clips.pt` sidecar, with its scope and schema READ."""

    def __init__(self, path):
        self.path = path
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.schema = d.get("schema", "UNSTATED")
        self.scope = d.get("scope", "UNSTATED")
        self.skeleton = d.get("skeleton", "UNSTATED")
        self.asset = d.get("weapon") or d.get("agent") or os.path.basename(path)
        self.events = d.get("events", {})
        self.note = (f"{self.asset}: {sum(len(v) for v in self.events.values())} "
                     f"clips over {len(self.events)} events, scope "
                     f"{self.scope!r}, schema {self.schema}")
        if self.schema != SCHEMA:
            # READ the field rather than trusting the filename. A sidecar at
            # a different schema may still load and mean something else.
            self.note += (f"  ⚠️ EXPECTED schema {SCHEMA}; this file says "
                          f"{self.schema!r}, so field meanings are not "
                          f"guaranteed and nothing here corrects for it")

    def has(self, event):
        return bool(self.events.get(event))

    def pick(self, event, key=0):
        """A clip for `event`: deterministic, and ABSOLUTE where one exists.

        Several clips per event is the norm (ak_47 ships 6 draws, 2 shoots).
        Which one the engine picked on a given tick is NOT on the wire, so
        the index is a STATED choice keyed on something stable -- never
        random, or two renders of one tick would disagree.

        ADDITIVE CLIPS HOLD DELTAS, NOT POSES (m_bIsAdditive). Applied as
        absolute bone transforms they render nonsense, and it presents as a
        pose bug rather than a semantics one -- the tell that exposed it
        upstream was a thigh "translating" a constant 136.1 mm at exactly
        180 degrees. This lane has no read of the base pose those deltas are
        against, so it does not composite them and does not guess one:
        it PREFERS an absolute sibling, and returns the additive clip only
        when the event has nothing else, flagged so the caller can refuse.

        The counts are why this is not theoretical. Viewmodel side: idle
        2 additive of 97, shoot 1 of 53, other 11 of 221. World side is far
        worse -- 123 of 132 idles -- and idle is the scripter's else-branch,
        so the default path is the one that would have broken.
        """
        lst = self.events.get(event) or []
        if not lst:
            return None
        absolute = [c for c in lst if not c.get("additive")]
        if absolute:
            return absolute[int(key) % len(absolute)]
        return lst[int(key) % len(lst)]

    def absolute_count(self, event):
        return sum(1 for c in (self.events.get(event) or [])
                   if not c.get("additive"))


def bone_track(block, name):
    """Track index of `name` in this block, or None. READ from bone_ids."""
    ids = block.get("bone_ids")
    if ids is None:
        return None
    try:
        return list(ids).index(name)
    except ValueError:
        return None


def sample_block(block, t_sec, duration, frames):
    """(t, q, s) for every track at time `t_sec`, linearly interpolated.

    q is xyzw and is SLERPed the short way; a lerp on quaternions is wrong
    at large angles and a recoil is a large angle. Frames outside the clip
    wrap, because every clip here is authored as a cycle or is clamped by
    the caller choosing t.
    """
    if frames <= 1 or duration <= 0:
        return block["t"][0], block["q"][0], block["s"][0]
    f = (float(t_sec) / float(duration)) * (frames - 1)
    f = f % (frames - 1) if f >= frames - 1 else max(0.0, f)
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, frames - 1)
    a = float(f - i0)
    t0, t1 = block["t"][i0], block["t"][i1]
    s0, s1 = block["s"][i0], block["s"][i1]
    q0, q1 = block["q"][i0].clone(), block["q"][i1].clone()
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)          # short way
    dot = dot.abs().clamp(max=1.0)
    th = torch.acos(dot)
    sin = torch.sin(th)
    w0 = torch.where(sin > 1e-6, torch.sin((1 - a) * th) / sin, 1.0 - a)
    w1 = torch.where(sin > 1e-6, torch.sin(a * th) / sin,
                     torch.full_like(sin, a))
    q = w0 * q0 + w1 * q1
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return t0 + (t1 - t0) * a, q, s0 + (s1 - s0) * a


class Scripter:
    """Per-entity animation state, advanced tick by tick."""

    def __init__(self, tick_rate=64.0):
        self.tick_rate = float(tick_rate)
        self.prev = {}
        self.dist = {}
        # (event, start_tick, clip) of a one-shot currently playing OUT.
        # The fire column is an EDGE -- 1 on the tick of the shot, 0 the
        # tick after -- while the shoot clip has DURATION (ak: 0.767 s,
        # 49 ticks). Mapping the column level to the event made the bolt
        # snap for one tick and return to idle 48 ticks early; the A/B
        # that exposed it showed 2 of 32 frames moving in a window with
        # four shots. The latch also PINS THE PICKED CLIP: pick() keys
        # shoot/draw on the tick, so re-picking mid-playthrough would flip
        # ak's two shoot clips against each other every tick.
        self.active = {}

    def choose(self, key, row, prev_row, clips):
        """(event, clip, t_sec, why) for one entity on one tick.

        Priority is the order the states actually override each other in
        play: a shot fired while walking shows the shoot clip on the arms.
        """
        tick = int(row.get("tick", 0))
        why = []
        ev = None

        if row.get("fire"):
            ev, why = "shoot", ["fire=1 (READ: weapon_fire column)"]
        elif prev_row is not None and \
                row.get("active_weapon_id") != prev_row.get("active_weapon_id"):
            ev = "draw"
            why = [f"active_weapon_id {prev_row.get('active_weapon_id')} -> "
                   f"{row.get('active_weapon_id')} (READ)"]
        elif prev_row is not None and \
                (row.get("ammo_clip") or 0) > (prev_row.get("ammo_clip") or 0):
            ev = "reload"
            why = [f"ammo_clip rose {prev_row.get('ammo_clip')} -> "
                   f"{row.get('ammo_clip')} (READ; there is no weapon_reload "
                   f"column in this corpus and none is invented)"]
        else:
            sp = 0.0
            if prev_row is not None:
                d = np.array([float(row.get(c, 0) or 0)
                              - float(prev_row.get(c, 0) or 0)
                              for c in ("x", "y", "z")])
                dt = max(1, tick - int(prev_row.get("tick", tick - 1)))
                sp = float(np.linalg.norm(d)) * self.tick_rate / dt
                self.dist[key] = self.dist.get(key, 0.0) + float(
                    np.linalg.norm(d[:2]))
            if row.get("duck_button") and sp <= STILL_UPS:
                ev, why = "crouch", ["duck_button set, speed "
                                     f"{sp:.1f} u/s (READ)"]
            elif sp > RUN_UPS:
                ev, why = "run", [f"speed {sp:.1f} u/s from x/y/z deltas "
                                  f"(READ) > {RUN_UPS} STATED threshold"]
            elif sp > STILL_UPS:
                ev, why = "walk", [f"speed {sp:.1f} u/s from x/y/z deltas "
                                   f"(READ) > {STILL_UPS} STATED threshold"]
            else:
                ev, why = "idle", ["no event and speed "
                                   f"{sp:.1f} u/s -- STATED idle loop"]

        if ev not in ("shoot", "draw", "reload"):
            # No one-shot on THIS row's columns -- but one may still be
            # playing out. Continue it for exactly its authored duration,
            # with the clip PINNED to the instance that started it.
            act = self.active.get(key)
            if act is not None:
                a_ev, a_st, a_clip = act
                el = (tick - a_st) / self.tick_rate
                if 0.0 <= el < float(a_clip["duration"]):
                    self.prev[key] = row
                    why = [f"playing out {a_ev!r} from tick {a_st} -- t "
                           f"{el:.3f}s of {a_clip['duration']:.3f}s (READ: "
                           f"the fire column marks the EDGE of the event; "
                           f"the clip's duration is the asset's)"]
                    return a_ev, a_clip, float(el), why
                self.active.pop(key, None)

        if not clips.has(ev):
            fallback = next((e for e in ("idle", "walk", "draw")
                             if clips.has(e)), None)
            why.append(f"{ev!r} has no clip in this sidecar; fell back to "
                       f"{fallback!r}")
            ev = fallback
            if ev is None:
                return None, None, 0.0, why

        clip = clips.pick(ev, key=tick if ev in ("shoot", "draw") else 0)
        if clip is not None and clip.get("additive"):
            # Every clip for this event is a DELTA set. Refuse it and say so
            # rather than apply deltas as poses; then look for any event that
            # does have an absolute clip, so the asset still animates.
            alt = next((e for e in ("idle", "walk", "draw", "lookat")
                        if clips.absolute_count(e)), None)
            why.append(f"ADDITIVE REFUSED: every {ev!r} clip in this sidecar "
                       f"is m_bIsAdditive (deltas against a base pose this "
                       f"lane has no read of). Substituted {alt!r}, which "
                       f"has an absolute clip. Nothing is composited against "
                       f"a guessed base.")
            if alt is None:
                return None, None, 0.0, why
            ev = alt
            clip = clips.pick(ev, key=0)
        # t within the clip. For a one-shot (shoot/draw/reload) it runs from
        # the tick the event STARTED, so the clip plays through rather than
        # restarting every tick. For a cycle it is phase from distance
        # travelled -- STATED, because stride phase is not on any wire.
        if ev in ("shoot", "draw", "reload"):
            act = self.active.get(key)
            fresh = (act is None or act[0] != ev
                     or (prev_row is not None
                         and not _same_event(row, prev_row, ev)))
            if fresh:
                self.active[key] = (ev, tick, clip)
                st = tick
            else:
                # same instance still running: keep its start AND its clip
                # -- pick() keys on the tick, so re-picking here would swap
                # clips mid-playthrough.
                st, clip = act[1], act[2]
            t = max(0.0, (tick - st) / self.tick_rate)
            why.append(f"t from event start tick {st} (READ)")
        else:
            stride = 75.0        # STATED: source units per full walk cycle
            t = (self.dist.get(key, 0.0) % stride) / stride \
                * float(clip["duration"])
            why.append(f"t is STATED phase from distance travelled "
                       f"{self.dist.get(key, 0.0):.0f} u over a {stride:.0f} u "
                       f"stride -- stride phase is not on the wire")
        self.prev[key] = row
        return ev, clip, float(t), why


def _same_event(row, prev_row, ev):
    if ev == "shoot":
        return bool(prev_row.get("fire"))
    if ev == "draw":
        return row.get("active_weapon_id") == prev_row.get("active_weapon_id")
    if ev == "reload":
        return (row.get("ammo_clip") or 0) <= (prev_row.get("ammo_clip") or 0)
    return True
