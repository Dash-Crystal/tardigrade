# ======================================================================
# 89: per-agent screen-space HUD, drawn by the RENDERER (owner: overlays
# "should be present in the screen space of each individual agent" --
# not composited after the fact by a harness tool nobody runs).
#
# Ported from harness/gpu_render/vector_hud.py (the validated batched
# vector rasterizer). Two schemes, chosen per agent by the hud doc:
#   cs2  crosshair (fire-reactive), scope lines, health bar + ammo pips
#        ONLY when the recording carries those columns -- tard21 carries
#        weapon/fire/scope and NO health/ammo, and that absence is
#        PRINTED once per agent, never substituted (the harness file's
#        own docstring claimed health/ammo were on the tard21 wire;
#        probing the recording falsified that -- nothing is invented).
#   ac   the PSX FCS wireframe animated by the adapter's lock ladder
#        (window, 4096-grid compass, radar ring + spoke, converging
#        bracket, dpad indicator).
#
# The sink calls hud_apply(chunk, agent_index, f0) on the CPU uint8
# frames just before the mux -- the overlay is part of the agent's
# rendered screen space, present in every demovideo by construction.
# ======================================================================

SESSION_HUD = {}          # agent index -> parsed hud doc (lazy, printed)
_HUD_SAID = set()


def _hud_doc(ai):
    if ai in SESSION_HUD:
        return SESSION_HUD[ai]
    doc = None
    if SESSION is not None:
        _ag = SESSION["agents"] if isinstance(SESSION, dict) else SESSION
        p = _ag[ai].get("hud_json") if ai < len(_ag) else None
        if p and os.path.exists(p):
            doc = json.load(open(p))
        elif ai not in _HUD_SAID:
            _HUD_SAID.add(ai)
            print(f"HUD agent {ai}: no hud_json in the session doc -- NO "
                  f"overlay is drawn for this agent and this line is the "
                  f"only record of that; emit one in gen()", flush=True)
    SESSION_HUD[ai] = doc
    return doc


def _hud_draw_segments(frames, segs, color, samples=96):
    """frames (B,H,W,3) uint8 CPU; segs (B,S,4) x0,y0,x1,y1; NaN = skip."""
    B = frames.shape[0]
    t = torch.linspace(0, 1, samples)
    x = segs[..., 0:1] * (1 - t) + segs[..., 2:3] * t
    y = segs[..., 1:2] * (1 - t) + segs[..., 3:4] * t
    ok = torch.isfinite(x) & torch.isfinite(y)
    xi = x.round().long().clamp(0, frames.shape[2] - 1)
    yi = y.round().long().clamp(0, frames.shape[1] - 1)
    fi = torch.arange(B)[:, None, None].expand_as(xi)
    fi, yi, xi = fi[ok], yi[ok], xi[ok]
    frames.index_put_((fi, yi, xi), color.expand(fi.shape[0], 3))
    frames.index_put_((fi, (yi + 1).clamp(max=frames.shape[1] - 1), xi),
                      color.expand(fi.shape[0], 3))
    return frames


def _hud_cs2_rows(rows, f0, B, w, h):
    cx, cy = w / 2.0, h / 2.0
    segs, said_absent = [], False
    for i in range(B):
        r = rows[min(f0 + i, len(rows) - 1)]
        g = 9 + (5 if r.get("fire") else 0)          # fire-reactive gap
        s = [(cx - g, cy, cx - 3, cy), (cx + 3, cy, cx + g, cy),
             (cx, cy - g, cx, cy - 3), (cx, cy + 3, cx, cy + g)]
        if r.get("scope"):
            s += [(0, cy, cx - 40, cy), (cx + 40, cy, w, cy),
                  (cx, 0, cx, cy - 40), (cx, cy + 40, cx, h)]
        hp = r.get("health")
        if isinstance(hp, (int, float)):
            bw = 1.2 * max(0.0, min(100.0, float(hp)))
            s += [(30, h - 30, 30 + bw, h - 30),
                  (30, h - 28, 30 + bw, h - 28),
                  (28, h - 34, 28, h - 24), (152, h - 34, 152, h - 24)]
        elif not said_absent:
            said_absent = True                      # printed by caller once
        am = r.get("ammo")
        if isinstance(am, (int, float)):
            for k in range(int(max(0, min(40, am)))):
                s.append((w - 40 - k * 6, h - 34, w - 40 - k * 6, h - 24))
        segs.append(s)
    S = max(len(s) for s in segs)
    out = torch.full((B, S, 4), float("nan"))
    for i, s in enumerate(segs):
        for j, seg in enumerate(s):
            out[i, j] = torch.tensor(seg, dtype=torch.float)
    return out, said_absent


def _hud_ac_rows(doc, f0, B, w, h):
    cx, cy = w / 2.0, h / 2.0
    lock, tape = doc.get("lock", []), doc.get("turn", [])
    yaws = doc.get("yaw_degrees", [])
    box_w, box_h = 240.0 * w / 960.0, 170.0 * h / 540.0
    segs = []
    for i in range(B):
        fi = f0 + i
        s = []
        x0, y0 = cx - box_w / 2, cy - box_h / 2
        x1, y1 = cx + box_w / 2, cy + box_h / 2
        s += [(x0, y0, x1, y0), (x1, y0, x1, y1),
              (x1, y1, x0, y1), (x0, y1, x0, y0)]
        yaw = yaws[min(fi, len(yaws) - 1)] if yaws else 0.0
        q = round(yaw / (360.0 / 4096.0))
        for k in range(-4, 5):
            tx = cx + ((q * (360.0 / 4096.0)) + k * 11.25 - yaw) * 9.0
            if 40 < tx < w - 40:
                s.append((tx, 26, tx, 26 + (10 if k % 2 == 0 else 5)))
        s.append((40, 24, w - 40, 24))
        rr, rx, ry = 52, w - 78, h - 78
        for a in range(0, 360, 30):
            a0, a1 = math.radians(a), math.radians(a + 30)
            s.append((rx + rr * math.cos(a0), ry + rr * math.sin(a0),
                      rx + rr * math.cos(a1), ry + rr * math.sin(a1)))
        sp = math.radians(-yaw)
        s.append((rx, ry, rx + rr * math.cos(sp), ry + rr * math.sin(sp)))
        st = lock[fi] if fi < len(lock) else 0
        if st > 0 and ((st == 2) or ((fi // 4) % 2 == 0)):
            prog = 1.0 if st == 2 else 0.55
            halfb = 26 + (1.0 - prog) * 30
            gap = 9 + (1.0 - prog) * 8
            for sx in (-1, 1):
                for sy in (-1, 1):
                    bx, by = cx + sx * halfb, cy + sy * halfb
                    s += [(bx, by, bx - sx * gap, by),
                          (bx, by, bx, by - sy * gap)]
            if st == 2:
                s += [(cx - 6, cy, cx + 6, cy), (cx, cy - 6, cx, cy + 6)]
        u = tape[fi] if fi < len(tape) else 0
        if u:
            ax = cx + u * (box_w / 2 + 26)
            s += [(ax, cy - 10, ax + u * 12, cy),
                  (ax, cy + 10, ax + u * 12, cy)]
        segs.append(s)
    S = max(len(s) for s in segs)
    out = torch.full((B, S, 4), float("nan"))
    for i, s in enumerate(segs):
        for j, seg in enumerate(s):
            out[i, j] = torch.tensor(seg, dtype=torch.float)
    return out


HUD_GREEN = torch.tensor([90, 255, 120], dtype=torch.uint8)
HUD_AMBER = torch.tensor([255, 200, 80], dtype=torch.uint8)


def hud_apply(chunk, ai, f0):
    """chunk (B,H,W,3) uint8 CPU -> overlaid in place; returns chunk."""
    doc = _hud_doc(ai)
    if doc is None:
        return chunk
    h, w = chunk.shape[1], chunk.shape[2]
    if doc.get("scheme") == "ac":
        return _hud_draw_segments(chunk, _hud_ac_rows(doc, f0,
                                                      chunk.shape[0], w, h),
                                  HUD_AMBER)
    segs, absent = _hud_cs2_rows(doc.get("rows", []), f0,
                                 chunk.shape[0], w, h)
    key = ("absent", ai)
    if absent and key not in _HUD_SAID:
        _HUD_SAID.add(key)
        print(f"HUD agent {ai}: recording carries NO health/ammo columns "
              f"-- bar and pips are not drawn and not invented; crosshair/"
              f"fire/scope are drawn from the columns that exist",
              flush=True)
    return _hud_draw_segments(chunk, segs, HUD_GREEN)
