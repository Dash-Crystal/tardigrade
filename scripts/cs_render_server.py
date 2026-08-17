"""REST boundary for explicitly selected CS2 or legacy CS:GO rendering.

    GET  /health                 -> {"ok": true, "views": [...]}
    POST /render  {json body}    -> render_match(...) manifest

Body fields:
    profile   : canonical render-profile JSON on the render node
    game_variant : optional cross-check; otherwise inherited from profile
    logstream : required only for cs2 (cs-tard21/tard3 .tard)
    snapshot  : required only for csgo_legacy (normalized state JSON)
    views     : ["ego","gow","sm64"]   (default all three)
    players   : "all" or [indices]
    stride    : tick decimation (default 16)
    width, height, tick_end

Runs ON the render node (needs torch+nvdiffrast+the ported renderer). stdlib
only -- no framework. Start:
    python cs_render_server.py --host 127.0.0.1 --port 8815 --out /tmp/renders
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cs_render_service as svc
from variant_config import public_variant_descriptors, resolve_profile_request_variant

OUT_ROOT = "/tmp/cs_render_out"
_REQUEST_FIELDS = {
    "logstream", "snapshot", "profile", "views", "players", "tick_begin", "tick_end",
    "stride", "width", "height", "game_variant",
}


def normalize_request(req):
    """Validate and canonicalize a wire request before assigning a job."""
    if not isinstance(req, dict):
        raise ValueError("request body must be a JSON object")
    if "world" in req:
        raise ValueError(
            "the 'world' request field is no longer accepted: create a "
            "canonical profile JSON containing --world and every other "
            "required asset flag, then send its path as 'profile'")
    unknown = sorted(set(req) - _REQUEST_FIELDS)
    if unknown:
        raise ValueError(f"unknown request fields: {', '.join(unknown)}")
    if not isinstance(req.get("profile"), str) or not req["profile"].strip():
        raise ValueError("profile is required and must be a path string")

    profile_variant = svc.profile_game_variant(req["profile"])
    selected_variant = resolve_profile_request_variant(
        profile_variant, req.get("game_variant")
    )
    normalized = {
        "profile": os.path.abspath(req["profile"]),
        "game_variant": selected_variant.id,
        "width": req.get("width", 640),
        "height": req.get("height", 360),
    }
    if selected_variant.id == "cs2":
        if not isinstance(req.get("logstream"), str) or not req["logstream"].strip():
            raise ValueError("cs2 requires logstream as a path string")
        if "snapshot" in req:
            raise ValueError("cs2 does not accept snapshot; use logstream")
        normalized.update({
            "logstream": os.path.abspath(req["logstream"]),
            "views": req.get("views", list(svc.VIEWS)),
            "players": req.get("players", "all"),
            "tick_begin": req.get("tick_begin", 0),
            "tick_end": req.get("tick_end"),
            "stride": req.get("stride", 16),
        })
        views, players = svc._validate_render_options(
            normalized["views"], normalized["players"],
            normalized["tick_begin"], normalized["tick_end"],
            normalized["stride"], normalized["width"], normalized["height"])
        normalized["views"] = list(views)
        normalized["players"] = players
    else:
        if not isinstance(req.get("snapshot"), str) or not req["snapshot"].strip():
            raise ValueError("csgo_legacy requires snapshot as a path string")
        if "logstream" in req:
            raise ValueError("csgo_legacy does not accept logstream; use snapshot")
        inapplicable = sorted(
            set(req).intersection({"views", "players", "tick_begin", "tick_end", "stride"})
        )
        if inapplicable:
            raise ValueError(
                "csgo_legacy snapshot request does not accept CS2 TARD options: "
                + ", ".join(inapplicable)
            )
        normalized["snapshot"] = os.path.abspath(req["snapshot"])
        for name in ("width", "height"):
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16384:
                raise ValueError(f"{name} must be an integer in 1..16384")
    return normalized


def job_identity(req):
    """Stable across processes, hosts, and PYTHONHASHSEED values."""
    wire = json.dumps(req, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(wire).hexdigest()


def job_output_dir(out_root, job_id):
    """Return a job directory that cannot escape the configured root."""
    root = os.path.realpath(os.path.abspath(out_root))
    os.makedirs(root, exist_ok=True)
    if not (len(job_id) == 64 and
            all(ch in "0123456789abcdef" for ch in job_id)):
        raise ValueError("invalid job identity")
    candidate = os.path.join(root, job_id)
    if os.path.lexists(candidate) and os.path.islink(candidate):
        raise RuntimeError(f"refusing symlinked render job directory: {candidate}")
    os.makedirs(candidate, exist_ok=True)
    resolved = os.path.realpath(candidate)
    if os.path.commonpath((root, resolved)) != root:
        raise RuntimeError("render job directory escaped configured output root")
    return resolved


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send(200, {
                "ok": True,
                "views": list(svc.VIEWS),
                "game_variants": public_variant_descriptors(),
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/render":
            self._send(404, {"error": "POST /render only"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n < 0 or n > 1024 * 1024:
                raise ValueError("request body must be at most 1 MiB")
            req = json.loads(self.rfile.read(n) or b"{}")
            req = normalize_request(req)
            out_dir = job_output_dir(OUT_ROOT, job_identity(req))
            if req["game_variant"] == "cs2":
                result = svc.render_match(
                    req["logstream"], req["profile"], out_dir, req["views"],
                    req["players"], req["tick_begin"], req["tick_end"],
                    req["stride"], req["width"], req["height"],
                    req["game_variant"])
            else:
                result = svc.render_snapshot(
                    req["snapshot"], req["profile"], out_dir,
                    req["width"], req["height"], req["game_variant"])
            result["job_id"] = os.path.basename(out_dir)
            self._send(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
        except svc.RenderRefusal as exc:
            self._send(422, {"error": f"RenderRefusal: {exc}"})
        except Exception as exc:  # noqa: BLE001 -- isolate request failures
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8815)
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()
    OUT_ROOT = os.path.realpath(os.path.abspath(a.out))
    os.makedirs(OUT_ROOT, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"cs_render_server on http://{a.host}:{a.port}  (POST /render, GET /health)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
