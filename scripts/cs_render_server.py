"""A boring REST HTTP server that renders demofiles through three view matrices.

    GET  /health                 -> {"ok": true, "views": [...]}
    POST /render  {json body}    -> render_match(...) manifest

Body fields (all but logstream+world optional):
    logstream : path to a cs-tard21/tard3 .tard on the render node
    world     : path to the world pack (.pt)
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
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cs_render_service as svc

OUT_ROOT = "/tmp/cs_render_out"


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
            self._send(200, {"ok": True, "views": list(svc.VIEWS)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/render":
            self._send(404, {"error": "POST /render only"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if "logstream" not in req or "world" not in req:
                raise ValueError("logstream and world are required")
            import os
            out_dir = os.path.join(OUT_ROOT, str(abs(hash(json.dumps(req, sort_keys=True))) % 10**9))
            views = tuple(req.get("views", svc.VIEWS))
            players = req.get("players", "all")
            result = svc.render_match(
                req["logstream"], req["world"], out_dir, views, players,
                int(req.get("tick_begin", 0)), req.get("tick_end"),
                int(req.get("stride", 16)),
                int(req.get("width", 640)), int(req.get("height", 360)))
            self._send(200, result)
        except Exception as exc:  # noqa: BLE001 -- report as JSON, don't 500-crash
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8815)
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()
    OUT_ROOT = a.out
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"cs_render_server on http://{a.host}:{a.port}  (POST /render, GET /health)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
