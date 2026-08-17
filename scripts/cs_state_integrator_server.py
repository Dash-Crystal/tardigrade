#!/usr/bin/env python3
"""Serve the renderer-facing state integrator over bounded JSON/HTTP.

This process does not attach to, inspect, inject into, or control CS2.  It only
accepts protocol documents emitted by a capture/parser process and writes its
own caller-selected artifact directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from state_replay import (  # noqa: E402
    ConflictError,
    NotFoundError,
    ProtocolError,
    StateIntegrator,
)

DEFAULT_MAX_BODY = 8 * 1024 * 1024
DEFAULT_MAX_BATCH = 4096


class IntegratorRegistry:
    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root
        self._streams: dict[str, StateIntegrator] = {}
        self._lock = threading.RLock()

    def get(self, stream_id: str) -> StateIntegrator:
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = StateIntegrator.open(
                    self.storage_root, stream_id)
            return self._streams[stream_id]

    def create(self, manifest: dict, checkpoint: dict) -> StateIntegrator:
        with self._lock:
            integrator = StateIntegrator.create(
                self.storage_root, manifest, checkpoint)
            self._streams[integrator.stream_id] = integrator
            return integrator


def make_handler(registry: IntegratorRegistry, max_body: int,
                 max_batch: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TardigradeStateIntegrator/1"

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write("state-integrator: " + fmt % args + "\n")

        def _json(self, status: int, payload: object) -> None:
            encoded = json.dumps(payload, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"ok": False, "error": code,
                                "message": message})

        def _body(self) -> dict:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ProtocolError("Content-Length is required")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ProtocolError("invalid Content-Length") from exc
            if length < 0 or length > max_body:
                raise OverflowError(
                    f"request body exceeds configured {max_body}-byte limit")
            raw = self.rfile.read(length)
            try:
                doc = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProtocolError(f"invalid JSON body: {exc}") from exc
            if not isinstance(doc, dict):
                raise ProtocolError("request body must be a JSON object")
            return doc

        @staticmethod
        def _one(query: dict[str, list[str]], name: str,
                 default: str | None = None) -> str:
            values = query.get(name)
            if not values:
                if default is not None:
                    return default
                raise ProtocolError(f"missing query parameter {name}")
            if len(values) != 1:
                raise ProtocolError(f"query parameter {name} must occur once")
            return values[0]

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if parsed.path == "/health":
                    self._json(HTTPStatus.OK, {
                        "ok": True,
                        "service": "tardigrade-state-integrator",
                        "schema": "tardigrade/state-snapshot/v1",
                    })
                    return
                stream_id = self._one(query, "stream_id")
                integrator = registry.get(stream_id)
                if parsed.path == "/state":
                    tick = int(self._one(query, "tick"))
                    subtick = int(self._one(query, "subtick", str(2**31 - 1)))
                    self._json(HTTPStatus.OK, integrator.state_at(tick, subtick))
                elif parsed.path == "/action-context":
                    action_id = self._one(query, "action_id")
                    self._json(HTTPStatus.OK,
                               integrator.action_context(action_id))
                elif parsed.path == "/stats":
                    self._json(HTTPStatus.OK, integrator.stats())
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not_found",
                                "unknown endpoint")
            except ValueError as exc:
                self._handle_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = self._body()
                if parsed.path == "/manifest":
                    integrator = registry.create(body.get("manifest"),
                                                 body.get("checkpoint"))
                    self._json(HTTPStatus.CREATED, {
                        "ok": True, "snapshot": integrator.current_snapshot()})
                elif parsed.path == "/ingest":
                    integrator = registry.get(body.get("stream_id"))
                    if "transactions" in body:
                        txs = body["transactions"]
                        if not isinstance(txs, list):
                            raise ProtocolError("transactions must be an array")
                        if len(txs) > max_batch:
                            raise OverflowError(
                                f"transaction batch exceeds {max_batch} items")
                        snapshots = integrator.ingest_many(txs)
                    else:
                        snapshots = [integrator.ingest(body.get("transaction"))]
                    self._json(HTTPStatus.OK, {
                        "ok": True, "snapshots": snapshots,
                        "current": integrator.current_snapshot()})
                elif parsed.path == "/checkpoint":
                    integrator = registry.get(body.get("stream_id"))
                    checkpoint = integrator.checkpoint(body.get("checkpoint"))
                    self._json(HTTPStatus.CREATED,
                               {"ok": True, "checkpoint": checkpoint})
                elif parsed.path == "/batch":
                    integrator = registry.get(body.get("stream_id"))
                    positions = body.get("positions")
                    if not isinstance(positions, list):
                        raise ProtocolError("positions must be an array")
                    if len(positions) > max_batch:
                        raise OverflowError(
                            f"state batch exceeds {max_batch} positions")
                    self._json(HTTPStatus.OK, {
                        "ok": True,
                        "snapshots": integrator.batch_states(positions),
                    })
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not_found",
                                "unknown endpoint")
            except OverflowError as exc:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            "request_too_large", str(exc))
            except (ProtocolError, ValueError, TypeError) as exc:
                self._handle_exception(exc)

        def _handle_exception(self, exc: Exception) -> None:
            if isinstance(exc, NotFoundError):
                self._error(HTTPStatus.NOT_FOUND, "not_found", str(exc))
            elif isinstance(exc, ConflictError):
                self._error(HTTPStatus.CONFLICT, "history_conflict", str(exc))
            else:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent Tardigrade state-integrator HTTP service")
    parser.add_argument("--storage-root", required=True,
                        help="caller-owned artifact directory (never source tree)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY)
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    args = parser.parse_args()
    if args.max_body_bytes < 1024 or args.max_batch < 1:
        parser.error("body and batch limits must be positive")
    root = Path(args.storage_root).expanduser().resolve()
    # Deployment must make the location explicit.  Refuse the checkout itself
    # because journals and checkpoints are run artifacts, not source.
    if root == _ROOT or _ROOT in root.parents:
        parser.error("--storage-root must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=True)
    registry = IntegratorRegistry(root)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(registry, args.max_body_bytes, args.max_batch))
    print(f"state-integrator listening on http://{args.host}:{args.port} "
          f"storage={root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
