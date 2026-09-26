from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .control import ControlPlane

MAX_BODY_BYTES = 1024 * 1024


class ControlRequestHandler(BaseHTTPRequestHandler):
    control_plane = ControlPlane()
    server_version = "InferenceDeckControl/1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        if not length:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/status":
                self._json(HTTPStatus.OK, self.control_plane.status())
            elif parsed.path == "/api/inventory":
                self._json(HTTPStatus.OK, self.control_plane.inventory())
            elif parsed.path == "/api/profiles":
                self._json(HTTPStatus.OK, {"profiles": self.control_plane.profiles()})
            elif parsed.path == "/api/logs":
                server_id = (query.get("server_id") or [""])[0]
                if not server_id:
                    self._json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "server_id is required"})
                    return
                lines = int((query.get("lines") or ["200"])[0])
                self._json(HTTPStatus.OK, self.control_plane.logs(server_id, lines=max(1, min(lines, 2000))))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"success": False, "error": "not found"})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/prepare":
                mode = str(body.get("mode") or "")
                payload = self.control_plane.prepare(mode, body.get("overrides"))
            elif parsed.path == "/api/start":
                mode = str(body.get("mode") or "")
                payload = self.control_plane.start(
                    mode,
                    body.get("overrides"),
                    stop_existing=bool(body.get("stop_existing", False)),
                )
            elif parsed.path == "/api/stop":
                payload = self.control_plane.stop(server_id=body.get("server_id"), mode=body.get("mode"))
            elif parsed.path == "/api/suspend":
                payload = self.control_plane.suspend(server_id=body.get("server_id"), mode=body.get("mode"))
            elif parsed.path == "/api/resume":
                payload = self.control_plane.resume(server_id=body.get("server_id"), mode=body.get("mode"))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"success": False, "error": "not found"})
                return
            code = HTTPStatus.OK if payload.get("success", True) else HTTPStatus.BAD_REQUEST
            self._json(code, payload)
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(exc)})


def serve(host: str = "127.0.0.1", port: int = 8717, control_plane: ControlPlane | None = None) -> None:
    handler = type("BoundControlRequestHandler", (ControlRequestHandler,), {})
    handler.control_plane = control_plane or ControlPlane()
    ThreadingHTTPServer((host, port), handler).serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="InferenceDeck local frontend/control API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8717)
    args = parser.parse_args()
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
