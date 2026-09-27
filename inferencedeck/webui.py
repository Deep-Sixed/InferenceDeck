from __future__ import annotations

import argparse
import ssl
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from .auth import AuthState, validate_bind_security
from .config import AppConfig
from .control import ControlPlane
from .control_api import ControlRequestHandler
from .telemetry_history import start_sampler


ASSET_TYPES = {
    "/app.js": "text/javascript; charset=utf-8",
    "/styles.css": "text/css; charset=utf-8",
    "/telemetry.js": "text/javascript; charset=utf-8",
}


class WebRequestHandler(ControlRequestHandler):
    def _asset(self, path: str, content_type: str) -> None:
        asset = files("inferencedeck.web").joinpath(path)
        body = asset.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        clean = self.path.split("?", 1)[0]
        if clean == "/":
            self._asset("index.html", "text/html; charset=utf-8")
            return
        if clean in ASSET_TYPES:
            self._asset(clean.lstrip("/"), ASSET_TYPES[clean])
            return
        super().do_GET()


def make_server(
    host: str = "127.0.0.1",
    port: int = 8716,
    control_plane: ControlPlane | None = None,
    auth_state: AuthState | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
) -> ThreadingHTTPServer:
    """Build the server; with ``certfile`` it speaks HTTPS and marks the session cookie Secure."""
    auth = auth_state or AuthState()
    validate_bind_security(host, auth)
    handler = type("BoundWebRequestHandler", (WebRequestHandler,), {})
    handler.control_plane = control_plane or ControlPlane()
    handler.auth_state = auth
    handler.secure_cookies = bool(certfile)
    server = ThreadingHTTPServer((host, port), handler)
    if certfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile, keyfile or None)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 8716,
    control_plane: ControlPlane | None = None,
    auth_state: AuthState | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
) -> None:
    server = make_server(host, port, control_plane, auth_state, certfile, keyfile)
    config = AppConfig.load()
    start_sampler(config.telemetry_sample_seconds, config.telemetry_retention_days)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="InferenceDeck local web control panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8716)
    parser.add_argument("--certfile", help="PEM certificate (with chain) to serve HTTPS; recommended for LAN/tailnet binds")
    parser.add_argument("--keyfile", help="PEM private key, if not included in --certfile")
    args = parser.parse_args()
    if args.keyfile and not args.certfile:
        parser.error("--keyfile needs --certfile")
    serve(args.host, args.port, certfile=args.certfile, keyfile=args.keyfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
