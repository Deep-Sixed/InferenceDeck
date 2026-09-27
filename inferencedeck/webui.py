from __future__ import annotations

import argparse
import ssl
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from .auth import AuthState, validate_bind_security
from .control import ControlPlane
from .control_api import ControlRequestHandler


ASSET_TYPES = {
    "/app.js": "text/javascript; charset=utf-8",
    "/styles.css": "text/css; charset=utf-8",
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
    *,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    allow_insecure_http: bool = False,
) -> ThreadingHTTPServer:
    if bool(tls_cert) != bool(tls_key):
        raise RuntimeError("--tls-cert and --tls-key must be given together.")
    auth = auth_state or AuthState()
    tls = bool(tls_cert)
    validate_bind_security(host, auth, tls=tls, allow_insecure_http=allow_insecure_http)
    handler = type("BoundWebRequestHandler", (WebRequestHandler,), {})
    handler.control_plane = control_plane or ControlPlane()
    handler.auth_state = auth
    handler.secure_cookies = tls
    server = ThreadingHTTPServer((host, port), handler)
    if tls:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(tls_cert, tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 8716,
    control_plane: ControlPlane | None = None,
    auth_state: AuthState | None = None,
    **transport: Any,
) -> None:
    make_server(host, port, control_plane, auth_state, **transport).serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="InferenceDeck web UI and control API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8716)
    parser.add_argument("--tls-cert", help="PEM certificate: serve HTTPS (needed off loopback unless on Tailscale)")
    parser.add_argument("--tls-key", help="PEM private key for --tls-cert")
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="allow a plain-HTTP non-loopback bind (credentials travel unencrypted)",
    )
    args = parser.parse_args()
    serve(
        args.host,
        args.port,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        allow_insecure_http=args.allow_insecure_http,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
