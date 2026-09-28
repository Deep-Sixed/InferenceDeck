"""HTTP handler for the InferenceDeck control API.

Served by ``inferencedeck-web`` (see webui.py), the single process that owns
server state. Browsers and both trays talk to that one process.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

from .api_params import validate_overrides
from .auth import SESSION_TTL_SECONDS, AuthState, host_header_ok, request_client
from .control import ControlPlane

MAX_BODY_BYTES = 1024 * 1024
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _bounded_int(body: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    """An integer field clamped to [low, high]; anything else is a 400, not a 500."""
    value = body.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None
    return max(low, min(number, high))


class ControlRequestHandler(BaseHTTPRequestHandler):
    control_plane = ControlPlane()
    auth_state = AuthState()
    server_version = "InferenceDeckControl/1"
    # Set when served over TLS, so the session cookie is never sent over plain HTTP.
    secure_cookies = False

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _cookie(self, name: str) -> str:
        for part in self.headers.get("Cookie", "").split(";"):
            key, _, value = part.strip().partition("=")
            if key == name:
                return value
        return ""

    def _host_ok(self) -> bool:
        # Without auth the API only binds to loopback. Reject any other Host
        # header so a DNS-rebinding page can't read or drive the local API.
        return host_header_ok(self.headers, self.auth_state)

    def _cookie_attrs(self) -> str:
        return "HttpOnly; SameSite=Strict; Path=/" + ("; Secure" if self.secure_cookies else "")

    def _client(self) -> str:
        return request_client(self)

    def _throttled(self, retry_after: int) -> None:
        self._json(
            HTTPStatus.TOO_MANY_REQUESTS,
            {"success": False, "error": f"too many failed attempts; retry in {retry_after}s"},
            headers={"Retry-After": str(retry_after)},
        )

    def _require_auth(self) -> bool:
        """True when the request may proceed; otherwise the error response has been sent."""
        if not self.auth_state.enabled or self.auth_state.session_ok(self._cookie("sid")):
            return True
        client = self._client()
        wait = self.auth_state.retry_after(client)
        if wait:
            self._throttled(wait)
            return False
        if self.headers.get("X-Auth-Token") or self.headers.get("Authorization"):
            if self.auth_state.supplied_token_ok(self.headers):
                self.auth_state.record_success(client)
                return True
            # A wrong token is a failed guess, same as a wrong login password.
            self.auth_state.record_failure(client)
        self._json(HTTPStatus.UNAUTHORIZED, {"success": False, "error": "unauthorized"})
        return False

    def _json(
        self,
        status: int,
        payload: Any,
        cookies: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        for cookie in cookies or []:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _text(self, status: int, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

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
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"success": False, "error": "host not allowed"})
            return
        if parsed.path == "/healthz":
            self._json(HTTPStatus.OK, {"ok": True})
            return
        if parsed.path == "/api/auth":
            # Not the login name: that would hand an unauthenticated caller half the credentials.
            self._json(HTTPStatus.OK, {"required": self.auth_state.enabled})
            return
        if not self._require_auth():
            return
        try:
            if parsed.path == "/api/status":
                self._json(HTTPStatus.OK, self.control_plane.status())
            elif parsed.path == "/api/inventory":
                self._json(HTTPStatus.OK, self.control_plane.inventory())
            elif parsed.path == "/api/profiles":
                self._json(HTTPStatus.OK, {"profiles": self.control_plane.profiles()})
            elif parsed.path == "/api/hardware":
                self._json(HTTPStatus.OK, self.control_plane.hardware())
            elif parsed.path == "/api/benchmarks":
                self._json(HTTPStatus.OK, {"benchmarks": self.control_plane.benchmark_history()})
            elif parsed.path == "/api/remotes":
                self._json(HTTPStatus.OK, self.control_plane.remote_endpoints())
            elif parsed.path == "/api/runtime":
                self._json(HTTPStatus.OK, self.control_plane.runtime())
            elif parsed.path == "/api/telemetry":
                self._json(HTTPStatus.OK, self.control_plane.telemetry())
            elif parsed.path == "/api/fleet":
                self._json(HTTPStatus.OK, self.control_plane.fleet())
            elif parsed.path == "/api/fleet/placement":
                try:
                    payload = self.control_plane.fleet_placement(
                        profile=(query.get("profile") or [""])[0], model=(query.get("model") or [""])[0]
                    )
                except ValueError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)})
                    return
                self._json(HTTPStatus.OK, payload)
            elif parsed.path == "/api/telemetry/history":
                range_name = (query.get("range") or ["1h"])[0]
                try:
                    payload = self.control_plane.telemetry_history(range_name)
                except ValueError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)})
                    return
                self._json(HTTPStatus.OK, payload)
            elif parsed.path == "/metrics":
                self._text(HTTPStatus.OK, self.control_plane.metrics(), PROMETHEUS_CONTENT_TYPE)
            elif parsed.path == "/api/hf/files":
                self._json(HTTPStatus.OK, self.control_plane.hf_files(str((query.get("repo_id") or [""])[0])))
            elif parsed.path == "/api/updates":
                # Cached answers only. Asking GitHub again is POST /api/updates: a
                # GET can be fired by any web page (an <img> tag), and each forced
                # check spends this machine's GitHub API rate limit.
                self._json(HTTPStatus.OK, self.control_plane.updates(refresh=False))
            elif parsed.path == "/api/logs":
                server_id = (query.get("server_id") or [""])[0]
                if not server_id:
                    self._json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "server_id is required"})
                    return
                try:
                    lines = int((query.get("lines") or ["200"])[0])
                except ValueError:
                    self._json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "lines must be an integer"})
                    return
                self._json(HTTPStatus.OK, self.control_plane.logs(server_id, lines=max(1, min(lines, 2000))))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"success": False, "error": "not found"})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"success": False, "error": "host not allowed"})
            return
        # Browsers can send text/plain cross-origin without a CORS preflight;
        # requiring application/json forces one, which this server never grants.
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"success": False, "error": "Content-Type must be application/json"})
            return
        try:
            body = self._body()
            if parsed.path == "/api/login":
                if not self.auth_state.enabled:
                    self._json(HTTPStatus.OK, {"success": True})
                    return
                client = self._client()
                wait = self.auth_state.retry_after(client)
                if wait:
                    self._throttled(wait)
                    return
                if not self.auth_state.credentials_ok(str(body.get("username", "")), str(body.get("password", ""))):
                    self.auth_state.record_failure(client)
                    self._json(HTTPStatus.UNAUTHORIZED, {"success": False, "error": "invalid credentials"})
                    return
                self.auth_state.record_success(client)
                sid = self.auth_state.issue_session()
                self._json(HTTPStatus.OK, {"success": True}, cookies=[f"sid={sid}; {self._cookie_attrs()}; Max-Age={SESSION_TTL_SECONDS}"])
                return
            if parsed.path == "/api/logout":
                self.auth_state.revoke_session(self._cookie("sid"))
                self._json(HTTPStatus.OK, {"success": True}, cookies=[f"sid=; {self._cookie_attrs()}; Max-Age=0"])
                return
            if not self._require_auth():
                return
            if parsed.path == "/api/prepare":
                mode = str(body.get("mode") or "")
                payload = self.control_plane.prepare(mode, validate_overrides(body.get("overrides")))
            elif parsed.path == "/api/start":
                mode = str(body.get("mode") or "")
                payload = self.control_plane.start(
                    mode,
                    validate_overrides(body.get("overrides")),
                    stop_existing=bool(body.get("stop_existing", False)),
                )
            elif parsed.path == "/api/stop":
                payload = self.control_plane.stop(server_id=body.get("server_id"), mode=body.get("mode"))
            elif parsed.path == "/api/suspend":
                payload = self.control_plane.suspend(server_id=body.get("server_id"), mode=body.get("mode"))
            elif parsed.path in ("/api/release", "/api/restore", "/api/restart"):
                server_id = str(body.get("server_id") or "")
                if not server_id:
                    raise ValueError("server_id is required")
                if parsed.path == "/api/release":
                    payload = self.control_plane.release_gpu(server_id=server_id)
                else:
                    # Only the context size may change on restore/restart; the rest of
                    # the saved spec is reused as it was.
                    extra = validate_overrides({"ctx_size": body["ctx_size"]}) if body.get("ctx_size") is not None else None
                    action = self.control_plane.restore if parsed.path == "/api/restore" else self.control_plane.restart
                    payload = action(server_id, extra)
            elif parsed.path == "/api/resume":
                payload = self.control_plane.resume(server_id=body.get("server_id"), mode=body.get("mode"))
            elif parsed.path == "/api/fit":
                payload = self.control_plane.fit(
                    str(body.get("mode") or ""),
                    validate_overrides(body.get("overrides")),
                    target_mib=_bounded_int(body, "target_mib", 1024, 0, 65536),
                )
            elif parsed.path == "/api/benchmark":
                payload = self.control_plane.benchmark(
                    str(body.get("mode") or ""),
                    validate_overrides(body.get("overrides")),
                    completion_tokens=_bounded_int(body, "completion_tokens", 128, 16, 2048),
                )
            elif parsed.path == "/api/hf/download":
                # Always into the HF cache: the API never picks a destination path.
                payload = self.control_plane.hf_download(
                    str(body.get("repo_id") or ""),
                    quant=str(body.get("quant") or "") or None,
                    pattern=str(body.get("pattern") or "") or None,
                    include_mmproj=bool(body.get("include_mmproj", True)),
                    dry_run=bool(body.get("dry_run", False)),
                )
            elif parsed.path == "/api/updates":
                payload = self.control_plane.updates(refresh=True)
            elif parsed.path == "/api/runtime":
                payload = self.control_plane.set_runtime(str(body.get("runtime") or ""))
            elif parsed.path == "/api/remote":
                action = str(body.get("action") or "")
                if action == "enable":
                    payload = self.control_plane.enable_remote(str(body.get("name") or ""))
                elif action == "disable":
                    payload = self.control_plane.disable_remotes()
                else:
                    payload = {"success": False, "error": "action must be enable or disable"}
            else:
                self._json(HTTPStatus.NOT_FOUND, {"success": False, "error": "not found"})
                return
            code = HTTPStatus.OK if payload.get("success", True) else HTTPStatus.BAD_REQUEST
            self._json(code, payload)
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(exc)})
