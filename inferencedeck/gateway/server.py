"""``inferencedeck-gateway``: one stable inference API in front of whatever is active.

Endpoints:
  POST /v1/chat/completions  OpenAI Chat Completions
  POST /v1/messages          Anthropic Messages
  GET  /v1/models            every model name the gateway can route (see router.py),
                             plus loadable profiles when switching is on
  GET  /healthz

A request's ``model`` picks the target it names (see router.py); ``--switch-models``
(or ``gateway_model_switching`` in config) also loads a named profile on demand.

Binding follows the control API's rule: loopback without a token, anything
else only with INFERENCEDECK_TOKEN set. Clients present the token the way their
SDK sends API keys (``Authorization: Bearer``, ``x-api-key``) or as
``X-Auth-Token``.
"""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from ..auth import AuthState, host_header_ok, request_client, validate_bind_security
from ..config import AppConfig
from . import anthropic_api, openai_api
from .ir import ChatRequest, GatewayError
from .router import Router
from .switching import ModelSwitcher

DEFAULT_PORT = 8717
MAX_BODY_BYTES = 32 * 1024 * 1024  # room for inline images


class _Api:
    """How one public API parses requests and renders replies."""

    def __init__(self, parse, render_result, render_stream, render_error) -> None:
        self.parse = parse
        self.render_result = render_result
        self.render_stream = render_stream
        self.render_error = render_error


APIS = {
    "/v1/chat/completions": _Api(
        openai_api.parse_request,
        openai_api.render_result,
        lambda events, model, body: openai_api.render_stream(events, model, openai_api.wants_usage_in_stream(body)),
        openai_api.render_error,
    ),
    "/v1/messages": _Api(
        anthropic_api.parse_request,
        anthropic_api.render_result,
        lambda events, model, body: anthropic_api.render_stream(events, model),
        anthropic_api.render_error,
    ),
}


class GatewayRequestHandler(BaseHTTPRequestHandler):
    auth_state = AuthState()
    router: Router = Router()
    server_version = "InferenceDeckGateway/1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _host_ok(self) -> bool:
        # Same DNS-rebinding guard as the control API.
        return host_header_ok(self.headers, self.auth_state)

    def _token_ok(self) -> bool:
        if not self.auth_state.enabled:
            return True
        bearer = self.headers.get("Authorization", "")
        supplied = (
            bearer[7:].strip() if bearer.lower().startswith("bearer ") else ""
        ) or self.headers.get("x-api-key", "") or self.headers.get("X-Auth-Token", "")
        return bool(supplied) and secrets.compare_digest(supplied, self.auth_state.token)

    def _client(self) -> str:
        # Same throttle key as the control API: X-Forwarded-For only from a trusted proxy.
        return request_client(self)

    def _send(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, chunks: Iterator[bytes], lease: Any = None) -> None:
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away; closing the generator closes the upstream
        except Exception:
            # The 200 and part of the body are already on the wire, so nothing
            # may escape to do_POST's error handler: it would write a second
            # status line into the stream. render_stream reports upstream
            # failures in-band; anything else just ends the stream here.
            self.close_connection = True
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
            if lease is not None:
                lease.__exit__(None, None, None)

    def _guard(self, render_error: Callable[[GatewayError], Any]) -> bool:
        if not self._host_ok():
            self._send(HTTPStatus.FORBIDDEN, render_error(GatewayError(403, "host not allowed", "authentication")))
            return False
        client = self._client()
        wait = self.auth_state.retry_after(client)
        if wait:
            self._send(HTTPStatus.TOO_MANY_REQUESTS,
                       render_error(GatewayError(429, f"too many failed attempts; retry in {wait}s", "rate_limit")),
                       headers={"Retry-After": str(wait)})
            return False
        if not self._token_ok():
            self.auth_state.record_failure(client)
            self._send(HTTPStatus.UNAUTHORIZED, render_error(GatewayError(401, "invalid or missing token", "authentication")))
            return False
        if self.auth_state.enabled:
            # Earlier typos (a stale key during setup) shouldn't linger and
            # lock out a client that now authenticates.
            self.auth_state.record_success(client)
        return True

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise GatewayError(400, "invalid Content-Length", "invalid_request") from exc
        if length <= 0:
            raise GatewayError(400, "request body is required", "invalid_request")
        if length > MAX_BODY_BYTES:
            raise GatewayError(413, "request body too large", "invalid_request")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise GatewayError(400, "invalid JSON", "invalid_request") from exc
        if not isinstance(value, dict):
            raise GatewayError(400, "JSON body must be an object", "invalid_request")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/healthz" and self._host_ok():
            self._send(HTTPStatus.OK, {"ok": True})
            return
        if not self._guard(openai_api.render_error):
            return
        if path == "/v1/models":
            try:
                targets = self.router.catalog()
            except GatewayError:
                targets = []
            loadable = getattr(self.router, "loadable", lambda: [])()
            self._send(HTTPStatus.OK, {"object": "list", "data": [
                {"id": t.model_id, "object": "model", "owned_by": "inferencedeck", "description": t.label,
                 "aliases": [name for name in t.names if name != t.model_id], "default": t.default, "loaded": True}
                for t in targets
            ] + [
                # With switching on: profiles a request can name to load them.
                {"id": str(p["mode"]), "object": "model", "owned_by": "inferencedeck",
                 "description": str(p.get("name") or p["mode"]), "aliases": [], "default": False, "loaded": False}
                for p in loadable
            ]})
        else:
            self._send(HTTPStatus.NOT_FOUND, openai_api.render_error(GatewayError(404, "not found", "not_found")))

    def do_POST(self) -> None:
        api = APIS.get(urlparse(self.path).path)
        if api is None:
            if self._guard(openai_api.render_error):
                self._send(HTTPStatus.NOT_FOUND, openai_api.render_error(GatewayError(404, "not found", "not_found")))
            return
        if not self._guard(api.render_error):
            return
        # Browsers send text/plain (or form) POSTs cross-origin without a CORS
        # preflight, so without this any web page could make a loopback gateway
        # run inference, spending GPU time or a cloud endpoint's API credits.
        # SDKs always send application/json; requiring it forces a preflight,
        # which this server never grants.
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, api.render_error(
                GatewayError(415, "Content-Type must be application/json", "invalid_request")))
            return
        try:
            body = self._body()
            request: ChatRequest = api.parse(body)
            target = self.router.resolve(request.model)
            model = request.model or target.model_id
            # The lease tells a model switch this server is still busy; for a
            # stream it is held until the last chunk is written.
            lease = target.lease()
            lease.__enter__()
            try:
                if request.stream:
                    events = target.engine.stream(request)
                    stream_lease, lease = lease, None
                    self._send_stream(api.render_stream(events, model, body), stream_lease)
                else:
                    result = target.engine.complete(request)
                    # Report the model the client asked for, so aliases stay stable.
                    result.model = model
                    self._send(HTTPStatus.OK, api.render_result(result))
            finally:
                if lease is not None:
                    lease.__exit__(None, None, None)
        except GatewayError as exc:
            self._send(exc.status, api.render_error(exc))
        except Exception as exc:  # pragma: no cover - last-resort guard
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, api.render_error(GatewayError(500, str(exc))))


def make_server(host: str, port: int, auth_state: AuthState | None = None,
                router: Router | None = None) -> ThreadingHTTPServer:
    auth = auth_state or AuthState()
    validate_bind_security(host, auth)
    attrs: dict[str, Any] = {"auth_state": auth}
    if router is not None:
        attrs["router"] = router
    handler = type("BoundGatewayRequestHandler", (GatewayRequestHandler,), attrs)
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="InferenceDeck API-mapping gateway (OpenAI and Anthropic APIs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--switch-models", action=argparse.BooleanOptionalAction, default=None,
        help="Load the profile a request's model names, releasing the loaded one "
             "(needs inferencedeck-web; default: gateway_model_switching in config).",
    )
    args = parser.parse_args()
    switching = AppConfig.load().gateway_model_switching if args.switch_models is None else args.switch_models
    router = Router(switcher=ModelSwitcher()) if switching else None
    make_server(args.host, args.port, router=router).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
