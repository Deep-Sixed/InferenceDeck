"""Engine adapters: canonical requests -> an inference server's own API.

- ``OpenAICompatibleEngine``: llama.cpp, vllm.cpp, KoboldCpp, vLLM, LM Studio, OpenRouter.
- ``OllamaEngine``: Ollama's native ``/api/chat``, which keeps its own options
  (``num_ctx``, ``keep_alive``, ...) and ``format`` structured output.
- ``AnthropicEngine``: the Anthropic Messages API (``/v1/messages``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from . import anthropic_api, ollama_api, openai_api
from .ir import ChatRequest, ChatResult, GatewayError, StreamEvent

# Local models can take minutes to prefill a long prompt.
DEFAULT_TIMEOUT_SECONDS = 600


def _upstream_error(exc: urllib.error.HTTPError) -> GatewayError:
    try:
        payload = json.loads(exc.read() or b"{}")
    except (json.JSONDecodeError, OSError):
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    message = (error.get("message") if isinstance(error, dict) else error) or exc.reason or "upstream error"
    kind = {400: "invalid_request", 401: "authentication", 403: "authentication", 404: "not_found",
            429: "rate_limit", 503: "overloaded", 529: "overloaded"}.get(exc.code, "api_error")
    # 401/403 from upstream is the gateway's credential problem, not the client's;
    # 529 (Anthropic "overloaded") is not a standard status, so clients get 503.
    status = {401: 502, 403: 502, 529: 503}.get(exc.code, exc.code)
    return GatewayError(status, f"upstream: {message}", kind)


@dataclass
class _HTTPEngine:
    api_base: str
    model: str = ""  # sent upstream; empty -> the client's model name
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    # Set by subclasses.
    path: ClassVar[str] = ""
    stream_accept: ClassVar[str] = ""
    to_wire: ClassVar[Callable[[ChatRequest, str], dict[str, Any]]]
    from_wire: ClassVar[Callable[[dict[str, Any], str], ChatResult]]
    parse_stream: ClassVar[Callable[[Iterable[bytes]], Iterator[StreamEvent]]]

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _open(self, request: ChatRequest) -> Any:
        body = type(self).to_wire(request, self.model or request.model)
        headers = {"Content-Type": "application/json",
                   "Accept": self.stream_accept if request.stream else "application/json",
                   **self._auth_headers()}
        http_request = urllib.request.Request(
            f"{self.api_base.rstrip('/')}{self.path}",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            return urllib.request.urlopen(http_request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise _upstream_error(exc) from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise GatewayError(502, f"cannot reach {self.api_base}: {reason}", "overloaded") from exc

    def complete(self, request: ChatRequest) -> ChatResult:
        with self._open(request) as response:
            try:
                payload = json.loads(response.read())
            except json.JSONDecodeError as exc:
                raise GatewayError(502, "upstream returned invalid JSON") from exc
        return type(self).from_wire(payload, self.model or request.model)

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        # Opened eagerly so connection and HTTP errors surface before the
        # gateway commits to a 200 streaming response.
        response = self._open(request)
        parse = type(self).parse_stream

        def events() -> Iterator[StreamEvent]:
            with response:
                try:
                    yield from parse(response)
                except OSError as exc:
                    raise GatewayError(502, f"upstream stream interrupted: {exc}") from exc

        return events()


@dataclass
class OpenAICompatibleEngine(_HTTPEngine):
    """``api_base`` ends in /v1."""

    path = "/chat/completions"
    stream_accept = "text/event-stream"
    to_wire = staticmethod(openai_api.to_wire)
    from_wire = staticmethod(openai_api.from_wire)
    parse_stream = staticmethod(openai_api.stream_events)


@dataclass
class OllamaEngine(_HTTPEngine):
    """``api_base`` is the server root, e.g. http://host:11434."""

    path = "/api/chat"
    stream_accept = "application/x-ndjson"
    to_wire = staticmethod(ollama_api.to_wire)
    from_wire = staticmethod(ollama_api.from_wire)
    parse_stream = staticmethod(ollama_api.stream_events)


@dataclass
class AnthropicEngine(_HTTPEngine):
    """``api_base`` is the API root, e.g. https://api.anthropic.com."""

    API_VERSION: ClassVar[str] = "2023-06-01"

    path = "/v1/messages"
    stream_accept = "text/event-stream"
    to_wire = staticmethod(anthropic_api.to_wire)
    from_wire = staticmethod(anthropic_api.from_wire)
    parse_stream = staticmethod(anthropic_api.stream_events)

    def _auth_headers(self) -> dict[str, str]:
        headers = {"anthropic-version": self.API_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
