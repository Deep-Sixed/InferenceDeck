"""Pick the inference target for a request.

The gateway follows the same single-active-target rule as the rest of
InferenceDeck: an enabled remote/cloud endpoint wins (local starts are refused
while one is enabled), otherwise the running local server is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..remotes import active_endpoint
from ..server_manager import http_base, list_servers
from .engines import OllamaEngine, OpenAICompatibleEngine
from .ir import GatewayError


@dataclass
class Target:
    engine: OpenAICompatibleEngine | OllamaEngine
    label: str       # e.g. "Qwen3-32B (Thanatos · Tailscale · Self-hosted)"
    model_id: str    # what /v1/models reports

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "model": self.model_id, "api_base": self.engine.api_base}


def _api_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _ollama_base(base_url: str) -> str:
    # Accept the server root, or a URL copied from its OpenAI layer (/v1) or native API (/api).
    base = base_url.rstrip("/")
    for suffix in ("/v1", "/api"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _local_server() -> dict[str, Any] | None:
    live = [s for s in list_servers() if s.get("running") and not s.get("suspended")]
    return live[0] if live else None


def resolve_target() -> Target:
    remote = active_endpoint()
    if remote is not None:
        if not remote.valid:
            raise GatewayError(503, f"remote endpoint {remote.name} is invalid: {remote.error}", "overloaded")
        key = os.environ.get(remote.api_key_env, "").strip() if remote.api_key_env else ""
        if remote.key_required and not key:
            raise GatewayError(503, f"{remote.display_name}: ${remote.api_key_env} is not set", "overloaded")
        label = f"{remote.display_name} ({remote.summary})" if remote.summary else remote.display_name
        if remote.provider.lower() == "ollama":
            engine: OpenAICompatibleEngine | OllamaEngine = OllamaEngine(
                _ollama_base(remote.base_url), model=remote.model, api_key=key)
        else:
            engine = OpenAICompatibleEngine(_api_base(remote.base_url), model=remote.model, api_key=key)
        return Target(engine, label, remote.model or remote.name)
    server = _local_server()
    if server is not None:
        mode = str(server.get("mode") or server.get("id") or "local")
        engine = OpenAICompatibleEngine(http_base(server.get("host"), int(server.get("port") or 8080)) + "/v1")
        return Target(engine, f"{mode} (local)", mode)
    raise GatewayError(503, "no inference target: start a local profile or enable a remote endpoint", "overloaded")
