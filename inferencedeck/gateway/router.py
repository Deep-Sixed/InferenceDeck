"""Pick the inference target for a request, by the model name it asks for.

The catalog holds every target the gateway may use:

- the default target, as before: the enabled remote/cloud endpoint, otherwise
  the first running local server;
- every other running (not paused) local server;
- every other remote endpoint marked routable. Self-hosted endpoints are
  routable unless they opt out; cloud endpoints must opt in with
  ``"routable": true``.

A requested model name is matched, case-insensitively, against each target's
names (endpoint aliases, configured model and file name; a local server's
profile, id and model file). ``<endpoint>/<model>`` picks a model on a named
endpoint, e.g. ``openrouter/meta-llama/llama-3.3-70b-instruct``. A name that
matches nothing, or no name, goes to the default target, so clients with a
hard-coded model name keep working.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..remotes import RemoteEndpoint, list_endpoints
from ..server_manager import http_base, list_servers
from .engines import OllamaEngine, OpenAICompatibleEngine
from .ir import GatewayError

Engine = OpenAICompatibleEngine | OllamaEngine


@dataclass
class Target:
    engine: Engine
    label: str       # e.g. "Qwen3-32B (Thanatos · Tailscale · Self-hosted)"
    model_id: str    # the name /v1/models reports
    names: tuple[str, ...] = ()  # every name that routes here
    endpoint: str = ""           # remote endpoint name, for <endpoint>/<model>
    default: bool = False
    server_id: str = ""          # local server id, for in-flight tracking (see inflight.py)

    def matches(self, name: str) -> bool:
        wanted = name.casefold()
        return any(candidate.casefold() == wanted for candidate in self.names)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "model": self.model_id, "names": list(self.names),
                "api_base": self.engine.api_base, "default": self.default, "server_id": self.server_id}


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


def _unique(names: list[str]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for name in names:
        if name and name.casefold() not in seen:
            seen[name.casefold()] = name
    return tuple(seen.values())


def _remote_key(remote: RemoteEndpoint) -> str:
    """The endpoint's API key; raises when it needs one that is not set."""
    if not remote.valid:
        raise GatewayError(503, f"remote endpoint {remote.name} is invalid: {remote.error}", "overloaded")
    key = os.environ.get(remote.api_key_env, "").strip() if remote.api_key_env else ""
    if remote.key_required and not key:
        raise GatewayError(503, f"{remote.display_name}: ${remote.api_key_env} is not set", "overloaded")
    return key


def remote_target(remote: RemoteEndpoint, default: bool = False) -> Target:
    key = _remote_key(remote)
    engine: Engine
    if remote.provider.lower() == "ollama":
        engine = OllamaEngine(_ollama_base(remote.base_url), model=remote.model, api_key=key)
    else:
        engine = OpenAICompatibleEngine(_api_base(remote.base_url), model=remote.model, api_key=key)
    label = f"{remote.display_name} ({remote.summary})" if remote.summary else remote.display_name
    names = _unique([*remote.aliases, remote.model, remote.name])
    return Target(engine, label, names[0], names, endpoint=remote.name, default=default)


def local_target(server: dict[str, Any], default: bool = False) -> Target:
    mode = str(server.get("mode") or server.get("id") or "local")
    model_path = server.get("model_path")
    names = _unique([mode, str(server.get("id") or ""), Path(model_path).stem if model_path else ""])
    engine = OpenAICompatibleEngine(http_base(server.get("host"), int(server.get("port") or 8080)) + "/v1")
    return Target(engine, f"{mode} (local)", mode, names, default=default, server_id=str(server.get("id") or ""))


@dataclass
class Router:
    """Builds the catalog fresh for each request, so it follows servers starting and stopping."""

    endpoints: Any = field(default=list_endpoints)
    servers: Any = field(default=list_servers)

    def catalog(self, remotes: list[RemoteEndpoint] | None = None) -> list[Target]:
        remotes = self.endpoints() if remotes is None else remotes
        locals_ = [s for s in self.servers() if s.get("running") and not s.get("suspended")]
        enabled = next((r for r in remotes if r.enabled), None)
        targets: list[Target] = []
        if enabled is not None:
            targets.append(remote_target(enabled, default=True))
        for index, server in enumerate(locals_):
            targets.append(local_target(server, default=enabled is None and index == 0))
        for remote in remotes:
            if remote is enabled or not remote.routable:
                continue
            try:
                targets.append(remote_target(remote))
            except GatewayError:
                continue  # invalid or missing its key: not offered
        return targets

    def resolve(self, model: str = "") -> Target:
        # An enabled endpoint that cannot be used is an error, not a reason to
        # silently send traffic somewhere else.
        remotes = self.endpoints()
        enabled = next((r for r in remotes if r.enabled), None)
        if enabled is not None:
            _remote_key(enabled)
        targets = self.catalog(remotes)
        if model:
            for target in targets:
                if target.matches(model):
                    return target
            prefix, _, rest = model.partition("/")
            if rest:
                for target in targets:
                    if target.endpoint and target.endpoint.casefold() == prefix.casefold():
                        engine = dataclasses.replace(target.engine, model=rest)
                        return dataclasses.replace(target, engine=engine)
        for target in targets:
            if target.default:
                return target
        raise GatewayError(503, "no inference target: start a local profile or enable a remote endpoint", "overloaded")


def resolve_target(model: str = "") -> Target:
    return Router().resolve(model)
