"""Pick the inference target for a request.

The gateway follows the same single-active-target rule as the rest of
InferenceDeck: an enabled remote/cloud endpoint wins (local starts are refused
while one is enabled). Otherwise the request's ``model`` picks the running
local server it names (profile mode, name or alias); with model switching on,
a named profile that isn't loaded is loaded first (see ``switching``). A model
name that matches nothing falls back to the running local server, so clients
that send a fixed name (``gpt-4o``) keep working.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

from ..remotes import active_endpoint
from ..server_manager import http_base, list_servers
from .engines import OllamaEngine, OpenAICompatibleEngine
from .ir import GatewayError
from .switching import ModelSwitcher, is_live, matches


@dataclass
class Target:
    engine: OpenAICompatibleEngine | OllamaEngine
    label: str       # e.g. "Qwen3-32B (Thanatos · Tailscale · Self-hosted)"
    model_id: str    # what /v1/models reports
    # Held while a request is served, so a model switch waits for it.
    lease: Callable[[], AbstractContextManager[Any]] = field(default=nullcontext, repr=False)

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


def _live_servers() -> list[dict[str, Any]]:
    return [s for s in list_servers() if is_live(s)]


def _local_target(server: dict[str, Any], switcher: ModelSwitcher | None) -> Target:
    mode = str(server.get("mode") or server.get("id") or "local")
    engine = OpenAICompatibleEngine(http_base(server.get("host"), int(server.get("port") or 8080)) + "/v1")
    lease: Callable[[], AbstractContextManager[Any]] = nullcontext
    if switcher is not None:
        server_id = str(server.get("id") or mode)
        lease = lambda: switcher.inflight.lease(server_id)  # noqa: E731
    return Target(engine, f"{mode} (local)", mode, lease)


def resolve_target(model: str | None = None, switcher: ModelSwitcher | None = None) -> Target:
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
    live = _live_servers()
    # Server records carry the mode (and any alias override); a profile's name and
    # alias are matched through the switcher's profile list below.
    named = next((s for s in live if matches(s, model)), None)
    if named is not None:
        return _local_target(named, switcher)
    if switcher is not None and model:
        profile = switcher.find_profile(model)
        if profile is not None:
            running = next((s for s in live if s.get("mode") == profile.get("mode")), None)
            server = running or switcher.ensure_loaded(profile, list_servers)
            return _local_target(server, switcher)
    if live:
        return _local_target(live[0], switcher)
    raise GatewayError(503, "no inference target: start a local profile or enable a remote endpoint", "overloaded")


def list_models(switcher: ModelSwitcher | None = None) -> list[dict[str, Any]]:
    """What /v1/models reports: the remote model, else the loaded local models and,
    with switching on, every launchable profile a request can name."""
    try:
        if active_endpoint() is not None:
            target = resolve_target()
            return [{"id": target.model_id, "description": target.label, "loaded": True}]
    except GatewayError:
        return []
    models = [{"id": str(s.get("mode") or s.get("id")), "description": f"{s.get('mode')} (local)", "loaded": True}
              for s in _live_servers()]
    if switcher is not None:
        seen = {m["id"] for m in models}
        try:
            profiles = switcher.profiles()
        except GatewayError:
            profiles = []
        for profile in profiles:
            if profile.get("mode") not in seen:
                models.append({"id": str(profile["mode"]), "description": str(profile.get("name") or profile["mode"]),
                               "loaded": False})
    return models
