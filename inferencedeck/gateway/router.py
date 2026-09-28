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

With a ``ModelSwitcher`` (``inferencedeck-gateway --switch-models``), a name that
routes nowhere but names a launchable profile (mode, display name or alias)
loads that profile first; see ``switching``. ``explain`` (``GET /route``)
reports that switch without making it.

When several targets match the same name and the fleet view is configured,
they are ordered by where the model runs best (see placement.py); otherwise
the first match in catalog order wins, as before. Placement only reorders
existing matches, so it never starts anything or overrides a model switch.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..remotes import RemoteEndpoint, list_endpoints
from ..server_manager import http_base, list_servers
from .engines import AnthropicEngine, OllamaEngine, OpenAICompatibleEngine
from .ir import GatewayError
from .placement import GROUP_LABELS, LOCAL, FleetRanker, rank_target, url_host
from .switching import ModelSwitcher, profile_names

Engine = OpenAICompatibleEngine | OllamaEngine | AnthropicEngine


@dataclass
class Target:
    engine: Engine
    label: str       # e.g. "Qwen3-32B (Thanatos · Tailscale · Self-hosted)"
    model_id: str    # the name /v1/models reports
    names: tuple[str, ...] = ()  # every name that routes here
    endpoint: str = ""           # remote endpoint name, for <endpoint>/<model>
    default: bool = False
    server_id: str = ""          # tracked local server; in-flight tracking keys on it (see inflight.py)
    # Keys that identify the machine behind the target in the fleet view.
    host_keys: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        wanted = name.casefold()
        return any(candidate.casefold() == wanted for candidate in self.names)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "model": self.model_id, "names": list(self.names),
                "api_base": self.engine.api_base, "default": self.default, "server_id": self.server_id}


def _api_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _root(base_url: str, suffixes: tuple[str, ...]) -> str:
    base = base_url.rstrip("/")
    for suffix in suffixes:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _ollama_base(base_url: str) -> str:
    # Accept the server root, or a URL copied from its OpenAI layer (/v1) or native API (/api).
    return _root(base_url, ("/v1", "/api"))


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
    provider = remote.provider.lower()
    if provider == "ollama":
        engine = OllamaEngine(_ollama_base(remote.base_url), model=remote.model, api_key=key)
    elif provider == "anthropic":
        # Accept https://api.anthropic.com, .../v1 or .../v1/messages.
        engine = AnthropicEngine(_root(remote.base_url, ("/v1/messages", "/v1")), model=remote.model, api_key=key)
    else:
        engine = OpenAICompatibleEngine(_api_base(remote.base_url), model=remote.model, api_key=key)
    label = f"{remote.display_name} ({remote.summary})" if remote.summary else remote.display_name
    names = _unique([*remote.aliases, remote.model, remote.name])
    keys = _unique([url_host(remote.base_url), remote.host.casefold(), remote.name.casefold()])
    return Target(engine, label, names[0], names, endpoint=remote.name, default=default, host_keys=keys)


def local_target(server: dict[str, Any], default: bool = False) -> Target:
    mode = str(server.get("mode") or server.get("id") or "local")
    model_path = server.get("model_path")
    names = _unique([mode, str(server.get("id") or ""), Path(model_path).stem if model_path else ""])
    engine = OpenAICompatibleEngine(http_base(server.get("host"), int(server.get("port") or 8080)) + "/v1")
    return Target(engine, f"{mode} (local)", mode, names, default=default, server_id=str(server.get("id") or ""),
                  host_keys=(LOCAL,))


@dataclass
class Router:
    """Builds the catalog fresh for each request, so it follows servers starting and stopping."""

    endpoints: Any = field(default=list_endpoints)
    servers: Any = field(default=list_servers)
    switcher: ModelSwitcher | None = None
    ranker: Any = field(default_factory=FleetRanker)

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

    def loadable(self) -> list[dict[str, Any]]:
        """Launchable profiles that aren't running; empty without switching."""
        if self.switcher is None:
            return []
        running = {s.get("mode") for s in self.servers() if s.get("running") and not s.get("suspended")}
        try:
            profiles = self.switcher.profiles()
        except GatewayError:
            return []
        return [p for p in profiles if p.get("mode") not in running]

    def resolve(self, model: str = "") -> Target:
        target, profile = self._pick(model)
        if profile is not None:
            assert self.switcher is not None
            return local_target(self.switcher.ensure_loaded(profile, self.servers))
        return target

    def _ordered_matches(self, targets: list[Target], model: str) -> list[tuple[Target, Any]]:
        """Targets answering to ``model``, best first, each with its fleet rank (or None)."""
        matches = [t for t in targets if t.matches(model)]
        ranks = self.ranker.ranks(model) if len(matches) > 1 and self.ranker is not None else None
        if not ranks:
            return [(t, None) for t in matches]
        # Stable: targets that rank equally keep catalog order.
        return sorted(((t, rank_target(t.host_keys, ranks)) for t in matches), key=lambda pair: pair[1].key())

    def explain(self, model: str = "") -> dict[str, Any]:
        """Which target a request for ``model`` would use, and the ranking behind it. Never loads anything."""
        ordered = self._ordered_matches(self.catalog(), model) if model else []
        target, profile = self._pick(model)
        if profile is not None:
            mode = str(profile["mode"])
            running = [str(s.get("mode") or s.get("id")) for s in self.servers()
                       if s.get("running") and s.get("mode") != mode]
            chosen: dict[str, Any] = {"label": f"{mode} (local)", "model": mode, "names": sorted(profile_names(profile)),
                                      "api_base": None, "default": False, "server_id": None, "loaded": False}
            switch: dict[str, Any] = {"action": "switch", "would_load": mode, "would_release": running}
        else:
            assert target is not None
            chosen, switch = {**target.to_dict(), "loaded": True}, {"action": "route"}
        return {
            "model": model or None,
            "target": chosen,
            **switch,
            "placement_used": any(rank is not None for _, rank in ordered),
            "candidates": [
                {**t.to_dict(), "group": GROUP_LABELS[r.group] if r else None,
                 "fleet_host": r.host if r else None, "reason": r.reason if r else None}
                for t, r in ordered
            ],
        }

    def _pick(self, model: str) -> tuple[Target | None, dict[str, Any] | None]:
        """The target for ``model``, or (with switching) the profile that would be loaded for it.

        Has no side effects, so ``explain`` reports exactly what ``resolve`` would do.
        """
        # An enabled endpoint that cannot be used is an error, not a reason to
        # silently send traffic somewhere else.
        remotes = self.endpoints()
        enabled = next((r for r in remotes if r.enabled), None)
        if enabled is not None:
            _remote_key(enabled)
        targets = self.catalog(remotes)
        if model:
            ordered = self._ordered_matches(targets, model)
            if ordered:
                return ordered[0][0], None
            prefix, _, rest = model.partition("/")
            if rest:
                for target in targets:
                    if target.endpoint and target.endpoint.casefold() == prefix.casefold():
                        engine = dataclasses.replace(target.engine, model=rest)
                        return dataclasses.replace(target, engine=engine), None
            # Local starts are refused while an endpoint is enabled, so only switch without one.
            if self.switcher is not None and enabled is None:
                profile = self.switcher.find_profile(model)
                if profile is not None:
                    return None, profile
        for target in targets:
            if target.default:
                return target, None
        raise GatewayError(503, "no inference target: start a local profile or enable a remote endpoint", "overloaded")


def resolve_target(model: str = "") -> Target:
    return Router().resolve(model)
