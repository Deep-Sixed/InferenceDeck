from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .fileio import atomic_write_text, locked
from .paths import config_dir

LANE_REMOTE_HOST = "remote_host"
LANE_TRUE_CLOUD = "true_cloud"
VALID_LANES = {LANE_REMOTE_HOST, LANE_TRUE_CLOUD}
# Providers that are always someone's own server, so a config without a lane
# is treated as remote_host (and may omit apiKeyEnv).
SELF_HOSTED_PROVIDERS = {"llamacpp", "ollama"}

# Transport is how requests reach the endpoint, independent of the lane (who
# runs the model). A self-hosted llama.cpp on another box is remote_host
# whether it is reached over the LAN or a tailnet.
TRANSPORT_TAILSCALE = "tailscale"
TRANSPORT_LAN = "lan"
TRANSPORT_HTTPS = "https"
VALID_TRANSPORTS = {TRANSPORT_TAILSCALE, TRANSPORT_LAN, TRANSPORT_HTTPS}

LANE_LABELS = {LANE_REMOTE_HOST: "Self-hosted", LANE_TRUE_CLOUD: "Cloud"}
TRANSPORT_LABELS = {TRANSPORT_TAILSCALE: "Tailscale", TRANSPORT_LAN: "LAN", TRANSPORT_HTTPS: "HTTPS"}
PROVIDER_LABELS = {
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "llamacpp": "llama.cpp",
    "ollama": "Ollama",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "vllm": "vLLM",
}
# Tailscale assigns node addresses from the CGNAT range and MagicDNS names
# under ts.net.
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


@dataclass(frozen=True)
class RemoteEndpoint:
    path: Path
    provider: str
    model: str
    base_url: str
    api_key_env: str
    display_name: str
    lane: str
    enabled: bool
    valid: bool
    error: str = ""
    context_size: int | None = None
    tags: tuple[str, ...] = ()
    host: str = ""
    transport: str = ""

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def key_required(self) -> bool:
        # Cloud endpoints always name a key variable; a self-hosted endpoint
        # (e.g. llama-server without --api-key on a tailnet) may omit it.
        return bool(self.api_key_env)

    @property
    def key_present(self) -> bool:
        return self.key_required and bool(os.environ.get(self.api_key_env, "").strip())

    @property
    def selectable(self) -> bool:
        return self.valid and (self.key_present or not self.key_required)

    @property
    def summary(self) -> str:
        """Where the model runs, e.g. "Thanatos · Tailscale · Self-hosted"."""
        if self.lane == LANE_REMOTE_HOST:
            where = self.host
        else:
            where = PROVIDER_LABELS.get(self.provider.lower(), self.provider)
        parts = [where, TRANSPORT_LABELS.get(self.transport, "") if self.lane == LANE_REMOTE_HOST else "",
                 LANE_LABELS.get(self.lane, self.lane)]
        return " · ".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["name"] = self.name
        payload["summary"] = self.summary
        payload["key_required"] = self.key_required
        payload["key_present"] = self.key_present
        payload["selectable"] = self.selectable
        # Never serialize the actual environment value.
        return payload


def endpoints_dir() -> Path:
    return config_dir() / "remote_endpoints"


def _url_host(base_url: str) -> str:
    try:
        return urlsplit(base_url).hostname or ""
    except ValueError:
        return ""


def infer_transport(base_url: str, lane: str) -> str:
    """Best guess at the transport when the config does not name one."""
    hostname = _url_host(base_url).lower()
    if hostname.endswith(".ts.net"):
        return TRANSPORT_TAILSCALE
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (address in _TAILNET_V4 or address in _TAILNET_V6):
        return TRANSPORT_TAILSCALE
    if lane == LANE_TRUE_CLOUD:
        return TRANSPORT_HTTPS
    # A bare name such as http://thanatos:8080 may be MagicDNS or LAN DNS;
    # leave it unnamed rather than guess.
    return ""


def _parse(path: Path) -> RemoteEndpoint:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RemoteEndpoint(path, "", "", "", "", "", LANE_TRUE_CLOUD, False, False, f"bad JSON: {exc}")
    provider = str(data.get("provider") or "").strip()
    model = str(data.get("model") or "").strip()
    base_url = str(data.get("baseUrl") or "").strip()
    api_key_env = str(data.get("apiKeyEnv") or "").strip()
    display_name = str(data.get("displayName") or model or path.stem).strip()
    raw_enabled = data.get("enabled", False)
    lane = str(data.get("lane") or (LANE_REMOTE_HOST if provider in SELF_HOSTED_PROVIDERS else LANE_TRUE_CLOUD)).strip().lower()
    error = ""
    if "apiKey" in data or "api_key" in data:
        error = "API keys must not be stored in endpoint configuration"
    elif not isinstance(raw_enabled, bool):
        error = "enabled must be true/false"
    elif not base_url:
        error = "baseUrl is required"
    elif lane not in VALID_LANES:
        error = "lane must be remote_host or true_cloud"
    elif not api_key_env and lane != LANE_REMOTE_HOST:
        error = "apiKeyEnv is required for true_cloud endpoints"
    transport = str(data.get("transport") or "").strip().lower()
    if transport and transport not in VALID_TRANSPORTS:
        error = error or "transport must be tailscale, lan or https"
    elif not transport:
        transport = infer_transport(base_url, lane)
    host = str(data.get("host") or "").strip() or _url_host(base_url)
    context = data.get("contextSize")
    try:
        context_size = int(context) if context is not None else None
    except (TypeError, ValueError):
        context_size = None
        error = error or "contextSize must be an integer"
    tags = tuple(str(tag) for tag in (data.get("tags") or []) if str(tag).strip())
    return RemoteEndpoint(
        path=path,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        display_name=display_name,
        lane=lane,
        enabled=bool(raw_enabled) if isinstance(raw_enabled, bool) else False,
        valid=not error,
        error=error,
        context_size=context_size,
        tags=tags,
        host=host,
        transport=transport,
    )


def list_endpoints(directory: Path | None = None) -> list[RemoteEndpoint]:
    root = directory or endpoints_dir()
    if not root.is_dir():
        return []
    return [_parse(path) for path in sorted(root.glob("*.json")) if not path.name.endswith(".example.json")]


def active_endpoint(directory: Path | None = None) -> RemoteEndpoint | None:
    enabled = [cfg for cfg in list_endpoints(directory) if cfg.enabled]
    return enabled[0] if enabled else None


def _write_enabled(path: Path, enabled: bool) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["enabled"] = bool(enabled)
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def _switch_lock(root: Path):
    # One lock over the whole directory: enabling one endpoint rewrites several files.
    return locked(root / ".switch.lock")


def enable_endpoint(name: str, directory: Path | None = None) -> RemoteEndpoint:
    root = directory or endpoints_dir()
    with _switch_lock(root):
        configs = list_endpoints(root)
        target = next((cfg for cfg in configs if cfg.name == name), None)
        if target is None:
            raise ValueError(f"Unknown remote endpoint: {name}")
        if not target.valid:
            raise ValueError(f"Invalid endpoint {name}: {target.error}")
        if target.key_required and not target.key_present:
            raise ValueError(f"{target.display_name}: set ${target.api_key_env} before enabling")
        # Disable the others before enabling the target, so a failure part-way
        # leaves at most one endpoint enabled, never two.
        for cfg in configs:
            if cfg.enabled and cfg.name != name:
                _write_enabled(cfg.path, False)
        if not target.enabled:
            _write_enabled(target.path, True)
        return _parse(target.path)


def disable_all(directory: Path | None = None) -> None:
    root = directory or endpoints_dir()
    with _switch_lock(root):
        for cfg in list_endpoints(root):
            if cfg.enabled:
                _write_enabled(cfg.path, False)
