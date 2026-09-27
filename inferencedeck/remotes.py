from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fileio import atomic_write_text, locked
from .paths import config_dir

LANE_REMOTE_HOST = "remote_host"
LANE_TRUE_CLOUD = "true_cloud"
VALID_LANES = {LANE_REMOTE_HOST, LANE_TRUE_CLOUD}


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

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def key_present(self) -> bool:
        return bool(self.api_key_env) and bool(os.environ.get(self.api_key_env, "").strip())

    @property
    def selectable(self) -> bool:
        return self.valid and self.key_present

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["name"] = self.name
        payload["key_present"] = self.key_present
        payload["selectable"] = self.selectable
        # Never serialize the actual environment value.
        return payload


def endpoints_dir() -> Path:
    return config_dir() / "remote_endpoints"


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
    lane = str(data.get("lane") or (LANE_REMOTE_HOST if provider == "llamacpp" else LANE_TRUE_CLOUD)).strip().lower()
    error = ""
    if "apiKey" in data or "api_key" in data:
        error = "API keys must not be stored in endpoint configuration"
    elif not isinstance(raw_enabled, bool):
        error = "enabled must be true/false"
    elif not base_url:
        error = "baseUrl is required"
    elif not api_key_env:
        error = "apiKeyEnv is required"
    elif lane not in VALID_LANES:
        error = "lane must be remote_host or true_cloud"
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
        if not target.key_present:
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
