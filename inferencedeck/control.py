from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .inventory import build_inventory
from .profile_resolver import resolve_profiles
from .server_manager import (
    list_servers,
    prepare_launch_command,
    resume_server,
    server_logs,
    start_profile,
    stop_server,
    suspend_server,
)


@dataclass
class ControlPlane:
    """Stable frontend-facing facade over the portable InferenceDeck core.

    Frontends should use this API instead of reaching into discovery, profile,
    or process-management modules directly.  The methods intentionally return
    JSON-serializable dictionaries/lists so native and browser frontends can
    share the same contract.
    """

    project_root: str | Path | None = None
    model_dirs: list[str | Path] | None = None

    def _config(self) -> AppConfig:
        return AppConfig.load()

    def status(self) -> dict[str, Any]:
        servers = list_servers()
        running = [server for server in servers if server.get("running")]
        return {
            "version": 1,
            "running_count": len(running),
            "servers": servers,
            "capabilities": {
                "start": True,
                "stop": True,
                "suspend": True,
                "resume": True,
                "prepare": True,
                "logs": True,
            },
        }

    def inventory(self) -> dict[str, Any]:
        return build_inventory(project_root=self.project_root, model_dirs=self.model_dirs)

    def profiles(self) -> list[dict[str, Any]]:
        return [profile.to_dict() for profile in resolve_profiles(self.project_root, self.model_dirs)]

    def prepare(self, mode: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return prepare_launch_command(
            mode,
            project_root=self.project_root,
            model_dirs=self.model_dirs,
            overrides=overrides,
            config=self._config(),
        )

    def start(self, mode: str, overrides: dict[str, Any] | None = None, *, stop_existing: bool = False) -> dict[str, Any]:
        return start_profile(
            mode,
            project_root=self.project_root,
            model_dirs=self.model_dirs,
            overrides=overrides,
            stop_existing=stop_existing,
        )

    def stop(self, *, server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
        return stop_server(server_id=server_id, mode=mode)

    def suspend(self, *, server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
        return suspend_server(server_id=server_id, mode=mode)

    def resume(self, *, server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
        return resume_server(server_id=server_id, mode=mode)

    def logs(self, server_id: str, *, lines: int = 200) -> dict[str, Any]:
        return server_logs(server_id, lines=lines)
