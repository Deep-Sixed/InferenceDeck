from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import load_benchmark_results, run_profile_benchmark
from .config import AppConfig
from .fit import run_fit_test
from .hardware import detect_system_hardware
from .inventory import build_inventory
from .profile_resolver import resolve_profiles
from .remotes import active_endpoint, disable_all, enable_endpoint, list_endpoints
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
        remote = active_endpoint()
        return {
            "version": 1,
            "running_count": len(running),
            "servers": servers,
            "remote_active": remote.to_dict() if remote else None,
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

    def hardware(self) -> dict[str, Any]:
        return detect_system_hardware()

    def fit(self, mode: str, overrides: dict[str, Any] | None = None, *, target_mib: int = 1024) -> dict[str, Any]:
        return run_fit_test(
            mode,
            project_root=self.project_root,
            model_dirs=self.model_dirs,
            overrides=overrides,
            target_mib=target_mib,
        )

    def benchmark(
        self, mode: str, overrides: dict[str, Any] | None = None, *, completion_tokens: int = 128
    ) -> dict[str, Any]:
        return run_profile_benchmark(
            mode,
            project_root=self.project_root,
            model_dirs=self.model_dirs,
            overrides=overrides,
            completion_tokens=completion_tokens,
        )

    def benchmark_history(self) -> list[dict[str, Any]]:
        return load_benchmark_results()

    def remote_endpoints(self) -> dict[str, Any]:
        endpoints = list_endpoints()
        active = next((cfg for cfg in endpoints if cfg.enabled), None)
        return {
            "active": active.to_dict() if active else None,
            "endpoints": [cfg.to_dict() for cfg in endpoints],
        }

    def enable_remote(self, name: str) -> dict[str, Any]:
        cfg = enable_endpoint(name)
        return {"success": True, "active": cfg.to_dict()}

    def disable_remotes(self) -> dict[str, Any]:
        disable_all()
        return {"success": True, "active": None}

    def prepare(self, mode: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return prepare_launch_command(
            mode,
            project_root=self.project_root,
            model_dirs=self.model_dirs,
            overrides=overrides,
            config=self._config(),
        )

    def start(self, mode: str, overrides: dict[str, Any] | None = None, *, stop_existing: bool = False) -> dict[str, Any]:
        remote = active_endpoint()
        if remote is not None:
            return {
                "success": False,
                "error": f"Remote endpoint '{remote.display_name}' is active; disable it before starting a local profile.",
                "remote": remote.to_dict(),
            }
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
