from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import detect_all, detect_llama_cpp
from .benchmark import load_benchmark_results, run_profile_benchmark
from .capabilities import filter_profiles, profile_capabilities
from .config import AppConfig
from .config_check import check_all
from .fit import run_fit_test
from .hardware import detect_system_hardware
from .inflight import snapshot as inflight_snapshot
from .hf_download import download_model, repo_gguf_listing
from .inventory import build_inventory
from .live_config import rejected_files
from .paths import find_project_root
from .profile_resolver import resolve_profiles
from .remotes import active_endpoint, disable_all, enable_endpoint, list_endpoints
from .runtime_updates import check_runtime_updates
from .sampling import sampling_presets
from .server_manager import (
    CONTEXT_PRESETS,
    list_servers,
    plan_launch,
    prepare_launch_command,
    release_gpu,
    restart_server,
    restore_server,
    resume_server,
    server_log_paths,
    server_logs,
    set_idle_release,
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
        # Gateway requests running on each server right now (see inflight.py).
        gateway = inflight_snapshot()
        for server in servers:
            counts = gateway.get(str(server.get("id") or ""))
            if counts:
                server["in_flight"] = counts["in_flight"]
                server["last_request_at"] = counts["last_request_at"]
        remote = active_endpoint()
        return {
            "version": 1,
            # Config files whose latest edit was rejected; the previous version is in effect.
            "config_rejected": rejected_files(),
            "running_count": len(running),
            "servers": servers,
            "remote_active": remote.to_dict() if remote else None,
            "capabilities": {
                "start": True,
                "stop": True,
                # suspend/resume pause the process but keep its model in VRAM;
                # release/restore stop it (freeing VRAM) and start it again.
                "suspend": True,
                "resume": True,
                "release": True,
                "restore": True,
                "restart": True,
                "prepare": True,
                "logs": True,
                "idle_release": True,
                "hf_download": True,
            },
            "context_presets": list(CONTEXT_PRESETS),
            # Servers without their own idle_release_seconds use this; 0 = off.
            "idle_release_default_seconds": self._config().idle_release_seconds,
        }

    def inventory(self) -> dict[str, Any]:
        return build_inventory(project_root=self.project_root, model_dirs=self.model_dirs)

    def profiles(self, capability: str | None = None) -> list[dict[str, Any]]:
        """Resolved profiles with their capabilities; ``capability`` filters (see capabilities.QUERIES)."""
        profiles = []
        for resolved in resolve_profiles(self.project_root, self.model_dirs):
            item = resolved.to_dict()
            # Cache-only: listing profiles must never block on reading GGUF headers.
            item["capabilities"] = profile_capabilities(resolved.model, resolved.params, probe=False)
            profiles.append(item)
        return filter_profiles(profiles, capability) if capability else profiles

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
        # A benchmark may start a local server, so it obeys the same rule as Start.
        blocked = self._remote_blocks_local()
        if blocked:
            return blocked
        return run_profile_benchmark(
            mode,
            project_root=self.project_root,
            model_dirs=self.model_dirs,
            overrides=overrides,
            completion_tokens=completion_tokens,
        )

    def hf_files(self, repo_id: str) -> dict[str, Any]:
        """The GGUF quants and vision projectors a Hugging Face repo offers."""
        return repo_gguf_listing(repo_id)

    def hf_download(
        self,
        repo_id: str,
        *,
        quant: str | None = None,
        pattern: str | None = None,
        include_mmproj: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Download one quant (all shards, plus its mmproj) into the HF cache,
        which model discovery already scans."""
        return download_model(repo_id, pattern=pattern, quant=quant, include_mmproj=include_mmproj, dry_run=dry_run)

    def benchmark_history(self) -> list[dict[str, Any]]:
        return load_benchmark_results()

    def runtime(self) -> dict[str, Any]:
        """Which llama.cpp build will be used, why, and what else was found."""
        # Same root the launch path uses, so this reports the build Start will run.
        root = Path(self.project_root).expanduser().resolve() if self.project_root else find_project_root()
        env = detect_llama_cpp(root, config=self._config())
        return env.details["runtime_selection"]

    def updates(self, *, refresh: bool = False) -> dict[str, Any]:
        """Installed runtime versions against their latest upstream releases.

        GitHub answers are cached for an hour; ``refresh`` asks again. Nothing is
        ever downloaded or replaced.
        """
        root = Path(self.project_root).expanduser().resolve() if self.project_root else find_project_root()
        config = self._config()
        environments = [env.to_dict() for env in detect_all(root, config=config)]
        return check_runtime_updates(environments, channel=config.update_channel, force_refresh=refresh)

    def set_runtime(self, runtime: str) -> dict[str, Any]:
        """Pin a discovered, compatible build (by id), or go back to "auto"."""
        runtime = (runtime or "").strip()
        if runtime != "auto":
            match = next((c for c in self.runtime()["candidates"] if c["id"] == runtime), None)
            if match is None:
                raise ValueError(f"Unknown runtime: {runtime}")
            if not match["compatible"]:
                raise ValueError(f"{match['label']} can't run on this machine: {match['incompatible_reason']}")
        AppConfig.update(lambda config: setattr(config, "llama_runtime", runtime))
        return {"success": True, "runtime": self.runtime()}

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

    def _remote_blocks_local(self) -> dict[str, Any] | None:
        remote = active_endpoint()
        if remote is None:
            return None
        return {
            "success": False,
            "error": f"Remote endpoint '{remote.display_name}' is active; disable it before starting a local profile.",
            "remote": remote.to_dict(),
        }

    def start(
        self,
        mode: str,
        overrides: dict[str, Any] | None = None,
        *,
        stop_existing: bool = False,
        release_conflicts: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        blocked = self._remote_blocks_local()
        if blocked:
            return blocked
        return start_profile(
            mode,
            project_root=self.project_root,
            model_dirs=self.model_dirs,
            overrides=overrides,
            stop_existing=stop_existing,
            release_conflicts=release_conflicts,
            force=force,
        )

    def plan(self, mode: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return plan_launch(mode, project_root=self.project_root, model_dirs=self.model_dirs, overrides=overrides)

    def stop(self, *, server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
        return stop_server(server_id=server_id, mode=mode)

    def suspend(self, *, server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
        return suspend_server(server_id=server_id, mode=mode)

    def resume(self, *, server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
        return resume_server(server_id=server_id, mode=mode)

    def release_gpu(self, *, server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
        return release_gpu(server_id=server_id, mode=mode)

    def restore(
        self, server_id: str, overrides: dict[str, Any] | None = None, *, release_conflicts: bool = False, force: bool = False
    ) -> dict[str, Any]:
        blocked = self._remote_blocks_local()
        if blocked:
            return blocked
        return restore_server(
            server_id, overrides, project_root=self.project_root, model_dirs=self.model_dirs,
            release_conflicts=release_conflicts, force=force,
        )

    def restart(
        self, server_id: str, overrides: dict[str, Any] | None = None, *, release_conflicts: bool = False, force: bool = False
    ) -> dict[str, Any]:
        blocked = self._remote_blocks_local()
        if blocked:
            return blocked
        return restart_server(
            server_id, overrides, project_root=self.project_root, model_dirs=self.model_dirs,
            release_conflicts=release_conflicts, force=force,
        )

    def set_idle_release(self, server_id: str, seconds: int | None) -> dict[str, Any]:
        return set_idle_release(server_id, seconds)

    def config_check(self) -> dict[str, Any]:
        root = Path(self.project_root).expanduser() if self.project_root else None
        return check_all(project_root=root)

    def sampling_presets(self) -> dict[str, Any]:
        return {"presets": sampling_presets()}

    def log_paths(self, server_id: str) -> dict[str, str] | None:
        return server_log_paths(server_id)

    def logs(self, server_id: str, *, lines: int = 200) -> dict[str, Any]:
        return server_logs(server_id, lines=lines)
