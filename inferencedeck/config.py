from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .fileio import atomic_write_text, locked
from .live_config import Rejected, live_file
from .paths import config_dir


CONFIG_FILENAME = "config.json"


class ConfigRejected(ValueError):
    """config.json currently holds an edit that was rejected (see AppConfig.load)."""


def _parse_config(data: bytes) -> "AppConfig":
    # Imported here: config_check imports this module for AppConfig.
    from .config_check import CONFIG_SCHEMA, validate_schema

    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Rejected([f"not valid JSON: {exc}"]) from None
    if not isinstance(raw, dict):
        raise Rejected(["must be a JSON object"])
    errors = [f"{where}: {message}" for severity, where, message in validate_schema(raw, CONFIG_SCHEMA) if severity == "error"]
    if errors:
        raise Rejected(errors)
    allowed = {field_name for field_name in AppConfig.__dataclass_fields__}
    return AppConfig(**{key: value for key, value in raw.items() if key in allowed})


def _live(path: str | Path | None):
    return live_file(_config_path(path), _parse_config, "config.json")


def _config_path(path: str | Path | None) -> Path:
    return Path(path).expanduser() if path else config_dir() / CONFIG_FILENAME


@dataclass
class AppConfig:
    """Portable user configuration for the new control plane."""

    model_dirs: list[str] = field(default_factory=list)
    default_host: str = "127.0.0.1"
    default_port: int = 8080
    default_backend: str = "llama.cpp"
    runtime_dirs: list[str] = field(default_factory=list)
    llama_server_path: str = ""
    # "auto" picks the best llama.cpp build this CPU can run; a runtime id
    # (e.g. "standard", "cuda-avx1") or path pins one, if it is compatible.
    llama_runtime: str = "auto"
    llama_fit_params_path: str = ""
    extra_llama_args: list[str] = field(default_factory=list)
    # vllm.cpp's vllm-server; found under runtime_dirs, VLLM_CPP_HOME or PATH when unset.
    vllm_cpp_server_path: str = ""
    extra_vllm_cpp_args: list[str] = field(default_factory=list)
    # MLC LLM's mlc_llm command; found on PATH or run as python -m mlc_llm when unset.
    mlc_llm_path: str = ""
    extra_mlc_llm_args: list[str] = field(default_factory=list)
    update_channel: str = "stable"
    profile_names: dict[str, str] = field(default_factory=dict)
    server_history_limit: int = 5
    auto_generate_launch_scripts: bool = True
    auto_scan_on_startup: bool = True
    # Release a server's GPU (stop and park it, like Release GPU) after this
    # many seconds without requests; 0 turns it off. A profile param or launch
    # override named idle_release_seconds sets it for one server.
    idle_release_seconds: int = 0
    # Starting a server that fits the GPU on its own but not next to the ones
    # already running: "block" refuses (unless told to release them or forced),
    # "warn" starts with a warning, "off" skips the check. See gpu_budget.py.
    concurrent_vram_check: str = "block"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        """The config in effect: the file's last version that parsed and validated.

        An edit that breaks the file (bad JSON, not an object, a wrong type or
        value) is not applied; the previous version stays in effect, or the
        defaults if there is none, and live_config.rejected_files() reports it
        until it's fixed.
        """
        config = _live(path).get()
        return config if config is not None else cls()

    def save(self, path: str | Path | None = None) -> Path:
        config_path = _config_path(path)
        atomic_write_text(config_path, json.dumps(asdict(self), indent=2) + "\n")
        return config_path

    @classmethod
    def update(cls, change: Callable[["AppConfig"], None], path: str | Path | None = None) -> "AppConfig":
        """Load, apply ``change`` and save under a lock, so concurrent updates don't undo each other."""
        config_path = _config_path(path)
        with locked(config_path.with_name(config_path.name + ".lock")):
            live = _live(config_path)
            config = live.get() or cls()
            rejected = live.state()["rejected"]
            if rejected:
                # Saving now would overwrite the edit in progress with the old version.
                raise ConfigRejected(
                    f"{config_path} has errors, so settings can't be changed here until it is fixed: "
                    + "; ".join(rejected["errors"])
                )
            change(config)
            config.save(config_path)
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
