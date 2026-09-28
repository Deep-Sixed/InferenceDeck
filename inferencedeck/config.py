from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .fileio import atomic_write_text, locked
from .paths import config_dir


CONFIG_FILENAME = "config.json"


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
    # KoboldCpp's executable (or koboldcpp.py); found under runtime_dirs, KOBOLDCPP_HOME or PATH when unset.
    koboldcpp_path: str = ""
    extra_koboldcpp_args: list[str] = field(default_factory=list)
    update_channel: str = "stable"
    profile_names: dict[str, str] = field(default_factory=dict)
    server_history_limit: int = 5
    auto_generate_launch_scripts: bool = True
    auto_scan_on_startup: bool = True
    # Telemetry history: seconds between samples (0 turns history off) and days of
    # one-minute averages kept on disk.
    telemetry_sample_seconds: int = 5
    telemetry_retention_days: int = 7
    # Gateway loads the profile a request's model names, releasing the loaded one.
    gateway_model_switching: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        config_path = _config_path(path)
        if not config_path.is_file():
            return cls()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(data, dict):
            return cls()  # valid JSON but not a settings object; fall back to defaults like unreadable JSON
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)

    def save(self, path: str | Path | None = None) -> Path:
        config_path = _config_path(path)
        atomic_write_text(config_path, json.dumps(asdict(self), indent=2) + "\n")
        return config_path

    @classmethod
    def update(cls, change: Callable[["AppConfig"], None], path: str | Path | None = None) -> "AppConfig":
        """Load, apply ``change`` and save under a lock, so concurrent updates don't undo each other."""
        config_path = _config_path(path)
        with locked(config_path.with_name(config_path.name + ".lock")):
            config = cls.load(config_path)
            change(config)
            config.save(config_path)
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
