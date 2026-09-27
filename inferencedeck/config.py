from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import config_dir


CONFIG_FILENAME = "config.json"


@dataclass
class AppConfig:
    """Portable user configuration for the new control plane."""

    model_dirs: list[str] = field(default_factory=list)
    default_host: str = "127.0.0.1"
    default_port: int = 8080
    default_backend: str = "llama.cpp"
    runtime_dirs: list[str] = field(default_factory=list)
    llama_server_path: str = ""
    llama_fit_params_path: str = ""
    extra_llama_args: list[str] = field(default_factory=list)
    update_channel: str = "stable"
    profile_names: dict[str, str] = field(default_factory=dict)
    server_history_limit: int = 5
    auto_generate_launch_scripts: bool = True
    auto_scan_on_startup: bool = True

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        config_path = Path(path).expanduser() if path else config_dir() / CONFIG_FILENAME
        if not config_path.is_file():
            return cls()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)

    def save(self, path: str | Path | None = None) -> Path:
        config_path = Path(path).expanduser() if path else config_dir() / CONFIG_FILENAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = config_path.with_suffix(f"{config_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(config_path)
        return config_path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
