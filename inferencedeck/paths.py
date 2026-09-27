from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


APP_NAME = "inferencedeck"


def is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def split_env_paths(value: str | None) -> list[Path]:
    """Split an environment path list using the host OS path separator."""

    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def _path_key(path: Path) -> str:
    """A comparable identity for a path (resolved, case-folded on Windows)."""
    try:
        resolved = str(path.expanduser().resolve())
    except OSError:
        resolved = str(path.expanduser())
    return resolved.lower() if is_windows() else resolved


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path.expanduser())
    return result


def existing_dirs(paths: Iterable[Path]) -> list[Path]:
    return [path for path in dedupe_paths(paths) if path.is_dir()]


def config_dir() -> Path:
    override = os.environ.get("LCC_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if is_windows():
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / ".config" / APP_NAME


def cache_dir() -> Path:
    override = os.environ.get("LCC_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if is_windows():
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / APP_NAME
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / ".cache" / APP_NAME


def launch_scripts_dir() -> Path:
    """Directory where auto-generated launch scripts are stored locally."""

    override = os.environ.get("LCC_LAUNCH_SCRIPTS_DIR")
    if override:
        return Path(override).expanduser()
    return cache_dir() / "launch-scripts"


def huggingface_hub_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def default_lm_studio_dirs() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".lmstudio" / "models",
        home / ".cache" / "lm-studio" / "models",
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "LM Studio" / "models")
    return candidates


def default_llama_model_dirs(project_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(split_env_paths(os.environ.get("LCC_MODEL_DIRS")))
    candidates.extend(split_env_paths(os.environ.get("LLAMA_MODELS_DIR")))
    candidates.extend(split_env_paths(os.environ.get("LLAMA_CPP_MODEL_DIRS")))

    llama_cpp_home = os.environ.get("LLAMA_CPP_HOME")
    if llama_cpp_home:
        home = Path(llama_cpp_home).expanduser()
        candidates.extend([home / "models", home / ".cache" / "llm-models"])

    if project_root:
        candidates.append(project_root / "models")

    candidates.append(Path.cwd() / "models")
    candidates.extend(
        [
            Path.home() / "models",
            Path.home() / "Models",
            Path.home() / "llms",
            Path.home() / "LLMs",
        ]
    )
    candidates.append(Path.home() / ".cache" / "llm-models")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "llm-models")

    return candidates


def default_model_dirs(project_root: Path | None = None) -> list[tuple[str, Path]]:
    """Return targeted model scan roots with a source label.

    The list is intentionally conservative. It never scans a whole home drive.
    """

    pairs: list[tuple[str, Path]] = []
    for path in default_llama_model_dirs(project_root):
        pairs.append(("llama.cpp/custom", path))
    for path in default_lm_studio_dirs():
        pairs.append(("LM Studio", path))
    pairs.append(("Hugging Face cache", huggingface_hub_dir()))

    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for source, path in pairs:
        key = _path_key(path)
        if key not in seen and path.expanduser().is_dir():
            result.append((source, path.expanduser()))
            seen.add(key)
    return result


def executable_names(base_name: str) -> list[str]:
    if is_windows() and not base_name.lower().endswith(".exe"):
        return [f"{base_name}.exe", base_name]
    return [base_name]


def _looks_like_llama_runtime_root(path: Path) -> bool:
    """Return true when a directory contains a llama.cpp runtime marker."""

    if any((path / name).is_file() for name in executable_names("llama-server")):
        return True
    if (path / "switch-model.ps1").is_file():
        return True
    return False


def _runtime_ancestors(start: Path | None) -> list[Path]:
    if not start:
        return []
    try:
        current = start.expanduser().resolve()
    except OSError:
        current = start.expanduser()
    if current.is_file():
        current = current.parent
    return [path for path in [current, *current.parents] if _looks_like_llama_runtime_root(path)]


def find_project_root(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Find the nearest parent that looks like a InferenceDeck root."""

    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent

    markers = {"models.json", "llama-server.exe", "llama-server", "switch-model.ps1", "pyproject.toml"}
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent
    # Fallback: the installed package root (repo checkout) is a sane default.
    pkg_root = Path(__file__).resolve().parent.parent
    if (pkg_root / "pyproject.toml").exists():
        return pkg_root
    return None


def candidate_llama_roots(project_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    llama_cpp_home = os.environ.get("LLAMA_CPP_HOME")
    if llama_cpp_home:
        candidates.append(Path(llama_cpp_home).expanduser())

    if project_root:
        candidates.extend([project_root, project_root / "build", project_root / "build" / "bin", project_root / "bin"])
        candidates.extend(_runtime_ancestors(project_root))

    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.extend(_runtime_ancestors(cwd))
    return existing_dirs(candidates)
