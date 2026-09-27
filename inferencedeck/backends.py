from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import importlib.util
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

from .config import AppConfig
from .llama_runtimes import resolve_llama_runtime, sibling_tool
from .paths import candidate_llama_roots, executable_names, is_windows
from .schema import Environment
from .proc import run as run_hidden


DEFAULT_TIMEOUT_SECONDS = 0.8


def _request_json(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> tuple[bool, dict | list | None, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "inferencedeck/portable-core"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return True, json.loads(raw.decode("utf-8")), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, None, str(exc)


def _normalize_base_url(raw: str | None, default: str) -> str:
    value = (raw or default).strip()
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    host = parsed.hostname or ""
    if host in {"0.0.0.0", "::"}:
        replacement = "127.0.0.1"
        netloc = replacement
        if parsed.port:
            netloc = f"{replacement}:{parsed.port}"
        value = parsed._replace(netloc=netloc).geturl()
    return value.rstrip("/")


def _find_executable(base_name: str, env_vars: list[str] | None = None, roots: list[Path] | None = None) -> str | None:
    for env_var in env_vars or []:
        override = os.environ.get(env_var)
        if override and Path(override).expanduser().is_file():
            return str(Path(override).expanduser())

    for root in roots or []:
        for name in executable_names(base_name):
            for subdir in [Path("."), Path("bin"), Path("build") / "bin"]:
                candidate = root / subdir / name
                if candidate.is_file():
                    return str(candidate)

    for name in executable_names(base_name):
        found = shutil.which(name)
        if found:
            return found
    return None


def _env_file(*names: str) -> str | None:
    """An explicitly configured binary from the environment (no PATH search)."""
    for name in names:
        found = _configured_file(os.environ.get(name))
        if found:
            return found
    return None


def _configured_file(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    return str(path) if path.is_file() else None


def _configured_roots(config: AppConfig) -> list[Path]:
    roots: list[Path] = []
    for raw in config.runtime_dirs:
        if raw:
            roots.append(Path(raw).expanduser())
    for raw in [config.llama_server_path, config.llama_fit_params_path]:
        if raw:
            path = Path(raw).expanduser()
            roots.append(path.parent if path.suffix else path)
    return [path for path in roots if path.is_dir()]


def _binary_version(binary_path: str | None, timeout: float = 2.0) -> str | None:
    if not binary_path:
        return None
    try:
        result = run_hidden(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    if not output:
        return None
    for line in output.splitlines():
        if "version:" in line.lower():
            return line[:240]
    return output.splitlines()[0][:240]


def detect_llama_cpp(project_root: Path | None = None, config: AppConfig | None = None) -> Environment:
    app_config = config or AppConfig.load()
    roots = _configured_roots(app_config) + candidate_llama_roots(project_root)
    # Explicit paths are pins; the resolver still refuses them if this CPU
    # can't run them (e.g. an AVX2 build on an AVX-only Xeon).
    pinned_paths = [app_config.llama_server_path] + [
        os.environ.get(name, "") for name in ("LLAMA_SERVER", "LLAMA_SERVER_BIN")
    ]
    selection = resolve_llama_runtime(roots, app_config.llama_runtime, [p for p in pinned_paths if p])
    server = (selection.get("selected") or {}).get("path")
    # Companion tools come from the same build as the server, so an AVX2
    # llama-fit-params is never paired with an AVX1 llama-server.
    cli = _env_file("LLAMA_CLI", "LLAMA_CLI_BIN") or sibling_tool(server, "llama-cli")
    fit = (
        _configured_file(app_config.llama_fit_params_path)
        or _env_file("LLAMA_FIT_PARAMS", "LLAMA_FIT_PARAMS_BIN")
        or sibling_tool(server, "llama-fit-params")
    )
    if not server:
        cli = fit = None

    configured_url = os.environ.get("LLAMA_SERVER_URL")
    if configured_url:
        api_url = _normalize_base_url(configured_url, configured_url)
    else:
        port = os.environ.get("LLAMA_SERVER_PORT") or str(app_config.default_port or 8080)
        host = os.environ.get("LLAMA_SERVER_HOST") or app_config.default_host or "127.0.0.1"
        api_url = _normalize_base_url(f"{host}:{port}", "http://127.0.0.1:8080")

    ok, models_payload, error = _request_json(f"{api_url}/v1/models")
    model_count = None
    if ok and isinstance(models_payload, dict):
        model_count = len(models_payload.get("data", []) or [])

    warnings: list[str] = list(selection.get("warnings") or [])
    if not server:
        warnings.append(selection["reason"])

    return Environment(
        id="llama.cpp",
        kind="local_binary",
        name="llama.cpp",
        available=bool(server or ok),
        binary_path=server,
        api_url=api_url if ok else None,
        version=_binary_version(server),
        model_count=model_count,
        details={
            "llama_cli": cli,
            "llama_fit_params": fit,
            "probe_url": api_url,
            "probe_error": None if ok else error,
            "candidate_roots": [str(path) for path in roots],
            "runtime_selection": selection,
        },
        warnings=warnings,
    )


def detect_ollama() -> Environment:
    api_url = _normalize_base_url(os.environ.get("OLLAMA_HOST"), "http://127.0.0.1:11434")
    ok, payload, error = _request_json(f"{api_url}/api/tags")
    model_count = None
    if ok and isinstance(payload, dict):
        model_count = len(payload.get("models", []) or [])
    binary = _find_executable("ollama", ["OLLAMA_BIN"])
    version = _binary_version(binary) if binary else None
    return Environment(
        id="ollama",
        kind="api_runtime",
        name="Ollama",
        available=ok,
        binary_path=binary,
        api_url=api_url if ok else None,
        version=version,
        model_count=model_count,
        details={"probe_url": api_url, "probe_error": None if ok else error},
    )


def detect_lm_studio() -> Environment:
    api_url = _normalize_base_url(os.environ.get("LMSTUDIO_HOST"), "http://127.0.0.1:1234")
    ok, payload, error = _request_json(f"{api_url}/v1/models")
    model_count = None
    if ok and isinstance(payload, dict):
        model_count = len(payload.get("data", []) or [])
    return Environment(
        id="lm-studio",
        kind="api_runtime",
        name="LM Studio",
        available=ok,
        api_url=api_url if ok else None,
        model_count=model_count,
        details={"probe_url": api_url, "probe_error": None if ok else error},
    )


def detect_vllm() -> Environment:
    api_url = _normalize_base_url(os.environ.get("VLLM_HOST"), "http://127.0.0.1:8000")
    ok, payload, error = _request_json(f"{api_url}/v1/models")
    model_count = None
    if ok and isinstance(payload, dict):
        model_count = len(payload.get("data", []) or [])
    binary = _find_executable("vllm", ["VLLM_BIN"])
    return Environment(
        id="vllm",
        kind="api_or_binary_runtime",
        name="vLLM",
        available=bool(ok or binary),
        binary_path=binary,
        api_url=api_url if ok else None,
        model_count=model_count,
        details={"probe_url": api_url, "probe_error": None if ok else error},
    )


def _vllm_cpp_roots(config: AppConfig) -> list[Path]:
    roots: list[Path] = []
    home = os.environ.get("VLLM_CPP_HOME")
    if home:
        roots.append(Path(home).expanduser())
    roots.extend(Path(raw).expanduser() for raw in config.runtime_dirs if raw)
    # A source build puts vllm-server in build/examples; a release archive in bin.
    expanded: list[Path] = []
    for root in roots:
        expanded.extend([root, root / "build" / "examples"])
    return [path for path in expanded if path.is_dir()]


def detect_vllm_cpp(config: AppConfig | None = None) -> Environment:
    """vllm.cpp's ``vllm-server``: a standalone C++ engine with vLLM's serving core."""

    app_config = config or AppConfig.load()
    binary = (
        _configured_file(app_config.vllm_cpp_server_path)
        or _find_executable("vllm-server", ["VLLM_CPP_SERVER", "VLLM_CPP_SERVER_BIN"], _vllm_cpp_roots(app_config))
    )

    configured_url = os.environ.get("VLLM_CPP_SERVER_URL")
    api_url = _normalize_base_url(configured_url, "http://127.0.0.1:8000")
    ok, payload, error = _request_json(f"{api_url}/v1/models")
    model_count = None
    if ok and isinstance(payload, dict):
        model_count = len(payload.get("data", []) or [])
    version = None
    if ok:
        # The CLI has no documented --version; a running server answers /version.
        got_version, version_payload, _ = _request_json(f"{api_url}/version")
        if got_version and isinstance(version_payload, dict):
            version = version_payload.get("version")

    warnings: list[str] = []
    if not binary:
        warnings.append(
            "vllm-server was not found. Set vllm_cpp_server_path in config, VLLM_CPP_SERVER, "
            "or VLLM_CPP_HOME to a vllm.cpp build or release archive."
        )
    return Environment(
        id="vllm.cpp",
        kind="local_binary",
        name="vllm.cpp",
        available=bool(binary or ok),
        binary_path=binary,
        api_url=api_url if ok else None,
        version=str(version) if version else None,
        model_count=model_count,
        details={
            "probe_url": api_url,
            "probe_error": None if ok else error,
            "status": "alpha: CLI flags may change between vllm.cpp releases",
        },
        warnings=warnings,
    )


def _mlc_llm_package_version() -> str | None:
    """Version of the installed mlc-llm wheel (nightly and CUDA builds use suffixed names)."""

    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name == "mlc-llm" or name.startswith("mlc-llm-"):
            return dist.version
    return None


def mlc_llm_invocation(config: AppConfig | None = None) -> list[str] | None:
    """How to run the MLC LLM CLI: its console script, else ``python -m mlc_llm``."""

    app_config = config or AppConfig.load()
    script = _configured_file(app_config.mlc_llm_path) or _find_executable("mlc_llm", ["MLC_LLM_BIN"])
    if script:
        return [script]
    if importlib.util.find_spec("mlc_llm") is not None:
        return [sys.executable, "-m", "mlc_llm"]
    return None


def detect_mlc_llm(config: AppConfig | None = None) -> Environment:
    """MLC LLM's ``mlc_llm serve``: TVM-compiled models on CUDA, Metal, Vulkan, ROCm or OpenCL."""

    invocation = mlc_llm_invocation(config)
    api_url = _normalize_base_url(os.environ.get("MLC_LLM_SERVER_URL"), "http://127.0.0.1:8000")
    ok, payload, error = _request_json(f"{api_url}/v1/models")
    model_count = None
    if ok and isinstance(payload, dict):
        model_count = len(payload.get("data", []) or [])
    warnings: list[str] = []
    if not invocation:
        warnings.append(
            "MLC LLM was not found. Install the mlc-llm package, or set mlc_llm_path in config "
            "or MLC_LLM_BIN to its mlc_llm command."
        )
    return Environment(
        id="mlc-llm",
        kind="local_binary",
        name="MLC LLM",
        available=bool(invocation or ok),
        binary_path=invocation[0] if invocation else None,
        api_url=api_url if ok else None,
        version=_mlc_llm_package_version() if invocation else None,
        model_count=model_count,
        details={
            "invocation": invocation,
            "probe_url": api_url,
            "probe_error": None if ok else error,
            "model_format": "MLC weight folders (mlc-chat-config.json) or HF:// ids, not GGUF",
        },
        warnings=warnings,
    )


def detect_mlx() -> Environment:
    module_available = importlib.util.find_spec("mlx_lm") is not None
    is_macos = os.uname().sysname == "Darwin" if hasattr(os, "uname") else False
    server_binary = _find_executable("mlx_lm.server", ["MLX_LM_SERVER", "MLX_SERVER_BIN"])
    warnings: list[str] = []
    if not is_macos:
        warnings.append("MLX is intended for macOS with Apple Silicon.")
    if is_macos and not module_available and not server_binary:
        warnings.append("MLX was not found. Install mlx-lm in the Python environment to enable it.")
    return Environment(
        id="mlx",
        kind="local_runtime",
        name="MLX",
        available=bool(is_macos and (module_available or server_binary)),
        binary_path=server_binary,
        details={
            "python_module": "mlx_lm" if module_available else None,
            "acceleration_backend": "metal",
            "platform_supported": is_macos,
        },
        warnings=warnings,
    )


def detect_wsl_llama_cpp() -> Environment:
    if not is_windows():
        return Environment(
            id="wsl-llama.cpp",
            kind="wsl_runtime",
            name="WSL llama.cpp",
            available=False,
            details={"reason": "WSL detection is only relevant on Windows."},
        )

    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return Environment(
            id="wsl-llama.cpp",
            kind="wsl_runtime",
            name="WSL llama.cpp",
            available=False,
            warnings=["wsl.exe was not found on PATH."],
        )

    probe = "command -v llama-server || command -v llama.cpp/build/bin/llama-server || true"
    try:
        result = run_hidden(
            [wsl, "sh", "-lc", probe],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        binary = result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
    except (OSError, subprocess.SubprocessError) as exc:
        return Environment(
            id="wsl-llama.cpp",
            kind="wsl_runtime",
            name="WSL llama.cpp",
            available=True,
            binary_path=wsl,
            warnings=[f"WSL exists, but the llama-server probe failed: {exc}"],
        )

    return Environment(
        id="wsl-llama.cpp",
        kind="wsl_runtime",
        name="WSL llama.cpp",
        available=bool(binary),
        binary_path=binary,
        details={"wsl_binary": wsl, "probe": probe},
        warnings=[] if binary else ["WSL exists, but llama-server was not found in the default distro PATH."],
    )


def detect_all(project_root: Path | None = None, config: AppConfig | None = None) -> list[Environment]:
    return [
        detect_llama_cpp(project_root, config=config),
        detect_wsl_llama_cpp(),
        detect_ollama(),
        detect_lm_studio(),
        detect_vllm(),
        detect_vllm_cpp(config),
        detect_mlc_llm(config),
        detect_mlx(),
    ]


# Runtimes whose launch path is actually wired into prepare_launch_command.
# Others are detectable (and selectable in the UI) but cannot be started yet.
LAUNCHABLE_RUNTIMES = ("llama.cpp", "vllm.cpp", "mlc-llm")


def detect_runtime(
    runtime_id: str | None,
    project_root: Path | None = None,
    config: AppConfig | None = None,
) -> Environment | None:
    """Detect a single runtime by its environment id. Returns None if unknown."""

    if runtime_id in (None, "", "llama.cpp"):
        return detect_llama_cpp(project_root, config=config)
    if runtime_id == "wsl-llama.cpp":
        return detect_wsl_llama_cpp()
    if runtime_id == "vllm.cpp":
        return detect_vllm_cpp(config)
    if runtime_id == "mlc-llm":
        return detect_mlc_llm(config)
    detectors = {
        "ollama": detect_ollama,
        "lm-studio": detect_lm_studio,
        "vllm": detect_vllm,
        "mlx": detect_mlx,
    }
    detector = detectors.get(runtime_id)
    return detector() if detector else None
