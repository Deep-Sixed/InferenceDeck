"""Validate InferenceDeck's configuration files without starting anything.

Three files shape what InferenceDeck does, and all three used to fail quietly:
``AppConfig.load`` falls back to defaults on bad JSON and drops unknown keys,
and a typo'd profile param is simply never read. This checks them:

- ``config.json`` (app settings, see config.AppConfig),
- ``models.json`` (the profiles and their ``recommended_params``),
- ``remote_endpoints/*.json`` (remote and cloud endpoints, see remotes.py).

Each file is checked against a JSON Schema (``SCHEMAS``; the same documents are
committed under ``schemas/`` for editors) with a small validator for the subset
those schemas use, plus checks a schema cannot express: paths that do not
exist, duplicate profile modes, runtimes that cannot be launched, and unknown
keys with a "did you mean" hint. Nothing is launched, no hardware is probed and
no port is opened.

Problems are errors (the file will not work as written) or warnings (it will
work, but probably not as meant). ``check_all`` returns them as a report;
``inferencedeck config validate`` and ``inferencedeck-web --check-config`` print it.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .api_params import CACHE_TYPES
from .backends import LAUNCHABLE_RUNTIMES
from .capabilities import MODALITIES
from .config import CONFIG_FILENAME, AppConfig
from .paths import config_dir, find_project_root
from .remotes import VALID_LANES, VALID_TRANSPORTS, _parse as parse_endpoint, endpoints_dir
from .runtime_updates import SUPPORTED_CHANNELS
from .sampling import NO_PRESET, SAMPLING_PRESETS

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Runtimes detect_runtime knows; only LAUNCHABLE_RUNTIMES can be started.
KNOWN_RUNTIMES = (*LAUNCHABLE_RUNTIMES, "wsl-llama.cpp", "ollama", "lm-studio", "vllm", "mlx")

_POSITIVE = {"type": "integer", "minimum": 1}
_NON_NEGATIVE = {"type": "integer", "minimum": 0}
_UNIT = {"type": "number", "minimum": 0, "maximum": 1}
_STRING_LIST = {"type": "array", "items": {"type": "string"}}
# Lets a file name its schema for editors; InferenceDeck ignores it.
_SCHEMA_REF = {"type": "string"}

CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": SCHEMA_DIALECT,
    "title": "InferenceDeck config.json",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "$schema": _SCHEMA_REF,
        "model_dirs": {**_STRING_LIST, "description": "Extra folders scanned for GGUF models."},
        "default_host": {"type": "string", "minLength": 1},
        "default_port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "default_backend": {"type": "string"},
        "runtime_dirs": {**_STRING_LIST, "description": "Extra folders searched for runtime builds."},
        "llama_server_path": {"type": "string"},
        "llama_runtime": {"type": "string", "description": "\"auto\", a runtime id such as \"cuda-avx1\", or a path."},
        "llama_fit_params_path": {"type": "string"},
        "extra_llama_args": _STRING_LIST,
        "vllm_cpp_server_path": {"type": "string"},
        "extra_vllm_cpp_args": _STRING_LIST,
        "mlc_llm_path": {"type": "string"},
        "extra_mlc_llm_args": _STRING_LIST,
        "update_channel": {"enum": list(SUPPORTED_CHANNELS)},
        "profile_names": {"type": "object", "additionalProperties": {"type": "string"}},
        "server_history_limit": _NON_NEGATIVE,
        "auto_generate_launch_scripts": {"type": "boolean"},
        "auto_scan_on_startup": {"type": "boolean"},
        "idle_release_seconds": {**_NON_NEGATIVE, "description": "Release a server's GPU after this long idle; 0 = off."},
        "concurrent_vram_check": {"enum": ["block", "warn", "off"]},
    },
}

# Profile params with a known type. Other keys the launch code reads are in
# EXTRA_PARAMS; anything else gets an "unknown param" warning.
PARAM_PROPERTIES: dict[str, Any] = {
    "runtime": {"enum": list(KNOWN_RUNTIMES)},
    "host": {"type": "string", "minLength": 1},
    "port": {"type": "integer", "minimum": 1, "maximum": 65535},
    "alias": {"type": "string"},
    "ctx_size": _POSITIVE,
    "gpu_layers": {"anyOf": [_NON_NEGATIVE, {"enum": ["all", "auto", "max"]}]},
    "threads": _POSITIVE,
    "threads_batch": _POSITIVE,
    "batch_size": _POSITIVE,
    "ubatch_size": _POSITIVE,
    "cache_type_k": {"enum": sorted(CACHE_TYPES)},
    "cache_type_v": {"enum": sorted(CACHE_TYPES)},
    "cache_ram_mib": {"type": "integer", "minimum": -1},
    "cache_reuse": _NON_NEGATIVE,
    "flash_attn": {"type": "boolean"},
    "jinja": {"type": "boolean"},
    "reasoning": {"type": "boolean"},
    "reasoning_budget": {"type": "integer", "minimum": -1},
    "mmap": {"type": "boolean"},
    "kv_offload": {"type": "boolean"},
    "op_offload": {"type": "boolean"},
    "embedding": {"type": "boolean"},
    "reranking": {"type": "boolean"},
    "vision": {"type": "boolean"},
    "mmproj": {"type": "string", "minLength": 1},
    "n_predict": {"type": "integer", "minimum": -1},
    "seed": {"type": "integer", "minimum": -1},
    "temperature": {"type": "number", "minimum": 0, "maximum": 5},
    "top_k": _NON_NEGATIVE,
    "top_p": _UNIT,
    "min_p": _UNIT,
    "repeat_penalty": {"type": "number", "minimum": 0},
    "repeat_last_n": {"type": "integer", "minimum": -1},
    "presence_penalty": {"type": "number", "minimum": -2, "maximum": 2},
    "frequency_penalty": {"type": "number", "minimum": -2, "maximum": 2},
    "slot_prompt_similarity": _UNIT,
    "sampling_preset": {"enum": [NO_PRESET, *SAMPLING_PRESETS]},
    "chat_template_kwargs": {"type": ["object", "string"]},
    "idle_release_seconds": _NON_NEGATIVE,
    "draft_model": {"type": "string"},
    "draft_max": _NON_NEGATIVE,
    "draft_min": _NON_NEGATIVE,
    "draft_p_min": _UNIT,
    "spec_draft_n_max": _NON_NEGATIVE,
    "tensor_overrides": {"type": ["string", "array"], "items": {"type": "string"}},
    "fit_target_mib": _NON_NEGATIVE,
    "block_size": _POSITIVE,
    "num_blocks": _POSITIVE,
    "kv_cache_memory_mib": _POSITIVE,
    "max_num_seqs": _POSITIVE,
    "max_num_batched_tokens": _POSITIVE,
    "gpu_memory_utilization": _UNIT,
    "capabilities": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input": {"type": "array", "items": {"enum": list(MODALITIES)}},
            "output": {"type": "array", "items": {"enum": ["text", "embedding", "score"]}},
            "tools": {"type": "boolean"},
            "embedding": {"type": "boolean"},
            "reranker": {"type": "boolean"},
            "context_max": _POSITIVE,
        },
    },
}

# Read by the launch code (llama.cpp, vllm.cpp, MLC LLM) without a fixed type here.
EXTRA_PARAMS = {
    "acceleration_backend", "cuda_device", "device", "ot", "override_tensors", "target_mib",
    "spec_type", "speculative_config", "enable_prefix_caching", "kv_cache_dtype", "generation_config",
    "reasoning_parser", "scheduling_policy", "tokenizer_config", "tool_call_parser",
    "mlc_mode", "mlc_model", "model_lib", "context_window_size", "prefill_chunk_size",
    "max_num_sequence", "max_total_seq_length", "sliding_window_size", "tensor_parallel_shards",
}
KNOWN_PARAMS = set(PARAM_PROPERTIES) | EXTRA_PARAMS

PROFILES_SCHEMA: dict[str, Any] = {
    "$schema": SCHEMA_DIALECT,
    "title": "InferenceDeck models.json (profiles)",
    "type": "object",
    "required": ["models"],
    "properties": {
        "$schema": _SCHEMA_REF,
        "models": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["mode"],
                "properties": {
                    "mode": {"type": "string", "minLength": 1},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "script": {"type": "string"},
                    "model_size_gb": {"type": "number", "minimum": 0},
                    "recommended_params": {"type": "object", "properties": PARAM_PROPERTIES},
                },
            },
        },
    },
}

ENDPOINT_SCHEMA: dict[str, Any] = {
    "$schema": SCHEMA_DIALECT,
    "title": "InferenceDeck remote endpoint (remote_endpoints/*.json)",
    "type": "object",
    "required": ["baseUrl"],
    "additionalProperties": False,
    "properties": {
        "$schema": _SCHEMA_REF,
        "provider": {"type": "string"},
        "enabled": {"type": "boolean"},
        "lane": {"enum": sorted(VALID_LANES)},
        "model": {"type": ["string", "null"]},
        "baseUrl": {"type": "string", "minLength": 1},
        "apiKeyEnv": {"type": "string", "description": "Name of the environment variable holding the key, never the key."},
        "displayName": {"type": "string"},
        "contextSize": {"type": ["integer", "null"], "minimum": 1},
        "tags": _STRING_LIST,
        "transport": {"enum": sorted(VALID_TRANSPORTS)},
        "host": {"type": "string"},
    },
}

SCHEMAS = {"config": CONFIG_SCHEMA, "profiles": PROFILES_SCHEMA, "endpoint": ENDPOINT_SCHEMA}


@dataclass
class Problem:
    severity: str  # "error" or "warning"
    file: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "file": self.file, "path": self.path, "message": self.message}


_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else key


def _suggest(key: str, known: Any) -> str:
    close = difflib.get_close_matches(key.replace("-", "_"), list(known), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def validate_schema(value: Any, schema: dict[str, Any], path: str = "") -> list[tuple[str, str, str]]:
    """(severity, path, message) for ``value`` against a JSON Schema subset.

    Supports type, enum, minimum, maximum, minLength, anyOf, required, items,
    properties and additionalProperties. Unknown object keys are warnings;
    everything else is an error.
    """

    if "anyOf" in schema:
        if any(not [p for p in validate_schema(value, option, path) if p[0] == "error"] for option in schema["anyOf"]):
            return []
        return [("error", path, f"{value!r} does not match any allowed form")]
    if "enum" in schema:
        if value not in schema["enum"]:
            return [("error", path, f"{value!r} is not one of {', '.join(map(repr, schema['enum']))}")]
        return []
    types = schema.get("type")
    if types:
        allowed = [types] if isinstance(types, str) else list(types)
        if not any(_TYPES[t](value) for t in allowed):
            return [("error", path, f"must be {' or '.join(allowed)}, not {type(value).__name__}")]
    problems: list[tuple[str, str, str]] = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(("error", path, f"must be at least {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(("error", path, f"must be at most {schema['maximum']}"))
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        problems.append(("error", path, "must not be empty"))
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            problems.extend(validate_schema(item, schema["items"], _join(path, index)))
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                problems.append(("error", _join(path, key), "is required"))
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                problems.extend(validate_schema(item, properties[key], _join(path, key)))
            elif extra is False:
                problems.append(("warning", _join(path, key), f"unknown key{_suggest(key, properties)}; it is ignored"))
            elif isinstance(extra, dict):
                problems.extend(validate_schema(item, extra, _join(path, key)))
    return problems


def _read_json(path: Path, label: str, problems: list[Problem]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        problems.append(Problem("error", label, "", f"cannot be read: {exc}"))
    except json.JSONDecodeError as exc:
        problems.append(Problem("error", label, "", f"is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"))
    return None


def _schema_problems(data: Any, schema: dict[str, Any], label: str) -> list[Problem]:
    return [Problem(severity, label, path, message) for severity, path, message in validate_schema(data, schema)]


def check_config(path: Path | None = None) -> list[Problem]:
    config_path = path or config_dir() / CONFIG_FILENAME
    label = str(config_path)
    if not config_path.is_file():
        return []  # no file: defaults, which are valid
    problems: list[Problem] = []
    data = _read_json(config_path, label, problems)
    if data is None:
        problems.append(Problem("error", label, "", "InferenceDeck ignores the whole file and runs on defaults."))
        return problems
    problems += _schema_problems(data, CONFIG_SCHEMA, label)
    if not isinstance(data, dict):
        return problems
    for key in ("llama_server_path", "llama_fit_params_path", "vllm_cpp_server_path", "mlc_llm_path"):
        value = data.get(key)
        if isinstance(value, str) and value.strip() and not Path(value).expanduser().exists():
            problems.append(Problem("warning", label, key, f"{value} does not exist"))
    for key in ("model_dirs", "runtime_dirs"):
        values = data.get(key)
        for index, value in enumerate(values if isinstance(values, list) else []):
            if isinstance(value, str) and not Path(value).expanduser().is_dir():
                problems.append(Problem("warning", label, _join(key, index), f"{value} is not a folder"))
    return problems


def check_profiles(project_root: Path | None = None, manifest: Path | None = None) -> list[Problem]:
    root = project_root or find_project_root()
    path = manifest or (root / "models.json" if root else None)
    if not path or not path.is_file():
        return []
    label = str(path)
    problems: list[Problem] = []
    data = _read_json(path, label, problems)
    if data is None:
        problems.append(Problem("error", label, "", "No profiles can be loaded from it."))
        return problems
    problems += _schema_problems(data, PROFILES_SCHEMA, label)
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return problems
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        where = _join("models", index)
        mode = entry.get("mode")
        if isinstance(mode, str) and mode:
            if mode in seen:
                problems.append(Problem("error", label, _join(where, "mode"), f"{mode!r} is also used by models[{seen[mode]}]"))
            seen.setdefault(mode, index)
        params = entry.get("recommended_params")
        if not isinstance(params, dict):
            continue
        for key in params:
            if key not in KNOWN_PARAMS:
                problems.append(Problem(
                    "warning", label, _join(_join(where, "recommended_params"), key),
                    f"unknown param{_suggest(key, KNOWN_PARAMS)}; it is ignored",
                ))
        runtime = params.get("runtime")
        if isinstance(runtime, str) and runtime in KNOWN_RUNTIMES and runtime not in LAUNCHABLE_RUNTIMES:
            problems.append(Problem(
                "warning", label, _join(_join(where, "recommended_params"), "runtime"),
                f"{runtime} is detected but cannot be launched; use one of {', '.join(LAUNCHABLE_RUNTIMES)}",
            ))
        kwargs = params.get("chat_template_kwargs")
        if isinstance(kwargs, str) and kwargs.strip():
            try:
                parsed = json.loads(kwargs)
            except json.JSONDecodeError:
                parsed = None
            if not isinstance(parsed, dict):
                problems.append(Problem("error", label, _join(_join(where, "recommended_params"), "chat_template_kwargs"),
                                        "must be a JSON object"))
    return problems


def check_endpoints(directory: Path | None = None) -> list[Problem]:
    folder = directory or endpoints_dir()
    if not folder.is_dir():
        return []
    problems: list[Problem] = []
    for path in sorted(folder.glob("*.json")):
        label = str(path)
        data = _read_json(path, label, problems)
        if data is None:
            continue
        schema_problems = _schema_problems(data, ENDPOINT_SCHEMA, label)
        for secret in ("apiKey", "api_key"):
            if isinstance(data, dict) and secret in data:
                schema_problems = [p for p in schema_problems if p.path != secret]
                schema_problems.append(Problem("error", label, secret, "API keys must not be stored here; use apiKeyEnv"))
        problems += schema_problems
        endpoint = parse_endpoint(path)
        if not endpoint.valid and not any(p.severity == "error" for p in schema_problems):
            problems.append(Problem("error", label, "", endpoint.error))
    return problems


def check_all(
    project_root: Path | None = None,
    config_path: Path | None = None,
    endpoints: Path | None = None,
) -> dict[str, Any]:
    """Every problem in the config, profiles and remote endpoint files."""

    problems = check_config(config_path) + check_profiles(project_root) + check_endpoints(endpoints)
    errors = [p.to_dict() for p in problems if p.severity == "error"]
    warnings = [p.to_dict() for p in problems if p.severity == "warning"]
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def format_report(report: dict[str, Any]) -> str:
    lines = []
    for severity in ("errors", "warnings"):
        for item in report[severity]:
            where = f" {item['path']}" if item["path"] else ""
            lines.append(f"{item['severity'].upper():7} {item['file']}{where}: {item['message']}")
    if not lines:
        return "Configuration OK."
    lines.append(f"{len(report['errors'])} error(s), {len(report['warnings'])} warning(s).")
    return "\n".join(lines)


def config_field_names() -> set[str]:
    return {f.name for f in fields(AppConfig)}
