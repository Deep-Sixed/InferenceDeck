from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .estimates import estimate_tokens_per_second
from .hardware import detect_system_hardware
from .llama_args import normalize_gpu_layers
from .server_manager import prepare_launch_command


MEMORY_RE = re.compile(
    r"(?im)^(?P<device>(?:CUDA|MTL|METAL|ROCM|HIP|VULKAN|VK|SYCL|GPU)\d*)\s+"
    r"(?P<model>\d+)\s+(?P<context>\d+)\s+(?P<compute>\d+)"
)
FREE_RE = re.compile(r"projected to use (?P<used>\d+) MiB.* vs\. (?P<free>\d+) MiB", re.IGNORECASE)
LEAVE_RE = re.compile(r"will leave (?P<remaining>\d+).*?MiB", re.IGNORECASE)

INT_FLAGS = {
    "-c": "ctx_size",
    "--ctx-size": "ctx_size",
    "-t": "threads",
    "--threads": "threads",
    "-tb": "threads_batch",
    "--threads-batch": "threads_batch",
    "-b": "batch_size",
    "--batch-size": "batch_size",
    "-ub": "ubatch_size",
    "--ubatch-size": "ubatch_size",
    "--repeat-last-n": "repeat_last_n",
    "-s": "seed",
    "--seed": "seed",
    "-n": "n_predict",
    "--predict": "n_predict",
    "--n-predict": "n_predict",
}
FLOAT_FLAGS = {
    "--temp": "temperature",
    "--temperature": "temperature",
    "--top-p": "top_p",
    "--min-p": "min_p",
    "--repeat-penalty": "repeat_penalty",
    "--presence-penalty": "presence_penalty",
    "--frequency-penalty": "frequency_penalty",
}
STRING_FLAGS = {
    "-ctk": "cache_type_k",
    "--cache-type-k": "cache_type_k",
    "-ctv": "cache_type_v",
    "--cache-type-v": "cache_type_v",
}
GPU_LAYER_FLAGS = {"-ngl", "--gpu-layers", "--n-gpu-layers"}
BOOL_FLAGS = {"-fa": "flash_attn", "--flash-attn": "flash_attn", "--reasoning": "reasoning"}
BOOL_FLAGS.update({"-kvo": "kv_offload", "--kv-offload": "kv_offload", "--op-offload": "op_offload"})
NEG_BOOL_FLAGS = {
    "-nkvo": "kv_offload",
    "--no-kv-offload": "kv_offload",
    "--no-op-offload": "op_offload",
}
INT_OR_ZERO_FLAGS = {"--top-k": "top_k"}
FIT_APPLY_KEYS = {
    "ctx_size",
    "threads",
    "threads_batch",
    "batch_size",
    "ubatch_size",
    "gpu_layers",
    "cache_type_k",
    "cache_type_v",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "n_predict",
    "flash_attn",
    "reasoning",
    "kv_offload",
    "op_offload",
}


def build_fit_args(fit_binary: str, model_path: str, params: dict[str, Any], target_mib: int = 1024) -> list[str]:
    gpu_layers = normalize_gpu_layers(params.get("gpu_layers"))
    if gpu_layers is None:
        gpu_layers = 999
    gpu_layers_arg = "-2" if gpu_layers >= 999 else str(gpu_layers)
    args = [
        fit_binary,
        "-m",
        model_path,
        "-c",
        str(int(params.get("ctx_size", 4096))),
        "-ngl",
        gpu_layers_arg,
        "-t",
        str(int(params.get("threads", 4))),
        "-tb",
        str(int(params.get("threads_batch", params.get("threads", 4)))),
        "-ctk",
        str(params.get("cache_type_k", "q4_0")),
        "-ctv",
        str(params.get("cache_type_v", "q4_0")),
        "-b",
        str(int(params.get("batch_size", 512))),
        "-ub",
        str(int(params.get("ubatch_size", 512))),
        "-fit",
        "on",
        "-fitp",
        "on",
        "-fitt",
        str(int(target_mib)),
    ]
    if params.get("flash_attn", True):
        args.extend(["-fa", "on"])
    args.append("-kvo" if params.get("kv_offload", True) else "-nkvo")
    args.append("--op-offload" if params.get("op_offload", True) else "--no-op-offload")
    if "temperature" in params:
        args.extend(["--temp", str(params["temperature"])])
    if "top_k" in params:
        args.extend(["--top-k", str(params["top_k"])])
    if "top_p" in params:
        args.extend(["--top-p", str(params["top_p"])])
    if "min_p" in params:
        args.extend(["--min-p", str(params["min_p"])])
    if "repeat_penalty" in params:
        args.extend(["--repeat-penalty", str(params["repeat_penalty"])])
    if "repeat_last_n" in params:
        args.extend(["--repeat-last-n", str(params["repeat_last_n"])])
    if "presence_penalty" in params:
        args.extend(["--presence-penalty", str(params["presence_penalty"])])
    if "frequency_penalty" in params:
        args.extend(["--frequency-penalty", str(params["frequency_penalty"])])
    if "seed" in params:
        args.extend(["-s", str(params["seed"])])
    if "n_predict" in params:
        args.extend(["--predict", str(params["n_predict"])])
    if "reasoning" in params and params["reasoning"]:
        args.append("--reasoning")
    return args


def _find_fitted_args(stdout: str) -> str | None:
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        has_context = " -c " in f" {line} " or " --ctx-size " in f" {line} "
        has_layers = any(f" {flag} " in f" {line} " for flag in GPU_LAYER_FLAGS)
        if not has_context or not has_layers:
            continue
        # Slice from the earliest flag so a leading prose prefix is dropped but
        # no flag is lost if -ngl happens to precede -c in the tool's output.
        markers = ["-c ", "--ctx-size "] + [f"{flag} " for flag in GPU_LAYER_FLAGS]
        positions = [pos for pos in (line.find(marker) for marker in markers) if pos >= 0]
        if positions:
            return line[min(positions):]
    return None


def _tokens(command_line: str) -> list[str]:
    try:
        return shlex.split(command_line, posix=False)
    except ValueError:
        return command_line.split()


def _bool_value(value: str | None) -> bool:
    if value is None:
        return True
    return value.lower() not in {"0", "false", "off", "no"}


def parse_fitted_args(command_line: str) -> dict[str, Any]:
    suggestions: dict[str, Any] = {"fitted_args": command_line.strip()}
    tokens = _tokens(command_line)
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        next_value = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if token in GPU_LAYER_FLAGS and next_value is not None:
            suggestions["gpu_layers"] = 999 if next_value in {"-2", "all", "auto"} else int(next_value)
            idx += 2
            continue
        if token in INT_FLAGS and next_value is not None:
            suggestions[INT_FLAGS[token]] = int(float(next_value))
            idx += 2
            continue
        if token in INT_OR_ZERO_FLAGS and next_value is not None:
            suggestions[INT_OR_ZERO_FLAGS[token]] = int(float(next_value))
            idx += 2
            continue
        if token in FLOAT_FLAGS and next_value is not None:
            suggestions[FLOAT_FLAGS[token]] = float(next_value)
            idx += 2
            continue
        if token in STRING_FLAGS and next_value is not None:
            suggestions[STRING_FLAGS[token]] = str(next_value)
            idx += 2
            continue
        if token in BOOL_FLAGS:
            consumes_value = next_value is not None and not next_value.startswith("-")
            suggestions[BOOL_FLAGS[token]] = _bool_value(next_value if consumes_value else None)
            idx += 2 if consumes_value else 1
            continue
        if token in NEG_BOOL_FLAGS:
            suggestions[NEG_BOOL_FLAGS[token]] = False
            idx += 1
            continue
        idx += 1
    return suggestions


def apply_fit_suggestions(params: dict[str, Any], suggestions: dict[str, Any], target_mib: int | None = None) -> dict[str, Any]:
    applied = dict(params)
    for key in FIT_APPLY_KEYS:
        if key in suggestions:
            applied[key] = suggestions[key]
    if target_mib is not None:
        applied["fit_target_mib"] = int(target_mib)
    if "headroom_mib" in suggestions:
        applied["fit_headroom_mib"] = suggestions["headroom_mib"]
    if "cuda_memory_mib" in suggestions:
        applied["fit_cuda_memory_mib"] = suggestions["cuda_memory_mib"]
    return applied


def parse_fit_output(stdout: str, stderr: str = "") -> dict[str, Any]:
    combined = "\n".join(part for part in [stdout, stderr] if part)
    fitted_line = _find_fitted_args(stdout)
    memory = MEMORY_RE.search(combined)
    free_match = FREE_RE.search(combined)
    leave_match = LEAVE_RE.search(combined)
    suggestions: dict[str, Any] = {}
    notes: list[str] = []

    if fitted_line:
        suggestions.update(parse_fitted_args(fitted_line))
        notes.append("llama-fit-params returned fitted CLI arguments.")

    if memory:
        model_mib = int(memory.group("model"))
        context_mib = int(memory.group("context"))
        compute_mib = int(memory.group("compute"))
        suggestions["cuda_memory_mib"] = {
            "model": model_mib,
            "context": context_mib,
            "compute": compute_mib,
            "projected": model_mib + context_mib + compute_mib,
        }
        notes.append(f"Parsed {memory.group('device')} memory estimate from llama-fit-params output.")

    if free_match:
        used = int(free_match.group("used"))
        free = int(free_match.group("free"))
        suggestions["projected_device_mib"] = used
        suggestions["free_device_mib"] = free
        suggestions["headroom_mib"] = max(free - used, 0)
    if leave_match:
        suggestions["headroom_mib"] = int(leave_match.group("remaining"))

    if not suggestions:
        notes.append("No structured fit suggestion could be parsed; review stdout/stderr.")

    return {"suggestions": suggestions, "notes": notes}


def run_fit_test(
    mode: str,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    target_mib: int = 1024,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    prepared = prepare_launch_command(mode, project_root=project_root, model_dirs=model_dirs, overrides=overrides)
    if not prepared.get("success"):
        return prepared

    fit_binary = prepared.get("environment", {}).get("details", {}).get("llama_fit_params")
    if not fit_binary or not Path(fit_binary).is_file():
        return {"success": False, "error": "llama-fit-params was not found.", "prepared": prepared}

    model_path = prepared["profile"]["model"]["path"]
    args = build_fit_args(fit_binary, model_path, prepared["params"], target_mib)
    cwd = str(Path(fit_binary).parent)
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
            check=False,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc), "prepared": prepared, "argv": args}

    parsed = parse_fit_output(result.stdout, result.stderr)
    applied_params = apply_fit_suggestions(prepared["params"], parsed["suggestions"], target_mib)
    hardware = detect_system_hardware()
    speed_estimate = estimate_tokens_per_second(
        applied_params,
        prepared.get("profile", {}).get("model"),
        hardware,
    )
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "argv": args,
        "command_line": subprocess.list2cmdline(args),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "prepared": prepared,
        "applied_params": applied_params,
        "hardware": hardware,
        "speed_estimate": speed_estimate,
        **parsed,
    }
