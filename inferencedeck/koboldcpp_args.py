"""Build a KoboldCpp command line from normalized profile params.

KoboldCpp (https://github.com/LostRuins/koboldcpp) is a llama.cpp fork shipped as
one executable (or ``koboldcpp.py``). It loads the same GGUF files and serves an
OpenAI-compatible API next to its KoboldAI one, so llama.cpp profiles translate
almost one to one; only the flag names and a few value sets differ.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .llama_args import LaunchCommand, _add_optional, normalize_gpu_layers
from .vllm_cpp_args import PER_REQUEST_SAMPLING_KEYS

# --batchsize only accepts these; anything else makes KoboldCpp exit at startup.
BATCH_SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
# --quantkv's KV cache types (one setting for both K and V).
KV_CACHE_TYPES = ("f16", "bf16", "q8_0", "q5_1", "q4_0")
# --defaultgenamt's accepted range.
DEFAULT_GEN_RANGE = (64, 32768)

# llama.cpp profile keys KoboldCpp has no flag for. A profile that sets them
# gets a warning instead of a silently different launch.
UNSUPPORTED_KEYS = (
    "ubatch_size",
    "cache_ram_mib",
    "cache_reuse",
    "slot_prompt_similarity",
    "reasoning_budget",
    "spec_type",
)

# acceleration_backend -> KoboldCpp's backend flag. Anything else (auto, metal)
# leaves the choice to KoboldCpp, which picks CUDA, Vulkan or CPU itself and
# falls back to its no-AVX2 and failsafe modes on older CPUs.
BACKEND_FLAGS = {"cuda": "--usecuda", "rocm": "--usecuda", "hip": "--usecuda", "vulkan": "--usevulkan", "cpu": "--usecpu"}


def _snap_batch_size(value: Any, warnings: list[str]) -> int | None:
    try:
        size = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    if size in BATCH_SIZES:
        return size
    snapped = max((s for s in BATCH_SIZES if s <= size), default=BATCH_SIZES[0])
    warnings.append(f"KoboldCpp only accepts batch sizes {', '.join(map(str, BATCH_SIZES))}; using {snapped} for {size}.")
    return snapped


def _device_index(device: Any) -> str | None:
    """A GPU index from device values like 1, "1", "CUDA1", "cuda:1" or "Vulkan0"."""
    match = re.search(r"(\d+)$", str(device or "").strip())
    return match.group(1) if match else None


def build_koboldcpp_args(
    invocation: list[str],
    model_path: str,
    params: dict[str, Any],
    extra_args: list[str] | None = None,
) -> LaunchCommand:
    """Build a KoboldCpp argv list.

    ``invocation`` is the executable, or ``[python, "koboldcpp.py"]`` for a
    source checkout.
    """

    warnings: list[str] = []
    args = [
        *invocation,
        "--model",
        model_path,
        "--host",
        str(params.get("host", "127.0.0.1")),
        "--port",
        str(int(params.get("port", 5001))),
        # Never open the Tk launcher: this is a headless, managed server.
        "--skiplauncher",
    ]

    backend = str(params.get("acceleration_backend") or "").strip().lower()
    backend_flag = BACKEND_FLAGS.get(backend)
    if backend_flag:
        args.append(backend_flag)
        index = _device_index(params.get("device", params.get("cuda_device")))
        if index is not None and backend_flag != "--usecpu":
            args.append(index)

    mapping = [
        ("ctx_size", "--contextsize"),
        ("threads", "--threads"),
        ("threads_batch", "--blasthreads"),
        ("mmproj", "--mmproj"),
    ]
    for key, flag in mapping:
        _add_optional(args, flag, params.get(key))

    if params.get("batch_size") is not None:
        size = _snap_batch_size(params["batch_size"], warnings)
        if size is not None:
            args.extend(["--batchsize", str(size)])

    if backend == "cpu":
        args.extend(["--gpulayers", "0"])
    elif str(params.get("gpu_layers", "")).strip().lower() == "auto":
        pass  # KoboldCpp's default (-1) is its own autofit.
    else:
        gpu_layers = normalize_gpu_layers(params.get("gpu_layers"))
        if gpu_layers is not None:
            args.extend(["--gpulayers", str(gpu_layers)])

    cache_k, cache_v = params.get("cache_type_k"), params.get("cache_type_v")
    cache_type = cache_k or cache_v
    if cache_type:
        if cache_type not in KV_CACHE_TYPES:
            warnings.append(f"KoboldCpp's KV cache types are {', '.join(KV_CACHE_TYPES)}; {cache_type} was not applied.")
        else:
            args.extend(["--quantkv", str(cache_type)])
            if cache_k and cache_v and cache_k != cache_v:
                warnings.append(f"KoboldCpp uses one KV cache type for K and V; using {cache_type} for both.")

    # Flash attention is on by default in KoboldCpp; only turning it off needs a flag.
    if params.get("flash_attn") is False:
        args.append("--noflashattention")
    if params.get("kv_offload") is False:
        args.append("--lowvram")  # KoboldCpp's name for keeping the KV cache off the GPU
    if params.get("mmap"):
        args.append("--usemmap")

    # llama.cpp's --jinja uses the model's template and its tool-call parser;
    # KoboldCpp's equivalent is --jinja_tools (--jinja alone skips tool calls).
    if params.get("jinja"):
        args.append("--jinja_tools")
        if "reasoning" in params:
            args.extend(["--jinjathink", "true" if params["reasoning"] else "false"])

    draft_model = str(params.get("draft_model", "")).strip()
    if draft_model:
        args.extend(["--draftmodel", draft_model])
        draft_amount = params.get("spec_draft_n_max", params.get("draft_max"))
        _add_optional(args, "--draftamount", draft_amount)

    tensor_overrides = params.get("tensor_overrides") or params.get("override_tensors") or params.get("ot")
    if tensor_overrides:
        # One flag, comma-separated, as llama.cpp's -ot also accepts.
        joined = ",".join(map(str, tensor_overrides)) if isinstance(tensor_overrides, list) else str(tensor_overrides)
        args.extend(["--overridetensors", joined])

    n_predict = params.get("n_predict")
    if n_predict is not None:
        low, high = DEFAULT_GEN_RANGE
        try:
            amount = int(n_predict)
        except (TypeError, ValueError):
            amount = -1
        if low <= amount <= high:
            args.extend(["--defaultgenamt", str(amount)])
        else:
            warnings.append(f"n_predict {n_predict} is outside KoboldCpp's --defaultgenamt range {low}-{high}; not applied.")

    ignored = [key for key in UNSUPPORTED_KEYS if params.get(key) not in (None, "", [])]
    if params.get("op_offload") is False:
        ignored.append("op_offload")
    if ignored:
        warnings.append(f"KoboldCpp has no setting for: {', '.join(ignored)}.")
    sampling = [key for key in PER_REQUEST_SAMPLING_KEYS if key != "n_predict" and params.get(key) is not None]
    if sampling:
        warnings.append(
            "KoboldCpp takes sampling settings per request, so these profile values are not applied: "
            f"{', '.join(sampling)}."
        )

    if extra_args:
        args.extend(extra_args)

    cwd = str(Path(invocation[-1]).parent)
    return LaunchCommand(argv=args, cwd=cwd, warnings=warnings)
