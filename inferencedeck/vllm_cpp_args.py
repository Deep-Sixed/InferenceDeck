"""Build a ``vllm-server`` (vllm.cpp) command line from normalized profile params.

vllm.cpp (https://github.com/mudler/vllm.cpp) is a standalone C++ engine with
vLLM's serving core: continuous batching, a block-paged KV pool and prefix
caching. Its flags follow vLLM's, not llama.cpp's, so profiles are translated
here rather than passed through ``build_llama_server_args``.

The project is pre-release and its CLI moves between commits; only flags
documented in its ``docs/reference/server.md`` are emitted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .llama_args import LaunchCommand, _add_optional

# vllm-server's default --block-size; it must be a multiple of 16.
DEFAULT_BLOCK_SIZE = 32

# Profile keys that only mean something to llama-server. A vllm.cpp profile
# that sets them gets a warning instead of a silently different launch.
LLAMA_ONLY_KEYS = (
    "gpu_layers",
    "threads",
    "threads_batch",
    "batch_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
    "cache_ram_mib",
    "cache_reuse",
    "slot_prompt_similarity",
    "reasoning_budget",
    "kv_offload",
    "op_offload",
    "mmap",
    "draft_model",
    "spec_type",
    "tensor_overrides",
    "override_tensors",
    "ot",
    "device",
    "cuda_device",
)

# Sampling defaults are per-request in vllm.cpp (or come from the checkpoint's
# generation_config.json); vllm-server has no flags for them.
PER_REQUEST_SAMPLING_KEYS = (
    "n_predict",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_last_n",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
)


def _positive_int(value: Any) -> int | None:
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def build_vllm_cpp_server_args(
    vllm_server: str,
    model_path: str,
    params: dict[str, Any],
    extra_args: list[str] | None = None,
) -> LaunchCommand:
    """Build a ``vllm-server`` argv list from normalized profile params."""

    warnings: list[str] = []
    args = [
        vllm_server,
        "--model",
        model_path,
        "--host",
        str(params.get("host", "127.0.0.1")),
        "--port",
        str(int(params.get("port", 8000))),
        "--served-model-name",
        str(params.get("alias", Path(model_path).stem)),
    ]

    block_size = _positive_int(params.get("block_size"))
    if block_size is not None:
        if block_size % 16:
            warnings.append(f"block_size {block_size} is not a multiple of 16; vllm-server will refuse it.")
        args.extend(["--block-size", str(block_size)])

    ctx_size = _positive_int(params.get("ctx_size"))
    if ctx_size is not None:
        args.extend(["--max-model-len", str(ctx_size)])

    # The KV pool defaults to 256 blocks x 32 tokens = 8192 tokens, and
    # vllm-server exits at startup when --max-model-len can't fit in it. Unless
    # the profile sizes the pool itself, size it to hold one full-length
    # sequence so the Context presets work the way they do for llama.cpp.
    kv_cache_mib = _positive_int(params.get("kv_cache_memory_mib"))
    num_blocks = _positive_int(params.get("num_blocks"))
    if num_blocks is not None:
        args.extend(["--num-blocks", str(num_blocks)])
    elif kv_cache_mib is not None:
        args.extend(["--kv-cache-memory", str(kv_cache_mib * 1024 * 1024)])
    elif ctx_size is not None:
        blocks = math.ceil(ctx_size / (block_size or DEFAULT_BLOCK_SIZE))
        args.extend(["--num-blocks", str(blocks)])

    mapping = [
        ("max_num_seqs", "--max-num-seqs"),
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("kv_cache_dtype", "--kv-cache-dtype"),
        ("scheduling_policy", "--scheduling-policy"),
        ("tool_call_parser", "--tool-call-parser"),
        ("reasoning_parser", "--reasoning-parser"),
        ("generation_config", "--generation-config"),
        ("tokenizer_config", "--tokenizer-config"),
        ("mmproj", "--mmproj"),
    ]
    for key, flag in mapping:
        _add_optional(args, flag, params.get(key))

    if "enable_prefix_caching" in params:
        args.append("--enable-prefix-caching" if params["enable_prefix_caching"] else "--no-enable-prefix-caching")

    # Only emitted when the profile says so: with neither flag the chat
    # template keeps its own default, which is not the same as "off".
    if "reasoning" in params:
        args.append("--enable-thinking" if params["reasoning"] else "--no-enable-thinking")

    speculative = params.get("speculative_config")
    if speculative:
        args.extend(["--speculative-config", speculative if isinstance(speculative, str) else json.dumps(speculative)])

    ignored = [key for key in LLAMA_ONLY_KEYS if params.get(key) not in (None, "", [], "auto")]
    if ignored:
        warnings.append(f"vllm.cpp ignores llama.cpp-only settings: {', '.join(ignored)}.")
    sampling = [key for key in PER_REQUEST_SAMPLING_KEYS if params.get(key) is not None]
    if sampling:
        warnings.append(
            "vllm-server takes sampling settings per request (or from the checkpoint's "
            f"generation_config.json), so these profile values are not applied: {', '.join(sampling)}."
        )

    if extra_args:
        args.extend(extra_args)

    return LaunchCommand(argv=args, cwd=str(Path(vllm_server).parent), warnings=warnings)
