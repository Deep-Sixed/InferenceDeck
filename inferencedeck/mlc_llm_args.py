"""Build an ``mlc_llm serve`` (MLC LLM) command line from normalized profile params.

MLC LLM (https://github.com/mlc-ai/mlc-llm) compiles models ahead of time with
Apache TVM and serves its own weight folders (the ones with an
``mlc-chat-config.json``, e.g. ``mlc-ai/*-MLC`` on Hugging Face), not GGUF. A
profile therefore names its model with ``mlc_model``: a local MLC folder or an
``HF://org/repo`` id. Engine sizing goes through ``--overrides``, a
``key=value;key=value`` string.
"""

from __future__ import annotations

from typing import Any

from .llama_args import LaunchCommand, _add_optional
from .vllm_cpp_args import LLAMA_ONLY_KEYS, PER_REQUEST_SAMPLING_KEYS

MLC_MODES = ("local", "interactive", "server")

# profile key -> EngineConfig field accepted by ``--overrides``.
OVERRIDE_KEYS = (
    ("ctx_size", "context_window_size"),
    ("max_num_seqs", "max_num_sequence"),
    ("max_total_seq_length", "max_total_seq_length"),
    ("prefill_chunk_size", "prefill_chunk_size"),
    ("gpu_memory_utilization", "gpu_memory_utilization"),
    ("tensor_parallel_shards", "tensor_parallel_shards"),
    ("sliding_window_size", "sliding_window_size"),
)


def is_hf_model(value: str) -> bool:
    return value.strip().upper().startswith("HF://")


def build_mlc_llm_serve_args(
    invocation: list[str],
    model: str,
    params: dict[str, Any],
    extra_args: list[str] | None = None,
    cwd: str | None = None,
) -> LaunchCommand:
    """Build an ``mlc_llm serve`` argv list.

    ``invocation`` is how to run the CLI: ``[mlc_llm]`` for the console script,
    or ``[python, "-m", "mlc_llm"]`` when only the Python package was found.
    """

    warnings: list[str] = []
    args = [*invocation, "serve", model]
    args.extend(["--host", str(params.get("host", "127.0.0.1")), "--port", str(int(params.get("port", 8000)))])

    mode = params.get("mlc_mode")
    if mode not in (None, ""):
        if mode not in MLC_MODES:
            warnings.append(f"mlc_mode '{mode}' is not one of {', '.join(MLC_MODES)}; mlc_llm serve will refuse it.")
        args.extend(["--mode", str(mode)])

    device = params.get("device")
    if device not in (None, "", "auto"):
        args.extend(["--device", str(device)])
    _add_optional(args, "--model-lib", params.get("model_lib"))

    if "enable_prefix_caching" in params:
        args.extend(["--prefix-cache-mode", "radix" if params["enable_prefix_caching"] else "disable"])

    overrides = [f"{field}={params[key]}" for key, field in OVERRIDE_KEYS if params.get(key) not in (None, "")]
    if overrides:
        args.extend(["--overrides", ";".join(overrides)])

    # device is an MLC flag here, so it isn't "llama-only" for this runtime.
    ignored = [key for key in LLAMA_ONLY_KEYS if key != "device" and params.get(key) not in (None, "", [], "auto")]
    if ignored:
        warnings.append(f"MLC LLM ignores llama.cpp-only settings: {', '.join(ignored)}.")
    sampling = [key for key in PER_REQUEST_SAMPLING_KEYS if params.get(key) is not None]
    if sampling:
        warnings.append(
            "mlc_llm serve takes sampling settings per request (or from the model's "
            f"mlc-chat-config.json), so these profile values are not applied: {', '.join(sampling)}."
        )
    if extra_args:
        args.extend(extra_args)

    return LaunchCommand(argv=args, cwd=cwd, warnings=warnings)
