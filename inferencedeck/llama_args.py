from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LaunchCommand:
    argv: list[str]
    cwd: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def command_line(self) -> str:
        return subprocess.list2cmdline(self.argv)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command_line"] = self.command_line
        return data


def _bool_on(value: Any) -> str:
    return "on" if bool(value) else "off"


def normalize_gpu_layers(value: Any) -> int | None:
    """Coerce a gpu_layers param to an int. None/absent -> None (omit flag).

    Accepts the "offload everything" words other parts of the app already use
    ('all'/'auto'/'max', see estimates._layer_fraction and fit.parse_fitted_args)
    and float-ish strings like '32.0'. Unknown non-numeric -> 999 (all).
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"all", "auto", "max"}:
        return 999
    try:
        return int(float(text))
    except ValueError:
        return 999


def _add_optional(args: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    args.extend([flag, str(value)])


SPLIT_MODES = {"none", "layer", "row", "tensor"}
NUMA_STRATEGIES = {"distribute", "isolate", "numactl"}
ROPE_SCALING_TYPES = {"none", "linear", "yarn"}
KV_OVERRIDE_TYPES = {"int", "float", "bool", "str"}

# Scalar RoPE/YaRN knobs; only needed for models whose GGUF metadata doesn't
# already carry the right scaling (e.g. running past the trained context).
ROPE_FLAGS = [
    ("rope_scale", "--rope-scale"),
    ("rope_freq_base", "--rope-freq-base"),
    ("rope_freq_scale", "--rope-freq-scale"),
    ("yarn_orig_ctx", "--yarn-orig-ctx"),
    ("yarn_ext_factor", "--yarn-ext-factor"),
    ("yarn_attn_factor", "--yarn-attn-factor"),
    ("yarn_beta_slow", "--yarn-beta-slow"),
    ("yarn_beta_fast", "--yarn-beta-fast"),
]


def _as_list(value: Any) -> list[Any]:
    """A profile value that may be a list or a single comma-separated string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None and str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _enum_value(params: dict[str, Any], key: str, allowed: set[str], warnings: list[str]) -> str | None:
    value = params.get(key)
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().lower()
    if text not in allowed:
        warnings.append(f"{key}={value!r} is not one of {', '.join(sorted(allowed))}; flag not emitted.")
        return None
    return text


def multi_gpu_args(params: dict[str, Any], warnings: list[str] | None = None) -> list[str]:
    """--split-mode/--tensor-split/--main-gpu/--rpc from profile params."""
    warnings = [] if warnings is None else warnings
    args: list[str] = []
    split_mode = _enum_value(params, "split_mode", SPLIT_MODES, warnings)
    if split_mode:
        args.extend(["--split-mode", split_mode])
    tensor_split = _as_list(params.get("tensor_split"))
    if tensor_split:
        try:
            proportions = [float(part) for part in tensor_split]
        except (TypeError, ValueError):
            warnings.append(f"tensor_split={params.get('tensor_split')!r} must be numbers such as [3, 1]; flag not emitted.")
        else:
            args.extend(["--tensor-split", ",".join(f"{value:g}" for value in proportions)])
    main_gpu = params.get("main_gpu")
    if main_gpu is not None and str(main_gpu).strip():
        try:
            args.extend(["--main-gpu", str(int(main_gpu))])
        except (TypeError, ValueError):
            warnings.append(f"main_gpu={main_gpu!r} must be a GPU index; flag not emitted.")
    rpc = _as_list(params.get("rpc_servers", params.get("rpc")))
    if rpc:
        args.extend(["--rpc", ",".join(str(server) for server in rpc)])
    return args


def _kv_override_args(params: dict[str, Any], warnings: list[str]) -> list[str]:
    raw = params.get("override_kv", params.get("kv_overrides"))
    items = raw if isinstance(raw, (list, tuple)) else ([raw] if raw else [])
    valid: list[str] = []
    for item in items:
        text = str(item).strip()
        key, _, typed = text.partition("=")
        kind, _, _ = typed.partition(":")
        if not key or ":" not in typed or kind not in KV_OVERRIDE_TYPES:
            warnings.append(f"override_kv entry {text!r} must look like KEY=TYPE:VALUE (TYPE int/float/bool/str); skipped.")
            continue
        valid.append(text)
    # One flag per override: a str value may itself contain commas.
    args: list[str] = []
    for item in valid:
        args.extend(["--override-kv", item])
    return args


def _lora_args(params: dict[str, Any], warnings: list[str]) -> list[str]:
    raw = params.get("lora", params.get("lora_path"))
    items = raw if isinstance(raw, (list, tuple)) else ([raw] if raw else [])
    args: list[str] = []
    for item in items:
        if isinstance(item, dict):
            path = str(item.get("path") or "").strip()
            scale = item.get("scale")
        else:
            path, scale = str(item).strip(), None
        if not path:
            continue
        if scale is None:
            args.extend(["--lora", path])
            continue
        try:
            args.extend(["--lora-scaled", f"{path}:{float(scale):g}"])
        except (TypeError, ValueError):
            warnings.append(f"LoRA scale {scale!r} for {path} is not a number; adapter skipped.")
    return args


def _rope_args(params: dict[str, Any], warnings: list[str]) -> list[str]:
    args: list[str] = []
    scaling = _enum_value(params, "rope_scaling", ROPE_SCALING_TYPES, warnings)
    if scaling:
        args.extend(["--rope-scaling", scaling])
    for key, flag in ROPE_FLAGS:
        _add_optional(args, flag, params.get(key))
    return args


def build_llama_server_args(
    llama_server: str,
    model_path: str,
    params: dict[str, Any],
    extra_args: list[str] | None = None,
) -> LaunchCommand:
    """Build a modern llama-server argv list from normalized profile params."""

    warnings: list[str] = []
    args = [
        llama_server,
        "-m",
        model_path,
        "--host",
        str(params.get("host", "127.0.0.1")),
        "--port",
        str(int(params.get("port", 8080))),
        "--alias",
        str(params.get("alias", Path(model_path).stem)),
    ]

    mapping = [
        ("ctx_size", "--ctx-size"),
        ("threads", "--threads"),
        ("threads_batch", "--threads-batch"),
        ("batch_size", "--batch-size"),
        ("ubatch_size", "--ubatch-size"),
        ("cache_type_k", "--cache-type-k"),
        ("cache_type_v", "--cache-type-v"),
        ("cache_ram_mib", "--cache-ram"),
        ("cache_reuse", "--cache-reuse"),
        ("slot_prompt_similarity", "--slot-prompt-similarity"),
        ("reasoning_budget", "--reasoning-budget"),
        ("n_predict", "--predict"),
        ("seed", "--seed"),
        ("temperature", "--temp"),
        ("top_k", "--top-k"),
        ("top_p", "--top-p"),
        ("min_p", "--min-p"),
        ("repeat_last_n", "--repeat-last-n"),
        ("repeat_penalty", "--repeat-penalty"),
        ("presence_penalty", "--presence-penalty"),
        ("frequency_penalty", "--frequency-penalty"),
    ]
    for key, flag in mapping:
        _add_optional(args, flag, params.get(key))

    gpu_layers = 0 if str(params.get("acceleration_backend", "")).lower() == "cpu" else normalize_gpu_layers(params.get("gpu_layers"))
    if gpu_layers is not None:
        args.extend(["--gpu-layers", "all" if gpu_layers >= 999 else str(gpu_layers)])

    args.extend(["--flash-attn", _bool_on(params.get("flash_attn", True))])
    args.extend(["--reasoning", _bool_on(params.get("reasoning", False))])
    # --jinja is a presence flag (no on/off value). It makes llama.cpp use the
    # model's own chat template + tool-call parser; without it, tool results are
    # injected wrong and tool-capable models loop the same call forever.
    if params.get("jinja"):
        args.append("--jinja")
    args.append("--kv-offload" if params.get("kv_offload", True) else "--no-kv-offload")
    args.append("--op-offload" if params.get("op_offload", True) else "--no-op-offload")

    device = params.get("device", params.get("cuda_device"))
    if device not in (None, "", "auto"):
        args.extend(["--device", str(device)])
    if params.get("mmap", True):
        args.append("--mmap")
    else:
        args.append("--no-mmap")
    if params.get("embedding", False):
        args.append("--embedding")
    numa = params.get("numa")
    if numa is True:
        args.extend(["--numa", "distribute"])
    elif numa:
        numa_strategy = _enum_value(params, "numa", NUMA_STRATEGIES, warnings)
        if numa_strategy:
            args.extend(["--numa", numa_strategy])

    args.extend(multi_gpu_args(params, warnings))
    args.extend(_rope_args(params, warnings))
    args.extend(_kv_override_args(params, warnings))
    args.extend(_lora_args(params, warnings))

    draft_model = str(params.get("draft_model", "")).strip()
    spec_type = str(params.get("spec_type", "")).strip()
    if draft_model:
        args.extend(["--model-draft", draft_model])
        if spec_type:
            args.extend(["--spec-type", spec_type])
        if "spec_draft_n_max" in params:
            args.extend(["--spec-draft-n-max", str(params["spec_draft_n_max"])])
        elif "draft_max" in params:
            args.extend(["--draft-max", str(params["draft_max"])])
        if "draft_min" in params:
            args.extend(["--draft-min", str(params["draft_min"])])
        if "draft_p_min" in params:
            args.extend(["--draft-p-min", str(params["draft_p_min"])])
    elif spec_type:
        warnings.append("spec_type was set but draft_model was missing; speculative flags were not emitted.")

    tensor_overrides = params.get("tensor_overrides") or params.get("override_tensors") or params.get("ot")
    if tensor_overrides:
        if isinstance(tensor_overrides, list):
            for override in tensor_overrides:
                args.extend(["-ot", str(override)])
        else:
            args.extend(["-ot", str(tensor_overrides)])

    if extra_args:
        args.extend(extra_args)

    return LaunchCommand(argv=args, cwd=str(Path(llama_server).parent), warnings=warnings)
