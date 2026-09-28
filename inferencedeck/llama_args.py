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
LOAD_MODES = {"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"}
# How each load mode is spelled on builds from before --load-mode existed.
LEGACY_LOAD_MODE_ARGS = {
    "auto": [],
    "mmap": [],
    "none": ["--no-mmap"],
    "mlock": ["--no-mmap", "--mlock"],
    "mmap+mlock": ["--mlock"],
}
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


def _load_mode_args(params: dict[str, Any], flags: frozenset[str] | None, warnings: list[str]) -> list[str]:
    """Model-loading flags: ``--load-mode`` on current builds, else ``--no-mmap``/``--mlock``.

    Both spellings default to mmap, so the default emits nothing and works on any
    build. ``load_mode`` wins over the ``mmap``/``mlock`` booleans when set.
    """
    has_load_mode = flags is not None and "--load-mode" in flags
    if params.get("load_mode") not in (None, ""):
        mode = _enum_value(params, "load_mode", LOAD_MODES, warnings)
        if not mode:
            return []
        if has_load_mode:
            return ["--load-mode", mode]
        if mode == "dio":
            warnings.append("load_mode 'dio' needs a llama-server with --load-mode; flag not emitted.")
            return []
        return list(LEGACY_LOAD_MODE_ARGS[mode])
    mmap = bool(params.get("mmap", True))
    mlock = bool(params.get("mlock", False))
    if has_load_mode:
        if mmap and not mlock:
            return []
        return ["--load-mode", "mmap+mlock" if mmap and mlock else "mlock" if mlock else "none"]
    return (["--no-mmap"] if not mmap else []) + (["--mlock"] if mlock else [])


def _draft_args(params: dict[str, Any], flags: frozenset[str] | None) -> list[str]:
    """Speculative-decoding limits, using the ``--spec-draft-*`` names when the build has them.

    ``draft_max``/``draft_min`` are the older profile keys; current llama.cpp rejects
    ``--draft-max``/``--draft-min`` outright, so they are renamed when it can tell.
    """
    args: list[str] = []
    for spec_key, legacy_key, spec_flag, legacy_flag in (
        ("spec_draft_n_max", "draft_max", "--spec-draft-n-max", "--draft-max"),
        ("spec_draft_n_min", "draft_min", "--spec-draft-n-min", "--draft-min"),
        ("spec_draft_p_min", "draft_p_min", "--spec-draft-p-min", "--draft-p-min"),
    ):
        if spec_key in params:
            args.extend([spec_flag, str(params[spec_key])])
        elif legacy_key in params:
            renamed = flags is not None and spec_flag in flags
            args.extend([spec_flag if renamed else legacy_flag, str(params[legacy_key])])
    return args


def _unsupported_flag_warnings(argv: list[str], flags: frozenset[str] | None, binary: str) -> list[str]:
    if not flags:
        return []
    unknown = sorted({token for token in argv if token.startswith("--") and token not in flags})
    if not unknown:
        return []
    return [
        f"{Path(binary).name} does not list {', '.join(unknown)} in its --help; "
        "it will probably refuse to start. Update llama.cpp or remove those settings."
    ]


def build_llama_server_args(
    llama_server: str,
    model_path: str,
    params: dict[str, Any],
    extra_args: list[str] | None = None,
    flags: frozenset[str] | None = None,
) -> LaunchCommand:
    """Build a modern llama-server argv list from normalized profile params.

    ``flags`` is what the binary accepts (see ``llama_flags.supported_flags``);
    with it, renamed flags are spelled the way that build expects. ``None`` means
    unknown and keeps the older spellings.
    """

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
    # Serves llama-server's own Prometheus counters (tokens, requests, KV cache),
    # which InferenceDeck's telemetry reads for tokens/sec and request counts.
    # Only on builds known to accept it: an unknown flag would stop the server starting.
    if params.get("metrics", True) and (flags is None or "--metrics" in flags):
        args.append("--metrics")

    device = params.get("device", params.get("cuda_device"))
    if device not in (None, "", "auto"):
        args.extend(["--device", str(device)])
    args.extend(_load_mode_args(params, flags, warnings))
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
        args.extend(_draft_args(params, flags))
    elif spec_type:
        warnings.append("spec_type was set but draft_model was missing; speculative flags were not emitted.")

    tensor_overrides = params.get("tensor_overrides") or params.get("override_tensors") or params.get("ot")
    if tensor_overrides:
        if isinstance(tensor_overrides, list):
            for override in tensor_overrides:
                args.extend(["-ot", str(override)])
        else:
            args.extend(["-ot", str(tensor_overrides)])

    # Checked before extra_args: those are the user's own, verbatim.
    warnings.extend(_unsupported_flag_warnings(args, flags, llama_server))
    if extra_args:
        args.extend(extra_args)

    return LaunchCommand(argv=args, cwd=str(Path(llama_server).parent), warnings=warnings)
