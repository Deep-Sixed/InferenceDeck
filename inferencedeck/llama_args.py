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
