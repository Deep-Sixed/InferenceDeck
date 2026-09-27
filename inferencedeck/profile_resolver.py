from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .estimates import recommend_jinja
from .inventory import build_inventory
from .manifest import load_profiles
from .models import discover_models
from .paths import find_project_root
from .schema import ModelFile, ModelProfile


# Configuration for tokenization
DEFAULT_STOPWORDS = {
    "a", "agent", "and", "coder", "gguf", "instruct", "it", "large", 
    "mode", "no", "on", "reasoning", "server", "the"
}

@dataclass
class ResolvedProfile:
    mode: str
    name: str
    description: str
    profile: dict[str, Any]
    model: dict[str, Any] | None
    launchable: bool
    confidence: float
    warnings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(value: str, stopwords: set[str] = DEFAULT_STOPWORDS) -> set[str]:
    normalized = re.sub(r"(?i)(qwen)3[._-]?6", r"\g<1>36", value)
    normalized = re.sub(r"(?i)(qwen)3[._-]?5", r"\g<1>35", normalized)
    raw = re.findall(r"[a-zA-Z0-9]+", normalized.lower())
    tokens: set[str] = set()
    for token in raw:
        if token in stopwords or len(token) < 2:
            continue
        tokens.add(token)
        split = re.match(r"([a-z]+)(\d+)(b)?$", token)
        if split:
            tokens.add(split.group(1))
            tokens.add(split.group(2) + (split.group(3) or ""))
        quant_split = re.match(r"q(\d+)([a-z]+)?$", token)
        if quant_split:
            tokens.add(f"q{quant_split.group(1)}")
    return tokens


def _profile_text(profile: ModelProfile) -> str:
    parts = [profile.mode, profile.name, profile.description, profile.script or ""]
    params = profile.recommended_params
    for key in ["alias"]:
        value = params.get(key)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def _model_text(model: ModelFile) -> str:
    parts = [model.name, model.path, model.quant or ""]
    if model.params_b:
        if float(model.params_b).is_integer():
            parts.append(f"{int(model.params_b)}b")
        else:
            parts.append(f"{model.params_b}b")
    return " ".join(parts)


def _score(profile: ModelProfile, model: ModelFile) -> float:
    profile_tokens = _tokens(_profile_text(profile))
    model_tokens = _tokens(_model_text(model))
    if not profile_tokens or not model_tokens:
        return 0.0
    overlap = profile_tokens & model_tokens
    score = len(overlap) / max(3, len(profile_tokens))

    profile_params = profile.model_size_gb
    if profile_params and model.size_bytes:
        model_gb = model.size_bytes / (1024**3)
        if abs(float(profile_params) - model_gb) <= max(1.0, float(profile_params) * 0.1):
            score += 0.15

    if profile.mode.lower() in model.path.lower():
        score += 0.15
    return min(score, 1.0)


def _path_model(profile: ModelProfile, models: list[ModelFile]) -> ModelFile | None:
    if not profile.model_path:
        return None
    try:
        profile_path = Path(profile.model_path).expanduser().resolve()
    except OSError:
        return None
    for model in models:
        try:
            if Path(model.path).expanduser().resolve() == profile_path:
                return model
        except OSError:
            continue
    return None


def _best_model(profile: ModelProfile, models: list[ModelFile]) -> tuple[ModelFile | None, float, list[str]]:
    by_path = _path_model(profile, models)
    if by_path:
        return by_path, 1.0, []
    scored = sorted(((_score(profile, model), model) for model in models), key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 0.25:
        return None, 0.0, []
    warnings: list[str] = []
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08:
        warnings.append(
            f"Model match is ambiguous: also considered '{scored[1][1].name}' with score {scored[1][0]:.2f}."
        )
    return scored[0][1], scored[0][0], warnings


def _resolved_params(profile: ModelProfile) -> dict[str, Any]:
    params = dict(profile.recommended_params)
    description = profile.description.lower()
    mode = profile.mode.lower()
    reasoning_disabled = any(token in description for token in ["no reasoning", "reasoning off", "non-reasoning"])
    reasoning_enabled = any(token in description for token in ["reasoning on", "reasoning mode"])
    if not reasoning_disabled and (reasoning_enabled or mode.endswith("think") or "-think" in mode):
        params.setdefault("reasoning", True)
    else:
        params.setdefault("reasoning", False)
    params.setdefault("host", "127.0.0.1")
    params.setdefault("port", 8080)
    params.setdefault("alias", profile.mode or "local-model")
    params.setdefault("threads_batch", params.get("threads", 4))
    params.setdefault("batch_size", 512)
    params.setdefault("ubatch_size", min(int(params.get("batch_size", 512)), 512))
    params.setdefault("flash_attn", True)
    params.setdefault("kv_offload", True)
    params.setdefault("op_offload", True)
    # Detection-driven jinja default: on when the model's chat template advertises
    # tool calling (needed so tool-capable models don't loop tool calls). Cache-only
    # (probe=False) so resolving the profiles list never blocks on a GGUF read; an
    # explicit saved value always wins via setdefault.
    if "jinja" not in params and profile.model_path:
        params.setdefault("jinja", recommend_jinja(profile.model_path, probe=False)["recommended"])
    params.setdefault("acceleration_backend", "auto")
    params.setdefault("device", "auto")
    return params


def _validate_resolved(profile: ModelProfile, model: ModelFile | None, params: dict[str, Any], confidence: float) -> tuple[bool, list[str], list[str]]:
    warnings = list(profile.portable_warnings)
    missing: list[str] = []
    if not model:
        missing.append("model")
    if confidence < 0.45 and model:
        warnings.append(f"Low-confidence model match ({confidence:.2f}); confirm before launching.")

    text = " ".join([profile.mode, profile.name, profile.description]).lower()
    if "mtp" in text:
        draft_model = str(params.get("draft_model", "")).strip()
        if not draft_model:
            missing.append("draft_model")
        elif not Path(draft_model).expanduser().exists():
            missing.append("draft_model")
            warnings.append(f"Draft model does not exist: {draft_model}")
        if model and "mtp" not in model.path.lower() and "gemma" not in text:
            warnings.append("MTP profile matched a non-MTP model path; this may require a WSL or custom backend.")

    required = ["ctx_size", "threads", "cache_type_k", "cache_type_v", "gpu_layers"]
    for key in required:
        if key not in params:
            missing.append(f"param:{key}")

    return not missing, warnings, missing


def resolve_profiles(
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
) -> list[ResolvedProfile]:
    root = Path(project_root).expanduser().resolve() if project_root else find_project_root()
    model_paths = [Path(path).expanduser() for path in model_dirs] if model_dirs else None
    profiles = load_profiles(root)
    models = discover_models(model_paths, root)
    resolved: list[ResolvedProfile] = []
    for profile in profiles:
        model, confidence, match_warnings = _best_model(profile, models)
        params = _resolved_params(profile)
        launchable, warnings, missing = _validate_resolved(profile, model, params, confidence)
        warnings = match_warnings + warnings
        resolved.append(
            ResolvedProfile(
                mode=profile.mode,
                name=profile.name,
                description=profile.description,
                profile=profile.to_dict(),
                model=model.to_dict() if model else None,
                launchable=launchable,
                confidence=round(confidence, 3),
                warnings=warnings,
                missing=missing,
                params=params,
            )
        )
    return resolved


def resolved_inventory(
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    payload = build_inventory(project_root=project_root, model_dirs=model_dirs)
    payload["resolved_profiles"] = [profile.to_dict() for profile in resolve_profiles(project_root, model_dirs)]
    payload["summary"]["launchable_profile_count"] = len(
        [profile for profile in payload["resolved_profiles"] if profile["launchable"]]
    )
    return payload
