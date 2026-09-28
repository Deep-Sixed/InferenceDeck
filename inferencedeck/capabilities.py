"""What a profile's model can do: inputs, outputs, tools, embedding, reranking, context.

Capabilities come from four places, later ones winning field by field:

1. discovered from the GGUF header (cached, see estimates.gguf_model_info):
   tool-calling chat template, trained context length, and pooling type,
   which marks embedding (mean/cls/last) and reranking (rank) models;
2. the launch itself: image/audio input only when a projector (mmproj) is
   passed, since llama-server cannot see images without one;
3. a profile's own ``capabilities`` param, for anything detection gets wrong;
4. for a running server, what llama-server reports in /props (see
   server_manager.parse_props): the modalities and tool support it actually
   has, plus the per-request context.

Each field's origin is kept in ``sources`` so the UI can say where it came from.
"""

from __future__ import annotations

from typing import Any

from .estimates import gguf_model_info, model_supports_tools

FIELDS = ("input", "output", "tools", "embedding", "reranker", "context_max")
MODALITIES = ("text", "image", "audio", "video")
# llama.cpp pooling types; any pooling means the model produces vectors.
_POOLING_RANK = 4
_POOLING_EMBED = {1, 2, 3}
# Encoder-only architectures, which are embedding models even without a pooling key.
_EMBEDDING_ARCHES = {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "modern-bert", "neo-bert", "t5encoder"}
# Queries accepted by filter_profiles and GET /api/profiles?capability=...
QUERIES = ("tools", "image", "audio", "embedding", "reranker")


def _model_path(model: dict[str, Any] | None) -> str | None:
    if not model:
        return None
    return model.get("path") or model.get("model_path")


def resolve_projector(model: dict[str, Any] | None, params: dict[str, Any]) -> tuple[str | None, list[str]]:
    """The mmproj file a launch should pass, and any warning about it.

    An explicit ``mmproj`` param wins; ``vision: true`` uses the projector found
    next to the model during discovery.
    """
    explicit = str(params.get("mmproj") or "").strip()
    if explicit:
        return explicit, []
    if params.get("vision"):
        found = (model or {}).get("mmproj_path")
        if found:
            return str(found), []
        return None, ["vision is on but no mmproj projector file was found next to the model; it starts text-only."]
    return None, []


def _declared(params: dict[str, Any]) -> dict[str, Any]:
    raw = params.get("capabilities")
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, Any] = {}
    for key in ("input", "output"):
        value = raw.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            clean[key] = [item.lower() for item in value]
    for key in ("tools", "embedding", "reranker"):
        if isinstance(raw.get(key), bool):
            clean[key] = raw[key]
    context = raw.get("context_max")
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        clean["context_max"] = context
    return clean


def profile_capabilities(
    model: dict[str, Any] | None, params: dict[str, Any], probe: bool = False
) -> dict[str, Any]:
    """Capabilities of ``model`` launched with ``params``.

    With ``probe`` False the GGUF is never opened (cached metadata only), so
    listing profiles stays fast; fields not yet known are None.
    """

    path = _model_path(model)
    info = gguf_model_info(path, probe) if path else None
    tools = model_supports_tools(path, probe) if path else None
    caps: dict[str, Any] = {field: None for field in FIELDS}
    sources: dict[str, str] = {}

    def put(field: str, value: Any, source: str) -> None:
        caps[field] = value
        sources[field] = source

    put("input", ["text"], "default")
    put("output", ["text"], "default")
    if tools is not None:
        put("tools", tools, "gguf")
    if info:
        pooling = info.get("pooling_type")
        reranker = pooling == _POOLING_RANK
        embedding = not reranker and (pooling in _POOLING_EMBED or info.get("arch") in _EMBEDDING_ARCHES)
        put("reranker", reranker, "gguf")
        put("embedding", embedding, "gguf")
        if reranker:
            put("output", ["score"], "gguf")
        elif embedding:
            put("output", ["embedding"], "gguf")
        if info.get("context_length"):
            put("context_max", int(info["context_length"]), "gguf")

    projector, _ = resolve_projector(model, params)
    found = (model or {}).get("mmproj_path")
    if projector:
        projector_info = gguf_model_info(projector, probe) or {}
        inputs = ["text"]
        # A projector with no encoder flags (older files) is assumed to be vision.
        if projector_info.get("vision_encoder", not projector_info.get("audio_encoder", False)):
            inputs.append("image")
        if projector_info.get("audio_encoder"):
            inputs.append("audio")
        put("input", inputs, "launch")
    caps["vision_available"] = bool(found) and not projector

    for field, value in _declared(params).items():
        put(field, value, "profile")
    caps["sources"] = sources
    return caps


def with_served(caps: dict[str, Any] | None, served: dict[str, Any] | None) -> dict[str, Any] | None:
    """Overlay what a running llama-server reported in /props onto ``caps``."""

    if not served:
        return caps
    merged = dict(caps or {field: None for field in FIELDS})
    sources = dict(merged.get("sources") or {})
    inputs = served.get("input_modalities")
    if isinstance(inputs, list) and inputs:
        merged["input"] = list(inputs)
        sources["input"] = "server"
    if isinstance(served.get("tools"), bool):
        merged["tools"] = served["tools"]
        sources["tools"] = "server"
    for key in ("slot_ctx", "total_slots", "build_info"):
        if served.get(key) is not None:
            merged[key] = served[key]
    merged["sources"] = sources
    return merged


def matches(caps: dict[str, Any] | None, query: str) -> bool:
    """Whether ``caps`` has the capability named by ``query`` (see QUERIES)."""
    if not caps:
        return False
    if query in ("image", "audio"):
        return query in (caps.get("input") or [])
    return caps.get(query) is True


def filter_profiles(profiles: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    if query not in QUERIES:
        raise ValueError(f"capability must be one of {', '.join(QUERIES)}")
    return [profile for profile in profiles if matches(profile.get("capabilities"), query)]
