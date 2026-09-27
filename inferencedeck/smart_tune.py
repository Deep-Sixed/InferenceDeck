from __future__ import annotations

from typing import Any, Callable

from .estimates import estimate_memory_fit, estimate_tokens_per_second, _get_total_layers, prime_model_meta, recommend_jinja

# ponytail: greedy grid scan over the existing estimator (layers x ctx x kv-cache).
# No subprocess, no optimizer lib — ~100 cheap pure-Python fit evals. The ceiling
# is the estimator's own accuracy; a real benchmark should override these picks.
CTX_LADDER = [2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072]
CACHE_LADDER = ["f16", "q8_0", "q5_1", "q4_0"]  # highest quality -> most compact
_BF16_CACHE_LADDER = ["bf16", *CACHE_LADDER]
# higher rank == better KV quality, used to weight/break ties toward fidelity
_CACHE_RANK = {name: i for i, name in enumerate(reversed(CACHE_LADDER))}
_CACHE_RANK["bf16"] = _CACHE_RANK["f16"]
_MAX_CACHE_RANK = max(_CACHE_RANK.values())
_MAX_CTX_INDEX = len(CTX_LADDER) - 1

# K and V are tuned independently. The K cache is more sensitive to quantization
# than V (keys drive attention scores; values are just averaged), so we (a) never
# spend more bits on V than K and (b) weight K fidelity well above V when scoring.
# The search therefore sheds V bits first when memory is short, keeping asymmetric
# picks like q8_0 K / q4_0 V that preserve the precision that matters most.
_CACHE_K_WEIGHT = 0.7
_CACHE_V_WEIGHT = 0.3


_BF16_BACKEND_MARKERS = ("cuda", "nvidia", "geforce", "rtx", "quadro", "tesla")
_BF16_GPU_MARKERS = (
    "blackwell", "rtx 50", "rtx 5070", "rtx 5080", "rtx 5090",
    "ada", "rtx 40", "rtx 4070", "rtx 4080", "rtx 4090",
    "h100", "h200", "h800", "h20",
    "b100", "b200", "gb200",
    "a100", "a800",
    "l4", "l40", "l40s",
)
_SIXTEEN_BIT_CACHES = {"f16", "bf16"}


def _gpu_descriptor(hardware: dict[str, Any] | None) -> str:
    gpu = (hardware or {}).get("primary_gpu") or {}
    fields = (
        gpu.get("name"),
        gpu.get("vendor"),
        gpu.get("backend"),
        gpu.get("acceleration_backend"),
    )
    return " ".join(str(value).lower() for value in fields if value)


def _prefers_bf16_kv(hardware: dict[str, Any] | None) -> bool:
    """Return true when the detected accelerator is a known-good BF16 target."""
    descriptor = _gpu_descriptor(hardware)
    if not descriptor:
        return False
    if not any(marker in descriptor for marker in _BF16_BACKEND_MARKERS):
        return False
    return any(marker in descriptor for marker in _BF16_GPU_MARKERS)


def _cache_ladder(hardware: dict[str, Any] | None) -> list[str]:
    return list(_BF16_CACHE_LADDER if _prefers_bf16_kv(hardware) else CACHE_LADDER)


def _is_16bit_cache(cache: Any) -> bool:
    return str(cache).lower() in _SIXTEEN_BIT_CACHES


def _cache_fidelity_norm(cache_k: str, cache_v: str) -> float:
    rank_k = _CACHE_RANK.get(cache_k, 0)
    rank_v = _CACHE_RANK.get(cache_v, 0)
    return (_CACHE_K_WEIGHT * rank_k + _CACHE_V_WEIGHT * rank_v) / _MAX_CACHE_RANK

# The balanced pick leans toward KV quality: quant fidelity is weighted slightly
# above context size, so the tuner won't trade a better cache quant for a bigger
# window — it grows context only once a sensible quant is locked in.
_BALANCED_CACHE_WEIGHT = 0.55
_BALANCED_CTX_WEIGHT = 0.45

_TUNE_KEYS = (
    "gpu_layers", "ctx_size", "cache_type_k", "cache_type_v",
    "batch_size", "ubatch_size", "threads", "threads_batch",
)
_REASONS = {
    "gpu_layers": "offload as many layers to the accelerator as memory allows (biggest speed lever)",
    "ctx_size": "grow the context window into the remaining memory headroom",
    "cache_type_k": "pick the KV-cache quant that best balances fidelity and memory",
    "cache_type_v": "pick the KV-cache quant that best balances fidelity and memory (V sheds bits before K)",
    "batch_size": "grow the logical batch into leftover memory headroom for faster prompt processing",
    "ubatch_size": "grow the physical batch into leftover memory headroom for faster prompt processing",
    "threads": "match generation threads to the CPU (decode is memory-bound; extra threads add contention)",
    "threads_batch": "use more threads for prompt processing (it scales with logical cores)",
    "flash_attn": "enable flash attention (required for a quantized KV cache)",
    "jinja": "use the model's own chat template so tool calls parse correctly (no tool-call loops)",
}

# Prompt-processing batch pairs (batch, ubatch), largest first. Batch sizes only
# affect the compute buffer, so they're grown into whatever headroom is left
# AFTER the main grid pick — context and KV quality always take priority. The
# logical batch (-b) is kept >= the physical batch (-ub).
_BATCH_LADDER = [
    (2048, 2048), (2048, 1024), (2048, 512),
    (1024, 512), (512, 512), (512, 256), (256, 128),
]

# Named intents the tuner reports so a caller can pick by current need.
_INTENTS = (
    ("balanced", "Balanced",
     "Best overall fit; leans toward KV-cache quality, then grows context."),
    ("max_quality", "Max quality",
     "Highest-fidelity KV cache that fits, then as much context as fits."),
    ("max_context", "Max context",
     "Largest context window that fits, then the best KV quant for it."),
)


def _layer_options(model: dict[str, Any] | None) -> list[Any]:
    total = _get_total_layers(model)
    if total and total > 0:
        return ["all", int(total * 0.75), int(total * 0.5), int(total * 0.25), 0]
    # Unknown layer count: only full-GPU or pure-CPU are reliable to estimate.
    return ["all", 0]


def _int_or_none(value: Any) -> int | None:
    try:
        result = int(float(value))
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def _recommend_threads(hardware: dict[str, Any] | None, layer_fraction: float) -> dict[str, int] | None:
    """Recommend -t / -tb from the detected CPU and how much work stays on it.

    Decode (-t) is memory-bandwidth-bound, so hyper-threads don't help and
    oversubscription adds contention: use physical cores, minus one to keep the
    OS and server responsive when the CPU is doing real layer work. Prompt
    processing (-tb) is compute-bound and scales with logical cores. When the
    model is fully offloaded the CPU only orchestrates and samples, so a small
    fixed pool avoids busy-spin without costing speed.
    """
    cpu = (hardware or {}).get("cpu") or {}
    physical = _int_or_none(cpu.get("physical_cores"))
    logical = _int_or_none(cpu.get("logical_cores"))
    if not physical and logical:
        physical = max(1, logical // 2)  # assume SMT when only logical is known
    if not physical:
        return None
    if layer_fraction >= 1.0:
        threads = max(2, min(physical, 8))
        return {"threads": threads, "threads_batch": threads}
    threads = max(1, physical - 1)
    return {"threads": threads, "threads_batch": max(threads, logical or physical)}


def _tune_batch(
    params: dict[str, Any],
    model: dict[str, Any] | None,
    hardware: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Grow batch/ubatch into the headroom left over by the main grid pick.

    Tries (batch, ubatch) pairs largest-first and keeps the first that still
    fits; the caller's pair is included so the result is never smaller than
    what already fit. Returns the (possibly updated) params and the fit for
    the chosen pair (None when nothing beat the caller's own pair).
    """
    base_b = _int_or_none(params.get("batch_size")) or 512
    base_ub = _int_or_none(params.get("ubatch_size")) or min(base_b, 512)
    pairs = sorted(set(_BATCH_LADDER) | {(base_b, base_ub)}, key=lambda p: (p[1], p[0]), reverse=True)
    for batch, ubatch in pairs:
        if ubatch > batch:
            continue
        cand = {**params, "batch_size": batch, "ubatch_size": ubatch}
        fit = estimate_memory_fit(cand, model, hardware)
        if fit["status"] in ("near_limit", "unknown"):
            continue
        return cand, fit
    return dict(params), None


def _changes(base: dict[str, Any], tuned: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for key in (*_TUNE_KEYS, "flash_attn", "jinja"):
        before, after = base.get(key), tuned.get(key)
        if str(before) != str(after):
            changes.append({"field": key, "from": before, "to": after, "why": _REASONS[key]})
    return changes


def _candidate_params(base: dict[str, Any], layers: Any, ctx: int, cache_k: str, cache_v: str) -> dict[str, Any]:
    cand = {**base, "gpu_layers": layers, "ctx_size": ctx,
            "cache_type_k": cache_k, "cache_type_v": cache_v}
    # Quantized KV requires flash attention in llama.cpp; force it on so the
    # suggested config can actually launch (and so the estimate reflects it).
    if not _is_16bit_cache(cache_k) or not _is_16bit_cache(cache_v):
        cand["flash_attn"] = True
    return cand


def _collect_candidates(
    base: dict[str, Any],
    model: dict[str, Any] | None,
    hardware: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Evaluate the grid once and keep every config the estimator says fits."""
    candidates: list[dict[str, Any]] = []
    cache_ladder = _cache_ladder(hardware)
    for layers in _layer_options(model):
        for ctx in CTX_LADDER:
            for cache_k in cache_ladder:
                for cache_v in cache_ladder:
                    # Never spend more bits on V than K — K carries more signal.
                    if _CACHE_RANK[cache_v] > _CACHE_RANK[cache_k]:
                        continue
                    cand = _candidate_params(base, layers, ctx, cache_k, cache_v)
                    fit = estimate_memory_fit(cand, model, hardware)
                    if fit["status"] in ("near_limit", "unknown"):
                        continue
                    # Don't trust a GPU offload we can't size against real VRAM.
                    if fit["inputs"]["gpu_layer_fraction"] > 0 and fit["estimated"]["accelerator_capacity_mib"] is None:
                        continue
                    candidates.append({
                        "params": cand,
                        "fit": fit,
                        "cache_k": cache_k,
                        "cache_v": cache_v,
                        "lf": fit["inputs"]["gpu_layer_fraction"],
                        "ctx_norm": CTX_LADDER.index(ctx) / _MAX_CTX_INDEX,
                        "cache_norm": _cache_fidelity_norm(cache_k, cache_v),
                        "roomy": 1 if fit["status"] == "good" else 0,
                    })
    return candidates


# Each key maximizes GPU layers first (the dominant fit/speed lever), then the
# intent-specific trade-off between context and KV quant, then prefers the
# roomier (non-tight) fit as a final safety tiebreak.
_INTENT_KEYS: dict[str, Callable[[dict[str, Any]], tuple]] = {
    "balanced": lambda c: (
        c["lf"],
        _BALANCED_CACHE_WEIGHT * c["cache_norm"] + _BALANCED_CTX_WEIGHT * c["ctx_norm"],
        c["cache_norm"],  # tie toward fidelity
        c["roomy"],
    ),
    "max_quality": lambda c: (c["lf"], c["cache_norm"], c["ctx_norm"], c["roomy"]),
    "max_context": lambda c: (c["lf"], c["ctx_norm"], c["cache_norm"], c["roomy"]),
}


def _signature(cand: dict[str, Any]) -> tuple:
    p = cand["params"]
    return (str(p.get("gpu_layers")), p.get("ctx_size"), p.get("cache_type_k"), p.get("cache_type_v"))


def auto_tune_fit(
    params: dict[str, Any],
    model: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
    target_mib: int = 1024,
) -> dict[str, Any]:
    """Search launch params for the best memory fit, by named intent.

    Maximizes GPU offload first (the biggest speed lever), then trades context
    against KV-cache fidelity. The default ("balanced") pick leans toward quant
    quality so it never drops to a worse KV quant just to enlarge the cache.
    Returns the balanced pick as ``tuned_params`` plus a ``suggestions`` list
    (balanced / max quality / max context) so a caller can choose by need.
    Rejects any candidate the estimator flags as near_limit or can't verify.
    """
    base = dict(params or {})
    base.setdefault("fit_target_mib", target_mib)
    # Parse the GGUF once up front so every grid eval below reads exact KV dims
    # from cache instead of re-opening the (slow) header.
    prime_model_meta(model)
    # Jinja is orthogonal to the memory grid — recommend it from the model's chat
    # template (now cached by prime_model_meta) and apply it to every suggestion.
    jinja_rec = recommend_jinja(model)
    before_fit = estimate_memory_fit(base, model, hardware)
    before_speed = estimate_tokens_per_second(base, model, hardware)

    candidates = _collect_candidates(base, model, hardware)
    if not candidates:
        return {
            "success": False,
            "reason": "No configuration fit within memory, or accelerator capacity is unknown.",
            "before": {"params": base, "fit_status": before_fit, "speed_estimate": before_speed},
        }

    # Pick the best candidate per intent, de-duplicating identical configs and
    # merging the labels of any intent that lands on the same config.
    suggestions: list[dict[str, Any]] = []
    by_signature: dict[tuple, dict[str, Any]] = {}
    for intent_id, label, description in _INTENTS:
        best = max(candidates, key=_INTENT_KEYS[intent_id])
        sig = _signature(best)
        if sig in by_signature:
            entry = by_signature[sig]
            entry["intents"].append(intent_id)
            entry["labels"].append(label)
            entry["label"] = " / ".join(entry["labels"])
            continue
        # Refine the winner: grow batch sizes into the leftover headroom, then
        # match threads to the CPU and how much of the model stays on it.
        tuned, refined_fit = _tune_batch(best["params"], model, hardware)
        fit = refined_fit or best["fit"]
        threads_rec = _recommend_threads(hardware, fit["inputs"]["gpu_layer_fraction"])
        if threads_rec:
            tuned.update(threads_rec)
        tuned["jinja"] = jinja_rec["recommended"]
        entry = {
            "intent": intent_id,
            "intents": [intent_id],
            "label": label,
            "labels": [label],
            "description": description,
            "params": tuned,
            "changes": _changes(base, tuned),
            "fit_status": fit,
            "speed_estimate": estimate_tokens_per_second(tuned, model, hardware),
        }
        by_signature[sig] = entry
        suggestions.append(entry)

    primary = suggestions[0]  # balanced is listed first
    tuned, after_fit = primary["params"], primary["fit_status"]
    return {
        "success": True,
        "tuned_params": tuned,
        "changes": primary["changes"],
        "suggestions": suggestions,
        "jinja": jinja_rec,
        "before": {"params": base, "fit_status": before_fit, "speed_estimate": before_speed},
        "after": {"params": tuned, "fit_status": after_fit, "speed_estimate": primary["speed_estimate"]},
        "notes": [
            "Suggestions come from the memory estimator, not a live run — verify with a fit test or benchmark.",
            "Priority: max GPU layers, then a balance of KV-cache fidelity and context (quality-leaning).",
            "K and V caches are tuned independently; V sheds bits before K and is never more precise than K.",
            "Batch/ubatch grow into leftover headroom after context and KV quality are settled.",
            "Threads follow the CPU: physical cores for decode, logical cores for prompt batches.",
            "Pick 'Max quality' or 'Max context' from the suggestions when your need leans one way.",
            f"Jinja {'on' if jinja_rec['recommended'] else 'off'}: {jinja_rec['reason']}.",
        ],
    }


def demo() -> None:
    """ponytail self-check: tuning never recommends an overflowing config, a
    roomy GPU gets full offload, and quality/context intents are reported."""
    model = {"name": "test-7B", "params_b": 7, "quant": "Q4_K_M"}
    hw = {
        "primary_gpu": {"name": "RTX 4090", "vram_total_bytes": 24 * 1024**3,
                        "vram_free_bytes": 24 * 1024**3, "vram_bandwidth_gbps": 1000},
        "memory": {"total_bytes": 64 * 1024**3, "available_bytes": 48 * 1024**3},
        "cpu": {"logical_cores": 16},
    }
    out = auto_tune_fit({"gpu_layers": 0, "ctx_size": 2048, "cache_type_k": "q4_0",
                         "cache_type_v": "q4_0"}, model, hw)
    assert out["success"], out
    assert out["after"]["fit_status"]["status"] != "near_limit"
    assert str(out["tuned_params"]["gpu_layers"]) == "all", out["tuned_params"]
    assert out["after"]["fit_status"]["inputs"]["ctx_size"] >= 2048
    assert out["suggestions"], "expected named suggestions"

    # Max-context should never carry a higher-fidelity-but-smaller window than
    # max-quality, and max-quality should never use a more compact quant.
    quality = next((s for s in out["suggestions"] if "max_quality" in s["intents"]), None)
    context = next((s for s in out["suggestions"] if "max_context" in s["intents"]), None)
    if quality and context:
        assert context["params"]["ctx_size"] >= quality["params"]["ctx_size"]
        assert _CACHE_RANK[quality["params"]["cache_type_k"]] >= _CACHE_RANK[context["params"]["cache_type_k"]]

    # No suggestion may spend more bits on V than K.
    for s in out["suggestions"]:
        p = s["params"]
        assert _CACHE_RANK[p["cache_type_v"]] <= _CACHE_RANK[p["cache_type_k"]], p

    # Threads and batch sizes are recommended and stay coherent.
    for s in out["suggestions"]:
        p = s["params"]
        assert p.get("threads", 1) >= 1 and p.get("threads_batch", 1) >= p.get("threads", 1), p
        assert p.get("ubatch_size", 1) <= p.get("batch_size", 1), p
    # A roomy GPU should have grown the batch beyond the tiny starting point.
    assert out["tuned_params"]["ubatch_size"] >= 512, out["tuned_params"]

    # Tiny VRAM with unknown capacity falls back to a safe pick or reports failure.
    tiny = {"primary_gpu": {}, "memory": {"total_bytes": 8 * 1024**3, "available_bytes": 2 * 1024**3},
            "cpu": {"logical_cores": 8}}
    out2 = auto_tune_fit({"gpu_layers": "all", "ctx_size": 131072}, model, tiny)
    if out2["success"]:
        assert out2["after"]["fit_status"]["status"] != "near_limit"
    print("smart_tune demo OK")


if __name__ == "__main__":
    demo()
