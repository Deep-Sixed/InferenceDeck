from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import cache_dir
from .server_manager import DEFAULT_READY_TIMEOUT_SECONDS, list_servers, start_profile, stop_server


RESULTS_FILENAME = "benchmarks.json"
DEFAULT_PROMPT = (
    "Write a concise technical note explaining how local LLM inference speed changes "
    "with context length, batch size, GPU offload, and cache quantization."
)


def benchmark_results_path() -> Path:
    path = cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path / RESULTS_FILENAME


def load_benchmark_results() -> list[dict[str, Any]]:
    path = benchmark_results_path()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def save_benchmark_result(result: dict[str, Any]) -> None:
    results = load_benchmark_results()
    results.append(result)
    results = results[-100:]
    path = benchmark_results_path()
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _server_for_mode(mode: str) -> dict[str, Any] | None:
    for server in list_servers():
        if server.get("mode") == mode and server.get("running"):
            return server
    return None


def _api_base(server: dict[str, Any]) -> str:
    host = str(server.get("host") or "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{int(server.get('port') or 8080)}"


def _completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    return str(message.get("content") or first.get("text") or "")


def _fallback_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _round(value: Any, digits: int) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if number > 0 else None


def response_metrics(payload: dict[str, Any], text: str, elapsed: float) -> dict[str, Any]:
    """Token counts and speeds for one non-streaming completion response.

    llama-server adds a ``timings`` block that splits prompt processing from
    generation; its ``predicted_per_second`` is the true decode speed. Without
    it, the speed falls back to completion tokens over wall-clock time, which
    also counts prompt processing and HTTP overhead.
    """
    usage = payload.get("usage") or {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    completion = int(timings.get("predicted_n") or usage.get("completion_tokens") or _fallback_token_count(text))
    prompt = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
    wall_tps = completion / elapsed if completion else 0.0
    generation_tps = _round(timings.get("predicted_per_second"), 2)
    metrics: dict[str, Any] = {
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "tokens_per_second": generation_tps if generation_tps is not None else round(wall_tps, 2),
        "tokens_per_second_source": "timings" if generation_tps is not None else "wall_clock",
        "wall_tokens_per_second": round(wall_tps, 2),
    }
    if timings:
        metrics["prompt_tokens_per_second"] = _round(timings.get("prompt_per_second"), 2)
        metrics["prompt_ms"] = _round(timings.get("prompt_ms"), 1)
        metrics["generation_ms"] = _round(timings.get("predicted_ms"), 1)
        if timings.get("cache_n") is not None:
            # cache_n counts prompt tokens reused from the KV cache (not re-processed).
            metrics["cached_prompt_tokens"] = int(timings.get("cache_n") or 0)
    return metrics


def send_chat_prompt(
    mode: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Send one chat prompt to an already-running tracked server and return the reply.

    Unlike run_profile_benchmark, this never starts or restarts the server — it only
    talks to a server this app already has running for the given mode.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {"success": False, "error": "Prompt is empty."}
    server = _server_for_mode(mode)
    if not server:
        return {"success": False, "error": f"No running tracked server for '{mode}'. Start it first."}

    base_url = _api_base(server)
    request_payload = {
        "model": mode,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    raw = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=raw,
        headers={"Content-Type": "application/json", "User-Agent": "inferencedeck/test-prompt"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"success": False, "error": str(exc), "endpoint": f"{base_url}/v1/chat/completions"}
    elapsed = max(time.perf_counter() - started, 0.001)

    text = _completion_text(response_payload)
    return {
        "success": True,
        "reply": text,
        "endpoint": f"{base_url}/v1/chat/completions",
        "elapsed_seconds": round(elapsed, 3),
        **response_metrics(response_payload, text, elapsed),
    }


def run_profile_benchmark(
    mode: str,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    prompt: str | None = None,
    completion_tokens: int = 128,
    restart: bool = True,
    stop_after: bool = False,
    ready_timeout_seconds: int = DEFAULT_READY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    params = dict(overrides or {})
    params["n_predict"] = int(completion_tokens)
    start = start_profile(
        mode=mode,
        project_root=project_root,
        model_dirs=model_dirs,
        overrides=params,
        stop_existing=restart,
        wait_ready=True,
        ready_timeout_seconds=ready_timeout_seconds,
    )
    server = (start.get("server") if start.get("success") else None) or _server_for_mode(mode)
    if not server:
        return {"success": False, "error": start.get("error") or "No running tracked server was available.", "start": start}

    base_url = _api_base(server)
    request_payload = {
        "model": params.get("alias") or mode,
        "messages": [
            {"role": "system", "content": "You are benchmarking local inference. Answer directly."},
            {"role": "user", "content": prompt or DEFAULT_PROMPT},
        ],
        "temperature": float(params.get("temperature", 0.2) or 0.2),
        "max_tokens": int(completion_tokens),
        "stream": False,
    }
    raw = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=raw,
        headers={"Content-Type": "application/json", "User-Agent": "inferencedeck/benchmark"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=max(30, int(completion_tokens))) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"success": False, "error": str(exc), "server": server, "start": start}
    elapsed = max(time.perf_counter() - started, 0.001)

    usage = response_payload.get("usage") or {}
    text = _completion_text(response_payload)
    metrics = response_metrics(response_payload, text, elapsed)
    total_count = int(usage.get("total_tokens") or (metrics["prompt_tokens"] + metrics["completion_tokens"]))
    chars_per_second = len(text) / elapsed if text else 0.0

    benchmark = {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_id": server.get("id"),
        "endpoint": f"{base_url}/v1/chat/completions",
        "elapsed_seconds": round(elapsed, 3),
        **metrics,
        "total_tokens": total_count,
        "chars_per_second": round(chars_per_second, 1),
        "response_chars": len(text),
        "requested_max_tokens": int(completion_tokens),
        "usage": usage,
    }
    save_benchmark_result(benchmark)

    stop_result = None
    if stop_after and server.get("id"):
        stop_result = stop_server(server_id=server["id"])

    return {
        "success": True,
        "benchmark": benchmark,
        "server": server,
        "start": start,
        "stop": stop_result,
        "response_preview": text[:600],
    }
