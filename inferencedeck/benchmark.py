from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import cache_dir
from .fileio import atomic_write_text, locked
from .server_manager import http_base, list_servers, start_profile, stop_server
from .telemetry import emit as emit_event


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
    path = benchmark_results_path()
    # Locked read-modify-write: two benchmarks finishing together keep both results.
    with locked(path.with_name(path.name + ".lock")):
        results = load_benchmark_results()
        results.append(result)
        atomic_write_text(path, json.dumps(results[-100:], indent=2) + "\n")


def _server_for_mode(mode: str) -> dict[str, Any] | None:
    for server in list_servers():
        if server.get("mode") == mode and server.get("running"):
            return server
    return None


def _api_base(server: dict[str, Any]) -> str:
    return http_base(server.get("host"), int(server.get("port") or 8080))


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

    usage = response_payload.get("usage") or {}
    text = _completion_text(response_payload)
    completion_count = int(usage.get("completion_tokens") or usage.get("predicted_n") or _fallback_token_count(text))
    tokens_per_second = completion_count / elapsed if completion_count else 0.0

    return {
        "success": True,
        "reply": text,
        "endpoint": f"{base_url}/v1/chat/completions",
        "elapsed_seconds": round(elapsed, 3),
        "completion_tokens": completion_count,
        "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("prompt_n") or 0),
        "tokens_per_second": round(tokens_per_second, 2),
    }


def run_profile_benchmark(
    mode: str,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    prompt: str | None = None,
    completion_tokens: int = 128,
    restart: bool = False,
    stop_after: bool = False,
    # None: wait as long as this profile's runtime needs to load (see
    # server_manager.READY_TIMEOUT_SECONDS; vllm.cpp takes far longer than llama.cpp).
    ready_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Time one chat completion against the profile's server.

    A server already running for ``mode`` is benchmarked as it is; one is only
    started (with ``overrides``) when none is running or ``restart`` is set.
    The completion length is capped per request via ``max_tokens``, never as a
    server flag, so the server is left as the user configured it.
    """

    params = dict(overrides or {})
    server = None if restart else _server_for_mode(mode)
    start = None
    if server is None:
        start = start_profile(
            mode=mode,
            project_root=project_root,
            model_dirs=model_dirs,
            overrides=params or None,
            stop_existing=restart,
            wait_ready=True,
            ready_timeout_seconds=ready_timeout_seconds,
        )
        server = (start.get("server") if start.get("success") else None) or _server_for_mode(mode)
        if not server:
            return {"success": False, "error": start.get("error") or "No running tracked server was available.", "start": start}
    if server.get("suspended"):
        # A paused process can't answer; the request would just hang until timeout.
        return {"success": False, "error": f"Server for '{mode}' is paused; resume it before benchmarking.", "server": server}

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
    completion_count = int(usage.get("completion_tokens") or usage.get("predicted_n") or _fallback_token_count(text))
    prompt_count = int(usage.get("prompt_tokens") or usage.get("prompt_n") or 0)
    total_count = int(usage.get("total_tokens") or (prompt_count + completion_count))
    tokens_per_second = completion_count / elapsed if completion_count else 0.0
    chars_per_second = len(text) / elapsed if text else 0.0

    benchmark = {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_id": server.get("id"),
        "endpoint": f"{base_url}/v1/chat/completions",
        "elapsed_seconds": round(elapsed, 3),
        "completion_tokens": completion_count,
        "prompt_tokens": prompt_count,
        "total_tokens": total_count,
        "tokens_per_second": round(tokens_per_second, 2),
        "chars_per_second": round(chars_per_second, 1),
        "response_chars": len(text),
        "requested_max_tokens": int(completion_tokens),
        "usage": usage,
    }
    save_benchmark_result(benchmark)
    emit_event(
        "benchmark.completed",
        server,
        tokens_per_second=benchmark["tokens_per_second"],
        completion_tokens=completion_count,
        elapsed_seconds=benchmark["elapsed_seconds"],
    )

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
