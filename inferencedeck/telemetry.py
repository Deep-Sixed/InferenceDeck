"""Local telemetry: a live snapshot of this machine and its tracked servers, plus lifecycle counters.

Telemetry observes the lifecycle; the lifecycle never depends on it. Every
collector swallows its own errors and returns what it could read, and ``emit``
never raises, so a broken nvidia-smi or an unreadable /proc can't stop a server
starting or stopping.

Nothing is sent anywhere. The control API serves the snapshot as JSON
(``/api/telemetry``) and in Prometheus text format (``/metrics``), behind the
same authentication as the rest of the API.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import hardware
from .paths import is_windows

SNAPSHOT_MAX_AGE_SECONDS = 2.0
SERVER_METRICS_TIMEOUT_SECONDS = 1.0
MAX_RECENT_EVENTS = 200
MIB = 1024 * 1024


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp() if value else None
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        # nvidia-smi reports unsupported fields as "N/A" or "[Not Supported]".
        return None


# --- lifecycle events -------------------------------------------------------


class TelemetryRegistry:
    """In-memory lifecycle counters and a ring buffer of recent events.

    Counters live for the life of the control process, as Prometheus counters
    are expected to; a restart resets them to zero.
    """

    def __init__(self, max_events: int = MAX_RECENT_EVENTS) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str, str], int] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.started_at = time.time()

    def add_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Call ``listener`` with every event recorded from now on (the history store keeps them on disk)."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit(self, event: str, **fields: Any) -> None:
        record = {"type": event, "timestamp": _now_iso(), **{k: v for k, v in fields.items() if v is not None}}
        key = (event, str(fields.get("profile") or ""), str(fields.get("runtime") or ""))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            self._events.append(record)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(record)
            except Exception:
                pass

    def counters(self) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(self._counters.items())
        return [{"event": e, "profile": p, "runtime": r, "count": n} for (e, p, r), n in items]

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        return events[-limit:] if limit > 0 else []

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._events.clear()
            self.started_at = time.time()


REGISTRY = TelemetryRegistry()


def emit(event: str, server: dict[str, Any] | None = None, **fields: Any) -> None:
    """Record a lifecycle event. Never raises."""
    try:
        if server:
            fields = {
                "server_id": server.get("id"),
                "profile": server.get("mode"),
                "runtime": server.get("runtime"),
                "pid": server.get("pid"),
                **fields,
            }
        REGISTRY.emit(event, **fields)
    except Exception:
        pass


# --- collectors -------------------------------------------------------------

_cpu_lock = threading.Lock()
_last_cpu_times: tuple[int, int] | None = None


def _linux_cpu_times() -> tuple[int, int] | None:
    """(busy, total) jiffies across all CPUs from /proc/stat."""
    try:
        first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not first or first[0] != "cpu":
        return None
    values = [int(v) for v in first[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    total = sum(values[:8])  # guest time is already counted in user/nice
    return total - idle, total


def cpu_percent() -> float | None:
    """System CPU utilisation since the previous call; None on the first call or off Linux."""
    global _last_cpu_times
    current = _linux_cpu_times()
    if current is None:
        return None
    with _cpu_lock:
        previous, _last_cpu_times = _last_cpu_times, current
    if previous is None:
        return None
    busy, total = current[0] - previous[0], current[1] - previous[1]
    if total <= 0:
        return None
    return round(100.0 * busy / total, 1)


def system_memory() -> dict[str, int | None]:
    total = available = None
    try:
        # MemAvailable counts reclaimable cache; SC_AVPHYS_PAGES (free pages) badly underestimates it.
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(":")
            if name in ("MemTotal", "MemAvailable"):
                kib = int(rest.split()[0]) * 1024
                total, available = (kib, available) if name == "MemTotal" else (total, kib)
    except (OSError, ValueError, IndexError):
        pass
    if total is None:
        info = hardware._windows_memory_info() if is_windows() else hardware._posix_memory_info()
        total, available = info.get("total_bytes"), info.get("available_bytes")
    used = total - available if total is not None and available is not None else None
    return {"total_bytes": total, "available_bytes": available, "used_bytes": used}


def _load_average() -> list[float] | None:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def nvidia_gpu_samples() -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Live NVIDIA GPU readings, and GPU memory in bytes per process ID."""
    binary = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not binary:
        return [], {}
    result = hardware._run(
        [
            binary,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
        timeout=2.5,
    )
    gpus: list[dict[str, Any]] = []
    if result and result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            used, total = _float_or_none(parts[3]), _float_or_none(parts[4])
            clock = _float_or_none(parts[7])
            gpus.append(
                {
                    "index": hardware._int_or_none(parts[0]),
                    "name": parts[1],
                    "vendor": "NVIDIA",
                    "utilization_percent": _float_or_none(parts[2]),
                    "vram_used_bytes": int(used * MIB) if used is not None else None,
                    "vram_total_bytes": int(total * MIB) if total is not None else None,
                    "temperature_c": _float_or_none(parts[5]),
                    "power_watts": _float_or_none(parts[6]),
                    "sm_clock_mhz": int(clock) if clock is not None else None,
                }
            )
    per_pid: dict[int, int] = {}
    apps = hardware._run([binary, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], timeout=2.5)
    if apps and apps.returncode == 0:
        for line in apps.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            pid, used = (hardware._int_or_none(parts[0]), _float_or_none(parts[1])) if len(parts) >= 2 else (None, None)
            if pid is not None and used is not None:
                # A process spread over several GPUs is listed once per GPU.
                per_pid[pid] = per_pid.get(pid, 0) + int(used * MIB)
    return gpus, per_pid


def process_stats(pid: int | None) -> dict[str, Any]:
    """Resident memory and CPU time for one process (Linux only; empty elsewhere)."""
    if not pid:
        return {}
    stats: dict[str, Any] = {}
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                stats["rss_bytes"] = int(line.split()[1]) * 1024
                break
        # The command name (field 2) may contain spaces; the fields after its closing paren don't.
        fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").rpartition(")")[2].split()
        ticks = os.sysconf("SC_CLK_TCK")
        stats["cpu_seconds"] = round((int(fields[11]) + int(fields[12])) / ticks, 2)  # utime + stime
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    return stats


# Inference servers' own Prometheus counters, mapped to InferenceDeck's names.
# llama-server serves these with --metrics; the vllm: names are vLLM's, which
# vllm.cpp may follow.
SERVER_METRIC_NAMES = {
    "llamacpp:prompt_tokens_total": "prompt_tokens_total",
    "llamacpp:prompt_seconds_total": "prompt_seconds_total",
    "llamacpp:tokens_predicted_total": "tokens_generated_total",
    "llamacpp:tokens_predicted_seconds_total": "generation_seconds_total",
    "llamacpp:requests_processing": "requests_active",
    "llamacpp:requests_deferred": "requests_deferred",
    "llamacpp:kv_cache_usage_ratio": "kv_cache_usage_ratio",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "tokens_generated_total",
    "vllm:num_requests_running": "requests_active",
    "vllm:num_requests_waiting": "requests_deferred",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage_ratio",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_ratio",
}


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Sample values by metric name, summed across label sets. Comments and unparsable lines are skipped."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, _, rest = line.partition("{")
            rest = rest.rpartition("}")[2]  # a label value may itself contain "}"
        else:
            name, _, rest = line.partition(" ")
        fields = rest.split()
        number = _float_or_none(fields[0]) if fields else None
        if number is not None and number == number:  # skip NaN
            values[name] = values.get(name, 0.0) + number
    return values


def fetch_server_metrics(server: dict[str, Any], timeout: float = SERVER_METRICS_TIMEOUT_SECONDS) -> dict[str, float] | None:
    """The server's own counters, or None when it has no metrics endpoint or doesn't answer in time."""
    from .server_manager import http_base

    url = f"{http_base(server.get('host'), int(server.get('port') or 0))}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read(4 * 1024 * 1024).decode("utf-8", errors="replace")
    except Exception:
        return None
    raw = parse_prometheus_text(text)
    mapped = {ours: raw[theirs] for theirs, ours in SERVER_METRIC_NAMES.items() if theirs in raw}
    return mapped or None


class _RateTracker:
    """Turns cumulative token counters into rates between consecutive snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: dict[str, tuple[float, dict[str, float]]] = {}

    def rates(self, server_id: str, now: float, counters: dict[str, float]) -> dict[str, float | None]:
        with self._lock:
            previous = self._last.get(server_id)
            self._last[server_id] = (now, counters)
        out: dict[str, float | None] = {"tokens_per_second": None, "prompt_tokens_per_second": None, "throughput_tokens_per_second": None}
        if not previous:
            return out
        then, old = previous
        delta = {k: counters[k] - old[k] for k in counters if k in old}
        if any(v < 0 for v in delta.values()):
            return out  # the server restarted and its counters reset
        # Speed while working: tokens over the time the server spent on them, so
        # an idle gap doesn't dilute it. Idle means no reading, not zero.
        for tokens, seconds, key in (
            ("tokens_generated_total", "generation_seconds_total", "tokens_per_second"),
            ("prompt_tokens_total", "prompt_seconds_total", "prompt_tokens_per_second"),
        ):
            if delta.get(seconds, 0) > 0 and tokens in delta:
                out[key] = round(delta[tokens] / delta[seconds], 2)
        wall = now - then
        if wall > 0 and "tokens_generated_total" in delta:
            out["throughput_tokens_per_second"] = round(delta["tokens_generated_total"] / wall, 2)
            if out["tokens_per_second"] is None and "generation_seconds_total" not in counters and delta["tokens_generated_total"] > 0:
                # No time counter (vLLM-style metrics): wall-clock rate is the best available.
                out["tokens_per_second"] = out["throughput_tokens_per_second"]
        return out

    def forget_except(self, server_ids: set[str]) -> None:
        with self._lock:
            for key in [k for k in self._last if k not in server_ids]:
                del self._last[key]


_rates = _RateTracker()


def inference_stats(server: dict[str, Any], now: float) -> dict[str, Any]:
    """Request and token readings for one live server; empty when it exposes none."""
    counters = fetch_server_metrics(server)
    if counters is None:
        return {"metrics_available": False}
    stats: dict[str, Any] = {"metrics_available": True}
    for key in ("requests_active", "requests_deferred", "prompt_tokens_total", "tokens_generated_total"):
        if key in counters:
            stats[key] = int(counters[key])
    if "kv_cache_usage_ratio" in counters:
        stats["kv_cache_usage_percent"] = round(100 * counters["kv_cache_usage_ratio"], 1)
    stats.update(_rates.rates(str(server.get("id")), now, counters))
    return stats


def latest_benchmarks() -> list[dict[str, Any]]:
    """The most recent benchmark per profile."""
    from .benchmark import load_benchmark_results

    latest: dict[str, dict[str, Any]] = {}
    for result in load_benchmark_results():
        mode = result.get("mode")
        if mode and str(result.get("created_at") or "") >= str(latest.get(mode, {}).get("created_at") or ""):
            latest[mode] = result
    return [
        {
            "profile": mode,
            "tokens_per_second": r.get("tokens_per_second"),
            "completion_tokens": r.get("completion_tokens"),
            "elapsed_seconds": r.get("elapsed_seconds"),
            "created_at": r.get("created_at"),
        }
        for mode, r in sorted(latest.items())
    ]


def _server_entry(server: dict[str, Any], now: float, gpu_by_pid: dict[int, int]) -> dict[str, Any]:
    started, ready = _parse_iso(server.get("started_at")), _parse_iso(server.get("ready_at"))
    running = bool(server.get("running")) and server.get("status") not in ("parked", "restoring")
    pid = server.get("pid") if running else None
    entry = {
        "server_id": server.get("id"),
        "profile": server.get("mode"),
        "runtime": server.get("runtime") or "llama.cpp",
        "pid": pid,
        "status": server.get("status"),
        "running": running,
        "suspended": bool(server.get("suspended")),
        "context_size": hardware._int_or_none(server.get("ctx_size")),
        "uptime_seconds": round(now - started, 1) if running and started else None,
        "startup_seconds": round(ready - started, 3) if started and ready and ready >= started else None,
        "gpu_memory_bytes": gpu_by_pid.get(int(pid)) if pid else None,
    }
    entry.update(process_stats(pid) if running else {})
    # A paused (SIGSTOPped) server can't answer; asking would only wait out the timeout.
    if running and not entry["suspended"] and server.get("port"):
        entry.update(inference_stats(server, now))
    return entry


# --- snapshot ---------------------------------------------------------------


def _safe(collect: Callable[[], Any], fallback: Any) -> Any:
    try:
        return collect()
    except Exception:
        return fallback


def collect_snapshot(list_servers: Callable[[], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    if list_servers is None:
        from .server_manager import list_servers
    now = time.time()
    gpus, gpu_by_pid = _safe(nvidia_gpu_samples, ([], {}))
    servers = _safe(list_servers, [])
    _rates.forget_except({str(s.get("id")) for s in servers})
    return {
        "version": 1,
        "timestamp": _now_iso(),
        "control_uptime_seconds": round(now - REGISTRY.started_at, 1),
        "system": {
            "cpu_percent": _safe(cpu_percent, None),
            "load_average": _safe(_load_average, None),
            "memory": _safe(system_memory, {}),
        },
        "gpus": gpus,
        "servers": [_safe(lambda s=s: _server_entry(s, now, gpu_by_pid), {"server_id": s.get("id")}) for s in servers],
        "benchmarks": _safe(latest_benchmarks, []),
        "lifecycle": {"counters": REGISTRY.counters(), "recent_events": REGISTRY.recent_events()},
    }


_snapshot_lock = threading.Lock()
_snapshot_cache: tuple[float, dict[str, Any]] | None = None


def snapshot(max_age: float = SNAPSHOT_MAX_AGE_SECONDS) -> dict[str, Any]:
    """The current snapshot, reused for ``max_age`` seconds so a UI poll and a Prometheus scrape share one nvidia-smi call."""
    global _snapshot_cache
    with _snapshot_lock:
        if _snapshot_cache and time.monotonic() - _snapshot_cache[0] < max_age:
            return _snapshot_cache[1]
        result = collect_snapshot()
        _snapshot_cache = (time.monotonic(), result)
        return result


# --- Prometheus -------------------------------------------------------------


def _label_value(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass(frozen=True)
class Sample:
    """One metric reading, independent of export format (Prometheus text or OTLP)."""

    name: str
    kind: str  # "gauge" or "counter"
    help: str
    value: float
    labels: dict[str, Any] = field(default_factory=dict)
    # When a counter started counting (epoch seconds); OTLP needs it, Prometheus doesn't.
    start: float | None = None


class _MetricSet:
    def __init__(self) -> None:
        self.samples: list[Sample] = []

    def add(
        self,
        name: str,
        kind: str,
        help_text: str,
        value: Any,
        labels: dict[str, Any] | None = None,
        start: float | None = None,
    ) -> None:
        if value is not None:
            self.samples.append(Sample(name, kind, help_text, float(value), dict(labels or {}), start))


def render_prometheus(snap: dict[str, Any]) -> str:
    families: dict[str, tuple[Sample, list[str]]] = {}
    for sample in metric_samples(snap):
        family = families.setdefault(sample.name, (sample, []))
        label_text = ",".join(f'{k}="{_label_value(v)}"' for k, v in sample.labels.items())
        rendered = str(int(sample.value)) if sample.value.is_integer() else repr(sample.value)
        family[1].append(f"{sample.name}{{{label_text}}} {rendered}" if label_text else f"{sample.name} {rendered}")
    lines: list[str] = []
    for name, (first, rows) in families.items():
        lines += [f"# HELP {name} {first.help}", f"# TYPE {name} {first.kind}", *rows]
    return "\n".join(lines) + "\n"


def metric_samples(snap: dict[str, Any]) -> list[Sample]:
    """Every metric in a snapshot, in export order."""
    out = _MetricSet()
    taken_at = _parse_iso(snap.get("timestamp")) or time.time()
    out.add("inferencedeck_control_uptime_seconds", "gauge", "Seconds since the control process started.", snap.get("control_uptime_seconds"))

    system = snap.get("system") or {}
    memory = system.get("memory") or {}
    out.add("inferencedeck_cpu_utilization_percent", "gauge", "System CPU utilisation.", system.get("cpu_percent"))
    for i, window in enumerate(("1m", "5m", "15m")):
        load = system.get("load_average") or []
        out.add("inferencedeck_load_average", "gauge", "System load average.", load[i] if i < len(load) else None, {"window": window})
    out.add("inferencedeck_memory_total_bytes", "gauge", "Physical memory.", memory.get("total_bytes"))
    out.add("inferencedeck_memory_available_bytes", "gauge", "Memory available to new processes.", memory.get("available_bytes"))

    for gpu in snap.get("gpus") or []:
        labels = {"gpu": gpu.get("index"), "name": gpu.get("name")}
        out.add("inferencedeck_gpu_utilization_percent", "gauge", "GPU core utilisation.", gpu.get("utilization_percent"), labels)
        out.add("inferencedeck_gpu_vram_used_bytes", "gauge", "GPU memory in use.", gpu.get("vram_used_bytes"), labels)
        out.add("inferencedeck_gpu_vram_total_bytes", "gauge", "GPU memory capacity.", gpu.get("vram_total_bytes"), labels)
        out.add("inferencedeck_gpu_temperature_celsius", "gauge", "GPU temperature.", gpu.get("temperature_c"), labels)
        out.add("inferencedeck_gpu_power_watts", "gauge", "GPU power draw.", gpu.get("power_watts"), labels)
        out.add("inferencedeck_gpu_sm_clock_mhz", "gauge", "GPU core clock.", gpu.get("sm_clock_mhz"), labels)

    # One series per profile: a profile runs at most once, but a parked or stopped
    # record can sit beside the live one, and duplicate series are invalid.
    by_profile: dict[tuple[str, str], dict[str, Any]] = {}
    for server in snap.get("servers") or []:
        key = (str(server.get("profile") or ""), str(server.get("runtime") or ""))
        if key not in by_profile or (server.get("running") and not by_profile[key].get("running")):
            by_profile[key] = server
    for (profile, runtime), server in sorted(by_profile.items()):
        labels = {"profile": profile, "runtime": runtime}
        uptime = server.get("uptime_seconds")
        counting_since = taken_at - uptime if uptime is not None else None
        out.add("inferencedeck_server_up", "gauge", "1 when the profile's server process is running.", 1 if server.get("running") else 0, labels)
        out.add("inferencedeck_server_suspended", "gauge", "1 when the server is paused with its model still loaded.", 1 if server.get("suspended") else 0, labels)
        out.add("inferencedeck_server_uptime_seconds", "gauge", "Seconds since the server was started.", server.get("uptime_seconds"), labels)
        out.add("inferencedeck_server_startup_seconds", "gauge", "Seconds from launch until the server answered its health check.", server.get("startup_seconds"), labels)
        out.add("inferencedeck_server_context_size", "gauge", "Configured context size in tokens.", server.get("context_size"), labels)
        out.add("inferencedeck_server_resident_memory_bytes", "gauge", "Server process resident memory.", server.get("rss_bytes"), labels)
        out.add("inferencedeck_server_cpu_seconds_total", "counter", "Server process CPU time.", server.get("cpu_seconds"), labels, counting_since)
        out.add("inferencedeck_server_gpu_memory_bytes", "gauge", "GPU memory held by the server process.", server.get("gpu_memory_bytes"), labels)
        out.add("inferencedeck_server_requests_active", "gauge", "Requests the server is processing.", server.get("requests_active"), labels)
        out.add("inferencedeck_server_requests_deferred", "gauge", "Requests waiting for a free slot.", server.get("requests_deferred"), labels)
        out.add("inferencedeck_server_prompt_tokens_total", "counter", "Prompt tokens processed since the server started.", server.get("prompt_tokens_total"), labels, counting_since)
        out.add("inferencedeck_server_generated_tokens_total", "counter", "Tokens generated since the server started.", server.get("tokens_generated_total"), labels, counting_since)
        out.add("inferencedeck_server_generation_tokens_per_second", "gauge", "Generation speed while generating, since the previous reading.", server.get("tokens_per_second"), labels)
        out.add("inferencedeck_server_prompt_tokens_per_second", "gauge", "Prompt processing speed while processing, since the previous reading.", server.get("prompt_tokens_per_second"), labels)
        out.add("inferencedeck_server_kv_cache_usage_ratio", "gauge", "Share of the KV cache in use.", None if server.get("kv_cache_usage_percent") is None else server["kv_cache_usage_percent"] / 100, labels)

    for bench in snap.get("benchmarks") or []:
        out.add("inferencedeck_benchmark_tokens_per_second", "gauge", "Generation speed from the profile's most recent benchmark.", bench.get("tokens_per_second"), {"profile": bench.get("profile")})

    control_uptime = snap.get("control_uptime_seconds")
    control_start = taken_at - control_uptime if control_uptime is not None else None
    for counter in (snap.get("lifecycle") or {}).get("counters") or []:
        out.add(
            "inferencedeck_lifecycle_events_total",
            "counter",
            "Server lifecycle events seen by this control process.",
            counter["count"],
            {"event": counter["event"], "profile": counter["profile"], "runtime": counter["runtime"]},
            control_start,
        )
    return out.samples
