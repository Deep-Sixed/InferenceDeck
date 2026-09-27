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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import hardware
from .paths import is_windows

SNAPSHOT_MAX_AGE_SECONDS = 2.0
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
        self.started_at = time.time()

    def emit(self, event: str, **fields: Any) -> None:
        record = {"type": event, "timestamp": _now_iso(), **{k: v for k, v in fields.items() if v is not None}}
        key = (event, str(fields.get("profile") or ""), str(fields.get("runtime") or ""))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            self._events.append(record)

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


class _Exposition:
    def __init__(self) -> None:
        self._families: dict[str, tuple[str, str, list[str]]] = {}

    def add(self, name: str, kind: str, help_text: str, value: Any, labels: dict[str, Any] | None = None) -> None:
        if value is None:
            return
        family = self._families.setdefault(name, (kind, help_text, []))
        label_text = ",".join(f'{k}="{_label_value(v)}"' for k, v in (labels or {}).items())
        number = float(value)
        rendered = str(int(number)) if number.is_integer() else repr(number)
        family[2].append(f"{name}{{{label_text}}} {rendered}" if label_text else f"{name} {rendered}")

    def render(self) -> str:
        lines: list[str] = []
        for name, (kind, help_text, samples) in self._families.items():
            if samples:
                lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", *samples]
        return "\n".join(lines) + "\n"


def render_prometheus(snap: dict[str, Any]) -> str:
    out = _Exposition()
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
        out.add("inferencedeck_server_up", "gauge", "1 when the profile's server process is running.", 1 if server.get("running") else 0, labels)
        out.add("inferencedeck_server_suspended", "gauge", "1 when the server is paused with its model still loaded.", 1 if server.get("suspended") else 0, labels)
        out.add("inferencedeck_server_uptime_seconds", "gauge", "Seconds since the server was started.", server.get("uptime_seconds"), labels)
        out.add("inferencedeck_server_startup_seconds", "gauge", "Seconds from launch until the server answered its health check.", server.get("startup_seconds"), labels)
        out.add("inferencedeck_server_context_size", "gauge", "Configured context size in tokens.", server.get("context_size"), labels)
        out.add("inferencedeck_server_resident_memory_bytes", "gauge", "Server process resident memory.", server.get("rss_bytes"), labels)
        out.add("inferencedeck_server_cpu_seconds_total", "counter", "Server process CPU time.", server.get("cpu_seconds"), labels)
        out.add("inferencedeck_server_gpu_memory_bytes", "gauge", "GPU memory held by the server process.", server.get("gpu_memory_bytes"), labels)

    for counter in (snap.get("lifecycle") or {}).get("counters") or []:
        out.add(
            "inferencedeck_lifecycle_events_total",
            "counter",
            "Server lifecycle events seen by this control process.",
            counter["count"],
            {"event": counter["event"], "profile": counter["profile"], "runtime": counter["runtime"]},
        )
    return out.render()
