"""Telemetry history: periodic samples of the live snapshot, kept locally for graphs.

The web process runs one sampler thread. It keeps full-resolution samples for
the last hour in memory and one-minute averages on disk, in one JSON-lines file
per UTC day under the cache directory, so a restart doesn't lose the graphs.
Lifecycle events are written to the same files so charts can mark starts,
releases and failures.

Like the rest of telemetry, history never gates the lifecycle: the sampler
swallows its own errors and a failed write only loses that sample.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import telemetry
from .paths import cache_dir

HISTORY_DIRNAME = "telemetry"
RAW_WINDOW_SECONDS = 3600
MINUTE = 60
# Range name -> (span in seconds, bucket in seconds; None means the sample interval).
RANGES: dict[str, tuple[int, int | None]] = {
    "15m": (900, None),
    "1h": (3600, None),
    "6h": (6 * 3600, 60),
    "24h": (24 * 3600, 120),
    "7d": (7 * 86400, 900),
}

Row = tuple[float, dict[str, float]]


def flatten(snap: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Split a snapshot into numeric series values and the labels/limits that describe them."""
    values: dict[str, float] = {}
    meta: dict[str, Any] = {"labels": {}, "limits": {}}

    def put(key: str, value: Any) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values[key] = float(value)

    system = snap.get("system") or {}
    memory = system.get("memory") or {}
    put("cpu_percent", system.get("cpu_percent"))
    put("memory_used_bytes", memory.get("used_bytes"))
    if memory.get("total_bytes"):
        meta["limits"]["memory_used_bytes"] = memory["total_bytes"]
    for gpu in snap.get("gpus") or []:
        if gpu.get("index") is None:
            continue
        prefix = f"gpu{gpu['index']}"
        meta["labels"][prefix] = gpu.get("name") or prefix.upper()
        for field in ("utilization_percent", "vram_used_bytes", "temperature_c", "power_watts", "sm_clock_mhz"):
            put(f"{prefix}.{field}", gpu.get(field))
        if gpu.get("vram_total_bytes"):
            meta["limits"][f"{prefix}.vram_used_bytes"] = gpu["vram_total_bytes"]
    for server in snap.get("servers") or []:
        profile = server.get("profile")
        if not profile or not server.get("running"):
            continue
        for field in ("gpu_memory_bytes", "rss_bytes"):
            put(f"server:{profile}.{field}", server.get(field))
    return values, meta


def average(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for values in rows:
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: round(sums[key] / counts[key], 3) for key in sums}


def bucket(rows: list[Row], start: float, end: float, step: int) -> tuple[list[int], dict[str, list[float | None]]]:
    """Average ``rows`` into fixed ``step``-second slots from ``start`` to ``end``.

    Every slot gets a timestamp; a slot with no sample holds None for every
    series, so a chart shows a gap where the control process wasn't running.
    """
    first = int(start // step) * step
    slots = list(range(first, int(end // step) * step + 1, step))
    grouped: dict[int, list[dict[str, float]]] = {}
    for t, values in rows:
        if start <= t <= end:
            grouped.setdefault(int(t // step) * step, []).append(values)
    averaged = {slot: average(group) for slot, group in grouped.items()}
    keys = sorted({key for values in averaged.values() for key in values})
    series = {key: [averaged.get(slot, {}).get(key) for slot in slots] for key in keys}
    return slots, series


class TelemetryHistory:
    def __init__(
        self,
        directory: Path | None = None,
        interval: float = 5.0,
        retention_days: int = 7,
        collect: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = directory or cache_dir() / HISTORY_DIRNAME
        self.interval = max(1.0, float(interval))
        self.retention_days = max(1, int(retention_days))
        self._collect = collect or telemetry.snapshot
        self._clock = clock
        self._lock = threading.Lock()
        self._raw: deque[Row] = deque(maxlen=int(RAW_WINDOW_SECONDS / self.interval) + 2)
        self._minutes: deque[Row] = deque(maxlen=self.retention_days * 1440 + 2)
        self._events: deque[dict[str, Any]] = deque(maxlen=5000)
        self._pending: list[dict[str, float]] = []
        self._pending_minute: int | None = None
        self._meta: dict[str, Any] = {"labels": {}, "limits": {}}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- storage ------------------------------------------------------------

    def _file_for(self, t: float) -> Path:
        return self.directory / f"{datetime.fromtimestamp(t, timezone.utc):%Y-%m-%d}.jsonl"

    def _append(self, t: float, record: dict[str, Any]) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._file_for(t).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _prune(self, cutoff: float) -> list[Path]:
        """Delete day files wholly older than ``cutoff``; return the rest, oldest first."""
        oldest_kept = f"{datetime.fromtimestamp(cutoff, timezone.utc) - timedelta(days=1):%Y-%m-%d}.jsonl"
        kept = []
        try:
            files = sorted(self.directory.glob("*.jsonl"))
        except OSError:
            return []
        for path in files:
            if path.name >= oldest_kept:
                kept.append(path)
                continue
            try:
                path.unlink()
            except OSError:
                pass
        return kept

    def load(self) -> None:
        """Read the minute averages and events still inside the retention window, and delete older files."""
        cutoff = self._clock() - self.retention_days * 86400
        minutes: list[Row] = []
        events: list[dict[str, Any]] = []
        for path in self._prune(cutoff):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    record = json.loads(line)
                    t = float(record["t"])
                except (ValueError, KeyError, TypeError):
                    continue  # a torn last line after a crash
                if t < cutoff:
                    continue
                if record.get("k") == "m" and isinstance(record.get("v"), dict):
                    minutes.append((t, {k: float(v) for k, v in record["v"].items() if isinstance(v, (int, float))}))
                elif record.get("k") == "e" and isinstance(record.get("e"), dict):
                    events.append({**record["e"], "t": t})
        minutes.sort(key=lambda row: row[0])
        with self._lock:
            self._minutes.extend(minutes)
            self._events.extend(sorted(events, key=lambda e: e["t"]))

    # --- sampling -----------------------------------------------------------

    def record(self, snap: dict[str, Any], t: float | None = None) -> None:
        """Add one sample; closes out and persists the previous minute when a new one begins."""
        t = self._clock() if t is None else t
        values, meta = flatten(snap)
        minute = int(t // MINUTE) * MINUTE
        closed: Row | None = None
        with self._lock:
            self._raw.append((t, values))
            self._meta = meta
            if self._pending_minute is not None and minute != self._pending_minute and self._pending:
                closed = (float(self._pending_minute), average(self._pending))
                self._minutes.append(closed)
                self._pending = []
            self._pending_minute = minute
            self._pending.append(values)
        if closed:
            self._append(closed[0], {"k": "m", "t": closed[0], "v": closed[1]})
            if int(closed[0]) % 86400 == 0:  # a new UTC day's file is starting
                self._prune(t - self.retention_days * 86400)

    def record_event(self, event: dict[str, Any]) -> None:
        t = self._clock()
        with self._lock:
            self._events.append({**event, "t": t})
        self._append(t, {"k": "e", "t": t, "e": event})

    def sample_once(self) -> None:
        try:
            self.record(self._collect())
        except Exception:
            pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.load()
        telemetry.REGISTRY.add_listener(self.record_event)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="inferencedeck-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        telemetry.REGISTRY.remove_listener(self.record_event)
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.monotonic()
            self.sample_once()
            self._stop.wait(max(0.0, self.interval - (time.monotonic() - began)))

    # --- queries ------------------------------------------------------------

    def query(self, range_name: str = "1h") -> dict[str, Any]:
        if range_name not in RANGES:
            raise ValueError(f"range must be one of {', '.join(RANGES)}")
        span, step = RANGES[range_name]
        end = self._clock()
        start = end - span
        with self._lock:
            if step is None:
                rows = list(self._raw)
                step = int(self.interval)
            else:
                rows = list(self._minutes)
                if self._pending and self._pending_minute is not None:
                    rows.append((float(self._pending_minute), average(self._pending)))
            events = [e for e in self._events if start <= e["t"] <= end]
            meta = json.loads(json.dumps(self._meta))
        timestamps, series = bucket(rows, start, end, step)
        return {
            "range": range_name,
            "step_seconds": step,
            "sample_seconds": self.interval,
            "timestamps": timestamps,
            "series": series,
            "labels": meta["labels"],
            "limits": meta["limits"],
            "events": events,
        }


_history: TelemetryHistory | None = None
_history_lock = threading.Lock()


def start_sampler(interval: float, retention_days: int) -> TelemetryHistory | None:
    """Start the process-wide sampler (once). Returns None when history is turned off."""
    global _history
    if interval <= 0:
        return None
    with _history_lock:
        if _history is None:
            _history = TelemetryHistory(interval=interval, retention_days=retention_days)
            _history.start()
        return _history


def current() -> TelemetryHistory | None:
    return _history
