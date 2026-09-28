"""Which local servers have gateway requests running right now.

The gateway (``inferencedeck-gateway``) is its own process, and the code that
stops servers (idle release, making room for a new model) runs in
inferencedeck-web. So each gateway process keeps its counts in a small file,
``<cache>/inflight/gateway-<pid>.json``, rewritten whenever a request to a
local server starts or ends:

    {"pid": 4242, "pid_identity": "...", "servers": {
        "<server id>": {"in_flight": 2, "requests": 17, "last_request_at": "..."}}}

``requests`` only grows, so a reader can tell that requests came and went
between two looks even when ``in_flight`` is 0 both times. ``snapshot()``
merges the files of every gateway still running and deletes the files of ones
that are gone, so a gateway that crashed mid-request never pins a server as
busy.

Requests sent straight to a server's own port, not through the gateway, are
not counted here; idle release still sees them through llama-server's /slots.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fileio import atomic_write_text
from .paths import cache_dir

Snapshot = dict[str, dict[str, Any]]


def inflight_dir() -> Path:
    return cache_dir() / "inflight"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Tracker:
    """Counts one process's in-flight requests per server id and publishes them."""

    def __init__(self, path: Path | None = None, pid: int | None = None) -> None:
        self.pid = pid or os.getpid()
        self._path = path
        self._lock = threading.Lock()
        # File writes happen outside _lock, so requests never wait on disk I/O
        # (on Windows a replace can retry for seconds while a reader has the file).
        self._write_lock = threading.Lock()
        self._seq = 0
        self._written_seq = 0
        self._servers: Snapshot = {}
        self._identity: str | None = None
        self._identity_read = False

    @property
    def path(self) -> Path:
        return self._path or inflight_dir() / f"gateway-{self.pid}.json"

    def _snapshot_payload(self) -> tuple[int, str]:
        # Called with _lock held: numbers this state and serializes it.
        self._seq += 1
        payload = {"pid": self.pid, "pid_identity": self._identity, "updated_at": _now(),
                   "servers": self._servers}
        return self._seq, json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def _write(self, seq: int, text: str) -> None:
        with self._write_lock:
            if seq <= self._written_seq:
                return  # a newer state is already on disk; never go back in time
            try:
                atomic_write_text(self.path, text)
            except OSError:
                return  # a full or read-only cache must not fail the request itself
            self._written_seq = seq

    def _ensure_identity(self) -> None:
        if not self._identity_read:
            from .server_manager import process_identity

            self._identity, self._identity_read = process_identity(self.pid), True

    def begin(self, server_id: str) -> None:
        self._ensure_identity()
        with self._lock:
            entry = self._servers.setdefault(server_id, {"in_flight": 0, "requests": 0})
            entry["in_flight"] += 1
            entry["requests"] += 1
            entry["last_request_at"] = _now()
            seq, text = self._snapshot_payload()
        self._write(seq, text)

    def end(self, server_id: str) -> None:
        with self._lock:
            entry = self._servers.get(server_id)
            if entry is None:
                return
            entry["in_flight"] = max(0, entry["in_flight"] - 1)
            entry["last_request_at"] = _now()
            seq, text = self._snapshot_payload()
        self._write(seq, text)

    @contextmanager
    def track(self, server_id: str | None) -> Iterator[None]:
        """Count the enclosed request, streaming included, against ``server_id`` (None: not tracked)."""
        if not server_id:
            yield
            return
        self.begin(server_id)
        try:
            yield
        finally:
            self.end(server_id)

    def counts(self) -> Snapshot:
        with self._lock:
            return json.loads(json.dumps(self._servers))

    def close(self) -> None:
        with self._lock, self._write_lock:
            try:
                self.path.unlink()
            except OSError:
                pass


_tracker: Tracker | None = None
_tracker_lock = threading.Lock()


def tracker() -> Tracker:
    """This process's tracker; its file is removed when the process exits normally."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = Tracker()
            atexit.register(_tracker.close)
        return _tracker


# (pid, identity) -> when it was last confirmed to be that same process. Checking
# identity can spawn `ps` (macOS), and status polls read these files every few
# seconds, so a confirmed gateway is trusted for a while; liveness is still
# checked every time.
_IDENTITY_TTL_SECONDS = 30.0
_confirmed: dict[tuple[int, Any], float] = {}
_confirmed_lock = threading.Lock()


def _alive(payload: dict[str, Any]) -> bool:
    from .server_manager import _is_same_process, pid_is_running

    pid = payload.get("pid")
    if not isinstance(pid, int) or not pid_is_running(pid):
        return False
    key = (pid, payload.get("pid_identity"))
    now = time.monotonic()
    with _confirmed_lock:
        if now - _confirmed.get(key, float("-inf")) < _IDENTITY_TTL_SECONDS:
            return True
    if not _is_same_process(pid, payload.get("pid_identity")):
        return False
    with _confirmed_lock:
        _confirmed[key] = now
        for stale in [k for k, at in _confirmed.items() if now - at >= _IDENTITY_TTL_SECONDS]:
            del _confirmed[stale]
    return True


def snapshot(directory: Path | None = None) -> Snapshot:
    """Every running gateway's counts merged: ``{server id: {in_flight, requests, last_request_at}}``.

    ``requests`` is summed per gateway pid, so it still only grows while those
    gateways run; a gateway restarting can make it drop, which readers treat as
    a change like any other.
    """
    folder = directory or inflight_dir()
    merged: Snapshot = {}
    try:
        files = sorted(folder.glob("gateway-*.json"))
    except OSError:
        return merged
    for file in files:
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            continue  # mid-write on a platform without atomic replace; next look reads it
        if not isinstance(payload, dict):
            continue
        if not _alive(payload):
            try:
                file.unlink()
            except OSError:
                pass
            continue
        servers = payload.get("servers")
        if not isinstance(servers, dict):
            continue
        for server_id, entry in servers.items():
            if not isinstance(entry, dict):
                continue
            into = merged.setdefault(str(server_id), {"in_flight": 0, "requests": 0, "last_request_at": None})
            into["in_flight"] += max(0, int(entry.get("in_flight") or 0))
            into["requests"] += max(0, int(entry.get("requests") or 0))
            last = entry.get("last_request_at")
            if isinstance(last, str) and (into["last_request_at"] is None or last > into["last_request_at"]):
                into["last_request_at"] = last
    return merged


def busy_servers(counts: Snapshot | None = None) -> dict[str, int]:
    """Server id -> number of gateway requests running on it, for servers with any."""
    counts = snapshot() if counts is None else counts
    return {server_id: entry["in_flight"] for server_id, entry in counts.items() if entry.get("in_flight", 0) > 0}
