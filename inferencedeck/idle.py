"""Release a server's GPU after it has sat idle for a while.

InferenceDeck is not in the request path, so it cannot count requests itself.
Instead it polls llama-server's GET /slots: a slot that is processing means the
server is busy, and a slot whose task id changed since the last poll means a
request came and went in between. A server with neither for its whole idle
window is released exactly like the Release GPU control (stopped and parked),
so Restore brings it back with the same settings.

The idle window comes from the server's ``idle_release_seconds`` (a profile
param or launch override) or, when that is unset, the app config's
``idle_release_seconds``. Zero turns it off, which is the default.

When /slots cannot be read (disabled with --no-slots, behind --api-key, or the
server is briefly unresponsive) the server is treated as active: InferenceDeck
never stops a server it cannot see is idle.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from .config import AppConfig
from .server_manager import _probe_base, _update_server, list_servers, release_gpu

DEFAULT_POLL_SECONDS = 15
MAX_IDLE_RELEASE_SECONDS = 7 * 24 * 3600

Slots = list[dict[str, Any]]


def idle_window(server: dict[str, Any], config: AppConfig) -> int:
    """Seconds of inactivity before ``server`` is released; 0 means never."""
    value = server.get("idle_release_seconds")
    if value is None:
        value = config.idle_release_seconds
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def probe_slots(host: str, port: int, timeout: float = 2) -> Slots | None:
    """llama-server's slot list, or None when it cannot be read."""
    try:
        with urllib.request.urlopen(f"{_probe_base(host, port)}/slots", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    return [slot for slot in payload if isinstance(slot, dict)]


def slots_state(slots: Slots) -> tuple[bool, tuple[tuple[str, str], ...]]:
    """(busy, fingerprint). The fingerprint changes whenever a slot takes a new task."""
    busy = any(bool(slot.get("is_processing")) for slot in slots)
    fingerprint = tuple(sorted((str(slot.get("id")), str(slot.get("id_task"))) for slot in slots))
    return busy, fingerprint


def _watchable(server: dict[str, Any]) -> bool:
    # Only a ready, unpaused server: a paused one cannot answer /slots, and one
    # still loading or already released has nothing to release.
    return bool(server.get("running")) and server.get("status") == "running" and not server.get("suspended")


class IdleMonitor:
    """One ``check()`` per poll; keeps each server's last-activity time in memory.

    A server first seen by this monitor (including after inferencedeck-web
    restarts) gets a full idle window from that moment.
    """

    def __init__(
        self,
        *,
        probe: Callable[[str, int], Slots | None] = probe_slots,
        release: Callable[..., dict[str, Any]] = release_gpu,
        clock: Callable[[], float] = time.monotonic,
        config_loader: Callable[[], AppConfig] = AppConfig.load,
    ) -> None:
        self._probe = probe
        self._release = release
        self._clock = clock
        self._config_loader = config_loader
        # server id -> (slot fingerprint, monotonic time of last activity)
        self._seen: dict[str, tuple[Any, float]] = {}

    def _read(self, server: dict[str, Any]) -> tuple[bool, Any] | None:
        slots = self._probe(str(server.get("host") or "127.0.0.1"), int(server.get("port") or 8080))
        return None if slots is None else slots_state(slots)

    def check(self) -> list[dict[str, Any]]:
        """Release every server idle past its window; returns the release results."""
        now = self._clock()
        config = self._config_loader()
        released: list[dict[str, Any]] = []
        watched: set[str] = set()
        for server in list_servers():
            server_id = str(server.get("id") or "")
            if not server_id or not _watchable(server):
                continue
            window = idle_window(server, config)
            if window <= 0:
                continue
            watched.add(server_id)
            state = self._read(server)
            previous = self._seen.get(server_id)
            if state is None or state[0] or previous is None or previous[0] != state[1]:
                # Unreadable, busy, new to us, or served a request since the last poll.
                self._seen[server_id] = (None if state is None else state[1], now)
                continue
            if now - previous[1] < window:
                continue
            # Look once more right before stopping, so a request that has just
            # arrived is not cut off.
            if self._read(server) != (False, previous[0]):
                self._seen[server_id] = (previous[0], now)
                continue
            result = self._release(server_id=server_id)
            self._seen.pop(server_id, None)
            if result.get("success"):
                _update_server(server_id, {"released_by": "idle", "idle_seconds": int(now - previous[1])})
                released.append(result)
        for server_id in set(self._seen) - watched:
            del self._seen[server_id]
        return released


def start_idle_monitor(
    interval_seconds: float = DEFAULT_POLL_SECONDS,
    stop: threading.Event | None = None,
    monitor: IdleMonitor | None = None,
) -> threading.Thread:
    """Run ``IdleMonitor.check`` every ``interval_seconds`` on a daemon thread."""
    stop_event = stop or threading.Event()
    idle_monitor = monitor or IdleMonitor()

    def loop() -> None:
        while not stop_event.is_set():
            try:
                idle_monitor.check()
            except Exception:
                # A bad poll (state file mid-write, odd /slots payload) must not
                # kill the monitor; the next poll tries again.
                pass
            stop_event.wait(interval_seconds)

    thread = threading.Thread(target=loop, name="inferencedeck-idle-release", daemon=True)
    thread.start()
    return thread

