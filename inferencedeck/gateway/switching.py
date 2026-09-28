"""Pick, and optionally load, the local model a request names.

A request's ``model`` is matched against the tracked servers and the profiles
(by mode, display name or ``alias``). A running match serves it directly. With
switching on, a match that isn't loaded is loaded on demand, like Ollama or
llama-cpp-python's multi-model server: the other local servers are released
(stopped, settings kept for Restore) and the profile is started or restored.

The gateway never launches processes itself. Like the trays, it asks the
``inferencedeck-web`` control API (``INFERENCEDECK_URL``), the one process that
owns server state. Switches are serialized, and a switch waits (bounded) for
requests the gateway is still streaming from the server it is about to release.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ..tray import TIMEOUTS, ApiClient, ApiError
from .ir import GatewayError

PROFILE_CACHE_SECONDS = 30.0
DRAIN_TIMEOUT_SECONDS = 120.0


def _names(item: dict[str, Any]) -> set[str]:
    params = item.get("params") or item.get("overrides") or {}
    values = (item.get("mode"), item.get("name"), params.get("alias"))
    return {str(value).strip().lower() for value in values if value and str(value).strip()}


def matches(item: dict[str, Any], model: str | None) -> bool:
    return bool(model) and model.strip().lower() in _names(item)


def is_live(server: dict[str, Any]) -> bool:
    return bool(server.get("running")) and not server.get("suspended")


class InFlight:
    """Requests the gateway is serving per server, so a switch can wait for them."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._cond = threading.Condition()

    @contextmanager
    def lease(self, server_id: str) -> Iterator[None]:
        with self._cond:
            self._counts[server_id] += 1
        try:
            yield
        finally:
            with self._cond:
                self._counts[server_id] -= 1
                if self._counts[server_id] <= 0:
                    del self._counts[server_id]
                self._cond.notify_all()

    def count(self, server_id: str) -> int:
        with self._cond:
            return self._counts.get(server_id, 0)

    def wait_idle(self, server_ids: list[str], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while any(self._counts.get(sid, 0) for sid in server_ids):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True


class ModelSwitcher:
    def __init__(
        self,
        client: ApiClient | None = None,
        *,
        drain_timeout: float = DRAIN_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client or ApiClient()
        self.drain_timeout = drain_timeout
        self.clock = clock
        self.inflight = InFlight()
        self._switch_lock = threading.Lock()
        self._profiles: list[dict[str, Any]] = []
        self._profiles_at: float | None = None

    def profiles(self) -> list[dict[str, Any]]:
        """Launchable profiles, from the control API (cached briefly: discovery scans disks)."""
        now = self.clock()
        if self._profiles_at is None or now - self._profiles_at > PROFILE_CACHE_SECONDS:
            try:
                self._profiles = [p for p in self.client.profiles() if p.get("launchable")]
            except ApiError as exc:
                raise GatewayError(503, f"model switching needs inferencedeck-web: {exc}", "overloaded") from exc
            self._profiles_at = now
        return self._profiles

    def find_profile(self, model: str | None) -> dict[str, Any] | None:
        return next((p for p in self.profiles() if matches(p, model)), None)

    def ensure_loaded(self, profile: dict[str, Any], servers: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
        """Release every other running local server, then resume, restore or start ``profile``."""
        mode = str(profile["mode"])
        with self._switch_lock:
            current = servers()
            ready = next((s for s in current if s.get("mode") == mode and is_live(s)), None)
            if ready:
                return ready  # another request switched to it while we waited
            # Paused servers still hold their VRAM, so they are released too.
            others = [s for s in current if s.get("running") and s.get("mode") != mode]
            self.inflight.wait_idle([str(s["id"]) for s in others], self.drain_timeout)
            for server in others:
                self._call("release", {"server_id": server["id"]}, f"release {server.get('mode')}")
            paused = next((s for s in current if s.get("mode") == mode and s.get("running") and s.get("suspended")), None)
            parked = next((s for s in current if s.get("mode") == mode and s.get("status") == "parked"), None)
            if paused:
                result = self._call("resume", {"server_id": paused["id"]}, f"resume {mode}")
            elif parked:
                result = self._call("restore", {"server_id": parked["id"]}, f"restore {mode}")
            else:
                result = self._call("start", {"mode": mode}, f"start {mode}")
            server = result.get("server")
            if not isinstance(server, dict) or not is_live(server):
                server = next((s for s in servers() if s.get("mode") == mode and is_live(s)), None)
            if not server:
                raise GatewayError(503, f"{mode} did not come up: {result.get('error') or 'no running server'}", "overloaded")
            return server

    def _call(self, action: str, body: dict[str, Any], what: str) -> dict[str, Any]:
        try:
            result = self.client.request(f"/api/{action}", body, timeout=TIMEOUTS.get(action, 20))
        except ApiError as exc:
            raise GatewayError(503, f"could not {what}: {exc}", "overloaded") from exc
        if not result.get("success", True):
            raise GatewayError(503, f"could not {what}: {result.get('error') or result.get('message')}", "overloaded")
        return result
