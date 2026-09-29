"""Pick, and optionally load, the local model a request names.

A request's ``model`` is matched against the tracked servers and the profiles
(by mode, display name or ``alias``). A running match serves it directly. With
switching on, a match that isn't loaded is loaded on demand, like Ollama or
llama-cpp-python's multi-model server: the other local servers are released
(stopped, settings kept for Restore) and the profile is started or restored.

The gateway never launches or stops processes itself. It asks the
``inferencedeck-web`` control API (``INFERENCEDECK_URL``), the one process that
owns server state, to switch (``POST /api/switch``, see model_switch.py). The
switch runs there because several gateway processes may share a machine: it is
serialized machine-wide and waits (bounded) until no gateway has a request
running on a server it would release, then fails with a 503 rather than cut a
request off.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ..tray import TIMEOUTS, ApiClient, ApiError
from .ir import GatewayError

PROFILE_CACHE_SECONDS = 30.0
DRAIN_TIMEOUT_SECONDS = 120.0
# Long enough for the drain, releasing the other servers and a cold start.
SWITCH_TIMEOUT_PADDING = TIMEOUTS["restart"] + TIMEOUTS["release"]


def profile_names(item: dict[str, Any]) -> set[str]:
    params = item.get("params") or item.get("overrides") or {}
    values = (item.get("mode"), item.get("name"), params.get("alias"))
    return {str(value).strip().lower() for value in values if value and str(value).strip()}


def matches(item: dict[str, Any], model: str | None) -> bool:
    return bool(model) and model.strip().lower() in profile_names(item)


def is_live(server: dict[str, Any]) -> bool:
    return bool(server.get("running")) and not server.get("suspended")


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
        """Have inferencedeck-web release the other local servers and load ``profile``."""
        mode = str(profile["mode"])
        # Only saves this process's concurrent requests a round trip each; the
        # switch itself is serialized, machine-wide, by inferencedeck-web.
        with self._switch_lock:
            ready = next((s for s in servers() if s.get("mode") == mode and is_live(s)), None)
            if ready:
                return ready  # another request switched to it while we waited
            result = self._call("switch", {"mode": mode, "drain_timeout": self.drain_timeout}, f"switch to {mode}",
                                timeout=self.drain_timeout + SWITCH_TIMEOUT_PADDING)
            server = result.get("server")
            if not isinstance(server, dict) or not is_live(server):
                server = next((s for s in servers() if s.get("mode") == mode and is_live(s)), None)
            if not server:
                raise GatewayError(503, f"{mode} did not come up: {result.get('error') or 'no running server'}", "overloaded")
            return server

    def _call(self, action: str, body: dict[str, Any], what: str, timeout: float = 20) -> dict[str, Any]:
        try:
            result = self.client.request(f"/api/{action}", body, timeout=timeout)
        except ApiError as exc:
            raise GatewayError(503, f"could not {what}: {exc}", "overloaded") from exc
        if not result.get("success", True):
            raise GatewayError(503, f"could not {what}: {result.get('error') or result.get('message')}", "overloaded")
        return result
