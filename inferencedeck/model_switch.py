"""Switch this machine to one model: release the other servers, then load it.

This is what the gateway's ``--switch-models`` asks for (``POST /api/switch``).
It runs here, in inferencedeck-web, not in a gateway, because several gateway
processes may share the machine: one gateway's view of its own requests says
nothing about a stream another gateway is serving. So a switch

- is serialized for the whole machine (a thread lock plus a lock file, so two
  web processes or a CLI can't interleave either);
- re-reads server state once it holds that lock, so a request that waited
  behind another switch to the same model just uses it;
- waits (bounded) until no gateway has a request running on a server it is
  about to release, reading every gateway's counts (see inflight.py), and
  checks again immediately before each release;
- gives up with ``reason: "switch_busy"`` when that wait runs out, instead of
  stopping a server under a running request. Stopping one anyway is an
  explicit Release, never a side effect of a switch.

A gateway that picked a server just before it was released can still send it
one request that fails; the window is the gap between the last in-flight check
and the stop, not the drain time.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import server_manager as sm
from .fileio import lock_file, unlock_file
from .inflight import busy_servers

DRAIN_TIMEOUT_SECONDS = 120.0
MAX_DRAIN_TIMEOUT_SECONDS = 600.0
POLL_SECONDS = 0.25

_SWITCH_LOCK = threading.Lock()


@contextmanager
def switch_lock() -> Iterator[None]:
    with _SWITCH_LOCK:
        with open(sm.state_path().with_name("switch.lock"), "a+b") as fh:
            lock_file(fh)
            try:
                yield
            finally:
                unlock_file(fh)


def _live(server: dict[str, Any]) -> bool:
    return bool(server.get("running")) and not server.get("suspended")


def switch_to(
    mode: str,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    *,
    drain_timeout: float = DRAIN_TIMEOUT_SECONDS,
    busy: Callable[[], dict[str, int]] = busy_servers,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Make ``mode`` the one running local server, without cutting off gateway requests."""

    if not mode:
        return {"success": False, "error": "mode is required"}
    with switch_lock():
        current = sm.list_servers()
        ready = next((s for s in current if s.get("mode") == mode and _live(s)), None)
        if ready:
            return {"success": True, "server": ready, "released": [], "switched": False}
        # Paused servers still hold their VRAM, so they are released too.
        pending = [s for s in current if s.get("running") and s.get("mode") != mode]
        released: list[str] = []
        deadline = clock() + drain_timeout
        while pending:
            counts = busy()
            serving = {s.get("mode") or s["id"]: n for s in pending if (n := counts.get(str(s["id"]), 0))}
            if serving:
                if clock() >= deadline:
                    names = ", ".join(f"{name} ({n} request{'s' if n != 1 else ''})" for name, n in serving.items())
                    return {
                        "success": False,
                        "reason": "switch_busy",
                        "error": f"Not switching to {mode}: still serving {names} after {drain_timeout:g}s. Try again shortly.",
                        "busy": serving,
                        "released": released,
                    }
                sleep(POLL_SECONDS)
                continue
            # Nothing was running on them a moment ago. Check each one again right
            # before releasing it, since a request may have started since.
            while pending:
                server = pending[0]
                if busy().get(str(server["id"])):
                    break  # back to waiting for it
                result = sm.release_gpu(server_id=server["id"])
                if not result.get("success"):
                    return {
                        "success": False,
                        "error": f"Could not release {server.get('mode')}: {result.get('error')}",
                        "released": released,
                    }
                released.append(str(server["id"]))
                pending.pop(0)

        paused = next((s for s in current if s.get("mode") == mode and s.get("running") and s.get("suspended")), None)
        parked = next((s for s in current if s.get("mode") == mode and s.get("status") == sm.PARKED), None)
        if paused:
            result = sm.resume_server(server_id=paused["id"])
        elif parked:
            result = sm.restore_server(parked["id"], project_root=project_root, model_dirs=model_dirs)
        else:
            result = sm.start_profile(mode, project_root=project_root, model_dirs=model_dirs)
        return {**result, "released": released, "switched": True}
