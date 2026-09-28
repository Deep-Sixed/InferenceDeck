"""Will a new server fit in GPU memory next to the ones already running?

The fit estimator answers "does this profile fit on this GPU"; this module
answers "does it fit on this GPU *now*". It compares the new server's estimated
accelerator use with what is free:

- Live, when the driver can say: free VRAM from nvidia-smi, or available RAM
  on unified-memory machines (Apple silicon). Running servers are already
  counted there, except ones still loading, whose estimate is subtracted.
- Otherwise estimated: total VRAM minus the estimates of the tracked servers
  that are running (paused ones included, since they keep their VRAM).

The outcome is one of:

- ``fits`` / ``tight``: start it (tight adds a warning).
- ``conflict``: it would fit on its own but not next to what is running. Start
  refuses unless asked to release the servers in ``release`` first, or forced.
  Servers answering gateway requests are never in ``release``.
- ``busy``: it would fit, but only by releasing servers that are answering
  gateway requests right now (``busy``). Start refuses unless forced; trying
  again once they finish picks idle servers or finds them idle.
- ``too_big``: it does not fit even with nothing else running. That is the
  ordinary fit problem (see Fit / Smart Fit), so start only warns.
- ``unknown``: capacity or the estimate is unknown, or it runs on the CPU.
"""

from __future__ import annotations

import os
from typing import Any

from .estimates import estimate_memory_fit
from .hardware import _nvidia_smi_gpus, _posix_memory_info, _windows_memory_info
from .paths import is_windows

MIB = 1024 * 1024
# Statuses of tracked servers whose processes hold (or are taking) GPU memory.
# Paused ("suspended") servers are frozen, not stopped, so their VRAM stays allocated.
_HOLDING = {"running", "starting", "startup_timeout", "suspended"}


def estimate_server_vram_mib(params: dict[str, Any], model: dict[str, Any] | None) -> int | None:
    """Estimated accelerator memory a server launched with ``params`` will use."""
    try:
        estimate = estimate_memory_fit(params, model, None)
    except Exception:
        return None
    used = (estimate.get("estimated") or {}).get("accelerator_used_mib")
    if used is None:
        return None
    projector = str(params.get("mmproj") or "").strip()
    if projector and used > 0:
        # The vision/audio projector (--mmproj) is loaded onto the GPU as well.
        try:
            used += os.path.getsize(projector) // MIB
        except OSError:
            pass
    return int(used)


def live_free_mib(hardware: dict[str, Any]) -> int | None:
    """Accelerator memory free right now, when the platform can report it."""
    primary = hardware.get("primary_gpu") or {}
    memory = hardware.get("memory") or {}
    if primary.get("backend") == "nvidia-smi":
        for gpu in _nvidia_smi_gpus():
            if gpu.get("index") == primary.get("index") and gpu.get("vram_free_bytes") is not None:
                return int(gpu["vram_free_bytes"] // MIB)
        return None
    if memory.get("unified") or primary.get("unified_memory"):
        current = _windows_memory_info() if is_windows() else _posix_memory_info()
        available = current.get("available_bytes")
        return int(available // MIB) if available else None
    return None


def _capacity_mib(hardware: dict[str, Any]) -> int | None:
    primary = hardware.get("primary_gpu") or {}
    memory = hardware.get("memory") or {}
    total = primary.get("vram_total_bytes")
    if not total and (memory.get("unified") or primary.get("unified_memory")):
        total = memory.get("total_bytes")
    return int(total // MIB) if total else None


def _holding(servers: list[dict[str, Any]], exclude: set[str]) -> list[dict[str, Any]]:
    return [
        server
        for server in servers
        if server.get("running") and server.get("status") in _HOLDING and str(server.get("id")) not in exclude
    ]


def plan_start(
    need_mib: int | None,
    hardware: dict[str, Any],
    servers: list[dict[str, Any]],
    *,
    live_free: int | None = None,
    target_mib: int | None = None,
    exclude: set[str] | None = None,
    busy: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Decide whether a server needing ``need_mib`` fits next to ``servers``.

    ``busy`` maps server ids to the gateway requests running on them (see
    inflight.py); those servers are never picked for release. When only
    releasing a busy server would make room, the status is ``busy``.
    """

    target = int(target_mib or hardware.get("recommended_fit_target_mib") or 1024)
    holding = _holding(servers, exclude or set())
    others = [
        {"id": s.get("id"), "mode": s.get("mode"), "status": s.get("status"),
         "estimated_vram_mib": s.get("estimated_vram_mib"), "in_flight": (busy or {}).get(str(s.get("id")), 0)}
        for s in holding
    ]
    unestimated = [o for o in others if o["estimated_vram_mib"] is None]
    capacity = _capacity_mib(hardware)
    plan: dict[str, Any] = {
        "need_mib": need_mib,
        "capacity_mib": capacity,
        "target_headroom_mib": target,
        "running": others,
        "release": [],
        "warnings": [],
    }

    def done(status: str, message: str) -> dict[str, Any]:
        plan["status"] = status
        plan["message"] = message
        return plan

    if not need_mib:
        return done("unknown", "No GPU memory estimate for this launch (CPU-only or unknown model).")

    if live_free is not None:
        # The driver already counts running servers; ones still loading may not
        # have allocated yet, so hold their estimate back as well.
        loading = sum(o["estimated_vram_mib"] or 0 for o in others if o["status"] == "starting")
        available = live_free - loading
        plan["source"] = "live"
    elif capacity is not None:
        available = capacity - sum(o["estimated_vram_mib"] or 0 for o in others)
        plan["source"] = "estimated"
        if unestimated:
            plan["warnings"].append(
                f"{len(unestimated)} running server(s) have no memory estimate and were not counted."
            )
    else:
        return done("unknown", "GPU memory capacity is unknown, so concurrent fit was not checked.")

    headroom = available - need_mib
    freeable = sum(o["estimated_vram_mib"] or 0 for o in others)
    plan["available_mib"] = available
    plan["headroom_mib"] = headroom

    if headroom >= target:
        return done("fits", f"Fits: about {need_mib} MiB needed, {available} MiB free.")
    if headroom >= 0:
        return done(
            "tight",
            f"Tight fit: about {need_mib} MiB needed, {available} MiB free, "
            f"leaving {headroom} MiB (target {target} MiB).",
        )
    if headroom + freeable < 0:
        return done(
            "too_big",
            f"About {need_mib} MiB needed but only {available + freeable} MiB would be free even with "
            "nothing else running. Lower gpu_layers or ctx_size (Fit can suggest settings).",
        )

    # Release the fewest idle servers: biggest first until it fits. A server
    # with requests running is not released out from under them.
    release: list[dict[str, Any]] = []
    for other in sorted(others, key=lambda o: o["estimated_vram_mib"] or 0, reverse=True):
        if headroom >= 0:
            break
        if not other["estimated_vram_mib"] or other["in_flight"]:
            continue
        release.append(other)
        headroom += other["estimated_vram_mib"]
    if headroom < 0:
        serving = [o for o in others if o["in_flight"] and o["estimated_vram_mib"]]
        plan["busy"] = [o["id"] for o in serving]
        names = ", ".join(f"{o['mode'] or o['id']} ({o['in_flight']} running)" for o in serving)
        return done(
            "busy",
            f"Not enough GPU memory next to what is running: about {need_mib} MiB needed, {available} MiB free. "
            f"Making room would stop servers that are answering requests: {names}. Try again when they finish.",
        )
    plan["release"] = [o["id"] for o in release]
    names = ", ".join(str(o["mode"] or o["id"]) for o in release)
    return done(
        "conflict",
        f"Not enough GPU memory next to what is running: about {need_mib} MiB needed, {available} MiB free. "
        f"Releasing {names} would make room.",
    )
