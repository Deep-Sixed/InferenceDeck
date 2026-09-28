"""Fleet view: this machine's telemetry combined with other InferenceDeck machines'.

Each peer is another ``inferencedeck-web`` whose ``/api/telemetry`` this
process reads, with the peer's token taken from an environment variable. Peers
are asked in parallel with a short timeout, and the combined view is cached
for a few seconds, so a slow or offline machine shows as unreachable instead
of stalling the page.

``placement`` turns the view into advice on where to run a model: a machine
that already has it loaded first, then measured speed, current load and free
GPU memory. It only advises; nothing is started or routed from here.

Only ``/api/telemetry`` is read, never a peer's own ``/api/fleet``, so two
machines that list each other don't loop.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from . import telemetry

PEER_TIMEOUT_SECONDS = 3.0
FLEET_MAX_AGE_SECONDS = 5.0
MAX_PEER_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Peer:
    name: str
    url: str
    token_env: str = ""
    ca_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        # Which variable holds the token, and whether it is set; never the token.
        return {
            "name": self.name,
            "url": self.url,
            "token_env": self.token_env or None,
            "token_present": bool(self.token_env and os.environ.get(self.token_env, "").strip()),
        }


def parse_peers(entries: list[Any]) -> tuple[list[Peer], list[str]]:
    """Valid peers from config, and a message for each entry that was skipped."""
    peers: list[Peer] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            errors.append(f"fleet_peers[{index}] must be an object")
            continue
        url = str(entry.get("url") or "").strip().rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        name = str(entry.get("name") or parsed.hostname or "").strip()
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            errors.append(f"fleet_peers[{index}]: url must be http:// or https://, got {url!r}")
            continue
        if name.lower() in seen:
            errors.append(f"fleet_peers[{index}]: duplicate name {name!r}")
            continue
        seen.add(name.lower())
        peers.append(Peer(name, url, str(entry.get("tokenEnv") or entry.get("token_env") or "").strip(), str(entry.get("caFile") or entry.get("ca_file") or "").strip()))
    return peers, errors


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect would carry the peer's token to wherever it points.
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


def fetch_peer(peer: Peer, timeout: float = PEER_TIMEOUT_SECONDS) -> dict[str, Any]:
    """The peer's telemetry snapshot, or why it couldn't be read."""
    headers = {"Accept": "application/json", "User-Agent": "inferencedeck/fleet"}
    token = os.environ.get(peer.token_env, "").strip() if peer.token_env else ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    handlers: list[Any] = [_NoRedirect()]
    if peer.url.startswith("https://"):
        context = ssl.create_default_context(cafile=os.path.expanduser(peer.ca_file) if peer.ca_file else None)
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    started = time.monotonic()
    result: dict[str, Any] = {"reachable": False, "latency_ms": None, "error": None, "snapshot": None}
    try:
        with opener.open(urllib.request.Request(f"{peer.url}/api/telemetry", headers=headers), timeout=timeout) as response:
            body = response.read(MAX_PEER_RESPONSE_BYTES + 1)
        if len(body) > MAX_PEER_RESPONSE_BYTES:
            raise ValueError("response too large")
        snapshot = json.loads(body)
        if not isinstance(snapshot, dict):
            raise ValueError("not a telemetry snapshot")
        result.update(reachable=True, snapshot=snapshot)
    except urllib.error.HTTPError as exc:
        hint = " (check its token)" if exc.code == 401 else ""
        result["error"] = f"HTTP {exc.code}{hint}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        result["error"] = str(reason) or type(exc).__name__
    result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    return result


def _free_vram(gpus: list[dict[str, Any]]) -> int | None:
    free = [g["vram_total_bytes"] - g["vram_used_bytes"] for g in gpus if g.get("vram_total_bytes") is not None and g.get("vram_used_bytes") is not None]
    return sum(free) if free else None


def summarize_host(name: str, snapshot: dict[str, Any] | None, *, local: bool, reachable: bool = True, error: str | None = None, latency_ms: float | None = None) -> dict[str, Any]:
    """One machine's row in the fleet view, from its telemetry snapshot."""
    snap = snapshot or {}
    gpus = [
        {k: g.get(k) for k in ("index", "name", "utilization_percent", "vram_used_bytes", "vram_total_bytes", "temperature_c", "power_watts")}
        for g in snap.get("gpus") or []
    ]
    servers = [
        {
            k: s.get(k)
            for k in (
                "profile", "runtime", "model", "status", "running", "suspended", "context_size", "startup_seconds",
                "gpu_memory_bytes", "tokens_per_second", "prompt_tokens_per_second", "requests_active", "requests_deferred",
            )
        }
        for s in snap.get("servers") or []
    ]
    system = snap.get("system") or {}
    return {
        "name": name,
        "local": local,
        "reachable": reachable,
        "error": error,
        "latency_ms": latency_ms,
        "taken_at": snap.get("timestamp"),
        "cpu_percent": system.get("cpu_percent"),
        "memory": system.get("memory") or {},
        "gpus": gpus,
        "free_vram_bytes": _free_vram(gpus),
        "servers": servers,
        "benchmarks": [b for b in snap.get("benchmarks") or [] if isinstance(b, dict)],
    }


def local_name(configured: str = "") -> str:
    if configured.strip():
        return configured.strip()
    try:
        return socket.gethostname() or "this machine"
    except OSError:
        return "this machine"


class Fleet:
    def __init__(
        self,
        peers: list[Peer],
        name: str,
        collect_local: Callable[[], dict[str, Any]] | None = None,
        fetch: Callable[[Peer], dict[str, Any]] = fetch_peer,
        max_age: float = FLEET_MAX_AGE_SECONDS,
        config_errors: list[str] | None = None,
    ) -> None:
        self.peers = peers
        self.name = name
        self._collect_local = collect_local or telemetry.snapshot
        self._fetch = fetch
        self._max_age = max_age
        self._config_errors = list(config_errors or [])
        self._lock = threading.Lock()
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._refreshing = threading.Event()

    def _build(self) -> dict[str, Any]:
        hosts = [summarize_host(self.name, self._collect_local(), local=True)]
        if self.peers:
            with ThreadPoolExecutor(max_workers=min(8, len(self.peers))) as pool:
                results = list(pool.map(self._safe_fetch, self.peers))
            for peer, result in zip(self.peers, results):
                hosts.append(
                    summarize_host(
                        peer.name,
                        result.get("snapshot"),
                        local=False,
                        reachable=bool(result.get("reachable")),
                        error=result.get("error"),
                        latency_ms=result.get("latency_ms"),
                    )
                    | {"url": peer.url}
                )
        return {
            "version": 1,
            "timestamp": telemetry._now_iso(),
            "hosts": hosts,
            "peers": [p.to_dict() for p in self.peers],
            "config_errors": self._config_errors,
        }

    def _safe_fetch(self, peer: Peer) -> dict[str, Any]:
        try:
            return self._fetch(peer)
        except Exception as exc:
            return {"reachable": False, "error": str(exc) or type(exc).__name__}

    def overview(self) -> dict[str, Any]:
        with self._lock:
            if self._cache and time.monotonic() - self._cache[0] < self._max_age:
                return self._cache[1]
            result = self._build()
            self._cache = (time.monotonic(), result)
            return result

    def overview_nowait(self, max_stale: float = 60.0) -> dict[str, Any] | None:
        """The cached view without waiting on the network; None until a first one exists.

        A view older than the cache age is refreshed in the background, so a
        caller on a latency-sensitive path (the gateway routing a request)
        never waits for peers to answer.
        """
        cached = self._cache
        age = time.monotonic() - cached[0] if cached else None
        if (age is None or age >= self._max_age) and not self._refreshing.is_set():
            self._refreshing.set()
            threading.Thread(target=self._refresh, name="inferencedeck-fleet", daemon=True).start()
        return cached[1] if cached and age is not None and age < max_stale else None

    def _refresh(self) -> None:
        try:
            self.overview()
        except Exception:
            pass
        finally:
            self._refreshing.clear()


# --- placement --------------------------------------------------------------

TIER_LABELS = {0: "loaded", 1: "loaded, paused", 2: "known", 3: "not seen"}


def _matches(entry: dict[str, Any], profile: str, model: str) -> bool:
    served = str(entry.get("model") or "").lower()
    # A model may be named by its file (qwen3-30b.gguf) or its stem (qwen3-30b), as the gateway does.
    return bool(
        (profile and str(entry.get("profile") or "").lower() == profile)
        or (model and served and model in (served, served.removesuffix(".gguf")))
    )


def placement(overview: dict[str, Any], profile: str = "", model: str = "") -> dict[str, Any]:
    """Machines ranked for running ``profile`` (or the model file ``model``), best first."""
    profile, model = profile.strip().lower(), model.strip().lower()
    if not profile and not model:
        raise ValueError("profile or model is required")
    candidates, unreachable = [], []
    for host in overview.get("hosts") or []:
        if not host.get("reachable"):
            unreachable.append({"host": host["name"], "error": host.get("error")})
            continue
        servers = [s for s in host.get("servers") or [] if _matches(s, profile, model)]
        live = [s for s in servers if s.get("running")]
        benchmark = next((b for b in host.get("benchmarks") or [] if profile and str(b.get("profile") or "").lower() == profile), None)
        if any(not s.get("suspended") for s in live):
            tier = 0
        elif live:
            tier = 1
        elif servers or benchmark:
            tier = 2
        else:
            tier = 3
        # Measured speed: live speed if the server is generating now, else its last benchmark.
        live_speed = next((s["tokens_per_second"] for s in live if s.get("tokens_per_second")), None)
        speed = live_speed or (benchmark or {}).get("tokens_per_second")
        load = sum(int(s.get("requests_active") or 0) for s in host.get("servers") or [] if s.get("running"))
        reasons = {
            0: "model is loaded and serving",
            1: "model is loaded but paused; resume it first",
            2: "has run this model before; needs a start",
            3: "no record of this model here",
        }[tier]
        details = [reasons]
        if speed:
            details.append(f"{speed:g} tok/s {'now' if live_speed else 'in its last benchmark'}")
        if load:
            details.append(f"{load} request{'s' if load != 1 else ''} in progress")
        if host.get("free_vram_bytes") is not None:
            details.append(f"{host['free_vram_bytes'] / 1024**3:.1f} GB VRAM free")
        candidates.append(
            {
                "host": host["name"],
                "local": host.get("local", False),
                "tier": tier,
                "state": TIER_LABELS[tier],
                "tokens_per_second": speed,
                "requests_active": load,
                "free_vram_bytes": host.get("free_vram_bytes"),
                "reasons": details,
            }
        )
    candidates.sort(key=lambda c: (c["tier"], -(c["tokens_per_second"] or 0), c["requests_active"], -(c["free_vram_bytes"] or 0)))
    return {
        "profile": profile or None,
        "model": model or None,
        "recommended": candidates[0]["host"] if candidates and candidates[0]["tier"] < 3 else None,
        "candidates": candidates,
        "unreachable": unreachable,
    }


_fleet: Fleet | None = None
_fleet_key: tuple | None = None
_fleet_lock = threading.Lock()


def current(peers_config: list[Any], configured_name: str = "") -> Fleet:
    """The process-wide fleet, rebuilt when the peer config changes."""
    global _fleet, _fleet_key
    key = (json.dumps(peers_config, sort_keys=True, default=str), configured_name)
    with _fleet_lock:
        if _fleet is None or key != _fleet_key:
            peers, errors = parse_peers(peers_config)
            _fleet = Fleet(peers, local_name(configured_name), config_errors=errors)
            _fleet_key = key
        return _fleet
