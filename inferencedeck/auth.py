from __future__ import annotations

import ipaddress
import os
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

SESSION_TTL_SECONDS = 7 * 24 * 3600  # matches the cookie Max-Age
MAX_SESSIONS = 128
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 300
MAX_TRACKED_CLIENTS = 1024


def _token_from_environment() -> str:
    direct = os.environ.get("INFERENCEDECK_TOKEN", "").strip()
    if direct:
        return direct
    token_file = os.environ.get("INFERENCEDECK_TOKEN_FILE", "").strip()
    if not token_file:
        return ""
    try:
        return Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read INFERENCEDECK_TOKEN_FILE: {exc}") from exc


@dataclass
class AuthState:
    username: str = field(default_factory=lambda: os.environ.get("INFERENCEDECK_USER", "admin").strip() or "admin")
    token: str = field(default_factory=_token_from_environment)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    # sid -> expiry; insertion order is issue order, so the oldest is evicted first.
    _sessions: OrderedDict[str, float] = field(default_factory=OrderedDict, init=False, repr=False)
    # client -> recent failure times, for login/token throttling.
    _failures: OrderedDict[str, deque[float]] = field(default_factory=OrderedDict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def credentials_ok(self, username: str, password: str) -> bool:
        return (
            self.enabled
            and bool(username)
            and bool(password)
            and secrets.compare_digest(username, self.username)
            and secrets.compare_digest(password, self.token)
        )

    def issue_session(self) -> str:
        sid = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[sid] = self.clock() + SESSION_TTL_SECONDS
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
        return sid

    def session_ok(self, sid: str) -> bool:
        if not self.enabled:
            return True
        if not sid:
            return False
        with self._lock:
            expiry = self._sessions.get(sid)
            if expiry is None:
                return False
            if expiry <= self.clock():
                del self._sessions[sid]
                return False
            return True

    def revoke_session(self, sid: str) -> None:
        if not sid:
            return
        with self._lock:
            self._sessions.pop(sid, None)

    def supplied_token_ok(self, headers) -> bool:
        # Header only: a ?token= query parameter would leak into logs and history.
        supplied = headers.get("X-Auth-Token", "")
        return bool(supplied) and self.enabled and secrets.compare_digest(supplied, self.token)

    def retry_after(self, client: str) -> int:
        """Seconds until ``client`` may try credentials again; 0 when not throttled."""
        with self._lock:
            recent = self._recent_failures(client)
            if len(recent) < MAX_FAILURES:
                return 0
            return max(1, int(recent[0] + FAILURE_WINDOW_SECONDS - self.clock()) + 1)

    def record_failure(self, client: str) -> None:
        with self._lock:
            recent = self._recent_failures(client)
            recent.append(self.clock())
            self._failures[client] = recent
            self._failures.move_to_end(client)
            while len(self._failures) > MAX_TRACKED_CLIENTS:
                self._failures.popitem(last=False)

    def record_success(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)

    def _recent_failures(self, client: str) -> deque[float]:
        cutoff = self.clock() - FAILURE_WINDOW_SECONDS
        recent = self._failures.get(client, deque())
        while recent and recent[0] <= cutoff:
            recent.popleft()
        return recent


# Tailscale addresses: traffic to them is already WireGuard-encrypted.
TAILNET_NETWORKS = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))


def is_tailnet_address(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host.strip().strip("[]"))
    except ValueError:
        return False
    return any(address in network for network in TAILNET_NETWORKS)


def validate_bind_security(
    host: str, auth: AuthState, *, tls: bool = False, allow_insecure_http: bool = False
) -> None:
    """Refuse binds that would expose the control API unsafely.

    Off loopback the API needs authentication, and the password, token and
    session cookie must not cross the network in the clear: that needs TLS, a
    Tailscale address (encrypted by WireGuard), or an explicit opt-out.
    0.0.0.0 is every interface, including the ordinary LAN, not "the tailnet".
    """

    host = host.strip().lower()
    if host in LOOPBACK_HOSTS:
        return
    if not auth.enabled:
        raise RuntimeError(
            "Refusing non-loopback bind without authentication. Set INFERENCEDECK_TOKEN "
            "or INFERENCEDECK_TOKEN_FILE before binding to a LAN/tailnet address."
        )
    if tls or allow_insecure_http or is_tailnet_address(host):
        return
    raise RuntimeError(
        f"Refusing plain-HTTP bind on {host}: the login password, token and session cookie would "
        "cross the network unencrypted. Use --tls-cert/--tls-key, bind to this machine's Tailscale "
        "address (100.x.y.z) instead of 0.0.0.0, or pass --allow-insecure-http to accept the risk."
    )
