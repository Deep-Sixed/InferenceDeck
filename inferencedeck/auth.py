from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


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
    _sessions: set[str] = field(default_factory=set, init=False, repr=False)
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
            self._sessions.add(sid)
        return sid

    def session_ok(self, sid: str) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            return bool(sid) and sid in self._sessions

    def revoke_session(self, sid: str) -> None:
        if not sid:
            return
        with self._lock:
            self._sessions.discard(sid)

    def supplied_token_ok(self, path: str, headers) -> bool:
        supplied = headers.get("X-Auth-Token", "")
        if not supplied:
            supplied = (parse_qs(urlparse(path).query).get("token") or [""])[0]
        return bool(supplied) and self.enabled and secrets.compare_digest(supplied, self.token)


def validate_bind_security(host: str, auth: AuthState) -> None:
    if host.strip().lower() not in LOOPBACK_HOSTS and not auth.enabled:
        raise RuntimeError(
            "Refusing non-loopback bind without authentication. Set INFERENCEDECK_TOKEN "
            "or INFERENCEDECK_TOKEN_FILE before binding to a LAN/tailnet address."
        )
