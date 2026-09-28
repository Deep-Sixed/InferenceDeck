from __future__ import annotations

import ipaddress
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

SESSION_TTL_SECONDS = 7 * 24 * 3600  # matches the cookie Max-Age
MAX_SESSIONS = 128
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 300
MAX_TRACKED_CLIENTS = 1024
# Failed attempts allowed across all clients per window. Per-client buckets
# alone let anyone with many addresses (an IPv6 prefix, a botnet, rotating
# X-Forwarded-For values through a proxy) guess without limit; this bounds the
# total. It is well under MAX_TRACKED_CLIENTS, so no client that failed within
# the window is ever evicted from tracking.
MAX_GLOBAL_FAILURES = 50
# IPv6 clients are throttled by /64: one host usually holds the whole prefix.
IPV6_THROTTLE_PREFIX = 64

_Network = ipaddress.IPv4Network | ipaddress.IPv6Network
_LOOPBACK_NETWORKS = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))
_warned_proxy_entries: set[str] = set()


def parse_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """An IP from a peer address or X-Forwarded-For hop; None when it isn't one.

    Accepts the forms proxies write: ``1.2.3.4``, ``1.2.3.4:5678``, ``::1``,
    ``[::1]`` and ``[::1]:5678`` (optionally quoted). IPv4-mapped IPv6 comes back
    as IPv4, so ``::ffff:1.2.3.4`` and ``1.2.3.4`` are the same client.
    """
    text = value.strip().strip('"')
    if text.startswith("["):
        text = text[1:].partition("]")[0]
    elif text.count(":") == 1:
        text = text.partition(":")[0]  # IPv4 with a port
    try:
        address = ipaddress.ip_address(text.partition("%")[0])  # drop an IPv6 zone
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


class TrustedProxies:
    """The reverse proxies whose X-Forwarded-For we believe.

    Entries are IP addresses or CIDR ranges (``10.0.0.0/8``); ``localhost``
    means the loopback ranges. Anything else can never match a peer address,
    so it is reported on stderr rather than silently ignored.
    """

    def __init__(self, entries: Iterable[str] = (), warn: bool = False) -> None:
        networks: list[_Network] = []
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            if entry.lower() == "localhost":
                networks.extend(_LOOPBACK_NETWORKS)
                continue
            try:
                if "/" in entry:
                    network = ipaddress.ip_network(entry, strict=False)
                else:
                    address = parse_address(entry)
                    if address is None:
                        raise ValueError(entry)
                    network = ipaddress.ip_network(address)
            except ValueError:
                if warn and entry not in _warned_proxy_entries:
                    _warned_proxy_entries.add(entry)
                    sys.stderr.write(
                        f"inferencedeck: ignoring INFERENCEDECK_TRUSTED_PROXIES entry {entry!r}: "
                        "use IP addresses or CIDR ranges\n"
                    )
                continue
            if isinstance(network, ipaddress.IPv6Network) and network.prefixlen >= 96:
                mapped = network.network_address.ipv4_mapped
                if mapped is not None:
                    network = ipaddress.ip_network(f"{mapped}/{network.prefixlen - 96}")
            networks.append(network)
        self.networks = tuple(networks)

    def __contains__(self, address: object) -> bool:
        if not isinstance(address, str):
            return False
        parsed = parse_address(address)
        return parsed is not None and any(parsed in network for network in self.networks)

    def __bool__(self) -> bool:
        return bool(self.networks)

    def __repr__(self) -> str:
        return f"TrustedProxies({[str(n) for n in self.networks]!r})"


def throttle_key(address: str) -> str:
    """The bucket failed attempts from ``address`` count against."""
    parsed = parse_address(address)
    if parsed is None:
        return address
    if isinstance(parsed, ipaddress.IPv6Address):
        return str(ipaddress.ip_network(f"{parsed}/{IPV6_THROTTLE_PREFIX}", strict=False))
    return str(parsed)


def trusted_proxies_from_environment() -> TrustedProxies:
    """INFERENCEDECK_TRUSTED_PROXIES: comma-separated IPs or CIDR ranges of reverse proxies in front of us."""
    return TrustedProxies(os.environ.get("INFERENCEDECK_TRUSTED_PROXIES", "").split(","), warn=True)


def client_address(peer: str, forwarded_for: str, trusted_proxies: Iterable[str]) -> str:
    """The throttle key for failed logins (see ``throttle_key``).

    Behind a reverse proxy every request comes from the proxy, so one person's
    bad attempts would lock out everyone. X-Forwarded-For is honoured only when
    the connection comes from a configured trusted proxy (anyone else could set
    it to dodge the throttle), and is read right to left, skipping trusted hops,
    so a client can't spoof it by adding entries of its own. A hop that isn't an
    IP address (``unknown``, junk) ends the walk at the peer: it can't be told
    apart from other clients, so it shares the proxy's bucket.
    """
    trusted = trusted_proxies if isinstance(trusted_proxies, TrustedProxies) else TrustedProxies(trusted_proxies)
    if peer not in trusted or not forwarded_for:
        return throttle_key(peer)
    for hop in reversed([part.strip() for part in forwarded_for.split(",")]):
        if not hop:
            continue
        if parse_address(hop) is None:
            break
        if hop not in trusted:
            return throttle_key(hop)
    return throttle_key(peer)


def forwarded_for(headers) -> str:
    """Every X-Forwarded-For line, joined: some proxies (HAProxy) add a line
    instead of appending to the client's, and the last hop is on the last line."""
    get_all = getattr(headers, "get_all", None)
    lines = get_all("X-Forwarded-For") if get_all is not None else [headers.get("X-Forwarded-For", "")]
    return ",".join(line for line in lines or [] if line)


def request_client(handler) -> str:
    """The throttle key for a BaseHTTPRequestHandler with an ``auth_state``."""
    return client_address(str(handler.client_address[0]), forwarded_for(handler.headers), handler.auth_state.trusted_proxies)


def host_header_ok(headers, auth: "AuthState") -> bool:
    """DNS-rebinding guard: without auth the server only binds to loopback, so
    reject any Host header that doesn't name loopback."""
    if auth.enabled:
        return True
    host = headers.get("Host", "")
    if host.startswith("["):
        name = host[1:].partition("]")[0]
    else:
        name = host.rpartition(":")[0] if host.count(":") == 1 else host
    return name.lower() in LOOPBACK_HOSTS


def token_from_environment() -> str:
    """INFERENCEDECK_TOKEN, else the contents of INFERENCEDECK_TOKEN_FILE; "" when neither is set."""
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
    token: str = field(default_factory=token_from_environment)
    trusted_proxies: TrustedProxies = field(default_factory=trusted_proxies_from_environment)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    # sid -> expiry; insertion order is issue order, so the oldest is evicted first.
    _sessions: OrderedDict[str, float] = field(default_factory=OrderedDict, init=False, repr=False)
    # client -> recent failure times, for login/token throttling.
    _failures: OrderedDict[str, deque[float]] = field(default_factory=OrderedDict, init=False, repr=False)
    # Every recent failure, whoever made it (MAX_GLOBAL_FAILURES).
    _all_failures: deque[float] = field(
        default_factory=lambda: deque(maxlen=MAX_GLOBAL_FAILURES), init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.trusted_proxies, TrustedProxies):
            self.trusted_proxies = TrustedProxies(self.trusted_proxies)

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
            waits = [0]
            recent = self._recent_failures(client)
            if len(recent) >= MAX_FAILURES:
                waits.append(self._wait_for(recent[-MAX_FAILURES]))
            self._expire(self._all_failures)
            if len(self._all_failures) >= MAX_GLOBAL_FAILURES:
                waits.append(self._wait_for(self._all_failures[-MAX_GLOBAL_FAILURES]))
            return max(waits)

    def record_failure(self, client: str) -> None:
        with self._lock:
            now = self.clock()
            self._expire(self._all_failures)
            self._all_failures.append(now)
            recent = self._recent_failures(client)
            recent.append(now)
            self._failures[client] = recent
            self._failures.move_to_end(client)
            while len(self._failures) > MAX_TRACKED_CLIENTS:
                self._failures.popitem(last=False)

    def record_success(self, client: str) -> None:
        # Clears only this client's bucket; the global count still bounds
        # guesses interleaved with someone else's good requests from one address.
        with self._lock:
            self._failures.pop(client, None)

    def _wait_for(self, oldest: float) -> int:
        return max(1, int(oldest + FAILURE_WINDOW_SECONDS - self.clock()) + 1)

    def _expire(self, times: deque[float]) -> deque[float]:
        cutoff = self.clock() - FAILURE_WINDOW_SECONDS
        while times and times[0] <= cutoff:
            times.popleft()
        return times

    def _recent_failures(self, client: str) -> deque[float]:
        return self._expire(self._failures.get(client, deque()))


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
        "cross the network unencrypted. Use --certfile/--keyfile, bind to this machine's Tailscale "
        "address (100.x.y.z) instead of 0.0.0.0, or pass --allow-insecure-http to accept the risk."
    )
