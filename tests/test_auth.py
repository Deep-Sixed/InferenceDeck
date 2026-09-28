from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from inferencedeck.auth import AuthState, validate_bind_security
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler


class AuthStateTests(unittest.TestCase):
    def test_nonloopback_requires_token(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_bind_security("0.0.0.0", AuthState(token=""))
        validate_bind_security("127.0.0.1", AuthState(token=""))
        # With a token, a LAN bind still needs an encrypted transport or an explicit opt-out.
        with self.assertRaisesRegex(RuntimeError, "unencrypted"):
            validate_bind_security("0.0.0.0", AuthState(token="secret"))
        validate_bind_security("0.0.0.0", AuthState(token="secret"), tls=True)
        validate_bind_security("0.0.0.0", AuthState(token="secret"), allow_insecure_http=True)
        validate_bind_security("100.101.102.103", AuthState(token="secret"))  # Tailscale (WireGuard)
        with self.assertRaises(RuntimeError):
            validate_bind_security("100.101.102.103", AuthState(token=""))  # still needs auth

    def test_credentials_and_sessions(self) -> None:
        auth = AuthState(username="admin", token="secret")
        self.assertTrue(auth.credentials_ok("admin", "secret"))
        self.assertFalse(auth.credentials_ok("admin", "wrong"))
        sid = auth.issue_session()
        self.assertTrue(auth.session_ok(sid))
        auth.revoke_session(sid)
        self.assertFalse(auth.session_ok(sid))


class AuthHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("AuthHandler", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.status.return_value = {"version": 1, "running_count": 0, "servers": []}
        handler.auth_state = AuthState(username="admin", token="secret")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def test_status_requires_auth(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base + "/api/status", timeout=2)
        self.assertEqual(caught.exception.code, 401)

    def test_header_token_auth(self) -> None:
        req = urllib.request.Request(self.base + "/api/status", headers={"X-Auth-Token": "secret"})
        with urllib.request.urlopen(req, timeout=2) as response:
            self.assertEqual(response.status, 200)

    def test_login_sets_httponly_cookie(self) -> None:
        req = urllib.request.Request(
            self.base + "/api/login",
            data=json.dumps({"username": "admin", "password": "secret"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=2) as response:
            cookie = response.headers.get("Set-Cookie", "")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)

    def _code(self, req: urllib.request.Request) -> int:
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def _login(self, password: str) -> urllib.request.Request:
        return urllib.request.Request(
            self.base + "/api/login",
            data=json.dumps({"username": "admin", "password": password}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def test_query_string_token_is_not_accepted(self) -> None:
        self.assertEqual(self._code(urllib.request.Request(self.base + "/api/status?token=secret")), 401)

    def test_repeated_bad_logins_are_throttled(self) -> None:
        for _ in range(5):
            self.assertEqual(self._code(self._login("wrong")), 401)
        # Locked out now, even with the right password.
        self.assertEqual(self._code(self._login("secret")), 429)

    def test_proxy_basic_auth_header_is_not_a_failed_guess(self) -> None:
        # nginx/Caddy basic auth forwards this on every request; it isn't our token.
        for _ in range(10):
            req = urllib.request.Request(self.base + "/api/status", headers={"Authorization": "Basic dXNlcjpwYXNz"})
            self.assertEqual(self._code(req), 401)
        self.assertEqual(self._code(self._login("secret")), 200)

    def test_wrong_header_tokens_count_as_failures(self) -> None:
        for _ in range(5):
            req = urllib.request.Request(self.base + "/api/status", headers={"X-Auth-Token": "guess"})
            self.assertEqual(self._code(req), 401)
        req = urllib.request.Request(self.base + "/api/status", headers={"X-Auth-Token": "secret"})
        self.assertEqual(self._code(req), 429)


class SessionLimitTests(unittest.TestCase):
    def test_sessions_expire(self) -> None:
        now = [1000.0]
        auth = AuthState(username="admin", token="secret", clock=lambda: now[0])
        sid = auth.issue_session()
        now[0] += 7 * 24 * 3600 - 1
        self.assertTrue(auth.session_ok(sid))
        now[0] += 2
        self.assertFalse(auth.session_ok(sid))

    def test_session_count_is_capped_oldest_first(self) -> None:
        from inferencedeck.auth import MAX_SESSIONS

        auth = AuthState(username="admin", token="secret")
        first = auth.issue_session()
        rest = [auth.issue_session() for _ in range(MAX_SESSIONS)]
        self.assertFalse(auth.session_ok(first))
        self.assertTrue(all(auth.session_ok(sid) for sid in rest))

    def test_throttle_lifts_after_window(self) -> None:
        now = [0.0]
        auth = AuthState(username="admin", token="secret", clock=lambda: now[0])
        for _ in range(5):
            auth.record_failure("10.0.0.5")
        self.assertGreater(auth.retry_after("10.0.0.5"), 0)
        self.assertEqual(auth.retry_after("10.0.0.6"), 0)  # other clients unaffected
        now[0] += 301
        self.assertEqual(auth.retry_after("10.0.0.5"), 0)

    def test_many_clients_hit_the_global_cap(self) -> None:
        from inferencedeck.auth import MAX_FAILURES, MAX_GLOBAL_FAILURES, MAX_TRACKED_CLIENTS

        now = [0.0]
        auth = AuthState(username="admin", token="secret", clock=lambda: now[0])
        # Spread across clients so none reaches its own limit.
        for n in range(MAX_GLOBAL_FAILURES):
            client = f"10.0.{n // (MAX_FAILURES - 1)}.1"
            self.assertEqual(auth.retry_after(client), 0)
            auth.record_failure(client)
        self.assertGreater(auth.retry_after("198.51.100.99"), 0)  # a fresh address is throttled too
        self.assertLess(MAX_GLOBAL_FAILURES, MAX_TRACKED_CLIENTS)  # so recent failures are never evicted
        now[0] += 301
        self.assertEqual(auth.retry_after("198.51.100.99"), 0)

    def test_success_does_not_clear_the_global_count(self) -> None:
        from inferencedeck.auth import MAX_GLOBAL_FAILURES

        auth = AuthState(username="admin", token="secret")
        for _ in range(MAX_GLOBAL_FAILURES):
            auth.record_failure("shared")
            auth.record_success("shared")
        self.assertGreater(auth.retry_after("shared"), 0)

    def test_success_clears_failures(self) -> None:
        auth = AuthState(username="admin", token="secret")
        for _ in range(4):
            auth.record_failure("c")
        auth.record_success("c")
        auth.record_failure("c")
        self.assertEqual(auth.retry_after("c"), 0)
