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
        validate_bind_security("0.0.0.0", AuthState(token="secret"))
        validate_bind_security("127.0.0.1", AuthState(token=""))

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
