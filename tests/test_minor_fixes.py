from __future__ import annotations

import io
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import benchmark, webui
from inferencedeck.auth import AuthState, client_address
from inferencedeck.config import AppConfig
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler


def _code(req: urllib.request.Request, context: ssl.SSLContext | None = None) -> int:
    try:
        with urllib.request.urlopen(req, timeout=5, context=context) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


class ClientAddressTests(unittest.TestCase):
    PROXY = frozenset({"10.0.0.1"})

    def test_header_ignored_unless_peer_is_a_trusted_proxy(self) -> None:
        self.assertEqual(client_address("203.0.113.9", "198.51.100.7", self.PROXY), "203.0.113.9")
        self.assertEqual(client_address("10.0.0.1", "198.51.100.7", frozenset()), "10.0.0.1")

    def test_rightmost_untrusted_hop_is_the_client(self) -> None:
        # The client can prepend anything; only the hop the proxy appended counts.
        self.assertEqual(client_address("10.0.0.1", "1.2.3.4, 198.51.100.7", self.PROXY), "198.51.100.7")
        chain = frozenset({"10.0.0.1", "10.0.0.2"})
        self.assertEqual(client_address("10.0.0.1", "198.51.100.7, 10.0.0.2", chain), "198.51.100.7")

    def test_all_trusted_or_empty_falls_back_to_peer(self) -> None:
        self.assertEqual(client_address("10.0.0.1", "10.0.0.1", self.PROXY), "10.0.0.1")
        self.assertEqual(client_address("10.0.0.1", "", self.PROXY), "10.0.0.1")


class _Server:
    def __init__(self, auth: AuthState, control: ControlPlane | None = None) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = control or mock.Mock(spec=ControlPlane)
        handler.auth_state = auth
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class ProxyThrottleTests(unittest.TestCase):
    def _guess(self, srv: _Server, forwarded: str) -> int:
        req = urllib.request.Request(
            srv.base + "/api/status", headers={"X-Auth-Token": "wrong", "X-Forwarded-For": forwarded}
        )
        return _code(req)

    def test_trusted_proxy_throttles_each_client_separately(self) -> None:
        srv = _Server(AuthState(username="admin", token="secret", trusted_proxies=frozenset({"127.0.0.1"})))
        self.addCleanup(srv.close)
        for _ in range(5):
            self.assertEqual(self._guess(srv, "198.51.100.7"), 401)
        self.assertEqual(self._guess(srv, "198.51.100.7"), 429)
        self.assertEqual(self._guess(srv, "198.51.100.8"), 401)  # another user behind the proxy is not locked out

    def test_untrusted_forwarded_for_cannot_dodge_the_throttle(self) -> None:
        srv = _Server(AuthState(username="admin", token="secret", trusted_proxies=frozenset()))
        self.addCleanup(srv.close)
        for i in range(5):
            self._guess(srv, f"198.51.100.{i}")
        self.assertEqual(self._guess(srv, "198.51.100.99"), 429)


class ApiInputTests(unittest.TestCase):
    def setUp(self) -> None:
        control = mock.Mock(spec=ControlPlane)
        control.fit.return_value = {"success": True}
        control.benchmark.return_value = {"success": True}
        self.control = control
        self.srv = _Server(AuthState(username="admin", token=""), control)
        self.addCleanup(self.srv.close)

    def _post(self, path: str, body: dict) -> int:
        req = urllib.request.Request(
            self.srv.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        return _code(req)

    def test_bad_integer_fields_are_400_not_500(self) -> None:
        self.assertEqual(self._post("/api/fit", {"mode": "m", "target_mib": None}), 400)
        self.assertEqual(self._post("/api/fit", {"mode": "m", "target_mib": [1]}), 400)
        self.assertEqual(self._post("/api/benchmark", {"mode": "m", "completion_tokens": "lots"}), 400)
        self.assertEqual(self._post("/api/benchmark", {"mode": "m", "completion_tokens": True}), 400)
        self.control.fit.assert_not_called()
        self.control.benchmark.assert_not_called()

    def test_good_integer_fields_are_clamped(self) -> None:
        self.assertEqual(self._post("/api/benchmark", {"mode": "m", "completion_tokens": 999999}), 200)
        self.assertEqual(self.control.benchmark.call_args.kwargs["completion_tokens"], 2048)

    def test_auth_probe_does_not_reveal_the_login_name(self) -> None:
        srv = _Server(AuthState(username="operator", token="secret"))
        self.addCleanup(srv.close)
        with urllib.request.urlopen(srv.base + "/api/auth", timeout=5) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload, {"required": True})


class SmallFixTests(unittest.TestCase):
    def test_benchmark_keeps_temperature_zero(self) -> None:
        running = {"id": "m-1", "mode": "m", "pid": 1, "running": True, "host": "127.0.0.1", "port": 8080}
        reply = {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 4}}
        sent: list[dict] = []

        def fake_urlopen(req, timeout=None):
            sent.append(json.loads(req.data))
            return io.BytesIO(json.dumps(reply).encode())

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"LCC_CACHE_DIR": tmp}), \
                mock.patch.object(benchmark, "list_servers", return_value=[running]), \
                mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=fake_urlopen):
            benchmark.run_profile_benchmark("m", overrides={"temperature": 0})
            benchmark.run_profile_benchmark("m")
        self.assertEqual([body["temperature"] for body in sent], [0.0, 0.2])

    def test_config_that_is_not_an_object_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for text in ("[]", "42", '"x"', "null"):
                path.write_text(text, encoding="utf-8")
                self.assertEqual(AppConfig.load(path), AppConfig())

    def test_web_ui_tolerates_non_json_errors_and_no_login_name(self) -> None:
        app = (Path(webui.__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("r.json().catch(", app)
        self.assertNotIn("a.username", app)


@unittest.skipUnless(shutil.which("openssl"), "openssl not available to make a test certificate")
class TlsTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cert = os.path.join(tmp.name, "cert.pem")
        self.key = os.path.join(tmp.name, "key.pem")
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
             "-keyout", self.key, "-out", self.cert],
            check=True, capture_output=True,
        )

    def _login_cookie(self, server: ThreadingHTTPServer, scheme: str) -> str:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            f"{scheme}://127.0.0.1:{server.server_address[1]}/api/login",
            data=json.dumps({"username": "admin", "password": "secret"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5, context=context if scheme == "https" else None) as response:
            return response.headers["Set-Cookie"]

    def test_https_login_sets_a_secure_cookie(self) -> None:
        server = webui.make_server(
            "127.0.0.1", 0, mock.Mock(spec=ControlPlane), AuthState(username="admin", token="secret"),
            certfile=self.cert, keyfile=self.key,
        )
        cookie = self._login_cookie(server, "https")
        self.assertIn("Secure", cookie)
        self.assertIn("HttpOnly", cookie)

    def test_plain_http_cookie_is_not_marked_secure(self) -> None:
        server = webui.make_server("127.0.0.1", 0, mock.Mock(spec=ControlPlane), AuthState(username="admin", token="secret"))
        self.assertNotIn("Secure", self._login_cookie(server, "http"))


if __name__ == "__main__":
    unittest.main()
