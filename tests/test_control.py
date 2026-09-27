from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler


class ControlPlaneTests(unittest.TestCase):
    def test_status_contract_is_json_serializable(self) -> None:
        with mock.patch("inferencedeck.control.list_servers", return_value=[{"id": "one", "running": True}]):
            payload = ControlPlane().status()
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["running_count"], 1)
        json.dumps(payload)

    def test_suspend_is_delegated_by_id(self) -> None:
        with mock.patch("inferencedeck.control.suspend_server", return_value={"success": True}) as call:
            self.assertTrue(ControlPlane().suspend(server_id="abc")["success"])
        call.assert_called_once_with(server_id="abc", mode=None)


class ControlApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("TestHandler", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.status.return_value = {"version": 1, "running_count": 0, "servers": []}
        handler.control_plane.suspend.return_value = {"success": True, "message": "ok"}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.control = handler.control_plane

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_status_endpoint(self) -> None:
        with urllib.request.urlopen(self.base + "/api/status", timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["version"], 1)

    def test_suspend_endpoint(self) -> None:
        req = urllib.request.Request(
            self.base + "/api/suspend",
            data=json.dumps({"server_id": "abc"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            payload = json.load(response)
        self.assertTrue(payload["success"])
        self.control.suspend.assert_called_once_with(server_id="abc", mode=None)

    def test_invalid_json_is_400(self) -> None:
        req = urllib.request.Request(
            self.base + "/api/start", data=b"{", headers={"Content-Type": "application/json"}, method="POST"
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=2)
        self.assertEqual(caught.exception.code, 400)

    def _status_code(self, req: urllib.request.Request) -> int:
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_cross_site_simple_post_is_rejected(self) -> None:
        # A web page can send text/plain to localhost without a CORS preflight.
        req = urllib.request.Request(
            self.base + "/api/stop",
            data=json.dumps({"server_id": "abc"}).encode(),
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        self.assertEqual(self._status_code(req), 415)
        self.control.stop.assert_not_called()

    def test_foreign_host_header_is_rejected_without_auth(self) -> None:
        req = urllib.request.Request(self.base + "/api/status", headers={"Host": "rebind.example:80"})
        self.assertEqual(self._status_code(req), 403)
        self.control.status.assert_not_called()

    def test_loopback_host_headers_are_allowed(self) -> None:
        port = self.server.server_port
        for host in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}", "localhost"):
            req = urllib.request.Request(self.base + "/api/status", headers={"Host": host})
            self.assertEqual(self._status_code(req), 200, host)

    def test_non_integer_log_lines_is_400(self) -> None:
        req = urllib.request.Request(self.base + "/api/logs?server_id=abc&lines=lots")
        self.assertEqual(self._status_code(req), 400)

class PerformanceControlTests(unittest.TestCase):
    def test_hardware_delegates(self) -> None:
        with mock.patch("inferencedeck.control.detect_system_hardware", return_value={"cpu": {"model": "test"}}):
            self.assertEqual(ControlPlane().hardware()["cpu"]["model"], "test")

    def test_fit_delegates_with_scope(self) -> None:
        plane = ControlPlane(project_root="/tmp/project", model_dirs=["/tmp/models"])
        with mock.patch("inferencedeck.control.run_fit_test", return_value={"success": True}) as call:
            self.assertTrue(plane.fit("demo", target_mib=2048)["success"])
        call.assert_called_once_with("demo", project_root="/tmp/project", model_dirs=["/tmp/models"], overrides=None, target_mib=2048)

    def test_benchmark_delegates_with_scope(self) -> None:
        plane = ControlPlane(project_root="/tmp/project", model_dirs=["/tmp/models"])
        with mock.patch("inferencedeck.control.run_profile_benchmark", return_value={"success": True}) as call:
            self.assertTrue(plane.benchmark("demo", completion_tokens=64)["success"])
        call.assert_called_once_with("demo", project_root="/tmp/project", model_dirs=["/tmp/models"], overrides=None, completion_tokens=64)
