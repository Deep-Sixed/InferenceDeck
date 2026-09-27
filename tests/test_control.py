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
        req = urllib.request.Request(self.base + "/api/start", data=b"{", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=2)
        self.assertEqual(caught.exception.code, 400)
