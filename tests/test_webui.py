from __future__ import annotations

import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from inferencedeck.control import ControlPlane
from inferencedeck.webui import WebRequestHandler


class WebUiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("TestWebHandler", (WebRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.status.return_value = {"version": 1, "running_count": 0, "servers": []}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def test_index_is_served(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=2) as response:
            body = response.read().decode()
        self.assertIn("InferenceDeck", body)
        self.assertIn("Tracked servers", body)

    def test_static_assets_are_served(self) -> None:
        with urllib.request.urlopen(self.base + "/app.js", timeout=2) as response:
            self.assertIn("control API online", response.read().decode())

    def test_api_is_available_from_same_server(self) -> None:
        with urllib.request.urlopen(self.base + "/api/status", timeout=2) as response:
            self.assertEqual(response.status, 200)
