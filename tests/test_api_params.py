from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from inferencedeck.api_params import validate_overrides
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler


class ValidateOverridesTests(unittest.TestCase):
    def test_accepts_tuning_keys(self) -> None:
        clean = validate_overrides({"ctx_size": 32768, "gpu_layers": 999, "flash_attn": True, "cache_type_k": "q8_0", "top_p": 0.9})
        self.assertEqual(clean, {"ctx_size": 32768, "gpu_layers": 999, "flash_attn": True, "cache_type_k": "q8_0", "top_p": 0.9})
        self.assertIsNone(validate_overrides(None))

    def test_rejects_keys_that_change_what_runs_or_where(self) -> None:
        for key, value in [
            ("host", "0.0.0.0"),
            ("port", 80),
            ("draft_model", "/etc/passwd"),
            ("runtime", "vllm"),
            ("device", "CUDA1"),
            ("tensor_overrides", ["x"]),
            ("alias", "x"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_overrides({key: value})

    def test_rejects_bad_values(self) -> None:
        for bad in [
            {"ctx_size": 10},
            {"ctx_size": "32768"},
            {"ctx_size": True},
            {"ctx_size": 1.5},
            {"gpu_layers": -1},
            {"flash_attn": "yes"},
            {"cache_type_v": "q9_9"},
            {"temperature": 50},
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_overrides(bad)
        with self.assertRaises(ValueError):
            validate_overrides(["ctx_size"])


class OverridesOverHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.start.return_value = {"success": True}
        self.control = handler.control_plane
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _start(self, overrides: dict) -> int:
        req = urllib.request.Request(
            self.base + "/api/start",
            data=json.dumps({"mode": "m", "overrides": overrides}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_host_override_is_rejected(self) -> None:
        self.assertEqual(self._start({"host": "0.0.0.0"}), 400)
        self.control.start.assert_not_called()

    def test_allowed_override_passes_through(self) -> None:
        self.assertEqual(self._start({"ctx_size": 16384}), 200)
        self.control.start.assert_called_once_with("m", {"ctx_size": 16384}, stop_existing=False)
