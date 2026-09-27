from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck.control import ControlPlane
from inferencedeck.remotes import active_endpoint, disable_all, enable_endpoint, list_endpoints


class RemoteEndpointTests(unittest.TestCase):
    def _write(self, root: Path, name: str, *, enabled: bool = False) -> Path:
        path = root / f"{name}.json"
        path.write_text(json.dumps({
            "provider": "llamacpp" if name == "remote" else "openai",
            "enabled": enabled,
            "lane": "remote_host" if name == "remote" else "true_cloud",
            "model": None if name == "remote" else "model",
            "baseUrl": "http://example.invalid/v1",
            "apiKeyEnv": f"{name.upper()}_KEY",
            "displayName": name,
        }))
        return path

    def test_enable_is_exclusive_and_requires_environment_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self._write(root, "remote"); self._write(root, "cloud", enabled=True)
            with mock.patch.dict(os.environ, {"REMOTE_KEY": "secret"}, clear=False):
                enabled = enable_endpoint("remote", root)
            self.assertTrue(enabled.enabled)
            states = {c.name: c.enabled for c in list_endpoints(root)}
            self.assertEqual(states, {"cloud": False, "remote": True})

    def test_actual_key_is_never_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self._write(root, "remote")
            with mock.patch.dict(os.environ, {"REMOTE_KEY": "TOP-SECRET"}, clear=False):
                payload = list_endpoints(root)[0].to_dict()
            self.assertTrue(payload["key_present"])
            self.assertNotIn("TOP-SECRET", json.dumps(payload))

    def test_disable_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self._write(root, "remote", enabled=True)
            disable_all(root)
            self.assertIsNone(active_endpoint(root))

    def test_local_start_refuses_when_remote_active(self) -> None:
        fake = mock.Mock(display_name="Remote", to_dict=lambda: {"name": "remote"})
        with mock.patch("inferencedeck.control.active_endpoint", return_value=fake):
            result = ControlPlane().start("local")
        self.assertFalse(result["success"])
        self.assertIn("disable", result["error"])

    def test_benchmark_refuses_when_remote_active(self) -> None:
        fake = mock.Mock(display_name="Remote", to_dict=lambda: {"name": "remote"})
        with mock.patch("inferencedeck.control.active_endpoint", return_value=fake), \
                mock.patch("inferencedeck.control.run_profile_benchmark") as run:
            result = ControlPlane().benchmark("local")
        self.assertFalse(result["success"])
        self.assertIn("disable", result["error"])
        run.assert_not_called()
