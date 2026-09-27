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


class EndpointLocationTests(unittest.TestCase):
    def _parse(self, **fields):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep.json"
            path.write_text(json.dumps({"apiKeyEnv": "K", **fields}))
            return list_endpoints(Path(tmp))[0]

    def test_self_hosted_over_tailscale(self) -> None:
        cfg = self._parse(provider="llamacpp", lane="remote_host", host="Thanatos",
                          transport="tailscale", baseUrl="http://thanatos:8080/v1")
        self.assertTrue(cfg.valid)
        self.assertEqual(cfg.summary, "Thanatos · Tailscale · Self-hosted")
        self.assertEqual(cfg.to_dict()["summary"], cfg.summary)

    def test_transport_and_host_inferred_from_url(self) -> None:
        cases = {
            "http://thanatos.tail1234.ts.net:8080/v1": "tailscale",
            "http://100.101.102.103:8080/v1": "tailscale",
            "http://192.168.1.20:8080/v1": "",
        }
        for url, transport in cases.items():
            cfg = self._parse(provider="llamacpp", baseUrl=url)
            self.assertEqual(cfg.transport, transport, url)
        cfg = self._parse(provider="llamacpp", baseUrl="http://192.168.1.20:8080/v1")
        self.assertEqual(cfg.summary, "192.168.1.20 · Self-hosted")

    def test_cloud_provider_summary(self) -> None:
        cfg = self._parse(provider="openrouter", lane="true_cloud", baseUrl="https://openrouter.ai/api/v1")
        self.assertEqual(cfg.transport, "https")
        self.assertEqual(cfg.summary, "OpenRouter · Cloud")

    def test_unknown_transport_is_invalid(self) -> None:
        cfg = self._parse(provider="llamacpp", baseUrl="http://x:8080/v1", transport="carrier-pigeon")
        self.assertFalse(cfg.valid)
        self.assertIn("transport", cfg.error)

    def test_examples_parse(self) -> None:
        examples = Path(__file__).resolve().parents[1] / "examples" / "remote_endpoints"
        with tempfile.TemporaryDirectory() as tmp:
            for example in examples.glob("*.example.json"):
                (Path(tmp) / example.name.replace(".example", "")).write_text(example.read_text())
            configs = list_endpoints(Path(tmp))
        self.assertTrue(configs)
        for cfg in configs:
            self.assertTrue(cfg.valid, f"{cfg.name}: {cfg.error}")
