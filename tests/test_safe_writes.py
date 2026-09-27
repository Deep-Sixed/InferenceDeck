from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import benchmark, server_manager
from inferencedeck.config import AppConfig
from inferencedeck.fileio import atomic_write_text
from inferencedeck.remotes import enable_endpoint, list_endpoints


def _run_together(target, count: int = 16) -> None:
    barrier = threading.Barrier(count)
    errors: list[BaseException] = []

    def run(i: int) -> None:
        barrier.wait()
        try:
            target(i)
        except BaseException as exc:  # surface failures from worker threads
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    if errors:
        raise errors[0]


class HttpBaseTests(unittest.TestCase):
    def test_hosts(self) -> None:
        cases = {
            "127.0.0.1": "http://127.0.0.1:8080",
            "0.0.0.0": "http://127.0.0.1:8080",
            "": "http://127.0.0.1:8080",
            None: "http://127.0.0.1:8080",
            "::": "http://[::1]:8080",
            "::1": "http://[::1]:8080",
            "[::1]": "http://[::1]:8080",
            "fd00::5": "http://[fd00::5]:8080",
            "box.lan": "http://box.lan:8080",
        }
        for host, expected in cases.items():
            with self.subTest(host=host):
                self.assertEqual(server_manager.http_base(host, 8080), expected)

    def test_benchmark_uses_the_same_urls(self) -> None:
        self.assertEqual(benchmark._api_base({"host": "::1", "port": 9000}), "http://[::1]:9000")

    @unittest.skipUnless(socket.has_ipv6, "no IPv6")
    def test_server_on_ipv6_loopback_becomes_ready(self) -> None:
        class V6Server(HTTPServer):
            address_family = socket.AF_INET6

        class Models(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *_args) -> None:
                pass

        try:
            httpd = V6Server(("::1", 0), Models)
        except OSError:
            self.skipTest("IPv6 loopback not available")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            port = httpd.server_address[1]
            ready = server_manager.wait_until_ready("::1", port, os.getpid(), timeout_seconds=5)
            self.assertTrue(ready)
        finally:
            httpd.shutdown()
            httpd.server_close()


class ConcurrentWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        env = mock.patch.dict(os.environ, {"LCC_CACHE_DIR": str(self.root / "cache"), "LCC_CONFIG_DIR": str(self.root / "config")})
        env.start()
        self.addCleanup(env.stop)

    def test_atomic_write_leaves_no_temp_files(self) -> None:
        target = self.root / "out.json"
        _run_together(lambda i: atomic_write_text(target, json.dumps({"writer": i})))
        self.assertIn("writer", json.loads(target.read_text(encoding="utf-8")))
        self.assertEqual([p.name for p in self.root.iterdir()], ["out.json"])

    def test_concurrent_benchmark_results_are_all_kept(self) -> None:
        _run_together(lambda i: benchmark.save_benchmark_result({"mode": f"m{i}"}))
        modes = {r["mode"] for r in benchmark.load_benchmark_results()}
        self.assertEqual(modes, {f"m{i}" for i in range(16)})

    def test_concurrent_config_updates_are_all_kept(self) -> None:
        _run_together(lambda i: AppConfig.update(lambda c: c.model_dirs.append(f"/models/{i}")))
        self.assertEqual(set(AppConfig.load().model_dirs), {f"/models/{i}" for i in range(16)})

    def test_concurrent_enables_leave_exactly_one_endpoint_enabled(self) -> None:
        endpoints = self.root / "endpoints"
        endpoints.mkdir()
        env = {}
        for i in range(8):
            (endpoints / f"e{i}.json").write_text(json.dumps({
                "provider": "openai", "enabled": False, "lane": "true_cloud", "model": "m",
                "baseUrl": "http://example.invalid/v1", "apiKeyEnv": f"E{i}_KEY",
            }), encoding="utf-8")
            env[f"E{i}_KEY"] = "k"
        with mock.patch.dict(os.environ, env):
            _run_together(lambda i: enable_endpoint(f"e{i % 8}", endpoints))
        enabled = [cfg.name for cfg in list_endpoints(endpoints) if cfg.enabled]
        self.assertEqual(len(enabled), 1, enabled)


if __name__ == "__main__":
    unittest.main()
