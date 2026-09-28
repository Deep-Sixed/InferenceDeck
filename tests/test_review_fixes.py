"""Regressions for the post-merge review of main (ae707fd).

The PID-reuse and concurrent-start findings are covered by main's own tests
(pid identity and the start lock landed separately); these are the rest.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck.control import ControlPlane

ROOT = Path(__file__).resolve().parent.parent


class BenchmarkRemoteGuardTests(unittest.TestCase):
    def test_benchmark_is_blocked_while_a_remote_endpoint_is_active(self) -> None:
        remote = mock.Mock(display_name="OpenAI")
        remote.to_dict.return_value = {"name": "oai"}
        with mock.patch("inferencedeck.control.active_endpoint", return_value=remote), \
             mock.patch("inferencedeck.control.run_profile_benchmark") as bench:
            result = ControlPlane().benchmark("qwen")
        self.assertFalse(result["success"])
        self.assertIn("Remote endpoint 'OpenAI' is active", result["error"])
        bench.assert_not_called()


class PackageDataTests(unittest.TestCase):
    def test_every_non_python_file_in_the_package_is_declared(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = re.search(r"\[tool\.setuptools\.package-data\]\s*inferencedeck\s*=\s*\[([^\]]*)\]", text)
        self.assertIsNotNone(block, "pyproject.toml declares no package-data for inferencedeck")
        patterns = re.findall(r'"([^"]+)"', block.group(1))
        package = ROOT / "inferencedeck"
        missing = [
            str(path.relative_to(package)).replace("\\", "/")
            for path in package.rglob("*")
            if path.is_file() and path.suffix not in {".py", ".pyc"} and "__pycache__" not in path.parts
            and not any(fnmatch.fnmatch(str(path.relative_to(package)).replace("\\", "/"), pat) for pat in patterns)
        ]
        self.assertEqual(missing, [], "files that would be left out of built wheels")


class StableRuntimeIdTests(unittest.TestCase):
    """Second review: a saved runtime pin must keep naming the same binary."""

    def _binary(self, directory: Path) -> Path:
        from inferencedeck.paths import executable_names

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / executable_names("llama-server")[0]
        path.write_text("x", encoding="utf-8")
        return path

    def test_pin_survives_a_new_build_being_discovered_first(self) -> None:
        from inferencedeck.cpu_features import CpuFeatures
        from inferencedeck.llama_runtimes import resolve_llama_runtime

        avx2 = CpuFeatures(x86=True, features=frozenset({"sse4_2", "avx", "avx2", "fma", "f16c", "bmi2"}), source="t")
        with tempfile.TemporaryDirectory() as tmp, mock.patch("inferencedeck.llama_runtimes.shutil.which", return_value=None):
            root = Path(tmp)
            self._binary(root / "b" / "bin")
            b = self._binary(root / "c" / "bin")
            first = resolve_llama_runtime([root / "b", root / "c"], cpu=avx2, has_cuda=False)
            pin = next(c["id"] for c in first["candidates"] if c["path"] == str(b))
            # A new standard build appears earlier in the search order.
            self._binary(root / "a" / "bin")
            later = resolve_llama_runtime([root / "a", root / "b", root / "c"], requested=pin, cpu=avx2, has_cuda=False)
        self.assertEqual(later["selected"]["path"], str(b))
        self.assertEqual(later["policy"], "pinned")

    def test_old_order_based_pin_falls_back_to_automatic(self) -> None:
        from inferencedeck.cpu_features import CpuFeatures
        from inferencedeck.llama_runtimes import resolve_llama_runtime

        avx2 = CpuFeatures(x86=True, features=frozenset({"sse4_2", "avx", "avx2", "fma", "f16c", "bmi2"}), source="t")
        with tempfile.TemporaryDirectory() as tmp, mock.patch("inferencedeck.llama_runtimes.shutil.which", return_value=None):
            root = Path(tmp)
            self._binary(root / "a" / "bin")
            self._binary(root / "b" / "bin")
            result = resolve_llama_runtime([root / "a", root / "b"], requested="standard-2", cpu=avx2, has_cuda=False)
        self.assertEqual(result["policy"], "auto")
        self.assertTrue(any("not found" in w for w in result["warnings"]))


class BenchmarkHistoryTests(unittest.TestCase):
    def test_concurrent_saves_keep_every_result(self) -> None:
        from inferencedeck import benchmark

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(benchmark, "cache_dir", return_value=Path(tmp)):
            barrier = threading.Barrier(12)

            def save(i: int) -> None:
                barrier.wait()
                benchmark.save_benchmark_result({"mode": f"m{i}"})

            threads = [threading.Thread(target=save, args=(i,)) for i in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            modes = {r["mode"] for r in benchmark.load_benchmark_results()}
            leftovers = [p.name for p in Path(tmp).iterdir() if p.suffix == ".tmp"]
        self.assertEqual(modes, {f"m{i}" for i in range(12)})
        self.assertEqual(leftovers, [])


class TransportSecurityTests(unittest.TestCase):
    def test_plain_http_lan_bind_is_refused(self) -> None:
        from inferencedeck.auth import AuthState
        from inferencedeck.webui import make_server

        with self.assertRaisesRegex(RuntimeError, "unencrypted"):
            make_server("0.0.0.0", 0, mock.Mock(spec=ControlPlane), AuthState(token="secret"))

    def test_explicit_opt_out_allows_plain_http(self) -> None:
        from inferencedeck.auth import AuthState
        from inferencedeck.webui import make_server

        server = make_server("0.0.0.0", 0, mock.Mock(spec=ControlPlane), AuthState(token="secret"),
                             allow_insecure_http=True)
        server.server_close()

    @unittest.skipUnless(__import__("shutil").which("openssl"), "needs openssl to make a test certificate")
    def test_https_serves_and_marks_cookies_secure(self) -> None:
        import json
        import ssl
        import urllib.request

        from inferencedeck.auth import AuthState
        from inferencedeck.webui import make_server

        with tempfile.TemporaryDirectory() as tmp:
            cert, key = Path(tmp) / "c.pem", Path(tmp) / "k.pem"
            subprocess.run(
                ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
                 "-keyout", str(key), "-out", str(cert)],
                check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            server = make_server("127.0.0.1", 0, mock.Mock(spec=ControlPlane), AuthState(username="admin", token="s3cret"),
                                 certfile=str(cert), keyfile=str(key))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                context = ssl.create_default_context(cafile=str(cert))
                context.check_hostname = False
                req = urllib.request.Request(
                    f"https://127.0.0.1:{server.server_port}/api/login",
                    data=json.dumps({"username": "admin", "password": "s3cret"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=5, context=context) as response:
                    cookie = response.headers.get("Set-Cookie", "")
            finally:
                server.shutdown()
                server.server_close()
        self.assertIn("Secure", cookie)
        self.assertIn("HttpOnly", cookie)


if __name__ == "__main__":
    unittest.main()
