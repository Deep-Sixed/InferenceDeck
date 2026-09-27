from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from unittest import mock

from inferencedeck import benchmark

RUNNING = {"id": "demo-123", "mode": "demo", "pid": 123, "running": True, "host": "127.0.0.1", "port": 8080}
REPLY = {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 8, "prompt_tokens": 4}}


def _fake_urlopen(req, timeout=None):
    return io.BytesIO(json.dumps(REPLY).encode("utf-8"))


class RunProfileBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.dict(os.environ, {"LCC_CACHE_DIR": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_running_server_is_reused_not_restarted(self) -> None:
        with mock.patch.object(benchmark, "list_servers", return_value=[RUNNING]), \
                mock.patch.object(benchmark, "start_profile") as start, \
                mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=_fake_urlopen) as urlopen:
            result = benchmark.run_profile_benchmark("demo", completion_tokens=64)
        self.assertTrue(result["success"])
        start.assert_not_called()
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["max_tokens"], 64)

    def test_started_server_gets_no_predict_cap(self) -> None:
        with mock.patch.object(benchmark, "list_servers", return_value=[]), \
                mock.patch.object(benchmark, "start_profile", return_value={"success": True, "server": RUNNING}) as start, \
                mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=_fake_urlopen):
            result = benchmark.run_profile_benchmark("demo", completion_tokens=64)
        self.assertTrue(result["success"])
        # The completion cap goes in the request; a server flag would outlive the benchmark.
        overrides = start.call_args.kwargs["overrides"] or {}
        self.assertNotIn("n_predict", overrides)
        self.assertFalse(start.call_args.kwargs["stop_existing"])

    def test_start_waits_as_long_as_the_runtime_needs(self) -> None:
        with mock.patch.object(benchmark, "list_servers", return_value=[]), \
                mock.patch.object(benchmark, "start_profile", return_value={"success": True, "server": RUNNING}) as start, \
                mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=_fake_urlopen):
            benchmark.run_profile_benchmark("demo")
        # None lets start_profile use the runtime's own ready timeout.
        self.assertIsNone(start.call_args.kwargs["ready_timeout_seconds"])

    def test_vllm_cpp_start_gets_its_full_ready_timeout(self) -> None:
        from inferencedeck import server_manager

        prepared = {
            "success": True,
            "runtime": "vllm.cpp",
            "command": {"argv": ["vllm-server"], "cwd": None, "warnings": []},
            "params": {"host": "127.0.0.1", "port": 18095},
            "profile": {"model": None},
            "warnings": [],
        }
        waits: list[int] = []

        def fake_wait(host, port, pid, timeout_seconds=45):
            waits.append(timeout_seconds)
            return True

        with mock.patch.object(benchmark, "list_servers", return_value=[]), \
                mock.patch.object(server_manager, "prepare_launch_command", return_value=prepared), \
                mock.patch.object(server_manager.subprocess, "Popen", return_value=mock.Mock(pid=424242)), \
                mock.patch.object(server_manager, "process_identity", return_value=None), \
                mock.patch.object(server_manager, "wait_until_ready", side_effect=fake_wait), \
                mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=_fake_urlopen):
            benchmark.run_profile_benchmark("demo")
        self.assertEqual(waits, [server_manager.READY_TIMEOUT_SECONDS["vllm.cpp"]])

    def test_paused_server_is_refused(self) -> None:
        paused = {**RUNNING, "suspended": True}
        with mock.patch.object(benchmark, "list_servers", return_value=[paused]), \
                mock.patch.object(benchmark.urllib.request, "urlopen") as urlopen:
            result = benchmark.run_profile_benchmark("demo")
        self.assertFalse(result["success"])
        self.assertIn("paused", result["error"])
        urlopen.assert_not_called()

    def test_restart_requested_starts_fresh(self) -> None:
        with mock.patch.object(benchmark, "list_servers", return_value=[RUNNING]), \
                mock.patch.object(benchmark, "start_profile", return_value={"success": True, "server": RUNNING}) as start, \
                mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=_fake_urlopen):
            benchmark.run_profile_benchmark("demo", overrides={"ctx_size": 8192}, restart=True)
        self.assertTrue(start.call_args.kwargs["stop_existing"])
        self.assertEqual(start.call_args.kwargs["overrides"], {"ctx_size": 8192})


if __name__ == "__main__":
    unittest.main()
