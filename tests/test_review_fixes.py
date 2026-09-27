"""Regressions for the post-merge review of main (ae707fd)."""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import server_manager
from inferencedeck.control import ControlPlane

ROOT = Path(__file__).resolve().parent.parent


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        start_new_session=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


class _StateDir(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        patcher = mock.patch.object(server_manager, "cache_dir", return_value=Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.procs: list[subprocess.Popen] = []
        self.addCleanup(self._reap)

    def _reap(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)

    def sleeper(self) -> subprocess.Popen:
        p = _sleeper()
        self.procs.append(p)
        return p


class StalePidTests(_StateDir):
    """A PID in servers.json may have been reused by an unrelated program."""

    def test_stop_never_kills_a_process_with_a_different_start_time(self) -> None:
        victim = self.sleeper()
        identity = server_manager.process_identity(victim.pid)
        stale = {**identity, "start": identity["start"] + "0"}  # same PID, different process
        server_manager.write_state({"servers": [{"id": "old", "mode": "m", "pid": victim.pid, "process": stale}]})
        result = server_manager.stop_server(server_id="old")
        time.sleep(0.3)
        self.assertIsNone(victim.poll(), "an unrelated process was killed")
        self.assertTrue(result["success"])

    def test_legacy_record_is_matched_by_executable_name(self) -> None:
        # Records written before identity was stored have only the command line.
        victim = self.sleeper()
        server_manager.write_state({"servers": [
            {"id": "old", "mode": "m", "pid": victim.pid, "command_line": "/opt/llama/llama-server -m x.gguf --port 8080"}
        ]})
        server_manager.stop_server(server_id="old")
        time.sleep(0.3)
        self.assertIsNone(victim.poll(), "a python process was killed as if it were llama-server")
        self.assertEqual(server_manager.list_servers(), [])  # the stale record is dropped

    def test_pause_refuses_a_foreign_pid(self) -> None:
        victim = self.sleeper()
        server_manager.write_state({"servers": [{"id": "old", "mode": "m", "pid": victim.pid,
                                                 "process": {"start": "0", "image": "llama-server"}}]})
        self.assertFalse(server_manager.suspend_server(server_id="old")["success"])

    def test_the_real_server_is_still_recognised(self) -> None:
        own = self.sleeper()
        server_manager.write_state({"servers": [{"id": "s", "mode": "m", "pid": own.pid,
                                                 "process": server_manager.process_identity(own.pid)}]})
        self.assertTrue(server_manager.list_servers()[0]["running"])
        self.assertTrue(server_manager.stop_server(server_id="s")["success"])
        own.wait(timeout=10)


class ConcurrentStartTests(_StateDir):
    def _prepared(self) -> dict:
        return {
            "success": True,
            "command": {"argv": [sys.executable, "-c", "import time;time.sleep(60)"], "cwd": self.tmp},
            "params": {"host": "127.0.0.1", "port": 8080},
            "profile": {"model": {"path": "x.gguf"}},
        }

    def test_two_simultaneous_starts_launch_one_server(self) -> None:
        barrier = threading.Barrier(2)
        results: list[dict] = []

        def slow_prepare(*_a, **_k):
            time.sleep(0.2)  # both requests pass the "already running?" check together
            return self._prepared()

        def start() -> None:
            barrier.wait()
            results.append(server_manager.start_profile("m", wait_ready=False))

        with mock.patch.object(server_manager, "prepare_launch_command", slow_prepare):
            threads = [threading.Thread(target=start) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        live = [s for s in server_manager.list_servers() if s.get("running")]
        for s in live:
            self.addCleanup(_kill, s["pid"])
        self.assertEqual(sum(bool(r.get("success")) for r in results), 1, results)
        self.assertEqual(len(live), 1)
        self.assertRegex(next(r["error"] for r in results if not r.get("success")), "already (starting|has a tracked running server)")

    def test_abandoned_reservation_expires(self) -> None:
        old = time.time() - server_manager.LAUNCH_RESERVATION_SECONDS - 1
        server_manager.write_state({"servers": [{"id": "r", "mode": "m", "status": server_manager.LAUNCHING, "pid": None, "reserved_at": old}]})
        self.assertEqual(server_manager.list_servers(), [])


def _kill(pid: int) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            import os
            import signal

            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


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


if __name__ == "__main__":
    unittest.main()
