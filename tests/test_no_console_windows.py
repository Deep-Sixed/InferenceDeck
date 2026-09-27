from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import hardware, proc, server_manager

PACKAGE = Path(__file__).resolve().parent.parent / "inferencedeck"


class NoConsoleWindowTests(unittest.TestCase):
    def test_helper_processes_go_through_the_hidden_runner(self) -> None:
        # A bare subprocess.run of a console program (PowerShell, tasklist,
        # nvidia-smi, llama-server) flashes a window on Windows.
        offenders = [
            f"{path.name}:{lineno}"
            for path in PACKAGE.rglob("*.py")
            if path.name != "proc.py"
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"\bsubprocess\.(run|call|check_output|check_call)\(", line)
        ]
        self.assertEqual(offenders, [], "use inferencedeck.proc.run instead")

    def test_popen_calls_hide_their_window(self) -> None:
        for path in PACKAGE.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"subprocess\.Popen\(", text):
                call = text[match.start(): match.start() + 1200]
                self.assertIn("CREATE_NO_WINDOW", call.split("\n)\n")[0], f"{path.name}: Popen without CREATE_NO_WINDOW")

    def test_hidden_runner_passes_create_no_window(self) -> None:
        flags = {"creationflags": 0x08000000}
        with mock.patch.object(proc, "NO_WINDOW", flags), mock.patch.object(proc.subprocess, "run") as run:
            proc.run(["x"], capture_output=True)
        run.assert_called_once_with(["x"], creationflags=0x08000000, capture_output=True)

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_no_window_flag_is_set_on_windows(self) -> None:
        self.assertEqual(proc.NO_WINDOW, {"creationflags": subprocess.CREATE_NO_WINDOW})


class HardwareCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        hardware._hardware_cache = None
        self.addCleanup(setattr, hardware, "_hardware_cache", None)

    def test_repeated_requests_reuse_one_detection(self) -> None:
        with mock.patch.object(hardware, "_detect_system_hardware", return_value={"cpu": {"name": "x"}}) as detect:
            for _ in range(5):
                hardware.detect_system_hardware()
        detect.assert_called_once()

    def test_cache_expires_and_callers_get_copies(self) -> None:
        with mock.patch.object(hardware, "_detect_system_hardware", return_value={"cpu": {"name": "x"}}) as detect:
            first = hardware.detect_system_hardware()
            first["cpu"]["name"] = "mutated"
            self.assertEqual(hardware.detect_system_hardware()["cpu"]["name"], "x")
            hardware.detect_system_hardware(max_age=0)
        self.assertEqual(detect.call_count, 2)


@unittest.skipUnless(sys.platform == "win32", "Windows only")
class WindowsPidTests(unittest.TestCase):
    def test_pid_check_uses_the_kernel_not_tasklist(self) -> None:
        proc_ = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            with mock.patch.object(server_manager, "run_hidden", side_effect=AssertionError("spawned a process")):
                self.assertTrue(server_manager.pid_is_running(proc_.pid))
                proc_.kill()
                proc_.wait(timeout=10)
                self.assertFalse(server_manager.pid_is_running(proc_.pid))
        finally:
            if proc_.poll() is None:
                proc_.kill()


if __name__ == "__main__":
    unittest.main()
