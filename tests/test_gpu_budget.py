from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import gpu_budget, server_manager
from inferencedeck.config import AppConfig
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler

GIB = 1024 * 1024 * 1024
HW_24G = {"primary_gpu": {"name": "GPU", "vram_total_bytes": 24 * GIB}, "memory": {}, "recommended_fit_target_mib": 1024}


def _server(sid: str, mib: int | None, status: str = "running", running: bool = True, mode: str | None = None) -> dict:
    return {"id": sid, "mode": mode or sid, "status": status, "running": running, "estimated_vram_mib": mib}


class PlanStartTests(unittest.TestCase):
    def test_fits_and_tight(self) -> None:
        plan = gpu_budget.plan_start(8000, HW_24G, [_server("a", 10000)])
        self.assertEqual(plan["status"], "fits")
        self.assertEqual(plan["source"], "estimated")
        self.assertEqual(plan["available_mib"], 24576 - 10000)
        plan = gpu_budget.plan_start(14000, HW_24G, [_server("a", 10000)])
        self.assertEqual(plan["status"], "tight")
        self.assertEqual(plan["headroom_mib"], 576)

    def test_conflict_releases_fewest_biggest_first(self) -> None:
        servers = [_server("small", 3000), _server("big", 12000), _server("mid", 6000)]
        plan = gpu_budget.plan_start(10000, HW_24G, servers)
        self.assertEqual(plan["status"], "conflict")
        self.assertEqual(plan["release"], ["big"])  # 3576 free + 12000 is enough
        self.assertIn("big", plan["message"])

    def test_busy_servers_are_not_released(self) -> None:
        servers = [_server("small", 3000), _server("big", 12000), _server("mid", 8000)]
        plan = gpu_budget.plan_start(10000, HW_24G, servers, busy={"big": 2})
        self.assertEqual(plan["status"], "conflict")
        self.assertEqual(plan["release"], ["mid", "small"])  # the idle ones, biggest first
        self.assertEqual({s["id"]: s["in_flight"] for s in plan["running"]}, {"small": 0, "big": 2, "mid": 0})

    def test_only_busy_servers_would_make_room(self) -> None:
        servers = [_server("small", 3000), _server("big", 16000)]
        plan = gpu_budget.plan_start(10000, HW_24G, servers, busy={"big": 1})
        self.assertEqual(plan["status"], "busy")
        self.assertEqual(plan["release"], [])
        self.assertEqual(plan["busy"], ["big"])
        self.assertIn("big (1 running)", plan["message"])

    def test_too_big_even_alone(self) -> None:
        plan = gpu_budget.plan_start(30000, HW_24G, [_server("a", 4000)])
        self.assertEqual(plan["status"], "too_big")
        self.assertEqual(plan["release"], [])

    def test_parked_and_stopped_servers_do_not_count_but_paused_do(self) -> None:
        servers = [
            _server("parked", 20000, status="parked", running=False),
            _server("dead", 20000, running=False),
            {**_server("paused", 5000), "suspended": True},
        ]
        plan = gpu_budget.plan_start(8000, HW_24G, servers)
        self.assertEqual(plan["available_mib"], 24576 - 5000)
        self.assertEqual([s["id"] for s in plan["running"]], ["paused"])

    def test_excluded_server_is_ignored(self) -> None:
        plan = gpu_budget.plan_start(20000, HW_24G, [_server("same-mode", 20000)], exclude={"same-mode"})
        self.assertEqual(plan["status"], "fits")
        self.assertEqual(plan["running"], [])

    def test_live_free_counts_only_loading_servers_again(self) -> None:
        servers = [_server("ready", 10000), _server("loading", 4000, status="starting")]
        plan = gpu_budget.plan_start(6000, HW_24G, servers, live_free=12000)
        self.assertEqual(plan["source"], "live")
        self.assertEqual(plan["available_mib"], 8000)
        self.assertEqual(plan["status"], "fits")

    def test_unknown_cases(self) -> None:
        self.assertEqual(gpu_budget.plan_start(None, HW_24G, [])["status"], "unknown")
        self.assertEqual(gpu_budget.plan_start(0, HW_24G, [])["status"], "unknown")
        self.assertEqual(gpu_budget.plan_start(8000, {"primary_gpu": {}, "memory": {}}, [])["status"], "unknown")

    def test_unestimated_servers_are_warned_about(self) -> None:
        plan = gpu_budget.plan_start(8000, HW_24G, [_server("vllm", None)])
        self.assertEqual(len(plan["warnings"]), 1)

    def test_unified_memory_capacity(self) -> None:
        hw = {"primary_gpu": {"unified_memory": True}, "memory": {"unified": True, "total_bytes": 64 * GIB}}
        plan = gpu_budget.plan_start(30000, hw, [_server("a", 20000)])
        self.assertEqual(plan["capacity_mib"], 65536)
        self.assertEqual(plan["status"], "fits")


class LiveFreeTests(unittest.TestCase):
    def test_nvidia(self) -> None:
        hw = {"primary_gpu": {"backend": "nvidia-smi", "index": 1}, "memory": {}}
        gpus = [{"index": 0, "vram_free_bytes": GIB}, {"index": 1, "vram_free_bytes": 5 * GIB}]
        with mock.patch.object(gpu_budget, "_nvidia_smi_gpus", return_value=gpus):
            self.assertEqual(gpu_budget.live_free_mib(hw), 5120)
        with mock.patch.object(gpu_budget, "_nvidia_smi_gpus", return_value=[]):
            self.assertIsNone(gpu_budget.live_free_mib(hw))

    def test_unified_memory_uses_available_ram(self) -> None:
        hw = {"primary_gpu": {"unified_memory": True}, "memory": {"unified": True}}
        with mock.patch.object(gpu_budget, "_posix_memory_info", return_value={"available_bytes": 10 * GIB}), \
                mock.patch.object(gpu_budget, "_windows_memory_info", return_value={"available_bytes": 10 * GIB}):
            self.assertEqual(gpu_budget.live_free_mib(hw), 10240)

    def test_other_gpus_have_no_live_number(self) -> None:
        self.assertIsNone(gpu_budget.live_free_mib({"primary_gpu": {"backend": "wmi"}, "memory": {}}))

    def test_estimate_server_vram(self) -> None:
        model = {"path": "/m.gguf", "size_bytes": 8 * GIB, "params_b": 14}
        full = gpu_budget.estimate_server_vram_mib({"gpu_layers": 999, "ctx_size": 8192}, model)
        cpu = gpu_budget.estimate_server_vram_mib({"gpu_layers": 0, "ctx_size": 8192}, model)
        self.assertGreater(full, 8000)
        self.assertEqual(cpu, 0)
        with tempfile.TemporaryDirectory() as tmp:
            projector = Path(tmp) / "mmproj.gguf"
            projector.write_bytes(b"\0" * (3 * 1024 * 1024))
            with_projector = gpu_budget.estimate_server_vram_mib(
                {"gpu_layers": 999, "ctx_size": 8192, "mmproj": str(projector)}, model
            )
            cpu_with_projector = gpu_budget.estimate_server_vram_mib(
                {"gpu_layers": 0, "ctx_size": 8192, "mmproj": str(projector)}, model
            )
        self.assertEqual(with_projector, full + 3)
        self.assertEqual(cpu_with_projector, 0)


class StartProfileVramTests(unittest.TestCase):
    """start_profile with a real (sleeper) process and a fake 24 GiB GPU."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.procs: list[subprocess.Popen] = []
        self.config = AppConfig()
        sleeper = [sys.executable, "-c", "import time;time.sleep(60)"]
        prepared = {
            "success": True,
            "runtime": "llama.cpp",
            "profile": {"model": {"path": "/m.gguf"}},
            "params": {"host": "127.0.0.1", "port": 1},
            "command": {"argv": sleeper, "cwd": None, "warnings": []},
            "warnings": [],
        }
        for target, value in [
            ("cache_dir", Path(tmp.name)),
            ("prepare_launch_command", prepared),
            ("detect_system_hardware", HW_24G),
            ("live_free_mib", None),
            ("estimate_server_vram_mib", 14000),
        ]:
            patcher = mock.patch.object(server_manager, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(server_manager.AppConfig, "load", side_effect=lambda *a, **k: self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Gateway requests running per server id (inflight.busy_servers).
        self.busy: list[dict[str, int]] = [{}]
        patcher = mock.patch.object(server_manager, "busy_servers",
                                    side_effect=lambda: self.busy.pop(0) if len(self.busy) > 1 else self.busy[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        # Cleanups run last-in first-out: reap while the state dir is still patched,
        # so started servers are stopped before Windows is asked to delete their logs.
        self.addCleanup(self._reap)

    def _reap(self) -> None:
        pids = []
        for server in server_manager.read_state().get("servers", []):
            pid = server.get("pid") or server.get("last_pid")
            if pid and server_manager.pid_is_running(pid):
                pids.append(int(pid))
                if not server_manager.stop_server(server_id=server["id"]).get("success"):
                    os.kill(int(pid), signal.SIGTERM)
        # Windows keeps a log file locked until its process has fully exited, so
        # wait for that before the temp dir is deleted.
        deadline = time.monotonic() + 10
        for pid in pids:
            while server_manager.pid_is_running(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=5)

    def _track_other(self, mib: int = 16000) -> str:
        proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
        self.procs.append(proc)
        sid = f"other-{proc.pid}"
        server_manager._upsert_server(
            {"id": sid, "mode": "other", "pid": proc.pid, "status": "running", "estimated_vram_mib": mib, "overrides": {}}
        )
        return sid

    def _start(self, **kwargs) -> dict:
        return server_manager.start_profile("qwen", wait_ready=False, **kwargs)

    def test_conflict_is_refused_without_launching(self) -> None:
        self._track_other()
        result = self._start()
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "vram_conflict")
        self.assertEqual(result["vram_plan"]["release"][0].split("-")[0], "other")
        self.assertIsNone(server_manager._find_server(mode="qwen"))

    def test_release_conflicts_parks_the_other_server_then_starts(self) -> None:
        other = self._track_other()
        result = self._start(release_conflicts=True)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["vram_plan"]["released"], [other])
        self.assertEqual(server_manager._find_server(other)["status"], server_manager.PARKED)
        self.assertEqual(result["server"]["estimated_vram_mib"], 14000)

    def test_server_answering_requests_is_not_released(self) -> None:
        other = self._track_other()
        self.busy = [{other: 2}]
        for kwargs in ({}, {"release_conflicts": True}):
            result = self._start(**kwargs)
            self.assertFalse(result["success"])
            self.assertEqual(result["reason"], "vram_busy")
            self.assertEqual(result["vram_plan"]["busy"], [other])
        self.assertEqual(server_manager._find_server(other)["status"], "running")
        self.assertIsNone(server_manager._find_server(mode="qwen"))
        forced = self._start(force=True)
        self.assertTrue(forced["success"], forced)
        self.assertEqual(server_manager._find_server(other)["status"], "running")

    def test_request_arriving_after_the_plan_stops_the_release(self) -> None:
        other = self._track_other()
        self.busy = [{}, {other: 1}]  # idle when planned, busy when about to release
        result = self._start(release_conflicts=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "vram_busy")
        self.assertEqual(server_manager._find_server(other)["status"], "running")

    def test_force_starts_with_a_warning(self) -> None:
        other = self._track_other()
        result = self._start(force=True)
        self.assertTrue(result["success"], result)
        self.assertTrue(any("Not enough GPU memory" in w for w in result["server"]["warnings"]))
        self.assertEqual(server_manager._find_server(other)["status"], "running")

    def test_warn_and_off_modes(self) -> None:
        self._track_other()
        self.config = AppConfig(concurrent_vram_check="warn")
        warned = self._start()
        self.assertTrue(warned["success"], warned)
        self.assertTrue(any("Not enough GPU memory" in w for w in warned["server"]["warnings"]))
        server_manager.stop_server(mode="qwen")

        self.config = AppConfig(concurrent_vram_check="off")
        quiet = self._start()
        self.assertTrue(quiet["success"], quiet)
        self.assertIsNone(quiet["vram_plan"])

    def test_fits_starts_normally(self) -> None:
        self._track_other(mib=4000)
        result = self._start()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["vram_plan"]["status"], "fits")

    def test_restore_passes_flags_through(self) -> None:
        self._track_other()
        server_manager._upsert_server(
            {"id": "qwen-old", "mode": "qwen", "pid": None, "status": server_manager.PARKED,
             "restart": {"mode": "qwen", "overrides": {}}}
        )
        refused = server_manager.restore_server("qwen-old")
        self.assertEqual(refused["reason"], "vram_conflict")
        self.assertEqual(server_manager._find_server("qwen-old")["status"], server_manager.PARKED)
        with mock.patch.object(server_manager, "wait_until_ready", return_value=True), \
                mock.patch.object(server_manager, "probe_capabilities", return_value=None):
            restored = server_manager.restore_server("qwen-old", release_conflicts=True)
        self.assertTrue(restored["success"], restored)


class VramApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.create_autospec(ControlPlane, instance=True)
        self.control = handler.control_plane
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_conflict_is_409_and_flags_are_forwarded(self) -> None:
        self.control.start.return_value = {"success": False, "reason": "vram_conflict", "error": "no room"}
        status, payload = self._post("/api/start", {"mode": "m"})
        self.assertEqual(status, 409)
        self.assertEqual(payload["reason"], "vram_conflict")
        self.control.start.assert_called_with("m", None, stop_existing=False)

        self.control.start.return_value = {"success": True}
        self._post("/api/start", {"mode": "m", "release_conflicts": True})
        self.control.start.assert_called_with("m", None, stop_existing=False, release_conflicts=True)

    def test_busy_is_409(self) -> None:
        self.control.start.return_value = {"success": False, "reason": "vram_busy", "error": "busy"}
        status, payload = self._post("/api/start", {"mode": "m"})
        self.assertEqual((status, payload["reason"]), (409, "vram_busy"))

    def test_restore_restart_and_plan(self) -> None:
        self.control.restore.return_value = {"success": True}
        self.control.restart.return_value = {"success": True}
        self.control.plan.return_value = {"success": True, "vram_plan": {"status": "fits"}}
        self._post("/api/restore", {"server_id": "s", "force": True})
        self.control.restore.assert_called_with("s", None, force=True)
        self._post("/api/restart", {"server_id": "s", "ctx_size": 8192, "release_conflicts": "yes"})
        self.control.restart.assert_called_with("s", {"ctx_size": 8192})  # only literal true counts
        status, payload = self._post("/api/plan", {"mode": "m"})
        self.assertEqual((status, payload["vram_plan"]["status"]), (200, "fits"))


if __name__ == "__main__":
    unittest.main()
