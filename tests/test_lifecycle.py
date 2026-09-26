from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import server_manager
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(server_manager, "cache_dir", return_value=Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.procs: list[subprocess.Popen] = []
        self.addCleanup(self._reap)

    def _reap(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=5)

    def _track(self, mode: str = "qwen", overrides: dict | None = None) -> str:
        proc = _sleeper()
        self.procs.append(proc)
        server_id = f"{mode}-{proc.pid}"
        server_manager._upsert_server(
            {"id": server_id, "mode": mode, "pid": proc.pid, "status": "running", "overrides": overrides or {}}
        )
        return server_id

    def _fake_start(self, success: bool = True):
        calls: list[tuple] = []

        def fake(mode, project_root=None, model_dirs=None, overrides=None, **_kw):
            calls.append((mode, overrides))
            if not success:
                return {"success": False, "error": "boom"}
            return {"success": True, "server": {"id": self._track(mode, overrides), "running": True}}

        return calls, fake

    def test_release_stops_process_and_parks_restart_spec(self) -> None:
        sid = self._track(overrides={"ctx_size": 8192})
        pid = self.procs[-1].pid
        result = server_manager.release_gpu(server_id=sid)
        self.assertTrue(result["success"], result)
        self.assertFalse(server_manager.pid_is_running(pid))  # the process is gone, so its VRAM is freed
        parked = server_manager._find_server(sid)  # survives the stale-PID prune
        self.assertEqual(parked["status"], server_manager.PARKED)
        self.assertIsNone(parked["pid"])
        self.assertEqual(parked["restart"], {"mode": "qwen", "overrides": {"ctx_size": 8192}})

    def test_restore_starts_saved_spec_and_drops_parked_record(self) -> None:
        sid = self._track(overrides={"ctx_size": 8192, "gpu_layers": 20})
        server_manager.release_gpu(server_id=sid)
        calls, fake = self._fake_start()
        with mock.patch.object(server_manager, "start_profile", fake):
            result = server_manager.restore_server(sid)
        self.assertTrue(result["success"], result)
        self.assertEqual(calls, [("qwen", {"ctx_size": 8192, "gpu_layers": 20})])
        self.assertIsNone(server_manager._find_server(sid))

    def test_failed_restore_stays_parked_for_retry(self) -> None:
        sid = self._track()
        server_manager.release_gpu(server_id=sid)
        _calls, fake = self._fake_start(success=False)
        with mock.patch.object(server_manager, "start_profile", fake):
            self.assertFalse(server_manager.restore_server(sid)["success"])
        self.assertEqual(server_manager._find_server(sid)["status"], server_manager.PARKED)

    def test_restore_is_claimed_once(self) -> None:
        sid = self._track()
        server_manager.release_gpu(server_id=sid)
        server_manager._update_server(sid, {"status": server_manager.RESTORING})  # another request got there first
        calls, fake = self._fake_start()
        with mock.patch.object(server_manager, "start_profile", fake):
            self.assertFalse(server_manager.restore_server(sid)["success"])
        self.assertEqual(calls, [])

    def test_restart_changes_only_context_size(self) -> None:
        sid = self._track(overrides={"ctx_size": 8192, "gpu_layers": 20})
        old_pid = self.procs[-1].pid
        calls, fake = self._fake_start()
        with mock.patch.object(server_manager, "start_profile", fake):
            result = server_manager.restart_server(sid, {"ctx_size": 32768})
        self.assertTrue(result["success"], result)
        self.assertFalse(server_manager.pid_is_running(old_pid))
        self.assertEqual(calls, [("qwen", {"ctx_size": 32768, "gpu_layers": 20})])

    def test_stop_forgets_parked_record(self) -> None:
        sid = self._track()
        server_manager.release_gpu(server_id=sid)
        self.assertTrue(server_manager.stop_server(server_id=sid)["success"])
        self.assertIsNone(server_manager._find_server(sid))

    def test_trim_keeps_parked_records(self) -> None:
        sid = self._track()
        server_manager.release_gpu(server_id=sid)
        server_manager._upsert_server({"id": "history-1"})
        server_manager._upsert_server({"id": "history-2"})
        server_manager.trim_server_history(limit=1)
        ids = [s["id"] for s in server_manager.read_state()["servers"]]
        self.assertIn(sid, ids)

    def test_stop_by_mode_prefers_running_over_parked(self) -> None:
        parked = self._track()
        server_manager.release_gpu(server_id=parked)
        live = self._track()
        live_pid = self.procs[-1].pid
        server_manager.stop_server(mode="qwen")
        self.assertFalse(server_manager.pid_is_running(live_pid))
        self.assertEqual(server_manager._find_server(parked)["status"], server_manager.PARKED)
        self.assertIsNone(server_manager._find_server(live))


class LifecycleApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.restart.return_value = {"success": True}
        handler.control_plane.release_gpu.return_value = {"success": True}
        self.control = handler.control_plane
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _post(self, path: str, body: dict) -> int:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_restart_with_context_preset(self) -> None:
        self.assertEqual(self._post("/api/restart", {"server_id": "s", "ctx_size": 65536}), 200)
        self.control.restart.assert_called_once_with("s", {"ctx_size": 65536})

    def test_restart_rejects_bad_context_size(self) -> None:
        self.assertEqual(self._post("/api/restart", {"server_id": "s", "ctx_size": 12}), 400)
        self.control.restart.assert_not_called()

    def test_release_requires_server_id(self) -> None:
        self.assertEqual(self._post("/api/release", {}), 400)
        self.assertEqual(self._post("/api/release", {"server_id": "s"}), 200)
        self.control.release_gpu.assert_called_once_with(server_id="s")
