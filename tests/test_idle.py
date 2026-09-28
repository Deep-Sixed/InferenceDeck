from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import idle, server_manager
from inferencedeck.config import AppConfig
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)


def _slots(task: int, processing: bool = False) -> list[dict]:
    return [{"id": 0, "id_task": task, "is_processing": processing}]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class IdleMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(server_manager, "cache_dir", return_value=Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.procs: list[subprocess.Popen] = []
        self.addCleanup(self._reap)
        self.clock = _Clock()
        self.slots: list[dict] | None = _slots(1)
        self.gateway: dict = {}  # in-flight gateway counts (inflight.snapshot)
        self.config = AppConfig(idle_release_seconds=0)
        self.release = mock.Mock(return_value={"success": True})

    def _reap(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=5)

    def _track(self, **fields) -> str:
        proc = _sleeper()
        self.procs.append(proc)
        server_id = f"qwen-{proc.pid}"
        record = {"id": server_id, "mode": "qwen", "pid": proc.pid, "status": "running", "host": "127.0.0.1", "port": 1}
        record.update(fields)
        server_manager._upsert_server(record)
        return server_id

    def _monitor(self, release=None) -> idle.IdleMonitor:
        return idle.IdleMonitor(
            probe=lambda _host, _port: self.slots,
            release=release or self.release,
            clock=self.clock,
            config_loader=lambda: self.config,
            inflight=lambda: self.gateway,
        )

    def test_releases_after_idle_window(self) -> None:
        sid = self._track(idle_release_seconds=60)
        monitor = self._monitor(release=server_manager.release_gpu)
        self.assertEqual(monitor.check(), [])  # first sight starts the window
        self.clock.now += 59
        self.assertEqual(monitor.check(), [])
        self.clock.now += 2
        released = monitor.check()
        self.assertEqual(len(released), 1)
        record = server_manager._find_server(sid)
        self.assertEqual(record["status"], server_manager.PARKED)
        self.assertEqual(record["released_by"], "idle")
        self.assertEqual(record["idle_seconds"], 61)
        self.assertIsNotNone(self.procs[0].wait(timeout=10))

    def test_new_request_resets_the_window(self) -> None:
        self._track(idle_release_seconds=60)
        monitor = self._monitor()
        monitor.check()
        self.clock.now += 50
        self.slots = _slots(2)  # a request came and went between polls
        monitor.check()
        self.clock.now += 50
        monitor.check()
        self.release.assert_not_called()
        self.clock.now += 11
        monitor.check()
        self.release.assert_called_once()

    def test_busy_server_is_never_released(self) -> None:
        self._track(idle_release_seconds=60)
        self.slots = _slots(1, processing=True)
        monitor = self._monitor()
        for _ in range(5):
            monitor.check()
            self.clock.now += 100
        self.release.assert_not_called()

    def test_gateway_request_in_flight_keeps_the_server(self) -> None:
        # e.g. a long prompt still uploading: no slot is processing yet.
        sid = self._track(idle_release_seconds=60)
        self.gateway = {sid: {"in_flight": 1, "requests": 1, "last_request_at": None}}
        monitor = self._monitor()
        for _ in range(5):
            monitor.check()
            self.clock.now += 100
        self.release.assert_not_called()
        self.gateway = {sid: {"in_flight": 0, "requests": 1, "last_request_at": None}}
        monitor.check()  # the request finishing counts as activity
        self.clock.now += 59
        monitor.check()
        self.release.assert_not_called()
        self.clock.now += 2
        monitor.check()
        self.release.assert_called_once_with(server_id=sid)

    def test_finished_gateway_requests_reset_the_window(self) -> None:
        sid = self._track(idle_release_seconds=60)
        monitor = self._monitor()
        monitor.check()
        self.clock.now += 50
        self.gateway = {sid: {"in_flight": 0, "requests": 3, "last_request_at": None}}
        monitor.check()
        self.clock.now += 50
        monitor.check()
        self.release.assert_not_called()
        self.clock.now += 11
        monitor.check()
        self.release.assert_called_once()

    def test_gateway_snapshot_failure_is_ignored(self) -> None:
        self._track(idle_release_seconds=60)
        monitor = idle.IdleMonitor(probe=lambda _h, _p: self.slots, release=self.release, clock=self.clock,
                                   config_loader=lambda: self.config, inflight=mock.Mock(side_effect=OSError))
        monitor.check()
        self.clock.now += 61
        monitor.check()
        self.release.assert_called_once()

    def test_unreadable_slots_count_as_active(self) -> None:
        self._track(idle_release_seconds=60)
        monitor = self._monitor()
        monitor.check()
        self.clock.now += 100
        self.slots = None
        monitor.check()
        self.slots = _slots(1)
        self.clock.now += 30
        monitor.check()
        self.release.assert_not_called()

    def test_request_arriving_at_the_last_moment_cancels_release(self) -> None:
        self._track(idle_release_seconds=60)
        answers = iter([_slots(1), _slots(1), _slots(1, processing=True)])
        monitor = idle.IdleMonitor(
            probe=lambda _h, _p: next(answers),
            release=self.release,
            clock=self.clock,
            config_loader=lambda: self.config,
        )
        monitor.check()
        self.clock.now += 61
        monitor.check()
        self.release.assert_not_called()

    def test_config_default_and_per_server_override(self) -> None:
        self._track()
        monitor = self._monitor()
        monitor.check()
        self.clock.now += 10_000
        monitor.check()
        self.release.assert_not_called()  # default is off

        self.config = AppConfig(idle_release_seconds=30)
        monitor.check()
        self.clock.now += 31
        monitor.check()
        self.assertEqual(self.release.call_count, 1)

    def test_server_can_opt_out_of_the_default(self) -> None:
        self.config = AppConfig(idle_release_seconds=30)
        self._track(idle_release_seconds=0)
        monitor = self._monitor()
        monitor.check()
        self.clock.now += 1000
        monitor.check()
        self.release.assert_not_called()

    def test_paused_starting_and_parked_servers_are_skipped(self) -> None:
        self._track(idle_release_seconds=1, suspended=True)
        self._track(idle_release_seconds=1, status="starting")
        server_manager._upsert_server({"id": "parked", "status": server_manager.PARKED, "pid": None, "idle_release_seconds": 1})
        probe = mock.Mock(return_value=_slots(1))
        monitor = idle.IdleMonitor(probe=probe, release=self.release, clock=self.clock, config_loader=lambda: self.config)
        monitor.check()
        self.clock.now += 100
        monitor.check()
        probe.assert_not_called()
        self.release.assert_not_called()

    def test_set_idle_release_updates_overrides_and_restart_spec(self) -> None:
        sid = self._track(overrides={"ctx_size": 8192})
        result = server_manager.set_idle_release(sid, 900)
        self.assertTrue(result["success"])
        record = server_manager._find_server(sid)
        self.assertEqual(record["idle_release_seconds"], 900)
        self.assertEqual(record["overrides"], {"ctx_size": 8192, "idle_release_seconds": 900})

        released = server_manager.release_gpu(server_id=sid)
        self.assertTrue(released["success"])
        self.assertEqual(released["server"]["restart"]["overrides"]["idle_release_seconds"], 900)
        server_manager.set_idle_release(sid, None)
        parked = server_manager._find_server(sid)
        self.assertIsNone(parked["idle_release_seconds"])
        self.assertEqual(parked["restart"]["overrides"], {"ctx_size": 8192})

    def test_set_idle_release_rejects_bad_values(self) -> None:
        sid = self._track()
        for bad in (-1, 1.5, True, "60"):
            self.assertFalse(server_manager.set_idle_release(sid, bad)["success"])
        self.assertFalse(server_manager.set_idle_release("missing", 60)["success"])


class SlotsTests(unittest.TestCase):
    def test_slots_state(self) -> None:
        busy, fp = idle.slots_state([{"id": 1, "id_task": 7, "is_processing": False}, {"id": 0, "id_task": 3}])
        self.assertFalse(busy)
        self.assertEqual(fp, (("0", "3"), ("1", "7")))
        self.assertTrue(idle.slots_state(_slots(1, processing=True))[0])

    def test_probe_slots_reads_llama_server(self) -> None:
        replies = {"/slots": (200, _slots(4))}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_GET(self) -> None:
                status, body = replies.get(self.path, (404, {}))
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        self.assertEqual(idle.probe_slots("127.0.0.1", port), _slots(4))
        replies["/slots"] = (501, {"error": "slots disabled"})
        self.assertIsNone(idle.probe_slots("127.0.0.1", port))

    def test_monitor_thread_polls_until_stopped(self) -> None:
        polled = threading.Event()
        monitor = mock.Mock()
        monitor.check.side_effect = lambda: polled.set()
        stop = threading.Event()
        thread = idle.start_idle_monitor(interval_seconds=0.01, stop=stop, monitor=monitor)
        self.assertTrue(polled.wait(5))
        stop.set()
        thread.join(5)
        self.assertFalse(thread.is_alive())


class IdleApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.create_autospec(ControlPlane, instance=True)
        handler.control_plane.set_idle_release.return_value = {"success": True}
        self.control = handler.control_plane
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def _post(self, body: dict) -> int:
        req = urllib.request.Request(
            self.base + "/api/idle",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_sets_and_clears_window(self) -> None:
        self.assertEqual(self._post({"server_id": "s", "seconds": 1800}), 200)
        self.control.set_idle_release.assert_called_with("s", 1800)
        self.assertEqual(self._post({"server_id": "s", "seconds": None}), 200)
        self.control.set_idle_release.assert_called_with("s", None)

    def test_rejects_bad_input(self) -> None:
        self.assertEqual(self._post({"seconds": 60}), 400)
        self.assertEqual(self._post({"server_id": "s", "seconds": -5}), 400)
        self.assertEqual(self._post({"server_id": "s", "seconds": "60"}), 400)
        self.assertEqual(self._post({"server_id": "s", "seconds": 10**9}), 400)
        self.control.set_idle_release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
