from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

from inferencedeck import inflight
from inferencedeck.auth import AuthState
from inferencedeck.control import ControlPlane
from inferencedeck.gateway.engines import OpenAICompatibleEngine
from inferencedeck.gateway.router import Router, Target, local_target
from inferencedeck.gateway.server import make_server


class _Dir(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        patcher = mock.patch.object(inflight, "cache_dir", return_value=self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)


class TrackerTests(_Dir):
    def test_slow_disk_never_blocks_other_requests(self) -> None:
        t = inflight.Tracker(pid=os.getpid())
        release = threading.Event()
        real_write = inflight.atomic_write_text

        def slow_write(path, text):
            release.wait(5)
            real_write(path, text)

        with mock.patch.object(inflight, "atomic_write_text", slow_write):
            first = threading.Thread(target=t.begin, args=("a",))
            first.start()
            time.sleep(0.1)  # the first request is now stuck writing its file
            started = time.monotonic()
            t.counts()  # needs the request lock, not the write lock
            self.assertLess(time.monotonic() - started, 1.0)
            release.set()
            first.join(5)
        self.assertEqual(json.loads(t.path.read_text())["servers"]["a"]["in_flight"], 1)

    def test_stale_write_never_overwrites_a_newer_one(self) -> None:
        t = inflight.Tracker(pid=os.getpid())
        t.begin("a")
        t._write(1, "old")  # an earlier state arriving late
        self.assertEqual(json.loads(t.path.read_text())["servers"]["a"]["in_flight"], 1)

    def test_gateway_identity_is_checked_once_per_ttl(self) -> None:
        t = inflight.Tracker(pid=os.getpid())
        t.begin("a")
        inflight._confirmed.clear()
        with mock.patch("inferencedeck.server_manager._is_same_process", return_value=True) as same:
            for _ in range(5):
                self.assertEqual(inflight.busy_servers(), {"a": 1})
        self.assertEqual(same.call_count, 1)

    def test_counts_are_published_and_merged(self) -> None:
        tracker = inflight.Tracker()
        self.assertEqual(tracker.path, self.dir / "inflight" / f"gateway-{os.getpid()}.json")
        tracker.begin("a")
        tracker.begin("a")
        tracker.begin("b")
        tracker.end("b")
        snap = inflight.snapshot()
        self.assertEqual({k: (v["in_flight"], v["requests"]) for k, v in snap.items()}, {"a": (2, 2), "b": (0, 1)})
        self.assertIsNotNone(snap["a"]["last_request_at"])
        self.assertEqual(inflight.busy_servers(snap), {"a": 2})

        # A second gateway process with requests on the same server adds up.
        other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
        self.addCleanup(other.wait, 5)
        self.addCleanup(other.kill)
        other_file = inflight.Tracker(pid=other.pid)
        other_file.begin("a")
        self.assertEqual(inflight.snapshot()["a"]["in_flight"], 3)
        self.assertEqual(inflight.snapshot()["a"]["requests"], 3)

    def test_track_ends_even_on_error(self) -> None:
        tracker = inflight.Tracker()
        with self.assertRaises(RuntimeError):
            with tracker.track("a"):
                self.assertEqual(inflight.busy_servers(), {"a": 1})
                raise RuntimeError("upstream failed")
        self.assertEqual(inflight.busy_servers(), {})
        with tracker.track(""):  # remote targets are not tracked
            pass
        self.assertEqual(set(tracker.counts()), {"a"})

    def test_files_of_gone_gateways_are_dropped(self) -> None:
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait(timeout=10)
        stale = inflight.Tracker(pid=dead.pid)
        with mock.patch("inferencedeck.server_manager.process_identity", return_value=None):
            stale.begin("a")  # a gateway that crashed mid-request
        self.assertTrue(stale.path.exists())
        self.assertEqual(inflight.snapshot(), {})
        self.assertFalse(stale.path.exists())

    def test_reused_pid_is_not_a_live_gateway(self) -> None:
        path = self.dir / "inflight" / f"gateway-{os.getpid()}.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"pid": os.getpid(), "pid_identity": "an earlier process",
                                    "servers": {"a": {"in_flight": 1, "requests": 1}}}))
        self.assertEqual(inflight.snapshot(), {})

    def test_junk_is_ignored(self) -> None:
        folder = self.dir / "inflight"
        folder.mkdir()
        (folder / "gateway-1.json").write_text("{")
        (folder / "gateway-2.json").write_text("[]")
        (folder / f"gateway-{os.getpid()}.json").write_text(json.dumps(
            {"pid": os.getpid(), "servers": {"a": "x", "b": {"in_flight": -3, "requests": 1}}}))
        self.assertEqual(inflight.snapshot(), {"b": {"in_flight": 0, "requests": 1, "last_request_at": None}})
        self.assertEqual(inflight.snapshot(self.dir / "missing"), {})

    def test_close_removes_the_file(self) -> None:
        tracker = inflight.Tracker()
        tracker.begin("a")
        tracker.close()
        self.assertFalse(tracker.path.exists())

    def test_unwritable_cache_does_not_fail_requests(self) -> None:
        tracker = inflight.Tracker()
        with mock.patch.object(inflight, "atomic_write_text", side_effect=OSError("read-only")):
            with tracker.track("a"):
                self.assertEqual(tracker.counts()["a"]["in_flight"], 1)
        self.assertEqual(tracker.counts()["a"]["in_flight"], 0)


class StatusTests(unittest.TestCase):
    def test_status_shows_requests_per_server(self) -> None:
        servers = [{"id": "a", "running": True}, {"id": "b", "running": True}]
        counts = {"a": {"in_flight": 2, "requests": 5, "last_request_at": "2026-01-01T00:00:00+00:00"}}
        with mock.patch("inferencedeck.control.list_servers", return_value=servers), \
                mock.patch("inferencedeck.control.active_endpoint", return_value=None), \
                mock.patch("inferencedeck.control.inflight_snapshot", return_value=counts):
            status = ControlPlane().status()
        by_id = {s["id"]: s for s in status["servers"]}
        self.assertEqual(by_id["a"]["in_flight"], 2)
        self.assertEqual(by_id["a"]["last_request_at"], "2026-01-01T00:00:00+00:00")
        self.assertNotIn("in_flight", by_id["b"])


class _SlowUpstream(BaseHTTPRequestHandler):
    """Streams one chunk, then waits for ``release`` before finishing."""

    release = threading.Event()

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n')
        self.wfile.flush()
        type(self).release.wait(10)
        self.wfile.write(b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n')


class GatewayTrackingTests(_Dir):
    def setUp(self) -> None:
        super().setUp()
        _SlowUpstream.release = threading.Event()
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _SlowUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        # Runs first: let a waiting handler finish so server_close can join it.
        self.addCleanup(_SlowUpstream.release.set)
        self.server = {"id": "qwen-1-x", "mode": "qwen", "running": True, "host": "127.0.0.1",
                       "port": upstream.server_address[1]}
        self.tracker = inflight.Tracker()
        gateway = make_server("127.0.0.1", 0, AuthState(token=""),
                              Router(endpoints=lambda: [], servers=lambda: [self.server]), inflight=self.tracker)
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        self.base = f"http://127.0.0.1:{gateway.server_address[1]}"

    def _wait_for(self, predicate) -> None:
        deadline = time.monotonic() + 10
        while not predicate():
            if time.monotonic() > deadline:
                self.fail(f"timed out; counts: {inflight.snapshot()}")
            time.sleep(0.02)

    def test_streaming_request_counts_until_it_finishes(self) -> None:
        self.assertEqual(local_target(self.server).server_id, "qwen-1-x")
        body = json.dumps({"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=body.encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        replies: list[bytes] = []
        client = threading.Thread(target=lambda: replies.append(urllib.request.urlopen(request, timeout=10).read()))
        client.start()
        self._wait_for(lambda: inflight.busy_servers() == {"qwen-1-x": 1})
        _SlowUpstream.release.set()
        client.join(10)
        self.assertIn(b"[DONE]", replies[0])
        self._wait_for(lambda: inflight.busy_servers() == {})
        self.assertEqual(inflight.snapshot()["qwen-1-x"]["requests"], 1)

    def test_remote_targets_are_not_tracked(self) -> None:
        target = Target(OpenAICompatibleEngine("http://example.invalid/v1"), "remote", "m", ("m",), endpoint="box")
        self.assertEqual(target.server_id, "")
        self.assertEqual(target.to_dict()["server_id"], "")


if __name__ == "__main__":
    unittest.main()
