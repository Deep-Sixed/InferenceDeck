from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler
from inferencedeck.logstream import LogFollower, tail_bytes
from inferencedeck.server_manager import tail_file


class _TmpDir(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, name: str, text: str, mode: str = "a") -> Path:
        path = self.dir / name
        with open(path, mode, encoding="utf-8", newline="") as fh:
            fh.write(text)
        return path


class TailTests(_TmpDir):
    def test_tail_bytes_starts_at_a_line(self) -> None:
        path = self.write("a.log", "".join(f"line {i}\n" for i in range(100)))
        text = tail_bytes(path, 30)
        self.assertTrue(text.startswith("line "))
        self.assertTrue(text.endswith("line 99\n"))
        self.assertLessEqual(len(text), 30)
        self.assertEqual(tail_bytes(self.dir / "missing.log", 30), "")

    def test_tail_file_reads_only_the_end(self) -> None:
        path = self.write("big.log", "".join(f"{i:08d} {'x' * 90}\n" for i in range(50_000)))  # ~5 MB
        text = tail_file(path, lines=3)
        self.assertEqual([line[:8] for line in text.splitlines()], ["00049997", "00049998", "00049999"])
        self.assertEqual(tail_file(self.dir / "missing.log"), "")

    def test_tail_file_small_file(self) -> None:
        path = self.write("s.log", "a\nb\nc\n")
        self.assertEqual(tail_file(path, lines=2), "b\nc\n")
        self.assertEqual(tail_file(path, lines=10), "a\nb\nc\n")


class FollowerTests(_TmpDir):
    def _lines(self, events: list[dict]) -> list[str]:
        return [line for event in events if event["type"] == "lines" for line in event["lines"]]

    def test_history_then_new_lines(self) -> None:
        path = self.write("err.log", "".join(f"old {i}\n" for i in range(1000)))
        follower = LogFollower({"stderr": path}, history_bytes=40)
        history = follower.start()
        self.assertTrue(history[0]["history"])
        self.assertEqual(history[0]["lines"][-1], "old 999")
        self.assertLessEqual(sum(len(l) + 1 for l in history[0]["lines"]), 40)
        self.assertEqual(follower.poll(), [])
        self.write("err.log", "new 1\nnew 2\n")
        self.assertEqual(self._lines(follower.poll()), ["new 1", "new 2"])

    def test_partial_line_waits_for_newline(self) -> None:
        path = self.write("err.log", "")
        follower = LogFollower({"stderr": path})
        follower.start()
        self.write("err.log", "loading mod")
        self.assertEqual(follower.poll(), [])
        self.write("err.log", "el... done\n")
        self.assertEqual(self._lines(follower.poll()), ["loading model... done"])

    def test_very_long_partial_line_is_flushed(self) -> None:
        path = self.write("err.log", "")
        follower = LogFollower({"stderr": path}, max_line=10)
        follower.start()
        self.write("err.log", "x" * 25)
        self.assertEqual(self._lines(follower.poll()), ["x" * 25])

    def test_truncation_is_a_reset(self) -> None:
        path = self.write("err.log", "first run line\n" * 10)
        follower = LogFollower({"stderr": path})
        follower.start()
        self.write("err.log", "second run\n", mode="w")
        events = follower.poll()
        self.assertEqual(events[0], {"type": "reset", "stream": "stderr"})
        self.assertEqual(self._lines(events), ["second run"])

    def test_falling_behind_skips_and_resyncs(self) -> None:
        path = self.write("err.log", "")
        follower = LogFollower({"stderr": path}, history_bytes=100, max_backlog_bytes=1000, read_bytes=10_000)
        follower.start()
        self.write("err.log", "".join(f"burst {i:05d}\n" for i in range(500)))  # 6000 bytes
        events = follower.poll()
        skipped = [e for e in events if e["type"] == "skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertGreater(skipped[0]["bytes"], 5000)
        lines = self._lines(events)
        self.assertTrue(all(line.startswith("burst ") and len(line) == 11 for line in lines), lines)
        self.assertEqual(lines[-1], "burst 00499")

    def test_reads_are_capped_per_poll(self) -> None:
        path = self.write("err.log", "")
        follower = LogFollower({"stderr": path}, read_bytes=100, max_backlog_bytes=10_000)
        follower.start()
        self.write("err.log", "".join(f"row {i:04d}\n" for i in range(50)))  # 450 bytes
        first = self._lines(follower.poll())
        self.assertLessEqual(len(first), 12)
        rest: list[str] = []
        for _ in range(10):
            rest += self._lines(follower.poll())
        self.assertEqual(first + rest, [f"row {i:04d}" for i in range(50)])

    def test_two_streams_and_missing_file(self) -> None:
        err = self.write("err.log", "")
        follower = LogFollower({"stderr": err, "stdout": self.dir / "not-yet.log"})
        follower.start()
        self.write("not-yet.log", "hello\n")
        self.write("err.log", "warn\n")
        events = follower.poll()
        self.assertEqual({(e["stream"], tuple(e["lines"])) for e in events}, {("stderr", ("warn",)), ("stdout", ("hello",))})


class StreamEndpointTests(_TmpDir):
    def setUp(self) -> None:
        super().setUp()
        self.err = self.write("qwen-stderr.log", "".join(f"boot {i}\n" for i in range(5)))
        self.out = self.write("qwen-stdout.log", "")
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.create_autospec(ControlPlane, instance=True)
        handler.log_poll_seconds = 0.02
        handler.log_keepalive_seconds = 0.2
        handler.log_check_seconds = 0.05
        handler.log_stream_slots = threading.BoundedSemaphore(2)
        self.handler = handler
        self.paths = {"stderr": str(self.err), "stdout": str(self.out)}
        handler.control_plane.log_paths.side_effect = lambda sid: dict(self.paths) if sid == "qwen-1" and self.paths else None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        # Runs before the temp dir is removed: end every stream so no handler
        # thread still has a log file open (Windows cannot delete open files).
        self.addCleanup(self._end_streams)

    def _end_streams(self) -> None:
        self.paths = {}
        for _ in range(200):
            if self.handler.log_stream_slots._value == 2:
                return
            threading.Event().wait(0.02)

    def _open(self, query: str) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=5)
        self.addCleanup(conn.close)
        conn.request("GET", "/api/logs/stream?" + query)
        return conn.getresponse()

    def _events(self, response: http.client.HTTPResponse, until: str):
        """Yield (event, data) pairs until an event named ``until`` has been read."""
        event = None
        while True:
            line = response.fp.readline().decode("utf-8")
            if line == "":
                return
            line = line.rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                yield event, json.loads(line[6:])
                if event == until:
                    return

    def test_history_live_lines_and_end(self) -> None:
        response = self._open("server_id=qwen-1")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream; charset=utf-8")
        seen = list(self._events(response, until="ready"))
        self.assertEqual(seen[0][1]["lines"], [f"boot {i}" for i in range(5)])
        self.assertEqual(seen[-1][1]["streams"], ["stderr", "stdout"])

        self.write("qwen-stdout.log", "request served\n")
        event, data = next(self._events(response, until="lines"))
        self.assertEqual((event, data["stream"], data["lines"]), ("lines", "stdout", ["request served"]))

        self.paths = {}  # the server is no longer tracked
        tail = list(self._events(response, until="end"))
        self.assertEqual(tail[-1], ("end", {"reason": "server_gone"}))

    def test_single_stream_and_no_history(self) -> None:
        response = self._open("server_id=qwen-1&stream=stdout&history=0")
        seen = list(self._events(response, until="ready"))
        self.assertEqual(seen, [("ready", {"server_id": "qwen-1", "streams": ["stdout"]})])

    def test_errors(self) -> None:
        self.assertEqual(self._open("server_id=nope").status, 404)
        self.assertEqual(self._open("").status, 400)
        self.assertEqual(self._open("server_id=qwen-1&stream=both2").status, 400)
        self.assertEqual(self._open("server_id=qwen-1&history=lots").status, 400)

    def test_stream_limit(self) -> None:
        first = self._open("server_id=qwen-1")
        second = self._open("server_id=qwen-1")
        list(self._events(first, until="ready"))
        list(self._events(second, until="ready"))
        self.assertEqual(self._open("server_id=qwen-1").status, 429)
        # Closing a viewer frees its slot once the server notices (next write fails).
        first.close()
        self.paths = {}
        list(self._events(second, until="end"))
        self._end_streams()
        self.assertEqual(self.handler.log_stream_slots._value, 2)
        self.paths = {"stderr": str(self.err), "stdout": str(self.out)}
        self.assertEqual(self._open("server_id=qwen-1").status, 200)


if __name__ == "__main__":
    unittest.main()
